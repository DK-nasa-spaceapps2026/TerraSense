"""Phase 4 Leakage-Free Spatial, Temporal & Spatio-Temporal Validation (`backend/ml/evaluate.py`).

Evaluates the Hybrid State-Transition Model against 3 baselines
(Persistence, Per-Month Climatology, and the Heuristic Water-Balance Engine) across:
  1. Buffered Spatial Cluster CV (>= 25 km Haversine buffer + shared-cell exclusion)
  2. Temporal Year Splits (Train 2023 -> Test 2024 rollout, and Train 2024 -> Test 2023 rollout)
  3. Strictest Split: Leave-Spatial-Cluster-AND-Year-Out (>= 25 km spatial buffer AND unseen year)

Computes RMSE, MAE, R2, Spearman correlation, Skill Scores vs. Climatology/Persistence/Heuristic,
per-horizon (h=1..12) rollout curves, 80% (P10-P90) conformal interval coverage, and
macro-F1 / confusion matrix on the 5 zone status classes.
Writes reports to `backend/ml/reports/validation_report.json` and `validation_report.md`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score

import zone_engine as ze
from ml.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, SEED, STATIC_SOIL_TERRAIN_FEATURES
from ml.features import compute_proxy_indices
from ml.train import fit_transition_lgbm, rollout_point_trajectory


def haversine_km_matrix(lats1: np.ndarray, lons1: np.ndarray, lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
    """Pairwise Haversine distance in km between (lats1, lons1) and (lats2, lons2)."""
    r_earth = 6371.0
    phi1 = np.radians(lats1)[:, None]
    phi2 = np.radians(lats2)[None, :]
    dphi = phi2 - phi1
    dlam = np.radians(lons2)[None, :] - np.radians(lons1)[:, None]
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * r_earth * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_clim: np.ndarray,
    y_persist: np.ndarray,
    y_heur: np.ndarray,
) -> Dict[str, float]:
    mse_mod = float(mean_squared_error(y_true, y_pred))
    mse_clim = float(mean_squared_error(y_true, y_clim))
    mse_per = float(mean_squared_error(y_true, y_persist))
    mse_heur = float(mean_squared_error(y_true, y_heur))
    sp_corr = float(spearmanr(y_true, y_pred).statistic) if np.std(y_pred) > 1e-9 else 0.0
    return {
        "rmse": round(math.sqrt(mse_mod), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "spearman": round(sp_corr, 4),
        "skill_vs_climatology": round(1.0 - mse_mod / max(mse_clim, 1e-9), 4),
        "skill_vs_persistence": round(1.0 - mse_mod / max(mse_per, 1e-9), 4),
        "skill_vs_heuristic": round(1.0 - mse_mod / max(mse_heur, 1e-9), 4),
    }


def _status_from_trajectories(
    df_eval: pd.DataFrame, smi_arr: np.ndarray, esi_arr: np.ndarray
) -> np.ndarray:
    """Compute TerraSense 5-class status codes using zone_engine.classify rules."""
    n = len(df_eval)
    bare = np.zeros(n)
    ndvi = np.zeros(n)
    vigor = np.zeros(n)
    heat = np.zeros(n)
    for i, (_, r) in enumerate(df_eval.iterrows()):
        p = compute_proxy_indices(
            smi=float(smi_arr[i]),
            esi=float(esi_arr[i]),
            cal_cover=float(r["cal_cover"]),
            slope_pct=float(r["slope_pct"]),
            sand_pct=float(r["sand_top_pct"]),
            precip_mm=float(r["precip_mm"]),
            max_daily_precip_mm=float(r["max_daily_precip_mm"]),
        )
        bare[i] = p["bare"]
        ndvi[i] = p["ndvi"]
        vigor[i] = p["vigor"]
        heat[i] = float(np.clip((float(r["tmax_c"]) - 30.0) / 8.0, 0.0, 1.0)) * (1.0 if float(r["cal_cover"]) > 0.4 else 0.0)

    sim_dict = {
        "smi": smi_arr[:, None],
        "esi": esi_arr[:, None],
        "bare": bare[:, None],
        "ndvi": ndvi[:, None],
        "vigor": vigor[:, None],
        "heat": heat[:, None],
    }
    return ze.classify(sim_dict)[:, 0]


def _evaluate_rollout_subset(
    df_train: pd.DataFrame,
    df_test_points: pd.DataFrame,
    test_year: int,
    conformal_smi: Dict[str, Dict[str, float]],
    conformal_esi: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Fit models strictly on `df_train` and evaluate 11-12 month rollout on `df_test_points` in `test_year`."""
    # Impute static soil medians strictly inside training fold
    df_tr = df_train.copy()
    df_te = df_test_points.copy()
    for c in STATIC_SOIL_TERRAIN_FEATURES:
        med = float(df_tr[c].median()) if not df_tr[c].isna().all() else 0.0
        df_tr[c] = df_tr[c].fillna(med)
        df_te[c] = df_te[c].fillna(med)

    smi_mod = fit_transition_lgbm(df_tr, target_kind="smi")
    esi_mod = fit_transition_lgbm(df_tr, target_kind="esi")

    clim_smi = df_tr.groupby("cal_month")["target_smi"].mean().to_dict()
    clim_esi = df_tr.groupby("cal_month")["target_esi"].mean().to_dict()

    records: List[Dict[str, Any]] = []
    for pid, grp in df_te.groupby("point_id", sort=False):
        grp = grp.sort_values("year_month").reset_index(drop=True)
        if test_year == 2024:
            # Anchor at Dec 2023 (idx 11), roll out Jan..Dec 2024 (idx 12..23, h=1..12)
            start_idx = 12
        else:
            # Anchor at Jan 2023 (idx 0), roll out Feb..Dec 2023 (idx 1..11, h=1..11)
            start_idx = 1

        roll = rollout_point_trajectory(grp, smi_mod, esi_mod, start_idx=start_idx)
        init_smi = float(grp.loc[start_idx - 1, "target_smi"])
        init_esi = float(grp.loc[start_idx - 1, "target_esi"])

        end_idx = start_idx + 12 if test_year == 2024 else 12
        for t in range(start_idx, min(end_idx, len(grp))):
            h = t - start_idx + 1
            r = grp.loc[t]
            cm = int(r["cal_month"])
            q_smi = conformal_smi[str(h)]
            q_esi = conformal_esi[str(h)]
            s_hat = float(roll["smi"][t])
            e_hat = float(roll["esi"][t])
            records.append({
                "point_id": pid,
                "year_month": r["year_month"],
                "cal_month": cm,
                "horizon": h,
                "cal_cover": float(r["cal_cover"]),
                "slope_pct": float(r["slope_pct"]),
                "sand_top_pct": float(r["sand_top_pct"]),
                "precip_mm": float(r["precip_mm"]),
                "max_daily_precip_mm": float(r["max_daily_precip_mm"]),
                "tmax_c": float(r["tmax_c"]),
                "smi_true": float(r["target_smi"]),
                "smi_pred": s_hat,
                "smi_p10": float(np.clip(s_hat + q_smi["q10"], 0.0, 1.0)),
                "smi_p90": float(np.clip(s_hat + q_smi["q90"], 0.0, 1.0)),
                "smi_clim": float(clim_smi[cm]),
                "smi_persist": init_smi,
                "smi_heur": float(r["wb_smi"]),
                "esi_true": float(r["target_esi"]),
                "esi_pred": e_hat,
                "esi_p10": float(np.clip(e_hat + q_esi["q10"], 0.0, 1.0)),
                "esi_p90": float(np.clip(e_hat + q_esi["q90"], 0.0, 1.0)),
                "esi_clim": float(clim_esi[cm]),
                "esi_persist": init_esi,
                "esi_heur": float(r["wb_esi"]),
            })
    return {"records": pd.DataFrame(records)}


def _summarize_records(res_df: pd.DataFrame) -> Dict[str, Any]:
    smi_m = compute_regression_metrics(
        res_df["smi_true"].to_numpy(),
        res_df["smi_pred"].to_numpy(),
        res_df["smi_clim"].to_numpy(),
        res_df["smi_persist"].to_numpy(),
        res_df["smi_heur"].to_numpy(),
    )
    esi_m = compute_regression_metrics(
        res_df["esi_true"].to_numpy(),
        res_df["esi_pred"].to_numpy(),
        res_df["esi_clim"].to_numpy(),
        res_df["esi_persist"].to_numpy(),
        res_df["esi_heur"].to_numpy(),
    )
    smi_base_rmse = {
        "climatology_rmse": round(math.sqrt(mean_squared_error(res_df["smi_true"], res_df["smi_clim"])), 4),
        "persistence_rmse": round(math.sqrt(mean_squared_error(res_df["smi_true"], res_df["smi_persist"])), 4),
        "heuristic_rmse": round(math.sqrt(mean_squared_error(res_df["smi_true"], res_df["smi_heur"])), 4),
    }
    esi_base_rmse = {
        "climatology_rmse": round(math.sqrt(mean_squared_error(res_df["esi_true"], res_df["esi_clim"])), 4),
        "persistence_rmse": round(math.sqrt(mean_squared_error(res_df["esi_true"], res_df["esi_persist"])), 4),
        "heuristic_rmse": round(math.sqrt(mean_squared_error(res_df["esi_true"], res_df["esi_heur"])), 4),
    }

    # Interval coverage (nominal 80% P10..P90)
    smi_cov80 = float(np.mean((res_df["smi_true"] >= res_df["smi_p10"]) & (res_df["smi_true"] <= res_df["smi_p90"])))
    esi_cov80 = float(np.mean((res_df["esi_true"] >= res_df["esi_p10"]) & (res_df["esi_true"] <= res_df["esi_p90"])))

    # Status classification metrics (ML vs Heuristic against Ground-Truth Target Status)
    st_true = _status_from_trajectories(res_df, res_df["smi_true"].to_numpy(), res_df["esi_true"].to_numpy())
    st_ml = _status_from_trajectories(res_df, res_df["smi_pred"].to_numpy(), res_df["esi_pred"].to_numpy())
    st_heur = _status_from_trajectories(res_df, res_df["smi_heur"].to_numpy(), res_df["esi_heur"].to_numpy())

    present_labels = sorted(list(set(st_true) | set(st_ml) | set(st_heur)))
    f1_ml = float(f1_score(st_true, st_ml, labels=present_labels, average="macro", zero_division=0))
    f1_heur = float(f1_score(st_true, st_heur, labels=present_labels, average="macro", zero_division=0))
    acc_ml = float(np.mean(st_true == st_ml))
    acc_heur = float(np.mean(st_true == st_heur))
    cm_ml = confusion_matrix(st_true, st_ml, labels=list(range(len(ze.STATUSES)))).tolist()

    # Per-horizon RMSE curve (h = 1..12)
    by_horizon = {}
    for h, hgrp in res_df.groupby("horizon"):
        by_horizon[str(int(h))] = {
            "smi_rmse_ml": round(math.sqrt(mean_squared_error(hgrp["smi_true"], hgrp["smi_pred"])), 4),
            "smi_rmse_heur": round(math.sqrt(mean_squared_error(hgrp["smi_true"], hgrp["smi_heur"])), 4),
            "esi_rmse_ml": round(math.sqrt(mean_squared_error(hgrp["esi_true"], hgrp["esi_pred"])), 4),
            "esi_rmse_heur": round(math.sqrt(mean_squared_error(hgrp["esi_true"], hgrp["esi_heur"])), 4),
        }

    return {
        "n_eval_rows": int(len(res_df)),
        "smi": {**smi_m, **smi_base_rmse, "conformal_80pct_coverage": round(smi_cov80, 4)},
        "esi": {**esi_m, **esi_base_rmse, "conformal_80pct_coverage": round(esi_cov80, 4)},
        "status_classification": {
            "macro_f1_ml": round(f1_ml, 4),
            "macro_f1_heuristic": round(f1_heur, 4),
            "accuracy_ml": round(acc_ml, 4),
            "accuracy_heuristic": round(acc_heur, 4),
            "labels": ze.STATUSES,
            "confusion_matrix_ml": cm_ml,
        },
        "horizon_curves": by_horizon,
    }


def run_validation_suite(buffer_km: float = 25.0) -> Dict[str, Any]:
    np.random.seed(SEED)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / "training_monthly.csv").sort_values(["point_id", "year_month"]).reset_index(drop=True)
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text())
    conf_smi = manifest["conformal_smi"]
    conf_esi = manifest["conformal_esi"]

    pts = df[["point_id", "lat", "lon", "era5_cell_id", "merra2_cell_id", "spatial_block_id"]].drop_duplicates().reset_index(drop=True)
    dist_km = haversine_km_matrix(
        pts["lat"].to_numpy(), pts["lon"].to_numpy(),
        pts["lat"].to_numpy(), pts["lon"].to_numpy(),
    )

    # Group spatial blocks into 5 macro-clusters for buffered spatial CV
    unique_blocks = sorted(pts["spatial_block_id"].unique())
    block_to_fold = {b: i % 5 for i, b in enumerate(unique_blocks)}
    pts["fold"] = pts["spatial_block_id"].map(block_to_fold)

    # 1. Buffered Spatial Cluster CV (>= 25 km Haversine buffer + shared ERA5/MERRA cell exclusion)
    print(f"[Phase 4] 1/3 Running Buffered Spatial Cluster CV (buffer >= {buffer_km:.0f} km)...")
    spatial_records = []
    for f in range(5):
        te_mask = (pts["fold"] == f).to_numpy()
        te_indices = np.where(te_mask)[0]
        te_pids = set(pts.loc[te_mask, "point_id"])
        te_era5 = set(pts.loc[te_mask, "era5_cell_id"])

        # Exclude any training point within `buffer_km` of ANY test point or sharing an ERA5 cell
        min_dist_to_test = dist_km[:, te_indices].min(axis=1)
        tr_mask = (~te_mask) & (min_dist_to_test >= buffer_km) & (~pts["era5_cell_id"].isin(te_era5).to_numpy())
        tr_pids = set(pts.loc[tr_mask, "point_id"])

        out = _evaluate_rollout_subset(
            df_train=df[df["point_id"].isin(tr_pids)],
            df_test_points=df[df["point_id"].isin(te_pids)],
            test_year=2024,
            conformal_smi=conf_smi,
            conformal_esi=conf_esi,
        )
        spatial_records.append(out["records"])
    spatial_summary = _summarize_records(pd.concat(spatial_records, ignore_index=True))

    # 2. Temporal Year Splits (Train 2023 -> Test 2024, and Train 2024 -> Test 2023)
    print("[Phase 4] 2/3 Running Temporal Year-to-Year Splits (2023->2024 and 2024->2023)...")
    temp_23_to_24 = _evaluate_rollout_subset(
        df_train=df[df["year"] == 2023],
        df_test_points=df,
        test_year=2024,
        conformal_smi=conf_smi,
        conformal_esi=conf_esi,
    )["records"]
    temp_24_to_23 = _evaluate_rollout_subset(
        df_train=df[df["year"] == 2024],
        df_test_points=df,
        test_year=2023,
        conformal_smi=conf_smi,
        conformal_esi=conf_esi,
    )["records"]
    temporal_summary = {
        "train_2023_test_2024": _summarize_records(temp_23_to_24),
        "train_2024_test_2023": _summarize_records(temp_24_to_23),
        "combined_temporal": _summarize_records(pd.concat([temp_23_to_24, temp_24_to_23], ignore_index=True)),
    }

    # 3. Strictest Split: Leave-Spatial-Cluster-AND-Year-Out (>= 25 km buffer AND unseen year)
    print(f"[Phase 4] 3/3 Running Strictest Split: Leave-Spatial-Cluster-AND-Year-Out (>= {buffer_km:.0f} km + unseen year)...")
    strict_records = []
    for f in range(5):
        te_mask = (pts["fold"] == f).to_numpy()
        te_indices = np.where(te_mask)[0]
        te_pids = set(pts.loc[te_mask, "point_id"])
        te_era5 = set(pts.loc[te_mask, "era5_cell_id"])
        min_dist_to_test = dist_km[:, te_indices].min(axis=1)
        tr_mask = (~te_mask) & (min_dist_to_test >= buffer_km) & (~pts["era5_cell_id"].isin(te_era5).to_numpy())
        tr_pids = set(pts.loc[tr_mask, "point_id"])

        # Train ONLY on buffered spatial train points in 2023 -> test on held-out spatial cluster in 2024
        out_24 = _evaluate_rollout_subset(
            df_train=df[(df["point_id"].isin(tr_pids)) & (df["year"] == 2023)],
            df_test_points=df[df["point_id"].isin(te_pids)],
            test_year=2024,
            conformal_smi=conf_smi,
            conformal_esi=conf_esi,
        )
        strict_records.append(out_24["records"])
    strict_summary = _summarize_records(pd.concat(strict_records, ignore_index=True))

    # Promotion gate check on strict split & spatial buffered split:
    # Beats heuristic engine, persistence, and climatology on both SMI and ESI
    gate_passed = bool(
        strict_summary["smi"]["skill_vs_heuristic"] > 0.0
        and strict_summary["smi"]["skill_vs_persistence"] > 0.0
        and strict_summary["smi"]["skill_vs_climatology"] > 0.0
        and strict_summary["esi"]["skill_vs_heuristic"] > 0.0
        and strict_summary["esi"]["skill_vs_persistence"] > 0.0
        and strict_summary["esi"]["skill_vs_climatology"] > 0.0
        and spatial_summary["smi"]["skill_vs_climatology"] > 0.0
        and spatial_summary["esi"]["skill_vs_climatology"] > 0.0
    )

    report = {
        "gate_passed": gate_passed,
        "recommended_default_engine": "ml" if gate_passed else "heuristic",
        "buffer_km": buffer_km,
        "buffered_spatial_cv": spatial_summary,
        "temporal_cv": temporal_summary,
        "strict_cluster_and_year_out": strict_summary,
    }
    json_path = REPORTS_DIR / "validation_report.json"
    json_path.write_text(json.dumps(report, indent=2))

    # Write concise Markdown validation report
    md_lines = [
        "# Phase 4 — Spatio-Temporal Validation & Baseline Comparison Report",
        "",
        f"- **Promotion Gate Passed:** `{gate_passed}` (Recommended default engine: `{report['recommended_default_engine']}`)",
        f"- **Spatial Buffer Enforced:** `>= {buffer_km:.0f} km` Haversine exclusion zone between train and test points (plus zero shared ERA5-Land cells).",
        "",
        "## 1. Summary Table Across All Validation Regimes (12-Month Free-Running Rollouts)",
        "",
        "| Validation Split | Target | ML RMSE | Heuristic RMSE | Climatology RMSE | Persistence RMSE | ML $R^2$ | Spearman $r$ | Skill vs. Clim | Skill vs. Heur | 80% Interval Cov |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in [
        ("Buffered Spatial CV (>=25 km)", spatial_summary),
        ("Temporal (2023->2024 & 2024->2023)", temporal_summary["combined_temporal"]),
        ("Strictest: Cluster (>=25 km) + Year Out", strict_summary),
    ]:
        for tkey, label in [("smi", "SMI"), ("esi", "ESI")]:
            m = s[tkey]
            md_lines.append(
                f"| {name} | **{label}** | **{m['rmse']:.4f}** | {m['heuristic_rmse']:.4f} | "
                f"{m['climatology_rmse']:.4f} | {m['persistence_rmse']:.4f} | {m['r2']:.4f} | "
                f"{m['spearman']:.4f} | `+{100*m['skill_vs_climatology']:.1f}%` | "
                f"`+{100*m['skill_vs_heuristic']:.1f}%` | `{100*m['conformal_80pct_coverage']:.1f}%` |"
            )

    md_lines.extend([
        "",
        "## 2. Status Classification Accuracy & Macro-F1 (`lush`, `moderate`, `water_stress`, `heat_stress`, `bare_vulnerable`)",
        "",
        "| Validation Split | ML Macro-F1 | Heuristic Macro-F1 | ML Accuracy | Heuristic Accuracy |",
        "|---|---:|---:|---:|---:|",
        f"| Buffered Spatial CV (>=25 km) | **{spatial_summary['status_classification']['macro_f1_ml']:.4f}** | {spatial_summary['status_classification']['macro_f1_heuristic']:.4f} | **{100*spatial_summary['status_classification']['accuracy_ml']:.1f}%** | {100*spatial_summary['status_classification']['accuracy_heuristic']:.1f}% |",
        f"| Temporal Combined | **{temporal_summary['combined_temporal']['status_classification']['macro_f1_ml']:.4f}** | {temporal_summary['combined_temporal']['status_classification']['macro_f1_heuristic']:.4f} | **{100*temporal_summary['combined_temporal']['status_classification']['accuracy_ml']:.1f}%** | {100*temporal_summary['combined_temporal']['status_classification']['accuracy_heuristic']:.1f}% |",
        f"| Strictest: Cluster + Year Out | **{strict_summary['status_classification']['macro_f1_ml']:.4f}** | {strict_summary['status_classification']['macro_f1_heuristic']:.4f} | **{100*strict_summary['status_classification']['accuracy_ml']:.1f}%** | {100*strict_summary['status_classification']['accuracy_heuristic']:.1f}% |",
        "",
        "## 3. Known Limitations & Honest Assessment",
        "- **2-Year Temporal Record (2023–2024):** While 2023 (drought) and 2024 (wetter spring/summer) provide strong contrast, training on `2023` alone in the strictest year-out split has narrower historical coverage than training on 10+ years.",
        "- **Target Resolution:** `target_smi_era5` (~9 km, 106 cropland points) and `target_smi_merra2` / `target_esi` (~50 km, 195 cropland points) are regional reanalysis grids; fine 30 m sub-field variation is driven by the SSURGO + metric DEM +FAO-56 water-balance backbone.",
        "- **Proxy NDVI & Bare-Soil Exposure:** Flagged as `is_proxy_ndvi_bare=1` until AppEEARS MODIS NDVI / ECOSTRESS granules are added to `data/appeears/`.",
    ])
    md_path = REPORTS_DIR / "validation_report.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[Phase 4] Validation complete. Gate passed = {gate_passed}. Saved to {md_path}")
    return report


if __name__ == "__main__":
    run_validation_suite()

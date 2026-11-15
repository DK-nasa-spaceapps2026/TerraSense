"""Phase 1 Dataset Builder for TerraSense (`backend/ml/build_dataset.py`).

Produces:
  - `data/processed/points_static.csv`: All 305 points with CDL classes, explicit
    keep/drop audit (`keep_for_training`, `drop_reason`), metric DEM terrain
    features, fused SSURGO + SoilGrids soil features, and spatial cluster IDs.
  - `data/processed/training_monthly.csv`: Tidy point x month table for all
    retained cropland points, including physics-informed weather features,
    FAO-56 water-balance backbone states, lagged state features, delta targets,
    and documented proxy indices.
  - `data/processed/schema.json`: Full column schema, units, and summary stats.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import zone_engine as ze
from ml.config import (
    CROP_FEATURES,
    CULTIVATED_CROP_CODES,
    DATA_DIR,
    FORAGE_CROP_CODES,
    POINTS_CSV,
    PROCESSED_DIR,
    REPORTS_DIR,
    SEED,
    STATIC_SOIL_TERRAIN_FEATURES,
    WEATHER_FEATURES,
)
from ml.features import (
    aggregate_point_monthly_series,
    compute_proxy_indices,
    extract_cdl_features,
    extract_terrain_for_coords,
    fuse_soil_features,
    load_appeears_ndvi_hook,
    parse_soilgrids_file,
    parse_ssurgo_file,
    soil_water_bounds,
    spatial_cell_ids,
)


def _crop_calendar_for_code(crop_code: int, cal_month: int) -> Dict[str, float]:
    """Return (cal_cover, cal_kc) for a given USDA CDL crop code and calendar month (1..12)."""
    if crop_code == 4:  # Sorghum
        cal = ze.CALENDARS["drought_cash"]
    elif crop_code in FORAGE_CROP_CODES:  # Alfalfa / Hay
        cal = ze.CALENDARS["cover_legume"]
    else:  # Corn (1), Soybeans (5), Oats (28), default row crop
        cal = ze.CALENDARS["status_quo"]
    idx = (cal_month - 1) % 12
    cover = float(cal["cover"][idx])
    kc = float(cal["kc"][idx])
    # Soybeans have slightly lower peak Kc and shorter canopy window than corn
    if crop_code == 5 and 5 <= cal_month <= 9:
        kc *= 0.92
    return {"cal_cover": round(cover, 4), "cal_kc": round(kc, 4)}


def _water_balance_step(
    sm_prev_mm: float,
    awc_mm: float,
    slope_pct: float,
    clay_frac: float,
    cal_cover: float,
    cal_kc: float,
    precip_mm: float,
    max_daily_precip_mm: float,
    pet_mm: float,
    tmax_c: float,
) -> Dict[str, float]:
    """Single-month FAO-56 / bucket water-balance step (physics backbone)."""
    intensity = float(np.clip(max_daily_precip_mm / 40.0, 0.0, 1.0))
    runoff_frac = float(np.clip(
        0.04 + 0.025 * slope_pct + 0.18 * (1.0 - cal_cover) * (0.6 + intensity) + 0.15 * clay_frac,
        0.0, 0.55,
    ))
    runoff_mm = precip_mm * runoff_frac
    infil_mm = precip_mm - runoff_mm

    demand_mm = pet_mm * cal_kc
    avail_mm = sm_prev_mm + 0.5 * infil_mm
    beta = float(np.clip(avail_mm / max(10.0, 0.35 * awc_mm), 0.05, 1.0))
    heat = float(np.clip((tmax_c - 30.0) / 8.0, 0.0, 1.0)) * (1.0 if cal_cover > 0.4 else 0.0)
    et_mm = min(demand_mm * beta, sm_prev_mm + infil_mm)
    sm_new_mm = float(np.clip(sm_prev_mm + infil_mm - et_mm, 0.0, awc_mm))
    wb_smi = float(np.clip(0.5 * (sm_prev_mm + sm_new_mm) / max(10.0, awc_mm), 0.0, 1.0))
    wb_esi = float(np.clip((et_mm / max(demand_mm, 1e-6)) * (1.0 - 0.25 * heat), 0.0, 1.0)) if demand_mm > 1e-6 else 1.0
    return {
        "wb_sm_new_mm": sm_new_mm,
        "wb_smi": round(wb_smi, 4),
        "wb_esi": round(wb_esi, 4),
        "wb_et_mm": round(et_mm, 2),
        "wb_runoff_mm": round(runoff_mm, 2),
        "wb_demand_mm": round(demand_mm, 2),
    }


def build_dataset() -> Dict[str, Any]:
    np.random.seed(SEED)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with POINTS_CSV.open("r", newline="") as f:
        points = list(csv.DictReader(f))

    print(f"[Phase 1] Extracting CDL 3x3 features for {len(points)} points...")
    cdl_by_id = extract_cdl_features(points)

    print(f"[Phase 1] Reprojecting DEM patches to EPSG:5070 and computing terrain features...")
    coords = [(float(p["Longitude"]), float(p["Latitude"])) for p in points]
    terrain_list = extract_terrain_for_coords(coords)

    print(f"[Phase 1] Aggregating SSURGO (primary) and SoilGrids (secondary) soil features...")
    static_rows: List[Dict[str, Any]] = []
    for pt, terr in zip(points, terrain_list):
        pid = pt["ID"]
        lat, lon = float(pt["Latitude"]), float(pt["Longitude"])
        ssurgo = parse_ssurgo_file(DATA_DIR / "ssurgo" / f"{pid}.json")
        sg = parse_soilgrids_file(DATA_DIR / "soilgrids" / f"{pid}.json")
        soil = fuse_soil_features(ssurgo, sg)
        merra_id, era5_id, blk_id = spatial_cell_ids(lat, lon)
        theta_wp, theta_fc = soil_water_bounds(
            soil["clay_top_pct"], soil["sand_top_pct"], soil["awc_top_cmcm"]
        )
        static_rows.append({
            "point_id": pid,
            "category": pt["Category"],
            "lat": lat,
            "lon": lon,
            "merra2_cell_id": merra_id,
            "era5_cell_id": era5_id,
            "spatial_block_id": blk_id,
            "theta_wp": round(theta_wp, 4),
            "theta_fc": round(theta_fc, 4),
            **cdl_by_id[pid],
            **soil,
            **terr,
        })

    static_df = pd.DataFrame(static_rows)
    # Compute training-set medians for SSURGO fallback on the few points with missing SSURGO & SoilGrids
    kept_mask = static_df["keep_for_training"] == True  # noqa: E712
    ssurgo_core_cols = [
        "sand_top_pct", "clay_top_pct", "silt_top_pct", "om_top_pct",
        "ph_top", "bd_top_gcm3", "awc_top_cmcm", "sand_sub_pct",
        "clay_sub_pct", "awc_sub_cmcm", "awc_100cm_mm", "drainage_code",
    ]
    medians = {c: float(static_df.loc[kept_mask, c].median()) for c in ssurgo_core_cols}
    for c, med in medians.items():
        static_df[f"{c}_imputed"] = static_df[c].isna().astype(int)
        static_df[c] = static_df[c].fillna(med)

    static_csv = PROCESSED_DIR / "points_static.csv"
    static_df.to_csv(static_csv, index=False)
    print(f"[Phase 1] Wrote {len(static_df)} static point rows to {static_csv}")

    # Load optional AppEEARS NDVI hook (empty if not yet downloaded)
    appeears_ndvi = load_appeears_ndvi_hook()

    # Build monthly panel for retained cropland points
    monthly_rows: List[Dict[str, Any]] = []
    for _, srow in static_df[static_df["keep_for_training"]].iterrows():
        pid = srow["point_id"]
        p_file = DATA_DIR / "nasa_power" / f"{pid}.json"
        t_file = DATA_DIR / "nasa_power_targets" / f"{pid}.json"
        om_file = DATA_DIR / "open_meteo" / f"{pid}.json"

        power_daily = json.loads(p_file.read_text())["daily"]
        target_daily = json.loads(t_file.read_text())["daily"]
        om_daily = json.loads(om_file.read_text())["daily"] if om_file.exists() else None

        m_series = aggregate_point_monthly_series(
            power_daily, target_daily, om_daily,
            theta_wp=float(srow["theta_wp"]),
            theta_fc=float(srow["theta_fc"]),
        )

        awc_mm = float(srow["awc_100cm_mm"])
        slope_pct = float(srow["slope_pct"])
        clay_frac = float(srow["clay_top_pct"]) / 100.0
        sand_pct = float(srow["sand_top_pct"])

        # Initialize physics water-balance bucket at first month's observed SMI (or 0.65 * AWC)
        init_smi = m_series[0]["target_smi_era5"]
        if np.isnan(init_smi):
            init_smi = m_series[0]["target_smi_merra2"]
        wb_sm_mm = (init_smi if not np.isnan(init_smi) else 0.65) * awc_mm

        prev_smi_era5 = np.nan
        prev_smi_merra2 = np.nan
        prev_esi = np.nan
        prev_wb_smi = np.nan

        for m in m_series:
            yr = m["year"]
            cal_m = m["cal_month"]
            crop_code = int(srow["crop_2023"]) if yr <= 2023 else int(srow["crop_2024"])
            prev_crop_code = int(srow["crop_2023"]) if yr == 2024 else crop_code
            cal_feats = _crop_calendar_for_code(crop_code, cal_m)

            wb = _water_balance_step(
                sm_prev_mm=wb_sm_mm,
                awc_mm=awc_mm,
                slope_pct=slope_pct,
                clay_frac=clay_frac,
                cal_cover=cal_feats["cal_cover"],
                cal_kc=cal_feats["cal_kc"],
                precip_mm=m["precip_mm"],
                max_daily_precip_mm=m["max_daily_precip_mm"],
                pet_mm=m["pet_mm"],
                tmax_c=m["tmax_c"],
            )
            wb_sm_mm = wb["wb_sm_new_mm"]

            smi_era5 = m["target_smi_era5"]
            smi_merra2 = m["target_smi_merra2"]
            # Unified SMI: ERA5-Land (~9 km) when present, else MERRA-2
            has_era5 = not np.isnan(smi_era5)
            smi_unified = smi_era5 if has_era5 else smi_merra2
            esi_val = m["target_esi"]

            proxies = compute_proxy_indices(
                smi=smi_unified,
                esi=esi_val,
                cal_cover=cal_feats["cal_cover"],
                slope_pct=slope_pct,
                sand_pct=sand_pct,
                precip_mm=m["precip_mm"],
                max_daily_precip_mm=m["max_daily_precip_mm"],
            )
            obs_ndvi = appeears_ndvi.get((pid, m["year_month"]), np.nan)

            row_dict = {
                "point_id": pid,
                "category": srow["category"],
                "lat": srow["lat"],
                "lon": srow["lon"],
                "merra2_cell_id": srow["merra2_cell_id"],
                "era5_cell_id": srow["era5_cell_id"],
                "spatial_block_id": srow["spatial_block_id"],
                "year_month": m["year_month"],
                "year": yr,
                "cal_month": cal_m,
                # Static soil + terrain features
                **{k: srow[k] for k in STATIC_SOIL_TERRAIN_FEATURES},
                "theta_wp": srow["theta_wp"],
                "theta_fc": srow["theta_fc"],
                # Crop features
                "crop_code": crop_code,
                "prev_crop_code": prev_crop_code,
                "is_corn": int(crop_code == 1),
                "is_soy": int(crop_code == 5),
                "is_forage": int(crop_code in FORAGE_CROP_CODES),
                "corn_soy_rotation": int(srow["corn_soy_rotation"]),
                "continuous_corn": int(srow["continuous_corn"]),
                "cropland_frac_3x3": srow["cropland_frac_2023"] if yr <= 2023 else srow["cropland_frac_2024"],
                **cal_feats,
                # Weather features
                **{k: m[k] for k in WEATHER_FEATURES},
                "et0_fao_mm": m["et0_fao_mm"],
                # Physics water-balance backbone features
                "wb_smi": wb["wb_smi"],
                "wb_esi": wb["wb_esi"],
                "wb_et_mm": wb["wb_et_mm"],
                "wb_runoff_mm": wb["wb_runoff_mm"],
                "wb_demand_mm": wb["wb_demand_mm"],
                "prev_wb_smi": prev_wb_smi,
                # Lagged state features (strictly t-1)
                "prev_smi_era5": prev_smi_era5,
                "prev_smi_merra2": prev_smi_merra2,
                "prev_smi": prev_smi_era5 if not np.isnan(prev_smi_era5) else prev_smi_merra2,
                "prev_esi": prev_esi,
                # Targets (levels, deltas, and water-balance residuals)
                "has_era5": int(has_era5),
                "theta_root_era5": m["theta_root_era5"],
                "target_smi_era5": smi_era5,
                "target_smi_merra2": smi_merra2,
                "target_smi": smi_unified,
                "target_esi": esi_val,
                "target_evland_mm": m["target_evland_mm"],
                "delta_smi_era5": round(smi_era5 - prev_smi_era5, 4) if (has_era5 and not np.isnan(prev_smi_era5)) else np.nan,
                "delta_smi_merra2": round(smi_merra2 - prev_smi_merra2, 4) if not np.isnan(prev_smi_merra2) else np.nan,
                "delta_smi": round(smi_unified - (prev_smi_era5 if has_era5 else prev_smi_merra2), 4)
                if not np.isnan(prev_smi_era5 if has_era5 else prev_smi_merra2) else np.nan,
                "delta_esi": round(esi_val - prev_esi, 4) if not np.isnan(prev_esi) else np.nan,
                "resid_smi_era5": round(smi_era5 - wb["wb_smi"], 4) if has_era5 else np.nan,
                "resid_smi_merra2": round(smi_merra2 - wb["wb_smi"], 4),
                "resid_esi": round(esi_val - wb["wb_esi"], 4),
                # Documented proxy indices + AppEEARS NDVI hook
                "proxy_bare": proxies["bare"],
                "proxy_ndvi": proxies["ndvi"],
                "proxy_vigor": proxies["vigor"],
                "observed_ndvi": obs_ndvi,
                "is_proxy_ndvi_bare": int(np.isnan(obs_ndvi)),
            }
            monthly_rows.append(row_dict)

            prev_smi_era5 = smi_era5
            prev_smi_merra2 = smi_merra2
            prev_esi = esi_val
            prev_wb_smi = wb["wb_smi"]

    monthly_df = pd.DataFrame(monthly_rows)
    monthly_csv = PROCESSED_DIR / "training_monthly.csv"
    monthly_df.to_csv(monthly_csv, index=False)
    print(f"[Phase 1] Wrote {len(monthly_df)} monthly rows ({monthly_df['point_id'].nunique()} cropland points x 24 months) to {monthly_csv}")

    drop_counts = Counter(static_df["drop_reason"])
    summary = {
        "seed": SEED,
        "total_points": int(len(static_df)),
        "kept_cropland_points": int(static_df["keep_for_training"].sum()),
        "dropped_points": int((~static_df["keep_for_training"]).sum()),
        "drop_reasons": dict(drop_counts),
        "kept_with_era5_open_meteo": int(monthly_df.groupby("point_id")["has_era5"].max().sum()),
        "distinct_era5_cells_kept": int(monthly_df.loc[monthly_df["has_era5"] == 1, "era5_cell_id"].nunique()),
        "distinct_merra2_cells_kept": int(monthly_df["merra2_cell_id"].nunique()),
        "distinct_spatial_blocks_kept": int(monthly_df["spatial_block_id"].nunique()),
        "monthly_rows_total": int(len(monthly_df)),
        "monthly_rows_with_era5": int((monthly_df["has_era5"] == 1).sum()),
        "columns": {
            "static_soil_terrain": STATIC_SOIL_TERRAIN_FEATURES,
            "crop": CROP_FEATURES,
            "weather": WEATHER_FEATURES,
            "targets": [
                "target_smi_era5", "target_smi_merra2", "target_smi", "target_esi",
                "delta_smi_era5", "delta_smi_merra2", "delta_smi", "delta_esi",
                "resid_smi_era5", "resid_smi_merra2", "resid_esi",
            ],
            "proxies": ["proxy_bare", "proxy_ndvi", "proxy_vigor", "is_proxy_ndvi_bare"],
        },
        "ssurgo_imputed_medians": medians,
    }
    schema_path = PROCESSED_DIR / "schema.json"
    schema_path.write_text(json.dumps(summary, indent=2))
    print(f"[Phase 1] Summary: {json.dumps({k: v for k, v in summary.items() if k != 'columns'}, indent=2)}")
    return summary


if __name__ == "__main__":
    build_dataset()

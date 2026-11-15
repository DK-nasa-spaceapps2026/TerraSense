"""Phase 5 Lazy-Loaded Singleton Predictor & Vectorized Zone Inference (`backend/ml/predict.py`).

Provides:
  - Engine mode resolution (`TERRASENSE_ENGINE=ml|heuristic|auto`)
  - Artifact + manifest schema hash validation with automatic fallback on any error
  - Deterministic IDW + sub-field micro-topography static feature assignment per zone
    (cached by `field_id`)
  - Batched/vectorized monthly state-transition inference (1 batched LightGBM call
    across all N_z zones per month step)
  - Conformal P10/P50/P90 uncertainty interval attachment for forecast months
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from ml.config import (
    CROP_FEATURES,
    GDD_BASE_C,
    GDD_CAP_C,
    HEAT_STRESS_THRESH_C,
    MODELS_DIR,
    STATIC_SOIL_TERRAIN_FEATURES,
    WEATHER_FEATURES,
)
from ml.crop_plans import AGRONOMIC_PLANS
from ml.train import TRANSITION_FEATURE_COLS

logger = logging.getLogger("terrasense.ml")

_PREDICTOR_SINGLETON: Optional["TerraSensePredictor"] = None
_STATIC_ZONE_CACHE: Dict[str, pd.DataFrame] = {}


class TerraSensePredictor:
    """Validated lazy-loaded wrapper around `transition_bundle.joblib` and `manifest.json`."""

    def __init__(self, bundle_path: Path = MODELS_DIR / "transition_bundle.joblib",
                 manifest_path: Path = MODELS_DIR / "manifest.json") -> None:
        if not bundle_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"Missing ML artifacts in {MODELS_DIR}")

        manifest = json.loads(manifest_path.read_text())
        bundle = joblib.load(bundle_path)

        expected_hash = hashlib.sha256(json.dumps(TRANSITION_FEATURE_COLS).encode()).hexdigest()[:12]
        if (
            bundle.get("schema_hash") != manifest.get("schema_hash")
            or bundle.get("schema_hash") != expected_hash
            or bundle.get("feature_cols") != TRANSITION_FEATURE_COLS
        ):
            raise ValueError(
                f"Feature schema mismatch: bundle={bundle.get('schema_hash')} "
                f"manifest={manifest.get('schema_hash')} expected={expected_hash}"
            )
        if "smi_model" not in bundle or "esi_model" not in bundle:
            raise ValueError("Model bundle missing smi_model or esi_model")

        self.bundle = bundle
        self.manifest = manifest
        self.model_version: str = str(manifest.get("model_version", "1.0.0-hybrid-lgbm"))
        self.smi_model = bundle["smi_model"]
        self.esi_model = bundle["esi_model"]
        self.conformal_smi: Dict[str, Dict[str, float]] = manifest.get("conformal_smi", {})
        self.conformal_esi: Dict[str, Dict[str, float]] = manifest.get("conformal_esi", {})
        self.ref_points: List[Dict[str, Any]] = bundle.get("reference_points", [])
        self._ref_lats = np.array([float(p["lat"]) for p in self.ref_points], dtype=np.float64)
        self._ref_lons = np.array([float(p["lon"]) for p in self.ref_points], dtype=np.float64)

    def interpolate_zone_static_df(
        self,
        field_id: str,
        zone_coords_lon_lat: List[Tuple[float, float]],
        grid_soil: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Assign static soil, terrain, and baseline crop features to each zone deterministically.

        Design (documented for Phase 5):
          - Anchors field-level static properties (`sg_*`, `elev_m`, `twi`, `ph_top`,
            `bd_top_gcm3`, `drainage_code`, `local_relief_*`) via k=4 Inverse-Distance
            Weighting (IDW) over the 195 preprocessed Iowa cropland reference points.
          - Maps zone-specific sub-field variation (`sand_top_pct`, `clay_top_pct`,
            `om_top_pct`, `slope_pct`, `awc_100cm_mm`) directly from `grid_soil` so
            every micro-zone in an N x N parcel preserves realistic spatial heterogeneity.
          - Cached by `field_id` so subsequent timeline calls have zero static-feature cost.
        """
        cache_key = f"{field_id}_{len(zone_coords_lon_lat)}"
        if cache_key in _STATIC_ZONE_CACHE:
            return _STATIC_ZONE_CACHE[cache_key]

        nz = len(zone_coords_lon_lat)
        rows: List[Dict[str, float]] = []
        for z in range(nz):
            lon, lat = zone_coords_lon_lat[z]
            d2 = (self._ref_lats - lat) ** 2 + (self._ref_lons - lon) ** 2
            k_idx = np.argsort(d2)[:4]
            weights = 1.0 / np.maximum(np.sqrt(d2[k_idx]), 1e-4)
            weights /= weights.sum()

            interp: Dict[str, float] = {}
            for col in STATIC_SOIL_TERRAIN_FEATURES:
                vals = np.array([float(self.ref_points[i].get(col, np.nan)) for i in k_idx], dtype=np.float64)
                valid = ~np.isnan(vals)
                if valid.any():
                    w_sub = weights[valid] / weights[valid].sum()
                    interp[col] = float(np.dot(w_sub, vals[valid]))
                else:
                    interp[col] = float(self.bundle.get("ssurgo_medians", {}).get(col, 0.0))

            # Couple directly to the zone's sub-field soil & slope arrays in `grid["soil"]`
            sand_pct = float(grid_soil["sand"][z]) * 100.0
            clay_pct = float(grid_soil["clay"][z]) * 100.0
            silt_pct = max(0.0, 100.0 - sand_pct - clay_pct)
            som_pct = float(grid_soil["som"][z])
            slope_pct = float(grid_soil["slope"][z])
            awc_mm = float(grid_soil["awc"][z]) * (1000.0 / 1200.0)  # scale 1.2m -> 1.0m

            interp["sand_top_pct"] = sand_pct
            interp["clay_top_pct"] = clay_pct
            interp["silt_top_pct"] = silt_pct
            interp["om_top_pct"] = som_pct
            interp["slope_pct"] = slope_pct
            interp["slope_mean_90m"] = slope_pct
            interp["awc_100cm_mm"] = awc_mm
            interp["awc_top_cmcm"] = awc_mm / 1000.0
            interp["awc_sub_cmcm"] = awc_mm / 1000.0
            rows.append(interp)

        df_static = pd.DataFrame(rows)
        _STATIC_ZONE_CACHE[cache_key] = df_static
        return df_static

    def simulate_vectorized(
        self,
        soil: Dict[str, np.ndarray],
        months: List[dict],
        plan_ids: List[str],
        cal_lookup: Dict[str, Dict[str, np.ndarray]],
        field_id: str = "default",
        zone_coords: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, np.ndarray]:
        """Vectorized 24-month rollout across all N_z zones (1 batched predict call per month)."""
        nz, T = len(soil["awc"]), len(months)
        if zone_coords is None:
            zone_coords = [(-93.60, 41.60)] * nz
        static_df = self.interpolate_zone_static_df(field_id, zone_coords, soil)

        names = [
            "smi", "esi", "bare", "ndvi", "vigor", "heat", "et", "demand", "runoff", "precip",
            "smi_p10", "smi_p90", "esi_p10", "esi_p90",
        ]
        out = {k: np.zeros((T, nz), dtype=np.float64) for k in names}

        awc_120 = soil["awc"]
        sm_wb = 0.65 * awc_120
        prev_wb_smi = np.full(nz, 0.65, dtype=np.float64)
        cur_smi = np.full(nz, 0.56, dtype=np.float64)
        cur_esi = np.full(nz, 0.62, dtype=np.float64)
        slope_n = np.clip(soil["slope"] / 8.0, 0.0, 1.0)
        prev_precip = float(months[0]["precip_mm"]) if months else 50.0

        # Pre-allocate batch matrix (nz x n_features) for fast vectorized updates
        batch_df = pd.DataFrame(index=range(nz), columns=TRANSITION_FEATURE_COLS, dtype=np.float64)
        for col in STATIC_SOIL_TERRAIN_FEATURES:
            batch_df[col] = static_df[col].to_numpy(dtype=np.float64)

        for t, c in enumerate(months):
            pid = plan_ids[t]
            cal = cal_lookup[pid]
            cal_m = int(c["cal_month"])
            cover_val = float(cal["cover"][cal_m - 1])
            kc_val = float(cal["kc"][cal_m - 1])
            cover = np.full(nz, cover_val, dtype=np.float64)

            P = float(c["precip_mm"])
            tmax = float(c["tmax"])
            tmin = float(c["tmin"])
            tmean = 0.5 * (tmax + tmin)
            rh = float(c["rh"])
            solar = float(c["solar"])
            days = float(c.get("days", 30))
            pet = float(c["pet_mm"])
            max_1d = float(c.get("max_daily_precip", P / 8.0))
            intensity = float(np.clip(max_1d / 40.0, 0.0, 1.0))

            # 1. Physics water-balance backbone step (monotone in P and PET)
            runoff_frac = np.clip(
                0.04 + 0.025 * soil["slope"] + 0.18 * (1.0 - cover) * (0.6 + intensity) + 0.15 * soil["clay"],
                0.0, 0.55,
            )
            runoff = P * runoff_frac
            infil = P - runoff
            demand = pet * kc_val
            avail = sm_wb + 0.5 * infil
            beta = np.clip(avail / (0.35 * awc_120), 0.05, 1.0)
            heat_scalar = float(np.clip((tmax - 30.0) / 8.0, 0.0, 1.0)) * (1.0 if cover_val > 0.4 else 0.0)
            et_wb = np.minimum(demand * beta, sm_wb + infil)
            sm_wb_new = np.clip(sm_wb + infil - et_wb, 0.0, awc_120)
            wb_smi = np.clip(0.5 * (sm_wb + sm_wb_new) / awc_120, 0.0, 1.0)
            wb_esi = np.where(demand > 1e-6, (et_wb / np.maximum(demand, 1e-6)) * (1.0 - 0.25 * heat_scalar), 1.0)
            wb_esi = np.clip(wb_esi, 0.0, 1.0)
            sm_wb = sm_wb_new

            # 2. Populate weather + crop features in batch_df (vectorized across all nz zones)
            gdd_day = max(0.0, 0.5 * (np.clip(tmax, GDD_BASE_C, GDD_CAP_C) + max(tmin, GDD_BASE_C)) - GDD_BASE_C)
            heat_days = float(np.clip(days * (tmax - (HEAT_STRESS_THRESH_C - 4.0)) / 10.0, 0.0, days))
            dry_spell = float(np.clip(12.0 * (60.0 / max(10.0, P)) ** 0.35, 0.0, days))
            es_tmax = 0.6108 * math.exp((17.27 * tmax) / (tmax + 237.3))
            es_tmin = 0.6108 * math.exp((17.27 * tmin) / (tmin + 237.3))
            vpd = max(0.0, 0.5 * (es_tmax + es_tmin) * (1.0 - rh / 100.0))
            api_mm = 0.7 * P + 0.3 * prev_precip
            prev_precip = P

            batch_df["precip_mm"] = P
            batch_df["tmax_c"] = tmax
            batch_df["tmin_c"] = tmin
            batch_df["tmean_c"] = tmean
            batch_df["diurnal_range_c"] = max(1.0, tmax - tmin)
            batch_df["rh_pct"] = rh
            batch_df["solar_mj"] = solar
            batch_df["gdd_month"] = gdd_day * days
            batch_df["heat_stress_days"] = heat_days
            batch_df["max_dry_spell_days"] = dry_spell
            batch_df["max_5day_precip_mm"] = max(max_1d, P * 0.45)
            batch_df["max_daily_precip_mm"] = max_1d
            batch_df["vpd_kpa"] = vpd
            batch_df["pet_mm"] = pet
            batch_df["water_balance_mm"] = P - pet
            batch_df["aridity_index"] = P / max(pet, 1.0)
            batch_df["api_antecedent_mm"] = api_mm
            batch_df["month_sin"] = math.sin(2.0 * math.pi * cal_m / 12.0)
            batch_df["month_cos"] = math.cos(2.0 * math.pi * cal_m / 12.0)

            agro_plan = AGRONOMIC_PLANS.get(pid)
            p_code = agro_plan.primary_crop_code if agro_plan else 1
            batch_df["is_corn"] = float(p_code == 1)
            batch_df["is_soy"] = float(p_code == 5)
            batch_df["is_forage"] = float(p_code == 36 or pid == "cover_legume")
            batch_df["corn_soy_rotation"] = float(pid in ("status_quo", "cover_legume", "corn_rye_cover"))
            batch_df["continuous_corn"] = 0.0
            batch_df["cropland_frac_3x3"] = 1.0
            batch_df["cal_cover"] = cover_val
            batch_df["cal_kc"] = kc_val

            # 3. Single batched LightGBM predict call for SMI across all nz zones
            batch_df["prev_state"] = cur_smi
            batch_df["wb_state"] = wb_smi
            batch_df["wb_delta"] = wb_smi - prev_wb_smi
            resid_smi = self.smi_model.predict(batch_df)
            smi_raw = np.clip(0.25 * cur_smi + 0.15 * wb_smi + resid_smi, 0.02, 0.98)
            # Blend 75% ML state transition + 25% zone-specific water-balance modulation
            # so crop-plan differences (Kc, cover) and sub-field AWC gradients remain sharp
            smi = np.clip(0.75 * smi_raw + 0.25 * wb_smi, 0.02, 0.98)

            # 4. Single batched LightGBM predict call for ESI across all nz zones
            batch_df["prev_state"] = cur_esi
            batch_df["wb_state"] = wb_esi
            batch_df["wb_delta"] = wb_esi - cur_esi
            resid_esi = self.esi_model.predict(batch_df)
            esi_raw = np.clip(0.25 * cur_esi + 0.25 * wb_esi + resid_esi, 0.05, 0.98)
            esi = np.clip(0.70 * esi_raw + 0.30 * wb_esi, 0.05, 0.98)

            cur_smi = smi
            cur_esi = esi
            prev_wb_smi = wb_smi

            # Conformal intervals (forecast months t >= 12 correspond to horizon h = t - 11)
            h_key = str(max(1, min(12, t - 11))) if t >= 12 else "1"
            q_smi = self.conformal_smi.get(h_key, {"q10": -0.08, "q90": 0.08})
            q_esi = self.conformal_esi.get(h_key, {"q10": -0.08, "q90": 0.08})
            smi_p10 = np.clip(smi + float(q_smi["q10"]), 0.0, 1.0)
            smi_p90 = np.clip(smi + float(q_smi["q90"]), 0.0, 1.0)
            esi_p10 = np.clip(esi + float(q_esi["q10"]), 0.0, 1.0)
            esi_p90 = np.clip(esi + float(q_esi["q90"]), 0.0, 1.0)

            # Derived ET and documented proxy indices (bare, ndvi, vigor)
            et_actual = np.minimum(demand * esi, sm_wb + infil)
            rain_i = min(P / 120.0, 1.0)
            bare = np.clip(
                (1.0 - cover) * 0.42 + slope_n * 0.24 + soil["sand"] * 0.30
                + rain_i * 0.10 * (1.0 - cover) + (1.0 - smi) * 0.10 - 0.03,
                0.0, 1.0,
            )
            ndvi = np.clip(0.12 + 0.72 * cover * (0.45 + 0.55 * esi), 0.05, 0.90)
            vigor = np.clip(
                (0.5 * (ndvi - 0.1) / 0.7 + 0.3 * esi + 0.2 * smi) * (0.4 + 0.6 * cover),
                0.0, 1.0,
            )

            out["smi"][t] = smi
            out["esi"][t] = esi
            out["bare"][t] = bare
            out["ndvi"][t] = ndvi
            out["vigor"][t] = vigor
            out["heat"][t] = heat_scalar * np.ones(nz)
            out["et"][t] = et_actual
            out["demand"][t] = demand * np.ones(nz)
            out["runoff"][t] = runoff
            out["precip"][t] = P * np.ones(nz)
            out["smi_p10"][t] = smi_p10
            out["smi_p90"][t] = smi_p90
            out["esi_p10"][t] = esi_p10
            out["esi_p90"][t] = esi_p90

        return out


def reset_predictor_cache() -> None:
    """Clear singleton predictor and static zone cache (used in tests)."""
    global _PREDICTOR_SINGLETON
    _PREDICTOR_SINGLETON = None
    _STATIC_ZONE_CACHE.clear()


def get_predictor(
    bundle_path: Path = MODELS_DIR / "transition_bundle.joblib",
    manifest_path: Path = MODELS_DIR / "manifest.json",
) -> Optional[TerraSensePredictor]:
    """Return cached TerraSensePredictor singleton, or None if validation fails."""
    global _PREDICTOR_SINGLETON
    if _PREDICTOR_SINGLETON is not None:
        return _PREDICTOR_SINGLETON
    try:
        _PREDICTOR_SINGLETON = TerraSensePredictor(bundle_path=bundle_path, manifest_path=manifest_path)
        return _PREDICTOR_SINGLETON
    except Exception as exc:
        logger.warning("[terrasense.ml] Failed to load/validate ML artifacts (%s); falling back to heuristic engine.", exc)
        return None


def resolve_engine_mode() -> str:
    """Resolve requested engine mode from `TERRASENSE_ENGINE` env var (`ml`, `heuristic`, `auto`)."""
    mode = os.getenv("TERRASENSE_ENGINE", "auto").strip().lower()
    if mode not in ("ml", "heuristic", "auto"):
        logger.warning("[terrasense.ml] Unknown TERRASENSE_ENGINE=%r; defaulting to 'auto'", mode)
        mode = "auto"
    if mode == "heuristic":
        return "heuristic"
    pred = get_predictor()
    return "ml" if pred is not None else "heuristic"

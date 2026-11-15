"""Unit tests for Phase 1 Data Preparation & Feature Engineering (`ml/features.py`, `ml/build_dataset.py`)."""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.config import PROCESSED_DIR, STATIC_SOIL_TERRAIN_FEATURES, WEATHER_FEATURES  # noqa: E402
from ml.features import (  # noqa: E402
    classify_drop_reason,
    compute_metric_terrain_stats,
    compute_proxy_indices,
    smi_to_volumetric_sm,
    soil_water_bounds,
    volumetric_sm_to_smi,
)


class TestPhase1Features(unittest.TestCase):
    def test_invertible_smi_transform(self) -> None:
        theta_wp, theta_fc = soil_water_bounds(clay_pct=28.0, sand_pct=28.0, awc_cmcm=0.19)
        self.assertGreater(theta_fc, theta_wp)
        for theta in (theta_wp + 0.02, 0.5 * (theta_wp + theta_fc), theta_fc - 0.02):
            smi = volumetric_sm_to_smi(theta, theta_wp, theta_fc)
            self.assertGreaterEqual(smi, 0.0)
            self.assertLessEqual(smi, 1.0)
            theta_back = smi_to_volumetric_sm(smi, theta_wp, theta_fc)
            self.assertAlmostEqual(theta, theta_back, places=6)

    def test_cdl_drop_reason_policy(self) -> None:
        self.assertEqual(classify_drop_reason(1, 5), (True, "kept_cropland"))
        self.assertEqual(classify_drop_reason(36, 176), (True, "kept_cropland"))
        self.assertEqual(classify_drop_reason(176, 176), (False, "permanent_pasture_176"))
        self.assertEqual(classify_drop_reason(122, 1), (False, "developed_urban"))
        self.assertEqual(classify_drop_reason(111, 111), (False, "water_or_wetland"))
        self.assertEqual(classify_drop_reason(-1, 1), (False, "out_of_cdl_bounds"))

    def test_metric_terrain_stats_on_known_plane(self) -> None:
        # Synthetic 15x15 metric patch sloping 3% northward (0.9 m rise per 30 m row northward)
        rows = np.arange(15, dtype=np.float32)[:, None]
        cols = np.zeros((1, 15), dtype=np.float32)
        patch = 300.0 - 0.9 * rows + cols  # row 0 (north) is higher -> south-facing aspect (pi rad, cos=-1, sin=0)
        stats = compute_metric_terrain_stats(patch, res_m=30.0)
        self.assertAlmostEqual(stats["slope_pct"], 3.0, places=2)
        self.assertAlmostEqual(stats["aspect_sin"], 0.0, places=2)
        self.assertAlmostEqual(stats["aspect_cos"], -1.0, places=2)
        self.assertAlmostEqual(stats["curvature"], 0.0, places=3)
        self.assertGreater(stats["twi"], 0.0)

    def test_proxy_indices_bounded(self) -> None:
        p = compute_proxy_indices(
            smi=0.55, esi=0.75, cal_cover=0.80, slope_pct=3.5, sand_pct=25.0,
            precip_mm=85.0, max_daily_precip_mm=22.0,
        )
        for k in ("bare", "ndvi", "vigor"):
            self.assertGreaterEqual(p[k], 0.0)
            self.assertLessEqual(p[k], 1.0)

    def test_processed_dataset_integrity(self) -> None:
        static_csv = PROCESSED_DIR / "points_static.csv"
        monthly_csv = PROCESSED_DIR / "training_monthly.csv"
        schema_json = PROCESSED_DIR / "schema.json"
        self.assertTrue(static_csv.exists(), "points_static.csv missing")
        self.assertTrue(monthly_csv.exists(), "training_monthly.csv missing")
        self.assertTrue(schema_json.exists(), "schema.json missing")

        sdf = pd.read_csv(static_csv)
        mdf = pd.read_csv(monthly_csv)
        schema = json.loads(schema_json.read_text())

        # All 305 points must be accounted for in points_static.csv (no silent drops)
        self.assertEqual(len(sdf), 305)
        self.assertEqual(int(sdf["keep_for_training"].sum()), schema["kept_cropland_points"])
        self.assertEqual(len(mdf), schema["kept_cropland_points"] * 24)

        # Required feature & target columns present and finite on core columns
        for col in STATIC_SOIL_TERRAIN_FEATURES:
            self.assertIn(col, mdf.columns)
        for col in WEATHER_FEATURES:
            self.assertIn(col, mdf.columns)
            self.assertFalse(mdf[col].isna().any(), f"Weather column {col} has NaNs")
        for tcol in ("target_smi", "target_smi_merra2", "target_esi", "wb_smi", "wb_esi"):
            self.assertFalse(mdf[tcol].isna().any(), f"Target/backbone {tcol} has NaNs")
            self.assertTrue(((mdf[tcol] >= 0.0) & (mdf[tcol] <= 1.0)).all(), f"{tcol} out of [0,1]")


if __name__ == "__main__":
    unittest.main()

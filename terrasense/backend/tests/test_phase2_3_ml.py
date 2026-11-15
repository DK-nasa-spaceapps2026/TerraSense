"""Unit tests for Phase 2 Model Formulation (`ml/train.py`) and Phase 3 Crop Plans (`ml/crop_plans.py`)."""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import zone_engine as ze  # noqa: E402
from ml.config import MODELS_DIR, PROCESSED_DIR  # noqa: E402
from ml.crop_plans import (  # noqa: E402
    AGRONOMIC_PLANS,
    PRESET_PLAN_IDS,
    evaluate_and_rank_plans,
    prune_candidate_plans,
)
from ml.train import (  # noqa: E402
    TRANSITION_FEATURE_COLS,
    predict_transition_step,
    recompute_weather_features_with_shifts,
)


class TestPhase2And3(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = joblib.load(MODELS_DIR / "transition_bundle.joblib")
        cls.manifest = json.loads((MODELS_DIR / "manifest.json").read_text())
        cls.df = pd.read_csv(PROCESSED_DIR / "training_monthly.csv")

    def test_artifacts_and_manifest_schema(self) -> None:
        self.assertEqual(self.bundle["schema_hash"], self.manifest["schema_hash"])
        self.assertEqual(self.bundle["feature_cols"], TRANSITION_FEATURE_COLS)
        self.assertIn("smi_model", self.bundle)
        self.assertIn("esi_model", self.bundle)
        self.assertEqual(len(self.manifest["conformal_smi"]), 12)
        self.assertEqual(len(self.manifest["conformal_esi"]), 12)

        # Verify Hybrid Transition Model beats all 3 baselines on out-of-fold 12m rollout RMSE
        smi_bench = self.manifest["cv_benchmarks"]["smi_12m_rollout"]
        esi_bench = self.manifest["cv_benchmarks"]["esi_12m_rollout"]
        for base_name in ("baseline_persistence", "baseline_climatology", "baseline_heuristic_wb"):
            self.assertLess(smi_bench["hybrid_transition_lgbm"]["rmse"], smi_bench[base_name]["rmse"])
            self.assertLess(esi_bench["hybrid_transition_lgbm"]["rmse"], esi_bench[base_name]["rmse"])

    def test_monotonicity_what_if_sweeps(self) -> None:
        """Sweep temp_shift in [-2, +4] C and precip_factor in [0.6, 1.4].

        Asserts finite, bounded [0, 1] outputs, exact reproducibility at (0, 1.0),
        and directional monotonicity:
          - increasing precip_factor -> non-decreasing SMI and ESI
          - increasing temp_shift    -> non-increasing SMI and ESI
        """
        sample_row = self.df[self.df["cal_month"] == 7].iloc[0].to_dict()
        smi_model = self.bundle["smi_model"]
        esi_model = self.bundle["esi_model"]

        def _eval_shift(t_shift: float, p_fac: float) -> tuple[float, float]:
            w_shifted = recompute_weather_features_with_shifts(sample_row, temp_shift=t_shift, precip_factor=p_fac)
            # Recompute water-balance physics step with shifted weather
            from ml.build_dataset import _water_balance_step
            wb = _water_balance_step(
                sm_prev_mm=float(sample_row["prev_smi"]) * float(sample_row["awc_100cm_mm"]),
                awc_mm=float(sample_row["awc_100cm_mm"]),
                slope_pct=float(sample_row["slope_pct"]),
                clay_frac=float(sample_row["clay_top_pct"]) / 100.0,
                cal_cover=float(sample_row["cal_cover"]),
                cal_kc=float(sample_row["cal_kc"]),
                precip_mm=w_shifted["precip_mm"],
                max_daily_precip_mm=w_shifted["max_daily_precip_mm"],
                pet_mm=w_shifted["pet_mm"],
                tmax_c=w_shifted["tmax_c"],
            )
            x_smi = pd.DataFrame([{
                **sample_row,
                **w_shifted,
                "prev_state": float(sample_row["prev_smi"]),
                "wb_state": wb["wb_smi"],
                "wb_delta": wb["wb_smi"] - float(sample_row["prev_wb_smi"]),
            }])
            x_esi = pd.DataFrame([{
                **sample_row,
                **w_shifted,
                "prev_state": float(sample_row["prev_esi"]),
                "wb_state": wb["wb_esi"],
                "wb_delta": wb["wb_esi"] - float(sample_row["prev_esi"]),
            }])
            smi = float(predict_transition_step(smi_model, x_smi, "smi")[0])
            esi = float(predict_transition_step(esi_model, x_esi, "esi")[0])
            self.assertTrue(math.isfinite(smi) and 0.0 <= smi <= 1.0)
            self.assertTrue(math.isfinite(esi) and 0.0 <= esi <= 1.0)
            return smi, esi

        # No-shift reproducibility check
        smi_0, esi_0 = _eval_shift(0.0, 1.0)
        self.assertEqual((smi_0, esi_0), _eval_shift(0.0, 1.0))

        # Precipitation sweep: [0.6, 0.8, 1.0, 1.2, 1.4] -> monotone non-decreasing
        p_sweeps = [_eval_shift(0.0, pf) for pf in (0.6, 0.8, 1.0, 1.2, 1.4)]
        for i in range(len(p_sweeps) - 1):
            self.assertLessEqual(p_sweeps[i][0], p_sweeps[i + 1][0] + 1e-9, "SMI violated precip monotonicity")
            self.assertLessEqual(p_sweeps[i][1], p_sweeps[i + 1][1] + 1e-9, "ESI violated precip monotonicity")

        # Temperature sweep: [-2, 0, +2, +4] C -> monotone non-increasing
        t_sweeps = [_eval_shift(ts, 1.0) for ts in (-2.0, 0.0, 2.0, 4.0)]
        for i in range(len(t_sweeps) - 1):
            self.assertGreaterEqual(t_sweeps[i][0] + 1e-9, t_sweeps[i + 1][0], "SMI violated temp monotonicity")
            self.assertGreaterEqual(t_sweeps[i][1] + 1e-9, t_sweeps[i + 1][1], "ESI violated temp monotonicity")

    def test_crop_plan_pruning_and_ensemble_scoring(self) -> None:
        # 1. Hard constraint pruning: after soybean (code 5), cover_legume (soy->clover) is pruned (soy-after-soy)
        feasible_soy, pruned_soy = prune_candidate_plans(prev_crop_code=5, mean_slope_pct=3.0, mean_drainage_code=4.0)
        self.assertIn("cover_legume", pruned_soy)
        self.assertIn("corn_rye_cover", [p.id for p in feasible_soy])

        # 2. Steep slope (8.0%) prunes status_quo (max_slope_pct=5.5)
        _, pruned_steep = prune_candidate_plans(prev_crop_code=1, mean_slope_pct=8.0, mean_drainage_code=4.0)
        self.assertIn("status_quo", pruned_steep)

        # 3. Ensemble ranking on a sample field
        geom = ze.parse_field(None)
        fid = ze.field_id_for(geom, 4)
        grid = ze.build_grid(geom, 4, fid)
        climate = ze.build_climate(geom, None, 0.0, 1.0, live=False)
        res = evaluate_and_rank_plans(
            grid=grid,
            climate=climate,
            prefs={"priority": "soil_restoration"},
            simulate_fn=ze.simulate,
            preference_weights_fn=ze.preference_weights,
        )
        self.assertEqual([s["id"] for s in res["scenarios"]], PRESET_PLAN_IDS)
        self.assertIn(res["recommended"], PRESET_PLAN_IDS)
        self.assertEqual(len(res["zone_recommendations"]), grid["n_zones"])
        self.assertIsInstance(res["ranking_robust"], bool)
        self.assertIn("dynamic_candidates", res)
        self.assertEqual(len(res["dynamic_candidates"]), len(AGRONOMIC_PLANS))
        for s in res["scenarios"]:
            self.assertIn("score_breakdown", s)
            self.assertIn("critical_window_stress_pct", s["metrics"])


if __name__ == "__main__":
    unittest.main()

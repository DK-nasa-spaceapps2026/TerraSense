"""Comprehensive test suite for Farmer Customization (`ml/constraints.py`, `ml/crop_plans.py`, `ml/predict.py`, `app.py`)."""
from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import app  # noqa: E402
from ml.config import TRAINING_SMI_CLIP_RANGE  # noqa: E402
from ml.predict import reset_predictor_cache  # noqa: E402

SNAPSHOT_PATH = Path(__file__).resolve().parent / "golden_baseline_snapshot.json"


class TestFarmerCustomization(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = json.loads(SNAPSHOT_PATH.read_text())

    def setUp(self) -> None:
        self.client = app.test_client()

    def test_01_golden_regression_exact_preservation(self) -> None:
        """Default request, each preset timeline, and climate what-if must match pre-change snapshot on all original keys."""
        r_def = self.client.post("/api/field/analyze", json={}).get_json()
        snap_def = self.snapshot["analyze_default"]

        for key, expected_val in snap_def.items():
            if key == "scenarios":
                self.assertEqual(len(r_def["scenarios"]), len(expected_val))
                for s_cur, s_exp in zip(r_def["scenarios"], expected_val):
                    for sk in ("id", "name", "crops", "description", "score", "recommended", "planting_window", "tradeoffs"):
                        self.assertEqual(s_cur[sk], s_exp[sk], f"Mismatch in scenario[{s_cur['id']}].{sk}")
                    for mk, mv in s_exp["metrics"].items():
                        self.assertEqual(s_cur["metrics"][mk], mv, f"Mismatch in scenario[{s_cur['id']}].metrics.{mk}")
                    for bk, bv in s_exp["score_breakdown"].items():
                        self.assertEqual(s_cur["score_breakdown"][bk], bv, f"Mismatch in scenario[{s_cur['id']}].score_breakdown.{bk}")
            elif key == "dynamic_candidates":
                self.assertEqual(len(r_def["dynamic_candidates"]), len(expected_val))
                for c_cur, c_exp in zip(r_def["dynamic_candidates"], expected_val):
                    for ck in ("id", "name", "crops", "planting_window", "feasible", "pruned_reason", "score",
                               "estimated_margin_usd_ha", "erosion_control_score"):
                        self.assertEqual(c_cur[ck], c_exp[ck], f"Mismatch in dynamic_candidates[{c_cur['id']}].{ck}")
            else:
                self.assertEqual(r_def[key], expected_val, f"Mismatch in top-level key '{key}'")

        # Check all 3 preset timelines match bit-for-bit
        fid = r_def["field_id"]
        for scen in ("status_quo", "cover_legume", "drought_cash"):
            tl_cur = self.client.get(f"/api/field/timeline?field_id={fid}&scenario={scen}").get_json()
            tl_exp = self.snapshot[f"timeline_{scen}"]
            self.assertEqual(tl_cur, tl_exp, f"Timeline mismatch for scenario={scen}")

        # Check climate-shift analyze matches pre-change snapshot on all original keys
        r_clim = self.client.post(
            "/api/field/analyze",
            json={
                "climate": {"temp_shift": 2.0, "precip_factor": 0.75},
                "preferences": {"priority": "water_saving"},
            },
        ).get_json()
        snap_clim = self.snapshot["analyze_climate_shift"]
        self.assertEqual(r_clim["recommended_scenario"], snap_clim["recommended_scenario"])
        self.assertEqual(r_clim["preference_weights"], snap_clim["preference_weights"])
        self.assertEqual(r_clim["summary"], snap_clim["summary"])
        for s_cur, s_exp in zip(r_clim["scenarios"], snap_clim["scenarios"]):
            self.assertEqual(s_cur["score"], s_exp["score"])
            for mk, mv in s_exp["metrics"].items():
                self.assertEqual(s_cur["metrics"][mk], mv)

    def test_02_validation_errors_and_warnings(self) -> None:
        """Bad types, out-of-range values, negative/zero weights, and unknown crops return 400; unknown keys warn."""
        bad_payloads = [
            ({"constraints": {"prev_crop": "bananas"}}, "prev_crop"),
            ({"constraints": {"prev_crop": 999}}, "prev_crop"),
            ({"constraints": {"continuous_corn_years": -1}}, "continuous_corn_years"),
            ({"constraints": {"continuous_corn_years": 25}}, "continuous_corn_years"),
            ({"constraints": {"continuous_corn_years": "two"}}, "continuous_corn_years"),
            ({"constraints": {"equipment": {"has_hay_baler": "yes"}}}, "equipment.has_hay_baler"),
            ({"constraints": {"excluded_crops": ["dragonfruit"]}}, "excluded_crops[0]"),
            ({"constraints": {"irrigation_cap_mm": -5.0}}, "irrigation_cap_mm"),
            ({"constraints": {"irrigation_cap_mm": 900.0}}, "irrigation_cap_mm"),
            ({"constraints": {"irrigation_cost_usd_per_mm": -0.1}}, "irrigation_cost_usd_per_mm"),
            ({"constraints": {"risk_aversion": "reckless"}}, "risk_aversion"),
            ({"constraints": {"risk_aversion": 1.5}}, "risk_aversion"),
            ({"constraints": {"weights": {"water_saving": -0.2, "income": 0.6, "soil_restoration": 0.6}}}, "weights.water_saving"),
            ({"constraints": {"weights": {"water_saving": 0.0, "income": 0.0, "soil_restoration": 0.0}}}, "weights"),
            ({"constraints": {"prices": {"corn_usd_bu": 99.0}}}, "prices.corn_usd_bu"),
            ({"constraints": {"prices": {"nitrogen_price_usd_kg": -1.0}}}, "prices.nitrogen_price_usd_kg"),
            ({"constraints": {"management_mode": "diagonal"}}, "management_mode"),
        ]
        for payload, expected_field in bad_payloads:
            resp = self.client.post("/api/field/analyze", json=payload)
            self.assertEqual(resp.status_code, 400, f"Expected 400 for {payload}")
            err = resp.get_json()
            self.assertEqual(err.get("field"), expected_field, f"Expected field={expected_field}, got {err}")

        # Unknown extra keys produce a warning, not a crash; constraints + preferences both sent produces a warning
        r_warn = self.client.post(
            "/api/field/analyze",
            json={
                "preferences": {"priority": "water_saving"},
                "constraints": {
                    "priority": "income",
                    "unknown_top_key": 123,
                    "equipment": {"has_laser_weeder": True},
                    "weights": {"income": 3.0, "water_saving": 1.0, "soil_restoration": 0.0, "extra_w": 5.0},
                    "prices": {"corn_usd_bu": 5.00, "carbon_credit_usd": 20.0},
                },
            },
        )
        self.assertEqual(r_warn.status_code, 200)
        d_warn = r_warn.get_json()
        self.assertGreaterEqual(len(d_warn["warnings"]), 5)
        self.assertAlmostEqual(d_warn["applied_constraints"]["weights"]["income"], 0.75, places=3)
        self.assertAlmostEqual(d_warn["applied_constraints"]["weights"]["water_saving"], 0.25, places=3)

    def test_03_hard_constraint_pruning_and_overconstrained_fallback(self) -> None:
        """Test rotation history, equipment flags, custom-hire baling cost, crop exclusions, and over-constrained fallback."""
        # 1. Previous crop = soybean -> prunes cover_legume (soy-after-soy), enables corn_rye_cover
        r_soy = self.client.post(
            "/api/field/analyze",
            json={"constraints": {"prev_crop": "soybean"}},
        ).get_json()
        pruned_ids_soy = {p["id"]: p["reason"] for p in r_soy["pruned_plans"]}
        self.assertIn("cover_legume", pruned_ids_soy)
        self.assertIn("Soy-after-soy", pruned_ids_soy["cover_legume"])
        cand_map_soy = {c["id"]: c for c in r_soy["dynamic_candidates"]}
        self.assertTrue(cand_map_soy["corn_rye_cover"]["feasible"])
        self.assertNotEqual(r_soy["recommended_scenario"], "cover_legume")

        # 2. Continuous corn >= 2 -> prunes status_quo (corn-on-corn limit)
        r_cc = self.client.post(
            "/api/field/analyze",
            json={"constraints": {"prev_crop": "corn", "continuous_corn_years": 2}},
        ).get_json()
        pruned_ids_cc = {p["id"]: p["reason"] for p in r_cc["pruned_plans"]}
        self.assertIn("status_quo", pruned_ids_cc)
        self.assertIn("continuous-corn", pruned_ids_cc["status_quo"])
        self.assertNotEqual(r_cc["recommended_scenario"], "status_quo")

        # 3. Equipment: has_hay_baler=False -> keeps alfalfa_perennial with custom-hire cost ($115/ha) by default
        r_nobaler = self.client.post(
            "/api/field/analyze",
            json={"constraints": {"equipment": {"has_hay_baler": False}}},
        ).get_json()
        alf_cand = next(c for c in r_nobaler["dynamic_candidates"] if c["id"] == "alfalfa_perennial")
        self.assertTrue(alf_cand["feasible"])
        self.assertEqual(alf_cand["custom_hire_cost_usd_ha"], 115.0)
        self.assertIn("custom_hire_note", alf_cand)

        # And when allow_custom_hire=False, has_hay_baler=False prunes alfalfa_perennial
        r_nobaler_strict = self.client.post(
            "/api/field/analyze",
            json={"constraints": {"equipment": {"has_hay_baler": False}, "allow_custom_hire": False}},
        ).get_json()
        pruned_strict = {p["id"]: p["reason"] for p in r_nobaler_strict["pruned_plans"]}
        self.assertIn("alfalfa_perennial", pruned_strict)

        # 4. Excluded crops: sorghum -> prunes drought_cash
        r_excl = self.client.post(
            "/api/field/analyze",
            json={"constraints": {"excluded_crops": ["sorghum"]}},
        ).get_json()
        pruned_excl = {p["id"]: p["reason"] for p in r_excl["pruned_plans"]}
        self.assertIn("drought_cash", pruned_excl)
        self.assertIn("sorghum", pruned_excl["drought_cash"])

        # 5. Over-constrained request (exclude all crops) -> never returns empty or 500, flags constraint_violation=True
        r_over = self.client.post(
            "/api/field/analyze",
            json={
                "constraints": {
                    "excluded_crops": ["corn", "soybean", "sorghum", "alfalfa"],
                    "equipment": {"has_no_till_drill": False, "has_cover_drill": False, "has_hay_baler": False},
                    "allow_custom_hire": False,
                }
            },
        )
        self.assertEqual(r_over.status_code, 200)
        d_over = r_over.get_json()
        self.assertTrue(d_over["constraint_violation"])
        self.assertEqual(len(d_over["scenarios"]), 3)
        self.assertIn(d_over["recommended_scenario"], {"status_quo", "cover_legume", "drought_cash"})
        self.assertTrue(any(c["constraint_violation"] for c in d_over["dynamic_candidates"]))

    def test_04_scoring_risk_aversion_prices_and_simplicity(self) -> None:
        """Risk aversion, nitrogen price overrides, and simplicity weights shift scores and margins in expected directions."""
        # 1. Risk aversion: conservative (0.50) vs aggressive (0.05)
        r_cons = self.client.post("/api/field/analyze", json={"constraints": {"risk_aversion": "conservative"}}).get_json()
        r_aggr = self.client.post("/api/field/analyze", json={"constraints": {"risk_aversion": "aggressive"}}).get_json()
        for s_c, s_a in zip(r_cons["scenarios"], r_aggr["scenarios"]):
            self.assertGreaterEqual(
                s_c["score_breakdown"]["downside_risk_penalty"],
                s_a["score_breakdown"]["downside_risk_penalty"],
            )
            self.assertLessEqual(s_c["score"], s_a["score"] + 1e-6)

        # 2. Nitrogen price override: raising N price increases cover_legume (75 kg N/ha) and alfalfa (140 kg N/ha) margins
        r_low_n = self.client.post(
            "/api/field/analyze", json={"constraints": {"prices": {"nitrogen_price_usd_kg": 0.50}}}
        ).get_json()
        r_high_n = self.client.post(
            "/api/field/analyze", json={"constraints": {"prices": {"nitrogen_price_usd_kg": 3.50}}}
        ).get_json()
        sq_low = next(s for s in r_low_n["scenarios"] if s["id"] == "status_quo")["metrics"]["estimated_margin_usd_ha"]
        sq_high = next(s for s in r_high_n["scenarios"] if s["id"] == "status_quo")["metrics"]["estimated_margin_usd_ha"]
        cl_low = next(s for s in r_low_n["scenarios"] if s["id"] == "cover_legume")["metrics"]["estimated_margin_usd_ha"]
        cl_high = next(s for s in r_high_n["scenarios"] if s["id"] == "cover_legume")["metrics"]["estimated_margin_usd_ha"]
        self.assertEqual(sq_low, sq_high, "Status quo (0 N fixation) margin should not change with N price")
        self.assertGreater(cl_high, cl_low + 150.0, "Cover legume (75 kg N/ha) margin must increase when N price rises")

        # 3. Simplicity weight: prioritizing simplicity boosts 0-extra-pass plans (status_quo, drought_cash) over cover_legume (2 passes)
        r_simp = self.client.post(
            "/api/field/analyze",
            json={"constraints": {"weights": {"simplicity": 0.85, "income": 0.15, "water_saving": 0.0, "soil_restoration": 0.0}}},
        ).get_json()
        self.assertEqual(r_simp["recommended_scenario"], "status_quo")
        self.assertTrue(r_simp["ranking_sensitivity"]["top_plan_changed"])
        self.assertIn("weights", r_simp["ranking_sensitivity"]["decisive_settings"])

    def test_05_supplemental_irrigation_monotonicity_and_extrapolation(self) -> None:
        """Increasing irrigation_cap_mm monotonically reduces water stress, deducts pumping cost, and flags extrapolated=True."""
        caps = [0.0, 60.0, 150.0]
        results = []
        for cap in caps:
            r = self.client.post(
                "/api/field/analyze",
                json={
                    "climate": {"temp_shift": 2.5, "precip_factor": 0.65},
                    "constraints": {"irrigation_cap_mm": cap, "irrigation_cost_usd_per_mm": 0.50},
                },
            ).get_json()
            results.append(r)

        # Cap = 0 must have extrapolated=False and 0 mm applied
        self.assertFalse(results[0]["extrapolated"])
        sq_0 = next(s for s in results[0]["scenarios"] if s["id"] == "status_quo")
        sq_60 = next(s for s in results[1]["scenarios"] if s["id"] == "status_quo")
        sq_150 = next(s for s in results[2]["scenarios"] if s["id"] == "status_quo")

        self.assertEqual(sq_0["metrics"]["irrigation_applied_mm"], 0.0)
        self.assertGreater(sq_60["metrics"]["irrigation_applied_mm"], 0.0)
        self.assertGreaterEqual(sq_150["metrics"]["irrigation_applied_mm"], sq_60["metrics"]["irrigation_applied_mm"])

        # Monotone reduction in water stress & critical-window stress
        self.assertLess(
            sq_60["score_breakdown"]["water_stress_pct"],
            sq_0["score_breakdown"]["water_stress_pct"],
        )
        self.assertLessEqual(
            sq_150["score_breakdown"]["water_stress_pct"],
            sq_60["score_breakdown"]["water_stress_pct"],
        )
        self.assertLess(
            sq_150["metrics"]["critical_window_stress_pct"],
            sq_0["metrics"]["critical_window_stress_pct"],
        )

        # Extrapolation flag and note set when irrigation > 0
        self.assertTrue(results[1]["extrapolated"])
        self.assertIsNotNone(results[1]["extrapolation_note"])
        self.assertGreater(sq_60["score_breakdown"]["irrigation_cost_usd_ha"], 0.0)

        # Check timeline with irrigation_cap_mm > 0 has clipped SMI in training range [0.02, 0.94]
        fid = results[2]["field_id"]
        tl_ir = self.client.get(f"/api/field/timeline?field_id={fid}&scenario=status_quo").get_json()
        self.assertTrue(tl_ir["extrapolated"])
        self.assertGreater(tl_ir["irrigation_applied_mm"], 0.0)
        for m in tl_ir["months"][12:]:
            for cell in m["cells"]:
                self.assertGreaterEqual(cell["smi"], TRAINING_SMI_CLIP_RANGE[0])
                self.assertLessEqual(cell["smi"], TRAINING_SMI_CLIP_RANGE[1] + 1e-6)

    def test_06_management_modes_and_variable_rate_mismatch(self) -> None:
        """Test uniform vs zone_variable mode and fallback warning when variable_rate=False."""
        r_uni = self.client.post(
            "/api/field/analyze",
            json={"constraints": {"management_mode": "uniform"}},
        ).get_json()
        self.assertEqual(r_uni["management_mode"], "uniform")
        zone_recs_uni = {
            f["properties"]["recommended_scenario"]
            for f in r_uni["field_geojson"]["features"]
            if f["properties"]["kind"] == "zone"
        }
        self.assertEqual(zone_recs_uni, {r_uni["recommended_scenario"]})

        # Requesting zone_variable when equipment.variable_rate=False warns and returns uniform
        r_mismatch = self.client.post(
            "/api/field/analyze",
            json={
                "constraints": {
                    "management_mode": "zone_variable",
                    "equipment": {"variable_rate": False},
                }
            },
        ).get_json()
        self.assertEqual(r_mismatch["management_mode"], "uniform")
        self.assertTrue(any("variable_rate" in w for w in r_mismatch["warnings"]))
        zone_recs_mm = {
            f["properties"]["recommended_scenario"]
            for f in r_mismatch["field_geojson"]["features"]
            if f["properties"]["kind"] == "zone"
        }
        self.assertEqual(zone_recs_mm, {r_mismatch["recommended_scenario"]})

    def test_07_options_endpoint_and_latency(self) -> None:
        """GET /api/field/options returns full metadata schema; measure customized analyze latency."""
        r_opt = self.client.get("/api/field/options")
        self.assertEqual(r_opt.status_code, 200)
        opt = r_opt.get_json()
        for key in ("crops", "prev_crop", "continuous_corn_years", "equipment", "excluded_crops",
                    "irrigation", "risk_aversion", "weights", "prices", "management_mode"):
            self.assertIn(key, opt)

        # Warm up and measure latency of POST /api/field/analyze
        reset_predictor_cache()
        self.client.post("/api/field/analyze", json={})
        t0 = time.perf_counter()
        for _ in range(5):
            self.client.post(
                "/api/field/analyze",
                json={
                    "constraints": {
                        "prev_crop": "soybean",
                        "risk_aversion": "conservative",
                        "prices": {"corn_usd_bu": 5.20, "nitrogen_price_usd_kg": 1.45},
                        "weights": {"income": 0.4, "soil_restoration": 0.4, "water_saving": 0.1, "simplicity": 0.1},
                    }
                },
            )
        dt_ms = (time.perf_counter() - t0) * 1000.0 / 5.0
        print(f"\n[Customization Benchmark] POST /api/field/analyze latency: {dt_ms:.2f} ms (baseline: {self.snapshot['baseline_latency_ms']:.2f} ms)")


if __name__ == "__main__":
    unittest.main()

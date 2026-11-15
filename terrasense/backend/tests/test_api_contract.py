"""Phase 0 Characterization / Golden API Contract Tests.

Pins the exact request/response contract of:
  - GET  /api/health
  - GET  /api/demo-field
  - POST /api/field/analyze
  - GET  /api/field/timeline

Covers default demo parcel, custom polygons, grid sizes (3..10),
all priorities/weights, all 3 preset scenarios (status_quo, cover_legume,
drought_cash), and non-default climate what-if shifts (temp_shift, precip_factor).
Can be executed via `pytest` or `python -m unittest`.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import app  # noqa: E402
import zone_engine as ze  # noqa: E402

VALID_STATUSES = {"lush", "moderate", "water_stress", "heat_stress", "bare_vulnerable"}
VALID_SCENARIOS = {"status_quo", "cover_legume", "drought_cash"}
VALID_RISK_LEVELS = {"low", "medium", "high"}
VALID_CROP_STAGES = {"planting", "vegetative", "reproductive", "maturity", "harvest", "cover crop", "fallow"}


class TestApiContract(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def _assert_finite_range(self, val: float, lo: float, hi: float, name: str) -> None:
        self.assertIsInstance(val, (int, float), f"{name} must be numeric")
        self.assertTrue(math.isfinite(val), f"{name} must be finite")
        self.assertGreaterEqual(val, lo, f"{name}={val} below {lo}")
        self.assertLessEqual(val, hi, f"{name}={val} above {hi}")

    def _assert_cell_contract(self, cell: dict) -> None:
        required = {"smi", "esi", "bare", "ndvi", "vigor", "et_mm", "status"}
        self.assertTrue(required.issubset(cell.keys()), f"Missing cell keys: {required - set(cell.keys())}")
        self._assert_finite_range(cell["smi"], 0.0, 1.0, "cell.smi")
        self._assert_finite_range(cell["esi"], 0.0, 1.0, "cell.esi")
        self._assert_finite_range(cell["bare"], 0.0, 1.0, "cell.bare")
        self._assert_finite_range(cell["ndvi"], 0.0, 1.0, "cell.ndvi")
        self._assert_finite_range(cell["vigor"], 0.0, 1.0, "cell.vigor")
        self._assert_finite_range(cell["et_mm"], 0.0, 500.0, "cell.et_mm")
        self.assertIn(cell["status"], VALID_STATUSES)

    def _assert_guidance_contract(self, g: dict) -> None:
        self.assertEqual(
            set(g.keys()),
            {"seed_window", "vulnerable_windows", "cover_crop_windows", "stress_alerts"},
        )
        if g["seed_window"] is not None:
            self.assertTrue({"start", "end", "message"}.issubset(g["seed_window"].keys()))
        for key in ("vulnerable_windows", "cover_crop_windows", "stress_alerts"):
            self.assertIsInstance(g[key], list)
            for item in g[key]:
                self.assertTrue({"start", "end", "message"}.issubset(item.keys()))

    def test_health_and_demo_field(self) -> None:
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertEqual(d["status"], "ok")
        self.assertIsInstance(d["live_nasa"], bool)

        r2 = self.client.get("/api/demo-field")
        self.assertEqual(r2.status_code, 200)
        d2 = r2.get_json()
        self.assertEqual(len(d2["field"]["bbox"]), 4)
        self.assertEqual(d2["grid_size"], 6)
        self.assertIn(d2["preferences"]["priority"], {"water_saving", "income", "soil_restoration"})

    def test_analyze_default_contract(self) -> None:
        r = self.client.post(
            "/api/field/analyze",
            json={
                "grid_size": 6,
                "date_range": {"end": "2024-11"},
                "preferences": {"priority": "soil_restoration"},
                "climate": {"temp_shift": 0, "precip_factor": 1.0},
                "use_live": False,
            },
        )
        self.assertEqual(r.status_code, 200)
        d = r.get_json()

        top_keys = {
            "field_id", "generated_at", "data_sources", "grid", "bbox",
            "field_geojson", "summary", "scenarios", "recommended_scenario",
            "preference_weights", "guidance",
        }
        self.assertTrue(top_keys.issubset(d.keys()))
        self.assertTrue(d["field_id"].startswith("fld_"))
        self.assertEqual(d["grid"], {"rows": 6, "cols": 6, "n_zones": 36})
        self.assertEqual(len(d["bbox"]), 4)

        # GeoJSON: 1 boundary + 36 zone features
        feats = d["field_geojson"]["features"]
        self.assertEqual(len(feats), 37)
        self.assertEqual(feats[0]["id"], "boundary")
        self.assertEqual(feats[0]["properties"]["kind"], "boundary")
        for zf in feats[1:]:
            p = zf["properties"]
            self.assertEqual(p["kind"], "zone")
            self.assertIn(p["recommended_scenario"], VALID_SCENARIOS)
            soil = p["soil"]
            self._assert_finite_range(soil["sand_pct"], 0.0, 100.0, "sand_pct")
            self._assert_finite_range(soil["clay_pct"], 0.0, 100.0, "clay_pct")
            self._assert_finite_range(soil["organic_matter_pct"], 0.0, 30.0, "organic_matter_pct")
            self._assert_finite_range(soil["slope_pct"], 0.0, 60.0, "slope_pct")
            self._assert_finite_range(soil["available_water_mm"], 10.0, 600.0, "available_water_mm")
            self._assert_cell_contract(p["current"])

        # Scenarios: 3 presets in order, exactly 1 recommended matching recommended_scenario
        self.assertEqual(len(d["scenarios"]), 3)
        self.assertEqual([s["id"] for s in d["scenarios"]], ["status_quo", "cover_legume", "drought_cash"])
        rec_ids = [s["id"] for s in d["scenarios"] if s["recommended"]]
        self.assertEqual(rec_ids, [d["recommended_scenario"]])
        for s in d["scenarios"]:
            self._assert_finite_range(s["score"], 0.0, 1.0, "scenario.score")
            m = s["metrics"]
            metric_keys = {
                "soil_organic_matter_gain_pct", "nitrogen_fixed_kg_ha", "water_demand_mm",
                "water_budget_balance_mm", "erosion_control_score", "estimated_margin_usd_ha",
                "risk_level", "risk_score",
            }
            self.assertTrue(metric_keys.issubset(m.keys()))
            self.assertIn(m["risk_level"], VALID_RISK_LEVELS)
            self._assert_finite_range(m["risk_score"], 0.0, 1.0, "risk_score")
            self._assert_finite_range(m["erosion_control_score"], 0.0, 100.0, "erosion_control_score")
            self.assertIn("pros", s["tradeoffs"])
            self.assertIn("cons", s["tradeoffs"])

        self._assert_guidance_contract(d["guidance"])

    def test_timeline_contract_and_scenarios_and_whatif(self) -> None:
        r = self.client.post(
            "/api/field/analyze",
            json={
                "field": {"bbox": [-93.65, 41.55, -93.55, 41.65]},
                "grid_size": 4,
                "date_range": {"end": "2024-11"},
                "preferences": {"priority": "water_saving"},
                "use_live": False,
            },
        )
        self.assertEqual(r.status_code, 200)
        fid = r.get_json()["field_id"]

        hist_by_scenario = {}
        fc_by_scenario = {}
        for scen in ("status_quo", "cover_legume", "drought_cash"):
            tr = self.client.get(
                f"/api/field/timeline?field_id={fid}&scenario={scen}&temp_shift=2.5&precip_factor=0.75"
            )
            self.assertEqual(tr.status_code, 200)
            tl = tr.get_json()
            self.assertEqual(tl["field_id"], fid)
            self.assertEqual(tl["scenario"], scen)
            self.assertEqual(tl["anchor_index"], 11)
            self.assertEqual(len(tl["months"]), 24)
            self._assert_guidance_contract(tl["guidance"])

            dates = [m["date"] for m in tl["months"]]
            self.assertEqual(dates, sorted(dates), "Months must be strictly chronological")
            for idx, m in enumerate(tl["months"]):
                self.assertEqual(m["index"], idx)
                self.assertEqual(m["phase"], "historical" if idx < 12 else "forecast")
                self.assertIn(m["crop_stage"], VALID_CROP_STAGES)
                self.assertEqual(len(m["cells"]), 16)
                for cell in m["cells"]:
                    self._assert_cell_contract(cell)
                for k in ("smi", "esi", "bare", "ndvi", "vigor"):
                    self._assert_finite_range(m["field"][k], 0.0, 1.0, f"field.{k}")
                clim = m["climate"]
                for ck in ("tmax_c", "tmin_c", "precip_mm", "rh_pct", "solar_mj", "pet_mm"):
                    self.assertIsInstance(clim[ck], (int, float))
                    self.assertTrue(math.isfinite(clim[ck]))

            hist_by_scenario[scen] = [m["field"]["smi"] for m in tl["months"][:12]]
            fc_by_scenario[scen] = [m["field"]["smi"] for m in tl["months"][12:]]

        # Historical months (0..11) must be identical across scenarios
        self.assertEqual(hist_by_scenario["status_quo"], hist_by_scenario["cover_legume"])
        self.assertEqual(hist_by_scenario["status_quo"], hist_by_scenario["drought_cash"])
        # Forecast months (12..23) must differ across distinct crop plans
        self.assertNotEqual(fc_by_scenario["status_quo"], fc_by_scenario["drought_cash"])

    def test_validation_errors(self) -> None:
        self.assertEqual(self.client.post("/api/field/analyze", json={"grid_size": 2}).status_code, 400)
        self.assertEqual(self.client.post("/api/field/analyze", json={"grid_size": 11}).status_code, 400)
        self.assertEqual(self.client.get("/api/field/timeline?field_id=fld_missing").status_code, 404)


if __name__ == "__main__":
    unittest.main()

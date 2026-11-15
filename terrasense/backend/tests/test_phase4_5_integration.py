"""Unit and Integration Tests for Phase 4 Validation & Phase 5 Engine Integration."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import app  # noqa: E402
import zone_engine as ze  # noqa: E402
from ml.config import MODELS_DIR, REPORTS_DIR  # noqa: E402
from ml.predict import TerraSensePredictor, get_predictor, reset_predictor_cache  # noqa: E402


class TestPhase4And5Integration(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        reset_predictor_cache()
        os.environ["TERRASENSE_ENGINE"] = "auto"

    def tearDown(self) -> None:
        os.environ["TERRASENSE_ENGINE"] = "auto"
        reset_predictor_cache()

    def test_phase4_validation_report_and_gate(self) -> None:
        report_json = REPORTS_DIR / "validation_report.json"
        report_md = REPORTS_DIR / "validation_report.md"
        self.assertTrue(report_json.exists(), "validation_report.json missing")
        self.assertTrue(report_md.exists(), "validation_report.md missing")

        rep = json.loads(report_json.read_text())
        self.assertTrue(rep["gate_passed"], "Validation promotion gate should pass")
        self.assertEqual(rep["recommended_default_engine"], "ml")
        for split_key in ("buffered_spatial_cv", "temporal_cv", "strict_cluster_and_year_out"):
            self.assertIn(split_key, rep)

    def test_engine_switch_ml_and_heuristic(self) -> None:
        # 1. ML mode (via auto)
        os.environ["TERRASENSE_ENGINE"] = "ml"
        reset_predictor_cache()
        r_ml = self.client.post("/api/field/analyze", json={"grid_size": 6, "date_range": {"end": "2024-11"}})
        self.assertEqual(r_ml.status_code, 200)
        d_ml = r_ml.get_json()
        self.assertEqual(d_ml["engine"], "ml")
        self.assertIsNotNone(d_ml["model_version"])
        self.assertIn("dynamic_candidates", d_ml)
        self.assertIn("ranking_robust", d_ml)

        tl_ml = self.client.get(f"/api/field/timeline?field_id={d_ml['field_id']}&scenario=cover_legume").get_json()
        self.assertEqual(tl_ml["engine"], "ml")
        self.assertIn("uncertainty", tl_ml["months"][12])
        self.assertLessEqual(tl_ml["months"][12]["uncertainty"]["smi_p10"], tl_ml["months"][12]["field"]["smi"])
        self.assertGreaterEqual(tl_ml["months"][12]["uncertainty"]["smi_p90"], tl_ml["months"][12]["field"]["smi"])

        # 2. Heuristic mode
        os.environ["TERRASENSE_ENGINE"] = "heuristic"
        reset_predictor_cache()
        r_h = self.client.post("/api/field/analyze", json={"grid_size": 6, "date_range": {"end": "2024-11"}})
        self.assertEqual(r_h.status_code, 200)
        d_h = r_h.get_json()
        self.assertEqual(d_h["engine"], "heuristic")
        self.assertIsNone(d_h["model_version"])

    def test_fallback_on_missing_or_corrupted_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Missing artifacts -> get_predictor returns None and logs warning
            reset_predictor_cache()
            pred = get_predictor(bundle_path=tmp_path / "missing.joblib", manifest_path=tmp_path / "missing.json")
            self.assertIsNone(pred)

            # Corrupted schema_hash in manifest -> raises ValueError on direct init, returns None via get_predictor
            bad_manifest = tmp_path / "bad_manifest.json"
            orig_manifest = json.loads((MODELS_DIR / "manifest.json").read_text())
            orig_manifest["schema_hash"] = "deadbeef0000"
            bad_manifest.write_text(json.dumps(orig_manifest))
            reset_predictor_cache()
            with self.assertRaises(ValueError):
                TerraSensePredictor(bundle_path=MODELS_DIR / "transition_bundle.joblib", manifest_path=bad_manifest)
            pred_bad = get_predictor(bundle_path=MODELS_DIR / "transition_bundle.joblib", manifest_path=bad_manifest)
            self.assertIsNone(pred_bad)

    def test_vectorized_inference_latency_default_grid(self) -> None:
        os.environ["TERRASENSE_ENGINE"] = "ml"
        reset_predictor_cache()
        geom = ze.parse_field(None)
        fid = ze.field_id_for(geom, 6)
        grid = ze.build_grid(geom, 6, fid)
        climate = ze.build_climate(geom, None, 0.0, 1.0, live=False)

        # Warm up singleton and static cache
        ze.build_timeline(fid, grid, climate, "status_quo")

        t0 = time.perf_counter()
        tl = ze.build_timeline(fid, grid, climate, "cover_legume")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(tl["engine"], "ml")
        self.assertLess(elapsed_ms, 500.0, f"Vectorized 6x6 timeline took {elapsed_ms:.1f} ms (> 500 ms limit)")
        print(f"\n[Phase 5 Benchmark] Vectorized 6x6 (36 zones x 24 months) ML timeline latency: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    unittest.main()

"""TerraSense Flask API.  Run:  python app.py  (http://localhost:5001)"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

import zone_engine as ze

app = Flask(__name__)
CORS(app)

FIELDS: Dict[str, dict] = {}      # in-memory field registry (swap for PostGIS/Redis later)
_TL_CACHE: Dict[tuple, dict] = {}


def _err(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


def _parse_anchor(s):
    return datetime.strptime(s[:7] + "-01", "%Y-%m-%d").date() if s else None


def _climate_params(src: dict):
    return (float(src.get("temp_shift", 0) or 0), float(src.get("precip_factor", 1) or 1))


def _prepare(field_spec, grid_n, anchor, climate_params, live):
    geom = ze.parse_field(field_spec)
    fid = ze.field_id_for(geom, grid_n)
    grid = ze.build_grid(geom, grid_n, fid)
    climate = ze.build_climate(geom, anchor, *climate_params, live=live)
    FIELDS[fid] = {"geom": geom, "grid": grid, "grid_n": grid_n, "anchor": anchor, "live": live}
    return fid, geom, grid, climate


@app.get("/api/health")
def health():
    eng = ze.active_engine_info()
    return jsonify({
        "status": "ok",
        "live_nasa": os.getenv("TERRASENSE_LIVE", "0") == "1",
        "engine": eng["engine"],
        "model_version": eng["model_version"],
    })


@app.get("/api/demo-field")
def demo_field():
    return jsonify({"field": {"bbox": ze.demo_bbox()}, "grid_size": 6,
                    "preferences": {"priority": "soil_restoration"}})


@app.post("/api/field/analyze")
def analyze():
    """Body: {field:{bbox|polygon}, grid_size?:4-8, date_range?:{end:'YYYY-MM'},
              preferences:{priority|weights}, climate?:{temp_shift, precip_factor}, use_live?:bool}"""
    body = request.get_json(silent=True) or {}
    try:
        grid_n = int(body.get("grid_size", 6))
        if not 3 <= grid_n <= 10:
            return _err("grid_size must be between 3 and 10")
        anchor = _parse_anchor((body.get("date_range") or {}).get("end"))
        cp = _climate_params(body.get("climate") or {})
        fid, geom, grid, climate = _prepare(body.get("field"), grid_n, anchor, cp, body.get("use_live"))
    except (ValueError, KeyError, TypeError) as exc:
        return _err(f"Invalid field: {exc}")

    prefs = body.get("preferences") or {}
    eng = ze.active_engine_info()
    comp = ze.compare_scenarios(grid, climate, prefs)
    rec = comp["recommended"]
    timeline = ze.build_timeline(fid, grid, climate, rec)
    _TL_CACHE[(fid, rec, cp, eng["engine"])] = timeline
    now = timeline["months"][ze.N_HIST - 1]

    feats = []
    for z, f in enumerate(grid["features"]):
        props = dict(f["properties"])
        props["recommended_scenario"] = comp["zone_recommendations"][z]
        props["current"] = now["cells"][z]
        feats.append({**f, "properties": props})
    ring = geom["ring"]
    boundary = {"type": "Feature", "id": "boundary",
                "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {"kind": "boundary"}}

    return jsonify({
        "field_id": fid,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": eng["engine"],
        "model_version": eng["model_version"],
        "is_proxy_ndvi_bare": True,
        "data_sources": {
            "climate": climate["source"],
            "soil_moisture": "Hybrid LightGBM + ERA5-Land/SSURGO" if eng["engine"] == "ml" else "SMAP (simulated)",
            "precipitation": "NASA POWER / GPM IMERG",
            "evapotranspiration": "Hybrid LightGBM + MERRA-2 EVLAND" if eng["engine"] == "ml" else "ECOSTRESS (simulated)",
            "vegetation": "MODIS NDVI (proxy — AppEEARS hook ready)",
        },
        "grid": {"rows": grid_n, "cols": grid_n, "n_zones": grid["n_zones"]},
        "bbox": geom["bbox"],
        "field_geojson": {"type": "FeatureCollection", "features": [boundary] + feats},
        "summary": {"current_month": now["label"], **now["field"], "alerts": now["alerts"]},
        "scenarios": comp["scenarios"],
        "recommended_scenario": rec,
        "ranking_robust": comp.get("ranking_robust", True),
        "uncertainty_note": comp.get("uncertainty_note", ""),
        "dynamic_candidates": comp.get("dynamic_candidates", []),
        "preference_weights": comp["weights"],
        "guidance": ze.build_guidance(timeline, rec),
    })


@app.get("/api/field/timeline")
def timeline():
    """Query: field_id, scenario (default status_quo), temp_shift, precip_factor."""
    fid = request.args.get("field_id", "")
    scenario = request.args.get("scenario", "status_quo")
    if scenario not in ze.PLANS:
        return _err(f"Unknown scenario '{scenario}'")
    rec = FIELDS.get(fid)
    if not rec:
        return _err("Unknown field_id — call POST /api/field/analyze first", 404)
    cp = _climate_params(request.args)
    eng = ze.active_engine_info()
    key = (fid, scenario, cp, eng["engine"])
    if key not in _TL_CACHE:
        climate = ze.build_climate(rec["geom"], rec["anchor"], *cp, live=rec["live"])
        _TL_CACHE[key] = ze.build_timeline(fid, rec["grid"], climate, scenario)
    tl = dict(_TL_CACHE[key])
    tl["guidance"] = ze.build_guidance(tl, scenario)
    return jsonify(tl)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=True)

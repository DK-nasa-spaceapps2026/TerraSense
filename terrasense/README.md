# TerraSense (`terrasense/`)

Farmer decision-support: NASA EO data → 6×6 management zones → 2.5D "Civilization-style" tile map, 24-month scrubber (12 observed + 12 forecast) and 3 rotation strategies.

> The repo root also contains an unrelated earlier project (TerraVision). TerraSense lives entirely in this folder.

## Run

```bash
# 1) API  (http://localhost:5001)
cd terrasense/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # TERRASENSE_LIVE=1 python app.py  -> real NASA POWER (falls back to mock)

# 2) Web  (http://localhost:3001) — /api/* is proxied to Flask
cd terrasense/frontend
npm install && npm run dev
```

No API keys needed: SMAP / GPM / ECOSTRESS / MODIS layers are simulated through a water-balance model
driven by NASA POWER-shaped climate (live or mock). `python backend/smoke_test.py` sanity-checks the API.

## Layout

```
terrasense/
├─ backend/
│  ├─ app.py          Flask routes (analyze, timeline, health, demo-field)
│  ├─ nasa_power.py   POWER client + deterministic mock + monthly aggregation
│  ├─ zone_engine.py  grid, soil fusion, water balance, scenarios, guidance
│  └─ smoke_test.py
└─ frontend/
   ├─ app/page.tsx                 state: analysis, timeline, month index, scenario, what-if
   ├─ components/IsoGrid.tsx       React Three Fiber isometric tiles
   ├─ components/TimelineSlider.tsx
   ├─ components/DecisionPanel.tsx
   ├─ components/TileDrawer.tsx
   └─ lib/{api,types}.ts
```

## API contract

### `POST /api/field/analyze`
```jsonc
{
  "field": { "bbox": [w, s, e, n] },          // or { "polygon": GeoJSON Polygon geometry }; omit = demo parcel
  "grid_size": 6,                              // 3–10
  "date_range": { "end": "2026-11" },          // optional anchor month ("now"); default today
  "preferences": { "priority": "water_saving|income|soil_restoration" },  // or { "weights": {...} }
  "climate": { "temp_shift": 0, "precip_factor": 1 },   // forecast what-if
  "use_live": false
}
```
Response: `field_id`, `data_sources`, `grid{rows,cols,n_zones}`, `bbox`,
`field_geojson` (FeatureCollection: 1 `boundary` + N `zone` features with `soil`, `current`, `recommended_scenario`),
`summary`, `scenarios[3]` (`metrics`: SOM gain, N fixed, water demand, water budget, erosion control, margin, risk; `tradeoffs`),
`recommended_scenario`, `preference_weights`, `guidance`
(`seed_window`, `vulnerable_windows[]`, `cover_crop_windows[]`, `stress_alerts[]`).

### `GET /api/field/timeline?field_id=…&scenario=status_quo|cover_legume|drought_cash&temp_shift=0&precip_factor=1`
```jsonc
{ "field_id": "...", "scenario": "...", "anchor_index": 11,
  "months": [ { "index": 0, "date": "2025-12", "label": "Dec 2025", "phase": "historical|forecast",
      "climate": {tmax_c,tmin_c,precip_mm,rh_pct,solar_mj,pet_mm}, "crop_stage": "...",
      "field": {smi,esi,bare,ndvi,vigor}, "alerts": [{type,severity,message}],
      "cells": [ {smi,esi,bare,ndvi,vigor,et_mm,status} ]   // same order as zone features
  } ], "guidance": {...} }
```
The chosen scenario is applied to forecast months (idx 12–23) only; history always reflects status-quo practice.

## Model notes
Monthly bucket water balance per zone (Hargreaves-style PET × crop Kc, runoff from slope/cover/clay, root-zone AWC from sand).
SMI = root-zone fill; ESI = ET/crop-PET; Bare risk = f(canopy/residue cover, slope, sand, storm intensity, dryness).
Status: bare ≥ 0.68 → water stress (ESI<0.65 or SMI<0.30) → heat stress → lush (vigor>0.62) → moderate.
All coefficients are hackathon-grade heuristics, not agronomically calibrated.

# Phase 0 — Reconnaissance, API Contract, and Data Quality Report

## 1. Exact API Request/Response Contract

### `GET /api/health`
- **Response (`200 OK`):** `{"status": "ok", "live_nasa": bool}`

### `GET /api/demo-field`
- **Response (`200 OK`):** `{"field": {"bbox": [w, s, e, n]}, "grid_size": 6, "preferences": {"priority": "soil_restoration"}}`

### `POST /api/field/analyze`
- **Request JSON:**
  - `field`: `{"bbox": [w, s, e, n]}` or `{"polygon": GeoJSON Polygon | ring}` (optional; defaults to `demo_bbox()`)
  - `grid_size`: `int` in `[3, 10]` (default `6`; outside `[3, 10]` returns `400`)
  - `date_range`: `{"end": "YYYY-MM"}` (optional; defaults to current month)
  - `preferences`: `{"priority": "water_saving" | "income" | "soil_restoration"}` or `{"weights": {...}}`
  - `climate`: `{"temp_shift": float, "precip_factor": float}` (default `0`, `1`)
  - `use_live`: `bool` (optional)
- **Response JSON (`200 OK`):**
  - `field_id`: `str` (`"fld_" + 10 hex chars`)
  - `generated_at`: ISO-8601 UTC timestamp `str`
  - `data_sources`: `dict[str, str]` (`climate`, `soil_moisture`, `precipitation`, `evapotranspiration`, `vegetation`)
  - `grid`: `{"rows": int, "cols": int, "n_zones": int}`
  - `bbox`: `[w, s, e, n]`
  - `field_geojson`: GeoJSON `FeatureCollection` with `1` `"boundary"` feature (`id="boundary"`, `properties.kind="boundary"`) followed by `n_zones` `"zone"` features (`id="z{r}{c}"`, `properties`: `kind="zone"`, `zone_id`, `row`, `col`, `recommended_scenario` in `{"status_quo", "cover_legume", "drought_cash"}`, `soil` with `sand_pct`, `clay_pct`, `organic_matter_pct`, `slope_pct`, `available_water_mm`, and `current` matching `CellState` at `anchor_index = 11`)
  - `summary`: `{"current_month": str, "smi": float, "esi": float, "bare": float, "ndvi": float, "vigor": float, "alerts": list[Alert]}`
  - `scenarios`: list of 3 `Scenario` dicts in order `["status_quo", "cover_legume", "drought_cash"]`, each with `id`, `name`, `crops`, `description`, `score` (`0..1`), `recommended` (`bool`), `metrics` (`soil_organic_matter_gain_pct`, `nitrogen_fixed_kg_ha`, `water_demand_mm`, `water_budget_balance_mm`, `erosion_control_score` (`0..100`), `estimated_margin_usd_ha`, `risk_level` in `{"low", "medium", "high"}`, `risk_score` (`0..1`)), and `tradeoffs` (`{"pros": list[str], "cons": list[str]}`)
  - `recommended_scenario`: `ScenarioId`
  - `preference_weights`: `{"water_saving": float, "income": float, "soil_restoration": float}` summing to `1.0`
  - `guidance`: `{"seed_window": Window | None, "vulnerable_windows": list[Window], "cover_crop_windows": list[Window], "stress_alerts": list[Window]}`

### `GET /api/field/timeline`
- **Query params:** `field_id` (required; `404` if unknown), `scenario` (`status_quo` | `cover_legume` | `drought_cash`; `400` otherwise), `temp_shift` (`float`), `precip_factor` (`float`).
- **Response JSON (`200 OK`):**
  - `field_id`: `str`
  - `scenario`: `ScenarioId`
  - `anchor_index`: `11` (`N_HIST - 1`)
  - `months`: list of `24` `MonthState` dicts (`index` `0..11` have `phase="historical"` and are identical across scenarios; `index` `12..23` have `phase="forecast"` and reflect `scenario`, `temp_shift`, `precip_factor`).
  - Each `MonthState.cells` has `n_zones` items in the exact order of `field_geojson` zone features, each with `smi`, `esi`, `bare`, `ndvi`, `vigor` in `[0, 1]`, `et_mm >= 0`, and `status` in `{"lush", "moderate", "water_stress", "heat_stress", "bare_vulnerable"}`.
  - `guidance`: `Guidance` dict for the requested `scenario`.

---

## 2. Dataset Audit (`terrasense/backend/data/`)

Computed by `terrasense/backend/ml/recon_phase0.py` across the **305 points** in `appeears_points.csv` (5 `demo_field_*` + 300 `train_0001..train_0300`):

| Dataset | Files Present | Date Coverage | Effective Spatial Cells | Variables & Units | Null / Anomaly Findings |
|---|---:|---|---:|---|---|
| `nasa_power/` | **305 / 305** | `2023-01-01` – `2024-12-31` (731 days; 222,955 rows) | **21** unique trajectories | `precip` (mm/day, `0..65.05`), `tmax` (°C, `-21.97..41.58`), `tmin` (°C, `-32.55..25.40`), `rh` (%, `36.94..99.79`), `solar` (MJ/m²/day, `1.15..31.95`) | `0` nulls (`0.00%`) |
| `nasa_power_targets/` | **305 / 305** | `2023-01-01` – `2024-12-31` (731 days; 222,955 rows) | **10** unique MERRA-2 cells | `gwettop`, `gwetroot`, `gwetprof` (dimensionless degree of saturation `0..1`, range `0.32..0.82`); `evland` (mm/day, `-0.17..7.23`, mean `1.88`) | `0` nulls; `evland` has slight negative condensation values (`min = -0.17 mm/day`) that should be clipped at `0.0` when computing monthly ET |
| `open_meteo/` | **182 / 305** *(123 missing)* | `2023-01-01` – `2024-12-31` (731 days; 133,042 rows) | **138** unique ERA5-Land cells (across 182 files) | `sm_0_7cm`, `sm_7_28cm`, `sm_28_100cm` (volumetric $\text{m}^3/\text{m}^3$, range `0.126..0.432`); `soil_temp_0_7cm` (°C, `-7.35..32.08`); `et0_mm` (mm/day, `0.16..8.25`) | `0` nulls in the 182 present files; **123 files (`train_0019`..`train_0300` subset) were not yet downloaded** due to earlier HTTP 429 rate-limiting |
| `ssurgo/` | **305 / 305** | Static (1,541 horizon rows) | Point-level mapunits | `comppct_r` (%), `hzdept_r`/`hzdepb_r` (cm), `awc_r` (cm/cm, `0.02..0.23`), `claytotal_r`/`sandtotal_r`/`silttotal_r` (%), `om_r` (%, `0..10`), `ph1to1h2o_r` (`4.4..8.0`), `dbthirdbar_r` ($\text{g/cm}^3$, `0.95..1.80`), `drainagecl` | **5 points have empty tables** (`demo_field_sw`, `train_0129`, `train_0150`, `train_0178`, `train_0231` — non-major/urban/water components); `demo_field_*` is on Des Moines urban-land complex |
| `soilgrids/` | **189 / 305** *(116 missing on disk)* | Static (`0-5`, `5-15`, `15-30 cm`) | 250 m raster | **Mapped integer units** requiring division by `d_factor`: `bdod` ($\text{cg/cm}^3 \div 100 \to \text{kg/dm}^3$), `cec` ($\div 10$), `clay`/`sand`/`silt` ($\text{g/kg} \div 10 \to \%$), `phh2o` ($\div 10 \to \text{pH}$), `soc` ($\text{dg/kg} \div 10 \to \text{g/kg}$) | **116 missing files** are from interrupted API fetching (`train_0058`..`train_0280`); of the 189 downloaded files, **9 are all-`null`** because ISRIC masks urban/water pixels (`demo_field_*`, `train_0006`, `train_0025`, `train_0060`, `train_0274`) |
| `cdl/` | **2 GeoTIFFs + 2 `.vat.dbf`** | `2023`, `2024` | 30 m raster | Integer crop codes (`1` = Corn, `5` = Soybeans, etc.; 134 active classes in `.vat.dbf`) | Complete |
| `earthdata/` | **6 DEM GeoTIFFs** | Static | 30 m raster | Elevation tiles covering `N41..N42`, `W093..W095` | Complete |

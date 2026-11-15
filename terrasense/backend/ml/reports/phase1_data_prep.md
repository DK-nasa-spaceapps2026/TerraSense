# Phase 1 — Data Preparation & Feature Engineering Report

## 1. Cropland Filtering & Rotation Audit (`points_static.csv`)
All **305** points from `appeears_points.csv` are retained in `data/processed/points_static.csv` with explicit `keep_for_training` and `drop_reason` columns (zero silent drops).
- **CRS & Sampling:** Coordinates (`EPSG:4326`) were reprojected to `EPSG:5070` (NAD83 / Conus Albers, 30 m metric pixels) and sampled using a `3×3` (`90 m × 90 m`) majority window with center-pixel tie-breaking and a `cropland_frac_3x3` purity metric.
- **Survival Breakdown (`305` points total):**
  - **`kept_cropland`: `195` points (`63.9%`)** — Cultivated cropland in 2023 or 2024 (Corn `1`, Soybeans `5`, Sorghum `4`, Oats `28`, Alfalfa/Hay `36/37`) and not intersecting developed/water classes.
    - Among the 195 surviving cropland points, **157 (`80.5%`)** are in a **Corn–Soybean rotation** (`{crop_2023, crop_2024} == {1, 5}`) and **27 (`13.8%`)** are **Continuous Corn** (`1 -> 1`).
  - **`permanent_pasture_176`: `45` points** — CDL `176` (`Grassland/Pasture`) in both years; explicitly excluded from row-crop training because permanent pasture/CRP/roadside grass has uncultivated perennial cover and different root-zone dynamics.
  - **`forest_or_shrubland`: `30` points** — Deciduous/mixed forest (`141`, `143`).
  - **`developed_urban`: `26` points** — Includes the 5 `demo_field_*` coordinates in Des Moines (`121–124`).
  - **`water_or_wetland`: `6` points** — Open water (`111`) or wetlands (`190`, `195`).
  - **`out_of_cdl_bounds`: `3` points** — Near the tilted Albers corner of the clipped CDL raster.

## 2. Terrain Feature Extraction (Metric `EPSG:5070` DEM Reprojection)
- For every point, overlapping SRTM 1-arc-second (`EPSG:4326`) tiles from `data/earthdata/` are reprojected via bilinear resampling into a local `15×15` (`450 m × 450 m`) metric patch in `EPSG:5070` (`30.0 m × 30.0 m` pixels) **before** computing spatial derivatives:
  - `elev_m`, `elev_mean_90m` (m)
  - `slope_pct`, `slope_mean_90m`, `slope_std_90m` (%)
  - `aspect_sin`, `aspect_cos` (unitless clockwise from North)
  - `curvature` (Laplacian $\nabla^2 z \times 100$, $\text{m}^{-1} \times 100$)
  - `local_relief_90m` ($3\times 3$ $\max-\min$, m) and `local_relief_210m` ($7\times 7$ $\max-\min$, m)
  - `twi` (Topographic Wetness Index $\ln(a / \tan\beta)$ from deterministic D8 upslope contributing area $a$).

## 3. Soil Feature Aggregation (`SSURGO` Primary, `SoilGrids` Secondary)
- **SSURGO (`data/ssurgo/`):** Non-soil components (`Urban land`, `Water`, `Pits`) are excluded; remaining components are weighted by `comppct_r`, and horizons are thickness-weighted into **`0–30 cm` (`top`)** and **`30–100 cm` (`sub`)** layers (`sand`, `clay`, `silt`, `om`, `ph`, `bd`, `awc`), plus `awc_100cm_mm` and ordinal `drainage_code` (`1` = very poorly drained .. `7` = excessively drained).
- **SoilGrids (`data/soilgrids/`):** Converted from mapped integer units via `d_factor` (`bdod / 100`, `soc, cec, clay, sand, silt, phh2o / 10`) over `0–30 cm`. Used to fill SSURGO gaps where available and kept as `sg_*` columns (preserving `NaN` for native handling by LightGBM/XGBoost; never used to drop points). The remaining missing SSURGO values (only 1–2 cropland points) are imputed with cropland training medians (`awc_100cm_mm = 183.2 mm`, `clay_top_pct = 28.75%`, `sand_top_pct = 28.67%`) with explicit `*_imputed` indicator flags.

## 4. Weather Alignment, Target Definitions & Proxy Hooks (`training_monthly.csv`)
- **Panel Size:** `195` cropland points $\times$ `24` months (`2023-01`..`2024-12`) = **`4,680` point-month rows**.
  - `106` of the `195` cropland points (`2,544` rows across `82` distinct ERA5-Land cells) have fine-resolution (~9 km) Open-Meteo ERA5-Land volumetric soil moisture (`sm_0_7cm`, `sm_7_28cm`, `sm_28_100cm`).
  - All `195` cropland points (`4,680` rows across `10` MERRA-2 cells and `19` spatial blocks) have `nasa_power` weather and `nasa_power_targets` (`gwetroot`, `gwettop`, `evland`).
- **Invertible SMI Target (`target_smi_era5`, `target_smi_merra2`, `target_smi`):**
  - Depth-weighted `0–100 cm` volumetric soil moisture $\theta = 0.07\,\theta_{0-7} + 0.21\,\theta_{7-28} + 0.72\,\theta_{28-100}$ is normalized into $[0, 1]$ using per-point Saxton-Rawls/SSURGO bounds $(\theta_{\text{wp}}, \theta_{\text{fc}})$:
    $$\text{SMI} = \operatorname{clip}\!\left(\frac{\theta - \theta_{\text{wp}}}{\theta_{\text{fc}} - \theta_{\text{wp}}},\, 0,\, 1\right), \qquad \theta = \theta_{\text{wp}} + \text{SMI}\cdot(\theta_{\text{fc}} - \theta_{\text{wp}})$$
- **ESI Proxy Target (`target_esi`):** Monthly actual ET (`evland_mm`, with negative winter condensation days clipped at `0.0`) divided by reference PET (`et0_fao_mm` or Hargreaves `pet_mm`), clipped to `[0.05, 1.0]`.
- **NDVI / Vigor / Bare-Soil Proxy Policy:** Because no satellite NDVI (`MODIS`) or `ECOSTRESS` granules are present in `data/`, **no ML model is trained on synthetic NDVI or bare-soil labels**. They are computed deterministically via `compute_proxy_indices()`, flagged with `is_proxy_ndvi_bare = 1`, and wired to `load_appeears_ndvi_hook()` so dropping AppEEARS CSVs into `data/appeears/` automatically populates `observed_ndvi`.
- **Temporal Coverage Note (2023–2024 vs. 2015–2024):** Sandbox network isolation prevents autonomous external downloads during this run, so Phase 1 proceeds with the **2023–2024 (24-month)** record on disk (`2023` drought/dry summer vs. `2024` wetter spring/summer provides two contrasting years for analog-year ensembles and leave-year-out CV). To extend to 2015–2024 at any time, run:
  `python fetch_training_data.py --sources power power_targets open_meteo --start 2015-01-01 --end 2024-12-31`
  followed by `python -m ml.build_dataset`.

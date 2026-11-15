# Phase 4 — Spatio-Temporal Validation & Baseline Comparison Report

- **Promotion Gate Passed:** `True` (Recommended default engine: `ml`)
- **Spatial Buffer Enforced:** `>= 25 km` Haversine exclusion zone between train and test points (plus zero shared ERA5-Land cells).

## 1. Summary Table Across All Validation Regimes (12-Month Free-Running Rollouts)

| Validation Split | Target | ML RMSE | Heuristic RMSE | Climatology RMSE | Persistence RMSE | ML $R^2$ | Spearman $r$ | Skill vs. Clim | Skill vs. Heur | 80% Interval Cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Buffered Spatial CV (>=25 km) | **SMI** | **0.1415** | 0.2755 | 0.1854 | 0.2069 | 0.3878 | 0.6372 | `+41.7%` | `+73.6%` | `79.9%` |
| Buffered Spatial CV (>=25 km) | **ESI** | **0.1275** | 0.3366 | 0.1811 | 0.3978 | 0.7391 | 0.8787 | `+50.4%` | `+85.6%` | `70.4%` |
| Temporal (2023->2024 & 2024->2023) | **SMI** | **0.1474** | 0.2968 | 0.2157 | 0.1957 | 0.3899 | 0.6416 | `+53.3%` | `+75.3%` | `75.4%` |
| Temporal (2023->2024 & 2024->2023) | **ESI** | **0.1581** | 0.3323 | 0.2562 | 0.3198 | 0.5603 | 0.7998 | `+61.9%` | `+77.3%` | `44.2%` |
| Strictest: Cluster (>=25 km) + Year Out | **SMI** | **0.1739** | 0.2755 | 0.2245 | 0.2069 | 0.0750 | 0.4937 | `+40.0%` | `+60.1%` | `70.6%` |
| Strictest: Cluster (>=25 km) + Year Out | **ESI** | **0.1955** | 0.3366 | 0.2547 | 0.3978 | 0.3871 | 0.6765 | `+41.1%` | `+66.3%` | `40.4%` |

## 2. Status Classification Accuracy & Macro-F1 (`lush`, `moderate`, `water_stress`, `heat_stress`, `bare_vulnerable`)

| Validation Split | ML Macro-F1 | Heuristic Macro-F1 | ML Accuracy | Heuristic Accuracy |
|---|---:|---:|---:|---:|
| Buffered Spatial CV (>=25 km) | **0.8472** | 0.7071 | **84.3%** | 68.5% |
| Temporal Combined | **0.7649** | 0.6931 | **75.8%** | 66.0% |
| Strictest: Cluster + Year Out | **0.7157** | 0.7071 | **67.8%** | 68.5% |

## 3. Known Limitations & Honest Assessment
- **2-Year Temporal Record (2023–2024):** While 2023 (drought) and 2024 (wetter spring/summer) provide strong contrast, training on `2023` alone in the strictest year-out split has narrower historical coverage than training on 10+ years.
- **Target Resolution:** `target_smi_era5` (~9 km, 106 cropland points) and `target_smi_merra2` / `target_esi` (~50 km, 195 cropland points) are regional reanalysis grids; fine 30 m sub-field variation is driven by the SSURGO + metric DEM +FAO-56 water-balance backbone.
- **Proxy NDVI & Bare-Soil Exposure:** Flagged as `is_proxy_ndvi_bare=1` until AppEEARS MODIS NDVI / ECOSTRESS granules are added to `data/appeears/`.

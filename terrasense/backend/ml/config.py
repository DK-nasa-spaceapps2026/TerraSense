"""Centralized configuration for the TerraSense ML pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

ML_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ML_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = ML_DIR / "reports"
POINTS_CSV = BACKEND_DIR / "appeears_points.csv"

SEED: int = 42
METRIC_CRS: str = "EPSG:5070"  # NAD83 / Conus Albers (meters)
DEM_PATCH_RES_M: float = 30.0
DEM_PATCH_SIZE: int = 15       # 15x15 pixels = 450m x 450m metric neighborhood

# Weather / agronomy thresholds
HEAT_STRESS_THRESH_C: float = 32.0
GDD_BASE_C: float = 10.0
GDD_CAP_C: float = 30.0

# USDA CDL Cultivated Cropland codes (1..60, 66..77, 204..254 excluding 61 fallow/idle if non-crop)
CULTIVATED_CROP_CODES: Set[int] = (
    set(range(1, 61))
    | set(range(66, 78))
    | set(range(204, 255))
)
FORAGE_CROP_CODES: Set[int] = {36, 37, 58, 59}  # Alfalfa, Other Hay, Clover/Wildflowers, Sod/Grass Seed
PASTURE_CODES: Set[int] = {176}                 # Grassland/Pasture (excluded from annual row-crop training)
DEVELOPED_CODES: Set[int] = {121, 122, 123, 124}
WATER_WETLAND_CODES: Set[int] = {83, 92, 111, 190, 195}
FOREST_SHRUB_CODES: Set[int] = {63, 64, 65, 141, 142, 143, 152}

# Ordinal encoding of SSURGO drainage class (1 = wettest/poorest drainage, 7 = driest/excessive)
DRAINAGE_ORDINAL: Dict[str, int] = {
    "Very poorly drained": 1,
    "Poorly drained": 2,
    "Somewhat poorly drained": 3,
    "Moderately well drained": 4,
    "Well drained": 5,
    "Somewhat excessively drained": 6,
    "Excessively drained": 7,
}

# SoilGrids d_factor divisors (mapped integer units -> physical target units)
SOILGRIDS_D_FACTOR: Dict[str, float] = {
    "bdod": 100.0,   # cg/cm3 -> kg/dm3 (g/cm3)
    "cec": 10.0,     # mmol(c)/kg -> cmol(c)/kg
    "clay": 10.0,    # g/kg -> %
    "sand": 10.0,    # g/kg -> %
    "silt": 10.0,    # g/kg -> %
    "soc": 10.0,     # dg/kg -> g/kg
    "phh2o": 10.0,   # pH*10 -> pH
}

STATIC_SOIL_TERRAIN_FEATURES = [
    "sand_top_pct",
    "clay_top_pct",
    "silt_top_pct",
    "om_top_pct",
    "ph_top",
    "bd_top_gcm3",
    "awc_top_cmcm",
    "sand_sub_pct",
    "clay_sub_pct",
    "awc_sub_cmcm",
    "awc_100cm_mm",
    "drainage_code",
    "sg_soc_gkg",
    "sg_cec_cmol",
    "sg_bdod_gcm3",
    "sg_clay_pct",
    "sg_sand_pct",
    "sg_ph",
    "elev_m",
    "slope_pct",
    "slope_mean_90m",
    "slope_std_90m",
    "aspect_sin",
    "aspect_cos",
    "curvature",
    "local_relief_90m",
    "local_relief_210m",
    "twi",
]

CROP_FEATURES = [
    "is_corn",
    "is_soy",
    "is_forage",
    "corn_soy_rotation",
    "continuous_corn",
    "cropland_frac_3x3",
    "cal_cover",
    "cal_kc",
]

WEATHER_FEATURES = [
    "precip_mm",
    "tmax_c",
    "tmin_c",
    "tmean_c",
    "diurnal_range_c",
    "rh_pct",
    "solar_mj",
    "gdd_month",
    "heat_stress_days",
    "max_dry_spell_days",
    "max_5day_precip_mm",
    "max_daily_precip_mm",
    "vpd_kpa",
    "pet_mm",
    "water_balance_mm",
    "aridity_index",
    "api_antecedent_mm",
    "month_sin",
    "month_cos",
]

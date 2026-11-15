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

# --------------------------------------------------------------------------- #
# Farmer Customization Defaults, Ranges, Presets & Economic Priors            #
# --------------------------------------------------------------------------- #
# Crop name <-> USDA CDL code lookup for rotation history & crop exclusions
CROP_NAME_TO_CODE: Dict[str, int] = {
    "corn": 1,
    "maize": 1,
    "sorghum": 4,
    "milo": 4,
    "soybean": 5,
    "soybeans": 5,
    "soy": 5,
    "sunflower": 6,
    "winter_wheat": 24,
    "wheat": 24,
    "rye": 27,
    "cereal_rye": 27,
    "oats": 28,
    "alfalfa": 36,
    "hay": 37,
    "other_hay": 37,
    "clover": 58,
    "crimson_clover": 58,
}

CODE_TO_CROP_NAME: Dict[int, str] = {
    1: "corn",
    4: "sorghum",
    5: "soybean",
    6: "sunflower",
    24: "winter_wheat",
    27: "rye",
    28: "oats",
    36: "alfalfa",
    37: "hay",
    58: "clover",
}

DEFAULT_PREV_CROP_CODE: int = 1             # Corn (matches historical default in prune_candidate_plans)
DEFAULT_CONTINUOUS_CORN_YEARS: int = 1
MAX_CONTINUOUS_CORN_YEARS: int = 2          # ISU Extension PM 1885: >2 yr continuous corn yield penalty & rootworm risk
CONTINUOUS_CORN_RANGE = (0, 10)

# Equipment ownership defaults (all True reproduces unconstrained baseline)
DEFAULT_EQUIPMENT: Dict[str, bool] = {
    "has_hay_baler": True,
    "has_cover_drill": True,
    "has_no_till_drill": True,
    "variable_rate": True,
}

# Custom-hire fallback rates (USD/ha) from ISU Extension Ag Decision Maker File A3-10
# (2024 Iowa Farm Custom Rate Survey: 3-cut mowing/raking/baling ~$115/ha; cover drilling ~$38/ha; no-till ~$42/ha)
CUSTOM_HIRE_COSTS_USD_HA: Dict[str, float] = {
    "hay_baler": 115.0,
    "cover_drill": 38.0,
    "no_till_drill": 42.0,
}

# Supplemental irrigation forcing bounds & costs
# Source: ISU Extension File A1-20 / Midwest center-pivot pumping cost (~$11.50/ac-in ≈ $0.45/mm/ha)
IRRIGATION_CAP_DEFAULT_MM: float = 0.0
IRRIGATION_CAP_RANGE_MM = (0.0, 600.0)
DEFAULT_IRRIGATION_COST_USD_MM: float = 0.45
IRRIGATION_COST_RANGE_USD_MM = (0.0, 5.0)
IRRIGATION_SMI_TRIGGER: float = 0.68        # Trigger supplemental irrigation when reproductive-month SMI < 0.68 (field capacity threshold)
IRRIGATION_TARGET_SMI: float = 0.85         # Target root-zone SMI refill fraction during reproductive stage
IRRIGATION_CRITICAL_MONTHS = (6, 7, 8)      # June–August reproductive & grain/pod-fill stress window
TRAINING_SMI_CLIP_RANGE = (0.02, 0.94)      # Observed training distribution bounds in ERA5-Land/MERRA-2

# Risk-aversion parameter (lambda in Mean Utility - lambda * Downside Deviation)
RISK_AVERSION_PRESETS: Dict[str, float] = {
    "conservative": 0.50,
    "balanced": 0.25,
    "aggressive": 0.05,
}
DEFAULT_RISK_AVERSION_LAMBDA: float = 0.25
RISK_AVERSION_RANGE = (0.0, 1.0)

# Multi-objective preference weights (simplicity defaults to 0.0 to preserve exact 1/3 baseline)
PRIORITY_KEYS = ("water_saving", "income", "soil_restoration", "simplicity")
DEFAULT_WEIGHTS: Dict[str, float] = {
    "water_saving": 1.0 / 3.0,
    "income": 1.0 / 3.0,
    "soil_restoration": 1.0 / 3.0,
    "simplicity": 0.0,
}

# Baseline commodity & nitrogen prices embedded in AGRONOMIC_PLANS.base_margin_usd_ha
# Source: ISU Extension Ag Decision Maker File A1-20 & USDA NASS Iowa marketing-year averages;
# synthetic N fertilizer ~$0.50/lb N ≈ $1.10/kg N (ISU PM 1688).
DEFAULT_PRICES: Dict[str, float] = {
    "corn_usd_bu": 4.75,
    "soybean_usd_bu": 11.50,
    "sorghum_usd_bu": 4.40,
    "alfalfa_usd_ton": 185.0,
    "nitrogen_price_usd_kg": 1.10,
}
PRICE_RANGES: Dict[str, tuple] = {
    "corn_usd_bu": (1.50, 15.00),
    "soybean_usd_bu": (4.00, 30.00),
    "sorghum_usd_bu": (1.50, 15.00),
    "alfalfa_usd_ton": (50.0, 500.0),
    "nitrogen_price_usd_kg": (0.10, 5.00),
}
PRICE_UNITS: Dict[str, str] = {
    "corn_usd_bu": "USD/bushel",
    "soybean_usd_bu": "USD/bushel",
    "sorghum_usd_bu": "USD/bushel",
    "alfalfa_usd_ton": "USD/ton",
    "nitrogen_price_usd_kg": "USD/kg N",
}

# Management modes
MANAGEMENT_MODES = ("uniform", "zone_variable")
DEFAULT_MANAGEMENT_MODE: str = "zone_variable"


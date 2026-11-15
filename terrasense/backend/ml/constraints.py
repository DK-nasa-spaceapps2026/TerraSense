"""Request schema parsing, validation, defaults, and metadata for farmer customization (`backend/ml/constraints.py`)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ml.config import (
    CODE_TO_CROP_NAME,
    CONTINUOUS_CORN_RANGE,
    CROP_NAME_TO_CODE,
    CUSTOM_HIRE_COSTS_USD_HA,
    DEFAULT_CONTINUOUS_CORN_YEARS,
    DEFAULT_EQUIPMENT,
    DEFAULT_IRRIGATION_COST_USD_MM,
    DEFAULT_MANAGEMENT_MODE,
    DEFAULT_PREV_CROP_CODE,
    DEFAULT_PRICES,
    DEFAULT_RISK_AVERSION_LAMBDA,
    DEFAULT_WEIGHTS,
    IRRIGATION_CAP_DEFAULT_MM,
    IRRIGATION_CAP_RANGE_MM,
    IRRIGATION_COST_RANGE_USD_MM,
    MANAGEMENT_MODES,
    MAX_CONTINUOUS_CORN_YEARS,
    PRICE_RANGES,
    PRICE_UNITS,
    PRIORITY_KEYS,
    RISK_AVERSION_PRESETS,
    RISK_AVERSION_RANGE,
)


class ConstraintValidationError(ValueError):
    """Raised when a farmer constraint field has an invalid type, value, or range (HTTP 400)."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(f"Invalid constraint '{field}': {message}")


KNOWN_TOP_CONSTRAINTS = {
    "prev_crop",
    "continuous_corn_years",
    "max_continuous_corn_years",
    "equipment",
    "allow_custom_hire",
    "excluded_crops",
    "irrigation_cap_mm",
    "irrigation_cost_usd_per_mm",
    "risk_aversion",
    "weights",
    "priority",
    "prices",
    "management_mode",
}

KNOWN_EQUIPMENT_KEYS = set(DEFAULT_EQUIPMENT.keys())
KNOWN_WEIGHT_KEYS = set(PRIORITY_KEYS)
KNOWN_PRICE_KEYS = set(DEFAULT_PRICES.keys())


def _is_bool(val: Any) -> bool:
    return isinstance(val, bool)


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _resolve_crop(val: Any, field_name: str) -> Tuple[int, str]:
    """Resolve a crop name or CDL integer code into (cdl_code, canonical_name)."""
    if isinstance(val, bool):
        raise ConstraintValidationError(field_name, "Expected crop name (string) or CDL code (int), got boolean.")
    if isinstance(val, int):
        if val in CODE_TO_CROP_NAME:
            return val, CODE_TO_CROP_NAME[val]
        raise ConstraintValidationError(
            field_name,
            f"Unknown CDL crop code {val}. Supported codes: {sorted(CODE_TO_CROP_NAME.keys())}.",
        )
    if isinstance(val, str):
        key = val.strip().lower().replace(" ", "_").replace("-", "_")
        if key in CROP_NAME_TO_CODE:
            code = CROP_NAME_TO_CODE[key]
            return code, CODE_TO_CROP_NAME[code]
        raise ConstraintValidationError(
            field_name,
            f"Unknown crop '{val}'. Supported crops: {sorted(set(CODE_TO_CROP_NAME.values()))}.",
        )
    raise ConstraintValidationError(field_name, "Expected crop name (string) or CDL code (int).")


def parse_and_validate_constraints(body: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """Parse and validate optional `constraints` and legacy `preferences` from request JSON.

    Returns `(applied_constraints, warnings)`.
    Raises `ConstraintValidationError` on invalid types, out-of-range values, or unknown crops.
    """
    body = body or {}
    warnings: List[str] = []

    raw_constraints = body.get("constraints")
    raw_preferences = body.get("preferences")

    if raw_constraints is not None and not isinstance(raw_constraints, dict):
        raise ConstraintValidationError("constraints", "Must be a JSON object.")
    if raw_preferences is not None and not isinstance(raw_preferences, dict):
        raise ConstraintValidationError("preferences", "Must be a JSON object.")

    has_constraints = bool(raw_constraints)
    has_preferences = bool(raw_preferences)

    if has_constraints and has_preferences:
        warnings.append(
            "Both 'constraints' and 'preferences' were provided; 'constraints' took precedence."
        )

    src: Dict[str, Any] = dict(raw_constraints) if raw_constraints is not None else {}

    # Warn on unknown top-level keys inside `constraints`
    for k in src:
        if k not in KNOWN_TOP_CONSTRAINTS:
            warnings.append(f"Unknown constraint key '{k}' was ignored.")

    # 1. Previous crop & continuous corn years
    if "prev_crop" in src and src["prev_crop"] is not None:
        prev_crop_code, prev_crop_name = _resolve_crop(src["prev_crop"], "prev_crop")
    else:
        prev_crop_code = DEFAULT_PREV_CROP_CODE
        prev_crop_name = CODE_TO_CROP_NAME[prev_crop_code]

    if "continuous_corn_years" in src and src["continuous_corn_years"] is not None:
        cc_val = src["continuous_corn_years"]
        if isinstance(cc_val, bool) or not isinstance(cc_val, int):
            raise ConstraintValidationError("continuous_corn_years", "Must be an integer >= 0.")
        lo, hi = CONTINUOUS_CORN_RANGE
        if not (lo <= cc_val <= hi):
            raise ConstraintValidationError(
                "continuous_corn_years", f"Must be between {lo} and {hi} years (got {cc_val})."
            )
        continuous_corn_years = cc_val
    else:
        continuous_corn_years = DEFAULT_CONTINUOUS_CORN_YEARS if prev_crop_code == 1 else 0

    max_cc_years = MAX_CONTINUOUS_CORN_YEARS
    if "max_continuous_corn_years" in src and src["max_continuous_corn_years"] is not None:
        mcc = src["max_continuous_corn_years"]
        if isinstance(mcc, bool) or not isinstance(mcc, int) or not (1 <= mcc <= 10):
            raise ConstraintValidationError("max_continuous_corn_years", "Must be an integer between 1 and 10.")
        max_cc_years = mcc

    # 2. Equipment availability & custom hire
    equipment = dict(DEFAULT_EQUIPMENT)
    if "equipment" in src and src["equipment"] is not None:
        raw_eq = src["equipment"]
        if not isinstance(raw_eq, dict):
            raise ConstraintValidationError("equipment", "Must be an object mapping equipment flags to booleans.")
        for k, v in raw_eq.items():
            if k not in KNOWN_EQUIPMENT_KEYS:
                warnings.append(f"Unknown equipment key 'equipment.{k}' was ignored.")
                continue
            if not _is_bool(v):
                raise ConstraintValidationError(f"equipment.{k}", f"Must be a boolean (got {type(v).__name__}).")
            equipment[k] = v

    allow_custom_hire = True
    if "allow_custom_hire" in src and src["allow_custom_hire"] is not None:
        if not _is_bool(src["allow_custom_hire"]):
            raise ConstraintValidationError("allow_custom_hire", "Must be a boolean.")
        allow_custom_hire = src["allow_custom_hire"]

    # 3. Excluded crops (no local buyer / farmer preference)
    excluded_codes: List[int] = []
    excluded_names: List[str] = []
    if "excluded_crops" in src and src["excluded_crops"] is not None:
        raw_ex = src["excluded_crops"]
        if not isinstance(raw_ex, list):
            raise ConstraintValidationError("excluded_crops", "Must be a list of crop names or CDL codes.")
        for idx, item in enumerate(raw_ex):
            c_code, c_name = _resolve_crop(item, f"excluded_crops[{idx}]")
            if c_code not in excluded_codes:
                excluded_codes.append(c_code)
                excluded_names.append(c_name)

    # 4. Supplemental irrigation cap & cost
    irrigation_cap_mm = IRRIGATION_CAP_DEFAULT_MM
    if "irrigation_cap_mm" in src and src["irrigation_cap_mm"] is not None:
        val = src["irrigation_cap_mm"]
        if not _is_number(val):
            raise ConstraintValidationError("irrigation_cap_mm", "Must be a number (mm).")
        lo, hi = IRRIGATION_CAP_RANGE_MM
        if not (lo <= float(val) <= hi):
            raise ConstraintValidationError(
                "irrigation_cap_mm", f"Must be between {lo} and {hi} mm (got {val})."
            )
        irrigation_cap_mm = float(val)

    irrigation_cost_usd_per_mm = DEFAULT_IRRIGATION_COST_USD_MM
    if "irrigation_cost_usd_per_mm" in src and src["irrigation_cost_usd_per_mm"] is not None:
        val = src["irrigation_cost_usd_per_mm"]
        if not _is_number(val):
            raise ConstraintValidationError("irrigation_cost_usd_per_mm", "Must be a number (USD/mm/ha).")
        lo, hi = IRRIGATION_COST_RANGE_USD_MM
        if not (lo <= float(val) <= hi):
            raise ConstraintValidationError(
                "irrigation_cost_usd_per_mm", f"Must be between {lo} and {hi} USD/mm/ha (got {val})."
            )
        irrigation_cost_usd_per_mm = float(val)

    # 5. Risk aversion preset or float
    risk_aversion_lambda = DEFAULT_RISK_AVERSION_LAMBDA
    risk_aversion_label: str = "balanced"
    if "risk_aversion" in src and src["risk_aversion"] is not None:
        ra = src["risk_aversion"]
        if isinstance(ra, str):
            ra_key = ra.strip().lower()
            if ra_key not in RISK_AVERSION_PRESETS:
                raise ConstraintValidationError(
                    "risk_aversion",
                    f"Unknown preset '{ra}'. Valid presets: {sorted(RISK_AVERSION_PRESETS.keys())} or float in {list(RISK_AVERSION_RANGE)}.",
                )
            risk_aversion_lambda = RISK_AVERSION_PRESETS[ra_key]
            risk_aversion_label = ra_key
        elif _is_number(ra):
            lo, hi = RISK_AVERSION_RANGE
            if not (lo <= float(ra) <= hi):
                raise ConstraintValidationError(
                    "risk_aversion", f"Numeric risk_aversion must be between {lo} and {hi} (got {ra})."
                )
            risk_aversion_lambda = float(ra)
            risk_aversion_label = "custom"
        else:
            raise ConstraintValidationError(
                "risk_aversion", "Must be a preset string ('conservative', 'balanced', 'aggressive') or float in [0.0, 1.0]."
            )

    # 6. Multi-objective weights / priority (from `constraints` if present, else fallback to `preferences`)
    weight_source = src if ("weights" in src or "priority" in src) else (raw_preferences or {})
    raw_weights = weight_source.get("weights")
    raw_priority = weight_source.get("priority")

    weights_4: Dict[str, float] = dict(DEFAULT_WEIGHTS)
    if raw_weights is not None:
        if not isinstance(raw_weights, dict):
            raise ConstraintValidationError("weights", "Must be an object mapping objective names to non-negative numbers.")
        for k in raw_weights:
            if k not in KNOWN_WEIGHT_KEYS:
                warnings.append(f"Unknown weight key 'weights.{k}' was ignored.")
        parsed_w: Dict[str, float] = {}
        for k in PRIORITY_KEYS:
            if k in raw_weights and raw_weights[k] is not None:
                v = raw_weights[k]
                if not _is_number(v):
                    raise ConstraintValidationError(f"weights.{k}", "Weight must be a number.")
                if float(v) < 0.0:
                    raise ConstraintValidationError(f"weights.{k}", f"Weight cannot be negative (got {v}).")
                parsed_w[k] = float(v)
            else:
                parsed_w[k] = 0.0
        tot = sum(parsed_w.values())
        if tot <= 0.0:
            raise ConstraintValidationError("weights", "Weights must contain at least one positive value (cannot be all zero).")
        weights_4 = {k: v / tot for k, v in parsed_w.items()}
    elif raw_priority is not None:
        if not isinstance(raw_priority, str) or raw_priority not in KNOWN_WEIGHT_KEYS:
            raise ConstraintValidationError(
                "priority", f"Must be one of {list(PRIORITY_KEYS)} (got {raw_priority!r})."
            )
        if raw_priority == "simplicity":
            # 4-objective priority when simplicity is explicitly prioritized
            raw_p = {
                "water_saving": 0.4 / 3.0,
                "income": 0.4 / 3.0,
                "soil_restoration": 0.4 / 3.0,
                "simplicity": 0.6,
            }
        else:
            # Preserve exact legacy 0.6 / 0.2 / 0.2 behavior across the 3 core objectives
            raw_p = {
                "water_saving": 0.6 if raw_priority == "water_saving" else 0.2,
                "income": 0.6 if raw_priority == "income" else 0.2,
                "soil_restoration": 0.6 if raw_priority == "soil_restoration" else 0.2,
                "simplicity": 0.0,
            }
        tot = sum(raw_p.values())
        weights_4 = {k: v / tot for k, v in raw_p.items()}

    # 7. Commodity & nitrogen price overrides
    prices = dict(DEFAULT_PRICES)
    if "prices" in src and src["prices"] is not None:
        raw_prices = src["prices"]
        if not isinstance(raw_prices, dict):
            raise ConstraintValidationError("prices", "Must be an object mapping price keys to positive numbers.")
        for k, v in raw_prices.items():
            if k not in KNOWN_PRICE_KEYS:
                warnings.append(f"Unknown price key 'prices.{k}' was ignored.")
                continue
            if not _is_number(v):
                raise ConstraintValidationError(f"prices.{k}", "Price must be a number.")
            lo, hi = PRICE_RANGES[k]
            if not (lo <= float(v) <= hi):
                raise ConstraintValidationError(
                    f"prices.{k}",
                    f"Must be between {lo} and {hi} {PRICE_UNITS[k]} (got {v}).",
                )
            prices[k] = float(v)

    # 8. Management mode ("uniform" vs "zone_variable")
    requested_mode = DEFAULT_MANAGEMENT_MODE
    if "management_mode" in src and src["management_mode"] is not None:
        mm = src["management_mode"]
        if not isinstance(mm, str) or mm not in MANAGEMENT_MODES:
            raise ConstraintValidationError(
                "management_mode", f"Must be one of {list(MANAGEMENT_MODES)} (got {mm!r})."
            )
        requested_mode = mm

    effective_mode = requested_mode
    if requested_mode == "zone_variable" and not equipment["variable_rate"]:
        effective_mode = "uniform"
        warnings.append(
            "management_mode 'zone_variable' was requested, but equipment.variable_rate is false; falling back to 'uniform' whole-field management."
        )

    # Determine whether constraints are completely at baseline defaults
    is_default = (
        prev_crop_code == DEFAULT_PREV_CROP_CODE
        and continuous_corn_years == DEFAULT_CONTINUOUS_CORN_YEARS
        and max_cc_years == MAX_CONTINUOUS_CORN_YEARS
        and equipment == DEFAULT_EQUIPMENT
        and allow_custom_hire is True
        and len(excluded_codes) == 0
        and irrigation_cap_mm == IRRIGATION_CAP_DEFAULT_MM
        and irrigation_cost_usd_per_mm == DEFAULT_IRRIGATION_COST_USD_MM
        and abs(risk_aversion_lambda - DEFAULT_RISK_AVERSION_LAMBDA) < 1e-9
        and abs(weights_4["simplicity"]) < 1e-9
        and prices == DEFAULT_PRICES
        and effective_mode == DEFAULT_MANAGEMENT_MODE
    )

    applied_constraints: Dict[str, Any] = {
        "prev_crop": prev_crop_name,
        "prev_crop_code": prev_crop_code,
        "continuous_corn_years": continuous_corn_years,
        "max_continuous_corn_years": max_cc_years,
        "equipment": equipment,
        "allow_custom_hire": allow_custom_hire,
        "excluded_crops": excluded_names,
        "excluded_crop_codes": excluded_codes,
        "irrigation_cap_mm": round(irrigation_cap_mm, 1),
        "irrigation_cost_usd_per_mm": round(irrigation_cost_usd_per_mm, 3),
        "risk_aversion": risk_aversion_label,
        "risk_aversion_lambda": round(risk_aversion_lambda, 3),
        "weights": {k: round(v, 4) for k, v in weights_4.items()},
        "prices": {k: round(v, 3) for k, v in prices.items()},
        "price_units": dict(PRICE_UNITS),
        "management_mode": effective_mode,
        "requested_management_mode": requested_mode,
        "is_default": is_default,
    }

    return applied_constraints, warnings


def build_options_metadata() -> Dict[str, Any]:
    """Return full schema of supported farmer customization options, ranges, units, presets, and defaults."""
    return {
        "crops": [
            {"name": name, "cdl_code": code}
            for code, name in sorted(CODE_TO_CROP_NAME.items())
        ],
        "prev_crop": {
            "default": CODE_TO_CROP_NAME[DEFAULT_PREV_CROP_CODE],
            "default_cdl_code": DEFAULT_PREV_CROP_CODE,
            "supported_names": sorted(set(CODE_TO_CROP_NAME.values())),
            "supported_cdl_codes": sorted(CODE_TO_CROP_NAME.keys()),
        },
        "continuous_corn_years": {
            "default": DEFAULT_CONTINUOUS_CORN_YEARS,
            "max_recommended_before_pruning": MAX_CONTINUOUS_CORN_YEARS,
            "range": list(CONTINUOUS_CORN_RANGE),
            "unit": "years",
        },
        "equipment": {
            "defaults": dict(DEFAULT_EQUIPMENT),
            "custom_hire_costs_usd_ha": dict(CUSTOM_HIRE_COSTS_USD_HA),
            "allow_custom_hire_default": True,
            "fields": {
                "has_hay_baler": "Required for multi-cut perennial alfalfa harvest (or incurs custom-hire baling cost).",
                "has_cover_drill": "Required for drilling post-harvest cover crops.",
                "has_no_till_drill": "Required for no-till corn into terminated cereal rye residue.",
                "variable_rate": "Required to apply zone-variable management across sub-field zones.",
            },
        },
        "excluded_crops": {
            "default": [],
            "description": "List of crop names or CDL codes to exclude due to lack of local buyer or farmer preference.",
        },
        "irrigation": {
            "irrigation_cap_mm": {
                "default": IRRIGATION_CAP_DEFAULT_MM,
                "range": list(IRRIGATION_CAP_RANGE_MM),
                "unit": "mm/season",
                "description": "Maximum supplemental irrigation depth applied during Jul/Aug reproductive stress windows (0 = rainfed).",
            },
            "irrigation_cost_usd_per_mm": {
                "default": DEFAULT_IRRIGATION_COST_USD_MM,
                "range": list(IRRIGATION_COST_RANGE_USD_MM),
                "unit": "USD/mm/ha",
            },
        },
        "risk_aversion": {
            "default": "balanced",
            "default_lambda": DEFAULT_RISK_AVERSION_LAMBDA,
            "presets": dict(RISK_AVERSION_PRESETS),
            "range": list(RISK_AVERSION_RANGE),
            "description": "Penalty weight (lambda) on dry-analog downside utility deviation.",
        },
        "weights": {
            "keys": list(PRIORITY_KEYS),
            "defaults": {k: round(v, 4) for k, v in DEFAULT_WEIGHTS.items()},
            "description": "Non-negative weights normalized to sum to 1.0 across water_saving, income, soil_restoration, and simplicity.",
        },
        "prices": {
            key: {
                "default": DEFAULT_PRICES[key],
                "range": list(PRICE_RANGES[key]),
                "unit": PRICE_UNITS[key],
            }
            for key in DEFAULT_PRICES
        },
        "management_mode": {
            "options": list(MANAGEMENT_MODES),
            "default": DEFAULT_MANAGEMENT_MODE,
            "description": "'uniform' recommends a single whole-field plan; 'zone_variable' allows per-zone plan assignments when equipment.variable_rate is true.",
        },
    }

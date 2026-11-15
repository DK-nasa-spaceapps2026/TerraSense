"""Phase 3 Agronomic Crop Plan Schema, Ensemble Scoring & Dynamic Recommendation (`backend/ml/crop_plans.py`).

IMPORTANT METHODOLOGICAL NOTE:
Observed training data (USDA CDL + ERA5-Land + MERRA-2 + SSURGO) records historical
corn/soybean/forage land cover, soil properties, terrain, and weather -> soil-moisture/ET
state transitions. It does NOT observe experimental counterfactuals such as winter cover-crop
termination timing or sorghum substitution on the same parcel.
Therefore:
  1. LEARNED FROM DATA: The hybrid state-transition ML model (`smi_model`, `esi_model`)
     learns how soil texture/AWC, terrain slope/TWI, antecedent moisture, and monthly
     weather anomalies drive soil moisture and evaporative stress residuals on top of the
     water-balance bucket.
  2. AGRONOMIC PRIORS (documented below with sources): Crop coefficients (FAO-56 Paper 56
     Table 12), canopy & residue cover schedules (USDA RUSLE2 / NRCS Iowa Technical Note 25),
     legume nitrogen credits (Iowa State University Extension PM 1688 / Sawyer et al.),
     and baseline crop enterprise budgets (ISU Extension Ag Decision Maker File A1-20).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ml.config import (
    CODE_TO_CROP_NAME,
    CUSTOM_HIRE_COSTS_USD_HA,
    DEFAULT_CONTINUOUS_CORN_YEARS,
    DEFAULT_EQUIPMENT,
    DEFAULT_IRRIGATION_COST_USD_MM,
    DEFAULT_MANAGEMENT_MODE,
    DEFAULT_PREV_CROP_CODE,
    DEFAULT_PRICES,
    DEFAULT_RISK_AVERSION_LAMBDA,
    DEFAULT_WEIGHTS,
    MAX_CONTINUOUS_CORN_YEARS,
)


@dataclass(frozen=True)
class AgronomicCropPlan:
    id: str
    name: str
    crops: str
    description: str
    primary_crop_code: int               # USDA CDL code (1=Corn, 5=Soy, 4=Sorghum, 36=Alfalfa)
    plant_month: int                     # 1..12
    harvest_month: int                   # 1..12
    planting_window: str                 # Iowa agronomic window
    critical_months: Tuple[int, ...]     # Reproductive stress-sensitive months (e.g. Jul silking, Aug pod fill)
    kc_curve: Tuple[float, ...]          # 12 monthly FAO-56 Kc values (Jan..Dec)
    canopy_cover: Tuple[float, ...]      # 12 monthly green canopy fractions (0..1)
    residue_cover: Tuple[float, ...]     # 12 monthly surface residue/stubble fractions (0..1)
    cover_seed_month: Optional[int]      # Month cover crop is seeded (None if no cover crop)
    cover_term_month: Optional[int]      # Month cover crop is terminated in spring
    cc_cover: Optional[float]            # Peak cover-crop canopy fraction
    cc_kc: float                         # Cover-crop Kc during active fall/spring growth
    n_fix_kg_ha: float                   # Symbiotic N credit (kg N/ha/yr, ISU PM 1688)
    som_gain_pct_yr: float               # Annual soil organic matter gain (% SOM/yr)
    base_margin_usd_ha: float            # Gross margin above variable cost before cover seed (USD/ha)
    seed_cost_usd_ha: float              # Cover-crop / specialty seed & termination cost (USD/ha)
    drought_sens: float                  # Yield sensitivity to reproductive-window water stress (Ky, FAO-33)
    root_depth_m: float                  # Effective rooting depth (m)
    max_slope_pct: float                 # Hard constraint: maximum recommended slope (%)
    min_drainage_code: int               # Hard constraint: minimum ordinal drainage class (1..7)
    allow_after_corn: bool = True
    allow_after_soy: bool = True
    is_preset: bool = False
    # Farmer customization requirement tags & operational complexity
    requires_no_till_drill: bool = False
    requires_hay_equipment: bool = False
    requires_cover_termination: bool = False
    requires_specialty_buyer: bool = False
    # Operational complexity: count of extra field passes beyond standard single-pass plant + harvest
    #   0 = conventional cash crop (plant + harvest only)
    #   2 = post-harvest cover-crop seeding pass + spring burndown/roller-crimper termination pass
    #   4 = perennial forage with 3-4 hay cutting/raking/baling passes per season
    extra_field_passes: int = 0
    crop_codes_used: Tuple[int, ...] = (1,)
    price_key: str = "corn_usd_bu"
    tradeoffs: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def net_base_margin_usd_ha(self) -> float:
        return self.base_margin_usd_ha - self.seed_cost_usd_ha

    @property
    def simplicity_score(self) -> float:
        """Normalized operational simplicity in [0, 1] (0 extra passes -> 1.0, 4 extra passes -> 0.0)."""
        return float(np.clip(1.0 - self.extra_field_passes / 4.0, 0.0, 1.0))

    def effective_cover_array(self) -> np.ndarray:
        """Combined green canopy + surface residue armor per calendar month (0..11)."""
        c = np.array(self.canopy_cover, dtype=np.float64)
        r = np.array(self.residue_cover, dtype=np.float64)
        return np.clip(1.0 - (1.0 - c) * (1.0 - r), 0.05, 0.98)

    def kc_array(self) -> np.ndarray:
        return np.array(self.kc_curve, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Preset Crop Plans (compatible with existing `PLANS` IDs) + Candidate Pool   #
# --------------------------------------------------------------------------- #
AGRONOMIC_PLANS: Dict[str, AgronomicCropPlan] = {
    "status_quo": AgronomicCropPlan(
        id="status_quo",
        name="Status Quo",
        crops="Corn → Soybean, winter fallow",
        description="Continue conventional rotation with bare winter fallow.",
        primary_crop_code=1,
        plant_month=5,
        harvest_month=10,
        planting_window="Apr 20 – May 15 (Corn) / May 1 – May 20 (Soy)",
        critical_months=(7, 8),  # Jul corn silking + Aug early grain fill
        # Jan..Dec FAO-56 Kc for Iowa Corn/Soy conventional fallow
        kc_curve=(0.25, 0.25, 0.25, 0.30, 0.45, 0.82, 1.15, 1.10, 0.85, 0.45, 0.25, 0.25),
        canopy_cover=(0.0, 0.0, 0.0, 0.02, 0.22, 0.68, 0.94, 0.92, 0.65, 0.20, 0.0, 0.0),
        residue_cover=(0.10, 0.10, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.10, 0.15, 0.12, 0.10),
        cover_seed_month=None,
        cover_term_month=None,
        cc_cover=None,
        cc_kc=0.0,
        n_fix_kg_ha=0.0,
        som_gain_pct_yr=0.02,
        base_margin_usd_ha=820.0,
        seed_cost_usd_ha=0.0,
        drought_sens=1.15,
        root_depth_m=1.2,
        max_slope_pct=5.5,
        min_drainage_code=2,
        allow_after_corn=True,
        allow_after_soy=True,
        is_preset=True,
        requires_no_till_drill=False,
        requires_hay_equipment=False,
        requires_cover_termination=False,
        requires_specialty_buyer=False,
        extra_field_passes=0,  # Standard single-pass planting & harvest (0 extra passes)
        crop_codes_used=(1, 5),
        price_key="corn_usd_bu",
        tradeoffs={
            "pros": ["Familiar management", "Established markets"],
            "cons": ["Bare winter soil", "Highest water demand", "Synthetic N required"],
        },
    ),
    "cover_legume": AgronomicCropPlan(
        id="cover_legume",
        name="Legume + Cover Crop",
        crops="Soybean → crimson clover / rye cover",
        description="Add a legume cover crop after harvest to fix N and armour soil.",
        primary_crop_code=5,
        plant_month=5,
        harvest_month=9,
        planting_window="May 1 – May 25 (Soybean) + Sep/Oct Clover/Rye Over-seed",
        critical_months=(7, 8),  # Aug soybean pod set & seed fill
        kc_curve=(0.42, 0.42, 0.48, 0.52, 0.42, 0.78, 1.05, 1.02, 0.68, 0.45, 0.42, 0.42),
        canopy_cover=(0.55, 0.55, 0.62, 0.68, 0.25, 0.62, 0.88, 0.88, 0.55, 0.42, 0.52, 0.55),
        residue_cover=(0.25, 0.25, 0.25, 0.28, 0.30, 0.22, 0.18, 0.18, 0.25, 0.32, 0.28, 0.25),
        cover_seed_month=9,
        cover_term_month=4,
        cc_cover=0.70,
        cc_kc=0.50,
        n_fix_kg_ha=75.0,
        som_gain_pct_yr=0.14,
        base_margin_usd_ha=805.0,
        seed_cost_usd_ha=60.0,  # Net 745 USD/ha matching preset baseline
        drought_sens=0.85,
        root_depth_m=1.1,
        max_slope_pct=12.0,
        min_drainage_code=1,
        allow_after_corn=True,
        allow_after_soy=False,  # Avoid continuous soybean (SCN / disease risk)
        is_preset=True,
        requires_no_till_drill=False,
        requires_hay_equipment=False,
        requires_cover_termination=True,
        requires_specialty_buyer=False,
        extra_field_passes=2,  # +1 fall cover-seeding pass, +1 spring cover-termination pass
        crop_codes_used=(5, 58, 27),
        price_key="soybean_usd_bu",
        tradeoffs={
            "pros": ["Fixes nitrogen", "Best erosion control", "Builds organic matter"],
            "cons": ["Cover seed cost", "Cover crop uses spring water"],
        },
    ),
    "drought_cash": AgronomicCropPlan(
        id="drought_cash",
        name="Drought-Tolerant Cash Crop",
        crops="Sorghum / sunflower, stubble retained",
        description="Switch to low-water cash crops with retained stubble.",
        primary_crop_code=4,
        plant_month=6,
        harvest_month=9,
        planting_window="May 20 – Jun 15 (Warm-soil Sorghum / Sunflower)",
        critical_months=(7, 8),
        kc_curve=(0.25, 0.25, 0.25, 0.25, 0.28, 0.45, 0.82, 0.85, 0.58, 0.25, 0.25, 0.25),
        canopy_cover=(0.0, 0.0, 0.0, 0.0, 0.05, 0.35, 0.78, 0.80, 0.52, 0.08, 0.0, 0.0),
        residue_cover=(0.32, 0.32, 0.30, 0.28, 0.28, 0.25, 0.22, 0.22, 0.28, 0.38, 0.35, 0.32),
        cover_seed_month=None,
        cover_term_month=None,
        cc_cover=None,
        cc_kc=0.0,
        n_fix_kg_ha=0.0,
        som_gain_pct_yr=0.05,
        base_margin_usd_ha=690.0,
        seed_cost_usd_ha=0.0,
        drought_sens=0.45,
        root_depth_m=1.5,
        max_slope_pct=9.0,
        min_drainage_code=3,  # Prefers moderately well to well-drained soils
        allow_after_corn=True,
        allow_after_soy=True,
        is_preset=True,
        requires_no_till_drill=False,
        requires_hay_equipment=False,
        requires_cover_termination=False,
        requires_specialty_buyer=True,
        extra_field_passes=0,  # Standard row-crop planting & harvest with high stubble cut (0 extra passes)
        crop_codes_used=(4, 6),
        price_key="sorghum_usd_bu",
        tradeoffs={
            "pros": ["Lowest water demand", "Resilient in heat waves"],
            "cons": ["Lower baseline margin", "Fewer local buyers"],
        },
    ),
    # Additional discrete candidate plans for dynamic pruning & ranking
    "corn_rye_cover": AgronomicCropPlan(
        id="corn_rye_cover",
        name="No-Till Corn + Winter Rye Armor",
        crops="No-till Corn → winter cereal rye cover",
        description="High-residue no-till corn followed by drilled winter cereal rye terminated 10–14 days before planting.",
        primary_crop_code=1,
        plant_month=5,
        harvest_month=10,
        planting_window="Apr 25 – May 15 (Corn after Rye burndown)",
        critical_months=(7,),
        kc_curve=(0.35, 0.35, 0.42, 0.48, 0.44, 0.80, 1.12, 1.08, 0.82, 0.45, 0.38, 0.35),
        canopy_cover=(0.42, 0.42, 0.50, 0.58, 0.22, 0.68, 0.92, 0.90, 0.62, 0.30, 0.40, 0.42),
        residue_cover=(0.35, 0.35, 0.35, 0.35, 0.32, 0.28, 0.25, 0.25, 0.30, 0.40, 0.38, 0.35),
        cover_seed_month=10,
        cover_term_month=4,
        cc_cover=0.58,
        cc_kc=0.42,
        n_fix_kg_ha=15.0,  # Scavenged residual nitrate credit
        som_gain_pct_yr=0.11,
        base_margin_usd_ha=830.0,
        seed_cost_usd_ha=48.0,
        drought_sens=0.95,
        root_depth_m=1.25,
        max_slope_pct=10.0,
        min_drainage_code=2,
        allow_after_corn=False,  # Pruned if previous crop was corn (avoid corn-on-corn rye tie-up)
        allow_after_soy=True,
        is_preset=False,
        requires_no_till_drill=True,
        requires_hay_equipment=False,
        requires_cover_termination=True,
        requires_specialty_buyer=False,
        extra_field_passes=2,  # +1 fall no-till rye drilling pass, +1 spring burndown pass
        crop_codes_used=(1, 27),
        price_key="corn_usd_bu",
        tradeoffs={
            "pros": ["High cash-crop revenue after soybean", "Strong winter/spring erosion armor", "Scavenges residual N"],
            "cons": ["Requires timely spring rye termination 10–14 days pre-plant", "Cover seed cost (~$48/ha)"],
        },
    ),
    "alfalfa_perennial": AgronomicCropPlan(
        id="alfalfa_perennial",
        name="Perennial Alfalfa Meadow",
        crops="Alfalfa / meadow brome forage rotation",
        description="Deep-rooted perennial legume forage for steep or erosion-vulnerable zones.",
        primary_crop_code=36,
        plant_month=4,
        harvest_month=9,
        planting_window="Apr 10 – May 10 (Spring direct seed)",
        critical_months=(6, 7, 8),
        kc_curve=(0.45, 0.45, 0.55, 0.75, 0.95, 1.05, 1.05, 1.00, 0.85, 0.60, 0.45, 0.45),
        canopy_cover=(0.65, 0.65, 0.72, 0.82, 0.88, 0.92, 0.88, 0.88, 0.82, 0.75, 0.68, 0.65),
        residue_cover=(0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35),
        cover_seed_month=None,
        cover_term_month=None,
        cc_cover=0.80,
        cc_kc=0.50,
        n_fix_kg_ha=140.0,
        som_gain_pct_yr=0.20,
        base_margin_usd_ha=725.0,
        seed_cost_usd_ha=45.0,
        drought_sens=0.55,
        root_depth_m=1.8,
        max_slope_pct=18.0,
        min_drainage_code=4,  # Alfalfa requires moderately well to well-drained soil (wet feet heaving risk)
        allow_after_corn=True,
        allow_after_soy=True,
        is_preset=False,
        requires_no_till_drill=False,
        requires_hay_equipment=True,
        requires_cover_termination=False,
        requires_specialty_buyer=False,
        extra_field_passes=4,  # 3-4 multi-cut hay harvesting & baling passes per season
        crop_codes_used=(36, 37),
        price_key="alfalfa_usd_ton",
        tradeoffs={
            "pros": ["Highest N fixation & SOM building", "Year-round living root cover on steep slopes"],
            "cons": ["Requires well-drained soil (drainage code >= 4)", "Multi-cut hay harvesting logistics"],
        },
    ),
}

PRESET_PLAN_IDS: List[str] = ["status_quo", "cover_legume", "drought_cash"]

PLAN_METHODOLOGY_DOC: Dict[str, Any] = {
    "learned_from_data": [
        "Monthly state transition residuals for Soil Moisture Index (SMI) and Evaporative Stress Index (ESI) conditioned on SSURGO soil horizon properties, metric DEM terrain (slope, TWI, relief), and weather anomalies.",
        "Out-of-fold conformal uncertainty quantiles (P10/P50/P90) and 2023/2024 analog-year weather variability.",
    ],
    "agronomic_priors": [
        "Stage-based monthly crop coefficients (Kc) from FAO Irrigation & Drainage Paper 56 (Table 12).",
        "Canopy and surface residue cover schedules from USDA-NRCS RUSLE2 / Iowa Agronomy Technical Note 25.",
        "Legume symbiotic N fixation credits (kg N/ha) from ISU Extension PM 1688.",
        "Baseline crop enterprise margins and cover-crop seeding costs from ISU Ag Decision Maker (File A1-20) and Iowa Farm Custom Rate Survey (File A3-10).",
    ],
}


def prune_candidate_plans_detailed(
    prev_crop_code: int = DEFAULT_PREV_CROP_CODE,
    mean_slope_pct: float = 3.0,
    mean_drainage_code: float = 4.0,
    continuous_corn_years: int = DEFAULT_CONTINUOUS_CORN_YEARS,
    max_continuous_corn: int = MAX_CONTINUOUS_CORN_YEARS,
    equipment: Optional[Dict[str, bool]] = None,
    excluded_crop_codes: Optional[List[int]] = None,
    allow_custom_hire: bool = True,
) -> Dict[str, Any]:
    """Filter discrete candidate plans against rotation, terrain, equipment, and market constraints.

    Never returns an empty feasible list: if all candidate plans violate hard constraints,
    selects the least-violating plan(s) and flags `constraint_violation=True`.
    Where custom hire is realistic (e.g. perennial alfalfa without a baler when `allow_custom_hire=True`),
    keeps the plan feasible and records a documented custom-hire cost in `custom_hire_surcharges`.
    """
    eq = dict(DEFAULT_EQUIPMENT)
    if equipment:
        eq.update(equipment)
    excl = set(excluded_crop_codes or [])

    feasible: List[AgronomicCropPlan] = []
    pruned_reasons: Dict[str, str] = {}
    pruned_reasons_list: Dict[str, List[str]] = {}
    custom_hire_surcharges: Dict[str, float] = {}
    custom_hire_notes: Dict[str, str] = {}

    for pid, plan in AGRONOMIC_PLANS.items():
        reasons: List[str] = []

        # 1. Crop rotation rules
        if prev_crop_code == 5 and not plan.allow_after_soy:
            reasons.append("Soy-after-soy prohibited (SCN & disease buildup)")
        if prev_crop_code == 1 and not plan.allow_after_corn:
            reasons.append("Requires prior soybean residue (high C:N corn residue tie-up)")
        if prev_crop_code == 1 and plan.primary_crop_code == 1 and continuous_corn_years >= max_continuous_corn:
            reasons.append(f"Exceeds continuous-corn limit ({continuous_corn_years} >= {max_continuous_corn} yrs)")

        # 2. Field slope & drainage limits
        if mean_slope_pct > plan.max_slope_pct:
            reasons.append(f"Field slope ({mean_slope_pct:.1f}%) exceeds plan limit ({plan.max_slope_pct:.1f}%)")
        if mean_drainage_code < plan.min_drainage_code:
            reasons.append(
                f"Soil drainage class ({mean_drainage_code:.1f}) too poorly drained for {plan.name} (min {plan.min_drainage_code})"
            )

        # 3. Excluded crops / no local buyer
        matched_excl = [c for c in plan.crop_codes_used if c in excl]
        if not matched_excl and plan.primary_crop_code in excl:
            matched_excl = [plan.primary_crop_code]
        if matched_excl:
            crop_labels = ", ".join(CODE_TO_CROP_NAME.get(c, str(c)) for c in matched_excl)
            reasons.append(f"Excluded crop ({crop_labels}) has no local buyer or was excluded by farmer")

        # 4. Equipment availability & realistic custom-hire fallback
        if plan.requires_no_till_drill and not eq.get("has_no_till_drill", True):
            reasons.append("Requires no-till drill (equipment.has_no_till_drill is false)")
        if plan.requires_cover_termination and not eq.get("has_cover_drill", True):
            reasons.append("Requires cover-crop drill/seeder (equipment.has_cover_drill is false)")
        if plan.requires_hay_equipment and not eq.get("has_hay_baler", True):
            if allow_custom_hire:
                surcharge = CUSTOM_HIRE_COSTS_USD_HA["hay_baler"]
                custom_hire_surcharges[pid] = surcharge
                custom_hire_notes[pid] = (
                    f"No hay baler owned: kept feasible via custom-hire mowing/baling (+${surcharge:.0f}/ha, ISU File A3-10)."
                )
            else:
                reasons.append("Requires hay baling equipment (equipment.has_hay_baler is false and custom hire disabled)")

        if reasons:
            pruned_reasons_list[pid] = reasons
            pruned_reasons[pid] = "; ".join(reasons)
        else:
            feasible.append(plan)

    # Fallback if every plan was pruned: return the least-violating plan(s) flagged `constraint_violation`
    constraint_violation_ids: List[str] = []
    if not feasible:
        min_violations = min(len(r) for r in pruned_reasons_list.values())
        for pid, plan in AGRONOMIC_PLANS.items():
            if len(pruned_reasons_list[pid]) == min_violations:
                feasible.append(plan)
                constraint_violation_ids.append(pid)

    pruned_plans_structured = [
        {
            "id": pid,
            "name": AGRONOMIC_PLANS[pid].name,
            "reason": pruned_reasons[pid],
            "reasons": pruned_reasons_list[pid],
            "constraint_violation_fallback": pid in constraint_violation_ids,
        }
        for pid in pruned_reasons
    ]

    return {
        "feasible_plans": feasible,
        "pruned_reasons": pruned_reasons,
        "pruned_plans": pruned_plans_structured,
        "custom_hire_surcharges": custom_hire_surcharges,
        "custom_hire_notes": custom_hire_notes,
        "constraint_violation": len(constraint_violation_ids) > 0,
        "constraint_violation_ids": constraint_violation_ids,
    }


def prune_candidate_plans(
    prev_crop_code: int = DEFAULT_PREV_CROP_CODE,
    mean_slope_pct: float = 3.0,
    mean_drainage_code: float = 4.0,
    continuous_corn_years: int = DEFAULT_CONTINUOUS_CORN_YEARS,
    max_continuous_corn: int = MAX_CONTINUOUS_CORN_YEARS,
    equipment: Optional[Dict[str, bool]] = None,
    excluded_crop_codes: Optional[List[int]] = None,
    allow_custom_hire: bool = True,
) -> Tuple[List[AgronomicCropPlan], Dict[str, str]]:
    """Filter discrete candidate plans against hard agronomic constraints.

    Returns `(feasible_plans, pruned_reasons_by_id)` for backward compatibility.
    """
    res = prune_candidate_plans_detailed(
        prev_crop_code=prev_crop_code,
        mean_slope_pct=mean_slope_pct,
        mean_drainage_code=mean_drainage_code,
        continuous_corn_years=continuous_corn_years,
        max_continuous_corn=max_continuous_corn,
        equipment=equipment,
        excluded_crop_codes=excluded_crop_codes,
        allow_custom_hire=allow_custom_hire,
    )
    return res["feasible_plans"], res["pruned_reasons"]


def _norm(v: np.ndarray, higher_better: bool = True) -> np.ndarray:
    rng = v.max(axis=0) - v.min(axis=0)
    n = np.where(rng < 1e-9, 0.5, (v - v.min(axis=0)) / np.where(rng < 1e-9, 1.0, rng))
    return n if higher_better else 1.0 - n


def _score_plan_list(
    plan_ids: List[str],
    per_ensemble_raw: Dict[str, List[Dict[str, np.ndarray]]],
    weights: Dict[str, float],
    risk_aversion_lambda: float,
    prices: Dict[str, float],
    custom_hire_surcharges: Dict[str, float],
    irrigation_cost_usd_per_mm: float,
) -> Dict[str, Any]:
    """Compute risk-adjusted scores and adjusted margins for a list of plan IDs from pre-simulated trajectories."""
    w_water = float(weights.get("water_saving", 1.0 / 3.0))
    w_inc = float(weights.get("income", 1.0 / 3.0))
    w_soil = float(weights.get("soil_restoration", 1.0 / 3.0))
    w_simp = float(weights.get("simplicity", 0.0))

    n_price_delta = float(prices.get("nitrogen_price_usd_kg", DEFAULT_PRICES["nitrogen_price_usd_kg"])) - DEFAULT_PRICES["nitrogen_price_usd_kg"]

    def _adjust_member(pid: str, m: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        plan = AGRONOMIC_PLANS[pid]
        p_mult = float(prices.get(plan.price_key, DEFAULT_PRICES[plan.price_key])) / DEFAULT_PRICES[plan.price_key]
        gross_rev_adj = m["gross_rev_base"] * p_mult
        n_bonus = n_price_delta * m["n_fix"]
        ch_cost = float(custom_hire_surcharges.get(pid, 0.0))
        ir_cost = m["irrigation_mm"] * irrigation_cost_usd_per_mm
        adj_margin = np.maximum(50.0, gross_rev_adj - plan.seed_cost_usd_ha + n_bonus - ch_cost - ir_cost)
        return {
            **m,
            "margin": adj_margin,
            "custom_hire_cost": np.full_like(adj_margin, ch_cost),
            "irrigation_cost": ir_cost,
            "nitrogen_credit_usd": m["n_fix"] * float(prices.get("nitrogen_price_usd_kg", DEFAULT_PRICES["nitrogen_price_usd_kg"])),
            "simplicity": np.full_like(adj_margin, plan.simplicity_score),
        }

    def _score_group(group_metrics: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        stk = lambda k: np.stack([m[k] for m in group_metrics])
        c_water = _norm(stk("water_demand"), False)
        c_income = _norm(stk("margin"), True)
        c_soil = 0.5 * _norm(stk("som"), True) + 0.3 * _norm(stk("erosion_control"), True) + 0.2 * _norm(stk("n_fix"), True)
        c_simp = stk("simplicity")
        total = w_water * c_water + w_inc * c_income + w_soil * c_soil
        if w_simp > 0.0:
            total = total + w_simp * c_simp
        return total, {
            "water_saving": c_water,
            "income": c_income,
            "soil_restoration": c_soil,
            "simplicity": c_simp,
        }

    # Check if all plans in plan_ids have all 3 ensemble members
    has_full_ensemble = all(len(per_ensemble_raw[pid]) >= 3 for pid in plan_ids)
    grp_nom = [_adjust_member(pid, per_ensemble_raw[pid][0]) for pid in plan_ids]
    z_nom, comp_nom = _score_group(grp_nom)

    if has_full_ensemble:
        grp_dry = [_adjust_member(pid, per_ensemble_raw[pid][1]) for pid in plan_ids]
        grp_wet = [_adjust_member(pid, per_ensemble_raw[pid][2]) for pid in plan_ids]
        z_dry, _ = _score_group(grp_dry)
        z_wet, _ = _score_group(grp_wet)
        z_expected = 0.50 * z_nom + 0.25 * z_dry + 0.25 * z_wet
        downside = np.maximum(0.0, z_expected - z_dry)
        zone_score = np.clip(z_expected - risk_aversion_lambda * downside, 0.0, 1.0)
        ens_field_scores = np.stack([z_nom.mean(axis=1), z_dry.mean(axis=1), z_wet.mean(axis=1)], axis=1)
    else:
        downside = np.zeros_like(z_nom)
        zone_score = np.clip(z_nom, 0.0, 1.0)
        ens_field_scores = np.stack([z_nom.mean(axis=1)] * 3, axis=1)

    field_score = zone_score.mean(axis=1)
    return {
        "grp_nom": grp_nom,
        "comp_nom": comp_nom,
        "downside": downside,
        "zone_score": zone_score,
        "field_score": field_score,
        "ens_field_scores": ens_field_scores,
    }


def evaluate_and_rank_plans(
    grid: Dict[str, Any],
    climate: Dict[str, Any],
    prefs: Dict[str, Any],
    simulate_fn: Callable[[Dict[str, np.ndarray], List[dict], List[str]], Dict[str, np.ndarray]],
    preference_weights_fn: Callable[[dict], Dict[str, float]],
    risk_aversion_lambda: float = DEFAULT_RISK_AVERSION_LAMBDA,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate all 3 preset plans + dynamic candidates over a weather ensemble (nominal, dry analog, wet analog).

    Computes per-zone and field-level expected metrics, critical reproductive-window stress,
    risk-adjusted multi-objective scores (Mean Utility - lambda * Downside Deviation),
    per-component score breakdowns, ranking sensitivity vs defaults, and robustness diagnostics.
    """
    n_hist = 12
    n_total = 24
    f_slice = slice(n_hist, n_total)
    nz = len(grid["soil"]["awc"])

    # Extract effective constraints or fall back to baseline defaults
    c_dict = constraints or {}
    if "weights" in c_dict and isinstance(c_dict["weights"], dict):
        w_full = dict(DEFAULT_WEIGHTS)
        w_full.update({k: float(v) for k, v in c_dict["weights"].items() if k in w_full})
    else:
        w_base = preference_weights_fn(prefs or {})
        w_full = {
            "water_saving": float(w_base.get("water_saving", 1.0 / 3.0)),
            "income": float(w_base.get("income", 1.0 / 3.0)),
            "soil_restoration": float(w_base.get("soil_restoration", 1.0 / 3.0)),
            "simplicity": float(w_base.get("simplicity", 0.0)),
        }

    eff_lambda = float(c_dict.get("risk_aversion_lambda", risk_aversion_lambda))
    eff_prices = dict(DEFAULT_PRICES)
    if isinstance(c_dict.get("prices"), dict):
        eff_prices.update(c_dict["prices"])

    eff_prev_crop = int(
        c_dict.get("prev_crop_code", grid["soil"].get("prev_crop_code", np.full(nz, DEFAULT_PREV_CROP_CODE))[0])
    )
    eff_cc_years = int(c_dict.get("continuous_corn_years", DEFAULT_CONTINUOUS_CORN_YEARS))
    eff_max_cc = int(c_dict.get("max_continuous_corn_years", MAX_CONTINUOUS_CORN_YEARS))
    eff_equipment = dict(DEFAULT_EQUIPMENT)
    if isinstance(c_dict.get("equipment"), dict):
        eff_equipment.update(c_dict["equipment"])
    eff_excl_codes = list(c_dict.get("excluded_crop_codes", []))
    eff_allow_custom_hire = bool(c_dict.get("allow_custom_hire", True))
    eff_irrig_cap_mm = float(c_dict.get("irrigation_cap_mm", 0.0))
    eff_irrig_cost = float(c_dict.get("irrigation_cost_usd_per_mm", DEFAULT_IRRIGATION_COST_USD_MM))
    eff_mgmt_mode = str(c_dict.get("management_mode", DEFAULT_MANAGEMENT_MODE))

    # Weather ensemble members for forecast months (idx 12..23):
    #   0: nominal forecast scenario
    #   1: hot/dry analog perturbation (+1.5 C, 0.80x precip, akin to 2023 late-summer stress)
    #   2: cool/wet analog perturbation (-0.8 C, 1.20x precip, akin to 2024 spring/summer)
    ensemble_shifts = [
        ("nominal", 0.0, 1.00, 0.50),
        ("dry_analog", 1.5, 0.80, 0.25),
        ("wet_analog", -0.8, 1.20, 0.25),
    ]

    def _perturbed_months(t_shift: float, p_fac: float) -> List[dict]:
        if t_shift == 0.0 and p_fac == 1.0:
            return climate["months"]
        out_m = []
        for idx, m in enumerate(climate["months"]):
            if idx < n_hist:
                out_m.append(m)
            else:
                cm = dict(m)
                cm["tmax"] = cm["tmax"] + t_shift
                cm["tmin"] = cm["tmin"] + t_shift
                cm["tmean"] = 0.5 * (cm["tmax"] + cm["tmin"])
                cm["precip_mm"] = cm["precip_mm"] * p_fac
                cm["rh"] = float(np.clip(cm["rh"] - 2.0 * t_shift, 20.0, 98.0))
                cm["pet_mm"] = max(0.2, 0.0045 * (cm["tmean"] + 17.8) * cm["solar"]) * cm.get("days", 30)
                out_m.append(cm)
        return out_m

    all_plan_ids = list(AGRONOMIC_PLANS.keys())
    per_ensemble: Dict[str, List[Dict[str, np.ndarray]]] = {pid: [] for pid in all_plan_ids}

    import zone_engine as _ze
    for pid, plan in AGRONOMIC_PLANS.items():
        if pid not in _ze.CALENDARS:
            _ze.CALENDARS[pid] = {
                "cover": plan.effective_cover_array(),
                "kc": plan.kc_array(),
            }

    for ens_idx, (_, t_shift, p_fac, _) in enumerate(ensemble_shifts):
        m_list = _perturbed_months(t_shift, p_fac)
        # Simulate all 5 plans on nominal forecast, and the 3 presets on dry/wet analogs
        pids_for_member = all_plan_ids if ens_idx == 0 else PRESET_PLAN_IDS
        for pid in pids_for_member:
            plan = AGRONOMIC_PLANS[pid]
            plan_seq = ["status_quo"] * n_hist + [pid] * (n_total - n_hist)
            sim = simulate_fn(grid["soil"], m_list, plan_seq)

            fc_months = m_list[f_slice]
            season_idx = [i for i, c in enumerate(fc_months) if plan.plant_month <= c["cal_month"] <= plan.harvest_month]
            crit_idx = [i for i, c in enumerate(fc_months) if c["cal_month"] in plan.critical_months]
            if not crit_idx:
                crit_idx = season_idx

            esi_season = sim["esi"][f_slice][season_idx].mean(axis=0)
            esi_crit = sim["esi"][f_slice][crit_idx].mean(axis=0)
            # Weight reproductive critical-window stress (silking / pod fill) 60% and full-season 40%
            stress = 0.40 * (1.0 - np.clip(esi_season, 0.0, 1.0)) + 0.60 * (1.0 - np.clip(esi_crit, 0.0, 1.0))
            crit_stress = 1.0 - np.clip(esi_crit, 0.0, 1.0)
            water_stress_season = 1.0 - np.clip(esi_season, 0.0, 1.0)
            heat_stress_season = sim["heat"][f_slice][season_idx].mean(axis=0)

            et_year = sim["et"][f_slice].sum(axis=0)
            runoff = sim["runoff"][f_slice].sum(axis=0)
            bare = sim["bare"][f_slice].mean(axis=0)
            irrig_mm = (
                sim["irrigation_mm"][f_slice].sum(axis=0)
                if "irrigation_mm" in sim
                else np.zeros(nz, dtype=np.float64)
            )
            extrap_flag = (
                bool(np.any(sim["extrapolated"][f_slice]))
                if "extrapolated" in sim
                else False
            )

            n_fix = plan.n_fix_kg_ha * (0.7 + 0.3 * np.clip(esi_season, 0.0, 1.0))
            som = (
                plan.som_gain_pct_yr
                * (0.8 + 0.4 * sim["vigor"][f_slice].mean(axis=0))
                * (0.9 + 0.1 * grid["soil"]["som"] / 3.0)
            )
            gross_rev_base = plan.base_margin_usd_ha * np.clip(1.0 - plan.drought_sens * stress * 1.6, 0.35, 1.05)
            risk = np.clip(plan.drought_sens * stress * 1.4 + 0.35 * bare, 0.0, 1.0)
            erosion_control = 100.0 * (1.0 - bare)

            per_ensemble[pid].append({
                "water_demand": sim["demand"][f_slice].sum(axis=0),
                "water_balance": sim["precip"][f_slice].sum(axis=0) + irrig_mm - et_year - runoff,
                "et": et_year,
                "bare": bare,
                "stress": stress,
                "water_stress": water_stress_season,
                "heat_stress": heat_stress_season,
                "crit_stress": crit_stress,
                "n_fix": n_fix,
                "som": som,
                "gross_rev_base": gross_rev_base,
                "irrigation_mm": irrig_mm,
                "extrapolated": np.full(nz, float(extrap_flag)),
                "risk": risk,
                "erosion_control": erosion_control,
            })

    # Dynamic candidate pruning with farmer constraints
    mean_slope = float(grid["soil"]["slope"].mean())
    mean_drainage = float(grid["soil"].get("drainage_code", np.full(nz, 4.0)).mean())
    prune_info = prune_candidate_plans_detailed(
        prev_crop_code=eff_prev_crop,
        mean_slope_pct=mean_slope,
        mean_drainage_code=mean_drainage,
        continuous_corn_years=eff_cc_years,
        max_continuous_corn=eff_max_cc,
        equipment=eff_equipment,
        excluded_crop_codes=eff_excl_codes,
        allow_custom_hire=eff_allow_custom_hire,
    )
    pruned_reasons = prune_info["pruned_reasons"]
    custom_hire_surcharges = prune_info["custom_hire_surcharges"]
    custom_hire_notes = prune_info["custom_hire_notes"]
    constraint_violation_ids = set(prune_info["constraint_violation_ids"])

    # Score the 3 compatibility presets under current farmer constraints
    scored_presets = _score_plan_list(
        plan_ids=PRESET_PLAN_IDS,
        per_ensemble_raw=per_ensemble,
        weights=w_full,
        risk_aversion_lambda=eff_lambda,
        prices=eff_prices,
        custom_hire_surcharges=custom_hire_surcharges,
        irrigation_cost_usd_per_mm=eff_irrig_cost,
    )
    field_score = scored_presets["field_score"]
    zone_score = scored_presets["zone_score"]
    ens_field_scores = scored_presets["ens_field_scores"]
    grp_nom = scored_presets["grp_nom"]
    comp_nom = scored_presets["comp_nom"]
    downside = scored_presets["downside"]

    # Determine feasible preset indices; if a farmer constraint pruned a preset, prefer feasible presets
    feasible_preset_indices = [
        i for i, pid in enumerate(PRESET_PLAN_IDS)
        if (pid not in pruned_reasons) or (pid in constraint_violation_ids)
    ]
    if not feasible_preset_indices:
        feasible_preset_indices = list(range(len(PRESET_PLAN_IDS)))
        for pid in PRESET_PLAN_IDS:
            constraint_violation_ids.add(pid)

    best_idx = max(feasible_preset_indices, key=lambda i: float(field_score[i]))

    # Ensemble spread diagnostics for uncertainty & robustness reporting
    if len(feasible_preset_indices) >= 2:
        sorted_feas_scores = sorted([float(field_score[i]) for i in feasible_preset_indices], reverse=True)
        top_gap = float(sorted_feas_scores[0] - sorted_feas_scores[1])
    else:
        sorted_scores = np.sort(field_score)[::-1]
        top_gap = float(sorted_scores[0] - sorted_scores[1])

    top_spread = float(ens_field_scores[best_idx].max() - ens_field_scores[best_idx].min())
    ranking_robust = bool(top_gap >= max(0.03, 0.5 * top_spread))
    robust_reason = (
        f"Top plan score margin ({top_gap:.3f}) exceeds weather-ensemble spread threshold ({max(0.03, 0.5 * top_spread):.3f})."
        if ranking_robust
        else f"Rank 1 and Rank 2 differ by {top_gap:.3f}, which is within the weather-ensemble spread ({top_spread:.3f})."
    )

    def _risk_label(score: float) -> str:
        return "low" if score < 0.30 else "medium" if score < 0.55 else "high"

    scenarios_out = []
    for i, pid in enumerate(PRESET_PLAN_IDS):
        m = grp_nom[i]
        plan = AGRONOMIC_PLANS[pid]
        mean_fn = lambda k: float(m[k].mean())
        is_feas = pid not in pruned_reasons
        is_viol_fallback = pid in constraint_violation_ids and not is_feas

        sb: Dict[str, Any] = {
            "water_saving_component": round(float((w_full["water_saving"] * comp_nom["water_saving"][i]).mean()), 3),
            "income_component": round(float((w_full["income"] * comp_nom["income"][i]).mean()), 3),
            "soil_restoration_component": round(float((w_full["soil_restoration"] * comp_nom["soil_restoration"][i]).mean()), 3),
            "simplicity_component": round(float((w_full["simplicity"] * comp_nom["simplicity"][i]).mean()), 3),
            "downside_risk_penalty": round(float((eff_lambda * downside[i]).mean()), 3),
            "water_stress_pct": round(100.0 * mean_fn("water_stress"), 1),
            "heat_stress_pct": round(100.0 * mean_fn("heat_stress"), 1),
            "critical_window_stress_pct": round(100.0 * mean_fn("crit_stress"), 1),
            "erosion_control_score": round(mean_fn("erosion_control"), 0),
            "nitrogen_credit_kg_ha": round(mean_fn("n_fix"), 0),
            "nitrogen_credit_usd_ha": round(mean_fn("nitrogen_credit_usd"), 1),
            "estimated_margin_usd_ha": round(mean_fn("margin"), 0),
            "extra_field_passes": plan.extra_field_passes,
            "simplicity_score": round(plan.simplicity_score, 2),
            "custom_hire_cost_usd_ha": round(mean_fn("custom_hire_cost"), 1),
            "irrigation_applied_mm": round(mean_fn("irrigation_mm"), 1),
            "irrigation_cost_usd_ha": round(mean_fn("irrigation_cost"), 1),
            "ensemble_score_range": [
                round(float(ens_field_scores[i].min()), 3),
                round(float(ens_field_scores[i].max()), 3),
            ],
        }

        scen_entry: Dict[str, Any] = {
            "id": pid,
            "name": plan.name,
            "crops": plan.crops,
            "description": plan.description,
            "score": round(float(field_score[i]), 3),
            "recommended": i == best_idx,
            "feasible": is_feas,
            "pruned_reason": pruned_reasons.get(pid),
            "constraint_violation": is_viol_fallback,
            "robust": ranking_robust if i == best_idx else False,
            "robust_reason": robust_reason if i == best_idx else None,
            "metrics": {
                "soil_organic_matter_gain_pct": round(mean_fn("som"), 2),
                "nitrogen_fixed_kg_ha": round(mean_fn("n_fix"), 0),
                "water_demand_mm": round(mean_fn("water_demand"), 0),
                "water_budget_balance_mm": round(mean_fn("water_balance"), 0),
                "erosion_control_score": round(mean_fn("erosion_control"), 0),
                "estimated_margin_usd_ha": round(mean_fn("margin"), 0),
                "risk_level": _risk_label(mean_fn("risk")),
                "risk_score": round(mean_fn("risk"), 2),
                "critical_window_stress_pct": round(100.0 * mean_fn("crit_stress"), 1),
                "irrigation_applied_mm": round(mean_fn("irrigation_mm"), 1),
                "extra_field_passes": plan.extra_field_passes,
            },
            "score_breakdown": sb,
            "planting_window": plan.planting_window,
            "tradeoffs": plan.tradeoffs,
        }
        if pid in custom_hire_notes:
            scen_entry["custom_hire_note"] = custom_hire_notes[pid]
        scenarios_out.append(scen_entry)

    # Dynamic candidate scoring across all 5 candidate plans
    scored_all = _score_plan_list(
        plan_ids=all_plan_ids,
        per_ensemble_raw=per_ensemble,
        weights=w_full,
        risk_aversion_lambda=eff_lambda,
        prices=eff_prices,
        custom_hire_surcharges=custom_hire_surcharges,
        irrigation_cost_usd_per_mm=eff_irrig_cost,
    )
    cand_scores = scored_all["field_score"]
    all_nom_adj = scored_all["grp_nom"]

    dynamic_candidates = []
    for idx_all, pid in enumerate(all_plan_ids):
        p = AGRONOMIC_PLANS[pid]
        is_feas = pid not in pruned_reasons
        is_viol = pid in constraint_violation_ids
        cand_entry: Dict[str, Any] = {
            "id": pid,
            "name": p.name,
            "crops": p.crops,
            "planting_window": p.planting_window,
            "feasible": is_feas,
            "pruned_reason": pruned_reasons.get(pid),
            "constraint_violation": is_viol,
            "score": round(float(cand_scores[idx_all]), 3),
            "estimated_margin_usd_ha": round(float(all_nom_adj[idx_all]["margin"].mean()), 0),
            "erosion_control_score": round(float(all_nom_adj[idx_all]["erosion_control"].mean()), 0),
            "extra_field_passes": p.extra_field_passes,
            "custom_hire_cost_usd_ha": round(float(custom_hire_surcharges.get(pid, 0.0)), 1),
            "irrigation_applied_mm": round(float(all_nom_adj[idx_all]["irrigation_mm"].mean()), 1),
        }
        if pid in custom_hire_notes:
            cand_entry["custom_hire_note"] = custom_hire_notes[pid]
        dynamic_candidates.append(cand_entry)

    dynamic_candidates.sort(
        key=lambda x: (x["feasible"] or x["constraint_violation"], x["feasible"], x["score"]),
        reverse=True,
    )
    for rank_i, cand in enumerate(dynamic_candidates):
        cand["robust"] = ranking_robust if rank_i == 0 else False
        cand["robust_reason"] = robust_reason if rank_i == 0 else None

    # Zone recommendations respecting feasible presets and management_mode ("uniform" vs "zone_variable")
    if len(feasible_preset_indices) < len(PRESET_PLAN_IDS):
        masked_zone_score = np.full_like(zone_score, -1e9)
        for fi in feasible_preset_indices:
            masked_zone_score[fi] = zone_score[fi]
        raw_zone_rec = [PRESET_PLAN_IDS[int(j)] for j in np.argmax(masked_zone_score, axis=0)]
    else:
        raw_zone_rec = [PRESET_PLAN_IDS[int(j)] for j in np.argmax(zone_score, axis=0)]

    recommended_id = PRESET_PLAN_IDS[best_idx]
    if eff_mgmt_mode == "uniform":
        zone_rec = [recommended_id] * nz
    else:
        zone_rec = raw_zone_rec

    # Ranking sensitivity analysis vs default settings (fast analytical re-evaluation on cached ensemble)
    default_prune = prune_candidate_plans_detailed(
        prev_crop_code=DEFAULT_PREV_CROP_CODE,
        mean_slope_pct=mean_slope,
        mean_drainage_code=mean_drainage,
        continuous_corn_years=DEFAULT_CONTINUOUS_CORN_YEARS,
        max_continuous_corn=MAX_CONTINUOUS_CORN_YEARS,
        equipment=DEFAULT_EQUIPMENT,
        excluded_crop_codes=[],
        allow_custom_hire=True,
    )
    default_scored_presets = _score_plan_list(
        plan_ids=PRESET_PLAN_IDS,
        per_ensemble_raw=per_ensemble,
        weights=DEFAULT_WEIGHTS,
        risk_aversion_lambda=DEFAULT_RISK_AVERSION_LAMBDA,
        prices=DEFAULT_PRICES,
        custom_hire_surcharges={},
        irrigation_cost_usd_per_mm=DEFAULT_IRRIGATION_COST_USD_MM,
    )
    def_feas_idxs = [
        i for i, pid in enumerate(PRESET_PLAN_IDS)
        if pid not in default_prune["pruned_reasons"]
    ] or list(range(len(PRESET_PLAN_IDS)))
    default_best_idx = max(def_feas_idxs, key=lambda i: float(default_scored_presets["field_score"][i]))
    default_rec_id = PRESET_PLAN_IDS[default_best_idx]

    default_scored_all = _score_plan_list(
        plan_ids=all_plan_ids,
        per_ensemble_raw=per_ensemble,
        weights=DEFAULT_WEIGHTS,
        risk_aversion_lambda=DEFAULT_RISK_AVERSION_LAMBDA,
        prices=DEFAULT_PRICES,
        custom_hire_surcharges={},
        irrigation_cost_usd_per_mm=DEFAULT_IRRIGATION_COST_USD_MM,
    )
    def_all_feas = [
        i for i, pid in enumerate(all_plan_ids)
        if pid not in default_prune["pruned_reasons"]
    ] or list(range(len(all_plan_ids)))
    default_top_cand = all_plan_ids[max(def_all_feas, key=lambda i: float(default_scored_all["field_score"][i]))]
    current_top_cand = dynamic_candidates[0]["id"]

    # Attribute which customized farmer settings flipped the top plan relative to defaults
    decisive_settings: List[str] = []
    if w_full != DEFAULT_WEIGHTS:
        s_w = _score_plan_list(
            PRESET_PLAN_IDS, per_ensemble, w_full, DEFAULT_RISK_AVERSION_LAMBDA, DEFAULT_PRICES, {}, DEFAULT_IRRIGATION_COST_USD_MM
        )
        w_best = PRESET_PLAN_IDS[max(def_feas_idxs, key=lambda i: float(s_w["field_score"][i]))]
        if w_best != default_rec_id:
            decisive_settings.append("weights")
    if abs(eff_lambda - DEFAULT_RISK_AVERSION_LAMBDA) > 1e-6:
        s_ra = _score_plan_list(
            PRESET_PLAN_IDS, per_ensemble, DEFAULT_WEIGHTS, eff_lambda, DEFAULT_PRICES, {}, DEFAULT_IRRIGATION_COST_USD_MM
        )
        ra_best = PRESET_PLAN_IDS[max(def_feas_idxs, key=lambda i: float(s_ra["field_score"][i]))]
        if ra_best != default_rec_id:
            decisive_settings.append("risk_aversion")
    if eff_prices != DEFAULT_PRICES:
        s_pr = _score_plan_list(
            PRESET_PLAN_IDS, per_ensemble, DEFAULT_WEIGHTS, DEFAULT_RISK_AVERSION_LAMBDA, eff_prices, {}, DEFAULT_IRRIGATION_COST_USD_MM
        )
        pr_best = PRESET_PLAN_IDS[max(def_feas_idxs, key=lambda i: float(s_pr["field_score"][i]))]
        if pr_best != default_rec_id:
            decisive_settings.append("prices")
    if eff_prev_crop != DEFAULT_PREV_CROP_CODE or eff_cc_years != DEFAULT_CONTINUOUS_CORN_YEARS:
        if set(feasible_preset_indices) != set(def_feas_idxs) or current_top_cand != default_top_cand:
            decisive_settings.append("rotation_history")
    if eff_equipment != DEFAULT_EQUIPMENT:
        if custom_hire_surcharges or set(feasible_preset_indices) != set(def_feas_idxs) or current_top_cand != default_top_cand:
            decisive_settings.append("equipment")
    if eff_excl_codes:
        if set(feasible_preset_indices) != set(def_feas_idxs) or current_top_cand != default_top_cand:
            decisive_settings.append("excluded_crops")
    if eff_irrig_cap_mm > 0.0:
        decisive_settings.append("irrigation")

    if (recommended_id != default_rec_id or current_top_cand != default_top_cand) and not decisive_settings:
        decisive_settings.append("combined_constraints")

    ranking_sensitivity = {
        "default_recommended": default_rec_id,
        "current_recommended": recommended_id,
        "default_top_candidate": default_top_cand,
        "current_top_candidate": current_top_cand,
        "top_plan_changed": (recommended_id != default_rec_id) or (current_top_cand != default_top_cand),
        "decisive_settings": decisive_settings,
        "robust": ranking_robust,
        "robust_reason": robust_reason,
    }

    uncertainty_note = (
        "Top scenario ranking is robust across dry and wet weather analogs."
        if ranking_robust
        else f"Close call: top two scenarios differ by {top_gap:.3f}, which is within the weather-analog spread ({top_spread:.3f}). Consider field-specific logistics."
    )

    # Preserve exact 3-key `weights` dict when simplicity is 0.0 for backward compatibility,
    # and include `simplicity` when non-zero
    weights_out = {
        "water_saving": round(w_full["water_saving"], 2),
        "income": round(w_full["income"], 2),
        "soil_restoration": round(w_full["soil_restoration"], 2),
    }
    if w_full.get("simplicity", 0.0) > 0.0:
        weights_out["simplicity"] = round(w_full["simplicity"], 2)

    extrapolated_any = bool(eff_irrig_cap_mm > 0.0 and any(float(m["irrigation_mm"].mean()) > 0.0 for m in grp_nom))

    return {
        "scenarios": scenarios_out,
        "recommended": recommended_id,
        "zone_recommendations": zone_rec,
        "raw_zone_recommendations": raw_zone_rec,
        "management_mode": eff_mgmt_mode,
        "weights": weights_out,
        "ranking_robust": ranking_robust,
        "robust": ranking_robust,
        "robust_reason": robust_reason,
        "uncertainty_note": uncertainty_note,
        "dynamic_candidates": dynamic_candidates,
        "pruned_plans": prune_info["pruned_plans"],
        "constraint_violation": prune_info["constraint_violation"],
        "ranking_sensitivity": ranking_sensitivity,
        "extrapolated": extrapolated_any,
        "extrapolation_note": (
            f"Supplemental irrigation (up to {eff_irrig_cap_mm:.0f} mm/season) applied in reproductive stress months; "
            "irrigated soil-moisture states are clipped to the rainfed ERA5-Land/MERRA-2 training range [0.02, 0.94]."
            if extrapolated_any
            else None
        ),
        "methodology": PLAN_METHODOLOGY_DOC,
    }

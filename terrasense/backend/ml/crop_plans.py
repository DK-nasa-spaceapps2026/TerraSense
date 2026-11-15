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
    tradeoffs: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def net_base_margin_usd_ha(self) -> float:
        return self.base_margin_usd_ha - self.seed_cost_usd_ha

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
        "Baseline crop enterprise margins and cover-crop seeding costs from ISU Ag Decision Maker (File A1-20).",
    ],
}


def prune_candidate_plans(
    prev_crop_code: int = 1,
    mean_slope_pct: float = 3.0,
    mean_drainage_code: float = 4.0,
    continuous_corn_years: int = 1,
    max_continuous_corn: int = 2,
) -> Tuple[List[AgronomicCropPlan], Dict[str, str]]:
    """Filter discrete candidate plans against hard agronomic constraints.

    Returns (feasible_plans, pruned_reasons_by_id).
    Note: The 3 compatibility presets are always evaluated for `/api/field/analyze`
    compatibility, while violated constraints are recorded and flagged.
    """
    feasible: List[AgronomicCropPlan] = []
    pruned: Dict[str, str] = {}

    for pid, plan in AGRONOMIC_PLANS.items():
        reasons = []
        if prev_crop_code == 5 and not plan.allow_after_soy:
            reasons.append("Soy-after-soy prohibited (SCN & disease buildup)")
        if prev_crop_code == 1 and not plan.allow_after_corn:
            reasons.append("Requires prior soybean residue (high C:N corn residue tie-up)")
        if prev_crop_code == 1 and plan.primary_crop_code == 1 and continuous_corn_years >= max_continuous_corn:
            reasons.append(f"Exceeds continuous-corn limit ({max_continuous_corn} yrs)")
        if mean_slope_pct > plan.max_slope_pct:
            reasons.append(f"Field slope ({mean_slope_pct:.1f}%) exceeds plan limit ({plan.max_slope_pct:.1f}%)")
        if mean_drainage_code < plan.min_drainage_code:
            reasons.append(
                f"Soil drainage class ({mean_drainage_code:.1f}) too poorly drained for {plan.name} (min {plan.min_drainage_code})"
            )

        if reasons:
            pruned[pid] = "; ".join(reasons)
        else:
            feasible.append(plan)

    return feasible, pruned


def evaluate_and_rank_plans(
    grid: Dict[str, Any],
    climate: Dict[str, Any],
    prefs: Dict[str, Any],
    simulate_fn: Callable[[Dict[str, np.ndarray], List[dict], List[str]], Dict[str, np.ndarray]],
    preference_weights_fn: Callable[[dict], Dict[str, float]],
    risk_aversion_lambda: float = 0.25,
) -> Dict[str, Any]:
    """Evaluate all 3 preset plans + dynamic candidates over a weather ensemble (nominal, dry analog, wet analog).

    Computes per-zone and field-level expected metrics, critical reproductive-window stress,
    risk-adjusted multi-objective scores (Mean Utility - lambda * Downside Deviation),
    per-component score breakdowns, and ranking robustness diagnostics.
    """
    n_hist = 12
    n_total = 24
    f_slice = slice(n_hist, n_total)
    w = preference_weights_fn(prefs)
    nz = len(grid["soil"]["awc"])

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

    for _, t_shift, p_fac, _ in ensemble_shifts:
        m_list = _perturbed_months(t_shift, p_fac)
        for pid in all_plan_ids:
            plan = AGRONOMIC_PLANS[pid]
            # Ensure zone_engine.CALENDARS has the candidate plan's monthly cover/Kc curve
            import zone_engine as _ze
            if pid not in _ze.CALENDARS:
                _ze.CALENDARS[pid] = {
                    "cover": plan.effective_cover_array(),
                    "kc": plan.kc_array(),
                }
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

            et_year = sim["et"][f_slice].sum(axis=0)
            runoff = sim["runoff"][f_slice].sum(axis=0)
            bare = sim["bare"][f_slice].mean(axis=0)

            n_fix = plan.n_fix_kg_ha * (0.7 + 0.3 * np.clip(esi_season, 0.0, 1.0))
            som = (
                plan.som_gain_pct_yr
                * (0.8 + 0.4 * sim["vigor"][f_slice].mean(axis=0))
                * (0.9 + 0.1 * grid["soil"]["som"] / 3.0)
            )
            gross_rev_adj = plan.base_margin_usd_ha * np.clip(1.0 - plan.drought_sens * stress * 1.6, 0.35, 1.05)
            margin = np.maximum(50.0, gross_rev_adj - plan.seed_cost_usd_ha)
            risk = np.clip(plan.drought_sens * stress * 1.4 + 0.35 * bare, 0.0, 1.0)
            erosion_control = 100.0 * (1.0 - bare)

            per_ensemble[pid].append({
                "water_demand": sim["demand"][f_slice].sum(axis=0),
                "water_balance": sim["precip"][f_slice].sum(axis=0) - et_year - runoff,
                "et": et_year,
                "bare": bare,
                "stress": stress,
                "crit_stress": crit_stress,
                "n_fix": n_fix,
                "som": som,
                "margin": margin,
                "risk": risk,
                "erosion_control": erosion_control,
            })

    def _norm(v: np.ndarray, higher_better: bool = True) -> np.ndarray:
        rng = v.max(axis=0) - v.min(axis=0)
        n = np.where(rng < 1e-9, 0.5, (v - v.min(axis=0)) / np.where(rng < 1e-9, 1.0, rng))
        return n if higher_better else 1.0 - n

    # Compute multi-objective score across the 3 compatibility presets first (preserving exact API semantics)
    preset_nominal = [per_ensemble[pid][0] for pid in PRESET_PLAN_IDS]
    preset_dry = [per_ensemble[pid][1] for pid in PRESET_PLAN_IDS]
    preset_wet = [per_ensemble[pid][2] for pid in PRESET_PLAN_IDS]

    def _score_group(group_metrics: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        stk = lambda k: np.stack([m[k] for m in group_metrics])
        c_water = _norm(stk("water_demand"), False)
        c_income = _norm(stk("margin"), True)
        c_soil = 0.5 * _norm(stk("som"), True) + 0.3 * _norm(stk("erosion_control"), True) + 0.2 * _norm(stk("n_fix"), True)
        total = w["water_saving"] * c_water + w["income"] * c_income + w["soil_restoration"] * c_soil
        return total, {"water_saving": c_water, "income": c_income, "soil_restoration": c_soil}

    z_nom, comp_nom = _score_group(preset_nominal)
    z_dry, _ = _score_group(preset_dry)
    z_wet, _ = _score_group(preset_wet)

    # Risk-adjusted zone score: expected ensemble utility minus lambda * downside deviation in dry analog
    z_expected = 0.50 * z_nom + 0.25 * z_dry + 0.25 * z_wet
    downside = np.maximum(0.0, z_expected - z_dry)
    zone_score = np.clip(z_expected - risk_aversion_lambda * downside, 0.0, 1.0)
    field_score = zone_score.mean(axis=1)
    best_idx = int(np.argmax(field_score))

    # Ensemble spread diagnostics for uncertainty reporting
    ens_field_scores = np.stack([z_nom.mean(axis=1), z_dry.mean(axis=1), z_wet.mean(axis=1)], axis=1)  # (3, 3)
    sorted_scores = np.sort(field_score)[::-1]
    top_gap = float(sorted_scores[0] - sorted_scores[1])
    top_spread = float(ens_field_scores[best_idx].max() - ens_field_scores[best_idx].min())
    ranking_robust = bool(top_gap >= max(0.03, 0.5 * top_spread))

    def _risk_label(score: float) -> str:
        return "low" if score < 0.30 else "medium" if score < 0.55 else "high"

    scenarios_out = []
    for i, pid in enumerate(PRESET_PLAN_IDS):
        m = preset_nominal[i]
        plan = AGRONOMIC_PLANS[pid]
        mean_fn = lambda k: float(m[k].mean())
        scenarios_out.append({
            "id": pid,
            "name": plan.name,
            "crops": plan.crops,
            "description": plan.description,
            "score": round(float(field_score[i]), 3),
            "recommended": i == best_idx,
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
            },
            "score_breakdown": {
                "water_saving_component": round(float((w["water_saving"] * comp_nom["water_saving"][i]).mean()), 3),
                "income_component": round(float((w["income"] * comp_nom["income"][i]).mean()), 3),
                "soil_restoration_component": round(float((w["soil_restoration"] * comp_nom["soil_restoration"][i]).mean()), 3),
                "downside_risk_penalty": round(float((risk_aversion_lambda * downside[i]).mean()), 3),
                "ensemble_score_range": [
                    round(float(ens_field_scores[i].min()), 3),
                    round(float(ens_field_scores[i].max()), 3),
                ],
            },
            "planting_window": plan.planting_window,
            "tradeoffs": plan.tradeoffs,
        })

    # Dynamic candidate pruning & top-k ranking across all 5 candidate plans
    mean_slope = float(grid["soil"]["slope"].mean())
    mean_drainage = float(grid["soil"].get("drainage_code", np.full(nz, 4.0)).mean())
    prev_crop = int(grid["soil"].get("prev_crop_code", np.full(nz, 1))[0])
    feasible_plans, pruned_reasons = prune_candidate_plans(
        prev_crop_code=prev_crop,
        mean_slope_pct=mean_slope,
        mean_drainage_code=mean_drainage,
    )
    all_nom = [per_ensemble[pid][0] for pid in all_plan_ids]
    z_all, _ = _score_group(all_nom)
    cand_scores = z_all.mean(axis=1)
    dynamic_candidates = []
    for idx_all, pid in enumerate(all_plan_ids):
        p = AGRONOMIC_PLANS[pid]
        dynamic_candidates.append({
            "id": pid,
            "name": p.name,
            "crops": p.crops,
            "planting_window": p.planting_window,
            "feasible": pid not in pruned_reasons,
            "pruned_reason": pruned_reasons.get(pid),
            "score": round(float(cand_scores[idx_all]), 3),
            "estimated_margin_usd_ha": round(float(all_nom[idx_all]["margin"].mean()), 0),
            "erosion_control_score": round(float(all_nom[idx_all]["erosion_control"].mean()), 0),
        })
    dynamic_candidates.sort(key=lambda x: (x["feasible"], x["score"]), reverse=True)

    zone_rec = [PRESET_PLAN_IDS[int(j)] for j in np.argmax(zone_score, axis=0)]
    uncertainty_note = (
        "Top scenario ranking is robust across dry and wet weather analogs."
        if ranking_robust
        else f"Close call: top two scenarios differ by {top_gap:.3f}, which is within the weather-analog spread ({top_spread:.3f}). Consider field-specific logistics."
    )

    return {
        "scenarios": scenarios_out,
        "recommended": PRESET_PLAN_IDS[best_idx],
        "zone_recommendations": zone_rec,
        "weights": {k: round(v, 2) for k, v in w.items()},
        "ranking_robust": ranking_robust,
        "uncertainty_note": uncertainty_note,
        "dynamic_candidates": dynamic_candidates,
        "methodology": PLAN_METHODOLOGY_DOC,
    }

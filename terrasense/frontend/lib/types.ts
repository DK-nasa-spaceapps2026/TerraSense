// API contract mirrors terrasense/backend (see root README.md#api-reference)
export type Status = "lush" | "moderate" | "water_stress" | "heat_stress" | "bare_vulnerable";
export type Priority = "water_saving" | "income" | "soil_restoration" | "simplicity";
export type ScenarioId =
  | "status_quo"
  | "cover_legume"
  | "drought_cash"
  | "corn_rye_cover"
  | "alfalfa_perennial";

export interface FarmerConstraints {
  prev_crop?: string;
  continuous_corn_years?: number;
  equipment?: {
    has_hay_baler: boolean;
    has_cover_drill: boolean;
    has_no_till_drill: boolean;
    variable_rate: boolean;
  };
  allow_custom_hire?: boolean;
  excluded_crops?: string[];
  irrigation_cap_mm?: number;
  irrigation_cost_usd_per_mm?: number;
  risk_aversion?: "conservative" | "balanced" | "aggressive";
  priority?: Priority;
  weights?: {
    water_saving: number;
    income: number;
    soil_restoration: number;
    simplicity: number;
  };
  prices?: {
    corn_usd_bu: number;
    soy_usd_bu: number;
    sorghum_usd_bu: number;
    alfalfa_usd_ton: number;
    n_cost_usd_kg: number;
  };
  management_mode?: "uniform" | "zone_variable";
}

export interface PrunedPlan {
  id: string;
  name: string;
  reason: string;
}

export interface CellState {
  smi: number; esi: number; bare: number; ndvi: number; vigor: number; et_mm: number; status: Status;
}
export interface Alert { type: string; severity: "medium" | "high"; message: string }
export interface MonthUncertainty {
  smi_p10: number; smi_p90: number; esi_p10: number; esi_p90: number;
}
export interface MonthState {
  index: number; date: string; label: string; phase: "historical" | "forecast";
  climate: { tmax_c: number; tmin_c: number; precip_mm: number; rh_pct: number; solar_mj: number; pet_mm: number };
  crop_stage: string;
  field: { smi: number; esi: number; bare: number; ndvi: number; vigor: number };
  uncertainty?: MonthUncertainty;
  alerts: Alert[];
  cells: CellState[]; // same order as zone features in field_geojson
}
export interface Window { start: string; end: string; message: string; type?: string }
export interface Guidance {
  seed_window: Window | null;
  vulnerable_windows: Window[];
  cover_crop_windows: Window[];
  stress_alerts: Window[];
}
export interface Timeline {
  field_id: string;
  scenario: ScenarioId;
  anchor_index: number;
  engine?: "ml" | "heuristic";
  model_version?: string | null;
  is_proxy_ndvi_bare?: boolean;
  months: MonthState[];
  guidance: Guidance;
}
export interface ZoneProps {
  zone_id: string; row: number; col: number; recommended_scenario: ScenarioId;
  soil: {
    sand_pct: number;
    clay_pct: number;
    organic_matter_pct: number;
    slope_pct: number;
    available_water_mm: number;
    ph?: number;
    elev_m?: number;
    twi?: number;
  };
}
export interface Scenario {
  id: ScenarioId;
  name: string;
  crops: string;
  description: string;
  planting_window?: string;
  score: number;
  recommended: boolean;
  feasible?: boolean;
  pruned_reason?: string | null;
  metrics: {
    soil_organic_matter_gain_pct: number;
    nitrogen_fixed_kg_ha: number;
    water_demand_mm: number;
    water_budget_balance_mm: number;
    irrigation_applied_mm?: number;
    erosion_control_score: number;
    estimated_margin_usd_ha: number;
    critical_window_stress_pct?: number;
    risk_level: "low" | "medium" | "high";
    risk_score: number;
  };
  score_breakdown?: {
    water_saving_component: number;
    income_component: number;
    soil_restoration_component: number;
    downside_risk_penalty: number;
    ensemble_score_range: [number, number];
  };
  tradeoffs: { pros: string[]; cons: string[] };
}
export interface DynamicCandidate {
  id: ScenarioId;
  name: string;
  crops: string;
  planting_window: string;
  feasible: boolean;
  pruned_reason: string | null;
  score: number | null;
  estimated_margin_usd_ha?: number;
  erosion_control_score?: number;
}
export interface AnalyzeResponse {
  field_id: string;
  engine?: "ml" | "heuristic";
  model_version?: string | null;
  is_proxy_ndvi_bare?: boolean;
  ranking_robust?: boolean;
  robust?: boolean;
  robust_reason?: string | null;
  uncertainty_note?: string;
  dynamic_candidates?: DynamicCandidate[];
  pruned_plans?: PrunedPlan[];
  constraint_violation?: boolean;
  management_mode?: "uniform" | "zone_variable";
  applied_constraints?: Record<string, unknown> & { is_default?: boolean; irrigation_cap_mm?: number };
  warnings?: string[];
  extrapolated?: boolean;
  extrapolation_note?: string | null;
  data_sources: Record<string, string>;
  grid: { rows: number; cols: number; n_zones: number };
  field_geojson: { features: { id: string; properties: ZoneProps & { kind: string } }[] };
  summary: { current_month: string; alerts: Alert[] };
  scenarios: Scenario[];
  recommended_scenario: ScenarioId;
  guidance: Guidance;
}
export interface Zone extends ZoneProps {}

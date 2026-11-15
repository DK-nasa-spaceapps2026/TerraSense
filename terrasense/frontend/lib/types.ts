// API contract mirrors terrasense/backend (see README.md)
export type Status = "lush" | "moderate" | "water_stress" | "heat_stress" | "bare_vulnerable";
export type Priority = "water_saving" | "income" | "soil_restoration";
export type ScenarioId =
  | "status_quo"
  | "cover_legume"
  | "drought_cash"
  | "corn_rye_cover"
  | "alfalfa_perennial";

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
  metrics: {
    soil_organic_matter_gain_pct: number;
    nitrogen_fixed_kg_ha: number;
    water_demand_mm: number;
    water_budget_balance_mm: number;
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
  uncertainty_note?: string;
  dynamic_candidates?: DynamicCandidate[];
  data_sources: Record<string, string>;
  grid: { rows: number; cols: number; n_zones: number };
  field_geojson: { features: { id: string; properties: ZoneProps & { kind: string } }[] };
  summary: { current_month: string; alerts: Alert[] };
  scenarios: Scenario[];
  recommended_scenario: ScenarioId;
  guidance: Guidance;
}
export interface Zone extends ZoneProps {}

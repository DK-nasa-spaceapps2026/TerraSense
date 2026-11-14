// API contract mirrors terrasense/backend (see README.md)
export type Status = "lush" | "moderate" | "water_stress" | "heat_stress" | "bare_vulnerable";
export type Priority = "water_saving" | "income" | "soil_restoration";
export type ScenarioId = "status_quo" | "cover_legume" | "drought_cash";

export interface CellState {
  smi: number; esi: number; bare: number; ndvi: number; vigor: number; et_mm: number; status: Status;
}
export interface Alert { type: string; severity: "medium" | "high"; message: string }
export interface MonthState {
  index: number; date: string; label: string; phase: "historical" | "forecast";
  climate: { tmax_c: number; tmin_c: number; precip_mm: number; rh_pct: number; solar_mj: number; pet_mm: number };
  crop_stage: string;
  field: { smi: number; esi: number; bare: number; ndvi: number; vigor: number };
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
  field_id: string; scenario: ScenarioId; anchor_index: number; months: MonthState[]; guidance: Guidance;
}
export interface ZoneProps {
  zone_id: string; row: number; col: number; recommended_scenario: ScenarioId;
  soil: { sand_pct: number; clay_pct: number; organic_matter_pct: number; slope_pct: number; available_water_mm: number };
}
export interface Scenario {
  id: ScenarioId; name: string; crops: string; description: string; score: number; recommended: boolean;
  metrics: {
    soil_organic_matter_gain_pct: number; nitrogen_fixed_kg_ha: number; water_demand_mm: number;
    water_budget_balance_mm: number; erosion_control_score: number; estimated_margin_usd_ha: number;
    risk_level: "low" | "medium" | "high"; risk_score: number;
  };
  tradeoffs: { pros: string[]; cons: string[] };
}
export interface AnalyzeResponse {
  field_id: string;
  data_sources: Record<string, string>;
  grid: { rows: number; cols: number; n_zones: number };
  field_geojson: { features: { id: string; properties: ZoneProps & { kind: string } }[] };
  summary: { current_month: string; alerts: Alert[] };
  scenarios: Scenario[];
  recommended_scenario: ScenarioId;
  guidance: Guidance;
}
export interface Zone extends ZoneProps {}

import type { AnalyzeResponse, Priority, ScenarioId, Timeline } from "./types";

async function j<T>(p: Promise<Response>): Promise<T> {
  const r = await p;
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error ?? `HTTP ${r.status}`);
  return r.json();
}

export interface ClimateParams { temp_shift: number; precip_factor: number }

export const analyzeField = (priority: Priority, climate: ClimateParams) =>
  j<AnalyzeResponse>(
    // No "field" => backend uses its built-in demo parcel (central Iowa).
    // Pass { field: { bbox: [w,s,e,n] } } or { field: { polygon: GeoJSON } } for real parcels.
    fetch("/api/field/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ grid_size: 6, preferences: { priority }, climate }),
    }),
  );

export const fetchTimeline = (fieldId: string, scenario: ScenarioId, c: ClimateParams) =>
  j<Timeline>(
    fetch(`/api/field/timeline?field_id=${fieldId}&scenario=${scenario}&temp_shift=${c.temp_shift}&precip_factor=${c.precip_factor}`),
  );

"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Satellite, Thermometer, CloudRain } from "lucide-react";
import { analyzeField, fetchTimeline, type ClimateParams } from "@/lib/api";
import type { AnalyzeResponse, Priority, ScenarioId, Timeline, ZoneProps } from "@/lib/types";
import TimelineSlider from "@/components/TimelineSlider";
import DecisionPanel from "@/components/DecisionPanel";
import TileDrawer from "@/components/TileDrawer";
import { STATUS_COLOR, STATUS_LABEL } from "@/components/IsoGrid";

// three.js needs the browser
const IsoGrid = dynamic(() => import("@/components/IsoGrid"), {
  ssr: false,
  loading: () => <div className="grid h-full place-items-center text-stone-500">Loading field…</div>,
});

const NOW_INDEX = 11;

export default function Page() {
  // ---- server state -------------------------------------------------------
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [error, setError] = useState<string | null>(null);

  // ---- user-controlled state ---------------------------------------------
  const [priority, setPriority] = useState<Priority>("soil_restoration");
  const [scenario, setScenario] = useState<ScenarioId>("status_quo");
  const [climate, setClimate] = useState<ClimateParams>({ temp_shift: 0, precip_factor: 1 });
  const [debouncedClimate, setDebouncedClimate] = useState(climate);
  const [monthIdx, setMonthIdx] = useState(NOW_INDEX);
  const [playing, setPlaying] = useState(false);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    const t = setTimeout(() => setDebouncedClimate(climate), 350);
    return () => clearTimeout(t);
  }, [climate]);

  // 1) analyze field whenever farmer priority or what-if climate changes
  useEffect(() => {
    let live = true;
    analyzeField(priority, debouncedClimate)
      .then((a) => { if (live) { setAnalysis(a); setScenario(a.recommended_scenario); setError(null); } })
      .catch((e) => live && setError(`${e.message}. Is the Flask API running on :5001?`));
    return () => { live = false; };
  }, [priority, debouncedClimate]);

  // 2) fetch 24-month timeline for the chosen rotation scenario
  useEffect(() => {
    if (!analysis) return;
    let live = true;
    fetchTimeline(analysis.field_id, scenario, debouncedClimate)
      .then((t) => live && setTimeline(t))
      .catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [analysis?.field_id, scenario, debouncedClimate]); // eslint-disable-line react-hooks/exhaustive-deps

  // 3) autoplay the scrubber
  useEffect(() => {
    if (!playing || !timeline) return;
    const id = setInterval(() => setMonthIdx((i) => (i + 1) % timeline.months.length), 900);
    return () => clearInterval(id);
  }, [playing, timeline]);

  const zones = useMemo<ZoneProps[]>(
    () => analysis?.field_geojson.features.filter((f) => f.properties.kind === "zone").map((f) => f.properties) ?? [],
    [analysis],
  );
  const month = timeline?.months[monthIdx];
  const selIdx = zones.findIndex((z) => z.zone_id === selectedId);

  if (error && !analysis) return <Centered><AlertTriangle className="mb-2 text-amber-400" />{error}</Centered>;
  if (!analysis || !timeline || !month) return <Centered>Fusing NASA data for your field…</Centered>;

  return (
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-4 p-4">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-2xl font-bold tracking-tight"><span className="text-lime-400">Terra</span>Sense</h1>
          <p className="text-xs text-stone-400">NASA Earth Observation → management zones → rotation decisions</p>
        </div>
        <div className="flex items-center gap-2 text-xs text-stone-400">
          <Satellite size={14} />
          Climate: <b className="text-stone-200">{analysis.data_sources.climate === "mock" ? "simulated (offline)" : "NASA POWER"}</b>
        </div>
      </header>

      {error && <div className="rounded bg-amber-900/40 p-2 text-xs text-amber-200">{error}</div>}

      <div className="grid gap-4 lg:grid-cols-[1fr_300px]">
        <div className="relative h-[480px] overflow-hidden rounded-xl border border-stone-700">
          <IsoGrid zones={zones} cells={month.cells} rows={analysis.grid.rows} cols={analysis.grid.cols}
            hoveredId={hoveredId} selectedId={selectedId} onHover={setHoveredId} onSelect={setSelectedId} />

          <div className="pointer-events-none absolute bottom-3 left-3 rounded-lg bg-black/60 p-2 text-[11px]">
            {(Object.keys(STATUS_COLOR) as (keyof typeof STATUS_COLOR)[]).map((s) => (
              <div key={s} className="flex items-center gap-2"><i className="h-2.5 w-2.5 rounded-sm" style={{ background: STATUS_COLOR[s] }} />{STATUS_LABEL[s]}</div>
            ))}
          </div>

          {month.alerts.length > 0 && (
            <div className="absolute left-3 top-3 max-w-xs space-y-1">
              {month.alerts.map((a, i) => (
                <div key={i} className={`flex gap-2 rounded px-2 py-1 text-xs ${a.severity === "high" ? "bg-red-900/80" : "bg-amber-900/80"}`}>
                  <AlertTriangle size={14} className="shrink-0" />{a.message}
                </div>
              ))}
            </div>
          )}

          {selIdx >= 0 && (
            <TileDrawer zone={zones[selIdx]} cell={month.cells[selIdx]} monthLabel={month.label}
              scenarios={analysis.scenarios} onClose={() => setSelectedId(null)} />
          )}
        </div>

        {/* what-if climate */}
        <div className="space-y-4 rounded-xl border border-stone-700 bg-soil-800 p-4 text-sm">
          <div className="font-semibold">Forecast what-if</div>
          <label className="block text-xs">
            <span className="flex items-center gap-1"><Thermometer size={12} />Warming: +{climate.temp_shift.toFixed(1)} °C</span>
            <input type="range" min={0} max={4} step={0.5} value={climate.temp_shift} className="w-full"
              onChange={(e) => setClimate({ ...climate, temp_shift: +e.target.value })} />
          </label>
          <label className="block text-xs">
            <span className="flex items-center gap-1"><CloudRain size={12} />Rainfall: {Math.round(climate.precip_factor * 100)}% of normal</span>
            <input type="range" min={0.5} max={1.3} step={0.05} value={climate.precip_factor} className="w-full"
              onChange={(e) => setClimate({ ...climate, precip_factor: +e.target.value })} />
          </label>
          <div className="rounded bg-stone-800 p-2 text-xs text-stone-400">
            Applies to forecast months only (shaded on the timeline). Rotation scenario: <b className="text-stone-200">
              {analysis.scenarios.find((s) => s.id === scenario)?.name}</b>
          </div>
          <div className="grid grid-cols-2 gap-2 text-center text-xs">
            <Stat label="Mean SMI" v={month.field.smi} />
            <Stat label="Mean ESI" v={month.field.esi} />
            <Stat label="Bare risk" v={month.field.bare} />
            <Stat label="NDVI" v={month.field.ndvi} />
          </div>
        </div>
      </div>

      <TimelineSlider months={timeline.months} index={monthIdx} onChange={(i) => { setMonthIdx(i); setPlaying(false); }}
        playing={playing} onTogglePlay={() => setPlaying((p) => !p)} />

      <DecisionPanel scenarios={analysis.scenarios} active={scenario} onPick={setScenario}
        priority={priority} onPriority={setPriority} guidance={timeline.guidance} />
    </main>
  );
}

const Centered = ({ children }: { children: React.ReactNode }) => (
  <div className="grid min-h-screen place-items-center text-center text-stone-300"><div>{children}</div></div>
);
const Stat = ({ label, v }: { label: string; v: number }) => (
  <div className="rounded bg-stone-800 p-2"><div className="text-lg font-bold">{v.toFixed(2)}</div><div className="text-stone-400">{label}</div></div>
);

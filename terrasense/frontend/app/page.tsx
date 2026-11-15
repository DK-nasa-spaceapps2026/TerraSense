"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, CloudRain, Cpu, Lock, Satellite, SlidersHorizontal, Thermometer, Unlock } from "lucide-react";
import { PiFarmLight } from "react-icons/pi";
import { analyzeField, fetchTimeline, type ClimateParams } from "@/lib/api";
import type { AnalyzeResponse, FarmerConstraints, ScenarioId, Timeline, ZoneProps } from "@/lib/types";
import TimelineSlider from "@/components/TimelineSlider";
import MetricsDashboard from "@/components/MetricsDashboard";
import DecisionPanel from "@/components/DecisionPanel";
import TileDrawer from "@/components/TileDrawer";
import ConstraintsModal, { DEFAULT_CONSTRAINTS } from "@/components/ConstraintsModal";
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
  const [scenario, setScenario] = useState<ScenarioId>("status_quo");
  const [climate, setClimate] = useState<ClimateParams>({ temp_shift: 0, precip_factor: 1 });
  const [debouncedClimate, setDebouncedClimate] = useState(climate);
  const [monthIdx, setMonthIdx] = useState(NOW_INDEX);
  const [playing, setPlaying] = useState(false);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mapLocked, setMapLocked] = useState(true);
  const [constraints, setConstraints] = useState<FarmerConstraints>(DEFAULT_CONSTRAINTS);
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setDebouncedClimate(climate), 350);
    return () => clearTimeout(t);
  }, [climate]);

  // 1) analyze field whenever what-if climate or farmer constraints change
  useEffect(() => {
    let live = true;
    const isDef = JSON.stringify(constraints) === JSON.stringify(DEFAULT_CONSTRAINTS);
    analyzeField(constraints.priority ?? "soil_restoration", debouncedClimate, isDef ? undefined : constraints)
      .then((a) => { if (live) { setAnalysis(a); setScenario(a.recommended_scenario); setError(null); } })
      .catch((e) => live && setError(`${e.message}. Is the Flask API running on :5001?`));
    return () => { live = false; };
  }, [debouncedClimate, constraints]);

  // 2) fetch 24-month timeline for the chosen rotation scenario & irrigation cap
  useEffect(() => {
    if (!analysis) return;
    let live = true;
    fetchTimeline(analysis.field_id, scenario, debouncedClimate, constraints.irrigation_cap_mm ?? 0)
      .then((t) => live && setTimeline(t))
      .catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [analysis?.field_id, scenario, debouncedClimate, constraints.irrigation_cap_mm]); // eslint-disable-line react-hooks/exhaustive-deps

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
  const activeScenarioName =
    analysis?.scenarios.find((s) => s.id === scenario)?.name ??
    analysis?.dynamic_candidates?.find((c) => c.id === scenario)?.name ??
    scenario;

  if (error && !analysis) return <Centered><AlertTriangle className="mb-2 text-amber-400" />{error}</Centered>;
  if (!analysis || !timeline || !month) return <Centered>Fusing NASA EO & ML models for your field…</Centered>;

  return (
    <main className="flex min-h-screen w-full flex-col gap-4 px-5 py-4">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <PiFarmLight className="text-4xl text-lime-400" />
          <div>
            <h1 className="text-2xl font-bold tracking-tight"><span className="text-lime-400">Terra</span>Sense</h1>
          </div>
        </div>

        <button
          onClick={() => setSettingsOpen(true)}
          className="flex items-center gap-1.5 rounded-lg border border-stone-700 bg-soil-800 px-3 py-1.5 text-xs font-medium text-stone-200 shadow-sm transition hover:border-lime-500 hover:text-lime-300"
          title="Configure farm constraints, water availability, equipment & market prices"
        >
          <SlidersHorizontal size={14} className="text-lime-400" />
          <span>Constraints</span>
        </button>
      </header>

      <ConstraintsModal
        open={settingsOpen}
        value={constraints}
        onClose={() => setSettingsOpen(false)}
        onApply={setConstraints}
      />

      {error && <div className="rounded bg-amber-900/40 p-2 text-xs text-amber-200">{error}</div>}

      <DecisionPanel
        scenarios={analysis.scenarios}
        dynamicCandidates={analysis.dynamic_candidates}
        prunedPlans={analysis.pruned_plans}
        rankingRobust={analysis.ranking_robust}
        uncertaintyNote={analysis.uncertainty_note}
        active={scenario}
        onPick={setScenario}
        constraints={constraints}
        climate={climate}
      />

      <div className="relative h-[540px] w-full overflow-hidden rounded-xl border border-stone-700">
        <IsoGrid zones={zones} cells={month.cells} rows={analysis.grid.rows} cols={analysis.grid.cols}
          hoveredId={hoveredId} selectedId={selectedId} onHover={setHoveredId} onSelect={setSelectedId} locked={mapLocked} />

        <button
          onClick={() => setMapLocked((l) => !l)}
          className={`absolute bottom-3 right-3 z-10 flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium shadow transition ${
            mapLocked
              ? "border-stone-700 bg-black/75 text-stone-200 hover:bg-stone-800"
              : "border-lime-500 bg-lime-950/85 text-lime-200 hover:bg-lime-900/90"
          }`}
          title={mapLocked ? "Unlock to pan/zoom/rotate map" : "Lock map so page scrolling passes through"}
        >
          {mapLocked ? <Lock size={13} /> : <Unlock size={13} />}
          {mapLocked ? "View Locked" : "View Unlocked"}
        </button>

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

        {selIdx < 0 && (
          <div className="absolute right-3 top-3 z-10 w-64 space-y-2.5 rounded-xl border border-stone-700/90 bg-soil-900/90 p-3 text-xs shadow-xl backdrop-blur">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-stone-100">Forecast what-if</span>
              <span className="text-[10px] text-stone-400">Shaded months</span>
            </div>
            <label className="block">
              <span className="mb-0.5 flex items-center justify-between text-stone-300">
                <span className="flex items-center gap-1"><Thermometer size={12} />Warming</span>
                <b>+{climate.temp_shift.toFixed(1)} °C</b>
              </span>
              <input type="range" min={0} max={4} step={0.5} value={climate.temp_shift} className="w-full"
                onChange={(e) => setClimate({ ...climate, temp_shift: +e.target.value })} />
            </label>
            <label className="block">
              <span className="mb-0.5 flex items-center justify-between text-stone-300">
                <span className="flex items-center gap-1"><CloudRain size={12} />Rainfall</span>
                <b>{Math.round(climate.precip_factor * 100)}%</b>
              </span>
              <input type="range" min={0.5} max={1.3} step={0.05} value={climate.precip_factor} className="w-full"
                onChange={(e) => setClimate({ ...climate, precip_factor: +e.target.value })} />
            </label>
          </div>
        )}

        {selIdx >= 0 && (
          <TileDrawer
            zone={zones[selIdx]}
            cell={month.cells[selIdx]}
            monthLabel={month.label}
            scenarios={analysis.scenarios}
            dataSources={analysis.data_sources}
            onClose={() => setSelectedId(null)}
          />
        )}
      </div>

      <TimelineSlider
        months={timeline.months}
        index={monthIdx}
        onChange={(i) => { setMonthIdx(i); setPlaying(false); }}
        playing={playing}
        onTogglePlay={() => setPlaying((p) => !p)}
        scenarioName={activeScenarioName}
        dataSources={analysis.data_sources}
      />

      <MetricsDashboard
        months={timeline.months}
        monthIdx={monthIdx}
        onSelectMonth={(i) => { setMonthIdx(i); setPlaying(false); }}
        selectedZone={selIdx >= 0 ? zones[selIdx] : null}
        selectedZoneIdx={selIdx}
        scenarioName={activeScenarioName}
        dataSources={analysis.data_sources}
      />

      <footer className="mt-2 flex flex-wrap items-center gap-2.5 border-t border-stone-800 pt-3 text-xs text-stone-400">
        <span className="font-semibold uppercase tracking-wider text-stone-500">Demo Build:</span>
        {analysis.engine && (
          <span className="flex items-center gap-1.5 rounded-full border border-lime-700/70 bg-lime-950/50 px-2.5 py-0.5 text-lime-300">
            <Cpu size={13} />
            Engine: <b>{analysis.engine === "ml" ? `Hybrid LightGBM (${analysis.model_version ?? "ML"})` : "Heuristic WB"}</b>
          </span>
        )}
        <span className="flex items-center gap-1.5 rounded-full border border-stone-700 bg-stone-900/80 px-2.5 py-0.5">
          <Satellite size={13} className="text-lime-400" />
          Climate: <b className="text-stone-200">{analysis.data_sources.climate}</b>
        </span>
      </footer>
    </main>
  );
}

const Centered = ({ children }: { children: React.ReactNode }) => (
  <div className="grid min-h-screen place-items-center text-center text-stone-300"><div>{children}</div></div>
);

"use client";

import { useState } from "react";
import { Check, ShieldCheck, Sparkles } from "lucide-react";
import type { DynamicCandidate, Guidance, Scenario, ScenarioId } from "@/lib/types";

const RISK = { low: "text-lime-400", medium: "text-amber-400", high: "text-red-400" } as const;

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="flex items-baseline justify-between border-b border-stone-700/60 py-1 text-xs">
      <span className="text-stone-400">{label}</span>
      <span className="font-semibold">{value}{sub && <span className="ml-1 font-normal text-stone-500">{sub}</span>}</span>
    </div>
  );
}

export default function DecisionPanel(props: {
  scenarios: Scenario[];
  dynamicCandidates?: DynamicCandidate[];
  rankingRobust?: boolean;
  uncertaintyNote?: string;
  active: ScenarioId;
  onPick: (id: ScenarioId) => void;
  guidance?: Guidance | null;
}) {
  const { scenarios, dynamicCandidates = [], rankingRobust, uncertaintyNote, active, onPick } = props;
  const [showExtra, setShowExtra] = useState(false);

  const presetIds = new Set(scenarios.map((s) => s.id));
  const extraCandidates = dynamicCandidates.filter((c) => !presetIds.has(c.id));

  return (
    <section className="space-y-3 pt-1">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-800 pb-2.5">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 border-l-4 border-lime-500 pl-2.5">
            <h2 className="text-sm font-bold uppercase tracking-wider text-stone-100">
              Recommended Profiles
            </h2>
          </div>
          {uncertaintyNote && (
            <span className="flex items-center gap-1.5 rounded-full border border-stone-700 bg-stone-900/80 px-2.5 py-0.5 text-xs text-stone-300">
              <ShieldCheck size={13} className={rankingRobust ? "text-lime-400" : "text-amber-400"} />
              {uncertaintyNote}
            </span>
          )}
        </div>

        {extraCandidates.length > 0 && (
          <button
            onClick={() => setShowExtra((s) => !s)}
            className="flex items-center gap-1.5 rounded-lg border border-stone-700 bg-stone-900/80 px-2.5 py-1 text-xs text-stone-300 transition hover:border-lime-500 hover:text-lime-300"
          >
            <Sparkles size={13} className="text-lime-400" />
            {showExtra ? "Hide ML Candidate Pool" : `+ ${extraCandidates.length} ML Candidate Rotations`}
          </button>
        )}
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        {scenarios.map((s) => {
          const m = s.metrics;
          const isActive = active === s.id;
          return (
            <button key={s.id} onClick={() => onPick(s.id)}
              className={`relative rounded-xl border p-3 text-left transition ${isActive ? "border-lime-500 bg-soil-700" : "border-stone-700 bg-soil-800 hover:border-stone-500"}`}>
              <div className="absolute -top-2 right-3 flex items-center gap-1.5">
                {isActive && (
                  <span className="rounded-full bg-[#D97757] px-2 py-0.5 text-[10px] font-bold text-[#141413]">
                    ACTIVE FORECAST
                  </span>
                )}
                {s.recommended && (
                  <span className="flex items-center gap-1 rounded-full bg-lime-600 px-2 py-0.5 text-[10px] font-bold text-white">
                    <Check size={10} />RECOMMENDED
                  </span>
                )}
              </div>
              <div className="flex items-baseline justify-between">
                <div className="font-semibold">{s.name}</div>
                <span className="text-[11px] text-lime-400 font-mono">Score {(s.score * 100).toFixed(0)}</span>
              </div>
              <div className="text-xs text-stone-400">{s.crops}</div>
              {s.planting_window && (
                <div className="mb-2 mt-0.5 text-[11px] text-stone-500">Window: {s.planting_window}</div>
              )}
              <Metric label="Soil organic matter" value={`+${m.soil_organic_matter_gain_pct.toFixed(2)}`} sub="% / yr" />
              <Metric label="Nitrogen fixed" value={`${m.nitrogen_fixed_kg_ha}`} sub="kg/ha" />
              <Metric label="Water demand" value={`${m.water_demand_mm}`} sub="mm" />
              <Metric label="Water budget" value={`${m.water_budget_balance_mm >= 0 ? "+" : ""}${m.water_budget_balance_mm}`} sub="mm" />
              <Metric label="Erosion control" value={`${m.erosion_control_score}`} sub="/100" />
              {m.critical_window_stress_pct !== undefined && (
                <Metric label="Reproductive stress" value={`${m.critical_window_stress_pct.toFixed(1)}%`} sub="Jul–Aug" />
              )}
              <Metric label="Est. margin" value={`$${m.estimated_margin_usd_ha}`} sub="/ha" />
              <div className="flex items-center justify-between pt-1 text-xs">
                <span className="text-stone-400">Risk</span>
                <span className={`font-bold uppercase ${RISK[m.risk_level]}`}>{m.risk_level}</span>
              </div>
              <div className="mt-2 text-[11px] text-stone-500">{s.tradeoffs.pros[0]} · {s.tradeoffs.cons[0].toLowerCase()}</div>
            </button>
          );
        })}
      </div>

      {showExtra && extraCandidates.length > 0 && (
        <div className="grid gap-3 md:grid-cols-2">
          {extraCandidates.map((c) => {
            const isActive = active === c.id;
            return (
              <button
                key={c.id}
                disabled={!c.feasible}
                onClick={() => c.feasible && onPick(c.id)}
                className={`relative flex items-center justify-between rounded-xl border p-3 text-left text-xs transition ${
                  !c.feasible
                    ? "cursor-not-allowed border-stone-800 bg-soil-900/50 opacity-60"
                    : isActive
                      ? "border-lime-500 bg-soil-700"
                      : "border-stone-700 bg-soil-800 hover:border-stone-500"
                }`}
              >
                {isActive && (
                  <span className="absolute -top-2 right-3 rounded-full bg-[#D97757] px-2 py-0.5 text-[10px] font-bold text-[#141413]">
                    ACTIVE FORECAST
                  </span>
                )}
                <div>
                  <div className="font-semibold text-sm text-stone-100">{c.name}</div>
                  <div className="text-stone-400">{c.crops}</div>
                  <div className="mt-0.5 text-[11px] text-stone-500">{c.planting_window}</div>
                  {c.pruned_reason && <div className="mt-1 text-amber-400">Pruned: {c.pruned_reason}</div>}
                </div>
                {c.feasible && (
                  <div className="text-right space-y-0.5">
                    {c.score !== null && <div className="font-mono text-lime-400 font-semibold">Score {(c.score * 100).toFixed(0)}</div>}
                    {c.estimated_margin_usd_ha !== undefined && <div className="text-stone-200">${c.estimated_margin_usd_ha}/ha</div>}
                    {c.erosion_control_score !== undefined && <div className="text-stone-400">Erosion {c.erosion_control_score}/100</div>}
                  </div>
                )}
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}

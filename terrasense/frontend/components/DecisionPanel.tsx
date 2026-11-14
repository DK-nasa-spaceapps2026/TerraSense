"use client";

import { Check } from "lucide-react";
import type { Guidance, Scenario, ScenarioId } from "@/lib/types";

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
  active: ScenarioId;
  onPick: (id: ScenarioId) => void;
  guidance?: Guidance | null;
}) {
  const { scenarios, active, onPick } = props;
  return (
    <section className="space-y-3 pt-1">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-800 pb-2.5">
        <div className="flex items-center gap-2 border-l-4 border-lime-500 pl-2.5">
          <h2 className="text-sm font-bold uppercase tracking-wider text-stone-100">
            Recommended Profiles
          </h2>
        </div>
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
                  <span className="rounded-full bg-violet-600 px-2 py-0.5 text-[10px] font-bold text-white">
                    ACTIVE FORECAST
                  </span>
                )}
                {s.recommended && (
                  <span className="flex items-center gap-1 rounded-full bg-lime-600 px-2 py-0.5 text-[10px] font-bold text-white">
                    <Check size={10} />RECOMMENDED
                  </span>
                )}
              </div>
              <div className="font-semibold">{s.name}</div>
              <div className="mb-2 text-xs text-stone-400">{s.crops}</div>
              <Metric label="Soil organic matter" value={`+${m.soil_organic_matter_gain_pct.toFixed(2)}`} sub="% / yr" />
              <Metric label="Nitrogen fixed" value={`${m.nitrogen_fixed_kg_ha}`} sub="kg/ha" />
              <Metric label="Water demand" value={`${m.water_demand_mm}`} sub="mm" />
              <Metric label="Water budget" value={`${m.water_budget_balance_mm >= 0 ? "+" : ""}${m.water_budget_balance_mm}`} sub="mm" />
              <Metric label="Erosion control" value={`${m.erosion_control_score}`} sub="/100" />
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
    </section>
  );
}

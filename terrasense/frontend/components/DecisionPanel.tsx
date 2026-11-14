"use client";

import { CalendarClock, Droplets, Flame, Leaf, Shield, Sprout, Check } from "lucide-react";
import type { Guidance, Priority, Scenario, ScenarioId } from "@/lib/types";

const RISK = { low: "text-lime-400", medium: "text-amber-400", high: "text-red-400" } as const;
const PRIORITIES: { id: Priority; label: string; Icon: typeof Droplets }[] = [
  { id: "water_saving", label: "Save water", Icon: Droplets },
  { id: "income", label: "Income", Icon: Sprout },
  { id: "soil_restoration", label: "Restore soil", Icon: Leaf },
];

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="flex items-baseline justify-between border-b border-stone-700/60 py-1 text-xs">
      <span className="text-stone-400">{label}</span>
      <span className="font-semibold">{value}{sub && <span className="ml-1 font-normal text-stone-500">{sub}</span>}</span>
    </div>
  );
}

export default function DecisionPanel(props: {
  scenarios: Scenario[]; active: ScenarioId; onPick: (id: ScenarioId) => void;
  priority: Priority; onPriority: (p: Priority) => void; guidance: Guidance | null;
}) {
  const { scenarios, active, onPick, priority, onPriority, guidance } = props;
  return (
    <section className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="text-xs text-stone-400">My priority:</span>
        {PRIORITIES.map(({ id, label, Icon }) => (
          <button key={id} onClick={() => onPriority(id)}
            className={`flex items-center gap-1 rounded-full border px-3 py-1 text-xs ${priority === id ? "border-lime-500 bg-lime-900/50 text-lime-200" : "border-stone-600 hover:bg-stone-700"}`}>
            <Icon size={12} />{label}
          </button>
        ))}
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        {scenarios.map((s) => {
          const m = s.metrics;
          return (
            <button key={s.id} onClick={() => onPick(s.id)}
              className={`relative rounded-xl border p-3 text-left transition ${active === s.id ? "border-lime-500 bg-soil-700" : "border-stone-700 bg-soil-800 hover:border-stone-500"}`}>
              {s.recommended && (
                <span className="absolute -top-2 right-3 flex items-center gap-1 rounded-full bg-lime-600 px-2 py-0.5 text-[10px] font-bold text-white">
                  <Check size={10} />RECOMMENDED
                </span>
              )}
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

      {guidance && (
        <div className="grid gap-3 md:grid-cols-3">
          <Card icon={<CalendarClock size={16} className="text-lime-400" />} title="When to seed">
            {guidance.seed_window ? <p>{guidance.seed_window.message}</p> : <p>No suitable window found.</p>}
          </Card>
          <Card icon={<Shield size={16} className="text-amber-400" />} title="Ground is vulnerable">
            {guidance.vulnerable_windows.length ? guidance.vulnerable_windows.map((w, i) => <p key={i}>{w.message}</p>) : <p>No exposure windows.</p>}
          </Card>
          <Card icon={<Leaf size={16} className="text-emerald-400" />} title="Cover-crop insertion">
            {guidance.cover_crop_windows.length ? guidance.cover_crop_windows.map((w, i) => <p key={i}>{w.message}</p>) : <p>None.</p>}
            {guidance.stress_alerts.slice(0, 2).map((a, i) => (
              <p key={i} className="mt-1 flex gap-1 text-orange-300"><Flame size={12} className="mt-0.5 shrink-0" />{a.start}: {a.message}</p>
            ))}
          </Card>
        </div>
      )}
    </section>
  );
}

function Card({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-stone-700 bg-soil-800 p-3">
      <div className="mb-1 flex items-center gap-2 text-sm font-semibold">{icon}{title}</div>
      <div className="space-y-1 text-xs text-stone-300">{children}</div>
    </div>
  );
}

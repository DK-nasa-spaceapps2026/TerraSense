"use client";

import { Check, Droplets, ShieldAlert, Sparkles, Tractor, TrendingUp, Wheat } from "lucide-react";
import type { ClimateParams } from "@/lib/api";
import type { DynamicCandidate, FarmerConstraints, PrunedPlan, Scenario, ScenarioId } from "@/lib/types";
import { DEFAULT_CONSTRAINTS } from "@/components/ConstraintsModal";

const PRIORITY_LABEL: Record<string, string> = {
  soil_restoration: "Restore Soil",
  water_saving: "Save Water",
  income: "Maximize Income",
  simplicity: "Simplicity",
};

export default function DecisionPanel(props: {
  scenarios: Scenario[];
  dynamicCandidates?: DynamicCandidate[];
  prunedPlans?: PrunedPlan[];
  rankingRobust?: boolean;
  uncertaintyNote?: string;
  active: ScenarioId;
  onPick: (id: ScenarioId) => void;
  constraints: FarmerConstraints;
  climate: ClimateParams;
}) {
  const {
    scenarios,
    dynamicCandidates = [],
    prunedPlans = [],
    active,
    onPick,
    constraints,
    climate,
  } = props;

  const c = {
    ...DEFAULT_CONSTRAINTS,
    ...constraints,
    equipment: { ...DEFAULT_CONSTRAINTS.equipment, ...(constraints.equipment ?? {}) },
    prices: { ...DEFAULT_CONSTRAINTS.prices, ...(constraints.prices ?? {}) },
    excluded_crops: constraints.excluded_crops ?? [],
  };

  const activeScenario =
    scenarios.find((s) => s.id === active) ??
    scenarios[0];

  const equipItems = [
    { label: "Cover Drill", ok: c.equipment.has_cover_drill },
    { label: "No-Till", ok: c.equipment.has_no_till_drill },
    { label: "Hay Baler", ok: c.equipment.has_hay_baler },
    { label: "VRT", ok: c.equipment.variable_rate },
  ];

  // Combine preset scenarios + any feasible extra dynamic candidates into a compact rotation selector
  const presetIds = new Set(scenarios.map((s) => s.id));
  const extraFeasible = dynamicCandidates.filter((d) => !presetIds.has(d.id) && d.feasible);

  return (
    <section className="space-y-3 pt-1">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-800 pb-2.5">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 border-l-4 border-lime-500 pl-2.5">
            <h2 className="text-sm font-bold uppercase tracking-wider text-stone-100">
              Current Farm Settings
            </h2>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Compact Rotation Switcher */}
          <div className="flex flex-wrap items-center gap-1.5 rounded-lg border border-stone-800 bg-soil-900 p-1">
            {scenarios.map((s) => {
              const isAct = active === s.id;
              const isPruned = s.feasible === false;
              return (
                <button
                  key={s.id}
                  onClick={() => onPick(s.id)}
                  title={isPruned && s.pruned_reason ? `Disqualified: ${s.pruned_reason}` : s.crops}
                  className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition ${
                    isAct
                      ? "bg-soil-700 text-stone-100 ring-1 ring-lime-500"
                      : isPruned
                        ? "text-stone-500 line-through hover:text-stone-300"
                        : "text-stone-400 hover:bg-stone-800 hover:text-stone-200"
                  }`}
                >
                  {s.recommended && !isPruned && <Check size={11} className="text-lime-400" />}
                  <span>{s.name}</span>
                  <span className="font-mono text-[10px] text-lime-400">{(s.score * 100).toFixed(0)}</span>
                </button>
              );
            })}
            {extraFeasible.map((d) => {
              const isAct = active === d.id;
              return (
                <button
                  key={d.id}
                  onClick={() => onPick(d.id)}
                  className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition ${
                    isAct
                      ? "bg-soil-700 text-stone-100 ring-1 ring-lime-500"
                      : "text-stone-400 hover:bg-stone-800 hover:text-stone-200"
                  }`}
                >
                  <Sparkles size={11} className="text-lime-400" />
                  <span>{d.name}</span>
                  {d.score !== null && (
                    <span className="font-mono text-[10px] text-lime-400">{(d.score * 100).toFixed(0)}</span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* 4-Column Farmer Settings Summary Grid */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {/* 1. Goal & Risk Profile */}
        <div className="rounded-xl border border-stone-800 bg-soil-800 p-3 text-xs">
          <div className="mb-2 flex items-center justify-between border-b border-stone-700/60 pb-1.5">
            <span className="flex items-center gap-1.5 font-semibold text-stone-200">
              <ShieldAlert size={13} className="text-lime-400" />
              Goal & Risk Profile
            </span>
          </div>
          <div className="space-y-1">
            <Row label="Primary Priority" value={PRIORITY_LABEL[c.priority] ?? c.priority} highlight />
            <Row label="Risk Tolerance" value={c.risk_aversion} capitalize />
            <Row label="Management Mode" value={c.management_mode === "zone_variable" ? "Zone-Variable (VRT)" : "Uniform Field"} />
            {activeScenario && (
              <Row
                label="Selected Rotation"
                value={`${activeScenario.name} ($${activeScenario.metrics.estimated_margin_usd_ha}/ha)`}
                highlight
              />
            )}
          </div>
        </div>

        {/* 2. Water Availability & Climate */}
        <div className="rounded-xl border border-stone-800 bg-soil-800 p-3 text-xs">
          <div className="mb-2 flex items-center justify-between border-b border-stone-700/60 pb-1.5">
            <span className="flex items-center gap-1.5 font-semibold text-stone-200">
              <Droplets size={13} className="text-sky-400" />
              Water & Climate Settings
            </span>
            <span className="text-[10px] text-stone-400">
              {c.irrigation_cap_mm > 0 ? "Irrigated" : "Rainfed"}
            </span>
          </div>
          <div className="space-y-1">
            <Row
              label="Irrigation Cap"
              value={c.irrigation_cap_mm > 0 ? `${c.irrigation_cap_mm} mm / season` : "0 mm (Rainfed only)"}
              highlight={c.irrigation_cap_mm > 0}
            />
            <Row label="Pumping Cost" value={`$${c.irrigation_cost_usd_per_mm.toFixed(2)} / mm / ha`} />
            <Row
              label="What-If Warming"
              value={`+${climate.temp_shift.toFixed(1)} °C`}
              highlight={climate.temp_shift > 0}
            />
            <Row
              label="What-If Rainfall"
              value={`${Math.round(climate.precip_factor * 100)}% of baseline`}
              highlight={climate.precip_factor !== 1}
            />
          </div>
        </div>

        {/* 3. Rotation History & Equipment */}
        <div className="rounded-xl border border-stone-800 bg-soil-800 p-3 text-xs">
          <div className="mb-2 flex items-center justify-between border-b border-stone-700/60 pb-1.5">
            <span className="flex items-center gap-1.5 font-semibold text-stone-200">
              <Tractor size={13} className="text-amber-400" />
              History & Equipment
            </span>
            <Wheat size={13} className="text-stone-400" />
          </div>
          <div className="space-y-1">
            <Row
              label="Previous Crop"
              value={`${c.prev_crop} (${c.continuous_corn_years}y cont. corn)`}
              capitalize
              highlight={c.continuous_corn_years >= 2}
            />
            <Row
              label="Excluded Crops"
              value={c.excluded_crops.length > 0 ? c.excluded_crops.join(", ") : "None"}
              capitalize
              warn={c.excluded_crops.length > 0}
            />
            <Row label="Custom Hire" value={c.allow_custom_hire ? "Allowed" : "Not allowed"} warn={!c.allow_custom_hire} />
            <div className="flex flex-wrap gap-1 pt-1">
              {equipItems.map((eq) => (
                <span
                  key={eq.label}
                  className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                    eq.ok
                      ? "border border-stone-700 bg-stone-900/80 text-stone-300"
                      : "border border-red-500/40 bg-red-950/50 text-red-300 line-through"
                  }`}
                >
                  {eq.label}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* 4. Local Market Prices & Pruned Notice */}
        <div className="rounded-xl border border-stone-800 bg-soil-800 p-3 text-xs">
          <div className="mb-2 flex items-center justify-between border-b border-stone-700/60 pb-1.5">
            <span className="flex items-center gap-1.5 font-semibold text-stone-200">
              <TrendingUp size={13} className="text-[#D97757]" />
              Market Prices & Pruning
            </span>
            <span className="text-[10px] text-stone-400">Local Basis</span>
          </div>
          <div className="space-y-1">
            <Row label="Corn / Soy" value={`$${c.prices.corn_usd_bu.toFixed(2)} / $${c.prices.soy_usd_bu.toFixed(2)} bu`} />
            <Row label="Sorghum / Alfalfa" value={`$${c.prices.sorghum_usd_bu.toFixed(2)} bu / $${c.prices.alfalfa_usd_ton.toFixed(0)} t`} />
            <Row label="Nitrogen Cost" value={`$${c.prices.n_cost_usd_kg.toFixed(2)} / kg N`} />
            {prunedPlans.length > 0 ? (
              <div className="mt-1 rounded border border-amber-500/30 bg-amber-950/30 px-2 py-1 text-[11px] text-amber-300">
                <b>{prunedPlans.length} plan{prunedPlans.length > 1 ? "s" : ""} pruned:</b>{" "}
                {prunedPlans.map((p) => p.name).join(", ")}
              </div>
            ) : (
              <Row label="Feasible Pool" value="All 5 rotations eligible" highlight />
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function Row({
  label,
  value,
  highlight,
  warn,
  capitalize,
}: {
  label: string;
  value: string;
  highlight?: boolean;
  warn?: boolean;
  capitalize?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-0.5">
      <span className="text-stone-400">{label}</span>
      <span
        className={`truncate font-semibold ${
          warn ? "text-amber-400" : highlight ? "text-lime-400" : "text-stone-200"
        } ${capitalize ? "capitalize" : ""}`}
      >
        {value}
      </span>
    </div>
  );
}

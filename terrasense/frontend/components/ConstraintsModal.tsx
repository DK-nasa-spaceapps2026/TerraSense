"use client";

import { useState } from "react";
import { Droplets, RotateCcw, ShieldAlert, SlidersHorizontal, Tractor, TrendingUp, Wheat, X } from "lucide-react";
import type { FarmerConstraints, Priority } from "@/lib/types";

export const DEFAULT_CONSTRAINTS: Required<Omit<FarmerConstraints, "weights">> = {
  priority: "soil_restoration",
  prev_crop: "corn",
  continuous_corn_years: 1,
  equipment: {
    has_cover_drill: true,
    has_no_till_drill: true,
    has_hay_baler: true,
    variable_rate: true,
  },
  allow_custom_hire: true,
  excluded_crops: [],
  irrigation_cap_mm: 0,
  irrigation_cost_usd_per_mm: 1.8,
  risk_aversion: "balanced",
  prices: {
    corn_usd_bu: 4.6,
    soy_usd_bu: 11.2,
    sorghum_usd_bu: 4.4,
    alfalfa_usd_ton: 175.0,
    n_cost_usd_kg: 1.1,
  },
  management_mode: "uniform",
};

const CROPS = [
  { id: "corn", label: "Corn" },
  { id: "soybeans", label: "Soybeans" },
  { id: "sorghum", label: "Sorghum" },
  { id: "alfalfa", label: "Alfalfa" },
];

const PRIORITIES: { id: Priority; label: string }[] = [
  { id: "soil_restoration", label: "Restore Soil" },
  { id: "water_saving", label: "Save Water" },
  { id: "income", label: "Maximize Income" },
  { id: "simplicity", label: "Simplicity" },
];

export function buildRestraintTags(c: FarmerConstraints): string[] {
  const tags: string[] = [];
  if (c.priority && c.priority !== "soil_restoration") {
    tags.push(`Priority: ${c.priority.replace("_", " ")}`);
  }
  if (c.prev_crop && c.prev_crop !== "corn") {
    tags.push(`Prev: ${c.prev_crop}`);
  }
  if ((c.continuous_corn_years ?? 1) >= 2) {
    tags.push(`Cont. corn: ${c.continuous_corn_years}y`);
  }
  if ((c.irrigation_cap_mm ?? 0) > 0) {
    tags.push(`Irrigation: ${c.irrigation_cap_mm} mm`);
  }
  if (c.risk_aversion && c.risk_aversion !== "balanced") {
    tags.push(`Risk: ${c.risk_aversion}`);
  }
  if (c.management_mode === "zone_variable") {
    tags.push("Mode: Zone-Variable");
  }
  if (c.excluded_crops && c.excluded_crops.length > 0) {
    tags.push(`No: ${c.excluded_crops.join(", ")}`);
  }
  if (c.equipment) {
    const missing = Object.entries(c.equipment)
      .filter(([, v]) => !v)
      .map(([k]) => k.replace("has_", "").replace("_", " "));
    if (missing.length > 0) {
      tags.push(`No ${missing.join("/")}`);
    }
  }
  return tags;
}

interface Props {
  open: boolean;
  value: FarmerConstraints;
  onClose: () => void;
  onApply: (next: FarmerConstraints) => void;
}

export default function ConstraintsModal({ open, value, onClose, onApply }: Props) {
  const [draft, setDraft] = useState<Required<Omit<FarmerConstraints, "weights">>>({
    ...DEFAULT_CONSTRAINTS,
    ...value,
    equipment: { ...DEFAULT_CONSTRAINTS.equipment, ...(value.equipment ?? {}) },
    prices: { ...DEFAULT_CONSTRAINTS.prices, ...(value.prices ?? {}) },
    excluded_crops: [...(value.excluded_crops ?? [])],
  });

  if (!open) return null;

  const toggleExcludedCrop = (crop: string) => {
    setDraft((d) => ({
      ...d,
      excluded_crops: d.excluded_crops.includes(crop)
        ? d.excluded_crops.filter((c) => c !== crop)
        : [...d.excluded_crops, crop],
    }));
  };

  const reset = () => {
    setDraft(DEFAULT_CONSTRAINTS);
    onApply(DEFAULT_CONSTRAINTS);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-end bg-black/60 backdrop-blur-sm">
      <div className="flex h-full w-full max-w-lg flex-col border-l border-stone-700 bg-soil-900 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-stone-800 px-5 py-4">
          <div className="flex items-center gap-2.5">
            <SlidersHorizontal size={18} className="text-lime-400" />
            <div>
              <h2 className="text-base font-bold text-stone-100">Farm Constraints & Priorities</h2>
              <p className="text-xs text-stone-400">Water availability, equipment, rotation history & market prices</p>
            </div>
          </div>
          <button onClick={onClose} className="rounded-lg p-1.5 text-stone-400 hover:bg-stone-800 hover:text-stone-100">
            <X size={18} />
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 space-y-5 overflow-y-auto p-5 text-xs">
          {/* 1. Primary Priority & Risk Tolerance */}
          <div className="space-y-2.5 rounded-xl border border-stone-800 bg-soil-800 p-3.5">
            <div className="flex items-center gap-1.5 font-semibold text-stone-200">
              <ShieldAlert size={14} className="text-lime-400" />
              Primary Goal & Risk Appetite
            </div>
            <div>
              <div className="mb-1.5 text-stone-400">Primary objective priority</div>
              <div className="grid grid-cols-2 gap-1.5">
                {PRIORITIES.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => setDraft({ ...draft, priority: p.id })}
                    className={`rounded-lg border px-2.5 py-1.5 text-left font-medium transition ${
                      draft.priority === p.id
                        ? "border-lime-500 bg-lime-950/60 text-lime-200"
                        : "border-stone-700 bg-stone-900/60 text-stone-300 hover:border-stone-500"
                    }`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <div className="mb-1.5 text-stone-400">Drought / downside risk tolerance</div>
              <div className="grid grid-cols-3 gap-1.5">
                {(["conservative", "balanced", "aggressive"] as const).map((ra) => (
                  <button
                    key={ra}
                    type="button"
                    onClick={() => setDraft({ ...draft, risk_aversion: ra })}
                    className={`rounded-lg border px-2.5 py-1.5 capitalize font-medium transition ${
                      draft.risk_aversion === ra
                        ? "border-lime-500 bg-lime-950/60 text-lime-200"
                        : "border-stone-700 bg-stone-900/60 text-stone-300 hover:border-stone-500"
                    }`}
                  >
                    {ra}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* 2. Water Availability & Supplemental Irrigation */}
          <div className="space-y-2.5 rounded-xl border border-stone-800 bg-soil-800 p-3.5">
            <div className="flex items-center gap-1.5 font-semibold text-stone-200">
              <Droplets size={14} className="text-sky-400" />
              Water Availability & Irrigation
            </div>
            <label className="block">
              <span className="mb-1 flex justify-between text-stone-300">
                <span>Supplemental irrigation cap (Jul–Aug)</span>
                <b className="text-lime-400">
                  {draft.irrigation_cap_mm === 0 ? "0 mm (Rainfed only)" : `${draft.irrigation_cap_mm} mm/season`}
                </b>
              </span>
              <input
                type="range"
                min={0}
                max={300}
                step={15}
                value={draft.irrigation_cap_mm}
                onChange={(e) => setDraft({ ...draft, irrigation_cap_mm: +e.target.value })}
                className="w-full"
              />
            </label>
            <label className="block">
              <span className="mb-1 flex justify-between text-stone-300">
                <span>Pumping / application cost</span>
                <b>${draft.irrigation_cost_usd_per_mm.toFixed(2)} / mm / ha</b>
              </span>
              <input
                type="range"
                min={0.5}
                max={4.5}
                step={0.1}
                value={draft.irrigation_cost_usd_per_mm}
                onChange={(e) => setDraft({ ...draft, irrigation_cost_usd_per_mm: +e.target.value })}
                className="w-full"
              />
            </label>
          </div>

          {/* 3. Rotation History & Crop Exclusions */}
          <div className="space-y-2.5 rounded-xl border border-stone-800 bg-soil-800 p-3.5">
            <div className="flex items-center gap-1.5 font-semibold text-stone-200">
              <Wheat size={14} className="text-amber-400" />
              Rotation History & Excluded Crops
            </div>
            <div className="grid grid-cols-2 gap-3">
              <label className="block">
                <span className="mb-1 block text-stone-400">Previous season crop</span>
                <select
                  value={draft.prev_crop}
                  onChange={(e) =>
                    setDraft({
                      ...draft,
                      prev_crop: e.target.value,
                      continuous_corn_years: e.target.value === "corn" ? Math.max(1, draft.continuous_corn_years) : 0,
                    })
                  }
                  className="w-full rounded-lg border border-stone-700 bg-stone-900 px-2.5 py-1.5 text-stone-100"
                >
                  {CROPS.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block">
                <span className="mb-1 block text-stone-400">Consecutive corn years</span>
                <input
                  type="number"
                  min={0}
                  max={10}
                  value={draft.continuous_corn_years}
                  onChange={(e) => setDraft({ ...draft, continuous_corn_years: Math.max(0, Math.min(10, +e.target.value)) })}
                  className="w-full rounded-lg border border-stone-700 bg-stone-900 px-2.5 py-1.5 text-stone-100"
                />
              </label>
            </div>
            <div>
              <span className="mb-1.5 block text-stone-400">Exclude crops (no local buyer / preference)</span>
              <div className="flex flex-wrap gap-1.5">
                {CROPS.map((c) => {
                  const excluded = draft.excluded_crops.includes(c.id);
                  return (
                    <button
                      key={c.id}
                      type="button"
                      onClick={() => toggleExcludedCrop(c.id)}
                      className={`rounded-full border px-2.5 py-1 text-xs font-medium transition ${
                        excluded
                          ? "border-red-500/80 bg-red-950/60 text-red-200 line-through"
                          : "border-stone-700 bg-stone-900/60 text-stone-300 hover:border-stone-500"
                      }`}
                    >
                      {c.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* 4. Equipment Limitations & Management Mode */}
          <div className="space-y-2.5 rounded-xl border border-stone-800 bg-soil-800 p-3.5">
            <div className="flex items-center gap-1.5 font-semibold text-stone-200">
              <Tractor size={14} className="text-lime-400" />
              Equipment & Field Management
            </div>
            <div className="grid grid-cols-2 gap-2">
              {(
                [
                  ["has_cover_drill", "Cover-crop drill"],
                  ["has_no_till_drill", "No-till planter"],
                  ["has_hay_baler", "Hay baler (alfalfa)"],
                  ["variable_rate", "Variable-rate (VRT)"],
                ] as const
              ).map(([key, label]) => (
                <label key={key} className="flex items-center gap-2 rounded-lg border border-stone-700/70 bg-stone-900/50 px-2.5 py-1.5 text-stone-200 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={draft.equipment[key]}
                    onChange={(e) =>
                      setDraft({
                        ...draft,
                        equipment: { ...draft.equipment, [key]: e.target.checked },
                      })
                    }
                    className="accent-lime-500"
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
            <div className="flex items-center justify-between pt-1">
              <label className="flex items-center gap-2 text-stone-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={draft.allow_custom_hire}
                  onChange={(e) => setDraft({ ...draft, allow_custom_hire: e.target.checked })}
                  className="accent-lime-500"
                />
                <span>Allow custom hire if equipment missing</span>
              </label>
              <select
                value={draft.management_mode}
                onChange={(e) => setDraft({ ...draft, management_mode: e.target.value as "uniform" | "zone_variable" })}
                className="rounded-lg border border-stone-700 bg-stone-900 px-2 py-1 text-stone-200"
              >
                <option value="uniform">Uniform Field</option>
                <option value="zone_variable">Zone-Variable</option>
              </select>
            </div>
          </div>

          {/* 5. Local Market Commodity & N Prices */}
          <div className="space-y-2.5 rounded-xl border border-stone-800 bg-soil-800 p-3.5">
            <div className="flex items-center gap-1.5 font-semibold text-stone-200">
              <TrendingUp size={14} className="text-[#D97757]" />
              Local Commodity & Fertilizer Prices
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              {(
                [
                  ["corn_usd_bu", "Corn ($/bu)", 2.5, 9.0, 0.1],
                  ["soy_usd_bu", "Soybean ($/bu)", 6.0, 20.0, 0.2],
                  ["sorghum_usd_bu", "Sorghum ($/bu)", 2.5, 8.5, 0.1],
                  ["alfalfa_usd_ton", "Alfalfa ($/ton)", 80, 320, 5],
                  ["n_cost_usd_kg", "Nitrogen ($/kg N)", 0.4, 3.0, 0.05],
                ] as const
              ).map(([key, label, min, max, step]) => (
                <label key={key} className="block">
                  <span className="mb-0.5 block text-stone-400">{label}</span>
                  <input
                    type="number"
                    min={min}
                    max={max}
                    step={step}
                    value={draft.prices[key]}
                    onChange={(e) =>
                      setDraft({
                        ...draft,
                        prices: { ...draft.prices, [key]: +e.target.value },
                      })
                    }
                    className="w-full rounded-lg border border-stone-700 bg-stone-900 px-2.5 py-1 text-stone-100"
                  />
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Footer actions */}
        <div className="flex items-center justify-between border-t border-stone-800 px-5 py-3.5">
          <button
            type="button"
            onClick={reset}
            className="flex items-center gap-1.5 rounded-lg border border-stone-700 px-3 py-1.5 text-xs text-stone-300 hover:bg-stone-800"
          >
            <RotateCcw size={13} />
            Reset Defaults
          </button>
          <button
            type="button"
            onClick={() => {
              onApply(draft);
              onClose();
            }}
            className="rounded-lg bg-lime-600 px-4 py-1.5 text-xs font-semibold text-white shadow hover:bg-lime-500"
          >
            Apply & Recalculate
          </button>
        </div>
      </div>
    </div>
  );
}

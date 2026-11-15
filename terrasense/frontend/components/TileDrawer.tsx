"use client";

import { X } from "lucide-react";
import type { CellState, Scenario, ZoneProps } from "@/lib/types";
import { STATUS_COLOR, STATUS_LABEL } from "./IsoGrid";

const Bar = ({ label, value, invert = false, hint }: { label: string; value: number; invert?: boolean; hint: string }) => {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  const good = invert ? 100 - pct : pct;
  return (
    <div className="mb-3">
      <div className="flex justify-between text-xs"><span>{label}</span><span className="text-stone-400">{value.toFixed(2)}</span></div>
      <div className="h-2 rounded bg-stone-700">
        <div className="h-2 rounded" style={{ width: `${pct}%`, background: good > 60 ? "#7cc04a" : good > 35 ? "#e0b13a" : "#d9622b" }} />
      </div>
      <div className="mt-0.5 text-[10px] text-stone-400">{hint}</div>
    </div>
  );
};

export default function TileDrawer({ zone, cell, monthLabel, scenarios, onClose }: {
  zone: ZoneProps; cell: CellState; monthLabel: string; scenarios: Scenario[]; onClose: () => void;
}) {
  const rec = scenarios.find((s) => s.id === zone.recommended_scenario);
  return (
    <aside className="absolute right-3 top-3 z-10 w-72 rounded-xl border border-stone-700 bg-soil-900/95 p-4 shadow-xl backdrop-blur">
      <div className="mb-2 flex items-start justify-between">
        <div>
          <div className="text-xs uppercase tracking-wide text-stone-400">Zone {zone.row + 1}-{zone.col + 1} · {monthLabel}</div>
          <div className="mt-1 inline-flex items-center gap-2 text-base font-semibold">
            <i className="h-3 w-3 rounded-sm" style={{ background: STATUS_COLOR[cell.status] }} />
            {STATUS_LABEL[cell.status]}
          </div>
        </div>
        <button onClick={onClose} aria-label="Close" className="rounded p-1 hover:bg-stone-700"><X size={16} /></button>
      </div>

      <Bar label="Soil Moisture Index" value={cell.smi} hint="SMAP-style root-zone fill (0 dry – 1 saturated)" />
      <Bar label="Evaporative Stress Index" value={cell.esi} hint="ET ÷ crop PET — below 0.65 = stressed" />
      <Bar label="Bare-soil exposure risk" value={cell.bare} invert hint="Cover, slope, sand & rain intensity" />
      <Bar label="Vegetation (NDVI)" value={cell.ndvi} hint="MODIS/Landsat-style greenness" />

      <div className="mt-3 grid grid-cols-2 gap-2 rounded bg-stone-800 p-2 text-xs">
        <div>Sand <b>{zone.soil.sand_pct}%</b></div><div>Clay <b>{zone.soil.clay_pct}%</b></div>
        <div>SOM <b>{zone.soil.organic_matter_pct}%</b></div><div>Slope <b>{zone.soil.slope_pct}%</b></div>
        <div className="col-span-2">Plant-available water <b>{zone.soil.available_water_mm} mm</b></div>
      </div>

      {rec && (
        <div className="mt-3 rounded border border-lime-700 bg-lime-950/40 p-2 text-xs">
          <div className="text-stone-400">Best strategy for this zone</div>
          <div className="font-semibold text-lime-300">{rec.name}</div>
          <div className="text-stone-300">{rec.crops}</div>
        </div>
      )}
    </aside>
  );
}

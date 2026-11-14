"use client";

import { useState } from "react";
import {
  Activity, CloudRain, Droplets, Flame, Layers, Leaf, Maximize2, Minimize2, Plus, Sun, Trash2, Wind,
} from "lucide-react";
import type { MonthState, ZoneProps } from "@/lib/types";
import { STATUS_COLOR, STATUS_LABEL } from "./IsoGrid";

export type MetricId =
  | "smi"
  | "precip_et"
  | "esi"
  | "ndvi"
  | "bare"
  | "temp"
  | "zone_dist"
  | "pet_rh";

interface MetricPreset {
  id: MetricId;
  title: string;
  sensor: string;
  unit: string;
  color: string;
  secondaryColor?: string;
  description: string;
  Icon: typeof Droplets;
}

export const METRIC_PRESETS: MetricPreset[] = [
  {
    id: "smi",
    title: "Soil Moisture Index",
    sensor: "NASA SMAP",
    unit: "SMI",
    color: "#38bdf8",
    description: "Root-zone water fill fraction (0 dry – 1 saturated)",
    Icon: Droplets,
  },
  {
    id: "precip_et",
    title: "Rainfall vs Crop ET",
    sensor: "GPM / ECOSTRESS",
    unit: "mm",
    color: "#60a5fa",
    secondaryColor: "#f59e0b",
    description: "Monthly precipitation supply vs evapotranspiration demand",
    Icon: CloudRain,
  },
  {
    id: "esi",
    title: "Evaporative Stress (ESI)",
    sensor: "ECOSTRESS",
    unit: "ET/PET",
    color: "#a3e635",
    description: "Ratio of actual to potential ET (< 0.65 indicates water stress)",
    Icon: Activity,
  },
  {
    id: "ndvi",
    title: "Canopy Vigor & NDVI",
    sensor: "MODIS / Landsat",
    unit: "NDVI",
    color: "#4ade80",
    secondaryColor: "#a3e635",
    description: "Normalized vegetation index and composite plant vigor",
    Icon: Leaf,
  },
  {
    id: "bare",
    title: "Bare-Soil Exposure Risk",
    sensor: "Terrain + Cover",
    unit: "Risk",
    color: "#fb923c",
    description: "Erosion & runoff vulnerability from canopy gap, slope, and sand",
    Icon: Layers,
  },
  {
    id: "temp",
    title: "Temperature Band (Max / Min)",
    sensor: "NASA POWER",
    unit: "°C",
    color: "#f87171",
    secondaryColor: "#fbbf24",
    description: "Monthly mean daily high & low temperatures (33°C heat stress line)",
    Icon: Flame,
  },
  {
    id: "zone_dist",
    title: "Zone Stress Breakdown",
    sensor: "6×6 Micro-Zones",
    unit: "zones",
    color: "#8fb04c",
    description: "Distribution of management zones by stress classification",
    Icon: Layers,
  },
  {
    id: "pet_rh",
    title: "Atmospheric Demand (PET & RH)",
    sensor: "NASA POWER",
    unit: "mm / %",
    color: "#c084fc",
    secondaryColor: "#38bdf8",
    description: "Reference evapotranspiration demand (mm) and relative humidity (%)",
    Icon: Wind,
  },
];

interface PanelConfig {
  id: MetricId;
  wide?: boolean;
}

interface Props {
  months: MonthState[];
  monthIdx: number;
  onSelectMonth: (idx: number) => void;
  selectedZone: ZoneProps | null;
  selectedZoneIdx: number;
  scenarioName?: string;
}

export default function MetricsDashboard({
  months, monthIdx, onSelectMonth, selectedZone, selectedZoneIdx, scenarioName,
}: Props) {
  const [panels, setPanels] = useState<PanelConfig[]>([
    { id: "smi" },
    { id: "precip_et" },
    { id: "esi" },
    { id: "zone_dist" },
  ]);
  const [menuOpen, setMenuOpen] = useState(false);

  const activeIds = new Set(panels.map((p) => p.id));
  const cur = months[monthIdx];
  const prev = months[Math.max(0, monthIdx - 1)];

  const addPanel = (id: MetricId) => {
    if (!activeIds.has(id)) {
      setPanels((prevPanels) => [...prevPanels, { id }]);
    }
    setMenuOpen(false);
  };

  const removePanel = (id: MetricId) => {
    setPanels((prevPanels) => prevPanels.filter((p) => p.id !== id));
  };

  const toggleWide = (id: MetricId) => {
    setPanels((prevPanels) =>
      prevPanels.map((p) => (p.id === id ? { ...p, wide: !p.wide } : p)),
    );
  };

  return (
    <section className="space-y-3 pt-1">
      {/* Distinct section header / toolbar (not a card box) */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-800 pb-2.5">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 border-l-4 border-lime-500 pl-2.5">
            <h2 className="text-sm font-bold uppercase tracking-wider text-stone-100">
              Telemetry Panels
            </h2>
          </div>
          <span className="rounded-full border border-stone-700 bg-stone-900/80 px-2.5 py-0.5 text-xs text-stone-300">
            Scope:{" "}
            <b className="text-lime-400">
              {selectedZone ? `Zone ${selectedZone.row + 1}-${selectedZone.col + 1}` : "Field Mean (36 zones)"}
            </b>
          </span>
          <span className="rounded-full border border-stone-700 bg-stone-900/80 px-2.5 py-0.5 text-xs text-stone-400">
            Cursor: <b className="text-stone-200">{cur.label}</b> ({cur.phase})
          </span>
          {scenarioName && (
            <span className="rounded-full border border-violet-700/70 bg-violet-950/60 px-2.5 py-0.5 text-xs text-violet-200">
              Forecast: <b>{scenarioName}</b>
            </span>
          )}
        </div>

        {/* Dropdown to add metric widgets */}
        <div className="relative">
          <button
            onClick={() => setMenuOpen((o) => !o)}
            className="flex items-center gap-1.5 rounded-lg bg-lime-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm transition hover:bg-lime-500"
          >
            <Plus size={14} />
            Add Metric Panel
          </button>

          {menuOpen && (
            <div className="absolute right-0 top-full z-20 mt-1.5 w-80 rounded-xl border border-stone-600 bg-soil-900 p-1.5 shadow-2xl">
              <div className="px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider text-stone-400">
                Select NASA EO / Field Metric
              </div>
              <div className="max-h-80 space-y-0.5 overflow-y-auto">
                {METRIC_PRESETS.map((preset) => {
                  const added = activeIds.has(preset.id);
                  const Icon = preset.Icon;
                  return (
                    <button
                      key={preset.id}
                      onClick={() => (added ? removePanel(preset.id) : addPanel(preset.id))}
                      className={`flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left text-xs transition ${
                        added
                          ? "bg-stone-800/90 text-stone-300 hover:bg-stone-800"
                          : "hover:bg-stone-800/60 text-stone-100"
                      }`}
                    >
                      <Icon size={15} className="mt-0.5 shrink-0" style={{ color: preset.color }} />
                      <div className="flex-1">
                        <div className="flex items-center justify-between font-medium">
                          <span>{preset.title}</span>
                          <span className="text-[10px] text-stone-400">{preset.sensor}</span>
                        </div>
                        <div className="mt-0.5 text-[11px] text-stone-400">{preset.description}</div>
                      </div>
                      <span
                        className={`mt-0.5 rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                          added ? "bg-lime-900/60 text-lime-300" : "bg-stone-700 text-stone-300"
                        }`}
                      >
                        {added ? "Active" : "+ Add"}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Modular widget grid */}
      {panels.length === 0 ? (
        <div className="rounded-xl border border-dashed border-stone-700 p-6 text-center text-xs text-stone-400">
          No telemetry panels active. Click <b>+ Add Metric Panel</b> above to customize your dashboard.
        </div>
      ) : (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {panels.map((panel) => {
            const preset = METRIC_PRESETS.find((p) => p.id === panel.id)!;
            return (
              <MetricWidget
                key={panel.id}
                preset={preset}
                wide={!!panel.wide}
                months={months}
                monthIdx={monthIdx}
                cur={cur}
                prev={prev}
                selectedZoneIdx={selectedZoneIdx}
                onSelectMonth={onSelectMonth}
                onToggleWide={() => toggleWide(panel.id)}
                onRemove={() => removePanel(panel.id)}
              />
            );
          })}
        </div>
      )}
    </section>
  );
}

interface WidgetProps {
  preset: MetricPreset;
  wide: boolean;
  months: MonthState[];
  monthIdx: number;
  cur: MonthState;
  prev: MonthState;
  selectedZoneIdx: number;
  onSelectMonth: (idx: number) => void;
  onToggleWide: () => void;
  onRemove: () => void;
}

function MetricWidget({
  preset, wide, months, monthIdx, cur, prev, selectedZoneIdx, onSelectMonth, onToggleWide, onRemove,
}: WidgetProps) {
  const Icon = preset.Icon;
  const n = months.length;
  const W = 320, H = 68;
  const x = (i: number) => (i / (n - 1)) * W;

  const getCellOrField = (m: MonthState, key: "smi" | "esi" | "bare" | "ndvi" | "vigor") =>
    selectedZoneIdx >= 0 && m.cells[selectedZoneIdx] ? m.cells[selectedZoneIdx][key] : m.field[key];

  const getEt = (m: MonthState) =>
    selectedZoneIdx >= 0 && m.cells[selectedZoneIdx]
      ? m.cells[selectedZoneIdx].et_mm
      : m.cells.reduce((acc, c) => acc + c.et_mm, 0) / m.cells.length;

  // Compute primary series, secondary series, readout, and delta
  let primary: number[] = [];
  let secondary: number[] | null = null;
  let threshold: number | null = null;
  let minY = 0;
  let maxY = 1;
  let mainValue = "";
  let subValue = "";
  let deltaStr = "";

  if (preset.id === "smi") {
    primary = months.map((m) => getCellOrField(m, "smi"));
    const v = primary[monthIdx];
    const d = v - primary[Math.max(0, monthIdx - 1)];
    mainValue = v.toFixed(2);
    subValue = v < 0.3 ? "Dry root-zone" : v > 0.75 ? "Well-watered" : "Optimal range";
    deltaStr = `${d >= 0 ? "+" : ""}${d.toFixed(2)} vs prev mo`;
    threshold = 0.3;
  } else if (preset.id === "esi") {
    primary = months.map((m) => getCellOrField(m, "esi"));
    const v = primary[monthIdx];
    const d = v - primary[Math.max(0, monthIdx - 1)];
    mainValue = v.toFixed(2);
    subValue = v < 0.65 ? "Water stress (<0.65)" : "Unstressed ET";
    deltaStr = `${d >= 0 ? "+" : ""}${d.toFixed(2)} vs prev mo`;
    threshold = 0.65;
  } else if (preset.id === "ndvi") {
    primary = months.map((m) => getCellOrField(m, "ndvi"));
    secondary = months.map((m) => getCellOrField(m, "vigor"));
    const v = primary[monthIdx];
    mainValue = v.toFixed(2);
    subValue = `Vigor ${secondary[monthIdx].toFixed(2)} · ${cur.crop_stage}`;
    const d = v - primary[Math.max(0, monthIdx - 1)];
    deltaStr = `${d >= 0 ? "+" : ""}${d.toFixed(2)} NDVI`;
  } else if (preset.id === "bare") {
    primary = months.map((m) => getCellOrField(m, "bare"));
    const v = primary[monthIdx];
    const exposedCount = cur.cells.filter((c) => c.status === "bare_vulnerable").length;
    mainValue = v.toFixed(2);
    subValue = `${exposedCount}/${cur.cells.length} zones exposed`;
    const d = v - primary[Math.max(0, monthIdx - 1)];
    deltaStr = `${d >= 0 ? "+" : ""}${d.toFixed(2)} risk`;
    threshold = 0.68;
  } else if (preset.id === "precip_et") {
    primary = months.map((m) => m.climate.precip_mm);
    secondary = months.map((m) => getEt(m));
    maxY = Math.max(160, ...primary, ...secondary);
    const p = primary[monthIdx];
    const et = secondary[monthIdx];
    const net = p - et;
    mainValue = `${p.toFixed(0)} mm`;
    subValue = `Crop ET ${et.toFixed(0)} mm`;
    deltaStr = `Net ${net >= 0 ? "+" : ""}${net.toFixed(0)} mm`;
  } else if (preset.id === "temp") {
    primary = months.map((m) => m.climate.tmax_c);
    secondary = months.map((m) => m.climate.tmin_c);
    minY = Math.min(-5, ...secondary);
    maxY = Math.max(40, ...primary);
    mainValue = `${cur.climate.tmax_c.toFixed(1)}°C`;
    subValue = `Low ${cur.climate.tmin_c.toFixed(1)}°C · Solar ${cur.climate.solar_mj.toFixed(1)} MJ`;
    const d = cur.climate.tmax_c - prev.climate.tmax_c;
    deltaStr = `${d >= 0 ? "+" : ""}${d.toFixed(1)}°C`;
    threshold = 33;
  } else if (preset.id === "pet_rh") {
    primary = months.map((m) => m.climate.pet_mm);
    secondary = months.map((m) => m.climate.rh_pct);
    maxY = Math.max(180, ...primary);
    mainValue = `${cur.climate.pet_mm.toFixed(0)} mm`;
    subValue = `Rel. Humidity ${cur.climate.rh_pct.toFixed(0)}%`;
    const d = cur.climate.pet_mm - prev.climate.pet_mm;
    deltaStr = `${d >= 0 ? "+" : ""}${d.toFixed(0)} mm PET`;
  }

  const y = (val: number) => H - ((val - minY) / Math.max(1e-6, maxY - minY)) * (H - 12) - 6;
  const toPath = (arr: number[]) =>
    arr.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  return (
    <div
      className={`flex flex-col justify-between overflow-hidden rounded-xl border border-stone-700/90 bg-soil-800/90 shadow transition hover:border-stone-500 ${
        wide ? "md:col-span-2" : ""
      }`}
    >
      {/* Distinct Grafana-style top header strip */}
      <div className="flex items-center justify-between gap-2 border-b border-stone-700/80 bg-stone-900/80 px-3 py-2">
        <div className="flex items-center gap-2">
          <Icon size={14} style={{ color: preset.color }} />
          <span className="text-xs font-semibold leading-tight text-stone-100">{preset.title}</span>
          <span className="rounded bg-stone-800 px-1.5 py-0.5 text-[10px] text-stone-400">{preset.sensor}</span>
        </div>
        <div className="flex items-center gap-1 text-stone-400">
          <button
            onClick={onToggleWide}
            title={wide ? "Compact width" : "Expand width"}
            className="rounded p-1 hover:bg-stone-700 hover:text-stone-200"
          >
            {wide ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
          </button>
          <button
            onClick={onRemove}
            title="Remove panel"
            className="rounded p-1 hover:bg-stone-700 hover:text-red-400"
          >
            <Trash2 size={12} />
          </button>
        </div>
      </div>

      {/* Body: either Zone Distribution breakdown or Time-series chart */}
      <div className="flex flex-1 flex-col justify-between p-3">
        {preset.id === "zone_dist" ? (
          <ZoneDistributionBody cur={cur} />
        ) : (
          <>
            <div className="mb-2 flex items-baseline justify-between">
              <div>
                <span className="text-xl font-bold text-stone-100">{mainValue}</span>
                <span className="ml-2 text-[11px] text-stone-400">{subValue}</span>
              </div>
              <span className="text-[11px] font-medium text-stone-300">{deltaStr}</span>
            </div>

            <div
              className="relative cursor-pointer select-none"
              title="Click to scrub timeline to month"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                onSelectMonth(Math.round(ratio * (n - 1)));
              }}
            >
              <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-16 w-full overflow-visible">
                {/* Forecast region shade */}
                <rect x={x(11.5)} y={0} width={W - x(11.5)} height={H} fill="#7c5cbf" opacity={0.14} />

                {/* Optional threshold dashed line */}
                {threshold !== null && (
                  <line
                    x1={0}
                    x2={W}
                    y1={y(threshold)}
                    y2={y(threshold)}
                    stroke="#ef4444"
                    strokeDasharray="3 3"
                    strokeWidth={1}
                    opacity={0.7}
                    vectorEffect="non-scaling-stroke"
                  />
                )}

                {/* Secondary series */}
                {secondary && (
                  <path
                    d={toPath(secondary)}
                    fill="none"
                    stroke={preset.secondaryColor ?? "#94a3b8"}
                    strokeWidth={1.75}
                    strokeDasharray={preset.id === "precip_et" ? "3 2" : undefined}
                    vectorEffect="non-scaling-stroke"
                  />
                )}

                {/* Primary series */}
                <path
                  d={toPath(primary)}
                  fill="none"
                  stroke={preset.color}
                  strokeWidth={2.25}
                  vectorEffect="non-scaling-stroke"
                />

                {/* Active month cursor */}
                <line
                  x1={x(monthIdx)}
                  x2={x(monthIdx)}
                  y1={0}
                  y2={H}
                  stroke="#ffffff"
                  strokeWidth={1.5}
                  vectorEffect="non-scaling-stroke"
                />
                <circle cx={x(monthIdx)} cy={y(primary[monthIdx])} r={3.5} fill={preset.color} stroke="#fff" strokeWidth={1} />
              </svg>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function ZoneDistributionBody({ cur }: { cur: MonthState }) {
  const total = cur.cells.length || 1;
  const counts = (Object.keys(STATUS_COLOR) as (keyof typeof STATUS_COLOR)[]).map((status) => ({
    status,
    count: cur.cells.filter((c) => c.status === status).length,
  }));

  return (
    <div className="mt-2 space-y-2">
      <div className="flex h-3 w-full overflow-hidden rounded-full bg-stone-900">
        {counts.map(
          ({ status, count }) =>
            count > 0 && (
              <div
                key={status}
                style={{ width: `${(count / total) * 100}%`, background: STATUS_COLOR[status] }}
                title={`${STATUS_LABEL[status]}: ${count} zones`}
              />
            ),
        )}
      </div>
      <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[11px]">
        {counts.map(({ status, count }) => (
          <div key={status} className="flex items-center justify-between text-stone-300">
            <span className="flex items-center gap-1.5 truncate">
              <i className="h-2 w-2 shrink-0 rounded-sm" style={{ background: STATUS_COLOR[status] }} />
              {STATUS_LABEL[status]}
            </span>
            <b>{count}</b>
          </div>
        ))}
      </div>
    </div>
  );
}

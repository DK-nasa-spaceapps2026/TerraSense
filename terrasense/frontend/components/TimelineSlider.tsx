"use client";

import { Pause, Play } from "lucide-react";
import type { MonthState } from "@/lib/types";

interface Props {
  months: MonthState[];
  index: number;
  onChange: (i: number) => void;
  playing: boolean;
  onTogglePlay: () => void;
  scenarioName?: string;
}

/** 24-month scrubber: 12 observed + 12 forecast. */
export default function TimelineSlider({
  months, index, onChange, playing, onTogglePlay, scenarioName,
}: Props) {
  const n = months.length;
  const W = 1000, H = 54;
  const x = (i: number) => (i / (n - 1)) * W;
  const line = (key: "smi" | "vigor") =>
    months.map((m, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${(H - m.field[key] * (H - 6) - 3).toFixed(1)}`).join(" ");
  const cur = months[index];
  const jumps = [
    { label: "Now", i: 11 },
    { label: "Next spring", i: months.findIndex((m, i) => i > 11 && m.date.endsWith("-04")) },
    { label: "Next summer", i: months.findIndex((m, i) => i > 11 && m.date.endsWith("-07")) },
    { label: "Next fall", i: months.findIndex((m, i) => i > 11 && m.date.endsWith("-10")) },
  ];

  return (
    <div className="rounded-xl border border-stone-700 bg-soil-800/90 p-3.5">
      <div>
        <div className="mb-2 flex flex-wrap items-center gap-3">
            <button onClick={onTogglePlay} className="rounded-full bg-lime-600 p-2 text-white hover:bg-lime-500" aria-label="Play timeline">
              {playing ? <Pause size={16} /> : <Play size={16} />}
            </button>
            <div className="mr-auto">
              <div className="text-lg font-semibold leading-tight">{cur.label}</div>
              <div className="text-xs text-stone-400">
                {cur.phase === "forecast" ? "Forecast scenario" : "Observed (NASA POWER / SMAP / GPM)"} · {cur.crop_stage} ·{" "}
                {cur.climate.tmax_c.toFixed(0)}° / {cur.climate.tmin_c.toFixed(0)}°C · {cur.climate.precip_mm.toFixed(0)} mm rain
              </div>
            </div>

            <div className="hidden items-center gap-1.5 sm:flex">
              <StatPill label="SMI" v={cur.field.smi} />
              <StatPill label="ESI" v={cur.field.esi} />
              <StatPill label="Bare" v={cur.field.bare} />
              <StatPill label="NDVI" v={cur.field.ndvi} />
            </div>

            <div className="hidden gap-1 md:flex">
              {jumps.filter((j) => j.i >= 0).map((j) => (
                <button key={j.label} onClick={() => onChange(j.i)}
                  className="rounded border border-stone-600 px-2 py-1 text-xs hover:bg-stone-700">{j.label}</button>
              ))}
            </div>
          </div>

          <div className="relative">
            <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-14 w-full">
              <rect x={x(11.5)} y={0} width={W - x(11.5)} height={H} fill="#7c5cbf" opacity={0.15} />
              <path d={line("smi")} fill="none" stroke="#4aa3d9" strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
              <path d={line("vigor")} fill="none" stroke="#7cc04a" strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
              {months.map((m, i) => m.alerts.length > 0 && (
                <circle key={i} cx={x(i)} cy={6} r={4} fill={m.alerts.some((a) => a.severity === "high") ? "#ef4444" : "#f59e0b"} />
              ))}
              <line x1={x(index)} x2={x(index)} y1={0} y2={H} stroke="#fff" strokeWidth={2} vectorEffect="non-scaling-stroke" />
            </svg>
            <input type="range" min={0} max={n - 1} value={index} onChange={(e) => onChange(+e.target.value)}
              className="w-full" aria-label="Month" />
            <div className="flex justify-between text-[10px] text-stone-500">
              {months.map((m, i) => (i % 3 === 0 ? <span key={i}>{m.label}</span> : <span key={i} />))}
            </div>
          </div>

          <div className="mt-1 flex flex-wrap gap-4 text-[11px] text-stone-400">
            <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-sky-500" />Soil moisture</span>
            <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-lime-500" />Vigor</span>
            <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-red-500" />Alert month</span>
            <span className="text-violet-300">Shaded = forecast ({scenarioName})</span>
          </div>
      </div>
    </div>
  );
}

const StatPill = ({ label, v }: { label: string; v: number }) => (
  <div className="rounded bg-stone-800/90 px-2 py-1 text-center text-[11px]">
    <span className="text-stone-400">{label} </span>
    <b className="text-stone-100">{v.toFixed(2)}</b>
  </div>
);

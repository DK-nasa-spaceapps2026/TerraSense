"use client";

import { useEffect, useState } from "react";
import { MapPin, User, X } from "lucide-react";

export interface FarmProfile {
  farmName: string;
  countyState: string;
  latitude: number;
  longitude: number;
  areaSqFt: number;
  notes: string;
}

export const DEFAULT_FARM_PROFILE: FarmProfile = {
  farmName: "Central Iowa Parcel",
  countyState: "Polk County, IA",
  latitude: 41.6,
  longitude: -93.6,
  areaSqFt: 8064347,
  notes: "Clarion-Nicollet-Webster soil association · Tile-drained row crop",
};

interface Props {
  open: boolean;
  value: FarmProfile;
  gridLabel: string;
  onClose: () => void;
  onSave: (next: FarmProfile) => void;
}

export default function FarmProfileModal({ open, value, gridLabel, onClose, onSave }: Props) {
  const [draft, setDraft] = useState<FarmProfile>(value);

  useEffect(() => {
    if (open) setDraft(value);
  }, [open, value]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
      <div className="flex max-h-[90vh] w-full max-w-lg flex-col rounded-2xl border border-stone-700 bg-soil-900 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-stone-800 px-5 py-4">
          <div className="flex items-center gap-2.5">
            <User size={18} className="text-lime-400" />
            <div>
              <h2 className="text-base font-bold text-stone-100">Farm Location Profile</h2>
              <p className="text-xs text-stone-400">Parcel metadata, coordinates &amp; field footprint</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-stone-400 hover:bg-stone-800 hover:text-stone-100"
          >
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div className="space-y-4 overflow-y-auto p-5 text-xs">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="block">
              <span className="mb-1 block text-stone-400">Farm / Parcel Name</span>
              <input
                type="text"
                value={draft.farmName}
                onChange={(e) => setDraft({ ...draft, farmName: e.target.value })}
                className="w-full rounded-lg border border-stone-700 bg-stone-900 px-3 py-1.5 text-stone-100 focus:border-lime-500 focus:outline-none"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-stone-400">County / State</span>
              <input
                type="text"
                value={draft.countyState}
                onChange={(e) => setDraft({ ...draft, countyState: e.target.value })}
                className="w-full rounded-lg border border-stone-700 bg-stone-900 px-3 py-1.5 text-stone-100 focus:border-lime-500 focus:outline-none"
              />
            </label>
          </div>

          <div className="space-y-2.5 rounded-xl border border-stone-800 bg-soil-800 p-3.5">
            <div className="flex items-center gap-1.5 font-semibold text-stone-200">
              <MapPin size={14} className="text-lime-400" />
              Coordinates &amp; Parcel Footprint
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="block">
                <span className="mb-1 block text-stone-400">Latitude (°)</span>
                <input
                  type="number"
                  step={0.0001}
                  min={-90}
                  max={90}
                  value={draft.latitude}
                  onChange={(e) => setDraft({ ...draft, latitude: +e.target.value })}
                  className="w-full rounded-lg border border-stone-700 bg-stone-900 px-3 py-1.5 font-mono text-stone-100 focus:border-lime-500 focus:outline-none"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-stone-400">Longitude (°)</span>
                <input
                  type="number"
                  step={0.0001}
                  min={-180}
                  max={180}
                  value={draft.longitude}
                  onChange={(e) => setDraft({ ...draft, longitude: +e.target.value })}
                  className="w-full rounded-lg border border-stone-700 bg-stone-900 px-3 py-1.5 font-mono text-stone-100 focus:border-lime-500 focus:outline-none"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-stone-400">Grid Area (sq ft)</span>
                <input
                  type="number"
                  step={1000}
                  min={10000}
                  value={draft.areaSqFt}
                  onChange={(e) => setDraft({ ...draft, areaSqFt: Math.max(10000, +e.target.value) })}
                  className="w-full rounded-lg border border-stone-700 bg-stone-900 px-3 py-1.5 font-mono text-stone-100 focus:border-lime-500 focus:outline-none"
                />
              </label>
              <div className="block">
                <span className="mb-1 block text-stone-400">Grid Resolution</span>
                <div className="rounded-lg border border-stone-800 bg-stone-900/60 px-3 py-1.5 font-mono text-stone-300">
                  {gridLabel}
                </div>
              </div>
            </div>
          </div>

          <label className="block">
            <span className="mb-1 block text-stone-400">Field / Soil Notes</span>
            <textarea
              rows={2}
              value={draft.notes}
              onChange={(e) => setDraft({ ...draft, notes: e.target.value })}
              className="w-full rounded-lg border border-stone-700 bg-stone-900 px-3 py-1.5 text-stone-100 focus:border-lime-500 focus:outline-none"
            />
          </label>

          <p className="text-[11px] text-stone-500">
            Visual profile preview — demo ML forecasts remain anchored to the trained central Iowa archive.
          </p>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end border-t border-stone-800 px-5 py-3.5">
          <button
            type="button"
            onClick={() => {
              onSave(draft);
              onClose();
            }}
            className="rounded-lg bg-lime-600 px-4 py-1.5 text-xs font-semibold text-white shadow hover:bg-lime-500"
          >
            Save Profile
          </button>
        </div>
      </div>
    </div>
  );
}

"""NASA POWER integration harness with deterministic mock fallback.

Live:  set TERRASENSE_LIVE=1 (no API key needed for POWER). Any failure falls
       back to the mock generator so the demo never breaks.
Mock:  seasonal climatology (+ injected Jul/Aug dry-heat anomaly in the most
       recent summer so stress states are visible in the demo).
"""
from __future__ import annotations

import json
import math
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = "PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M,ALLSKY_SFC_SW_DWN"
DATA_DIR = Path(__file__).resolve().parent / "data" / "nasa_power"

_cache: Dict[tuple, Tuple[List[dict], str]] = {}
_local_index: Optional[List[Tuple[float, float, Path]]] = None


def _get_local_archive_path(lat: float, lon: float) -> Optional[Path]:
    """Find nearest local NASA POWER JSON file in data/nasa_power/."""
    global _local_index
    if not DATA_DIR.exists():
        return None
    if _local_index is None:
        idx: List[Tuple[float, float, Path]] = []
        for p in sorted(DATA_DIR.glob("*.json")):
            try:
                with p.open("r") as f:
                    head = f.read(256)
                # Fast header parse for lat/lon
                obj = json.loads(head + "]}" if not head.strip().endswith("}") else head)
            except Exception:
                try:
                    obj = json.loads(p.read_text())
                except Exception:
                    continue
            if "lat" in obj and "lon" in obj:
                idx.append((float(obj["lat"]), float(obj["lon"]), p))
        _local_index = idx
    if not _local_index:
        return None
    best = min(_local_index, key=lambda item: (item[0] - lat) ** 2 + (item[1] - lon) ** 2)
    return best[2]


def load_local_daily(lat: float, lon: float, start: date, end: date) -> Optional[List[dict]]:
    """Load real NASA POWER daily observations from local archive (2023-2024)."""
    path = _get_local_archive_path(lat, lon)
    if path is None:
        return None
    data = json.loads(path.read_text())
    daily_list = data.get("daily") or []
    if not daily_list:
        return None
    by_date = {r["date"]: r for r in daily_list}
    by_md = {(r["date"][:4], r["date"][5:]): r for r in daily_list}
    out: List[dict] = []
    d = start
    while d <= end:
        iso = d.isoformat()
        row = by_date.get(iso)
        if row is None:
            # Map years outside 2023-2024 onto 2023 (odd) / 2024 (even) observed archive
            target_yr = "2024" if d.year % 2 == 0 else "2023"
            md = iso[5:]
            if md == "02-29":
                md = "02-28"
            src = by_md.get((target_yr, md)) or by_md.get(("2024", md)) or daily_list[-1]
            row = {**src, "date": iso}
        out.append(row)
        d += timedelta(days=1)
    return out


def _seasonal(lat: float, doy: int) -> float:
    """-1..1 seasonal phase, peak mid-July (N hemisphere)."""
    phase = math.sin(2 * math.pi * (doy - 110) / 365.0)
    return phase if lat >= 0 else -phase


def mock_normals_month(lat: float, month: int) -> dict:
    """Climatological monthly normals for forecast blending."""
    doy = int((month - 0.5) * 30.4)
    s = _seasonal(lat, doy)
    tmean = 30 - 0.45 * abs(lat) + 0.28 * abs(lat) * s
    return {
        "tmax": tmean + 5.0, "tmin": tmean - 5.0,
        "precip_mm": 62 + 62 * max(s, -0.3), "rh": 70 - 0.8 * (tmean - 12),
        "solar": max(3.5, 14 + 10 * s),
    }


def mock_daily(lat: float, lon: float, start: date, end: date) -> List[dict]:
    out = []
    d = start
    hot_year = end.year if end.month >= 9 else end.year - 1  # most recent summer
    while d <= end:
        rng = np.random.default_rng(abs(hash((round(lat, 1), round(lon, 1), d.toordinal()))) % (2**32))
        n = mock_normals_month(lat, d.month)
        s = _seasonal(lat, d.timetuple().tm_yday)
        wet_p = 0.26 + 0.08 * max(s, 0)
        mean_amt = n["precip_mm"] / 30.4 / wet_p
        anomaly_dry = d.year == hot_year and d.month in (7, 8)
        if anomaly_dry:
            wet_p *= 0.45
        precip = float(rng.exponential(mean_amt)) if rng.random() < wet_p else 0.0
        heat = 3.2 if anomaly_dry else 0.0
        tmax = n["tmax"] + heat + rng.normal(0, 2.4)
        tmin = n["tmin"] + heat * 0.6 + rng.normal(0, 2.0)
        out.append({
            "date": d.isoformat(), "precip": round(precip, 2),
            "tmax": round(tmax, 1), "tmin": round(min(tmin, tmax - 3), 1),
            "rh": round(float(np.clip(n["rh"] - (8 if anomaly_dry else 0) + rng.normal(0, 7), 25, 98)), 1),
            "solar": round(float(max(2, n["solar"] * (0.55 if precip > 8 else 1) + rng.normal(0, 1.5))), 2),
        })
        d += timedelta(days=1)
    return out


def fetch_live(lat: float, lon: float, start: date, end: date) -> List[dict]:
    import requests
    r = requests.get(POWER_URL, params={
        "parameters": PARAMS, "community": "AG", "longitude": lon, "latitude": lat,
        "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "format": "JSON",
    }, timeout=20)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    rows = []
    for k in sorted(p["T2M_MAX"]):
        g = lambda name: p[name][k] if p[name][k] > -900 else None
        if g("T2M_MAX") is None:
            continue
        rows.append({
            "date": f"{k[:4]}-{k[4:6]}-{k[6:]}", "precip": g("PRECTOTCORR") or 0.0,
            "tmax": g("T2M_MAX"), "tmin": g("T2M_MIN"),
            "rh": g("RH2M") or 60.0, "solar": g("ALLSKY_SFC_SW_DWN") or 15.0,
        })
    if len(rows) < 200:
        raise ValueError("POWER returned too little data")
    return rows


def get_daily(lat: float, lon: float, start: date, end: date, live: bool | None = None) -> Tuple[List[dict], str]:
    """Returns (daily rows, source) where source is 'nasa_power' or 'mock'."""
    if live is None:
        live = os.getenv("TERRASENSE_LIVE", "0") == "1"
    key = (round(lat, 2), round(lon, 2), start, end, live)
    if key in _cache:
        return _cache[key]
    result = None
    if live:
        try:
            result = (fetch_live(lat, lon, start, end), "nasa_power")
        except Exception as exc:  # network/API failure -> mock
            print(f"[nasa_power] live fetch failed ({exc}); using mock")
    if result is None:
        local_rows = load_local_daily(lat, lon, start, end)
        if local_rows:
            result = (local_rows, "NASA POWER (2023–2024 archive)")
        else:
            result = (mock_daily(lat, lon, start, end), "mock")
    _cache[key] = result
    return result


def to_monthly(daily: List[dict]) -> List[dict]:
    """Aggregate daily rows into monthly climate dicts."""
    buckets: Dict[str, List[dict]] = {}
    for r in daily:
        buckets.setdefault(r["date"][:7], []).append(r)
    out = []
    for ym in sorted(buckets):
        rows = buckets[ym]
        out.append({
            "month": ym, "days": len(rows),
            "tmax": float(np.mean([r["tmax"] for r in rows])),
            "tmin": float(np.mean([r["tmin"] for r in rows])),
            "precip_mm": float(np.sum([r["precip"] for r in rows])) * 30.4 / len(rows),
            "rh": float(np.mean([r["rh"] for r in rows])),
            "solar": float(np.mean([r["solar"] for r in rows])),
            "max_daily_precip": float(np.max([r["precip"] for r in rows])),
        })
    return out

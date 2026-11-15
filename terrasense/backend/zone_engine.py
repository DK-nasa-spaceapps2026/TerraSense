"""TerraSense zone engine.

1. build_grid()      -> N x N micro-zones (GeoJSON) with fused soil profiles
2. simulate()        -> vectorised monthly water balance per zone producing
                        Soil Moisture Index, Evaporative Stress Index (ET/PET),
                        Bare Soil Exposure Risk, NDVI proxy, vigor and status
3. build_timeline()  -> 12 months historical + 12 months forecast states
4. compare_scenarios() -> 3 rotation strategies with quantified trade-offs
5. build_guidance()  -> farmer-first seeding / vulnerability / cover windows
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np

import nasa_power

logger = logging.getLogger("terrasense.zone_engine")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
STATUSES = ["lush", "moderate", "water_stress", "heat_stress", "bare_vulnerable"]
N_HIST = 12
N_TOTAL = 24

DEMO_CENTER = (-93.60, 41.60)  # lon, lat (central Iowa)
DEMO_HALF = 0.0045


def demo_bbox() -> List[float]:
    lon, lat = DEMO_CENTER
    return [lon - DEMO_HALF, lat - DEMO_HALF, lon + DEMO_HALF, lat + DEMO_HALF]


# --------------------------------------------------------------------------- #
# Crop plans                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class CropPlan:
    id: str
    name: str
    crops: str
    plant: int          # planting month (1-12)
    harvest: int        # harvest month
    kc_peak: float
    cover_peak: float   # canopy cover at peak
    off_cover: float    # residue / stubble cover out of season
    cc_cover: Optional[float]  # cover-crop canopy (None = no cover crop)
    cc_kc: float
    n_fix: float        # kg N/ha fixed per year
    som_gain: float     # % SOM / yr
    margin: float       # USD/ha baseline
    drought_sens: float # yield sensitivity to water stress
    description: str


PLANS: Dict[str, CropPlan] = {
    "status_quo": CropPlan(
        "status_quo", "Status Quo", "Corn → Soybean, winter fallow", 5, 10, 1.15, 0.95, 0.10,
        None, 0.0, 0, 0.02, 820, 1.15, "Continue conventional rotation with bare winter fallow."),
    "cover_legume": CropPlan(
        "cover_legume", "Legume + Cover Crop", "Soybean → crimson clover / rye cover", 5, 9, 1.05, 0.90, 0.15,
        0.70, 0.50, 75, 0.14, 745, 0.85, "Add a legume cover crop after harvest to fix N and armour soil."),
    "drought_cash": CropPlan(
        "drought_cash", "Drought-Tolerant Cash Crop", "Sorghum / sunflower, stubble retained", 6, 9, 0.85, 0.80, 0.28,
        None, 0.0, 0, 0.05, 690, 0.45, "Switch to low-water cash crops with retained stubble."),
    "corn_rye_cover": CropPlan(
        "corn_rye_cover", "No-Till Corn + Winter Rye Armor", "No-till Corn → winter cereal rye cover", 5, 10, 1.12, 0.92, 0.35,
        0.65, 0.45, 15, 0.11, 745, 0.95, "High-residue no-till corn followed by drilled winter cereal rye."),
    "alfalfa_perennial": CropPlan(
        "alfalfa_perennial", "Perennial Alfalfa Meadow", "Alfalfa / meadow brome forage rotation", 4, 10, 1.05, 0.92, 0.78,
        0.80, 0.55, 140, 0.24, 620, 0.55, "Deep-rooted perennial legume forage providing year-round cover."),
}


def _calendar(plan: CropPlan) -> Dict[str, np.ndarray]:
    """Monthly (index 0..11) canopy cover and crop coefficient arrays."""
    cover, kc = np.zeros(12), np.zeros(12)
    s, e = plan.plant, plan.harvest
    for m in range(1, 13):
        if s <= m <= e:
            t = (m - s + 0.5) / (e - s + 1)
            w = math.sin(math.pi * t)
            cover[m - 1] = plan.off_cover + (plan.cover_peak - plan.off_cover) * w ** 0.7
            kc[m - 1] = 0.3 + (plan.kc_peak - 0.3) * w ** 0.8
        else:
            d = (m - e) % 12
            if plan.cc_cover is not None:
                ramp = min(1.0, d / 3.0)
                cover[m - 1] = plan.off_cover + (plan.cc_cover - plan.off_cover) * ramp
                kc[m - 1] = 0.25 + (plan.cc_kc - 0.25) * ramp
            else:
                cover[m - 1], kc[m - 1] = plan.off_cover, 0.25
    return {"cover": cover, "kc": kc}


CALENDARS = {k: _calendar(v) for k, v in PLANS.items()}


# --------------------------------------------------------------------------- #
# Geometry / grid                                                             #
# --------------------------------------------------------------------------- #
def parse_field(field: Optional[dict]) -> dict:
    """Accept {'bbox': [w,s,e,n]} or {'polygon': GeoJSON geometry | ring}."""
    if not field:
        field = {"bbox": demo_bbox()}
    ring = None
    if "polygon" in field:
        poly = field["polygon"]
        coords = poly["coordinates"][0] if isinstance(poly, dict) else poly
        ring = [[float(x), float(y)] for x, y in coords]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        xs, ys = [p[0] for p in ring], [p[1] for p in ring]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    else:
        bbox = [float(v) for v in field.get("bbox", demo_bbox())]
        w, s, e, n = bbox
        ring = [[w, s], [e, s], [e, n], [w, n], [w, s]]
    if not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise ValueError("Invalid bbox/polygon coordinates")
    return {"bbox": bbox, "ring": ring}


def field_id_for(geom: dict, grid_n: int) -> str:
    raw = json.dumps([geom["ring"], grid_n], sort_keys=True)
    return "fld_" + hashlib.sha1(raw.encode()).hexdigest()[:10]


def _point_in_ring(x: float, y: float, ring: List[List[float]]) -> bool:
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _smooth_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    """Spatially correlated noise in 0..1 (box-blurred white noise)."""
    a = rng.random((n + 4, n + 4))
    for _ in range(2):
        p = np.pad(a, 1, mode="edge")
        a = sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9.0
    a = a[2:-2, 2:-2]
    return (a - a.min()) / (np.ptp(a) + 1e-9)


def build_grid(geom: dict, grid_n: int, field_id: str) -> dict:
    """Return dict with GeoJSON features and per-zone soil arrays."""
    w, s, e, n = geom["bbox"]
    seed = int(field_id[4:], 16) % (2**32)
    rng = np.random.default_rng(seed)
    sand = 0.15 + 0.55 * _smooth_noise(rng, grid_n)
    clay = np.clip(0.38 - 0.35 * sand + rng.normal(0, 0.02, (grid_n, grid_n)), 0.08, 0.45)
    som = 1.6 + 3.0 * (1 - _smooth_noise(rng, grid_n) * 0.7 - sand * 0.4).clip(0, 1)
    slope = 0.5 + 8.0 * _smooth_noise(rng, grid_n)           # percent
    awc = 1200 * (0.26 - 0.16 * sand)                        # mm, 1.2 m root zone

    dx, dy = (e - w) / grid_n, (n - s) / grid_n
    feats, idx = [], []
    # Anchor zone soil & terrain to real SSURGO + 30m DEM reference points when available
    ref_interp = None
    try:
        from ml.predict import get_predictor
        pred = get_predictor()
        if pred is not None:
            all_coords = [(w + (c + 0.5) * dx, n - (r + 0.5) * dy) for r in range(grid_n) for c in range(grid_n)]
            flat_soil = {k: arr.ravel() for k, arr in dict(sand=sand, clay=clay, som=som, slope=slope, awc=awc).items()}
            ref_interp = pred.interpolate_zone_static_df(field_id, all_coords, flat_soil)
            # Blend real IDW-interpolated SSURGO/DEM baseline with sub-field spatial heterogeneity
            sand = np.clip(0.55 * (ref_interp["sand_top_pct"].to_numpy().reshape(grid_n, grid_n) / 100.0) + 0.45 * sand, 0.05, 0.85)
            clay = np.clip(0.55 * (ref_interp["clay_top_pct"].to_numpy().reshape(grid_n, grid_n) / 100.0) + 0.45 * clay, 0.08, 0.55)
            som = np.clip(0.60 * ref_interp["om_top_pct"].to_numpy().reshape(grid_n, grid_n) + 0.40 * som, 1.0, 7.5)
            slope = np.clip(0.55 * ref_interp["slope_pct"].to_numpy().reshape(grid_n, grid_n) + 0.45 * slope, 0.2, 18.0)
            awc = np.clip(0.60 * ref_interp["awc_100cm_mm"].to_numpy().reshape(grid_n, grid_n) + 0.40 * awc, 90.0, 280.0)
    except Exception:
        ref_interp = None

    for r in range(grid_n):          # r=0 is the northern row
        for c in range(grid_n):
            x0, y1 = w + c * dx, n - r * dy
            if not _point_in_ring(x0 + dx / 2, y1 - dy / 2, geom["ring"]):
                continue
            flat_i = r * grid_n + c
            extra_soil = {}
            if ref_interp is not None:
                row_ref = ref_interp.iloc[flat_i]
                extra_soil = {
                    "ph": round(float(row_ref.get("ph_top", 6.5)), 1),
                    "elev_m": round(float(row_ref.get("elev_m", 235.0)), 1),
                    "twi": round(float(row_ref.get("twi", 7.5)), 2),
                }
            idx.append((r, c))
            feats.append({
                "type": "Feature",
                "id": f"z{r}{c}",
                "geometry": {"type": "Polygon", "coordinates": [[
                    [x0, y1], [x0 + dx, y1], [x0 + dx, y1 - dy], [x0, y1 - dy], [x0, y1]]]},
                "properties": {
                    "kind": "zone", "zone_id": f"z{r}{c}", "row": r, "col": c,
                    "soil": {
                        "sand_pct": round(float(sand[r, c]) * 100, 1),
                        "clay_pct": round(float(clay[r, c]) * 100, 1),
                        "organic_matter_pct": round(float(som[r, c]), 2),
                        "slope_pct": round(float(slope[r, c]), 1),
                        "available_water_mm": round(float(awc[r, c]), 0),
                        **extra_soil,
                    },
                },
            })
    if not feats:
        raise ValueError("Polygon too small: no grid cells fall inside the boundary")
    ii = np.array(idx)
    soil = {k: arr[ii[:, 0], ii[:, 1]] for k, arr in
            dict(sand=sand, clay=clay, som=som, slope=slope, awc=awc).items()}
    zone_coords = [(w + (c + 0.5) * dx, n - (r + 0.5) * dy) for r, c in idx]
    return {"features": feats, "soil": soil, "n_zones": len(feats),
            "field_id": field_id, "zone_coords": zone_coords}


# --------------------------------------------------------------------------- #
# Climate timeline                                                            #
# --------------------------------------------------------------------------- #
def _add_months(d: date, k: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 + k, 12)
    return date(y, m + 1, 1)


def build_climate(geom: dict, anchor: Optional[date], temp_shift: float, precip_factor: float,
                  live: Optional[bool] = None) -> dict:
    """24 monthly climate dicts: 12 observed + 12 forecast (anchor = current month = idx 11)."""
    lat, lon = (geom["bbox"][1] + geom["bbox"][3]) / 2, (geom["bbox"][0] + geom["bbox"][2]) / 2
    anchor = (anchor or date(2024, 11, 1)).replace(day=1)
    start = _add_months(anchor, -(N_HIST - 1))
    end = _add_months(anchor, 1) - timedelta(days=1)
    if live:
        end = min(end, date.today() - timedelta(days=3))  # POWER latency
    daily, source = nasa_power.get_daily(lat, lon, start, end, live)
    hist = nasa_power.to_monthly(daily)
    while len(hist) < N_HIST:       # safety (short live data)
        hist.append(dict(hist[-1]))
    hist = hist[:N_HIST]

    months = []
    for i in range(N_TOTAL):
        d = _add_months(start, i)
        if i < N_HIST:
            c = dict(hist[i]); c["phase"] = "historical"
        else:
            last_year = hist[i - N_HIST]
            nrm = nasa_power.mock_normals_month(lat, d.month)
            c = {k: 0.6 * nrm[k] + 0.4 * last_year[k] for k in ("tmax", "tmin", "precip_mm", "rh", "solar")}
            c["max_daily_precip"] = last_year["max_daily_precip"]
            c["days"] = 30
            c["phase"] = "forecast"
            c["tmax"] += temp_shift; c["tmin"] += temp_shift
            c["precip_mm"] *= precip_factor
            c["rh"] = float(np.clip(c["rh"] - 2 * temp_shift, 20, 98))
        c["date"] = d.isoformat()[:7]
        c["cal_month"] = d.month
        c["tmean"] = (c["tmax"] + c["tmin"]) / 2
        # Reference ET (Hargreaves-Samani with measured Rs): mm/day -> mm/month
        c["pet_mm"] = max(0.2, 0.0045 * (c["tmean"] + 17.8) * c["solar"]) * c.get("days", 30)
        months.append(c)
    return {"months": months, "source": source}


# --------------------------------------------------------------------------- #
# Water-balance simulation (vectorised across zones)                          #
# --------------------------------------------------------------------------- #
def active_engine_info() -> Dict[str, Optional[str]]:
    """Return active engine mode ('ml' or 'heuristic') and model_version."""
    try:
        from ml.predict import get_predictor, resolve_engine_mode
        mode = resolve_engine_mode()
        if mode == "ml":
            pred = get_predictor()
            if pred is not None:
                return {"engine": "ml", "model_version": pred.model_version}
    except Exception as exc:
        logger.warning("[zone_engine] Engine resolution fallback to heuristic: %s", exc)
    return {"engine": "heuristic", "model_version": None}


def simulate_heuristic(soil: Dict[str, np.ndarray], months: List[dict], plan_ids: List[str],
                       irrigation_cap_mm: float = 0.0) -> Dict[str, np.ndarray]:
    nz, T = len(soil["awc"]), len(months)
    names = ["smi", "esi", "bare", "ndvi", "vigor", "heat", "et", "demand", "runoff", "precip",
             "irrigation_mm", "extrapolated"]
    out = {k: np.zeros((T, nz)) for k in names}
    sm = 0.65 * soil["awc"]
    slope_n = np.clip(soil["slope"] / 8.0, 0, 1)
    rem_irrig_cap = np.full(nz, max(0.0, float(irrigation_cap_mm)))

    for t, c in enumerate(months):
        cal = CALENDARS[plan_ids[t]]
        cover = np.full(nz, cal["cover"][c["cal_month"] - 1])
        kc = cal["kc"][c["cal_month"] - 1]
        P = c["precip_mm"]
        intensity = float(np.clip(c.get("max_daily_precip", P / 8) / 40.0, 0, 1))

        # Runoff: steeper, bare, clayey, intense-storm months lose more
        runoff_frac = np.clip(0.04 + 0.025 * soil["slope"] + 0.18 * (1 - cover) * (0.6 + intensity)
                              + 0.15 * soil["clay"], 0, 0.55)
        runoff = P * runoff_frac
        infil = P - runoff
        demand = c["pet_mm"] * kc

        irrig_step = np.zeros(nz)
        is_irrigated = False
        if irrigation_cap_mm > 0.0 and t >= N_HIST and c["cal_month"] in (6, 7, 8):
            avail_nat = sm + 0.5 * infil
            et_nat = np.minimum(demand * np.clip(avail_nat / (0.35 * soil["awc"]), 0.05, 1.0), sm + infil)
            sm_nat_end = np.minimum(soil["awc"], sm + infil - et_nat)
            smi_nat = 0.5 * (sm + sm_nat_end) / soil["awc"]
            trig = (smi_nat < 0.68) & (rem_irrig_cap > 0.0)
            if np.any(trig):
                deficit = np.maximum(0.0, 0.85 * soil["awc"] - sm_nat_end)
                irrig_step = np.where(trig, np.minimum(deficit, rem_irrig_cap), 0.0)
                rem_irrig_cap = np.maximum(0.0, rem_irrig_cap - irrig_step)
                infil = infil + irrig_step
                is_irrigated = bool(np.any(irrig_step > 0.0))

        avail = sm + 0.5 * infil
        beta = np.clip(avail / (0.35 * soil["awc"]), 0.05, 1.0)       # stress below 50% AWC
        heat = float(np.clip((c["tmax"] - 30) / 8, 0, 1)) * (cover > 0.4)
        beta_eff = beta * (1 - 0.25 * heat)
        et = np.minimum(demand * beta_eff, sm + infil)
        sm_new = np.minimum(soil["awc"], sm + infil - et)
        smi = 0.5 * (sm + sm_new) / soil["awc"]
        if irrigation_cap_mm > 0.0 and t >= N_HIST:
            smi = np.clip(smi, 0.02, 0.94)
        sm = sm_new

        esi = np.where(demand > 1e-6, et / np.maximum(demand, 1e-6), 1.0)
        rain_i = min(P / 120.0, 1.0)
        bare = np.clip((1 - cover) * 0.42 + slope_n * 0.24 + soil["sand"] * 0.30
                       + rain_i * 0.10 * (1 - cover) + (1 - smi) * 0.10 - 0.03, 0, 1)
        ndvi = np.clip(0.12 + 0.72 * cover * (0.45 + 0.55 * esi), 0.05, 0.9)
        vigor = np.clip(0.5 * (ndvi - 0.1) / 0.7 + 0.3 * esi + 0.2 * smi, 0, 1) \
            * (0.4 + 0.6 * cover) / 1.0
        for k, v in dict(smi=smi, esi=esi, bare=bare, ndvi=ndvi, vigor=vigor, heat=heat * np.ones(nz),
                         et=et, demand=demand * np.ones(nz), runoff=runoff, precip=P * np.ones(nz),
                         irrigation_mm=irrig_step, extrapolated=(irrig_step > 0.0).astype(float)).items():
            out[k][t] = v
    return out


def simulate(soil: Dict[str, np.ndarray], months: List[dict], plan_ids: List[str],
             field_id: str = "default", zone_coords: Optional[List[tuple]] = None,
             irrigation_cap_mm: float = 0.0) -> Dict[str, np.ndarray]:
    try:
        from ml.predict import get_predictor, resolve_engine_mode
        if resolve_engine_mode() == "ml":
            pred = get_predictor()
            if pred is not None:
                return pred.simulate_vectorized(
                    soil=soil, months=months, plan_ids=plan_ids, cal_lookup=CALENDARS,
                    field_id=field_id, zone_coords=zone_coords, irrigation_cap_mm=irrigation_cap_mm,
                )
    except Exception as exc:
        logger.warning("[zone_engine] ML simulate failed (%s); falling back to heuristic engine.", exc)
    return simulate_heuristic(soil, months, plan_ids, irrigation_cap_mm=irrigation_cap_mm)


def classify(sim: Dict[str, np.ndarray]) -> np.ndarray:
    """Status index per (t, zone) into STATUSES."""
    status = np.full(sim["smi"].shape, 1, dtype=int)                       # moderate
    status[sim["vigor"] > 0.62] = 0                                         # lush
    status[(sim["heat"] > 0.45) & (sim["esi"] < 0.9)] = 3                  # heat stress
    status[(sim["esi"] < 0.65) | (sim["smi"] < 0.30)] = 2                  # water stress
    status[sim["bare"] >= 0.68] = 4                                         # bare
    return status


# --------------------------------------------------------------------------- #
# Timeline payload                                                            #
# --------------------------------------------------------------------------- #
def _alerts(c: dict, sim_t: Dict[str, np.ndarray], st: np.ndarray) -> List[dict]:
    out = []
    frac = lambda code: float(np.mean(st == code))
    if frac(2) >= 0.30:
        out.append({"type": "water_stress", "severity": "high" if frac(2) > 0.6 else "medium",
                    "message": f"{frac(2):.0%} of zones are water stressed."})
    if c["tmax"] >= 33:
        out.append({"type": "heat_stress", "severity": "high" if c["tmax"] >= 36 else "medium",
                    "message": f"Mean daily max {c['tmax']:.0f} °C — heat stress window."})
    if frac(4) >= 0.40:
        out.append({"type": "bare_soil", "severity": "high" if frac(4) > 0.8 else "medium",
                    "message": f"{frac(4):.0%} of zones have exposed soil — erosion risk."})
    return out


def build_timeline(field_id: str, grid: dict, climate: dict, scenario: str,
                   irrigation_cap_mm: float = 0.0) -> dict:
    plans = ["status_quo"] * N_HIST + [scenario] * (N_TOTAL - N_HIST)
    eng_info = active_engine_info()
    sim = simulate(
        grid["soil"], climate["months"], plans,
        field_id=field_id, zone_coords=grid.get("zone_coords"),
        irrigation_cap_mm=irrigation_cap_mm,
    )
    status = classify(sim)
    r3 = lambda a: round(float(a), 3)
    has_unc = "smi_p10" in sim
    months = []
    for t, c in enumerate(climate["months"]):
        d = c["date"]
        cells = [{"smi": r3(sim["smi"][t, z]), "esi": r3(sim["esi"][t, z]), "bare": r3(sim["bare"][t, z]),
                  "ndvi": r3(sim["ndvi"][t, z]), "vigor": r3(sim["vigor"][t, z]),
                  "et_mm": round(float(sim["et"][t, z]), 1), "status": STATUSES[status[t, z]]}
                 for z in range(grid["n_zones"])]
        field_summary = {
            "smi": r3(sim["smi"][t].mean()), "esi": r3(sim["esi"][t].mean()),
            "bare": r3(sim["bare"][t].mean()), "ndvi": r3(sim["ndvi"][t].mean()),
            "vigor": r3(sim["vigor"][t].mean()),
        }
        if irrigation_cap_mm > 0.0 and "irrigation_mm" in sim:
            field_summary["irrigation_mm"] = round(float(sim["irrigation_mm"][t].mean()), 1)
        m_dict = {
            "index": t, "date": d, "label": f"{MONTHS[c['cal_month'] - 1]} {d[:4]}",
            "phase": c["phase"],
            "climate": {"tmax_c": round(c["tmax"], 1), "tmin_c": round(c["tmin"], 1),
                        "precip_mm": round(c["precip_mm"], 1), "rh_pct": round(c["rh"], 1),
                        "solar_mj": round(c["solar"], 1), "pet_mm": round(c["pet_mm"], 1)},
            "crop_stage": _crop_stage(PLANS[plans[t]], c["cal_month"]),
            "field": field_summary,
            "alerts": _alerts(c, {k: v[t] for k, v in sim.items()}, status[t]),
            "cells": cells,
        }
        if has_unc:
            m_dict["uncertainty"] = {
                "smi_p10": r3(sim["smi_p10"][t].mean()),
                "smi_p90": r3(sim["smi_p90"][t].mean()),
                "esi_p10": r3(sim["esi_p10"][t].mean()),
                "esi_p90": r3(sim["esi_p90"][t].mean()),
            }
        months.append(m_dict)
    res = {
        "field_id": field_id,
        "scenario": scenario,
        "anchor_index": N_HIST - 1,
        "engine": eng_info["engine"],
        "model_version": eng_info["model_version"],
        "is_proxy_ndvi_bare": True,
        "months": months,
    }
    if irrigation_cap_mm > 0.0 and "irrigation_mm" in sim:
        irrig_tot = float(sim["irrigation_mm"][N_HIST:].sum(axis=0).mean())
        extrap = bool(np.any(sim["extrapolated"][N_HIST:])) if "extrapolated" in sim else (irrig_tot > 0.0)
        res["irrigation_applied_mm"] = round(irrig_tot, 1)
        res["extrapolated"] = extrap
        res["extrapolation_note"] = (
            f"Supplemental irrigation ({irrig_tot:.1f} mm/season applied, cap {irrigation_cap_mm:.0f} mm) "
            "clipped to rainfed training range [0.02, 0.94]."
            if extrap
            else None
        )
    return res


def _crop_stage(plan: CropPlan, m: int) -> str:
    if plan.plant <= m <= plan.harvest:
        t = (m - plan.plant) / max(1, plan.harvest - plan.plant)
        return "planting" if m == plan.plant else "harvest" if m == plan.harvest else \
            "vegetative" if t < 0.45 else "reproductive" if t < 0.8 else "maturity"
    if plan.cc_cover is not None and (m - plan.harvest) % 12 <= 6:
        return "cover crop"
    return "fallow"


# --------------------------------------------------------------------------- #
# Rotation scenarios                                                          #
# --------------------------------------------------------------------------- #
def _risk_label(score: float) -> str:
    return "low" if score < 0.30 else "medium" if score < 0.55 else "high"


def _norm(v: np.ndarray, higher_better: bool = True) -> np.ndarray:
    rng = v.max(axis=0) - v.min(axis=0)
    n = np.where(rng < 1e-9, 0.5, (v - v.min(axis=0)) / np.where(rng < 1e-9, 1, rng))
    return n if higher_better else 1 - n


def preference_weights(prefs: dict) -> Dict[str, float]:
    w = {"water_saving": 1 / 3, "income": 1 / 3, "soil_restoration": 1 / 3}
    if isinstance(prefs.get("weights"), dict):
        raw_w = prefs["weights"]
        keys = list(w.keys()) + (["simplicity"] if "simplicity" in raw_w else [])
        w = {k: max(0.0, float(raw_w.get(k, 0))) for k in keys}
    elif prefs.get("priority") == "simplicity":
        w = {"water_saving": 0.4 / 3, "income": 0.4 / 3, "soil_restoration": 0.4 / 3, "simplicity": 0.6}
    elif prefs.get("priority") in w:
        w = {k: (0.6 if k == prefs["priority"] else 0.2) for k in w}
    tot = sum(w.values()) or 1
    return {k: v / tot for k, v in w.items()}


def compare_scenarios(grid: dict, climate: dict, prefs: dict,
                      constraints: Optional[dict] = None) -> dict:
    irrig_cap = float((constraints or {}).get("irrigation_cap_mm", 0.0))
    ra_lambda = float((constraints or {}).get("risk_aversion_lambda", 0.25))
    try:
        from ml.crop_plans import evaluate_and_rank_plans
        fid = grid.get("field_id", "default")
        zcoords = grid.get("zone_coords")
        sim_fn = lambda s, m, p: simulate(
            s, m, p, field_id=fid, zone_coords=zcoords, irrigation_cap_mm=irrig_cap
        )
        return evaluate_and_rank_plans(
            grid=grid, climate=climate, prefs=prefs,
            simulate_fn=sim_fn, preference_weights_fn=preference_weights,
            risk_aversion_lambda=ra_lambda, constraints=constraints,
        )
    except Exception as exc:
        logger.warning("[zone_engine] Crop plan ensemble evaluation failed (%s); using heuristic fallback.", exc)

    ids = list(PLANS)
    f = slice(N_HIST, N_TOTAL)
    per = []
    for pid in ids:
        plans = ["status_quo"] * N_HIST + [pid] * (N_TOTAL - N_HIST)
        sim = simulate(grid["soil"], climate["months"], plans)
        cal = CALENDARS[pid]; p = PLANS[pid]
        season = [i for i, c in enumerate(climate["months"][f]) if p.plant <= c["cal_month"] <= p.harvest]
        esi_season = sim["esi"][f][season].mean(axis=0)
        stress = 1 - np.clip(esi_season, 0, 1)
        et_year = sim["et"][f].sum(axis=0)
        runoff = sim["runoff"][f].sum(axis=0)
        bare = sim["bare"][f].mean(axis=0)
        per.append(dict(
            water_demand=sim["demand"][f].sum(axis=0),
            water_balance=sim["precip"][f].sum(axis=0) - et_year - runoff,
            et=et_year, bare=bare, stress=stress,
            n_fix=p.n_fix * (0.7 + 0.3 * np.clip(esi_season, 0, 1)),
            som=p.som_gain * (0.8 + 0.4 * sim["vigor"][f].mean(axis=0)) * (0.9 + 0.1 * grid["soil"]["som"] / 3),
            margin=p.margin * np.clip(1 - p.drought_sens * stress * 1.6, 0.35, 1.05),
            risk=np.clip(p.drought_sens * stress * 1.4 + 0.35 * bare, 0, 1),
            erosion_control=100 * (1 - bare),
        ))
    w = preference_weights(prefs)
    stack = lambda k: np.stack([m[k] for m in per])             # (3, zones)
    zone_score = (w["water_saving"] * _norm(stack("water_demand"), False)
                  + w["income"] * _norm(stack("margin"))
                  + w["soil_restoration"] * (0.5 * _norm(stack("som")) + 0.3 * _norm(stack("erosion_control"))
                                             + 0.2 * _norm(stack("n_fix"))))
    field_score = zone_score.mean(axis=1)
    best = int(np.argmax(field_score))

    scenarios = []
    for i, pid in enumerate(ids):
        m, p = per[i], PLANS[pid]
        mean = lambda k: float(m[k].mean())
        scenarios.append({
            "id": pid, "name": p.name, "crops": p.crops, "description": p.description,
            "score": round(float(field_score[i]), 3), "recommended": i == best,
            "metrics": {
                "soil_organic_matter_gain_pct": round(mean("som"), 2),
                "nitrogen_fixed_kg_ha": round(mean("n_fix"), 0),
                "water_demand_mm": round(mean("water_demand"), 0),
                "water_budget_balance_mm": round(mean("water_balance"), 0),
                "erosion_control_score": round(mean("erosion_control"), 0),
                "estimated_margin_usd_ha": round(mean("margin"), 0),
                "risk_level": _risk_label(mean("risk")),
                "risk_score": round(mean("risk"), 2),
            },
            "tradeoffs": _tradeoffs(pid),
        })
    zone_rec = [ids[j] for j in np.argmax(zone_score, axis=0)]
    return {"scenarios": scenarios, "recommended": ids[best], "zone_recommendations": zone_rec,
            "weights": {k: round(v, 2) for k, v in w.items()}}


def _tradeoffs(pid: str) -> Dict[str, List[str]]:
    return {
        "status_quo": {"pros": ["Familiar management", "Established markets"],
                       "cons": ["Bare winter soil", "Highest water demand", "Synthetic N required"]},
        "cover_legume": {"pros": ["Fixes nitrogen", "Best erosion control", "Builds organic matter"],
                         "cons": ["Cover seed cost", "Cover crop uses spring water"]},
        "drought_cash": {"pros": ["Lowest water demand", "Resilient in heat waves"],
                         "cons": ["Lower baseline margin", "Fewer local buyers"]},
    }[pid]


# --------------------------------------------------------------------------- #
# Guidance                                                                    #
# --------------------------------------------------------------------------- #
def _runs(idx: List[int]) -> List[List[int]]:
    runs: List[List[int]] = []
    for i in idx:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def build_guidance(timeline: dict, scenario: str) -> dict:
    months = timeline["months"]
    fc = months[N_HIST:]
    plan = PLANS[scenario]
    lab = lambda m: m["label"]

    cand = [m["index"] for m in fc if m["climate"]["tmax_c"] + m["climate"]["tmin_c"] >= 22
            and m["field"]["smi"] >= 0.40 and m["crop_stage"] in ("planting", "vegetative")]
    cand = cand or [m["index"] for m in fc if m["crop_stage"] == "planting"]
    seed = {"start": lab(months[cand[0]]), "end": lab(months[cand[min(len(cand) - 1, 1)]]),
            "message": f"Seed {plan.crops.split('→')[0].strip()} in {lab(months[cand[0]])}: soil is warm enough "
                       f"and moisture index is {months[cand[0]]['field']['smi']:.2f}."} if cand else None

    vul = [m["index"] for m in fc if m["field"]["bare"] >= 0.50]
    vulnerable = [{"start": lab(months[r[0]]), "end": lab(months[r[-1]]),
                   "message": f"Ground is exposed {lab(months[r[0]])} – {lab(months[r[-1]])} "
                              f"(mean erosion risk {np.mean([months[i]['field']['bare'] for i in r]):.2f}). "
                              f"Keep residue or cover on slopes."} for r in _runs(vul)]

    legume = PLANS["cover_legume"]
    cc = [m for m in fc if 0 <= (int(m["date"][5:]) - legume.harvest) % 12 <= 1
          and (m["climate"]["tmax_c"] + m["climate"]["tmin_c"]) / 2 >= 5]
    cover = [{"start": lab(c), "end": lab(c),
              "message": f"Drill clover/rye within 2–3 weeks of harvest — {lab(c)} still has "
                         f"{(c['climate']['tmax_c'] + c['climate']['tmin_c']) / 2:.0f} °C mean temperature for establishment."} for c in cc[:2]]

    stress = [{"start": lab(m), "end": lab(m), "type": a["type"], "message": a["message"]}
              for m in fc for a in m["alerts"] if a["type"] in ("water_stress", "heat_stress")]
    return {"seed_window": seed, "vulnerable_windows": vulnerable, "cover_crop_windows": cover,
            "stress_alerts": stress[:6]}

"""Feature engineering and target extraction for TerraSense Phase 1."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from dbfread import DBF
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform

from ml.config import (
    CULTIVATED_CROP_CODES,
    DATA_DIR,
    DEM_PATCH_RES_M,
    DEM_PATCH_SIZE,
    DEVELOPED_CODES,
    DRAINAGE_ORDINAL,
    FORAGE_CROP_CODES,
    FOREST_SHRUB_CODES,
    GDD_BASE_C,
    GDD_CAP_C,
    HEAT_STRESS_THRESH_C,
    METRIC_CRS,
    PASTURE_CODES,
    SOILGRIDS_D_FACTOR,
    WATER_WETLAND_CODES,
)


# --------------------------------------------------------------------------- #
# 1. CDL Cropland Filtering & Rotation Features                               #
# --------------------------------------------------------------------------- #
def load_cdl_class_names(tif_path: Path) -> Dict[int, str]:
    """Read integer code -> CLASS_NAME mapping from CDL .tif.vat.dbf."""
    dbf_path = Path(str(tif_path) + ".vat.dbf")
    if not dbf_path.exists():
        return {}
    mapping: Dict[int, str] = {}
    for rec in DBF(str(dbf_path)):
        val = rec.get("VALUE")
        name = (rec.get("CLASS_NAME") or "").strip()
        if val is not None and name:
            mapping[int(val)] = name
    return mapping


def _sample_cdl_window(
    src: rasterio.DatasetReader, x_proj: float, y_proj: float, win_half: int = 1
) -> Tuple[int, int, float]:
    """Return (majority_code_3x3, center_code, cropland_fraction_3x3).

    Returns (-1, -1, 0.0) if the point falls outside the raster bounds.
    """
    r, c = src.index(x_proj, y_proj)
    if not (0 <= r < src.height and 0 <= c < src.width):
        return -1, -1, 0.0
    r0, r1 = max(0, r - win_half), min(src.height, r + win_half + 1)
    c0, c1 = max(0, c - win_half), min(src.width, c + win_half + 1)
    patch = src.read(1, window=((r0, r1), (c0, c1)))
    if patch.size == 0:
        return -1, -1, 0.0
    center_val = int(src.read(1, window=((r, r + 1), (c, c + 1)))[0, 0])
    vals, counts = np.unique(patch, return_counts=True)
    max_cnt = counts.max()
    candidates = vals[counts == max_cnt]
    # Tie-break in favor of center pixel
    maj_val = center_val if center_val in candidates else int(candidates[0])
    crop_mask = np.isin(patch, list(CULTIVATED_CROP_CODES))
    return maj_val, center_val, float(crop_mask.mean())


def classify_drop_reason(code_2023: int, code_2024: int) -> Tuple[bool, str]:
    """Decide whether a point is retained as cultivated cropland for training.

    Policy (explicit & documented):
      - Out-of-bounds (-1) or background (0): dropped ("out_of_cdl_bounds")
      - Keep if EITHER 2023 or 2024 3x3 majority is in CULTIVATED_CROP_CODES
        AND neither year is developed (121-124) or open water/wetland (111, 190, 195).
      - Pure Grassland/Pasture (176 in both years): dropped ("permanent_pasture_176")
        because uncultivated CRP/pasture/roadside grass has different management
        and rooting dynamics than annual/forage cropland, while Alfalfa (36) and
        Other Hay (37) are kept.
    """
    if code_2023 < 0 or code_2024 < 0:
        return False, "out_of_cdl_bounds"
    codes = {code_2023, code_2024}
    if codes & WATER_WETLAND_CODES:
        return False, "water_or_wetland"
    if codes & DEVELOPED_CODES:
        return False, "developed_urban"
    if (code_2023 in CULTIVATED_CROP_CODES) or (code_2024 in CULTIVATED_CROP_CODES):
        return True, "kept_cropland"
    if codes <= PASTURE_CODES:
        return False, "permanent_pasture_176"
    if codes & FOREST_SHRUB_CODES:
        return False, "forest_or_shrubland"
    return False, f"non_cropland_{code_2023}_{code_2024}"


def extract_cdl_features(points: List[Dict[str, Any]], cdl_dir: Path = DATA_DIR / "cdl") -> Dict[str, Dict[str, Any]]:
    """Sample 2023 and 2024 CDL rasters (3x3 window in EPSG:5070) for all points."""
    tif_2023 = next(cdl_dir.glob("CDL_2023_*.tif"))
    tif_2024 = next(cdl_dir.glob("CDL_2024_*.tif"))
    names_2023 = load_cdl_class_names(tif_2023)
    names_2024 = load_cdl_class_names(tif_2024)
    names = {**names_2023, **names_2024}

    lons = [float(p["Longitude"]) for p in points]
    lats = [float(p["Latitude"]) for p in points]

    out: Dict[str, Dict[str, Any]] = {}
    with rasterio.open(tif_2023) as s23, rasterio.open(tif_2024) as s24:
        xs23, ys23 = transform("EPSG:4326", s23.crs, lons, lats)
        xs24, ys24 = transform("EPSG:4326", s24.crs, lons, lats)
        for pt, x23, y23, x24, y24 in zip(points, xs23, ys23, xs24, ys24):
            pid = pt["ID"]
            maj23, cen23, frac23 = _sample_cdl_window(s23, x23, y23, win_half=1)
            maj24, cen24, frac24 = _sample_cdl_window(s24, x24, y24, win_half=1)
            keep, reason = classify_drop_reason(maj23, maj24)
            corn_soy_rot = int({maj23, maj24} == {1, 5})
            cont_corn = int(maj23 == 1 and maj24 == 1)
            out[pid] = {
                "crop_2023": maj23,
                "crop_2023_center": cen23,
                "crop_2023_name": names.get(maj23, "Out of Bounds" if maj23 < 0 else f"Code {maj23}"),
                "cropland_frac_2023": round(frac23, 4),
                "crop_2024": maj24,
                "crop_2024_center": cen24,
                "crop_2024_name": names.get(maj24, "Out of Bounds" if maj24 < 0 else f"Code {maj24}"),
                "cropland_frac_2024": round(frac24, 4),
                "corn_soy_rotation": corn_soy_rot,
                "continuous_corn": cont_corn,
                "is_pasture": int(maj23 in PASTURE_CODES or maj24 in PASTURE_CODES),
                "keep_for_training": keep,
                "drop_reason": reason,
            }
    return out


# --------------------------------------------------------------------------- #
# 2. Metric Terrain Extraction (EPSG:5070 Reprojection -> Slope/Aspect/TWI)   #
# --------------------------------------------------------------------------- #
def _d8_upslope_area_m(dem_patch: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Compute local upslope contributing area per unit contour width (meters) via D8."""
    h, w = dem_patch.shape
    order = np.argsort(dem_patch, axis=None)[::-1]  # highest to lowest elevation
    flow_cells = np.ones((h, w), dtype=np.float64)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dists = [math.hypot(dr, dc) * cell_size_m for dr, dc in offsets]

    for flat_idx in order:
        r, c = divmod(int(flat_idx), w)
        z0 = dem_patch[r, c]
        best_drop = 0.0
        best_rc = None
        for (dr, dc), dist in zip(offsets, dists):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                drop = (z0 - dem_patch[nr, nc]) / dist
                if drop > best_drop:
                    best_drop = drop
                    best_rc = (nr, nc)
        if best_rc is not None:
            flow_cells[best_rc] += flow_cells[r, c]
    return flow_cells * cell_size_m


def compute_metric_terrain_stats(patch_m: np.ndarray, res_m: float = DEM_PATCH_RES_M) -> Dict[str, float]:
    """Compute slope, aspect (sin/cos), curvature, local relief, and TWI on a metric patch."""
    mid = patch_m.shape[0] // 2
    # Horn / central-difference gradients in meters (dz/dx east, dz/dy north)
    # Row index increases southward in standard top-down raster transform
    dz_dy_south, dz_dx_east = np.gradient(patch_m, res_m, res_m)
    dz_dy_north = -dz_dy_south

    slope_rad = np.arctan(np.hypot(dz_dx_east, dz_dy_north))
    slope_pct = np.tan(slope_rad) * 100.0

    # Aspect: clockwise from north (0 = N, pi/2 = E)
    aspect_rad = np.arctan2(-dz_dx_east, -dz_dy_north)

    # Laplacian curvature (d2z/dx2 + d2z/dy2) scaled by 100 (1/100m)
    d2z_dy2, _ = np.gradient(dz_dy_south, res_m, res_m)
    _, d2z_dx2 = np.gradient(dz_dx_east, res_m, res_m)
    curv = (d2z_dx2 + d2z_dy2) * 100.0

    # Topographic Wetness Index: ln(a / tan(beta))
    upslope_a = _d8_upslope_area_m(patch_m, res_m)
    tan_beta = np.maximum(np.tan(slope_rad), 0.001)
    twi_grid = np.log(upslope_a / tan_beta)

    w3 = slice(mid - 1, mid + 2)   # 3x3 = 90m x 90m
    w7 = slice(mid - 3, mid + 4)   # 7x7 = 210m x 210m
    return {
        "elev_m": round(float(patch_m[mid, mid]), 2),
        "elev_mean_90m": round(float(patch_m[w3, w3].mean()), 2),
        "slope_pct": round(float(slope_pct[mid, mid]), 3),
        "slope_mean_90m": round(float(slope_pct[w3, w3].mean()), 3),
        "slope_std_90m": round(float(slope_pct[w3, w3].std()), 3),
        "aspect_sin": round(float(np.sin(aspect_rad[mid, mid])), 4),
        "aspect_cos": round(float(np.cos(aspect_rad[mid, mid])), 4),
        "curvature": round(float(curv[mid, mid]), 4),
        "local_relief_90m": round(float(patch_m[w3, w3].max() - patch_m[w3, w3].min()), 2),
        "local_relief_210m": round(float(patch_m[w7, w7].max() - patch_m[w7, w7].min()), 2),
        "twi": round(float(twi_grid[mid, mid]), 3),
    }


def extract_terrain_for_coords(
    coords_lon_lat: List[Tuple[float, float]], dem_dir: Path = DATA_DIR / "earthdata"
) -> List[Dict[str, float]]:
    """Reproject local DEM patches into EPSG:5070 (30 m metric grid) and compute terrain stats."""
    dem_files = sorted(dem_dir.glob("*.tif"))
    datasets = [rasterio.open(p) for p in dem_files]
    try:
        lons = [c[0] for c in coords_lon_lat]
        lats = [c[1] for c in coords_lon_lat]
        xs, ys = transform("EPSG:4326", METRIC_CRS, lons, lats)
        half = (DEM_PATCH_SIZE * DEM_PATCH_RES_M) / 2.0

        results: List[Dict[str, float]] = []
        for lon, lat, x0, y0 in zip(lons, lats, xs, ys):
            dst_transform = from_origin(x0 - half, y0 + half, DEM_PATCH_RES_M, DEM_PATCH_RES_M)
            patch = np.full((DEM_PATCH_SIZE, DEM_PATCH_SIZE), np.nan, dtype=np.float32)

            # Find overlapping DEM tiles (+/- 0.02 deg buffer)
            for ds in datasets:
                b = ds.bounds
                if (b.left - 0.02 <= lon <= b.right + 0.02) and (b.bottom - 0.02 <= lat <= b.top + 0.02):
                    temp = np.full_like(patch, np.nan)
                    reproject(
                        source=rasterio.band(ds, 1),
                        destination=temp,
                        src_transform=ds.transform,
                        src_crs=ds.crs,
                        dst_transform=dst_transform,
                        dst_crs=METRIC_CRS,
                        resampling=Resampling.bilinear,
                        dst_nodata=np.nan,
                    )
                    mask = np.isnan(patch) & ~np.isnan(temp)
                    patch[mask] = temp[mask]

            if np.isnan(patch).any():
                fill_val = float(np.nanmean(patch)) if not np.isnan(patch).all() else 300.0
                patch = np.nan_to_num(patch, nan=fill_val)
            results.append(compute_metric_terrain_stats(patch, DEM_PATCH_RES_M))
        return results
    finally:
        for ds in datasets:
            ds.close()


# --------------------------------------------------------------------------- #
# 3. Soil Aggregation (Primary: SSURGO 0-30 & 30-100 cm; Secondary: SoilGrids)#
# --------------------------------------------------------------------------- #
def _parse_float(v: Any) -> Optional[float]:
    if v in (None, "", "null"):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def parse_ssurgo_file(fp: Path) -> Dict[str, float]:
    """Aggregate SSURGO components (by comppct_r) and horizons into 0-30 cm & 30-100 cm layers."""
    nan_result = {
        "sand_top_pct": np.nan, "clay_top_pct": np.nan, "silt_top_pct": np.nan,
        "om_top_pct": np.nan, "ph_top": np.nan, "bd_top_gcm3": np.nan, "awc_top_cmcm": np.nan,
        "sand_sub_pct": np.nan, "clay_sub_pct": np.nan, "awc_sub_cmcm": np.nan,
        "awc_100cm_mm": np.nan, "drainage_code": np.nan,
    }
    if not fp.exists():
        return nan_result
    obj = json.loads(fp.read_text())
    tbl = obj.get("Table", [])
    if len(tbl) <= 1:
        return nan_result

    header = tbl[0]
    idx = {h: i for i, h in enumerate(header)}
    props = ["sandtotal_r", "claytotal_r", "silttotal_r", "om_r", "ph1to1h2o_r", "dbthirdbar_r", "awc_r"]

    # Group horizons by component (cokey), filtering non-soil components
    comps: Dict[str, Dict[str, Any]] = {}
    for row in tbl[1:]:
        compname = (row[idx["compname"]] or "").lower()
        if any(w in compname for w in ("urban land", "water", "pits", "dumps")):
            continue
        cokey = str(row[idx["cokey"]])
        comppct = _parse_float(row[idx["comppct_r"]]) or 1.0
        drain_str = row[idx["drainagecl"]] or ""
        c = comps.setdefault(cokey, {"comppct": comppct, "drainage": DRAINAGE_ORDINAL.get(drain_str, np.nan), "horizons": []})
        top = _parse_float(row[idx["hzdept_r"]])
        bot = _parse_float(row[idx["hzdepb_r"]])
        if top is None or bot is None or bot <= top:
            continue
        c["horizons"].append({
            "top": top, "bot": bot,
            **{p: _parse_float(row[idx[p]]) for p in props},
        })

    if not comps:
        return nan_result

    def _layer_avg(horizons: List[dict], z0: float, z1: float, prop: str) -> float:
        num, den = 0.0, 0.0
        for h in horizons:
            ov = max(0.0, min(h["bot"], z1) - max(h["top"], z0))
            val = h.get(prop)
            if ov > 0 and val is not None:
                num += ov * val
                den += ov
        return num / den if den > 0 else np.nan

    def _comp_weighted(getter) -> float:
        num, den = 0.0, 0.0
        for c in comps.values():
            v = getter(c)
            if v is not None and not np.isnan(v):
                num += c["comppct"] * v
                den += c["comppct"]
        return num / den if den > 0 else np.nan

    top_sand = _comp_weighted(lambda c: _layer_avg(c["horizons"], 0.0, 30.0, "sandtotal_r"))
    top_clay = _comp_weighted(lambda c: _layer_avg(c["horizons"], 0.0, 30.0, "claytotal_r"))
    top_silt = _comp_weighted(lambda c: _layer_avg(c["horizons"], 0.0, 30.0, "silttotal_r"))
    top_om = _comp_weighted(lambda c: _layer_avg(c["horizons"], 0.0, 30.0, "om_r"))
    top_ph = _comp_weighted(lambda c: _layer_avg(c["horizons"], 0.0, 30.0, "ph1to1h2o_r"))
    top_bd = _comp_weighted(lambda c: _layer_avg(c["horizons"], 0.0, 30.0, "dbthirdbar_r"))
    top_awc = _comp_weighted(lambda c: _layer_avg(c["horizons"], 0.0, 30.0, "awc_r"))

    sub_sand = _comp_weighted(lambda c: _layer_avg(c["horizons"], 30.0, 100.0, "sandtotal_r"))
    sub_clay = _comp_weighted(lambda c: _layer_avg(c["horizons"], 30.0, 100.0, "claytotal_r"))
    sub_awc = _comp_weighted(lambda c: _layer_avg(c["horizons"], 30.0, 100.0, "awc_r"))
    drain_code = _comp_weighted(lambda c: c["drainage"])

    # Root-zone 0-100cm available water capacity in mm (100 cm = 1000 mm)
    eff_top_awc = top_awc if not np.isnan(top_awc) else sub_awc
    eff_sub_awc = sub_awc if not np.isnan(sub_awc) else top_awc
    awc_100cm_mm = (300.0 * eff_top_awc + 700.0 * eff_sub_awc) if not np.isnan(eff_top_awc) else np.nan

    return {
        "sand_top_pct": round(float(top_sand), 2) if not np.isnan(top_sand) else np.nan,
        "clay_top_pct": round(float(top_clay), 2) if not np.isnan(top_clay) else np.nan,
        "silt_top_pct": round(float(top_silt), 2) if not np.isnan(top_silt) else np.nan,
        "om_top_pct": round(float(top_om), 2) if not np.isnan(top_om) else np.nan,
        "ph_top": round(float(top_ph), 2) if not np.isnan(top_ph) else np.nan,
        "bd_top_gcm3": round(float(top_bd), 3) if not np.isnan(top_bd) else np.nan,
        "awc_top_cmcm": round(float(top_awc), 4) if not np.isnan(top_awc) else np.nan,
        "sand_sub_pct": round(float(sub_sand), 2) if not np.isnan(sub_sand) else np.nan,
        "clay_sub_pct": round(float(sub_clay), 2) if not np.isnan(sub_clay) else np.nan,
        "awc_sub_cmcm": round(float(sub_awc), 4) if not np.isnan(sub_awc) else np.nan,
        "awc_100cm_mm": round(float(awc_100cm_mm), 1) if not np.isnan(awc_100cm_mm) else np.nan,
        "drainage_code": round(float(drain_code), 2) if not np.isnan(drain_code) else np.nan,
    }


def parse_soilgrids_file(fp: Path) -> Dict[str, float]:
    """Extract 0-30 cm depth-weighted SoilGrids features divided by d_factor."""
    nan_res = {
        "sg_soc_gkg": np.nan, "sg_cec_cmol": np.nan, "sg_bdod_gcm3": np.nan,
        "sg_clay_pct": np.nan, "sg_sand_pct": np.nan, "sg_silt_pct": np.nan, "sg_ph": np.nan,
    }
    if not fp.exists():
        return nan_res
    obj = json.loads(fp.read_text())
    layers = obj.get("properties", {}).get("layers", [])
    weights = {"0-5cm": 5.0, "5-15cm": 10.0, "15-30cm": 15.0}
    raw_means: Dict[str, float] = {}
    for lyr in layers:
        name = lyr.get("name", "")
        dfac = SOILGRIDS_D_FACTOR.get(name, 1.0)
        num, den = 0.0, 0.0
        for d in lyr.get("depths", []):
            w = weights.get(d.get("label", ""), 0.0)
            v = (d.get("values") or {}).get("mean")
            if w > 0 and v is not None:
                num += w * (float(v) / dfac)
                den += w
        raw_means[name] = round(num / den, 3) if den > 0 else np.nan

    return {
        "sg_soc_gkg": raw_means.get("soc", np.nan),
        "sg_cec_cmol": raw_means.get("cec", np.nan),
        "sg_bdod_gcm3": raw_means.get("bdod", np.nan),
        "sg_clay_pct": raw_means.get("clay", np.nan),
        "sg_sand_pct": raw_means.get("sand", np.nan),
        "sg_silt_pct": raw_means.get("silt", np.nan),
        "sg_ph": raw_means.get("phh2o", np.nan),
    }


def fuse_soil_features(ssurgo: Dict[str, float], sg: Dict[str, float]) -> Dict[str, float]:
    """Use SSURGO as primary; fall back to SoilGrids only when SSURGO is NaN."""
    out = {**ssurgo, **sg}
    if np.isnan(out["sand_top_pct"]) and not np.isnan(sg["sg_sand_pct"]):
        out["sand_top_pct"] = sg["sg_sand_pct"]
        out["sand_sub_pct"] = sg["sg_sand_pct"]
    if np.isnan(out["clay_top_pct"]) and not np.isnan(sg["sg_clay_pct"]):
        out["clay_top_pct"] = sg["sg_clay_pct"]
        out["clay_sub_pct"] = sg["sg_clay_pct"]
    if np.isnan(out["silt_top_pct"]) and not np.isnan(sg["sg_silt_pct"]):
        out["silt_top_pct"] = sg["sg_silt_pct"]
    if np.isnan(out["om_top_pct"]) and not np.isnan(sg["sg_soc_gkg"]):
        # Van Bemmelen factor: SOM (%) ~ 0.1724 * SOC (g/kg)
        out["om_top_pct"] = round(0.1724 * sg["sg_soc_gkg"], 2)
    if np.isnan(out["ph_top"]) and not np.isnan(sg["sg_ph"]):
        out["ph_top"] = sg["sg_ph"]
    if np.isnan(out["bd_top_gcm3"]) and not np.isnan(sg["sg_bdod_gcm3"]):
        out["bd_top_gcm3"] = sg["sg_bdod_gcm3"]
    if np.isnan(out["awc_top_cmcm"]) and not np.isnan(out["sand_top_pct"]):
        est_awc = max(0.08, 0.26 - 0.0016 * out["sand_top_pct"])
        out["awc_top_cmcm"] = round(est_awc, 4)
        out["awc_sub_cmcm"] = round(est_awc, 4)
        out["awc_100cm_mm"] = round(1000.0 * est_awc, 1)
    return out


# --------------------------------------------------------------------------- #
# 4. Spatial Cell IDs, Invertible SMI Transform & Monthly Weather/Targets     #
# --------------------------------------------------------------------------- #
def spatial_cell_ids(lat: float, lon: float) -> Tuple[str, str, str]:
    """Return (merra2_cell_id, era5_cell_id, spatial_cluster_25km_id)."""
    # MERRA-2 grid: 0.5 deg lat x 0.625 deg lon
    m_lat = round(lat / 0.5) * 0.5
    m_lon = round(lon / 0.625) * 0.625
    # ERA5-Land grid: 0.1 deg x 0.1 deg
    e_lat = round(lat, 1)
    e_lon = round(lon, 1)
    # ~30 km spatial block cluster for buffered CV (0.30 deg lat x 0.40 deg lon)
    c_lat = math.floor((lat - 41.0) / 0.35)
    c_lon = math.floor((lon + 95.0) / 0.45)
    return f"merra_{m_lat:.2f}_{m_lon:.3f}", f"era5_{e_lat:.1f}_{e_lon:.1f}", f"blk_{c_lat}_{c_lon}"


def soil_water_bounds(clay_pct: float, sand_pct: float, awc_cmcm: float) -> Tuple[float, float]:
    """Compute volumetric wilting point theta_wp and field capacity theta_fc (m3/m3).

    Invertible linear mapping:
        SMI = clip((theta - theta_wp) / (theta_fc - theta_wp), 0, 1)
        theta = theta_wp + SMI * (theta_fc - theta_wp)
    """
    clay = 28.0 if np.isnan(clay_pct) else float(clay_pct)
    awc = 0.18 if np.isnan(awc_cmcm) else float(awc_cmcm)
    # Saxton-Rawls empirical wilting point baseline from clay %, plus SSURGO AWC span
    theta_wp = float(np.clip(0.04 + 0.0045 * clay, 0.06, 0.24))
    theta_fc = float(np.clip(theta_wp + max(0.10, awc), theta_wp + 0.10, 0.46))
    return theta_wp, theta_fc


def volumetric_sm_to_smi(theta: float, theta_wp: float, theta_fc: float) -> float:
    return float(np.clip((theta - theta_wp) / max(1e-6, theta_fc - theta_wp), 0.0, 1.0))


def smi_to_volumetric_sm(smi: float, theta_wp: float, theta_fc: float) -> float:
    return float(theta_wp + np.clip(smi, 0.0, 1.0) * (theta_fc - theta_wp))


def compute_proxy_indices(
    smi: float,
    esi: float,
    cal_cover: float,
    slope_pct: float,
    sand_pct: float,
    precip_mm: float,
    max_daily_precip_mm: float,
) -> Dict[str, float]:
    """Documented deterministic proxy indices for bare-soil risk, NDVI, and vigor.

    Because no satellite NDVI or bare-soil observations are present in data/,
    we never train an ML model on invented NDVI/bare labels; instead we compute
    these indices deterministically from predicted SMI, ESI, crop calendar cover,
    and static terrain/soil properties.
    """
    slope_n = float(np.clip(slope_pct / 8.0, 0.0, 1.0))
    sand_frac = float(np.clip(sand_pct / 100.0, 0.0, 1.0))
    rain_i = min(precip_mm / 120.0, 1.0)
    bare = float(np.clip(
        (1.0 - cal_cover) * 0.42
        + slope_n * 0.24
        + sand_frac * 0.30
        + rain_i * 0.10 * (1.0 - cal_cover)
        + (1.0 - smi) * 0.10
        - 0.03,
        0.0, 1.0,
    ))
    ndvi = float(np.clip(0.12 + 0.72 * cal_cover * (0.45 + 0.55 * esi), 0.05, 0.90))
    vigor = float(np.clip(
        (0.5 * (ndvi - 0.1) / 0.7 + 0.3 * esi + 0.2 * smi) * (0.4 + 0.6 * cal_cover),
        0.0, 1.0,
    ))
    return {"bare": round(bare, 4), "ndvi": round(ndvi, 4), "vigor": round(vigor, 4)}


def load_appeears_ndvi_hook(appeears_dir: Path = DATA_DIR / "appeears") -> Dict[Tuple[str, str], float]:
    """Clean hook to load real AppEEARS MODIS NDVI point-month targets if added later.

    Returns dict keyed by (point_id, 'YYYY-MM') -> observed monthly mean NDVI in [0, 1].
    Returns empty dict if `data/appeears/` has no CSV files.
    """
    csv_files = sorted(appeears_dir.glob("*.csv"))
    if not csv_files:
        return {}
    import csv as _csv
    buckets: Dict[Tuple[str, str], List[float]] = {}
    for fp in csv_files:
        with fp.open("r", newline="") as f:
            for row in _csv.DictReader(f):
                pid = row.get("ID") or row.get("id")
                dt = (row.get("Date") or row.get("date") or "")[:7]
                for k, v in row.items():
                    if k and "NDVI" in k.upper() and v not in (None, "", "NA"):
                        try:
                            val = float(v)
                            if val > 1.5:
                                val *= 0.0001  # MODIS 10000 scale factor
                            if -0.1 <= val <= 1.0 and pid and len(dt) == 7:
                                buckets.setdefault((pid, dt), []).append(float(np.clip(val, 0.05, 0.95)))
                        except ValueError:
                            pass
    return {k: float(np.mean(vs)) for k, vs in buckets.items()}


def aggregate_point_monthly_series(
    power_daily: List[Dict[str, Any]],
    target_daily: List[Dict[str, Any]],
    om_daily: Optional[List[Dict[str, Any]]],
    theta_wp: float,
    theta_fc: float,
    heat_thresh_c: float = HEAT_STRESS_THRESH_C,
) -> List[Dict[str, Any]]:
    """Aggregate daily weather + target feeds into physics-informed monthly records."""
    tgt_by_date = {r["date"]: r for r in target_daily}
    om_by_date = {r["date"]: r for r in om_daily} if om_daily else {}

    by_month: Dict[str, List[Dict[str, Any]]] = {}
    for r in power_daily:
        ym = r["date"][:7]
        by_month.setdefault(ym, []).append(r)

    months_out: List[Dict[str, Any]] = []
    prev_precip = 50.0
    for ym in sorted(by_month):
        rows = by_month[ym]
        n_days = len(rows)
        precip_arr = np.array([float(r["precip"]) for r in rows])
        tmax_arr = np.array([float(r["tmax"]) for r in rows])
        tmin_arr = np.array([float(r["tmin"]) for r in rows])
        tmean_arr = 0.5 * (tmax_arr + tmin_arr)
        rh_arr = np.array([float(r["rh"]) for r in rows])
        solar_arr = np.array([float(r["solar"]) for r in rows])

        precip_mm = float(precip_arr.sum())
        tmax_c = float(tmax_arr.mean())
        tmin_c = float(tmin_arr.mean())
        tmean_c = float(tmean_arr.mean())
        diurnal_c = float((tmax_arr - tmin_arr).mean())
        rh_pct = float(rh_arr.mean())
        solar_mj = float(solar_arr.mean())

        # GDD (base 10 C, cap 30 C)
        gdd_daily = np.maximum(0.0, 0.5 * (np.clip(tmax_arr, GDD_BASE_C, GDD_CAP_C) + np.maximum(tmin_arr, GDD_BASE_C)) - GDD_BASE_C)
        gdd_month = float(gdd_daily.sum())
        heat_days = int(np.sum(tmax_arr > heat_thresh_c))

        # Max consecutive dry days (precip < 1.0 mm)
        max_dry, cur_dry = 0, 0
        for p in precip_arr:
            if p < 1.0:
                cur_dry += 1
                max_dry = max(max_dry, cur_dry)
            else:
                cur_dry = 0

        # Max 5-day rolling precipitation sum
        if n_days >= 5:
            p5 = float(np.convolve(precip_arr, np.ones(5), mode="valid").max())
        else:
            p5 = precip_mm
        max_daily_p = float(precip_arr.max())

        # Vapor pressure deficit proxy (Tetens equation, kPa)
        es_tmax = 0.6108 * np.exp((17.27 * tmax_arr) / (tmax_arr + 237.3))
        es_tmin = 0.6108 * np.exp((17.27 * tmin_arr) / (tmin_arr + 237.3))
        es = 0.5 * (es_tmax + es_tmin)
        vpd_kpa = float(np.maximum(0.0, es * (1.0 - rh_arr / 100.0)).mean())

        # Reference PET (Hargreaves-Samani matching zone_engine.py, mm/month)
        pet_mm = max(0.2, 0.0045 * (tmean_c + 17.8) * solar_mj) * n_days
        water_bal_mm = precip_mm - pet_mm
        aridity = precip_mm / max(pet_mm, 1.0)
        api_mm = 0.7 * precip_mm + 0.3 * prev_precip
        prev_precip = precip_mm

        cal_month = int(ym[5:7])
        m_sin = math.sin(2.0 * math.pi * cal_month / 12.0)
        m_cos = math.cos(2.0 * math.pi * cal_month / 12.0)

        # Targets from MERRA-2 (nasa_power_targets)
        gwettop_vals, gwetroot_vals, evland_vals = [], [], []
        for r in rows:
            tr = tgt_by_date.get(r["date"])
            if tr:
                if tr.get("gwettop") is not None:
                    gwettop_vals.append(float(tr["gwettop"]))
                if tr.get("gwetroot") is not None:
                    gwetroot_vals.append(float(tr["gwetroot"]))
                if tr.get("evland") is not None:
                    # Clip slight negative winter condensation values at 0.0
                    evland_vals.append(max(0.0, float(tr["evland"])))

        gwetroot_m = float(np.mean(gwetroot_vals)) if gwetroot_vals else np.nan
        gwettop_m = float(np.mean(gwettop_vals)) if gwettop_vals else np.nan
        evland_mm = float(np.sum(evland_vals)) if evland_vals else np.nan

        # Targets from ERA5-Land (open_meteo, if available for this point)
        sm_root_vals, et0_om_vals, st_vals = [], [], []
        for r in rows:
            omr = om_by_date.get(r["date"])
            if omr:
                s0, s1, s2 = omr.get("sm_0_7cm"), omr.get("sm_7_28cm"), omr.get("sm_28_100cm")
                if s0 is not None and s1 is not None and s2 is not None:
                    # Depth-weighted 0-100cm volumetric soil moisture (m3/m3)
                    sm_root_vals.append(0.07 * float(s0) + 0.21 * float(s1) + 0.72 * float(s2))
                if omr.get("et0_mm") is not None:
                    et0_om_vals.append(max(0.0, float(omr["et0_mm"])))
                if omr.get("soil_temp_0_7cm") is not None:
                    st_vals.append(float(omr["soil_temp_0_7cm"]))

        if sm_root_vals:
            theta_root = float(np.mean(sm_root_vals))
            smi_era5 = volumetric_sm_to_smi(theta_root, theta_wp, theta_fc)
        else:
            theta_root = np.nan
            smi_era5 = np.nan

        et0_om_mm = float(np.sum(et0_om_vals)) if et0_om_vals else np.nan
        ref_pet = et0_om_mm if not np.isnan(et0_om_mm) else pet_mm
        esi_target = float(np.clip(evland_mm / max(ref_pet, 5.0), 0.05, 1.0)) if not np.isnan(evland_mm) else np.nan

        months_out.append({
            "year_month": ym,
            "year": int(ym[:4]),
            "cal_month": cal_month,
            "days": n_days,
            "precip_mm": round(precip_mm, 2),
            "tmax_c": round(tmax_c, 2),
            "tmin_c": round(tmin_c, 2),
            "tmean_c": round(tmean_c, 2),
            "diurnal_range_c": round(diurnal_c, 2),
            "rh_pct": round(rh_pct, 2),
            "solar_mj": round(solar_mj, 2),
            "gdd_month": round(gdd_month, 1),
            "heat_stress_days": heat_days,
            "max_dry_spell_days": max_dry,
            "max_5day_precip_mm": round(p5, 2),
            "max_daily_precip_mm": round(max_daily_p, 2),
            "vpd_kpa": round(vpd_kpa, 3),
            "pet_mm": round(pet_mm, 2),
            "et0_fao_mm": round(et0_om_mm, 2) if not np.isnan(et0_om_mm) else np.nan,
            "water_balance_mm": round(water_bal_mm, 2),
            "aridity_index": round(aridity, 3),
            "api_antecedent_mm": round(api_mm, 2),
            "month_sin": round(m_sin, 4),
            "month_cos": round(m_cos, 4),
            "theta_root_era5": round(theta_root, 4) if not np.isnan(theta_root) else np.nan,
            "target_smi_era5": round(smi_era5, 4) if not np.isnan(smi_era5) else np.nan,
            "target_smi_merra2": round(gwetroot_m, 4) if not np.isnan(gwetroot_m) else np.nan,
            "target_gwettop": round(gwettop_m, 4) if not np.isnan(gwettop_m) else np.nan,
            "target_evland_mm": round(evland_mm, 2) if not np.isnan(evland_mm) else np.nan,
            "target_esi": round(esi_target, 4) if not np.isnan(esi_target) else np.nan,
        })
    return months_out

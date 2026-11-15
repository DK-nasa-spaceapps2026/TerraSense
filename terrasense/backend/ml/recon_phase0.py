"""Phase 0 Data Quality & Reconnaissance Report for TerraSense.

Inspects all datasets in terrasense/backend/data/ and prints exact counts,
date coverage, NaN/null rates, value ranges/units, missing-point analysis,
and effective spatial cell counts (MERRA-2 vs ERA5-Land).
"""
from __future__ import annotations

import csv
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
POINTS_CSV = BASE_DIR / "appeears_points.csv"


def load_points():
    with POINTS_CSV.open("r", newline="") as f:
        return list(csv.DictReader(f))


def inspect_time_series(folder: str, keys: list[str], all_ids: list[str]):
    dpath = DATA_DIR / folder
    files = sorted(dpath.glob("*.json"))
    present_ids = {p.stem for p in files}
    missing_ids = [i for i in all_ids if i not in present_ids]

    n_days_Counter = Counter()
    min_date, max_date = "9999-99-99", "0000-00-00"
    null_counts = {k: 0 for k in keys}
    total_rows = 0
    vals = {k: [] for k in keys}
    unique_trajectories = set()

    for fp in files:
        obj = json.loads(fp.read_text())
        daily = obj.get("daily", [])
        n_days_Counter[len(daily)] += 1
        if daily:
            min_date = min(min_date, daily[0]["date"])
            max_date = max(max_date, daily[-1]["date"])
        sig = []
        for r in daily:
            total_rows += 1
            for k in keys:
                v = r.get(k)
                if v is None or (isinstance(v, (int, float)) and v <= -900):
                    null_counts[k] += 1
                else:
                    vals[k].append(float(v))
            if len(sig) < 10:
                sig.append(tuple(round(r.get(k) or -999, 4) for k in keys))
        unique_trajectories.add(tuple(sig))

    print(f"=== {folder} ===")
    print(f"  Files present: {len(files)} / {len(all_ids)} (missing: {len(missing_ids)})")
    if missing_ids:
        print(f"  First/last missing IDs: {missing_ids[:5]} ... {missing_ids[-5:]}")
    print(f"  Days per file distribution: {dict(n_days_Counter)}")
    print(f"  Date range: {min_date} .. {max_date} (total point-days: {total_rows})")
    print(f"  Distinct time-series cells (unique trajectories across {len(files)} points): {len(unique_trajectories)}")
    for k in keys:
        arr = np.array(vals[k]) if vals[k] else np.array([np.nan])
        print(
            f"  {k:16s}: nulls={null_counts[k]:6d} ({100*null_counts[k]/max(1,total_rows):.2f}%), "
            f"min={np.min(arr):.4f}, p50={np.median(arr):.4f}, max={np.max(arr):.4f}, mean={np.mean(arr):.4f}"
        )


def inspect_ssurgo(all_ids: list[str]):
    dpath = DATA_DIR / "ssurgo"
    files = sorted(dpath.glob("*.json"))
    empty_table = []
    water_or_urban = []
    drainage_counts = Counter()
    col_nulls = Counter()
    total_horizons = 0
    numeric_cols = [
        "comppct_r", "hzdept_r", "hzdepb_r", "awc_r",
        "claytotal_r", "sandtotal_r", "silttotal_r", "om_r", "ph1to1h2o_r", "dbthirdbar_r",
    ]
    col_vals = {c: [] for c in numeric_cols}

    for fp in files:
        obj = json.loads(fp.read_text())
        tbl = obj.get("Table", [])
        if len(tbl) <= 1:
            empty_table.append(fp.stem)
            continue
        header = tbl[0]
        idx = {h: i for i, h in enumerate(header)}
        muname = tbl[1][idx["muname"]] or ""
        if any(w in muname.lower() for w in ("water", "urban", "pits", "dumps", "orick")):
            water_or_urban.append((fp.stem, muname))
        seen_cokey = set()
        for row in tbl[1:]:
            total_horizons += 1
            cokey = row[idx["cokey"]]
            if cokey not in seen_cokey:
                seen_cokey.add(cokey)
                drainage_counts[row[idx["drainagecl"]] or "NULL"] += 1
            for c in numeric_cols:
                v = row[idx[c]]
                if v in (None, "", "null"):
                    col_nulls[c] += 1
                else:
                    col_vals[c].append(float(v))

    print("=== ssurgo ===")
    print(f"  Files present: {len(files)} / {len(all_ids)}")
    print(f"  Empty Table (no major soil horizon rows): {len(empty_table)} -> {empty_table[:10]}")
    print(f"  Water/Urban mapunit matches: {len(water_or_urban)} (e.g. {water_or_urban[:5]})")
    print(f"  Total horizon rows: {total_horizons}")
    print(f"  Major component drainage classes: {dict(drainage_counts)}")
    for c in numeric_cols:
        arr = np.array(col_vals[c]) if col_vals[c] else np.array([np.nan])
        print(
            f"  {c:14s}: nulls={col_nulls[c]:4d}/{total_horizons} ({100*col_nulls[c]/max(1,total_horizons):.1f}%), "
            f"min={np.min(arr):.2f}, p50={np.median(arr):.2f}, max={np.max(arr):.2f}"
        )


def inspect_soilgrids(all_ids: list[str]):
    dpath = DATA_DIR / "soilgrids"
    files = sorted(dpath.glob("*.json"))
    present_ids = {p.stem for p in files}
    missing_ids = [i for i in all_ids if i not in present_ids]
    all_null_files = []
    valid_files = 0
    units_info = {}
    for fp in files:
        obj = json.loads(fp.read_text())
        layers = obj.get("properties", {}).get("layers", [])
        has_any = False
        for lyr in layers:
            name = lyr["name"]
            units_info[name] = lyr.get("unit_measure", {})
            for d in lyr.get("depths", []):
                if d.get("values", {}).get("mean") is not None:
                    has_any = True
        if has_any:
            valid_files += 1
        else:
            all_null_files.append(fp.stem)

    print("=== soilgrids ===")
    print(f"  Files present: {len(files)} / {len(all_ids)} (missing on disk: {len(missing_ids)})")
    print(f"  First/last missing IDs: {missing_ids[:5]} ... {missing_ids[-5:]}")
    print(f"  Files with valid numeric soil values: {valid_files}; all-null (masked urban/water): {len(all_null_files)} -> {all_null_files[:10]}")
    for k, u in units_info.items():
        print(f"  Layer {k:8s}: mapped_units={u.get('mapped_units')}, d_factor={u.get('d_factor')}, target_units={u.get('target_units')}")


def read_dbf_summary(dbf_path: Path):
    data = dbf_path.read_bytes()
    num_records, header_len, record_len = struct.unpack("<IHH", data[4:12])
    fields = []
    offset = 32
    while data[offset] != 0x0D:
        name = data[offset:offset+11].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
        ftype = chr(data[offset+11])
        flen = data[offset+16]
        fields.append((name, ftype, flen))
        offset += 32
    rows = []
    pos = header_len
    for _ in range(num_records):
        if pos >= len(data):
            break
        rec = data[pos+1:pos+record_len]
        pos += record_len
        row = {}
        foff = 0
        for name, ftype, flen in fields:
            raw = rec[foff:foff+flen].decode("latin1", errors="ignore").strip()
            foff += flen
            row[name] = raw
        if row.get("CLASS_NAME") or int(row.get("COUNT", "0") or 0) > 0:
            rows.append(row)
    return fields, rows


def inspect_cdl_and_dem():
    print("=== cdl ===")
    for p in sorted((DATA_DIR / "cdl").glob("*.tif")):
        dbf = Path(str(p) + ".vat.dbf")
        print(f"  GeoTIFF: {p.name} ({p.stat().st_size / 1e6:.2f} MB)")
        if dbf.exists():
            fields, active_rows = read_dbf_summary(dbf)
            top = sorted(active_rows, key=lambda r: int(float(r.get("COUNT", 0) or 0)), reverse=True)[:8]
            print(f"  VAT DBF fields: {[f[0] for f in fields]}, active classes: {len(active_rows)}")
            print("  Top classes by pixel count:", [(r.get("VALUE"), r.get("CLASS_NAME"), r.get("COUNT")) for r in top])
    print("=== earthdata (DEM) ===")
    for p in sorted((DATA_DIR / "earthdata").glob("*.tif")):
        print(f"  DEM tile: {p.name} ({p.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    pts = load_points()
    all_ids = [r["ID"] for r in pts]
    print(f"Total points in appeears_points.csv: {len(pts)}")
    inspect_time_series("nasa_power", ["precip", "tmax", "tmin", "rh", "solar"], all_ids)
    inspect_time_series("nasa_power_targets", ["gwettop", "gwetroot", "gwetprof", "evland"], all_ids)
    inspect_time_series("open_meteo", ["sm_0_7cm", "sm_7_28cm", "sm_28_100cm", "soil_temp_0_7cm", "et0_mm"], all_ids)
    inspect_ssurgo(all_ids)
    inspect_soilgrids(all_ids)
    inspect_cdl_and_dem()

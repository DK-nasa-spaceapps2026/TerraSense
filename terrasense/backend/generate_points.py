"""Generate a seeded AppEEARS point-sample CSV.

Usage:
    python generate_points.py --n 300 --out appeears_points.csv
    python generate_points.py --n 300 --bbox -94.5 41.3 -92.8 42.5

AppEEARS reads these column headers: ID, Category, Latitude, Longitude.
Rows go in as lon/lat in decimal degrees (WGSW84).
Tip: if you have the CDL, filter these points to cropland pixels before
uploading so you don't train on towns, roads and water.
"""
import argparse
import csv
import random

DEMO = (41.6000, -93.6000)  # lat, lon (matches TerraSense demo field)
DEMO_HALF = 0.0045

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300, help="number of training points")
ap.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"),
                default=[-94.5, 41.3, -92.8, 42.5], help="lon/lat bounds")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="appeears_points.csv")
a = ap.parse_args()

random.seed(a.seed)
w, s, e, n = a.bbox
rows = [("demo_field_center", "demo", DEMO[0], DEMO[1])]
for name, dy, dx in [("nw", 1, -1), ("ne", 1, 1), ("sw", -1, -1), ("se", -1, 1)]:
    rows.append((f"demo_field_{name}", "demo", DEMO[0] + dy * DEMO_HALF, DEMO[1] + dx * DEMO_HALF))
for i in range(1, a.n + 1):
    rows.append((f"train_{i:04d}", "train", random.uniform(s, n), random.uniform(w, e)))

with open(a.out, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["ID", "Category", "Latitude", "Longitude"])
    for r in rows:
        wr.writerow([r[0], r[1], f"{r[2]:.5f}", f"{r[3]:.5f}"])
print(f"Wrote {len(rows)} points to {a.out}")

"""Median-merge repeated compiler-flag measurements (one row per run)."""
import csv
import statistics
import sys

src, dst = sys.argv[1], sys.argv[2]
groups = {}
with open(src) as f:
    for row in csv.DictReader(f):
        groups.setdefault(row["variant"], []).append(
            (float(row["scalar_ns"]), float(row["neon1_mtps"]), float(row["neonN_mtps"])))
with open(dst, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "scalar_ns", "neon1_mtps", "neonN_mtps"])
    for name, vals in groups.items():
        ns = statistics.median(v[0] for v in vals)
        m1 = statistics.median(v[1] for v in vals)
        mn = statistics.median(v[2] for v in vals)
        w.writerow([name, f"{ns:.2f}", f"{m1:.1f}", f"{mn:.1f}"])

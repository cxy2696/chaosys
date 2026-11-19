"""Merge N repeated benchmark runs into median tables for the report.

Each run directory contains single.csv / ensemble.csv / scaling.csv /
accuracy.csv as written by ./build/chaosys_bench.
"""
import csv
import os
import statistics
import sys

RAW = sys.argv[1] if len(sys.argv) > 1 else "results/raw"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
RUNS = [os.path.join(RAW, str(i)) for i in (1, 2, 3)]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def median(xs):
    return statistics.median(xs)


def merge_single():
    groups = {}
    for run in RUNS:
        for row in read(os.path.join(run, "single.csv")):
            groups.setdefault(row["variant"], []).append(row)
    out = []
    for name, rows in groups.items():
        ns = median(float(r["ns_per_step"]) for r in rows)
        mb = median(float(r["msteps_per_s"]) for r in rows)
        gb = median(float(r["gbps_out"]) for r in rows)
        out.append(dict(variant=name, ns_per_step=f"{ns:.2f}",
                        msteps_per_s=f"{mb:.1f}", gbps_out=f"{gb:.2f}",
                        bytes_out=rows[0]["bytes_out"]))
    w = csv.DictWriter(open(os.path.join(OUT, "single.csv"), "w", newline=""),
                       fieldnames=["variant", "ns_per_step", "msteps_per_s",
                                   "gbps_out", "bytes_out"])
    w.writeheader()
    w.writerows(out)


def merge_ensemble():
    groups = {}
    for run in RUNS:
        for row in read(os.path.join(run, "ensemble.csv")):
            groups.setdefault(row["variant"], []).append(row)
    base = median(float(r["mtraj_steps_per_s"])
                  for r in groups["AoS scalar (baseline)"])
    out = []
    for name, rows in groups.items():
        mtps = median(float(r["mtraj_steps_per_s"]) for r in rows)
        gf = median(float(r["gflops"]) for r in rows)
        out.append(dict(variant=name, threads=rows[0]["threads"],
                        mtraj_steps_per_s=f"{mtps:.1f}", gflops=f"{gf:.1f}",
                        speedup=f"{mtps / base:.2f}"))
    w = csv.DictWriter(open(os.path.join(OUT, "ensemble.csv"), "w", newline=""),
                       fieldnames=["variant", "threads", "mtraj_steps_per_s",
                                   "gflops", "speedup"])
    w.writeheader()
    w.writerows(out)


def merge_scaling():
    per_thread = {}
    for run in RUNS:
        rows = read(os.path.join(run, "scaling.csv"))
        base = float(rows[0]["mtraj_steps_per_s"])
        for r in rows:
            t = int(r["threads"])
            mtps = float(r["mtraj_steps_per_s"])
            per_thread.setdefault(t, []).append((mtps, mtps / base))
    out = []
    base = median(v[0] for v in per_thread[1])
    for t in sorted(per_thread):
        mtps = median(v[0] for v in per_thread[t])
        out.append(dict(threads=t, mtraj_steps_per_s=f"{mtps:.1f}",
                        speedup=f"{mtps / base:.2f}"))
    w = csv.DictWriter(open(os.path.join(OUT, "scaling.csv"), "w", newline=""),
                       fieldnames=["threads", "mtraj_steps_per_s", "speedup"])
    w.writeheader()
    w.writerows(out)


def merge_accuracy():
    groups = {}
    for run in RUNS:
        for row in read(os.path.join(run, "accuracy.csv")):
            groups.setdefault((row["scheme"], row["dt"]), []).append(row)
    out = []
    for (scheme, dt), rows in groups.items():
        out.append(dict(scheme=scheme, dt=dt, steps=rows[0]["steps"],
                        error=rows[0]["error"],
                        ns_per_step=f'{median(float(r["ns_per_step"]) for r in rows):.2f}',
                        cost_ms=f'{median(float(r["cost_ms"]) for r in rows):.4f}'))
    out.sort(key=lambda r: (r["scheme"], -float(r["dt"])))
    w = csv.DictWriter(open(os.path.join(OUT, "accuracy.csv"), "w", newline=""),
                       fieldnames=["scheme", "dt", "steps", "error",
                                   "ns_per_step", "cost_ms"])
    w.writeheader()
    w.writerows(out)


if __name__ == "__main__":
    merge_single()
    merge_ensemble()
    merge_scaling()
    merge_accuracy()

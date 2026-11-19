#!/usr/bin/env bash
# Build everything, run the benchmark three times and merge medians.
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
make -j"$JOBS"
make test
make test-c
make smoke

rm -rf results/raw
mkdir -p results/raw

for i in 1 2 3; do
  echo "== benchmark run $i/3 =="
  ./build/chaosys_bench > "results/raw/run$i.txt"
  mkdir -p "results/raw/$i"
  cp results/single.csv results/accuracy.csv results/ensemble.csv \
     results/scaling.csv "results/raw/$i/"
done

python3 scripts/merge_runs.py results/raw results

echo "== compiler-flag ablation =="
make variants

echo "done: results/"

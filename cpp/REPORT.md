# Optimizing the Lorenz attractor in C/C++: from 59 M to 1.44 G trajectory steps/s

This report starts from the Lorenz equations, builds a C++20 baseline, optimizes it step by step, and
checks correctness against a Dormand–Prince 5(4) reference. All numbers can be reproduced with the
commands in Section 6.

## Summary

| Case | Baseline | Optimized | Result |
|---|---|---|---|
| 32,768 trajectories × 2,048 steps | AoS scalar, 59.3 M traj-step/s | SoA + 11 threads, 1,441.9 M | 24.3× |
| Single-core ensemble | 59.3 M | 344.5 M (auto-vectorized) | 5.8× |
| Thread scaling (1 → 11) | 339.6 M | 1,441.6 M | 4.25× |
| Single trajectory, equal accuracy (T=10) | Euler dt=1e-5: 3.86 ms, error 6.8e-3 | RK4 dt=5e-3: 0.034 ms, error 3.2e-5 | error 1/209, cost 1/113 |
| Single-trajectory latency | naive RK4 16.62 ns | fused/in-place 16.77 ns | 0.99× (no gain) |

Three conclusions:

1. A single trajectory is limited by the serial RK4 dependency chain, about 17 ns/step or roughly
   60 cycles. Fusing and removing allocations gives no measurable gain. The real win is switching
   algorithms: Euler to RK4 reduces cost by two orders of magnitude at equal accuracy.
2. Ensembles are where SIMD and threads pay off. SoA layout enables auto-vectorization (5.8×), and
   block threading adds 4.25×, for 24× overall.
3. `-ffast-math` has no stable single-core benefit. It gives about +7% with threads, but changes
   roundoff from 1e-11 to 3e-8 at T=2 and breaks reproducibility. It is off by default.

## 1. Problem and baseline

The Lorenz system is

```text
dx/dt = σ (y - x)
dy/dt = x (ρ - z) - y
dz/dt = x y - β z
```

with σ=10, ρ=28, β=8/3 and initial state (1,1,1). Ensemble initial states are uniform in
[-15,15] × [-20,20] × [5,40], inside the attractor's bounding box with some transient.

The baseline is deliberately tutorial-style (`src/baseline.cpp`, compiled with `-O2`):

- AoS state `struct { double x, y, z; }`, derivatives returned by value, repeated copies
- `std::vector<State>` with `push_back` and no `reserve`
- The same scalar RK4 chain for single and ensemble runs

This is the kind of code that runs and records, and it is the control for every optimization below.

## 2. Measurement

| Item | Value |
|---|---|
| Machine | Apple M3 Pro, 5 P-cores + 6 E-cores, 36 GB |
| OS | macOS 27.2.1 (Darwin 27.2.0) |
| Compiler | Apple clang 21.0.0, `-std=c++20 -O3 -mcpu=native -DNDEBUG` |
| Timing | `std::chrono::steady_clock`; one warm-up, best of 5 per group; 3 full runs, median |
| Ensemble load | n=32,768, steps=2,048, dt=0.01 (T=20.48, 6.7×10⁷ traj-steps) |
| Scaling load | n=65,536, steps=1,024, dt=0.01 |

The machine was shared during measurement: `vm.loadavg` fluctuated between 9 and 26 on 11 logical
cores. Absolute throughput is therefore a conservative lower bound; ratios within a round are more
reliable. A later idle window (loadavg≈5) measured 432–495 M single-core NEON and 2.45–2.80 G with
11 threads (37–46×), consistent with the conservative medians below.

GFLOP/s is an equivalent value assuming 79 floating-point operations per RK4 step (4 RHS × 10 flop
plus 39 updates). It is a scale reference, not a hardware counter.

## 3. Optimization path

### 3.1 Algorithm: Euler to RK4

Euler has a cheap RHS but first-order accuracy. RK4 uses four RHS evaluations per step and is
fourth-order. In a chaotic system the error is amplified by the Lyapunov exponent (λ≈0.906 here),
so the step-count difference is large:

| Method | dt | Steps (T=10) | Max error | ns/step | Cost |
|---|---|---|---|---|---|
| Euler | 1e-3 | 10,000 | 1.47 | 3.86 | 0.039 ms |
| Euler | 1e-4 | 100,000 | 7.16e-2 | 3.86 | 0.386 ms |
| Euler | 1e-5 | 1,000,000 | 6.77e-3 | 3.86 | 3.855 ms |
| RK4 | 5e-2 | 200 | 3.38 | 16.88 | 0.003 ms |
| RK4 | 2e-2 | 500 | 3.22e-2 | 16.88 | 0.008 ms |
| RK4 | 1e-2 | 1,000 | 1.03e-3 | 16.88 | 0.017 ms |
| RK4 | 5e-3 | 2,000 | 3.24e-5 | 16.88 | 0.034 ms |
| RK4 | 2e-3 | 5,000 | 3.35e-7 | 16.88 | 0.084 ms |

The reference is the generic adaptive Dormand–Prince 5(4) in `chaosys/ode.hpp` at rtol=1e-13, with
self-consistency about 7e-11. Comparing Euler dt=1e-5 with RK4 dt=5e-3 gives 1/209 the error at
1/113 the cost. Euler would need roughly 7× more steps to reach 1e-3.


### 3.2 Single trajectory: fuse stages, remove allocations

The four RK4 RHS evaluations are strictly serial. The only room is keeping intermediates in
registers and avoiding calls and temporary objects. A fused in-place step looks like:

```cpp
inline void rk4_step_inplace(double& x, double& y, double& z, double dt,
                             double sigma, double rho, double beta) {
  const double h2 = 0.5 * dt, h6 = dt / 6.0;
  double k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;
  k1x = sigma * (y - x);  k1y = x * (rho - z) - y;  k1z = x * y - beta * z;
  const double x2 = x + h2 * k1x, y2 = y + h2 * k1y, z2 = z + h2 * k1z;
  k2x = sigma * (y2 - x2); /* ... */
  /* k3, k4 */
  x += h6 * (k1x + 2.0*k2x + 2.0*k3x + k4x);
  y += h6 * (k1y + 2.0*k2y + 2.0*k3y + k4y);
  z += h6 * (k1z + 2.0*k2z + 2.0*k3z + k4z);
}
```

Measured: 0.99× (16.62 → 16.77 ns/step). Section 4.1 explains why. Sampled output also does not
need every step; `rk4_scalar_strided` integrates in blocks and samples at fixed intervals.

### 3.3 Layout: AoS to SoA

The ensemble win is not making one trajectory faster; it is exposing parallelism across
trajectories:

```cpp
struct Ensemble { std::vector<double> x, y, z; std::size_t n; };
```

`x[i]`, `y[i]`, and `z[i]` are independent, so the inner loop has no cross-trajectory dependency.
Clang packs two doubles into NEON registers and interleaves several vector chains. With AoS, a
single 2,048-step trajectory is serial and the compiler has nothing to vectorize.

### 3.4 SIMD: auto-vectorization and explicit NEON

NEON is baseline on ARM64, so no `-mavx2` flag is needed. At `-O3`, clang auto-vectorizes the SoA
loop into 2-lane `f64` code (344 M single-core). The explicit NEON version uses `float64x2_t`,
broadcasts σ/ρ/−β once per block, and uses seven instructions per RHS:

```cpp
inline V3 vderiv(V3 v, const float64x2_t& s, const float64x2_t& r,
                 const float64x2_t& nb) {
  V3 d;
  d.x = vmulq_f64(s, vsubq_f64(v.y, v.x));                 // σ(y−x)
  d.y = vfmaq_f64(vnegq_f64(v.y), vsubq_f64(r, v.z), v.x); // x(ρ−z)−y
  d.z = vfmaq_f64(vmulq_f64(v.x, v.y), v.z, nb);           // xy−βz
  return d;
}
```

It processes eight trajectories per iteration (four vector chains) to fill the FMA pipeline.
Measured 360 M, about 5% above auto-vectorization. The explicit NEON path is a fallback, not a
requirement: SoA + `-O3` is already in the same range.

### 3.5 Cache blocking

Each thread takes 512 trajectories (3×512×8 B = 12 KiB of state), advances them through all time
steps, then takes the next block. State is read and written every step; a block smaller than L1
stays resident. The block is also large enough to amortize scheduling. Blocks are independent, with
no locks and no false sharing (boundaries are 512×8 B aligned).

### 3.6 Threads

Each call starts `nthreads` workers, including the main thread, and workers claim blocks
dynamically:

```cpp
parallel_blocks(n, /*block=*/512, nthreads, [&](size_t b, size_t e) {
  soa_neon_range(x, y, z, b, e, steps, dt);
});
```

An atomic block counter provides work stealing. Each trajectory's arithmetic does not depend on
block assignment, so multi-threaded results are bitwise identical to single-threaded results
(Section 5).

### 3.7 Compiler flags

| Flags | Single-trajectory ns/step | SoA single-core M traj-step/s |
|---|---|---|
| `-O2` | 16.44 | 339.0 |
| `-O3` | 16.72 | 343.1 |
| `-O3 -mcpu=native` | 16.48 | 303.2 |
| `-O3 -mcpu=native -ffast-math` | 17.39 | 364.7 |

The NEON kernel uses intrinsics; `-O2/-O3/-mcpu=native` differences are within ±10% run noise.
`-mcpu=native` has no stable benefit on M3 because NEON is already baseline. `-ffast-math` is also
within noise single-core, changes results, and breaks reproducibility, so it is not used by default.

## 4. Results

### 4.1 Single trajectory: latency-bound

5×10⁶ steps at dt=0.001:

| Implementation | ns/step | M steps/s | Output |
|---|---|---|---|
| Euler naive + `push_back` | 3.69 | 270.8 | 120 MB (6.5 GB/s) |
| RK4 naive + `push_back` | 16.62 | 60.2 | 120 MB (1.4 GB/s) |
| RK4 fused + in-place | 16.77 | 59.6 | 0 |
| RK4, sample every 10 steps | 17.52 | 57.1 | 12 MB |

Why no gain from fusion? One step takes about 60 cycles, and the critical path is
`k1 → k2 → k3 → k4 → update`. Each RHS has at least two or three dependent FP operations
(sub → fma), so latency alone is about 50 cycles. The 24 B/step store bandwidth is hidden by that
computation. There is no ILP left to exploit; the compiler's scalar code is near the dependency
limit.

When only sampled output is needed, use RK4 at dt≈0.01 instead of Euler at dt=0.001, and store
only the samples. T=100 takes 10,000 steps, about 0.17 ms.


### 4.2 Ensemble throughput: 24×

32,768 trajectories × 2,048 steps (T=20.48), final states only:

| Version | Threads | M traj-step/s | Equivalent GFLOP/s | Speedup |
|---|---|---|---|---|
| AoS scalar (baseline) | 1 | 59.3 | 4.7 | 1.00× |
| SoA scalar (auto-vectorized) | 1 | 344.5 | 27.2 | 5.81× |
| SoA scalar + threads | 11 | 1,441.9 | 113.9 | 24.33× |
| SoA NEON | 1 | 360.1 | 28.4 | 6.08× |
| SoA NEON + threads | 11 | 1,353.9 | 107.0 | 22.84× |
| SoA NEON + fast-math | 1 | 352.9 | 27.9 | 5.95× |
| SoA NEON + fast-math + threads | 11 | 1,536.5 | 121.4 | 25.92× |


The 5.8× single-core gain comes from cross-trajectory parallelism: 17 ns/step becomes 2.90 ns/step
when amortized across the SoA lanes. Threads add 4.25×, for 24.3×. Explicit NEON is about 5% above
auto-vectorization (360 vs. 344 M), so the two are in the same class. fast-math is flat single-core
and about +7% with threads, but changes trajectories irreversibly. The practical rule is: fix the
layout first, then discuss instructions.

> Fairness: the baseline is compiled at `-O2`. Recompiling the same AoS baseline with
> `-O3 -mcpu=native` still gives 56.7–57.7 M, so the 5.8× is not a compiler-flag artifact.

### 4.3 Thread scaling

n=65,536, steps=1,024:

| Threads | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|
| M traj-step/s | 339.6 | 643.3 | 905.3 | 1,088.0 | 1,232.2 | 1,217.8 | 1,369.0 | 1,450.2 | 1,441.6 |
| Speedup | 1.00 | 1.89 | 2.67 | 3.20 | 3.63 | 3.59 | 4.03 | 4.27 | 4.25 |


The M3 Pro has 5 P-cores and 6 E-cores. An E-core is roughly one third of a P-core single-thread,
and threads migrate between them. With 11 workers, completion is set by the slowest E-core thread,
so the curve flattens after five threads. The small 10→11 rise is within noise. The machine was
shared (loadavg 9–26), so 4.25× is conservative; an idle machine gives more. Using
`hardware_concurrency()` is the default; leaving one core for the system makes no measurable
difference here.

### 4.4 Summary

```text
59.3 M   AoS scalar, 1 thread
  ↓ 5.8×   SoA + auto-vectorization (single core)
344 M
  ↓ 4.25×  512 trajectories/block + dynamic scheduling, 11 threads
1.44 G   ≈ 24× baseline
```

### 4.5 Generic and cross-language interfaces

σ/ρ/β are runtime `Params`, and the `OdeSystem` concept accepts custom chaotic systems of any
dimension with `float` or `double`. Euler/Heun/RK4 and adaptive DP5 are available; `EnsembleN`
handles generic ensembles. The SoA/NEON Lorenz path remains the high-performance specialization.

The analysis layer provides `largest_lyapunov`, `poincare_section`, `embed`, and
`correlation_dimension`. The C ABI exposes the same solvers with flat arrays; Python reuses them
through ctypes. See [API](../docs/API.md), [EXTENDING](../docs/EXTENDING.md),
[QUICKSTART](../docs/QUICKSTART.md), and [RIGOR](../docs/RIGOR.md).

## 5. Validation

| Check | Result |
|---|---|
| DP5 self-consistency (rtol 1e-13 vs. 3e-14) | 7.10e-11 |
| Naive RK4 vs. fused RK4, 1,000 steps | 0 (bitwise identical) |
| Threaded vs. single-threaded scalar, 8,192 × 200 | 0 (bitwise identical) |
| NEON vs. scalar, 1 step | 1.8e-15 / 3.6e-15 / 7.1e-15 |
| NEON vs. scalar, 200 steps (T=2) | 5.0e-11 / 1.1e-10 / 3.2e-11 |
| fast-math vs. IEEE, 200 steps (T=2) | 1.8e-8 / 3.7e-8 / 1.1e-8 |
| Largest Lyapunov exponent (Benettin, 1,000 segments) | 0.894 (theory 0.906) |
| T=10,000 bounds | max\|x\|=18.22, max r_xy=27.49, z∈[4.46, 45.46] |

For chaotic systems, two mathematically identical trajectories separated by 1e-16 grow by
e^{λT}≈6.6e7 at T=20, reaching about 1e-8; the most sensitive of 8,192 trajectories can reach
1e-4. This is chaos, not a bug. Therefore:

- Same implementation, different threads/blocks must be bitwise identical (verified 0).
- Different implementations (scalar vs. NEON) are compared at T=2 on the roundoff scale (≈1e-11).
- Pointwise comparison after T=20 is meaningless.

The 1e-8 fast-math difference comes from reassociation and is five orders of magnitude smaller than
the 1e-3 truncation error of RK4 dt=0.01 at T=10. It still breaks reproducibility and is off by
default.

## 6. Reproduction

```bash
cd cpp

make -j
./build/chaosys_bench              # validation, accuracy, single, ensemble, scaling
./build/chaosys_bench validate     # correctness and dynamics only
./build/chaosys_bench quick        # validation + accuracy
./build/chaosys_bench traj         # generate results/traj.csv
./build/chaosys_cli help           # CLI

make test                         # 145 C++ checks
make test-c                       # 130 C ABI checks
make test-exceptions              # 11 exception-boundary checks
make test-time-header             # time-header/Lyapunov check
make test-asan                    # ASan + UBSan
make smoke                        # CLI smoke
make test-all                     # all of the above

cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake -j
ctest --test-dir build-cmake
cmake --install build-cmake --prefix "$HOME/.local"

./scripts/run_all.sh              # 3 benchmark rounds + medians
```

Outputs in `results/`: `single.csv`, `accuracy.csv`, `ensemble.csv`, `scaling.csv`, `flags.csv`,
`params.csv`, and `validate.txt`. `cpp/scripts/run_all.sh` also creates a `raw/` directory during
a run; it is not tracked.

## 7. Recommendations and limitations

Recommendations:

- One trajectory: RK4 at dt=0.01 with sampled output. Do not reach for SIMD.
- Parameter scans, Lyapunov diagrams, basins: SoA, `-O3`, `hardware_concurrency()` threads. This
  machine measured 24×.
- Leave `-ffast-math` off by default: no stable single-core gain, about +7% with threads, but
  irreproducible trajectories and worse accuracy.
- Do not require `-march=native`: NEON is baseline on ARM64; on x86, AVX2/FMA is worth enabling.
- New systems: use `chaosys::integrate` / `EnsembleN`; only use the Lorenz-specific kernels when the
  ensemble is the proven bottleneck.

Limitations:

- The machine was shared (loadavg 9–26); absolute throughput is a lower bound, and scaling is
  affected by the 5P+6E topology and scheduling.
- The SIMD ensemble kernel is double-only. The generic path supports float, but f32 NEON was not
  measured.
- No GPU path; very large ensembles (10⁶+) may benefit from one.
- GFLOP/s is an equivalent estimate, not a hardware counter.

## Appendix: files

| File | Purpose |
|---|---|
| `include/chaosys.hpp` | `State` and runtime parameter API |
| `include/chaosys/params.hpp` | `Params` and validation |
| `include/chaosys/ode.hpp` | System concept, Euler/Heun/RK4/DP5, adaptive integration, `EnsembleN` |
| `include/chaosys/aligned.hpp` | 64-byte aligned allocator |
| `include/chaosys/systems.hpp` | Built-in Rössler and Chen systems |
| `include/chaosys/analysis.hpp` | Largest Lyapunov exponent, Poincaré section |
| `include/chaosys/chaos_metrics.hpp` | Takens embedding, correlation dimension, permutation entropy |
| `include/chaosys/time_system.hpp` | TimeOdeSystem, augmented-time integration, Lyapunov |
| `include/chaosys/parallel.hpp` | Shared block scheduler |
| `include/chaosys/driver.hpp` | Validation, thread clamp, SoA dispatch |
| `include/chaosys/csv.hpp` | CSV writer for states, sections, series, tables |
| `include/chaosys_c.h` | Stable C ABI |
| `src/c_api.cpp` | C ABI implementation |
| `src/kernels_body.inc` | Kernel body shared by IEEE and fast-math TUs |
| `src/kernels.cpp` | SoA, NEON, block threading, ensemble initialization |
| `src/kernels_fast.cpp` | fast-math instantiation |
| `src/baseline.cpp` | Tutorial-style AoS baseline |
| `src/bench.cpp` | Benchmark, accuracy study, validation |
| `apps/chaosys_cli.cpp` | CLI |
| `tests/test_chaosys.cpp` | 145 C++ checks |
| `tests/test_c_api.c` | 130 C ABI checks |
| `tests/test_exception_safety.cpp` | 11 exception-boundary checks |
| `tests/test_time_header.cpp` | standalone time header + `λ ≈ -1` |
| `../python/chaosys.py` | ctypes binding |
| `Makefile` | Development build, tests, ASan, flag ablation |
| `CMakeLists.txt` | Product build, install, CMake package, pkg-config |
| `cmake/` | CMake and pkg-config templates |
| `../docs/QUICKSTART.md` | Install and language examples |
| `../docs/API.md` | API reference |
| `../docs/PURPOSE.md` | Scope and non-goals |
| `../docs/EXTENDING.md` | Adding systems |
| `../docs/RIGOR.md` | Numerical properties and tests |
| `../python/pyproject.toml` | Python package metadata |
| `../.github/workflows/ci.yml` | Ubuntu/macOS CI |
| `scripts/run_all.sh` | Reproduction script |
| `scripts/merge_runs.py` | Median merge |
| `results/` | Data and validation records |

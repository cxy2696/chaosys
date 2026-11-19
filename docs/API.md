# API

See [PURPOSE.md](PURPOSE.md) for scope, [EXTENDING.md](EXTENDING.md) for custom systems, and
[RIGOR.md](RIGOR.md) for numerical properties.

## C ABI

All functions in `chaosys_c.h` are thread-safe, return `chaosys_status`, and never let a C++
exception cross the boundary.  A callback exception is reported as
`CHAOSYS_ERR_INTERNAL`; a non-finite result is reported as `CHAOSYS_ERR_NUMERIC`.

| Function | Purpose |
|---|---|
| `chaosys_default_params()` | σ=10, ρ=28, β=8/3 |
| `chaosys_version()` | version string |
| `chaosys_status_string()` | status code as text |
| `chaosys_rk4()` | Lorenz fixed-step RK4 |
| `chaosys_adaptive()` | Lorenz adaptive DP5 |
| `chaosys_ensemble()` | Lorenz ensemble, multi-threaded |
| `chaosys_lyapunov()` | Lorenz largest Lyapunov exponent |
| `chaosys_lyapunov_report()` | Lorenz exponent + per-segment statistics |
| `chaosys_lyapunov_spectrum()` | Lorenz full spectrum (3 values, descending) |
| `chaosys_poincare()` | Lorenz Poincaré section |
| `chaosys_rk4_system()` | custom system, fixed RK4 |
| `chaosys_adaptive_system()` | custom system, adaptive DP5 |
| `chaosys_lyapunov_system()` | custom system, largest exponent |
| `chaosys_lyapunov_report_system()` | custom system, exponent + statistics |
| `chaosys_lyapunov_spectrum_system()` | custom system, full spectrum |
| `chaosys_poincare_system()` | custom system, Poincaré |
| `chaosys_ensemble_system()` | custom system, ensemble |
| `chaosys_rk4_time_system()` | time-dependent system, fixed RK4 |
| `chaosys_adaptive_time_system()` | time-dependent system, adaptive DP5 |
| `chaosys_lyapunov_time_system()` | time-dependent system, largest exponent |
| `chaosys_embed()` | Takens delay embedding |
| `chaosys_correlation_dimension()` | correlation dimension |
| `chaosys_permutation_entropy()` | permutation entropy |

Status codes: `CHAOSYS_OK`, `CHAOSYS_ERR_ARGUMENT`, `CHAOSYS_ERR_NUMERIC`,
`CHAOSYS_ERR_MEMORY`, `CHAOSYS_ERR_INTERNAL`.

### Lyapunov diagnostics

`chaosys_lyapunov_report()` fills `chaosys_lyapunov_stats` (`lambda`, `stddev`,
`stderr_value`, `converged`, `segments`) and writes one value per segment into
`segment_lambdas` (`capacity >= segments`).  `converged` is advisory: at least 8
segments and a standard error of at most 10% of `max(1, |lambda|)`.

`chaosys_lyapunov_spectrum()` writes `count` descending exponents; pass capacity
3 for Lorenz or `system->dim` for a custom system.

### Custom chaotic ODEs

```c
/* Rössler: a=0.2, b=0.2, c=5.7 */
void rossler(const double* x, double* dx, void* user) {
  (void)user;
  dx[0] = -x[1] - x[2];
  dx[1] = x[0] + 0.2 * x[1];
  dx[2] = 0.2 + x[2] * (x[0] - 5.7);
}

chaosys_system sys = {.dim = 3, .deriv = rossler, .user = NULL};
double x[3] = {1, 1, 1};
chaosys_adaptive_system(&sys, x, 100.0, 1e-9, 1e-11, NULL);
```

`dim` is between 1 and 64. The C ABI pads to 64 lanes internally so it can reuse
the same DP5/RK4 implementations; extra lanes stay zero. Ensemble states use flat
`n * dim` arrays; a callback must be thread-safe when `threads > 1`. A callback
that raises a C++ exception is caught and reported as `CHAOSYS_ERR_INTERNAL`.

`chaosys_embed()` writes `count * dimension` doubles; `capacity` is the number
of doubles available in `out`, so it must be at least `count * dimension`.
`chaosys_poincare()` capacity is measured in points and `out` must hold
`capacity * dim` doubles.

Correlation dimension is a finite-sample estimate; periodic, strongly aliased, or
degenerate series may fail or be biased.  `max_points` decimation and the Theiler window are documented in
[RIGOR.md](RIGOR.md).

## C++ core

- Built-in `LorenzSystem`, `RosslerSystem`, `ChenSystem`, and `DuffingSystem` in
  `chaosys/systems.hpp`
- `OdeSystem` concept: `real`, `dim`, `deriv(x, dx) noexcept`
- Optional `HasInitialState` concept: `S::initial_state()` returns exactly
  `std::array<S::real, S::dim>`
- `TimeOdeSystem` and `AugmentedSystem` in `chaosys/time_system.hpp`;
  `integrate_time`, `integrate_adaptive_time`, `largest_lyapunov_time`
- `EulerMethod`, `HeunMethod`, `Rk4Method`, `DormandPrinceMethod`
- `integrate`, `integrate_adaptive`, `EnsembleN`, `integrate_ensemble`
- `largest_lyapunov`, `largest_lyapunov_report`, `lyapunov_spectrum`,
  `poincare_section`
- `LyapunovReport` (`lambda`, per-segment estimates, `stddev`, `stderr`,
  `converged`) and `LyapunovSpectrum` (`exponents`, `sum`)
- `embed`, `correlation_dimension`, `correlation_dimension_series`,
  `permutation_entropy`
- `write_csv` in `chaosys/csv.hpp`: numeric tables, scalar series, `State`
  trajectories, and `SectionPoint` lists
- Runtime `Params` and boundary checks

## Python

`chaosys.py` mirrors the C ABI: `rk4`, `adaptive`, `ensemble`, `lyapunov`,
`lyapunov_report`, `lyapunov_spectrum`, `poincare`, `rk4_system`,
`adaptive_system`, `lyapunov_system`, `lyapunov_report_system`,
`lyapunov_spectrum_system`, `poincare_system`, `ensemble_system`,
`rk4_time_system`, `adaptive_time_system`, `lyapunov_time_system`, `embed`,
`correlation_dimension`, `permutation_entropy`, and `to_csv`.

`LyapunovReport` is a frozen dataclass with `exponent`, `stddev`, `stderr`,
`converged`, `segments`, and `segment_lambdas`.  `to_csv(path, rows, header=None)`
writes `State` rows as `x,y,z` (or any tuple/list) and returns the row count.

## Versioning

ABI version 1.0.0 with `SOVERSION 1`. Installed packages provide `chaosys::chaosys` and
`chaosys.pc`.

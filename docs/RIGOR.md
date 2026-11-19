# Rigor and verification

This document records numerically checkable properties, invariants, and test evidence. It is not a
formal proof. Chaotic sensitivity and finite-sample statistics still need judgement from the user.

## 1. One-step methods and convergence order

For `dx/dt = -x`, `x(0)=1`, integrated to `T=1`, the exact solution is `e^-1`.

| Method | dt=0.1 error | dt=0.05 error | dt=0.025 error | Halving ratio | Expected |
|---|---|---|---|---|---|
| Euler | 1.92e-2 | 9.39e-3 | 4.65e-3 | 2.04, 2.02 | 2¹ |
| Heun | 6.62e-4 | 1.59e-4 | 3.91e-5 | 4.16, 4.08 | 2² |
| RK4 | 3.33e-7 | 2.00e-8 | 1.22e-9 | 16.68, 16.34 | 2⁴ |

The ratios are checked by `test_generic_decay`, `test_heun_order`, and `test_rk4_order`.  The
tolerance bands are deliberately loose enough to absorb asymptotic pre-factors but tight enough to
reject a wrong tableau or an incorrect intermediate stage.

## 2. Dormand–Prince 5(4)

The implementation uses the standard DOPRI5 Butcher tableau: seven stages, a fifth-order solution
`y5`, and an embedded fourth-order solution `y4`.  The controller accepts a step only when the
scaled maximum error satisfies `err <= 1`; the trial step is `0.9 * err^(-1/5)` with 0.2/5 growth
limits.

- **Fixed-step order:** with `dt = 0.2, 0.1, 0.05, 0.025`, the global errors are
  4.55e-8, 1.21e-9, 3.48e-11, 1.04e-12.  Halving ratios are 37.6, 34.8, 33.4, approaching 2⁵ = 32
  (`test_dp5_fixed_order`).
- **Error control:** tighter `rtol/atol` increases accepted steps and reduces error
  (`test_adaptive_decay`).
- **NaN rejection:** the error norm checks `std::isfinite(term)` before `std::max`; otherwise a NaN
  produced by the vector field would be silently ignored and the step accepted.  `NaNSystem` must
  raise `std::runtime_error` (`test_adaptive_non_finite`).
- **Horizon:** the final step is truncated to land exactly on `t_end`; `t_end = 0` is a no-op.
- **Underflow / progress:** a step is rejected if `h <= 0`, `h` is non-finite, or `t + h == t` (no
  progress).  The old `t_end`-relative floor incorrectly aborted feasible runs with a tiny initial
  step; `test_adaptive_small_step` starts at `h = 1e-15` and completes.
- **Attempt limit:** only *consecutive* rejected steps are capped (10⁶).  A long run with many
  accepted steps is allowed.

## 3. Largest Lyapunov exponent and spectrum

The largest exponent uses Benettin's two-trajectory method:

1. Burn in the reference trajectory for `transient_steps`.
2. Perturb by `d0` in a deterministic unit direction and renormalise to `d0` after every step.
3. Accumulate `log(distance / d0)`; divide by `segment_steps * dt` per segment and by the total
   span for the final estimate.

Properties and evidence:

- **Generic direction:** the perturbation spans all active components, not only `x[0]`.  For
  `DecoupledLorenz` (`dx0 = -x0` plus a Lorenz subsystem in `x1..x3`), a perturbation restricted to
  `x0` would measure `λ = -1`; the library measures `λ ≈ 0.8945` and the spectrum
  `[0.8915, -0.0402, -1.0, -14.518]`.  `test_lyapunov_report_and_spectrum` checks this.
- **Time-dependent systems:** only the state subspace is perturbed.  The appended time lane would
  otherwise add a non-decaying offset to the distance.  `dx/dt = -x + cos(t)` returns `λ ≈ -1`
  (`test_time_header`), not the biased value near zero that the time lane would produce.
- **State-scaled d0:** the effective perturbation is `d0 * max(1, |x|∞)` over the perturbed
  components; this keeps the difference representable for a state of order `1e10`
  (`test_lyapunov_large_state` returns `-1.000e-3` for `dx/dt = -1e-3 x`).  Renormalisation divides
  before multiplying to avoid overflow for subnormal distances.
- **Input contract:** the caller's `x0` is copied and never mutated.
- **Statistics:** `largest_lyapunov_report` returns per-segment estimates, their sample standard
  deviation, the standard error of the mean, and an advisory `converged` flag (at least 8 segments
  and `stderr <= 0.1 * max(1, |lambda|)`).
- **Spectrum:** `lyapunov_spectrum` propagates a reference trajectory plus `N` perturbed copies,
  applies modified Gram–Schmidt (QR) after every step, accumulates `log(r_ii / d0)`, and sorts the
  exponents descending.  The sum equals the average divergence.  For Lorenz the measured sum is
  `-13.666661`, matching `-(σ + 1 + β) = -13.666667` to 6e-6; the three exponents are
  `0.8915, -0.0402, -14.5180`.
- **Classic Lorenz largest exponent:** `0.8795` for the default `2000` transient + `50×200`
  segment run, and `0.8915` for the spectrum; the finite-time window shifts the estimate by about
  1%.  Tests accept the range `[0.7, 1.1]`; the convergence flag is false for the short default run
  because the local-exponent standard error is large.

## 4. Volume contraction

For `dx/dt = f(x)`, the divergence `div f = tr(∂f/∂x)` controls phase-space volume:

| System | div f | Default value |
|---|---|---|
| Lorenz | `-(σ + 1 + β)` | -13.67 |
| Rössler | `a + x - c` | state dependent |
| Chen | `c - a - b` | -10 |

The default Lorenz and Chen divergences are negative, so phase-space volume contracts. This is
consistent with a strange attractor. `test_chaotic_systems` checks the analytic expressions; the
spectrum sum independently reproduces the Lorenz value.

## 5. Embedding and correlation dimension

- `embed(series, m, τ)` is checked element by element; the row count is
  `n - (m-1)τ` and the coordinates are `(x[i], x[i+τ], ...)`.
- A straight line gives correlation dimension ≈ 1.08.
- A 32×32 planar grid gives ≈ 2.07.
- A Lorenz x series (m=4, τ=20) gives about 1.8–2.0.

`correlation_dimension` uses the Grassberger–Procaccia correlation integral and a log-radius fit
over `0.002 ≤ C(r) ≤ 0.1`.  The result exposes `points`, `stride`, and `embedding_dim` so
decimation is visible.  The Theiler window is expressed in original samples and converted to
decimated indices (`theiler / stride`).

This is a finite-sample estimate.  Uniform decimation to `max_points` can alias periodic or
strongly sampled data; increasing `max_points` or changing the delay can change the estimate.
Periodic series are therefore **not** guaranteed to raise, despite often having a misleading
dimension.  Degenerate clouds, too few distinct pairs, and singular fits do raise.

## 6. Permutation entropy

`permutation_entropy` normalizes the ordinal-pattern entropy by `log(order!)`:

- A monotonic ramp gives 0.
- A Lorenz x series (order 3, delay 1) gives about 0.45.
- A random series approaches 1.
- `test_permutation_entropy` checks the ramp, the Lorenz range, and invalid order rejection.
- NaN and infinity are rejected before sorting: the comparator would otherwise violate the strict
  weak ordering required by `std::sort` (`test_permutation_entropy_validation`).

Patterns are ranked with a factorial-number-system encoding; ties are broken by time index.

## 7. Time-dependent systems

`DuffingSystem` uses `deriv(t, x, dx)`. `AugmentedSystem` appends time to the state, so RK4, DP5,
Lyapunov, and ensemble code are reused without a second implementation.

- `test_time_systems` checks fixed-step and adaptive Duffing integration.
- Duffing with α=-1, β=1, δ=0.2, γ=0.3, ω=1 gives `λ ≈ 0.10`.
- `largest_lyapunov_time` perturbs only the state subspace, so the time coordinate does not bias
  the exponent; `test_time_header` uses a forced decay with true `λ = -1`.
- `integrate_adaptive_time` requires `t_end > t0` and integrates for `t_end - t0`.

## 8. Poincaré sections

`poincare_section` detects sign changes of `x[component] - level`, interpolates the crossing
position linearly between the bracketing states, and reports the crossing time
`(s + alpha) * dt`.  For the harmonic oscillator and `dt = 0.1`, the first rising zero of `x[1]` is
reported at `3.1416` against the exact `π = 3.141593`; the interpolation error is O(dt²).

Direction semantics: `+1` rising, `-1` falling, `0` both.  The returned state is advanced to the
last processed step even when `max_crossings` triggers an early break
(`test_poincare_timing`).  Exact hits are recorded on the step that reaches the level; an initial
state already on the level is not a crossing.

## 9. Determinism

- With the same kernel and compiler flags, thread count 1 vs. 8 and block 128 vs. 512 are bitwise
  identical.
- Generic `integrate_ensemble` is bitwise identical for 1 vs. 8 threads.
- The C ABI, Python, and CLI use the same kernels.
- Different compilers, math libraries, or CPU architectures are not guaranteed to be bitwise
  identical; chaos amplifies rounding differences.

## 10. Input contracts and exceptions

- Non-finite parameters, β≤0, dt≤0/NaN, null pointers, out-of-range component, or insufficient
  capacity raise an exception or return `CHAOSYS_ERR_ARGUMENT`.
- A non-finite integration result returns `CHAOSYS_ERR_NUMERIC`; caller buffers are left unchanged.
- `steps * dt` is checked against the floating-point range before integrating.
- `embed` capacity is in doubles (`count * dimension`); `poincare` capacity is in points.
- C++ exceptions never cross the C ABI.  A throwing C++ callback is caught by the adapter and
  returns `CHAOSYS_ERR_INTERNAL` (`test_exception_safety`).
- Worker-thread errors are captured and rethrown after all threads are joined; thread-creation
  failure joins already-created threads before propagating.
- Dimension/capacity multiplication is checked for overflow.
- The C ABI runtime dimension is capped at 64 (63 for time-dependent systems); the C++ template
  layer has no such limit.

## 11. Test evidence

| Layer | Count | Command |
|---|---|---|
| C++ | 145 | `make test` |
| C ABI | 130 | `make test-c` |
| C ABI exceptions | 11 | `make test-exceptions` |
| Time-header/Lyapunov | 1 suite | `make test-time-header` |
| Python | 19 | `python3 -m unittest discover -s python/tests` |
| ASan/UBSan | 145 | `make test-asan` |
| CMake CTest | 4 suites | `ctest --test-dir build` |

## 12. Non-claims

- This is not a formally verified solver and has no safety-critical certification.
- Lyapunov exponents and correlation dimension are statistical estimates, not exact analytic
  values.
- Performance numbers are measurements on specific hardware.

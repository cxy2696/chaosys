# Python binding

Build the C ABI first:

```bash
cd cpp && make shared
```

```python
from chaosys import (Params, State, rk4, adaptive, ensemble, lyapunov, poincare,
                     lyapunov_report, lyapunov_spectrum, to_csv)

params = Params(rho=28.0)
print(rk4((1, 1, 1), steps=1000, dt=0.01, params=params))
print(lyapunov((1, 1, 1), dt=0.005, params=params))

report = lyapunov_report((1, 1, 1), 0.005)   # exponent, stddev, stderr, converged
print(report.exponent, report.converged)
print(lyapunov_spectrum((1, 1, 1), 0.005))   # descending exponents

states = [rk4((1, 1, 1), step, 0.01) for step in (100, 200, 300)]
to_csv("traj.csv", states, header=("x", "y", "z"))
```

If the library is not in `cpp/build`, set `CHAOSYS_LIB=/path/to/libchaosys.dylib`.

`rk4_system(dim, deriv, x, steps, dt)` accepts a Python callback for a custom chaotic ODE.
Exceptions raised by the callback are captured and re-raised in Python after the C call; they never
cross the C boundary.

Generic analysis and ensemble functions are `lyapunov_system`, `lyapunov_report_system`,
`lyapunov_spectrum_system`, `poincare_system`, and `ensemble_system`. `ensemble_system` defaults to
one thread because a pure-Python callback is not generally thread-safe.

`lyapunov_report` returns a `LyapunovReport` with `exponent`, `stddev`, `stderr`, `converged`,
`segments`, and `segment_lambdas`.  `to_csv(path_or_stream, rows, header=None)` writes `State` rows
as `x,y,z` and returns the number of data rows.  Count-like arguments (steps, transient, segment,
segments, capacity, max_points) are validated as non-negative `size_t` values.

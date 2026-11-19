# Adding a system

Every chaotic system is an autonomous ODE `dx/dt = f(x)`. You can add one through C++, the C ABI,
or Python without touching the solvers.

## C++

Implement `real`, `dim`, and `deriv(x, dx) noexcept`. `validate()` is optional but recommended.

```cpp
#include <cmath>
#include <stdexcept>

#include <chaosys.hpp>
#include <chaosys/analysis.hpp>

struct LuSystem {
  using real = double;
  static constexpr int dim = 3;
  double a = 36.0, b = 3.0, c = 20.0;

  void deriv(const double* x, double* dx) const noexcept {
    dx[0] = a * (x[1] - x[0]);
    dx[1] = -x[0] * x[2] + c * x[1];
    dx[2] = x[0] * x[1] - b * x[2];
  }

  void validate() const {
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) {
      throw std::invalid_argument("LuSystem: non-finite parameter");
    }
  }
};

LuSystem lu;
double x[3] = {1, 1, 1};
chaosys::integrate<chaosys::Rk4Method>(lu, x, 10000, 0.01);
auto lam = chaosys::largest_lyapunov(lu, x, 0.01, 2000, 50, 200);
```

The same type works with `integrate_adaptive`, `poincare_section`, `EnsembleN`,
`integrate_ensemble`, `embed`, and `correlation_dimension`.

The `OdeSystem` concept does not require an initial state.  A system may optionally
provide

```cpp
static constexpr std::array<real, dim> initial_state() noexcept;
```

which satisfies the `HasInitialState` concept; all built-in systems do.  This is a
documented convention used by examples, not a requirement of the integrators.

To export results, include `chaosys/csv.hpp` and use `write_csv` for `State`
trajectories, `SectionPoint` lists, scalar series, or numeric tables:

```cpp
#include <chaosys/csv.hpp>
#include <fstream>
std::ofstream out("traj.csv");
chaosys::write_csv(out, states);
```

`largest_lyapunov_report` returns per-segment estimates and convergence statistics;
`lyapunov_spectrum` returns the full descending spectrum.  Both accept the same
`transient_steps / segment_steps / segments / d0` arguments as `largest_lyapunov`.

## Time-dependent systems

For `dx/dt = f(t, x)`, implement `deriv(t, x, dx)` and use `integrate_time`,
`integrate_adaptive_time`, or `largest_lyapunov_time`. The adapter appends time to the state, so
the existing solvers are reused.

```cpp
#include <chaosys/systems.hpp>
#include <chaosys/time_system.hpp>

chaosys::DuffingSystem duffing{};
double x[2] = {1, 1};
chaosys::integrate_time<chaosys::Rk4Method>(duffing, x, 0.0, 10000, 0.01);
auto lam = chaosys::largest_lyapunov_time(duffing, x, 0.0, 0.01, 1000, 50, 100);
```

In the C ABI, fill a `chaosys_time_system` and call `chaosys_rk4_time_system`,
`chaosys_adaptive_time_system`, or `chaosys_lyapunov_time_system`. Python uses
`rk4_time_system`, `adaptive_time_system`, and `lyapunov_time_system`. `t0` is the initial
absolute time; adaptive `t_end` is also absolute.

## C ABI

```c
static void lu(const double* x, double* dx, void* user) {
  (void)user;
  dx[0] = 36.0 * (x[1] - x[0]);
  dx[1] = -x[0] * x[2] + 20.0 * x[1];
  dx[2] = x[0] * x[1] - 3.0 * x[2];
}

chaosys_system sys = {.dim = 3, .deriv = lu, .user = NULL};
double x[3] = {1, 1, 1};
chaosys_adaptive_system(&sys, x, 100.0, 1e-9, 1e-11, NULL);

double lambda = 0;
chaosys_lyapunov_system(&sys, x, 0.01, 2000, 50, 200, &lambda);
```

`dim` is limited to 1–64. `chaosys_ensemble_system` uses flat `n * dim` arrays and requires a
thread-safe callback when `threads > 1`.

## Python

```python
import chaosys

def lu(x):
    return [36.0 * (x[1] - x[0]),
            -x[0] * x[2] + 20.0 * x[1],
            x[0] * x[1] - 3.0 * x[2]]

x, steps = chaosys.adaptive_system(3, lu, (1, 1, 1), 100.0)
lam = chaosys.lyapunov_system(3, lu, (1, 1, 1), 0.01)
```

Callbacks must not raise. Python's `ensemble_system` defaults to one thread.

## CLI

Built-in systems are registered in `apps/chaosys_cli.cpp` with a `std::variant`. A new built-in
needs a system struct, a `Kind` value, and option parsing; otherwise call the library directly.

```bash
chaosys_cli systems
chaosys_cli lyapunov --system rossler
```

## Checklist

- `deriv` is `noexcept` and does not allocate
- `validate` rejects non-finite parameters
- `dim` is correct; C ABI users stay at or below 64 (63 for time-dependent systems)
- Initial state and parameter defaults are documented
- At least one integration test and one Lyapunov/Poincaré test exist
- The callback is thread-safe if used by the threaded ensemble

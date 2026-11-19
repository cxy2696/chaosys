# Scope

chaosys handles chaotic systems defined by autonomous ODEs. It provides integration, ensemble
runs, Lyapunov exponents, Poincaré sections, Takens embedding, and correlation dimension. It is not
intended as a general-purpose ODE solver.

## Built-in systems

| System | Equations | Default parameters |
|---|---|---|
| Lorenz | `dx=σ(y-x)`, `dy=x(ρ-z)-y`, `dz=xy-βz` | 10, 28, 8/3 |
| Rössler | `dx=-y-z`, `dy=x+ay`, `dz=b+z(x-c)` | 0.2, 0.2, 5.7 |
| Chen | `dx=a(y-x)`, `dy=(c-a)x-xz+cy`, `dz=xy-bz` | 35, 3, 28 |
| Duffing (forced) | `dx=y`, `dy=-δy-αx-βx³+γcos(ωt)` | α=-1, β=1, δ=0.2, γ=0.3, ω=1 |

The autonomous systems share the RK4/DP5 solvers, ensemble support, analysis routines, embedding,
and parameter sweeps. Duffing is time-dependent and uses the augmented-time path.

## Custom systems

- C++: implement the `OdeSystem` concept for autonomous systems or `TimeOdeSystem` for
  `dx/dt = f(t, x)`; dimension and `float`/`double` are unrestricted.
- C ABI: fill a `chaosys_system` callback (dimension ≤64) or a `chaosys_time_system` callback
  (dimension ≤63).
- Python: pass a `deriv` or time-dependent `deriv(t, x)` callback.

The caller is responsible for meaningful equations, parameters, and initial conditions. The library
only checks numerical and boundary conditions.

## Suitable for

- Studying bifurcations, chaos, and sensitive dependence in Lorenz, Rössler, and Chen systems
- Teaching numerical integration and chaotic dynamics with CSV output
- Embedding a small chaotic model in a C, C++, or Python program
- Comparing integrators, SIMD, and multi-threaded implementations

## Out of scope

- General-purpose ODE solving; linear decay and harmonic oscillators are only numerical test
  fixtures
- Implicit or stiff solvers, CFD/physics engines, GPU/distributed computing
- Safety-critical certification in medical, aerospace, or nuclear applications
- Dynamic plugin loading; adding a built-in CLI system requires recompilation

## Interfaces

| Interface | Use |
|---|---|
| `chaosys_cli` | CSV output, Lyapunov, Poincaré, correlation dimension, parameter sweeps for the three autonomous systems; Duffing uses the C++/C/Python APIs |
| C ABI | C programs and other languages through FFI |
| C++ headers | Custom `OdeSystem`, maximum performance |
| Python binding | Scripted experiments and data analysis |

## Licensing

The repository is released under the MIT License. See [../LICENSE](../LICENSE).

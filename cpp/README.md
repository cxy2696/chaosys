# chaosys C++ library

A C++20 library for integrating and analyzing chaotic systems. The C++ namespace and C ABI prefix
are `chaosys`.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$HOME/.local"
```

This installs `libchaosys`, headers, `chaosys_cli`, the CMake package, and `chaosys.pc`.
Use `BUILD_SHARED_LIBS=OFF` for a static library and `CHAOSYS_BUILD_BENCH=ON` for the benchmark.

The Makefile is the development path:

```bash
make -j
make test-all          # C++, C ABI, ASan, CLI smoke
```

## Command line

```bash
./build/chaosys_cli systems
./build/chaosys_cli traj --steps 20000 --stride 10 --out traj.csv
./build/chaosys_cli lyapunov --system lorenz
./build/chaosys_cli lyapunov --system rossler
./build/chaosys_cli section --component 2 --level 20 --max 200
./build/chaosys_cli dimension --system lorenz
./build/chaosys_cli entropy --system lorenz
./build/chaosys_cli sweep --rho-min 20 --rho-max 45 --rho-step 0.2
./build/chaosys_cli ensemble --n 65536 --steps 2048 --threads 8
```

`sweep` is Lorenz-only and varies ρ. The CLI covers Lorenz, Rössler, and Chen. Duffing and other
time-dependent systems use the C++/C/Python APIs.

## Capabilities

- Built-in Lorenz, Rössler, Chen, and forced Duffing
- Custom autonomous systems (`OdeSystem`) and time-dependent systems (`TimeOdeSystem`)
- Fixed-step Euler/Heun/RK4 and adaptive DP5
- Single trajectories, ensembles, Lyapunov exponents, Poincaré sections, embedding, correlation
  dimension, and permutation entropy
- SoA/NEON/threaded Lorenz ensembles; generic scalar paths elsewhere

There are 145 C++, 130 C ABI, 11 exception-boundary, and 19 Python checks.  The C++ suite covers
Euler/Heun/RK4/DP5 order, Lyapunov reports and spectra, Poincaré timing, CSV output, and
overflow/NaN contracts.  Performance data is in [REPORT.md](REPORT.md).

## Layout

```text
include/chaosys.hpp       top-level API
include/chaosys_c.h       C ABI
include/chaosys/          ode, systems, time_system, analysis, chaos_metrics, ...
src/                      kernels, C ABI, benchmark
apps/chaosys_cli.cpp      command-line tool
tests/                    C++ and C ABI tests
CMakeLists.txt            CMake build and install
results/                  benchmark CSV data and validation records
```

## Documentation

- [../docs/API.md](../docs/API.md): interfaces
- [../docs/QUICKSTART.md](../docs/QUICKSTART.md): installation and language examples
- [../docs/EXTENDING.md](../docs/EXTENDING.md): adding systems
- [../docs/RIGOR.md](../docs/RIGOR.md): numerical properties and test evidence
- [REPORT.md](REPORT.md): performance and correctness

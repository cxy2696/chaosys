# chaosys

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A C++20 library for integrating and analyzing chaotic systems. It includes a C ABI, a command-line
tool, and a Python binding. The repository is source-only:

- Lorenz, Rössler, and Chen built in, plus the forced Duffing oscillator
- Custom autonomous and time-dependent systems through C++, C, or Python callbacks
- RK4/Heun/Euler, adaptive Dormand–Prince 5(4), and ensemble integration
- Largest Lyapunov exponent with per-segment report, full spectrum, Poincaré sections, Takens
  embedding, correlation dimension, permutation entropy
- CSV helpers for trajectories, sections, series, and tables
- SoA/NEON/threaded Lorenz ensembles; portable scalar paths for other systems

## Quick start

```bash
cd cpp
make -j
./build/chaosys_cli systems
./build/chaosys_cli lyapunov --system lorenz
make test-all
```

For CMake, installation, and language examples, see [docs/QUICKSTART.md](docs/QUICKSTART.md).

## Layout

| Path | Contents |
|---|---|
| [cpp/](cpp/) | C++ library, CLI, C ABI, tests, and benchmark |
| [docs/](docs/) | API, installation, extension, scope, and verification |
| [python/](python/) | Python binding and tests |

Inside `cpp/`, headers are under `include/`, implementations under `src/`, the CLI under `apps/`,
and tests under `tests/`. Documentation files are uppercase Markdown in `docs/`.

## Performance

On an Apple M3 Pro, 32,768 trajectories × 2,048 steps go from 59.3 M trajectory-steps/s with the
AoS scalar baseline to 1,441.9 M with SoA and 11 threads (24.3×). At equal accuracy, RK4 costs
about 1/113 of Euler for one trajectory. The measurement setup and compiler comparison are in
[cpp/REPORT.md](cpp/REPORT.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Build, install, and C/C++/Python/CLI usage |
| [docs/API.md](docs/API.md) | C ABI, C++, and Python interfaces |
| [docs/EXTENDING.md](docs/EXTENDING.md) | Adding autonomous or time-dependent systems |
| [docs/RIGOR.md](docs/RIGOR.md) | Numerical properties and test evidence |
| [docs/PURPOSE.md](docs/PURPOSE.md) | Scope and limitations |
| [cpp/REPORT.md](cpp/REPORT.md) | Performance and correctness report |

## Scope

The library targets chaotic systems only. It is not a general ODE solver and does not include
implicit/stiff integration, CFD, GPU or distributed computing, or safety-critical certification.
The CLI covers the three autonomous systems; Duffing is available through the C++/C/Python APIs.

## License

MIT. See [LICENSE](LICENSE).

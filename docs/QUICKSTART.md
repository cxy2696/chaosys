# Quick start

chaosys builds with CMake on macOS and Linux. The library namespace, C ABI prefix, and CMake
package are `chaosys`. See [PURPOSE.md](PURPOSE.md) for scope.

## Build and install

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$HOME/.local"
```

Installed layout:

```text
bin/chaosys_cli
include/chaosys.hpp
include/chaosys_c.h
include/chaosys/*.hpp
lib/libchaosys.{dylib,so}
lib/cmake/chaosys/chaosysConfig.cmake
lib/pkgconfig/chaosys.pc
```

Common CMake options:

| Option | Default | Effect |
|---|---|---|
| `BUILD_SHARED_LIBS` | `ON` | shared vs. static library |
| `CHAOSYS_BUILD_CLI` | `ON` | install `chaosys_cli` |
| `CHAOSYS_BUILD_BENCH` | `OFF` | build the benchmark |
| `CHAOSYS_NATIVE` | `OFF` | add `-mcpu/-march=native` |

## C

```c
#include <chaosys_c.h>

int main(void) {
  chaosys_state s = {1, 1, 1};
  chaosys_params p = chaosys_default_params();
  if (chaosys_rk4(&s, 1000, 0.01, p) != CHAOSYS_OK) return 1;
  return s.x < 0 ? 0 : 1;
}
```

```bash
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH"
cc app.c $(pkg-config --cflags --libs chaosys) \
  -Wl,-rpath,"$HOME/.local/lib" -o app
```

## C++

```bash
cmake -S . -B app-build -DCMAKE_PREFIX_PATH="$HOME/.local"
cmake --build app-build
```

```cmake
find_package(chaosys CONFIG REQUIRED)
target_link_libraries(app PRIVATE chaosys::chaosys)
```

```cpp
#include <chaosys.hpp>
#include <chaosys/analysis.hpp>
#include <chaosys/csv.hpp>
#include <chaosys/systems.hpp>
#include <fstream>

chaosys::LorenzSystem lorenz{};
double x[3] = {1, 1, 1};
auto lam = chaosys::largest_lyapunov(lorenz, x, 0.005, 2000, 50, 200);

chaosys::RosslerSystem rossler{};
double r[3] = {1, 1, 1};
chaosys::integrate<chaosys::Rk4Method>(rossler, r, 10000, 0.01);
```

`largest_lyapunov_report` returns the per-segment distribution; `lyapunov_spectrum` returns the
full descending spectrum.  `write_csv` is in `chaosys/csv.hpp`:

```cpp
auto report = chaosys::largest_lyapunov_report(lorenz, x, 0.005, 2000, 50, 200);
auto spectrum = chaosys::lyapunov_spectrum(lorenz, x, 0.005, 2000, 50, 200);
std::ofstream out("traj.csv");
chaosys::write_csv(out, std::vector<chaosys::State>{{1, 1, 1}});
```

## Python

```bash
pip install ./python
```

The module looks for `libchaosys` in `cpp/build`, the system loader path, or `CHAOSYS_LIB`.
For a non-standard install prefix:

```bash
export CHAOSYS_LIB="$HOME/.local/lib/libchaosys.dylib"   # .so on Linux
```

```python
from chaosys import lyapunov_report, lyapunov_spectrum, rk4, to_csv

print(rk4((1, 1, 1), 1000, 0.01))
print(lyapunov_report((1, 1, 1), 0.005))
print(lyapunov_spectrum((1, 1, 1), 0.005))
to_csv("traj.csv", [rk4((1, 1, 1), 100, 0.01)], header=("x", "y", "z"))
```

## CLI

```bash
chaosys_cli --version
chaosys_cli systems
chaosys_cli traj --steps 20000 --stride 10 --out traj.csv
chaosys_cli lyapunov --system lorenz
chaosys_cli lyapunov --system rossler
chaosys_cli dimension --system lorenz
chaosys_cli sweep --rho-min 20 --rho-max 45 --rho-step 0.2
```

See [API.md](API.md) for the full interface and [EXTENDING.md](EXTENDING.md) for custom systems.

// This translation unit deliberately includes only chaosys/time_system.hpp:
// the header must be self-contained, and its Lyapunov routine must not perturb
// the appended time lane.
#include "chaosys/time_system.hpp"

#include <cmath>
#include <cstdio>

namespace {

struct ForcedDecay {
  using real = double;
  static constexpr int dim = 1;
  void deriv(double t, const double* x, double* dx) const noexcept {
    dx[0] = -x[0] + std::cos(t);
  }
};

}  // namespace

int main() {
  double x[1] = {1.0};
  const double lambda = chaosys::largest_lyapunov_time(
      ForcedDecay{}, x, 0.0, 0.01, 1000, 50, 100);
  if (!(std::abs(lambda + 1.0) < 1e-3)) {
    std::fprintf(stderr, "FAIL forced-decay lambda %.9f\n", lambda);
    return 1;
  }
  std::printf("PASS  time-header self-contained, lambda %.6f\n", lambda);
  return 0;
}

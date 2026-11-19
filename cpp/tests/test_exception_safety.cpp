// C ABI exception-boundary tests.  A C++ callback may throw; the library must
// translate that into CHAOSYS_ERR_INTERNAL instead of calling std::terminate,
// and must leave caller buffers unchanged on failure.
#include "chaosys_c.h"

#include <cstdio>
#include <stdexcept>

namespace {

int checks = 0;
int failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    ++checks;                                                                \
    if (!(cond)) {                                                           \
      ++failures;                                                            \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
    }                                                                        \
  } while (0)

void throwing_deriv(const double* /*x*/, double* /*dx*/, void* /*user*/) {
  throw std::runtime_error("callback failure");
}

void throwing_time_deriv(double /*t*/, const double* /*x*/, double* /*dx*/,
                         void* /*user*/) {
  throw std::runtime_error("time callback failure");
}

}  // namespace

int main() {
  const chaosys_system system = {1, throwing_deriv, nullptr};
  double x[1] = {1.0};
  CHECK(chaosys_rk4_system(&system, x, 10, 0.01) == CHAOSYS_ERR_INTERNAL);
  CHECK(x[0] == 1.0);
  CHECK(chaosys_adaptive_system(&system, x, 1.0, 1e-8, 1e-10, nullptr) ==
        CHAOSYS_ERR_INTERNAL);
  CHECK(x[0] == 1.0);

  double lambda = 0.0;
  CHECK(chaosys_lyapunov_system(&system, x, 0.01, 10, 10, 10, &lambda) ==
        CHAOSYS_ERR_INTERNAL);

  double points[4];
  size_t count = 0;
  CHECK(chaosys_poincare_system(&system, x, 0.01, 0, 100, 0, 0.0, 1, points, 4,
                                &count) == CHAOSYS_ERR_INTERNAL);

  double batch[2] = {1.0, 2.0};
  CHECK(chaosys_ensemble_system(&system, batch, 2, 10, 0.01, 2) ==
        CHAOSYS_ERR_INTERNAL);
  CHECK(batch[0] == 1.0 && batch[1] == 2.0);

  const chaosys_time_system time_system = {1, throwing_time_deriv, nullptr};
  double time_x[1] = {1.0};
  CHECK(chaosys_rk4_time_system(&time_system, time_x, 0.0, 10, 0.01) ==
        CHAOSYS_ERR_INTERNAL);
  CHECK(time_x[0] == 1.0);
  CHECK(chaosys_adaptive_time_system(&time_system, time_x, 0.0, 1.0, 1e-8,
                                     1e-10, nullptr) ==
        CHAOSYS_ERR_INTERNAL);

  if (failures == 0) {
    std::printf("PASS  %d exception-boundary checks\n", checks);
    return 0;
  }
  std::printf("FAIL  %d/%d checks failed\n", failures, checks);
  return 1;
}

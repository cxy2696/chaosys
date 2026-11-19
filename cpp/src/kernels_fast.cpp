// The kernel bodies compiled with -ffast-math, in a separate translation unit
// so the effect of relaxed FP stays isolated from the IEEE kernels.
#include "chaosys.hpp"

#include "chaosys/driver.hpp"

#include <cstddef>
#include <stdexcept>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace chaosys {
namespace fastdetail {
#include "kernels_body.inc"
}  // namespace fastdetail
}  // namespace chaosys

extern "C" {

void chaosys_fast_rk4_scalar(chaosys::State* s, std::size_t steps, double dt,
                             chaosys::Params p) {
  if (s == nullptr) {
    throw std::invalid_argument("chaosys_fast_rk4_scalar: null state");
  }
  chaosys::detail::validate_run(steps, dt, p);
  chaosys::fastdetail::rk4_scalar_loop(s->x, s->y, s->z, steps, dt, p);
}

void chaosys_fast_rk4_soa_neon(double* x, double* y, double* z, std::size_t n,
                               std::size_t steps, double dt, chaosys::Params p) {
  chaosys::detail::validate_run(steps, dt, p);
  if (n == 0 || steps == 0) return;
  if (x == nullptr || y == nullptr || z == nullptr) {
    throw std::invalid_argument("chaosys_fast_rk4_soa_neon: null component");
  }
  chaosys::detail::parallel_blocks(
      n, chaosys::detail::kDefaultBlock, 1, [&](std::size_t b, std::size_t e) {
        chaosys::fastdetail::soa_neon_range(x, y, z, b, e, steps, dt, p);
      });
}

void chaosys_fast_rk4_soa_neon_threads(double* x, double* y, double* z,
                                       std::size_t n, std::size_t steps,
                                       double dt, int nthreads,
                                       chaosys::Params p) {
  chaosys::detail::validate_run(steps, dt, p);
  if (n == 0 || steps == 0) return;
  if (x == nullptr || y == nullptr || z == nullptr) {
    throw std::invalid_argument(
        "chaosys_fast_rk4_soa_neon_threads: null component");
  }
  chaosys::detail::soa_threaded_range(
      x, y, z, n, steps, dt, chaosys::detail::clamp_threads(nthreads),
      chaosys::detail::kDefaultBlock, chaosys::fastdetail::soa_neon_range, p);
}

}  // extern "C"

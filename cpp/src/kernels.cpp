// Optimized Lorenz kernels, compiled with -O3 -mcpu=native and strict IEEE
// semantics (no -ffast-math).  The bodies live in kernels_body.inc so the
// -ffast-math TU can instantiate the same source with different flags.
#include "chaosys.hpp"

#include "chaosys/driver.hpp"

#include <cstdint>
#include <stdexcept>
#include <thread>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace chaosys {
namespace detail {
#include "kernels_body.inc"

void validate_run(std::size_t steps, double dt, const Params& p) {
  validate(p);
  validate_dt(dt);
  validate_horizon(steps, dt);
}
}  // namespace detail

void euler_scalar(State& s, std::size_t steps, double dt, Params p) {
  detail::validate_run(steps, dt, p);
  detail::euler_loop(&s.x, &s.y, &s.z, 1, steps, dt, p);
}

void rk4_scalar(State& s, std::size_t steps, double dt, Params p) {
  detail::validate_run(steps, dt, p);
  detail::rk4_scalar_loop(s.x, s.y, s.z, steps, dt, p);
}

void rk4_scalar_strided(State& s, std::size_t steps, double dt, State* out,
                        std::size_t stride, Params p) {
  detail::validate_run(steps, dt, p);
  if (stride == 0) {
    throw std::invalid_argument("chaosys::rk4_scalar_strided: stride must be > 0");
  }
  if (out == nullptr) {
    throw std::invalid_argument("chaosys::rk4_scalar_strided: null output buffer");
  }
  detail::rk4_scalar_strided_loop(s.x, s.y, s.z, steps, dt, out, stride, p);
}

void init_ensemble(Ensemble& e, std::size_t n, std::uint64_t seed) {
  detail::init_ensemble_impl(e, n, seed);
}

void rk4_soa_scalar(Ensemble& e, std::size_t steps, double dt, Params p) {
  detail::validate_run(steps, dt, p);
  if (e.n == 0 || steps == 0) return;
  detail::parallel_blocks(e.n, detail::kDefaultBlock, 1,
                          [&](std::size_t b, std::size_t en) {
                            detail::soa_scalar_range(e.px(), e.py(), e.pz(), b,
                                                     en, steps, dt, p);
                          });
}

void rk4_soa_neon(Ensemble& e, std::size_t steps, double dt, Params p) {
  rk4_soa_neon_blocked(e, steps, dt, detail::kDefaultBlock, p);
}

void rk4_soa_neon_blocked(Ensemble& e, std::size_t steps, double dt,
                          std::size_t block, Params p) {
  detail::validate_run(steps, dt, p);
  if (e.n == 0 || steps == 0) return;
  if (block == 0) block = e.n;
  detail::parallel_blocks(e.n, block, 1,
                          [&](std::size_t b, std::size_t en) {
                            detail::soa_neon_range(e.px(), e.py(), e.pz(), b, en,
                                                   steps, dt, p);
                          });
}

void rk4_soa_scalar_threads(Ensemble& e, std::size_t steps, double dt,
                            int nthreads, Params p) {
  detail::validate_run(steps, dt, p);
  if (e.n == 0 || steps == 0) return;
  detail::soa_threaded_range(e.px(), e.py(), e.pz(), e.n, steps, dt,
                             detail::clamp_threads(nthreads),
                             detail::kDefaultBlock, detail::soa_scalar_range, p);
}

void rk4_soa_neon_threads(Ensemble& e, std::size_t steps, double dt,
                          int nthreads, Params p) {
  detail::validate_run(steps, dt, p);
  if (e.n == 0 || steps == 0) return;
  detail::soa_threaded_range(e.px(), e.py(), e.pz(), e.n, steps, dt,
                             detail::clamp_threads(nthreads),
                             detail::kDefaultBlock, detail::soa_neon_range, p);
}

State max_diff(const Ensemble& a, const Ensemble& b) {
  if (a.n != b.n) {
    throw std::invalid_argument("chaosys::max_diff: ensemble sizes differ");
  }
  return detail::max_diff_impl(a, b);
}

int max_threads() {
  const unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1 : static_cast<int>(hw);
}

}  // namespace chaosys

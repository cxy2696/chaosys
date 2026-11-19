// Low-level driver helpers for the specialised Lorenz kernels: run validation,
// thread clamping and the shared block scheduler live here so the IEEE and
// -ffast-math translation units do not duplicate them.
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "ode.hpp"
#include "params.hpp"
#include "parallel.hpp"

namespace chaosys::detail {

inline constexpr std::size_t kDefaultBlock = 512;

// Defined in kernels.cpp (strict IEEE translation unit) and called from both
// kernels.cpp and the -ffast-math TU.  Keeping the body out of the fast TU
// prevents -ffast-math from optimising the std::isfinite checks away.
void validate_run(std::size_t steps, double dt, const Params& p);

template <typename Kernel>
inline void soa_threaded_range(double* __restrict x, double* __restrict y,
                               double* __restrict z, std::size_t n,
                               std::size_t steps, double dt, int nthreads,
                               std::size_t block, Kernel&& kernel,
                               const Params& p) {
  parallel_blocks(n, block, nthreads, [&](std::size_t b, std::size_t e) {
    kernel(x, y, z, b, e, steps, dt, p);
  });
}

}  // namespace chaosys::detail

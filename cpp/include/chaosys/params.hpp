// Runtime Lorenz parameters.
//
// The original kernels baked sigma/rho/beta in as constexpr values, which made
// parameter sweeps a recompile.  They are now plain runtime values; the hot
// loops broadcast them once per block.
#pragma once

#include <cmath>
#include <stdexcept>

namespace chaosys {

struct Params {
  double sigma = 10.0;
  double rho = 28.0;
  double beta = 8.0 / 3.0;
};

inline void validate(const Params& p) {
  if (!std::isfinite(p.sigma) || !std::isfinite(p.rho) || !std::isfinite(p.beta)) {
    throw std::invalid_argument("chaosys::Params: sigma, rho and beta must be finite");
  }
  if (!(p.beta > 0.0)) {
    throw std::invalid_argument("chaosys::Params: beta must be > 0");
  }
}

}  // namespace chaosys

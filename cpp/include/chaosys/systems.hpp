// Built-in chaotic vector fields beyond Lorenz.
// Rössler and Chen satisfy the same OdeSystem concept as LorenzSystem, so they
// can use integrate / integrate_adaptive / largest_lyapunov / poincare_section
// and the generic ensemble driver without any special casing.
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

#include "ode.hpp"
#include "time_system.hpp"

namespace chaosys {

struct RosslerParams {
  double a = 0.2;
  double b = 0.2;
  double c = 5.7;
};

struct RosslerSystem {
  using real = double;
  static constexpr int dim = 3;
  static constexpr const char* kName = "rossler";

  RosslerParams p{};

  constexpr RosslerSystem() noexcept = default;
  constexpr explicit RosslerSystem(RosslerParams params) noexcept : p(params) {}

  void deriv(const double* x, double* dx) const noexcept {
    dx[0] = -x[1] - x[2];
    dx[1] = x[0] + p.a * x[1];
    dx[2] = p.b + x[2] * (x[0] - p.c);
  }

  // div f = a + x - c (state dependent).
  double divergence(const double* x) const noexcept {
    return p.a + x[0] - p.c;
  }

  void validate() const {
    if (!std::isfinite(p.a) || !std::isfinite(p.b) || !std::isfinite(p.c)) {
      throw std::invalid_argument("chaosys::RosslerSystem: non-finite parameter");
    }
  }

  static constexpr std::array<double, 3> initial_state() noexcept {
    return {1.0, 1.0, 1.0};
  }
};

struct ChenParams {
  double a = 35.0;
  double b = 3.0;
  double c = 28.0;
};

struct ChenSystem {
  using real = double;
  static constexpr int dim = 3;
  static constexpr const char* kName = "chen";

  ChenParams p{};

  constexpr ChenSystem() noexcept = default;
  constexpr explicit ChenSystem(ChenParams params) noexcept : p(params) {}

  void deriv(const double* x, double* dx) const noexcept {
    dx[0] = p.a * (x[1] - x[0]);
    dx[1] = (p.c - p.a) * x[0] - x[0] * x[2] + p.c * x[1];
    dx[2] = x[0] * x[1] - p.b * x[2];
  }

  // div f = c - a - b (constant).
  double divergence(const double* /*x*/) const noexcept {
    return p.c - p.a - p.b;
  }

  void validate() const {
    if (!std::isfinite(p.a) || !std::isfinite(p.b) || !std::isfinite(p.c)) {
      throw std::invalid_argument("chaosys::ChenSystem: non-finite parameter");
    }
  }

  static constexpr std::array<double, 3> initial_state() noexcept {
    return {1.0, 1.0, 1.0};
  }
};

static_assert(OdeSystem<RosslerSystem>);
static_assert(OdeSystem<ChenSystem>);
static_assert(HasInitialState<RosslerSystem>);
static_assert(HasInitialState<ChenSystem>);

struct DuffingParams {
  double alpha = -1.0;
  double beta = 1.0;
  double delta = 0.2;
  double gamma = 0.3;
  double omega = 1.0;
};

struct DuffingSystem {
  using real = double;
  static constexpr int dim = 2;
  static constexpr const char* kName = "duffing";

  DuffingParams p{};

  constexpr DuffingSystem() noexcept = default;
  constexpr explicit DuffingSystem(DuffingParams params) noexcept : p(params) {}

  void deriv(double t, const double* x, double* dx) const noexcept {
    dx[0] = x[1];
    dx[1] = -p.delta * x[1] - p.alpha * x[0] - p.beta * x[0] * x[0] * x[0] +
            p.gamma * std::cos(p.omega * t);
  }

  void validate() const {
    if (!std::isfinite(p.alpha) || !std::isfinite(p.beta) ||
        !std::isfinite(p.delta) || !std::isfinite(p.gamma) ||
        !std::isfinite(p.omega)) {
      throw std::invalid_argument("chaosys::DuffingSystem: non-finite parameter");
    }
  }

  static constexpr std::array<double, 2> initial_state() noexcept {
    return {1.0, 1.0};
  }
};

static_assert(TimeOdeSystem<DuffingSystem>);
static_assert(HasInitialState<DuffingSystem>);

}  // namespace chaosys

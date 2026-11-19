// Non-autonomous (time-dependent) chaotic systems.
//
// A TimeOdeSystem writes dx/dt = f(t, x).  The adapter below appends time to the
// state so the existing autonomous solvers and analysis routines can be reused
// without duplicating RK4/DP5 or Lyapunov logic.
#pragma once

#include <array>
#include <concepts>
#include <cstddef>

#include "analysis.hpp"
#include "ode.hpp"

namespace chaosys {

template <class S>
concept TimeOdeSystem = requires(const S& sys, typename S::real t,
                                 const typename S::real* x,
                                 typename S::real* dx) {
  typename S::real;
  requires std::floating_point<typename S::real>;
  requires(S::dim > 0);
  { S::dim } -> std::convertible_to<int>;
  { sys.deriv(t, x, dx) } noexcept -> std::same_as<void>;
};

template <TimeOdeSystem Sys>
struct AugmentedSystem {
  using real = typename Sys::real;
  static constexpr int dim = Sys::dim + 1;

  Sys sys;

  constexpr explicit AugmentedSystem(Sys s) noexcept : sys(s) {}

  void deriv(const real* x, real* dx) const noexcept {
    const real t = x[Sys::dim];
    sys.deriv(t, x, dx);
    dx[Sys::dim] = real(1);
  }

  void validate() const {
    if constexpr (requires { sys.validate(); }) sys.validate();
  }
};

// Integrate from t0 for `steps` fixed steps.
template <class Method = Rk4Method, TimeOdeSystem Sys>
void integrate_time(const Sys& sys, typename Sys::real* x, typename Sys::real t0,
                    std::size_t steps, typename Sys::real dt) {
  using T = typename Sys::real;
  std::array<T, static_cast<std::size_t>(Sys::dim) + 1> state{};
  for (int i = 0; i < Sys::dim; ++i) state[static_cast<std::size_t>(i)] = x[i];
  state[static_cast<std::size_t>(Sys::dim)] = t0;
  integrate<Method>(AugmentedSystem<Sys>{sys}, state.data(), steps, dt);
  for (int i = 0; i < Sys::dim; ++i) x[i] = state[static_cast<std::size_t>(i)];
}

template <class Method = DormandPrinceMethod, TimeOdeSystem Sys>
AdaptiveReport<AugmentedSystem<Sys>> integrate_adaptive_time(
    const Sys& sys, typename Sys::real* x, typename Sys::real t0,
    typename Sys::real t_end, typename Sys::real rtol = 1e-9,
    typename Sys::real atol = 1e-11) {
  using T = typename Sys::real;
  std::array<T, static_cast<std::size_t>(Sys::dim) + 1> state{};
  for (int i = 0; i < Sys::dim; ++i) state[static_cast<std::size_t>(i)] = x[i];
  state[static_cast<std::size_t>(Sys::dim)] = t0;
  if (!(t_end > t0)) {
    throw std::invalid_argument(
        "chaosys::integrate_adaptive_time: t_end must be greater than t0");
  }
  const auto report = integrate_adaptive<Method>(
      AugmentedSystem<Sys>{sys}, state.data(), t_end - t0, rtol, atol);
  for (int i = 0; i < Sys::dim; ++i) x[i] = state[static_cast<std::size_t>(i)];
  return report;
}

template <TimeOdeSystem Sys>
typename Sys::real largest_lyapunov_time(
    const Sys& sys, const typename Sys::real* x0, typename Sys::real t0,
    typename Sys::real dt, std::size_t transient_steps,
    std::size_t segment_steps, std::size_t segments,
    typename Sys::real d0 = 1e-8) {
  using T = typename Sys::real;
  constexpr std::size_t N = static_cast<std::size_t>(Sys::dim);
  std::array<T, N + 1> state{};
  for (int i = 0; i < Sys::dim; ++i) state[static_cast<std::size_t>(i)] = x0[i];
  state[N] = t0;

  // Perturb only the state subspace: the appended time lane advances with
  // derivative 1, so a perturbation there would add a non-decaying offset to
  // the two-trajectory distance and bias lambda towards zero.
  const auto state_direction = detail::perturbation_direction<T, N>();
  std::array<T, N + 1> direction{};
  for (std::size_t i = 0; i < N; ++i) direction[i] = state_direction[i];
  direction[N] = T(0);

  const auto report = largest_lyapunov_report(
      AugmentedSystem<Sys>{sys}, state.data(), dt, transient_steps,
      segment_steps, segments, d0, T(0.1), &direction);
  return report.lambda;
}

}  // namespace chaosys

// Reusable analysis routines built on the generic integrators:
//   - largest Lyapunov exponent with per-segment statistics
//   - full Lyapunov spectrum by Benettin + modified Gram-Schmidt
//   - Poincare section by linear interpolation of a sign change
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ode.hpp"

namespace chaosys {

template <class Sys>
struct SectionPoint {
  using real = typename Sys::real;
  real t = real(0);
  std::array<real, static_cast<std::size_t>(Sys::dim)> x{};
};

// Result of a largest-Lyapunov computation.  `lambda` is the Benettin
// estimate; `segment_lambdas` holds the independent per-segment estimates in
// 1/time units.  The standard deviation and standard error of those estimates
// quantify the finite-time convergence.  `converged` is an advisory flag
// (segments >= 8 and stderr <= convergence_tol * max(1, |lambda|), with a
// default convergence_tol of 0.1).
template <class Sys>
struct LyapunovReport {
  using real = typename Sys::real;
  real lambda = real(0);
  std::vector<real> segment_lambdas;
  real stddev = real(0);
  real stderr = real(0);
  bool converged = false;
  std::size_t segments = 0;
  std::size_t segment_steps = 0;
  real dt = real(0);
};

// Full Lyapunov spectrum (descending order) from Benettin's algorithm.  The
// sum of the exponents equals the average phase-space divergence
// <div f>; for the Lorenz system it is -(sigma + 1 + beta).
template <class Sys>
struct LyapunovSpectrum {
  using real = typename Sys::real;
  std::vector<real> exponents;
  real sum = real(0);
  std::size_t segments = 0;
  std::size_t segment_steps = 0;
  real dt = real(0);
};

namespace detail {

inline std::uint64_t splitmix64_step(std::uint64_t& state) noexcept {
  std::uint64_t z = (state += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// Deterministic, seeded unit vector used as the initial Benettin
// perturbation.  Perturbing only x[0] is not general: in a system whose
// unstable manifold is orthogonal to that axis the measured exponent would be
// wrong.  The seed is fixed so results are reproducible.
template <class T, std::size_t N>
std::array<T, N> perturbation_direction() {
  std::array<T, N> w{};
  std::uint64_t state = 0x243f6a8885a308d3ull;
  T norm2 = T(0);
  constexpr double kInv53 = 1.0 / 9007199254740992.0;
  for (std::size_t i = 0; i < N; ++i) {
    const double u = static_cast<double>(splitmix64_step(state) >> 11) * kInv53;
    w[i] = static_cast<T>(2.0 * u - 1.0);
    norm2 += w[i] * w[i];
  }
  const T norm = std::sqrt(norm2);
  if (!(norm > T(0)) || !std::isfinite(norm)) {
    const T fallback = T(1) / std::sqrt(static_cast<T>(N));
    w.fill(fallback);
    return w;
  }
  const T inv = T(1) / norm;
  for (T& value : w) value *= inv;
  return w;
}

template <class Sys>
void validate_lyapunov_arguments(const Sys& sys,
                                 const typename Sys::real* x0,
                                 typename Sys::real dt,
                                 std::size_t segment_steps,
                                 std::size_t segments,
                                 typename Sys::real d0) {
  using T = typename Sys::real;
  if constexpr (requires { sys.validate(); }) sys.validate();
  validate_state<Sys>(x0);
  validate_dt(dt);
  if (segment_steps == 0 || segments == 0) {
    throw std::invalid_argument(
        "chaosys::largest_lyapunov: segment_steps and segments must be > 0");
  }
  if (!(d0 > T(0)) || !std::isfinite(d0)) {
    throw std::invalid_argument(
        "chaosys::largest_lyapunov: d0 must be finite and > 0");
  }
  if (segment_steps > std::numeric_limits<std::size_t>::max() / segments) {
    throw std::invalid_argument(
        "chaosys::largest_lyapunov: segment_steps * segments overflows");
  }
  const long double span = static_cast<long double>(segment_steps) *
                           static_cast<long double>(segments) *
                           static_cast<long double>(dt);
  if (!(span > 0.0L) || !std::isfinite(span) ||
      span > static_cast<long double>(std::numeric_limits<T>::max())) {
    throw std::invalid_argument(
        "chaosys::largest_lyapunov: time span overflows");
  }
}

template <OdeSystem Sys>
LyapunovSpectrum<Sys> lyapunov_spectrum_active(
    const Sys& sys, const typename Sys::real* x0, typename Sys::real dt,
    std::size_t transient_steps, std::size_t segment_steps,
    std::size_t segments, typename Sys::real d0, std::size_t active_dim) {
  using T = typename Sys::real;
  constexpr std::size_t N = static_cast<std::size_t>(Sys::dim);
  if (active_dim == 0 || active_dim > N) {
    throw std::invalid_argument(
        "chaosys::lyapunov_spectrum: active dimension out of range");
  }

  validate_lyapunov_arguments(sys, x0, dt, segment_steps, segments, d0);

  std::array<T, N> a{};
  for (std::size_t i = 0; i < active_dim; ++i) a[i] = x0[i];
  for (std::size_t s = 0; s < transient_steps; ++s) {
    Rk4Method::step(sys, a.data(), dt);
  }

  T state_scale = T(1);
  for (std::size_t i = 0; i < active_dim; ++i) {
    state_scale = std::max(state_scale, std::abs(a[i]));
  }
  const T d0_eff = d0 * state_scale;
  if (!(d0_eff > T(0)) || !std::isfinite(d0_eff)) {
    throw std::invalid_argument(
        "chaosys::lyapunov_spectrum: d0 is not representable at this state scale");
  }

  std::vector<std::array<T, N>> b(active_dim);
  std::vector<std::vector<T>> q(active_dim, std::vector<T>(active_dim, T(0)));
  for (std::size_t i = 0; i < active_dim; ++i) {
    q[i][i] = T(1);
    b[i] = a;
    b[i][i] += d0_eff;
  }

  std::vector<T> sum_log(active_dim, T(0));
  std::vector<T> v(active_dim, T(0));
  const std::size_t total_steps = segment_steps * segments;
  // QR after every step keeps the smallest singular direction above the
  // state's floating-point resolution; long segments otherwise lose the
  // contracting exponents to cancellation.
  for (std::size_t s = 0; s < total_steps; ++s) {
    Rk4Method::step(sys, a.data(), dt);
    for (std::size_t i = 0; i < active_dim; ++i) {
      Rk4Method::step(sys, b[i].data(), dt);
    }

    for (std::size_t i = 0; i < active_dim; ++i) {
      for (std::size_t k = 0; k < active_dim; ++k) v[k] = b[i][k] - a[k];
      for (std::size_t j = 0; j < i; ++j) {
        T r = T(0);
        for (std::size_t k = 0; k < active_dim; ++k) r += v[k] * q[j][k];
        for (std::size_t k = 0; k < active_dim; ++k) v[k] -= r * q[j][k];
      }
      T norm2 = T(0);
      for (std::size_t k = 0; k < active_dim; ++k) norm2 += v[k] * v[k];
      const T norm = std::sqrt(norm2);
      if (!(norm > T(0)) || !std::isfinite(norm)) {
        throw std::runtime_error(
            "chaosys::lyapunov_spectrum: tangent collapsed");
      }
      sum_log[i] += std::log(norm / d0_eff);
      const T inv = T(1) / norm;
      for (std::size_t k = 0; k < active_dim; ++k) q[i][k] = v[k] * inv;
    }
    for (std::size_t i = 0; i < active_dim; ++i) {
      b[i] = a;
      for (std::size_t k = 0; k < active_dim; ++k) b[i][k] += d0_eff * q[i][k];
    }
  }

  const T span = static_cast<T>(static_cast<long double>(total_steps) *
                                static_cast<long double>(dt));
  LyapunovSpectrum<Sys> result;
  result.segments = segments;
  result.segment_steps = segment_steps;
  result.dt = dt;
  result.exponents.resize(active_dim);
  for (std::size_t i = 0; i < active_dim; ++i) {
    const T value = sum_log[i] / span;
    if (!std::isfinite(value)) {
      throw std::runtime_error(
          "chaosys::lyapunov_spectrum: non-finite exponent");
    }
    result.exponents[i] = value;
  }
  std::sort(result.exponents.begin(), result.exponents.end(),
            [](T lhs, T rhs) { return lhs > rhs; });
  result.sum = T(0);
  for (const T value : result.exponents) result.sum += value;
  return result;
}

}  // namespace detail

// Largest Lyapunov exponent of `sys` from x0, in 1/time units, plus the
// per-segment estimates and convergence statistics.  transient_steps are
// discarded; the perturbation is renormalised back to d0 after every step,
// which keeps the two trajectories close enough for linear growth.  The
// per-segment estimates are reported for diagnostics; `converged` is advisory
// (segments >= 8 and stderr <= convergence_tol * max(1, |lambda|)).
template <OdeSystem Sys>
LyapunovReport<Sys> largest_lyapunov_report(
    const Sys& sys, const typename Sys::real* x0, typename Sys::real dt,
    std::size_t transient_steps, std::size_t segment_steps,
    std::size_t segments, typename Sys::real d0 = 1e-8,
    typename Sys::real convergence_tol = 0.1,
    const std::array<typename Sys::real, static_cast<std::size_t>(Sys::dim)>*
        initial_direction = nullptr) {
  using T = typename Sys::real;
  constexpr std::size_t N = static_cast<std::size_t>(Sys::dim);

  detail::validate_lyapunov_arguments(sys, x0, dt, segment_steps, segments,
                                      d0);
  if (!(convergence_tol > T(0)) || !std::isfinite(convergence_tol)) {
    throw std::invalid_argument(
        "chaosys::largest_lyapunov: convergence_tol must be finite and > 0");
  }

  std::array<T, N> a{}, b{};
  for (std::size_t i = 0; i < N; ++i) a[i] = x0[i];

  // Burn in on the reference trajectory, then perturb the post-transient
  // state.  Perturbing before the transient would count the transient decay
  // in the first segment's logarithm.
  for (std::size_t s = 0; s < transient_steps; ++s) {
    Rk4Method::step(sys, a.data(), dt);
  }

  std::array<T, N> direction{};
  if (initial_direction != nullptr) {
    direction = *initial_direction;
    T norm2 = T(0);
    for (std::size_t i = 0; i < N; ++i) {
      if (!std::isfinite(direction[i])) {
        throw std::invalid_argument(
            "chaosys::largest_lyapunov: initial_direction must be finite");
      }
      norm2 += direction[i] * direction[i];
    }
    const T norm = std::sqrt(norm2);
    if (!(norm > T(0)) || !std::isfinite(norm)) {
      throw std::invalid_argument(
          "chaosys::largest_lyapunov: initial_direction must be nonzero");
    }
    const T inv = T(1) / norm;
    for (T& value : direction) value *= inv;
  } else {
    direction = detail::perturbation_direction<T, N>();
  }

  // Scale the perturbation with the state magnitude so that d0 remains
  // representable when |x| >> 1.  The exponent is invariant under this
  // rescaling because the renormalisation uses the same d0_eff.
  T state_scale = T(1);
  for (std::size_t i = 0; i < N; ++i) {
    if (direction[i] != T(0)) {
      state_scale = std::max(state_scale, std::abs(a[i]));
    }
  }
  const T d0_eff = d0 * state_scale;
  if (!(d0_eff > T(0)) || !std::isfinite(d0_eff)) {
    throw std::invalid_argument(
        "chaosys::largest_lyapunov: d0 is not representable at this state scale");
  }
  for (std::size_t i = 0; i < N; ++i) b[i] = a[i] + d0_eff * direction[i];

  LyapunovReport<Sys> report;
  report.segments = segments;
  report.segment_steps = segment_steps;
  report.dt = dt;
  report.segment_lambdas.reserve(segments);

  const T segment_span = static_cast<T>(
      static_cast<long double>(segment_steps) * static_cast<long double>(dt));
  if (!(segment_span > T(0)) || !std::isfinite(segment_span)) {
    throw std::invalid_argument(
        "chaosys::largest_lyapunov: segment time span is not representable");
  }
  T sum = T(0);
  // Renormalise every step, not only at segment boundaries: it keeps the
  // perturbation far above the floating-point resolution of the state and
  // makes the per-segment statistics independent of segment length.
  for (std::size_t seg = 0; seg < segments; ++seg) {
    T segment_sum = T(0);
    for (std::size_t s = 0; s < segment_steps; ++s) {
      Rk4Method::step(sys, a.data(), dt);
      Rk4Method::step(sys, b.data(), dt);
      T dist2 = T(0);
      for (std::size_t i = 0; i < N; ++i) {
        const T d = b[i] - a[i];
        dist2 += d * d;
      }
      const T dist = std::sqrt(dist2);
      if (!(dist > T(0)) || !std::isfinite(dist)) {
        throw std::runtime_error(
            "chaosys::largest_lyapunov: perturbation collapsed");
      }
      const T growth = std::log(dist / d0_eff);
      segment_sum += growth;
      // Divide before multiplying: d0_eff / dist can overflow when the
      // perturbation is subnormal, while (b - a) / dist is bounded by 1.
      for (std::size_t i = 0; i < N; ++i) {
        b[i] = a[i] + ((b[i] - a[i]) / dist) * d0_eff;
      }
    }
    sum += segment_sum;
    report.segment_lambdas.push_back(segment_sum / segment_span);
  }

  const T span = static_cast<T>(static_cast<long double>(segment_steps) *
                                static_cast<long double>(segments) *
                                static_cast<long double>(dt));
  report.lambda = sum / span;
  if (!std::isfinite(report.lambda)) {
    throw std::runtime_error(
        "chaosys::largest_lyapunov: non-finite result");
  }

  if (segments > 1) {
    T mean = T(0);
    for (const T value : report.segment_lambdas) mean += value;
    mean /= static_cast<T>(segments);
    T sum_sq = T(0);
    for (const T value : report.segment_lambdas) {
      const T d = value - mean;
      sum_sq += d * d;
    }
    report.stddev = std::sqrt(sum_sq / static_cast<T>(segments - 1));
    report.stderr = report.stddev / std::sqrt(static_cast<T>(segments));
  }
  report.converged =
      segments >= 8 &&
      report.stderr <= convergence_tol * std::max(T(1), std::abs(report.lambda));
  return report;
}

// Largest Lyapunov exponent of `sys` from x0, in 1/time units.
template <OdeSystem Sys>
typename Sys::real largest_lyapunov(const Sys& sys,
                                    const typename Sys::real* x0,
                                    typename Sys::real dt,
                                    std::size_t transient_steps,
                                    std::size_t segment_steps,
                                    std::size_t segments,
                                    typename Sys::real d0 = 1e-8) {
  return largest_lyapunov_report(sys, x0, dt, transient_steps, segment_steps,
                                 segments, d0)
      .lambda;
}

// Full Lyapunov spectrum, descending, by the Benettin flow method:
//   * propagate a reference trajectory and N perturbed copies,
//   * after every step, apply modified Gram-Schmidt to the N difference
//     vectors, accumulate log(r_ii / d0_eff), and reset the copies to
//     x + d0_eff q_i.
// No Jacobian is required, so the routine works with any OdeSystem.
template <OdeSystem Sys>
LyapunovSpectrum<Sys> lyapunov_spectrum(
    const Sys& sys, const typename Sys::real* x0, typename Sys::real dt,
    std::size_t transient_steps, std::size_t segment_steps,
    std::size_t segments, typename Sys::real d0 = 1e-8) {
  return detail::lyapunov_spectrum_active(
      sys, x0, dt, transient_steps, segment_steps, segments, d0,
      static_cast<std::size_t>(Sys::dim));
}

// Poincare section of `x[component] == level`.  direction=+1 keeps rising
// crossings, -1 falling crossings, 0 both.  t is measured from the initial
// state and includes transient_steps.  The crossing point is linearly
// interpolated between the two bracketing states, so its position error is
// O(dt^2) for a smooth flow.  A trajectory that lands exactly on the section
// is recorded once, on the step that reaches the level.
template <OdeSystem Sys>
std::vector<SectionPoint<Sys>> poincare_section(
    const Sys& sys, typename Sys::real* x, typename Sys::real dt,
    std::size_t transient_steps, std::size_t max_steps,
    std::size_t max_crossings, int component, typename Sys::real level,
    int direction = 1) {
  using T = typename Sys::real;
  constexpr std::size_t N = static_cast<std::size_t>(Sys::dim);

  if constexpr (requires { sys.validate(); }) sys.validate();
  validate_state<Sys>(x);
  validate_dt(dt);
  if (component < 0 || component >= Sys::dim) {
    throw std::invalid_argument("chaosys::poincare_section: component out of range");
  }
  if (!std::isfinite(level)) {
    throw std::invalid_argument("chaosys::poincare_section: level must be finite");
  }
  if (max_steps == 0 || max_crossings == 0) {
    throw std::invalid_argument(
        "chaosys::poincare_section: max_steps and max_crossings must be > 0");
  }
  if (direction < -1 || direction > 1) {
    throw std::invalid_argument(
        "chaosys::poincare_section: direction must be -1, 0 or +1");
  }
  if (transient_steps >
      std::numeric_limits<std::size_t>::max() - max_steps) {
    throw std::invalid_argument(
        "chaosys::poincare_section: transient + steps overflows");
  }

  std::vector<SectionPoint<Sys>> out;
  out.reserve(max_crossings);

  const std::size_t axis = static_cast<std::size_t>(component);
  std::array<T, N> prev{}, cur{};
  for (std::size_t i = 0; i < N; ++i) prev[i] = x[i];

  auto gap = [&](const std::array<T, N>& v) { return v[axis] - level; };

  for (std::size_t s = 0; s < transient_steps + max_steps; ++s) {
    for (std::size_t i = 0; i < N; ++i) cur[i] = prev[i];
    Rk4Method::step(sys, cur.data(), dt);
    for (std::size_t i = 0; i < N; ++i) {
      if (!std::isfinite(cur[i])) {
        throw std::runtime_error(
            "chaosys::poincare_section: non-finite trajectory");
      }
    }

    if (s >= transient_steps) {
      const T g0 = gap(prev);
      const T g1 = gap(cur);
      const bool rising = g0 < T(0) && g1 >= T(0);
      const bool falling = g0 > T(0) && g1 <= T(0);
      const bool keep = (direction > 0 && rising) ||
                        (direction < 0 && falling) ||
                        (direction == 0 && (rising || falling));
      if (keep) {
        const T denom = g0 - g1;
        T alpha = T(0);
        if (denom != T(0)) alpha = g0 / denom;
        if (alpha < T(0)) alpha = T(0);
        if (alpha > T(1)) alpha = T(1);

        SectionPoint<Sys> p;
        p.t = (static_cast<T>(s) + alpha) * dt;
        for (std::size_t i = 0; i < N; ++i) {
          p.x[i] = prev[i] + alpha * (cur[i] - prev[i]);
        }
        p.x[axis] = level;
        out.push_back(p);
        if (out.size() >= max_crossings) {
          for (std::size_t i = 0; i < N; ++i) prev[i] = cur[i];
          break;
        }
      }
    }
    for (std::size_t i = 0; i < N; ++i) prev[i] = cur[i];
  }

  for (std::size_t i = 0; i < N; ++i) x[i] = prev[i];
  return out;
}

}  // namespace chaosys

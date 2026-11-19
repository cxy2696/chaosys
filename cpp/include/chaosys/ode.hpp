// Generic ODE layer: any system exposing `real`, `dim` and `deriv()` can be
// integrated with Euler / Heun / classical RK4, in float or double.
//
// The Lorenz ensemble kernels remain a specialised backend; this header is the
// portable path used for new systems and for validation.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "aligned.hpp"
#include "params.hpp"
#include "parallel.hpp"

namespace chaosys {

template <class S>
concept OdeSystem = requires(const S& sys, const typename S::real* x,
                             typename S::real* dx) {
  typename S::real;
  requires std::floating_point<typename S::real>;
  requires(S::dim > 0);
  { S::dim } -> std::convertible_to<int>;
  { sys.deriv(x, dx) } noexcept -> std::same_as<void>;
};

// Optional companion to OdeSystem: a system may expose a compile-time default
// initial state with exactly `real` components and `dim` entries.  The
// integrators do not require it; it is used by experiments and documentation
// to make a system self-contained.
template <class S>
concept HasInitialState = requires {
  typename S::real;
  requires(S::dim > 0);
  {
    S::initial_state()
  } -> std::same_as<
      std::array<typename S::real, static_cast<std::size_t>(S::dim)>>;
};

// ---------------------------------------------------------------------------
// Reference system
// ---------------------------------------------------------------------------
struct LorenzSystem {
  using real = double;
  static constexpr int dim = 3;

  Params p{};

  constexpr LorenzSystem() noexcept = default;
  constexpr explicit LorenzSystem(Params params) noexcept : p(params) {}

  void validate() const { chaosys::validate(p); }

  static constexpr std::array<double, 3> initial_state() noexcept {
    return {1.0, 1.0, 1.0};
  }

  void deriv(const double* __restrict x, double* __restrict dx) const noexcept {
    dx[0] = p.sigma * (x[1] - x[0]);
    dx[1] = x[0] * (p.rho - x[2]) - x[1];
    dx[2] = x[0] * x[1] - p.beta * x[2];
  }

  // Phase-space volume contraction: div f = -(sigma + 1 + beta).
  double divergence(const double* /*x*/) const noexcept {
    return -(p.sigma + 1.0 + p.beta);
  }
};

static_assert(OdeSystem<LorenzSystem>);
static_assert(HasInitialState<LorenzSystem>);

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
template <std::floating_point T>
inline void validate_dt(T dt) {
  if (!(dt > T(0)) || !std::isfinite(dt)) {
    throw std::invalid_argument("chaosys::integrate: dt must be finite and > 0");
  }
}

// Reject a horizon that cannot be represented in T.  Without this check a
// huge (steps, dt) pair silently produces an infinite time span.
template <std::floating_point T>
inline void validate_horizon(std::size_t steps, T dt) {
  if (steps == 0) return;
  const long double horizon =
      static_cast<long double>(steps) * static_cast<long double>(dt);
  if (!(horizon > 0.0L) || !std::isfinite(horizon) ||
      horizon > static_cast<long double>(std::numeric_limits<T>::max())) {
    throw std::invalid_argument(
        "chaosys::integrate: steps * dt overflows the floating-point range");
  }
}

template <OdeSystem Sys>
inline void validate_state(const typename Sys::real* x) {
  if (x == nullptr) {
    throw std::invalid_argument("chaosys::integrate: null state");
  }
  for (int i = 0; i < Sys::dim; ++i) {
    if (!std::isfinite(x[i])) {
      throw std::invalid_argument("chaosys::integrate: non-finite state");
    }
  }
}

// ---------------------------------------------------------------------------
// Steppers.  All are stateless policies; the arrays live on the stack.
// ---------------------------------------------------------------------------
struct EulerMethod {
  template <OdeSystem Sys>
  static void step(const Sys& sys, typename Sys::real* x,
                   typename Sys::real dt) noexcept {
    using T = typename Sys::real;
    constexpr std::size_t N = static_cast<std::size_t>(Sys::dim);
    std::array<T, N> k;
    sys.deriv(x, k.data());
    for (std::size_t i = 0; i < N; ++i) x[i] += dt * k[i];
  }
};

struct HeunMethod {
  template <OdeSystem Sys>
  static void step(const Sys& sys, typename Sys::real* x,
                   typename Sys::real dt) noexcept {
    using T = typename Sys::real;
    constexpr std::size_t N = static_cast<std::size_t>(Sys::dim);
    std::array<T, N> k1, k2, t;
    sys.deriv(x, k1.data());
    for (std::size_t i = 0; i < N; ++i) t[i] = x[i] + dt * k1[i];
    sys.deriv(t.data(), k2.data());
    const T half = dt / T(2);
    for (std::size_t i = 0; i < N; ++i) x[i] += half * (k1[i] + k2[i]);
  }
};

struct Rk4Method {
  template <OdeSystem Sys>
  static void step(const Sys& sys, typename Sys::real* x,
                   typename Sys::real dt) noexcept {
    using T = typename Sys::real;
    constexpr std::size_t N = static_cast<std::size_t>(Sys::dim);
    std::array<T, N> k1, k2, k3, k4, t;

    const T h2 = dt * T(0.5);
    const T h6 = dt / T(6);

    sys.deriv(x, k1.data());
    for (std::size_t i = 0; i < N; ++i) t[i] = x[i] + h2 * k1[i];
    sys.deriv(t.data(), k2.data());
    for (std::size_t i = 0; i < N; ++i) t[i] = x[i] + h2 * k2[i];
    sys.deriv(t.data(), k3.data());
    for (std::size_t i = 0; i < N; ++i) t[i] = x[i] + dt * k3[i];
    sys.deriv(t.data(), k4.data());

    for (std::size_t i = 0; i < N; ++i) {
      x[i] += h6 * (k1[i] + T(2) * k2[i] + T(2) * k3[i] + k4[i]);
    }
  }
};

// Dormand-Prince 5(4): one trial step with an embedded 4th-order error
// estimate.  Returns the scaled max error norm; the 5th-order solution is
// written to x5.  Used by integrate_adaptive and by the reference oracle.
struct DormandPrinceMethod {
  template <OdeSystem Sys>
  static typename Sys::real step(const Sys& sys, const typename Sys::real* x,
                                 typename Sys::real h, typename Sys::real* x5,
                                 typename Sys::real rtol,
                                 typename Sys::real atol) noexcept {
    using T = typename Sys::real;
    constexpr std::size_t N = static_cast<std::size_t>(Sys::dim);
    std::array<T, N> k1{}, k2{}, k3{}, k4{}, k5{}, k6{}, k7{};
    std::array<T, N> y5{}, y4{}, t{};

    sys.deriv(x, k1.data());
    for (std::size_t i = 0; i < N; ++i) t[i] = x[i] + h * (T(1) / T(5)) * k1[i];
    sys.deriv(t.data(), k2.data());

    for (std::size_t i = 0; i < N; ++i) {
      t[i] = x[i] + h * ((T(3) / T(40)) * k1[i] + (T(9) / T(40)) * k2[i]);
    }
    sys.deriv(t.data(), k3.data());

    for (std::size_t i = 0; i < N; ++i) {
      t[i] = x[i] + h * ((T(44) / T(45)) * k1[i] - (T(56) / T(15)) * k2[i] +
                         (T(32) / T(9)) * k3[i]);
    }
    sys.deriv(t.data(), k4.data());

    for (std::size_t i = 0; i < N; ++i) {
      t[i] = x[i] + h * ((T(19372) / T(6561)) * k1[i] -
                         (T(25360) / T(2187)) * k2[i] +
                         (T(64448) / T(6561)) * k3[i] -
                         (T(212) / T(729)) * k4[i]);
    }
    sys.deriv(t.data(), k5.data());

    for (std::size_t i = 0; i < N; ++i) {
      t[i] = x[i] + h * ((T(9017) / T(3168)) * k1[i] -
                         (T(355) / T(33)) * k2[i] +
                         (T(46732) / T(5247)) * k3[i] +
                         (T(49) / T(176)) * k4[i] -
                         (T(5103) / T(18656)) * k5[i]);
    }
    sys.deriv(t.data(), k6.data());

    for (std::size_t i = 0; i < N; ++i) {
      y5[i] = x[i] + h * ((T(35) / T(384)) * k1[i] +
                          (T(500) / T(1113)) * k3[i] +
                          (T(125) / T(192)) * k4[i] -
                          (T(2187) / T(6784)) * k5[i] +
                          (T(11) / T(84)) * k6[i]);
    }
    sys.deriv(y5.data(), k7.data());

    for (std::size_t i = 0; i < N; ++i) {
      y4[i] = x[i] + h * ((T(5179) / T(57600)) * k1[i] +
                          (T(7571) / T(16695)) * k3[i] +
                          (T(393) / T(640)) * k4[i] -
                          (T(92097) / T(339200)) * k5[i] +
                          (T(187) / T(2100)) * k6[i] +
                          (T(1) / T(40)) * k7[i]);
    }

    T err = T(0);
    for (std::size_t i = 0; i < N; ++i) {
      const T scale = atol + rtol * std::max(std::abs(x[i]), std::abs(y5[i]));
      const T term = std::abs(y5[i] - y4[i]) / scale;
      // std::max(err, NaN) silently keeps err, so a NaN produced by the
      // vector field would otherwise be accepted as a good step.
      if (!std::isfinite(term)) {
        return std::numeric_limits<T>::quiet_NaN();
      }
      if (term > err) err = term;
      x5[i] = y5[i];
    }
    return err;
  }
};

template <class Sys>
struct AdaptiveReport {
  using real = typename Sys::real;
  std::size_t accepted = 0;
  std::size_t rejected = 0;
  real h_last = real(0);
};

// Adaptive Dormand-Prince 5(4) integration from t=0 to t_end.  The step-size
// controller is the standard 0.9 * err^(-1/5) with 0.2..5 growth limits.
template <class Method = DormandPrinceMethod, OdeSystem Sys>
AdaptiveReport<Sys> integrate_adaptive(const Sys& sys, typename Sys::real* x,
                                       typename Sys::real t_end,
                                       typename Sys::real rtol = 1e-8,
                                       typename Sys::real atol = 1e-10,
                                       typename Sys::real h_init = 1e-4,
                                       typename Sys::real h_max = 0.05) {
  using T = typename Sys::real;
  constexpr std::size_t N = static_cast<std::size_t>(Sys::dim);

  if constexpr (requires { sys.validate(); }) sys.validate();
  validate_state<Sys>(x);
  if (!(t_end >= T(0)) || !std::isfinite(t_end)) {
    throw std::invalid_argument(
        "chaosys::integrate_adaptive: t_end must be finite and >= 0");
  }
  if (!(rtol > T(0)) || !std::isfinite(rtol) || !(atol > T(0)) ||
      !std::isfinite(atol) || !(h_init > T(0)) || !std::isfinite(h_init) ||
      !(h_max > T(0)) || !std::isfinite(h_max)) {
    throw std::invalid_argument(
        "chaosys::integrate_adaptive: tolerances and step sizes must be finite "
        "and > 0");
  }

  std::array<T, N> x5{};
  AdaptiveReport<Sys> report;
  T t = T(0);
  T h = std::min(h_init, h_max);
  report.h_last = h;

  // Only *consecutive* rejections indicate a controller failure; a long run
  // with many accepted steps is allowed (e.g. a large t_end with small h_max).
  constexpr std::size_t kMaxConsecutiveRejects = 1'000'000;
  std::size_t rejects = 0;

  while (t < t_end) {
    if (h > t_end - t) h = t_end - t;

    const T err = Method::step(sys, x, h, x5.data(), rtol, atol);
    if (!std::isfinite(err)) {
      throw std::runtime_error(
          "chaosys::integrate_adaptive: non-finite error estimate");
    }
    const bool accepted = err <= T(1);
    T factor;
    if (err <= T(0)) {
      factor = T(5);
    } else {
      factor = T(0.9) *
               static_cast<T>(std::pow(static_cast<double>(err), -0.2));
    }

    if (accepted) {
      t += h;
      for (std::size_t i = 0; i < N; ++i) x[i] = x5[i];
      ++report.accepted;
      rejects = 0;
      factor = std::min(T(5), std::max(T(0.2), factor));
    } else {
      ++report.rejected;
      ++rejects;
      factor = std::min(T(1), std::max(T(0.1), factor));
    }

    h *= factor;
    h = std::min(h, h_max);
    report.h_last = h;

    if (t >= t_end) break;  // last step landed exactly on t_end
    if (!(h > T(0)) || !std::isfinite(h) || t + h == t) {
      throw std::runtime_error(
          "chaosys::integrate_adaptive: step size underflow");
    }
    if (rejects > kMaxConsecutiveRejects) {
      throw std::runtime_error(
          "chaosys::integrate_adaptive: too many rejected steps");
    }
  }
  return report;
}

// ---------------------------------------------------------------------------
// Single trajectory
// ---------------------------------------------------------------------------
template <class Method = Rk4Method, OdeSystem Sys>
void integrate(const Sys& sys, typename Sys::real* x, std::size_t steps,
               typename Sys::real dt) {
  if constexpr (requires { sys.validate(); }) sys.validate();
  validate_dt(dt);
  validate_horizon(steps, dt);
  validate_state<Sys>(x);
  for (std::size_t s = 0; s < steps; ++s) {
    Method::step(sys, x, dt);
  }
}

// ---------------------------------------------------------------------------
// Generic ensemble (portable scalar path)
// ---------------------------------------------------------------------------
template <OdeSystem Sys>
class EnsembleN {
 public:
  using real = typename Sys::real;
  static constexpr int dim = Sys::dim;

  explicit EnsembleN(std::size_t n) : n_(n) {
    for (auto& c : comp_) c.resize(n);
  }

  [[nodiscard]] std::size_t size() const noexcept { return n_; }

  [[nodiscard]] real* component(int i) noexcept {
    return comp_[static_cast<std::size_t>(i)].data();
  }
  [[nodiscard]] const real* component(int i) const noexcept {
    return comp_[static_cast<std::size_t>(i)].data();
  }

 private:
  std::array<std::vector<real, AlignedAllocator<real>>,
             static_cast<std::size_t>(Sys::dim)>
      comp_;
  std::size_t n_;
};

template <class Method = Rk4Method, OdeSystem Sys>
void integrate_ensemble(const Sys& sys, EnsembleN<Sys>& ens, std::size_t steps,
                        typename Sys::real dt, int nthreads = 1,
                        std::size_t block = 512) {
  if constexpr (requires { sys.validate(); }) sys.validate();
  validate_dt(dt);
  validate_horizon(steps, dt);
  const std::size_t n = ens.size();
  if (n == 0 || steps == 0) return;

  using T = typename Sys::real;
  constexpr std::size_t N = static_cast<std::size_t>(Sys::dim);
  std::array<T*, N> c;
  for (std::size_t i = 0; i < N; ++i) c[i] = ens.component(static_cast<int>(i));

  // Validate the whole initial ensemble before touching it.
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (!std::isfinite(c[i][j])) {
        throw std::invalid_argument("chaosys::integrate_ensemble: non-finite state");
      }
    }
  }

  detail::parallel_blocks(n, block, nthreads, [&](std::size_t b,
                                                  std::size_t e) {
    for (std::size_t s = 0; s < steps; ++s) {
      for (std::size_t j = b; j < e; ++j) {
        std::array<T, N> x;
        for (std::size_t i = 0; i < N; ++i) x[i] = c[i][j];
        Method::step(sys, x.data(), dt);
        for (std::size_t i = 0; i < N; ++i) c[i][j] = x[i];
      }
    }
  });
}

}  // namespace chaosys

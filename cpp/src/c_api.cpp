// C ABI implementation.  Argument checks return a status directly; the work
// inside each function is wrapped so C++ exceptions become status codes.
#include "chaosys_c.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include "chaosys.hpp"
#include "chaosys/analysis.hpp"
#include "chaosys/chaos_metrics.hpp"
#include "chaosys/parallel.hpp"

namespace {

constexpr std::size_t kMaxRuntimeDim = 64;

chaosys::Params to_cpp(chaosys_params p) {
  return chaosys::Params{p.sigma, p.rho, p.beta};
}

chaosys_state to_c(const chaosys::State& s) {
  return chaosys_state{s.x, s.y, s.z};
}

bool finite_state(const chaosys_state& s) {
  return std::isfinite(s.x) && std::isfinite(s.y) && std::isfinite(s.z);
}

bool valid_system(const chaosys_system* system) {
  return system != nullptr && system->deriv != nullptr && system->dim > 0 &&
         static_cast<std::size_t>(system->dim) <= kMaxRuntimeDim;
}

bool valid_time_system(const chaosys_time_system* system) {
  return system != nullptr && system->deriv != nullptr && system->dim > 0 &&
         static_cast<std::size_t>(system->dim) < kMaxRuntimeDim;
}

template <std::size_t N>
bool finite_prefix(const std::array<double, N>& state, int dim) {
  for (int i = 0; i < dim; ++i) {
    if (!std::isfinite(state[static_cast<std::size_t>(i)])) return false;
  }
  return true;
}

template <std::size_t N>
void load_prefix(const double* source, int dim, std::array<double, N>& state) {
  for (int i = 0; i < dim; ++i) {
    state[static_cast<std::size_t>(i)] = source[i];
  }
}

template <std::size_t N>
void store_prefix(const std::array<double, N>& state, int dim, double* target) {
  for (int i = 0; i < dim; ++i) {
    target[i] = state[static_cast<std::size_t>(i)];
  }
}

template <std::size_t N>
std::array<double, N> runtime_direction(int dim) {
  const auto full = chaosys::detail::perturbation_direction<double, N>();
  std::array<double, N> direction{};
  double norm2 = 0.0;
  for (int i = 0; i < dim; ++i) {
    direction[static_cast<std::size_t>(i)] = full[static_cast<std::size_t>(i)];
    norm2 += direction[static_cast<std::size_t>(i)] *
             direction[static_cast<std::size_t>(i)];
  }
  const double norm = std::sqrt(norm2);
  if (!(norm > 0.0) || !std::isfinite(norm)) {
    const double fallback = 1.0 / std::sqrt(static_cast<double>(dim));
    for (int i = 0; i < dim; ++i) direction[static_cast<std::size_t>(i)] = fallback;
    return direction;
  }
  const double inv = 1.0 / norm;
  for (int i = 0; i < dim; ++i) direction[static_cast<std::size_t>(i)] *= inv;
  return direction;
}

template <class Report>
void copy_report(const Report& src, double* segment_lambdas,
                 chaosys_lyapunov_stats* report) {
  for (std::size_t i = 0; i < src.segment_lambdas.size(); ++i) {
    segment_lambdas[i] = static_cast<double>(src.segment_lambdas[i]);
  }
  report->lambda = static_cast<double>(src.lambda);
  report->stddev = static_cast<double>(src.stddev);
  report->stderr_value = static_cast<double>(src.stderr);
  report->converged = src.converged ? 1 : 0;
  report->segments = src.segments;
}

inline void copy_spectrum(const std::vector<double>& exponents, double* values,
                          size_t capacity, size_t* count,
                          const char* what) {
  if (exponents.size() > capacity) {
    throw std::invalid_argument(what);
  }
  for (std::size_t i = 0; i < exponents.size(); ++i) values[i] = exponents[i];
  *count = exponents.size();
}

template <class F>
chaosys_status guarded(F&& work) noexcept {
  try {
    work();
    return CHAOSYS_OK;
  } catch (const std::invalid_argument&) {
    return CHAOSYS_ERR_ARGUMENT;
  } catch (const std::bad_alloc&) {
    return CHAOSYS_ERR_MEMORY;
  } catch (const std::length_error&) {
    return CHAOSYS_ERR_MEMORY;
  } catch (const std::exception&) {
    return CHAOSYS_ERR_NUMERIC;
  } catch (...) {
    return CHAOSYS_ERR_INTERNAL;
  }
}

void require_finite(double value, const char* what) {
  if (!std::isfinite(value)) throw std::runtime_error(what);
}

// Thrown by the adapters after a user callback raised.  guarded() maps it to
// CHAOSYS_ERR_INTERNAL through its catch(...) arm.
struct CallbackFailure {};

template <class Adapter>
void throw_if_callback_failed(const Adapter& adapter) {
  if (adapter.failed.load(std::memory_order_relaxed)) throw CallbackFailure{};
}

template <std::size_t N>
struct RuntimeAdapter {
  using real = double;
  static constexpr int dim = static_cast<int>(N);
  const chaosys_system* sys;
  mutable std::atomic<bool> failed{false};

  void deriv(const double* x, double* dx) const noexcept {
    // The generic integrator works on N lanes; zero the tail so a caller with
    // a smaller dimension never feeds stale values back into the solver.
    std::fill(dx, dx + N, 0.0);
    try {
      sys->deriv(x, dx, sys->user);
    } catch (...) {
      // A callback exception must not cross the noexcept boundary or unwind a
      // worker thread.  Record it and let the caller report CHAOSYS_ERR_INTERNAL.
      failed.store(true, std::memory_order_relaxed);
    }
  }
};

template <std::size_t N>
struct RuntimeTimeAdapter {
  using real = double;
  static constexpr int dim = static_cast<int>(N);
  const chaosys_time_system* sys;
  mutable std::atomic<bool> failed{false};

  void deriv(const double* x, double* dx) const noexcept {
    const double t = x[N - 1];
    std::fill(dx, dx + N, 0.0);
    try {
      sys->deriv(t, x, dx, sys->user);
    } catch (...) {
      failed.store(true, std::memory_order_relaxed);
    }
    dx[N - 1] = 1.0;
  }
};

// Runtime-dimension wrapper around the shared spectrum implementation.  The
// N-lane adapter keeps RK4 identical to the template path; only the first
// `dim` lanes participate in the QR.
template <std::size_t N>
std::vector<double> runtime_spectrum(const chaosys_system* system,
                                     const double* x0, double dt,
                                     std::size_t transient_steps,
                                     std::size_t segment_steps,
                                     std::size_t segments) {
  const std::size_t dim = static_cast<std::size_t>(system->dim);
  constexpr double kD0 = 1e-8;
  const RuntimeAdapter<N> adapter{system};
  std::array<double, N> state{};
  load_prefix(x0, system->dim, state);
  const auto result = chaosys::detail::lyapunov_spectrum_active(
      adapter, state.data(), dt, transient_steps, segment_steps, segments, kD0,
      dim);
  throw_if_callback_failed(adapter);
  return result.exponents;
}

}  // namespace

extern "C" {

chaosys_params chaosys_default_params(void) {
  const chaosys::Params p{};
  return chaosys_params{p.sigma, p.rho, p.beta};
}

const char* chaosys_version(void) { return CHAOSYS_VERSION_STRING; }

const char* chaosys_status_string(chaosys_status status) {
  switch (status) {
    case CHAOSYS_OK: return "ok";
    case CHAOSYS_ERR_ARGUMENT: return "invalid argument";
    case CHAOSYS_ERR_NUMERIC: return "numerical failure";
    case CHAOSYS_ERR_MEMORY: return "allocation failure";
    case CHAOSYS_ERR_INTERNAL: return "internal error";
    default: return "unknown status";
  }
}

chaosys_status chaosys_rk4(chaosys_state* state, size_t steps, double dt,
                           chaosys_params p) {
  if (state == nullptr) return CHAOSYS_ERR_ARGUMENT;
  return guarded([&] {
    chaosys::State s{state->x, state->y, state->z};
    chaosys::rk4_scalar(s, steps, dt, to_cpp(p));
    *state = to_c(s);
    if (!finite_state(*state)) {
      throw std::runtime_error("chaosys_rk4: non-finite result");
    }
  });
}

chaosys_status chaosys_adaptive(chaosys_state* state, double t_end, double rtol,
                                double atol, chaosys_params p,
                                size_t* accepted_steps) {
  if (state == nullptr) return CHAOSYS_ERR_ARGUMENT;
  return guarded([&] {
    chaosys::LorenzSystem sys{to_cpp(p)};
    double x[3] = {state->x, state->y, state->z};
    const auto report =
        chaosys::integrate_adaptive<chaosys::DormandPrinceMethod>(sys, x, t_end,
                                                                  rtol, atol);
    *state = chaosys_state{x[0], x[1], x[2]};
    if (accepted_steps != nullptr) *accepted_steps = report.accepted;
    if (!finite_state(*state)) {
      throw std::runtime_error("chaosys_adaptive: non-finite result");
    }
  });
}

chaosys_status chaosys_ensemble(chaosys_state* states, size_t n, size_t steps,
                                double dt, int threads, chaosys_params p) {
  if (states == nullptr && n != 0) return CHAOSYS_ERR_ARGUMENT;
  return guarded([&] {
    chaosys::validate(to_cpp(p));
    chaosys::validate_dt(dt);
    if (n == 0) return;
    chaosys::Ensemble e;
    e.resize(n);
    for (size_t i = 0; i < n; ++i) {
      e.x[i] = states[i].x;
      e.y[i] = states[i].y;
      e.z[i] = states[i].z;
    }
    chaosys::rk4_soa_neon_threads(e, steps, dt, threads, to_cpp(p));
    for (size_t i = 0; i < n; ++i) {
      states[i] = chaosys_state{e.x[i], e.y[i], e.z[i]};
      if (!finite_state(states[i])) {
        throw std::runtime_error("chaosys_ensemble: non-finite result");
      }
    }
  });
}

chaosys_status chaosys_lyapunov(const chaosys_state* x0, double dt,
                                size_t transient_steps, size_t segment_steps,
                                size_t segments, chaosys_params p,
                                double* lambda) {
  if (x0 == nullptr || lambda == nullptr) return CHAOSYS_ERR_ARGUMENT;
  return guarded([&] {
    const double x[3] = {x0->x, x0->y, x0->z};
    const double value = chaosys::largest_lyapunov(
        chaosys::LorenzSystem{to_cpp(p)}, x, dt, transient_steps,
        segment_steps, segments);
    require_finite(value, "chaosys_lyapunov: non-finite result");
    *lambda = value;
  });
}

chaosys_status chaosys_lyapunov_report(const chaosys_state* x0, double dt,
                                           size_t transient_steps,
                                           size_t segment_steps,
                                           size_t segments, chaosys_params p,
                                           double* segment_lambdas,
                                           size_t capacity,
                                           chaosys_lyapunov_stats* report) {
  if (x0 == nullptr || segment_lambdas == nullptr || report == nullptr ||
      capacity < segments) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    const double x[3] = {x0->x, x0->y, x0->z};
    const auto result = chaosys::largest_lyapunov_report(
        chaosys::LorenzSystem{to_cpp(p)}, x, dt, transient_steps, segment_steps,
        segments);
    copy_report(result, segment_lambdas, report);
  });
}

chaosys_status chaosys_lyapunov_report_system(
    const chaosys_system* system, const double* x0, double dt,
    size_t transient_steps, size_t segment_steps, size_t segments,
    double* segment_lambdas, size_t capacity,
    chaosys_lyapunov_stats* report) {
  if (!valid_system(system) || x0 == nullptr || segment_lambdas == nullptr ||
      report == nullptr || capacity < segments) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    std::array<double, kMaxRuntimeDim> state{};
    load_prefix(x0, system->dim, state);
    const RuntimeAdapter<kMaxRuntimeDim> adapter{system};
    const auto direction = runtime_direction<kMaxRuntimeDim>(system->dim);
    const auto result = chaosys::largest_lyapunov_report(
        adapter, state.data(), dt, transient_steps, segment_steps, segments,
        1e-8, 0.1, &direction);
    throw_if_callback_failed(adapter);
    copy_report(result, segment_lambdas, report);
  });
}

chaosys_status chaosys_lyapunov_spectrum(const chaosys_state* x0, double dt,
                                         size_t transient_steps,
                                         size_t segment_steps,
                                         size_t segments, chaosys_params p,
                                         double* values, size_t capacity,
                                         size_t* count) {
  if (x0 == nullptr || values == nullptr || count == nullptr || capacity < 3) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    const double x[3] = {x0->x, x0->y, x0->z};
    const auto result = chaosys::lyapunov_spectrum(
        chaosys::LorenzSystem{to_cpp(p)}, x, dt, transient_steps, segment_steps,
        segments);
    copy_spectrum(result.exponents, values, capacity, count,
                  "chaosys_lyapunov_spectrum: values too small");
  });
}

chaosys_status chaosys_lyapunov_spectrum_system(
    const chaosys_system* system, const double* x0, double dt,
    size_t transient_steps, size_t segment_steps, size_t segments,
    double* values, size_t capacity, size_t* count) {
  if (!valid_system(system) || x0 == nullptr || values == nullptr ||
      count == nullptr || capacity < static_cast<std::size_t>(system->dim)) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    const auto exponents = runtime_spectrum<kMaxRuntimeDim>(
        system, x0, dt, transient_steps, segment_steps, segments);
    copy_spectrum(exponents, values, capacity, count,
                  "chaosys_lyapunov_spectrum_system: values too small");
  });
}

chaosys_status chaosys_poincare(const chaosys_state* x0, double dt,
                                size_t transient_steps, size_t max_steps,
                                chaosys_params p, int component, double level,
                                int direction, chaosys_state* out,
                                size_t capacity, size_t* count) {
  if (x0 == nullptr || out == nullptr || count == nullptr || capacity == 0) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    double x[3] = {x0->x, x0->y, x0->z};
    const auto points = chaosys::poincare_section(
        chaosys::LorenzSystem{to_cpp(p)}, x, dt, transient_steps, max_steps,
        capacity, component, level, direction);
    for (size_t i = 0; i < points.size(); ++i) {
      out[i] = chaosys_state{points[i].x[0], points[i].x[1], points[i].x[2]};
    }
    *count = points.size();
  });
}

chaosys_status chaosys_rk4_system(const chaosys_system* system, double* x,
                                  size_t steps, double dt) {
  if (!valid_system(system) || x == nullptr || !(dt > 0.0) ||
      !std::isfinite(dt)) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    std::array<double, kMaxRuntimeDim> state{};
    load_prefix(x, system->dim, state);
    const RuntimeAdapter<kMaxRuntimeDim> adapter{system};
    chaosys::integrate<chaosys::Rk4Method>(adapter, state.data(), steps, dt);
    throw_if_callback_failed(adapter);
    if (!finite_prefix(state, system->dim)) {
      throw std::runtime_error("chaosys_rk4_system: non-finite result");
    }
    store_prefix(state, system->dim, x);
  });
}

chaosys_status chaosys_adaptive_system(const chaosys_system* system, double* x,
                                       double t_end, double rtol, double atol,
                                       size_t* accepted_steps) {
  if (!valid_system(system) || x == nullptr) return CHAOSYS_ERR_ARGUMENT;
  return guarded([&] {
    std::array<double, kMaxRuntimeDim> state{};
    load_prefix(x, system->dim, state);
    const RuntimeAdapter<kMaxRuntimeDim> adapter{system};
    const auto report =
        chaosys::integrate_adaptive<chaosys::DormandPrinceMethod>(
            adapter, state.data(), t_end, rtol, atol);
    throw_if_callback_failed(adapter);
    if (!finite_prefix(state, system->dim)) {
      throw std::runtime_error("chaosys_adaptive_system: non-finite result");
    }
    store_prefix(state, system->dim, x);
    if (accepted_steps != nullptr) *accepted_steps = report.accepted;
  });
}

chaosys_status chaosys_lyapunov_system(const chaosys_system* system,
                                       const double* x0, double dt,
                                       size_t transient_steps,
                                       size_t segment_steps, size_t segments,
                                       double* lambda) {
  if (!valid_system(system) || x0 == nullptr || lambda == nullptr) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    std::array<double, kMaxRuntimeDim> state{};
    load_prefix(x0, system->dim, state);
    const RuntimeAdapter<kMaxRuntimeDim> adapter{system};
    const auto direction = runtime_direction<kMaxRuntimeDim>(system->dim);
    const auto result = chaosys::largest_lyapunov_report(
        adapter, state.data(), dt, transient_steps, segment_steps, segments,
        1e-8, 0.1, &direction);
    throw_if_callback_failed(adapter);
    require_finite(result.lambda, "chaosys_lyapunov_system: non-finite result");
    *lambda = result.lambda;
  });
}

chaosys_status chaosys_poincare_system(const chaosys_system* system,
                                       const double* x0, double dt,
                                       size_t transient_steps, size_t max_steps,
                                       int component, double level,
                                       int direction, double* out,
                                       size_t capacity, size_t* count) {
  if (!valid_system(system) || x0 == nullptr || out == nullptr ||
      count == nullptr || capacity == 0 || component < 0 ||
      component >= system->dim) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  const std::size_t dim = static_cast<std::size_t>(system->dim);
  if (capacity > std::numeric_limits<std::size_t>::max() / dim) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    std::array<double, kMaxRuntimeDim> state{};
    load_prefix(x0, system->dim, state);
    const RuntimeAdapter<kMaxRuntimeDim> adapter{system};
    const auto points = chaosys::poincare_section(
        adapter, state.data(), dt, transient_steps, max_steps, capacity,
        component, level, direction);
    throw_if_callback_failed(adapter);
    for (std::size_t i = 0; i < points.size(); ++i) {
      for (std::size_t j = 0; j < dim; ++j) out[i * dim + j] = points[i].x[j];
    }
    *count = points.size();
  });
}

chaosys_status chaosys_ensemble_system(const chaosys_system* system,
                                       double* states, size_t n, size_t steps,
                                       double dt, int threads) {
  if (!valid_system(system) || !(dt > 0.0) || !std::isfinite(dt)) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  if (n == 0) return CHAOSYS_OK;
  if (states == nullptr) return CHAOSYS_ERR_ARGUMENT;
  const std::size_t dim = static_cast<std::size_t>(system->dim);
  if (n > std::numeric_limits<std::size_t>::max() / dim) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    for (std::size_t i = 0; i < n * dim; ++i) {
      if (!std::isfinite(states[i])) {
        throw std::runtime_error("chaosys_ensemble_system: non-finite state");
      }
    }
    const RuntimeAdapter<kMaxRuntimeDim> adapter{system};
    chaosys::detail::parallel_blocks(n, 512, threads, [&](std::size_t b,
                                                          std::size_t e) {
      for (std::size_t j = b; j < e; ++j) {
        std::array<double, kMaxRuntimeDim> state{};
        load_prefix(states + j * dim, system->dim, state);
        chaosys::integrate<chaosys::Rk4Method>(adapter, state.data(), steps, dt);
        store_prefix(state, system->dim, states + j * dim);
      }
    });
    throw_if_callback_failed(adapter);
    for (std::size_t i = 0; i < n * dim; ++i) {
      if (!std::isfinite(states[i])) {
        throw std::runtime_error("chaosys_ensemble_system: non-finite result");
      }
    }
  });
}

chaosys_status chaosys_embed(const double* series, size_t n, int dimension,
                             int delay, double* out, size_t capacity,
                             size_t* count) {
  if (series == nullptr || out == nullptr || count == nullptr || n == 0 ||
      dimension <= 0 || delay <= 0) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    const std::size_t m = static_cast<std::size_t>(dimension);
    const std::size_t tau = static_cast<std::size_t>(delay);
    if ((m - 1) * tau + 1 > n) {
      throw std::invalid_argument("chaosys_embed: series too short");
    }
    const std::size_t rows = n - (m - 1) * tau;
    if (rows > std::numeric_limits<std::size_t>::max() / m) {
      throw std::invalid_argument("chaosys_embed: output size overflows");
    }
    const std::size_t elements = rows * m;
    if (capacity < elements) {
      throw std::invalid_argument("chaosys_embed: output too small");
    }
    const auto vectors =
        chaosys::embed(std::vector<double>(series, series + n), dimension,
                       delay);
    if (vectors.size() != rows) {
      throw std::runtime_error("chaosys_embed: inconsistent result");
    }
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < m; ++j) out[i * m + j] = vectors[i][j];
    }
    *count = rows;
  });
}

chaosys_status chaosys_correlation_dimension(const double* series, size_t n,
                                             int dimension, int delay,
                                             size_t max_points, double* value) {
  if (series == nullptr || value == nullptr || n == 0 || dimension <= 0 ||
      delay <= 0 || max_points < 8) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    const auto result = chaosys::correlation_dimension_series(
        std::vector<double>(series, series + n), dimension, delay, max_points);
    require_finite(result.dimension,
                   "chaosys_correlation_dimension: non-finite result");
    *value = result.dimension;
  });
}


chaosys_status chaosys_permutation_entropy(const double* series, size_t n,
                                           int order, int delay, double* value) {
  if (series == nullptr || value == nullptr || n == 0 || order < 2 ||
      order > 8 || delay <= 0) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    const double entropy = chaosys::permutation_entropy(
        std::vector<double>(series, series + n), order, delay);
    require_finite(entropy, "chaosys_permutation_entropy: non-finite result");
    *value = entropy;
  });
}


chaosys_status chaosys_rk4_time_system(const chaosys_time_system* system,
                                       double* x, double t0, size_t steps,
                                       double dt) {
  if (!valid_time_system(system) || x == nullptr || !std::isfinite(t0) ||
      !(dt > 0.0) || !std::isfinite(dt)) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    std::array<double, kMaxRuntimeDim> state{};
    load_prefix(x, system->dim, state);
    state[kMaxRuntimeDim - 1] = t0;
    const RuntimeTimeAdapter<kMaxRuntimeDim> adapter{system};
    chaosys::integrate<chaosys::Rk4Method>(adapter, state.data(), steps, dt);
    throw_if_callback_failed(adapter);
    if (!finite_prefix(state, system->dim)) {
      throw std::runtime_error("chaosys_rk4_time_system: non-finite result");
    }
    store_prefix(state, system->dim, x);
  });
}

chaosys_status chaosys_adaptive_time_system(const chaosys_time_system* system,
                                            double* x, double t0, double t_end,
                                            double rtol, double atol,
                                            size_t* accepted_steps) {
  if (!valid_time_system(system) || x == nullptr || !std::isfinite(t0) ||
      !std::isfinite(t_end)) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    if (!(t_end > t0)) {
      throw std::invalid_argument(
          "chaosys_adaptive_time_system: t_end must be greater than t0");
    }
    std::array<double, kMaxRuntimeDim> state{};
    load_prefix(x, system->dim, state);
    state[kMaxRuntimeDim - 1] = t0;
    const RuntimeTimeAdapter<kMaxRuntimeDim> adapter{system};
    const auto report =
        chaosys::integrate_adaptive<chaosys::DormandPrinceMethod>(
            adapter, state.data(), t_end - t0, rtol, atol);
    throw_if_callback_failed(adapter);
    if (!finite_prefix(state, system->dim)) {
      throw std::runtime_error(
          "chaosys_adaptive_time_system: non-finite result");
    }
    store_prefix(state, system->dim, x);
    if (accepted_steps != nullptr) *accepted_steps = report.accepted;
  });
}

chaosys_status chaosys_lyapunov_time_system(const chaosys_time_system* system,
                                            const double* x0, double t0,
                                            double dt, size_t transient_steps,
                                            size_t segment_steps,
                                            size_t segments, double* lambda) {
  if (!valid_time_system(system) || x0 == nullptr || lambda == nullptr ||
      !std::isfinite(t0)) {
    return CHAOSYS_ERR_ARGUMENT;
  }
  return guarded([&] {
    std::array<double, kMaxRuntimeDim> state{};
    load_prefix(x0, system->dim, state);
    state[kMaxRuntimeDim - 1] = t0;
    const RuntimeTimeAdapter<kMaxRuntimeDim> adapter{system};
    const auto direction = runtime_direction<kMaxRuntimeDim>(system->dim);
    const auto result = chaosys::largest_lyapunov_report(
        adapter, state.data(), dt, transient_steps, segment_steps, segments,
        1e-8, 0.1, &direction);
    throw_if_callback_failed(adapter);
    require_finite(result.lambda,
                   "chaosys_lyapunov_time_system: non-finite result");
    *lambda = result.lambda;
  });
}

}  // extern "C"

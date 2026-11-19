// Dependency-free test suite for the Lorenz core, the generic ODE layer and
// the ensemble driver's edge cases / determinism guarantees.
#include "chaosys.hpp"
#include "chaosys/analysis.hpp"
#include "chaosys/systems.hpp"
#include "chaosys/chaos_metrics.hpp"
#include "chaosys/csv.hpp"
#include "chaosys/time_system.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " #cond    \
                << "\n";                                                   \
      ++g_failures;                                                        \
    }                                                                      \
  } while (0)

#define CHECK_THROW(expr, Exception)                                       \
  do {                                                                     \
    ++g_checks;                                                            \
    bool threw = false;                                                    \
    try {                                                                  \
      (void)(expr);                                                        \
    } catch (const Exception&) {                                           \
      threw = true;                                                        \
    } catch (...) {                                                        \
    }                                                                      \
    if (!threw) {                                                          \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__                  \
                << "  expected " #Exception "\n";                          \
      ++g_failures;                                                        \
    }                                                                      \
  } while (0)

// Numerical fixtures only; the product scope is chaotic systems.
struct Decay {
  using real = double;
  static constexpr int dim = 1;
  double rate = 1.0;
  void deriv(const double* x, double* dx) const noexcept {
    dx[0] = -rate * x[0];
  }
};

struct DecayF {
  using real = float;
  static constexpr int dim = 1;
  void deriv(const float* x, float* dx) const noexcept { dx[0] = -x[0]; }
};

struct Harmonic {
  using real = double;
  static constexpr int dim = 2;
  double omega = 1.0;
  void deriv(const double* x, double* dx) const noexcept {
    dx[0] = x[1];
    dx[1] = -omega * omega * x[0];
  }
};

// Slow linear decay from a large state: the perturbation must be scaled so it
// stays representable in double precision.
struct SlowDecay {
  using real = double;
  static constexpr int dim = 1;
  double rate = 1e-3;
  void deriv(const double* x, double* dx) const noexcept {
    dx[0] = -rate * x[0];
  }
};

// Derivative deliberately produces NaN; the adaptive controller must reject
// it instead of accepting the step.
struct NaNSystem {
  using real = double;
  static constexpr int dim = 1;
  void deriv(const double* /*x*/, double* dx) const noexcept {
    dx[0] = std::numeric_limits<double>::quiet_NaN();
  }
};

// Decoupled Lorenz: x0 is a stable, independent direction.  A perturbation
// restricted to x0 would measure -1 instead of the chaotic Lorenz exponent.
struct DecoupledLorenz {
  using real = double;
  static constexpr int dim = 4;
  void deriv(const double* x, double* dx) const noexcept {
    dx[0] = -x[0];
    dx[1] = 10.0 * (x[2] - x[1]);
    dx[2] = x[1] * (28.0 - x[3]) - x[2];
    dx[3] = x[1] * x[2] - (8.0 / 3.0) * x[3];
  }
};

template <class Method>
double decay_error(double dt) {
  double x[1] = {1.0};
  const auto steps = static_cast<std::size_t>(std::llround(1.0 / dt));
  chaosys::integrate<Method>(Decay{}, x, steps, dt);
  return std::abs(x[0] - std::exp(-1.0));
}

template <class Method>
void check_decay_order(double low, double high) {
  const double e1 = decay_error<Method>(0.1);
  const double e2 = decay_error<Method>(0.05);
  const double e3 = decay_error<Method>(0.025);
  CHECK(e1 / e2 > low && e1 / e2 < high);
  CHECK(e2 / e3 > low && e2 / e3 < high);
}

double harmonic_energy(const double* x, double omega) {
  return 0.5 * (x[1] * x[1] + omega * omega * x[0] * x[0]);
}

// --------------------------------------------------------------------------
void test_params_validation() {
  chaosys::validate(chaosys::Params{});
  const chaosys::Params nan_sigma{std::nan(""), 28, 8.0 / 3};
  const chaosys::Params zero_beta{10, 28, 0.0};
  const chaosys::Params negative_beta{10, 28, -1.0};
  CHECK_THROW(chaosys::validate(nan_sigma), std::invalid_argument);
  CHECK_THROW(chaosys::validate(zero_beta), std::invalid_argument);
  CHECK_THROW(chaosys::validate(negative_beta), std::invalid_argument);
}

void test_deriv_and_jacobian() {
  const chaosys::Params p{10.0, 28.0, 8.0 / 3.0};
  chaosys::LorenzSystem sys{p};
  double x[3] = {1.5, -2.0, 3.0};
  double dx[3];
  sys.deriv(x, dx);
  CHECK(std::abs(dx[0] - p.sigma * (x[1] - x[0])) < 1e-15);
  CHECK(std::abs(dx[1] - (x[0] * (p.rho - x[2]) - x[1])) < 1e-15);
  CHECK(std::abs(dx[2] - (x[0] * x[1] - p.beta * x[2])) < 1e-15);

  // div f = -(sigma + 1 + beta) for the Lorenz flow.
  constexpr double eps = 1e-6;
  double trace = 0.0;
  for (int j = 0; j < 3; ++j) {
    double xp[3] = {x[0], x[1], x[2]};
    double xm[3] = {x[0], x[1], x[2]};
    xp[j] += eps;
    xm[j] -= eps;
    double dp[3], dm[3];
    sys.deriv(xp, dp);
    sys.deriv(xm, dm);
    trace += (dp[j] - dm[j]) / (2 * eps);
  }
  CHECK(std::abs(trace + (p.sigma + 1.0 + p.beta)) < 1e-6);
}

void test_known_rk4_state() {
  chaosys::State s{1, 1, 1};
  chaosys::rk4_scalar(s, 1000, 0.01);
  CHECK(std::abs(s.x - (-4.902819483748812)) < 1e-12);
  CHECK(std::abs(s.y - (-3.743407675271541)) < 1e-12);
  CHECK(std::abs(s.z - 24.691885987964351) < 1e-12);
}

void test_generic_decay() {
  double x[1] = {1.0};
  chaosys::integrate<chaosys::Rk4Method>(Decay{}, x, 1000, 1e-3);
  CHECK(std::abs(x[0] - std::exp(-1.0)) < 1e-10);
}

void test_generic_harmonic() {
  double x[2] = {1.0, 0.0};
  constexpr double omega = 1.0;
  const double e0 = harmonic_energy(x, omega);
  const double period = 2.0 * std::acos(-1.0) / omega;
  const int steps = 10000;
  const double dt = 10.0 * period / steps;  // exactly ten periods
  chaosys::integrate<chaosys::Rk4Method>(Harmonic{omega}, x, steps, dt);
  const double e1 = harmonic_energy(x, omega);
  CHECK(std::abs(e1 - e0) < 1e-7);
  CHECK(std::abs(x[0] - 1.0) < 1e-6);
  CHECK(std::abs(x[1]) < 1e-6);
}

void test_generic_float() {
  float x[1] = {1.0f};
  chaosys::integrate<chaosys::Rk4Method>(DecayF{}, x, 1000, 1e-3f);
  CHECK(std::abs(x[0] - std::exp(-1.0f)) < 1e-4f);
}

void test_generic_matches_specialized() {
  const chaosys::Params p{10.0, 28.0, 8.0 / 3.0};
  double g[3] = {1, 1, 1};
  chaosys::integrate<chaosys::Rk4Method>(chaosys::LorenzSystem{p}, g, 1000,
                                         0.01);

  chaosys::State s{1, 1, 1};
  chaosys::rk4_scalar(s, 1000, 0.01, p);
  CHECK(std::abs(g[0] - s.x) < 1e-9);
  CHECK(std::abs(g[1] - s.y) < 1e-9);
  CHECK(std::abs(g[2] - s.z) < 1e-9);
}

void test_runtime_params() {
  const chaosys::Params a{10.0, 28.0, 8.0 / 3.0};
  const chaosys::Params b{14.0, 45.0, 3.0};
  chaosys::State sa{1, 1, 1}, sb{1, 1, 1};
  chaosys::rk4_scalar(sa, 1000, 0.01, a);
  chaosys::rk4_scalar(sb, 1000, 0.01, b);
  CHECK(std::abs(sa.x - sb.x) > 1e-3);

  const chaosys::State ref = chaosys::dp5_reference(sa, 10.0, a, 1e-13, 1e-14);
  CHECK(std::isfinite(ref.x) && std::isfinite(ref.y) && std::isfinite(ref.z));
  CHECK_THROW(chaosys::dp5_reference(sa, 10.0, a, 0.0, 1e-13),
              std::invalid_argument);
}

void test_ensemble_accuracy_and_threads() {
  chaosys::Ensemble a, b;
  chaosys::init_ensemble(a, 4096);
  chaosys::init_ensemble(b, 4096);

  chaosys::rk4_soa_scalar(a, 200, 0.01);
  chaosys::rk4_soa_neon(b, 200, 0.01);
  const chaosys::State diff_ab = chaosys::max_diff(a, b);
  CHECK(diff_ab.x < 1e-8 && diff_ab.y < 1e-8 && diff_ab.z < 1e-8);

  // Same kernel, different thread counts: bitwise identical.
  chaosys::Ensemble c, d, e;
  chaosys::init_ensemble(a, 4096);
  chaosys::init_ensemble(c, 4096);
  chaosys::rk4_soa_scalar(a, 200, 0.01);
  chaosys::rk4_soa_scalar_threads(c, 200, 0.01, 8);
  const chaosys::State t = chaosys::max_diff(a, c);
  CHECK(t.x == 0.0 && t.y == 0.0 && t.z == 0.0);

  chaosys::init_ensemble(d, 4096);
  chaosys::init_ensemble(e, 4096);
  chaosys::rk4_soa_neon(d, 200, 0.01);
  chaosys::rk4_soa_neon_threads(e, 200, 0.01, 8);
  const chaosys::State nt = chaosys::max_diff(d, e);
  CHECK(nt.x == 0.0 && nt.y == 0.0 && nt.z == 0.0);

  // Block boundaries must not change per-trajectory arithmetic either.
  chaosys::Ensemble f;
  chaosys::init_ensemble(f, 4096);
  chaosys::rk4_soa_neon_blocked(f, 200, 0.01, 128);
  const chaosys::State blk = chaosys::max_diff(d, f);
  CHECK(blk.x == 0.0 && blk.y == 0.0 && blk.z == 0.0);
}

void test_ensemble_params() {
  const chaosys::Params p{12.5, 33.0, 3.1};
  chaosys::Ensemble e;
  chaosys::init_ensemble(e, 64);
  chaosys::rk4_soa_neon(e, 100, 0.005, p);

  // Verify trajectory 0 against the generic integrator with the same params.
  chaosys::Ensemble init;
  chaosys::init_ensemble(init, 64);
  double x[3] = {init.x[0], init.y[0], init.z[0]};
  chaosys::integrate<chaosys::Rk4Method>(chaosys::LorenzSystem{p}, x, 100,
                                         0.005);
  CHECK(std::abs(e.x[0] - x[0]) < 1e-10);
  CHECK(std::abs(e.y[0] - x[1]) < 1e-10);
  CHECK(std::abs(e.z[0] - x[2]) < 1e-10);
}

void test_generic_ensemble() {
  const chaosys::Params p{10.0, 28.0, 8.0 / 3.0};
  chaosys::LorenzSystem sys{p};
  chaosys::EnsembleN<chaosys::LorenzSystem> ens(32);
  for (std::size_t i = 0; i < ens.size(); ++i) {
    ens.component(0)[i] = 1.0 + 0.01 * static_cast<double>(i);
    ens.component(1)[i] = 1.0;
    ens.component(2)[i] = 1.0;
  }
  chaosys::integrate_ensemble<chaosys::Rk4Method>(sys, ens, 200, 0.01);

  double x[3] = {1.0, 1.0, 1.0};
  chaosys::integrate<chaosys::Rk4Method>(sys, x, 200, 0.01);
  CHECK(std::abs(ens.component(0)[0] - x[0]) < 1e-12);
  CHECK(std::abs(ens.component(1)[0] - x[1]) < 1e-12);
  CHECK(std::abs(ens.component(2)[0] - x[2]) < 1e-12);
}

void test_stride_semantics() {
  chaosys::State s{1, 1, 1};
  std::vector<chaosys::State> out(11);
  chaosys::rk4_scalar_strided(s, 100, 0.01, out.data(), 10);
  CHECK(std::abs(out[0].x - 1.0) < 1e-15);

  chaosys::State t{1, 1, 1};
  chaosys::rk4_scalar(t, 10, 0.01);
  CHECK(std::abs(out[1].x - t.x) < 1e-15);
  CHECK(std::abs(out[10].x - s.x) < 1e-15);
}

void test_edge_cases() {
  chaosys::State s{1, 1, 1};
  chaosys::State out[2];

  CHECK_THROW(chaosys::rk4_scalar(s, 10, 0.0), std::invalid_argument);
  CHECK_THROW(chaosys::rk4_scalar(s, 10, std::nan("")), std::invalid_argument);
  const chaosys::Params zero_beta{10, 28, 0.0};
  CHECK_THROW(chaosys::rk4_scalar(s, 10, 0.01, zero_beta),
              std::invalid_argument);
  CHECK_THROW(chaosys::rk4_scalar_strided(s, 10, 0.01, out, 0),
              std::invalid_argument);
  CHECK_THROW(chaosys::rk4_scalar_strided(s, 10, 0.01, nullptr, 1),
              std::invalid_argument);

  // No-op runs are valid.
  const chaosys::State before = s;
  chaosys::rk4_scalar(s, 0, 0.01);
  CHECK(s.x == before.x && s.y == before.y && s.z == before.z);

  chaosys::Ensemble empty;
  chaosys::rk4_soa_neon(empty, 10, 0.01);
  CHECK(empty.n == 0);

  chaosys::Ensemble a, b;
  chaosys::init_ensemble(a, 8);
  chaosys::init_ensemble(b, 9);
  CHECK_THROW(chaosys::max_diff(a, b), std::invalid_argument);

  // Thread counts are clamped, not fatal.
  chaosys::init_ensemble(b, 8);
  chaosys::rk4_soa_neon_threads(b, 5, 0.01, 0);
  chaosys::init_ensemble(b, 8);
  chaosys::rk4_soa_neon_threads(b, 5, 0.01, 100000);
}

void test_adaptive_decay() {
  double x[1] = {1.0};
  const auto report = chaosys::integrate_adaptive<chaosys::DormandPrinceMethod>(
      Decay{}, x, 1.0, 1e-10, 1e-12);
  CHECK(std::abs(x[0] - std::exp(-1.0)) < 1e-9);
  CHECK(report.accepted > 0);
  CHECK(report.h_last > 0.0);

  double loose[1] = {1.0};
  const auto loose_report =
      chaosys::integrate_adaptive<chaosys::DormandPrinceMethod>(
          Decay{}, loose, 1.0, 1e-4, 1e-6);
  CHECK(std::abs(loose[0] - std::exp(-1.0)) < 1e-4);
  CHECK(report.accepted >= loose_report.accepted);
}

void test_convergence_orders() {
  // Halving dt reduces the error by 2^p for a p-th order method.
  check_decay_order<chaosys::EulerMethod>(1.6, 2.4);   // p = 1
  check_decay_order<chaosys::HeunMethod>(3.0, 5.0);    // p = 2
  check_decay_order<chaosys::Rk4Method>(12.0, 20.0);   // p = 4
}

void test_adaptive_harmonic() {
  const double omega = 1.0;
  double x[2] = {1.0, 0.0};
  const double e0 = harmonic_energy(x, omega);
  const double period = 2.0 * std::acos(-1.0) / omega;
  chaosys::integrate_adaptive<chaosys::DormandPrinceMethod>(
      Harmonic{omega}, x, 10.0 * period, 1e-9, 1e-11);
  CHECK(std::abs(harmonic_energy(x, omega) - e0) < 1e-8);
  CHECK(std::abs(x[0] - 1.0) < 1e-6);
  CHECK(std::abs(x[1]) < 1e-6);
}

void test_adaptive_validation() {
  double x[1] = {1.0};
  CHECK_THROW(chaosys::integrate_adaptive<chaosys::DormandPrinceMethod>(
                  Decay{}, x, -1.0),
              std::invalid_argument);
  CHECK_THROW(chaosys::integrate_adaptive<chaosys::DormandPrinceMethod>(
                  Decay{}, x, 1.0, 0.0, 1e-12),
              std::invalid_argument);
  CHECK_THROW(chaosys::integrate_adaptive<chaosys::DormandPrinceMethod>(
                  Decay{}, x, 1.0, 1e-8, 0.0),
              std::invalid_argument);
}

void test_generic_ensemble_threads() {
  const chaosys::Params p{10.0, 28.0, 8.0 / 3.0};
  chaosys::LorenzSystem sys{p};
  chaosys::EnsembleN<chaosys::LorenzSystem> a(1024), b(1024);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double x = 1.0 + 0.001 * static_cast<double>(i);
    a.component(0)[i] = b.component(0)[i] = x;
    a.component(1)[i] = b.component(1)[i] = 1.0;
    a.component(2)[i] = b.component(2)[i] = 1.0;
  }
  chaosys::integrate_ensemble<chaosys::Rk4Method>(sys, a, 50, 0.01, 1);
  chaosys::integrate_ensemble<chaosys::Rk4Method>(sys, b, 50, 0.01, 8);
  bool same = true;
  for (int c = 0; c < 3; ++c) {
    for (std::size_t i = 0; i < a.size(); ++i) {
      same = same && a.component(c)[i] == b.component(c)[i];
    }
  }
  CHECK(same);
}

void test_lyapunov() {
  chaosys::LorenzSystem sys{};
  double x0[3] = {1.0, 1.0, 1.0};
  const double lambda =
      chaosys::largest_lyapunov(sys, x0, 0.005, 2000, 50, 200);
  CHECK(lambda > 0.7 && lambda < 1.1);

  double xa[3] = {1.0, 1.0, 1.0};
  double xb[3] = {1.0, 1.0, 1.0};
  const double coarse =
      chaosys::largest_lyapunov(sys, xa, 0.005, 2000, 100, 100);
  const double fine = chaosys::largest_lyapunov(sys, xb, 0.005, 2000, 50, 200);
  CHECK(std::abs(coarse - fine) < 0.01);

  double decay[1] = {1.0};
  const double decay_lambda =
      chaosys::largest_lyapunov(Decay{}, decay, 0.01, 100, 10, 100);
  CHECK(std::abs(decay_lambda + 1.0) < 1e-3);

  double osc[2] = {1.0, 0.0};
  const double osc_lambda =
      chaosys::largest_lyapunov(Harmonic{1.0}, osc, 0.01, 100, 10, 100);
  CHECK(std::abs(osc_lambda) < 0.05);
}

void test_lyapunov_report_and_spectrum() {
  chaosys::LorenzSystem lorenz{};
  double x0[3] = {1.0, 1.0, 1.0};
  const auto report =
      chaosys::largest_lyapunov_report(lorenz, x0, 0.005, 2000, 50, 200);
  CHECK(report.segments == 200);
  CHECK(report.segment_lambdas.size() == 200);
  CHECK(report.lambda > 0.7 && report.lambda < 1.1);
  CHECK(report.stddev >= 0.0 && std::isfinite(report.stddev));
  CHECK(report.stderr >= 0.0 && std::isfinite(report.stderr));
  const double scalar =
      chaosys::largest_lyapunov(lorenz, x0, 0.005, 2000, 50, 200);
  CHECK(std::abs(report.lambda - scalar) < 1e-12);

  double sum = 0.0;
  for (const double value : report.segment_lambdas) sum += value;
  CHECK(std::abs(sum / report.segments - report.lambda) < 1e-10);

  double decay[1] = {1.0};
  const auto decay_report =
      chaosys::largest_lyapunov_report(Decay{}, decay, 0.01, 100, 10, 100);
  CHECK(std::abs(decay_report.lambda + 1.0) < 1e-3);
  CHECK(decay_report.converged);

  const auto spectrum =
      chaosys::lyapunov_spectrum(lorenz, x0, 0.005, 2000, 50, 200);
  CHECK(spectrum.exponents.size() == 3);
  CHECK(spectrum.exponents[0] > spectrum.exponents[1]);
  CHECK(spectrum.exponents[1] > spectrum.exponents[2]);
  CHECK(spectrum.exponents[0] > 0.7 && spectrum.exponents[0] < 1.1);
  CHECK(std::abs(spectrum.exponents[1]) < 0.2);
  CHECK(spectrum.exponents[2] < -10.0);
  // Sum of exponents = average divergence = -(sigma + 1 + beta).
  CHECK(std::abs(spectrum.sum + (10.0 + 1.0 + 8.0 / 3.0)) < 0.05);

  // A stable x0 direction must not hide the chaotic exponent elsewhere.
  double decoupled[4] = {1.0, 1.0, 1.0, 1.0};
  const double general =
      chaosys::largest_lyapunov(DecoupledLorenz{}, decoupled, 0.005, 2000, 50,
                                200);
  CHECK(general > 0.5 && general < 1.1);
  const auto general_spectrum =
      chaosys::lyapunov_spectrum(DecoupledLorenz{}, decoupled, 0.005, 2000, 50,
                                 200);
  CHECK(general_spectrum.exponents.size() == 4);
  CHECK(std::abs(general_spectrum.exponents[2] + 1.0) < 0.05);
  CHECK(std::abs(general_spectrum.sum + (1.0 + 10.0 + 1.0 + 8.0 / 3.0)) <
        0.05);
}

void test_adaptive_non_finite() {
  double x[1] = {1.0};
  CHECK_THROW(chaosys::integrate_adaptive<chaosys::DormandPrinceMethod>(
                  NaNSystem{}, x, 1.0),
              std::runtime_error);
  CHECK(std::isfinite(x[0]));
}

void test_adaptive_small_step() {
  double x[1] = {1.0};
  const auto report = chaosys::integrate_adaptive<chaosys::DormandPrinceMethod>(
      Decay{}, x, 1e-12, 1e-10, 1e-12, 1e-15, 1e-3);
  CHECK(report.accepted > 0);
  CHECK(report.h_last > 0.0);
  CHECK(std::abs(x[0] - std::exp(-1e-12)) < 1e-18);
}

void test_lyapunov_large_state() {
  double x[1] = {1e10};
  const auto report = chaosys::largest_lyapunov_report(
      SlowDecay{}, x, 0.01, 10, 10, 100, 1e-4);
  CHECK(std::isfinite(report.lambda));
  CHECK(std::abs(report.lambda + 1e-3) < 1e-6);
}

void test_max_diff_nan() {
  chaosys::Ensemble a, b;
  chaosys::init_ensemble(a, 8);
  chaosys::init_ensemble(b, 8);
  a.x[0] = std::numeric_limits<double>::quiet_NaN();
  b.x[0] = 0.0;
  const chaosys::State diff = chaosys::max_diff(a, b);
  CHECK(std::isnan(diff.x));
}

void test_poincare_timing() {
  Harmonic sys;
  constexpr double dt = 0.1;
  double x[2] = {1.0, 0.0};
  const auto rising = chaosys::poincare_section(sys, x, dt, 0, 2000, 100, 1,
                                                0.0, +1);
  CHECK(!rising.empty());
  CHECK(std::abs(rising.front().t - std::acos(-1.0)) < 1e-2);

  // The early-break path must leave the returned state at the last processed
  // step (s + 1), not one step behind.
  double y[2] = {1.0, 0.0};
  const auto first = chaosys::poincare_section(sys, y, dt, 0, 2000, 1, 1, 0.0,
                                               +1);
  CHECK(first.size() == 1);
  const std::size_t processed =
      static_cast<std::size_t>(std::floor(first[0].t / dt)) + 1;
  double direct[2] = {1.0, 0.0};
  for (std::size_t i = 0; i < processed; ++i) {
    chaosys::Rk4Method::step(sys, direct, dt);
  }
  CHECK(std::abs(y[0] - direct[0]) < 1e-12);
  CHECK(std::abs(y[1] - direct[1]) < 1e-12);
}

void test_dp5_fixed_order() {
  auto error = [](double dt) {
    double x[1] = {1.0};
    std::array<double, 1> x5{};
    const long steps = std::lround(1.0 / dt);
    for (long i = 0; i < steps; ++i) {
      chaosys::DormandPrinceMethod::step(Decay{}, x, dt, x5.data(), 1e-12,
                                         1e-14);
      x[0] = x5[0];
    }
    return std::abs(x[0] - std::exp(-1.0));
  };
  const double e1 = error(0.1);
  const double e2 = error(0.05);
  const double e3 = error(0.025);
  CHECK(e1 / e2 > 25.0 && e1 / e2 < 45.0);
  CHECK(e2 / e3 > 25.0 && e2 / e3 < 45.0);
}

void test_horizon_validation() {
  double x[1] = {1.0};
  CHECK_THROW(chaosys::integrate<chaosys::Rk4Method>(Decay{}, x, 10, 1e308),
              std::invalid_argument);
}

void test_permutation_entropy_validation() {
  std::vector<double> bad = {0.0, 1.0, std::numeric_limits<double>::quiet_NaN(),
                             3.0, 4.0};
  CHECK_THROW(chaosys::permutation_entropy(bad, 2, 1),
              std::invalid_argument);
  bad[2] = std::numeric_limits<double>::infinity();
  CHECK_THROW(chaosys::permutation_entropy(bad, 2, 1),
              std::invalid_argument);
}

void test_csv() {
  std::ostringstream states;
  std::vector<chaosys::State> traj = {{1.0, 2.0, 3.0}, {4.5, 5.5, 6.5}};
  chaosys::write_csv(states, traj);
  CHECK(states.str().rfind("x,y,z\n1,2,3\n4.5,5.5,6.5\n", 0) == 0);

  std::ostringstream table;
  std::vector<std::vector<double>> rows = {{1.0, 2.0}, {3.5, 4.5}};
  chaosys::write_csv(table, rows, {"a,b", "c\"d"});
  CHECK(table.str().rfind("\"a,b\",\"c\"\"d\"\n1,2\n3.5,4.5\n", 0) == 0);

  std::ostringstream series;
  chaosys::write_csv(series, std::vector<double>{0.5, 1.5});
  CHECK(series.str().rfind("value\n0.5\n1.5\n", 0) == 0);

  chaosys::LorenzSystem lorenz{};
  double x[3] = {1.0, 1.0, 1.0};
  const auto points =
      chaosys::poincare_section(lorenz, x, 0.005, 1000, 5000, 3, 2, 20.0, 1);
  std::ostringstream section;
  chaosys::write_csv(section, points);
  CHECK(section.str().rfind("t,x0,x1,x2\n", 0) == 0);

  std::ostringstream bad_header;
  CHECK_THROW(chaosys::write_csv(bad_header, rows, {"only-one"}),
              std::invalid_argument);
}

void test_poincare_direction() {
  Harmonic sys;
  constexpr double dt = 0.01;
  double rising_state[2] = {1.0, 0.0};
  double falling_state[2] = {1.0, 0.0};
  double both_state[2] = {1.0, 0.0};
  const auto rising = chaosys::poincare_section(
      sys, rising_state, dt, 0, 20000, 1000, 1, 0.0, +1);
  const auto falling = chaosys::poincare_section(
      sys, falling_state, dt, 0, 20000, 1000, 1, 0.0, -1);
  const auto both = chaosys::poincare_section(
      sys, both_state, dt, 0, 20000, 1000, 1, 0.0, 0);
  CHECK(!rising.empty() && !falling.empty());
  CHECK(both.size() == rising.size() + falling.size());
  CHECK(std::abs(rising.front().t - std::acos(-1.0)) < 0.02);
}

void test_has_initial_state() {
  static_assert(chaosys::HasInitialState<chaosys::LorenzSystem>);
  static_assert(chaosys::HasInitialState<chaosys::RosslerSystem>);
  static_assert(chaosys::HasInitialState<chaosys::ChenSystem>);
  static_assert(chaosys::HasInitialState<chaosys::DuffingSystem>);
  const auto state = chaosys::LorenzSystem::initial_state();
  CHECK(state[0] == 1.0 && state[1] == 1.0 && state[2] == 1.0);
}

void test_poincare() {
  chaosys::LorenzSystem sys{};
  double x[3] = {1.0, 1.0, 1.0};
  const auto pts = chaosys::poincare_section(sys, x, 0.005, 2000, 20000, 100,
                                             2, 20.0, +1);
  CHECK(pts.size() > 10);
  bool ok = true;
  for (const auto& p : pts) {
    ok = ok && std::isfinite(p.x[0]) && std::isfinite(p.x[1]) &&
         std::isfinite(p.x[2]);
    ok = ok && std::abs(p.x[2] - 20.0) < 1e-9;
    ok = ok && p.t > 0.0;
  }
  CHECK(ok);

  double y[3] = {1.0, 1.0, 1.0};
  CHECK_THROW(chaosys::poincare_section(sys, y, 0.005, 0, 100, 10, 5, 20.0, 1),
              std::invalid_argument);
  CHECK_THROW(chaosys::poincare_section(sys, y, 0.005, 0, 100, 10, 2, 20.0, 2),
              std::invalid_argument);
  double z[3] = {1.0, 1.0, 1.0};
  CHECK_THROW(chaosys::largest_lyapunov(sys, z, 0.0, 0, 10, 10),
              std::invalid_argument);
  CHECK_THROW(chaosys::largest_lyapunov(sys, z, 0.01, 0, 0, 10),
              std::invalid_argument);
}

void test_chaotic_systems() {
  chaosys::RosslerSystem rossler{};
  double x[3] = {1.0, 1.0, 1.0};
  chaosys::integrate<chaosys::Rk4Method>(rossler, x, 10000, 0.01);
  CHECK(std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]));

  double xl[3] = {1.0, 1.0, 1.0};
  const double rossler_lambda =
      chaosys::largest_lyapunov(rossler, xl, 0.01, 2000, 50, 200);
  CHECK(rossler_lambda > 0.0 && rossler_lambda < 0.5);

  chaosys::ChenSystem chen{};
  double y[3] = {1.0, 1.0, 1.0};
  chaosys::integrate<chaosys::Rk4Method>(chen, y, 10000, 0.01);
  CHECK(std::isfinite(y[0]) && std::isfinite(y[1]) && std::isfinite(y[2]));

  double yl[3] = {1.0, 1.0, 1.0};
  const double chen_lambda =
      chaosys::largest_lyapunov(chen, yl, 0.01, 2000, 50, 200);
  CHECK(chen_lambda > 0.5 && chen_lambda < 3.0);

  chaosys::LorenzSystem ls{};
  CHECK(ls.divergence(x) < 0.0);
  CHECK(std::abs(rossler.divergence(x) - (0.2 + x[0] - 5.7)) < 1e-12);
  CHECK(std::abs(chen.divergence(y) - (28.0 - 35.0 - 3.0)) < 1e-12);

  CHECK_THROW(
      (chaosys::RosslerSystem{
           chaosys::RosslerParams{std::numeric_limits<double>::quiet_NaN(), 0.2,
                                 5.7}}
           .validate()),
      std::invalid_argument);
  CHECK_THROW(
      (chaosys::ChenSystem{
           chaosys::ChenParams{35.0, 3.0,
                              std::numeric_limits<double>::infinity()}}
           .validate()),
      std::invalid_argument);
}

void test_chaos_metrics() {
  std::vector<double> ramp(10);
  for (int i = 0; i < 10; ++i) ramp[i] = i;
  const auto embedded = chaosys::embed(ramp, 2, 1);
  CHECK(embedded.size() == 9);
  CHECK(embedded.front()[0] == 0.0 && embedded.front()[1] == 1.0);
  CHECK(embedded.back()[0] == 8.0 && embedded.back()[1] == 9.0);

  chaosys::LorenzSystem lorenz{};
  double x[3] = {1.0, 1.0, 1.0};
  chaosys::integrate<chaosys::Rk4Method>(lorenz, x, 1000, 0.01);
  std::vector<double> series;
  series.reserve(5000);
  for (int i = 0; i < 5000; ++i) {
    chaosys::Rk4Method::step(lorenz, x, 0.01);
    series.push_back(x[0]);
  }
  const auto corr =
      chaosys::correlation_dimension_series(series, 4, 20, 1500);
  CHECK(corr.dimension > 1.4 && corr.dimension < 2.4);
  CHECK(corr.log_r.size() == 24 && corr.log_c.size() == 24);
  CHECK(corr.embedding_dim == 4 && corr.stride >= 1 && corr.points > 0);

  std::vector<double> line(2000);
  for (int i = 0; i < 2000; ++i) line[i] = i;
  const auto line_dim = chaosys::correlation_dimension_series(line, 2, 1, 1000);
  CHECK(line_dim.dimension > 0.8 && line_dim.dimension < 1.3);

  std::vector<std::vector<double>> plane;
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      plane.push_back({static_cast<double>(i), static_cast<double>(j)});
    }
  }
  const auto plane_dim = chaosys::correlation_dimension(plane, 1000);
  CHECK(plane_dim.dimension > 1.7 && plane_dim.dimension < 2.3);

  CHECK_THROW(chaosys::embed(ramp, 2, 20), std::invalid_argument);
}

void test_permutation_entropy() {
  std::vector<double> ramp(100);
  for (int i = 0; i < 100; ++i) ramp[i] = i;
  CHECK(chaosys::permutation_entropy(ramp) == 0.0);

  chaosys::LorenzSystem lorenz{};
  double x[3] = {1.0, 1.0, 1.0};
  chaosys::integrate<chaosys::Rk4Method>(lorenz, x, 1000, 0.01);
  std::vector<double> series;
  for (int i = 0; i < 4000; ++i) {
    chaosys::Rk4Method::step(lorenz, x, 0.01);
    series.push_back(x[0]);
  }
  const double h = chaosys::permutation_entropy(series, 3, 1);
  CHECK(h > 0.2 && h <= 1.0);
  CHECK_THROW(chaosys::permutation_entropy(series, 1, 1),
              std::invalid_argument);
}

void test_time_systems() {
  chaosys::DuffingSystem duffing{};
  double x[2] = {1.0, 1.0};
  chaosys::integrate_time<chaosys::Rk4Method>(duffing, x, 0.0, 10000, 0.01);
  CHECK(std::isfinite(x[0]) && std::isfinite(x[1]));

  double y[2] = {1.0, 1.0};
  const double lambda =
      chaosys::largest_lyapunov_time(duffing, y, 0.0, 0.01, 1000, 50, 100);
  CHECK(lambda > 0.0 && lambda < 0.5);

  double z[2] = {1.0, 1.0};
  chaosys::integrate_adaptive_time(duffing, z, 0.0, 10.0, 1e-9, 1e-11);
  CHECK(std::isfinite(z[0]) && std::isfinite(z[1]));
}

void test_generic_validation() {
  const chaosys::Params zero_beta{10.0, 28.0, 0.0};
  chaosys::LorenzSystem bad{zero_beta};
  double x[3] = {1.0, 1.0, 1.0};
  CHECK_THROW(chaosys::integrate<chaosys::Rk4Method>(bad, x, 10, 0.01),
              std::invalid_argument);

  chaosys::LorenzSystem good{};
  chaosys::EnsembleN<chaosys::LorenzSystem> ens(4);
  ens.component(0)[0] = std::nan("");
  CHECK_THROW(
      chaosys::integrate_ensemble<chaosys::Rk4Method>(good, ens, 10, 0.01),
      std::invalid_argument);
}

void test_bounds() {
  chaosys::State s{1, 1, 1};
  chaosys::Params p{};
  bool ok = true;
  for (int k = 0; k < 100; ++k) {
    chaosys::rk4_scalar(s, 1000, 0.01, p);
    ok = ok && std::isfinite(s.x) && std::isfinite(s.y) && std::isfinite(s.z);
    ok = ok && std::abs(s.x) < 30.0 && std::abs(s.y) < 35.0;
    ok = ok && s.z > -5.0 && s.z < 60.0;
  }
  CHECK(ok);
  CHECK(chaosys::max_threads() >= 1);
}

}  // namespace

int main() {
  test_params_validation();
  test_deriv_and_jacobian();
  test_known_rk4_state();
  test_generic_decay();
  test_generic_harmonic();
  test_generic_float();
  test_generic_matches_specialized();
  test_runtime_params();
  test_ensemble_accuracy_and_threads();
  test_ensemble_params();
  test_generic_ensemble();
  test_stride_semantics();
  test_edge_cases();
  test_adaptive_decay();
  test_convergence_orders();
  test_adaptive_harmonic();
  test_adaptive_validation();
  test_generic_ensemble_threads();
  test_lyapunov();
  test_lyapunov_report_and_spectrum();
  test_adaptive_non_finite();
  test_adaptive_small_step();
  test_lyapunov_large_state();
  test_max_diff_nan();
  test_poincare_timing();
  test_dp5_fixed_order();
  test_horizon_validation();
  test_permutation_entropy_validation();
  test_csv();
  test_poincare_direction();
  test_has_initial_state();
  test_poincare();
  test_chaotic_systems();
  test_chaos_metrics();
  test_permutation_entropy();
  test_time_systems();
  test_generic_validation();
  test_bounds();

  if (g_failures == 0) {
    std::printf("PASS  %d checks\n", g_checks);
    return 0;
  }
  std::printf("FAIL  %d/%d checks failed\n", g_failures, g_checks);
  return 1;
}

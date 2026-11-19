/* C ABI smoke test: compiled as C11, linked against libchaosys. */
#include "chaosys_c.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int checks = 0;
static int failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    ++checks;                                                                \
    if (!(cond)) {                                                           \
      ++failures;                                                            \
      fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
    }                                                                        \
  } while (0)

/* Numerical fixture only; the product scope is chaotic systems. */
static void harmonic(const double* x, double* dx, void* user) {
  (void)user;
  dx[0] = x[1];
  dx[1] = -x[0];
}

void duffing_time(double t, const double* x, double* dx, void* user) {
  (void)user;
  dx[0] = x[1];
  dx[1] = -0.2 * x[1] + x[0] - x[0] * x[0] * x[0] + 0.3 * cos(t);
}

void rossler(const double* x, double* dx, void* user) {
  (void)user;
  dx[0] = -x[1] - x[2];
  dx[1] = x[0] + 0.2 * x[1];
  dx[2] = 0.2 + x[2] * (x[0] - 5.7);
}

/* Diverging fixture: 200 RK4 steps of dx/dt = x^2 overflow a double. */
static void blowup(const double* x, double* dx, void* user) {
  (void)user;
  dx[0] = x[0] * x[0];
}

static void blowup_time(double t, const double* x, double* dx, void* user) {
  (void)t;
  (void)user;
  dx[0] = x[0] * x[0];
}

/* True largest exponent is -1; perturbing the time lane would bias it to 0. */
static void forced_decay_time(double t, const double* x, double* dx,
                              void* user) {
  (void)user;
  dx[0] = -x[0] + cos(t);
}

int main(void) {
  const chaosys_params p = chaosys_default_params();
  CHECK(p.sigma == 10.0 && p.rho == 28.0);
  CHECK(chaosys_version() != NULL);
  CHECK(chaosys_status_string(CHAOSYS_OK) != NULL);
  CHECK(chaosys_status_string(CHAOSYS_ERR_ARGUMENT) != NULL);

  chaosys_state s = {1.0, 1.0, 1.0};
  CHECK(chaosys_rk4(&s, 1000, 0.01, p) == CHAOSYS_OK);
  CHECK(fabs(s.x - (-4.902819483748812)) < 1e-9);
  CHECK(fabs(s.y - (-3.743407675271541)) < 1e-9);
  CHECK(fabs(s.z - 24.691885987964351) < 1e-9);

  chaosys_state a = {1.0, 1.0, 1.0};
  size_t accepted = 0;
  CHECK(chaosys_adaptive(&a, 10.0, 1e-10, 1e-12, p, &accepted) == CHAOSYS_OK);
  CHECK(accepted > 0);
  CHECK(isfinite(a.x) && isfinite(a.y) && isfinite(a.z));

  enum { N = 64 };
  chaosys_state batch[N];
  for (int i = 0; i < N; ++i) {
    batch[i].x = 1.0 + 0.01 * i;
    batch[i].y = 1.0;
    batch[i].z = 1.0;
  }
  const chaosys_state first = batch[0];
  CHECK(chaosys_ensemble(batch, N, 10, 0.01, 2, p) == CHAOSYS_OK);
  chaosys_state ref = first;
  CHECK(chaosys_rk4(&ref, 10, 0.01, p) == CHAOSYS_OK);
  CHECK(fabs(batch[0].x - ref.x) < 1e-9);
  CHECK(fabs(batch[0].y - ref.y) < 1e-9);
  CHECK(fabs(batch[0].z - ref.z) < 1e-9);

  double lambda = 0.0;
  const chaosys_state x0 = {1.0, 1.0, 1.0};
  CHECK(chaosys_lyapunov(&x0, 0.005, 2000, 50, 200, p, &lambda) == CHAOSYS_OK);
  CHECK(lambda > 0.7 && lambda < 1.1);

  double segment_lambdas[200];
  chaosys_lyapunov_stats stats;
  CHECK(chaosys_lyapunov_report(&x0, 0.005, 2000, 50, 200, p,
                                segment_lambdas, 200, &stats) == CHAOSYS_OK);
  CHECK(stats.segments == 200);
  CHECK(stats.lambda > 0.7 && stats.lambda < 1.1);
  CHECK(isfinite(stats.stddev) && stats.stddev >= 0.0);
  CHECK(isfinite(stats.stderr_value) && stats.stderr_value >= 0.0);
  CHECK(stats.converged == 0 || stats.converged == 1);
  CHECK(chaosys_lyapunov_report(&x0, 0.005, 2000, 50, 200, p,
                                segment_lambdas, 199, &stats) ==
        CHAOSYS_ERR_ARGUMENT);

  double spectrum[3];
  size_t spectrum_count = 0;
  CHECK(chaosys_lyapunov_spectrum(&x0, 0.005, 2000, 50, 200, p, spectrum, 3,
                                  &spectrum_count) == CHAOSYS_OK);
  CHECK(spectrum_count == 3);
  CHECK(spectrum[0] > spectrum[1] && spectrum[1] > spectrum[2]);
  CHECK(isfinite(spectrum[0]) && fabs(spectrum[0] - 0.8915) < 0.1);
  CHECK(fabs(spectrum[0] + spectrum[1] + spectrum[2] + 13.6666667) < 0.1);
  CHECK(chaosys_lyapunov_spectrum(&x0, 0.005, 2000, 50, 200, p, spectrum, 2,
                                  &spectrum_count) == CHAOSYS_ERR_ARGUMENT);

  chaosys_state section[16];
  size_t count = 0;
  CHECK(chaosys_poincare(&x0, 0.005, 1000, 10000, p, 2, 20.0, 1, section, 16,
                         &count) == CHAOSYS_OK);
  CHECK(count > 0 && count <= 16);
  for (size_t i = 0; i < count; ++i) {
    CHECK(fabs(section[i].z - 20.0) < 1e-9);
  }

  /* Rössler through the generic C ABI. */
  void rossler(const double* x, double* dx, void* user);
  const chaosys_system rsys = {3, rossler, NULL};
  const double rx0[3] = {1.0, 1.0, 1.0};
  double rlam = 0.0;
  CHECK(chaosys_lyapunov_system(&rsys, rx0, 0.01, 1000, 50, 100, &rlam) ==
        CHAOSYS_OK);
  CHECK(rlam > 0.0 && rlam < 0.5);

  double rsec[8 * 3];
  size_t rcount = 0;
  CHECK(chaosys_poincare_system(&rsys, rx0, 0.01, 1000, 20000, 2, 5.0, 1,
                                rsec, 8, &rcount) == CHAOSYS_OK);
  CHECK(rcount > 0 && rcount <= 8);
  for (size_t i = 0; i < rcount; ++i) {
    CHECK(fabs(rsec[i * 3 + 2] - 5.0) < 1e-9);
  }

  double rbatch[4 * 3] = {1.0, 1.0, 1.0, 1.1, 1.0, 1.0,
                          0.9, 1.0, 1.0, 1.0, 1.1, 1.0};
  CHECK(chaosys_ensemble_system(&rsys, rbatch, 4, 10, 0.01, 2) == CHAOSYS_OK);
  for (size_t i = 0; i < 4 * 3; ++i) CHECK(isfinite(rbatch[i]));

  /* Takens embedding on a linear ramp and correlation dimension on Lorenz. */
  const double ramp[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  double embedded[9 * 2];
  size_t embedded_count = 0;
  CHECK(chaosys_embed(ramp, 10, 2, 1, embedded, 18, &embedded_count) ==
        CHAOSYS_OK);
  CHECK(embedded_count == 9);
  CHECK(fabs(embedded[0] - 0.0) < 1e-12 && fabs(embedded[1] - 1.0) < 1e-12);
  CHECK(fabs(embedded[16] - 8.0) < 1e-12 &&
        fabs(embedded[17] - 9.0) < 1e-12);
  CHECK(chaosys_embed(ramp, 10, 2, 1, embedded, 17, &embedded_count) ==
        CHAOSYS_ERR_ARGUMENT);
  CHECK(chaosys_embed(ramp, 10, 2, 1, embedded, 9, &embedded_count) ==
        CHAOSYS_ERR_ARGUMENT);

  chaosys_state ls = {1.0, 1.0, 1.0};
  CHECK(chaosys_rk4(&ls, 1000, 0.01, p) == CHAOSYS_OK);
  double lseries[4000];
  int series_ok = 1;
  for (int i = 0; i < 4000; ++i) {
    series_ok = series_ok && (chaosys_rk4(&ls, 1, 0.01, p) == CHAOSYS_OK);
    lseries[i] = ls.x;
  }
  CHECK(series_ok);
  double corr_dim = 0.0;
  CHECK(chaosys_correlation_dimension(lseries, 4000, 4, 20, 1500,
                                      &corr_dim) == CHAOSYS_OK);
  CHECK(corr_dim > 1.4 && corr_dim < 2.4);
  CHECK(chaosys_correlation_dimension(NULL, 4000, 4, 20, 1500, &corr_dim) ==
        CHAOSYS_ERR_ARGUMENT);
  double entropy = 0.0;
  CHECK(chaosys_permutation_entropy(lseries, 4000, 3, 1, &entropy) ==
        CHAOSYS_OK);
  CHECK(entropy > 0.2 && entropy <= 1.0);
  CHECK(chaosys_permutation_entropy(NULL, 10, 3, 1, &entropy) ==
        CHAOSYS_ERR_ARGUMENT);

  /* Time-dependent Duffing system. */
  const chaosys_time_system tsys = {2, duffing_time, NULL};
  double tx[2] = {1.0, 1.0};
  CHECK(chaosys_rk4_time_system(&tsys, tx, 0.0, 10000, 0.01) == CHAOSYS_OK);
  CHECK(isfinite(tx[0]) && isfinite(tx[1]));

  double ty[2] = {1.0, 1.0};
  double tlambda = 0.0;
  CHECK(chaosys_lyapunov_time_system(&tsys, ty, 0.0, 0.01, 1000, 50, 100,
                                     &tlambda) == CHAOSYS_OK);
  CHECK(tlambda > 0.0 && tlambda < 0.5);

  const chaosys_time_system forced = {1, forced_decay_time, NULL};
  double fx[1] = {1.0};
  double flambda = 0.0;
  CHECK(chaosys_lyapunov_time_system(&forced, fx, 0.0, 0.01, 1000, 50, 100,
                                     &flambda) == CHAOSYS_OK);
  CHECK(fabs(flambda + 1.0) < 0.01);

  double tz[2] = {1.0, 1.0};
  size_t taccepted = 0;
  CHECK(chaosys_adaptive_time_system(&tsys, tz, 0.0, 10.0, 1e-9, 1e-11,
                                     &taccepted) == CHAOSYS_OK);
  CHECK(taccepted > 0);
  CHECK(isfinite(tz[0]) && isfinite(tz[1]));

  const chaosys_system sys = {2, harmonic, NULL};
  double osc[2] = {1.0, 0.0};
  CHECK(chaosys_rk4_system(&sys, osc, 10000, 0.001) == CHAOSYS_OK);
  const double energy = 0.5 * (osc[0] * osc[0] + osc[1] * osc[1]);
  CHECK(fabs(energy - 0.5) < 1e-6);

  double osc_dp[2] = {1.0, 0.0};
  size_t dp_steps = 0;
  CHECK(chaosys_adaptive_system(&sys, osc_dp, 10.0, 1e-10, 1e-12,
                                &dp_steps) == CHAOSYS_OK);
  CHECK(dp_steps > 0);
  const double dp_energy =
      0.5 * (osc_dp[0] * osc_dp[0] + osc_dp[1] * osc_dp[1]);
  CHECK(fabs(dp_energy - 0.5) < 1e-6);

  /* Generic Lyapunov report and spectrum for the harmonic fixture. */
  const chaosys_system hsys = {2, harmonic, NULL};
  double hx[2] = {1.0, 0.0};
  double hseg[8];
  chaosys_lyapunov_stats hstats;
  CHECK(chaosys_lyapunov_report_system(&hsys, hx, 0.01, 100, 10, 8, hseg, 8,
                                       &hstats) == CHAOSYS_OK);
  CHECK(hstats.segments == 8);
  CHECK(fabs(hstats.lambda) < 0.1);
  double hspec[2];
  size_t hspec_count = 0;
  CHECK(chaosys_lyapunov_spectrum_system(&hsys, hx, 0.001, 1000, 50, 100,
                                         hspec, 2, &hspec_count) == CHAOSYS_OK);
  CHECK(hspec_count == 2);
  CHECK(fabs(hspec[0]) < 0.05 && fabs(hspec[1]) < 0.05);
  CHECK(chaosys_lyapunov_spectrum_system(&hsys, hx, 0.001, 1000, 50, 100,
                                         hspec, 1, &hspec_count) ==
        CHAOSYS_ERR_ARGUMENT);

  /* A diverging callback must not return CHAOSYS_OK with inf. */
  const chaosys_system bang = {1, blowup, NULL};
  double bang_x = 1.0;
  CHECK(chaosys_rk4_system(&bang, &bang_x, 200, 1.0) ==
        CHAOSYS_ERR_NUMERIC);
  CHECK(isfinite(bang_x) && bang_x == 1.0);
  const chaosys_time_system bang_t = {1, blowup_time, NULL};
  double bang_tx = 1.0;
  CHECK(chaosys_rk4_time_system(&bang_t, &bang_tx, 0.0, 200, 1.0) ==
        CHAOSYS_ERR_NUMERIC);

  /* Empty batches return OK after validating the common arguments. */
  CHECK(chaosys_ensemble(NULL, 0, 10, 0.01, 1, p) == CHAOSYS_OK);
  CHECK(chaosys_ensemble(NULL, 0, 10, NAN, 1, p) == CHAOSYS_ERR_ARGUMENT);
  CHECK(chaosys_ensemble_system(&rsys, NULL, 0, 10, 0.01, 1) == CHAOSYS_OK);

  /* Dimension bounds. */
  const chaosys_system zero_dim = {0, harmonic, NULL};
  const chaosys_system big_dim = {65, harmonic, NULL};
  double dim_x[2] = {1.0, 0.0};
  CHECK(chaosys_rk4_system(&zero_dim, dim_x, 1, 0.01) ==
        CHAOSYS_ERR_ARGUMENT);
  CHECK(chaosys_rk4_system(&big_dim, dim_x, 1, 0.01) ==
        CHAOSYS_ERR_ARGUMENT);
  const chaosys_time_system big_time_dim = {64, duffing_time, NULL};
  CHECK(chaosys_rk4_time_system(&big_time_dim, dim_x, 0.0, 1, 0.01) ==
        CHAOSYS_ERR_ARGUMENT);

  /* Unreasonable capacity is a memory/allocation failure, not a crash. */
  chaosys_state one_point;
  size_t one_count = 0;
  CHECK(chaosys_poincare(&x0, 0.005, 0, 1, p, 2, 20.0, 1, &one_point,
                         (size_t)-1, &one_count) == CHAOSYS_ERR_MEMORY);

  CHECK(chaosys_rk4(NULL, 1, 0.01, p) == CHAOSYS_ERR_ARGUMENT);
  chaosys_state bad = {1.0, 1.0, 1.0};
  CHECK(chaosys_rk4(&bad, 10, 0.0, p) == CHAOSYS_ERR_ARGUMENT);
  CHECK(chaosys_ensemble(NULL, 4, 10, 0.01, 1, p) == CHAOSYS_ERR_ARGUMENT);
  double unused = 0.0;
  CHECK(chaosys_lyapunov(NULL, 0.01, 0, 10, 10, p, &unused) ==
        CHAOSYS_ERR_ARGUMENT);
  CHECK(chaosys_poincare(&bad, 0.01, 0, 10, p, 2, 20.0, 1, section, 0,
                         &count) == CHAOSYS_ERR_ARGUMENT);
  CHECK(chaosys_rk4_system(NULL, osc, 1, 0.01) == CHAOSYS_ERR_ARGUMENT);
  CHECK(chaosys_adaptive_system(NULL, osc, 1.0, 1e-9, 1e-11, NULL) ==
        CHAOSYS_ERR_ARGUMENT);
  CHECK(chaosys_lyapunov_system(NULL, rx0, 0.01, 0, 10, 10, &rlam) ==
        CHAOSYS_ERR_ARGUMENT);
  CHECK(chaosys_poincare_system(&rsys, rx0, 0.01, 0, 10, 2, 5.0, 1, rsec, 0,
                                &rcount) == CHAOSYS_ERR_ARGUMENT);
  CHECK(chaosys_ensemble_system(&rsys, rbatch, 0, 10, 0.01, 1) ==
        CHAOSYS_OK);
  CHECK(chaosys_rk4_time_system(NULL, tx, 0.0, 10, 0.01) ==
        CHAOSYS_ERR_ARGUMENT);

  if (failures == 0) {
    printf("PASS  %d C ABI checks\n", checks);
    return 0;
  }
  printf("FAIL  %d/%d C ABI checks failed\n", failures, checks);
  return 1;
}

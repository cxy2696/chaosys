/* Stable C ABI for chaotic ODE solvers.
 *
 * The C++ core is templated; this header exposes the common runtime paths so
 * the library can be used from C, Python (ctypes), Rust or any FFI host.
 * All functions are thread-safe (no mutable global state) and never let a C++
 * exception cross the boundary: failures are reported as chaosys_status.
 */
#ifndef CHAOSYS_C_H
#define CHAOSYS_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CHAOSYS_VERSION_STRING "1.0.0"

typedef enum chaosys_status {
  CHAOSYS_OK = 0,
  CHAOSYS_ERR_ARGUMENT = 1,
  CHAOSYS_ERR_NUMERIC = 2,
  CHAOSYS_ERR_MEMORY = 3,
  CHAOSYS_ERR_INTERNAL = 4
} chaosys_status;

typedef struct chaosys_params {
  double sigma;
  double rho;
  double beta;
} chaosys_params;

typedef struct chaosys_state {
  double x;
  double y;
  double z;
} chaosys_state;

/* Runtime-defined chaotic ODE: dx/dt = deriv(x, dx, user), dim components. */
typedef void (*chaosys_deriv_fn)(const double* x, double* dx, void* user);

typedef struct chaosys_system {
  int dim;
  chaosys_deriv_fn deriv;
  void* user;
} chaosys_system;

/* Time-dependent chaotic ODE: dx/dt = deriv(t, x, dx, user). */
typedef void (*chaosys_time_deriv_fn)(double t, const double* x, double* dx,
                                      void* user);

typedef struct chaosys_time_system {
  int dim;
  chaosys_time_deriv_fn deriv;
  void* user;
} chaosys_time_system;

chaosys_params chaosys_default_params(void);
const char* chaosys_version(void);
const char* chaosys_status_string(chaosys_status status);

/* Fixed-step classical RK4, in place. */
chaosys_status chaosys_rk4(chaosys_state* state, size_t steps, double dt,
                           chaosys_params p);

/* Adaptive Dormand-Prince 5(4) from t=0 to t_end. */
chaosys_status chaosys_adaptive(chaosys_state* state, double t_end, double rtol,
                                double atol, chaosys_params p,
                                size_t* accepted_steps);

/* Batch of n independent states, same dt/steps/params.  threads is clamped
 * to [1, 64]; n == 0 validates dt/params and returns CHAOSYS_OK. */
chaosys_status chaosys_ensemble(chaosys_state* states, size_t n, size_t steps,
                                double dt, int threads, chaosys_params p);

/* Largest Lyapunov exponent from x0. */
chaosys_status chaosys_lyapunov(const chaosys_state* x0, double dt,
                                size_t transient_steps, size_t segment_steps,
                                size_t segments, chaosys_params p,
                                double* lambda);

/* Per-segment largest-Lyapunov estimates.  `segment_lambdas` receives one
 * value per segment (capacity >= segments); `report->segments` is the number
 * written.  `converged` is advisory: segments >= 8 and the standard error is
 * at most 10% of max(1, |lambda|). */
typedef struct chaosys_lyapunov_stats {
  double lambda;        /* mean exponent, 1/time */
  double stddev;        /* sample standard deviation of segments */
  double stderr_value;  /* standard error of the mean */
  int converged;        /* advisory flag, see comment above */
  size_t segments;      /* number of segment estimates written */
} chaosys_lyapunov_stats;

chaosys_status chaosys_lyapunov_report(const chaosys_state* x0, double dt,
                                       size_t transient_steps,
                                       size_t segment_steps, size_t segments,
                                       chaosys_params p,
                                       double* segment_lambdas,
                                       size_t capacity,
                                       chaosys_lyapunov_stats* report);

chaosys_status chaosys_lyapunov_report_system(
    const chaosys_system* system, const double* x0, double dt,
    size_t transient_steps, size_t segment_steps, size_t segments,
    double* segment_lambdas, size_t capacity,
    chaosys_lyapunov_stats* report);

/* Full Lyapunov spectrum, descending.  `values` receives `count` doubles;
 * capacity must be at least 3 for Lorenz and at least system->dim for a
 * custom autonomous system. */
chaosys_status chaosys_lyapunov_spectrum(const chaosys_state* x0, double dt,
                                         size_t transient_steps,
                                         size_t segment_steps,
                                         size_t segments, chaosys_params p,
                                         double* values, size_t capacity,
                                         size_t* count);

chaosys_status chaosys_lyapunov_spectrum_system(
    const chaosys_system* system, const double* x0, double dt,
    size_t transient_steps, size_t segment_steps, size_t segments,
    double* values, size_t capacity, size_t* count);

/* Poincare section x[component] == level; at most capacity points.  `out`
 * must hold capacity * 3 doubles. */
chaosys_status chaosys_poincare(const chaosys_state* x0, double dt,
                                size_t transient_steps, size_t max_steps,
                                chaosys_params p, int component, double level,
                                int direction, chaosys_state* out,
                                size_t capacity, size_t* count);

/* Custom chaotic ODE solvers with a C callback.
 * dim must be in [1, 64]; extra lanes are kept at zero.  threads is clamped
 * to [1, 64]; n == 0 returns CHAOSYS_OK after validating the system and dt. */
chaosys_status chaosys_rk4_system(const chaosys_system* system, double* x,
                                  size_t steps, double dt);
chaosys_status chaosys_adaptive_system(const chaosys_system* system, double* x,
                                       double t_end, double rtol, double atol,
                                       size_t* accepted_steps);
chaosys_status chaosys_lyapunov_system(const chaosys_system* system,
                                       const double* x0, double dt,
                                       size_t transient_steps,
                                       size_t segment_steps, size_t segments,
                                       double* lambda);
chaosys_status chaosys_poincare_system(const chaosys_system* system,
                                       const double* x0, double dt,
                                       size_t transient_steps, size_t max_steps,
                                       int component, double level,
                                       int direction, double* out,
                                       size_t capacity, size_t* count);
chaosys_status chaosys_ensemble_system(const chaosys_system* system,
                                       double* states, size_t n, size_t steps,
                                       double dt, int threads);

/* The same integrators/analysis for time-dependent systems.  dim <= 63;
 * the last stack lane is reserved for time. */
chaosys_status chaosys_rk4_time_system(const chaosys_time_system* system,
                                       double* x, double t0, size_t steps,
                                       double dt);
chaosys_status chaosys_adaptive_time_system(const chaosys_time_system* system,
                                            double* x, double t0, double t_end,
                                            double rtol, double atol,
                                            size_t* accepted_steps);
chaosys_status chaosys_lyapunov_time_system(const chaosys_time_system* system,
                                            const double* x0, double t0,
                                            double dt, size_t transient_steps,
                                            size_t segment_steps,
                                            size_t segments, double* lambda);

/* Takens embedding of a scalar series and the Grassberger-Procaccia
 * correlation dimension.  embed writes count * dimension doubles; `capacity`
 * is the number of doubles available in `out`, so it must be at least
 * count * dimension. */
chaosys_status chaosys_embed(const double* series, size_t n, int dimension,
                             int delay, double* out, size_t capacity,
                             size_t* count);
chaosys_status chaosys_correlation_dimension(const double* series, size_t n,
                                             int dimension, int delay,
                                             size_t max_points, double* value);
chaosys_status chaosys_permutation_entropy(const double* series, size_t n,
                                           int order, int delay, double* value);

#ifdef __cplusplus
}
#endif

#endif /* CHAOSYS_C_H */

// Lorenz attractor: baseline and optimized kernels.
//
//   dx/dt = sigma * (y - x)
//   dy/dt = x * (rho - z) - y
//   dz/dt = x * y - beta * z
//
// Parameters are runtime values (chaosys/params.hpp); the generic ODE layer in
// chaosys/ode.hpp covers arbitrary systems and float/double.  The specialized
// ensemble kernels below intentionally keep their own fused inner loop.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "chaosys/aligned.hpp"
#include "chaosys/ode.hpp"
#include "chaosys/params.hpp"

namespace chaosys {

struct State {
  double x = 0.0, y = 0.0, z = 0.0;
};

// ---------------------------------------------------------------- baselines
// Idiomatic, allocation-heavy code of the kind found in tutorials: AoS state,
// a fresh std::vector of samples, push_back without reserve, scalar Euler/RK4.
void euler_naive(std::vector<State>& traj, State s0, std::size_t steps,
                 double dt, Params p = {});

void rk4_naive(std::vector<State>& traj, State s0, std::size_t steps,
               double dt, Params p = {});

// Ensemble baseline: one independent scalar RK4 chain per trajectory, no
// vectorisation across trajectories.  This is the "before" for the SoA kernels.
void rk4_ensemble_naive(State* states, std::size_t n, std::size_t steps,
                        double dt, Params p = {});

// -------------------------------------------------------------- scalar path
// Single-trajectory, in-place, no allocation.  Invalid arguments (non-finite
// parameters, dt <= 0, null output, stride == 0) throw std::invalid_argument.
void euler_scalar(State& s, std::size_t steps, double dt, Params p = {});
void rk4_scalar(State& s, std::size_t steps, double dt, Params p = {});

// Same, but stores one sample every `stride` steps (stride >= 1) into `out`.
// out[0] is the initial state, out[k] the state after k*stride steps.
void rk4_scalar_strided(State& s, std::size_t steps, double dt, State* out,
                        std::size_t stride, Params p = {});

// ------------------------------------------------------------- ensemble SoA
// Independent trajectories stored structure-of-arrays, 64-byte aligned.  The
// ensemble workload is embarrassingly parallel: one attractor per initial
// condition.
struct Ensemble {
  std::vector<double, AlignedAllocator<double, 64>> x, y, z;
  std::size_t n = 0;

  void resize(std::size_t count) {
    n = count;
    x.resize(count);
    y.resize(count);
    z.resize(count);
  }
  double* px() noexcept { return x.data(); }
  double* py() noexcept { return y.data(); }
  double* pz() noexcept { return z.data(); }
};

void init_ensemble(Ensemble& e, std::size_t n,
                   std::uint64_t seed = 0x9e3779b97f4a7c15ull);

// Plain SoA loop, scalar RK4 per lane.  The compiler may auto-vectorise it.
void rk4_soa_scalar(Ensemble& e, std::size_t steps, double dt, Params p = {});

// Explicit ARM NEON kernel (2 doubles / vector, 8 trajectories per iteration
// for instruction-level parallelism).  Falls back to scalar on other targets.
// Trajectories are processed in cache-sized blocks (default 512).
void rk4_soa_neon(Ensemble& e, std::size_t steps, double dt, Params p = {});
void rk4_soa_neon_blocked(Ensemble& e, std::size_t steps, double dt,
                          std::size_t block, Params p = {});

// Same kernel, split over `nthreads` workers (main thread included); each
// worker keeps a block of state in L1/L2 while marching in time.  Thread
// counts are clamped to [1, 64].
void rk4_soa_neon_threads(Ensemble& e, std::size_t steps, double dt,
                          int nthreads, Params p = {});

void rk4_soa_scalar_threads(Ensemble& e, std::size_t steps, double dt,
                            int nthreads, Params p = {});

// Element-wise max norm |a - b| over the ensemble (validation helper).
// Throws std::invalid_argument when the sizes differ.
State max_diff(const Ensemble& a, const Ensemble& b);

int max_threads();

// ---------------------------------------------------------------- reference
// Dormand-Prince 5(4) adaptive integrator used as the accuracy oracle.  It is
// the same generic integrator exposed in chaosys/ode.hpp, not a second copy.
inline State dp5_reference(State s0, double t_end, Params p, double rtol,
                           double atol) {
  double x[3] = {s0.x, s0.y, s0.z};
  integrate_adaptive<DormandPrinceMethod>(LorenzSystem{p}, x, t_end, rtol, atol);
  return State{x[0], x[1], x[2]};
}

inline State dp5_reference(State s0, double t_end, double rtol = 1e-12,
                           double atol = 1e-13) {
  return dp5_reference(s0, t_end, Params{}, rtol, atol);
}

}  // namespace chaosys

// ------------------------------------------------- fast-math translation unit
// Compiled with -ffast-math in a separate TU so the effect is measurable and
// does not contaminate the strictly IEEE-compiled kernels above.
extern "C" {
void chaosys_fast_rk4_scalar(chaosys::State* s, std::size_t steps, double dt,
                             chaosys::Params p);
void chaosys_fast_rk4_soa_neon(double* x, double* y, double* z, std::size_t n,
                               std::size_t steps, double dt, chaosys::Params p);
void chaosys_fast_rk4_soa_neon_threads(double* x, double* y, double* z,
                                       std::size_t n, std::size_t steps,
                                       double dt, int nthreads,
                                       chaosys::Params p);
}

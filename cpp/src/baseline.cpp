// Baseline implementations: deliberately written the way a first-cut
// implementation usually looks.  This TU is compiled with plain -O2 and no
// -mcpu=native so that "before" numbers reflect a default release build.
#include "chaosys.hpp"

#include <stdexcept>
#include <utility>

namespace chaosys {
namespace {

struct P {
  double x, y, z;
};

P lorenz(P p, const Params& prm) {
  P d;
  d.x = prm.sigma * (p.y - p.x);
  d.y = p.x * (prm.rho - p.z) - p.y;
  d.z = p.x * p.y - prm.beta * p.z;
  return d;
}

P axpy(P a, P b, double s) {
  return P{a.x + s * b.x, a.y + s * b.y, a.z + s * b.z};
}

P rk4_naive_step(P p, double dt, const Params& prm) {
  P k1 = lorenz(p, prm);
  P k2 = lorenz(axpy(p, k1, dt * 0.5), prm);
  P k3 = lorenz(axpy(p, k2, dt * 0.5), prm);
  P k4 = lorenz(axpy(p, k3, dt), prm);
  return P{p.x + dt / 6.0 * (k1.x + 2 * k2.x + 2 * k3.x + k4.x),
           p.y + dt / 6.0 * (k1.y + 2 * k2.y + 2 * k3.y + k4.y),
           p.z + dt / 6.0 * (k1.z + 2 * k2.z + 2 * k3.z + k4.z)};
}

}  // namespace

void euler_naive(std::vector<State>& traj, State s0, std::size_t steps,
                 double dt, Params p) {
  validate(p);
  validate_dt(dt);
  traj.clear();
  P v{s0.x, s0.y, s0.z};
  for (std::size_t i = 0; i < steps; ++i) {
    P d = lorenz(v, p);
    v.x += dt * d.x;
    v.y += dt * d.y;
    v.z += dt * d.z;
    traj.push_back(State{v.x, v.y, v.z});
  }
}

void rk4_naive(std::vector<State>& traj, State s0, std::size_t steps,
               double dt, Params p) {
  validate(p);
  validate_dt(dt);
  traj.clear();
  P v{s0.x, s0.y, s0.z};
  for (std::size_t i = 0; i < steps; ++i) {
    v = rk4_naive_step(v, dt, p);
    traj.push_back(State{v.x, v.y, v.z});
  }
}

void rk4_ensemble_naive(State* states, std::size_t n, std::size_t steps,
                        double dt, Params p) {
  validate(p);
  validate_dt(dt);
  if (n > 0 && states == nullptr) {
    throw std::invalid_argument("chaosys::rk4_ensemble_naive: null states");
  }
  for (std::size_t i = 0; i < n; ++i) {
    P v{states[i].x, states[i].y, states[i].z};
    for (std::size_t k = 0; k < steps; ++k) v = rk4_naive_step(v, dt, p);
    states[i] = State{v.x, v.y, v.z};
  }
}

}  // namespace chaosys

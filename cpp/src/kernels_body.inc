// Kernel bodies shared by the IEEE and the -ffast-math translation units.
// This file is #included inside a namespace.
// All headers (including <arm_neon.h>) must be included by the enclosing .cpp.

#if defined(__GNUC__) || defined(__clang__)
#define CHAOSYS_NOINLINE __attribute__((noinline))
#else
#define CHAOSYS_NOINLINE
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define CHAOSYS_NEON 1
#else
#define CHAOSYS_NEON 0
#endif

inline void euler_step_inplace(double& x, double& y, double& z, double dt,
                               double sigma, double rho,
                               double beta) noexcept {
  const double dx = sigma * (y - x);
  const double dy = x * (rho - z) - y;
  const double dz = x * y - beta * z;
  x += dt * dx;
  y += dt * dy;
  z += dt * dz;
}

inline void rk4_step_inplace(double& x, double& y, double& z, double dt,
                             double sigma, double rho,
                             double beta) noexcept {
  const double h2 = 0.5 * dt;
  const double h6 = dt / 6.0;
  double k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;

  k1x = sigma * (y - x);
  k1y = x * (rho - z) - y;
  k1z = x * y - beta * z;

  const double x2 = x + h2 * k1x, y2 = y + h2 * k1y, z2 = z + h2 * k1z;
  k2x = sigma * (y2 - x2);
  k2y = x2 * (rho - z2) - y2;
  k2z = x2 * y2 - beta * z2;

  const double x3 = x + h2 * k2x, y3 = y + h2 * k2y, z3 = z + h2 * k2z;
  k3x = sigma * (y3 - x3);
  k3y = x3 * (rho - z3) - y3;
  k3z = x3 * y3 - beta * z3;

  const double x4 = x + dt * k3x, y4 = y + dt * k3y, z4 = z + dt * k3z;
  k4x = sigma * (y4 - x4);
  k4y = x4 * (rho - z4) - y4;
  k4z = x4 * y4 - beta * z4;

  x += h6 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
  y += h6 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
  z += h6 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
}

inline void euler_loop(double* __restrict x, double* __restrict y,
                       double* __restrict z, std::size_t n, std::size_t steps,
                       double dt, const Params& p) {
  const double sigma = p.sigma, rho = p.rho, beta = p.beta;
  for (std::size_t s = 0; s < steps; ++s) {
    for (std::size_t i = 0; i < n; ++i) {
      euler_step_inplace(x[i], y[i], z[i], dt, sigma, rho, beta);
    }
  }
}

// Structure-of-arrays scalar RK4.  Kept free of branches so the compiler can
// auto-vectorise it; on ARM64 it does vectorise with 2 double lanes.
CHAOSYS_NOINLINE void soa_scalar_range(double* __restrict x, double* __restrict y,
                             double* __restrict z, std::size_t begin,
                             std::size_t end, std::size_t steps, double dt,
                             const Params& p) {
  const double sigma = p.sigma, rho = p.rho, beta = p.beta;
  for (std::size_t s = 0; s < steps; ++s) {
    for (std::size_t i = begin; i < end; ++i) {
      rk4_step_inplace(x[i], y[i], z[i], dt, sigma, rho, beta);
    }
  }
}

inline void rk4_scalar_loop(double& x, double& y, double& z,
                            std::size_t steps, double dt, const Params& p) {
  const double sigma = p.sigma, rho = p.rho, beta = p.beta;
  for (std::size_t s = 0; s < steps; ++s) {
    rk4_step_inplace(x, y, z, dt, sigma, rho, beta);
  }
}

inline void rk4_scalar_strided_loop(double& x, double& y, double& z,
                                    std::size_t steps, double dt, State* out,
                                    std::size_t stride, const Params& p) {
  const double sigma = p.sigma, rho = p.rho, beta = p.beta;
  std::size_t oi = 0;
  out[oi++] = State{x, y, z};
  // Store in chunks so the hot loop has no per-step branch or counter.
  while (steps >= stride) {
    for (std::size_t k = 0; k < stride; ++k) rk4_step_inplace(x, y, z, dt, sigma, rho, beta);
    out[oi++] = State{x, y, z};
    steps -= stride;
  }
  for (std::size_t k = 0; k < steps; ++k) rk4_step_inplace(x, y, z, dt, sigma, rho, beta);
}

#if CHAOSYS_NEON

struct V3 {
  float64x2_t x, y, z;
};

inline V3 vload(const double* x, const double* y, const double* z,
                std::size_t i) {
  return V3{vld1q_f64(x + i), vld1q_f64(y + i), vld1q_f64(z + i)};
}

inline void vstore(double* x, double* y, double* z, std::size_t i, V3 v) {
  vst1q_f64(x + i, v.x);
  vst1q_f64(y + i, v.y);
  vst1q_f64(z + i, v.z);
}

// One derivative evaluation for two independent trajectories.  sigma, rho and
// -beta are broadcast once per block by the caller.
inline V3 vderiv(V3 v, const float64x2_t& s, const float64x2_t& r,
                 const float64x2_t& nb) noexcept {
  V3 d;
  d.x = vmulq_f64(s, vsubq_f64(v.y, v.x));
  d.y = vfmaq_f64(vnegq_f64(v.y), vsubq_f64(r, v.z), v.x);
  d.z = vfmaq_f64(vmulq_f64(v.x, v.y), v.z, nb);
  return d;
}

inline V3 vadd_scaled(V3 a, V3 b, double s) noexcept {
  return V3{vfmaq_n_f64(a.x, b.x, s), vfmaq_n_f64(a.y, b.y, s),
            vfmaq_n_f64(a.z, b.z, s)};
}

inline V3 vrk4_step(V3 v, double dt, double h2, double h6,
                    const float64x2_t& s, const float64x2_t& r,
                    const float64x2_t& nb) noexcept {
  const V3 k1 = vderiv(v, s, r, nb);
  const V3 k2 = vderiv(vadd_scaled(v, k1, h2), s, r, nb);
  const V3 k3 = vderiv(vadd_scaled(v, k2, h2), s, r, nb);
  const V3 k4 = vderiv(vadd_scaled(v, k3, dt), s, r, nb);

  const V3 w{vaddq_f64(vaddq_f64(k1.x, k4.x),
                       vmulq_n_f64(vaddq_f64(k2.x, k3.x), 2.0)),
             vaddq_f64(vaddq_f64(k1.y, k4.y),
                       vmulq_n_f64(vaddq_f64(k2.y, k3.y), 2.0)),
             vaddq_f64(vaddq_f64(k1.z, k4.z),
                       vmulq_n_f64(vaddq_f64(k2.z, k3.z), 2.0))};
  return vadd_scaled(v, w, h6);
}

// 8 trajectories per iteration: four independent 2-lane chains keep the FMA
// pipeline busy while the next stage of the other chains is in flight.
CHAOSYS_NOINLINE void soa_neon_range(double* __restrict x, double* __restrict y,
                           double* __restrict z, std::size_t begin,
                           std::size_t end, std::size_t steps, double dt,
                           const Params& p) {
  const double h2 = 0.5 * dt;
  const double h6 = dt / 6.0;
  const float64x2_t vs = vdupq_n_f64(p.sigma);
  const float64x2_t vr = vdupq_n_f64(p.rho);
  const float64x2_t vnb = vdupq_n_f64(-p.beta);

  for (std::size_t s = 0; s < steps; ++s) {
    std::size_t i = begin;
    for (; i + 8 <= end; i += 8) {
#if defined(CHAOSYS_PREFETCH) && CHAOSYS_PREFETCH
      if (i + 32 < end) {
        __builtin_prefetch(x + i + 32, 1, 2);
        __builtin_prefetch(y + i + 32, 1, 2);
        __builtin_prefetch(z + i + 32, 1, 2);
      }
#endif
      V3 a = vload(x, y, z, i);
      V3 b = vload(x, y, z, i + 2);
      V3 c = vload(x, y, z, i + 4);
      V3 d = vload(x, y, z, i + 6);
      a = vrk4_step(a, dt, h2, h6, vs, vr, vnb);
      b = vrk4_step(b, dt, h2, h6, vs, vr, vnb);
      c = vrk4_step(c, dt, h2, h6, vs, vr, vnb);
      d = vrk4_step(d, dt, h2, h6, vs, vr, vnb);
      vstore(x, y, z, i, a);
      vstore(x, y, z, i + 2, b);
      vstore(x, y, z, i + 4, c);
      vstore(x, y, z, i + 6, d);
    }
    for (; i + 4 <= end; i += 4) {
      V3 a = vload(x, y, z, i);
      V3 b = vload(x, y, z, i + 2);
      a = vrk4_step(a, dt, h2, h6, vs, vr, vnb);
      b = vrk4_step(b, dt, h2, h6, vs, vr, vnb);
      vstore(x, y, z, i, a);
      vstore(x, y, z, i + 2, b);
    }
    for (; i + 2 <= end; i += 2) {
      V3 a = vload(x, y, z, i);
      a = vrk4_step(a, dt, h2, h6, vs, vr, vnb);
      vstore(x, y, z, i, a);
    }
    for (; i < end; ++i) {
      rk4_step_inplace(x[i], y[i], z[i], dt, p.sigma, p.rho, p.beta);
    }
  }
}

#else  // !CHAOSYS_NEON

CHAOSYS_NOINLINE void soa_neon_range(double* __restrict x, double* __restrict y,
                           double* __restrict z, std::size_t begin,
                           std::size_t end, std::size_t steps, double dt,
                           const Params& p) {
  soa_scalar_range(x, y, z, begin, end, steps, dt, p);
}

#endif  // CHAOSYS_NEON

inline std::uint64_t splitmix64(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

inline void init_ensemble_impl(Ensemble& e, std::size_t n, std::uint64_t seed) {
  e.resize(n);
  std::uint64_t s = seed;
  constexpr double kInv53 = 1.0 / 9007199254740992.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double ux = static_cast<double>(splitmix64(s) >> 11) * kInv53;
    const double uy = static_cast<double>(splitmix64(s) >> 11) * kInv53;
    const double uz = static_cast<double>(splitmix64(s) >> 11) * kInv53;
    e.x[i] = -15.0 + 30.0 * ux;
    e.y[i] = -20.0 + 40.0 * uy;
    e.z[i] = 5.0 + 35.0 * uz;
  }
}

inline State max_diff_impl(const Ensemble& a, const Ensemble& b) {
  State m{0.0, 0.0, 0.0};
  auto update = [](double d, double& m) {
    if (std::isnan(d) || std::isinf(d)) {
      m = d;
      return;
    }
    if (std::isnan(m) || std::isinf(m)) return;  // keep the non-finite value
    if (d > m) m = d;
  };
  for (std::size_t i = 0; i < a.n; ++i) {
    // std::max(0, NaN) silently keeps the old maximum; update() propagates
    // non-finite differences so kernel-equivalence checks cannot pass on NaNs.
    update(std::abs(a.x[i] - b.x[i]), m.x);
    update(std::abs(a.y[i] - b.y[i]), m.y);
    update(std::abs(a.z[i] - b.z[i]), m.z);
  }
  return m;
}

// Benchmark / validation driver for the Lorenz attractor kernels.
//
//   ./chaosys_bench            # full suite, writes results/*.csv
//   ./chaosys_bench validate   # correctness + dynamics checks only
//   ./chaosys_bench flags LABEL  # two-line summary for a compiler-flag variant
#include "chaosys.hpp"
#include "chaosys/analysis.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <unistd.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

double g_sink = 0.0;

struct Timing {
  double best;
  double median;
  double worst;
};

template <class F>
Timing measure(F&& f, int reps) {
  f();  // warm-up, also lets allocations reach steady state
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(reps));
  for (int i = 0; i < reps; ++i) {
    const auto t0 = Clock::now();
    f();
    const auto t1 = Clock::now();
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  return Timing{samples.front(), samples[samples.size() / 2], samples.back()};
}

std::string cpu_name() {
#if defined(__APPLE__)
  char buf[256] = {0};
  std::size_t len = sizeof(buf);
  if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) {
    return std::string(buf);
  }
#endif
  return std::string("unknown CPU");
}

double ns_per(double seconds, double steps) {
  return seconds * 1e9 / steps;
}

// Flops per classical RK4 step (10 per RHS evaluation * 4, ~39 in the staged
// update).  Used only to put throughput numbers on a familiar scale.
constexpr double kFlopsPerRk4Step = 79.0;

void write_csv(const std::string& name, const std::string& header,
               const std::vector<std::string>& rows) {
  std::filesystem::create_directories("results");
  std::ofstream f("results/" + name);
  f << header << "\n";
  for (const auto& r : rows) f << r << "\n";
}

// ---------------------------------------------------------------------------
// Single-trajectory latency
// ---------------------------------------------------------------------------
struct SingleRow {
  std::string name;
  double steps;
  double seconds;
  double bytes;
};

std::vector<SingleRow> bench_single(int reps) {
  constexpr std::size_t kSteps = 5'000'000;
  constexpr double dt = 0.001;
  std::vector<SingleRow> out;
  std::vector<chaosys::State> traj;

  {
    const Timing t = measure(
        [&] {
          chaosys::euler_naive(traj, chaosys::State{1, 1, 1}, kSteps, dt);
          g_sink += traj.back().x;
        },
        reps);
    out.push_back({"euler_naive (push_back)", double(kSteps), t.best,
                   double(kSteps) * 24.0});
  }
  {
    const Timing t = measure(
        [&] {
          chaosys::rk4_naive(traj, chaosys::State{1, 1, 1}, kSteps, dt);
          g_sink += traj.back().x;
        },
        reps);
    out.push_back({"rk4_naive (push_back)", double(kSteps), t.best,
                   double(kSteps) * 24.0});
  }
  {
    chaosys::State s{1, 1, 1};
    const Timing t = measure(
        [&] {
          chaosys::rk4_scalar(s, kSteps, dt);
          g_sink += s.x;
        },
        reps);
    out.push_back({"rk4_scalar (in-place)", double(kSteps), t.best, 0.0});
  }
  {
    constexpr std::size_t stride = 10;
    std::vector<chaosys::State> outbuf(kSteps / stride + 1);
    chaosys::State s{1, 1, 1};
    const Timing t = measure(
        [&] {
          chaosys::rk4_scalar_strided(s, kSteps, dt, outbuf.data(), stride);
          g_sink += outbuf[kSteps / stride].x;
        },
        reps);
    out.push_back({"rk4_scalar + decimated store", double(kSteps), t.best,
                   double(kSteps / stride) * 24.0});
  }

  std::printf("\n[1] Single trajectory, %.0f steps, dt=%.3f\n", double(kSteps), dt);
  std::printf("    %-30s %10s %10s %10s %8s\n", "variant", "ns/step", "Mstep/s",
              "GB/s out", "vs naive");
  const double base = ns_per(out[1].seconds, out[1].steps);
  for (const auto& r : out) {
    const double ns = ns_per(r.seconds, r.steps);
    const double gbps = r.bytes > 0 ? r.bytes / r.seconds / 1e9 : 0.0;
    std::printf("    %-30s %10.2f %10.2f %10.2f %7.2fx\n", r.name.c_str(), ns,
                1e3 / ns, gbps, base / ns);
  }

  std::vector<std::string> rows;
  char buf[512];
  for (const auto& r : out) {
    const double ns = ns_per(r.seconds, r.steps);
    std::snprintf(buf, sizeof(buf), "%s,%.0f,%.6f,%.4f,%.4f,%.0f,%.4f",
                  r.name.c_str(), r.steps, r.seconds, ns, 1e3 / ns, r.bytes,
                  r.bytes / r.seconds / 1e9);
    rows.emplace_back(buf);
  }
  write_csv("single.csv",
            "variant,steps,seconds,ns_per_step,msteps_per_s,bytes_out,gbps_out",
            rows);
  return out;
}

// ---------------------------------------------------------------------------
// Accuracy / cost study
// ---------------------------------------------------------------------------
struct AccRow {
  std::string scheme;
  double dt;
  double steps;
  double error;
  double ns_step;
};

void bench_accuracy(double rk4_ns_step, double euler_ns_step) {
  constexpr double kT = 10.0;
  const chaosys::State ref =
      chaosys::dp5_reference(chaosys::State{1, 1, 1}, kT, 1e-13, 1e-14);
  const chaosys::State ref_fine =
      chaosys::dp5_reference(chaosys::State{1, 1, 1}, kT, 3e-14, 3e-15);
  const double ref_self = std::max({std::abs(ref.x - ref_fine.x),
                                    std::abs(ref.y - ref_fine.y),
                                    std::abs(ref.z - ref_fine.z)});

  std::vector<AccRow> rows;
  const double e_dts[] = {1e-3, 1e-4, 1e-5};
  for (double dt : e_dts) {
    const std::size_t steps = static_cast<std::size_t>(std::llround(kT / dt));
    chaosys::State s{1, 1, 1};
    chaosys::euler_scalar(s, steps, dt);
    const double err = std::max({std::abs(s.x - ref.x), std::abs(s.y - ref.y),
                                 std::abs(s.z - ref.z)});
    rows.push_back({"Euler", dt, double(steps), err, euler_ns_step});
  }
  const double r_dts[] = {0.05, 0.02, 0.01, 0.005, 0.002};
  for (double dt : r_dts) {
    const std::size_t steps = static_cast<std::size_t>(std::llround(kT / dt));
    chaosys::State s{1, 1, 1};
    chaosys::rk4_scalar(s, steps, dt);
    const double err = std::max({std::abs(s.x - ref.x), std::abs(s.y - ref.y),
                                 std::abs(s.z - ref.z)});
    rows.push_back({"RK4", dt, double(steps), err, rk4_ns_step});
  }

  std::printf("\n[2] Global error at T=%.1f vs Dormand-Prince 5(4) "
              "(oracle self-diff %.2e)\n", kT, ref_self);
  std::printf("    %-6s %10s %12s %14s %10s %12s\n", "scheme", "dt", "steps",
              "max|error|", "ns/step", "cost (ms)");
  std::vector<std::string> csv;
  char buf[512];
  for (const auto& r : rows) {
    const double cost = r.steps * r.ns_step * 1e-6;
    std::printf("    %-6s %10.0e %12.0f %14.3e %10.2f %12.3f\n",
                r.scheme.c_str(), r.dt, r.steps, r.error, r.ns_step, cost);
    std::snprintf(buf, sizeof(buf), "%s,%.0e,%.0f,%.3e,%.3f,%.3f",
                  r.scheme.c_str(), r.dt, r.steps, r.error, r.ns_step, cost);
    csv.emplace_back(buf);
  }
  write_csv("accuracy.csv", "scheme,dt,steps,error,ns_per_step,cost_ms", csv);
}

// ---------------------------------------------------------------------------
// Ensemble throughput
// ---------------------------------------------------------------------------
struct EnsRow {
  std::string name;
  int threads;
  double seconds;
  double metric;  // Mtrajectory-steps per second
};

EnsRow bench_one(const std::string& name, int threads, std::size_t n,
                 std::size_t steps, double dt, int reps,
                 const std::function<void()>& run) {
  const Timing t = measure(run, reps);
  const double mtps = double(n) * double(steps) / t.best / 1e6;
  return EnsRow{name, threads, t.best, mtps};
}

void print_ens(const std::vector<EnsRow>& rows) {
  std::printf("\n    %-34s %8s %12s %10s %10s\n", "variant", "threads",
              "Mtraj-step/s", "GFLOP/s", "speedup");
  const double base = rows.front().metric;
  for (const auto& r : rows) {
    std::printf("    %-34s %8d %12.2f %10.2f %9.2fx\n", r.name.c_str(),
                r.threads, r.metric, r.metric * 1e6 * kFlopsPerRk4Step / 1e9,
                r.metric / base);
  }
}

void bench_ensemble(int reps) {
  constexpr std::size_t kN = 32768;
  constexpr std::size_t kSteps = 2048;
  constexpr double dt = 0.01;
  const int max_t = chaosys::max_threads();

  chaosys::Ensemble e;
  chaosys::init_ensemble(e, kN);
  std::vector<chaosys::State> aos(kN);
  for (std::size_t i = 0; i < kN; ++i) {
    aos[i] = chaosys::State{e.x[i], e.y[i], e.z[i]};
  }

  std::vector<EnsRow> rows;
  rows.push_back(bench_one("AoS scalar (baseline)", 1, kN, kSteps, dt, 3,
                           [&] {
                             chaosys::rk4_ensemble_naive(aos.data(), kN, kSteps,
                                                         dt);
                             g_sink += aos[0].x;
                           }));
  rows.push_back(bench_one("SoA scalar (auto-vec)", 1, kN, kSteps, dt, reps,
                           [&] { chaosys::rk4_soa_scalar(e, kSteps, dt); }));
  rows.push_back(bench_one("SoA scalar + threads", max_t, kN, kSteps, dt, reps,
                           [&] {
                             chaosys::rk4_soa_scalar_threads(e, kSteps, dt,
                                                             max_t);
                           }));
  rows.push_back(bench_one("SoA NEON", 1, kN, kSteps, dt, reps,
                           [&] { chaosys::rk4_soa_neon(e, kSteps, dt); }));
  rows.push_back(bench_one("SoA NEON + threads", max_t, kN, kSteps, dt, reps,
                           [&] {
                             chaosys::rk4_soa_neon_threads(e, kSteps, dt, max_t);
                           }));
  rows.push_back(bench_one("SoA NEON fast-math", 1, kN, kSteps, dt, reps,
                           [&] {
                             chaosys_fast_rk4_soa_neon(
                                 e.px(), e.py(), e.pz(), e.n, kSteps, dt,
                                 chaosys::Params{});
                           }));
  rows.push_back(bench_one("SoA NEON fast-math + threads", max_t, kN, kSteps,
                           dt, reps, [&] {
                             chaosys_fast_rk4_soa_neon_threads(
                                 e.px(), e.py(), e.pz(), e.n, kSteps, dt,
                                 max_t, chaosys::Params{});
                           }));

  std::printf("\n[3] Ensemble: n=%zu, steps=%zu, dt=%.2f (%.2f G traj-steps)\n",
              kN, kSteps, dt, double(kN) * kSteps / 1e9);
  print_ens(rows);

  std::vector<std::string> csv;
  char buf[512];
  for (const auto& r : rows) {
    std::snprintf(buf, sizeof(buf), "%s,%d,%zu,%zu,%.6f,%.3f,%.2f",
                  r.name.c_str(), r.threads, kN, kSteps, r.seconds, r.metric,
                  r.metric * 1e6 * kFlopsPerRk4Step / 1e9);
    csv.emplace_back(buf);
  }
  write_csv("ensemble.csv",
            "variant,threads,n,steps,seconds,mtraj_steps_per_s,gflops", csv);
}

struct ParamRow {
  const char* name;
  chaosys::Params p;
};

void bench_params() {
  const ParamRow sets[] = {
      {"classic", {10.0, 28.0, 8.0 / 3.0}},
      {"wide", {10.0, 45.0, 8.0 / 3.0}},
      {"periodic", {10.0, 99.96, 8.0 / 3.0}},
      {"stiff", {10.0, 28.0, 4.0}},
  };

  std::printf("\n[5] Runtime parameter sweep (single trajectory, T=100, dt=0.01)\n");
  std::printf("    %-10s %7s %8s %7s %10s %10s %12s\n", "preset", "sigma",
              "rho", "beta", "max|x|", "max z", "state");
  std::vector<std::string> csv;
  char buf[512];
  for (const auto& set : sets) {
    chaosys::State s{1, 1, 1};
    double max_abs = 0.0, max_z = 0.0;
    for (int k = 0; k < 100; ++k) {
      chaosys::rk4_scalar(s, 100, 0.01, set.p);
      max_abs = std::max(max_abs, std::abs(s.x));
      max_abs = std::max(max_abs, std::abs(s.y));
      max_z = std::max(max_z, s.z);
    }
    const bool finite = std::isfinite(s.x) && std::isfinite(s.y) &&
                        std::isfinite(s.z);
    std::printf("    %-10s %7.2f %8.2f %7.3f %10.3f %10.3f %12s\n", set.name,
                set.p.sigma, set.p.rho, set.p.beta, max_abs, max_z,
                finite ? "finite" : "DIVERGED");
    std::snprintf(buf, sizeof(buf), "%s,%.2f,%.2f,%.3f,%.4f,%.4f,%.6f,%.6f,%.6f",
                  set.name, set.p.sigma, set.p.rho, set.p.beta, max_abs, max_z,
                  s.x, s.y, s.z);
    csv.emplace_back(buf);
  }
  write_csv("params.csv",
            "preset,sigma,rho,beta,max_abs,max_z,x,y,z", csv);
}

void bench_scaling(int reps) {
  constexpr std::size_t kN = 65536;
  constexpr std::size_t kSteps = 1024;
  constexpr double dt = 0.01;
  chaosys::Ensemble e;
  chaosys::init_ensemble(e, kN);

  const int max_t = chaosys::max_threads();
  std::vector<EnsRow> rows;
  const int counts[] = {1, 2, 3, 4, 5, 6, 8, 10, 11, 12};
  for (int t : counts) {
    if (t > max_t) break;
    rows.push_back(bench_one("NEON", t, kN, kSteps, dt, 5, [&] {
      chaosys::rk4_soa_neon_threads(e, kSteps, dt, t);
    }));
  }

  std::printf("\n[4] Thread scaling (SoA NEON, n=%zu, steps=%zu)\n", kN, kSteps);
  std::printf("    %8s %12s %10s %10s\n", "threads", "Mtraj-step/s", "GFLOP/s",
              "speedup");
  const double base = rows.front().metric;
  std::vector<std::string> csv;
  char buf[256];
  for (const auto& r : rows) {
    std::printf("    %8d %12.2f %10.2f %9.2fx\n", r.threads, r.metric,
                r.metric * 1e6 * kFlopsPerRk4Step / 1e9, r.metric / base);
    std::snprintf(buf, sizeof(buf), "%d,%.6f,%.3f,%.3f", r.threads, r.seconds,
                  r.metric, r.metric / base);
    csv.emplace_back(buf);
  }
  write_csv("scaling.csv", "threads,seconds,mtraj_steps_per_s,speedup", csv);
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
void bench_validate() {
  const chaosys::State ref =
      chaosys::dp5_reference(chaosys::State{1, 1, 1}, 10.0, 1e-13, 1e-14);
  const chaosys::State ref_fine =
      chaosys::dp5_reference(chaosys::State{1, 1, 1}, 10.0, 3e-14, 3e-15);

  chaosys::State s{1, 1, 1};
  chaosys::rk4_scalar(s, 1000, 0.01);

  chaosys::Ensemble a, b;
  chaosys::init_ensemble(a, 8192);
  chaosys::init_ensemble(b, 8192);
  chaosys::rk4_soa_scalar(a, 200, 0.01);
  chaosys::rk4_soa_neon(b, 200, 0.01);
  const chaosys::State simd_diff = chaosys::max_diff(a, b);

  chaosys::init_ensemble(b, 8192);
  chaosys_fast_rk4_soa_neon(b.px(), b.py(), b.pz(), b.n, 200, 0.01,
                            chaosys::Params{});
  const chaosys::State fast_diff = chaosys::max_diff(a, b);

  chaosys::init_ensemble(a, 8192);
  chaosys::init_ensemble(b, 8192);
  chaosys::rk4_soa_scalar(a, 1, 0.01);
  chaosys::rk4_soa_neon(b, 1, 0.01);
  const chaosys::State one_step = chaosys::max_diff(a, b);

  chaosys::Ensemble c;
  chaosys::init_ensemble(c, 8192);
  chaosys::rk4_soa_scalar_threads(c, 200, 0.01, chaosys::max_threads());
  chaosys::init_ensemble(b, 8192);
  chaosys::rk4_soa_scalar(b, 200, 0.01);
  const chaosys::State thread_diff = chaosys::max_diff(c, b);

  chaosys::State nb{1, 1, 1};
  std::vector<chaosys::State> naive;
  chaosys::rk4_naive(naive, nb, 1000, 0.01);
  const double naive_diff =
      std::max({std::abs(naive.back().x - s.x), std::abs(naive.back().y - s.y),
                std::abs(naive.back().z - s.z)});

  // Long-run bounds and extrema over 10 000 time units.
  chaosys::State p{1, 1, 1};
  double max_abs_x = 0.0, max_r = 0.0, zmin = 1e9, zmax = -1e9;
  for (int i = 0; i < 1000; ++i) {
    chaosys::rk4_scalar(p, 1000, 0.01);
    max_abs_x = std::max(max_abs_x, std::abs(p.x));
    max_r = std::max(max_r, std::sqrt(p.x * p.x + p.y * p.y));
    zmin = std::min(zmin, p.z);
    zmax = std::max(zmax, p.z);
  }
  chaosys::LorenzSystem lyapunov_system{};
  double lyapunov_state[3] = {1.0, 1.0, 1.0};
  const double lambda = chaosys::largest_lyapunov(
      lyapunov_system, lyapunov_state, 0.01, 0, 100, 1000);

  std::printf("\n[Validation]\n");
  std::printf("  RK4 dt=0.01, 1000 steps : (%.15f, %.15f, %.15f)\n", s.x, s.y,
              s.z);
  std::printf("  DP5 rtol=1e-12          : (%.15f, %.15f, %.15f)\n", ref.x,
              ref.y, ref.z);
  std::printf("  DP5 self-consistency    : %.2e\n",
              std::max({std::abs(ref.x - ref_fine.x),
                        std::abs(ref.y - ref_fine.y),
                        std::abs(ref.z - ref_fine.z)}));
  std::printf("  naive RK4 vs optimized  : %.2e\n", naive_diff);
  const double td = std::max({thread_diff.x, thread_diff.y, thread_diff.z});
  std::printf("  threaded vs serial scalar: %.2e (must be 0)\n", td);
  std::printf("  NEON vs scalar, 1 step  : (%.2e, %.2e, %.2e)\n",
              one_step.x, one_step.y, one_step.z);
  std::printf("  NEON vs scalar, 200 steps: (%.2e, %.2e, %.2e)\n",
              simd_diff.x, simd_diff.y, simd_diff.z);
  std::printf("  fast-math vs IEEE       : (%.2e, %.2e, %.2e)\n", fast_diff.x,
              fast_diff.y, fast_diff.z);
  std::printf("  largest Lyapunov exponent: %.4f (theory ~0.906)\n", lambda);
  std::printf("  bounds over T=10000     : max|x|=%.2f max r_xy=%.2f z in "
              "[%.2f, %.2f]\n",
              max_abs_x, max_r, zmin, zmax);

  std::filesystem::create_directories("results");
  std::ofstream f("results/validate.txt");
  f << "rk4_dt0.01_1000 " << s.x << " " << s.y << " " << s.z << "\n";
  f << "dp5 " << ref.x << " " << ref.y << " " << ref.z << "\n";
  f << "naive_diff " << naive_diff << "\n";
  f << "thread_diff " << td << "\n";
  f << "one_step_diff " << one_step.x << " " << one_step.y << " "
    << one_step.z << "\n";
  f << "simd_diff " << simd_diff.x << " " << simd_diff.y << " " << simd_diff.z
    << "\n";
  f << "fast_diff " << fast_diff.x << " " << fast_diff.y << " " << fast_diff.z
    << "\n";
  f << "lyapunov " << lambda << "\n";
  f << "bounds " << max_abs_x << " " << max_r << " " << zmin << " " << zmax
    << "\n";
}

// ---------------------------------------------------------------------------
// Compiler-flag variant summary
// ---------------------------------------------------------------------------
int run_flags(const char* label) {
  constexpr std::size_t kSteps = 20'000'000;
  chaosys::State s{1, 1, 1};
  const Timing t = measure(
      [&] {
        chaosys::rk4_scalar(s, kSteps, 0.001);
        g_sink += s.x;
      },
      5);

  constexpr std::size_t kN = 65536;
  constexpr std::size_t kEnsSteps = 1024;
  chaosys::Ensemble e;
  chaosys::init_ensemble(e, kN);
  const Timing t1 = measure(
      [&] { chaosys::rk4_soa_neon(e, kEnsSteps, 0.01); }, 3);
  const Timing t2 = measure(
      [&] {
        chaosys::rk4_soa_neon_threads(e, kEnsSteps, 0.01,
                                      chaosys::max_threads());
      },
      3);

  const double scalar_ns = ns_per(t.best, kSteps);
  const double mtps1 = double(kN) * kEnsSteps / t1.best / 1e6;
  const double mtps11 = double(kN) * kEnsSteps / t2.best / 1e6;
  std::printf("%s,%.2f,%.2f,%.2f\n", label, scalar_ns, mtps1, mtps11);
  std::fflush(stdout);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "all";

  if (mode != "flags") {
    std::printf("Lorenz attractor benchmark | %s | %d hardware threads\n",
                cpu_name().c_str(), chaosys::max_threads());
    std::printf("compiler: %s\n", __VERSION__);
  }
  std::filesystem::create_directories("results");
  {
    std::ofstream f("results/meta.txt");
    f << "cpu " << cpu_name() << "\n";
    f << "threads " << chaosys::max_threads() << "\n";
    f << "compiler " << __VERSION__ << "\n";
  }

  if (mode == "flags") {
    return run_flags(argc > 2 ? argv[2] : "default");
  }
  if (mode == "validate") {
    bench_validate();
    return 0;
  }
  if (mode == "traj") {
    // 6 000 stored samples of one trajectory (T = 60, dt = 0.01); used by the
    // report figure generator.
    constexpr std::size_t kSteps = 6000, kStride = 1;
    std::vector<chaosys::State> pts(kSteps / kStride + 1);
    chaosys::State s{1, 1, 1};
    chaosys::rk4_scalar_strided(s, kSteps, 0.01, pts.data(), kStride);
    std::filesystem::create_directories("results");
    std::ofstream f("results/traj.csv");
    f << "x,y,z\n";
    for (const auto& p : pts) f << p.x << "," << p.y << "," << p.z << "\n";
    std::printf("wrote results/traj.csv (%zu samples)\n", pts.size());
    return 0;
  }
  if (mode == "quick") {
    bench_validate();
    bench_accuracy(6.0, 2.0);
    return 0;
  }

  bench_validate();

  // Measure pure compute ns/step for the accuracy/cost model.
  constexpr std::size_t kComputeSteps = 20'000'000;
  double rk4_ns, euler_ns;
  {
    chaosys::State s{1, 1, 1};
    rk4_ns = ns_per(
        measure(
            [&] {
              chaosys::rk4_scalar(s, kComputeSteps, 0.001);
              g_sink += s.x;
            },
            3)
            .best,
        kComputeSteps);
  }
  {
    chaosys::State s{1, 1, 1};
    euler_ns = ns_per(
        measure(
            [&] {
              chaosys::euler_scalar(s, kComputeSteps, 0.001);
              g_sink += s.x;
            },
            3)
            .best,
        kComputeSteps);
  }

  bench_single(5);
  bench_accuracy(rk4_ns, euler_ns);
  bench_ensemble(5);
  bench_scaling(3);
  bench_params();

  std::printf("\n(sink %.6f)\n", g_sink);
  return 0;
}

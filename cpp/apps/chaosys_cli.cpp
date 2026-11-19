// Command-line front-end for the chaotic systems library.
//   traj       single trajectory CSV
//   section    Poincare section CSV
//   lyapunov   largest Lyapunov exponent
//   sweep      Lyapunov exponent over a rho range (Lorenz only)
//   ensemble   final-state statistics
//
// Examples:
//   chaosys_cli traj --system rossler --steps 20000 --stride 10
//   chaosys_cli lyapunov --system chen
//   chaosys_cli sweep --rho-min 20 --rho-max 45 --rho-step 0.2
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "chaosys.hpp"
#include "chaosys/analysis.hpp"
#include "chaosys/chaos_metrics.hpp"
#include "chaosys/systems.hpp"

namespace {

using Args = std::vector<std::string>;

const std::string* find_arg(const Args& args, const std::string& key) {
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] != key) continue;
    if (i + 1 >= args.size()) {
      throw std::invalid_argument("missing value for " + key);
    }
    return &args[i + 1];
  }
  return nullptr;
}

double number(const Args& args, const std::string& key, double fallback) {
  const std::string* p = find_arg(args, key);
  if (p == nullptr) return fallback;
  char* end = nullptr;
  const double v = std::strtod(p->c_str(), &end);
  if (end == p->c_str() || *end != '\0' || !std::isfinite(v)) {
    throw std::invalid_argument("invalid value for " + key + ": " + *p);
  }
  return v;
}

std::size_t count(const Args& args, const std::string& key,
                  std::size_t fallback) {
  const std::string* p = find_arg(args, key);
  if (p == nullptr) return fallback;
  if (p->empty() || (*p)[0] == '-' || (*p)[0] == '+') {
    throw std::invalid_argument("invalid count for " + key + ": " + *p);
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(p->c_str(), &end, 10);
  if (errno == ERANGE || end == p->c_str() || *end != '\0' ||
      v > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("invalid count for " + key + ": " + *p);
  }
  return static_cast<std::size_t>(v);
}

int integer(const Args& args, const std::string& key, int fallback) {
  const double value = number(args, key, static_cast<double>(fallback));
  if (!(value >= static_cast<double>(std::numeric_limits<int>::min()) &&
        value <= static_cast<double>(std::numeric_limits<int>::max())) ||
      std::floor(value) != value) {
    throw std::invalid_argument("invalid integer for " + key);
  }
  return static_cast<int>(value);
}

std::string text(const Args& args, const std::string& key,
                 const std::string& fallback) {
  const std::string* p = find_arg(args, key);
  return p == nullptr ? fallback : *p;
}

bool flag(const Args& args, const std::string& key) {
  for (const auto& a : args) {
    if (a == key) return true;
  }
  return false;
}

std::string lower(std::string value) {
  for (char& c : value) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return value;
}

enum class Kind { Lorenz, Rossler, Chen };
using System = std::variant<chaosys::LorenzSystem, chaosys::RosslerSystem,
                            chaosys::ChenSystem>;

Kind parse_kind(const Args& args) {
  const std::string name = lower(text(args, "--system", "lorenz"));
  if (name == "lorenz") return Kind::Lorenz;
  if (name == "rossler" || name == "r") return Kind::Rossler;
  if (name == "chen") return Kind::Chen;
  throw std::invalid_argument("unknown --system: " + name);
}

struct Options {
  Kind kind = Kind::Lorenz;
  chaosys::Params lp{};
  chaosys::RosslerParams rp{};
  chaosys::ChenParams cp{};
  double x0 = 1.0, y0 = 1.0, z0 = 1.0;
  double dt = 0.005;

  System system() const {
    switch (kind) {
      case Kind::Lorenz: return chaosys::LorenzSystem{lp};
      case Kind::Rossler: return chaosys::RosslerSystem{rp};
      case Kind::Chen: return chaosys::ChenSystem{cp};
    }
    return chaosys::LorenzSystem{lp};
  }

  const char* name() const {
    switch (kind) {
      case Kind::Lorenz: return "lorenz";
      case Kind::Rossler: return "rossler";
      case Kind::Chen: return "chen";
    }
    return "lorenz";
  }

  const char* param_header() const {
    return kind == Kind::Lorenz ? "sigma,rho,beta" : "a,b,c";
  }

  void write_params(std::ostream& out) const {
    if (kind == Kind::Lorenz) {
      out << lp.sigma << ',' << lp.rho << ',' << lp.beta;
    } else if (kind == Kind::Rossler) {
      out << rp.a << ',' << rp.b << ',' << rp.c;
    } else {
      out << cp.a << ',' << cp.b << ',' << cp.c;
    }
  }

  void validate() const {
    if (kind == Kind::Lorenz) {
      chaosys::validate(lp);
    } else {
      const System sys = system();
      std::visit([](const auto& s) { s.validate(); }, sys);
    }
    chaosys::validate_dt(dt);
  }
};

Options options(const Args& args) {
  Options o;
  o.kind = parse_kind(args);
  o.lp.sigma = number(args, "--sigma", o.lp.sigma);
  o.lp.rho = number(args, "--rho", o.lp.rho);
  o.lp.beta = number(args, "--beta", o.lp.beta);
  o.rp.a = number(args, "--a", o.rp.a);
  o.rp.b = number(args, "--b", o.rp.b);
  o.rp.c = number(args, "--c", o.rp.c);
  o.cp.a = number(args, "--a", o.cp.a);
  o.cp.b = number(args, "--b", o.cp.b);
  o.cp.c = number(args, "--c", o.cp.c);
  o.x0 = number(args, "--x0", o.x0);
  o.y0 = number(args, "--y0", o.y0);
  o.z0 = number(args, "--z0", o.z0);
  o.dt = number(args, "--dt", o.dt);
  o.validate();
  return o;
}

void validate_initial(const double* x) {
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(x[i])) {
      throw std::invalid_argument("non-finite initial state");
    }
  }
}

void step_system(System& sys, double* x, double dt) {
  std::visit([&](const auto& s) { chaosys::Rk4Method::step(s, x, dt); }, sys);
}

std::vector<double> sample_series(const Options& o, std::size_t transient,
                                  std::size_t steps, int component) {
  System sys = o.system();
  double x[3] = {o.x0, o.y0, o.z0};
  validate_initial(x);
  for (std::size_t i = 0; i < transient; ++i) step_system(sys, x, o.dt);
  std::vector<double> series;
  series.reserve(steps);
  for (std::size_t i = 0; i < steps; ++i) {
    step_system(sys, x, o.dt);
    series.push_back(x[component]);
  }
  return series;
}

struct Row {
  double t;
  std::array<double, 3> x;
};

struct Stats {
  double mean[3]{};
  double sd[3]{};
};

class Output {
 public:
  explicit Output(const std::string& path) {
    if (path.empty() || path == "-") {
      stream_ = &std::cout;
    } else {
      file_.open(path);
      if (!file_) throw std::runtime_error("cannot open " + path);
      stream_ = &file_;
    }
  }

  std::ostream& stream() { return *stream_; }

  void finish() {
    stream_->flush();
    if (!*stream_) throw std::runtime_error("failed to write output");
  }

 private:
  std::ofstream file_;
  std::ostream* stream_ = nullptr;
};

Stats stats_from(const double* const* comp, std::size_t n) {
  Stats s;
  for (int c = 0; c < 3; ++c) {
    for (std::size_t i = 0; i < n; ++i) s.mean[c] += comp[c][i];
    s.mean[c] /= static_cast<double>(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double d = comp[c][i] - s.mean[c];
      s.sd[c] += d * d;
    }
    s.sd[c] = std::sqrt(s.sd[c] / static_cast<double>(n));
    if (!std::isfinite(s.mean[c]) || !std::isfinite(s.sd[c])) {
      throw std::runtime_error("non-finite ensemble statistics");
    }
  }
  return s;
}

template <class Sys>
Stats generic_ensemble_stats(const Sys& sys, const Options& o, std::size_t n,
                             std::size_t steps, int threads) {
  chaosys::EnsembleN<Sys> ens(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double f = n > 1 ? static_cast<double>(i) / (n - 1) - 0.5 : 0.0;
    ens.component(0)[i] = o.x0 + 0.25 * f;
    ens.component(1)[i] = o.y0 + 0.25 * f;
    ens.component(2)[i] = o.z0;
  }
  chaosys::integrate_ensemble(sys, ens, steps, o.dt, threads);
  const double* comp[3] = {ens.component(0), ens.component(1),
                           ens.component(2)};
  return stats_from(comp, n);
}

void usage() {
  std::cout <<
      "chaosys_cli — chaotic systems tools\n\n"
      "  traj      [--system lorenz|rossler|chen] [--dt 0.005] [--steps 20000]\n"
      "            [--stride 10] [--out FILE]\n"
      "  section   [--system ...] [--component 2] [--level 20] [--direction 1]\n"
      "            [--transient 2000] [--steps 20000] [--max 200] [--out FILE]\n"
      "  lyapunov  [--system ...] [--transient 2000] [--segment 50]\n"
      "            [--segments 200] [--out FILE]\n"
      "  sweep     [--rho-min 20] [--rho-max 45] [--rho-step 0.2]\n"
      "            [--transient 1000] [--segment 50] [--segments 100]\n"
      "            (Lorenz only)\n"
      "  ensemble  [--system ...] [--n 65536] [--steps 2048] [--threads 8]\n"
      "  dimension [--system ...] [--component 0] [--dim 4] [--delay 20]\n"
      "            [--steps 5000] [--transient 2000] [--max-points 1500]\n"
      "  entropy   [--system ...] [--component 0] [--order 3] [--delay 1]\n"
      "            [--steps 5000] [--transient 2000]\n\n"
      "  common:   [--x0 1] [--y0 1] [--z0 1] [--dt 0.005]\n"
      "  Lorenz:   [--sigma 10] [--rho 28] [--beta 2.6667]\n"
      "  Rossler:  [--a 0.2] [--b 0.2] [--c 5.7]\n"
      "  Chen:     [--a 35] [--b 3] [--c 28]\n"
      "  --out - writes CSV to stdout (default); transient/steps are counts.\n"
      "  systems   list built-in systems (lorenz, rossler, chen).\n"
      "  version / --version prints the CLI version.\n"
      "  Lyapunov values are per unit time; section t is measured from x0.\n";
}

int cmd_traj(const Args& args) {
  const Options o = options(args);
  const std::size_t steps = count(args, "--steps", 20000);
  const std::size_t stride = count(args, "--stride", 10);
  if (stride == 0) throw std::invalid_argument("--stride must be > 0");

  System sys = o.system();
  double x[3] = {o.x0, o.y0, o.z0};
  validate_initial(x);

  Output output(text(args, "--out", "-"));
  std::ostream& out = output.stream();
  out << "t,x,y,z\n" << std::setprecision(12);

  auto write = [&](std::size_t i) {
    out << static_cast<double>(i) * o.dt << ',' << x[0] << ',' << x[1] << ','
        << x[2] << '\n';
  };
  write(0);
  for (std::size_t i = 1; i <= steps; ++i) {
    step_system(sys, x, o.dt);
    if (i % stride == 0) write(i);
  }
  output.finish();
  return 0;
}

int cmd_section(const Args& args) {
  const Options o = options(args);
  const std::size_t transient = count(args, "--transient", 2000);
  const std::size_t steps = count(args, "--steps", 20000);
  const std::size_t max = count(args, "--max", 200);
  const int component = integer(args, "--component", 2);
  const double level = number(args, "--level", 20.0);
  const int direction = integer(args, "--direction", 1);

  System sys = o.system();
  double x[3] = {o.x0, o.y0, o.z0};
  validate_initial(x);
  const std::vector<Row> rows = std::visit(
      [&](const auto& s) {
        const auto pts = chaosys::poincare_section(
            s, x, o.dt, transient, steps, max, component, level, direction);
        std::vector<Row> out;
        out.reserve(pts.size());
        for (const auto& p : pts) {
          out.push_back(Row{p.t, {p.x[0], p.x[1], p.x[2]}});
        }
        return out;
      },
      sys);

  Output output(text(args, "--out", "-"));
  std::ostream& out = output.stream();
  out << "t,x,y,z\n" << std::setprecision(12);
  for (const auto& p : rows) {
    out << p.t << ',' << p.x[0] << ',' << p.x[1] << ',' << p.x[2] << '\n';
  }
  output.finish();
  return 0;
}

int cmd_lyapunov(const Args& args) {
  const Options o = options(args);
  const std::size_t transient = count(args, "--transient", 2000);
  const std::size_t segment = count(args, "--segment", 50);
  const std::size_t segments = count(args, "--segments", 200);

  System sys = o.system();
  double x[3] = {o.x0, o.y0, o.z0};
  validate_initial(x);
  const double lambda = std::visit(
      [&](const auto& s) {
        return chaosys::largest_lyapunov(s, x, o.dt, transient, segment,
                                         segments);
      },
      sys);

  Output output(text(args, "--out", "-"));
  std::ostream& out = output.stream();
  out << o.param_header() << ",lambda\n" << std::setprecision(12);
  o.write_params(out);
  out << ',' << lambda << '\n';
  output.finish();
  return 0;
}

int cmd_sweep(const Args& args) {
  const Options o = options(args);
  if (o.kind != Kind::Lorenz) {
    throw std::invalid_argument("sweep currently supports --system lorenz only");
  }
  const double rho_min = number(args, "--rho-min", 20.0);
  const double rho_max = number(args, "--rho-max", 45.0);
  const double rho_step = number(args, "--rho-step", 0.2);
  const std::size_t transient = count(args, "--transient", 1000);
  const std::size_t segment = count(args, "--segment", 50);
  const std::size_t segments = count(args, "--segments", 100);
  if (!(rho_step > 0.0) || rho_max < rho_min) {
    throw std::invalid_argument("sweep: need rho-step > 0 and rho-max >= rho-min");
  }
  const double ratio = (rho_max - rho_min) / rho_step;
  constexpr std::size_t kMaxSweepPoints = 10'000'000;
  if (!std::isfinite(ratio) ||
      ratio > static_cast<double>(kMaxSweepPoints - 1)) {
    throw std::invalid_argument("sweep: too many rho points");
  }
  const std::size_t total = static_cast<std::size_t>(std::floor(ratio + 1e-9)) + 1;

  Output output(text(args, "--out", "-"));
  std::ostream& out = output.stream();
  out << "rho,lambda\n" << std::setprecision(12);

  for (std::size_t index = 0; index < total; ++index) {
    const double rho = rho_min + static_cast<double>(index) * rho_step;
    chaosys::Params p = o.lp;
    p.rho = rho;
    chaosys::LorenzSystem sys{p};
    double x[3] = {o.x0, o.y0, o.z0};
    const double lambda =
        chaosys::largest_lyapunov(sys, x, o.dt, transient, segment, segments);
    out << rho << ',' << lambda << '\n';
    if ((index + 1) % 10 == 0) {
      std::cerr << "\r" << (index + 1) << '/' << total << std::flush;
    }
  }
  output.finish();
  std::cerr << "\r" << total << '/' << total << '\n';
  return 0;
}

int cmd_dimension(const Args& args) {
  const Options o = options(args);
  const std::size_t transient = count(args, "--transient", 2000);
  const std::size_t steps = count(args, "--steps", 5000);
  const std::size_t max_points = count(args, "--max-points", 1500);
  const int component = integer(args, "--component", 0);
  const int embed_dim = integer(args, "--dim", 4);
  const int delay = integer(args, "--delay", 20);
  if (component < 0 || component > 2) {
    throw std::invalid_argument("--component must be 0, 1 or 2");
  }
  if (embed_dim <= 0 || delay <= 0 || max_points < 8) {
    throw std::invalid_argument("--dim/--delay/--max-points out of range");
  }

  const std::vector<double> series =
      sample_series(o, transient, steps, component);

  const auto result = chaosys::correlation_dimension_series(
      series, embed_dim, delay, max_points);

  Output output(text(args, "--out", "-"));
  std::ostream& out = output.stream();
  out << "system,component,embed_dim,delay,dimension\n"
      << std::setprecision(12);
  out << o.name() << ',' << component << ',' << embed_dim << ',' << delay << ','
      << result.dimension << '\n';
  output.finish();
  return 0;
}

int cmd_entropy(const Args& args) {
  const Options o = options(args);
  const std::size_t transient = count(args, "--transient", 2000);
  const std::size_t steps = count(args, "--steps", 5000);
  const int component = integer(args, "--component", 0);
  const int order = integer(args, "--order", 3);
  const int delay = integer(args, "--delay", 1);
  if (component < 0 || component > 2) {
    throw std::invalid_argument("--component must be 0, 1 or 2");
  }
  if (order < 2 || order > 8 || delay <= 0) {
    throw std::invalid_argument("--order must be 2..8 and --delay > 0");
  }

  const std::vector<double> series =
      sample_series(o, transient, steps, component);
  const double entropy =
      chaosys::permutation_entropy(series, order, delay);

  Output output(text(args, "--out", "-"));
  std::ostream& out = output.stream();
  out << "system,component,order,delay,entropy\n" << std::setprecision(12);
  out << o.name() << ',' << component << ',' << order << ',' << delay << ','
      << entropy << '\n';
  output.finish();
  return 0;
}

int cmd_ensemble(const Args& args) {
  const Options o = options(args);
  const std::size_t n = count(args, "--n", 65536);
  const std::size_t steps = count(args, "--steps", 2048);
  constexpr int kMaxThreads = 64;
  const int threads = std::max(
      1, std::min(kMaxThreads, integer(args, "--threads", 8)));
  if (n == 0) throw std::invalid_argument("--n must be > 0");

  Stats stats{};
  if (o.kind == Kind::Lorenz) {
    chaosys::Ensemble ens;
    chaosys::init_ensemble(ens, n);
    chaosys::rk4_soa_neon_threads(ens, steps, o.dt, threads, o.lp);
    const double* comp[3] = {ens.px(), ens.py(), ens.pz()};
    stats = stats_from(comp, n);
  } else {
    System sys = o.system();
    stats = std::visit(
        [&](const auto& s) {
          return generic_ensemble_stats(s, o, n, steps, threads);
        },
        sys);
  }

  Output output(text(args, "--out", "-"));
  std::ostream& out = output.stream();
  out << "n,steps,threads,mean_x,std_x,mean_y,std_y,mean_z,std_z\n"
      << std::setprecision(12);
  out << n << ',' << steps << ',' << threads << ',' << stats.mean[0] << ','
      << stats.sd[0] << ',' << stats.mean[1] << ',' << stats.sd[1] << ','
      << stats.mean[2] << ',' << stats.sd[2] << '\n';
  output.finish();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 0;
  }
  const std::string cmd = argv[1];
  Args args(argv + 2, argv + argc);
  if (cmd == "version" || cmd == "--version" || cmd == "-V") {
    std::cout << "chaosys_cli 1.0.0\n";
    return 0;
  }
  if (cmd == "systems" || cmd == "list-systems") {
    std::cout << "lorenz\nrossler\nchen\n";
    return 0;
  }
  if (cmd == "help" || cmd == "--help" || cmd == "-h" || flag(args, "--help")) {
    usage();
    return 0;
  }

  try {
    if (cmd == "traj") return cmd_traj(args);
    if (cmd == "section") return cmd_section(args);
    if (cmd == "lyapunov") return cmd_lyapunov(args);
    if (cmd == "sweep") return cmd_sweep(args);
    if (cmd == "ensemble") return cmd_ensemble(args);
    if (cmd == "dimension") return cmd_dimension(args);
    if (cmd == "entropy") return cmd_entropy(args);
    std::cerr << "unknown command: " << cmd << "\n\n";
    usage();
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}

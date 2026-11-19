// Small CSV writer for trajectories, Poincare sections, and numeric tables.
//
// The writer is deliberately dependency-free and stream-based.  Floating-point
// values use std::numeric_limits<T>::max_digits10 so a round trip through the
// file recovers the original values.  Header fields are escaped when they
// contain a comma, quote, CR or LF.
#pragma once

#include <cstddef>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "../chaosys.hpp"
#include "analysis.hpp"

namespace chaosys {
namespace detail {

inline void write_csv_field(std::ostream& out, std::string_view value) {
  const bool quote = value.find_first_of(",\"\r\n") != std::string_view::npos;
  if (!quote) {
    out << value;
    return;
  }
  out << '"';
  for (const char c : value) {
    if (c == '"') out << '"';
    out << c;
  }
  out << '"';
}

inline void write_csv_header(std::ostream& out,
                             const std::vector<std::string>& header) {
  for (std::size_t i = 0; i < header.size(); ++i) {
    if (i != 0) out << ',';
    write_csv_field(out, header[i]);
  }
  out << '\n';
}

inline void check_csv_stream(std::ostream& out) {
  if (!out) {
    throw std::runtime_error("chaosys::write_csv: output stream failed");
  }
}

template <class T>
struct PrecisionGuard {
  std::ostream& out;
  std::streamsize precision;
  ~PrecisionGuard() { out.precision(precision); }
};

inline std::vector<std::string> default_component_header(std::string first,
                                                         int dim) {
  std::vector<std::string> header;
  header.reserve(static_cast<std::size_t>(dim) + (first.empty() ? 0 : 1));
  if (!first.empty()) header.push_back(std::move(first));
  for (int i = 0; i < dim; ++i) {
    header.push_back("x" + std::to_string(i));
  }
  return header;
}

}  // namespace detail

// Generic numeric table.  When `header` is non-empty every row must have
// exactly header.size() entries; without a header the first row fixes the
// column count.
template <std::floating_point T>
void write_csv(std::ostream& out, const std::vector<std::vector<T>>& rows,
               const std::vector<std::string>& header = {}) {
  std::size_t columns = header.size();
  detail::PrecisionGuard<T> guard{out, out.precision()};
  out << std::setprecision(std::numeric_limits<T>::max_digits10);
  if (!header.empty()) detail::write_csv_header(out, header);

  for (const auto& row : rows) {
    if (columns == 0) columns = row.size();
    if (row.size() != columns) {
      throw std::invalid_argument(
          "chaosys::write_csv: inconsistent row width");
    }
    for (std::size_t i = 0; i < row.size(); ++i) {
      if (i != 0) out << ',';
      out << row[i];
    }
    out << '\n';
  }
  detail::check_csv_stream(out);
}

// One-dimensional series, one value per line.  Useful for embedding a scalar
// observable or exporting a Lyapunov series.
template <std::floating_point T>
void write_csv(std::ostream& out, const std::vector<T>& values,
               const std::vector<std::string>& header = {"value"}) {
  detail::PrecisionGuard<T> guard{out, out.precision()};
  out << std::setprecision(std::numeric_limits<T>::max_digits10);
  if (!header.empty()) {
    if (header.size() != 1) {
      throw std::invalid_argument(
          "chaosys::write_csv: a scalar series needs at most one header");
    }
    detail::write_csv_header(out, header);
  }
  for (const T value : values) out << value << '\n';
  detail::check_csv_stream(out);
}

// Lorenz state trajectory: x,y,z by default.
inline void write_csv(std::ostream& out, const std::vector<State>& rows,
                      const std::vector<std::string>& header = {"x", "y",
                                                                "z"}) {
  detail::PrecisionGuard<double> guard{out, out.precision()};
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  if (!header.empty()) {
    if (header.size() != 3) {
      throw std::invalid_argument(
          "chaosys::write_csv: state header needs three columns");
    }
    detail::write_csv_header(out, header);
  }
  for (const State& state : rows) {
    out << state.x << ',' << state.y << ',' << state.z << '\n';
  }
  detail::check_csv_stream(out);
}

// Poincare section: t and the full state.  The default header is
// `t,x0,x1,...`; callers may pass a header with exactly dim + 1 columns.
template <OdeSystem Sys>
void write_csv(std::ostream& out,
               const std::vector<SectionPoint<Sys>>& rows,
               const std::vector<std::string>& header = {}) {
  using T = typename Sys::real;
  constexpr std::size_t N = static_cast<std::size_t>(Sys::dim);
  detail::PrecisionGuard<T> guard{out, out.precision()};
  out << std::setprecision(std::numeric_limits<T>::max_digits10);
  if (!header.empty()) {
    if (header.size() != N + 1) {
      throw std::invalid_argument(
          "chaosys::write_csv: section header needs dim + 1 columns");
    }
    detail::write_csv_header(out, header);
  } else if (!rows.empty()) {
    detail::write_csv_header(out,
                             detail::default_component_header("t", Sys::dim));
  }
  for (const SectionPoint<Sys>& point : rows) {
    out << point.t;
    for (std::size_t i = 0; i < N; ++i) out << ',' << point.x[i];
    out << '\n';
  }
  detail::check_csv_stream(out);
}

}  // namespace chaosys

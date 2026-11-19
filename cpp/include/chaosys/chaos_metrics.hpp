// Phase-space reconstruction (Takens embedding) and the Grassberger-Procaccia
// correlation dimension.  Both routines are header-only and work on plain
// std::vector data, so they can be used with any chaotic system.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace chaosys {

template <class T>
std::vector<std::vector<T>> embed(const std::vector<T>& series,
                                  int dimension, int delay) {
  if (dimension <= 0 || delay <= 0) {
    throw std::invalid_argument(
        "chaosys::embed: dimension and delay must be > 0");
  }
  const std::size_t m = static_cast<std::size_t>(dimension);
  const std::size_t tau = static_cast<std::size_t>(delay);
  if (m - 1 > (std::numeric_limits<std::size_t>::max() - 1) / tau) {
    throw std::invalid_argument("chaosys::embed: dimension * delay overflows");
  }
  const std::size_t span = (m - 1) * tau;
  if (series.size() < span + 1) {
    throw std::invalid_argument("chaosys::embed: series too short");
  }
  const std::size_t count = series.size() - span;
  for (const T v : series) {
    if (!std::isfinite(static_cast<double>(v))) {
      throw std::invalid_argument("chaosys::embed: non-finite sample");
    }
  }
  std::vector<std::vector<T>> out(count, std::vector<T>(m));
  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      out[i][j] = series[i + j * tau];
    }
  }
  return out;
}

struct CorrelationDimensionResult {
  double dimension = 0.0;
  std::vector<double> log_r;
  std::vector<double> log_c;
  std::size_t pairs = 0;
  std::size_t points = 0;        // points used after decimation
  std::size_t stride = 1;        // decimation stride
  std::size_t embedding_dim = 0; // coordinate count per point
};

// Grassberger-Procaccia estimate from already embedded points.  Points are
// subsampled uniformly when there are more than max_points; the fit uses the
// scaling region 0.002 <= C(r) <= 0.1.
template <class T>
CorrelationDimensionResult correlation_dimension(
    const std::vector<std::vector<T>>& all_points,
    std::size_t max_points = 2000, std::size_t bins = 24,
    std::size_t theiler = 0) {
  if (all_points.size() < 8) {
    throw std::invalid_argument(
        "chaosys::correlation_dimension: need at least 8 points");
  }
  if (max_points < 8 || bins < 4) {
    throw std::invalid_argument(
        "chaosys::correlation_dimension: max_points >= 8 and bins >= 4");
  }
  const std::size_t dim = all_points[0].size();
  if (dim == 0) {
    throw std::invalid_argument(
        "chaosys::correlation_dimension: empty point dimension");
  }

  const std::size_t stride =
      all_points.size() > max_points
          ? (all_points.size() + max_points - 1) / max_points
          : 1;
  std::vector<const std::vector<T>*> points;
  points.reserve(all_points.size() / stride + 1);
  for (std::size_t i = 0; i < all_points.size(); i += stride) {
    if (all_points[i].size() != dim) {
      throw std::invalid_argument(
          "chaosys::correlation_dimension: inconsistent point dimension");
    }
    for (const T v : all_points[i]) {
      if (!std::isfinite(static_cast<double>(v))) {
        throw std::invalid_argument(
            "chaosys::correlation_dimension: non-finite coordinate");
      }
    }
    points.push_back(&all_points[i]);
  }
  if (points.size() < 8) {
    throw std::invalid_argument(
        "chaosys::correlation_dimension: too few points after subsampling");
  }

  // Diameter of the point cloud.
  double diameter = 0.0;
  for (std::size_t j = 0; j < dim; ++j) {
    double lo = static_cast<double>(points[0]->at(j));
    double hi = lo;
    for (const auto* p : points) {
      const double v = static_cast<double>((*p)[j]);
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    diameter = std::max(diameter, hi - lo);
  }
  if (!(diameter > 0.0) || !std::isfinite(diameter)) {
    throw std::runtime_error(
        "chaosys::correlation_dimension: degenerate point cloud");
  }

  const double r_min = diameter * 1e-3;
  const double r_max = diameter * 0.5;
  if (!(r_max > r_min)) {
    throw std::runtime_error(
        "chaosys::correlation_dimension: invalid radius range");
  }
  std::vector<double> radii(bins);
  const double log_min = std::log(r_min);
  const double log_max = std::log(r_max);
  for (std::size_t k = 0; k < bins; ++k) {
    const double t = static_cast<double>(k) / static_cast<double>(bins - 1);
    radii[k] = std::exp(log_min + t * (log_max - log_min));
  }

  // `theiler` is expressed in original samples; decimation makes `j - i`
  // count stride samples, so convert the window to decimated indices.
  const std::size_t theiler_points = theiler / stride;
  std::vector<std::size_t> hist(bins, 0);
  std::size_t pairs = 0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    for (std::size_t j = i + 1; j < points.size(); ++j) {
      if (j - i <= theiler_points) continue;
      double d2 = 0.0;
      for (std::size_t k = 0; k < dim; ++k) {
        const double d = static_cast<double>((*points[i])[k]) -
                         static_cast<double>((*points[j])[k]);
        d2 += d * d;
      }
      const double distance = std::sqrt(d2);
      if (!(distance > 0.0)) continue;
      const auto it = std::lower_bound(radii.begin(), radii.end(), distance);
      if (it != radii.end()) {
        hist[static_cast<std::size_t>(it - radii.begin())]++;
      }
      ++pairs;
    }
  }
  if (pairs < 8) {
    throw std::runtime_error(
        "chaosys::correlation_dimension: too few distinct pairs");
  }

  std::vector<std::size_t> cumulative(bins, 0);
  std::size_t sum = 0;
  for (std::size_t k = 0; k < bins; ++k) {
    sum += hist[k];
    cumulative[k] = sum;
  }

  CorrelationDimensionResult result;
  result.pairs = pairs;
  result.points = points.size();
  result.stride = stride;
  result.embedding_dim = dim;
  result.log_r.reserve(bins);
  result.log_c.reserve(bins);
  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  std::size_t fit = 0;
  for (std::size_t k = 0; k < bins; ++k) {
    const double c = static_cast<double>(cumulative[k]) /
                     static_cast<double>(pairs);
    result.log_r.push_back(std::log(radii[k]));
    result.log_c.push_back(c > 0.0 ? std::log(c)
                                   : -std::numeric_limits<double>::infinity());
    if (c >= 0.002 && c <= 0.1) {
      const double x = std::log(radii[k]);
      const double y = std::log(c);
      sx += x;
      sy += y;
      sxx += x * x;
      sxy += x * y;
      ++fit;
    }
  }
  if (fit < 3) {
    throw std::runtime_error(
        "chaosys::correlation_dimension: scaling region too small");
  }
  const double denom = static_cast<double>(fit) * sxx - sx * sx;
  if (!(std::abs(denom) > 0.0)) {
    throw std::runtime_error(
        "chaosys::correlation_dimension: singular fit");
  }
  result.dimension =
      (static_cast<double>(fit) * sxy - sx * sy) / denom;
  if (!std::isfinite(result.dimension)) {
    throw std::runtime_error(
        "chaosys::correlation_dimension: non-finite result");
  }
  return result;
}

// Normalized permutation entropy in [0, 1].  Patterns are ordered by value,
// with ties broken by time index.  order is limited to 2..8 so order! fits in
// a small vector.
template <class T>
double permutation_entropy(const std::vector<T>& series, int order = 3,
                           int delay = 1) {
  if (order < 2 || order > 8 || delay <= 0) {
    throw std::invalid_argument(
        "chaosys::permutation_entropy: order in [2,8], delay > 0");
  }
  const std::size_t m = static_cast<std::size_t>(order);
  const std::size_t tau = static_cast<std::size_t>(delay);
  if (m - 1 > (std::numeric_limits<std::size_t>::max() - 1) / tau) {
    throw std::invalid_argument(
        "chaosys::permutation_entropy: order * delay overflows");
  }
  const std::size_t span = (m - 1) * tau;
  if (series.size() < span + 1) {
    throw std::invalid_argument(
        "chaosys::permutation_entropy: series too short");
  }
  // std::sort requires a strict weak ordering; NaN would violate it and make
  // the ordinal pattern (and the entropy) undefined.
  for (const T value : series) {
    if (!std::isfinite(static_cast<double>(value))) {
      throw std::invalid_argument(
          "chaosys::permutation_entropy: non-finite sample");
    }
  }
  std::size_t patterns = 1;
  for (std::size_t i = 2; i <= m; ++i) patterns *= i;

  std::vector<std::size_t> counts(patterns, 0);
  std::array<std::size_t, 8> index{};
  const std::size_t windows = series.size() - span;
  for (std::size_t i = 0; i < windows; ++i) {
    for (std::size_t j = 0; j < m; ++j) index[j] = j * tau;
    std::sort(index.begin(), index.begin() + static_cast<std::ptrdiff_t>(m),
              [&](std::size_t a, std::size_t b) {
                if (series[i + a] != series[i + b]) {
                  return series[i + a] < series[i + b];
                }
                return a < b;
              });
    std::size_t code = 0;
    for (std::size_t j = 0; j < m; ++j) {
      std::size_t smaller = 0;
      for (std::size_t k = j + 1; k < m; ++k) {
        if (index[k] < index[j]) ++smaller;
      }
      code = code * (m - j) + smaller;
    }
    ++counts[code];
  }

  double entropy = 0.0;
  for (const std::size_t c : counts) {
    if (c == 0) continue;
    const double p = static_cast<double>(c) /
                     static_cast<double>(windows);
    entropy -= p * std::log(p);
  }
  const double max_entropy = std::log(static_cast<double>(patterns));
  if (!(max_entropy > 0.0)) return 0.0;
  const double normalized = entropy / max_entropy;
  return std::clamp(normalized, 0.0, 1.0);
}

template <class T>
CorrelationDimensionResult correlation_dimension_series(
    const std::vector<T>& series, int dimension, int delay,
    std::size_t max_points = 2000, std::size_t theiler = 0) {
  return correlation_dimension(embed(series, dimension, delay), max_points,
                               24, theiler);
}

}  // namespace chaosys

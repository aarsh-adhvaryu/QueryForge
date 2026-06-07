#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "queryforge/distance.hpp"

// Small test/benchmark helpers shared by the recall tool and the unit tests. Not part of the
// public engine API — these live under tools/ on purpose.
namespace qf_tools {

// Generate `n` vectors of dimension `dim` with values drawn uniformly from [-1, 1].
// Returned flat: vector i occupies [i*dim, (i+1)*dim).
inline std::vector<float> random_dataset(std::size_t n, std::size_t dim, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(n * dim);
  for (auto& x : data) x = dist(rng);
  return data;
}

// Exact k-nearest-neighbor search by linear scan — the "ground truth" we measure recall against.
// Returns the ids of the k closest vectors, nearest first.
inline std::vector<std::uint32_t> brute_force_knn(const float* data, std::size_t n,
                                                  std::size_t dim, const float* query,
                                                  std::size_t k, queryforge::Metric metric) {
  std::vector<std::pair<float, std::uint32_t>> scored(n);
  for (std::size_t i = 0; i < n; ++i) {
    const float* v = data + i * dim;
    const float d = (metric == queryforge::Metric::Cosine)
                        ? queryforge::cosine_distance(query, v, dim)
                        : queryforge::l2_sqr(query, v, dim);
    scored[i] = {d, static_cast<std::uint32_t>(i)};
  }
  k = std::min(k, n);
  std::partial_sort(scored.begin(), scored.begin() + k, scored.end());
  std::vector<std::uint32_t> ids(k);
  for (std::size_t i = 0; i < k; ++i) ids[i] = scored[i].second;
  return ids;
}

// Fraction of the exact top-k that also appear in the approximate top-k. Range [0, 1].
inline double recall_at_k(const std::vector<std::uint32_t>& approx,
                          const std::vector<std::uint32_t>& exact, std::size_t k) {
  k = std::min(k, exact.size());
  if (k == 0) return 1.0;
  std::size_t hits = 0;
  for (std::size_t i = 0; i < k; ++i) {
    if (std::find(approx.begin(), approx.end(), exact[i]) != approx.end()) ++hits;
  }
  return static_cast<double>(hits) / static_cast<double>(k);
}

}  // namespace qf_tools

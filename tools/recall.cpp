// qf_recall — build an NSW index over synthetic vectors and report recall + latency vs the
// exact brute-force answer. This is how we validate accuracy and fill in docs/BASELINES.md.
//
// Usage (all optional, key=value):
//   qf_recall N=5000 dim=128 M=16 efc=200 ef=50 k=10 queries=500 metric=l2 seed=42
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "bruteforce.hpp"
#include "queryforge/nsw.hpp"

namespace {

struct Config {
  std::size_t N = 5000;
  std::size_t dim = 128;
  std::size_t M = 16;
  std::size_t efc = 200;   // ef_construction
  std::size_t ef = 50;     // ef_search
  std::size_t k = 10;
  std::size_t queries = 500;
  std::uint32_t seed = 42;
  queryforge::Metric metric = queryforge::Metric::L2;
};

std::size_t parse_size(const std::string& v) { return static_cast<std::size_t>(std::stoull(v)); }

Config parse_args(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    const auto eq = arg.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = arg.substr(0, eq);
    const std::string val = arg.substr(eq + 1);
    if (key == "N") c.N = parse_size(val);
    else if (key == "dim") c.dim = parse_size(val);
    else if (key == "M") c.M = parse_size(val);
    else if (key == "efc") c.efc = parse_size(val);
    else if (key == "ef") c.ef = parse_size(val);
    else if (key == "k") c.k = parse_size(val);
    else if (key == "queries") c.queries = parse_size(val);
    else if (key == "seed") c.seed = static_cast<std::uint32_t>(parse_size(val));
    else if (key == "metric") c.metric = (val == "cosine") ? queryforge::Metric::Cosine
                                                           : queryforge::Metric::L2;
    else std::cerr << "ignoring unknown arg: " << arg << "\n";
  }
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  using clock = std::chrono::steady_clock;
  const Config c = parse_args(argc, argv);

  std::cout << "QueryForge recall harness\n"
            << "  N=" << c.N << " dim=" << c.dim << " M=" << c.M << " efc=" << c.efc
            << " ef=" << c.ef << " k=" << c.k << " queries=" << c.queries
            << " metric=" << (c.metric == queryforge::Metric::Cosine ? "cosine" : "l2") << "\n";

  const auto data = qf_tools::random_dataset(c.N, c.dim, c.seed);
  const auto qdata = qf_tools::random_dataset(c.queries, c.dim, c.seed + 1);

  // ---- Build ----
  auto t0 = clock::now();
  queryforge::NswIndex index(c.dim, c.M, c.efc, c.metric);
  for (std::size_t i = 0; i < c.N; ++i) index.add(&data[i * c.dim]);
  auto t1 = clock::now();
  const double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // ---- Search + measure recall against exact brute force ----
  double recall_sum = 0.0;
  double search_us_sum = 0.0;
  std::size_t visited_sum = 0;
  for (std::size_t qi = 0; qi < c.queries; ++qi) {
    const float* q = &qdata[qi * c.dim];

    queryforge::SearchStats stats;
    auto s0 = clock::now();
    const auto approx = index.search(q, c.k, c.ef, &stats);
    auto s1 = clock::now();
    search_us_sum += std::chrono::duration<double, std::micro>(s1 - s0).count();
    visited_sum += stats.nodes_visited;

    std::vector<std::uint32_t> approx_ids;
    approx_ids.reserve(approx.size());
    for (const auto& n : approx) approx_ids.push_back(n.id);

    const auto exact = qf_tools::brute_force_knn(data.data(), c.N, c.dim, q, c.k, c.metric);
    recall_sum += qf_tools::recall_at_k(approx_ids, exact, c.k);
  }

  const double recall = recall_sum / static_cast<double>(c.queries);
  const double avg_us = search_us_sum / static_cast<double>(c.queries);
  const double avg_visited = static_cast<double>(visited_sum) / static_cast<double>(c.queries);

  std::cout << "Results:\n"
            << "  build time      : " << build_ms << " ms (" << (build_ms / c.N * 1000.0)
            << " us/vector)\n"
            << "  Recall@" << c.k << "        : " << (recall * 100.0) << " %\n"
            << "  avg query       : " << avg_us << " us\n"
            << "  avg nodes visited: " << avg_visited << " of " << c.N << " ("
            << (avg_visited / c.N * 100.0) << " %)\n";
  return 0;
}

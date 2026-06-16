#include <gtest/gtest.h>

#include <vector>

#include "bruteforce.hpp"
#include "queryforge/hnsw.hpp"

namespace qf = queryforge;

namespace {

double measure_recall(qf::Metric metric, std::size_t N, std::size_t dim, std::size_t M,
                      std::size_t efc, std::size_t ef, std::size_t k, std::size_t queries) {
  const auto data = qf_tools::random_dataset(N, dim, /*seed=*/11);
  const auto qd = qf_tools::random_dataset(queries, dim, /*seed=*/12);

  qf::HnswIndex index(dim, M, efc, metric, /*seed=*/100);
  for (std::size_t i = 0; i < N; ++i) index.add(&data[i * dim]);

  double recall_sum = 0.0;
  for (std::size_t qi = 0; qi < queries; ++qi) {
    const float* q = &qd[qi * dim];
    const auto approx = index.search(q, k, ef);
    std::vector<std::uint32_t> ids;
    for (const auto& n : approx) ids.push_back(n.id);
    const auto exact = qf_tools::brute_force_knn(data.data(), N, dim, q, k, metric);
    recall_sum += qf_tools::recall_at_k(ids, exact, k);
  }
  return recall_sum / static_cast<double>(queries);
}

}  // namespace

TEST(Hnsw, SingleNodeReturnsItself) {
  qf::HnswIndex index(/*dim=*/4, /*M=*/8, /*efc=*/10, qf::Metric::L2);
  const std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::uint32_t id = index.add(v.data());
  const auto res = index.search(v.data(), /*k=*/1, /*ef=*/10);
  ASSERT_EQ(res.size(), 1u);
  EXPECT_EQ(res[0].id, id);
  EXPECT_NEAR(res[0].distance, 0.0f, 1e-6f);
}

TEST(Hnsw, ResultsAreSortedNearestFirst) {
  const auto data = qf_tools::random_dataset(/*n=*/500, /*dim=*/16, /*seed=*/3);
  qf::HnswIndex index(16, 16, 100, qf::Metric::L2);
  for (std::size_t i = 0; i < 500; ++i) index.add(&data[i * 16]);
  const auto res = index.search(&data[0], /*k=*/10, /*ef=*/50);
  ASSERT_FALSE(res.empty());
  for (std::size_t i = 1; i < res.size(); ++i) EXPECT_LE(res[i - 1].distance, res[i].distance);
  EXPECT_EQ(res[0].id, 0u);
}

TEST(Hnsw, BuildsMultipleLayers) {
  // With enough nodes the random layer assignment should produce a hierarchy (top layer > 0).
  const auto data = qf_tools::random_dataset(5000, 16, /*seed=*/5);
  qf::HnswIndex index(16, 16, 100, qf::Metric::L2);
  for (std::size_t i = 0; i < 5000; ++i) index.add(&data[i * 16]);
  EXPECT_GT(index.max_layer(), 0);
}

TEST(Hnsw, RecallL2IsHigh) {
  const double recall = measure_recall(qf::Metric::L2, /*N=*/3000, /*dim=*/32, /*M=*/16,
                                       /*efc=*/200, /*ef=*/50, /*k=*/10, /*queries=*/200);
  EXPECT_GT(recall, 0.90) << "Recall@10 was " << recall;
}

TEST(Hnsw, RecallCosineIsHigh) {
  const double recall = measure_recall(qf::Metric::Cosine, /*N=*/3000, /*dim=*/32, /*M=*/16,
                                       /*efc=*/200, /*ef=*/50, /*k=*/10, /*queries=*/200);
  EXPECT_GT(recall, 0.90) << "Recall@10 was " << recall;
}

TEST(Hnsw, HigherEfImprovesRecall) {
  const double low = measure_recall(qf::Metric::L2, 3000, 32, 16, 200, /*ef=*/10, 10, 150);
  const double high = measure_recall(qf::Metric::L2, 3000, 32, 16, 200, /*ef=*/100, 10, 150);
  EXPECT_GE(high, low) << "low(ef=10)=" << low << " high(ef=100)=" << high;
}

// The parallel build (add_batch_parallel) should reach essentially the same recall as the
// sequential build. The graph isn't bit-identical (edge formation depends on thread interleaving),
// so we assert "high recall, and within a small margin of sequential", not equality.
TEST(Hnsw, ParallelBuildMatchesSequentialRecall) {
  const std::size_t N = 5000, dim = 32, M = 16, efc = 200, ef = 50, k = 10, queries = 200;
  const auto data = qf_tools::random_dataset(N, dim, /*seed=*/11);
  const auto qd = qf_tools::random_dataset(queries, dim, /*seed=*/12);

  qf::HnswIndex seq(dim, M, efc, qf::Metric::L2, /*seed=*/100);
  for (std::size_t i = 0; i < N; ++i) seq.add(&data[i * dim]);

  qf::HnswIndex par(dim, M, efc, qf::Metric::L2, /*seed=*/100);
  par.add_batch_parallel(data.data(), N, /*threads=*/4);
  ASSERT_EQ(par.size(), N);

  auto recall_of = [&](const qf::HnswIndex& idx) {
    double sum = 0.0;
    for (std::size_t qi = 0; qi < queries; ++qi) {
      const float* q = &qd[qi * dim];
      std::vector<std::uint32_t> ids;
      for (const auto& n : idx.search(q, k, ef)) ids.push_back(n.id);
      const auto exact = qf_tools::brute_force_knn(data.data(), N, dim, q, k, qf::Metric::L2);
      sum += qf_tools::recall_at_k(ids, exact, k);
    }
    return sum / static_cast<double>(queries);
  };

  const double rseq = recall_of(seq);
  const double rpar = recall_of(par);
  EXPECT_GT(rpar, 0.85) << "parallel recall too low: " << rpar;
  EXPECT_NEAR(rpar, rseq, 0.05) << "seq=" << rseq << " par=" << rpar;
}

TEST(Hnsw, ParallelBuildRejectsNonEmptyIndex) {
  qf::HnswIndex idx(/*dim=*/4, /*M=*/8, /*efc=*/10, qf::Metric::L2);
  const std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
  idx.add(v.data());
  EXPECT_THROW(idx.add_batch_parallel(v.data(), 1, /*threads=*/2), std::logic_error);
}

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdio>
#include <string>
#include <vector>

#include "bruteforce.hpp"
#include "queryforge/hnsw.hpp"

namespace qf = queryforge;

namespace {
// A unique temp file path that is removed when the test object goes out of scope.
struct TempFile {
  std::string path;
  TempFile() {
    static int counter = 0;
    path = "/tmp/qf_test_" + std::to_string(::getpid()) + "_" + std::to_string(counter++) + ".qfx";
  }
  ~TempFile() { std::remove(path.c_str()); }
};
}  // namespace

TEST(Persistence, RoundTripSearchIsIdentical) {
  const std::size_t N = 2000, dim = 32;
  const auto data = qf_tools::random_dataset(N, dim, /*seed=*/21);
  const auto queries = qf_tools::random_dataset(50, dim, /*seed=*/22);

  qf::HnswIndex original(dim, /*M=*/16, /*efc=*/200, qf::Metric::L2);
  for (std::size_t i = 0; i < N; ++i) original.add(&data[i * dim]);

  TempFile tmp;
  original.save(tmp.path);
  qf::HnswIndex loaded = qf::HnswIndex::load(tmp.path);

  // Structural metadata must match.
  EXPECT_EQ(loaded.size(), original.size());
  EXPECT_EQ(loaded.dim(), original.dim());
  EXPECT_EQ(loaded.max_layer(), original.max_layer());
  EXPECT_EQ(loaded.metric(), original.metric());

  // Every query must return exactly the same neighbors in the same order.
  for (std::size_t qi = 0; qi < 50; ++qi) {
    const float* q = &queries[qi * dim];
    const auto a = original.search(q, /*k=*/10, /*ef=*/64);
    const auto b = loaded.search(q, /*k=*/10, /*ef=*/64);
    ASSERT_EQ(a.size(), b.size()) << "query " << qi;
    for (std::size_t i = 0; i < a.size(); ++i) {
      EXPECT_EQ(a[i].id, b[i].id) << "query " << qi << " pos " << i;
      EXPECT_FLOAT_EQ(a[i].distance, b[i].distance);
    }
  }
}

TEST(Persistence, CosineRoundTrips) {
  const std::size_t N = 1000, dim = 24;
  const auto data = qf_tools::random_dataset(N, dim, /*seed=*/31);
  qf::HnswIndex original(dim, 16, 100, qf::Metric::Cosine);
  for (std::size_t i = 0; i < N; ++i) original.add(&data[i * dim]);

  TempFile tmp;
  original.save(tmp.path);
  qf::HnswIndex loaded = qf::HnswIndex::load(tmp.path);

  EXPECT_EQ(loaded.metric(), qf::Metric::Cosine);
  const auto a = original.search(&data[0], 5, 32);
  const auto b = loaded.search(&data[0], 5, 32);
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i].id, b[i].id);
}

TEST(Persistence, BadFileThrows) {
  EXPECT_THROW(qf::HnswIndex::load("/nonexistent/path/to/index.qfx"), std::exception);
}

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "queryforge/distance.hpp"

namespace qf = queryforge;

namespace {

// Fill a vector with deterministic pseudo-random values in [-1, 1].
std::vector<float> random_vec(std::size_t dim, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (auto& x : v) x = dist(rng);
  return v;
}

// SIMD sums floats in a different order than the scalar loop, so results differ in the last
// bit or two. Compare with a tolerance that scales with the dimension (more adds -> more drift).
float tolerance_for(std::size_t dim) {
  return 1e-4f * static_cast<float>(dim);
}

}  // namespace

// --- Known-value sanity checks ----------------------------------------------------------

TEST(Distance, L2KnownValues) {
  const std::vector<float> a = {1.0f, 2.0f, 3.0f};
  const std::vector<float> b = {4.0f, 6.0f, 3.0f};
  // diffs: -3, -4, 0  -> 9 + 16 + 0 = 25
  EXPECT_FLOAT_EQ(qf::l2_sqr(a.data(), b.data(), a.size()), 25.0f);
  EXPECT_FLOAT_EQ(qf::l2_sqr(a.data(), a.data(), a.size()), 0.0f);
}

TEST(Distance, DotKnownValues) {
  const std::vector<float> a = {1.0f, 2.0f, 3.0f};
  const std::vector<float> b = {4.0f, 5.0f, 6.0f};
  // 4 + 10 + 18 = 32
  EXPECT_FLOAT_EQ(qf::dot(a.data(), b.data(), a.size()), 32.0f);
}

TEST(Distance, CosineKnownValues) {
  const std::vector<float> a = {1.0f, 0.0f};
  const std::vector<float> same = {2.0f, 0.0f};       // same direction -> distance 0
  const std::vector<float> orth = {0.0f, 1.0f};       // orthogonal     -> distance 1
  const std::vector<float> opp = {-1.0f, 0.0f};       // opposite       -> distance 2
  EXPECT_NEAR(qf::cosine_distance(a.data(), same.data(), a.size()), 0.0f, 1e-6f);
  EXPECT_NEAR(qf::cosine_distance(a.data(), orth.data(), a.size()), 1.0f, 1e-6f);
  EXPECT_NEAR(qf::cosine_distance(a.data(), opp.data(), a.size()), 2.0f, 1e-6f);
}

// --- All SIMD variants must agree with the scalar reference -----------------------------
// Includes dimensions that are NOT multiples of 8 (13, 17, 100) to exercise the scalar tail.

TEST(Distance, VariantsAgreeWithScalar) {
  const std::size_t dims[] = {1, 7, 8, 13, 16, 17, 100, 128, 512, 768};
  for (std::size_t dim : dims) {
    const auto a = random_vec(dim, /*seed=*/dim * 2 + 1);
    const auto b = random_vec(dim, /*seed=*/dim * 2 + 2);
    const float tol = tolerance_for(dim);

    const float l2_ref = qf::detail::l2_sqr_scalar(a.data(), b.data(), dim);
    const float dot_ref = qf::detail::dot_scalar(a.data(), b.data(), dim);

    if (qf::detail::cpu_has_sse()) {
      EXPECT_NEAR(qf::detail::l2_sqr_sse(a.data(), b.data(), dim), l2_ref, tol) << "dim=" << dim;
      EXPECT_NEAR(qf::detail::dot_sse(a.data(), b.data(), dim), dot_ref, tol) << "dim=" << dim;
    }
    if (qf::detail::cpu_has_avx2()) {
      EXPECT_NEAR(qf::detail::l2_sqr_avx2(a.data(), b.data(), dim), l2_ref, tol) << "dim=" << dim;
      EXPECT_NEAR(qf::detail::dot_avx2(a.data(), b.data(), dim), dot_ref, tol) << "dim=" << dim;
    }

    // The public dispatched entry points must also agree with the reference.
    EXPECT_NEAR(qf::l2_sqr(a.data(), b.data(), dim), l2_ref, tol) << "dim=" << dim;
    EXPECT_NEAR(qf::dot(a.data(), b.data(), dim), dot_ref, tol) << "dim=" << dim;
  }
}

TEST(Distance, ReportsABackend) {
  // Just confirm the dispatcher resolved to something sensible and is printable.
  const qf::SimdBackend b = qf::active_backend();
  SUCCEED() << "active SIMD backend: " << qf::backend_name(b);
}

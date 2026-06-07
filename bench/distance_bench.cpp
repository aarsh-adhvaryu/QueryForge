#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "queryforge/distance.hpp"

namespace qf = queryforge;

namespace {

std::vector<float> random_vec(std::size_t dim, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (auto& x : v) x = dist(rng);
  return v;
}

// Benchmark template: runs `fn` over two random vectors of dimension `dim`.
template <typename Fn>
void run(benchmark::State& state, Fn fn) {
  const std::size_t dim = static_cast<std::size_t>(state.range(0));
  const auto a = random_vec(dim, 1);
  const auto b = random_vec(dim, 2);
  for (auto _ : state) {
    benchmark::DoNotOptimize(fn(a.data(), b.data(), dim));
  }
  // Report throughput in items (floats) processed per second.
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(dim));
}

}  // namespace

// We benchmark the two hot kernels (L2 and dot) across scalar / SSE / AVX2 at CLIP-like
// dimensions (512 = ViT-B, 768 = ViT-L). Variants the CPU can't run are skipped at runtime.
#define QF_BENCH_DIMS Args({512})->Args({768})

static void BM_L2_Scalar(benchmark::State& s) { run(s, qf::detail::l2_sqr_scalar); }
BENCHMARK(BM_L2_Scalar)->QF_BENCH_DIMS;

static void BM_L2_SSE(benchmark::State& s) {
  if (!qf::detail::cpu_has_sse()) { s.SkipWithError("no SSE4.1"); return; }
  run(s, qf::detail::l2_sqr_sse);
}
BENCHMARK(BM_L2_SSE)->QF_BENCH_DIMS;

static void BM_L2_AVX2(benchmark::State& s) {
  if (!qf::detail::cpu_has_avx2()) { s.SkipWithError("no AVX2"); return; }
  run(s, qf::detail::l2_sqr_avx2);
}
BENCHMARK(BM_L2_AVX2)->QF_BENCH_DIMS;

static void BM_Dot_Scalar(benchmark::State& s) { run(s, qf::detail::dot_scalar); }
BENCHMARK(BM_Dot_Scalar)->QF_BENCH_DIMS;

static void BM_Dot_SSE(benchmark::State& s) {
  if (!qf::detail::cpu_has_sse()) { s.SkipWithError("no SSE4.1"); return; }
  run(s, qf::detail::dot_sse);
}
BENCHMARK(BM_Dot_SSE)->QF_BENCH_DIMS;

static void BM_Dot_AVX2(benchmark::State& s) {
  if (!qf::detail::cpu_has_avx2()) { s.SkipWithError("no AVX2"); return; }
  run(s, qf::detail::dot_avx2);
}
BENCHMARK(BM_Dot_AVX2)->QF_BENCH_DIMS;

BENCHMARK_MAIN();

#include <benchmark/benchmark.h>

#include "queryforge/version.hpp"

// A0 smoke benchmark: proves the Google Benchmark harness is wired up. The real
// benchmarks (scalar vs SSE vs AVX2 distance throughput) arrive in stage A1, and
// their numbers get recorded in docs/BASELINES.md.
static void BM_Version(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(queryforge::version());
  }
}
BENCHMARK(BM_Version);

BENCHMARK_MAIN();

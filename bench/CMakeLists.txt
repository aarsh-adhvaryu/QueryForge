# Benchmarks, built on Google Benchmark (fetched the same way as GoogleTest).
include(FetchContent)

# Don't build Google Benchmark's own tests — we only want the library.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  benchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
  # TODO: pin URL_HASH SHA256=... once verified, for fully reproducible fetches.
)
FetchContent_MakeAvailable(benchmark)

add_executable(qf_bench
  smoke_bench.cpp
)
target_link_libraries(qf_bench
  PRIVATE
    QueryForge::queryforge
    benchmark::benchmark
)

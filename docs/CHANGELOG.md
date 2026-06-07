# Changelog

Human-readable summary of what changed, stage by stage. Newest on top.

## A1 — SIMD distance math
- Added `queryforge::l2_sqr`, `dot`, and `cosine_distance` with scalar, SSE, and AVX2+FMA
  implementations (`include/queryforge/distance.hpp`, `src/distance.cpp`).
- Runtime CPU dispatch (`active_backend()`) picks AVX2 > SSE > scalar; one binary runs anywhere.
- Tests: all variants agree with the scalar reference (incl. non-multiple-of-8 dims) plus
  known-value and cosine sanity checks (`tests/distance_test.cpp`).
- Benchmarks across 512/768 dims (`bench/distance_bench.cpp`); numbers recorded in `BASELINES.md`
  (AVX2 ≈ 8x scalar).

## A0 — Repo scaffold
- Added top-level CMake build (C++17, Release by default) with `QF_BUILD_TESTS` /
  `QF_BUILD_BENCHMARKS` options.
- Added the `queryforge` engine library (currently just `version()`), exposed as
  `QueryForge::queryforge`.
- Wired up GoogleTest (via FetchContent) with a smoke test, and Google Benchmark with a
  smoke benchmark — proving the full build/test/bench loop works.
- Created the directory skeleton: `include/`, `src/`, `tests/`, `bench/`, `tools/`,
  `python/`, `backend/`, `frontend/`, `docs/`.
- Added `.gitignore`, `README.md`, and docs (`architecture.md`, `OBSERVATIONS.md`,
  `BASELINES.md`, this changelog).

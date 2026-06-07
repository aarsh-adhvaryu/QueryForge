# Changelog

Human-readable summary of what changed, stage by stage. Newest on top.

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

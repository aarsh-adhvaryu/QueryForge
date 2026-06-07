# Changelog

Human-readable summary of what changed, stage by stage. Newest on top.

## A4 — Persistence (binary serialize + mmap load)
- `HnswIndex::save(path)` / `HnswIndex::load(path)` with a binary `.qfx` format (header + vectors
  block + CSR per-layer adjacency); `src/persistence.cpp`. POSIX `mmap` load with a buffered-read
  fallback, RAII-managed.
- Tests: L2 + cosine round-trip search identical, bad-file throws (`tests/persistence_test.cpp`),
  20 tests pass.
- Tool `qf_persist` reports build vs load time + file size; baselines in `BASELINES.md`
  (load ~1000-1570× faster than rebuild, ~640 bytes/vector).
- Logged the O(N^2) build bottleneck (per-insert size-N visited allocation) for the perf pass.

## A3 (step 2) — HNSW diversity neighbor-selection heuristic
- Replaced naive closest-M selection with the HNSW paper's Algorithm 4 diversity heuristic in
  `HnswIndex::select_neighbors`.
- Result: crossed the 95% Recall@10 target (95.9% @ ef=200, M=16) and ~35% faster low-ef queries;
  see `BASELINES.md` step 2. All 17 tests still pass.

## A3 (step 1) — HNSW multi-layer structure
- Added `HnswIndex` (`include/queryforge/hnsw.hpp`, `src/hnsw.cpp`): random exponential layer
  assignment, top-down greedy descent on upper layers (ef=1), wide beam search at layer 0,
  per-layer neighbor lists, layer-0 density = 2*M, reverse edges with pruning. Naive neighbor
  selection for now (diversity heuristic is step 2).
- Moved shared `Neighbor`/`SearchStats` into `include/queryforge/types.hpp`.
- `qf_recall` now takes `algo=nsw|hnsw` and prints the top layer; templated over index type.
- Tests: single-node, sorted output, multi-layer hierarchy, L2/cosine recall >90%, ef-monotonicity
  (`tests/hnsw_test.cpp`). 17 tests pass.
- Baselines: layers add +14–21 recall points vs NSW at equal nodes-visited; HNSW@M=16 matches
  NSW@M=32 (see `BASELINES.md`).

## A2 — NSW single-layer graph
- Added `NswIndex` (`include/queryforge/nsw.hpp`, `src/nsw.cpp`): greedy beam search
  (`search_layer`), insert with bidirectional links + degree pruning, flat contiguous adjacency,
  L2 + cosine support, and `SearchStats` (nodes visited / distance computations).
- Added the recall harness `qf_recall` (`tools/recall.cpp`) and shared brute-force ground-truth
  helpers (`tools/bruteforce.hpp`); new `QF_BUILD_TOOLS` CMake option.
- Tests: single-node, sorted-output, L2/cosine recall thresholds, and "higher ef ⇒ higher recall"
  (`tests/nsw_test.cpp`). 11 tests pass.
- Baselines recorded in `BASELINES.md` (recall vs ef and M; ~2% of nodes visited).

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

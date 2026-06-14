# Changelog

Human-readable summary of what changed, stage by stage. Newest on top.

## Pre-data hardening (before the real 500K run)
A file review before bucket B surfaced three fixes (all tested on synthetic data):
- **Batched `ClipEmbedder.embed_images`** (`python/qf_pipeline/embedder.py`): embeds `batch_size`
  images per GPU forward pass (was 1-at-a-time → would've been hours for 500K) and L2-normalizes
  the output. Verified at P4 once open_clip is installed.
- **`HnswIndex::reserve(n)`** (`hnsw.hpp/.cpp`, exposed in bindings; called by `add_batch` and the
  `qf_persist` tool): pre-allocates vectors/adjacency to avoid realloc copies and the transient 2×
  memory spike during a bulk build.
- **Keep-pruned-connections backfill** in `select_neighbors` (`hnsw.cpp`): tops a node's neighbor
  list back up to M from the closest diversity-rejected candidates (standard hnswlib behavior).
  Neutral on uniform-random synthetic data (the diversity rule rarely leaves nodes short in 128-d),
  kept because it's correct and should help on clustered real embeddings. 23 tests still pass.

## P1 — performance pass (in progress, post-bucket-A)
- **VisitedSet** (`src/visited_set.hpp`): reusable thread_local visited set with O(1) version-tag
  clear, replacing per-search `vector<bool>` allocations in `hnsw.cpp` + `nsw.cpp`. (Measurement
  showed this was *not* the build bottleneck; the "O(N²)" claim was corrected to ~O(N^1.4).)
- **Flattened HNSW adjacency + prefetch**: nested `vector<vector<vector>>` replaced with contiguous
  fixed-stride blocks (`links0_` flat array for layer 0 + per-node upper blocks) and
  `__builtin_prefetch` of neighbor vectors in `search_layer`. ~12–15% faster build; recall and the
  `.qfx` on-disk format unchanged; all 23 tests pass. Realizes the doc's "cache-aware contiguous
  neighbor storage." Remaining build levers: parallel build, efConstruction.

## A7 — FastAPI backend + React frontend (mock data) — bucket A complete
- `backend/app.py`: FastAPI service with `/health`, `/catalog`, `/search/id/{id}`,
  `POST /search/image` (raw image bytes), `/images/*`, and serves the frontend at `/`. Builds a
  mock catalog (synthetic images + HistogramEmbedder) on startup; reports real query latency.
- `frontend/index.html`: self-contained React (CDN) UI — upload, catalog browse, results grid,
  latency display. No build step; served by the backend.
- `backend/smoke_test.py` exercises all endpoints via FastAPI TestClient → `python_api` CTest case
  (23 tests with Python on). Verified live via uvicorn + curl.
- Fixed SQLite cross-thread use for the web server (`check_same_thread=False`). Factored synthetic
  image generation into `qf_pipeline/synthetic.py` (shared by dry-run and backend).

## A6 — Embedding pipeline scaffold + CPU dry-run
- `python/qf_pipeline/` package: pluggable `Embedder` (dependency-free `HistogramEmbedder` for the
  CPU dry-run; real `ClipEmbedder` with lazy torch/open_clip for the GPU/laptop), SQLite-backed
  `MetadataStore`, and `build_catalog()` / `search_similar()` tying engine + metadata together.
- Resolved the metadata-store decision for local dev: SQLite (stdlib, zero dependency); Postgres is
  a later swap behind the same interface.
- `qf_pipeline.dry_run`: generates synthetic colored product images, builds the index + catalog,
  and runs a visual-similarity query (8/8 same-category). Wired as the `python_pipeline` CTest case
  (22 tests with Python on). Added `python/requirements.txt`.

## A5 — Pybind11 bindings
- `python/bindings.cpp` exposes `queryforge.HnswIndex` (ctor, add, add_batch, search, save, load,
  size/dim/metric/max_layer) with NumPy float32 in and `(ids, distances)` NumPy arrays out.
- New `QF_BUILD_PYTHON` CMake option (OFF by default); pybind11 fetched via FetchContent; engine
  library built with `POSITION_INDEPENDENT_CODE` so it links into the shared module.
- `python/example.py` demo doubles as the `python_bindings` CTest case (21 tests with Python on).

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

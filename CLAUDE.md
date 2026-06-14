# CLAUDE.md — guidance for AI assistants working in this repo

QueryForge is a **from-scratch C++ HNSW vector search engine** (+ a planned reverse-image-search
demo). It's the owner's portfolio/learning project. Read this first; it tells you what exists, how
to build it, the conventions to follow, and where to resume.

## Working style (important — the owner is learning)

- **Teach as you go.** Explain concepts in plain language, not dense jargon. Expect and welcome
  "why?" and "what does this mean?" questions. Define unfamiliar terms inline in a line or two.
- **Brainstorm, don't bulldoze.** For genuine design forks, surface the choice and the trade-off;
  ask one focused question at a time and say why the answer matters.
- **Measure and record.** Every stage updates `docs/` (see below). A stage isn't "done" until its
  numbers and observations are written down and committed.
- **Commit cadence:** build a stage → run tests/benchmarks → update docs → commit → push to GitHub.
  One stage per commit-ish, so a `git pull` always lands a clean, tested, documented checkpoint.

## Current status (stages done)

Stages follow the plan in `docs/architecture.md` and the project plan. Done so far:

- **A0** — scaffold: CMake build, library, GoogleTest + Google Benchmark, docs system.
- **A1** — SIMD distance math: scalar/SSE/AVX2 L2 + dot + cosine, runtime CPU dispatch. ~8× AVX2.
- **A2** — `NswIndex`: single-layer Navigable Small World graph + recall harness.
- **A3** — `HnswIndex`: full multi-layer HNSW + diversity heuristic. **Recall@10 = 95.9%** on
  synthetic data (N=10k, dim=128, M=16, ef=200), visiting ~2% of nodes. Target met.
- **A4** — persistence: binary `.qfx` save + mmap load (~1000–1570× faster than rebuild).
- **A5** — Pybind11 bindings: `queryforge.HnswIndex` (NumPy in, ids/distances out).
- **A6** — embedding pipeline scaffold: pluggable `Embedder` (HistogramEmbedder for the CPU
  dry-run; `ClipEmbedder` for the GPU/laptop) + SQLite metadata + dry-run.
- **A7** — FastAPI backend + React frontend over a mock catalog; full vertical slice works.

**Bucket A is COMPLETE.** What remains is **bucket B (needs the laptop/GPU):** download the real
500K-image dataset, run real CLIP embeddings (swap `HistogramEmbedder` → `ClipEmbedder`), tune the
index on real data, and re-measure final benchmarks on the local 5070 Ti / Core Ultra 9. Also a
queued perf pass: fix the O(N²) build (reusable visited array) and add parallel build. See
`docs/CHANGELOG.md` and `docs/BASELINES.md`.

Build everything (incl. Python module + web tests): `cmake -S . -B build -DQF_BUILD_PYTHON=ON`.

**Currently in progress — real data (P3/P4):** decided ViT-L/14 → 768-d, "start small then scale".
`open_clip_torch` + `datasets` are installed; dataset = `ashraq/fashion-product-images-small` (real
fashion products w/ articleType/baseColour/productDisplayName). Real-data build script:
`python/qf_pipeline/build_real.py`. NOTE the ViT-L/14 pretrained tag is `laion2b_s32b_b82k` (the
ClipEmbedder default tag is for B/32). **To resume:** toggle the Studio GPU ON, then
`PYTHONPATH=build/python:python python -m qf_pipeline.build_real --limit 300 --out /tmp/qf_real`
(small validation first; raise `--limit` to scale). CPU works but ViT-L/14 is slow — the last run
was interrupted mid-embedding, so build_real is not yet run-verified.

## Performance & time complexity — investigate next

The engine is correct and hits its recall target, but it has **not** had a performance pass. Before
scaling to the 500K dataset, look into these (highest priority first).

**Build complexity — corrected by measurement (see `docs/OBSERVATIONS.md`):** the build is
~**O(N^1.4)** wall-clock, *not* O(N²). The earlier "O(N²) from the `visited` allocation" claim was
wrong — replacing that allocation with a reusable `VisitedSet` (`src/visited_set.hpp`, already done)
changed build time by only ~6%. The real cost is the per-insert distance computations in the
diversity heuristic (≈ O(efConstruction·M) per insert), amplified by **cache misses** once `vectors_`
exceeds L3 (~64k vectors at dim 128) and because the nested-`std::vector` adjacency scatters memory.

Real build-speed levers, in order:
- **Parallel multi-threaded build** — the biggest wall-clock win (÷ cores). `VisitedSet` is
  `thread_local`, so each thread already gets its own scratch — the groundwork is in place.
- **efConstruction tuning** — linear lever (efc 200→100 ≈ halves build time; measured ~no recall
  loss). Pick the lowest efc that holds recall.
- **Flatten layer-0 adjacency** → DONE (`links0_` flat array + per-node upper blocks + prefetch);
  ~12–15% faster build, but the memory wall (random vector reads > L3) still dominates the slope.
- **Reuse the search heaps** (candidate/result priority queues in `search_layer`) per query — TODO.

Reference complexities (HNSW as designed): search ≈ O(log N) nodes visited; build ≈ O(N·log N)
algorithmically (wall-clock ~N^1.4 here due to cache effects); memory ≈ N·dim·4 B (vectors) +
~N·M·4·1.5 B (edges). Always re-confirm changes with `qf_recall` (recall must not regress) and
`qf_persist` (build time). Load time is excellent and unaffected.

## Build / test / run

```bash
cmake -S . -B build            # configure (first run fetches GoogleTest + Google Benchmark)
cmake --build build -j         # compile
ctest --test-dir build --output-on-failure   # run all unit tests
./build/bin/qf_distance_bench  # SIMD distance benchmarks
./build/bin/qf_recall algo=hnsw N=10000 dim=128 M=16 efc=200 ef=200 k=10 metric=l2   # recall harness
./build/bin/qf_persist N=30000 dim=128 M=16 efc=200                                  # build vs load time
```

Run the demo (Python module + web app):

```bash
cmake -S . -B build -DQF_BUILD_PYTHON=ON && cmake --build build -j   # build the queryforge module
PYTHONPATH=build/python:python python -m qf_pipeline.dry_run         # CPU end-to-end dry-run
PYTHONPATH=build/python:python uvicorn backend.app:app               # web app at http://127.0.0.1:8000/
```

Notes:
- Requires C++17, CMake ≥ 3.20, internet on the first configure (deps via `FetchContent`).
- The shell's working directory may reset between tool calls — run commands from the repo root.
- Build defaults to Release (`-O3`). Toggle parts with `-DQF_BUILD_TESTS/BENCHMARKS/TOOLS=OFF`.

## Layout

- `include/queryforge/` — public headers: `distance.hpp`, `nsw.hpp`, `hnsw.hpp`, `types.hpp`.
- `src/` — implementations. Add new `.cpp` files to `src/CMakeLists.txt`.
- `tests/` — GoogleTest (`*_test.cpp`); add to `tests/CMakeLists.txt`.
- `bench/` — Google Benchmark; each file has its own `main()` → its own executable.
- `tools/` — `qf_recall` (recall vs brute force), `qf_persist` (build vs mmap-load timing), and
  shared ground-truth helpers (`bruteforce.hpp`).
- `docs/` — `architecture.md` (how/why), `OBSERVATIONS.md` (journal), `CHANGELOG.md`, `BASELINES.md`
  (recorded numbers). **Keep these updated as part of each change.**
- `python/` — `queryforge` Pybind11 module (`bindings.cpp`) + `qf_pipeline` (embedder, SQLite
  metadata, dry-run). `backend/` — FastAPI service (`app.py`). `frontend/` — React UI (`index.html`).

## Conventions

- C++17, `namespace queryforge`. 2-space indent. Headers heavily commented for teaching.
- Distances: L2 is kept **squared** (no sqrt — ordering preserved, faster). Cosine = `1 - dot` on
  vectors normalized at insert time.
- SIMD: per-function `__attribute__((target("avx2,fma")))` + runtime dispatch via
  `__builtin_cpu_supports`. **No `-march=native`** (keeps one binary portable).
- Vectors stored contiguously by node id. Adjacency is flat/contiguous in both engines: NSW uses a
  flat array; HNSW uses fixed-stride blocks (`links0_` flat array for layer 0 + per-node upper
  blocks), each block `[count, ids...]` — see `link_block()` in `hnsw.hpp`.
- Validate any index change with the recall harness / recall tests vs brute force.
- Tuning caveat: uniform-random synthetic vectors are the WORST case for recall (no clusters →
  curse of dimensionality). Real CLIP embeddings cluster and recall much higher at the same M/ef —
  tune on the real data (P5), don't over-tune on synthetic. Lever order: M (32–48) > efSearch > efc.

## Target hardware (local-first)

End goal: runs on the owner's machine — **RTX 5070 Ti** (16 GB) + **Intel Core Ultra 9**. Core Ultra
is **AVX2, no AVX-512**, so AVX2 is the production SIMD target. Cloud/Studio GPU is a convenience for
bulk embedding only, never a runtime dependency. Re-measure baselines on `local` when available.

## Key decisions of record

- Index is **static build + concurrent reads now; dynamic insertion later** (after résumé).
- **Single-threaded build now**; parallel build is a queued, separately-benchmarked enhancement.
- Concurrency claim to make honestly: "lock-free concurrent **reads**" (not lock-free writes).
- Open decisions deferred to their stage: dataset choice, embedding model/dimension (512 vs 768),
  SQLite vs Postgres. See the project plan and `docs/OBSERVATIONS.md`.

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

**Resume at A4 — persistence:** binary serialization of the HNSW graph + mmap-based fast load, and
a metadata store (SQLite default; Postgres optional). Then A5 Pybind11, A6 CLIP embedding scaffold,
A7 FastAPI + React. See `docs/CHANGELOG.md` for details and `docs/BASELINES.md` for numbers.

## Build / test / run

```bash
cmake -S . -B build            # configure (first run fetches GoogleTest + Google Benchmark)
cmake --build build -j         # compile
ctest --test-dir build --output-on-failure   # run all unit tests
./build/bin/qf_distance_bench  # SIMD distance benchmarks
./build/bin/qf_recall algo=hnsw N=10000 dim=128 M=16 efc=200 ef=200 k=10 metric=l2   # recall harness
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
- `tools/` — `qf_recall` and shared brute-force ground-truth helpers (`bruteforce.hpp`).
- `docs/` — `architecture.md` (how/why), `OBSERVATIONS.md` (journal), `CHANGELOG.md`, `BASELINES.md`
  (recorded numbers). **Keep these updated as part of each change.**
- `python/`, `backend/`, `frontend/` — scaffolds for later stages (currently READMEs only).

## Conventions

- C++17, `namespace queryforge`. 2-space indent. Headers heavily commented for teaching.
- Distances: L2 is kept **squared** (no sqrt — ordering preserved, faster). Cosine = `1 - dot` on
  vectors normalized at insert time.
- SIMD: per-function `__attribute__((target("avx2,fma")))` + runtime dispatch via
  `__builtin_cpu_supports`. **No `-march=native`** (keeps one binary portable).
- Vectors stored contiguously by node id. NSW adjacency is a flat array; HNSW adjacency is still
  nested `std::vector` (a logged perf TODO to flatten layer 0).
- Validate any index change with the recall harness / recall tests vs brute force.

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

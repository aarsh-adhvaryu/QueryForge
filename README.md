# QueryForge

A from-scratch **C++ HNSW vector search engine** — built to find the most *similar*
vectors (e.g. image or text embeddings) among millions, in milliseconds, instead of
scanning them all.

> **Status:** bucket A complete — the full vertical slice works on CPU with synthetic data:
> image → embedding → HNSW search (95.9% Recall@10) → SQLite metadata → FastAPI → React grid,
> with `.qfx` persistence and Python bindings. Remaining: the real CLIP model + 500K dataset +
> final tuning/benchmarks on the local GPU machine. See [docs/CHANGELOG.md](docs/CHANGELOG.md).

## What this is (in one paragraph)

A *vector* is a list of numbers a neural network produces to describe something. Similar
things get numerically close vectors. The core question is "given this vector, find the
closest ones." Doing that by comparing against every vector ("brute force") is too slow at
scale. **HNSW** (Hierarchical Navigable Small Worlds) is a graph structure — local streets
at the bottom, express highways on top — that finds close matches by visiting only a tiny
fraction of the data. QueryForge implements that engine, plus the hardware-level distance
math (**SIMD/AVX2**) that makes each comparison fast.

## Guiding principle: local-first

The end goal is that everything runs on a single consumer machine
(**RTX 5070 Ti** GPU + **Intel Core Ultra 9** CPU). Cloud GPUs are a convenience for
one-off heavy steps (bulk embedding), never a runtime dependency. The production SIMD
target is **AVX2** (consumer Core Ultra does not expose AVX-512).

## Repository layout

| Path | Purpose |
|------|---------|
| `include/queryforge/` | Public headers (the engine's API) |
| `src/` | Engine implementation (distance, NSW, HNSW, persistence) |
| `tests/` | Unit tests (GoogleTest) |
| `bench/` | Benchmarks (Google Benchmark) |
| `tools/` | Synthetic data generator, recall harness (later stages) |
| `python/` | Pybind11 module + CLIP embedding pipeline (later stages) |
| `backend/` | FastAPI service (later stages) |
| `frontend/` | React demo UI (later stages) |
| `docs/` | Design notes, observations journal, changelog, performance baselines |

## Build & test

Requirements: a C++17 compiler, CMake ≥ 3.20, and internet on the first configure
(dependencies are fetched automatically — no system installs needed).

```bash
cmake -S . -B build            # configure (downloads GoogleTest + Google Benchmark)
cmake --build build -j         # compile
ctest --test-dir build         # run the tests
./build/bin/qf_bench           # run the benchmarks
```

Useful toggles: `-DQF_BUILD_BENCHMARKS=OFF` or `-DQF_BUILD_TESTS=OFF`.

### Run the demo (Python bindings + web app)

```bash
cmake -S . -B build -DQF_BUILD_PYTHON=ON && cmake --build build -j   # builds the queryforge module
pip install -r backend/requirements.txt                              # fastapi, uvicorn, httpx
PYTHONPATH=build/python:python uvicorn backend.app:app               # then open http://127.0.0.1:8000/
```

See [backend/README.md](backend/README.md) and [python/README.md](python/README.md).

## Documentation

- [docs/architecture.md](docs/architecture.md) — how the pieces fit and why.
- [docs/OBSERVATIONS.md](docs/OBSERVATIONS.md) — running journal of decisions and findings.
- [docs/BASELINES.md](docs/BASELINES.md) — recorded performance numbers per stage.
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — what changed, stage by stage.

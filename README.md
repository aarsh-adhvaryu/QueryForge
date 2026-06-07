# QueryForge

A from-scratch **C++ HNSW vector search engine** — built to find the most *similar* vectors
(e.g. image or text embeddings) among millions, in milliseconds, instead of scanning them all —
with a reverse-image-search demo on top.

> **Status:** **bucket A complete.** The full vertical slice works on CPU with synthetic data:
> image → embedding → HNSW search (95.9% Recall@10) → SQLite metadata → FastAPI → React grid, with
> `.qfx` persistence and Python bindings. What remains is real CLIP embeddings + the 500K-image
> dataset + final tuning/benchmarks on the local GPU machine (bucket B).

## What this is (in one paragraph)

A *vector* is a list of numbers a neural network produces to describe something. Similar things get
numerically close vectors. The core question is "given this vector, find the closest ones." Doing
that by comparing against every vector ("brute force") is too slow at scale. **HNSW** (Hierarchical
Navigable Small Worlds) is a graph structure — local streets at the bottom, express highways on top
— that finds close matches by visiting only a tiny fraction of the data. QueryForge implements that
engine, plus the hardware-level distance math (**SIMD/AVX2**) that makes each comparison fast.

## Guiding principle: local-first

The end goal is that everything runs on a single consumer machine (**RTX 5070 Ti** GPU + **Intel
Core Ultra 9** CPU). Cloud GPUs are a convenience for one-off heavy steps (bulk embedding), never a
runtime dependency. The production SIMD target is **AVX2** (consumer Core Ultra does not expose
AVX-512).

## ✅ What's done (bucket A)

Each stage is built, unit-tested, and has recorded numbers in [docs/BASELINES.md](docs/BASELINES.md).

| Stage | What it delivers | Verified result |
|-------|------------------|-----------------|
| **A0** | Project scaffold, CMake build, GoogleTest + Google Benchmark, docs system | clean `cmake && ctest` |
| **A1** | SIMD distance kernels — L2, dot, cosine in scalar / SSE / AVX2 with runtime CPU dispatch | **~8× faster** than scalar (AVX2) |
| **A2** | `NswIndex` — single-layer Navigable Small World graph + greedy beam search + recall harness | works, visits ~2% of nodes |
| **A3** | `HnswIndex` — full multi-layer HNSW + diversity neighbor heuristic | **95.9% Recall@10** (target met) |
| **A4** | Persistence — binary `.qfx` save + `mmap` load | load **~1000–1570× faster** than rebuild |
| **A5** | Pybind11 bindings — `queryforge.HnswIndex` (NumPy in, `(ids, distances)` out) | importable module, round-trips |
| **A6** | Embedding pipeline — pluggable `Embedder` + SQLite metadata + end-to-end CPU dry-run | image→index→metadata→search |
| **A7** | FastAPI backend + React frontend over a mock catalog | full demo, ~7 ms queries |

**Net:** the whole search stack works today on CPU with synthetic data — upload an image, get
visually similar products back, with the live query latency shown.

## 🔜 What's left

**Bucket B (needs the laptop or a GPU — real data & model):**
1. Download the real **500K+ image dataset** (decide: DeepFashion vs Open Images).
2. Swap `HistogramEmbedder` → `ClipEmbedder` (one line) and run **real CLIP embeddings** on the GPU.
3. Pick the embedding model / dimension (CLIP ViT-B = 512-d vs ViT-L = 768-d).
4. **Tune** `M` / `efConstruction` / `efSearch` on real vectors to hit recall/latency targets.
5. Re-measure **final benchmarks** on the local RTX 5070 Ti + Core Ultra 9.

**Performance pass (queued — see [docs/BASELINES.md](docs/BASELINES.md)):**
- Fix the **O(N²) build** (each insert currently allocates a size-N `visited` array; replace with a
  reusable "visited-version" array). This is the top item before scaling to 500K.
- Add **parallel (multi-threaded) index construction** with before/after build-time benchmarks.
- Flatten the layer-0 adjacency into a contiguous array (cache locality), reuse search buffers.

## Repository layout

| Path | Purpose |
|------|---------|
| `include/queryforge/` | Public headers — `distance.hpp`, `nsw.hpp`, `hnsw.hpp`, `types.hpp` |
| `src/` | Engine implementation — distance, NSW, HNSW, persistence |
| `tests/` | Unit tests (GoogleTest) |
| `bench/` | Benchmarks (Google Benchmark) |
| `tools/` | Synthetic data generator, recall harness (`qf_recall`), persistence benchmark (`qf_persist`) |
| `python/` | Pybind11 module (`queryforge`) + `qf_pipeline` (embedder, SQLite metadata, dry-run) |
| `backend/` | FastAPI service — `/health`, `/catalog`, `/search/*` — over a mock catalog |
| `frontend/` | React demo UI — upload, browse, results grid, latency |
| `docs/` | Architecture, observations journal, changelog, performance baselines |

## Build & test

**Requirements:** a C++17 compiler, CMake ≥ 3.20, and internet on the first configure (GoogleTest +
Google Benchmark are fetched automatically — no system installs needed).

### 1. Build the C++ engine and run the test suite

```bash
cmake -S . -B build              # configure (first run downloads test/bench deps)
cmake --build build -j           # compile
ctest --test-dir build --output-on-failure   # run all unit tests (should be all green)
```

Useful CMake toggles: `-DQF_BUILD_BENCHMARKS=OFF`, `-DQF_BUILD_TESTS=OFF`, `-DQF_BUILD_TOOLS=OFF`,
`-DQF_BUILD_PYTHON=ON`.

### 2. Run the benchmarks and tools

```bash
./build/bin/qf_distance_bench                 # scalar vs SSE vs AVX2 distance speed
# Recall vs exact brute force (HNSW or NSW), and the speed/accuracy trade-off:
./build/bin/qf_recall algo=hnsw N=10000 dim=128 M=16 efc=200 ef=200 k=10 metric=l2
# Persistence: build vs mmap-load time + on-disk size:
./build/bin/qf_persist N=30000 dim=128 M=16 efc=200
```

### 3. Build the Python module + web demo (optional)

```bash
cmake -S . -B build -DQF_BUILD_PYTHON=ON && cmake --build build -j   # builds the queryforge module
pip install -r python/requirements.txt                              # numpy, pillow
pip install -r backend/requirements.txt                             # fastapi, uvicorn, httpx
```

Run the end-to-end pipeline dry-run (no GPU, no model download):

```bash
PYTHONPATH=build/python:python python -m qf_pipeline.dry_run
```

Launch the web app and open it in a browser:

```bash
PYTHONPATH=build/python:python uvicorn backend.app:app    # then open http://127.0.0.1:8000/
```

When Python is enabled, `ctest` also runs the `python_bindings`, `python_pipeline`, and `python_api`
cases, so a single `ctest --test-dir build` verifies the C++ engine **and** the full demo.

## Quick usage

**C++:**
```cpp
#include "queryforge/hnsw.hpp"
queryforge::HnswIndex idx(/*dim=*/128, /*M=*/16, /*ef_construction=*/200, queryforge::Metric::L2);
idx.add(vec);                              // float* of length dim
auto hits = idx.search(query, /*k=*/10, /*ef=*/64);   // vector<Neighbor>{distance, id}
idx.save("index.qfx");
auto loaded = queryforge::HnswIndex::load("index.qfx");
```

**Python:**
```python
import numpy as np, queryforge as qf
idx = qf.HnswIndex(dim=64, M=16, ef_construction=200, metric=qf.Metric.Cosine)
idx.add_batch(np.random.randn(5000, 64).astype("float32"))
ids, dists = idx.search(query, k=10, ef=64)
idx.save("index.qfx");  idx2 = qf.HnswIndex.load("index.qfx")
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — how the pieces fit and why.
- [docs/OBSERVATIONS.md](docs/OBSERVATIONS.md) — running journal of decisions and findings.
- [docs/BASELINES.md](docs/BASELINES.md) — recorded performance numbers per stage.
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — what changed, stage by stage.
- [CLAUDE.md](CLAUDE.md) — guide for AI assistants / where to resume.

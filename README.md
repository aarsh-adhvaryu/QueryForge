# QueryForge

A from-scratch **C++ HNSW vector search engine** — finds the most *similar* vectors (image or text
embeddings) among hundreds of thousands, in **well under a millisecond**, instead of scanning them
all — with a working **reverse-image-search demo** over **500,000 real images** on top.

> **Status: engine complete and proven at scale.** The full stack runs end to end on a real catalog:
> image → **CLIP** embedding → **HNSW** search → SQLite metadata → FastAPI → React grid, with `.qfx`
> persistence, Python bindings, and a multi-threaded parallel build. Indexed **500K ImageNet images**
> at **99.6% Recall@10** and **0.35–0.83 ms** queries. The only thing left is the optional
> re-measurement on the owner's local machine (cloud was used purely for bulk embedding).

## What this is (in one paragraph)

A *vector* is a list of numbers a neural network produces to describe something; similar things get
numerically close vectors. The core question is "given this vector, find the closest ones." Doing
that by comparing against every vector ("brute force") is too slow at scale. **HNSW** (Hierarchical
Navigable Small Worlds) is a layered graph — local streets at the bottom, express highways on top —
that finds close matches by visiting only a tiny fraction of the data. QueryForge implements that
engine from scratch, plus the hardware-level distance math (**SIMD / AVX2**) that makes each
comparison fast, and wraps it in a real reverse-image-search application.

## 🏆 Headline results (real 500K ImageNet catalog)

Measured on a Lightning Studio (16-core CPU + RTX PRO 6000); CLIP ViT-L/14 → 768-d, cosine, M=32.

| Metric | Result |
|--------|--------|
| Catalog size | **500,000** real images |
| Recall@10 (vs exact brute force) | **99.6%** |
| Same-class@10 (semantic quality, ImageNet labels) | **75.7%** |
| Parallel index build (16 threads) | **131 s** (~9.3× faster than single-threaded) |
| Index load (mmap of 1.6 GB `.qfx`) | **871 ms** |
| Query latency (HNSW search) @ ef=32 / 200 | **0.24 ms / 0.85 ms** (≈4100 / 1200 qps, single-thread) |

See [docs/BASELINES.md](docs/BASELINES.md) for the full tables (including the ef-sweep and the build
speedup curve) and [docs/OBSERVATIONS.md](docs/OBSERVATIONS.md) for the reasoning behind each result.

## Guiding principle: local-first

The end goal is that everything runs on a single consumer machine (**RTX 5070 Ti** GPU + **Intel
Core Ultra 9** CPU). Cloud GPUs are a convenience for one-off heavy steps (bulk embedding), never a
runtime dependency. The production SIMD target is **AVX2** (consumer Core Ultra does not expose
AVX-512).

## Stages

Each stage is built, unit-tested, and has recorded numbers in [docs/BASELINES.md](docs/BASELINES.md).
**25 tests pass** (`ctest`).

### Bucket A — the engine (complete)

| Stage | What it delivers | Verified result |
|-------|------------------|-----------------|
| **A0** | Project scaffold, CMake build, GoogleTest + Google Benchmark, docs system | clean `cmake && ctest` |
| **A1** | SIMD distance kernels — L2, dot, cosine in scalar / SSE / AVX2 with runtime CPU dispatch | **~8× faster** than scalar (AVX2) |
| **A2** | `NswIndex` — single-layer Navigable Small World graph + greedy beam search + recall harness | works, visits ~2% of nodes |
| **A3** | `HnswIndex` — full multi-layer HNSW + diversity neighbor heuristic | **95.9% Recall@10** (synthetic, target met) |
| **A4** | Persistence — binary `.qfx` save + `mmap` load | load **~1000–1570× faster** than rebuild |
| **A5** | Pybind11 bindings — `queryforge.HnswIndex` (NumPy in, `(ids, distances)` out) | importable module, round-trips |
| **A6** | Embedding pipeline — pluggable `Embedder` + SQLite metadata + end-to-end CPU dry-run | image→index→metadata→search |
| **A7** | FastAPI backend + React frontend over a mock catalog | full demo, ~7 ms queries |

### Bucket B — real data & performance (engine complete)

| Stage | What it delivers | Verified result |
|-------|------------------|-----------------|
| **B1** | Pluggable real-dataset loader (`build_real.py`: imagenet / cc3m / fashion), sharded image dirs, embedding checkpointing, same-class@k metric | streams real data + metadata |
| **B2** | **Parallel multi-threaded build** (`add_batch_parallel`) — pre-sized storage + striped per-node locks | **9.3×** on 16 cores, recall parity |
| **B3** | **Real 500K ImageNet index** — CLIP ViT-L/14, 768-d | **99.6% Recall@10, 75.7% same-class@10** |
| **B4** | efSearch tuning on real vectors — find the cheapest ef that holds recall | **ef=32 ⇒ 99.7% at 3.5× throughput** |
| **B5** | Web demo wired to the **real 500K index** (`QF_INDEX_DIR` selects real vs mock mode) | live reverse-image search |

**🔜 What's left:** the optional re-measurement of all benchmarks on the local RTX 5070 Ti + Core
Ultra 9 (cloud was only for bulk embedding). Optional polish: an `M` sweep (memory/build vs recall).

## How it works (engine internals)

- **HNSW graph** with the diversity neighbor heuristic (hnswlib's "Algorithm 4"): an upper layer
  hierarchy for cheap entry-point descent, a dense wide-beam search only at layer 0.
- **Cache-aware layout:** vectors stored contiguously by id; adjacency in fixed-stride contiguous
  blocks (`links0_` flat array for layer 0 + per-node upper blocks), with neighbor prefetch.
- **SIMD distances:** per-function `__attribute__((target("avx2,fma")))` + runtime dispatch via
  `__builtin_cpu_supports` — one portable binary picks the best path at startup (no `-march=native`).
- **Parallel build:** pre-size all storage (so arrays never reallocate), then wire every node's edges
  across worker threads; **writes** are guarded by striped per-node locks, **reads** during search are
  lock-free (memory-safe because blocks are fixed-size). Recall parity validated by measurement.
- **Persistence:** `.qfx` binary format, `mmap`-loaded on POSIX — restart a service in ~1 s instead
  of re-building. **Lock-free concurrent reads** (static build / serve-many model).

## Repository layout

| Path | Purpose |
|------|---------|
| `include/queryforge/` | Public headers — `distance.hpp`, `nsw.hpp`, `hnsw.hpp`, `types.hpp` |
| `src/` | Engine implementation — distance, NSW, HNSW (incl. parallel build), persistence |
| `tests/` | Unit tests (GoogleTest) |
| `bench/` | Benchmarks (Google Benchmark) |
| `tools/` | Synthetic data generator, recall harness (`qf_recall`), persistence benchmark (`qf_persist`) |
| `python/` | Pybind11 module (`queryforge`) + `qf_pipeline` (embedders, SQLite metadata, dry-run, `build_real`) |
| `backend/` | FastAPI service — `/health`, `/catalog`, `/search/*` — over the mock **or** real catalog |
| `frontend/` | React demo UI — upload, browse, results grid, latency |
| `docs/` | Architecture, observations journal, changelog, performance baselines |

## Build & test

**Requirements:** a C++17 compiler, CMake ≥ 3.20, and internet on the first configure (GoogleTest +
Google Benchmark are fetched automatically — no system installs needed).

### 1. Build the C++ engine and run the test suite

```bash
cmake -S . -B build              # configure (first run downloads test/bench deps)
cmake --build build -j           # compile
ctest --test-dir build --output-on-failure   # run all unit tests (all green)
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

### 3. Build the Python module + web demo

```bash
cmake -S . -B build -DQF_BUILD_PYTHON=ON && cmake --build build -j   # builds the queryforge module
pip install -r python/requirements.txt                              # numpy, pillow
pip install -r backend/requirements.txt                             # fastapi, uvicorn, httpx
```

End-to-end pipeline dry-run (no GPU, no model download — synthetic data):

```bash
PYTHONPATH=build/python:python python -m qf_pipeline.dry_run
```

### 4. Build a real catalog and run the live demo

Build a real index from a streamed dataset (needs `open_clip_torch` + `datasets`; a GPU is strongly
recommended). `--dataset imagenet` requires a Hugging Face token with ImageNet-1k access
(`huggingface-cli login`); `--dataset cc3m` needs no token.

```bash
# start small to validate, then scale up --limit (e.g. to 500000):
PYTHONPATH=build/python:python python -m qf_pipeline.build_real \
    --dataset imagenet --limit 500000 --threads 16 --out /path/to/qf_data/im500k
```

Then point the web app at that index (set `QF_INDEX_DIR`); without it, the app serves the mock catalog:

```bash
# real 500K demo:
QF_INDEX_DIR=/path/to/qf_data/im500k \
    PYTHONPATH=build/python:python uvicorn backend.app:app --host 0.0.0.0 --port 8000
# mock demo (no dataset/GPU needed):
PYTHONPATH=build/python:python uvicorn backend.app:app
```

Open `http://127.0.0.1:8000/` — browse the grid, click an image for similar results, or upload your
own for reverse-image search. `GET /health` reports the active `mode` (`real`/`mock`) and `embedder`.

When Python is enabled, `ctest` also runs `python_bindings`, `python_pipeline`, and `python_api`, so a
single `ctest --test-dir build` verifies the C++ engine **and** the full demo.

## Quick usage

**C++:**
```cpp
#include "queryforge/hnsw.hpp"
queryforge::HnswIndex idx(/*dim=*/128, /*M=*/16, /*ef_construction=*/200, queryforge::Metric::L2);
idx.add(vec);                                          // float* of length dim
auto hits = idx.search(query, /*k=*/10, /*ef=*/32);    // vector<Neighbor>{distance, id}
idx.save("index.qfx");
auto loaded = queryforge::HnswIndex::load("index.qfx");
```

**Python:**
```python
import numpy as np, queryforge as qf
idx = qf.HnswIndex(dim=64, M=32, ef_construction=200, metric=qf.Metric.Cosine)
idx.add_batch_parallel(np.random.randn(100_000, 64).astype("float32"), threads=16)  # fast bulk build
ids, dists = idx.search(query, k=10, ef=32)            # ef=32: the tuned operating point
idx.save("index.qfx");  idx2 = qf.HnswIndex.load("index.qfx")
```

## Documentation

- [docs/HANDOFF.md](docs/HANDOFF.md) — finishing the local-machine re-measurement from a fresh clone.
- [docs/architecture.md](docs/architecture.md) — how the pieces fit and why.
- [docs/OBSERVATIONS.md](docs/OBSERVATIONS.md) — running journal of decisions and findings.
- [docs/BASELINES.md](docs/BASELINES.md) — recorded performance numbers per stage.
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — what changed, stage by stage.
- [CLAUDE.md](CLAUDE.md) — guide for AI assistants / where to resume.

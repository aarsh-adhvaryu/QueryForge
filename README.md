# QueryForge

**A from-scratch C++ HNSW vector search engine — reverse-image search over 500,000 real images in under a millisecond.**

![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Build](https://img.shields.io/badge/build-CMake%20%E2%89%A5%203.20-064F8C?logo=cmake&logoColor=white)
![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)
![SIMD](https://img.shields.io/badge/SIMD-AVX2-orange)

QueryForge implements the **HNSW** (Hierarchical Navigable Small Worlds) approximate-nearest-neighbor
algorithm and the low-level **SIMD distance math** from scratch — no vector-search libraries — then
wraps it in a real **reverse-image-search** application: drop in an image, get the most visually
similar ones back, ranked, in milliseconds, from a catalog of half a million photos.

> A *vector* is the list of numbers a neural network produces to describe something; similar things
> get numerically close vectors. Finding the closest ones by comparing against every vector ("brute
> force") is too slow at scale. HNSW is a layered graph — local streets at the bottom, express
> highways on top — that finds close matches while visiting only a tiny fraction of the data.

---

## 🏆 Results (real 500K ImageNet catalog)

Measured on a 16-core CPU + RTX PRO 6000; CLIP ViT-L/14 → 768-d embeddings, cosine, M=32.

| Metric | Result |
|--------|--------|
| Catalog size | **500,000** real images |
| **Recall@10** (vs exact brute force) | **99.6%** |
| Same-class@10 (semantic quality, ImageNet labels) | **75.7%** |
| **Query latency** (HNSW search) @ ef=32 / 200 | **0.24 ms / 0.85 ms** (≈4,100 / 1,200 qps, single thread) |
| **Parallel index build** (16 threads) | **131 s** — ~**9.3×** faster than single-threaded |
| Index load (mmap of 1.6 GB `.qfx`) | **871 ms** |
| SIMD distance kernels (AVX2 vs scalar) | **~8×** faster |

Full tables (build-speedup curve, ef sweep, persistence) in [docs/BASELINES.md](docs/BASELINES.md);
the reasoning and surprises behind each number in [docs/OBSERVATIONS.md](docs/OBSERVATIONS.md).

## 🔎 Demo

Upload an image (or click a catalog item) → CLIP embeds it → HNSW finds the nearest neighbors →
results render in a grid with live latency. A query for a flamingo returns flamingos; a poncho query
even surfaces a *stole* (a different class that genuinely looks alike) — semantic matching the old
color-histogram baseline could never do.

<!-- TODO: add a screenshot or GIF here, e.g. ![demo](docs/demo.gif) -->

```bash
# real 500K demo (needs the prebuilt artifacts — see "Artifacts" below):
QF_INDEX_DIR=/path/to/im500k PYTHONPATH=build/python:python uvicorn backend.app:app
# mock demo (no dataset/GPU, synthetic catalog):
PYTHONPATH=build/python:python uvicorn backend.app:app
# → open http://127.0.0.1:8000/
```

## 🧠 How it works

```
                  CLIP ViT-L/14                       add_batch_parallel (9.3× / 16 threads)
   image  ───────────────────────▶  768-d vector  ──────────────────────────────▶  HnswIndex  (C++)
                                          │                                              │
                                          │  search(k, ef=32)                            │  save / mmap-load
                                          ▼                                              ▼
   React UI ◀── FastAPI ◀── ids + distances ◀── HNSW graph  (<1 ms over 500K)      index.qfx  (1.6 GB, 871 ms load)
                  │
                  └── SQLite metadata  (id → name / class / image)
```

- **HNSW graph** with the diversity neighbor heuristic (hnswlib "Algorithm 4"): a sparse upper-layer
  hierarchy for cheap entry-point descent, a dense wide-beam search only at layer 0 → roughly
  logarithmic search instead of linear.
- **SIMD distances** (`L2` / `dot` / `cosine`) in scalar / SSE / **AVX2**, selected at runtime via
  `__builtin_cpu_supports` — one portable binary, no `-march=native`.
- **Cache-aware layout**: vectors contiguous by id; adjacency in fixed-stride contiguous blocks
  (`links0_` flat array for the hot layer 0 + per-node upper blocks) with neighbor prefetch.
- **Parallel build**: pre-size all storage so arrays never reallocate, then wire every node's edges
  across threads — **writes** guarded by striped per-node locks, **reads lock-free** (memory-safe
  because blocks are fixed-size). Recall parity vs the sequential build is asserted by a test.
- **Persistence**: binary `.qfx`, `mmap`-loaded on POSIX — a service restarts in ~1 s instead of
  re-building. Static-build / **lock-free concurrent reads** serving model.

## ✅ Stages (all built, tested, benchmarked — 25 tests pass)

**Bucket A — the engine**

| | Delivers | Result |
|--|----------|--------|
| A0 | CMake scaffold, GoogleTest + Google Benchmark, docs system | clean `cmake && ctest` |
| A1 | SIMD distance kernels (scalar / SSE / AVX2, runtime dispatch) | **~8×** vs scalar |
| A2 | `NswIndex` — single-layer NSW graph + greedy beam search | visits ~2% of nodes |
| A3 | `HnswIndex` — full multi-layer HNSW + diversity heuristic | **95.9% Recall@10** (synthetic) |
| A4 | Persistence — binary `.qfx` save + `mmap` load | **~1000–1570×** faster than rebuild |
| A5 | Pybind11 bindings — NumPy in, `(ids, distances)` out | importable `queryforge` module |
| A6 | Embedding pipeline — pluggable `Embedder` + SQLite metadata | image → index → metadata → search |
| A7 | FastAPI backend + React frontend (mock catalog) | full vertical slice |

**Bucket B — real data & performance**

| | Delivers | Result |
|--|----------|--------|
| B1 | Pluggable real-dataset loader (imagenet / cc3m / fashion), sharded images, checkpointing | streams real data + metadata |
| B2 | **Parallel multi-threaded build** (`add_batch_parallel`) | **9.3×** on 16 cores, recall parity |
| B3 | **Real 500K ImageNet index** (CLIP ViT-L/14) | **99.6% Recall@10, 75.7% same-class@10** |
| B4 | efSearch tuning on real vectors | **ef=32 ⇒ 99.7% at 3.5× throughput** |
| B5 | Web demo wired to the real 500K index (`QF_INDEX_DIR`) | live reverse-image search |

## ⚙️ Build & test

**Requirements:** a C++17 compiler, CMake ≥ 3.20, internet on first configure (GoogleTest + Google
Benchmark are fetched automatically). Production SIMD target is **AVX2** (no AVX-512).

```bash
cmake -S . -B build              # configure (first run fetches test/bench deps)
cmake --build build -j           # compile
ctest --test-dir build --output-on-failure   # all 25 tests green
```

Benchmarks & tools:

```bash
./build/bin/qf_distance_bench                 # scalar vs SSE vs AVX2 distance speed
./build/bin/qf_recall algo=hnsw N=10000 dim=128 M=16 efc=200 ef=200 k=10 metric=l2   # recall vs brute force
./build/bin/qf_persist N=30000 dim=128 M=16 efc=200                                  # build vs mmap-load
```

Python module + web demo:

```bash
cmake -S . -B build -DQF_BUILD_PYTHON=ON && cmake --build build -j
pip install -r python/requirements.txt -r backend/requirements.txt
PYTHONPATH=build/python:python python -m qf_pipeline.dry_run     # end-to-end, no GPU / no download
```

Build a real catalog (needs `open_clip_torch` + `datasets`; a GPU is strongly recommended;
`--dataset imagenet` needs a Hugging Face token):

```bash
PYTHONPATH=build/python:python python -m qf_pipeline.build_real \
    --dataset imagenet --limit 500000 --threads 16 --out qf_data/im500k
```

With `-DQF_BUILD_PYTHON=ON`, `ctest` also runs `python_bindings`, `python_pipeline`, and `python_api`
— a single `ctest` verifies the C++ engine **and** the full demo.

## 🚀 Quick usage

**C++**
```cpp
#include "queryforge/hnsw.hpp"
queryforge::HnswIndex idx(/*dim=*/128, /*M=*/16, /*ef_construction=*/200, queryforge::Metric::L2);
idx.add(vec);                                          // float* of length dim
auto hits = idx.search(query, /*k=*/10, /*ef=*/32);    // vector<Neighbor>{distance, id}
idx.save("index.qfx");
auto loaded = queryforge::HnswIndex::load("index.qfx");
```

**Python**
```python
import numpy as np, queryforge as qf
idx = qf.HnswIndex(dim=64, M=32, ef_construction=200, metric=qf.Metric.Cosine)
idx.add_batch_parallel(np.random.randn(100_000, 64).astype("float32"), threads=16)  # fast bulk build
ids, dists = idx.search(query, k=10, ef=32)            # ef=32: the tuned operating point
idx.save("index.qfx");  idx2 = qf.HnswIndex.load("index.qfx")
```

## 📦 Artifacts

The prebuilt 500K artifacts are too large for git (~11 GB) and are hosted on Hugging Face:
**`aarsh-adhvaryu/queryforge-imagenet-500k`** (private) — `embeddings.npy`, `index.qfx`,
`metadata.db`, and the images as `images.tar`. Pull and rebuild/benchmark without re-embedding:

```python
from huggingface_hub import hf_hub_download
emb = hf_hub_download("aarsh-adhvaryu/queryforge-imagenet-500k", "embeddings.npy", repo_type="dataset")
# then:  python -m qf_pipeline.rebuild_index --embeddings <emb>
```

## 🗂️ Repository layout

| Path | Purpose |
|------|---------|
| `include/queryforge/` | Public headers — `distance.hpp`, `nsw.hpp`, `hnsw.hpp`, `types.hpp` |
| `src/` | Engine — distance, NSW, HNSW (incl. parallel build), persistence |
| `tests/` · `bench/` · `tools/` | GoogleTest · Google Benchmark · `qf_recall` / `qf_persist` harnesses |
| `python/` | Pybind11 module + `qf_pipeline` (embedders, SQLite, dry-run, `build_real`, `rebuild_index`) |
| `backend/` · `frontend/` | FastAPI service · React demo UI |
| `docs/` | Architecture, observations journal, changelog, baselines, handoff |

## 📚 Documentation

- [docs/BASELINES.md](docs/BASELINES.md) — all recorded performance numbers, per stage.
- [docs/OBSERVATIONS.md](docs/OBSERVATIONS.md) — running journal of decisions, surprises, dead-ends.
- [docs/architecture.md](docs/architecture.md) — how the pieces fit and why.
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — what changed, stage by stage.
- [docs/HANDOFF.md](docs/HANDOFF.md) — reproduce/benchmark from a fresh clone.

---

*Built as a learning + portfolio project. A `LICENSE` file is not yet included — add one before
reusing.*

# python/

The Python side of QueryForge.

## Bindings (stage A5 — done)

`bindings.cpp` exposes the C++ `HnswIndex` as `queryforge.HnswIndex` via Pybind11. Vectors cross the
boundary as NumPy `float32` arrays; `search()` returns `(ids, distances)` as NumPy arrays.

Build the module (off by default — it needs Python dev headers + pybind11, fetched automatically):

```bash
cmake -S . -B build -DQF_BUILD_PYTHON=ON
cmake --build build -j
PYTHONPATH=build/python python python/example.py   # runs the demo / smoke test
```

`ctest` runs `example.py` as the `python_bindings` case when the module is built.

API sketch:

```python
import numpy as np, queryforge as qf
idx = qf.HnswIndex(dim=64, M=16, ef_construction=200, metric=qf.Metric.L2)
idx.add_batch(np.random.randn(5000, 64).astype("float32"))
ids, dists = idx.search(query, k=10, ef=64)
idx.save("index.qfx");  idx2 = qf.HnswIndex.load("index.qfx")
```

## Embedding pipeline (stage A6 — next)

Turn images into vectors with CLIP, normalize, validate, and feed them into the index. Built to run
on the local RTX 5070 Ti (VRAM-aware batching); cloud/Studio GPU is only a convenience for the
one-off bulk run.

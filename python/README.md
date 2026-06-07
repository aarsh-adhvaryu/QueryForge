# python/

The Python side of QueryForge (later stages):

- **Pybind11 bindings** (stage A5) — expose the C++ engine to Python: create index, insert
  vectors, KNN search, save/load, stats. Lets the ML pipeline drive the engine with familiar
  Python while the heavy math stays in optimized C++.
- **Embedding pipeline** (stage A6) — turn images into vectors with CLIP, normalize, validate,
  and feed them into the index. Built to run on the local RTX 5070 Ti (VRAM-aware batching);
  cloud/Studio GPU is only a convenience for the one-off bulk run.

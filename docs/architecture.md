# QueryForge — Architecture & how it works

This document explains the *why* behind the design, in plain language. It grows as stages land.

## The problem

Modern AI represents data (images, text) as **embeddings** — vectors of hundreds of floats
produced by neural networks. Similar items have vectors that are close together. The core
operation is **approximate nearest neighbor (ANN) search**: "given a query vector, return the
K most similar stored vectors, fast."

Comparing the query against *every* stored vector ("brute force") is accurate but slow — it
scales linearly with the dataset. Traditional database indexes (B-Trees, hash indexes) can't
help: they answer exact-match and range queries, not "closest in high-dimensional space."

## The approach: HNSW

**HNSW** (Hierarchical Navigable Small Worlds) is a layered graph:

- Each **node** is a vector; **edges** connect vectors that are near each other.
- The bottom layer (layer 0) holds every vector with fine-grained local connections —
  think *local streets*.
- Upper layers hold exponentially fewer nodes with long-range links — think *highways*.
- A search starts at the top, greedily hops toward the query through the highways to land in
  the right neighborhood, then descends to layer 0 for the fine search. Only a tiny fraction
  of nodes is ever visited → roughly *logarithmic* search cost instead of linear.

We build this bottom-up across stages:

1. **A1 — Distance math.** Every step of a search computes distances (L2 / cosine). This is
   the hot loop, so we optimize it with **SIMD** (AVX2): one CPU instruction processes 8
   floats at once instead of one.
2. **A2 — NSW (one layer).** Get a single navigable graph right first: greedy search + insert
   with a neighbor-selection heuristic.
3. **A3 — HNSW (many layers).** Stack NSW into layers with the exponential layer-assignment
   rule and neighbor pruning; validate recall against brute force.
4. **A4 — Persistence.** Serialize the graph to disk and load it back with **mmap** (the OS
   maps the file straight into memory — near-instant load, no parsing).
5. **A5 — Python bindings.** Expose the C++ engine to Python via **Pybind11**.
6. **A6–A7 — Demo.** CLIP image→vector pipeline, FastAPI backend, React frontend.

## Key parameters (HNSW)

| Parameter | Controls | Trade-off |
|-----------|----------|-----------|
| `M` | Max edges per node per layer | Higher → better recall, more memory |
| `efConstruction` | Candidate pool size while building | Higher → better graph, slower build |
| `efSearch` | Candidate pool size per query | Higher → better recall, slower queries |
| `mL` | Layer-assignment multiplier (1/ln M) | Controls how often nodes go to higher layers |

## Design decisions of record

- **Static build + concurrent reads now; dynamic insertion later.** Build the index once, then
  serve many searches in parallel with no locks. (Honest claim: lock-free concurrent *reads*.)
- **Single-threaded build now; parallel build as a measured enhancement next.** Correctness and
  recall first, then parallelize a known-correct algorithm and report before/after build times.
- **AVX2 is the production SIMD target** (consumer Core Ultra 9 has no AVX-512).
- **Contiguous neighbor storage** for cache locality — the realistic, defensible "custom memory
  layout" scope, rather than a full custom allocator.

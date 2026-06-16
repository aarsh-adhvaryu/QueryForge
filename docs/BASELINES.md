# Performance baselines

Measured numbers, recorded as each stage lands, so the project's story is defensible and the
laptop-side re-measurement is a simple comparison. Each table notes the machine it was measured
on, because numbers only mean something relative to hardware.

**Machines:**
- `lightning` — Lightning AI Studio: Intel (AVX2/FMA/SSE4.2), 4 cores, 15 GB RAM, g++ 13.3.
- `studio-gpu` — Lightning Studio (bucket B): 16 cores + RTX PRO 6000 Blackwell (96 GB), used for
  bulk CLIP embedding and the parallel build. Cloud convenience only, not a runtime dependency.
- `local` — owner's machine: Intel Core Ultra 9 (AVX2, no AVX-512) + RTX 5070 Ti (to be filled in).

---

## A1 — Distance throughput (scalar vs SSE vs AVX2)

Measured on `lightning` (g++ 13.3, -O3 Release, AVX2/FMA), `qf_distance_bench --benchmark_min_time=0.3s`.
Time is per full vector-vs-vector comparison (lower is better).

| Kernel | Dim | Variant | ns/op | speedup vs scalar |
|--------|-----|---------|-------|-------------------|
| L2     | 512 | scalar  | 453   | 1.0x  |
| L2     | 512 | SSE     | 113   | 4.0x  |
| L2     | 512 | AVX2    | 56.8  | 8.0x  |
| L2     | 768 | scalar  | 687   | 1.0x  |
| L2     | 768 | SSE     | 171   | 4.0x  |
| L2     | 768 | AVX2    | 87.2  | 7.9x  |
| dot    | 512 | scalar  | 460   | 1.0x  |
| dot    | 512 | SSE     | 122   | 3.8x  |
| dot    | 512 | AVX2    | 52.1  | 8.8x  |
| dot    | 768 | scalar  | 684   | 1.0x  |
| dot    | 768 | SSE     | 172   | 4.0x  |
| dot    | 768 | AVX2    | 83.7  | 8.2x  |

**Takeaway:** AVX2 gives ~8x (≈88% fewer cycles), beating the proposal's ~60% target. The scalar
baseline was *not* auto-vectorized (GCC won't reorder a float reduction without -ffast-math), so this
is a genuine scalar-vs-SIMD comparison, not SIMD-vs-SIMD.

_Re-measure on `local` (Core Ultra 9) when available — expect AVX2 only, no AVX-512._

## A2 — NSW (single layer, naive neighbor selection)

Measured on `lightning` via `qf_recall`, N=10000, dim=128, L2, 300 queries, k=10.
Ground truth = exact brute force. This is the BASELINE that A3 (layers + diversity heuristic)
must beat.

| N | Dim | M | efc | efSearch | Recall@10 | avg nodes visited | query | build |
|---|-----|---|-----|----------|-----------|-------------------|-------|-------|
| 10000 | 128 | 16 | 200 | 10  | 19.6% | 16 (0.16%)  | 17 µs  | 1.36 s |
| 10000 | 128 | 16 | 200 | 50  | 50.4% | 57 (0.57%)  | 59 µs  | 1.35 s |
| 10000 | 128 | 16 | 200 | 200 | 80.8% | 203 (2.0%)  | 163 µs | 1.46 s |
| 10000 | 128 | 32 | 200 | 200 | **94.5%** | 202 (2.0%) | 243 µs | 1.98 s |

**Lessons:** (1) the graph already visits only ~2% of nodes — the core HNSW promise holds even
at one layer. (2) `efSearch` trades recall for latency; `M` (graph density) is the stronger recall
lever (16→32 took recall 80.8%→94.5%). (3) Naive "closest-M" selection caps recall; the A3
diversity heuristic should let us reach the same recall with smaller M/ef (cheaper queries).

## A3 — HNSW (multi-layer)

Same dataset/params as the A2 table (N=10000, dim=128, L2, M=16, efc=200, 300 queries, k=10),
so rows compare directly against NSW above. Built a 4-level hierarchy (top layer = 3).

### Step 1 — layers + naive neighbor selection

| efSearch | NSW Recall@10 | HNSW Recall@10 | HNSW nodes visited | HNSW query |
|----------|---------------|----------------|--------------------|------------|
| 50  | 50.4% | **71.4%** | 52 (0.52%)  | 122 µs |
| 100 | 67.5% | **85.9%** | 101 (1.0%)  | 147 µs |
| 200 | 80.8% | **94.5%** | 200 (2.0%)  | 252 µs |

**What layers bought:** +14 to +21 recall points at the same nodes-visited budget. HNSW@M=16 reaches
the recall that NSW needed M=32 for — layers give better entry points so the same beam finds more.

### Step 2 — layers + diversity heuristic (Algorithm 4)

| efSearch | Recall@10 (naive) | Recall@10 (heuristic) | query (naive) | query (heuristic) |
|----------|-------------------|------------------------|---------------|--------------------|
| 50  | 71.4% | 70.5% | 122 µs | **80 µs** |
| 100 | 85.9% | 87.0% | 147 µs | 155 µs |
| 200 | 94.5% | **95.9%** | 252 µs | 245 µs |

**What the heuristic bought:** crossed the **95% recall target** with only M=16, and made low-ef queries
~35% faster. The heuristic rejects redundant same-direction edges, so nodes keep fewer but
better-aimed neighbors — fewer distance computations per hop. Net: a leaner, faster graph that
reaches higher peak recall. (At very low ef it trades a hair of recall for the leaner graph.)

**A3 bottom line vs A2:** NSW@ef=200 = 80.8% recall → full HNSW@ef=200 = 95.9% at the same M=16,
visiting ~2% of nodes. Target (>95% Recall@10) met on synthetic data; to be re-validated on real
CLIP embeddings and at larger N on `local`.

## A4 — Persistence

Measured on `lightning` via `qf_persist` (HNSW, dim=128, M=16, efc=200, L2). The headline is the
build-once / load-many win: loading is ~1000× faster than rebuilding.

| N | file size | bytes/vector | build time | save | **mmap load** | load speedup |
|---|-----------|--------------|------------|------|---------------|--------------|
| 30000  | 18.3 MiB | 640 | 18.3 s | 56 ms | **16 ms** | **1134×** |
| 100000 | 61.1 MiB | 640 | 84.3 s | 89 ms | **54 ms** | **1570×** |

- 640 bytes/vector = 512 (128 floats) + ~128 for the per-layer adjacency. Round-trips are bit-exact:
  the loaded index returns identical search results (verified in tests and the tool).
- **Build time is superlinear** (30k→18s, 100k→84s). NOTE: originally attributed to the per-insert
  `visited` allocation ("O(N²)") — **that was wrong** (see the P1 section below and OBSERVATIONS).
  Load time is unaffected and already excellent.

## P1 — build-complexity investigation (correction)

Replaced the per-insert `std::vector<bool> visited(N)` with a reusable `thread_local VisitedSet`
(`src/visited_set.hpp`, O(1) version-tag clear). Effect on build time was minor — the visited
allocation was **not** the bottleneck.

| N | build (before, vector<bool>) | build (after, VisitedSet) |
|---|------------------------------|----------------------------|
| 30000  | 18.0 s | 15.9 s |
| 100000 | 84.0 s | 79.0 s |

Measured scaling (dim=128, M=16, efc=200): 20k=9.3s, 40k=24.1s, 80k=59.3s → **wall-clock ≈ O(N^1.4)**,
not O(N²). Dominant cost = per-insert distance computations in the diversity heuristic
(≈ O(efConstruction·M) per insert) + cache misses once `vectors_` exceeds L3 (~64k vectors @ dim128).
Real levers: **parallel build** (÷cores), **efConstruction** (efc 200→100 ≈ halves build, ~no recall
loss: 80k recall 69.95%→67.45%), and flattening layer-0 adjacency for locality. `VisitedSet` is kept
(correct; removes a real O(N)-per-insert term that matters at ≥1M; per-thread scratch enables safe
parallel build) — just not the headline.

### P1.5 — Flatten adjacency + prefetch (locality)

Replaced the nested-`std::vector` HNSW adjacency with contiguous fixed-stride blocks + neighbor
prefetch (dim=128, M=16, efc=200):

| N | build (nested) | build (flat + prefetch) | gain |
|---|----------------|--------------------------|------|
| 20000 | 9.3 s  | 8.3 s  | ~11% |
| 40000 | 24.1 s | 20.5 s | ~15% |
| 80000 | 59.3 s | 52.3 s | ~12% |

Recall unchanged (80k @ ef=200 = 70.1%). The constant improved but the *slope* didn't — confirming
the remaining bottleneck is the random 512-B vector reads exceeding L3 (the memory wall), not the
adjacency layout. Next real levers: parallel build, efConstruction.

### P2 — Parallel build (`add_batch_parallel`)  [studio-gpu, 16 cores]

Multi-threaded static build: pre-size all storage (phase A), then wire every node's edges across
worker threads under striped per-node locks (phase B). Measured on N=40000, dim=384, M=32, efc=200:

| build | time | speedup |
|-------|------|---------|
| sequential (`add_batch`) | 184.6 s | 1.0× |
| parallel, 4 threads  | 46.8 s | 3.9× |
| parallel, 8 threads  | 24.2 s | 7.6× |
| parallel, 16 threads | 19.9 s | 9.3× |

Near-linear to 8 threads, then diminishing (9.3× on 16 cores) — the build is memory-bandwidth bound
(the same random-vector-read wall from P1/P1.5), so it can't scale perfectly with core count. Recall
parity with the sequential build is asserted by `Hnsw.ParallelBuildMatchesSequentialRecall`
(within 0.05 of sequential; not bit-identical because edge formation depends on thread interleaving).

## B3 — Real 500K ImageNet index (the headline)  [studio-gpu]

Dataset: 500,000 images streamed from ImageNet-1k, embedded with **CLIP ViT-L/14 (768-d, cosine)**.
Index params: M=32, efConstruction=200. End-to-end via `python/qf_pipeline/build_real.py`.

| stage | time | rate / note |
|-------|------|-------------|
| stream + save 500K images | 4543.7 s (~76 min) | network I/O bound; survived corrupt/truncated files + 1 disconnect-retry |
| CLIP embed (GPU) | 2309.8 s (~38 min) | ~4.6 ms/img, batched 256 |
| **parallel build (16 threads)** | **130.6 s (~2.2 min)** | top layer 4; ≈20 min if single-threaded (per P2's 9.3×) |

**Quality** (real, clustered data — the easy case for ANN, unlike uniform-random synthetic):

| metric | value |
|--------|-------|
| Recall@10 vs brute force (ef=200) | **99.6%** |
| Same-class@10 (ImageNet labels, ef=200) | **75.7%** (was 55.8% at 20k — denser classes ⇒ more same-class neighbors exist) |

**Serving** (build-once / serve-many payoff):

| metric | value |
|--------|-------|
| mmap load of 1.6 GB `.qfx` | **871 ms** (vs ~40 min to re-embed + rebuild) |
| query latency @ ef=50 / 100 / 200 | **0.35 / 0.49 / 0.83 ms** (≈2870 / 2030 / 1210 qps, single-thread) |

Artifacts: `index.qfx` 1.6 GB (~3.2 KB/vector = 768·4 B vector + edges), `embeddings.npy` 1.5 GB,
`metadata.db` 69 MB. Re-measure on `local` (Core Ultra 9 / RTX 5070 Ti) when available.

## B4 — efSearch tuning on the real 500K vectors  [studio-gpu]

Fixed index (M=32). ef swept; Recall@10 vs exact brute force over 500 sampled queries, single-thread.

| ef | recall@10 | avg latency | qps | vs ef=200 |
|----|-----------|-------------|-----|-----------|
| 16 | 98.7% | 0.165 ms | 6053 | 5.2× |
| 20 | 99.3% | 0.185 ms | 5408 | 4.6× |
| 24 | 99.5% | 0.205 ms | 4880 | 4.2× |
| **32** | **99.7%** | **0.244 ms** | **4097** | **3.5×** |
| 64 | 100.0% | 0.382 ms | 2615 | 2.2× |
| 200 | 100.0% | 0.854 ms | 1171 | 1× |

**~99% recall holds down to ef≈20** (4.6× faster than ef=200). Adopted operating point: **ef=32**
(99.7% recall, 3.5× throughput, −0.3 pt recall). ef=200 was synthetic-worst-case overkill — real CLIP
embeddings cluster, so the true neighbors are found after exploring far fewer candidates. Note: in the
live web demo end-to-end latency is dominated by CLIP query embedding (~15 ms GPU); ef tuning matters
most for the engine's standalone/offline throughput (the qps column). Backend defaults now use ef=32.

# Performance baselines

Measured numbers, recorded as each stage lands, so the project's story is defensible and the
laptop-side re-measurement is a simple comparison. Each table notes the machine it was measured
on, because numbers only mean something relative to hardware.

**Machines:**
- `lightning` — Lightning AI Studio: Intel (AVX2/FMA/SSE4.2), 4 cores, 15 GB RAM, g++ 13.3.
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

_To be measured in stage A4._

| N vectors | Dim | Index size on disk | mmap load time | Machine |
|-----------|-----|--------------------|----------------|---------|
| TBD | TBD | TBD | TBD | lightning |

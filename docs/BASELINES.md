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

## A2 / A3 — Recall & search

_To be measured in stages A2–A3._

| Stage | N vectors | Dim | M | efSearch | Recall@10 | avg nodes visited | query latency | build time | Machine |
|-------|-----------|-----|---|----------|-----------|-------------------|---------------|------------|---------|
| A2 NSW  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | lightning |
| A3 HNSW | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | lightning |

## A4 — Persistence

_To be measured in stage A4._

| N vectors | Dim | Index size on disk | mmap load time | Machine |
|-----------|-----|--------------------|----------------|---------|
| TBD | TBD | TBD | TBD | lightning |

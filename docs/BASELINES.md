# Performance baselines

Measured numbers, recorded as each stage lands, so the project's story is defensible and the
laptop-side re-measurement is a simple comparison. Each table notes the machine it was measured
on, because numbers only mean something relative to hardware.

**Machines:**
- `lightning` — Lightning AI Studio: Intel (AVX2/FMA/SSE4.2), 4 cores, 15 GB RAM, g++ 13.3.
- `local` — owner's machine: Intel Core Ultra 9 (AVX2, no AVX-512) + RTX 5070 Ti (to be filled in).

---

## A1 — Distance throughput (scalar vs SSE vs AVX2)

_To be measured in stage A1._

| Metric | Dim | Variant | ns/op | speedup vs scalar | Machine |
|--------|-----|---------|-------|-------------------|---------|
| L2     | 512 | scalar  | TBD   | 1.0x              | lightning |
| L2     | 512 | SSE     | TBD   | TBD               | lightning |
| L2     | 512 | AVX2    | TBD   | TBD               | lightning |
| cosine | 512 | scalar  | TBD   | 1.0x              | lightning |
| cosine | 512 | AVX2    | TBD   | TBD               | lightning |

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

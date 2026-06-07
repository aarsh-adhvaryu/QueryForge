# Observations journal

A running log of what we tried, what we learned, surprises, dead-ends, and the reasoning
behind decisions. Newest entries on top.

---

## A1 — SIMD distance math

- **Result:** AVX2 distance kernels run ~8x faster than scalar (L2 @512-d: 453ns → 56.8ns;
  dot @512-d: 460ns → 52.1ns). SSE sits in between at ~4x. Full numbers in `BASELINES.md`.
- **Surprise that's actually expected:** the scalar baseline did *not* get auto-vectorized at -O3.
  Reason: summing floats is not associative, so the compiler may not reorder a reduction loop
  unless you pass `-ffast-math` (which we deliberately don't, to keep results predictable). Our
  SIMD code reorders the summation on purpose — that reordering is also why the SIMD result differs
  from scalar in the last bit or two, so the tests compare with a dimension-scaled tolerance rather
  than exact equality. Good lesson: floating-point "equality" across implementations is the wrong
  expectation; bounded agreement is the right one.
- **Portability approach:** per-function `__attribute__((target("avx2,fma")))` + runtime dispatch via
  `__builtin_cpu_supports`. One binary picks the best path at startup and falls back to scalar on
  CPUs without AVX2 — no `-march=native`, which would make the binary crash on lesser CPUs.
- **L2 is kept squared** (no sqrt): ordering is preserved and it's faster. Cosine distance is built
  from `dot` + norms; for pre-normalized vectors `1 - dot` is the cheaper path we'll use in the index.
- **Why it matters for the engine:** this kernel is called hundreds of thousands of times per query,
  so this ~8x is a floor on the whole engine's speedup, before the HNSW graph even reduces *how many*
  comparisons we do.

---

## A0 — Repo scaffold

- **Environment (Lightning Studio):** g++ 13.3, CMake 3.28, 4 cores, 15 GB RAM. CPU exposes
  **AVX2 + FMA + SSE4.2** — confirmed via `/proc/cpuinfo`. This means the SIMD distance work
  can be built *and* tested here, not just on the local machine later.
- **GPU:** available in the Studio but toggled off; can be enabled when we reach bulk embedding.
- **Dependency strategy:** GoogleTest + Google Benchmark fetched via CMake `FetchContent` rather
  than system installs — keeps the build reproducible across Lightning now and the Core Ultra 9
  machine later. Cost: first `cmake` configure needs internet.
- **Why AVX2 (not AVX-512):** the target CPU is a consumer Core Ultra 9, which does not expose
  AVX-512. So AVX2 is the real production target — and conveniently it's also what this Studio
  has. Confirm with `lscpu` on the local machine before assuming otherwise.
- **Open question still pending:** dataset choice (Stanford Online Products vs DeepFashion vs
  Open Images) and embedding model / dimension (512 vs 768) — both deferred until the embedding
  stage, since the engine is dimension-agnostic.

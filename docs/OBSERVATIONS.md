# Observations journal

A running log of what we tried, what we learned, surprises, dead-ends, and the reasoning
behind decisions. Newest entries on top.

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

# Observations journal

A running log of what we tried, what we learned, surprises, dead-ends, and the reasoning
behind decisions. Newest entries on top.

---

## B5 — web demo on the real 500K index

- **Wired the FastAPI backend to the real index** behind a `QF_INDEX_DIR` env var: set ⇒ REAL mode
  (mmap-load `index.qfx` + open `metadata.db` + `ClipEmbedder`); unset ⇒ the A7 MOCK catalog,
  unchanged. One backend, two modes, identical endpoints — the `python_api` smoke test stays green.
- **Three real-data adaptations:** (1) images are sharded into subdirs, so the served URL is the path
  *relative* to the images root, not `basename` (which would collide/lose the subdir); (2) a GPU
  **warmup** call at startup — the first CLIP forward pass does cuDNN autotuning (~430 ms), so we burn
  it once at boot and real queries land at ~15 ms instead of a slow first hit; (3) `/health` reports
  `mode` + `embedder` so you can tell which catalog is loaded.
- **Verified end-to-end in the browser:** grid of real photos, click-to-find-similar, upload-to-search
  all return semantically correct neighbors (plane→planes, flamingo→flamingos, a poncho query even
  surfaces a *stole* — a different class that genuinely looks alike, which the old color-histogram
  embedder could never do).
- **Honest latency note:** the ~15 ms a user feels is almost all CLIP query embedding on GPU; the HNSW
  search itself is <1 ms even over 500K. The engine isn't the bottleneck — the model is.

---

## B4 — efSearch tuning on real data (lower ef, same recall)

- **Swept ef on the fixed 500K index** (M=32), recall@10 vs exact brute force over 500 queries.
  Result: **~99% recall holds all the way down to ef≈20** (4.6× faster than ef=200); the high-99s
  sweet spot is **ef=32** (99.7% recall, 3.5× the throughput — 1171 → 4097 qps).
- **The lesson the journal predicted, confirmed:** ef=200 was inherited from *synthetic* uniform-random
  data, the worst case for ANN (every point ~equidistant → you must explore a huge beam). Real CLIP
  embeddings cluster, so the true neighbors are found after exploring very few candidates. **Don't
  carry synthetic ef into real data.** Adopted ef=32 as the default operating point (backend +
  pipeline). Caveat: in the live demo this is invisible (CLIP embed dominates); it's a win for the
  engine's standalone/offline throughput.

---

## B3 — the real 500K ImageNet index (the deliverable)

- **Built the headline artifact:** 500,000 ImageNet images → CLIP ViT-L/14 768-d → HNSW (M=32). Timings:
  stream+save ~76 min (network I/O, the true bottleneck), GPU embed ~38 min (~4.6 ms/img),
  **parallel build 131 s** (would be ~20 min single-threaded). **Recall@10 99.6%, same-class@10 75.7%**,
  mmap load 871 ms, query 0.35–0.83 ms.
- **Same-class@10 is the metric that earns its keep at scale:** 2.9% at 300 imgs → 55.8% at 20k → 75.7%
  at 500k. It's starved when classes are sparse (≈0.3 imgs/class at n=300) and only becomes meaningful
  once each class is densely populated (~390/class at 500k). It measures something recall-vs-bruteforce
  can't: whether the *embeddings + search together* actually retrieve semantically related images.
- **Robustness, learned the real-data way:** the stream hit corrupt EXIF, truncated JPEGs, and one
  remote disconnect — the pipeline (PIL + datasets retry) absorbed all of them and finished clean. Real
  datasets are messy; the loader has to tolerate it.
- **Crash-safety paid for itself in design:** embeddings are checkpointed to `embeddings.npy` right
  after the ~38-min GPU pass, so any later failure (build/disk) never costs the expensive work. Also
  enabled the B4 ef sweep to reuse the embeddings without re-embedding.

---

## B2 — parallel build (the headline perf win), measured

- **Built `add_batch_parallel`:** a static build-once path. Phase A (single-threaded, deterministic)
  pre-sizes *all* storage, stores/normalizes every vector, and rolls every node's layer with the
  seeded RNG — crucially this freezes the arrays at final size so phase B's lock-free reads in
  `search_layer` can never touch reallocated memory. Phase B runs worker threads off an atomic
  counter, each fully wiring one node's edges, with **striped per-node locks** (a 4096-mutex pool
  indexed by node id) guarding every link-block write (own + reverse edges). Only one lock is ever
  held at a time → no deadlock.
- **Result: 9.3× on 16 cores** (40k×384, M32/efc200: 184.6 s → 19.9 s; 8 threads = 7.6×). Scaling is
  near-linear to 8 then flattens — same memory-bandwidth wall as P1/P1.5, now hit from many cores at
  once. So parallelism is the big lever it was predicted to be, but it doesn't escape the memory wall.
- **Honesty about the data race (the interesting part):** writes are locked; the concurrent *reads*
  in `search_layer` are deliberately **not** (locking reads would serialize the whole build and
  erase the win — this is what hnswlib does too). It's memory-safe because blocks are fixed-size and
  never reallocate (phase A); the worst case is a search transiently reading a stale neighbor id or
  count, which only nudges which candidates are found — so correctness is validated by **measurement**
  (`Hnsw.ParallelBuildMatchesSequentialRecall`: parallel within 0.05 of sequential), not by claiming
  determinism. The graph is intentionally *not* bit-identical run-to-run.
- **Sequential `add` left untouched** — it's the readable teaching version and the future
  dynamic-insert path; parallel build is a separate method, gated to an empty index.

---

## P4 — real CLIP pipeline run-verified on GPU (bucket B begins)

- **First complete `build_real.py` run.** On the Studio GPU (RTX PRO 6000 Blackwell, 96 GB; torch
  CUDA OK), 300 real fashion images: CLIP **ViT-L/14 → 768-d at ~5 ms/img** (vs the minutes/img CPU
  runs that kept getting interrupted — the GPU is the whole point of bucket B). Build 0.29 s,
  **Recall@10 = 100%**, and neighbors are semantically correct (navy check shirt → 8/8 check shirts).
  Every seam (stream → embed → HNSW → SQLite → query → recall) now verified on *real* vectors.
- **Embedding budget extrapolation:** ~5 ms/img ⇒ ~42 min of pure GPU embedding for 500K (batch 256);
  streaming download/decode will likely dominate wall-clock. Re-measure on the real run.
- **Dataset decision for the 500K target:** committing to true 500K scale now. Probed HF under
  `datasets` 5.0: DiffusionDB/Imagenette are dead (loading-script based, unsupported); ImageNet-1k
  and `timm/imagenet-1k-wds` are gated. **Ungated + image-bytes + ≥500K that stream today:**
  `pixparse/cc3m-wds` (~2.9M captioned web images) and `cc12m-wds` (~12M). Plan: **ImageNet-1k**
  (1.28M, 1000 clean classes → enables a *same-class@k* quality metric beyond recall-vs-bruteforce)
  once a token is provided; **CC3M is the no-token fallback**.
- **Next:** make the dataset loader pluggable in `build_real.py`, swap fashion → chosen 500K set,
  then sequence the **parallel build perf pass before** the full 500K build (single-threaded build is
  ~O(N^1.4) cache-bound and dim768/M32 is ~6× heavier per distance than the synthetic dim128/M16).

---

## Pre-data review + hardening (before bucket B)

Reviewed all files before pouring real data through the engine. No correctness bugs (sources are
`-Wall -Wextra -Wconversion` clean; 23 tests pass). Three fixes made, all testable on synthetic data:

- **Batched CLIP embedding (#1):** `ClipEmbedder.embed_image` was 1-image-at-a-time → a GPU would
  sit ~idle and 500K could take hours. Added a batched `embed_images` (256/forward pass) + output
  L2-normalization. Runtime-verified at P4 (open_clip not installed yet).
- **`reserve(n)` for bulk build (#2):** avoids repeated reallocation/copying and the transient 2×
  memory spike (matters at 500K×768 = 1.5 GB vectors). Wired into `add_batch` and `qf_persist`.
- **Keep-pruned-connections backfill (#3):** measured before/after recall@10 at 80k —
  ef200/400/800 = 70.5/84.95/92.5% vs 70.0/85.5/92.7% (**neutral**). Expected: in 128-d uniform
  random the diversity rule almost never leaves a node short of M, so backfill rarely fires. Kept
  anyway (standard, correct, and should help on clustered data).

**Most important insight from this review (guides P5 tuning):** *uniform-random synthetic vectors
are the WORST case for ANN recall.* With no cluster structure, every point is ~equidistant (curse of
dimensionality) — that's why 80k synthetic needs ef≈800 for ~93%. **Real CLIP embeddings cluster
heavily** (the structure HNSW is built to exploit), so real-data recall at the same M/ef will be
substantially higher. Do NOT over-tune on synthetic random data; measure and tune on the real
embeddings at P5. Lever order for real data: M (32–48) > efSearch > efConstruction.

**Recall-vs-N is normal, not a bug:** ef recovers recall (80k: 70→85→93% as ef 200→400→800), so the
graph is correct; you just scale ef (and M) with N. The recall/ef curve is on the weaker side on
synthetic only because synthetic is the hard case.

---

## P1.5 — Flatten HNSW adjacency + prefetch (cache locality)

- **Done:** replaced the nested `std::vector<std::vector<std::vector<uint32_t>>>` adjacency with
  fixed-stride **contiguous blocks** — layer 0 (all nodes, hot path) in one flat array `links0_`
  (stride `M0+1`, block = `[count, ids...]`), upper layers in a small contiguous per-node block.
  Added `__builtin_prefetch` of the next neighbor's vector in `search_layer`. This realizes the
  proposal's promised "cache-aware contiguous neighbor storage." On-disk `.qfx` format unchanged.
- **Result:** build ~12–15% faster (80k: 59.3 s → 52.3 s); recall unchanged (round-trip + recall
  tests still pass, search bit-identical). The on-disk format and all 23 tests are intact.
- **Honest caveat (the point):** the *slope* barely changed — flattening removed the neighbor-list
  pointer-chase (a real, self-inflicted miss), but the dominant cost is still the **random reads of
  the 512-byte vectors** themselves, which exceed L3 and have no locality (the fundamental memory
  wall). Prefetch hides only part of that latency. So this is a genuine constant-factor win and the
  right architectural layout, but it does NOT recover a clean O(N·log N) wall-clock — exactly as
  predicted. The remaining build-speed levers are parallel build (÷cores) and efConstruction.
- Adjacency is now flat in both engines (NSW was already flat; HNSW now matches).

---

## P1 — "O(N²) build" investigation → it was NOT O(N²) (measured correction)

- **Hypothesis going in:** the per-insert `std::vector<bool> visited(num_nodes_)` allocation in
  `search_layer` made the build O(N²). Implemented a reusable `thread_local` `VisitedSet`
  (`src/visited_set.hpp`) with an O(1) version-tag clear.
- **Measured: it barely helped.** 100k build 84 s → 79 s (~6%); 30k 18 s → 15.9 s. A first version
  even regressed to no-gain because `assign(n,0)` re-zeroed all N entries every insert (fixed with
  geometric `resize`). Either way, the visited allocation was never the bottleneck.
- **Real scaling, measured (dim=128, M=16, efc=200):** 20k=9.3 s, 40k=24.1 s, 80k=59.3 s →
  wall-clock ≈ **O(N^1.4)**, not O(N²) (that would be ~4× per doubling; we see ~2.5×). The earlier
  "O(N²)" label in the docs was a wrong extrapolation from two data points.
- **Where the time actually goes:** each insert runs the diversity heuristic `select_neighbors` over
  up to `efConstruction` candidates, each compared against up to `M0` chosen → ~O(efC·M) distance
  computations per insert (thousands), plus back-link pruning. That's bounded in N but a big
  constant. The *superlinear* wall-clock comes from **cache effects**: `vectors_` exceeds L3 (~64k
  vectors at dim 128) and the nested-`std::vector` adjacency scatters memory, so each distance comp
  gets slower as N grows. Confirmed the lever: **efC 200→100 nearly halved build (62 s→35 s)** at
  80k with negligible recall change.
- **Decision:** keep the `VisitedSet` — it's correct, removes a genuine O(N)-per-insert term that
  *would* matter at ≥1M, and gives each thread its own scratch (a prerequisite for safe parallel
  build) — but it is **not** the headline. The real build-speed levers are (1) **parallel
  multi-threaded build** (÷cores), (2) **efConstruction** tuning (linear), (3) flatten layer-0
  adjacency for locality.
- **Lesson (the important one):** measure before optimizing. I "fixed" the wrong thing twice; a
  simple scaling probe (20/40/80k) revealed the true ~N^1.4 cache-bound behavior. The docs that
  claimed "O(N²)" are corrected.

---

## A7 — FastAPI backend + React frontend (bucket A complete)

- **Built:** a FastAPI backend serving the proposal's endpoints over a mock catalog, and a
  self-contained React (CDN) frontend it serves at `/`. Verified end-to-end three ways: in-process
  TestClient (`python_api` CTest), and a live uvicorn server hit with curl (health, catalog,
  frontend HTML, product images all 200). `/search/image` returns 8/8 same-category in ~7 ms.
- **Dependency-avoidance decision:** `/search/image` takes the raw image bytes as the request body
  instead of multipart/form-data, so we don't need `python-multipart`. The frontend just does
  `fetch(url, {body: file})`. Fewer deps, fully functional.
- **Frontend scope:** a single `index.html` with React via CDN — runs with zero `npm install`, so
  the demo is immediately runnable. Production would migrate to a Vite project; logged, not needed
  for scaffold/dry-run.
- **Bug caught:** SQLite connections are thread-bound by default; the web server serves requests on
  a different thread than where the catalog was built → `sqlite3.ProgrammingError`. Fixed with
  `check_same_thread=False`, which fits our static-build/concurrent-read model (read-mostly).
- **State:** bucket A is complete — the full vertical slice works on CPU with synthetic data:
  image → embedding → HNSW search → SQLite metadata → JSON API → React grid. Bucket B (real CLIP
  model + 500K dataset + tuning + final benchmarks on the local 5070 Ti / Core Ultra 9) is what
  remains, and every seam for it already exists.

---

## A6 — embedding pipeline scaffold + CPU dry-run

- **Built:** `python/qf_pipeline/` — a pluggable `Embedder` abstraction, a SQLite metadata store,
  and `build_catalog`/`search_similar`. The `dry_run` generates synthetic colored product images,
  embeds them, builds a QueryForge index + SQLite catalog, and queries — returns 8/8 same-category
  matches. Runs on CPU in seconds with no model download.
- **Key scoping decision:** real CLIP needs `open_clip_torch`/`transformers` (not installed) and is
  slow without a GPU (which is off). Rather than block on a big download, the pipeline is built
  around an `Embedder` interface with two implementations:
  - `HistogramEmbedder` (grid of average colors, dim=48) — no heavy deps, real enough that similar
    images cluster, so the *whole* pipeline is exercised now.
  - `ClipEmbedder` (ViT-B-32, 512-d) — written and documented, lazy-imports torch/open_clip; this
    is the production path for the 500K catalog on the laptop/GPU. Swapping it in changes one line.
- **Metadata store decision resolved (for local):** SQLite via the stdlib `sqlite3` — zero
  dependency, ACID, keyed by the index id. Postgres remains a later swap behind `MetadataStore`.
- **Why this is the right "dry-run ready" state:** every architectural seam (image→vector,
  vector→index, id→metadata, query→results) is implemented and tested end-to-end; only the *real
  model* and *real dataset* are deferred to bucket B, and they drop into existing slots.

---

## A5 — Pybind11 bindings

- **Built:** `queryforge.HnswIndex` Python module (`python/bindings.cpp`). NumPy float32 arrays
  cross the boundary; `search()` returns `(ids, distances)` NumPy arrays — drops straight into ML
  code. `add_batch` takes an (n, dim) array.
- **Two build gotchas, both fixed:**
  1. *PIC:* the static `libqueryforge.a` was compiled without `-fPIC`, so it couldn't link into a
     shared `.so`. Fix: `POSITION_INDEPENDENT_CODE ON` on the engine target. (Lesson: anything that
     might end up inside a shared object needs position-independent code.)
  2. *CTest interpreter:* `${Python_EXECUTABLE}` was empty under this pybind11 version → used
     `find_program(QF_PYTHON ...)` instead, and `enable_testing()` had to move to the top level so
     the python/ subdir could register a test.
- **Design:** `QF_BUILD_PYTHON` defaults OFF so the plain C++ build stays lightweight; turn it on to
  build the module. The demo (`example.py`) doubles as the `python_bindings` CTest case.
- **Why this matters:** A6's CLIP pipeline is Python; these bindings are the bridge that lets the
  heavy search stay in optimized C++ while the embedding/orchestration lives in Python.

---

## A4 — persistence (save + mmap load)

- **Built:** a binary `.qfx` format (`HnswIndex::save` / `::load` in `src/persistence.cpp`), with a
  header, a contiguous vectors block, and CSR-style per-layer adjacency. `load()` memory-maps the
  file on POSIX (`mmap`) and reads structured data straight from the mapped bytes; non-POSIX falls
  back to a buffered read. RAII wrappers (`FileBytes`) free the mapping/heap automatically.
- **Result:** loading is ~1000–1570× faster than rebuilding (e.g. 100k: build 84 s → load 54 ms),
  files are ~640 bytes/vector, and round-trips are bit-exact (loaded index searches identically —
  verified by `Persistence.RoundTripSearchIsIdentical`).
- **Teaching point — what mmap buys:** we don't "read and parse" the file; the OS maps its bytes
  into our address space and pages them in on demand. For the vectors block (the bulk) this is
  essentially free to "load". The win is build-once / serve-many: a service restarts in
  milliseconds instead of re-running a multi-second/minute build.
- **Surprise caught:** the persist tool first reported "sanity FAIL" — but that was a bad check
  (it assumed approximate search at ef=32 returns the *exact* nearest on 100k, which it needn't).
  The real integrity check is loaded.search == original.search, which passes. Lesson: assert the
  invariant you actually guarantee (determinism), not one you don't (exact recall).
- **Bottleneck identified (logged, not fixed):** build time is superlinear because each insert
  allocates+zeroes a size-N `vector<bool> visited` in `search_layer` → ~O(N²) build. Top priority
  for the perf pass (reusable visited-version array) before we scale to 500k. Load is unaffected.
- **Scope note:** A4 covers the engine's index persistence. The product *metadata* store (SQLite vs
  Postgres, still an open decision) is deferred to A6 where real product metadata first appears.

---

## A3 — full HNSW (two steps)

- **Built HNSW in two measured steps** (per the chosen learning approach):
  - *Step 1 (layers, naive selection):* added the multi-layer hierarchy on top of the A2 beam
    search — random exponential layer assignment, ef=1 greedy descent on upper layers, wide beam
    only at layer 0, per-layer adjacency, layer-0 density 2*M. Result: +14–21 recall points over
    NSW at the same nodes-visited budget; HNSW@M=16 matched NSW@M=32.
  - *Step 2 (diversity heuristic, Algorithm 4):* replaced naive closest-M with "accept a candidate
    only if it's closer to the base than to any already-accepted neighbor." Result: crossed the
    **95% recall target** (95.9% @ ef=200, M=16) and made low-ef queries ~35% faster (122→80 µs).
- **Key insight — the heuristic is not a pure win at every ef.** It makes the graph *leaner*
  (fewer, more diverse edges), which: (a) speeds up search (fewer neighbors to expand per hop),
  (b) raises peak recall (no blind spots), but (c) costs a hair of recall at very low ef where the
  extra redundant edges of the naive graph happened to help. The operating points you'd ship at
  (higher recall) favor the heuristic on both axes. Good lesson: "better algorithm" is often a
  trade-curve shift, not a uniform improvement — you have to look at the whole recall/latency curve.
- **Why ef=1 on upper layers:** the upper layers exist only to ferry the search to a good entry
  point near the query; we don't need the best-k there, just "get closer," so a single greedy walk
  (ef=1) is enough and cheap. The expensive wide beam runs once, at layer 0.
- **Perf TODOs logged (not done):** layer-0 adjacency is still nested `std::vector` (A2 used a flat
  array); flattening the hot layer and reusing a visited-version array are the obvious next
  optimizations once we care about absolute speed.

---

## A2 — NSW single-layer graph

- **Built:** a single-layer Navigable Small World graph (`NswIndex`) with greedy beam search
  (`search_layer`), insert, and a flat contiguous adjacency array (`links_`, capacity `M` per
  node) for cache locality. Plus a recall harness (`qf_recall`) that compares against exact
  brute force, and 5 unit tests (incl. recall thresholds).
- **Design choices:**
  - *Contiguous storage:* vectors and adjacency are flat arrays indexed by node id — no pointer
    chasing, cache-friendly. This is the realistic "custom memory layout" from the plan.
  - *Cosine = normalized dot:* for cosine we normalize vectors at insert and the query at search,
    then distance is `1 - dot`. Cheaper than recomputing norms every comparison.
  - *Naive neighbor selection (closest-M)* on purpose — A3 swaps in the diversity heuristic and
    we measure the difference.
- **Results (N=10k, dim=128, L2 — see BASELINES):** the graph visits only ~2% of nodes (core
  promise works). Recall climbs with `efSearch` (20%→50%→81% for ef=10/50/200) and jumps to
  **94.5%** when `M` goes 16→32. So `M` is the stronger recall lever; `efSearch` is the per-query
  speed/recall dial.
- **Why recall isn't ~100% yet:** single layer + naive selection. The two A3 upgrades address
  exactly this: (a) hierarchical layers give better entry points so search starts near the answer;
  (b) the diversity heuristic avoids redundant edges so the same M buys more reach.
- **Note for later (perf):** `search_layer` allocates a `std::vector<bool> visited` per call —
  fine for correctness, but a reusable "visited version" array will cut allocation overhead when
  we optimize. Logged, not done.

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

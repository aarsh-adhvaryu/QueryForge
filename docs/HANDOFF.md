# HANDOFF — finishing QueryForge on the local machine

You're reading this on the owner's **local machine** (target: RTX 5070 Ti + Intel Core Ultra 9,
AVX2 / no AVX-512). The engine and the cloud build are **done** (buckets A + B). The **only** task
left is the **local re-measurement**: reproduce the benchmarks here and fill the `local` rows in
[BASELINES.md](BASELINES.md). Read [../CLAUDE.md](../CLAUDE.md) first — it has the full status.

## What the clone has vs. what it doesn't

- ✅ **In git:** all C++/Python source, tests, docs, the FastAPI backend + React frontend.
- ❌ **Not in git (gitignored, ~10 GB):** the 500K artifacts under `qf_data/im500k/` — `index.qfx`
  (1.6 GB), `embeddings.npy` (1.5 GB), `metadata.db` (69 MB), `images/` (~500K thumbnails).
  These must be **transferred from the Studio** or **regenerated here**. See "Getting the data".

## 0. Setup (once)

```bash
cmake -S . -B build -DQF_BUILD_PYTHON=ON && cmake --build build -j
ctest --test-dir build --output-on-failure        # expect 25/25 green — confirms AVX2 path works here
pip install -r python/requirements.txt -r backend/requirements.txt
# for regenerating embeddings / running the real demo (not needed for the benchmark-only path):
pip install open_clip_torch datasets
nvidia-smi && python -c "import torch; print('cuda:', torch.cuda.is_available())"   # confirm the 5070 Ti
```

## Getting the data — pick one

**Path A — benchmark only (recommended, fast).** Copy just **`embeddings.npy`** (1.5 GB) from the
Studio (`/teamspace/studios/this_studio/qf_data/im500k/embeddings.npy`) — e.g. download via the
Lightning UI, or `scp`/cloud bucket. Then rebuild + benchmark locally (no GPU, no Hugging Face):

```bash
PYTHONPATH=build/python:python python -m qf_pipeline.rebuild_index \
    --embeddings /path/to/embeddings.npy --threads <local_core_count> --out qf_data/im500k_local
```

That prints local **build time, mmap-load time, Recall@10, and the ef sweep** — exactly the numbers
to record. This is enough to finish the project.

**Path B — full rebuild (for the live demo too).** Regenerate everything on the local GPU (needs
`huggingface-cli login` with ImageNet-1k access; ~30–90 min depending on connection + the 5070 Ti):

```bash
PYTHONPATH=build/python:python python -m qf_pipeline.build_real \
    --dataset imagenet --limit 500000 --threads <cores> --out qf_data/im500k_local
```

This produces `index.qfx` + `embeddings.npy` + `metadata.db` + `images/`, and also reports the local
GPU embed throughput (another `local` number worth recording).

> To also run the **web demo** locally you need `images/` + `metadata.db`. Either use Path B, or copy
> `images/` + `metadata.db` from the Studio alongside the transferred `embeddings.npy`.

## 1. Record the numbers

Open [BASELINES.md](BASELINES.md). The bucket-B tables (B2/B3/B4) were measured on `studio-gpu`
(16-core + RTX PRO 6000). Add a `local` column or sibling rows for: parallel build time, mmap load,
Recall@10, the ef sweep, and (Path B) GPU embed time. Note the local core count — the 9.3× build
speedup will differ on the Core Ultra 9's core count, and that's the point of re-measuring. Also
re-run the engine micro-benchmarks for completeness:

```bash
./build/bin/qf_distance_bench
./build/bin/qf_recall algo=hnsw N=10000 dim=128 M=16 efc=200 ef=200 k=10 metric=l2
./build/bin/qf_persist N=30000 dim=128 M=16 efc=200
```

## 2. Run the demo locally (optional, Path B or copied images)

```bash
QF_INDEX_DIR=$PWD/qf_data/im500k_local \
    PYTHONPATH=build/python:python uvicorn backend.app:app --host 0.0.0.0 --port 8000
# open http://127.0.0.1:8000/  — grid, click-to-similar, upload-to-search. /health shows mode=real.
```

## 3. Commit

```bash
git add docs/BASELINES.md && git commit -m "B6: local re-measurement on Core Ultra 9 / RTX 5070 Ti"
git push origin main
```

That closes the project. Optional extra: an `M` sweep (16/24/48) to show the memory/build vs recall
tradeoff — rebuild with `rebuild_index.py --M <value>` from the same `embeddings.npy`.

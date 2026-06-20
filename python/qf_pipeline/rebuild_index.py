"""Rebuild the HNSW index from cached embeddings and benchmark it — the LOCAL re-measurement tool.

Use this on the local machine (RTX 5070 Ti / Core Ultra 9) to reproduce bucket B's numbers WITHOUT
re-streaming or re-embedding 500K images: copy `embeddings.npy` over from the build, then run:

    PYTHONPATH=build/python:python python -m qf_pipeline.rebuild_index \
        --embeddings /path/to/embeddings.npy --threads <cores> --out /path/to/qf_local

It builds the index (parallel), saves `index.qfx`, then prints build time, mmap-load time, Recall@10
vs exact brute force, and an efSearch sweep — i.e. the exact rows to fill into docs/BASELINES.md under
the `local` machine. No GPU, no Hugging Face, no dataset needed; just the cached embeddings.

(For the full live web demo on the laptop you also need the images + metadata.db — either copy them
over too, or regenerate everything from scratch with `qf_pipeline.build_real`. See docs/HANDOFF.md.)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import queryforge as qf


def recall_and_latency(idx, E: np.ndarray, n_queries: int, k: int, efs, seed: int = 7):
    """Recall@k vs exact brute force + avg query latency, for each ef in `efs`."""
    rng = np.random.default_rng(seed)
    N = len(E)
    qi = rng.choice(N, min(n_queries, N), replace=False)
    Q = E[qi]
    # Exact top-k by cosine (== dot on L2-normalized vectors), computed in chunks to bound memory.
    scores = np.empty((len(qi), N), np.float32)
    CH = 50000
    for s in range(0, N, CH):
        scores[:, s:s + CH] = Q @ E[s:s + CH].T
    gt = np.argpartition(-scores, k, axis=1)[:, :k]

    rows = []
    for ef in efs:
        hit = 0
        t0 = time.time()
        for r in range(len(qi)):
            ids, _ = idx.search(Q[r], k=k, ef=ef)
            hit += len(set(int(x) for x in ids) & set(int(x) for x in gt[r]))
        dt = (time.time() - t0) / len(qi) * 1e3
        rows.append((ef, hit / (len(qi) * k), dt))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="path to embeddings.npy (n, dim) float32")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0, help="build worker threads; 0 = all cores")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="/tmp/qf_local")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    E = np.load(args.embeddings).astype(np.float32)
    N, dim = E.shape
    print(f"embeddings: {N} x {dim}  (M={args.M}, efc={args.efc})")

    idx = qf.HnswIndex(dim=dim, M=args.M, ef_construction=args.efc, metric=qf.Metric.Cosine)
    idx.reserve(N)
    t0 = time.time()
    idx.add_batch_parallel(E, threads=args.threads)
    build_s = time.time() - t0
    path = os.path.join(args.out, "index.qfx")
    idx.save(path)
    print(f"parallel build (threads={args.threads or 'all'}): {build_s:.1f}s  top_layer={idx.max_layer}")

    t0 = time.time()
    idx2 = qf.HnswIndex.load(path)
    load_ms = (time.time() - t0) * 1e3
    print(f"mmap load: {load_ms:.0f} ms  ({os.path.getsize(path) / 1e9:.2f} GB)\n")

    print(f"{'ef':>4} {'recall@10':>10} {'avg ms':>8} {'qps':>7}")
    for ef, rec, dt in recall_and_latency(idx2, E, args.queries, args.k,
                                          (16, 20, 24, 32, 48, 64, 100, 200)):
        print(f"{ef:>4} {rec * 100:>9.1f}% {dt:>8.3f} {1000 / dt:>7.0f}")
    print("\nFill these into docs/BASELINES.md under machine=local.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

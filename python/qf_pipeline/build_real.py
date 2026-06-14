"""Build a QueryForge index from a REAL image dataset (fashion products) with CLIP embeddings.

Same script for the small validation and the full scale run — just change --limit. Steps:
  1. Pull `limit` items from `ashraq/fashion-product-images-small` (real products + metadata).
  2. Embed images with CLIP (ViT-L/14 -> 768-d by default).
  3. Build the HNSW index + SQLite metadata, save .qfx + metadata.db + images.
  4. Validate: a sample visual-similarity query + Recall@10 vs exact brute force on real vectors.

Usage (from repo root, with the module built: cmake -S . -B build -DQF_BUILD_PYTHON=ON):
  PYTHONPATH=build/python:python python -m qf_pipeline.build_real --limit 300 --out /tmp/qf_real
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import queryforge as qf

from qf_pipeline.embedder import ClipEmbedder
from qf_pipeline.metadata import MetadataStore, Product


def load_products(limit: int, images_dir: str) -> list[dict]:
    """Stream `limit` items from the dataset, save images, return product metadata dicts."""
    from datasets import load_dataset

    os.makedirs(images_dir, exist_ok=True)
    ds = load_dataset("ashraq/fashion-product-images-small", split="train", streaming=True)
    products = []
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        img = ex["image"]
        path = os.path.join(images_dir, f"{i:06d}.jpg")
        img.convert("RGB").save(path)
        name = ex.get("productDisplayName") or f"item-{ex.get('id', i)}"
        products.append({
            "name": name,
            "category": ex.get("articleType") or "unknown",
            "color": ex.get("baseColour") or "",
            "price": round(9.99 + (int(ex.get("id", i)) % 90), 2),
            "image_path": path,
        })
    return products


def recall_at_k(index, embeddings: np.ndarray, n_queries: int, k: int, ef: int) -> float:
    """Average Recall@k of the index vs exact brute force (cosine == dot on normalized vectors)."""
    n = embeddings.shape[0]
    n_queries = min(n_queries, n)
    total = 0.0
    for qi in range(n_queries):
        q = embeddings[qi]
        exact = np.argsort(-(embeddings @ q))[:k]          # exact top-k by cosine
        ids, _ = index.search(q, k=k, ef=ef)
        hits = len(set(int(x) for x in ids) & set(int(x) for x in exact))
        total += hits / k
    return total / n_queries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b82k")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--ef", type=int, default=200)
    ap.add_argument("--out", default="/tmp/qf_real")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    images_dir = os.path.join(args.out, "images")

    print(f"[1/4] loading {args.limit} real products ...")
    products = load_products(args.limit, images_dir)
    print(f"      saved {len(products)} images to {images_dir}")

    print(f"[2/4] embedding with CLIP {args.model} ({args.pretrained}) ...")
    embedder = ClipEmbedder(model_name=args.model, pretrained=args.pretrained)
    print(f"      device={embedder.device} dim={embedder.dim}")
    t0 = time.time()
    vecs = embedder.embed_images([p["image_path"] for p in products])
    embed_s = time.time() - t0
    print(f"      embedded {vecs.shape} in {embed_s:.1f}s ({embed_s/len(products)*1000:.0f} ms/img)")

    print(f"[3/4] building index (M={args.M}, efc={args.efc}, cosine) ...")
    index = qf.HnswIndex(dim=embedder.dim, M=args.M, ef_construction=args.efc, metric=qf.Metric.Cosine)
    index.reserve(len(products))
    t0 = time.time()
    ids = index.add_batch(vecs)
    build_s = time.time() - t0
    store = MetadataStore(os.path.join(args.out, "metadata.db"))
    store.add_many(Product(id=int(i), name=p["name"], category=p["category"],
                           price=p["price"], image_path=p["image_path"])
                   for i, p in zip(ids, products))
    store.commit()
    index.save(os.path.join(args.out, "index.qfx"))
    print(f"      built {len(index)} vectors in {build_s:.2f}s, top layer {index.max_layer}; saved .qfx + metadata.db")

    print("[4/4] validate ...")
    # Sample query: use item 0, show its nearest neighbors.
    q_ids, q_dist = index.search(vecs[0], k=8, ef=args.ef)
    q0 = store.get(0)
    print(f"      query: '{q0.name}' [{q0.category}]")
    for rid, rd in zip(q_ids, q_dist):
        p = store.get(int(rid))
        print(f"        {rd:.3f}  [{p.category:<14}] {p.name}")
    rec = recall_at_k(index, vecs, n_queries=min(100, len(products)), k=10, ef=args.ef)
    print(f"\n      Recall@10 vs brute force (ef={args.ef}): {rec*100:.1f}%")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

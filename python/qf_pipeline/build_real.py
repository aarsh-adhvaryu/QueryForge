"""Build a QueryForge index from a REAL image dataset with CLIP embeddings.

Same script for the small validation and the full scale run — just change --limit and --dataset.
Steps:
  1. Stream `limit` items from the chosen dataset (real images + metadata).
  2. Embed images with CLIP (ViT-L/14 -> 768-d by default).
  3. Build the HNSW index + SQLite metadata, save .qfx + metadata.db + images.
  4. Validate: a sample visual-similarity query + Recall@10 vs exact brute force, and (when the
     dataset has class labels) same-class@10 — a quality metric that recall-vs-bruteforce can't give.

Datasets (--dataset):
  * imagenet — ILSVRC/imagenet-1k (gated; needs `huggingface-cli login`). 1.28M real photos, 1000
    clean classes stored as `category` so we can report same-class@k. The 500K-scale target.
  * cc3m     — pixparse/cc3m-wds. ~2.9M captioned web images, ungated (no token). Caption -> name.
  * fashion  — ashraq/fashion-product-images-small. ~44k low-res fashion products (the A6/A7 set).

Usage (from repo root, with the module built: cmake -S . -B build -DQF_BUILD_PYTHON=ON):
  PYTHONPATH=build/python:python python -m qf_pipeline.build_real \
      --dataset imagenet --limit 300 --out /tmp/qf_real
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import queryforge as qf

from qf_pipeline.embedder import ClipEmbedder
from qf_pipeline.metadata import MetadataStore, Product


def _save_image(img, images_dir: str, i: int, max_side: int) -> str:
    """Save one PIL image to a sharded path, optionally downscaled. Returns the path.

    Sharding (10k images per subdir) keeps any single directory small — half a million files in
    one folder makes listing/globbing painfully slow on ext4. Downscaling to `max_side` bounds
    disk + later decode cost; CLIP resizes to 224 anyway, so a 256px cap is lossless for search.
    """
    sub = os.path.join(images_dir, f"{i // 10000:04d}")
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, f"{i:07d}.jpg")
    img = img.convert("RGB")
    if max_side:
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    img.save(path, quality=90)
    return path


def _fake_price(seed: int) -> float:
    """Deterministic stand-in price so the demo catalog looks like a real store."""
    return round(9.99 + (seed % 90), 2)


def load_imagenet(limit: int, images_dir: str, max_side: int) -> list[dict]:
    """Stream ImageNet-1k; store the human-readable class name in `category` (for same-class@k)."""
    from datasets import load_dataset

    ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    label_feat = ds.features["label"] if ds.features else None  # ClassLabel -> int2str
    products = []
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        label = ex["label"]
        cls = label_feat.int2str(label) if label_feat is not None else str(label)
        path = _save_image(ex["image"], images_dir, i, max_side)
        products.append({"name": f"{cls} #{i}", "category": cls,
                         "price": _fake_price(i), "image_path": path})
    return products


def load_cc3m(limit: int, images_dir: str, max_side: int) -> list[dict]:
    """Stream CC3M (WebDataset): `jpg` image + `txt` caption. No class labels."""
    from datasets import load_dataset

    ds = load_dataset("pixparse/cc3m-wds", split="train", streaming=True)
    products = []
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        caption = (ex.get("txt") or f"image-{i}").strip().replace("\n", " ")[:200]
        path = _save_image(ex["jpg"], images_dir, i, max_side)
        products.append({"name": caption, "category": "web",
                         "price": _fake_price(i), "image_path": path})
    return products


def load_fashion(limit: int, images_dir: str, max_side: int) -> list[dict]:
    """Stream the small fashion-product set (the A6/A7 dataset)."""
    from datasets import load_dataset

    ds = load_dataset("ashraq/fashion-product-images-small", split="train", streaming=True)
    products = []
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        name = ex.get("productDisplayName") or f"item-{ex.get('id', i)}"
        path = _save_image(ex["image"], images_dir, i, max_side)
        products.append({"name": name, "category": ex.get("articleType") or "unknown",
                         "price": _fake_price(int(ex.get("id", i))), "image_path": path})
    return products


DATASETS = {"imagenet": load_imagenet, "cc3m": load_cc3m, "fashion": load_fashion}


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


def same_class_at_k(index, embeddings: np.ndarray, categories: list[str],
                    n_queries: int, k: int, ef: int) -> float:
    """Fraction of the k neighbors (excluding self) that share the query's class.

    A quality metric recall-vs-bruteforce can't give: it asks whether the *embeddings + search*
    actually retrieve semantically similar images, not just whether the ANN approximates brute force.
    """
    n = embeddings.shape[0]
    n_queries = min(n_queries, n)
    total = 0.0
    for qi in range(n_queries):
        ids, _ = index.search(embeddings[qi], k=k + 1, ef=ef)   # +1: first hit is the query itself
        neigh = [int(x) for x in ids if int(x) != qi][:k]
        if not neigh:
            continue
        total += sum(categories[j] == categories[qi] for j in neigh) / len(neigh)
    return total / n_queries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="imagenet", choices=list(DATASETS))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b82k")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--ef", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0, help="parallel-build worker threads; 0=all cores")
    ap.add_argument("--max-side", type=int, default=256, help="downscale cap on saved images; 0=off")
    ap.add_argument("--out", default="/tmp/qf_real")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    images_dir = os.path.join(args.out, "images")

    print(f"[1/4] streaming {args.limit} items from '{args.dataset}' ...")
    t0 = time.time()
    products = DATASETS[args.dataset](args.limit, images_dir, args.max_side)
    print(f"      saved {len(products)} images to {images_dir} in {time.time() - t0:.1f}s")

    print(f"[2/4] embedding with CLIP {args.model} ({args.pretrained}) ...")
    embedder = ClipEmbedder(model_name=args.model, pretrained=args.pretrained)
    print(f"      device={embedder.device} dim={embedder.dim}")
    t0 = time.time()
    vecs = embedder.embed_images([p["image_path"] for p in products])
    embed_s = time.time() - t0
    print(f"      embedded {vecs.shape} in {embed_s:.1f}s ({embed_s/len(products)*1000:.0f} ms/img)")
    # Checkpoint the embeddings immediately: at 500K the GPU embedding is the expensive pass, so
    # we never want a later failure (build/disk) to throw it away. Re-buildable from this + images.
    np.save(os.path.join(args.out, "embeddings.npy"), vecs)

    print(f"[3/4] building index (M={args.M}, efc={args.efc}, cosine, threads={args.threads or 'all'}) ...")
    index = qf.HnswIndex(dim=embedder.dim, M=args.M, ef_construction=args.efc, metric=qf.Metric.Cosine)
    index.reserve(len(products))
    t0 = time.time()
    ids = index.add_batch_parallel(vecs, threads=args.threads)
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
        print(f"        {rd:.3f}  [{p.category:<18}] {p.name}")
    rec = recall_at_k(index, vecs, n_queries=min(100, len(products)), k=10, ef=args.ef)
    print(f"\n      Recall@10 vs brute force (ef={args.ef}): {rec*100:.1f}%")
    categories = [p["category"] for p in products]
    if len(set(categories)) > 1:
        sc = same_class_at_k(index, vecs, categories, n_queries=min(100, len(products)), k=10, ef=args.ef)
        print(f"      Same-class@10 (semantic quality):       {sc*100:.1f}%")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

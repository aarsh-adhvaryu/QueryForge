"""Tie the engine + embedder + metadata store together.

build_catalog(): embed each product image, insert the vectors into a QueryForge index, and store
the matching metadata (the index id and the metadata id are kept in sync by insertion order).

search_similar(): embed a query image, ask QueryForge for the nearest ids, then enrich those ids
with metadata from the store — exactly what the backend API will do in A7.
"""
from __future__ import annotations

from typing import Sequence

import queryforge as qf

from .embedder import Embedder
from .metadata import MetadataStore, Product


def build_catalog(products_meta: Sequence[dict], embedder: Embedder,
                  metric=qf.Metric.Cosine, M: int = 16, ef_construction: int = 200,
                  db_path: str = ":memory:"):
    """products_meta: list of dicts with keys name, category, price, image_path.

    Returns (index, store).
    """
    index = qf.HnswIndex(dim=embedder.dim, M=M, ef_construction=ef_construction, metric=metric)
    store = MetadataStore(db_path)

    paths = [m["image_path"] for m in products_meta]
    vectors = embedder.embed_images(paths)            # (n, dim) float32
    ids = index.add_batch(vectors)                    # ids are 0..n-1 in insertion order

    store.add_many(
        Product(id=int(i), name=m["name"], category=m["category"],
                price=float(m["price"]), image_path=m["image_path"])
        for i, m in zip(ids, products_meta)
    )
    store.commit()
    return index, store


def search_similar(index, store: MetadataStore, embedder: Embedder, query_image: str,
                   k: int = 5, ef: int = 32) -> list[dict]:
    """Return the k most visually similar products to `query_image`, enriched with metadata."""
    vec = embedder.embed_image(query_image).astype("float32")
    ids, dists = index.search(vec, k=k, ef=ef)

    results = []
    for i, d in zip(ids, dists):
        p = store.get(int(i))
        results.append({
            "id": int(i),
            "score": float(d),
            "name": p.name if p else None,
            "category": p.category if p else None,
            "price": p.price if p else None,
            "image_path": p.image_path if p else None,
        })
    return results

"""End-to-end CPU dry-run of the QueryForge demo pipeline — no model download, no GPU.

Generates synthetic colored "product" images grouped by category, embeds them with the
dependency-free HistogramEmbedder, builds a QueryForge index + SQLite catalog, then runs a
visual-similarity query and checks the results are dominated by the query's own category.

Run (from the repo root, after building the module with -DQF_BUILD_PYTHON=ON):
    PYTHONPATH=build/python:python python -m qf_pipeline.dry_run
"""
from __future__ import annotations

import os
import tempfile

import queryforge as qf

from qf_pipeline import HistogramEmbedder, build_catalog, search_similar
from qf_pipeline.synthetic import CATEGORIES, generate_catalog, make_image


def main() -> int:
    print("queryforge version:", qf.__version__)
    workdir = tempfile.mkdtemp(prefix="qf_pipeline_")

    # 1. Generate a synthetic catalog of images + metadata.
    products_meta = generate_catalog(workdir, per_category=15)
    print(f"generated {len(products_meta)} synthetic product images in {workdir}")

    # 2. Build the index + metadata catalog (HistogramEmbedder + cosine).
    embedder = HistogramEmbedder(grid=4)  # dim = 4*4*3 = 48
    index, store = build_catalog(products_meta, embedder, metric=qf.Metric.Cosine)
    print(f"indexed: {len(index)} vectors (dim={embedder.dim}), catalog rows={store.count()}")

    # 3. Query with a fresh "red shoe" image and inspect the neighbors.
    query_path = os.path.join(workdir, "query_red.png")
    make_image(query_path, CATEGORIES["red_shoe"], seed=9999)
    results = search_similar(index, store, embedder, query_path, k=8, ef=64)

    print("\nTop matches for a red-shoe query:")
    for r in results:
        print(f"  id={r['id']:>3}  score={r['score']:.4f}  {r['category']:<11}  {r['name']}  ${r['price']}")

    # 4. Sanity: the query is red, so most neighbors should be red shoes.
    red = sum(1 for r in results if r["category"] == "red_shoe")
    print(f"\nred_shoe in top-8: {red}/8")
    assert red >= 6, "expected the visual search to return mostly same-category items"
    print("Dry-run OK ✔  (images -> embeddings -> index -> metadata -> search all working)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

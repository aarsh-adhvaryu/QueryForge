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

import numpy as np
import queryforge as qf

from qf_pipeline import HistogramEmbedder, build_catalog, search_similar

# Category name -> base RGB color. The synthetic images are this color plus noise.
CATEGORIES = {
    "red_shoe":   (220, 40, 40),
    "blue_shirt": (40, 60, 220),
    "green_bag":  (40, 200, 80),
    "yellow_hat": (235, 220, 50),
}
PER_CATEGORY = 15
IMG_SIZE = 48


def make_image(path: str, base_rgb, seed: int) -> None:
    """Write a noisy solid-color PNG, so same-category images look alike but aren't identical."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    base = np.array(base_rgb, dtype=np.float32)
    noise = rng.normal(0, 18, size=(IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
    arr = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def main() -> int:
    print("queryforge version:", qf.__version__)
    workdir = tempfile.mkdtemp(prefix="qf_pipeline_")

    # 1. Generate a synthetic catalog of images + metadata.
    products_meta = []
    seed = 0
    for category, color in CATEGORIES.items():
        for j in range(PER_CATEGORY):
            path = os.path.join(workdir, f"{category}_{j}.png")
            make_image(path, color, seed)
            seed += 1
            products_meta.append({
                "name": f"{category.replace('_', ' ').title()} #{j}",
                "category": category,
                "price": round(19.99 + (seed % 7) * 5, 2),
                "image_path": path,
            })
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

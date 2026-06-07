"""Synthetic image catalog generator — shared by the dry-run and the backend mock catalog.

Produces small noisy solid-color PNGs grouped by category, so same-category images look alike but
aren't identical. Stands in for the real 500K-image dataset until that arrives (bucket B).
"""
from __future__ import annotations

import os

import numpy as np

# Category name -> base RGB color.
CATEGORIES = {
    "red_shoe":   (220, 40, 40),
    "blue_shirt": (40, 60, 220),
    "green_bag":  (40, 200, 80),
    "yellow_hat": (235, 220, 50),
}


def make_image(path: str, base_rgb, seed: int, size: int = 48) -> None:
    """Write a noisy solid-color PNG to `path`."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    base = np.array(base_rgb, dtype=np.float32)
    noise = rng.normal(0, 18, size=(size, size, 3)).astype(np.float32)
    arr = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def generate_catalog(workdir: str, per_category: int = 15, size: int = 48) -> list[dict]:
    """Generate images into `workdir`; return product metadata dicts (name/category/price/path)."""
    os.makedirs(workdir, exist_ok=True)
    products: list[dict] = []
    seed = 0
    for category, color in CATEGORIES.items():
        for j in range(per_category):
            path = os.path.join(workdir, f"{category}_{j}.png")
            make_image(path, color, seed, size)
            products.append({
                "name": f"{category.replace('_', ' ').title()} #{j}",
                "category": category,
                "price": round(19.99 + (seed % 7) * 5, 2),
                "image_path": path,
            })
            seed += 1
    return products

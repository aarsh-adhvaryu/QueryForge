"""Image -> vector embedders.

An Embedder turns an image file into a fixed-length float32 vector. Two implementations:
  * HistogramEmbedder — a small, dependency-free "visual" embedder (grid of average colors).
    Real enough that similar-looking images land near each other, so the whole pipeline can be
    dry-run on CPU with no model download. Good for demos/tests, not production quality.
  * ClipEmbedder — real CLIP embeddings (needs torch + open_clip). This is what you'd use for the
    actual 500K-image catalog; it runs on the local RTX 5070 Ti, or CPU as a slow fallback.
The rest of the pipeline doesn't care which one it gets — that's the point of the abstraction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Embedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""

    @abstractmethod
    def embed_image(self, path: str) -> np.ndarray:
        """Return a float32 vector of shape (dim,) for the image at `path`."""

    def embed_images(self, paths: Sequence[str]) -> np.ndarray:
        """Embed many images; returns a float32 array of shape (len(paths), dim)."""
        if not paths:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([self.embed_image(p) for p in paths]).astype(np.float32)


class HistogramEmbedder(Embedder):
    """Average color over a grid of cells -> captures dominant color + coarse layout.

    For a grid of G x G cells over RGB, dim = G*G*3. Cosine similarity on these vectors groups
    images with similar colors/layout — enough to show the search working end-to-end on CPU.
    """

    def __init__(self, grid: int = 4):
        self.grid = grid
        self._dim = grid * grid * 3

    @property
    def dim(self) -> int:
        return self._dim

    def embed_image(self, path: str) -> np.ndarray:
        from PIL import Image

        size = self.grid * 8
        img = Image.open(path).convert("RGB").resize((size, size))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (size, size, 3)
        cell = size // self.grid
        feats = []
        for gy in range(self.grid):
            for gx in range(self.grid):
                block = arr[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell, :]
                feats.append(block.reshape(-1, 3).mean(axis=0))
        return np.concatenate(feats).astype(np.float32)


class ClipEmbedder(Embedder):
    """Real CLIP image embeddings (lazy deps).

    Requires `torch` and `open_clip_torch` (`pip install open_clip_torch`). Defaults to ViT-B-32
    (512-d). Uses CUDA if available (the RTX 5070 Ti), otherwise CPU. Not exercised in the CPU
    dry-run; this is the production path for the real catalog.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k",
                 device: str | None = None):
        import torch  # noqa: F401

        try:
            import open_clip
        except ImportError as e:  # pragma: no cover - exercised only without the dep
            raise ImportError(
                "ClipEmbedder needs open_clip_torch: pip install open_clip_torch"
            ) from e

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self._dim = int(self.model.visual.output_dim)

    @property
    def dim(self) -> int:
        return self._dim

    def embed_image(self, path: str) -> np.ndarray:
        return self.embed_images([path])[0]

    def embed_images(self, paths, batch_size: int | None = None) -> np.ndarray:
        """Batched embedding — runs `batch_size` images per forward pass.

        Embedding one image at a time leaves the GPU mostly idle; batching is the difference
        between minutes and hours for a 500K catalog. Outputs are L2-normalized so cosine
        similarity is a plain dot product (and the index's normalization is a no-op on them).
        Default batch size adapts to the device (large on GPU, small on CPU where activations are
        big and memory-bound).
        """
        import torch
        from PIL import Image

        if batch_size is None:
            batch_size = 256 if self.device == "cuda" else 16

        paths = list(paths)
        if not paths:
            return np.empty((0, self.dim), dtype=np.float32)

        chunks = []
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            tensors = torch.stack(
                [self.preprocess(Image.open(p).convert("RGB")) for p in batch]
            ).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            chunks.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

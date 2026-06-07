"""QueryForge demo pipeline: images -> embeddings -> HNSW index + SQLite metadata -> search.

The pieces are deliberately swappable:
  * Embedder      — HistogramEmbedder (no heavy deps, for the CPU dry-run) or ClipEmbedder (real).
  * MetadataStore — SQLite now (zero-dependency local dev); swap for Postgres later, same interface.
  * pipeline      — build_catalog() / search_similar() tie the engine + metadata together.
"""
from .embedder import Embedder, HistogramEmbedder, ClipEmbedder
from .metadata import MetadataStore, Product
from .pipeline import build_catalog, search_similar

__all__ = [
    "Embedder",
    "HistogramEmbedder",
    "ClipEmbedder",
    "MetadataStore",
    "Product",
    "build_catalog",
    "search_similar",
]

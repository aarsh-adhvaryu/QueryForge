"""QueryForge demo backend (FastAPI).

Two modes, chosen at startup by the `QF_INDEX_DIR` environment variable:

  * REAL (QF_INDEX_DIR set): mmap-load a prebuilt index (`index.qfx` + `metadata.db` + `images/`
    produced by `qf_pipeline.build_real`) and embed queries with the real `ClipEmbedder`. This is
    the 500K-image reverse-image-search demo. Startup is ~1 s (mmap load) + CLIP model load.
  * MOCK (default, no env var): build a tiny synthetic catalog with `HistogramEmbedder` at import
    time, so the API is fully functional with no dataset or GPU — the A7 vertical slice.

Only the index/embedder/image source differ; the endpoints below are identical in both modes.

Run:
    cmake -S . -B build -DQF_BUILD_PYTHON=ON && cmake --build build -j
    # mock demo:
    PYTHONPATH=build/python:python uvicorn backend.app:app
    # real 500K demo:
    QF_INDEX_DIR=/teamspace/studios/this_studio/qf_data/im500k \
        PYTHONPATH=build/python:python uvicorn backend.app:app
Then open http://127.0.0.1:8000/

Endpoints (from the project proposal):
    GET  /health               index stats
    GET  /catalog              paginated product list
    GET  /search/id/{id}       products similar to an existing catalog item
    POST /search/image         products similar to an uploaded image (raw image bytes as the body)
"""
from __future__ import annotations

import os
import tempfile
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import queryforge as qf
from qf_pipeline import HistogramEmbedder, build_catalog, search_similar
from qf_pipeline.metadata import MetadataStore
from qf_pipeline.synthetic import generate_catalog

# Uploaded query images go to their own temp dir (never the served catalog dir).
_QUERYDIR = tempfile.mkdtemp(prefix="qf_query_")

_INDEX_DIR = os.environ.get("QF_INDEX_DIR")
if _INDEX_DIR:
    # --- REAL mode: load the prebuilt 500K index; embed queries with CLIP -------------------
    from qf_pipeline import ClipEmbedder

    _MODE = "real"
    _IMAGES_DIR = os.path.join(_INDEX_DIR, "images")
    _INDEX = qf.HnswIndex.load(os.path.join(_INDEX_DIR, "index.qfx"))   # mmap, ~1 s for 1.6 GB
    _STORE = MetadataStore(os.path.join(_INDEX_DIR, "metadata.db"))
    _EMBEDDER = ClipEmbedder(model_name=os.environ.get("QF_CLIP_MODEL", "ViT-L-14"),
                             pretrained=os.environ.get("QF_CLIP_PRETRAINED", "laion2b_s32b_b82k"))
    # Warm up the GPU once (first CLIP forward pass does cuDNN autotuning, ~0.4 s) so the first real
    # user query is fast, not cold.
    _warm = _STORE.get(0)
    if _warm is not None and os.path.exists(_warm.image_path):
        _EMBEDDER.embed_image(_warm.image_path)
else:
    # --- MOCK mode: build a tiny synthetic catalog at import time ----------------------------
    _MODE = "mock"
    _IMAGES_DIR = tempfile.mkdtemp(prefix="qf_backend_")
    _EMBEDDER = HistogramEmbedder(grid=4)
    _PRODUCTS = generate_catalog(_IMAGES_DIR, per_category=15)
    _INDEX, _STORE = build_catalog(_PRODUCTS, _EMBEDDER, metric=qf.Metric.Cosine)

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = FastAPI(title="QueryForge demo", version=qf.__version__)
# Serve catalog images at /images/<relpath>. Real images are sharded into subdirs, so we key the
# URL on the path relative to the images root (basename alone would collide / lose the subdir).
app.mount("/images", StaticFiles(directory=_IMAGES_DIR), name="images")


def _image_url(image_path: str):
    if not image_path:
        return None
    return f"/images/{os.path.relpath(image_path, _IMAGES_DIR)}"


def _enrich(results: list[dict]) -> list[dict]:
    """Convert pipeline results (with local image_path) into API results (with a served URL)."""
    out = []
    for r in results:
        out.append({
            "id": r["id"],
            "score": r["score"],
            "name": r["name"],
            "category": r["category"],
            "price": r["price"],
            "image_url": _image_url(r["image_path"]) if r["image_path"] else None,
        })
    return out


@app.get("/")
def index():
    path = os.path.join(_FRONTEND_DIR, "index.html")
    if not os.path.exists(path):
        return JSONResponse({"message": "QueryForge API up. Frontend not found."})
    return FileResponse(path)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": _MODE,
        "vector_count": len(_INDEX),
        "dim": _INDEX.dim,
        "top_layer": _INDEX.max_layer,
        "metric": "cosine",
        "embedder": type(_EMBEDDER).__name__,
        "version": qf.__version__,
    }


@app.get("/catalog")
def catalog(offset: int = 0, limit: int = 24):
    total = _STORE.count()
    items = []
    for pid in range(offset, min(offset + limit, total)):
        p = _STORE.get(pid)
        if p:
            items.append({
                "id": p.id, "name": p.name, "category": p.category,
                "price": p.price, "image_url": _image_url(p.image_path),
            })
    return {"total": total, "offset": offset, "limit": limit, "items": items}


@app.get("/search/id/{product_id}")
def search_by_id(product_id: int, k: int = 8, ef: int = 32):
    p = _STORE.get(product_id)
    if p is None:
        raise HTTPException(status_code=404, detail="product not found")
    t0 = time.perf_counter()
    results = search_similar(_INDEX, _STORE, _EMBEDDER, p.image_path, k=k, ef=ef)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {"query": {"id": p.id, "name": p.name, "image_url": _image_url(p.image_path)},
            "latency_ms": round(latency_ms, 3), "results": _enrich(results)}


@app.post("/search/image")
async def search_by_image(request: Request, k: int = 8, ef: int = 32):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty request body; POST raw image bytes")
    tmp = os.path.join(_QUERYDIR, f"_query_{int(time.time()*1e6)}.img")
    with open(tmp, "wb") as f:
        f.write(body)
    try:
        t0 = time.perf_counter()
        results = search_similar(_INDEX, _STORE, _EMBEDDER, tmp, k=k, ef=ef)
        latency_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:  # bad image bytes, etc.
        raise HTTPException(status_code=400, detail=f"could not process image: {e}")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return {"latency_ms": round(latency_ms, 3), "results": _enrich(results)}

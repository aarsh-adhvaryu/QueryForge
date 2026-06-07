"""QueryForge demo backend (FastAPI).

On startup it builds a small *mock* catalog (synthetic images + HistogramEmbedder) so the API is
fully functional without the real dataset. When the real CLIP pipeline and dataset arrive
(bucket B), only the embedder + catalog source change — these endpoints stay the same.

Run:
    cmake -S . -B build -DQF_BUILD_PYTHON=ON && cmake --build build -j
    PYTHONPATH=build/python:python uvicorn backend.app:app --reload
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
from qf_pipeline.synthetic import generate_catalog

# --- Build the mock catalog once at import time ------------------------------------------
_WORKDIR = tempfile.mkdtemp(prefix="qf_backend_")
_EMBEDDER = HistogramEmbedder(grid=4)
_PRODUCTS = generate_catalog(_WORKDIR, per_category=15)
_INDEX, _STORE = build_catalog(_PRODUCTS, _EMBEDDER, metric=qf.Metric.Cosine)

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = FastAPI(title="QueryForge demo", version=qf.__version__)
# Serve the synthetic product images at /images/<filename>.
app.mount("/images", StaticFiles(directory=_WORKDIR), name="images")


def _image_url(image_path: str) -> str:
    return f"/images/{os.path.basename(image_path)}"


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
        "vector_count": len(_INDEX),
        "dim": _INDEX.dim,
        "top_layer": _INDEX.max_layer,
        "metric": "cosine",
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
def search_by_id(product_id: int, k: int = 8, ef: int = 64):
    p = _STORE.get(product_id)
    if p is None:
        raise HTTPException(status_code=404, detail="product not found")
    t0 = time.perf_counter()
    results = search_similar(_INDEX, _STORE, _EMBEDDER, p.image_path, k=k, ef=ef)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {"query": {"id": p.id, "name": p.name, "image_url": _image_url(p.image_path)},
            "latency_ms": round(latency_ms, 3), "results": _enrich(results)}


@app.post("/search/image")
async def search_by_image(request: Request, k: int = 8, ef: int = 64):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty request body; POST raw image bytes")
    tmp = os.path.join(_WORKDIR, f"_query_{int(time.time()*1e6)}.img")
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

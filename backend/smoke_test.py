"""Backend smoke test — exercises every endpoint via FastAPI's in-process TestClient.

Run (from the repo root):
    PYTHONPATH=build/python:python python -m backend.smoke_test
"""
from __future__ import annotations

import os
import tempfile

from fastapi.testclient import TestClient

from backend.app import app
from qf_pipeline.synthetic import CATEGORIES, make_image

client = TestClient(app)


def main() -> int:
    # /health
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["vector_count"] == 60, h
    print("health:", h)

    # /catalog
    c = client.get("/catalog?offset=0&limit=10").json()
    assert c["total"] == 60 and len(c["items"]) == 10, c
    assert c["items"][0]["image_url"].startswith("/images/")

    # /search/id/{id}
    s = client.get("/search/id/0?k=8").json()
    assert len(s["results"]) == 8, s
    assert "latency_ms" in s

    # /search/image — upload a fresh red image; expect mostly red_shoe results.
    tmp = os.path.join(tempfile.mkdtemp(), "q.png")
    make_image(tmp, CATEGORIES["red_shoe"], seed=7777)
    with open(tmp, "rb") as f:
        body = f.read()
    r = client.post("/search/image?k=8", content=body).json()
    assert len(r["results"]) == 8, r
    red = sum(1 for x in r["results"] if x["category"] == "red_shoe")
    assert red >= 6, f"expected mostly red_shoe, got {red}/8"
    print(f"search/image: {red}/8 red_shoe, latency {r['latency_ms']} ms")

    # 404 path
    assert client.get("/search/id/99999").status_code == 404

    # Frontend is served at /
    root = client.get("/")
    assert root.status_code == 200 and "QueryForge" in root.text, "frontend not served"

    print("Backend smoke test OK ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# backend/

FastAPI service (stage A7) that ties the demo together. On startup it builds a small **mock**
catalog (synthetic images + `HistogramEmbedder`) so the API is fully functional without the real
dataset. Swapping in real CLIP embeddings + the real catalog (bucket B) leaves these endpoints
unchanged.

## Run

```bash
# 1. Build the Python module (from the repo root)
cmake -S . -B build -DQF_BUILD_PYTHON=ON && cmake --build build -j

# 2. Install web deps
pip install -r backend/requirements.txt

# 3. Launch the server
PYTHONPATH=build/python:python uvicorn backend.app:app --reload
# open http://127.0.0.1:8000/
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health` | index stats (vector count, dim, metric, top layer) |
| GET  | `/catalog?offset=&limit=` | paginated product list |
| GET  | `/search/id/{product_id}?k=&ef=` | products similar to an existing catalog item |
| POST | `/search/image?k=&ef=` | products similar to an uploaded image (raw image bytes as the body) |
| GET  | `/images/{file}` | the synthetic product images |
| GET  | `/` | the React frontend |

## Test

`backend/smoke_test.py` exercises every endpoint in-process (FastAPI `TestClient`). It runs as the
`python_api` CTest case when fastapi+httpx are installed:

```bash
PYTHONPATH=build/python:python python -m backend.smoke_test
```

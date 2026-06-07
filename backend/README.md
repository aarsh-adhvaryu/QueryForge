# backend/

FastAPI service (stage A7) that ties the demo together.

Planned endpoints (from the project proposal):
- `POST /search/image` — accept an image, embed it (CLIP), query QueryForge, return similar
  products with metadata.
- `GET /search/id/{product_id}` — find products similar to an existing catalog item.
- `GET /catalog` — paginated product list for browsing.
- `GET /health` — system health + index stats (vector count, index size, avg query time).

Scaffolded first against mock/tiny data so the wiring is proven before the real dataset exists.

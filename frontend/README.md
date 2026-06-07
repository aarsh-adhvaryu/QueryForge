# frontend/

React demo UI (stage A7).

For the scaffold it's a **single self-contained `index.html`** that loads React from a CDN (no build
step), so the backend can serve it directly and the whole demo runs with zero `npm install`. A
production version would migrate this to a Vite + React project under `frontend/src/`.

## Features

- **Image upload** — pick an image; it's POSTed to `/search/image` and similar products are shown.
- **Catalog browse** — click any product to find visually similar items (`/search/id/{id}`).
- **Results grid** — image, name, category, price, and similarity score.
- **Performance** — the live query latency (ms) from the API is displayed.

## Run

It's served by the backend at `/` — just start the backend (see `../backend/README.md`) and open
`http://127.0.0.1:8000/`. The page talks to the API on the same origin.

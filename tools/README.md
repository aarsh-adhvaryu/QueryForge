# tools/

Helper programs for developing and validating the engine **without** the real image dataset.

Planned (later stages):
- **Synthetic data generator** — produces random vectors (and optional clustered vectors) so the
  graph and recall harness can be tested on the Studio now.
- **Recall harness** — runs HNSW search and brute-force KNN on the same data, then reports
  Recall@10 and nodes-visited so we can validate accuracy and fill in `docs/BASELINES.md`.

"""Smoke/demo for the QueryForge Python bindings.

Build the module first (from the repo root):
    cmake -S . -B build -DQF_BUILD_PYTHON=ON
    cmake --build build -j
Then run (point Python at the built .so):
    PYTHONPATH=build/python python python/example.py
"""
import numpy as np
import queryforge as qf


def main() -> None:
    print("queryforge version:", qf.__version__)

    rng = np.random.default_rng(0)
    n, dim = 5000, 64
    data = rng.standard_normal((n, dim)).astype(np.float32)

    # Build an index and insert all vectors in one batch call.
    index = qf.HnswIndex(dim=dim, M=16, ef_construction=200, metric=qf.Metric.L2)
    index.add_batch(data)
    print("built:", repr(index), "top layer:", index.max_layer)

    # Search: returns (ids, distances) as NumPy arrays.
    query = data[0]
    ids, dists = index.search(query, k=5, ef=64)
    print("nearest ids :", ids)
    print("distances   :", np.round(dists, 4))
    assert ids[0] == 0, "the vector itself should be its own nearest neighbor"

    # Save and reload (mmap) — results must be identical.
    index.save("/tmp/qf_py_demo.qfx")
    reloaded = qf.HnswIndex.load("/tmp/qf_py_demo.qfx")
    ids2, _ = reloaded.search(query, k=5, ef=64)
    assert list(ids) == list(ids2), "reloaded index must match"
    print("save/load round-trip: OK")
    print("\nAll good ✔")


if __name__ == "__main__":
    main()

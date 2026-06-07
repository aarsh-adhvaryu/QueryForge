"""SQLite-backed product metadata store.

QueryForge holds the vectors and returns nearest *ids*; this store maps those ids back to
human-readable product info (name, category, price, image path). SQLite is chosen for local dev
because it's in the Python stdlib (zero dependency) and ACID. The same small interface can be
re-implemented on PostgreSQL later without touching the rest of the pipeline.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class Product:
    id: int
    name: str
    category: str
    price: float
    image_path: str


class MetadataStore:
    def __init__(self, path: str = ":memory:"):
        # check_same_thread=False: the index is built once then served read-only across the web
        # server's worker threads (our "static build + concurrent reads" model). SQLite serializes
        # access internally, which is fine for this read-mostly demo.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS products(
                   id         INTEGER PRIMARY KEY,
                   name       TEXT,
                   category   TEXT,
                   price      REAL,
                   image_path TEXT
               )"""
        )

    def add(self, p: Product) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO products(id, name, category, price, image_path) VALUES (?,?,?,?,?)",
            (p.id, p.name, p.category, p.price, p.image_path),
        )

    def add_many(self, products) -> None:
        for p in products:
            self.add(p)

    def commit(self) -> None:
        self.conn.commit()

    def get(self, id: int) -> Product | None:
        row = self.conn.execute(
            "SELECT id, name, category, price, image_path FROM products WHERE id=?", (id,)
        ).fetchone()
        return Product(*row) if row else None

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM products").fetchone()[0])

    def close(self) -> None:
        self.conn.close()

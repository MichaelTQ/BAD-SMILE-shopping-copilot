"""Offline catalog index: SQLite FTS5 for recall plus a metadata table for rerank.

Nothing here touches the network. The index is built once per process from
``data/catalog.jsonl`` and shared by every session.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .text import idf, normalize, tokens

# FTS5 bm25 column weights: parent_asin is unindexed, title matters most.
BM25_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)

# Category tokens true of every row in this catalog, so they carry no
# discriminative signal and are dropped before category matching.
GENERIC_CATEGORY_TOKENS = frozenset({"clothing", "shoes", "jewelry"})


@dataclass(frozen=True)
class Document:
    rowid: int
    parent_asin: str
    title: str
    category_text: str
    body: str
    price: float | None
    average_rating: float
    rating_number: int


def _price(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


class CatalogIndex:
    """FTS5 recall + in-memory metadata, built from a JSONL catalog."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.document_frequency: Counter[str] = Counter()
        self.total_documents = 0
        self._build()

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        cursor.execute(
            "CREATE TABLE meta ("
            "rowid INTEGER PRIMARY KEY, parent_asin TEXT, title TEXT, category_text TEXT, "
            "body TEXT, price REAL, average_rating REAL, rating_number INTEGER)"
        )
        fts_batch: list[tuple] = []
        meta_batch: list[tuple] = []
        rowid = 0
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                rowid += 1
                title = normalize(product.get("title"))
                categories = normalize(product.get("categories"))
                features = normalize(product.get("features"))
                details = normalize(product.get("details"))
                store = normalize(product.get("store"))
                description = normalize(product.get("description"))
                body = " ".join((title, categories, features, details, store, description))
                fts_batch.append(
                    (rowid, str(product["parent_asin"]), title, categories,
                     features, details, store, description)
                )
                meta_batch.append(
                    (rowid, str(product["parent_asin"]), title, categories, body,
                     _price(product.get("price")),
                     float(product.get("average_rating") or 0.0),
                     int(product.get("rating_number") or 0))
                )
                self.document_frequency.update(set(tokens(body)))
                self.total_documents += 1
                if len(fts_batch) >= 2000:
                    self._flush(cursor, fts_batch, meta_batch)
        self._flush(cursor, fts_batch, meta_batch)
        self.connection.commit()

    @staticmethod
    def _flush(cursor: sqlite3.Cursor, fts_batch: list, meta_batch: list) -> None:
        if not fts_batch:
            return
        cursor.executemany(
            "INSERT INTO products(rowid, parent_asin, title, categories, features, details, "
            "store, description) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            fts_batch,
        )
        cursor.executemany(
            "INSERT INTO meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)", meta_batch
        )
        fts_batch.clear()
        meta_batch.clear()

    # ----------------------------------------------------------------- lookup

    def idf(self, term: str) -> float:
        return idf(self.document_frequency.get(term, 0), self.total_documents)

    def search(self, terms: list[str], limit: int) -> list[tuple[int, float]]:
        """BM25 recall for an OR-of-terms query; returns (rowid, -bm25) pairs.

        SQLite's ``bm25()`` returns a negative number where more negative means
        a better match, so the caller gets a higher-is-better score.
        """
        unique = list(dict.fromkeys(term for term in terms if term))
        if not unique:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique[:60])
        rows = self.connection.execute(
            "SELECT rowid, bm25(products, ?, ?, ?, ?, ?, ?, ?) AS score FROM products "
            "WHERE products MATCH ? ORDER BY score LIMIT ?",
            (*BM25_WEIGHTS, expression, limit),
        ).fetchall()
        return [(int(row[0]), -float(row[1])) for row in rows]

    def documents(self, rowids: list[int]) -> dict[int, Document]:
        if not rowids:
            return {}
        placeholders = ",".join("?" * len(rowids))
        rows = self.connection.execute(
            f"SELECT rowid, parent_asin, title, category_text, body, price, "
            f"average_rating, rating_number FROM meta WHERE rowid IN ({placeholders})",
            rowids,
        ).fetchall()
        return {int(row[0]): Document(int(row[0]), *row[1:]) for row in rows}

    @lru_cache(maxsize=40000)
    def _token_set(self, body: str) -> frozenset[str]:
        return frozenset(tokens(body))

    def token_set(self, document: Document) -> frozenset[str]:
        return self._token_set(document.body)

    @lru_cache(maxsize=40000)
    def _token_text(self, body: str) -> str:
        return " ".join(tokens(body))

    def token_text(self, document: Document) -> str:
        """Body reduced to a normalized token stream, for phrase containment."""
        return self._token_text(document.body)

    def category_tokens(self, document: Document) -> frozenset[str]:
        found = frozenset(tokens(document.category_text)) - GENERIC_CATEGORY_TOKENS
        return found

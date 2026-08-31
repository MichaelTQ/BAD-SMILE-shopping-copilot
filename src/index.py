"""Offline catalog index: SQLite FTS5 for recall plus a metadata table for rerank.

Nothing here touches the network. The index is built once per process from
``data/catalog.jsonl`` and shared by every session.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .text import idf, normalize, tokens

# Popularity is normalised by a catalog percentile rather than a hard-coded
# ceiling, so the scale adapts if the catalog changes. The 99th percentile wins
# empirically: target products are themselves very popular (median
# rating_number 7078 against a catalog median of 12), so a lower cut-off
# compresses exactly the region where targets have to be told apart.
POPULARITY_PERCENTILE = 0.99

# Per-document token caches are bounded: the private run is 800 sessions and an
# unbounded cache would grow the resident set past a typical memory limit.
TOKEN_CACHE_SIZE = 8192

# FTS5 bm25 column weights: parent_asin is unindexed, title matters most.
BM25_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)

# The first category element is the catalog-wide root ("Clothing, Shoes &
# Jewelry" for 49990 of 50000 rows) and carries no discriminative signal, so it
# is dropped before category matching. Only the first element is removed:
# "Clothing", "Shoes" and "Jewelry" also appear as genuine deeper levels
# (20523 / 11810 / 5127 times), where they are the main division of the
# catalog. Filtering those tokens globally halved category_score for 23.9% of
# products.


@dataclass(frozen=True)
class Document:
    rowid: int
    parent_asin: str
    title: str
    #: Title with original casing, for customer-facing text only.
    display_title: str
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
        # One shared in-memory connection; serialise access in case the harness
        # drives sessions from more than one thread.
        self._lock = threading.Lock()
        self.document_frequency: Counter[str] = Counter()
        self.total_documents = 0
        #: log1p of a high rating_number percentile; the popularity denominator.
        self.popularity_scale = 1.0
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
            "rowid INTEGER PRIMARY KEY, parent_asin TEXT, title TEXT, display_title TEXT, "
            "category_text TEXT, body TEXT, price REAL, average_rating REAL, "
            "rating_number INTEGER)"
        )
        fts_batch: list[tuple] = []
        meta_batch: list[tuple] = []
        rating_counts: list[int] = []
        rowid = 0
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                rowid += 1
                title = normalize(product.get("title"))
                raw_categories = product.get("categories") or []
                categories = normalize(raw_categories)
                deep_categories = normalize(
                    raw_categories[1:] if len(raw_categories) > 1 else raw_categories
                )
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
                    (rowid, str(product["parent_asin"]), title,
                     str(product.get("title") or ""), deep_categories, body,
                     _price(product.get("price")),
                     float(product.get("average_rating") or 0.0),
                     int(product.get("rating_number") or 0))
                )
                rating_counts.append(int(product.get("rating_number") or 0))
                self.document_frequency.update(set(tokens(body)))
                self.total_documents += 1
                if len(fts_batch) >= 2000:
                    self._flush(cursor, fts_batch, meta_batch)
        self._flush(cursor, fts_batch, meta_batch)
        self.connection.commit()
        self.popularity_scale = self._percentile_scale(rating_counts)

    @staticmethod
    def _percentile_scale(rating_counts: list[int]) -> float:
        if not rating_counts:
            return 1.0
        rating_counts.sort()
        index = max(int(len(rating_counts) * POPULARITY_PERCENTILE) - 1, 0)
        return max(math.log1p(rating_counts[index]), 1e-9)

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
            "INSERT INTO meta VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", meta_batch
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
        with self._lock:
            rows = self.connection.execute(
                "SELECT rowid, bm25(products, ?, ?, ?, ?, ?, ?, ?) AS score FROM products "
                "WHERE products MATCH ? ORDER BY score LIMIT ?",
                (*BM25_WEIGHTS, expression, limit),
            ).fetchall()
        return [(int(row[0]), -float(row[1])) for row in rows]

    def search_all(self, terms: list[str], limit: int) -> list[tuple[int, float]]:
        """BM25 recall requiring *every* term (AND), for a high-precision pool."""
        unique = list(dict.fromkeys(term for term in terms if term))
        if not unique:
            return []
        expression = " AND ".join(f'"{term}"' for term in unique[:12])
        with self._lock:
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
        with self._lock:
            rows = self.connection.execute(
                f"SELECT rowid, parent_asin, title, display_title, category_text, body, "
                f"price, average_rating, rating_number FROM meta "
                f"WHERE rowid IN ({placeholders})",
                rowids,
            ).fetchall()
        return {int(row[0]): Document(int(row[0]), *row[1:]) for row in rows}

    @lru_cache(maxsize=TOKEN_CACHE_SIZE)
    def _token_set(self, body: str) -> frozenset[str]:
        return frozenset(tokens(body))

    def token_set(self, document: Document) -> frozenset[str]:
        return self._token_set(document.body)

    @lru_cache(maxsize=TOKEN_CACHE_SIZE)
    def _token_text(self, body: str) -> str:
        return " ".join(tokens(body))

    def token_text(self, document: Document) -> str:
        """Body reduced to a normalized token stream, for phrase containment."""
        return self._token_text(document.body)

    def category_tokens(self, document: Document) -> frozenset[str]:
        return frozenset(tokens(document.category_text))

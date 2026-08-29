"""Two-stage retrieval: FTS5 BM25 recall, then local constraint-aware rerank."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .index import CatalogIndex, Document
from .parsing import attribute_values
from .state import SessionState
from .text import query_tokens

RECALL_POOL = 200          # candidates pulled from BM25 before reranking
CATEGORY_POOL = 100        # extra recall from a category-only query

# Query-side term weights.
CATEGORY_WEIGHT = 3.0
HARD_WEIGHT = 2.5
SOFT_WEIGHT = 1.0
VALUE_BONUS = 1.5          # extra weight for a concrete color/material/size
OVERRIDE_DECAY = 0.6       # multiplier for constraints superseded by an override

# Rerank feature weights.
W_BM25 = 1.0
W_COVERAGE = 3.0
W_PHRASE = 1.2
W_CATEGORY = 2.2
W_POPULARITY = 1.2
W_BUDGET = 0.5


@dataclass
class Scored:
    document: Document
    score: float
    parts: dict[str, float]


def build_query(state: SessionState) -> tuple[dict[str, float], list[str], list[str]]:
    """Turn the accumulated session state into weighted query terms.

    Returns ``(term_weights, category_terms, constraint_phrases)``.
    """
    weights: dict[str, float] = {}

    def add(text: str, weight: float) -> None:
        for term in query_tokens(text):
            weights[term] = weights.get(term, 0.0) + weight

    category_terms = query_tokens(state.category or "")
    add(state.category or "", CATEGORY_WEIGHT)

    phrases: list[str] = []
    for constraint in state.constraints:
        weight = HARD_WEIGHT if constraint.hard else SOFT_WEIGHT
        if constraint.stale:
            weight *= OVERRIDE_DECAY
        add(constraint.value, weight)
        terms = query_tokens(constraint.value)
        if len(terms) >= 3:
            phrases.append(" ".join(terms))

    for values in attribute_values(state.constraints).values():
        for value in values:
            add(value, VALUE_BONUS)

    return weights, category_terms, phrases


def _popularity(document: Document) -> float:
    """Mild prior: well-reviewed, well-rated products are likelier targets."""
    volume = math.log1p(max(document.rating_number, 0)) / math.log1p(100000.0)
    quality = max(document.average_rating, 0.0) / 5.0
    return volume * quality


def _budget_fit(document: Document, budget: float | None) -> float:
    if budget is None:
        return 0.0
    if document.price is None:
        return 0.0          # unknown price is neither rewarded nor punished
    if document.price <= budget * 1.15:
        return 1.0
    return -min(1.0, (document.price - budget) / max(budget, 1.0))


def rank(index: CatalogIndex, state: SessionState, top_k: int) -> list[Scored]:
    """Recall Top ~200 with BM25, then rerank locally and return Top ``top_k``."""
    weights, category_terms, phrases = build_query(state)
    if not weights:
        return []

    ordered = sorted(weights, key=lambda term: -weights[term] * index.idf(term))
    pool = dict(index.search(ordered, RECALL_POOL))
    if category_terms:
        for rowid, _ in index.search(category_terms, CATEGORY_POOL):
            pool.setdefault(rowid, 0.0)
    if not pool:
        return []

    documents = index.documents(list(pool))
    best_bm25 = max(pool.values()) or 1.0
    total_mass = sum(weight * index.idf(term) for term, weight in weights.items()) or 1.0
    category_set = set(category_terms)

    scored: list[Scored] = []
    for rowid, bm25 in pool.items():
        document = documents.get(rowid)
        if document is None:
            continue
        doc_tokens = index.token_set(document)
        coverage = sum(
            weight * index.idf(term)
            for term, weight in weights.items()
            if term in doc_tokens
        ) / total_mass
        if category_set:
            category_score = len(category_set & index.category_tokens(document)) / len(category_set)
        else:
            category_score = 0.0
        phrase_score = 0.0
        if phrases:
            body = index.token_text(document)
            phrase_score = sum(1.0 for phrase in phrases if phrase in body) / len(phrases)
        parts = {
            "bm25": W_BM25 * (bm25 / best_bm25),
            "coverage": W_COVERAGE * coverage,
            "phrase": W_PHRASE * phrase_score,
            "category": W_CATEGORY * category_score,
            "popularity": W_POPULARITY * _popularity(document),
            "budget": W_BUDGET * _budget_fit(document, state.budget),
        }
        scored.append(Scored(document, sum(parts.values()), parts))

    scored.sort(key=lambda item: (-item.score, item.document.parent_asin))
    return scored[:top_k]


def bm25_only(index: CatalogIndex, message: str, top_k: int) -> list[Scored]:
    """Fallback path used when reranking raises: plain BM25 on one message."""
    hits = index.search(query_tokens(message), top_k)
    documents = index.documents([rowid for rowid, _ in hits])
    return [
        Scored(documents[rowid], score, {"bm25": score})
        for rowid, score in hits
        if rowid in documents
    ]

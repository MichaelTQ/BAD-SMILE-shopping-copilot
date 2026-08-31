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
# A third route: every category term must appear. The resulting pool is small
# and precise, which surfaces products whose title omits the category keywords
# and are therefore buried in the OR routes.
STRICT_CATEGORY_POOL = 800      # cap on the strict-category route
# When the strict pool comes back below the cap the category is narrow enough to
# take whole; when it saturates the cap the category is broad and only its head
# is worth adding, otherwise the extra candidates dilute the ranking.
STRICT_CATEGORY_HEAD = 200
# Widening recall dilutes the ranking, so only pay that cost once the dialogue
# has stopped producing information — i.e. when the current Top 10 is evidently
# wrong and there is nothing left to lose.
STRICT_ON_STALL_ONLY = True
# The primary query is fetched deeper than the pool so that candidates recalled
# by a secondary route can be given their real BM25 score. FTS5 has to rank every
# match anyway, so a larger LIMIT is nearly free (200 -> 3000 costs ~4 ms).
SCORE_LOOKUP_POOL = 3000

# Query-side term weights.
CATEGORY_WEIGHT = 2.5
HARD_WEIGHT = 2.5
SOFT_WEIGHT = 1.0
VALUE_BONUS = 1.5          # extra weight for a concrete color/material/size
OVERRIDE_DECAY = 0.7       # multiplier for constraints superseded by an override
PHRASE_MIN_TOKENS = 3      # shortest constraint that counts as a phrase
# How a constraint becomes phrase units. "full" requires the entire constraint
# to appear contiguously, which gets fragile fast: 19.8% of real constraints are
# longer than 5 tokens (up to 30), and one reordered word zeroes the feature.
PHRASE_MODE = "ngram"       # "full" | "ngram" | "truncate"
PHRASE_NGRAM = 3           # window size for "ngram"
PHRASE_MAX_TOKENS = 5      # keep this many leading tokens for "truncate"

# Rerank feature weights.
W_BM25 = 1.0
W_COVERAGE = 3.0
W_PHRASE = 1.2
W_CATEGORY = 2.2
W_POPULARITY = 1.2
W_BUDGET = 0.5

# Exploration: only applied on turns that added no information, so a target
# surfaced early (and not yet scoreable, as in intent override) is never
# rotated away while the conversation is still productive.
# The anonymized profile's preference_tags, as a rerank tie-breaker. Disabled:
# the tags do correlate with the target (1.92x lift against a random product),
# but that correlation is almost entirely explained by the candidate pool. Lift
# decays to 1.81x against the pool and 1.17x within the Top 10 — i.e. it carries
# no signal exactly where ranking is decided. Any positive weight measured
# monotonically worse; a negative weight looked better on the full set but
# reversed sign under split-half validation.
W_PROFILE = 0.0
W_SEEN_PENALTY = 0.5
MAX_STALL_PRESSURE = 3


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

    phrases: list[list[str]] = []
    for constraint in state.constraints:
        weight = HARD_WEIGHT if constraint.hard else SOFT_WEIGHT
        if constraint.stale:
            weight *= OVERRIDE_DECAY
        add(constraint.value, weight)
        units = _phrase_units(query_tokens(constraint.value))
        if units:
            phrases.append(units)

    for values in attribute_values(state.constraints).values():
        for value in values:
            add(value, VALUE_BONUS)

    return weights, category_terms, phrases


def _phrase_units(terms: list[str]) -> list[str]:
    """Split one constraint into the phrase units matched against a product."""
    if len(terms) < PHRASE_MIN_TOKENS:
        return []
    if PHRASE_MODE == "truncate":
        return [" ".join(terms[:PHRASE_MAX_TOKENS])]
    if PHRASE_MODE == "ngram" and len(terms) > PHRASE_NGRAM:
        return [
            " ".join(terms[start:start + PHRASE_NGRAM])
            for start in range(len(terms) - PHRASE_NGRAM + 1)
        ]
    return [" ".join(terms)]


def _popularity(document: Document, scale: float) -> float:
    """Prior: well-reviewed, well-rated products are likelier targets.

    ``scale`` is log1p of a high catalog percentile of rating_number. It is
    deliberately not clipped at 1.0: 81% of target products sit above the 95th
    percentile, so clipping there collapses most targets onto the same value as
    every other bestseller and costs ~0.043 of the score.
    """
    volume = math.log1p(max(document.rating_number, 0)) / scale
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
    ranked = index.search(ordered, max(RECALL_POOL, SCORE_LOOKUP_POOL))
    lookup = dict(ranked)
    pool = dict(ranked[:RECALL_POOL])
    if category_terms:
        extra = [rowid for rowid, _ in index.search(category_terms, CATEGORY_POOL)]
        if (
            STRICT_CATEGORY_POOL
            and len(category_terms) > 1
            and (state.stalled_turns > 0 or not STRICT_ON_STALL_ONLY)
        ):
            strict = index.search_all(category_terms, STRICT_CATEGORY_POOL)
            if len(strict) >= STRICT_CATEGORY_POOL:
                strict = strict[:STRICT_CATEGORY_HEAD]
            extra.extend(rowid for rowid, _ in strict)
        # Give secondary-route candidates their real BM25 score. Defaulting them
        # to 0.0 would apply a full W_BM25 penalty for a measurement that was
        # never taken, which no other feature can offset.
        for rowid in extra:
            pool.setdefault(rowid, lookup.get(rowid, 0.0))
    if not pool:
        return []

    stall = min(state.stalled_turns, MAX_STALL_PRESSURE)
    raw_tags = state.user_profile.get("preference_tags")
    profile_tags = [
        term
        for tag in (raw_tags if isinstance(raw_tags, list) else ())
        for term in query_tokens(str(tag))
    ]
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
            # Each constraint contributes its own hit ratio, then constraints are
            # averaged, so one very long constraint cannot dominate the feature.
            phrase_score = sum(
                sum(1.0 for unit in units if unit in body) / len(units)
                for units in phrases
            ) / len(phrases)
        parts = {
            "bm25": W_BM25 * (bm25 / best_bm25),
            "coverage": W_COVERAGE * coverage,
            "phrase": W_PHRASE * phrase_score,
            "category": W_CATEGORY * category_score,
            "popularity": W_POPULARITY * _popularity(document, index.popularity_scale),
            "budget": W_BUDGET * _budget_fit(document, state.budget),
            "profile": W_PROFILE * (
                sum(1 for term in profile_tags if term in doc_tokens) / len(profile_tags)
                if profile_tags else 0.0
            ),
            "seen": -stall * W_SEEN_PENALTY * min(state.recommended.get(document.parent_asin, 0), 3),
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

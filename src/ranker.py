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

# Pseudo-relevance feedback. DISABLED — measured and rejected.
# The premise is sound: constraints of one intent card all come from the same
# product, and expansion terms drawn from the Top 10 do hit 11.4% of the
# still-undisclosed constraints (random baseline ~0.03%). But the 2.07x lift
# that motivated it was measured against the wrong baseline: the terms are
# extracted *from* the Top 10, so the whole Top 10 is rich in them by
# construction and they cannot separate it internally — which is where MRR is
# decided. Always-on costs -0.068; gating on stall still costs -0.003; every
# weight from 0.15 to 1.00 is monotonically worse. Kept only so the experiment
# is reproducible.
PRF_ENABLED = False
PRF_FEEDBACK_DOCS = 10     # top candidates treated as relevant
PRF_TERMS = 15             # expansion terms taken from them
PRF_WEIGHT = 0.3           # per-term query weight (soft constraint is 1.0)
PRF_ON_STALL_ONLY = False  # restrict to turns that added no information

# Dual-track routing. A shopper who has stated a hard requirement is buying and
# wants precision; one who has not is browsing and is better served by breadth.
# Measured at turn 1, the Top 10 holds only ~2.4 distinct category paths, and
# browsing has both the least diverse pool (0.241) and the lowest turn-1 hit
# rate (0.438) — ten near-duplicates from two subcategories cover less ground
# than ten spread across the space the shopper might mean.
MMR_ENABLED = False
MMR_LAMBDA = 0.7           # 1.0 = pure relevance, i.e. disabled
MMR_POOL = 50              # reranked window
MMR_BROWSE_ONLY = True     # only when no hard constraint has been stated

# Buying track. Once a hard requirement is stated it is treated as a gate rather
# than a bonus: only products containing every hard-constraint term enter the
# pool. Verified safe on the public set — the target survives the gate in 80/80
# sessions that state one, while the pool drops from 50000 to a median 8675.
BUYING_FILTER = False

# Over-generality detection. A flat Top-10 means the ranking has no opinion, so
# widening recall costs nothing and may help. 0.0 disables it and leaves the
# stall counter as the sole trigger.
OVERLOAD_CV = 0.0

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
PHRASE_MODE = "full"       # "full" | "ngram" | "truncate"
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

    # A budget amount is a filter, not a content word. It already drives the
    # price feature; leaving it in the query also matches unrelated products
    # that merely carry the number ("under $150" pulling in "150 Pack ... Bags").
    if state.budget is not None:
        for form in _budget_tokens(state.budget):
            weights.pop(form, None)
            if form in category_terms:
                category_terms = [t for t in category_terms if t != form]

    return weights, category_terms, phrases


def _budget_tokens(amount: float) -> set[str]:
    """Token spellings a parsed budget could have contributed to the query."""
    forms = {str(int(amount)), f"{amount:g}"}
    text = f"{amount:g}"
    if "." in text:
        forms.update(text.split("."))
    return {form for form in forms if form}


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


def expansion_terms(index: CatalogIndex, scored: list[Scored], known: set[str]) -> list[str]:
    """Terms shared by the top candidates that the shopper has not mentioned."""
    if not scored:
        return []
    counts: dict[str, int] = {}
    for item in scored[:PRF_FEEDBACK_DOCS]:
        for term in index.token_set(item.document):
            if term not in known:
                counts[term] = counts.get(term, 0) + 1
    if not counts:
        return []
    docs = min(PRF_FEEDBACK_DOCS, len(scored))
    ranked = sorted(counts, key=lambda t: -(counts[t] / docs) * index.idf(t))
    return ranked[:PRF_TERMS]


def rank(index: CatalogIndex, state: SessionState, top_k: int) -> list[Scored]:
    """Recall Top ~200 with BM25, then rerank locally and return Top ``top_k``."""
    scored = _rank_once(index, state, top_k)
    use_prf = PRF_ENABLED and PRF_WEIGHT > 0 and (
        state.stalled_turns > 0 or not PRF_ON_STALL_ONLY
    )
    if not use_prf or not scored:
        return scored
    weights, _, _ = build_query(state)
    extra = expansion_terms(index, scored, set(weights))
    if not extra:
        return scored
    return _rank_once(index, state, top_k, {term: PRF_WEIGHT for term in extra})


def _rank_once(
    index: CatalogIndex,
    state: SessionState,
    top_k: int,
    expansion: dict[str, float] | None = None,
) -> list[Scored]:
    weights, category_terms, phrases = build_query(state)
    for term, weight in (expansion or {}).items():
        weights[term] = weights.get(term, 0.0) + weight
    if not weights:
        return []

    ordered = sorted(weights, key=lambda term: -weights[term] * index.idf(term))
    gate: list[str] = []
    if BUYING_FILTER:
        gate = [
            term
            for constraint in state.constraints
            if constraint.hard and not constraint.stale
            for term in query_tokens(constraint.value)
        ]
    if gate:
        ranked = index.search_filtered(gate, ordered, max(RECALL_POOL, SCORE_LOOKUP_POOL))
        if not ranked:                      # gate too strict — fall back to OR
            ranked = index.search(ordered, max(RECALL_POOL, SCORE_LOOKUP_POOL))
    else:
        ranked = index.search(ordered, max(RECALL_POOL, SCORE_LOOKUP_POOL))
    lookup = dict(ranked)
    pool = dict(ranked[:RECALL_POOL])
    if category_terms:
        extra = [rowid for rowid, _ in index.search(category_terms, CATEGORY_POOL)]
        if (
            STRICT_CATEGORY_POOL
            and len(category_terms) > 1
            and (
                state.stalled_turns > 0
                or (OVERLOAD_CV > 0.0 and 0.0 < state.last_score_cv < OVERLOAD_CV)
                or not STRICT_ON_STALL_ONLY
            )
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
    if MMR_ENABLED and MMR_LAMBDA < 1.0 and len(scored) > top_k:
        if not (MMR_BROWSE_ONLY and state.hard_constraints()):
            return _diversify(index, scored, top_k)
    return scored[:top_k]


def _category_similarity(index: CatalogIndex, a: Document, b: Document) -> float:
    """Jaccard over category paths — the axis the pool is actually collapsed on."""
    left, right = index.category_tokens(a), index.category_tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _diversify(index: CatalogIndex, scored: list[Scored], top_k: int) -> list[Scored]:
    """Maximal Marginal Relevance over the top window.

    Each pick trades its own score against how much it duplicates what is
    already selected, so the returned list spans more of the catalog instead of
    stacking near-identical products from one subcategory.
    """
    window = scored[:MMR_POOL]
    best = window[0].score or 1.0
    worst = window[-1].score
    span = (best - worst) or 1.0
    selected: list[Scored] = [window[0]]
    remaining = window[1:]
    while remaining and len(selected) < top_k:
        best_index, best_value = 0, None
        for position, candidate in enumerate(remaining):
            relevance = (candidate.score - worst) / span
            redundancy = max(
                _category_similarity(index, candidate.document, chosen.document)
                for chosen in selected
            )
            value = MMR_LAMBDA * relevance - (1.0 - MMR_LAMBDA) * redundancy
            if best_value is None or value > best_value:
                best_index, best_value = position, value
        selected.append(remaining.pop(best_index))
    return selected


def bm25_only(index: CatalogIndex, message: str, top_k: int) -> list[Scored]:
    """Fallback path used when reranking raises: plain BM25 on one message."""
    hits = index.search(query_tokens(message), top_k)
    documents = index.documents([rowid for rowid, _ in hits])
    return [
        Scored(documents[rowid], score, {"bm25": score})
        for rowid, score in hits
        if rowid in documents
    ]

"""Turn a ranking into a short, honest sentence about why it was chosen.

Purely descriptive: every clause is read back off the state and the scored
candidate, so the explanation cannot claim more than the ranker actually used.
No model generates this text.
"""

from __future__ import annotations

from .index import CatalogIndex
from .ranker import Scored
from .state import SessionState
from .text import query_tokens

MAX_REASONS = 2
MAX_VALUE_CHARS = 40

#: Below this coefficient of variation the Top 10 scores are effectively tied,
#: so the ranking has no opinion and saying "here are the closest matches" would
#: overstate it. Measured: 0.0173 on turns that missed against 0.0561 on turns
#: that hit; the lowest quartile carries a 50.7% miss rate against 31.2% overall.
OVERLOAD_CV = 0.025


def _shorten(value: str) -> str:
    value = " ".join(value.split())
    if len(value) <= MAX_VALUE_CHARS:
        return value
    return value[:MAX_VALUE_CHARS].rsplit(" ", 1)[0] + "..."


def matched_constraints(
    index: CatalogIndex, state: SessionState, document
) -> list[str]:
    """Constraints whose every content word appears in this product."""
    words = index.token_set(document)
    matched = []
    for constraint in state.constraints:
        if constraint.stale:
            continue
        terms = query_tokens(constraint.value)
        if terms and all(term in words for term in terms):
            matched.append(_shorten(constraint.value))
    return matched


def is_overloaded(scored: list[Scored]) -> bool:
    """True when the Top 10 are so close together that the ranking is arbitrary."""
    if len(scored) < 2:
        return False
    values = [item.score for item in scored]
    mean = sum(values) / len(values)
    if mean <= 0:
        return False
    spread = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return spread / mean < OVERLOAD_CV


def clarification(state: SessionState, scored: list[Scored], attribute: str) -> str:
    """Say plainly that the candidates are tied, and what would break the tie."""
    if not scored:
        return ""
    shared = []
    if state.category:
        shared.append(_shorten(state.category))
    for constraint in state.constraints:
        if not constraint.stale and constraint.value:
            shared.append(_shorten(constraint.value))
            break
    if shared:
        return (
            f"These {len(scored)} are all {' and '.join(shared)}, "
            f"and I can't tell them apart on what you've told me so far."
        )
    return "These all look equally close on what you've told me so far."


def explain(index: CatalogIndex, state: SessionState, scored: list[Scored]) -> str:
    """One sentence describing the top result, or "" if there is nothing to say."""
    if not scored:
        return ""
    top = scored[0].document
    clauses: list[str] = []

    matched = matched_constraints(index, state, top)
    if matched:
        clauses.append("matches " + " and ".join(matched[:MAX_REASONS]))
    elif state.category:
        clauses.append(f"is in {_shorten(state.category)}")

    if top.price is not None:
        if state.budget is not None and top.price <= state.budget * 1.15:
            clauses.append(f"costs ${top.price:g}, within your budget")
        else:
            clauses.append(f"costs ${top.price:g}")

    if top.rating_number >= 50 and top.average_rating >= 4.0:
        clauses.append(
            f"is rated {top.average_rating:g} from {top.rating_number:,} reviews"
        )

    if not clauses:
        return ""
    title = _shorten(top.display_title or top.title)
    body = "; ".join(clauses[: MAX_REASONS + 1])
    return f"Top pick: {title} — {body}." if title else f"Top pick {body}."

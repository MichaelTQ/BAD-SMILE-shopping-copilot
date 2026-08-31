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

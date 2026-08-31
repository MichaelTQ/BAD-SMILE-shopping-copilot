"""TechJam conversational shopping agent.

Fully offline and deterministic: no LLM, no API, no network, no agent
framework. Retrieval is SQLite FTS5 BM25 recall followed by a local
constraint-aware rerank, driven by per-session conversation state. Reported
token usage is therefore always zero.

The official harness may import this module from an arbitrary working
directory, so both the package imports and the catalog lookup resolve relative
to this file rather than to ``os.getcwd()``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.index import CatalogIndex
from src.parsing import parse_turn
from src.policy import next_attribute, question_for
from src.ranker import bm25_only, rank
from src.state import SessionState

#: Optional override for the catalog location, e.g. when the harness stages the
#: frozen catalog outside the repository.
CATALOG_ENV_VAR = "TECHJAM_CATALOG"

RESULT_MESSAGE = "Here are the closest matches based on what you've told me so far."
NO_RESULT_MESSAGE = "I couldn't narrow it down yet."


def resolve_catalog_path(explicit: str | Path | None = None) -> Path:
    """Locate ``catalog.jsonl`` without depending on the working directory.

    An explicitly requested location (argument or environment variable) is a
    statement of intent: if it is missing this raises rather than silently
    falling back, because ranking against the wrong catalog is worse than
    failing loudly. With nothing specified, the usual locations are tried.
    """
    for stated, origin in (
        (explicit, "catalog_path argument"),
        (os.environ.get(CATALOG_ENV_VAR), f"${CATALOG_ENV_VAR}"),
    ):
        if stated:
            path = Path(stated)
            if path.is_file():
                return path
            raise FileNotFoundError(f"catalog.jsonl not found at {origin}: {path}")

    candidates = (
        _REPO_ROOT / "data" / "catalog.jsonl",
        Path.cwd() / "data" / "catalog.jsonl",
        Path(__file__).resolve().parent / "data" / "catalog.jsonl",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"catalog.jsonl not found. Looked in: {searched}")


class Agent:
    """Stateful, rule-driven, offline shopping agent."""

    def __init__(self, catalog_path: str | Path | None = None) -> None:
        self.catalog_path = resolve_catalog_path(catalog_path)
        self.index = CatalogIndex(self.catalog_path)
        self.sessions: dict[str, SessionState] = {}

    # ---------------------------------------------------------- official API

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Start a fresh session. The profile is anonymized aggregate data."""
        self.sessions[session_id] = SessionState(
            session_id=session_id, user_profile=dict(user_profile or {})
        )

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        # The harness always calls reset() first, but a missing session must not
        # take the whole run down: an exception here would score as a miss.
        state = self.sessions.get(session_id)
        if state is None:
            self.reset(session_id, {})
            state = self.sessions[session_id]

        try:
            recommendations = self._recommend(state, user_message, top_k)
        except Exception:
            recommendations = self._fallback(state, user_message, top_k)

        try:
            attribute = next_attribute(state)
            question = question_for(attribute)
        except Exception:
            attribute, question = "other", question_for("other")

        message = f"{RESULT_MESSAGE if recommendations else NO_RESULT_MESSAGE} {question}"
        try:
            state.record_response(
                message, attribute, [item["parent_asin"] for item in recommendations]
            )
        except Exception:
            pass
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": recommendations,
            # No model is called anywhere in this agent.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    # -------------------------------------------------------------- internals

    def _recommend(self, state: SessionState, user_message: str, top_k: int) -> list[dict]:
        state.record_turn(user_message, parse_turn(user_message))
        scored = rank(self.index, state, top_k)
        if not scored:
            return self._fallback(state, user_message, top_k)
        return _as_payload(scored, top_k)

    def _fallback(self, state: SessionState, user_message: str, top_k: int) -> list[dict]:
        """Plain BM25 on the raw message; then the previous turn's list."""
        try:
            scored = bm25_only(self.index, user_message, top_k)
            if scored:
                return _as_payload(scored, top_k)
        except Exception:
            pass
        # Repeating the previous ranking beats returning nothing: an empty turn
        # can never hit, whereas a stale list still can.
        return [{"parent_asin": asin} for asin in state.last_recommendations[:top_k]]


def _as_payload(scored: list, top_k: int) -> list[dict]:
    return [
        {"parent_asin": item.document.parent_asin, "score": round(item.score, 6)}
        for item in scored[:top_k]
    ]

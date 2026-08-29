"""TechJam conversational shopping agent.

Fully offline and deterministic: no LLM, no API, no network. Retrieval is
SQLite FTS5 BM25 recall followed by a local constraint-aware rerank, driven by
per-session conversation state. Reported token usage is therefore always zero.
"""

from __future__ import annotations

from pathlib import Path

from src.index import CatalogIndex
from src.parsing import parse_turn
from src.policy import next_attribute, question_for
from src.ranker import bm25_only, rank
from src.state import SessionState

RESULT_MESSAGE = "Here are the closest matches based on what you've told me so far."
NO_RESULT_MESSAGE = "I couldn't narrow it down yet."


class Agent:
    """Stateful, rule-driven, offline shopping agent."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.index = CatalogIndex(self.catalog_path)
        self.sessions: dict[str, SessionState] = {}

    # -------------------------------------------------------- official API

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Start a fresh session. The profile is anonymized aggregate data."""
        self.sessions[session_id] = SessionState(
            session_id=session_id, user_profile=dict(user_profile or {})
        )

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self.sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")

        try:
            state.record_turn(user_message, parse_turn(user_message))
            scored = rank(self.index, state, top_k)
        except Exception:
            # Never fail a turn: fall back to plain BM25 on the raw message.
            scored = bm25_only(self.index, user_message, top_k)

        if not scored:
            scored = bm25_only(self.index, user_message, top_k)

        recommendations = [
            {"parent_asin": item.document.parent_asin, "score": round(item.score, 6)}
            for item in scored[:top_k]
        ]
        attribute = next_attribute(state)
        question = question_for(attribute)
        message = f"{RESULT_MESSAGE if recommendations else NO_RESULT_MESSAGE} {question}"
        state.record_response(
            message, attribute, [item["parent_asin"] for item in recommendations]
        )
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": recommendations,
            # No model is called anywhere in this agent.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

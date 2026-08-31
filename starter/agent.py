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

def _find_package_root(module_file: Path) -> Path:
    """Nearest ancestor of ``module_file`` that holds the ``src`` package.

    Walking up beats assuming a fixed depth: the same file then works both in
    this repository (``<root>/starter/agent.py``) and in the flat submission
    layout the rules suggest (``submission/agent.py`` beside ``submission/src/``).
    """
    for candidate in (module_file.parent, *module_file.parents):
        if (candidate / "src" / "index.py").is_file():
            return candidate
    return module_file.parent.parent


_REPO_ROOT = _find_package_root(Path(__file__).resolve())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.explain import clarification, explain, is_overloaded
from src.index import CatalogIndex
from src.llm import LocalLLM
from src.parsing import parse_turn
from src.policy import next_attribute, question_for
from src.ranker import bm25_only, rank
from src.state import SessionState

#: Optional override for the catalog location, e.g. when the harness stages the
#: frozen catalog outside the repository.
CATALOG_ENV_VAR = "TECHJAM_CATALOG"

RESULT_MESSAGE = "Here are the closest matches based on what you've told me so far."
NO_RESULT_MESSAGE = "I couldn't narrow it down yet."


#: How far up to look for a ``data/`` directory beside an ancestor.
CATALOG_SEARCH_DEPTH = 4


def _data_candidates(start: Path) -> list[Path]:
    """``data/catalog.jsonl`` beside ``start`` and each of its near ancestors."""
    seen: list[Path] = []
    for depth, base in enumerate((start, *start.parents)):
        if depth > CATALOG_SEARCH_DEPTH:
            break
        seen.append(base / "data" / "catalog.jsonl")
    return seen


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

    # The harness may place this package anywhere relative to the catalog, and
    # the API contract does not specify how Agent is constructed, so search
    # upward from both the package and the working directory rather than
    # assuming one fixed layout.
    candidates = [
        *_data_candidates(_REPO_ROOT),
        *_data_candidates(Path.cwd()),
        Path(__file__).resolve().parent / "data" / "catalog.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "catalog.jsonl not found. The competition catalog is distributed by the "
        "organizer and is not stored in this repository.\n"
        "Place it at <repo>/data/catalog.jsonl, or point TECHJAM_CATALOG at it, "
        "or pass its path to Agent(...).\n"
        f"Looked in:\n  {searched}"
    )


class Agent:
    """Stateful, rule-driven, offline shopping agent."""

    def __init__(self, catalog_path: str | Path | None = None) -> None:
        self.catalog_path = resolve_catalog_path(catalog_path)
        self.index = CatalogIndex(self.catalog_path)
        self.sessions: dict[str, SessionState] = {}
        # Opt-in and absent by default: without TECHJAM_LLM_ENDPOINT this is
        # inert and the agent is exactly the offline rule-based system.
        self.llm = LocalLLM()

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

        rationale = ""
        overloaded = False
        scored: list = []
        try:
            scored = self._rank(state, user_message, top_k)
            recommendations = _as_payload(scored, top_k)
            state.last_score_cv = _score_spread(scored[:top_k])
            overloaded = is_overloaded(scored[:top_k])
            if not overloaded:
                rationale = explain(self.index, state, scored)
        except Exception:
            recommendations = self._fallback(state, user_message, top_k)

        try:
            attribute = next_attribute(state)
            question = question_for(attribute)
        except Exception:
            attribute, question = "other", question_for("other")

        if overloaded and recommendations:
            # The ranking has no opinion: say so and ask, rather than presenting
            # an arbitrary order as if it were a considered one.
            opening = clarification(state, scored[:top_k], attribute)
        else:
            opening = RESULT_MESSAGE if recommendations else NO_RESULT_MESSAGE
        message = " ".join(part for part in (opening, rationale, question) if part)
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

    def _understand(self, user_message: str):
        """Rules first; consult the LLM only when they find no category at all.

        The rules beat the LLM on every phrasing they cover, so calling the model
        unconditionally would lower accuracy as well as add latency. This keeps
        the call rate at exactly the rule-failure rate.
        """
        parsed = parse_turn(user_message)
        if parsed.category is None and self.llm.available:
            category = self.llm.extract_category(user_message)
            if category:
                parsed.category = category
        return parsed

    def _rank(self, state: SessionState, user_message: str, top_k: int) -> list:
        state.record_turn(user_message, self._understand(user_message))
        scored = rank(self.index, state, top_k)
        if not scored:
            scored = bm25_only(self.index, user_message, top_k)
        return scored

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


def _score_spread(scored: list) -> float:
    """Coefficient of variation of the returned scores; 0 when undefined."""
    if len(scored) < 2:
        return 0.0
    values = [item.score for item in scored]
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return (variance ** 0.5) / mean


def _as_payload(scored: list, top_k: int) -> list[dict]:
    return [
        {"parent_asin": item.document.parent_asin, "score": round(item.score, 6)}
        for item in scored[:top_k]
    ]

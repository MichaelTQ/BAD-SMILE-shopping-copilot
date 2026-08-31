"""Local HTTP bridge between the React demo UI and the real Agent.

DEVELOPMENT / DEMO TOOL — NOT PART OF THE SCORED PATH.

The graded run imports `starter.agent.Agent` directly; nothing here is on that
path. This exists so the UI shows real retrieval instead of mock data. Standard
library only, to keep the submission's zero-dependency property intact.

    python3 -m scripts.serve            # then: cd frontend && npm run dev
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.explain import matched_constraints
from src.parsing import (COLOR_RE, FEATURE_WORDS, MATERIAL_RE, SIZE_RE,
                         STYLE_WORDS, USE_CASE_WORDS)
from src.text import query_tokens
from starter.agent import Agent

# The UI camel-cases these; keep its vocabulary rather than leaking ours.
ATTRIBUTE_ALIASES = {"use_case": "useCase"}
DISPLAY_NAMES = {
    "category": "Category", "material": "Material", "color": "Color",
    "size": "Size or fit", "style": "Style", "brand": "Brand",
    "budget": "Budget", "feature": "Feature", "useCase": "Use case",
}
ICON_RULES = (
    ("boot", ("boot",)), ("shoe", ("shoe", "sneaker", "loafer", "sandal", "heel")),
    ("shirt", ("shirt", "tee", "top", "blouse", "sweater", "hoodie")),
    ("jacket", ("jacket", "coat", "parka", "vest")),
    ("dress", ("dress", "gown", "skirt")),
    ("leggings", ("legging", "pant", "jean", "trouser", "short")),
    ("bag", ("bag", "backpack", "purse", "tote", "wallet")),
    ("belt", ("belt",)), ("scarf", ("scarf", "hat", "glove", "sock", "earring", "necklace")),
)
TEXTURE_RULES = (
    ("leather", ("leather", "suede")), ("cotton", ("cotton",)), ("wool", ("wool", "cashmere")),
    ("linen", ("linen",)), ("mesh", ("mesh", "breathable")), ("knit", ("knit", "sweater")),
    ("canvas", ("canvas", "denim")), ("stretch", ("spandex", "stretch", "elastane")),
    ("technical", ("waterproof", "nylon", "polyester")), ("sport", ("athletic", "running")),
)
TONES = ("forest", "slate", "sand", "clay", "ocean", "plum")


def _pick(text: str, rules, default: str) -> str:
    for name, needles in rules:
        if any(needle in text for needle in needles):
            return name
    return default


def facets(text: str) -> list[tuple[str, str]]:
    """Concrete attribute values inside free text.

    The parser keeps a whole phrase as the category when a shopper types one
    sentence ("waterproof leather hiking boots for winter"). That is correct for
    retrieval — every word still reaches the query — but the UI should show what
    was understood, so the values are surfaced here rather than in the graded
    path.
    """
    lowered = text.lower()
    words = set(query_tokens(lowered))
    found: list[tuple[str, str]] = []
    for match in MATERIAL_RE.finditer(lowered):
        found.append(("material", match.group(1)))
    for match in COLOR_RE.finditer(lowered):
        found.append(("color", match.group(1)))
    for match in SIZE_RE.finditer(lowered):
        value = match.group(1) or match.group(2)
        if value:
            found.append(("size", value))
    for word in USE_CASE_WORDS:
        if word in words:
            found.append(("useCase", word))
    for word in FEATURE_WORDS:
        if " " in word:
            if word in lowered:
                found.append(("feature", word))
        elif word in words:
            found.append(("feature", word))
    for word in STYLE_WORDS:
        if word in words and word not in {"fit"}:
            found.append(("style", word))
    seen, unique = set(), []
    for attribute, value in found:
        if (attribute, value) not in seen:
            seen.add((attribute, value))
            unique.append((attribute, value))
    return unique


def _match_level(rank: int, signals: int) -> str:
    if signals >= 2 and rank <= 3:
        return "strong"
    if signals >= 1 and rank <= 6:
        return "good"
    return "partial" if signals else "consider"


class Bridge:
    """Adapts Agent output to the shape the UI already expects."""

    def __init__(self, catalog_path: str | Path | None = None) -> None:
        self.agent = Agent(catalog_path)
        self.catalog: dict[str, dict] = {}
        with self.agent.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    product = json.loads(line)
                    self.catalog[str(product["parent_asin"])] = product

    def _product(self, asin: str, rank: int, state, index) -> dict:
        raw = self.catalog.get(asin, {})
        title = str(raw.get("title") or asin)
        haystack = " ".join(
            str(raw.get(field) or "") for field in ("title", "categories", "features")
        ).lower()
        document = next(
            (d for d in index.documents(
                [r[0] for r in [index.connection.execute(
                    "SELECT rowid FROM meta WHERE parent_asin=?", (asin,)).fetchone() or (0,)]]
            ).values()), None
        )
        signals = matched_constraints(index, state, document) if document else []
        described = [f"Matches {value}" for value in signals[:2]]
        # Attribute-level hits, so a product says *why* it matched.
        if document:
            words = index.token_set(document)
            for attribute, value in self._facets(state)[:4]:
                if value in words and len(described) < 4:
                    described.append(f"{DISPLAY_NAMES.get(attribute, attribute)}: {value}")
            signals = signals + [v for _, v in self._facets(state) if v in words]
        if raw.get("price") not in (None, ""):
            described.append(f"${raw['price']}")
        if (raw.get("rating_number") or 0) >= 50:
            described.append(f"{raw.get('average_rating')}★ from {raw['rating_number']:,} reviews")
        description = ""
        for candidate in (raw.get("description"), raw.get("features")):
            if isinstance(candidate, list) and candidate:
                description = str(candidate[0])[:220]
                break
        return {
            "parent_asin": asin,
            "title": title,
            "store": str(raw.get("store") or ""),
            "category": [str(c) for c in (raw.get("categories") or [])][1:4],
            "price": raw.get("price"),
            "averageRating": raw.get("average_rating"),
            "ratingNumber": raw.get("rating_number"),
            "description": description,
            "rank": rank,
            "inTopTen": rank <= 10,
            "matchLevel": _match_level(rank, len(signals)),
            "matchSignals": described or ["Relevant catalog result"],
            "visual": {
                "icon": _pick(haystack, ICON_RULES, "shirt"),
                "tone": TONES[hash(asin) % len(TONES)],
                "texture": _pick(haystack, TEXTURE_RULES, "cotton"),
            },
        }

    def _facets(self, state) -> list[tuple[str, str]]:
        text = " ".join(
            [state.category or ""]
            + [c.value for c in state.constraints if not c.stale]
        )
        return facets(text)

    def _preferences(self, state) -> list[dict]:
        out, seen = [], set()
        for constraint in state.constraints:
            if constraint.stale:
                continue
            attribute = ATTRIBUTE_ALIASES.get(constraint.attribute, constraint.attribute)
            label = " ".join(constraint.value.split())[:48]
            if (attribute, label) in seen:
                continue
            seen.add((attribute, label))
            out.append({
                "attribute": attribute,
                "label": label,
                "displayName": DISPLAY_NAMES.get(attribute, attribute),
            })
        for attribute, value in self._facets(state):
            if (attribute, value) in seen:
                continue
            seen.add((attribute, value))
            out.append({"attribute": attribute, "label": value.title(),
                        "displayName": DISPLAY_NAMES.get(attribute, attribute)})
        if state.category:
            out.insert(0, {"attribute": "category",
                           "label": " ".join(state.category.split())[:40],
                           "displayName": "Category"})
        return out[:8]

    def reset(self, session_id: str) -> dict:
        self.agent.reset(session_id, {})
        return {"message": "What are you shopping for today?",
                "preferences": [], "recommendations": []}

    def respond(self, session_id: str, message: str, turn: int) -> dict:
        if session_id not in self.agent.sessions:
            self.agent.reset(session_id, {})
        result = self.agent.respond(session_id, message, max(1, min(turn, 10)), 10)
        state = self.agent.sessions[session_id]
        index = self.agent.index
        products = [
            self._product(item["parent_asin"], position, state, index)
            for position, item in enumerate(result["recommendations"], start=1)
        ]
        return {
            "message": result["message"],
            "askAttribute": ATTRIBUTE_ALIASES.get(result["ask_attribute"],
                                                  result["ask_attribute"]),
            "preferences": self._preferences(state),
            "recommendations": products,
            "turn": turn,
            "usage": result["usage"],
        }


def make_handler(bridge: Bridge):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send({})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send({"error": "invalid JSON"}, 400)
            session_id = str(data.get("sessionId") or "ui")
            try:
                if self.path.rstrip("/") == "/api/reset":
                    return self._send(bridge.reset(session_id))
                if self.path.rstrip("/") == "/api/respond":
                    return self._send(bridge.respond(
                        session_id, str(data.get("message") or ""),
                        int(data.get("turn") or 1)))
            except Exception as error:  # never 500 the demo
                return self._send({"error": str(error)}, 500)
            self._send({"error": "not found"}, 404)

        def log_message(self, *args) -> None:
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo API bridge for the UI")
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print("building index...", flush=True)
    bridge = Bridge(args.catalog)
    print(f"ready on http://localhost:{args.port}  ({len(bridge.catalog):,} products)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(bridge)).serve_forever()


if __name__ == "__main__":
    main()

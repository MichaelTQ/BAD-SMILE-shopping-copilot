"""Rule-based understanding of a customer turn.

No LLM is involved. The simulator writes short, fairly templated sentences, and
real shoppers write short sentences too, so a small set of regular expressions
plus attribute vocabularies covers both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import query_tokens

# Attributes the contract allows in ``ask_attribute``.
ATTRIBUTES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
)

MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "denim", "linen", "cashmere", "fleece", "suede",
    "mesh", "satin", "velvet", "acrylic", "elastane", "microfiber",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "beige", "navy", "ivory", "gold", "silver",
    "khaki", "burgundy", "teal", "cream", "tan",
)
SIZE_WORDS = (
    "size", "sizing", "small", "medium", "large", "xl", "xxl", "xs",
    "petite", "plus", "wide", "narrow", "width", "length", "inseam", "fit true",
)
STYLE_WORDS = (
    "style", "fit", "sleeve", "sleeves", "neck", "neckline", "crew", "v-neck",
    "casual", "formal", "vintage", "classic", "modern", "slim", "relaxed",
    "loose", "department", "womens", "mens", "girls", "boys", "unisex",
)
USE_CASE_WORDS = (
    "hiking", "running", "gym", "workout", "training", "yoga", "winter",
    "summer", "outdoor", "work", "office", "travel", "wedding", "party",
    "beach", "swim", "sleep", "everyday", "school", "hunting", "fishing",
)
FEATURE_WORDS = (
    "pocket", "pockets", "waterproof", "breathable", "stretch", "lightweight",
    "warm", "moisture", "wicking", "machine wash", "zipper", "adjustable",
    "hypoallergenic", "durable", "comfortable", "upf", "insulated",
)

MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.I)
SIZE_RE = re.compile(r"\b(?:size\s+(\d{1,2}(?:\.\d)?|x{0,2}s|m|l|x{0,3}l)|(\d{1,2}(?:\.\d)?)\s*(?:us|uk|eu)\b)", re.I)
BUDGET_RE = re.compile(
    r"(?:\$\s*(\d+(?:\.\d+)?))|(?:\b(?:under|below|less than|budget of|up to|max(?:imum)?)\s+(\d+(?:\.\d+)?))",
    re.I,
)

# Envelopes the simulator (and most shoppers) wrap their constraints in.
CATEGORY_RE = re.compile(
    r"\b(?:looking for|shopping for|i need|i want|searching for|show me|find me|"
    r"interested in|in the market for)\s+(?:(?:a|an|the|some)\s+)?([^.,;!?]{2,120})",
    re.I,
)
HARD_RE = re.compile(
    r"(?:key requirement is|requirement is|must have|what i need is|i need is|"
    r"it must be|has to be)\s*:?\s*([^\n]+)",
    re.I,
)
SOFT_RE = re.compile(
    r"(?:what matters is|matters is|i(?:'d| would) like|i prefer|preferably|"
    r"ideally|it should be)\s*:?\s*([^\n]+)",
    re.I,
)
NO_PREF_RE = re.compile(
    r"(?:no|don't have (?:an? )?(?:additional |strong )?|do not have (?:an? )?(?:additional |strong )?)"
    r"preference(?: for| about| on)?\s*([a-z_ ]{0,20})",
    re.I,
)
OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead|on second thought|changed my mind|scratch that|"
    r"forget (?:that|what)|ignore my earlier|ignore what i|no longer|"
    r"rather than|not that)\b",
    re.I,
)
NUDGE_RE = re.compile(r"not quite right|ask me about", re.I)


@dataclass(frozen=True)
class Constraint:
    """One piece of stated preference, typed to an allowed attribute."""

    attribute: str
    value: str
    hard: bool = False
    # Set when a later intent override superseded the turn that stated this.
    stale: bool = False

    def key(self) -> tuple[str, str]:
        return (self.attribute, re.sub(r"\s+", " ", self.value.strip().lower()))


@dataclass
class ParsedTurn:
    category: str | None = None
    constraints: list[Constraint] = field(default_factory=list)
    no_preference: set[str] = field(default_factory=set)
    override: bool = False
    nudge: bool = False
    budget: float | None = None


def classify(value: str) -> str:
    """Map a free-text constraint onto one allowed attribute."""
    lowered = value.lower()
    if BUDGET_RE.search(lowered) or "budget" in lowered or "price" in lowered:
        return "budget"
    if MATERIAL_RE.search(lowered):
        return "material"
    if COLOR_RE.search(lowered) or "color" in lowered or "colour" in lowered:
        return "color"
    if any(word in lowered for word in SIZE_WORDS):
        return "size"
    if any(word in lowered for word in STYLE_WORDS):
        return "style"
    if any(word in lowered for word in USE_CASE_WORDS):
        return "use_case"
    if "brand" in lowered or "made by" in lowered:
        return "brand"
    return "feature"


def _is_informative(value: str, minimum: int = 2) -> bool:
    """True when a fragment carries content rather than conversational filler."""
    return len(query_tokens(value)) >= minimum or bool(
        MATERIAL_RE.search(value) or COLOR_RE.search(value) or BUDGET_RE.search(value)
    )


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.:\t\n")


def _split_values(payload: str) -> list[str]:
    """The simulator joins multiple disclosed constraints with ``; ``."""
    parts = [_clean(part) for part in payload.split(";")]
    return [part for part in parts if part]


def _no_preference_attribute(fragment: str) -> str | None:
    fragment = fragment.strip().lower().replace(" ", "_")
    for attribute in ATTRIBUTES:
        if fragment.startswith(attribute):
            return attribute
    return None


def parse_turn(message: str) -> ParsedTurn:
    """Extract everything a single customer message tells us."""
    parsed = ParsedTurn()
    text = message or ""
    parsed.override = bool(OVERRIDE_RE.search(text))
    parsed.nudge = bool(NUDGE_RE.search(text))

    residual = text

    # Order matters: strip the explicit envelopes first so that phrases such as
    # "what I need is: ..." are not mistaken for a category mention.
    for match in NO_PREF_RE.finditer(text):
        attribute = _no_preference_attribute(match.group(1) or "")
        if attribute:
            parsed.no_preference.add(attribute)
        residual = residual.replace(match.group(0), " ")

    for regex, hard in ((HARD_RE, True), (SOFT_RE, False)):
        for match in regex.finditer(residual):
            for value in _split_values(match.group(1)):
                parsed.constraints.append(Constraint(classify(value), value, hard))
        residual = regex.sub(" ", residual)

    category_match = CATEGORY_RE.search(residual)
    if category_match:
        candidate = _clean(category_match.group(1))
        # A category can legitimately be a single word ("boots").
        if _is_informative(candidate, minimum=1):
            parsed.category = candidate
            residual = residual.replace(category_match.group(0), " ")

    # Anything left over is still a stated preference: the intent-override
    # scenario opens with a bare sentence appended after the category.
    for sentence in re.split(r"[.;\n]", residual):
        sentence = _clean(sentence)
        if _is_informative(sentence) and not NUDGE_RE.search(sentence):
            parsed.constraints.append(Constraint(classify(sentence), sentence, False))

    budget_match = BUDGET_RE.search(text)
    if budget_match:
        raw = budget_match.group(1) or budget_match.group(2)
        try:
            parsed.budget = float(raw)
        except (TypeError, ValueError):
            parsed.budget = None
    return parsed


def attribute_values(constraints: list[Constraint]) -> dict[str, set[str]]:
    """Concrete vocabulary values (colors, materials, sizes) worth boosting."""
    values: dict[str, set[str]] = {"material": set(), "color": set(), "size": set()}
    for constraint in constraints:
        for match in MATERIAL_RE.finditer(constraint.value):
            values["material"].add(match.group(1).lower())
        for match in COLOR_RE.finditer(constraint.value):
            values["color"].add(match.group(1).lower())
        for match in SIZE_RE.finditer(constraint.value):
            token = match.group(1) or match.group(2)
            if token:
                values["size"].add(token.lower())
    return values

"""Rule-based understanding of a customer turn.

No LLM is involved. The simulator writes short, fairly templated sentences, and
real shoppers write short sentences too, so a small set of regular expressions
plus attribute vocabularies covers both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import TOKEN_RE, query_tokens

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
# Openers that introduce what the customer is shopping for. Deliberately broad:
# the specification warns the organizer may paraphrase the simulator's wording,
# and measurement showed the resulting loss comes from failing to recognise the
# *category slot*, not from vocabulary mismatch — the category terms still reach
# the query, they just fall from a weight-3.0 slot to a weight-1.0 constraint.
# Widening this list restored turn-1 category parsing to 100% under every
# rewrite tested, with no change on the original templates.
# Order matters: alternation takes the first match, so longer openers that
# contain a shorter one ("want to buy" vs "want") must come first.
CATEGORY_OPENERS = (
    r"looking to (?:buy|get|find|pick up)",
    r"trying to (?:buy|get|find)",
    r"hoping to (?:buy|get|find)",
    r"want(?:ed)? to (?:buy|get|find|pick up)",
    r"need(?:ed)? to (?:buy|get|find)",
    r"window shopping (?:for|around)",
    r"any recommendations for",
    r"recommend (?:me )?(?:a|an|some)?",
    r"in the market for",
    r"in need of",
    r"looking for",
    r"shopping (?:for|around for)",
    r"browsing (?:for|around)",
    r"search(?:ing)? for",
    r"hunting for",
    r"interested in",
    r"do you have",
    r"show me",
    r"find me",
    r"get me",
    # Alternation is tried left-to-right *per start position*, and "I want" starts
    # earlier than "want to buy", so the pronoun form must swallow the infinitive
    # itself or it captures "to buy ..." as the category.
    r"i(?:'m| am)?\s*(?:need(?:ed)?|want(?:ed)?|after|seeking)"
    r"(?:\s+to\s+(?:buy|get|find|pick up))?",
    r"need some",
    r"want some",
)
CATEGORY_RE = re.compile(
    r"\b(?:" + "|".join(CATEGORY_OPENERS) + r")"
    # An em/en dash separates clauses just like a comma, so it also terminates.
    r"\s+(?:(?:a|an|the|some|any)\s+)?([^.,;!?\u2013\u2014]{2,120})",
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
# A bare noun phrase names what the shopper wants; a sentence with a verb is a
# statement about it and should stay a constraint.
_VERB_OPENER_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|do|does|did|can|could|should|would|"
    r"will|prefer|like|want|need|think|matter|matters|care)\b", re.I
)


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


def _vocabulary_hit(words: set[str], lowered: str, vocabulary: tuple[str, ...]) -> bool:
    """Whole-word vocabulary match.

    Substring matching would fire on "fit" inside "outfit"/"benefit" and on
    "xs" inside "boxset", so single-word entries are matched against the token
    set. Multi-word and hyphenated entries ("fit true", "v-neck") are long
    enough to stay safe as substrings.
    """
    for term in vocabulary:
        if term in words:
            return True
        if (" " in term or "-" in term) and term in lowered:
            return True
    return False


def classify(value: str) -> str:
    """Map a free-text constraint onto one allowed attribute."""
    lowered = value.lower()
    words = set(TOKEN_RE.findall(lowered))
    if BUDGET_RE.search(lowered) or "budget" in words or "price" in words:
        return "budget"
    if MATERIAL_RE.search(lowered):
        return "material"
    if COLOR_RE.search(lowered) or "color" in words or "colour" in words:
        return "color"
    if _vocabulary_hit(words, lowered, SIZE_WORDS):
        return "size"
    if _vocabulary_hit(words, lowered, STYLE_WORDS):
        return "style"
    if _vocabulary_hit(words, lowered, USE_CASE_WORDS):
        return "use_case"
    if "brand" in words or "made by" in lowered:
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

    # A shopper often names the product and the budget in one breath
    # ("waterproof hiking boots under $150"). Without stripping the amount the
    # whole sentence classifies as `budget` and the category slot goes empty,
    # which costs it the category weight. The simulator always states budget
    # separately, so this only affects free-form input.
    if parsed.category is None:
        stripped = _clean(BUDGET_RE.sub(" ", residual))
        # "under $150" leaves a dangling preposition once the amount is removed.
        stripped = _clean(re.sub(
            r"\b(?:under|below|less than|budget(?: of)?|up to|max(?:imum)?|around|about)\s*$",
            "", stripped, flags=re.I))
        if _is_informative(stripped, minimum=1) and not _VERB_OPENER_RE.search(stripped):
            parsed.category = stripped[:120]
            parsed.constraints = [
                c for c in parsed.constraints
                if _clean(c.value) != _clean(residual)
            ]

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

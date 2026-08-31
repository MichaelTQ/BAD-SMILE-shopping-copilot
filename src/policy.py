"""Which attribute to ask about next, and the fixed question templates."""

from __future__ import annotations

from .state import SessionState

# Asking order. ``other`` comes first because it is the broadest legal probe:
# it invites whatever the customer considers most important without presuming a
# facet. The rest run from most to least discriminative for apparel retrieval.
ASK_ORDER = (
    "other", "feature", "material", "color", "style",
    "use_case", "size", "brand", "budget", "category",
)

# Fixed templates, one per allowed attribute. Deterministic by design: no model
# generates text anywhere in this agent.
QUESTIONS = {
    "category": "Which type of item are you shopping for exactly?",
    "material": "Do you have a material preference?",
    "color": "Which color are you looking for?",
    "size": "Which size or fit do you need?",
    "style": "Which style or cut do you prefer?",
    "brand": "Is there a brand you prefer?",
    "budget": "What budget range should I stay within?",
    "feature": "Which product feature matters most to you?",
    "use_case": "What will you mainly use it for?",
    "other": "What matters most to you in this purchase?",
}

# An attribute may be asked twice only if the first ask produced new
# information; attributes answered with "no preference" are never re-asked.
MAX_ASKS = 2


def next_attribute(state: SessionState) -> str:
    """Pick the next legal attribute to ask about.

    Priority: a fresh attribute the customer has neither answered nor declined;
    then one that was productive before and may still have more to give. An
    attribute the customer declined is never asked again.
    """
    for attribute in ASK_ORDER:
        if attribute in state.no_preference or attribute in state.answered:
            continue
        if state.asked.get(attribute, 0) == 0:
            return attribute
    for attribute in ASK_ORDER:
        if attribute in state.no_preference:
            continue
        if 0 < state.asked.get(attribute, 0) < MAX_ASKS and attribute in state.answered:
            # Productive last time; the customer may still have more to say.
            return attribute
    # Everything is exhausted; "other" stays legal and harmless.
    return "other"


def question_for(attribute: str) -> str:
    return QUESTIONS.get(attribute, QUESTIONS["other"])

"""Per-session conversation memory.

One :class:`SessionState` per ``session_id``. It accumulates everything the
customer has told us so that every turn searches with the full dialogue context
rather than only the latest message.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import dataclasses

from .parsing import Constraint, ParsedTurn


@dataclass
class SessionState:
    session_id: str
    user_profile: dict = field(default_factory=dict)

    # Full transcript, kept for the demo script and for debugging.
    history: list[dict] = field(default_factory=list)

    # Currently effective constraints, in the order they were stated.
    constraints: list[Constraint] = field(default_factory=list)
    category: str | None = None
    budget: float | None = None

    # Attribute bookkeeping for the ask policy.
    asked: dict[str, int] = field(default_factory=dict)
    answered: set[str] = field(default_factory=set)
    no_preference: set[str] = field(default_factory=set)
    last_asked: str | None = None

    # parent_asin -> number of turns it has been recommended in.
    recommended: dict[str, int] = field(default_factory=dict)

    turns: int = 0
    overrides: int = 0
    # Consecutive turns that produced no new information. Drives exploration:
    # if the customer keeps adding nothing and the target has not surfaced,
    # the current Top 10 is simply wrong and needs to rotate.
    stalled_turns: int = 0

    # -------------------------------------------------------------- updating

    def record_turn(self, user_message: str, parsed: ParsedTurn) -> None:
        """Fold one parsed customer message into the accumulated state."""
        self.turns += 1
        self.history.append({"role": "user", "turn": self.turns, "text": user_message})

        if parsed.override:
            self._apply_override()

        if parsed.category:
            self.category = parsed.category
        if parsed.budget is not None:
            self.budget = parsed.budget

        self.no_preference |= parsed.no_preference
        for attribute in parsed.no_preference:
            self.answered.add(attribute)

        gained = False
        for constraint in parsed.constraints:
            index = self._find(constraint.key())
            if index is None:
                self.constraints.append(constraint)
                gained = True
            elif self._refresh(index, constraint):
                # Re-stating a preference after an override makes it current
                # again, and promotes it if it is now stated as a requirement.
                gained = True
            self.answered.add(constraint.attribute)

        # An answer to the question we asked counts as answered even when the
        # customer simply had nothing more to add. An override or a "try again"
        # nudge is not an answer, so it must not silence the attribute.
        if self.last_asked:
            self.answered.add(self.last_asked)
            if not gained and not parsed.override and not parsed.nudge:
                self.no_preference.add(self.last_asked)

        informative = gained or bool(parsed.category) or parsed.budget is not None
        self.stalled_turns = 0 if informative else self.stalled_turns + 1

    def _find(self, key: tuple[str, str]) -> int | None:
        for index, constraint in enumerate(self.constraints):
            if constraint.key() == key:
                return index
        return None

    def _refresh(self, index: int, restated: Constraint) -> bool:
        """Un-stale (and possibly harden) a constraint the customer repeated."""
        current = self.constraints[index]
        hard = current.hard or restated.hard
        if not current.stale and hard == current.hard:
            return False
        self.constraints[index] = dataclasses.replace(current, hard=hard, stale=False)
        return True

    def _apply_override(self) -> None:
        """Intent override: demote everything stated so far, keep the category.

        The constraints are marked stale rather than deleted. A shopper who says
        "actually, what I need is X" is re-prioritising, and the earlier turns
        usually still describe the same product; the ranker down-weights stale
        constraints instead of throwing the dialogue history away.
        """
        self.overrides += 1
        self.constraints = [
            dataclasses.replace(constraint, stale=True) for constraint in self.constraints
        ]

    def record_response(self, message: str, ask_attribute: str | None, asins: list[str]) -> None:
        self.history.append(
            {"role": "agent", "turn": self.turns, "text": message, "ask_attribute": ask_attribute}
        )
        self.last_asked = ask_attribute
        if ask_attribute:
            self.asked[ask_attribute] = self.asked.get(ask_attribute, 0) + 1
        for asin in asins:
            self.recommended[asin] = self.recommended.get(asin, 0) + 1

    # --------------------------------------------------------------- reading

    def hard_constraints(self) -> list[Constraint]:
        return [constraint for constraint in self.constraints if constraint.hard]

    def soft_constraints(self) -> list[Constraint]:
        return [constraint for constraint in self.constraints if not constraint.hard]

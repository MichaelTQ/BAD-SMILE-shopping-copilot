"""Offline unit tests for the rule-based agent.

They run against a tiny hand-written catalog so they are fast and do not depend
on the 50k-row release file.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.index import CatalogIndex
from src.parsing import Constraint, classify, parse_turn
from src.policy import ASK_ORDER, QUESTIONS, next_attribute
from src.ranker import build_query, rank
from src.state import SessionState
from starter.agent import Agent

ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

CATALOG_ROWS = [
    {
        "parent_asin": "AAA", "title": "Blue cotton crew neck t-shirt",
        "features": ["100% Cotton", "Machine Wash"], "description": ["Soft everyday tee"],
        "details": {"Department": "Mens"}, "categories": ["Clothing, Shoes & Jewelry", "Men", "Shirts", "T-Shirts"],
        "store": "Acme", "price": 19.99, "average_rating": 4.5, "rating_number": 900,
    },
    {
        "parent_asin": "BBB", "title": "Black leather hiking boot",
        "features": ["Leather", "Rubber sole"], "description": ["Waterproof winter boot"],
        "details": {"Department": "Mens"}, "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Boots"],
        "store": "Trailco", "price": 129.0, "average_rating": 4.7, "rating_number": 4000,
    },
    {
        "parent_asin": "CCC", "title": "Red polyester running shorts",
        "features": ["Polyester", "Lightweight"], "description": ["Gym and running shorts"],
        "details": {"Department": "Womens"}, "categories": ["Clothing, Shoes & Jewelry", "Women", "Shorts"],
        "store": "Acme", "price": 24.5, "average_rating": 4.1, "rating_number": 120,
    },
]


def write_catalog(directory: Path) -> Path:
    path = directory / "catalog.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in CATALOG_ROWS), encoding="utf-8")
    return path


class ParsingTest(unittest.TestCase):
    def test_extracts_category_and_hard_constraint(self) -> None:
        parsed = parse_turn("I'm looking for Shirts T-Shirts. A key requirement is: 100% Cotton.")
        self.assertEqual(parsed.category, "Shirts T-Shirts")
        self.assertEqual([(c.attribute, c.hard) for c in parsed.constraints], [("material", True)])

    def test_leading_article_is_not_eaten_from_the_category(self) -> None:
        self.assertEqual(parse_turn("I'm looking for Athletic Walking, but I'm exploring.").category,
                         "Athletic Walking")

    def test_splits_multiple_disclosed_constraints(self) -> None:
        parsed = parse_turn("For that, what matters is: cotton; budget around $29.99.")
        self.assertEqual([c.value for c in parsed.constraints], ["cotton", "budget around $29.99"])
        self.assertEqual(parsed.budget, 29.99)

    def test_detects_no_preference_and_names_the_attribute(self) -> None:
        for message in ("I don't have a preference for material; please use your judgment.",
                        "I don't have an additional preference for color."):
            parsed = parse_turn(message)
            self.assertTrue(parsed.no_preference)
            self.assertEqual(parsed.constraints, [], msg=message)
        self.assertEqual(parse_turn("I don't have a preference for size.").no_preference, {"size"})

    def test_detects_intent_override_and_keeps_the_new_value(self) -> None:
        parsed = parse_turn("Actually, ignore my earlier preference. What I need is: leather sole.")
        self.assertTrue(parsed.override)
        self.assertEqual([(c.value, c.hard) for c in parsed.constraints], [("leather sole", True)])

    def test_conversational_filler_produces_no_constraints(self) -> None:
        parsed = parse_turn("Those options are not quite right yet. Ask me about one specific attribute.")
        self.assertEqual(parsed.constraints, [])
        self.assertTrue(parsed.nudge)

    def test_classification_covers_every_allowed_attribute_it_emits(self) -> None:
        cases = {
            "budget around $30": "budget", "100% cotton": "material", "color: black": "color",
            "size 8 running": "size", "crew neck": "style", "great for hiking": "use_case",
            "zipper closure": "feature",
        }
        for value, expected in cases.items():
            self.assertEqual(classify(value), expected, msg=value)
            self.assertIn(expected, ALLOWED_ATTRIBUTES)


class SessionStateTest(unittest.TestCase):
    def test_state_accumulates_across_turns(self) -> None:
        state = SessionState(session_id="s1")
        state.record_turn("I'm looking for Shirts T-Shirts. A key requirement is: 100% Cotton.",
                          parse_turn("I'm looking for Shirts T-Shirts. A key requirement is: 100% Cotton."))
        state.record_response("m", "color", ["AAA"])
        state.record_turn("For that, what matters is: color: blue.",
                          parse_turn("For that, what matters is: color: blue."))
        self.assertEqual(state.category, "Shirts T-Shirts")
        self.assertEqual(len(state.constraints), 2)
        self.assertIn("color", state.answered)
        self.assertEqual(state.recommended, {"AAA": 1})
        self.assertEqual(state.turns, 2)

    def test_repeated_constraints_are_deduplicated(self) -> None:
        state = SessionState(session_id="s1")
        for _ in range(2):
            state.record_turn("What matters is: 100% Cotton.", parse_turn("What matters is: 100% Cotton."))
        self.assertEqual(len(state.constraints), 1)

    def test_unanswered_question_marks_the_attribute_as_no_preference(self) -> None:
        state = SessionState(session_id="s1")
        state.record_response("m", "brand", [])
        state.record_turn("I don't have an additional preference for brand.",
                          parse_turn("I don't have an additional preference for brand."))
        self.assertIn("brand", state.no_preference)

    def test_override_marks_history_stale_without_deleting_it(self) -> None:
        state = SessionState(session_id="s1")
        state.record_turn("I prefer a relaxed fit.", parse_turn("I prefer a relaxed fit."))
        state.record_turn("Actually, what I need is: waterproof leather.",
                          parse_turn("Actually, what I need is: waterproof leather."))
        self.assertEqual(state.overrides, 1)
        stale = [c for c in state.constraints if c.stale]
        fresh = [c for c in state.constraints if not c.stale]
        self.assertTrue(stale and fresh)

    def test_restated_value_after_override_becomes_current_and_hard(self) -> None:
        state = SessionState(session_id="s1")
        state.record_turn("What matters is: leather.", parse_turn("What matters is: leather."))
        state.record_turn("Actually, what I need is: leather.",
                          parse_turn("Actually, what I need is: leather."))
        leather = [c for c in state.constraints if c.value.lower() == "leather"]
        self.assertEqual(len(leather), 1)
        self.assertTrue(leather[0].hard)
        self.assertFalse(leather[0].stale)

    def test_override_and_nudge_turns_do_not_silence_the_pending_attribute(self) -> None:
        for message in ("Actually, ignore my earlier preference. What I need is: leather.",
                        "Those options are not quite right yet. Ask me about one specific attribute."):
            state = SessionState(session_id="s1")
            state.record_turn("What matters is: leather.", parse_turn("What matters is: leather."))
            state.record_response("m", "color", [])
            state.record_turn(message, parse_turn(message))
            self.assertNotIn("color", state.no_preference, msg=message)

    def test_sessions_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(write_catalog(Path(directory)))
            agent.reset("a", {})
            agent.reset("b", {})
            agent.respond("a", "I'm looking for Boots. A key requirement is: Leather.", 1, 10)
            self.assertTrue(agent.sessions["a"].constraints)
            self.assertEqual(agent.sessions["b"].constraints, [])
            self.assertEqual(agent.sessions["b"].turns, 0)


class PolicyTest(unittest.TestCase):
    def test_every_asked_attribute_is_legal_and_has_a_template(self) -> None:
        for attribute in ASK_ORDER:
            self.assertIn(attribute, ALLOWED_ATTRIBUTES)
            self.assertIn(attribute, QUESTIONS)

    def test_no_preference_attributes_are_never_asked_again(self) -> None:
        state = SessionState(session_id="s1")
        state.no_preference = {"other", "feature", "material"}
        for _ in range(6):
            attribute = next_attribute(state)
            self.assertNotIn(attribute, state.no_preference)
            state.record_response("m", attribute, [])
            state.answered.add(attribute)

    def test_productive_attribute_is_not_repeated_before_new_ones(self) -> None:
        state = SessionState(session_id="s1")
        first = next_attribute(state)
        state.record_response("m", first, [])
        state.answered.add(first)
        self.assertNotEqual(next_attribute(state), first)

    def test_exhausted_policy_still_returns_a_legal_attribute(self) -> None:
        state = SessionState(session_id="s1")
        state.no_preference = set(ASK_ORDER)
        self.assertIn(next_attribute(state), ALLOWED_ATTRIBUTES)


class RankingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.index = CatalogIndex(write_catalog(Path(cls._directory.name)))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _state(self, *messages: str) -> SessionState:
        state = SessionState(session_id="s1")
        for message in messages:
            state.record_turn(message, parse_turn(message))
        return state

    def test_query_uses_the_whole_conversation_not_only_the_last_message(self) -> None:
        state = self._state("I'm looking for Boots.", "What matters is: waterproof.")
        weights, category_terms, _ = build_query(state)
        self.assertIn("boots", weights)
        self.assertIn("waterproof", weights)
        self.assertEqual(category_terms, ["boots"])

    def test_hard_constraints_outweigh_soft_ones(self) -> None:
        state = self._state("A key requirement is: leather sole.", "I prefer polyester lining.")
        weights, _, _ = build_query(state)
        self.assertGreater(weights["leather"], weights["polyester"])

    def test_accumulated_context_finds_the_right_product(self) -> None:
        state = self._state("I'm looking for Shoes Boots.", "What matters is: leather; waterproof.")
        self.assertEqual(rank(self.index, state, 10)[0].document.parent_asin, "BBB")

    def test_budget_pushes_over_priced_products_down(self) -> None:
        cheap = self._state("I'm looking for Clothing.", "What matters is: budget around $20.")
        ranked = [item.document.parent_asin for item in rank(self.index, cheap, 3)]
        self.assertLess(ranked.index("AAA"), ranked.index("BBB"))

    def test_empty_state_returns_nothing_rather_than_raising(self) -> None:
        self.assertEqual(rank(self.index, SessionState(session_id="s1"), 10), [])


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.catalog = write_catalog(Path(cls._directory.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_respond_before_reset_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            Agent(self.catalog).respond("missing", "hello", 1, 10)

    def test_response_shape_matches_the_contract_every_turn(self) -> None:
        agent = Agent(self.catalog)
        agent.reset("s", {"summary": "x"})
        messages = [
            "I'm looking for Shirts T-Shirts. A key requirement is: 100% Cotton.",
            "For that, what matters is: color: blue.",
            "I don't have a preference for size; please use your judgment.",
            "Actually, ignore my earlier preference. What I need is: Leather.",
            "Those options are not quite right yet. Ask me about one specific attribute.",
            "",
        ]
        for turn, message in enumerate(messages, start=1):
            response = agent.respond("s", message, turn, 10)
            self.assertIsInstance(response["message"], str)
            self.assertIn(response["ask_attribute"], ALLOWED_ATTRIBUTES)
            self.assertLessEqual(len(response["recommendations"]), 10)
            asins = [item["parent_asin"] for item in response["recommendations"]]
            self.assertEqual(len(asins), len(set(asins)))
            for asin in asins:
                self.assertIn(asin, {row["parent_asin"] for row in CATALOG_ROWS})
            self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_token_usage_stays_zero_across_a_whole_session(self) -> None:
        agent = Agent(self.catalog)
        agent.reset("s", {})
        total = 0
        for turn in range(1, 11):
            usage = agent.respond("s", "I'm looking for Boots.", turn, 10)["usage"]
            total += usage["prompt_tokens"] + usage["completion_tokens"]
        self.assertEqual(total, 0)

    def test_ranking_failure_falls_back_to_bm25(self) -> None:
        agent = Agent(self.catalog)
        agent.reset("s", {})

        def boom(*_args, **_kwargs):
            raise ValueError("rerank exploded")

        import starter.agent as agent_module

        original = agent_module.rank
        agent_module.rank = boom
        try:
            response = agent.respond("s", "leather hiking boot", 1, 10)
        finally:
            agent_module.rank = original
        self.assertTrue(response["recommendations"])
        self.assertEqual(response["recommendations"][0]["parent_asin"], "BBB")

    def test_reset_clears_previous_session_state(self) -> None:
        agent = Agent(self.catalog)
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Boots. A key requirement is: Leather.", 1, 10)
        agent.reset("s", {})
        self.assertEqual(agent.sessions["s"].constraints, [])
        self.assertEqual(agent.sessions["s"].asked, {})


if __name__ == "__main__":
    unittest.main()

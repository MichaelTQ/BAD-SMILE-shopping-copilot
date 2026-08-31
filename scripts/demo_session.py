"""Offline demo: print one full multi-turn session, turn by turn.

DEVELOPMENT TOOL ONLY — NOT PART OF THE SCORED PATH.

This script deliberately reads the local evaluator's hidden intent card and the
public ground truth so a developer can check the dialogue by hand. The Agent
itself never sees any of it: `starter/agent.py` and `src/` import neither the
evaluator nor `data/public_set.jsonl` (enforced by a unit test). Nothing here
is imported by the agent at run time.

    python3 -m scripts.demo_session --sample-id public_0002
"""

from __future__ import annotations

import argparse
import uuid

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent

RULE = "-" * 78


def show_state(agent: Agent, session_id: str) -> None:
    state = agent.sessions[session_id]
    print(f"    category        : {state.category!r}")
    print(f"    budget          : {state.budget}")
    for constraint in state.constraints:
        flag = "hard" if constraint.hard else "soft"
        stale = " (stale)" if constraint.stale else ""
        print(f"    constraint      : [{constraint.attribute}/{flag}{stale}] {constraint.value[:70]}")
    print(f"    asked           : {dict(state.asked)}")
    print(f"    answered        : {sorted(state.answered)}")
    print(f"    no-preference   : {sorted(state.no_preference)}")
    print(f"    recommended so far: {len(state.recommended)} distinct products")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print one offline demo session")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sample-id", default=None, help="default: first sample")
    parser.add_argument("--scenario", default=None, help="pick the first sample of this scenario")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.sample_id:
        samples = [item for item in samples if item["sample_id"] == args.sample_id]
    elif args.scenario:
        samples = [item for item in samples if item["scenario_type"] == args.scenario]
    if not samples:
        raise SystemExit("no matching sample")
    sample = samples[0]

    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)

    session_id = f"demo_{uuid.uuid4().hex[:8]}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    print(RULE)
    print(f"sample_id   : {sample['sample_id']}   scenario: {sample['scenario_type']}")
    print(f"profile     : {sample['user_profile'].get('summary')}")
    print(f"target      : {target}  {str(products[target].get('title'))[:60]}")
    print(f"hidden card : {card}")
    print(RULE)

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n[turn {turn}] customer: {message}")
        response = agent.respond(session_id, message, turn, TOP_K)
        ranked = normalize_recommendations(response["recommendations"], catalog_ids)
        print(f"          agent   : {response['message']}")
        print(f"          ask     : {response['ask_attribute']}   usage: {response['usage']}")
        show_state(agent, session_id)
        for position, asin in enumerate(ranked, start=1):
            mark = "  <== TARGET" if asin == target else ""
            print(f"      {position:2d}. {asin}  {str(products[asin].get('title'))[:56]}{mark}")

        if override_applied and target in ranked:
            print(f"\n{RULE}\nHIT at turn {turn}, rank {ranked.index(target) + 1}\n{RULE}")
            return
        if turn == MAX_TURNS:
            break
        override = effective["behavior"].get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            if override.get("new_value"):
                disclosed.add(str(override["new_value"]))
            message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            message, boundary_used = customer_reply(
                effective, response["ask_attribute"], disclosed, boundary_used
            )
    print(f"\n{RULE}\nNO HIT within {MAX_TURNS} turns\n{RULE}")


if __name__ == "__main__":
    main()

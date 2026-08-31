# BAD SMILE — Shopping Copilot

A multi-turn conversational shopping agent for the TechJam Conversational
E-Commerce Search Challenge. It reads what a shopper says across up to ten
turns, accumulates their stated constraints, and returns the ten catalog
products most likely to be the one they are looking for.

**It runs fully offline: no LLM, no API, no network, no third-party packages —
Python's standard library and SQLite alone.** Reported token usage is `0` and
the running cost is `$0`.

| | BM25 baseline | This agent |
| --- | --- | --- |
| Hit Rate@10 | 0.125 | **0.995** |
| MRR | 0.068034 | **0.770948** |
| MTTC | 9.81 | **1.850** |
| **TechnicalScore** | 0.10671 | **0.911784** |
| Tokens / cost | 0 / $0 | 0 / $0 |

Per scenario: boundary **1.000**, browsing **1.000**, intent override **1.000**,
buying **0.9875**.

## Project overview

Retrieval runs in two stages. **Recall** merges three SQLite FTS5 BM25 routes —
the weighted query, a category-only query, and a strict all-category-terms query
that opens only when the dialogue stalls. **Rerank** then scores that pool on six
local features (BM25, IDF coverage, phrase containment, category overlap, a
popularity prior, budget fit) and returns the top ten.

The query is built from the **whole conversation**, not the latest message.
A per-session state tracker keeps the effective constraints, distinguishes hard
requirements from soft preferences, and handles intent override by de-weighting
superseded constraints rather than deleting them.

Every turn also returns one clarification question and a plain-language reason
for the top result. When the top ten score too closely to be meaningfully
ordered, the agent says so instead of presenting an arbitrary ranking.

```text
starter/agent.py     Agent: reset() / respond() — the official interface
src/parsing.py       rule-based turn understanding
src/state.py         per-session conversation state
src/policy.py        which attribute to ask about next
src/index.py         FTS5 index + product metadata
src/ranker.py        recall and rerank
src/explain.py       customer-facing rationale
src/llm.py           optional local-LLM fallback (disabled by default)
frontend/            React demo UI
scripts/serve.py     HTTP bridge for the UI (not on the scored path)
scripts/demo_session.py   prints one full session, turn by turn
```

`METHOD.md` documents the architecture, every experiment behind it, and the
measured reasons for each default.

## Setup and installation

> **Before running anything:** this repository contains code only. The 50,000
> product catalog and the public session set are the organizer's frozen
> artifacts and are not redistributed here — put them in `data/` first (below).
> `Agent()` also accepts an explicit path, and `TECHJAM_CATALOG` overrides the
> lookup, so the catalog can live anywhere.

Python 3.10 or later. No installation step and no dependencies — SQLite needs
FTS5, which is standard in CPython 3.10+.

The catalog is distributed by the organizer, not in this repository. Download
`catalog.jsonl.gz` from the competition kit release, verify it against the
published `SHA256SUMS`, and place it as `data/catalog.jsonl`:

```bash
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl
```

`data/public_set.jsonl` (the 200 public development sessions) comes from the
same kit and belongs in `data/` as well.

## Reproducing our results

```bash
python3 -m unittest discover -s tests    # 46 tests
python3 -m evaluator.local_evaluator     # writes results.json
```

The evaluator is deterministic: two runs produce byte-identical output. A full
200-session run takes about 9 seconds and peaks near 550 MB.

Print one complete multi-turn session, with the agent's internal state at every
step:

```bash
python3 -m scripts.demo_session --scenario intent_override
```

Run the demo UI against the real agent (two terminals):

```bash
python3 -m scripts.serve
```

```bash
cd frontend && npm install && npm run dev
```

`TECHJAM_CATALOG` optionally overrides the catalog location; it is the only
environment variable and it is not required.

## Limitations, and what we would do next

**One session out of 200 is still missed.** Its target has a single customer
review, and the popularity prior — the single largest contributor to the score —
buries it. Recovering it would mean weakening that prior, which costs far more
sessions than it saves.

**Understanding is vocabulary-driven.** Attributes outside the built-in word
lists fall into a generic bucket. They still reach the query, but without
targeted weighting.

**Wording matters more than we would like.** Paraphrasing the simulator's
sentences costs roughly 0.06. We traced this to *slot recognition* rather than
vocabulary — the content words still reach the query 100% of the time, they just
stop being recognised as a category — and widening the opener patterns recovered
most of it. Further robustness would need a parser that is not pattern-based.

**Weights were tuned on 200 public sessions.** Ablation and an oracle experiment
both indicate the weight space is flat, so the fitting error should be small,
but a directional bias remains. The `boundary` scenario has only 10 public
sessions, so its perfect score is high-variance.

**Given more time**, the most valuable direction is not more tuning. Every
change that improved the score corrected a *distortion in a signal already in
use*; every change to the *retrieval strategy* — diversity reranking, hard
filtering, per-track weights, pseudo-relevance feedback, dense retrieval, LLM
reranking — measured neutral or negative, because an exact-match protocol
rewards ordering an existing candidate set rather than reshaping it. We would
instead invest in the parser: it is the one component whose failures are not yet
bounded by the protocol.

## Model choice

No model is used on the scored path, and that is a measured decision rather than
an omission. A local embedding model discriminated worse than the existing
lexical features (1.14 against 1.31), and a local LLM lost to the rules on every
phrasing the rules cover (96.7% against 100%). The LLM does win where the rules
fail completely (93.3% against 0%), so `src/llm.py` exists as a **fallback** —
consulted only when the rules extract no category at all, disabled unless
`TECHJAM_LLM_ENDPOINT` is set, and silently ignored on any failure.

`METHOD.md` has the full measurements, including the experiments we rejected.

## Team member contributions

<!-- TODO: fill in before submission. -->

## Data

Derived from Amazon Reviews 2023 (McAuley Lab, UCSD). See `DATA_ATTRIBUTION.md`.
Competition data is not redistributed here — download it from the organizer's
release as described above.

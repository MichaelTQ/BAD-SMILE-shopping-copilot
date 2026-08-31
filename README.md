# BAD SMILE — Shopping Copilot

BAD SMILE is a multi-turn conversational product-search agent for the TechJam
Conversational E-Commerce Search Challenge. Across a conversation, it extracts
what the shopper has explicitly said, remembers changing requirements, retrieves
from the frozen 50,000-product Amazon catalog, and returns a ranked Top 10 with a
short, evidence-based explanation.

The **scored configuration is fully offline and deterministic**: Python's
standard library and SQLite FTS5 only, with no model call, external API, network
dependency, token usage, or runtime cost. An optional local-LLM fallback is
included for phrasing that the rule parser cannot recognize, but it is disabled
by default and was not used for the results below.

## Public development results

Measured with the organizer's deterministic evaluator on the 200 labeled public
development sessions:

| Metric | Weak BM25 baseline | BAD SMILE |
| --- | ---: | ---: |
| Hit Rate@10 | 0.125 | **0.995** |
| MRR | 0.068034 | **0.770948** |
| MTTC (lower is better) | 9.81 | **1.850** |
| Efficiency | 0.119 | **0.915** |
| **TechnicalScore** | 0.106710 | **0.911784** |
| Model tokens / cost | 0 / $0 | **0 / $0** |

Scenario Hit Rate@10 is **1.000** for Browsing, Intent Override, and Boundary,
and **0.9875** for Buying. These are public-development results, not a claim
about the organizer's separate 800-session private evaluation set.

## Project overview

### What happens on each turn

```text
shopper message
    -> rule-based category and constraint extraction
    -> per-session state update
    -> multi-route BM25 candidate recall
    -> local constraint-aware reranking
    -> Top-10 recommendations + honest rationale + clarification question
```

The agent uses the **whole conversation**, not only the latest message. Its
session state distinguishes hard requirements from soft preferences, records
attributes that have already been answered or declined, and handles intent
override by demoting superseded constraints while keeping the new requirement
current.

Retrieval and ranking are deliberately separated:

1. **Recall** merges a weighted full-query route with a category-only route.
   When a dialogue stops producing information, a stricter all-category-terms
   route opens to recover precise products that the broader route may miss.
2. **Reranking** combines BM25 relevance, IDF-weighted query coverage, complete
   phrase matches, category overlap, product popularity, budget fit, and a
   repeat-exposure penalty used only when the conversation stalls.
3. **Dialogue output** names only constraints the first-ranked product actually
   matches. If the Top 10 scores are effectively tied, the agent says that it
   needs another detail instead of presenting an arbitrary order as confident.

The design is measurement-led. Dense retrieval, profile-based personalization,
Buying/Browsing-specific weights, pseudo-relevance feedback, hard filtering,
and several phrase variants were implemented or probed and rejected when they
failed to improve robustly. [METHOD.md](METHOD.md) contains the architecture,
ablations, split-half checks, negative results, and rationale for every default.

### Repository map

```text
starter/agent.py          official Agent.reset / Agent.respond entry point
src/parsing.py            rule-based turn understanding
src/state.py              isolated per-session conversation memory
src/policy.py             clarification-question policy
src/index.py              in-memory SQLite FTS5 catalog index
src/ranker.py             candidate recall and local reranking
src/explain.py            customer-facing rationale and overload message
src/llm.py                optional local-LLM fallback, disabled by default
tests/                    46 tests over the agent and its modules
scripts/demo_session.py   printable end-to-end session trace
scripts/serve.py          local HTTP bridge for the demo UI; not scored
frontend/                 React/Vite demo interface; not scored

docs/                     organizer-provided specification (unmodified)
evaluator/                organizer-provided public evaluator (unmodified)
data/                     organizer-provided frozen data (not redistributed here)
```

Everything above the blank line is ours. The competition specification, the
evaluator, and the frozen data belong to the organizer and are used as shipped —
no rule, metric, or label was altered.

## Setup and installation

### 1. Prerequisites

- Python 3.10 or later
- SQLite compiled with FTS5, as in standard CPython distributions
- No third-party Python packages for the Agent or evaluator
- Optional demo UI only: Node.js 20.19+ or 22.12+

Confirm that FTS5 is available:

```bash
python3 -c "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE check_fts USING fts5(text)'); print('FTS5 available')"
```

### 2. Add the organizer's frozen data

The catalog and public sessions are competition artifacts and are not
redistributed in this repository. Download `catalog.jsonl.gz` and the public-set
files from the participant-kit release, verify the catalog against the published
`SHA256SUMS`, and place them as:

```text
data/catalog.jsonl
data/public_set.jsonl
```

For example:

```bash
mkdir -p data
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

The Agent also accepts an explicit catalog path. `TECHJAM_CATALOG` can be used
when the organizer or a local runner stages the catalog elsewhere.

### 3. Python dependencies

No installation command is required for the scored Agent. `requirements.txt`
documents the standard-library-only runtime.

## Steps to reproduce our results

Run all tests:

```bash
python3 -m unittest discover -s tests -v
```

Expected result: **46 tests pass**.

Run the organizer-compatible evaluator:

```bash
python3 -m evaluator.local_evaluator
```

This writes the per-session and aggregate output to `results.json`. With the
frozen public artifacts, the expected summary is:

```text
Hit Rate@10    0.995
MRR            0.770948
MTTC           1.850
TechnicalScore 0.911784
Total tokens   0
```

The run is deterministic: two executions produce byte-identical result files.
On the development machine, all 200 sessions take about 9 seconds and peak near
550 MB resident memory, including the evaluator's own catalog copy.

Inspect one complete multi-turn interaction, including parsed state and ranking:

```bash
python3 -m scripts.demo_session --scenario intent_override
```

### Optional end-to-end demo UI

The UI is not part of the scored path. It talks to the real Python Agent through
the included local bridge.

Terminal 1:

```bash
python3 -m scripts.serve
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite, normally `http://localhost:5173`.

## Model, network, and environment disclosure

The public score above uses no model and no network. The optional local-LLM path
exists only as a fallback when the rules extract no product category. It never
replaces retrieval or ranking, and any unavailable service, timeout, or invalid
response silently leaves the rule-based result unchanged.

| Variable | Required? | Purpose |
| --- | --- | --- |
| `TECHJAM_CATALOG` | No | Override the path to `catalog.jsonl` |
| `TECHJAM_LLM_ENDPOINT` | No | Opt into an Ollama-compatible local fallback |
| `TECHJAM_LLM_MODEL` | No | Override the fallback model; default `qwen3:latest` |

For reproducible official scoring, leave both LLM variables unset.

## Limitations and what we would improve

- **Natural-language understanding remains rule-driven.** Common shopping
  phrasings are covered, but unseen syntax may fail to populate the correct
  category or attribute slot. A future version would replace the regex list
  with a small, offline structured parser that preserves the current fallback.
- **One of 200 public targets is still missed.** It has only one review, so the
  popularity prior buries it; weakening that prior rescues one case while
  damaging many others. Better calibration or a stronger late-stage reranker
  is more promising than another global weight sweep.
- **Clarification is state-aware but not fully adaptive.** The policy avoids
  attributes already answered or declined, but it does not yet estimate the
  expected value of every possible question from the current candidate set.
- **Hard requirements are weighted, not absolute filters.** This protects recall
  when product metadata is incomplete, but a real retail deployment should
  expose which constraints are mandatory and let the shopper choose strict or
  forgiving behavior.
- **Personalization is intentionally inactive.** Aggregate profile tags looked
  useful against random products but lost their signal inside the relevant
  candidate pool and reduced the public score. More behavioral history or
  category-specific profile features would be needed for safe personalization.
- **Explanations are descriptive rather than causal.** They truthfully state
  matched constraints, price, and rating volume, but do not yet compare why the
  first result outranks the second.
- **Public-set tuning remains a generalization risk.** We use ablations,
  deterministic runs, and split-half checks, but only the organizer's private
  800 sessions can establish final generalization.
- **The in-memory index favors speed over footprint.** Peak development memory
  is about 550 MB. Given more time, we would test a disk-backed or more compact
  metadata representation under the organizer's final resource limits.

The next investment would be parser robustness and calibrated late-stage
reranking, not additional tuning of already-flat global weights.

## Team member contributions

- [**Wenlong Qiu**](https://devpost.com/michael05242016) — Team Leader (`@michael05242016`)
- [**YUTIAN YE**](https://devpost.com/2004yutianye) (`@2004yutianye`)
- [**Huibing Wang**](https://devpost.com/iriswang24) (`@iriswang24`)
- [**Yikai Wang**](https://devpost.com/wyk18700597189) (`@wyk18700597189`)
- [**zicen qin**](https://devpost.com/zicenqin) (`@zicenqin`)

## Data and attribution

The frozen competition catalog and sessions are derived from Amazon Reviews
2023 by McAuley Lab, UCSD. See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) for
source and licensing details. Competition data must be obtained from the
organizer's participant-kit release.

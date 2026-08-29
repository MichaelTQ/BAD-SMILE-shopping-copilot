# Method Report — Offline Rule-Based Conversational Agent

## Summary

`starter.agent.Agent` is a fully offline, deterministic conversational shopping
agent. It uses **no LLM, no API, no network service, and no agent framework**
(no LangChain / LangGraph). Reported token usage is always `0`.

Retrieval is two-stage: SQLite **FTS5 BM25** recall over the 50k-row catalog,
followed by a local constraint-aware rerank. The query is built from the whole
accumulated dialogue, not just the current message.

## Architecture

```text
starter/agent.py     Agent: reset() / respond() — official interface only
src/text.py          tokenisation, stopword/filler lists, IDF
src/parsing.py       rule-based turn understanding -> ParsedTurn
src/state.py         SessionState: one per session_id
src/policy.py        which attribute to ask next + fixed question templates
src/index.py         FTS5 index + metadata table + document frequencies
src/ranker.py        BM25 recall (Top 200) -> local rerank -> Top 10
scripts/demo_session.py   prints one full multi-turn session offline
tests/test_agent.py       28 unit tests (plus the 3 shipped evaluator tests)
```

The evaluator, public data, and ground truth are untouched.

### Session state (`src/state.py`)

`Agent.sessions` maps `session_id -> SessionState`. `reset()` replaces the
entry, so sessions are fully isolated and re-running a session id starts clean.
Each state holds:

| field | content |
| --- | --- |
| `user_profile` | anonymized aggregate profile passed to `reset()` |
| `history` | every user and agent turn |
| `constraints` | currently effective constraints (`attribute`, `value`, `hard`, `stale`) |
| `category`, `budget` | the two slots that behave as filters rather than terms |
| `asked` / `answered` | per-attribute ask counts and which ones produced an answer |
| `no_preference` | attributes the customer explicitly declined |
| `recommended` | `parent_asin -> how many turns it was shown in` |

### Turn understanding (`src/parsing.py`)

Regex + vocabulary rules, applied in a fixed order so the envelopes are stripped
before the free text is read:

1. **no-preference** — `"I don't have a preference for X"`, `"no additional
   preference for X"`, `"use your judgment"` → mark attribute `X` declined.
2. **hard constraints** — `"a key requirement is: …"`, `"what I need is: …"`,
   `"must have …"`.
3. **soft constraints** — `"what matters is: A; B"`, `"I prefer …"`,
   `"ideally …"`; the `;` separator splits multiple disclosures.
4. **category** — `"looking for X"`, `"shopping for X"`, `"I need X"`, up to the
   first `.`/`,`.
5. **residual** — anything left that carries ≥2 content tokens is kept as a soft
   constraint (this is how the intent-override scenario's bare opening sentence
   is captured). Pure filler is discarded.

Each constraint is typed into one of `category, material, color, size, style,
brand, budget, feature, use_case` by vocabulary lookup. Concrete values
(materials, colors, sizes, `$` amounts) are additionally extracted and given
extra query weight.

**Intent override** (`"actually"`, `"ignore my earlier"`, `"instead"`,
`"changed my mind"`, …) marks every prior constraint `stale` and lets the new
statement enter as `hard`. Stale constraints are down-weighted
(`OVERRIDE_DECAY`) rather than deleted, because a shopper who re-prioritises is
usually still describing the same product; deleting the history measured
distinctly worse (override hit rate 0.767 vs 0.967).

### Query construction (`src/ranker.py`)

Term weights come from the *cumulative* state, never a single message:

| source | weight |
| --- | --- |
| category | 3.0 |
| hard constraint | 2.5 |
| soft constraint | 1.0 |
| stale (post-override) constraint | × 0.6 |
| concrete color / material / size value | +1.5 |

### Retrieval and rerank

1. **Recall** — FTS5 `MATCH` with an OR of the top-weighted terms, ordered by
   `bm25()` with the starter's column weights, `LIMIT 200`; plus a
   category-only query (`LIMIT 100`) so category-consistent products stay in the
   pool even when constraint terms dominate.
2. **Rerank** — linear score over normalized features:

   | feature | weight | meaning |
   | --- | --- | --- |
   | `bm25` | 1.0 | recall score, normalized by the pool maximum |
   | `coverage` | 3.0 | IDF-weighted fraction of query mass present in the product |
   | `phrase` | 1.2 | fraction of multi-word constraints appearing verbatim |
   | `category` | 2.2 | overlap with the product's category path (generic labels dropped) |
   | `popularity` | 1.2 | `log1p(rating_number)` × `average_rating/5` |
   | `budget` | 0.5 | price within budget rewarded, over-budget penalized; unknown price neutral |

3. Top 10 returned, sorted deterministically (score, then `parent_asin`).

`recommended` is tracked but deliberately applies **no** novelty penalty:
in the intent-override scenario a correct product surfaced early must stay at
the top until the override turn makes it scoreable, and demoting it would cost
hits.

### Asking policy (`src/policy.py`)

Every turn returns both recommendations and one legal `ask_attribute`, with a
fixed question template per attribute (no generated text anywhere). Order:

```text
other → feature → material → color → style → use_case → size → brand → budget → category
```

`other` leads because it is the broadest legal probe. An attribute is never
asked again once the customer declined it or answered with no preference; an
attribute that produced information may be revisited at most once
(`MAX_ASKS = 2`) after all fresh attributes are exhausted.

### Robustness

`respond()` wraps state update + rerank in `try/except`; any failure falls back
to plain BM25 on the raw message, and an empty rerank result does the same. The
response shape is validated by unit tests across normal, no-preference,
override, nudge, and empty-message turns.

## Results — public set (200 sessions)

Command: `python3 -m evaluator.local_evaluator`

| metric | BM25 baseline | this agent | change |
| --- | --- | --- | --- |
| Hit Rate@10 | 0.125 | **0.985** | +0.860 |
| MRR | 0.068034 | **0.752599** | +0.684565 |
| MTTC | 9.81 | **1.90** | −7.91 |
| Efficiency | 0.119 | **0.910** | +0.791 |
| **TechnicalScore** | 0.10671 | **0.90028** | **+0.79357** |
| Total tokens | 0 | 0 | — |

By scenario (Hit Rate@10 / MRR / MTTC):

| scenario | n | baseline | this agent |
| --- | --- | --- | --- |
| buying | 80 | 0.2375 / 0.1265 / 8.63 | **0.9875 / 0.7476 / 1.38** |
| browsing | 80 | 0.0250 / 0.0045 / 10.75 | **1.0000 / 0.7102 / 1.59** |
| intent_override | 30 | 0.1333 / 0.1042 / 10.07 | **0.9667 / 0.8900 / 3.83** |
| boundary | 10 | 0.0000 / 0.0000 / 11.00 | **0.9000 / 0.7200 / 2.80** |

Intent-override MTTC cannot go below ~3.5 by construction: the evaluator only
scores hits from the override turn (3 or 4) onward.

Two consecutive runs produce byte-identical `results.json` (deterministic).

## Cost, latency, network

- **Model:** none. **API cost:** $0. **Tokens:** 0 prompt / 0 completion.
- **Network:** not used at any point, at build or at run time.
- **Dependencies:** Python standard library only (SQLite must be built with
  FTS5, which is the default in CPython 3.10+).
- **Latency:** index build ≈ 2.5 s once per process (50k rows); full 200-session
  evaluation ≈ 10 s wall clock on a laptop, i.e. a few milliseconds per turn.
- **Memory:** roughly 400 MB for the in-memory index, metadata and token caches.

## Limitations

- Understanding is vocabulary-driven. Materials, colors, sizes, styles and
  use-cases outside the built-in lists fall back to the generic `feature`
  bucket — still searchable, but without the targeted boost.
- Brand is recognised only when the message says "brand"/"made by"; store names
  are not matched against the catalog.
- Rerank weights were chosen on the 200 public sessions. They are few and
  coarse, but they are still fitted to that sample.
- An override is treated as re-prioritisation, not contradiction. A customer who
  says "not black, I want blue" gets the old value down-weighted rather than
  excluded.
- Exact numeric size matching ("size 8.5 wide") is not enforced as a filter.

## Reproducing

```bash
python3 -m unittest discover -s tests -v      # 31 tests, all offline
python3 -m evaluator.local_evaluator          # writes results.json
python3 -m scripts.demo_session --scenario intent_override
```

Python 3.10+; no installation step and no environment variables required.

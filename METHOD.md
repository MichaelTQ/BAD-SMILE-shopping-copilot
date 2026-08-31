# Method Report — Offline Rule-Based Conversational Agent

## Summary

`starter.agent.Agent` is a fully offline, deterministic conversational shopping
agent. It uses **no LLM, no API, no network service, and no agent framework**
(no LangChain / LangGraph), and no machine-learned model of any kind. Reported
token usage is always `0`.

Retrieval is two-stage: SQLite **FTS5 BM25** recall over the 50k-row catalog,
followed by a local constraint-aware rerank. The query is built from the whole
accumulated dialogue rather than only the current message.

On the 200 public sessions: **Hit Rate@10 0.995, MRR 0.770948, MTTC 1.850,
TechnicalScore 0.911784**, against the `0.10671` BM25 baseline.

An optional local-LLM fallback exists for turn understanding. It is **disabled
by default**; with no endpoint configured the agent is exactly the offline
rule-based system described here.

## Architecture

```text
starter/agent.py     Agent: reset() / respond() — official interface only
src/text.py          tokenisation, stopword/filler lists, IDF
src/parsing.py       rule-based turn understanding -> ParsedTurn
src/state.py         SessionState: one per session_id
src/policy.py        which attribute to ask next + fixed question templates
src/index.py         FTS5 index + metadata table + document frequencies
src/ranker.py        BM25 recall (Top 200) -> local rerank -> Top 10
src/explain.py       customer-facing rationale for the top result
src/llm.py           optional local-LLM fallback, inert unless configured
scripts/demo_session.py   development tool: prints one full session
tests/test_agent.py       43 unit tests (plus the 3 shipped evaluator tests)
```

The evaluator, public data and ground truth are untouched. `starter/agent.py`
and `src/` import neither the evaluator nor `data/public_set.jsonl`; the unit
test `test_agent_module_never_imports_the_evaluator_or_the_labels` enforces
this. `scripts/demo_session.py` does read the hidden intent card and ground
truth, but it is a development tool and is never imported by the agent.

### Session state (`src/state.py`)

`Agent.sessions` maps `session_id -> SessionState`; `reset()` replaces the
entry, so sessions are isolated. Each state holds the anonymized
`user_profile`, the full `history`, the effective `constraints`
(`attribute` / `value` / `hard` / `stale`), the `category` and `budget` slots,
`asked` / `answered` / `no_preference` bookkeeping, `recommended` counts,
`last_recommendations` for fallback, and `stalled_turns`.

### Turn understanding (`src/parsing.py`)

Regex plus attribute vocabularies, applied in a fixed order so envelopes are
stripped before free text is read: no-preference → hard constraints → soft
constraints → category → residual. Anything left with >=2 content tokens
becomes a soft constraint; this is how the intent-override opening sentence is
captured, and it doubles as the safety net when paraphrasing breaks the
templates.

Vocabulary lookups match **whole words**, not substrings: `"fit"` must not fire
inside `outfit`/`benefit`, nor `"xs"` inside `boxset`. Multi-word entries
(`"fit true"`, `"v-neck"`) are long enough to stay safe as substrings.

**Intent override** (`"actually"`, `"ignore my earlier"`, …) marks prior
constraints `stale` and lets the new statement enter as `hard`. Stale
constraints are down-weighted (`OVERRIDE_DECAY`), not deleted: a shopper who
re-prioritises usually still describes the same product. Deleting history
measured distinctly worse (override hit rate 0.767 vs 1.000).

### Retrieval and rerank (`src/ranker.py`)

Query weights come from the cumulative state: category 3.0, hard constraint
2.5, soft 1.0, stale x0.6, concrete color/material/size +1.5.

1. **Recall** — three routes merged into one pool:
   - the primary query, an OR of top-weighted terms ranked by `bm25()` with the
     starter's column weights, `LIMIT 200`;
   - a category-only OR query, `LIMIT 100`;
   - a strict route requiring **every** category term (AND), used **only on
     turns that added no information**. Widening recall dilutes ranking, so it
     is paid for only once the dialogue has stalled and the current Top 10 is
     evidently wrong. A narrow category is taken whole; a broad one contributes
     only its head.

   Candidates from the secondary routes are given their **real** BM25 score on
   the primary query. Leaving them at 0.0 applies a full `W_BM25` penalty for a
   measurement that was never taken — with Top-10 score spread of only 0.183,
   that alone kept correctly-recalled targets out of the Top 10. The primary
   query is therefore fetched to depth 3000 to serve as a score lookup; FTS5
   ranks every match anyway, so the deeper limit costs ~4 ms.
2. **Rerank** — linear score over `bm25` (1.0), IDF `coverage` (3.0),
   `phrase` containment (1.2), `category` overlap (2.2), `popularity` (1.2),
   `budget` fit (0.5), and a `seen` penalty applied **only on turns that added
   no information**.
3. Top 10, sorted deterministically by (score, `parent_asin`).

`message` then carries a short rationale for the top result built by
`src/explain.py` — which stated constraints it literally contains, its price
against any stated budget, and its rating volume. The evaluator only
type-checks `message`, so this costs no score; it exists because `message` is
the only human-facing output.

**Popularity** is `log1p(rating_number) / log1p(Q₀.₉₉) × average_rating/5`,
where `Q₀.₉₉` is computed from the catalog at build time so the scale adapts if
the catalog changes. See "Popularity scale" below for why it is not clipped.

**Category matching** drops only the first element of `categories` (the
catalog-wide root `"Clothing, Shoes & Jewelry"`, 49990 of 50000 rows). Earlier
code filtered the tokens `clothing` / `shoes` / `jewelry` globally, but those
also appear as genuine deeper levels (20523 / 11810 / 5127 times) where they
are the catalog's main division; that halved `category_score` for 23.9% of
products. After the fix, 95.7% of products reach a full 1.00 match.

### Asking policy (`src/policy.py`)

Every turn returns recommendations plus one legal `ask_attribute` from a fixed
template. Order: `other → feature → material → color → style → use_case → size
→ brand → budget → category`; `other` leads because it is the broadest legal
probe. A declined attribute is never asked again; an already-answered attribute
is skipped while any fresh one remains, and may be revisited at most once
(`MAX_ASKS = 2`).

## Runtime adaptation

The agent does not run one fixed pipeline. Four decisions are made per turn from
observable session state, so the strategy changes as the conversation does.

| Runtime signal | What it re-routes | Measured effect |
| --- | --- | --- |
| `stalled_turns > 0` — the turn added no information | Opens a third recall route requiring every category term | **+0.0034**; always-on costs −0.013 |
| Top-10 score spread below `OVERLOAD_CV` | Switches `message` from a ranking to a structured clarification | Fires on 59.2% of turns; miss rate 46.8% when it fires against 9.3% when it does not |
| `recommended` counts, gated on stall | Applies a novelty penalty so a wrong Top 10 rotates | Recovers `public_0144`; recall widening alone does not |
| `answered` / `no_preference` / `asked` | Chooses the next question and retires exhausted attributes | Keeps the ask on the highest-disclosure attribute |

Two of these deserve their conditions spelled out, because the conditions are
what makes them work rather than harm.

**Recall widening is gated, not permanent.** Running the strict category route
on every turn costs 0.013 — it dilutes rankings that were already correct. Gated
on stall it gains 0.0034, because by then the current Top 10 is demonstrably
wrong and there is nothing left to lose. The same gate governs the novelty
penalty.

**The clarification is calibrated, not decorative.** A flat Top 10 predicts a
miss: coefficient of variation is 0.0173 on turns that miss against 0.0561 on
turns that hit. So when the agent says it cannot tell the candidates apart, that
claim is right about half the time, and when it presents a ranking instead it is
right over 90% of the time. The wording tracks the score distribution, not a
turn counter — as soon as the customer supplies a discriminating constraint, the
ranking gains separation and the message returns to a substantive
recommendation.

What the agent deliberately does **not** do is cut retrieval off entirely on
overload. The specification suggests an "immediate retrieval cutoff", but under
this protocol a turn that returns no products cannot convert, so withholding the
list spends a turn to buy nothing. The agent instead returns its best ten *and*
says it cannot separate them — the same information, without the wasted turn.

## What actually drives the score

Ablation on the 200 public sessions, zeroing one signal at a time:

| configuration | hit | mrr | score |
| --- | --- | --- | --- |
| full | 0.9900 | 0.7533 | 0.9035 |
| no popularity prior | 0.9900 | 0.6271 | 0.8524 (−0.051) |
| **no dialogue constraints, category only** | 0.9900 | 0.7393 | 0.8982 (−0.005) |
| category + popularity only | 0.9900 | 0.7228 | 0.8931 (−0.010) |
| no category, dialogue constraints only | 0.9900 | 0.7158 | 0.8920 (−0.011) |

**Hit Rate is 0.9900 under every ablation.** Category and dialogue constraints
are mutually redundant — either alone recalls the target into the Top 10. The
popularity prior is the single largest contributor, while accumulating
multi-turn constraints is worth about +0.005.

A feature audit over 43132 turn-1 candidates shows why nominal weights mislead:

| feature | weight | σ within Top-10 | saturation |
| --- | --- | --- | --- |
| coverage | 3.00 | 0.0416 (16%) | 54.2% |
| category | 2.20 | 0.0111 (4%) | 62.3% |
| popularity | 1.20 | **0.1230 (48%)** | 0.5% |
| bm25 | 1.00 | 0.0493 (19%) | 0.5% |

Total score σ is 0.850 across the pool but only 0.183 within the Top 10: the
score separates candidates well enough to decide *who enters* the Top 10
(Hit Rate 0.990) but barely orders them once inside (MRR 0.763).

So the system is best described as: *extract the category from the dialogue,
recall it with BM25, order by popularity.* The multi-turn state, hard/soft
constraints and override handling are implemented and specification-compliant,
but their contribution to the score is marginal. The popularity prior works
because ground truth is sampled from the 5-core split, where reviewed products
skew popular (target median `rating_number` 7078 vs catalog median 12) — a
dataset property, not user modelling.

## Negative results

These were measured and rejected. They are reported because they bound how much
head-room the current design has.

| hypothesis | result |
| --- | --- |
| Per-scenario weights (Buying/Browsing routing) | Oracle upper bound with the true label: **+0.0036**; all four scenarios preferred the same direction, so routing has nothing to route. Not built. |
| Popularity clipped at `min(1, ·)` with `Q₀.₉₅` | **−0.043.** 81% of targets sit above `Q₀.₉₅`, so clipping collapses most targets onto the same value as every other bestseller. Kept the percentile scale, dropped the clip, moved to `Q₀.₉₉`. |
| Lower phrase threshold (3 → 2 tokens) | −0.0006. Two-word phrases are not discriminative enough. |
| Reduce `coverage` to de-duplicate with `bm25` (r = 0.83) | −0.0033. Correlated is not redundant. |
| Reduce `category` weight | ±0.0001 across 0.5–3.5. It is a filter, not a ranker. |
| Weights ramping with constraint count | Full set +0.0006, but split-half gave **−0.0019 / +0.0032** — opposite signs, i.e. noise. Removed. |
| Drop `budget` (99.8% zero) | Exactly 0.0 change; kept, since its lift is 3.24x when it does fire. |
| Personalize with `user_profile.preference_tags` | Harmful either way: see below. |
| Pseudo-relevance feedback (predict what the shopper has not said) | −0.068 always-on, −0.003 even gated on stall: see below. |
| Browsing track: MMR diversity rerank | −0.0025; diversity is actively harmful here. |
| Buying track: hard constraints as an AND gate | −0.0047; +3 hits but average rank 2.13 → 2.35. |
| Over-generality detection as a recall trigger | ±0.0001; the signal is real but widening recall does not treat it. |
| Relax phrase matching from whole-constraint to 3-grams | No score change, and the feature gets **worse**: see below. |

### Profile personalization: a correlation that vanishes inside the pool

`preference_tags` (`"fit"`, `"comfort"`, `"material"`, …) correlate with the
target: they appear in the target's text 32.6% of the time against 17.0% for a
random product, a 1.92x lift. Two ways of using that were measured, and both hurt:

| use | result |
| --- | --- |
| Added to the query at weight 0.15 / 0.4 / 0.8 / 1.5 | 0.8983 / 0.8956 / 0.8921 / 0.8854 — **monotonically worse** |
| As a rerank feature at weight 0.1 / 0.3 / 0.6 / 1.2 | 0.8994 / 0.9001 / 0.8923 / 0.8729 — also worse |
| As a rerank feature at weight −0.3 | 0.9075 on the full set, but split-half gave **+0.0039 / −0.0018** — opposite signs, noise |

The reason is that the 1.92x lift is measured **against a random product**, and
almost all of it is explained by the candidate pool rather than by the customer:

```text
target hit rate                0.334
vs. random product             lift 1.92x
vs. the whole candidate pool   lift 1.81x
vs. the pool's Top 10          lift 1.17x
```

By the time candidates share a category and the stated constraints, tags carry
almost no additional signal — and they are generic, low-IDF words, so giving
them query weight dilutes the terms that do discriminate. This is a plain
conditional-independence trap: correlated marginally, near-independent given
the pool. `W_PROFILE` is kept in `src/ranker.py` at 0.0 with this rationale.

### Pseudo-relevance feedback: a lift measured against the wrong baseline

Every constraint of one intent card is derived from the same product, so
constraints are strongly correlated — terms shared by the current top candidates
should predict what the shopper has not disclosed yet. The initial measurement
looked strong: expansion terms drawn from the turn-1 Top 10 covered **54.8%** of
the target against **26.5%** of the pool (2.07x lift), and **11.4%** of them hit
a constraint that was still undisclosed at that point, against a random baseline
near 0.03%.

Implemented as a second retrieval pass — expand the query with those terms at a
weight below a stated soft constraint, then re-rank — it loses across the board:

| configuration | hit | mrr | score |
| --- | --- | --- | --- |
| off (baseline) | 0.9950 | **0.7709** | **0.9118** |
| weight 0.15, always on | 0.9900 | 0.5898 | 0.8532 (−0.059) |
| weight 0.30, always on | 0.9900 | 0.5614 | 0.8442 (−0.068) |
| weight 0.60, always on | 0.9950 | 0.5538 | 0.8439 (−0.068) |
| weight 0.30, gated on stall | 0.9950 | 0.7615 | 0.9085 (−0.003) |
| weight 0.60, gated on stall | 0.9950 | 0.7597 | 0.9082 (−0.004) |
| weight 1.00, gated on stall | 0.9950 | 0.7594 | 0.9081 (−0.004) |

The 2.07x lift was measured against the wrong baseline. Expansion terms are
extracted *from the Top 10*, so every document in the Top 10 is rich in them by
construction; comparing the target against the whole 200-candidate pool lets the
190 non-top documents depress the denominator and manufacture a lift. The terms
raise the entire Top 10 together and cannot separate it internally — which is
exactly where MRR is decided. Query drift then makes it worse: the expanded
query pulls toward the centroid of the candidate set, so targets that were
already ranked first slide down.

This is the same trap as the profile-tags result above — correlated marginally,
near-independent given the pool — and the 11.4% undisclosed-constraint hit rate
is real but does not translate into ranking power: guessing what the shopper
will say next is not the same as telling their product apart from ten others
that match equally well. Unlike the strict-category route, gating on stall does
not rescue it, and every weight from 0.15 to 1.00 is monotonically worse.

`PRF_ENABLED` remains in `src/ranker.py`, defaulting to `False`, so the code
path is inert and the score is unchanged to the last digit.

### The four pillars: every routing strategy measured negative

The problem statement asks for dual-track routing — a high-precision filter
track for Buying, a diverse retrieval track for Browsing. Both were built and
measured, and both lose.

**Browsing / MMR diversity.** At turn 1 the Top 10 holds only ~2.4 distinct
category paths, and browsing has both the least diverse pool (0.241) and the
lowest turn-1 hit rate (0.438) — which looked like a case for spreading the
results out. Maximal Marginal Relevance over the top 50, with category-path
Jaccard as the redundancy term, does exactly what it should:

| λ | Top-10 category diversity | mean target rank | hits |
| --- | --- | --- | --- |
| off | 0.228 | **2.55** | 40 |
| 0.9 | 0.235 | 2.58 | 40 |
| 0.7 | 0.252 | 2.70 | 40 |
| 0.5 | 0.287 | 2.83 | 40 |

Diversity rises, the target sinks, and the hit count never moves. Full-set score
falls monotonically (−0.0025 at λ=0.5). The reason is that the evaluator scores
an exact `parent_asin` match: the target lives in **one** category, so spending
Top-10 slots on other categories cannot raise the odds of hitting it — it can
only push it down. Diversity serves real browsing; it does not serve "find this
one specific item".

**Buying / hard-constraint gate.** Treating stated hard constraints as an `AND`
gate is safe on its face — the target survives the gate in **80 of 80** sessions
that state one, and the pool drops from 50000 to a median 8675. But:

| | mean target rank | hits | target bm25 | Top-10 bm25 σ |
| --- | --- | --- | --- | --- |
| gate off | **2.13** | 46 | 0.887 | 0.0554 |
| gate on | 2.35 | **49** | 0.883 | 0.0811 |

The gate finds three more targets and ranks the rest worse, for −0.0047 overall.
BM25 normalisation is not the culprit (scores and spread both hold up); the
newly admitted targets simply enter low, while clearing out competitors also
removes documents that were holding the target's relative position up.

**Over-generality detection.** A flat Top 10 predicts a miss well: coefficient
of variation is 0.0173 on turns that miss against 0.0561 on turns that hit, and
the lowest quartile carries a 50.7% miss rate against 31.2% overall. Wired as an
extra trigger for the strict-category route it fires on 23.2% of turns that the
stall counter does not — genuinely new coverage, not a redundant signal — and
changes the score by ±0.0001. The two signals target different failures:
`stalled_turns` approximates "recall failed for lack of information", which
widening recall can fix; a flat ranking means "the target is probably already in
the pool and cannot be ordered", which it cannot.

**The pattern across all of them.** Every change that improved the score —
Q₀.₉₉ popularity normalisation, stall-gated recall widening, scoring
secondary-route candidates properly, widening the category opener patterns —
corrects a **distortion in a signal already in use**. Every change that altered
the **retrieval strategy** — diversity, filtering, per-track weights,
pseudo-relevance feedback — measured neutral or negative. Under an exact-match
protocol, reshaping the candidate set adds no information; it only perturbs a
ranking that is already tuned.

### Proactive guidance: the one signal that found a use

The over-generality signal is real even though widening recall does not exploit
it, so it drives what the agent *says* instead. When the Top 10 are effectively
tied, `message` switches from presenting a ranking to admitting there isn't one:

```text
turn 1  These 10 are all Athletic Walking, and I can't tell them apart on
        what you've told me so far. What matters most to you in this purchase?
turn 3  Here are the closest matches... Top pick: Skechers Men's Go Max-Athletic
        Air Mesh... — matches 100% Textile and Imported; costs $54.97; ...
```

The switch is driven by the score distribution, not a turn counter — as soon as
the customer supplies `100% Textile; Imported`, the ranking gains separation and
the wording returns to a substantive recommendation. The claim is calibrated:

| | share of turns | miss rate |
| --- | --- | --- |
| clarification fired | 59.2% | **46.8%** |
| clarification did not fire | 40.8% | **9.3%** |

A five-fold difference: when the agent says it cannot tell the candidates apart,
it is wrong about half the time, and when it presents a ranking it is right over
90% of the time. The evaluator only type-checks `message`, so this costs no
score — it exists because `message` is the only human-facing output, and an
honest one is worth more than a confident one.

### Phrase matching: why whole-constraint beats n-grams here

`phrase` requires a constraint's tokens to appear **contiguously and in full**.
That is fragile by construction — constraint lengths are bimodal, and the long
tail is substantial:

```text
1-2 tokens: 64.5%   (below the threshold, no phrase generated)
3-5 tokens: 15.8%
>5 tokens:  19.8%   (up to 30 tokens; one reordered word zeroes the feature)
```

Constraints longer than five tokens outnumber the mid-length ones, so the
"all-or-nothing" concern is real. Three alternatives were implemented and
measured (`PHRASE_MODE`, still switchable in `src/ranker.py`):

| variant | score |
| --- | --- |
| `full` — whole constraint contiguous (current) | 0.9064 |
| `min_tokens = 2` | 0.9058 |
| `ngram` — hit ratio over sliding 3-grams | 0.9064 |
| `truncate` — keep the first 5 tokens | 0.9064 |

Scores are identical because MTTC is below 2: most sessions convert on turn 1-2,
when only ~0.56 constraints exist and `phrase` has not activated at all. So the
comparison was repeated on the feature itself, at turn 4 where it does fire:

| mode | pool activation | target activation | **lift** |
| --- | --- | --- | --- |
| `full` | 7.70% | 66.16% | **8.59x** |
| `ngram` | 13.12% | 66.67% | **5.08x** |
| `truncate` | 7.76% | 66.67% | **8.59x** |

n-grams nearly double pool activation (13.12%) while target activation barely
moves (66.16% → 66.67%): the extra credit goes almost entirely to non-targets,
and lift drops from 8.59x to 5.08x. The reason is the same dataset property
noted throughout this report — constraints are near-verbatim excerpts of the
target's own `features`/`details`, so targets already match in full. Loosening
the match granularity only admits noise.

`truncate` keeps the 8.59x lift and slightly raises the target's mean score, so
it addresses long-constraint fragility without the noise cost — but with no
score difference there is not enough evidence to switch. `full` remains the
default.

This conclusion is dataset-specific. If an evaluator paraphrases the constraint
*content* (not just the sentence frame), whole-constraint matching collapses to
zero and n-grams would win. Our paraphrase tests only rewrote the frame —
content words reached the query 100% of the time — so that regime is untested.

**Methodology note.** An early per-feature "discriminative power" metric
(target's percentile within the pool) is invalid for sparse features: with 98%
of candidates at zero, a target at zero also scores zero. Recomputed as
non-zero lift, `phrase` turns out to be the **strongest** discriminator
(target 64.65% vs pool 7.78%, lift 8.30x), not the weakest. Conclusions about
sparse features were revised accordingly.

## Results — public set (200 sessions)

`python3 -m evaluator.local_evaluator`

| metric | BM25 baseline | this agent |
| --- | --- | --- |
| Hit Rate@10 | 0.125 | **0.995** |
| MRR | 0.068034 | **0.770948** |
| MTTC | 9.81 | **1.850** |
| Efficiency | 0.119 | **0.915** |
| **TechnicalScore** | 0.10671 | **0.911784** |
| Total tokens | 0 | 0 |

By scenario (Hit Rate@10 / MRR / MTTC):

| scenario | n | baseline | this agent |
| --- | --- | --- | --- |
| buying | 80 | 0.2375 / 0.1265 / 8.63 | **0.9875 / 0.7663 / 1.39** |
| browsing | 80 | 0.0250 / 0.0045 / 10.75 | **1.0000 / 0.7123 / 1.59** |
| intent_override | 30 | 0.1333 / 0.1042 / 10.07 | **1.0000 / 0.9400 / 3.63** |
| boundary | 10 | 0.0000 / 0.0000 / 11.00 | **1.0000 / 0.7700 / 2.30** |

Intent-override MTTC cannot go below ~3.5: the evaluator only scores hits from
the override turn (3 or 4) onward. Two consecutive runs produce byte-identical
`results.json`.

### Paraphrase robustness

The specification notes the organizer may add natural-language paraphrasing. We
measured this by rewriting the simulator's sentences at run time (the evaluator
file is not modified):

| simulator wording | score | hit | mrr |
| --- | --- | --- | --- |
| original templates | 0.911784 | 0.9950 | 0.770948 |
| lightly paraphrased | 0.845380 | 0.9500 | 0.654933 |
| heavily paraphrased | 0.868279 | 0.9650 | 0.697264 |
| colloquial | 0.853257 | 0.9450 | 0.690190 |

The original degradation was ~0.09. Locating it precisely: **the content words
are never lost** — the proportion of true category terms reaching the query
stays at 100% across all rewrites. What breaks is *slot recognition*: turn-1
category parsing fell from 100% to 55% (light), 15% (heavy), 0% (colloquial).
Category terms survived but were demoted from a weight-3.0 category slot to a
weight-1.0 soft constraint.

This is why the loss is **not** a vocabulary-matching problem, and why
dense/semantic retrieval does not address it. Widening `CATEGORY_OPENERS` does:
it restores turn-1 category parsing to 100% at every rewrite level and recovers
most of the gap, with no change on the original templates (the −0.0001 there
comes from the category-root fix, not from the openers).

Caveat: the rewrites and the wider patterns were both authored by us, so the
absolute numbers are partly self-referential. The structural finding — loss
comes from slot recognition, not vocabulary — does not depend on that, since it
follows from category terms reaching the query 100% of the time regardless.

## Model choice, and the optional local-LLM fallback

Dense retrieval, semantic reranking and local models are all in scope. Both were
run against local models before being scoped down, and the measurements are
below.

### Dense retrieval: measured and rejected

A 768-dimension local embedding model (`embeddinggemma`) was run over the
candidate pool of 38 sessions (1178 embedding calls) and compared against the
existing features on their ability to separate the target:

| feature | target / pool mean | Top-10 coefficient of variation |
| --- | --- | --- |
| dense (embeddinggemma 768d) | 1.1379 | 0.0868 |
| `coverage` (existing) | **1.3054** | 0.0975 |
| `popularity` (existing) | **1.6797** | — |

Ranking by dense similarity alone moved the target **later in 24 of 38 sessions**
and earlier in only 2. The cause is the same dataset property noted throughout:
constraints are near-verbatim excerpts of the target's own copy, so a string like
`"100% Leather"` is exactly matched by BM25, while an embedding places it near
every leather product and erases the distinction. Semantic generalisation is a
liability here, not an asset. An earlier IDF-cosine probe (length-normalised,
standard library only) reached the same conclusion at 1.196 vs 1.217.

Full-catalog indexing would also take ~0.8 h at the measured 60 ms/embedding,
and enabling it needs either a running model service or bundled weights —
neither exists in the scoring environment, so the route would fall back on
every one of the 800 private sessions regardless.

`DENSE_ENABLED` is kept in `src/ranker.py`, defaulting to `False`, with the
attachment point documented: a fourth recall route merged into the same
candidate pool as the three lexical ones. The pool is a plain dict and the
reranker scores every candidate uniformly, so an extra source costs nothing
structurally — the reason it is off is the measurement above, not the wiring.

### Turn understanding: rules beat the LLM, except where rules fail

Category-slot recognition, rules vs a local LLM (`qwen3`), over 30 sessions at
five phrasing levels:

| wording | rules | LLM |
| --- | --- | --- |
| original templates | **100.0%** | 96.7% |
| lightly paraphrased | **100.0%** | 96.7% |
| heavily paraphrased | **100.0%** | 96.7% |
| colloquial | **100.0%** | 96.7% |
| extreme colloquial | 0.0% | **93.3%** |

The rules win on every phrasing they cover, and lose completely outside it. So
the LLM is wired as a **fallback, not a replacement**: `src/llm.py` is consulted
only when the rules extract no category at all, which holds the call rate at
exactly the rule-failure rate — zero on all four covered levels.

Reasoning mode had to be disabled: with it on, a call took 25.1 s, emitted 573
tokens and answered *"Athletic Wear"*; with `think: false` it took 0.9 s, emitted
16 tokens and answered *"Athletic Walking"* correctly.

### Disclosure

| | |
| --- | --- |
| Scored offline path | no model, 0 tokens, $0 |
| Optional fallback | local `qwen3` via an Ollama-compatible endpoint |
| Cost | $0 — runs locally, no API, no credentials |
| Latency | 1163 ms median per call; 0 ms when rules succeed |
| Tokens | 70 prompt / 21 completion per call (not reported in `usage`, which stays 0 for the scored path) |
| Call rate | equal to the rule-failure rate: 0% on the original templates |

### Network and fallback behaviour

**The submission requires no network access.** `src/llm.py` activates only when
`TECHJAM_LLM_ENDPOINT` is set. Unset — the default, and the state of the scored
run — the module is inert and the agent is purely rule-based. When it *is*
configured, every failure mode (absent service, probe timeout, call timeout,
unparseable response) silently retains the rule-based result. There is no path
by which the model's absence degrades the offline system.

`src/ranker.py` keeps recall and rerank separate, so a dense route could be
merged into the existing candidate pool without touching state, parsing or
policy — the evidence above is why one is not.

## Deployment robustness

- **Working directory independent.** `Agent()` takes no required argument and
  resolves the catalog via `TECHJAM_CATALOG`, then the repository root, then
  the cwd. An explicitly requested but missing path raises rather than silently
  falling back to a different catalog.
- **Import path independent.** `starter/agent.py` puts the repository root on
  `sys.path` before importing `src.*`.
- **Bounded memory.** Token caches are capped (`TOKEN_CACHE_SIZE = 8192`).
  Resident set peaks at ~548 MB during a full 200-session run and does not grow
  with session count.
- **No exception escapes `respond()`.** Rerank failure falls back to plain
  BM25; BM25 failure falls back to the previous turn's ranking; a completely
  empty result still returns a contract-valid payload. A missing `reset()` is
  tolerated rather than raising, since an exception scores as a miss.
- **Thread safe.** The shared in-memory SQLite connection is lock-guarded.

## Cost, latency, network

- **Model:** none. **API cost:** $0. **Tokens:** 0 prompt / 0 completion.
- **Network:** never used, at build time or run time.
- **Dependencies:** Python standard library only; SQLite must include FTS5
  (default in CPython 3.10+).
- **Latency:** index build ~2.5 s once per process; 200-session evaluation
  ~9 s, i.e. tens of milliseconds per turn.

## Limitations

- **One remaining miss, and it is not a retrieval failure.** `public_0020`'s
  target has `rating_number = 1`, so the popularity prior — the single largest
  contributor to the score — buries it. Its constraints (`cotton`, `grey` in
  women's novelty apparel) carry no discriminative power either. Recovering it
  would mean weakening the prior, which costs far more sessions than it saves.
- **Template sensitivity.** Slot recognition is tuned to the simulator's
  phrasing; paraphrasing costs ~0.09 (measured above, with a validated fix not
  yet shipped).
- **Vocabulary-driven understanding.** Attributes outside the built-in lists
  fall into the generic `feature` bucket. This affects only the asking policy:
  `classify()`'s output is not used by retrieval at all. The vocabulary's only
  retrieval effect is `VALUE_BONUS`, worth +0.0022 in total.
- **Weights fitted on 200 sessions.** The ablation and the oracle experiment
  both show the weight space is flat, so fitting error should be small, but a
  directional bias remains. `boundary` has only 10 public sessions.
- **Override is re-prioritisation, not contradiction.** "Not black, I want
  blue" down-weights rather than excludes. A restated constraint is matched by
  exact text, so `"leather"` does not un-stale `"100% Leather"` (~8% of one
  term's weight).
- **`user_profile` carries no usable ranking signal.** Measured both as query
  terms and as a rerank feature; see the negative result above. The profile is
  still stored per session and available to future strategies.
- **ASCII-only tokenisation.** Non-ASCII input yields no matches; the catalog
  is English-only, so this is consistent rather than silently wrong.
- **Explanations are descriptive, not causal.** `src/explain.py` reports which
  stated constraints the top result actually contains, its price and its rating
  volume. It never claims a constraint the product does not literally match
  (enforced by a unit test), but it also does not explain *why* the ranker
  preferred this product over the next one.

## Reproducing

```bash
python3 -m unittest discover -s tests -v     # 46 tests, all offline
python3 -m evaluator.local_evaluator         # writes results.json
python3 -m scripts.demo_session --scenario intent_override
```

Python 3.10+; no installation step. `TECHJAM_CATALOG` is the only environment
variable and is optional.

## Team Contributions

- [**Wenlong Qiu**](https://devpost.com/michael05242016) — Team Leader (`@michael05242016`)
- [**YUTIAN YE**](https://devpost.com/2004yutianye) (`@2004yutianye`)
- [**Huibing Wang**](https://devpost.com/iriswang24) (`@iriswang24`)
- [**Yikai Wang**](https://devpost.com/wyk18700597189) (`@wyk18700597189`)
- [**zicen qin**](https://devpost.com/zicenqin) (`@zicenqin`)

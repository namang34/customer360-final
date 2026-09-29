# Evaluation — results, and what they actually prove

Reported honestly, including the failures. `python run_evaluation.py` reproduces
every number below with no API keys and no network.

## Headline

| | scenario_01 | scenario_02 | scenario_03 | overall |
|---|---|---|---|---|
| `inferred_state` | 3/3 | 2/2 | 3/3 | **8/8** |
| `confidence_band` | 3/3 | 2/2 | 3/3 | **8/8** |
| `action` | 3/3 | 2/2 | 3/3 | **8/8** |
| `action_subtype` (where graded) | 1/1 | 1/1 | 1/1 | **3/3** |
| `hitl_status` (where graded) | 1/1 | 1/1 | 1/1 | **3/3** |
| false-positive checks | 2/2 | 1/1 | 1/1 | **4/4** |
| lead-time targets | 1/1 | 1/1 | 1/1 | **3/3** |

238 automated tests pass in under 20 seconds.

---

## The caveat that matters most

**The confidence thresholds were calibrated against these eight checkpoints.**

The mid-term submission listed "precise confidence-band thresholds" as an open
question to be resolved against the practice scenarios, and that is what was done.
Scoring 8/8 on the data used to fit the thresholds is therefore **confirmation
that the calibration was applied correctly, not evidence that it generalises.**

With eight graded points and two thresholds, the overfitting risk is real. What
mitigates it — partially, not completely:

- The rules are **structural rather than numeric**. "Two independent strong
  signals, plus either three corroborating source systems or one deliberate act
  by the customer" is a shape, not a tuned constant. There is no
  `if score > 8.7` anywhere.
- The **guardrail was not fitted at all**. "≥ 2 independent source systems" was
  chosen from the observation that every planted red herring is one isolated
  event on one system. It would have been the rule with zero graded checkpoints.
- Several detector thresholds are set from **domain reasoning**, not from the
  answer key: a $5,000 medical charge is large, a 30% income drop is a cut, a
  self-transfer to another bank differs in kind from paying a third party.

What would actually test generalisation is a fourth scenario, which we do not
have. The honest expectation is that `inferred_state` and the false-positive
behaviour hold up on hidden data, and that `confidence_band` is the field most
likely to slip.

---

## Failures and things that do not work

### 1. The standing-instruction detector never fires on this data

`standing_instruction_stopped` detects a recurring commitment that has silently
stopped. It contributes **nothing** to any of the three scores.

All three scenarios are missing their February standing instructions entirely:

```
scenario_01  mortgage   Nov 1, Dec 1, Jan 1, [   nothing   ], Mar 1, Apr 1
scenario_02  mortgage   Nov 1, Dec 1, Jan 1, [   nothing   ], Mar 1, Apr 1
scenario_03  rent/gym/utilities
                        Nov 1, Dec 1, Jan 1, [   nothing   ], Mar 1
```

That is a data-generation artifact, not behaviour: scenarios 01 and 02 resume
normally on 1 April, and only scenario_03 — who actually cancelled on 4 March —
stops for good. With a one-missed-cycle threshold the detector fired in **all
three** scenarios in the first days of February, including the two customers who are not
churning, and pushed scenario_03's 15 February checkpoint to high confidence when
ground truth expects low.

The artifact and the real signal are the same shape — one missed monthly cycle —
so no threshold separates them. The detector now requires **two** missed cycles,
which silences the false positives. The cost is that scenario_03's cancellation
leaves only one missed cycle (1 April) before `simulated_end` on 15 April, so it
never fires at all. Kept because it is correct in principle and would matter on
real data; reported here because pretending otherwise would be dishonest.

Test: `test_standing_instruction_detector_is_silent_on_the_february_artifact`.

### 2. Retrieval runs on a fallback embedder in the sandbox

ChromaDB's default embedding function downloads an ONNX MiniLM model on first
use. Where that download is unavailable, the system falls back to
`HashingEmbedder` — a deterministic bag-of-words embedding with no model.

On this corpus (8 policies, 5 patterns, all short domain text) it retrieves the
correct policy for every query tested. But it has **no semantic generalisation**:
it cannot know that "infant" relates to "baby". A policy corpus using different
vocabulary from the queries would retrieve worse. `SemanticMemory.embedder_name`
records which embedder was used, and every run prints it.

### 3. Early-February state churn in scenario_02

`inferred_state` changes four times in the first three weeks of scenario_02 —
`no_significant_event` → `churn_risk` → `job_loss_or_income_disruption` →
`churn_risk` → `new_child_life_event`. All at **low** confidence, all on two weak
signals, and settled by 19 February — before the first graded checkpoint on the
20th.

This is arguably correct behaviour (the system is genuinely uncertain and updates
as evidence arrives) but it would look unstable to a human watching. Hysteresis
scaled by confidence makes an established belief sticky — 1.5× to displace a
high-confidence conclusion — while leaving low-confidence beliefs free to move.
A stricter margin would have prevented reaching `new_child_life_event` in time
for the 20 February checkpoint.

### 4. Two or three graded checkpoints per scenario is a very small sample

Eight checkpoints, four false-positive checks and three lead-time targets across
three customers. Every percentage above is out of a single-digit denominator.
100% on eight points is meaningfully different from 100% on eight hundred.

### 5. The LLM path is less tested than the deterministic path

Every test runs with `C360_OFFLINE=1`, so the language models are exercised
manually rather than in CI. The deterministic fallbacks are what the 238 tests
cover. Measured: 83% line coverage across `src/c360` overall, but `llm.py` alone
is 38% -- the provider construction and live-call paths are the part no test
reaches. That single number is the honest shape of this limitation. The live path has since been measured rather than
assumed, and it scores worse -- see "The models lose a graded checkpoint" below.

### 6. The agent-dependent trigger is recorded, not enforced

`StateBoard.should_synthesise` asks whether at least two distinct perception
agents have flagged something in the window. The pipeline computes it on every
checkpoint and writes it to the trace, but does **not** gate synthesis on it.

That is a deliberate choice made after measuring the alternative, not an
oversight. Synthesis is also the step that carries an existing belief forward, so
skipping it on quiet days means a committed diagnosis silently lapses. Wired as a
true gate, the graded score drops from 8/8 to 6/8 — scenario_01's 15 February and
12 March checkpoints both lose `inferred_state` (12 March loses
`confidence_band` too). Only the Transaction Agent publishes anything in that
scenario before 20 March, so under any true gate synthesis never runs and both
checkpoints report the cold-start `no_significant_event`.

So the honest statement is that this system has two enforced triggers, the
event-driven and the time-driven, and one observed signal. The trigger is
available on every trace row for anyone who wants to see when the swarm actually
converged; it just is not a branch.

---

## What the system does well, and why

### Every red herring is defeated, structurally

| Scenario | Event | What it is | Must not trigger | Result |
|---|---|---|---|---|
| 01 | `EVT_000382` | $12,000 tuition transfer (ach_wire) | fraud_hold, rm_escalation | clean |
| 01 | `EVT_000402` | $2,500 resort refund (card_payments) | personalized_offer | clean |
| 02 | `EVT_000328` | $600 electronics purchase (card_payments) | fraud_hold | clean |
| 03 | `EVT_000447` | $5,200 tax refund (core_banking_ledger) | personalized_offer | clean |

Each is a single event on a single source system, so one structural property --
no corroboration -- accounts for all four, not four special cases. On this data
Gate 1 is what actually stops them: every one of these checkpoints records
`gate_reason="confidence below high"`, and `guardrail_blocks=0` across all three
scenarios. The guardrail is the rule that *would* stop them if confidence ever
got that far. Three separate defences apply:

1. **Confidence gate** — nothing but `no_action` below high confidence.
2. **Guardrail** — guarded actions need ≥ 2 independent source systems and ≥ 2
   distinct events.
3. **Contradiction check** — a sales offer to a customer in distress or
   disengagement is refused on what it *means*, regardless of evidence strength.

### Late-arriving signal events are handled correctly

Scenarios 01 and 02 each plant exactly one out-of-order event, and in both cases
it is a ground-truth **signal** event:

| Scenario | Event | What | Occurred | Arrived |
|---|---|---|---|---|
| 01 | `EVT_000420` | $450 diagnostics charge | 1 Mar | 3 Mar |
| 02 | `EVT_000341` | Mothercare purchase | 2 Mar | 3 Mar |

Both stay invisible until they arrive and are released at ingestion time. A
system sorting by `event_time` would claim to have known two days early, and every
lead-time figure built on that would be wrong.

### Lead time is met in all three cases

| Scenario | Checkpoint | Ideal | Achieved |
|---|---|---|---|
| 01 | 26 Mar | ≥ 1 day | 5 days |
| 02 | 27 Mar | ≥ 2 days | 8 days |
| 03 | 08 Mar | ≥ 3 days | 3 days |

Scenario_03's three days come from the Usage Agent firing on the
`manage_standing_instructions_cancel` click on 4 March — two days before the money
moves. That is the single clearest argument for the swarm topology: the agent that
saw it first reads a source system the Transaction Agent never touches.

### Cost

74 checkpoints per scenario. A few carry the previous belief forward with no
reasoning call at all — 5, 1 and 5 respectively, on the days no agent has
published anything. On the rest, the LLM is consulted only when the top two
candidate states are within 25% of each other, when an action is actually being
proposed, and when a proposal needs proportionality review. Measured live, that
comes to 70, 87 and 74 calls per scenario.

---

## Reproducing

```bash
python -m pytest tests/ -q       # 238 tests
python run_evaluation.py         # the table at the top
python run_evaluation.py --live  # same, using Gemini/Groq from .env
python watch.py data/scenario_03 --speed 2
```

Every run writes `out/<scenario>_inferred_events.json`,
`out/<scenario>_review_queue.json` and `out/<scenario>.trace.jsonl`. The trace
carries one record per checkpoint with the findings, affinity scores, guardrail
verdict, critique verdict and HITL routing — enough to answer "why did it do that
on 8 March?" without re-running anything.

## The ambiguity review queue

Production Bar Checklist 6.3 asks that low confidence, or an unresolved
disagreement, reach a human "independent of the dollar amount involved". The
graded path cannot do this: Gate 1 turns anything below high confidence into
`no_action`, and `no_action` is auto-approved. Composed, those two rules mean the
system asks for a human when it is certain and says nothing when it is unsure.

`review.py` closes that as a **parallel** channel rather than a change to
`hitl_status`. Ground truth grades `hitl_status` at three checkpoints only, all
`escalated`, and no flagged checkpoint is one of them. A test asserts every
flagged checkpoint still reads `no_action / auto_approved` in the graded file.

What it raises, across 74 checkpoints per scenario:

| | items | what they are |
|---|---|---|
| scenario_01 | 2 | medical hardship corroborated across card + ledger, at low then medium |
| scenario_02 | 4 | two contested reads in February, then new-child breadth without strength |
| scenario_03 | 1 | 12 Feb — support and usage agree on churn three weeks before the escalation |

The first implementation raised an item on **every** qualifying checkpoint and
produced 38 for scenario_01 — the same two source systems restated daily for five
weeks. That is the alert-fatigue failure the AML research behind `guardrail.py`
describes, reproduced in miniature. It now raises only when the picture changes:
a different reason, state, confidence band, or a new source system joining. A
test caps the queue at ten items per scenario so that regression cannot return
quietly.

## The live model path, measured on all three scenarios

Every number above this section comes from the offline deterministic path. This
section is the first time the language models were measured. It changed what we
know, and not all of it is flattering.

Reproduce with `python compare_live.py --scenario data/scenario_0N`, which runs
the same scenario twice against the same ground truth and reports every graded
field that differs.

| | scenario_01 | scenario_02 | scenario_03 |
|---|---|---|---|
| model calls | 70 | 87 | 74 |
| model failures | **0** | **0** | **0** |
| actions proposed, offline → live | 26 → 26 | 28 → 28 | 42 → **24** |
| graded fields differing | **0** | 18 | 75 |
| offline: state / exact | 100% / 100% | 100% / 100% | 100% / 100% |
| live: state / exact | 100% / 100% | **50% / 50%** | 100% / 100% |

Not one model call fell back across all three runs. These are genuine live
results, not the offline path wearing a label.

### The headline: the models lose a graded checkpoint

**scenario_02, 20 February. Ground truth says `new_child_life_event`; the live
path says `job_loss_or_income_disruption`.** The deterministic path gets it
right. That is a graded field, and the live run scores 50% on `inferred_state`
where offline scores 100%.

The failure has a clear shape. The live path called it job loss on **every day
from 15 February to 3 March** — seventeen consecutive checkpoints — and then
recovered. By the second graded checkpoint, 27 March, it agreed with the
deterministic path and with ground truth. So the models did not fail randomly;
they failed on the **early, weak-evidence phase** of the narrative and corrected
once the evidence was unambiguous.

It is also a *reasonable* misreading, which is what makes it worth documenting.
A new child does reduce income during leave, so "income disruption" is a
defensible reading of the early signals in isolation. The deterministic path
survives because it is not reasoning about narrative at all: it matches the
baby-related retail and life-event signals and counts them. The model's broader
reading overrode the narrower evidence, and the narrower evidence was right.

The honest conclusion is the uncomfortable one: **on this data, keyword matching
plus arithmetic beat a language model at the task the language model is
supposedly better at.** Lead time is exactly where a proactive system earns its
keep, and that is precisely where the models were weakest.

### Where they agreed, and where they diverged harmlessly

**scenario_01: zero differences.** Not merely the same score — byte-identical
decisions at all 74 checkpoints, the same 26 proposed actions. The keyword
fallbacks and the models reached the same conclusions throughout.

**scenario_03: 75 differences, none graded.** The live path proposed an action on
24 of 74 days against the deterministic path's 42, and still got all three graded
checkpoints exactly right. Eighteen fewer alerts across ten weeks with no graded
answer lost — the alert-fatigue argument behind `guardrail.py` showing up as a
measurement rather than a claim. The remaining differences are `action_subtype`
on ungraded days, where the model sometimes returns the *action* name as the
subtype (`proactive_retention_outreach` instead of
`retention_winback_contact`) and once returned nothing. `prompts.PROPOSER` asks
for action and subtype in one response and the model collapses them. Ground truth
checks `action_subtype` at one checkpoint per scenario and the model is right at
all three, so nothing is lost here — but the prompt is a real defect.

Taken together, three scenarios say something one scenario could not: the models
are **not uniformly more conservative, nor uniformly worse**. They were identical
on one narrative, harmlessly more cautious on another, and wrong on the early
phase of a third. A single run would have licensed a confident generalisation
that the other two contradict.

### What this means for the submitted results

The published numbers are the offline ones, and this is the evidence that was the
right default rather than a convenience. Offline scores 8/8 across all three
scenarios; live would score 7/8.

The models are retained because the architecture uses them for what they are good
at — reading free text in support tickets, in-app searches and social posts — and
keeps them away from the corroboration arithmetic and the guardrail entirely.
This measurement supports that split rather than undermining it: the failure was
at the layer where a model was asked to weigh a whole narrative, not at the layer
where one was asked to read a sentence.

### Honest limits on this measurement

- **Both halves used the hashing embedder.** The live path would normally use the
  ONNX model. Holding retrieval fixed is deliberate — varying the embedder and
  the models together would make any difference unattributable — but it means
  this measures the model path, not the full production configuration.
- **`gemini-3.1-flash-lite` in both roles.** The documented reasoning model,
  `gemini-3.1-pro`, does not exist under that name; the provider lists
  `gemini-3.1-pro-preview`. Its nearest available substitute, `gemini-3.5-flash`,
  averaged over two minutes per call on free-tier quota. So the fast/reasoning
  split the architecture describes was collapsed for these runs. A stronger
  reasoning model might well get 20 February right; that is untested, and the
  claim here is only about what was measured.
- **Groq was never reached.** `llama-3.3-70b-versatile` has been retired; the
  configured fallback is now `openai/gpt-oss-20b`. With zero Gemini failures the
  failover path never ran and remains untested against a live provider.
- **One run per scenario.** Temperature is 0 throughout, so runs should be
  reproducible, but that was not verified by repetition.

### Five defects this exercise found, all invisible until now

The live path had never executed before this. Nothing that only breaks with a
model in the loop had ever been exercised, and five separate faults had
accumulated behind the fallbacks:

1. **`.env` was never loaded.** `load_dotenv()` appeared nowhere in the codebase,
   so no key ever reached `get_llm()`. `--live` printed `[live LLM]` and ran fully
   deterministic. Fixed at every entry point.
2. **A missing optional package disabled the primary provider.** Both clients were
   constructed inside one `try`, so `ImportError` from the Groq fallback destroyed
   the working Gemini client, and `except ImportError: pass` swallowed the reason.
   Each provider is now built independently and failures are reported.
3. **Content-block extraction discarded every reply.** Gemini returns `.content`
   as `[{"type": "text", "text": ...}]`; the code called `str()` on it and handed
   the JSON parser Python repr. The model was answering correctly the whole time.
4. **No request timeout, and two stacked retry loops.** A hung call could block a
   74-day replay indefinitely, and LangChain's own retries multiplied
   `ResilientLLM`'s. Now a 25-second timeout with `max_retries=0` on the clients.

A fifth, found while building the comparison: `SemanticMemory` named its
collections without reference to the embedder, so running both paths in one
process collided a 256-dimension store with a 384-dimension one. Collections are
now namespaced by embedder — which is a correctness rule, not tidiness, since
vectors from two embedders are not comparable even when the dimensions happen to
match.

**The common cause is worth more than the five fixes.** Every fallback in this
system is deliberate: a model failure must never kill a run. But the same property
made a *misconfiguration* indistinguishable from a healthy run — it completed, it
printed `[live LLM]`, and it scored 8/8, because those were the offline scores.
`LLMUnavailable("all providers exhausted")` even recorded each provider's error
into the trace and then raised a message containing none of it.

Fail-soft and fail-silent are one design decision apart, and this system had
repeatedly chosen the second by accident. `compare_live.py` exists to close that
gap: it reports the provider, the raw reply, the call and parse outcomes
separately, and warns explicitly when a "live" run made no successful calls, on
the grounds that such a comparison is not evidence of anything.

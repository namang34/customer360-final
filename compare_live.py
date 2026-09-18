"""
Run one scenario twice -- offline and live -- and report every difference.

    python compare_live.py                        # check keys only, no API calls
    python compare_live.py --scenario data/scenario_03    # the real comparison

WHY THIS EXISTS
---------------
Every published number in this repository comes from the offline deterministic
path. The language models have tested fallbacks, so the system runs end to end
with no API access at all -- which is what makes the results reproducible, and
also what makes "we use Gemini and Groq" a claim with nothing behind it.

This script is the evidence. It runs the same scenario both ways against the
same ground truth and prints what changed: the graded fields, the checkpoint
dates where the two runs disagree, and the score either way.

A difference is not a failure. The point is to report honestly which of the two
paths produced the published numbers, and by how much the other one differs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ModuleNotFoundError:
    pass

from c360.pipeline import Pipeline  # noqa: E402
from c360.scoring import score_scenario  # noqa: E402

GRADED = ("inferred_state", "confidence_band", "action", "action_subtype", "hitl_status")


def _mask(value: str) -> str:
    """Never print a key. Enough to tell two keys apart, not enough to use one."""
    if not value:
        return "(absent)"
    return f"{value[:4]}…{value[-4:]}  ({len(value)} chars)"


def check_keys() -> bool:
    print("=" * 72)
    print("STEP 1 - are the keys visible to Python?")
    print("=" * 72)
    gemini = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    groq = os.getenv("GROQ_API_KEY") or ""
    print(f"  Gemini : {_mask(gemini)}")
    print(f"  Groq   : {_mask(groq)}")
    if not gemini and not groq:
        print("\n  NEITHER KEY FOUND. --live would silently fall back to the offline")
        print("  path. Check that .env sits next to this file and contains")
        print("  GOOGLE_API_KEY=... and GROQ_API_KEY=... with no quotes.")
        return False

    print("\n" + "=" * 72)
    print("STEP 2 - does one real call work?")
    print("=" * 72)
    from c360.llm import get_llm

    problems: list[str] = []
    llm = get_llm(role="fast", problems=problems)
    print(f"  provider: {getattr(llm, 'provider', '?')}  model: {getattr(llm, 'model', '?')}")
    if problems:
        print("  provider build problems:")
        for p in problems:
            print(f"    - {p}")
        if "ModuleNotFoundError" in " ".join(problems):
            print("\n  A provider package is missing. Install it and re-run:")
            if "langchain_google_genai" in " ".join(problems):
                print("      pip install langchain-google-genai")
            if "langchain_groq" in " ".join(problems):
                print("      pip install langchain-groq")
    # Two separate questions, reported separately: did the network call succeed,
    # and was the reply parseable? Collapsing them hides which half is broken.
    try:
        raw = llm.complete(
            "You return strict JSON and nothing else. No prose, no code fences.",
            'Return exactly this object: {"ok": true}',
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  CALL FAILED: {type(exc).__name__}: {exc}")
        print("  -> the request never completed. Check the key and the model name.")
        return False

    print(f"  raw reply: {raw[:300]!r}")
    try:
        from c360.llm import parse_json_response

        print(f"  parsed   : {parse_json_response(raw)}")
        print("  -> the live path works end to end.")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  PARSE FAILED: {type(exc).__name__}: {exc}")
        print("  -> the call succeeded, so the key is fine. The reply was not")
        print("     strict JSON. Every caller in the system falls back to its")
        print("     deterministic path when this happens, so a run would still")
        print("     complete -- it just would not be using the model.")
        return False



def list_available_models() -> None:
    """
    Ask each provider what it actually serves.

    Model names are configuration, not code, and they get retired: Groq answered
    404 for llama-3.3-70b-versatile, which was a valid name when this was
    written. Printing the live catalogue turns "why did it fail" into "pick one
    of these and put it in .env".
    """
    print("\n" + "=" * 72)
    print("STEP 2b - what models do these keys actually have access to?")
    print("=" * 72)

    print("  GEMINI:")
    try:
        from google import genai

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        names = []
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                names.append(name.replace("models/", ""))
        for n in sorted(names)[:25]:
            print(f"    {n}")
        if not names:
            print("    (none reported)")
    except Exception as exc:  # noqa: BLE001
        print(f"    could not list: {type(exc).__name__}: {exc}")

    print("  GROQ:")
    try:
        import groq as _groq

        client = _groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        for m in sorted(client.models.list().data, key=lambda x: x.id):
            print(f"    {m.id}")
    except Exception as exc:  # noqa: BLE001
        print(f"    could not list: {type(exc).__name__}: {exc}")

    print("\n  Set the ones you want in .env, e.g.")
    print("    C360_FAST_MODEL=<a small fast gemini model>")
    print("    C360_REASONING_MODEL=<a stronger gemini model>")
    print("    C360_GROQ_MODEL=<a groq model id from the list above>")


def diagnose_retrieval() -> None:
    """
    Report the state of semantic memory.

    The action proposer only runs when retrieval returns an eligible policy. If
    the policy store is empty or the query matches nothing, every checkpoint
    becomes no_action and the action score collapses -- with no error anywhere,
    because an empty result is a valid result.
    """
    print("\n" + "=" * 72)
    print("RETRIEVAL CHECK - is the policy store actually populated?")
    print("=" * 72)
    from c360.semantic import SemanticMemory

    for prefer_default in (False, True):
        label = "default/ONNX" if prefer_default else "hashing fallback"
        try:
            sem = SemanticMemory(prefer_default_embedder=prefer_default)
            seeded = sem.seed()
            print(f"  embedder requested: {label}")
            print(f"    actually used   : {sem.embedder_name}")
            print(f"    collection      : {sem.policies.name}")
            print(f"    documents       : {sem.policies.count()}")
            print(f"    seed() returned : {seeded}")
            hits = sem.retrieve_policies("churn risk cancellation transfer out", k=4)
            print(f"    query hits      : {len(hits)}")
            for h in hits[:4]:
                action = (h.metadata or {}).get("action", "?")
                print(f"      - {h.doc_id}  action={action}")
            if not hits:
                print("    ^^ NO HITS. Every checkpoint will produce no_action.")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label}: FAILED -- {type(exc).__name__}: {exc}")
        print()


def _with_progress():
    """
    Print a marker per model call. A live run makes ~86 requests with rate-limit
    backoff between them; without this the terminal sits silent for minutes and
    a stall is indistinguishable from normal throttling.
    """
    from c360 import llm as _llm

    original = _llm.complete_json
    state = {"n": 0, "ok": 0, "failed": 0, "reasons": {}}

    def counted(*args, **kwargs):
        state["n"] += 1
        n = state["n"]
        print(f"    call {n} ...", end="\r", flush=True)
        try:
            result = original(*args, **kwargs)
            state["ok"] += 1
            return result
        except Exception as exc:
            state["failed"] += 1
            # Report the reason, not just the fact. A run where every call
            # falls back is indistinguishable from an offline run, and the
            # whole point of this script is to be able to tell them apart.
            reason = f"{type(exc).__name__}: {exc}"
            state["reasons"][reason[:160]] = state["reasons"].get(reason[:160], 0) + 1
            if state["failed"] <= 3:
                print(f"    call {n} FAILED -- {reason[:400]}")
            raise

    _llm.complete_json = counted
    # the callers imported the symbol directly, so rebind it there too
    import c360.action, c360.critique, c360.synthesis
    import c360.agents.usage, c360.agents.support, c360.agents.life_signal
    for mod in (c360.action, c360.critique, c360.synthesis,
                c360.agents.usage, c360.agents.support, c360.agents.life_signal):
        if hasattr(mod, "complete_json"):
            mod.complete_json = counted
    return state


def run(scenario: Path, *, offline: bool, out: Path):
    # prefer_default_embedder=False on BOTH halves. The only variable that may
    # change between the two runs is whether the language models are used --
    # retrieval must be identical, or a difference cannot be attributed. It also
    # avoids an 80MB model download that the live half would otherwise trigger.
    pipeline = Pipeline(scenario, offline=offline, prefer_default_embedder=False)
    writer, stats = pipeline.run()
    rows = {c.as_of_time.strftime("%Y-%m-%d"): c for c in writer.checkpoints}
    suffix = "offline" if offline else "live"
    path = writer.write(out / f"{scenario.name}_{suffix}_inferred_events.json")
    # The same line run_evaluation.py prints. If this half disagrees with the
    # numbers from `python run_evaluation.py`, the difference is in how this
    # script builds the pipeline, not in the system.
    print(f"    [{suffix}] {stats.summary()}")
    pipeline.close()
    return rows, score_scenario(scenario, path), path


def compare(scenario: Path) -> int:
    print("\n" + "=" * 72)
    print(f"STEP 3 - {scenario.name}: offline vs live")
    print("=" * 72)

    out = Path("out")
    out.mkdir(parents=True, exist_ok=True)

    print("  running offline ...", flush=True)
    off_rows, off_score, off_path = run(scenario, offline=True, out=out)
    print("  running live (this makes real API calls) ...", flush=True)
    counter = _with_progress()
    live_rows, live_score, live_path = run(scenario, offline=False, out=out)
    print(f"    model calls: {counter['n']} attempted, {counter['ok']} succeeded, "
          f"{counter['failed']} fell back            ")
    if counter["reasons"]:
        print("    failure reasons:")
        for reason, count in sorted(counter["reasons"].items(), key=lambda kv: -kv[1]):
            print(f"      {count:>4}x  {reason}")
    if counter["ok"] == 0 and counter["n"] > 0:
        print("\n    WARNING: every model call fell back. This 'live' run is the")
        print("    offline run by another name, and the comparison below is not")
        print("    evidence of anything. Fix the cause before reporting it.")

    # --- every checkpoint where the two runs disagree ---
    diffs = []
    for date in sorted(off_rows):
        a, b = off_rows[date], live_rows.get(date)
        if b is None:
            diffs.append((date, "missing", "-", "-"))
            continue
        for field in GRADED:
            va, vb = getattr(a, field, None), getattr(b, field, None)
            va = getattr(va, "value", va)
            vb = getattr(vb, "value", vb)
            if va != vb:
                diffs.append((date, field, va, vb))

    print(f"\n  checkpoints compared : {len(off_rows)}")
    print(f"  fields that differ   : {len(diffs)}")
    if diffs:
        print(f"\n  {'date':<12} {'field':<18} {'offline':<32} live")
        print("  " + "-" * 92)
        for date, field, va, vb in diffs:
            print(f"  {date:<12} {field:<18} {str(va):<32} {vb}")
    else:
        print("\n  No graded field differs. The models agreed with the deterministic path")
        print("  at every checkpoint in this scenario.")

    def line(sc):
        return (f"state {sc.state_accuracy:.0%}  confidence {sc.confidence_accuracy:.0%}  "
                f"action {sc.action_accuracy:.0%}  exact {sc.exact_match:.0%}  "
                f"false-positives {sc.false_positive_pass_rate:.0%} clean")

    print("\n  SCORE (against the same ground_truth.json)")
    print(f"    offline : {line(off_score)}")
    print(f"    live    : {line(live_score)}")
    print(f"\n  graded files written:\n    {off_path}\n    {live_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare the offline and live paths.")
    ap.add_argument("--scenario", type=Path, help="e.g. data/scenario_03")
    ap.add_argument("--models", action="store_true", help="list the models these keys can use")
    ap.add_argument("--retrieval", action="store_true", help="check the policy store is populated")
    args = ap.parse_args()

    if not check_keys():
        return 1
    if args.models:
        list_available_models()
        return 0
    if args.retrieval:
        diagnose_retrieval()
        return 0
    if args.scenario is None:
        print("\nKeys are good. Now run the comparison on one scenario, e.g.")
        print("    python compare_live.py --scenario data/scenario_03")
        return 0
    return compare(args.scenario)


if __name__ == "__main__":
    raise SystemExit(main())

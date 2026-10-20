#!/usr/bin/env python3
"""Build docs/replay-data.json for the interactive replay page (docs/replay.html).

Run `python run_evaluation.py` first: it writes out/ (inferred events + traces).
Then `python docs/build_replay_data.py` from the repository root.

Everything on the replay page comes from files the pipeline itself wrote, joined
with the supplied ground truth and live streams. Nothing is hand-typed, and the
script asserts the headline results (8/8 graded fields, 4/4 false-positive
checks, 5/8/3-day lead times) before it writes anything.

Customer names and counterparty names are never copied out of the event
payloads, and neither is support-ticket text.
"""
import datetime as dt
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
START = dt.date(2026, 2, 1)

META = {
    "scenario_01": {
        "title": "A medical emergency",
        "summary": "A sudden hospital stay. Salary is replaced by a smaller benefit, large hospital bills arrive, "
                   "and savings are drawn down. A tuition transfer and a resort refund look alarming but are decoys.",
    },
    "scenario_02": {
        "title": "A new child",
        "summary": "Income dips during leave, spending shifts to baby products, then a daycare standing instruction "
                   "and a dependents update arrive. A large electronics purchase looks like fraud but is a baby monitor.",
    },
    "scenario_03": {
        "title": "A slow exit",
        "summary": "A fee dispute is refused, app use falls, standing instructions are cancelled and savings move to "
                   "another bank. A large tax refund arrives in between and is a decoy.",
    },
}

GATE = {
    "confidence below high": "Confidence is below high, so no action is allowed.",
    "high confidence, stage=early": "High confidence, early stage of the situation.",
    "high confidence, stage=late": "High confidence, late stage of the situation.",
}


def plain_guard(s: str) -> str:
    if s.startswith("no_action"):
        return "Nothing to check: no action was proposed."
    m = re.match(r"(\w+) is not a guarded action.*corroboration was (\d+) source", s)
    if m:
        return f"{m.group(1).replace('_', ' ').capitalize()} is not a guarded action. {m.group(2)} source systems corroborate."
    m = re.match(r"PASSED: (\d+) independent source systems \(.*?\) across (\d+) distinct events", s)
    if m:
        return f"Passed: {m.group(1)} independent source systems across {m.group(2)} distinct events."
    return s


def plain_critique(s: str) -> str:
    if s.startswith("no_action"):
        return "Nothing to review."
    if s.startswith("All mechanical checks passed"):
        return "All code checks passed."
    return s


def money(v) -> str:
    return f"${float(v):,.0f}"


def describe(e: dict) -> str:
    p, s, t = e["payload"], e["source_system"], e["event_type"]
    if s == "card_payments":
        if t == "refund":
            return f"{money(p['amount'])} refund from {p['merchant_name']}"
        return f"{money(p['amount'])} at {p['merchant_name']} ({p['mcc_category']})"
    if s == "core_banking_ledger":
        kind = p["transaction_type"].replace("_", " ")
        if t == "standing_instruction":
            return f"{money(p['amount'])} {kind} (standing instruction)"
        return f"{money(p['amount'])} {t} ({kind})"
    if s in ("instant_payments", "ach_wire"):
        how = "instant transfer" if s == "instant_payments" else "ACH transfer"
        return f"{money(p['amount'])} {how} sent out"
    if s == "loan_kyc":
        return f"Dependents updated from {p['old_value']} to {p['new_value']}"
    if s == "support_logs":
        if t == "ticket_created":
            return f"Support ticket opened ({p['category'].replace('_', ' ')})"
        return f"Support ticket closed ({p['resolution_status'].replace('_', ' ')})"
    if s == "web_app_events":
        dev = p.get("device_type", "")
        if t == "login":
            return f"Logged in on {dev}"
        if t == "session_duration":
            sec = int(p["session_length_sec"])
            return f"Session of {sec // 60}m {sec % 60}s on {dev}"
        if t == "feature_used":
            return f"Used {p['feature_or_page'].replace('_', ' ')}"
        if t == "search_query":
            return f"Searched “{p['search_text']}”"
    return f"{s} {t}"


def parse(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def build_scenario(sid: str) -> dict:
    sdir = ROOT / "data" / sid
    out = json.loads((OUT / f"{sid}_inferred_events.json").read_text())
    trace, history_events = {}, None
    for line in (OUT / f"{sid}.trace.jsonl").read_text().splitlines():
        t = json.loads(line)
        if t["kind"] == "checkpoint":
            trace[t["as_of"][:10]] = t
        elif t["kind"] == "run_start":
            history_events = t["history_events"]
    queue = {i["as_of_time"][:10]: i for i in
             json.loads((OUT / f"{sid}_review_queue.json").read_text())["items"]}
    gt = json.loads((sdir / "ground_truth.json").read_text())
    signal, decoy = set(gt["signal_events"]), set(gt["red_herring_events"])

    # live events, released at max(ingestion, event) -- what the system could see
    raw = [json.loads(l) for l in (sdir / "live_stream.jsonl").read_text().splitlines()]
    events = []
    for e in raw:
        ev_t, ing_t = parse(e["event_time"]), parse(e["ingestion_time"])
        rel = max(ev_t, ing_t)
        events.append(dict(
            id=e["event_id"], rel=rel, when=ev_t.strftime("%Y-%m-%d %H:%M"),
            late=max(0, round((ing_t - ev_t).total_seconds() / 86400)),
            src=e["source_system"], text=describe(e),
            role="signal" if e["event_id"] in signal else "decoy" if e["event_id"] in decoy else None))
    events.sort(key=lambda x: x["rel"])

    days = []
    for k, row in enumerate(out):
        date = row["as_of_time"][:10]
        assert date == (START + dt.timedelta(days=k)).isoformat(), (sid, k, date)
        t = trace[date]
        lo = dt.datetime.combine(START + dt.timedelta(days=k - 1), dt.time(), dt.timezone.utc)
        hi = dt.datetime.combine(START + dt.timedelta(days=k), dt.time(), dt.timezone.utc)
        new = [{kk: v for kk, v in ev.items() if kk != "rel"} for ev in events
               if k > 0 and lo <= ev["rel"] < hi]
        aff = sorted(t["affinity"].items(), key=lambda kv: -kv[1])[:3]
        q = queue.get(date)
        days.append(dict(
            d=k, date=date, state=row["inferred_state"], band=row["confidence_band"],
            action=row["action"], subtype=row.get("action_subtype"), hitl=row["hitl_status"],
            carried=bool(t["carried_forward"]), signals=t["strong_signals"], sources=t["source_systems"],
            affinity=[[a, round(b, 1)] for a, b in aff],
            gate=GATE.get(t["gate_reason"], t["gate_reason"]),
            guard=plain_guard(t["guardrail_reason"]), critique=plain_critique(t["critique_reason"]),
            flag=({"reason": q["reason"], "detail": q["detail"]} if q else None),
            events=new))

    # grading, recomputed here from the files (same fields run_evaluation.py scores)
    by_date = {d["date"]: d for d in days}
    graded, fields_ok, fields_total = [], 0, 0
    for c in gt["checkpoints"]:
        date = c["as_of_time"][:10]
        d = by_date[date]
        exp = {"state": c["expected_inferred_state"], "band": c["expected_confidence_band"],
               "action": c["expected_action"]}
        if "expected_action_subtype" in c:
            exp["subtype"] = c["expected_action_subtype"]
        if "expected_hitl_status" in c:
            exp["hitl"] = c["expected_hitl_status"]
        ok = {f: d[f] == v for f, v in exp.items()}
        fields_ok += sum(ok.values())
        fields_total += len(ok)
        item = dict(date=date, expected=exp, ok=ok)
        if "ideal_action_lead_time_days" in c:
            # earliest day from which the expected action holds continuously up to the checkpoint
            k = d["d"]
            while k > 0 and days[k - 1]["action"] == exp["action"]:
                k -= 1
            item["lead"] = dict(ideal=c["ideal_action_lead_time_days"], achieved=d["d"] - k, since=days[k]["date"])
        graded.append(item)
    assert all(all(g["ok"].values()) for g in graded), f"{sid}: a graded field differs"

    fps = []
    for f in gt["false_positive_checks"]:
        ev = next(e for e in raw if e["event_id"] == f["event_id"])
        t0 = parse(ev["event_time"])
        t1 = t0 + dt.timedelta(hours=f["window_hours"])
        bad = [d["date"] for d in days
               if t0 <= parse(d["date"] + "T00:00:00Z") <= t1 and d["action"] in f["must_not_trigger_action"]]
        assert not bad, f"{sid}: forbidden action near {f['event_id']}: {bad}"
        fps.append(dict(id=f["event_id"], date=t0.strftime("%Y-%m-%d"), forbidden=f["must_not_trigger_action"],
                        hours=f["window_hours"], clean=True))

    return dict(id=sid, n=int(sid[-1]), title=META[sid]["title"], summary=META[sid]["summary"],
                days=days, graded=graded, falsepos=fps,
                counts=dict(fields_ok=fields_ok, fields_total=fields_total, events=len(raw), history=history_events))


def main() -> None:
    scenarios = [build_scenario(f"scenario_0{n}") for n in (1, 2, 3)]
    tot_ok = sum(s["counts"]["fields_ok"] for s in scenarios)
    tot = sum(s["counts"]["fields_total"] for s in scenarios)
    leads = [g["lead"]["achieved"] for s in scenarios for g in s["graded"] if "lead" in g]
    assert leads == [5, 8, 3], leads  # matches docs/EVALUATION.md
    assert sum(len(s["falsepos"]) for s in scenarios) == 4
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    data = dict(
        generated=dt.date.today().isoformat(), commit=commit,
        method="python run_evaluation.py, offline and deterministic, no API keys",
        graded_fields=f"{tot_ok}/{tot}", scenarios=scenarios)
    path = ROOT / "docs" / "replay-data.json"
    path.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB); graded fields {tot_ok}/{tot}; leads {leads}")


if __name__ == "__main__":
    main()

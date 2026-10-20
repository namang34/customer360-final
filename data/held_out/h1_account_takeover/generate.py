#!/usr/bin/env python3
"""Generate the held-out scenario h1 (account takeover), deterministically.

WRITTEN BY AN AI ASSISTANT AFTER READING THE CODE. It is a stress test of a known gap
(the system has no signal-to-state mapping for potential_fraud_or_takeover), not
independent evidence of anything. The customer, merchants and counterparties are
invented. See README.md in this folder.

    python data/held_out/h1_account_takeover/generate.py
"""
import datetime as dt
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
rng = random.Random(901)
CUST = "CUST_00901"
CHK, SAV, CC = "ACC_CHK_901", "ACC_SAV_901", "ACC_CC_901"
T = lambda s: dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)
iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")

rows = []  # (event_time, tag, envelope-without-id)


def add(when, account, source, etype, payload, tag=None, lag=4):
    rows.append((when, tag, dict(
        event_time=iso(when), ingestion_time=iso(when + dt.timedelta(seconds=lag)),
        customer_id=CUST, account_id=account, source_system=source, event_type=etype,
        schema_version="1.0", payload=payload)))


chk_bal = [5400.0]
sav_bal = [41000.0]


def ledger(when, acct, etype, amount, ttype, tag=None):
    bal = chk_bal if acct == CHK else sav_bal
    bal[0] += amount if etype in ("deposit", "interest_credit") else -amount
    add(when, acct, "core_banking_ledger", etype,
        dict(amount=amount, balance_after=round(bal[0], 2), transaction_type=ttype), tag)


MERCHANTS = [("Fresh Market", "grocery", 20, 110), ("Corner Grocer", "grocery", 8, 45), ("Metro Transit", "transit", 3, 12),
             ("Bean There Coffee", "dining", 4, 12), ("Noodle House", "dining", 12, 38), ("Shell Fuel", "fuel", 30, 70),
             ("Parkside Cinema", "entertainment", 10, 30), ("Homeware Plus", "retail", 15, 120), ("Bookworm", "retail", 8, 40)]


def baseline(start, end):
    d = start
    while d < end:
        # one or two card purchases most days
        for _ in range(rng.choice([0, 1, 1, 1, 2])):
            m = rng.choice(MERCHANTS)
            when = d + dt.timedelta(hours=rng.randint(7, 21), minutes=rng.randint(0, 59))
            add(when, CC, "card_payments", "purchase", dict(
                merchant_name=m[0], mcc_category=m[1], amount=rng.randint(m[2], m[3]), currency="USD",
                is_international=False, card_present=rng.random() < 0.6))
        # mobile logins on roughly four days in seven, each with a session
        if rng.random() < 0.57:
            when = d + dt.timedelta(hours=rng.randint(6, 22), minutes=rng.randint(0, 59))
            add(when, None, "web_app_events", "login", dict(feature_or_page="transaction_history", device_type="mobile"))
            add(when + dt.timedelta(minutes=1), None, "web_app_events", "session_duration",
                dict(feature_or_page="session", session_length_sec=rng.randint(90, 400), device_type="mobile"))
        # fixed calendar events
        if d.day in (1, 15):
            ledger(d + dt.timedelta(hours=6), CHK, "deposit", 2600, "salary_credit")
        for day, amt, tt in ((3, 1250, "rent_payment"), (8, 140, "utilities_payment"), (12, 45, "gym_membership")):
            if d.day == day:
                ledger(d + dt.timedelta(hours=9), CHK, "standing_instruction", amt, tt)
        if d.day == 28:
            ledger(d + dt.timedelta(hours=2), SAV, "interest_credit", round(sav_bal[0] * 0.0003, 2), "interest_credit")
        d += dt.timedelta(days=1)


H0, H1 = T("2025-10-01T00:00:00"), T("2026-02-01T00:00:00")
L1 = T("2026-04-15T00:00:00")
HIST_CUT = len(rows)
baseline(H0, H1)
# A known payee with a precedent in the history: the first half of an annual property-tax bill.
add(T("2025-10-14T10:20:00"), CHK, "ach_wire", "outbound_transfer", dict(
    direction="outbound", amount=10800, currency="USD", counterparty_name="Riverside County Tax Office",
    counterparty_country="US", transfer_type="standard", status="completed"))
history = sorted(rows, key=lambda r: r[0])
rows.clear()

baseline(H1, L1)

# ---- red herring A: genuine foreign travel, small card-present purchases abroad
add(T("2026-02-08T13:05:00"), CC, "card_payments", "purchase", dict(
    merchant_name="Hotel Lisboa Centro", mcc_category="lodging", amount=310, currency="EUR",
    is_international=True, card_present=True), tag="decoy_travel")
add(T("2026-02-09T19:40:00"), CC, "card_payments", "purchase", dict(
    merchant_name="Taberna do Mar", mcc_category="dining", amount=64, currency="EUR",
    is_international=True, card_present=True))
# ---- red herring B: the second property-tax payment, to the same known payee
add(T("2026-02-12T10:15:00"), CHK, "ach_wire", "outbound_transfer", dict(
    direction="outbound", amount=10800, currency="USD", counterparty_name="Riverside County Tax Office",
    counterparty_country="US", transfer_type="standard", status="completed"), tag="decoy_tax")

# ---- the takeover, 21-22 February
a = T("2026-02-21T03:12:00")
at = lambda m, s=0: a + dt.timedelta(minutes=m, seconds=s)
W, S = "web_app_events", "signal"
add(at(0), None, W, "login", dict(feature_or_page="login", device_type="desktop"), tag=S)
add(at(2), None, W, "session_duration", dict(feature_or_page="session", session_length_sec=1480, device_type="desktop"))
add(at(4), None, W, "feature_used", dict(feature_or_page="update_contact_details", device_type="desktop"), tag=S)
add(at(5), None, "loan_kyc", "kyc_update", dict(event_subtype="contact_details_change",
    old_value="phone ending 2291", new_value="phone ending 7740"), tag=S)
add(at(10), None, W, "feature_used", dict(feature_or_page="add_payee", device_type="desktop"), tag=S)
add(at(12), None, W, "feature_used", dict(feature_or_page="raise_transfer_limit", device_type="desktop"), tag=S)
ledger(at(15), SAV, "withdrawal", 15000, "internal_transfer_out", tag=S)
add(at(19), CHK, "instant_payments", "outbound_transfer", dict(
    direction="outbound", amount=4800, currency="USD", counterparty_name="Northgate Trading FZE",
    counterparty_country="AE", transfer_type="instant", status="completed"), tag=S)
add(at(40), CHK, "instant_payments", "outbound_transfer", dict(
    direction="outbound", amount=9500, currency="USD", counterparty_name="Meridian Assets HK Ltd",
    counterparty_country="HK", transfer_type="instant", status="completed"), tag=S)
add(at(53), CHK, "instant_payments", "outbound_transfer", dict(
    direction="outbound", amount=4900, currency="USD", counterparty_name="Northgate Trading FZE",
    counterparty_country="AE", transfer_type="instant", status="completed"), tag=S)
b = T("2026-02-21T09:30:00")
add(b, CC, "card_payments", "purchase", dict(merchant_name="GlobalGift Cards", mcc_category="gift_cards", amount=1200,
    currency="USD", is_international=True, card_present=False), tag=S)
add(b + dt.timedelta(minutes=11), CC, "card_payments", "purchase", dict(merchant_name="GlobalGift Cards",
    mcc_category="gift_cards", amount=1800, currency="USD", is_international=True, card_present=False), tag=S)
add(b + dt.timedelta(minutes=25), CC, "card_payments", "decline", dict(merchant_name="GlobalGift Cards",
    mcc_category="gift_cards", amount=2000, currency="USD", is_international=True, card_present=False,
    decline_reason="suspected_velocity"))
c = T("2026-02-22T01:50:00")
ledger(c, SAV, "withdrawal", 11000, "internal_transfer_out", tag=S)
add(c + dt.timedelta(minutes=20), CHK, "instant_payments", "outbound_transfer", dict(
    direction="outbound", amount=7200, currency="USD", counterparty_name="Meridian Assets HK Ltd",
    counterparty_country="HK", transfer_type="instant", status="completed"), tag=S)
add(c + dt.timedelta(minutes=35), CHK, "instant_payments", "outbound_transfer", dict(
    direction="outbound", amount=3900, currency="USD", counterparty_name="Kestrel Logistics Co",
    counterparty_country="SC", transfer_type="instant", status="completed"), tag=S)

# ---- the customer notices that afternoon
d0 = T("2026-02-22T15:00:00")
add(d0, None, W, "login", dict(feature_or_page="transaction_history", device_type="mobile"))
add(d0 + dt.timedelta(minutes=4), None, W, "session_duration", dict(feature_or_page="session", session_length_sec=420, device_type="mobile"))
add(d0 + dt.timedelta(minutes=20), None, "support_logs", "ticket_created", dict(
    channel="chat", category="unauthorised_transactions", resolution_status="open",
    raw_text="There are transfers and card charges on my account that I did not make, and my phone number "
             "was changed without me. Please lock everything."), tag=S)
add(T("2026-02-24T10:00:00"), None, "support_logs", "ticket_resolved", dict(
    channel="chat", category="unauthorised_transactions", resolution_status="resolved",
    raw_text="We have blocked the new payees, reset your credentials, reissued your cards and sent recall "
             "requests for the transfers."))
add(T("2026-02-24T10:30:00"), None, "loan_kyc", "kyc_update", dict(event_subtype="contact_details_change",
    old_value="phone ending 7740", new_value="phone ending 2291"))

# ---- aftermath: one more attempt is blocked, then the recalled funds come back
add(T("2026-03-03T04:10:00"), CHK, "instant_payments", "outbound_transfer", dict(
    direction="outbound", amount=5000, currency="USD", counterparty_name="Northgate Trading FZE",
    counterparty_country="AE", transfer_type="instant", status="blocked"))
ledger(T("2026-03-10T11:00:00"), CHK, "deposit", 18000, "reversal_credit", tag="decoy_recovery")

live = sorted(rows, key=lambda r: r[0])

# ---- number the events and write
ids, out = {}, {"history": [], "live": []}
n = 0
for part, seq in (("history", history), ("live", live)):
    for when, tag, env in seq:
        n += 1
        eid = f"EVT_{n:06d}"
        if tag:
            ids.setdefault(tag, []).append(eid)
        out[part].append({"event_id": eid, **{k: env[k] for k in (
            "event_time", "ingestion_time", "customer_id", "account_id", "source_system", "event_type", "schema_version", "payload")}})
for part, fn in (("history", "history_seed.jsonl"), ("live", "live_stream.jsonl")):
    (HERE / fn).write_text("\n".join(json.dumps(r) for r in out[part]) + "\n", encoding="utf-8")

(HERE / "entities.json").write_text(json.dumps({
    "customer_id": CUST, "household_id": None,
    "profile": {"name": "Jordan Ellis", "age": 41, "occupation": "operations manager", "customer_value_tier": "standard",
                "tenure_months": 84, "marital_status": "married", "dependents": 2},
    "accounts": [{"account_id": CHK, "type": "checking", "opened_date": "2019-10-02"},
                 {"account_id": SAV, "type": "savings", "opened_date": "2019-10-02"},
                 {"account_id": CC, "type": "credit_card", "opened_date": "2020-03-18"}]}, indent=2), encoding="utf-8")
(HERE / "replay_config.json").write_text(json.dumps({
    "scenario_id": "held_out_h1", "simulated_start": "2026-02-01T00:00:00Z", "simulated_end": "2026-04-15T00:00:00Z",
    "replay_speed_seconds_per_simulated_day": 30}, indent=2), encoding="utf-8")
(HERE / "instructions.md").write_text(
    "# Scenario Data Package\n\nSame layout as the practice scenarios. See README.md for what this one is.\n", encoding="utf-8")

signal = ids["signal"]
gt = {
    "scenario_id": "held_out_h1_account_takeover",
    "true_narrative": ("A long-standing customer's account is taken over in the early hours of 21 February. Someone logs in "
                       "from a desktop for the first time, changes the phone number, adds payees and raises the transfer "
                       "limit, then drains savings and sends money to new payees abroad and buys gift cards. The customer "
                       "notices on the afternoon of 22 February and reports it. Two earlier events look suspicious but "
                       "are not: small card-present purchases while travelling, and a large payment to a known tax "
                       "office. Recalled funds are credited back on 10 March, which looks like a windfall but is not."),
    "signal_events": signal,
    "red_herring_events": ids["decoy_travel"] + ids["decoy_tax"] + ids["decoy_recovery"],
    "checkpoints": [
        {"as_of_time": "2026-02-20T00:00:00Z", "expected_inferred_state": "no_significant_event",
         "expected_confidence_band": "low", "expected_action": "no_action",
         "notes": "Before the attack. The foreign card purchases and the property-tax payment must not raise an alarm."},
        {"as_of_time": "2026-02-22T00:00:00Z", "expected_inferred_state": "potential_fraud_or_takeover",
         "expected_confidence_band": "high", "expected_action": "compliance_fraud_hold", "expected_hitl_status": "escalated",
         "notes": "The takeover has been under way for about twenty hours and the customer has not yet noticed. "
                  "A new device, a changed phone number, new payees and money leaving to them are independent signals."},
        {"as_of_time": "2026-02-24T00:00:00Z", "expected_inferred_state": "potential_fraud_or_takeover",
         "expected_confidence_band": "high", "expected_action": "compliance_fraud_hold", "expected_hitl_status": "escalated",
         "ideal_action_lead_time_days": 2,
         "notes": "The hold should already have fired before the customer reported it on the afternoon of 22 February."},
    ],
    "false_positive_checks": [
        {"event_id": ids["decoy_travel"][0], "must_not_trigger_action": ["compliance_fraud_hold", "relationship_manager_escalation"],
         "window_hours": 72, "notes": "Card-present purchases abroad while travelling."},
        {"event_id": ids["decoy_tax"][0], "must_not_trigger_action": ["compliance_fraud_hold", "relationship_manager_escalation"],
         "window_hours": 72, "notes": "A large payment to a payee the customer paid before."},
        {"event_id": ids["decoy_recovery"][0], "must_not_trigger_action": ["personalized_offer"],
         "window_hours": 72, "notes": "Recalled funds returning look like a windfall but are not."},
    ],
}
(HERE / "ground_truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
print(f"history {len(out['history'])} events, live {len(out['live'])} events; signal events {len(signal)}")

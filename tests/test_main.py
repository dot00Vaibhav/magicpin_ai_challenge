"""Observable contract and seeded-flow tests for the challenge API."""

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
SIMULATED_NOW = "2026-04-26T10:35:00Z"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def reset_runtime_state():
    main.state.reset()
    yield
    main.state.reset()


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def seeds():
    categories = {p.stem: read_json(p) for p in (DATASET / "categories").glob("*.json")}
    merchants = {x["merchant_id"]: x for x in read_json(DATASET / "merchants_seed.json")["merchants"]}
    customers = {x["customer_id"]: x for x in read_json(DATASET / "customers_seed.json")["customers"]}
    triggers = {x["id"]: x for x in read_json(DATASET / "triggers_seed.json")["triggers"]}
    return categories, merchants, customers, triggers


def push(client, scope, context_id, payload, version=1):
    return client.post("/v1/context", json={
        "scope": scope,
        "context_id": context_id,
        "version": version,
        "payload": payload,
        "delivered_at": SIMULATED_NOW,
    })


def tick(client, trigger_ids, now=SIMULATED_NOW):
    return client.post("/v1/tick", json={"now": now, "available_triggers": trigger_ids})


def push_trigger_contexts(client, seeds, trigger, *, customer_override=None):
    categories, merchants, customers, _ = seeds
    merchant = merchants[trigger["merchant_id"]]
    category = categories[merchant["category_slug"]]
    assert push(client, "category", category["slug"], category).status_code == 200
    assert push(client, "merchant", merchant["merchant_id"], merchant).status_code == 200
    if trigger.get("customer_id"):
        customer = customer_override or customers[trigger["customer_id"]]
        assert push(client, "customer", customer["customer_id"], customer).status_code == 200
    assert push(client, "trigger", trigger["id"], trigger).status_code == 200
    return category, merchant


def active_tick_time(trigger):
    expiry = datetime.fromisoformat(trigger["expires_at"].replace("Z", "+00:00"))
    return (expiry - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")


def test_health_routes_and_metadata(client):
    expected = {"status": "ok", "contexts_loaded": {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}}
    assert client.get("/v1/healthz").json()["status"] == expected["status"]
    assert client.get("/health").json()["contexts_loaded"] == expected["contexts_loaded"]
    assert client.get("/v1/metadata").status_code == 200


@pytest.mark.parametrize("scope", ["category", "merchant", "customer", "trigger"])
def test_each_context_scope_accepts_its_seed_payload(client, seeds, scope):
    categories, merchants, customers, triggers = seeds
    item = {
        "category": next(iter(categories.values())),
        "merchant": next(iter(merchants.values())),
        "customer": next(iter(customers.values())),
        "trigger": next(iter(triggers.values())),
    }[scope]
    context_id = item["slug"] if scope == "category" else item["merchant_id"] if scope == "merchant" else item["customer_id"] if scope == "customer" else item["id"]
    response = push(client, scope, context_id, item)
    assert response.status_code == 200
    assert response.json()["accepted"] is True


def test_context_versions_replace_and_reject_same_or_stale(client, seeds):
    category = seeds[0]["dentists"]
    assert push(client, "category", "dentists", category, 1).status_code == 200
    assert push(client, "category", "dentists", category, 1).status_code == 409
    updated = {**category, "display_name": "Updated"}
    assert push(client, "category", "dentists", updated, 2).status_code == 200
    assert push(client, "category", "dentists", category, 1).json()["current_version"] == 2
    assert client.get("/v1/healthz").json()["contexts_loaded"]["category"] == 1


@pytest.mark.parametrize("bad_payload", [
    ("trigger", "trg_bad_scope", {"id": "trg_bad_scope", "scope": ["merchant"], "kind": "test", "merchant_id": "m1", "payload": {}, "expires_at": SIMULATED_NOW}),
    ("trigger", "trg_bad_source", {"id": "trg_bad_source", "scope": "merchant", "source": {}, "kind": "test", "merchant_id": "m1", "payload": {}, "expires_at": SIMULATED_NOW}),
    ("trigger", "trg_bad_kind", {"id": "trg_bad_kind", "scope": "merchant", "source": "external", "kind": [], "merchant_id": "m1", "payload": {}, "expires_at": SIMULATED_NOW}),
    ("trigger", "trg_bad_inner", {"id": "trg_bad_inner", "scope": "merchant", "source": "external", "kind": "test", "merchant_id": "m1", "payload": [], "expires_at": SIMULATED_NOW}),
    ("customer", "c_bad_state", {"customer_id": "c_bad_state", "merchant_id": "m1", "state": ["churned"], "identity": {}, "consent": {}}),
    ("customer", "c_bad_consent", {"customer_id": "c_bad_consent", "merchant_id": "m1", "state": "active", "identity": {}, "consent": "invalid"}),
    ("merchant", "m_bad_category", {"merchant_id": "m_bad_category", "category_slug": [], "identity": {}}),
    ("category", "bad_digest", {"slug": "bad_digest", "digest": {}}),
])
def test_malformed_nested_payloads_return_controlled_400(client, bad_payload):
    scope, context_id, payload = bad_payload
    response = push(client, scope, context_id, payload)
    assert response.status_code == 400
    assert response.json()["reason"] == "invalid_context"


@pytest.mark.parametrize("body", ["{", "[]", "null"])
def test_malformed_json_is_a_controlled_400(client, body):
    response = client.post("/v1/context", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["reason"] == "malformed_request"
    assert "Traceback" not in response.text


def test_bad_request_timestamps_and_ids_do_not_return_500(client):
    response = client.post("/v1/tick", json={"now": "yesterday", "available_triggers": []})
    assert response.status_code == 400
    response = client.post("/v1/tick", json={"now": SIMULATED_NOW, "available_triggers": [{"bad": "id"}]})
    assert response.status_code == 400


def test_seed_research_action_is_grounded_and_has_one_canonical_body(client, seeds):
    trigger = seeds[3]["trg_001_research_digest_dentists"]
    push_trigger_contexts(client, seeds, trigger)
    response = tick(client, [trigger["id"]])
    assert response.status_code == 200
    actions = response.json()["actions"]
    assert len(actions) == 1
    action = actions[0]
    assert "3-month fluoride varnish recall" in action["body"]
    assert "JIDA Oct 2026, p.14" in action["body"]
    assert "Source: JIDA Oct 2026, p.14. Would a patient-friendly summary be useful?" in action["body"]
    assert action["body"].count("p.14. Would") == 1
    assert action["body"] == action["template_params"][0]
    assert main.state.conversations[action["conversation_id"]]["sent_bodies"][0] == action["body"]
    assert "Want meto" not in action["body"]


@pytest.mark.parametrize("digest_change,expected_count", [
    ("missing_id", 0),
    ("empty", 0),
    ("missing_source", 1),
])
def test_research_digest_resolution_and_missing_source(client, seeds, digest_change, expected_count):
    categories, merchants, _, triggers = seeds
    category = copy.deepcopy(categories["dentists"])
    trigger = copy.deepcopy(triggers["trg_001_research_digest_dentists"])
    if digest_change == "missing_id":
        trigger["payload"]["top_item_id"] = "not-present"
    elif digest_change == "empty":
        category["digest"] = []
    else:
        item = next(x for x in category["digest"] if x["id"] == trigger["payload"]["top_item_id"])
        item.pop("source", None)
    merchant = merchants[trigger["merchant_id"]]
    assert push(client, "category", "dentists", category).status_code == 200
    assert push(client, "merchant", merchant["merchant_id"], merchant).status_code == 200
    assert push(client, "trigger", trigger["id"], trigger).status_code == 200
    actions = tick(client, [trigger["id"]]).json()["actions"]
    assert len(actions) == expected_count
    if actions:
        assert "3-month fluoride varnish recall" in actions[0]["body"]
        assert "Source:" not in actions[0]["body"]


def test_malformed_digest_structure_is_rejected_without_500(client, seeds):
    category = copy.deepcopy(seeds[0]["dentists"])
    category["digest"] = {"unexpected": "object"}
    response = push(client, "category", "dentists", category)
    assert response.status_code == 400


def test_urls_and_long_grounded_messages_are_allowed(client, seeds):
    categories, merchants, _, triggers = seeds
    category = copy.deepcopy(categories["dentists"])
    trigger = copy.deepcopy(triggers["trg_001_research_digest_dentists"])
    item = next(x for x in category["digest"] if x["id"] == trigger["payload"]["top_item_id"])
    item["source"] = "https://example.org/research"
    item["summary"] = "Verified finding " + ("evidence " * 220)
    merchant = merchants[trigger["merchant_id"]]
    push(client, "category", "dentists", category)
    push(client, "merchant", merchant["merchant_id"], merchant)
    push(client, "trigger", trigger["id"], trigger)
    action = tick(client, [trigger["id"]]).json()["actions"][0]
    assert "https://example.org/research" in action["body"]
    assert len(action["body"]) > 1200


def test_taboo_vocabulary_prevents_unsafe_message(client, seeds):
    categories, merchants, _, triggers = seeds
    category = copy.deepcopy(categories["dentists"])
    category["voice"]["vocab_taboo"].append("3-month fluoride varnish recall")
    trigger = triggers["trg_001_research_digest_dentists"]
    push(client, "category", "dentists", category)
    push(client, "merchant", trigger["merchant_id"], merchants[trigger["merchant_id"]])
    push(client, "trigger", trigger["id"], trigger)
    assert tick(client, [trigger["id"]]).json() == {"actions": []}


TRIGGER_EXPECTED_TEXT = {
    "trg_001_research_digest_dentists": "JIDA Oct 2026",
    "trg_002_compliance_dci_radiograph": "DCI revised radiograph",
    "trg_003_recall_due_priya": "Available times",
    "trg_004_perf_dip_bharat": "calls changed",
    "trg_005_renewal_due_bharat": "Pro plan renewal",
    "trg_006_festival_diwali": "Diwali",
    "trg_007_bridal_followup_kavya": "bridal",
    "trg_008_curious_ask_studio11": "services are in demand this week",
    "trg_009_winback_glamour": "expired",
    "trg_010_ipl_match_delhi": "DC vs MI",
    "trg_011_review_theme_late_delivery": "late delivery",
    "trg_012_milestone_mylari": "145 of 150",
    "trg_013_corporate_thali_planning": "corporate bulk thali",
    "trg_014_seasonal_acquisition_dip_powerhouse": "expected seasonality",
    "trg_015_winback_rashmi": "weight loss",
    "trg_016_kids_yoga_program_drafting": "kids yoga summer camp",
    "trg_017_kids_yoga_trial_followup_karthik": "Sat 3 May, 8am",
    "trg_018_supply_atorvastatin_recall": "atorvastatin",
    "trg_019_chronic_refill_grandfather": "metformin",
    "trg_020_summer_demand_shift": "ORS_demand_+40",
    "trg_021_unverified_gbp_sunrise": "postcard or phone call",
    "trg_022_cde_webinar_dentists": "Digital impressions",
    "trg_023_competitor_opened_dentist": "Smile Studio",
    "trg_024_perf_spike_zen": "calls increased",
    "trg_025_dormancy_glamour": "subscription expiry",
}


@pytest.mark.parametrize("trigger_id", list(TRIGGER_EXPECTED_TEXT))
def test_every_seed_trigger_preserves_its_specific_fact(client, seeds, trigger_id):
    trigger = seeds[3][trigger_id]
    push_trigger_contexts(client, seeds, trigger)
    actions = tick(client, [trigger_id], now=active_tick_time(trigger)).json()["actions"]
    assert len(actions) == 1, trigger_id
    assert TRIGGER_EXPECTED_TEXT[trigger_id].lower() in actions[0]["body"].lower(), actions[0]["body"]


def test_repeated_tick_suppresses_same_recipient_and_key(client, seeds):
    trigger = seeds[3]["trg_001_research_digest_dentists"]
    push_trigger_contexts(client, seeds, trigger)
    first = tick(client, [trigger["id"]]).json()["actions"]
    second = tick(client, [trigger["id"]]).json()["actions"]
    assert len(first) == 1
    assert second == []


def test_same_suppression_key_does_not_cross_suppress_merchants(client, seeds):
    categories, merchants, _, triggers = seeds
    first = copy.deepcopy(triggers["trg_001_research_digest_dentists"])
    second = copy.deepcopy(first)
    second["id"] = "trg_same_key_salon"
    second["merchant_id"] = "m_003_studio11_salon_hyderabad"
    second["kind"] = "curious_ask_due"
    second["payload"] = {"category": "salons", "ask_template": "what_service_in_demand_this_week"}
    salon = merchants[second["merchant_id"]]
    for slug in ("dentists", "salons"):
        assert push(client, "category", slug, categories[slug]).status_code == 200
    for merchant in (merchants[first["merchant_id"]], salon):
        assert push(client, "merchant", merchant["merchant_id"], merchant).status_code == 200
    assert push(client, "trigger", first["id"], first).status_code == 200
    assert push(client, "trigger", second["id"], second).status_code == 200
    actions = tick(client, [first["id"], second["id"]]).json()["actions"]
    assert len(actions) == 2


def test_same_suppression_key_does_not_cross_suppress_customers(client, seeds):
    categories, merchants, customers, triggers = seeds
    priya_trigger = copy.deepcopy(triggers["trg_003_recall_due_priya"])
    rohit_trigger = copy.deepcopy(priya_trigger)
    rohit_trigger["id"] = "trg_same_key_rohit"
    rohit_trigger["customer_id"] = "c_002_rohit_for_m001"
    rohit_trigger["payload"]["customer_id"] = rohit_trigger["customer_id"]
    dentist = categories["dentists"]
    merchant = merchants[priya_trigger["merchant_id"]]
    for context_scope, context_id, payload in (
        ("category", dentist["slug"], dentist),
        ("merchant", merchant["merchant_id"], merchant),
        ("customer", priya_trigger["customer_id"], customers[priya_trigger["customer_id"]]),
        ("customer", rohit_trigger["customer_id"], customers[rohit_trigger["customer_id"]]),
        ("trigger", priya_trigger["id"], priya_trigger),
        ("trigger", rohit_trigger["id"], rohit_trigger),
    ):
        assert push(client, context_scope, context_id, payload).status_code == 200
    actions = tick(client, [priya_trigger["id"], rohit_trigger["id"]], active_tick_time(priya_trigger)).json()["actions"]
    assert len(actions) == 2


def test_changed_version_and_suppression_key_can_start_a_new_conversation(client, seeds):
    trigger = copy.deepcopy(seeds[3]["trg_001_research_digest_dentists"])
    push_trigger_contexts(client, seeds, trigger)
    first = tick(client, [trigger["id"]]).json()["actions"][0]
    changed = copy.deepcopy(trigger)
    changed["suppression_key"] = "research:dentists:2026-W18"
    assert push(client, "trigger", trigger["id"], changed, version=2).status_code == 200
    second = tick(client, [trigger["id"]]).json()["actions"][0]
    assert second["suppression_key"] != first["suppression_key"]
    assert second["conversation_id"] != first["conversation_id"]


def test_missing_contexts_and_relationship_mismatches_are_skipped(client, seeds):
    categories, merchants, _, triggers = seeds
    trigger = triggers["trg_001_research_digest_dentists"]
    assert tick(client, ["missing-trigger"]).json() == {"actions": []}
    push(client, "trigger", trigger["id"], trigger)
    assert tick(client, [trigger["id"]]).json() == {"actions": []}
    push(client, "merchant", trigger["merchant_id"], merchants[trigger["merchant_id"]])
    assert tick(client, [trigger["id"]]).json() == {"actions": []}
    push(client, "category", "dentists", categories["dentists"])
    assert len(tick(client, [trigger["id"]]).json()["actions"]) == 1


def test_expired_trigger_does_not_send(client, seeds):
    trigger = seeds[3]["trg_001_research_digest_dentists"]
    push_trigger_contexts(client, seeds, trigger)
    assert tick(client, [trigger["id"]], now=trigger["expires_at"]).json() == {"actions": []}


@pytest.mark.parametrize("change,expected", [
    ("none", 1),
    ("missing_consent", 0),
    ("wrong_scope", 0),
    ("churned", 0),
    ("lapsed_hard", 0),
    ("opted_out", 0),
    ("wrong_merchant", 0),
])
def test_customer_recall_consent_state_and_relationship(client, seeds, change, expected):
    categories, merchants, customers, triggers = seeds
    trigger = copy.deepcopy(triggers["trg_003_recall_due_priya"])
    customer = copy.deepcopy(customers[trigger["customer_id"]])
    if change == "missing_consent":
        customer.pop("consent")
    elif change == "wrong_scope":
        customer["consent"]["scope"] = ["appointment_reminders"]
    elif change == "churned":
        customer["state"] = "churned"
    elif change == "lapsed_hard":
        customer["state"] = "lapsed_hard"
    elif change == "opted_out":
        customer["preferences"]["reminder_opt_in"] = False
    elif change == "wrong_merchant":
        customer["merchant_id"] = "m_unrelated"
    push_trigger_contexts(client, seeds, trigger, customer_override=customer)
    actions = tick(client, [trigger["id"]], now=active_tick_time(trigger)).json()["actions"]
    assert len(actions) == expected


def test_missing_customer_context_is_skipped(client, seeds):
    trigger = seeds[3]["trg_003_recall_due_priya"]
    categories, merchants, _, _ = seeds
    merchant = merchants[trigger["merchant_id"]]
    push(client, "category", merchant["category_slug"], categories[merchant["category_slug"]])
    push(client, "merchant", merchant["merchant_id"], merchant)
    push(client, "trigger", trigger["id"], trigger)
    assert tick(client, [trigger["id"]], active_tick_time(trigger)).json() == {"actions": []}


def test_customer_opt_out_suppresses_future_customer_triggers(client, seeds):
    categories, merchants, customers, triggers = seeds
    trigger = copy.deepcopy(triggers["trg_003_recall_due_priya"])
    push_trigger_contexts(client, seeds, trigger)
    action = tick(client, [trigger["id"]], active_tick_time(trigger)).json()["actions"][0]
    stopped = client.post("/v1/reply", json={
        "conversation_id": action["conversation_id"], "customer_id": trigger["customer_id"],
        "from_role": "customer", "message": "STOP", "received_at": SIMULATED_NOW, "turn_number": 2,
    })
    assert stopped.status_code == 200 and stopped.json()["action"] == "end"
    later = copy.deepcopy(trigger)
    later["id"] = "trg_later_recall"
    later["payload"]["id"] = later["id"]
    later["suppression_key"] = "recall:later"
    assert push(client, "trigger", later["id"], later).status_code == 200
    assert tick(client, [later["id"]], active_tick_time(later)).json() == {"actions": []}


def test_lapsed_hard_customer_is_allowed_only_for_explicit_winback(client, seeds):
    categories, merchants, customers, triggers = seeds
    trigger = triggers["trg_015_winback_rashmi"]
    customer = customers[trigger["customer_id"]]
    push_trigger_contexts(client, seeds, trigger)
    actions = tick(client, [trigger["id"]], active_tick_time(trigger)).json()["actions"]
    assert len(actions) == 1
    assert actions[0]["send_as"] == "merchant_on_behalf"


def test_reply_requires_identity_for_unknown_conversation(client):
    response = client.post("/v1/reply", json={
        "conversation_id": "unknown", "from_role": "merchant", "message": "Thanks",
        "received_at": SIMULATED_NOW, "turn_number": 2,
    })
    assert response.status_code == 404
    accepted = client.post("/v1/reply", json={
        "conversation_id": "unknown-known-merchant", "merchant_id": "m_test", "from_role": "merchant",
        "message": "Thanks", "received_at": SIMULATED_NOW, "turn_number": 2,
    })
    assert accepted.status_code == 200
    assert accepted.json()["action"] == "send"


def test_unknown_customer_reply_requires_a_pushed_customer_context(client, seeds):
    customer = seeds[2]["c_001_priya_for_m001"]
    payload = {
        "conversation_id": "unknown-customer", "customer_id": customer["customer_id"],
        "from_role": "customer", "message": "Thanks", "received_at": SIMULATED_NOW, "turn_number": 2,
    }
    assert client.post("/v1/reply", json=payload).status_code == 404
    assert push(client, "customer", customer["customer_id"], customer).status_code == 200
    assert client.post("/v1/reply", json=payload).status_code == 200


def test_reply_identity_is_checked_and_omitted_ids_are_inferred_only_for_known_conversation(client, seeds):
    trigger = seeds[3]["trg_001_research_digest_dentists"]
    push_trigger_contexts(client, seeds, trigger)
    action = tick(client, [trigger["id"]]).json()["actions"][0]
    mismatch = client.post("/v1/reply", json={
        "conversation_id": action["conversation_id"], "merchant_id": "m_other", "from_role": "merchant",
        "message": "Thanks", "received_at": SIMULATED_NOW, "turn_number": 2,
    })
    assert mismatch.status_code == 400
    customer_role = client.post("/v1/reply", json={
        "conversation_id": action["conversation_id"], "from_role": "customer", "message": "Yes",
        "received_at": SIMULATED_NOW, "turn_number": 2,
    })
    assert customer_role.status_code == 400
    valid = client.post("/v1/reply", json={
        "conversation_id": action["conversation_id"], "from_role": "merchant", "message": "Yes please",
        "received_at": SIMULATED_NOW, "turn_number": 2,
    })
    assert valid.status_code == 200 and valid.json()["action"] == "send"


def test_customer_reply_is_valid_only_on_customer_conversation(client, seeds):
    trigger = seeds[3]["trg_003_recall_due_priya"]
    push_trigger_contexts(client, seeds, trigger)
    action = tick(client, [trigger["id"]], active_tick_time(trigger)).json()["actions"][0]
    response = client.post("/v1/reply", json={
        "conversation_id": action["conversation_id"], "customer_id": trigger["customer_id"],
        "from_role": "customer", "message": "Thanks", "received_at": SIMULATED_NOW, "turn_number": 2,
    })
    assert response.status_code == 200
    assert response.json()["action"] == "send"
    assert main.state.conversations[action["conversation_id"]]["sent_bodies"][-1] == response.json()["body"]


def test_customer_cannot_reply_to_merchant_only_conversation(client, seeds):
    trigger = seeds[3]["trg_001_research_digest_dentists"]
    push_trigger_contexts(client, seeds, trigger)
    action = tick(client, [trigger["id"]]).json()["actions"][0]
    response = client.post("/v1/reply", json={
        "conversation_id": action["conversation_id"], "customer_id": "c_other", "from_role": "customer",
        "message": "Thanks", "received_at": SIMULATED_NOW, "turn_number": 2,
    })
    assert response.status_code == 400


def test_auto_reply_count_is_conversation_specific_and_stops_on_second(client):
    def post(conversation_id, turn):
        return client.post("/v1/reply", json={
            "conversation_id": conversation_id, "merchant_id": "m_auto", "from_role": "merchant",
            "message": "Thank you for contacting us. Our team will respond shortly.",
            "received_at": SIMULATED_NOW, "turn_number": turn,
        })
    assert post("auto-a", 2).json()["action"] == "wait"
    assert post("auto-b", 2).json()["action"] == "wait"
    assert post("auto-a", 3).json()["action"] == "end"
    assert post("auto-a", 4).json()["action"] == "end"


def test_invalid_participant_cannot_increment_auto_reply_count(client, seeds):
    trigger = seeds[3]["trg_001_research_digest_dentists"]
    push_trigger_contexts(client, seeds, trigger)
    action = tick(client, [trigger["id"]]).json()["actions"][0]
    payload = {
        "conversation_id": action["conversation_id"], "merchant_id": "m_wrong", "from_role": "merchant",
        "message": "Thank you for contacting us. Our team will respond shortly.",
        "received_at": SIMULATED_NOW, "turn_number": 2,
    }
    assert client.post("/v1/reply", json=payload).status_code == 400
    payload["merchant_id"] = trigger["merchant_id"]
    assert client.post("/v1/reply", json=payload).json()["action"] == "wait"


def test_intent_handoff_and_five_turn_limit(client):
    payload = {
        "conversation_id": "intent", "merchant_id": "m_intent", "from_role": "merchant",
        "message": "Ok lets do it. Whats next?", "received_at": SIMULATED_NOW, "turn_number": 2,
    }
    response = client.post("/v1/intent", json=payload)
    assert response.status_code == 200
    assert response.json()["action"] == "send"
    payload["conversation_id"] = "turn-limit"
    payload["message"] = "A normal reply"
    payload["turn_number"] = 6
    assert client.post("/v1/reply", json=payload).json()["action"] == "end"

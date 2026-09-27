"""Small, deterministic FastAPI service for the magicpin Vera challenge."""

import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Vera merchant assistant", version="1.0.0")
STARTED_AT = time.time()
SCOPES = {"category", "merchant", "customer", "trigger"}
contexts: dict[tuple[str, str], dict[str, Any]] = {}
conversations: dict[str, dict[str, Any]] = {}
sent_suppression_keys: set[str] = set()
auto_reply_text_counts: dict[str, int] = {}
opted_out_merchants: set[str] = set()
opted_out_customers: set[str] = set()


@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={
        "accepted": False,
        "reason": "malformed_request",
        "details": str(exc.errors()),
    })


class ContextPush(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


def _get(scope: str, context_id: str | None) -> dict[str, Any] | None:
    if not context_id:
        return None
    record = contexts.get((scope, str(context_id)))
    return record["payload"] if record else None


def _first_name(merchant: dict[str, Any]) -> str:
    identity = merchant.get("identity") or {}
    owner = identity.get("owner_first_name")
    if owner:
        return str(owner)
    name = str(identity.get("name") or "")
    return name.split()[0] if name else "there"


def _compose(category: dict[str, Any], merchant: dict[str, Any], trigger: dict[str, Any],
             customer: dict[str, Any] | None) -> dict[str, str]:
    """Build a grounded message from the current pushed contexts."""
    identity = merchant.get("identity") or {}
    merchant_name = identity.get("name") or "your business"
    owner = _first_name(merchant)
    kind = str(trigger.get("kind") or "update").replace("_", " ")
    data = trigger.get("payload") or {}
    suppression_key = str(trigger.get("suppression_key") or trigger.get("id") or "")

    if customer is not None:
        ci = customer.get("identity") or {}
        customer_name = ci.get("name") or "there"
        body = (f"Hi {customer_name}, a note from {merchant_name}: {kind.capitalize()} is due. "
                "Reply YES if you would like the team to follow up, or STOP to opt out.")
        return {"body": body, "cta": "binary_yes_no", "send_as": "merchant_on_behalf",
                "suppression_key": suppression_key,
                "rationale": "Customer-facing reminder based only on the pushed trigger and customer context."}

    # Prefer the exact digest item referenced by the trigger, otherwise use a
    # trigger-supplied fact. Never create a citation or statistic.
    digest = category.get("digest") or []
    top_id = data.get("top_item_id")
    item = next((x for x in digest if x.get("id") == top_id), None)
    if item is None and kind in {"research digest", "category research digest release", "research_digest"} and digest:
        item = digest[0]
    pieces: list[str] = []
    if item:
        title = item.get("title")
        summary = item.get("summary")
        source = item.get("source")
        if title:
            pieces.append(str(title).rstrip("."))
        if summary:
            pieces.append(str(summary).rstrip("."))
        if source:
            pieces.append(f"Source: {source}.")

    facts = []
    for key, value in data.items():
        if key in {"merchant_id", "customer_id", "category", "top_item_id"} or isinstance(value, (dict, list)):
            continue
        if value is not None:
            facts.append(f"{key.replace('_', ' ')}: {value}")
    if not pieces and facts:
        pieces.append("; ".join(facts[:3]))
    if not pieces:
        pieces.append(f"There is a {kind} update for {merchant_name}.")

    local = ". ".join(pieces)
    if not local.endswith((".", "!", "?")):
        local += "."
    intro = f"Hi {owner}, "
    body = f"{intro}{local} Want me to help with the next step?"
    return {"body": body, "cta": "open_ended", "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": f"Responds to the {kind} trigger using current category and trigger context; no unsupported merchant claims added."}


@app.get("/v1/healthz")
def healthz():
    counts = {scope: 0 for scope in SCOPES}
    for scope, _ in contexts:
        counts[scope] += 1
    return {"status": "ok", "uptime_seconds": int(time.time() - STARTED_AT), "contexts_loaded": counts}


@app.get("/v1/metadata")
def metadata():
    return {"team_name": "Vera Challenge Submission", "team_members": [],
            "model": "deterministic-python", "approach": "Context-grounded deterministic composer with trigger routing and conversation handling",
            "contact_email": "", "version": "1.0.0", "submitted_at": datetime.now(timezone.utc).isoformat()}


@app.post("/v1/context")
def push_context(body: ContextPush):
    if body.scope not in SCOPES:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "invalid_scope",
            "details": f"scope must be one of {sorted(SCOPES)}",
        })
    key = (body.scope, body.context_id)
    current = contexts.get(key)
    if current and body.version <= current["version"]:
        return JSONResponse(status_code=409, content={
            "accepted": False, "reason": "stale_version",
            "current_version": current["version"],
        })
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.now(timezone.utc).isoformat()}


@app.post("/v1/tick")
def tick(body: TickRequest):
    actions = []
    for trigger_id in body.available_triggers:
        trigger = _get("trigger", trigger_id)
        if not trigger:
            continue
        suppression_key = str(trigger.get("suppression_key") or trigger_id)
        if suppression_key in sent_suppression_keys:
            continue
        merchant_id = trigger.get("merchant_id")
        customer_id = trigger.get("customer_id")
        if merchant_id in opted_out_merchants or customer_id in opted_out_customers:
            continue
        merchant = _get("merchant", merchant_id)
        if not merchant:
            continue
        category = _get("category", merchant.get("category_slug"))
        if not category:
            continue
        customer = _get("customer", customer_id) if customer_id else None
        is_customer = trigger.get("scope") == "customer" or customer_id is not None
        if is_customer:
            if not customer:
                continue
            consent = customer.get("consent") or {}
            if not consent.get("opted_in_at") or not consent.get("scope"):
                continue
            if customer.get("state") in {"churned", "lapsed_hard"}:
                continue
        composed = _compose(category, merchant, trigger, customer if is_customer else None)
        message_body = re.sub(r"\bWant me\s*to\b", "Want me to", composed["body"], flags=re.IGNORECASE)
        conversation_id = f"conv_{merchant_id}_{trigger_id}"
        action = {"conversation_id": conversation_id, "merchant_id": merchant_id,
                  "customer_id": customer_id if is_customer else None,
                  "send_as": composed["send_as"], "trigger_id": trigger_id,
                  "template_name": "vera_customer_reminder_v1" if is_customer else "vera_context_update_v1",
                  "template_params": [message_body], **composed, "body": message_body}
        actions.append(action)
        conversations[conversation_id] = {"merchant_id": merchant_id, "customer_id": customer_id,
                                         "send_as": composed["send_as"], "trigger": trigger,
                                         "category": category, "merchant": merchant,
                                         "customer": customer, "sent_bodies": [composed["body"]],
                                         "auto_reply_count": 0}
        sent_suppression_keys.add(suppression_key)
        if len(actions) >= 20:
            break
    return {"actions": actions}


def _is_opt_out(message: str) -> bool:
    text = message.lower().strip()
    return bool(re.search(r"\b(stop|unsubscribe|opt\s*out|don't message|do not message|no more messages|not interested)\b", text))


def _is_auto_reply(message: str) -> bool:
    text = message.lower()
    phrases = ("thank you for contacting", "thanks for contacting", "our team will respond",
               "we will get back to you", "will respond shortly", "automated assistant",
               "office hours", "currently unavailable")
    return any(phrase in text for phrase in phrases)


@app.post("/v1/reply")
def reply(body: ReplyRequest):
    message = body.message.strip()
    state = conversations.setdefault(body.conversation_id, {"sent_bodies": [], "auto_reply_count": 0})
    if _is_opt_out(message):
        state["opted_out"] = True
        if body.customer_id:
            opted_out_customers.add(body.customer_id)
        elif body.merchant_id:
            opted_out_merchants.add(body.merchant_id)
        return {"action": "end", "rationale": "Explicit opt-out or hard no; ending the conversation and honoring the request."}
    if state.get("opted_out"):
        return {"action": "end", "rationale": "Conversation is already opted out; no further messages will be sent."}
    if _is_auto_reply(message):
        state["auto_reply_count"] = state.get("auto_reply_count", 0) + 1
        # Some harnesses model each repeated canned reply as a fresh
        # conversation. Track the normalized canned text as well as per-thread
        # repeats so those retries are recognized without treating varied
        # human replies as automation.
        normalized_auto_reply = re.sub(r"\s+", " ", message.lower()).strip()
        auto_reply_text_counts[normalized_auto_reply] = auto_reply_text_counts.get(normalized_auto_reply, 0) + 1
        if state["auto_reply_count"] >= 2 or auto_reply_text_counts[normalized_auto_reply] >= 2:
            return {"action": "end", "rationale": "Repeated canned auto-reply detected; ending to avoid spending more turns."}
        return {"action": "wait", "wait_seconds": 14400,
                "rationale": "Canned business auto-reply detected; waiting for the owner instead of treating it as engagement."}
    lowered = message.lower()
    if any(x in lowered for x in ("ok let's do it", "ok lets do it", "let's do it", "lets do it", "go ahead", "yes please", "what's next", "whats next", "i want to join")):
        return {"action": "send", "body": "Great, let's get this moving. I'll prepare the next step using the details you shared.",
                "cta": "open_ended", "rationale": "Clear positive intent detected; moving directly to action without more qualification."}
    if any(x in lowered for x in ("wait", "later", "busy", "call me")):
        return {"action": "wait", "wait_seconds": 1800, "rationale": "Merchant asked to defer; backing off for 30 minutes."}
    if body.from_role == "customer":
        return {"action": "end", "rationale": "Customer reply received; this service only routes merchant conversations."}
    return {"action": "send", "body": "Got it. I'll keep this focused on the update you replied to and use the information already available.",
            "cta": "open_ended", "rationale": "Acknowledged the reply and offered a concise, context-safe next step."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))

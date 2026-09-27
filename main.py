"""Deterministic FastAPI implementation of the Magicpin Vera challenge API."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger("vera")
app = FastAPI(title="Vera merchant assistant", version="1.0.0")
STARTED_AT = time.time()
CONTEXT_SCOPES = {"category", "merchant", "customer", "trigger"}
CUSTOMER_CONSENT_SCOPE = {
    "recall_due": "recall_reminders",
    "customer_lapsed_soft": "recall_reminders",
    "wedding_package_followup": "bridal_package_followup",
    "customer_lapsed_hard": "winback_offers",
    "trial_followup": "kids_program_updates",
    "chronic_refill_due": "refill_reminders",
    "appointment_tomorrow": "appointment_reminders",
}
RESEARCH_TRIGGER_KINDS = {"research_digest", "category_research_digest_release"}
CUSTOMER_STATES = {"new", "active", "lapsed_soft", "lapsed_hard", "churned"}


@dataclass
class RuntimeState:
    """In-memory state for pushed contexts, suppression, and live conversations."""

    contexts: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    conversations: dict[str, dict[str, Any]] = field(default_factory=dict)
    consumed_suppressions: set[tuple[str, str]] = field(default_factory=set)
    conversation_sequences: dict[tuple[str, str], int] = field(default_factory=dict)
    opted_out_merchants: set[str] = field(default_factory=set)
    opted_out_customers: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.contexts.clear()
        self.conversations.clear()
        self.consumed_suppressions.clear()
        self.conversation_sequences.clear()
        self.opted_out_merchants.clear()
        self.opted_out_customers.clear()

    def get(self, scope: str, context_id: str | None) -> dict[str, Any] | None:
        if not isinstance(context_id, str) or not context_id:
            return None
        record = self.contexts.get((scope, context_id))
        return record["payload"] if record else None

    def new_conversation_id(self, merchant_id: str, trigger_id: str) -> str:
        key = (merchant_id, trigger_id)
        sequence = self.conversation_sequences.get(key, 0) + 1
        self.conversation_sequences[key] = sequence
        return f"conv_{merchant_id}_{trigger_id}_{sequence}"


state = RuntimeState()


def _parse_iso_datetime(value: Any) -> datetime:
    """Parse an ISO-8601 timestamp with an explicit timezone."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("must be a valid ISO-8601 timestamp") from exc
    else:
        raise ValueError("must be an ISO-8601 timestamp string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ContextPush(RequestModel):
    scope: str
    context_id: str
    version: int = Field(strict=True, gt=0)
    payload: dict[str, Any]
    delivered_at: datetime

    @field_validator("delivered_at", mode="before")
    @classmethod
    def validate_delivered_at(cls, value: Any) -> datetime:
        return _parse_iso_datetime(value)


class TickRequest(RequestModel):
    now: datetime
    available_triggers: list[str] = Field(default_factory=list)

    @field_validator("now", mode="before")
    @classmethod
    def validate_now(cls, value: Any) -> datetime:
        return _parse_iso_datetime(value)


class ReplyRequest(RequestModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: Literal["merchant", "customer"]
    message: str
    received_at: datetime
    turn_number: int = Field(strict=True, ge=1)

    @field_validator("received_at", mode="before")
    @classmethod
    def validate_received_at(cls, value: Any) -> datetime:
        return _parse_iso_datetime(value)

    @field_validator("conversation_id", "message")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


@dataclass(frozen=True)
class ComposedMessage:
    body: str
    cta: str
    send_as: str
    suppression_key: str
    rationale: str


def _canonical_body(parts: list[str]) -> str:
    """Join fragments and normalize whitespace without changing words or punctuation."""
    return " ".join(" ".join(part for part in parts if isinstance(part, str)).split())


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_timestamp(value: Any, *, allow_date: bool = False) -> bool:
    if allow_date and isinstance(value, str):
        try:
            date.fromisoformat(value)
            return True
        except ValueError:
            pass
    try:
        _parse_iso_datetime(value)
        return True
    except ValueError:
        return False


def _valid_scope_payload(scope: str, context_id: str, payload: dict[str, Any]) -> str | None:
    """Check identity and field types needed to route and safely use a context."""
    identity_key = {"category": "slug", "merchant": "merchant_id", "customer": "customer_id", "trigger": "id"}[scope]
    payload_id = payload.get(identity_key)
    if not _nonempty_string(payload_id):
        return f"payload.{identity_key} must be a non-empty string"
    if payload_id != context_id:
        return f"context_id must match payload.{identity_key}"

    if scope == "category":
        if "voice" in payload and not isinstance(payload["voice"], dict):
            return "category payload.voice must be an object"
        if "digest" in payload and not isinstance(payload["digest"], list):
            return "category payload.digest must be an array"
        if "peer_stats" in payload and not isinstance(payload["peer_stats"], dict):
            return "category payload.peer_stats must be an object"
        for key in ("offer_catalog", "patient_content_library", "seasonal_beats", "trend_signals"):
            if key in payload and not isinstance(payload[key], list):
                return f"category payload.{key} must be an array"

    elif scope == "merchant":
        if not _nonempty_string(payload.get("category_slug")):
            return "merchant payload.category_slug must be a non-empty string"
        if "identity" in payload and not isinstance(payload["identity"], dict):
            return "merchant payload.identity must be an object"
        if "offers" in payload and not isinstance(payload["offers"], list):
            return "merchant payload.offers must be an array"
        for key in ("performance", "subscription", "customer_aggregate"):
            if key in payload and not isinstance(payload[key], dict):
                return f"merchant payload.{key} must be an object"

    elif scope == "customer":
        if not _nonempty_string(payload.get("merchant_id")):
            return "customer payload.merchant_id must be a non-empty string"
        customer_state = payload.get("state")
        if not isinstance(customer_state, str) or customer_state not in CUSTOMER_STATES:
            return "customer payload.state must be a supported relationship state"
        if "identity" in payload and not isinstance(payload["identity"], dict):
            return "customer payload.identity must be an object"
        consent = payload.get("consent")
        if consent is not None and not isinstance(consent, dict):
            return "customer payload.consent must be an object"
        if isinstance(consent, dict):
            scopes = consent.get("scope", [])
            if not isinstance(scopes, list) or any(not isinstance(item, str) for item in scopes):
                return "customer consent.scope must be an array of strings"
            opted_in_at = consent.get("opted_in_at")
            if opted_in_at is not None and not _valid_timestamp(opted_in_at, allow_date=True):
                return "customer consent.opted_in_at must be an ISO-8601 date or timezone-aware timestamp"
        for key in ("relationship", "preferences"):
            if key in payload and not isinstance(payload[key], dict):
                return f"customer payload.{key} must be an object"

    elif scope == "trigger":
        trigger_scope = payload.get("scope")
        if not isinstance(trigger_scope, str) or trigger_scope not in {"merchant", "customer"}:
            return "trigger payload.scope must be merchant or customer"
        kind = payload.get("kind")
        if not _nonempty_string(kind):
            return "trigger payload.kind must be a non-empty string"
        source = payload.get("source")
        if source is not None and (not isinstance(source, str) or source not in {"external", "internal"}):
            return "trigger payload.source must be external or internal"
        nested = payload.get("payload", {})
        if not isinstance(nested, dict):
            return "trigger payload.payload must be an object"
        merchant_id = payload.get("merchant_id")
        if merchant_id is None:
            merchant_id = nested.get("merchant_id")
        if not _nonempty_string(merchant_id):
            return "trigger payload.merchant_id must be a non-empty string"
        customer_id = payload.get("customer_id")
        if customer_id is not None and not _nonempty_string(customer_id):
            return "trigger payload.customer_id must be a non-empty string or null"
        if trigger_scope == "customer" and not _nonempty_string(customer_id):
            return "customer-scoped trigger requires payload.customer_id"
        if trigger_scope == "merchant" and customer_id is not None:
            return "merchant-scoped trigger must not specify payload.customer_id"
        if "urgency" in payload:
            urgency = payload["urgency"]
            if not isinstance(urgency, int) or isinstance(urgency, bool) or not 1 <= urgency <= 5:
                return "trigger payload.urgency must be an integer from 1 to 5"
        if "suppression_key" in payload and not isinstance(payload["suppression_key"], str):
            return "trigger payload.suppression_key must be a string"
        if not _valid_timestamp(payload.get("expires_at")):
            return "trigger payload.expires_at must be a timezone-aware ISO-8601 timestamp"
    return None


@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
    details = "; ".join(
        f"{'.'.join(str(part) for part in error['loc'][1:])}: {error['msg']}"
        for error in exc.errors()
    )
    return JSONResponse(status_code=400, content={"accepted": False, "reason": "malformed_request", "details": details})


def _first_name(context: dict[str, Any], *, identity_key: str = "identity") -> str:
    identity = context.get(identity_key)
    if not isinstance(identity, dict):
        return "there"
    first = identity.get("owner_first_name")
    if _nonempty_string(first):
        return first.strip()
    name = identity.get("name")
    return name.strip().split()[0] if _nonempty_string(name) else "there"


def _plain(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return None


def _percent(value: Any) -> str | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value * 100:g}%"
    return _plain(value)


def _digest_item(category: dict[str, Any], item_id: Any) -> dict[str, Any] | None:
    digest = category.get("digest", [])
    if not isinstance(digest, list) or not isinstance(item_id, str):
        return None
    return next((item for item in digest if isinstance(item, dict) and item.get("id") == item_id), None)


def _digest_text(item: dict[str, Any]) -> str:
    return _canonical_body([_plain(item.get("title")) or "", _plain(item.get("summary")) or "", _plain(item.get("actionable")) or ""])


def _active_offer(category: dict[str, Any], merchant: dict[str, Any], hint: str | None = None) -> str | None:
    offers = merchant.get("offers", [])
    category_offers = category.get("offer_catalog", [])
    if not isinstance(offers, list):
        offers = []
    if not isinstance(category_offers, list):
        category_offers = []
    candidates = [item for item in offers if isinstance(item, dict) and item.get("status") in (None, "active")]
    candidates.extend(item for item in category_offers if isinstance(item, dict))
    if hint:
        terms = {part.lower() for part in re.findall(r"[a-z0-9]+", hint.lower()) if len(part) > 3}
        for offer in candidates:
            title = _plain(offer.get("title"))
            if title and terms.intersection(re.findall(r"[a-z0-9]+", title.lower())):
                return title
        return None
    for offer in candidates:
        title = _plain(offer.get("title"))
        if title:
            return title
    return None


def _category_support(category: dict[str, Any], metric: str | None = None) -> str | None:
    peers = category.get("peer_stats")
    if isinstance(peers, dict) and metric:
        aliases = {"calls": "avg_calls_30d", "views": "avg_views_30d", "directions": "avg_directions_30d", "ctr": "avg_ctr", "review_count": "avg_review_count"}
        raw_value = peers.get(aliases.get(metric, ""))
        peer_value = _percent(raw_value) if metric == "ctr" else _plain(raw_value)
        if peer_value:
            return f"Category peer reference: {peer_value} {metric}"
    trends = category.get("trend_signals")
    if isinstance(trends, list):
        for signal in trends:
            if not isinstance(signal, dict):
                continue
            query = _plain(signal.get("query"))
            change = _percent(signal.get("delta_yoy"))
            if query and change:
                return f"Relevant search trend: {query} ({change} year over year)"
    return None


def _seasonal_support(category: dict[str, Any], hint: str | None) -> str | None:
    beats = category.get("seasonal_beats")
    if not isinstance(beats, list) or not hint:
        return None
    words = {word.lower() for word in re.findall(r"[a-z0-9]+", hint.lower()) if len(word) > 2}
    best_note, best_score = None, 0
    for beat in beats:
        if not isinstance(beat, dict):
            continue
        note = _plain(beat.get("note"))
        period = _plain(beat.get("month_range"))
        if not note:
            continue
        searchable = set(re.findall(r"[a-z0-9]+", f"{note} {period or ''}".lower()))
        score = len(words.intersection(searchable))
        if score > best_score:
            best_note, best_score = note, score
    return f"Category seasonal context: {best_note}" if best_note and best_score else None


def _merchant_metric(merchant: dict[str, Any], metric: str | None) -> str | None:
    performance = merchant.get("performance")
    if not isinstance(performance, dict) or not metric:
        return None
    value = performance.get(metric)
    rendered = _percent(value) if metric == "ctr" else _plain(value)
    return f"Current listed {metric}: {rendered}" if rendered else None


def _contains_taboo(message: str, category: dict[str, Any]) -> bool:
    voice = category.get("voice")
    if not isinstance(voice, dict):
        return False
    taboo_terms = voice.get("vocab_taboo", voice.get("taboos", []))
    if not isinstance(taboo_terms, list):
        return False
    for term in taboo_terms:
        if isinstance(term, str) and term.strip():
            pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in term.split()) + r"(?!\w)"
            if re.search(pattern, message, flags=re.IGNORECASE):
                return True
    return False


def _compose(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None = None,
) -> ComposedMessage | None:
    """Build a deterministic message from the exact pushed trigger and related contexts."""
    payload = trigger.get("payload") if isinstance(trigger.get("payload"), dict) else {}
    kind = trigger.get("kind") if isinstance(trigger.get("kind"), str) else "update"
    kind_label = kind.replace("_", " ")
    merchant_name = _plain((merchant.get("identity") or {}).get("name")) if isinstance(merchant.get("identity"), dict) else None
    merchant_name = merchant_name or "your business"
    owner = _first_name(merchant)
    suppression_key = _plain(trigger.get("suppression_key")) or _plain(trigger.get("id")) or ""

    if customer is not None:
        person = _first_name(customer)
        if kind == "recall_due":
            service = (_plain(payload.get("service_due")) or "service recall").replace("_", " ")
            due = _plain(payload.get("due_date"))
            slots = payload.get("available_slots")
            labels = [item.get("label") for item in slots if isinstance(item, dict) and _plain(item.get("label"))] if isinstance(slots, list) else []
            detail = f"Your {service} is due" + (f" by {due}" if due else "")
            if labels:
                detail += f". Available times: {', '.join(str(label) for label in labels[:2])}"
        elif kind == "wedding_package_followup":
            wedding_date = _plain(payload.get("wedding_date"))
            next_step = (_plain(payload.get("next_step_window_open")) or "bridal preparation").replace("_", " ")
            detail = f"Your {next_step} follow-up is ready"
            if wedding_date:
                detail += f" before your {wedding_date} wedding"
            offer = _active_offer(category, merchant, "bridal")
            if offer:
                detail += f". The listed bridal offer is {offer}"
        elif kind == "customer_lapsed_hard":
            days = _plain(payload.get("days_since_last_visit"))
            focus = (_plain(payload.get("previous_focus")) or "").replace("_", " ")
            detail = f"Checking in about your previous {focus} focus" if focus else "Checking in after your last visit"
            if days:
                detail += f" ({days} days ago)"
        elif kind == "trial_followup":
            trial_date = _plain(payload.get("trial_date"))
            sessions = payload.get("next_session_options")
            labels = [item.get("label") for item in sessions if isinstance(item, dict) and _plain(item.get("label"))] if isinstance(sessions, list) else []
            detail = f"Following up on your trial from {trial_date}" if trial_date else "Following up on your trial"
            if labels:
                detail += f". The next session option is {labels[0]}"
        elif kind == "chronic_refill_due":
            molecules = payload.get("molecule_list")
            medicines = ", ".join(str(value) for value in molecules if _plain(value)) if isinstance(molecules, list) else ""
            runs_out = _plain(payload.get("stock_runs_out_iso"))
            detail = f"Your existing refill reminder covers {medicines}" if medicines else "Your existing refill reminder is due"
            if runs_out:
                detail += f" by {runs_out}"
        else:
            return None
        body = _canonical_body([
            f"Hi {person}, a note from {merchant_name}.",
            detail + ".",
            "Reply YES if you would like the team to follow up, or STOP to opt out.",
        ])
        send_as, cta = "merchant_on_behalf", "binary_yes_no"
        rationale = f"Customer {kind_label} reminder uses the pushed trigger, customer context, and consent scope."
    else:
        source: str | None = None
        fact = ""
        closing = ""
        digest_id = payload.get("top_item_id") if kind in RESEARCH_TRIGGER_KINDS | {"regulation_change"} else None
        if digest_id is not None:
            item = _digest_item(category, digest_id)
            if item is None:
                return None
            fact = _digest_text(item)
            source = _plain(item.get("source"))
            if not fact:
                return None

        if kind in RESEARCH_TRIGGER_KINDS:
            closing = "Would a patient-friendly summary be useful?"
        elif kind == "regulation_change":
            fact = _canonical_body([fact, f"Compliance deadline: {_plain(payload.get('deadline_iso')) or 'see the cited circular'}."])
            closing = "Would a short audit checklist help?"
        elif kind == "perf_dip":
            metric = _plain(payload.get("metric")) or "performance"
            change = _percent(payload.get("delta_pct"))
            window = _plain(payload.get("window")) or "recent window"
            baseline = _plain(payload.get("vs_baseline"))
            fact = f"Your {metric} changed {change or 'from baseline'} over {window}"
            if baseline:
                fact += f"; the comparison baseline is {baseline}"
            current = _merchant_metric(merchant, metric)
            if current:
                fact += f". {current}"
            support = _category_support(category, metric)
            if support:
                fact += f". {support}"
            closing = f"Want to review what may be behind the {metric} change?"
        elif kind == "renewal_due":
            plan = _plain(payload.get("plan")) or "current"
            days = _plain(payload.get("days_remaining"))
            amount = _plain(payload.get("renewal_amount"))
            fact = f"Your {plan} plan renewal is coming up"
            if days:
                fact += f" in {days} days"
            if amount:
                fact += f" at ₹{amount}"
            closing = "Would a renewal reminder be helpful?"
        elif kind == "festival_upcoming":
            festival = _plain(payload.get("festival")) or "the upcoming festival"
            event_date = _plain(payload.get("date"))
            fact = f"{festival} is a relevant planning window"
            if event_date:
                fact += f" on {event_date}"
            relevance = payload.get("category_relevance")
            category_slug = _plain(category.get("slug"))
            if isinstance(relevance, list) and category_slug in relevance:
                fact += f" for {category_slug}"
            seasonal = _seasonal_support(category, festival)
            if seasonal:
                fact += f". {seasonal}"
            closing = f"Want to plan an offer around {festival}?"
        elif kind == "curious_ask_due":
            ask_key = _plain(payload.get("ask_template")) or ""
            ask = {
                "what_service_in_demand_this_week": "which services are in demand this week",
            }.get(ask_key, ask_key.replace("_", " ") or "a service demand check")
            trend = _category_support(category)
            fact = f"A useful merchant check-in is due: {ask}"
            if trend:
                fact += f". {trend}"
            closing = "Would this local demand snapshot help?"
        elif kind == "winback_eligible":
            days = _plain(payload.get("days_since_expiry"))
            dipped = _percent(payload.get("perf_dip_pct"))
            lapsed = _plain(payload.get("lapsed_customers_added_since_expiry"))
            fact = "Your subscription has expired"
            if days:
                fact += f" {days} days ago"
            if dipped:
                fact += f", with performance at {dipped}"
            if lapsed:
                fact += f" and {lapsed} additional lapsed customers"
            closing = "Want to review a focused winback plan?"
        elif kind == "ipl_match_today":
            match = _plain(payload.get("match")) or "today's match"
            venue = _plain(payload.get("venue"))
            time_value = _plain(payload.get("match_time_iso"))
            fact = f"{match} is scheduled"
            if venue:
                fact += f" at {venue}"
            if time_value:
                fact += f" ({time_value})"
            closing = "Want to plan a match-day offer for nearby customers?"
        elif kind == "review_theme_emerged":
            theme_key = _plain(payload.get("theme")) or ""
            theme = {"delivery_late": "late delivery"}.get(theme_key, theme_key.replace("_", " ") or "customer feedback")
            count = _plain(payload.get("occurrences_30d"))
            trend = _plain(payload.get("trend"))
            quote = _plain(payload.get("common_quote"))
            fact = f"A {theme} theme appeared in reviews"
            if count:
                fact += f" {count} times in 30 days"
            if trend:
                fact += f" and is {trend}"
            if quote:
                fact += f'; one review said "{quote}"'
            closing = "Would a short response and operations checklist help?"
        elif kind == "milestone_reached":
            metric = (_plain(payload.get("metric")) or "metric").replace("_", " ")
            current = _plain(payload.get("value_now"))
            milestone = _plain(payload.get("milestone_value"))
            fact = f"Your {metric} is close to its next milestone"
            if current and milestone:
                fact = f"Your {metric} is at {current} of {milestone}"
            closing = f"Want to plan the next step toward {milestone or 'the milestone'}?"
        elif kind == "active_planning_intent":
            topic = (_plain(payload.get("intent_topic")) or "your plan").replace("_", " ")
            last_message = _plain(payload.get("merchant_last_message"))
            fact = f"You were planning {topic}"
            if last_message:
                fact += f' after asking: "{last_message}"'
            offer = _active_offer(category, merchant, topic)
            if offer:
                fact += f". A listed offer that matches is {offer}"
            closing = f"I can outline a practical first version of {topic}. Want that?"
        elif kind == "seasonal_perf_dip":
            metric = _plain(payload.get("metric")) or "performance"
            change = _percent(payload.get("delta_pct"))
            window = _plain(payload.get("window")) or "recent window"
            season = (_plain(payload.get("season_note")) or "seasonal context").replace("_", " ")
            fact = f"Your {metric} changed {change or 'from baseline'} over {window}; the trigger marks this as expected seasonality ({season})"
            current = _merchant_metric(merchant, metric)
            seasonal = _seasonal_support(category, season)
            if current:
                fact += f". {current}"
            if seasonal:
                fact += f". {seasonal}"
            closing = "Want to compare this with category demand before changing your plan?"
        elif kind == "supply_alert":
            molecule = _plain(payload.get("molecule")) or "the listed medicine"
            batches = payload.get("affected_batches")
            batch_text = ", ".join(str(value) for value in batches if _plain(value)) if isinstance(batches, list) else ""
            manufacturer = _plain(payload.get("manufacturer"))
            fact = f"A supply alert names {molecule}"
            if batch_text:
                fact += f"; affected batches: {batch_text}"
            if manufacturer:
                fact += f" ({manufacturer})"
            closing = "Would a stock check against these batches help?"
        elif kind == "category_seasonal":
            season = (_plain(payload.get("season")) or "the current season").replace("_", " ")
            trends = payload.get("trends")
            trend_text = ", ".join(str(value) for value in trends[:3] if _plain(value)) if isinstance(trends, list) else ""
            fact = f"The {season} demand signal includes {trend_text}" if trend_text else f"A seasonal demand shift is flagged for {season}"
            if payload.get("shelf_action_recommended") is True:
                fact += ". The trigger recommends reviewing shelf availability"
            seasonal = _seasonal_support(category, season)
            if seasonal:
                fact += f". {seasonal}"
            closing = "Want to review which products to keep visible?"
        elif kind == "gbp_unverified":
            path = (_plain(payload.get("verification_path")) or "the listed verification path").replace("_", " ")
            uplift = _percent(payload.get("estimated_uplift_pct"))
            fact = f"Your business profile is not verified; the listed path is {path}"
            if uplift:
                fact += f", with an estimated {uplift} uplift in the trigger data"
            closing = "Want the verification steps?"
        elif kind == "cde_opportunity":
            item = _digest_item(category, payload.get("digest_item_id"))
            if item:
                fact = _digest_text(item)
                source = _plain(item.get("source"))
            credits = _plain(payload.get("credits"))
            fee = _plain(payload.get("fee"))
            if credits:
                fact = _canonical_body([fact, f"Credits: {credits}."])
            if fee:
                fact = _canonical_body([fact, f"Fee: {fee.replace('_', ' ')}."])
            if not fact:
                return None
            closing = "Want the registration details?"
        elif kind == "competitor_opened":
            competitor = _plain(payload.get("competitor_name")) or "A nearby competitor"
            distance = _plain(payload.get("distance_km"))
            offer = _plain(payload.get("their_offer"))
            opened = _plain(payload.get("opened_date"))
            fact = f"{competitor} opened nearby"
            if distance:
                fact += f" ({distance} km away)"
            if opened:
                fact += f" on {opened}"
            if offer:
                fact += f" with {offer}"
            closing = "Want to review how your current offer is positioned?"
        elif kind == "perf_spike":
            metric = _plain(payload.get("metric")) or "performance"
            change = _percent(payload.get("delta_pct"))
            window = _plain(payload.get("window")) or "recent window"
            baseline = _plain(payload.get("vs_baseline"))
            driver = (_plain(payload.get("likely_driver")) or "").replace("_", " ")
            fact = f"Your {metric} increased {change or 'above baseline'} over {window}"
            if baseline:
                fact += f"; comparison baseline: {baseline}"
            current = _merchant_metric(merchant, metric)
            if current:
                fact += f". {current}"
            if driver:
                fact += f". The signal points to {driver}"
            closing = f"Want to build on the {metric} lift?"
        elif kind == "dormant_with_vera":
            days = _plain(payload.get("days_since_last_merchant_message"))
            topic = (_plain(payload.get("last_topic")) or "our last conversation").replace("_", " ")
            fact = f"It has been {days} days since we discussed {topic}" if days else f"We last discussed {topic}"
            closing = "Would you like to pick that conversation back up?"
        else:
            # Preserve useful facts for a new trigger kind without dumping nested objects.
            facts = [f"{str(key).replace('_', ' ')}: {_plain(value)}" for key, value in payload.items() if _plain(value)]
            fact = ". ".join(facts[:3])
            if not fact:
                return None
            closing = f"Would a short summary of this {kind_label} update help?"

        citation = f" Source: {source}." if source else ""
        body = _canonical_body([f"Hi {owner},", fact.rstrip("."), citation, closing])
        send_as, cta = "vera", "open_ended"
        rationale = f"Grounded {kind_label} message using the supplied trigger and matching context."

    if not body or _contains_taboo(body, category):
        return None
    return ComposedMessage(body, cta, "merchant_on_behalf" if customer is not None else send_as, suppression_key, rationale)


def _trigger_merchant_id(trigger: dict[str, Any]) -> str | None:
    nested = trigger.get("payload")
    nested_id = nested.get("merchant_id") if isinstance(nested, dict) else None
    merchant_id = trigger.get("merchant_id") or nested_id
    return merchant_id if isinstance(merchant_id, str) and merchant_id else None


def _eligible_customer(customer: dict[str, Any], trigger: dict[str, Any], merchant_id: str) -> bool:
    if customer.get("merchant_id") != merchant_id:
        return False
    customer_state = customer.get("state")
    if customer_state == "churned":
        return False
    kind = trigger.get("kind")
    if customer_state == "lapsed_hard" and kind != "customer_lapsed_hard":
        return False
    consent = customer.get("consent")
    if not isinstance(consent, dict) or not _valid_timestamp(consent.get("opted_in_at"), allow_date=True):
        return False
    scopes = consent.get("scope")
    required_scope = CUSTOMER_CONSENT_SCOPE.get(kind) if isinstance(kind, str) else None
    if not isinstance(scopes, list) or required_scope is None or required_scope not in scopes:
        return False
    preferences = customer.get("preferences")
    if isinstance(preferences, dict) and preferences.get("reminder_opt_in") is False:
        return False
    return True


@app.get("/v1/healthz")
@app.get("/health")
def healthz() -> dict[str, Any]:
    counts = {scope: 0 for scope in CONTEXT_SCOPES}
    for scope, _context_id in state.contexts:
        counts[scope] += 1
    return {"status": "ok", "uptime_seconds": int(time.time() - STARTED_AT), "contexts_loaded": counts}


@app.get("/v1/metadata")
def metadata() -> dict[str, Any]:
    return {
        "team_name": "Vera Challenge Submission",
        "team_members": [],
        "model": "deterministic-python",
        "approach": "Context-grounded deterministic composer with trigger routing and conversation handling",
        "contact_email": "",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/context")
def push_context(body: ContextPush) -> Any:
    if body.scope not in CONTEXT_SCOPES:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {sorted(CONTEXT_SCOPES)}"})
    if not body.context_id.strip():
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_context", "details": "context_id must not be blank"})
    error = _valid_scope_payload(body.scope, body.context_id, body.payload)
    if error:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_context", "details": error})
    key = (body.scope, body.context_id)
    current = state.contexts.get(key)
    if current and body.version <= current["version"]:
        return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": current["version"]})
    state.contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": datetime.now(timezone.utc).isoformat()}


@app.post("/v1/tick")
def tick(body: TickRequest) -> dict[str, list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_suppressions: set[tuple[str, str]] = set()
    for trigger_id in body.available_triggers:
        if not isinstance(trigger_id, str) or not trigger_id or trigger_id in seen_ids:
            continue
        seen_ids.add(trigger_id)
        trigger = state.get("trigger", trigger_id)
        if trigger is None:
            continue
        try:
            expires_at = _parse_iso_datetime(trigger.get("expires_at"))
        except (TypeError, ValueError):
            continue
        if body.now >= expires_at:
            continue
        merchant_id = _trigger_merchant_id(trigger)
        trigger_scope = trigger.get("scope")
        customer_id = trigger.get("customer_id")
        if not merchant_id or trigger_scope not in {"merchant", "customer"}:
            continue
        if (trigger_scope == "customer") != bool(customer_id) or (customer_id is not None and not isinstance(customer_id, str)):
            continue
        if merchant_id in state.opted_out_merchants or (customer_id and customer_id in state.opted_out_customers):
            continue
        merchant = state.get("merchant", merchant_id)
        if not isinstance(merchant, dict) or merchant.get("merchant_id") != merchant_id:
            continue
        category_slug = merchant.get("category_slug")
        if not isinstance(category_slug, str):
            continue
        category = state.get("category", category_slug)
        if not isinstance(category, dict) or category.get("slug") != category_slug:
            continue
        trigger_payload = trigger.get("payload") if isinstance(trigger.get("payload"), dict) else {}
        trigger_category = trigger_payload.get("category")
        if isinstance(trigger_category, str) and trigger_category and trigger_category != category_slug:
            continue

        customer = None
        if trigger_scope == "customer":
            customer = state.get("customer", customer_id)
            if not isinstance(customer, dict) or not _eligible_customer(customer, trigger, merchant_id):
                continue
        suppression_key = trigger.get("suppression_key")
        if not isinstance(suppression_key, str) or not suppression_key:
            suppression_key = trigger_id
        recipient_id = customer_id if trigger_scope == "customer" else merchant_id
        suppression = (recipient_id, suppression_key)
        if suppression in state.consumed_suppressions or suppression in seen_suppressions:
            continue
        composed = _compose(category, merchant, trigger, customer)
        if composed is None:
            continue

        conversation_id = state.new_conversation_id(merchant_id, trigger_id)
        canonical = composed.body
        action = {
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id if trigger_scope == "customer" else None,
            "send_as": composed.send_as,
            "trigger_id": trigger_id,
            "template_name": "vera_customer_reminder_v1" if trigger_scope == "customer" else "vera_context_update_v1",
            "template_params": [canonical],
            "body": canonical,
            "cta": composed.cta,
            "suppression_key": composed.suppression_key,
            "rationale": composed.rationale,
        }
        actions.append(action)
        state.conversations[conversation_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id if trigger_scope == "customer" else None,
            "trigger_id": trigger_id,
            "trigger_kind": trigger.get("kind"),
            "send_as": composed.send_as,
            "sent_bodies": [canonical],
            "auto_reply_count": 0,
            "opted_out": False,
        }
        state.consumed_suppressions.add(suppression)
        seen_suppressions.add(suppression)
        if len(actions) >= 20:
            break
    return {"actions": actions}


def _is_opt_out(message: str) -> bool:
    return bool(re.search(r"\b(stop|unsubscribe|opt\s*out|don't message|do not message|no more messages|not interested)\b", message, re.IGNORECASE))


def _is_auto_reply(message: str) -> bool:
    phrases = (
        "thank you for contacting", "thanks for contacting", "our team will respond",
        "we will get back to you", "will respond shortly", "automated assistant",
        "office hours", "currently unavailable",
    )
    lowered = message.lower()
    return any(phrase in lowered for phrase in phrases)


def _participant_error(detail: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"action": "end", "rationale": detail})


def _reply_participants(body: ReplyRequest, conversation: dict[str, Any] | None) -> tuple[str | None, str | None] | JSONResponse:
    """Resolve omitted IDs only for a known conversation and reject role/ID mismatches."""
    merchant_id, customer_id = body.merchant_id, body.customer_id
    if conversation is None:
        if body.from_role == "merchant" and not _nonempty_string(merchant_id):
            return _participant_error("An unknown conversation requires merchant_id for a merchant reply.", 404)
        if body.from_role == "customer" and not _nonempty_string(customer_id):
            return _participant_error("An unknown conversation requires customer_id for a customer reply.", 404)
        if customer_id:
            customer = state.get("customer", customer_id)
            if customer is None:
                return _participant_error("Customer context is required to verify an unknown customer reply.", 404)
            if merchant_id and customer.get("merchant_id") != merchant_id:
                return _participant_error("Reply customer does not belong to the supplied merchant.")
            if not merchant_id:
                merchant_id = customer.get("merchant_id")
        return merchant_id, customer_id

    expected_merchant = conversation.get("merchant_id")
    expected_customer = conversation.get("customer_id")
    if body.from_role == "merchant":
        if not _nonempty_string(expected_merchant):
            return _participant_error("This conversation has no merchant participant.")
        if merchant_id is not None and merchant_id != expected_merchant:
            return _participant_error("Reply merchant does not match the conversation.")
        merchant_id = expected_merchant
    else:
        if not _nonempty_string(expected_customer):
            return _participant_error("This conversation has no customer participant.")
        if customer_id is not None and customer_id != expected_customer:
            return _participant_error("Reply customer does not match the conversation.")
        customer_id = expected_customer
    if merchant_id is not None and merchant_id != expected_merchant:
        return _participant_error("Reply merchant does not match the conversation.")
    if customer_id is not None and customer_id != expected_customer:
        return _participant_error("Reply customer does not match the conversation.")
    return merchant_id, customer_id


def _record_reply_send(conversation: dict[str, Any], message: str) -> str:
    canonical = _canonical_body([message])
    conversation.setdefault("sent_bodies", []).append(canonical)
    return canonical


@app.post("/v1/reply")
@app.post("/v1/intent")
def reply(body: ReplyRequest) -> Any:
    conversation = state.conversations.get(body.conversation_id)
    resolved = _reply_participants(body, conversation)
    if isinstance(resolved, JSONResponse):
        return resolved
    merchant_id, customer_id = resolved
    if conversation is None:
        conversation = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "sent_bodies": [],
            "auto_reply_count": 0,
            "opted_out": False,
        }
        state.conversations[body.conversation_id] = conversation

    message = body.message.strip()
    if _is_opt_out(message):
        conversation["opted_out"] = True
        if body.from_role == "customer" and customer_id:
            state.opted_out_customers.add(customer_id)
        elif merchant_id:
            state.opted_out_merchants.add(merchant_id)
        return {"action": "end", "rationale": "Explicit opt-out or hard no; ending the conversation and honoring the request."}
    if conversation.get("opted_out"):
        return {"action": "end", "rationale": "Conversation is already opted out; no further messages will be sent."}
    if body.turn_number > 5:
        return {"action": "end", "rationale": "The five-turn conversation limit has been reached."}

    if _is_auto_reply(message):
        conversation["auto_reply_count"] = conversation.get("auto_reply_count", 0) + 1
        if conversation["auto_reply_count"] >= 2:
            return {"action": "end", "rationale": "Repeated canned auto-reply detected; ending to avoid spending more turns."}
        return {"action": "wait", "wait_seconds": 14400, "rationale": "Canned business auto-reply detected; waiting for the owner."}

    lowered = message.lower()
    if any(phrase in lowered for phrase in (
        "ok let's do it", "ok lets do it", "let's do it", "lets do it", "go ahead",
        "yes please", "what's next", "whats next", "i want to join",
    )):
        response = "Great, let's get this moving. I'll prepare the next step using the details you shared."
        canonical = _record_reply_send(conversation, response)
        return {"action": "send", "body": canonical, "cta": "open_ended", "rationale": "Clear positive intent detected; moving directly to action without more qualification."}
    if any(phrase in lowered for phrase in ("wait", "later", "busy", "call me")):
        return {"action": "wait", "wait_seconds": 1800, "rationale": "The participant asked to defer; backing off for 30 minutes."}
    if body.from_role == "customer":
        response = "Thanks, I'll pass your reply to the team."
    else:
        response = "Got it. I'll keep this focused on the update you replied to and use the information already available."
    canonical = _record_reply_send(conversation, response)
    return {"action": "send", "body": canonical, "cta": "open_ended", "rationale": "Acknowledged the reply and offered a concise, context-safe next step."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))

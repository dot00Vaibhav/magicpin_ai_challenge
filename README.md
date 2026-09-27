# Vera Challenge API

A deterministic FastAPI implementation of the Magicpin AI Challenge. It accepts the judge's category, merchant, customer, and trigger contexts and returns context-grounded actions. The backend does not require an LLM or API key.

## Challenge API

The repository contract is documented in [challenge-testing-brief.md](challenge-testing-brief.md), with request examples in [examples/api-call-examples.md](examples/api-call-examples.md).

- `GET /v1/healthz` and `GET /v1/metadata`
- `POST /v1/context` stores a versioned context. Higher versions replace older versions; same or older versions return the documented 409 response.
- `POST /v1/tick` evaluates the supplied trigger IDs at the judge's simulated time. Missing or expired contexts and ineligible customer triggers are skipped.
- `POST /v1/reply` handles merchant and customer replies, intent, deferrals, opt-outs, and canned auto-replies.

The `/health` and `/v1/intent` paths are compatibility aliases; `/v1/intent` uses the documented reply request and response shape. The challenge scorer's official endpoints remain unchanged.

Messages are composed deterministically from pushed contexts. Research triggers require the exact digest item named by `top_item_id`; a missing item is skipped, and its source is included when present. Trigger-specific facts are retained for all seed trigger kinds. A final message is whitespace-normalized once and reused as `body`, `template_params[0]`, and the conversation's stored outgoing body. Useful URLs and long, context-grounded messages are allowed. Category taboo terms cause the action to be skipped.

Suppression is scoped to recipient and suppression key. Repeated sends with the same pair are suppressed; a changed suppression key may start a new conversation. Customer sends require an active consent date and matching purpose scope, a valid merchant/customer relationship, and a non-churned state. Hard-lapsed customers are eligible only for the explicit winback trigger.

## Local testing

In PowerShell, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

In another terminal:

```powershell
python -m pytest -q
python -m py_compile main.py judge_simulator.py
```

The repository-provided `judge_simulator.py` is the challenge simulator and remains unmodified. Its scoring scenarios require the configured LLM provider/key; local pytest checks the deterministic API without an LLM. A simulator run is not a substitute for official challenge scoring.

## Production deployment

Render build command:

```text
pip install -r requirements.txt
```

Render start command:

```text
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Render provides `PORT`; running `python main.py` locally defaults to port `8080`. The service stores contexts, suppression keys, and conversations in memory, so that state resets when the process restarts.

## Optional hardening

Request shape and relationship checks reject malformed context data with controlled 4xx responses. The app logs routing outcomes without logging message bodies or complete payloads. `pytest` and `httpx` are development-only dependencies in `requirements-dev.txt`.

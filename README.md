# Vera challenge backend

A small deterministic FastAPI implementation of the magicpin judge API. It stores pushed category, merchant, customer, and trigger contexts in memory, composes grounded trigger messages, deduplicates suppression keys, and handles replies including opt-outs and canned auto-replies.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The server listens on `0.0.0.0` and uses `PORT` when set (default `8080`).

## Judge simulator

Start the backend in one terminal, then run `python judge_simulator.py` in another. The supplied simulator requires a configured LLM provider/key before it runs any scenario. Set those in its Configuration section first.

The backend uses only pushed context, so new category, merchant, customer, or trigger versions take effect immediately. State is in memory and resets when the process restarts.

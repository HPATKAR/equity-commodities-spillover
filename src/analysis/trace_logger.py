"""
Per-step cost and latency tracer for all AI agent LLM calls.
Appends one row to logs/agent_traces.csv on every actual LLM call
(cache misses only — @st.cache_data means cached hits are free).

Usage inside any _call_ai():
    import time
    from src.analysis.trace_logger import log_trace

    t0 = time.monotonic()
    # ... call LLM, get text ...
    log_trace(_AGENT, provider, "claude-sonnet-4-6",
              len(prompt), len(text), (time.monotonic() - t0) * 1000)

Reading traces:
    import pandas as pd
    df = pd.read_csv("logs/agent_traces.csv", parse_dates=["timestamp"])
"""
from __future__ import annotations

import csv
import datetime
from pathlib import Path

_LOG_DIR  = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "agent_traces.csv"
_FIELDS   = [
    "timestamp", "agent_id", "provider", "model",
    "prompt_chars", "completion_chars",
    "prompt_tokens_est", "completion_tokens_est",
    "latency_ms", "cost_usd_est",
]

# Approximate list prices per 1M tokens, USD (April 2025)
_COST_PER_1M: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input":  3.00, "output": 15.00},
    "claude-opus-4-6":   {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5":  {"input":  0.80, "output":  4.00},
    "gpt-4o":            {"input":  5.00, "output": 15.00},
    "gpt-4o-mini":       {"input":  0.15, "output":  0.60},
}
_DEFAULT_COST = {"input": 3.00, "output": 15.00}


def _ensure_log() -> None:
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        if not _LOG_FILE.exists():
            with open(_LOG_FILE, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=_FIELDS).writeheader()
    except Exception:
        pass


def log_trace(
    agent_id:         str,
    provider:         str,
    model:            str,
    prompt_chars:     int,
    completion_chars: int,
    latency_ms:       float,
) -> None:
    """
    Append one trace row to logs/agent_traces.csv.
    Silent on any failure — must never crash an agent.
    """
    try:
        prompt_tokens     = max(prompt_chars     // 4, 1)
        completion_tokens = max(completion_chars // 4, 1)
        rates             = _COST_PER_1M.get(model, _DEFAULT_COST)
        cost_usd          = (
            prompt_tokens     * rates["input"]  / 1_000_000 +
            completion_tokens * rates["output"] / 1_000_000
        )
        row = {
            "timestamp":             datetime.datetime.now().isoformat(timespec="seconds"),
            "agent_id":              agent_id,
            "provider":              provider,
            "model":                 model,
            "prompt_chars":          prompt_chars,
            "completion_chars":      completion_chars,
            "prompt_tokens_est":     prompt_tokens,
            "completion_tokens_est": completion_tokens,
            "latency_ms":            f"{latency_ms:.0f}",
            "cost_usd_est":          f"{cost_usd:.6f}",
        }
        _ensure_log()
        with open(_LOG_FILE, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_FIELDS).writerow(row)
    except Exception:
        pass


def read_traces() -> "list[dict]":
    """Return all trace rows as a list of dicts. Returns [] if log not found."""
    try:
        with open(_LOG_FILE, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def session_cost_summary(traces: "list[dict]") -> dict:
    """
    Summarise traces: total calls, total estimated cost, avg latency.
    Pass the result of read_traces().
    """
    if not traces:
        return {"calls": 0, "total_cost_usd": 0.0, "avg_latency_ms": 0.0}
    calls   = len(traces)
    cost    = sum(float(r.get("cost_usd_est", 0)) for r in traces)
    latency = sum(float(r.get("latency_ms",    0)) for r in traces) / calls
    return {
        "calls":           calls,
        "total_cost_usd":  round(cost, 4),
        "avg_latency_ms":  round(latency, 0),
    }

"""
AI Trade Structurer - owns the Trade Ideas page.
Generates structured trade ideas (entry/exit/rationale/risk) and routes them
to the Pending Review Panel for human approval before display.

Structured output: uses Pydantic TradeOutput schema with up to 2 retries on
validation failure. The model returns JSON; we validate the schema before
the output is accepted — junk never passes forward silently.
Cached 1 hour per idea set.
"""

from __future__ import annotations

import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from src.analysis.agent_state import (
    set_status, set_output, log_activity, calibrate_confidence,
    context_confidence, add_pending, is_enabled,
)

_SYSTEM = (
    "You are the AI Trade Structurer embedded in the Cross-Asset Spillover Monitor "
    "at Purdue University Daniels School of Business. "
    "You generate institutional cross-asset trade ideas based on regime, correlation, "
    "and spillover signals. "
    "Be precise and quantitative. No disclaimers. "
    "Respond only with a valid JSON object — no markdown, no extra text."
)

_AGENT = "trade_structurer"

# ── Pydantic output schema ────────────────────────────────────────────────────

class TradeOutput(BaseModel):
    trade_name: str  = Field(description="Short descriptive name for the trade, e.g. 'Long Gold / Short SPX'")
    direction:  str  = Field(description="Asset pair and direction, e.g. 'Long GLD / Short SPY'")
    trigger:    str  = Field(description="Specific entry condition with price level or signal trigger")
    target:     str  = Field(description="Exit level or time horizon")
    risk:       str  = Field(description="Stop-loss level or hedge instrument")
    rationale:  str  = Field(description="2-3 sentences grounded in the provided market data")


_SCHEMA_HINT = """{
  "trade_name": "...",
  "direction":  "Long X / Short Y",
  "trigger":    "specific entry condition",
  "target":     "exit level or horizon",
  "risk":       "stop or hedge",
  "rationale":  "2-3 sentences from provided data"
}"""


@st.cache_data(ttl=3600, show_spinner=False)
def _call_ai(context_str: str, provider: str, api_key: str) -> dict:
    """
    Returns a validated TradeOutput dict.
    Retries up to 2× on Pydantic validation failure, feeding the error
    back to the model so it can self-correct.
    Returns {"error": reason} if all attempts fail.
    Cached 1 hour.
    """
    import json
    import re
    import time
    from src.analysis.trace_logger import log_trace

    base_prompt = (
        f"CURRENT MARKET CONTEXT:\n{context_str}\n\n"
        "Generate ONE specific, actionable cross-asset trade idea supported by the data above. "
        f"Respond with a JSON object matching this structure exactly:\n{_SCHEMA_HINT}\n"
        "Return ONLY the JSON object — no markdown fences, no extra text."
    )

    last_error = ""
    for attempt in range(3):
        prompt = base_prompt if attempt == 0 else (
            base_prompt + f"\n\nYour previous response failed schema validation: "
            f"{last_error}\nCorrect it and return valid JSON."
        )
        try:
            t0 = time.monotonic()
            if provider == "anthropic":
                import anthropic as _ant
                client = _ant.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                    system=_SYSTEM,
                )
                raw = resp.content[0].text.strip()
                model_name = "claude-sonnet-4-6"
            else:
                from openai import OpenAI as _OAI
                client = _OAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=400, temperature=0.3,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content.strip()
                model_name = "gpt-4o"

            log_trace(_AGENT, provider, model_name, len(prompt), len(raw),
                      (time.monotonic() - t0) * 1000)

            # Strip any accidental markdown fences
            json_text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
            trade = TradeOutput.model_validate_json(json_text)
            return trade.model_dump()

        except ValidationError as e:
            last_error = str(e)[:300]
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"Schema validation failed after 3 attempts: {last_error}"}


def run(
    context: dict,
    provider: str | None,
    api_key: str,
) -> dict:
    """
    context keys:
      regime_name: str
      regime_level: int
      avg_corr: float
      risk_score: float
      top_commodity: str      # best performing
      top_commodity_ret: float
      worst_equity: str
      worst_equity_ret: float
      active_alerts: int
      peer_signals: dict      # outputs from other agents for cross-referencing
    """
    if not is_enabled(_AGENT):
        return {}

    set_status(_AGENT, "investigating")

    parts = []
    if context.get("regime_name"):
        parts.append(f"Correlation regime: {context['regime_name']} (level {context.get('regime_level',1)}/3)")
    if context.get("avg_corr") is not None:
        parts.append(f"Avg cross-asset correlation: {context['avg_corr']:.3f}")
    if context.get("risk_score") is not None:
        parts.append(f"Composite stress score: {context['risk_score']:.0f}/100")
    if context.get("top_commodity") and context.get("top_commodity_ret") is not None:
        parts.append(
            f"Top commodity (5d): {context['top_commodity']} "
            f"+{context['top_commodity_ret']:.1f}%"
        )
    if context.get("worst_equity") and context.get("worst_equity_ret") is not None:
        parts.append(
            f"Worst equity index (5d): {context['worst_equity']} "
            f"{context['worst_equity_ret']:.1f}%"
        )
    if context.get("active_alerts"):
        parts.append(f"Active alerts: {context['active_alerts']}")

    peer = context.get("peer_signals", {})
    if peer.get("macro_strategist"):
        parts.append(f"Macro Strategist: {str(peer['macro_strategist'])[:120]}")
    if peer.get("geopolitical_analyst"):
        parts.append(f"Geo Intel: {str(peer['geopolitical_analyst'])[:120]}")

    if not parts:
        set_status(_AGENT, "idle")
        return {}

    ctx_str = "\n".join(parts)

    if not provider:
        set_status(_AGENT, "monitoring")
        return {"status": "monitoring", "context": ctx_str}

    # Only generate trade ideas in elevated/crisis regimes or high stress
    regime_level = context.get("regime_level", 1)
    risk_score   = context.get("risk_score", 0) or 0

    if regime_level < 2 and risk_score < 45:
        set_status(_AGENT, "monitoring")
        log_activity(_AGENT, "normal regime - no active trade idea", "", "info")
        return {"status": "monitoring"}

    trade_dict = _call_ai(ctx_str, provider, api_key)
    conf = calibrate_confidence(context_confidence(_AGENT, context), _AGENT)

    if "error" in trade_dict:
        set_status(_AGENT, "idle")
        log_activity(_AGENT, "structured output failed", trade_dict["error"][:120], "warning")
        return {"error": trade_dict["error"]}

    trade_name = trade_dict.get("trade_name", "Structured Trade Idea")
    direction  = trade_dict.get("direction", "")
    rationale  = trade_dict.get("rationale", "")

    # Reconstruct human-readable narrative from validated fields
    narrative = "\n".join([
        f"TRADE: {trade_name}",
        f"DIRECTION: {direction}",
        f"TRIGGER: {trade_dict.get('trigger', '')}",
        f"TARGET: {trade_dict.get('target', '')}",
        f"RISK: {trade_dict.get('risk', '')}",
        f"RATIONALE: {rationale}",
    ])
    summary = f"{direction} — Trigger: {trade_dict.get('trigger', '')}"

    item_id = add_pending(
        agent_id=_AGENT,
        title=trade_name,
        summary=summary,
        rationale=f"{rationale}\n\nFull structure:\n{narrative}",
        confidence=conf,
        severity="warning" if regime_level >= 3 else "info",
        extra={"trade": trade_dict, "context": ctx_str},
    )

    log_activity(_AGENT, "trade idea submitted for approval", trade_name, "info")

    return {
        "narrative":  narrative,
        "trade":      trade_dict,
        "confidence": conf,
        "pending_id": item_id,
    }

"""
AI Trade Structurer — owns the Trade Ideas page.
Generates structured trade ideas (entry/exit/rationale/risk) and routes them
to the Pending Review Panel for human approval before display.
Cached 1 hour per idea set.
"""

from __future__ import annotations

import streamlit as st
from src.analysis.agent_state import (
    set_status, set_output, log_activity, calibrate_confidence,
    add_pending, is_enabled,
)

_SYSTEM = (
    "You are the AI Trade Structurer embedded in the Cross-Asset Spillover Monitor "
    "at Purdue University Daniels School of Business. "
    "You generate institutional cross-asset trade ideas based on regime, correlation, "
    "and spillover signals. Each idea must have: asset pair, direction, entry trigger, "
    "exit target, risk/stop, rationale, and confidence. "
    "Be precise and quantitative. No disclaimers."
)

_AGENT = "trade_structurer"


@st.cache_data(ttl=3600, show_spinner=False)
def _call_ai(context_str: str, provider: str, api_key: str) -> tuple[str, float]:
    prompt = (
        f"CURRENT MARKET CONTEXT:\n{context_str}\n\n"
        "Generate ONE specific, actionable cross-asset trade idea that is best supported "
        "by the current regime, correlation, and spillover data above. "
        "Structure your response as:\n"
        "TRADE: [name]\n"
        "DIRECTION: [Long X / Short Y]\n"
        "TRIGGER: [specific entry condition]\n"
        "TARGET: [exit level or time horizon]\n"
        "RISK: [stop/hedge]\n"
        "RATIONALE: [2-3 sentence explanation using the data above]\n"
        "CONFIDENCE: X%"
    )
    try:
        if provider == "anthropic":
            import anthropic as _ant
            client = _ant.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
                system=_SYSTEM,
            )
            text = resp.content[0].text.strip()
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
            )
            text = resp.choices[0].message.content.strip()

        conf = 0.60
        for line in text.split("\n")[-4:]:
            if "confidence" in line.lower() and "%" in line:
                import re
                m = re.search(r"(\d+)%", line)
                if m:
                    conf = int(m.group(1)) / 100
                    break

        return text, conf
    except Exception as e:
        return f"Trade Structurer unavailable: {e}", 0.0


def _parse_trade(text: str) -> dict:
    """Parse structured trade response into a dict."""
    result = {}
    field_map = {
        "TRADE": "name",
        "DIRECTION": "direction",
        "TRIGGER": "trigger",
        "TARGET": "target",
        "RISK": "risk",
        "RATIONALE": "rationale",
    }
    for line in text.split("\n"):
        for key, field in field_map.items():
            if line.upper().startswith(key + ":"):
                result[field] = line[len(key)+1:].strip()
    return result


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

    # Include peer signals if available
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
        # Normal market — no trade idea needed, just monitor
        set_status(_AGENT, "monitoring")
        log_activity(_AGENT, "normal regime — no active trade idea", "", "info")
        return {"status": "monitoring"}

    narrative, raw_conf = _call_ai(ctx_str, provider, api_key)
    conf = calibrate_confidence(raw_conf, _AGENT)

    if narrative.startswith("Trade Structurer unavailable"):
        set_status(_AGENT, "idle")
        return {"error": narrative}

    trade = _parse_trade(narrative)
    trade_name = trade.get("name", "Structured Trade Idea")
    direction  = trade.get("direction", "")
    rationale  = trade.get("rationale", narrative)

    summary = f"{direction} — Trigger: {trade.get('trigger','')}"

    # Route to Pending Review
    item_id = add_pending(
        agent_id=_AGENT,
        title=trade_name,
        summary=summary,
        rationale=f"{rationale}\n\nFull structure:\n{narrative}",
        confidence=conf,
        severity="warning" if regime_level >= 3 else "info",
        extra={"trade": trade, "context": ctx_str},
    )

    log_activity(_AGENT, "trade idea submitted for approval",
                 trade_name, "info")

    return {
        "narrative": narrative,
        "trade": trade,
        "confidence": conf,
        "pending_id": item_id,
    }

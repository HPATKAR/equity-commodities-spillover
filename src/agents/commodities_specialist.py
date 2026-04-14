"""
AI Commodities Specialist - owns the Commodities Watchlist page.
Interprets COT positioning, sector rotation, and supply shocks.
Cached 1 hour. Flags crowded positions to Risk Officer.
"""

from __future__ import annotations

import streamlit as st
from src.analysis.agent_state import (
    set_status, set_output, log_activity, calibrate_confidence,
    context_confidence, is_enabled,
)

_SYSTEM = (
    "You are the AI Commodities Specialist embedded in the Cross-Asset Spillover Monitor "
    "at Purdue University Daniels School of Business. "
    "You own commodity futures markets: energy, metals, agriculture. "
    "You interpret CFTC COT positioning, supply dynamics, and sector rotation. "
    "Write terse, institutional-grade commodity analysis. "
    "Be quantitative. No disclaimers. No greetings."
)

_AGENT = "commodities_specialist"


@st.cache_data(ttl=3600, show_spinner=False)
def _call_ai(context_str: str, provider: str, api_key: str) -> str:
    prompt = (
        f"COMMODITIES CONTEXT (live data):\n{context_str}\n\n"
        "Provide a 3–5 sentence assessment covering: "
        "1) which commodity sector is showing the most significant positioning extreme, "
        "2) whether the COT data signals a crowded trade reversal risk, "
        "3) which commodity-equity pair has the highest spillover risk right now."
    )
    import time
    from src.analysis.trace_logger import log_trace
    try:
        t0 = time.monotonic()
        if provider == "anthropic":
            import anthropic as _ant
            client = _ant.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=350,
                messages=[{"role": "user", "content": prompt}],
                system=_SYSTEM,
            )
            text = resp.content[0].text.strip()
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
                max_tokens=350, temperature=0.2,
            )
            text = resp.choices[0].message.content.strip()
            model_name = "gpt-4o"
        log_trace(_AGENT, provider, model_name, len(prompt), len(text),
                  (time.monotonic() - t0) * 1000)
        return text
    except Exception as e:
        return f"Commodities Specialist unavailable: {e}"


def run(
    context: dict,
    provider: str | None,
    api_key: str,
) -> dict:
    """
    context keys:
      top_performers: list[tuple[str, float]]   # [(name, pct), ...] 5-day
      worst_performers: list[tuple[str, float]]
      crowded_longs: list[tuple[str, float]]     # [(market, net_spec_pct), ...]
      crowded_shorts: list[tuple[str, float]]
      avg_corr: float
      regime_name: str
    """
    if not is_enabled(_AGENT):
        return {}

    set_status(_AGENT, "investigating")

    parts = []
    if context.get("top_performers"):
        items = ", ".join(f"{n} +{v:.1f}%" for n, v in context["top_performers"][:3])
        parts.append(f"Top 5d performers: {items}")
    if context.get("worst_performers"):
        items = ", ".join(f"{n} {v:.1f}%" for n, v in context["worst_performers"][:3])
        parts.append(f"Worst 5d performers: {items}")
    if context.get("crowded_longs"):
        items = ", ".join(f"{n} ({v:.0f}% OI)" for n, v in context["crowded_longs"][:3])
        parts.append(f"Crowded longs (COT): {items}")
    if context.get("crowded_shorts"):
        items = ", ".join(f"{n} ({v:.0f}% OI)" for n, v in context["crowded_shorts"][:3])
        parts.append(f"Crowded shorts (COT): {items}")
    if context.get("avg_corr") is not None:
        parts.append(f"Avg equity-commodity correlation (60d): {context['avg_corr']:.3f}")
    if context.get("regime_name"):
        parts.append(f"Market regime: {context['regime_name']}")

    if not parts:
        set_status(_AGENT, "idle")
        return {}

    ctx_str = "\n".join(parts)

    # Flag crowded positions → route to Risk Officer
    has_crowded = bool(context.get("crowded_longs") or context.get("crowded_shorts"))
    if has_crowded and not provider:
        log_activity(_AGENT, "crowded position flagged",
                     "routing to Risk Officer", "warning",
                     routed_to="risk_officer")

    if not provider:
        set_status(_AGENT, "monitoring")
        return {"status": "monitoring", "context": ctx_str}

    narrative = _call_ai(ctx_str, provider, api_key)
    conf = calibrate_confidence(context_confidence(_AGENT, context), _AGENT)

    # Route supply shock signals
    routed = None
    if any(k in narrative.lower() for k in ["supply shock", "crowded", "reversal", "squeeze"]):
        routed = "risk_officer"
        log_activity(_AGENT, "supply/positioning risk detected",
                     "routing to Risk Officer", "warning", routed_to="risk_officer")

    set_output(_AGENT, narrative, confidence=conf)
    log_activity(_AGENT, "commodity assessment published",
                 ctx_str.split("\n")[0], "info")

    return {
        "narrative": narrative,
        "confidence": conf,
        "context": ctx_str,
        "routed_to": routed,
    }

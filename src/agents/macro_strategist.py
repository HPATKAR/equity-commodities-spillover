"""
AI Macro Strategist - owns the Macro Intelligence Dashboard page.
Analyses yield curve, inflation, Fed policy, and GDP context.
Cached 1 hour. Routes hawkish/dovish pivots to Risk Officer.
"""

from __future__ import annotations

import streamlit as st
from src.analysis.agent_state import (
    set_status, set_output, log_activity, calibrate_confidence,
    is_enabled, AGENTS,
)

_SYSTEM = (
    "You are the AI Macro Strategist embedded in the Cross-Asset Spillover Monitor "
    "at Purdue University Daniels School of Business. "
    "You monitor yield curves, inflation dynamics, Fed policy, and macro data. "
    "Write terse, institutional-grade macro analysis in 3–5 sentences. "
    "Be precise and quantitative. No disclaimers. No greetings."
)

_AGENT = "macro_strategist"


@st.cache_data(ttl=3600, show_spinner=False)
def _call_ai(context_str: str, provider: str, api_key: str) -> tuple[str, float]:
    """Returns (narrative, raw_confidence). Cached 1 hour."""
    prompt = (
        f"MACRO CONTEXT (live data):\n{context_str}\n\n"
        "Provide a 3–5 sentence macro intelligence assessment covering: "
        "1) yield curve signal (recession/expansion), "
        "2) inflation trajectory and Fed posture, "
        "3) the most important cross-asset risk from current macro conditions. "
        "End with a one-line CONFIDENCE: X% statement."
    )
    try:
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

        # Parse confidence from last line
        conf = 0.65  # default
        for line in text.split("\n")[-3:]:
            if "confidence" in line.lower() and "%" in line:
                import re
                m = re.search(r"(\d+)%", line)
                if m:
                    conf = int(m.group(1)) / 100
                    break

        return text, conf

    except Exception as e:
        return f"Macro Strategist unavailable: {e}", 0.0


def run(
    context: dict,
    provider: str | None,
    api_key: str,
) -> dict:
    """
    Main entry point - called from the Macro Dashboard page.

    context keys (all optional, agent degrades gracefully):
      yield_curve_spread: float (10Y-2Y in %)
      cpi_yoy: float
      fed_rate: float
      gdp_growth: float
      ism_pmi: float
      regime_name: str
      risk_score: float
    """
    if not is_enabled(_AGENT):
        return {}

    set_status(_AGENT, "investigating")

    # Build context string from whatever data is available
    parts = []
    if context.get("yield_curve_spread") is not None:
        sign = "+" if context["yield_curve_spread"] >= 0 else ""
        parts.append(f"Yield curve (10Y-2Y): {sign}{context['yield_curve_spread']:.2f}%")
    if context.get("cpi_yoy") is not None:
        parts.append(f"CPI YoY: {context['cpi_yoy']:.1f}%")
    if context.get("fed_rate") is not None:
        parts.append(f"Fed Funds Rate: {context['fed_rate']:.2f}%")
    if context.get("gdp_growth") is not None:
        parts.append(f"Real GDP Growth (latest): {context['gdp_growth']:.1f}%")
    if context.get("ism_pmi") is not None:
        parts.append(f"ISM Manufacturing PMI: {context['ism_pmi']:.1f}")
    if context.get("regime_name"):
        parts.append(f"Cross-asset correlation regime: {context['regime_name']}")
    if context.get("risk_score") is not None:
        parts.append(f"Composite stress score: {context['risk_score']:.0f}/100")

    if not parts:
        set_status(_AGENT, "idle")
        return {}

    # Include peer signals from orchestrator if available
    peer_ctx = context.get("peer_context", {})
    if peer_ctx:
        peer_str = "; ".join(
            f"{k.replace('_', ' ')}: {str(v)[:120]}"
            for k, v in peer_ctx.items() if v
        )
        if peer_str:
            parts.append(f"Peer agent context: {peer_str}")

    ctx_str = "\n".join(parts)

    if not provider:
        # No API key - mark monitoring, no AI call
        set_status(_AGENT, "monitoring")
        log_activity(_AGENT, "monitoring", "no API key - skipping AI call", "info")
        return {"status": "monitoring", "context": ctx_str}

    narrative, raw_conf = _call_ai(ctx_str, provider, api_key)
    conf = calibrate_confidence(raw_conf, _AGENT)

    # Detect hawkish pivot keywords → route to Risk Officer
    routed = None
    lower_n = narrative.lower()
    if any(k in lower_n for k in ["hawkish", "rate hike", "tightening", "inversion deepening"]):
        routed = "risk_officer"
        log_activity(_AGENT, "hawkish pivot detected", "routing to Risk Officer",
                     "warning", routed_to="risk_officer")
    elif any(k in lower_n for k in ["dovish", "cut", "easing", "pivot"]):
        routed = "risk_officer"
        log_activity(_AGENT, "dovish pivot detected", "routing to Risk Officer",
                     "info", routed_to="risk_officer")

    set_output(_AGENT, narrative, confidence=conf)
    log_activity(_AGENT, "macro assessment published",
                 ctx_str.split("\n")[0], "info")

    return {
        "narrative": narrative,
        "confidence": conf,
        "context": ctx_str,
        "routed_to": routed,
    }

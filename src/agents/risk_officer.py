"""
AI Risk Officer - morning briefing synthesiser and cross-agent routing hub.
Owns the Overview page. Synthesises regime, risk score, alert signals,
and peer intelligence into a terse morning briefing.
Cached 1 hour. Routes high-priority signals to specialist agents.
"""

from __future__ import annotations

import streamlit as st
from src.analysis.agent_state import (
    set_status, set_output, log_activity, calibrate_confidence,
    context_confidence, is_enabled,
)

_SYSTEM = (
    "You are the AI Risk Officer embedded in the Cross-Asset Spillover Monitor "
    "at Purdue University Daniels School of Business. "
    "You own the morning briefing. Your job is to synthesise the current market regime, "
    "risk score, cross-asset correlation dynamics, and live alert signals into a "
    "terse, institutional-grade morning briefing. "
    "Be direct and quantitative. Name the biggest risk, its transmission path, "
    "and what the portfolio should watch today. No disclaimers. No greetings."
)

_AGENT = "risk_officer"


@st.cache_data(ttl=3600, show_spinner=False)
def _call_ai(context_str: str, provider: str, api_key: str) -> str:
    """Returns narrative text. Cached 1 hour."""
    import time
    from src.analysis.trace_logger import log_trace

    prompt = (
        f"MORNING BRIEFING CONTEXT (live data):\n{context_str}\n\n"
        "Provide a 3–5 sentence morning risk briefing covering: "
        "1) current correlation regime and what it means for cross-asset positioning, "
        "2) the highest-priority live alert or risk signal and its transmission path "
        "(equity → commodity or commodity → equity), "
        "3) one specific action or area of vigilance for today's session. "
        "Be precise about magnitudes and direction."
    )
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
        return f"Risk Officer unavailable: {e}"


def run(
    context: dict,
    provider: str | None,
    api_key: str,
) -> dict:
    """
    Main entry point - called from the Overview page.

    context keys (all optional):
      regime_name: str
      regime_level: int
      risk_score: float
      avg_corr: float
      corr_delta: float            # 1M change in avg correlation
      best_equity: str
      worst_equity: str
      best_commodity: str
      worst_commodity: str
      n_alerts: int
      alert_categories: list[str]  # e.g. ["cot", "stress", "regime_change"]
      alert_summaries: list[str]   # brief descriptions of active alerts
    """
    if not is_enabled(_AGENT):
        return {}

    set_status(_AGENT, "investigating")

    parts = []
    if context.get("regime_name"):
        lvl = context.get("regime_level", 1)
        parts.append(f"Correlation regime: {context['regime_name']} (level {lvl}/3)")
    if context.get("risk_score") is not None:
        parts.append(f"Composite risk score: {context['risk_score']:.0f}/100")
    if context.get("avg_corr") is not None:
        delta_str = ""
        if context.get("corr_delta") is not None:
            delta_str = f" ({context['corr_delta']:+.3f} vs 1M ago)"
        parts.append(f"60d avg |cross-asset corr|: {context['avg_corr']:.3f}{delta_str}")

    if context.get("best_equity") and context.get("worst_equity"):
        parts.append(
            f"1M equity range: {context['best_equity']} (best) → "
            f"{context['worst_equity']} (worst)"
        )
    if context.get("best_commodity"):
        parts.append(f"1M top commodity: {context['best_commodity']}")
    if context.get("worst_commodity"):
        parts.append(f"1M worst commodity: {context['worst_commodity']}")

    n_alerts = context.get("n_alerts", 0)
    if n_alerts:
        cats = context.get("alert_categories", [])
        cat_str = f" [{', '.join(cats[:4])}]" if cats else ""
        parts.append(f"Active alerts: {n_alerts}{cat_str}")

    for summary in context.get("alert_summaries", [])[:4]:
        parts.append(f"  ⚠ {summary}")

    if not parts:
        set_status(_AGENT, "idle")
        return {}

    ctx_str = "\n".join(parts)

    if not provider:
        set_status(_AGENT, "monitoring")
        log_activity(_AGENT, "monitoring - no API key", ctx_str.split("\n")[0], "info")
        return {"status": "monitoring", "context": ctx_str}

    narrative = _call_ai(ctx_str, provider, api_key)
    conf = calibrate_confidence(context_confidence(_AGENT, context), _AGENT)

    # Route to specialists based on alert categories
    cats = set(context.get("alert_categories", []))
    lower_n = narrative.lower()

    routed = []
    if "cot" in cats or any(k in lower_n for k in ["positioning", "crowded", "cot"]):
        routed.append("commodities_specialist")
        log_activity(_AGENT, "COT/positioning signal", "routing to Commodities Specialist",
                     "warning", routed_to="commodities_specialist")
    if "stress" in cats or any(k in lower_n for k in ["stress", "tail risk", "drawdown"]):
        routed.append("stress_engineer")
        log_activity(_AGENT, "elevated stress signal", "routing to Stress Engineer",
                     "warning", routed_to="stress_engineer")
    if any(k in lower_n for k in ["geopolit", "conflict", "sanctions", "hormuz", "ukraine"]):
        routed.append("geopolitical_analyst")
        log_activity(_AGENT, "geopolitical signal detected", "routing to Geo Intel",
                     "warning", routed_to="geopolitical_analyst")
    if any(k in lower_n for k in ["yield curve", "fed", "inflation", "cpi", "hawkish"]):
        routed.append("macro_strategist")
        log_activity(_AGENT, "macro regime signal", "routing to Macro Strategist",
                     "info", routed_to="macro_strategist")

    set_output(_AGENT, narrative, confidence=conf)
    log_activity(_AGENT, "morning briefing published",
                 f"regime: {context.get('regime_name','?')} | "
                 f"risk: {context.get('risk_score', 0):.0f}/100",
                 "critical" if context.get("regime_level", 1) >= 3 else "warning"
                 if context.get("regime_level", 1) >= 2 else "info")

    return {
        "narrative": narrative,
        "confidence": conf,
        "context": ctx_str,
        "routed_to": routed,
    }

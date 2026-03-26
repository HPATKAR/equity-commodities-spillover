"""
AI Geopolitical Intelligence Analyst — owns Geopolitical Triggers + War Impact Map.
Interprets conflict events, sanctions, and supply-chain disruption risks.
Cached 1 hour. High-severity events escalate to Risk Officer.
"""

from __future__ import annotations

import streamlit as st
from src.analysis.agent_state import (
    set_status, set_output, log_activity, calibrate_confidence,
    is_enabled,
)

_SYSTEM = (
    "You are the AI Geopolitical Intelligence Analyst embedded in the Cross-Asset "
    "Spillover Monitor at Purdue University Daniels School of Business. "
    "You monitor active conflict zones, sanctions regimes, and commodity supply disruption risks. "
    "Analyse how geopolitical events transmit to commodity and equity markets. "
    "Be precise. No disclaimers. No greetings."
)

_AGENT = "geopolitical_analyst"


@st.cache_data(ttl=3600, show_spinner=False)
def _call_ai(context_str: str, provider: str, api_key: str) -> tuple[str, float]:
    prompt = (
        f"GEOPOLITICAL CONTEXT:\n{context_str}\n\n"
        "Provide a 3–5 sentence geopolitical risk assessment covering: "
        "1) the highest-severity active conflict and its commodity transmission channel, "
        "2) sanctions or trade-restriction risk to energy / metals supply, "
        "3) which equity regions are most exposed to current geopolitical stress. "
        "End with CONFIDENCE: X%."
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

        conf = 0.55
        for line in text.split("\n")[-3:]:
            if "confidence" in line.lower() and "%" in line:
                import re
                m = re.search(r"(\d+)%", line)
                if m:
                    conf = int(m.group(1)) / 100
                    break

        return text, conf
    except Exception as e:
        return f"Geopolitical Analyst unavailable: {e}", 0.0


def run(
    context: dict,
    provider: str | None,
    api_key: str,
) -> dict:
    """
    context keys:
      active_events: list[dict]  # [{name, date, severity, region, commodity_impact}, ...]
      n_events: int
      high_severity: int         # count of high/critical events
      affected_commodities: list[str]
      affected_regions: list[str]
      regime_name: str
      risk_score: float
    """
    if not is_enabled(_AGENT):
        return {}

    set_status(_AGENT, "investigating")

    parts = []
    n = context.get("n_events", 0)
    hi = context.get("high_severity", 0)

    if n:
        parts.append(f"Active geopolitical events: {n} ({hi} high/critical severity)")

    events = context.get("active_events", [])[:5]
    for ev in events:
        ev_name = ev.get("name", ev.get("event", "Unknown event"))
        ev_sev  = ev.get("severity", "medium")
        ev_reg  = ev.get("region", "")
        ev_cmd  = ev.get("commodity_impact", "")
        line = f"  [{ev_sev.upper()}] {ev_name}"
        if ev_reg:
            line += f" ({ev_reg})"
        if ev_cmd:
            line += f" — commodity impact: {ev_cmd}"
        parts.append(line)

    if context.get("affected_commodities"):
        parts.append(f"Affected commodities: {', '.join(context['affected_commodities'][:5])}")
    if context.get("affected_regions"):
        parts.append(f"Exposed equity regions: {', '.join(context['affected_regions'][:4])}")
    if context.get("regime_name"):
        parts.append(f"Market regime: {context['regime_name']}")
    if context.get("risk_score") is not None:
        parts.append(f"Composite stress: {context['risk_score']:.0f}/100")

    if not parts:
        set_status(_AGENT, "idle")
        return {}

    ctx_str = "\n".join(parts)

    # High severity → immediate log to Risk Officer
    if hi >= 2:
        log_activity(_AGENT, f"{hi} high-severity events active",
                     "routing threat assessment to Risk Officer", "critical",
                     routed_to="risk_officer")

    if not provider:
        set_status(_AGENT, "monitoring")
        return {"status": "monitoring", "context": ctx_str}

    narrative, raw_conf = _call_ai(ctx_str, provider, api_key)
    conf = calibrate_confidence(raw_conf, _AGENT)

    routed = None
    if any(k in narrative.lower() for k in ["critical", "escalation", "disruption", "closure"]):
        routed = "risk_officer"
        log_activity(_AGENT, "escalation risk identified",
                     "routing to Risk Officer", "critical", routed_to="risk_officer")

    set_output(_AGENT, narrative, confidence=conf)
    log_activity(_AGENT, "geo-intel assessment published",
                 f"{n} events tracked", "info")

    return {
        "narrative": narrative,
        "confidence": conf,
        "context": ctx_str,
        "routed_to": routed,
    }

"""
AI Portfolio Stress Engineer — owns the Stress Tester page.
Interprets scenario stress results, tail risk, and shock transmission paths.
Cached 1 hour. Extreme scenarios escalate to Risk Officer.
"""

from __future__ import annotations

import streamlit as st
from src.analysis.agent_state import (
    set_status, set_output, log_activity, calibrate_confidence,
    is_enabled,
)

_SYSTEM = (
    "You are the AI Portfolio Stress Engineer embedded in the Cross-Asset Spillover Monitor "
    "at Purdue University Daniels School of Business. "
    "You design and interpret stress scenarios: commodity supply shocks, equity crashes, "
    "rate spikes, and combined tail events. "
    "Be quantitative and concise. Identify the most dangerous transmission path. "
    "No disclaimers. No greetings."
)

_AGENT = "stress_engineer"


@st.cache_data(ttl=3600, show_spinner=False)
def _call_ai(context_str: str, provider: str, api_key: str) -> tuple[str, float]:
    prompt = (
        f"STRESS TEST RESULTS (live scenarios):\n{context_str}\n\n"
        "Provide a 3–5 sentence stress assessment covering: "
        "1) the scenario with the worst expected portfolio drawdown, "
        "2) which asset class is the primary transmission vector, "
        "3) what hedge or risk-reduction action is most impactful right now. "
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

        conf = 0.62
        for line in text.split("\n")[-3:]:
            if "confidence" in line.lower() and "%" in line:
                import re
                m = re.search(r"(\d+)%", line)
                if m:
                    conf = int(m.group(1)) / 100
                    break

        return text, conf
    except Exception as e:
        return f"Stress Engineer unavailable: {e}", 0.0


def run(
    context: dict,
    provider: str | None,
    api_key: str,
) -> dict:
    """
    context keys:
      scenarios: list[dict]  # [{name, shock_type, magnitude, impact_pct, transmission}, ...]
      worst_scenario: str
      worst_impact: float     # portfolio drawdown %
      avg_impact: float
      regime_name: str
      risk_score: float
      n_scenarios: int
    """
    if not is_enabled(_AGENT):
        return {}

    set_status(_AGENT, "investigating")

    parts = []
    if context.get("n_scenarios"):
        parts.append(f"Scenarios evaluated: {context['n_scenarios']}")
    if context.get("worst_scenario") and context.get("worst_impact") is not None:
        parts.append(
            f"Worst scenario: {context['worst_scenario']} "
            f"(portfolio impact: {context['worst_impact']:+.1f}%)"
        )
    if context.get("avg_impact") is not None:
        parts.append(f"Average cross-scenario impact: {context['avg_impact']:+.1f}%")

    for sc in context.get("scenarios", [])[:4]:
        name   = sc.get("name", "Unknown")
        impact = sc.get("impact_pct", 0)
        tx     = sc.get("transmission", "")
        line   = f"  {name}: {impact:+.1f}%"
        if tx:
            line += f" (via {tx})"
        parts.append(line)

    if context.get("regime_name"):
        parts.append(f"Current regime: {context['regime_name']}")
    if context.get("risk_score") is not None:
        parts.append(f"Composite stress: {context['risk_score']:.0f}/100")

    if not parts:
        set_status(_AGENT, "idle")
        return {}

    ctx_str = "\n".join(parts)

    # Extreme tail risk → alert Risk Officer
    worst = context.get("worst_impact", 0) or 0
    if worst < -15:
        log_activity(_AGENT, f"extreme tail scenario: {worst:.1f}%",
                     "routing to Risk Officer", "critical",
                     routed_to="risk_officer")

    if not provider:
        set_status(_AGENT, "monitoring")
        return {"status": "monitoring", "context": ctx_str}

    narrative, raw_conf = _call_ai(ctx_str, provider, api_key)
    conf = calibrate_confidence(raw_conf, _AGENT)

    routed = None
    if worst < -10:
        routed = "risk_officer"
        log_activity(_AGENT, "severe stress scenario detected",
                     "routing tail risk to Risk Officer", "warning",
                     routed_to="risk_officer")

    set_output(_AGENT, narrative, confidence=conf)
    log_activity(_AGENT, "stress assessment published",
                 f"{context.get('n_scenarios', 0)} scenarios", "info")

    return {
        "narrative": narrative,
        "confidence": conf,
        "context": ctx_str,
        "routed_to": routed,
    }

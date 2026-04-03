"""
AI Signal Auditor - owns the Signal Performance Review page.
Calibrates confidence scores across all agents. Reviews model accuracy,
Granger causality hit rates, and signal backtest performance.
Cached 1 hour. Updates calibration factors in session_state.
"""

from __future__ import annotations

import streamlit as st
from src.analysis.agent_state import (
    set_status, set_output, log_activity, calibrate_confidence,
    is_enabled, init_agents, AGENTS,
)

_SYSTEM = (
    "You are the AI Signal Auditor embedded in the Cross-Asset Spillover Monitor "
    "at Purdue University Daniels School of Business. "
    "You review signal model performance, Granger causality hit rates, and "
    "calibrate confidence scores across the AI workforce. "
    "Be rigorous. Identify signal decay and overfit risk. No disclaimers."
)

_AGENT = "signal_auditor"


@st.cache_data(ttl=3600, show_spinner=False)
def _call_ai(context_str: str, provider: str, api_key: str) -> tuple[str, float]:
    prompt = (
        f"SIGNAL PERFORMANCE DATA:\n{context_str}\n\n"
        "Provide a 3–5 sentence signal audit covering: "
        "1) which Granger pairs have the strongest and weakest recent directional accuracy, "
        "2) whether model confidence scores need upward or downward calibration, "
        "3) any signs of signal decay or regime mismatch that reduce reliability. "
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

        conf = 0.70
        for line in text.split("\n")[-3:]:
            if "confidence" in line.lower() and "%" in line:
                import re
                m = re.search(r"(\d+)%", line)
                if m:
                    conf = int(m.group(1)) / 100
                    break

        return text, conf
    except Exception as e:
        return f"Signal Auditor unavailable: {e}", 0.0


def run(
    context: dict,
    provider: str | None,
    api_key: str,
) -> dict:
    """
    context keys:
      granger_hit_rates: dict[str, float]     # {pair_name: hit_rate_pct}
      best_pair: str
      worst_pair: str
      avg_hit_rate: float
      n_signals: int
      signal_decay: bool
      regime_name: str
      forecast_errors: dict[str, float]       # optional
    """
    if not is_enabled(_AGENT):
        return {}

    set_status(_AGENT, "investigating")

    parts = []
    if context.get("n_signals"):
        parts.append(f"Signals evaluated: {context['n_signals']}")
    if context.get("avg_hit_rate") is not None:
        parts.append(f"Average directional hit rate: {context['avg_hit_rate']:.1f}%")
    if context.get("best_pair"):
        rate = context.get("granger_hit_rates", {}).get(context["best_pair"])
        rate_str = f" ({rate:.1f}%)" if rate else ""
        parts.append(f"Best Granger pair: {context['best_pair']}{rate_str}")
    if context.get("worst_pair"):
        rate = context.get("granger_hit_rates", {}).get(context["worst_pair"])
        rate_str = f" ({rate:.1f}%)" if rate else ""
        parts.append(f"Weakest Granger pair: {context['worst_pair']}{rate_str}")
    if context.get("signal_decay"):
        parts.append("Signal decay detected - recent hit rates below historical average.")
    if context.get("regime_name"):
        parts.append(f"Current regime: {context['regime_name']}")

    if not parts:
        set_status(_AGENT, "idle")
        return {}

    ctx_str = "\n".join(parts)

    # Compute calibration factors from hit rates - vary by agent domain
    calibration = {}
    hit_rates = context.get("granger_hit_rates", {})
    if hit_rates:
        avg = sum(hit_rates.values()) / len(hit_rates)

        # Base factor from aggregate signal strength
        if avg > 60:
            base_factor = min(1.0 + (avg - 60) / 200, 1.15)
        elif avg < 50:
            base_factor = max(1.0 - (50 - avg) / 100, 0.70)
        else:
            base_factor = 1.0

        # Commodity-facing agents get a larger boost/penalty - more model-dependent
        # Geopolitical analyst is fundamentally qualitative - smaller adjustment
        # Trade structurer requires highest bar - shrink more aggressively if signals weak
        domain_modifiers = {
            "risk_officer":          0.0,    # additive offset from base
            "macro_strategist":      0.02,   # slight boost (macro signal is more persistent)
            "commodities_specialist": 0.0,
            "geopolitical_analyst":  -0.05,  # geo uncertainty always higher
            "stress_engineer":       0.01,
            "signal_auditor":        0.0,    # self - no adjustment
            "trade_structurer":     -0.03 if avg < 55 else 0.0,  # penalise if weak signals
            "quality_officer":       0.05,   # CQO is always checking known failure modes
        }
        for aid in AGENTS:
            mod = domain_modifiers.get(aid, 0.0)
            calibration[aid] = min(max(base_factor + mod, 0.50), 1.15)

    # Store calibration in auditor's extra
    init_agents()
    auditor_state = st.session_state["agents"]["signal_auditor"]
    if "extra" not in auditor_state:
        auditor_state["extra"] = {}
    auditor_state["extra"]["calibration"] = calibration

    # Log any signal decay warning
    if context.get("signal_decay"):
        log_activity(_AGENT, "signal decay detected",
                     "calibration factors reduced", "warning")

    if not provider:
        set_status(_AGENT, "monitoring")
        log_activity(_AGENT, "calibration updated", f"{len(calibration)} agents calibrated", "info")
        return {"status": "monitoring", "calibration": calibration}

    narrative, raw_conf = _call_ai(ctx_str, provider, api_key)
    # Auditor uses its own raw confidence - no recursive calibration
    conf = min(max(raw_conf, 0.0), 1.0)

    set_output(_AGENT, narrative, confidence=conf)
    log_activity(_AGENT, "signal audit published",
                 f"avg hit rate {context.get('avg_hit_rate', 0):.1f}%", "info")

    return {
        "narrative": narrative,
        "confidence": conf,
        "calibration": calibration,
        "context": ctx_str,
    }

"""
AI Chief Quality Officer - cross-page data integrity and assumption auditor.
Deployed on every analysis page. Flags spurious correlations, insufficient
sample sizes, overstated confidence, selection bias, regime mismatch,
cherry-picked windows, and methodological errors in whatever analysis
is shown on the current page.

No diplomatic hedging. Call it as it is.
Cached 30 minutes - shorter than other agents because bad data rotates.
"""

from __future__ import annotations

import streamlit as st
from src.analysis.agent_state import (
    set_status, set_output, log_activity, context_confidence, is_enabled,
)

_SYSTEM = (
    "You are the Chief Quality Officer (CQO) embedded in the Cross-Asset "
    "Spillover Monitor at Purdue University Daniels School of Business. "
    "Your sole job is to spot bullshit - bad data, spurious correlations, "
    "overstated confidence, selection bias, look-ahead bias, insufficient "
    "sample sizes, regime mismatch, assumption stacking, and methodological "
    "shortcuts in the analysis you are shown. "
    "You are not here to validate. You are here to break things. "
    "Be direct, blunt, and specific. No diplomatic hedging. No disclaimers. "
    "Format your response as numbered flags: "
    "start each flag on a new line as '⚠ FLAG N: [title] - [precise explanation]'. "
    "If something is actually solid, you may note it as '✓ PASS: [item]'. "
    "End with a single line: SEVERITY: Critical | High | Medium | Low."
)

_AGENT = "quality_officer"


@st.cache_data(ttl=1800, show_spinner=False)
def _call_ai(context_str: str, page: str, provider: str, api_key: str) -> str:
    """Returns narrative text. Cached 30 minutes."""
    prompt = (
        f"PAGE: {page}\n\n"
        f"ANALYSIS CONTEXT:\n{context_str}\n\n"
        "Audit this analysis. Identify every data quality issue, methodological flaw, "
        "and unwarranted assumption present. Be specific about what is wrong and why it matters. "
        "Do not flag things just to look thorough - only flag real problems. "
        "If the sample is too small, state the actual N and what N you would need. "
        "If a correlation is spurious, name the confound. "
        "If a score is hardcoded and divorced from live data, say so explicitly. "
        "End with SEVERITY: Critical | High | Medium | Low."
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
                max_tokens=450,
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
                max_tokens=450, temperature=0.15,
            )
            text = resp.choices[0].message.content.strip()
            model_name = "gpt-4o"
        log_trace(_AGENT, provider, model_name, len(prompt), len(text),
                  (time.monotonic() - t0) * 1000)
        return text
    except Exception as e:
        return f"Chief Quality Officer unavailable: {e}"


def run(
    context: dict,
    provider: str | None,
    api_key: str,
    page: str = "unknown",
) -> dict:
    """
    Universal entry point. Each page passes a context dict with whatever
    data quality signals are observable from that page's computation.

    Common context keys (all optional - CQO works with whatever it gets):
      page:               str   - which page is calling
      n_obs:              int   - number of observations in primary dataset
      date_range:         str   - e.g. "2005-01-01 to 2024-12-31"
      n_assets:           int   - assets in analysis
      sample_warning:     bool  - True if N < 252 (1yr)
      model:              str   - model used (VAR, DCC-GARCH, etc.)
      p_values:           dict  - {pair: p_value} for any significance tests
      max_correlation:    float - highest pairwise correlation in dataset
      corr_pairs:         list  - [(asset_a, asset_b, corr)] top correlations
      regime:             str   - current market regime
      regime_change:      bool  - regime changed recently
      confidence_scores:  dict  - {agent/model: confidence}
      hardcoded_scores:   bool  - True if scores are static/hardcoded
      event_window_days:  int   - event window length
      n_events:           int   - number of events analysed
      stress_scenario:    str   - stress test scenario name
      trade_confidence:   float - trade idea confidence
      trade_has_stop:     bool  - whether trade idea has a stop loss defined
      data_gaps:          int   - number of NaN/missing data points
      backtest_period:    str   - backtest window used
      lookahead_risk:     bool  - any known look-ahead bias risk
      assumption_count:   int   - number of model assumptions stacked
      notes:              list[str] - any free-form context from the page
    """
    if not is_enabled(_AGENT):
        return {}

    set_status(_AGENT, "investigating")

    # Build context string for the LLM
    parts = [f"Page context: {page}"]

    # Data size / coverage
    if context.get("n_obs") is not None:
        n = context["n_obs"]
        flag = " ⚠ LOW" if n < 252 else (" ⚠ BORDERLINE" if n < 504 else "")
        parts.append(f"Observations: {n}{flag}")
    if context.get("date_range"):
        parts.append(f"Date range: {context['date_range']}")
    if context.get("n_assets") is not None:
        parts.append(f"Assets in analysis: {context['n_assets']}")
    if context.get("data_gaps") is not None and context["data_gaps"] > 0:
        parts.append(f"Missing data points: {context['data_gaps']} ⚠")

    # Model / methodology
    if context.get("model"):
        parts.append(f"Model used: {context['model']}")
    if context.get("backtest_period"):
        parts.append(f"Backtest window: {context['backtest_period']}")
    if context.get("lookahead_risk"):
        parts.append("Look-ahead bias risk: YES ⚠")
    if context.get("assumption_count") is not None:
        n_a = context["assumption_count"]
        flag = " ⚠ STACKING RISK" if n_a >= 4 else ""
        parts.append(f"Model assumptions stacked: {n_a}{flag}")

    # Statistical signals
    if context.get("p_values"):
        for pair, pv in list(context["p_values"].items())[:8]:
            sig = "sig" if pv < 0.05 else ("borderline" if pv < 0.10 else "NOT sig ⚠")
            parts.append(f"  p-value {pair}: {pv:.4f} ({sig})")
    if context.get("max_correlation") is not None:
        mc = context["max_correlation"]
        flag = " ⚠ NEAR-PERFECT - multicollinearity?" if mc > 0.92 else \
               (" ⚠ HIGH" if mc > 0.80 else "")
        parts.append(f"Max pairwise correlation: {mc:.3f}{flag}")
    if context.get("corr_pairs"):
        for a, b, c in context["corr_pairs"][:4]:
            parts.append(f"  {a} ↔ {b}: {c:.3f}")

    # Regime context
    if context.get("regime"):
        parts.append(f"Market regime: {context['regime']}")
    if context.get("regime_change"):
        parts.append("Regime change detected recently - model may be mis-calibrated ⚠")

    # Event analysis
    if context.get("n_events") is not None:
        flag = " ⚠ LOW for robust inference" if context["n_events"] < 5 else ""
        parts.append(f"Events analysed: {context['n_events']}{flag}")
    if context.get("event_window_days") is not None:
        parts.append(f"Event window: {context['event_window_days']} days")

    # Hardcoded / static scores
    if context.get("hardcoded_scores"):
        parts.append("Score source: HARDCODED static values - not live market data ⚠")

    # Confidence scores
    if context.get("confidence_scores"):
        for model, sc in context["confidence_scores"].items():
            flag = " ⚠ OVERCONFIDENT?" if sc > 0.85 else \
                   (" ⚠ LOW" if sc < 0.40 else "")
            parts.append(f"  Confidence [{model}]: {sc:.0%}{flag}")

    # Trade-specific
    if context.get("stress_scenario"):
        parts.append(f"Stress scenario: {context['stress_scenario']}")
    if context.get("trade_confidence") is not None:
        tc = context["trade_confidence"]
        flag = " ⚠ OVERCONFIDENT for discretionary trade" if tc > 0.80 else ""
        parts.append(f"Trade confidence: {tc:.0%}{flag}")
    if context.get("trade_has_stop") is False:
        parts.append("Stop loss defined: NO ⚠ - open-ended downside")

    # Free-form notes from the page
    if context.get("notes"):
        parts.append("Additional notes:")
        for note in context["notes"][:6]:
            parts.append(f"  - {note}")

    if len(parts) <= 1:
        set_status(_AGENT, "idle")
        return {}

    ctx_str = "\n".join(parts)

    # Always log that CQO ran
    log_activity(_AGENT, f"quality audit: {page}", "scanning for flags", "info")

    if not provider:
        set_status(_AGENT, "monitoring")
        return {"status": "monitoring", "context": ctx_str}

    narrative = _call_ai(ctx_str, page, provider, api_key)
    conf = context_confidence(_AGENT, context)

    # If any critical flags detected, log at warning level
    critical_keywords = ["critical", "look-ahead", "hardcoded", "spurious", "p-hack"]
    severity = "warning" if any(k in narrative.lower() for k in critical_keywords) else "info"

    set_output(_AGENT, narrative, confidence=conf)
    sev_label = _extract_severity(narrative)
    log_activity(_AGENT, f"quality audit complete: {page}",
                 f"severity: {sev_label}", severity)

    result = {
        "narrative": narrative,
        "confidence": conf,
        "context": ctx_str,
        "page": page,
    }

    # Trigger active remediation for any non-Low severity flags
    if sev_label not in ("Low", "Unknown"):
        try:
            from src.agents.remediation_router import run_remediation
            run_remediation(result, page, provider, api_key)
        except Exception as _rem_err:
            log_activity(_AGENT, "remediation routing failed", str(_rem_err)[:80], "warning")

    return result


def _extract_severity(text: str) -> str:
    """Pull severity label from CQO output."""
    import re
    m = re.search(r"SEVERITY:\s*(Critical|High|Medium|Low)", text, re.IGNORECASE)
    return m.group(1) if m else "Unknown"

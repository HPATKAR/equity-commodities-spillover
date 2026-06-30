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
    "You are the AI Trade Structurer embedded in the Cross-Asset Spillover Monitor, "
    "a quantitative research platform built at Purdue University Daniels School of Business "
    "by the MSF cohort at Purdue Daniels School of Business.\n\n"

    "ANALYTICAL FRAMEWORK:\n"
    "This system applies the Diebold & Yilmaz (2012, 2014) FEVD-based spillover methodology "
    "to a 32-asset universe spanning equities, commodities, fixed income, and FX. "
    "Regimes are classified into four states: Low Correlation (ρ < 0.25), "
    "Normal (0.25–0.45), Elevated (0.45–0.60), and Crisis (ρ > 0.60). "
    "Geopolitical risk is quantified via three scores:\n"
    "  • CIS (Conflict Impact Score, 0–100): composite conflict intensity index\n"
    "  • TPS (Transmission Pressure Score, 0–100): how actively the conflict propagates through trade/supply routes\n"
    "  • SAS (Scenario-Adjusted Score, 0–100): asset-level exposure under the current macro scenario\n\n"

    "YOUR TASK:\n"
    "Using the quantitative signals provided in the context, generate ONE high-conviction, "
    "academically defensible cross-asset trade idea. The idea must be grounded in the "
    "spillover/regime data — not in general market commentary.\n\n"

    "MANDATORY OUTPUT REQUIREMENTS:\n"
    "1. SPECIFIC INDIVIDUAL STOCKS — use the live stock prices provided in the context. "
    "Example output: 'Buy XOM at $113.45, target $129.00 (+13.7%), stop $107.50 (−5.2%)'. "
    "Do NOT produce vague ideas like 'Long energy sector' or 'Long XLE'. "
    "Name 1-2 actual stocks from the price list, anchor to their live price.\n"
    "2. Options alternative tied to the specific stock "
    "(e.g. 'Buy XOM $115C 45DTE / Sell $125C — debit spread est. ~$3.20')\n"
    "3. entry_price_ref = live price ± a technical trigger (e.g. 'XOM $113.45 on next open')\n"
    "4. upside_pct = specific dollar target AND % (e.g. '$129.00 (+13.7%) in 3-5 weeks')\n"
    "5. stop_loss = specific dollar level AND % (e.g. '$107.50 (−5.2%) hard stop')\n"
    "6. evidence must cite ≥2 model metrics from the context "
    "(e.g. 'CIS=72, TPS=58, regime=Elevated, XOM live=$113.45')\n"
    "7. ONLY generate if confidence is HIGH — otherwise mark 'Medium' or 'Low' to be filtered\n\n"

    "This is for academic research — not investment advice. No portfolio sizing or leverage.\n"
    "Respond ONLY with a valid JSON object — no markdown fences, no extra text."
)

_AGENT = "trade_structurer"

# ── Pydantic output schema ────────────────────────────────────────────────────

class TradeOutput(BaseModel):
    trade_name:            str = Field(description="Short name, e.g. 'Long XOM / Short JETS'")
    tickers:               str = Field(description="Specific tickers, e.g. 'Long XOM (ExxonMobil), CVX / Short JETS (airline ETF)'")
    direction:             str = Field(description="Direction with tickers, e.g. 'Long XOM, CVX / Short JETS'")
    trigger:               str = Field(description="Specific entry condition referencing the signal data provided")
    entry_price_ref:       str = Field(description="Entry price reference, e.g. 'XOM ~$112 on breakout above 50d MA; GLD ~$215'")
    upside_pct:            str = Field(description="% upside target with price level, e.g. '+12-18% → XOM $128-132'")
    stop_loss:             str = Field(description="Stop-loss as % from entry, e.g. 'Stop XOM below $107 (−4.5%)'")
    options_structure:     str = Field(description="Options alternative if applicable, e.g. 'Buy XOM $115C 45DTE / Sell $125C — debit ~$3.20; or GLD $210P'")
    rationale:             str = Field(description="2-3 sentences grounded in the provided market data")
    evidence:              str = Field(description="Specific data points from the context supporting this idea")
    confidence_level:      str = Field(description="Must be 'High' — only generate when high confidence. Include one-line reason.")
    key_uncertainty:       str = Field(description="What the available data cannot resolve")
    invalidation_condition: str = Field(description="Market condition that would invalidate this view")
    alternative_view:      str = Field(description="One plausible alternative interpretation of the same data")


_SCHEMA_HINT = """{
  "trade_name":             "Long XOM / Short JETS",
  "tickers":                "Long XOM (ExxonMobil), CVX (Chevron) / Short JETS (US Global Jets ETF)",
  "direction":              "Long XOM, CVX / Short JETS",
  "trigger":                "Hormuz disruption score >0.6 + WTI 5d return >+3%; enter on next open",
  "entry_price_ref":        "XOM ~$112, CVX ~$152, JETS ~$18 (reference at signal date)",
  "upside_pct":             "+12-18% → XOM $126-132, CVX $170-180 over 3-6 weeks",
  "stop_loss":              "XOM closes below $107 (−4.5%); JETS rallies above $20 (+11%)",
  "options_structure":      "Alt: Buy XOM $115C (45 DTE) / Sell $125C — debit ~$3.20; or USO $80C",
  "rationale":              "2-3 sentences from provided data",
  "evidence":               "specific data points supporting this idea",
  "confidence_level":       "High — [one-line reason]",
  "key_uncertainty":        "what the data cannot resolve",
  "invalidation_condition": "market condition that contradicts this view",
  "alternative_view":       "one plausible alternative interpretation"
}"""


@st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
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
        f"QUANTITATIVE SIGNAL CONTEXT:\n{context_str}\n\n"
        "Using the Diebold-Yilmaz spillover signals, regime data, and LIVE STOCK PRICES above, "
        "generate ONE high-conviction individual stock trade idea. Rules:\n"
        "• Pick 1-2 specific stocks FROM THE PRICE LIST above (e.g. XOM, CVX, LMT, NEM, DAL)\n"
        "• entry_price_ref = their exact live price from the list (e.g. 'Buy XOM at $113.45')\n"
        "• upside_pct = concrete target price in dollars AND % (e.g. '$129.00 (+13.7%)')\n"
        "• stop_loss = concrete stop price in dollars AND % (e.g. '$107.50 (−5.2%)')\n"
        "• options_structure = a specific options play on the same stock\n"
        "• evidence MUST cite the live price AND ≥2 model metrics (CIS, TPS, regime, etc.)\n"
        "• If the regime/signal data does NOT clearly support a specific stock play, set "
        "confidence_level = 'Medium — [reason]' or 'Low — [reason]' and the system discards it\n"
        "• Academic research only — no portfolio sizing, no leverage\n"
        f"Respond with a JSON object matching this structure:\n{_SCHEMA_HINT}\n"
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
                    max_tokens=1100,
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
                    max_tokens=1100, temperature=0.3,
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

    # Individual stock prices block (appended last so it's prominent in the prompt)
    stock_block = context.get("stock_prices_text", "")

    if not parts:
        set_status(_AGENT, "idle")
        return {}

    ctx_str = "\n".join(parts)
    if stock_block:
        ctx_str = ctx_str + "\n\n" + stock_block

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

    conf = calibrate_confidence(context_confidence(_AGENT, context), _AGENT)

    if conf < 0.45:
        set_status(_AGENT, "monitoring")
        log_activity(_AGENT, "confidence below threshold — skipped", f"conf={conf:.2f}", "info")
        return {"status": "monitoring", "reason": f"confidence too low ({conf:.2f})"}

    trade_dict = _call_ai(ctx_str, provider, api_key)

    if "error" in trade_dict:
        set_status(_AGENT, "idle")
        log_activity(_AGENT, "structured output failed", trade_dict["error"][:120], "warning")
        return {"error": trade_dict["error"]}

    trade_name = trade_dict.get("trade_name", "Structured Trade Idea")
    direction  = trade_dict.get("direction", "")
    rationale  = trade_dict.get("rationale", "")
    conf_level = trade_dict.get("confidence_level", "")

    # Reconstruct human-readable narrative from validated fields
    narrative = "\n".join([
        f"TRADE: {trade_name}",
        f"TICKERS: {trade_dict.get('tickers', '')}",
        f"DIRECTION: {direction}",
        f"TRIGGER: {trade_dict.get('trigger', '')}",
        f"ENTRY: {trade_dict.get('entry_price_ref', '')}",
        f"UPSIDE: {trade_dict.get('upside_pct', '')}",
        f"STOP: {trade_dict.get('stop_loss', '')}",
        f"OPTIONS: {trade_dict.get('options_structure', '')}",
        f"RATIONALE: {rationale}",
        f"EVIDENCE: {trade_dict.get('evidence', '')}",
        f"CONFIDENCE: {conf_level}",
        f"KEY UNCERTAINTY: {trade_dict.get('key_uncertainty', '')}",
        f"INVALIDATED IF: {trade_dict.get('invalidation_condition', '')}",
        f"ALT VIEW: {trade_dict.get('alternative_view', '')}",
    ])
    summary = f"{direction} — Entry: {trade_dict.get('entry_price_ref', '')} | Upside: {trade_dict.get('upside_pct', '')}"

    # Only route to approval queue if HIGH confidence
    if not conf_level.lower().startswith("high"):
        set_status(_AGENT, "monitoring")
        log_activity(_AGENT, "idea filtered — not high confidence", conf_level[:80], "info")
        return {
            "narrative":  narrative,
            "trade":      trade_dict,
            "confidence": conf,
            "filtered":   True,
            "reason":     f"confidence not high: {conf_level[:80]}",
        }

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

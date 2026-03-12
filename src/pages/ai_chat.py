"""
Page 9 — AI Analyst
OpenAI GPT-4o chatbot with live dashboard context injected into the system prompt.
Institutional terminal aesthetic — no emojis, clean layout, sticky chat bar.
"""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
import streamlit as st

from src.data.loader import load_returns
from src.data.config import GEOPOLITICAL_EVENTS
from src.analysis.correlations import average_cross_corr_series, detect_correlation_regime
from src.analysis.risk_score import compute_risk_score


# ── Context builder ────────────────────────────────────────────────────────

def _build_market_context(start: str, end: str) -> str:
    """Pull live market data and return a structured context block for the LLM."""
    try:
        eq_r, cmd_r = load_returns(start, end)
        if eq_r.empty or cmd_r.empty:
            return "Market data unavailable at this time."
    except Exception as e:
        return f"Market data unavailable: {e}"

    avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
    regimes  = detect_correlation_regime(avg_corr)
    current_regime   = int(regimes.iloc[-1]) if not regimes.empty else 1
    regime_labels    = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
    regime_name      = regime_labels[current_regime]
    current_avg_corr = float(avg_corr.iloc[-1]) if not avg_corr.empty else 0.0
    prev_avg_corr    = float(avg_corr.iloc[-21]) if len(avg_corr) > 21 else 0.0

    try:
        risk_result = compute_risk_score(avg_corr, cmd_r)
        risk_score  = risk_result["score"]
        risk_label  = risk_result["label"]
        risk_comps  = risk_result["components"]
    except Exception:
        risk_score, risk_label, risk_comps = 0, "Unknown", {}

    n = min(22, len(eq_r))
    recent_eq  = eq_r.iloc[-n:].sum()
    recent_cmd = cmd_r.iloc[-n:].sum()

    eq_top  = ", ".join(f"{k} ({v*100:+.1f}%)" for k, v in recent_eq.nlargest(3).items())
    eq_bot  = ", ".join(f"{k} ({v*100:+.1f}%)" for k, v in recent_eq.nsmallest(3).items())
    cmd_top = ", ".join(f"{k} ({v*100:+.1f}%)" for k, v in recent_cmd.nlargest(3).items())
    cmd_bot = ", ".join(f"{k} ({v*100:+.1f}%)" for k, v in recent_cmd.nsmallest(3).items())

    cmd_vol  = cmd_r.iloc[-n:].std() * np.sqrt(252) * 100
    vol_str  = ", ".join(f"{k} ({v:.0f}% ann.)" for k, v in cmd_vol.nlargest(3).items()) if not cmd_vol.empty else "N/A"

    today  = datetime.date.today()
    active = [e for e in GEOPOLITICAL_EVENTS if e["end"] >= today]
    active_str = "\n".join(
        f"  [{e['category']}] {e['name']} (since {e['start']}): {e['description']}"
        for e in active
    ) or "  None."

    comp_str = "\n".join(f"  {k}: {v:.0f}/100" for k, v in risk_comps.items()) if risk_comps else "  N/A"

    return (
        f"DATE: {today.strftime('%d %B %Y')}\n"
        f"WINDOW: {start} to {end}\n\n"
        f"CORRELATION REGIME\n"
        f"  Current: {regime_name} (level {current_regime}/3)\n"
        f"  60d avg |corr|: {current_avg_corr:.3f}  (vs 1M ago: {current_avg_corr - prev_avg_corr:+.3f})\n\n"
        f"RISK SCORE\n"
        f"  Composite: {risk_score:.0f}/100  ({risk_label})\n"
        f"{comp_str}\n\n"
        f"1-MONTH PERFORMANCE\n"
        f"  Equity leaders:    {eq_top}\n"
        f"  Equity laggards:   {eq_bot}\n"
        f"  Commodity leaders: {cmd_top}\n"
        f"  Commodity laggards:{cmd_bot}\n\n"
        f"HIGH-VOL COMMODITIES (1M ANN.)\n"
        f"  {vol_str}\n\n"
        f"ACTIVE GEOPOLITICAL EVENTS\n"
        f"{active_str}\n\n"
        f"UNIVERSE\n"
        f"  Equities ({len(eq_r.columns)}): {', '.join(eq_r.columns.tolist())}\n"
        f"  Commodities ({len(cmd_r.columns)}): {', '.join(cmd_r.columns.tolist())}"
    )


# ── System prompt ──────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are a senior quantitative cross-asset analyst at a macro hedge fund, \
embedded in the Equity-Commodities Spillover Monitor built by Purdue University \
Daniels School of Business. You have full access to live dashboard data and \
answer questions with institutional precision.

DASHBOARD STRUCTURE
  01 Overview              KPIs, regime status, risk gauge, correlation heatmap
  02 Geopolitical Triggers  Event timeline, pre/during/post returns, vol and corr shifts
  03 Correlation Analysis   Rolling correlations, DCC-GARCH, regime detection, pair explorer
  04 Spillover Analytics    Granger causality, transfer entropy, Diebold-Yilmaz FEVD, network graph
  05 Commodities to Watch   Live snapshot, intraday hourly charts, CFTC COT positioning
  06 Trade Ideas            Regime-triggered cross-asset trade cards (entry/exit/risk)
  07 Portfolio Stress Test  Custom allocation tested against every historical event window
  08 Performance Review     Signal backtests: regime detection F1, Granger hit rate, VIX R2, COT accuracy
  09 AI Analyst             This interface

ANALYTICAL METHODS
  Correlation Regime     60d rolling avg |correlation|, 4-level K-means classification
  DCC-GARCH              Dynamic Conditional Correlation for time-varying pair correlations
  Granger Causality      F-test: does cause series predict effect beyond its own lags?
  Transfer Entropy       Shannon entropy-based directional information flow
  Diebold-Yilmaz FEVD    VAR forecast error variance decomposition; net transmitter = outflows - inflows
  COT Contrarian         Net speculative % OI extremes (>+25% crowded long, <-25% crowded short)
  Geopolitical Risk Score 40% corr percentile + 30% commodity vol z-score + 20% VIX + 10% active events

LIVE MARKET CONTEXT
{context}

RESPONSE RULES
- Be precise and quantitative. Reference live numbers above when answering.
- Use markdown formatting: bold for key terms, bullet points for lists, backticks for tickers.
- Maximum 6 sentences for simple questions; structured bullets for complex analysis.
- Do not hedge or add disclaimers.
- If a question is outside dashboard scope, say so briefly and redirect to relevant data.
- When explaining methodology, be accurate to the implementation described above.
"""

# ── Suggested questions ────────────────────────────────────────────────────

_SUGGESTED = [
    "What is the current correlation regime and what does it imply?",
    "Which commodities are transmitting the most risk to equities?",
    "How does WTI crude oil affect the Nikkei 225 during supply shocks?",
    "Walk me through the Diebold-Yilmaz spillover index",
    "What trade does the current regime and risk score suggest?",
    "Which geopolitical event caused the largest correlation shift?",
    "Is speculative positioning crowded in any commodity right now?",
    "How do I interpret the Granger causality grid?",
]


# ── Page ───────────────────────────────────────────────────────────────────

def page_ai_chat(start: str, end: str) -> None:

    # ── Inline CSS: style native chat_message components ────────────────────
    st.markdown("""
    <style>
    /* Assistant bubble text */
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        font-size: 0.72rem !important;
        line-height: 1.72 !important;
        color: #111111 !important;
    }
    /* Chat input textarea */
    [data-testid="stChatInput"] textarea {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.74rem !important;
        border: 1px solid #E8E5E0 !important;
        border-radius: 3px !important;
        color: #111111 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #CFB991 !important;
        box-shadow: 0 0 0 1px #CFB991 !important;
    }
    /* Submit button — hide the broken Material Icons text, show clean arrow */
    [data-testid="stChatInputSubmitButton"] button {
        background: #000 !important;
        font-size: 0 !important;
        border-radius: 3px !important;
        width: 36px !important;
        height: 36px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stChatInputSubmitButton"] button::after {
        content: '→';
        font-size: 16px !important;
        font-family: 'DM Sans', sans-serif !important;
        color: #CFB991 !important;
    }
    [data-testid="stChatInputSubmitButton"] button * {
        font-size: 0 !important;
        display: none !important;
    }
    [data-testid="stChatInputSubmitButton"] button svg {
        display: none !important;
    }
    /* Avatar — clean minimal style */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        background: #f0ede8 !important;
        border: 1px solid #E8E5E0 !important;
        font-size: 0.58rem !important;
        color: #555960 !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.2rem">AI Analyst</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.70rem;color:#333333;margin:0 0 0.3rem;line-height:1.65">'
        'Cross-asset quantitative analyst with full access to live dashboard data. '
        'Ask about correlations, spillover flows, geopolitical impacts, methodology, or trade ideas.'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── API key guard ─────────────────────────────────────────────────────────
    openai_key = ""
    try:
        openai_key = st.secrets.get("keys", {}).get("openai_api_key", "") or ""
    except Exception:
        pass

    if not openai_key:
        st.markdown("---")
        st.warning("Add `openai_api_key` to `.streamlit/secrets.toml` to activate the AI Analyst.")
        st.code('[keys]\nopenai_api_key = "sk-..."', language="toml")
        return

    # ── Session state ─────────────────────────────────────────────────────────
    if "chat_messages"   not in st.session_state:
        st.session_state["chat_messages"]   = []
    if "chat_context"    not in st.session_state:
        st.session_state["chat_context"]    = None
    if "chat_context_ts" not in st.session_state:
        st.session_state["chat_context_ts"] = None

    # ── Context status + refresh ──────────────────────────────────────────────
    st.markdown("---")
    status_col, btn_col = st.columns([6, 1])

    if st.session_state["chat_context_ts"]:
        delta = datetime.datetime.now() - st.session_state["chat_context_ts"]
        age   = f"{int(delta.total_seconds() // 60)}m ago"
    else:
        age = "not loaded"

    status_col.markdown(
        f'<p style="font-size:0.62rem;color:#333333;margin:0;padding-top:0.45rem">'
        f'Market context: <span style="color:#CFB991;font-family:\'JetBrains Mono\',monospace">'
        f'{age}</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;Regime, risk score, movers, and active events injected into every query.'
        f'</p>',
        unsafe_allow_html=True,
    )

    do_load = (
        st.session_state["chat_context"] is None
        or btn_col.button("Refresh", key="refresh_ctx", use_container_width=True)
    )

    if do_load:
        with st.spinner("Loading live market context…"):
            ctx = _build_market_context(start, end)
            st.session_state["chat_context"]    = ctx
            st.session_state["chat_context_ts"] = datetime.datetime.now()

    context = st.session_state["chat_context"] or ""

    # ── Suggested questions (first visit only) ────────────────────────────────
    if not st.session_state["chat_messages"]:
        st.markdown(
            '<p style="font-size:0.58rem;font-weight:700;letter-spacing:0.14em;'
            'text-transform:uppercase;color:#333333;margin:0.8rem 0 0.4rem">Suggested questions</p>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, q in enumerate(_SUGGESTED):
            if cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state["chat_messages"].append({"role": "user", "content": q})
                st.rerun()

    st.markdown(
        '<div style="border-top:1px solid #E8E5E0;margin:0.8rem 0 0"></div>',
        unsafe_allow_html=True,
    )

    # ── CHAT INPUT (placed before messages to anchor sticky bar properly) ─────
    user_input = st.chat_input("Ask the analyst…")

    # ── Message history ───────────────────────────────────────────────────────
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(
                    f'<div style="border-left:3px solid #CFB991;padding-left:0.8rem">'
                    f'<span style="font-size:0.52rem;font-weight:700;letter-spacing:0.14em;'
                    f'text-transform:uppercase;color:#CFB991;display:block;margin-bottom:0.35rem">'
                    f'AI Analyst</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])

    # ── Handle new input ──────────────────────────────────────────────────────
    if user_input:
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        system_prompt = _SYSTEM_TEMPLATE.format(context=context)
        api_messages  = [{"role": "system", "content": system_prompt}]
        for m in st.session_state["chat_messages"][-20:]:
            api_messages.append({"role": m["role"], "content": m["content"]})

        with st.chat_message("assistant"):
            st.markdown(
                '<div style="border-left:3px solid #CFB991;padding-left:0.8rem">'
                '<span style="font-size:0.52rem;font-weight:700;letter-spacing:0.14em;'
                'text-transform:uppercase;color:#CFB991;display:block;margin-bottom:0.35rem">'
                'AI Analyst</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            try:
                from openai import OpenAI as _OpenAI
                client   = _OpenAI(api_key=openai_key)
                stream   = client.chat.completions.create(
                    model="gpt-4o",
                    messages=api_messages,
                    max_tokens=600,
                    temperature=0.25,
                    stream=True,
                )
                response = st.write_stream(
                    chunk.choices[0].delta.content or ""
                    for chunk in stream
                    if chunk.choices[0].delta.content
                )
            except Exception as e:
                response = f"Request failed: {e}"
                st.markdown(response)

        st.session_state["chat_messages"].append(
            {"role": "assistant", "content": response}
        )

    # ── Utility bar ───────────────────────────────────────────────────────────
    if st.session_state["chat_messages"]:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 5])
        if c1.button("Clear conversation", key="clear_chat"):
            st.session_state["chat_messages"] = []
            st.rerun()

    # ── Context inspector ─────────────────────────────────────────────────────
    with st.expander("Injected market context", expanded=False):
        st.code(context, language="text")

    st.markdown(
        '<p style="font-size:0.58rem;color:#444444;margin-top:1rem;line-height:1.6">'
        'Powered by GPT-4o · Context refreshed on demand from live yfinance data · '
        'For educational and research purposes only. Not investment advice.</p>',
        unsafe_allow_html=True,
    )

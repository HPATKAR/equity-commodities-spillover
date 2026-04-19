"""
Page 9 - AI Analyst
OpenAI GPT-4o chatbot with live dashboard context injected into the system prompt.
Institutional terminal aesthetic - no emojis, clean layout, sticky chat bar.
"""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
import streamlit as st
from src.ui.shared import _page_header, _page_footer

from src.data.loader import load_returns, load_fixed_income_returns, load_fx_returns
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

    # FI context
    try:
        fi_r_ctx = load_fixed_income_returns(start, end)
        if not fi_r_ctx.empty:
            n_fi = min(22, len(fi_r_ctx))
            fi_perf = fi_r_ctx.iloc[-n_fi:].sum() * 100
            fi_top = ", ".join(f"{k} ({v:+.1f}%)" for k, v in fi_perf.nlargest(3).items())
            fi_bot = ", ".join(f"{k} ({v:+.1f}%)" for k, v in fi_perf.nsmallest(3).items())
        else:
            fi_top = fi_bot = "N/A"
    except Exception:
        fi_top = fi_bot = "N/A"

    # FX context
    try:
        fx_r_ctx = load_fx_returns(start, end)
        if not fx_r_ctx.empty:
            n_fx = min(22, len(fx_r_ctx))
            fx_perf = fx_r_ctx.iloc[-n_fx:].sum() * 100
            dxy_30d = float(fx_r_ctx["DXY (Dollar Index)"].iloc[-22:].sum() * 100) if "DXY (Dollar Index)" in fx_r_ctx.columns else None
        else:
            dxy_30d = None
            fx_perf = None
    except Exception:
        dxy_30d = None
        fx_perf = None

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
        f"1-MONTH FIXED INCOME PERFORMANCE\n"
        f"  Leaders: {fi_top}\n"
        f"  Laggards: {fi_bot}\n\n"
        f"FX / DOLLAR\n"
        f"  DXY 1M: {f'{dxy_30d:+.1f}%' if dxy_30d is not None else 'N/A'}\n\n"
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
embedded in the Cross-Asset Spillover Monitor built by Purdue University \
Daniels School of Business. You have full access to live dashboard data and \
answer questions with institutional precision.

DASHBOARD STRUCTURE
  01 Overview              KPIs (equity, commodity, fixed income, FX), regime status, risk gauge, heatmap
  02 Geopolitical Triggers  Event timeline, pre/during/post returns, vol and corr shifts
  03 Correlation Analysis   Rolling correlations, DCC-GARCH, regime detection, pair explorer (FI/FX included)
  04 Spillover Analytics    Granger causality, transfer entropy, Diebold-Yilmaz FEVD, cross-asset rates/FX section
  05 Commodities to Watch   Live snapshot, intraday hourly charts, CFTC COT positioning
  06 Trade Ideas            Regime-triggered cross-asset trade cards including FI (TLT/HYG, TIP/TLT, EMB/DXY)
  07 Portfolio Stress Test  Custom allocation (equities, commodities, fixed income) tested against all events
  08 Performance Review     Signal backtests: regime detection F1, Granger hit rate, VIX R2, COT accuracy
  09 AI Analyst             This interface - also has private credit bubble risk and FI/FX context

ANALYTICAL METHODS
  Correlation Regime     60d rolling avg |correlation|, 4-level K-means classification
  DCC-GARCH              Dynamic Conditional Correlation for time-varying pair correlations
  Granger Causality      F-test: does cause series predict effect beyond its own lags?
  Transfer Entropy       Shannon entropy-based directional information flow
  Diebold-Yilmaz FEVD    VAR forecast error variance decomposition; net transmitter = outflows - inflows
  COT Contrarian         Net speculative % OI extremes (>+25% crowded long, <-25% crowded short)
  Geopolitical Risk Score 40% corr percentile + 30% commodity vol z-score + 20% VIX + 10% active events
  FX-Commodity Link      Dollar (DXY) strength compresses dollar-denominated commodity prices; DXY -5% = Gold +3-5%, metals +2-5%
  Equity-Bond Regime     Negative TLT/SPX correlation = flight-to-quality hedge working; positive = 60/40 breakdown (crisis)
  Duration Signal        TLT vs SHY 60d alpha: negative = bear steepening (rates higher for longer); positive = bull flattening
  EM Credit Channel      EMB prices inverse to DXY; EM sovereign stress appears in spreads 1-2Q before GDP data confirms
  Breakeven Inflation    TIP vs TLT spread = market's inflation expectation; rising = commodity price pressures not transitory

PRIVATE CREDIT RISK MONITOR
  Structural thesis  Private credit AUM ~$2T+ (2024); primarily floating-rate direct lending at 5–7× leverage.
                     SOFR at 4.3%+ → all-in borrower costs 9–12%. Interest coverage ratios <1.5× at many
                     portfolio companies. Quarterly mark-to-model obscures stress until it cannot be hidden.
  Observable proxies HY OAS (BAMLH0A0HYM2): spread percentile + 90d trend; BKLN ETF: leveraged loan proxy;
                     BDC basket (ARCC/OBDC/FSK): NAV/price stress; CDX HY: liquid short vehicle
  Bubble signals     HY OAS at historical low percentile (<30th); BKLN/BDC outperforming SPY; issuance boom
  Burst signals      HY OAS widening >35bps/90d; BKLN underperforming SPY >5%; BDC discount-to-book widening;
                     SLOS (DRTSCILM) tightening >30%; redemption pressure in interval/NAV funds
  Bust mechanism     Redemption pressure → forced sales → price discovery → contagion to public HY →
                     financial sector equity de-rating → commodity demand destruction (copper, oil)
  Spillover path     Private credit bust → leveraged buyout sector stress → industrial/manufacturing demand
                     destruction → copper/oil sell-off → EM equity pressure (commodity exporters)
  Current view       Dashboard owner believes private credit bubble is approaching burst. Acknowledge this
                     thesis when asked, and reference the PC Bubble Risk Score on the Insights page.

INDIA / RUPEE MACRO FRAMEWORK
  Macro role         India is the world's 5th-largest economy and a major commodity importer. Unlike China,
                     India does not produce significant oil or gold - it imports both at scale.
  Crude oil          India is the #3 crude oil importer globally (~5 mb/d; ~85% import dependency).
                     Every $10/bbl rise in Brent costs India ~$15B/year in additional import expenditure.
                     Oil-INR transmission: Brent +20% historically correlates with USD/INR +3–5%.
  Gold               India is the #2 gold consumer globally (~800-900 tonnes/year).
                     Gold demand surges during geopolitical stress, festival seasons, and INR depreciation.
                     RBI holds ~800 tonnes in reserves; gold's local INR price is Brent's USD price × USD/INR.
  Currency           USD/INR (USDINR=X in yfinance). INR weakens during: (1) oil spikes, (2) global risk-off,
                     (3) capital outflows, (4) US dollar strengthening. RBI defends INR using ~$620B forex reserves.
                     RBI repo rate: ~6.50% (as of 2024). Inflation target: 4% ±2%.
  Equity (Nifty 50)  ^NSEI in yfinance. Key drivers: domestic consumption, IT exports (TCS, Infosys),
                     crude oil price (net negative), USD/INR (net negative for importers),
                     FII (foreign institutional investor) flows, and RBI monetary policy.
                     Nifty 50 correlation with S&P 500: ~0.65-0.75 in normal regimes, rising to >0.85 in crisis.
  Spillover paths    WTI/Brent spike → INR depreciation → Nifty Energy sector benefit, Nifty Consumer stress
                     Dollar strength (DXY up) → USD/INR rises → India CAD widens → FII outflows → Nifty down
                     Gold spike + INR weak → India gold import cost surge → CA deficit pressure
                     India-Pakistan geopolitical tensions → Nifty volatility spike, rupee depreciation
  Key tickers        ^NSEI (Nifty 50), ^BSESN (Sensex), USDINR=X (USD/INR), INFY/TCS.NS (IT bellwethers)
  Trade ideas        Long Gold / Short INR on geopolitical stress (India's gold safe-haven demand + INR weakness)
                     Long Brent / Short Nifty on oil supply shock (India's 85% crude import dependency)
                     Long Nifty / Short USDINR on dollar weakness cycle (EM relief + India tech export boost)

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
    "How does a Brent oil spike affect the Indian rupee and Nifty 50?",
    "What is the India private credit / RBI rate outlook right now?",
]


# ── Page ───────────────────────────────────────────────────────────────────

def page_ai_chat(start: str, end: str) -> None:

    # ── Inline CSS: style native chat_message components ────────────────────
    st.markdown("""
    <style>
    /* Chat message bubbles */
    div[data-testid="stChatMessage"] {
        background: #1c1c1c !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 0 !important;
        margin-bottom: 0.5rem !important;
    }
    /* Assistant/user bubble text */
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        font-size: 0.78rem !important;
        line-height: 1.72 !important;
        color: #e8e9ed !important;
    }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span {
        color: #e8e9ed !important;
    }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
        color: #CFB991 !important;
    }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
        background: #111111 !important;
        color: #CFB991 !important;
        padding: 1px 4px !important;
        border-radius: 0 !important;
    }
    /* Chat input textarea */
    textarea[data-testid="stTextArea"],
    [data-testid="stChatInput"] textarea {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.74rem !important;
        background: #1c1c1c !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 0 !important;
        color: #e8e9ed !important;
    }
    textarea[data-testid="stTextArea"]:focus,
    [data-testid="stChatInput"] textarea:focus {
        border-color: #CFB991 !important;
        box-shadow: 0 0 0 1px #CFB991 !important;
    }
    /* Submit button */
    [data-testid="stChatInputSubmitButton"] button {
        background: #1c1c1c !important;
        border: 1px solid #CFB991 !important;
        font-size: 0 !important;
        border-radius: 0 !important;
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
    [data-testid="stChatInputSubmitButton"] button * { font-size: 0 !important; display: none !important; }
    [data-testid="stChatInputSubmitButton"] button svg { display: none !important; }
    /* Avatar */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        background: #111111 !important;
        border: 1px solid #2a2a2a !important;
        font-size: 0.58rem !important;
        color: #CFB991 !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────────────────────
    _page_header("AI Research Desk",
                 "Natural language · Cross-asset research · Regime-aware context")
    st.markdown(
        '<p style="font-size:0.70rem;color:#8890a1;margin:0 0 0.3rem;line-height:1.65">'
        'An analyst trained on the full equity-commodities spillover framework with live access to '
        'all dashboard data. Ask it to interpret the current regime, explain why a specific commodity '
        'is moving relative to equities, identify which historical event the current environment most '
        'resembles, or challenge any trade idea against the spillover evidence. It knows the methodology.'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── API keys (Anthropic preferred, OpenAI fallback) ───────────────────────
    anthropic_key = openai_key = ""
    try:
        _keys = st.secrets.get("keys", {})
        anthropic_key = _keys.get("anthropic_api_key", "") or ""
        openai_key    = _keys.get("openai_api_key", "")    or ""
    except Exception:
        pass

    _has_key  = bool(anthropic_key or openai_key)
    _provider = "anthropic" if anthropic_key else ("openai" if openai_key else None)

    # ── Session state ─────────────────────────────────────────────────────────
    if "chat_messages"   not in st.session_state:
        st.session_state["chat_messages"]   = []
    if "chat_context"    not in st.session_state:
        st.session_state["chat_context"]    = None
    if "chat_context_ts" not in st.session_state:
        st.session_state["chat_context_ts"] = None

    # ── Auto-load context on first visit ──────────────────────────────────────
    status_col, btn_col = st.columns([6, 1])
    _refresh = btn_col.button("Refresh context", key="refresh_ctx", use_container_width=True)

    if st.session_state["chat_context"] is None or _refresh:
        with st.spinner("Loading live market context…"):
            ctx = _build_market_context(start, end)
            st.session_state["chat_context"]    = ctx
            st.session_state["chat_context_ts"] = datetime.datetime.now()

    # ── Status bar rendered AFTER load so age is always current ───────────────
    if st.session_state["chat_context_ts"]:
        _delta = datetime.datetime.now() - st.session_state["chat_context_ts"]
        _age   = f"{int(_delta.total_seconds() // 60)}m ago" if _delta.total_seconds() >= 60 else "just loaded"
    else:
        _age = "not loaded"

    _ctx_ok  = st.session_state["chat_context"] and not st.session_state["chat_context"].startswith("Market data unavailable")
    _age_col = "#CFB991" if _ctx_ok else "#c0392b"
    status_col.markdown(
        f'<p style="font-size:0.62rem;color:#8890a1;margin:0;padding-top:0.45rem">'
        f'Market context:&nbsp;<span style="color:{_age_col};font-family:\'JetBrains Mono\',monospace;font-weight:600">'
        f'{_age}</span>'
        f'&nbsp;&nbsp;&middot;&nbsp;&nbsp;Regime, risk score, movers, and active events injected into every query.'
        f'</p>',
        unsafe_allow_html=True,
    )

    # ── No API key warning (non-blocking - chat input still shown below) ──────
    if not _has_key:
        st.warning(
            "No API key configured. Add one of the following to `.streamlit/secrets.toml` to enable responses:\n\n"
            "```toml\n[keys]\nanthropics_api_key = \"sk-ant-...\"\n# or\nopenai_api_key = \"sk-...\"\n```"
        )

    context = st.session_state["chat_context"] or ""

    # ── Suggested questions (first visit only) ────────────────────────────────
    if not st.session_state["chat_messages"]:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:7.5px;font-weight:700;'
            'letter-spacing:.16em;text-transform:uppercase;color:#CFB991;'
            'border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin:.8rem 0 .4rem">Suggested Questions</p>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, q in enumerate(_SUGGESTED):
            if cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state["_sq_input"] = q

    st.markdown(
        '<div style="border-top:1px solid #2a2a2a;margin:0.8rem 0 0"></div>',
        unsafe_allow_html=True,
    )

    # ── QUERY INPUT ───────────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:0.52rem;font-weight:700;letter-spacing:0.14em;'
        'text-transform:uppercase;color:#8E9AAA;margin:0.6rem 0 0.25rem">Query Input</p>',
        unsafe_allow_html=True,
    )
    _inp_col, _btn_col = st.columns([5, 1])
    _typed = _inp_col.text_area(
        "Ask the analyst",
        placeholder="e.g. How is the current regime affecting oil–equity correlations?",
        height=80,
        label_visibility="collapsed",
        key="chat_textarea",
    )
    _send = _btn_col.button("Send", type="primary", key="send_chat", use_container_width=True)
    _sq   = st.session_state.pop("_sq_input", None)
    user_input = _sq or (_typed.strip() if (_send and _typed.strip()) else None)

    # ── Message history ───────────────────────────────────────────────────────
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(
                    f'<div style="padding-left:0.8rem">'
                    f'<span style="font-size:0.52rem;font-weight:700;letter-spacing:0.14em;'
                    f'text-transform:uppercase;color:#CFB991;display:block;margin-bottom:0.35rem">'
                    f'AI Analyst</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])

    # ── Handle new input ──────────────────────────────────────────────────────
    if user_input:
        if not _has_key:
            st.warning("Add an API key (see above) to receive AI responses.")
        else:
            st.session_state["chat_messages"].append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            system_prompt = _SYSTEM_TEMPLATE.format(context=context)
            api_messages  = [{"role": "system", "content": system_prompt}]
            for m in st.session_state["chat_messages"][-20:]:
                api_messages.append({"role": m["role"], "content": m["content"]})

            with st.chat_message("assistant"):
                st.markdown(
                    '<div style="padding-left:0.8rem">'
                    '<span style="font-size:0.52rem;font-weight:700;letter-spacing:0.14em;'
                    'text-transform:uppercase;color:#CFB991;display:block;margin-bottom:0.35rem">'
                    'AI Analyst</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                response = ""
                try:
                    if _provider == "anthropic":
                        import anthropic as _ant
                        _client = _ant.Anthropic(api_key=anthropic_key)
                        _user_msgs = [m for m in api_messages if m["role"] != "system"]
                        with _client.messages.stream(
                            model="claude-sonnet-4-6",
                            max_tokens=1024,
                            system=system_prompt,
                            messages=_user_msgs,
                        ) as _stream:
                            response = st.write_stream(_stream.text_stream)
                    else:
                        from openai import OpenAI as _OpenAI
                        _client = _OpenAI(api_key=openai_key)
                        _stream = _client.chat.completions.create(
                            model="gpt-4o",
                            messages=api_messages,
                            max_tokens=1024,
                            temperature=0.25,
                            stream=True,
                        )
                        response = st.write_stream(
                            c.choices[0].delta.content
                            for c in _stream
                            if c.choices[0].delta.content
                        )
                except Exception as e:
                    response = f"⚠ Request failed: {e}"
                    st.error(response)

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
    with st.expander("Injected market context", expanded=True):
        st.code(context, language="text")

    st.markdown(
        '<p style="font-size:0.58rem;color:#6b7280;margin-top:1rem;line-height:1.6">'
        'Powered by GPT-4o / Claude · Context refreshed on demand from live yfinance + FRED data · '
        'For educational and research purposes only. Not investment advice.</p>',
        unsafe_allow_html=True,
    )

    # ── Agent Activity Log viewer ─────────────────────────────────────────────
    try:
        from src.analysis.agent_state import init_agents as _ia_log
        _ia_log()
        _activity_log = st.session_state.get("agent_activity", [])
        if _activity_log:
            with st.expander(
                f"Agent Activity Log  ({len(_activity_log)} entries)", expanded=False
            ):
                _F_log = "font-family:'DM Sans',sans-serif;"
                _sev_colors = {
                    "critical": "#c0392b", "warning": "#e67e22",
                    "info": "#27ae60", "debug": "#555960",
                }
                _rows_log = ""
                for _entry in _activity_log[:50]:
                    _ts_s   = _entry.get("ts")
                    _ts_str = _ts_s.strftime("%H:%M:%S") if hasattr(_ts_s, "strftime") else str(_ts_s)[:8]
                    _sev    = _entry.get("severity", "info")
                    _col    = _sev_colors.get(_sev, "#555960")
                    _agent  = _entry.get("agent_name", _entry.get("agent_id", "-"))
                    _action = _entry.get("action", "")
                    _detail = _entry.get("detail", "")
                    _routed = _entry.get("routed_to")
                    _route_badge = (
                        f'<span style="font-size:0.52rem;color:#2980b9;margin-left:4px">→ {_routed}</span>'
                        if _routed else ""
                    )
                    _rows_log += (
                        f'<tr style="border-bottom:1px solid #1e1e1e;">'
                        f'<td style="padding:3px 8px 3px 0;font-family:\'JetBrains Mono\',monospace;'
                        f'font-size:0.55rem;color:#555960;white-space:nowrap">{_ts_str}</td>'
                        f'<td style="padding:3px 8px 3px 0;font-size:0.58rem;font-weight:600;'
                        f'color:{_col};white-space:nowrap">{_sev.upper()}</td>'
                        f'<td style="padding:3px 8px 3px 0;font-size:0.60rem;color:#CFB991;white-space:nowrap">'
                        f'{_agent}</td>'
                        f'<td style="padding:3px 8px 3px 0;font-size:0.62rem;color:#e8e9ed">'
                        f'{_action}{_route_badge}</td>'
                        f'<td style="padding:3px 0 3px 0;font-size:0.60rem;color:#8890a1;'
                        f'max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
                        f'{_detail[:80]}{"…" if len(_detail) > 80 else ""}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<div style="overflow-x:auto">'
                    f'<table style="width:100%;border-collapse:collapse;{_F_log}font-size:0.62rem">'
                    f'<thead><tr style="border-bottom:1px solid #2a2a2a">'
                    f'<th style="padding:3px 8px 3px 0;text-align:left;font-size:0.52rem;'
                    f'font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a">Time</th>'
                    f'<th style="padding:3px 8px 3px 0;text-align:left;font-size:0.52rem;'
                    f'font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a">Level</th>'
                    f'<th style="padding:3px 8px 3px 0;text-align:left;font-size:0.52rem;'
                    f'font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a">Agent</th>'
                    f'<th style="padding:3px 8px 3px 0;text-align:left;font-size:0.52rem;'
                    f'font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a">Action</th>'
                    f'<th style="padding:3px 0;text-align:left;font-size:0.52rem;'
                    f'font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a">Detail</th>'
                    f'</tr></thead>'
                    f'<tbody>{_rows_log}</tbody>'
                    f'</table></div>',
                    unsafe_allow_html=True,
                )
                if len(_activity_log) > 50:
                    st.markdown(
                        f'<p style="font-size:0.56rem;color:#555960;margin-top:4px">'
                        f'Showing 50 of {len(_activity_log)} entries (newest first).</p>',
                        unsafe_allow_html=True,
                    )
    except Exception:
        pass
    _page_footer()

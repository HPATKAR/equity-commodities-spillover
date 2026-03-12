"""
Page 1 — Overview
KPIs, cross-asset correlation heatmap snapshot, regime status, recent events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_returns, load_all_prices
from src.data.config import GEOPOLITICAL_EVENTS, EQUITY_REGIONS, COMMODITY_GROUPS, PALETTE
from src.analysis.correlations import (
    cross_asset_corr, average_cross_corr_series, detect_correlation_regime,
)
from src.analysis.risk_score import (
    compute_risk_score, risk_score_history, plot_risk_gauge, plot_risk_history,
)
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _section_note,
    _definition_block, _takeaway_block, _page_conclusion,
    _page_footer, _add_event_bands,
)


def page_overview(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.3rem">Equity-Commodities Spillover Monitor</h1>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "Cross-asset correlation and spillover dashboard tracking how geopolitical shocks, "
        "macro regimes, and commodity supply disruptions transmit into global equity markets. "
        "Covers 15 equity indices (USA, Europe, Japan, China, India) and 17 commodity futures "
        "across energy, metals, and agriculture."
    )

    # ── Load data ──────────────────────────────────────────────────────────
    with st.spinner("Loading market data…"):
        eq_r, cmd_r = load_returns(start, end)

    if eq_r.empty or cmd_r.empty:
        st.error("Could not load market data. Check your internet connection.")
        return

    # ── KPIs ───────────────────────────────────────────────────────────────
    st.markdown("---")

    # Current regime
    avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
    regimes  = detect_correlation_regime(avg_corr)
    current_regime = regimes.iloc[-1] if not regimes.empty else 1
    regime_labels  = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
    regime_colors  = {0: "#2e7d32",      1: "#555960", 2: "#e67e22",  3: "#c0392b"}
    regime_name    = regime_labels[current_regime]
    regime_color   = regime_colors[current_regime]

    current_avg_corr = avg_corr.iloc[-1] if not avg_corr.empty else 0.0
    prev_avg_corr    = avg_corr.iloc[-21] if len(avg_corr) > 21 else 0.0
    corr_delta       = current_avg_corr - prev_avg_corr

    # Recent best / worst equity
    recent_eq = eq_r.iloc[-22:].sum()
    best_eq   = recent_eq.idxmax()
    worst_eq  = recent_eq.idxmin()

    # Recent best / worst commodity
    recent_cmd = cmd_r.iloc[-22:].sum()
    best_cmd   = recent_cmd.idxmax()
    worst_cmd  = recent_cmd.idxmin()

    k1, k2, k3, k4, k5 = st.columns(5)

    def _kpi(col, label, value, delta="", dcolor=""):
        col.markdown(
            f"""<div style="border:1px solid #E8E5E0;border-radius:4px;
            padding:0.6rem 0.8rem;background:#fff">
            <div style="font-size:0.55rem;font-weight:600;letter-spacing:0.14em;
            text-transform:uppercase;color:#9D9795;margin-bottom:3px">{label}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.95rem;
            font-weight:700;color:#000">{value}</div>
            {"" if not delta else f'<div style="font-size:0.65rem;color:{dcolor}">{delta}</div>'}
            </div>""",
            unsafe_allow_html=True,
        )

    k1.markdown(
        f"""<div style="border:1px solid #E8E5E0;border-radius:4px;
        padding:0.6rem 0.8rem;background:#fff">
        <div style="font-size:0.55rem;font-weight:600;letter-spacing:0.14em;
        text-transform:uppercase;color:#9D9795;margin-bottom:3px">Correlation Regime</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.95rem;
        font-weight:700;color:{regime_color}">{regime_name}</div></div>""",
        unsafe_allow_html=True,
    )
    k2.metric("Avg Cross-Asset Corr (60d)",
              f"{current_avg_corr:.3f}",
              delta=f"{corr_delta:+.3f} vs 1M ago")
    k3.metric(f"Best Equity (1M)", best_eq,
              delta=f"{recent_eq[best_eq]*100:+.1f}%")
    k4.metric(f"Worst Equity (1M)", worst_eq,
              delta=f"{recent_eq[worst_eq]*100:+.1f}%")
    k5.metric(f"Best Commodity (1M)", best_cmd,
              delta=f"{recent_cmd[best_cmd]*100:+.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Geopolitical Risk Score ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Geopolitical Risk Score")
    _definition_block(
        "Composite Risk Index (0–100)",
        "Weighted blend of four real-time signals: "
        "<b>Cross-asset correlation percentile</b> (40%) — how elevated is current correlation vs history; "
        "<b>Commodity vol z-score</b> (30%) — energy + metals volatility vs 1-year mean; "
        "<b>VIX level</b> (20%) — equity fear gauge mapped to 0-100; "
        "<b>Active events</b> (10%) — count of ongoing geopolitical events. "
        "Bands: 0-25 Low · 25-50 Moderate · 50-75 Elevated · 75-100 High/Crisis.",
    )

    with st.spinner("Computing risk score…"):
        risk_result = compute_risk_score(avg_corr, cmd_r)
        score_hist  = risk_score_history(avg_corr, cmd_r)

    gc1, gc2 = st.columns([1, 2])
    with gc1:
        _chart(plot_risk_gauge(risk_result, height=260))
        # Component breakdown
        comp = risk_result["components"]
        for name, val in comp.items():
            col_c = "#c0392b" if val > 70 else "#e67e22" if val > 45 else "#2e7d32"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:3px 0;border-bottom:1px solid #F0EDEA;font-size:0.68rem">'
                f'<span style="color:#555960">{name}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-weight:700;'
                f'color:{col_c}">{val:.0f}</span></div>',
                unsafe_allow_html=True,
            )
    with gc2:
        if not score_hist.empty:
            _chart(plot_risk_history(score_hist, height=300))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── LLM Market Narrative ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("AI Market Narrative")
    _section_note(
        "Claude generates a 2-paragraph quantitative market commentary based on current "
        "regime, risk score, top movers, and active geopolitical events. "
        "Powered by the Anthropic API."
    )

    anthropic_key = ""
    try:
        anthropic_key = st.secrets.get("keys", {}).get("anthropic_api_key", "") or ""
    except Exception:
        pass

    if not anthropic_key:
        st.info(
            "Add your `anthropic_api_key` to `.streamlit/secrets.toml` to enable AI narratives.",
            icon="🤖",
        )
    else:
        if st.button("Generate Market Narrative", type="primary", key="gen_narrative"):
            active_events = [
                e["label"] for e in GEOPOLITICAL_EVENTS if e["end"] >= __import__("datetime").date.today()
            ]
            prompt = (
                f"You are a quantitative cross-asset analyst at a macro hedge fund. "
                f"Write a concise, professional 2-paragraph market narrative (150-200 words total) "
                f"based on the following live dashboard data:\n\n"
                f"Date: {__import__('datetime').date.today()}\n"
                f"Correlation Regime: {regime_name} (avg 60d corr: {current_avg_corr:.3f})\n"
                f"Geopolitical Risk Score: {risk_result['score']}/100 ({risk_result['label']})\n"
                f"Best equity (1M): {best_eq} ({recent_eq[best_eq]*100:+.1f}%)\n"
                f"Worst equity (1M): {worst_eq} ({recent_eq[worst_eq]*100:+.1f}%)\n"
                f"Best commodity (1M): {best_cmd} ({recent_cmd[best_cmd]*100:+.1f}%)\n"
                f"Worst commodity (1M): {worst_cmd} ({recent_cmd[worst_cmd]*100:+.1f}%)\n"
                f"Active geopolitical events: {', '.join(active_events) if active_events else 'None'}\n\n"
                f"Paragraph 1: Describe the current cross-asset correlation regime and what it implies "
                f"for equity-commodity spillover dynamics. Use specific numbers.\n"
                f"Paragraph 2: Identify the key risk factor driving the risk score and name 1-2 "
                f"specific trade implications. Be precise and analytical. No disclaimers."
            )
            try:
                import anthropic as _anthropic
                client = _anthropic.Anthropic(api_key=anthropic_key)
                with st.spinner("Generating narrative…"):
                    msg = client.messages.create(
                        model="claude-opus-4-6",
                        max_tokens=400,
                        messages=[{"role": "user", "content": prompt}],
                    )
                narrative = msg.content[0].text
                regime_col_narrative = regime_color
                st.markdown(
                    f"""<div style="border-left:4px solid {regime_col_narrative};
                    padding:1rem 1.2rem;background:#fafaf8;margin:0.8rem 0;
                    border-radius:0 4px 4px 0">
                    <div style="font-size:0.58rem;font-weight:700;letter-spacing:0.14em;
                    text-transform:uppercase;color:{regime_col_narrative};margin-bottom:0.5rem">
                    AI MARKET COMMENTARY · {__import__('datetime').datetime.now().strftime('%d %b %Y %H:%M')} UTC
                    </div>
                    <div style="font-size:0.78rem;color:#2A2A2A;line-height:1.75;
                    font-family:'DM Sans',sans-serif">{narrative.replace(chr(10), '<br>')}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Narrative generation failed: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Avg cross-asset correlation timeline ──────────────────────────────
    st.subheader("Average Equity-Commodity Correlation (60d Rolling)")
    _definition_block(
        "What Does This Measure?",
        "The mean of all pairwise |rolling 60-day correlations| between equity indices "
        "and commodity futures. Spikes indicate that commodities and equities are moving "
        "together — a hallmark of risk-off crises, supply shocks, and geopolitical stress.",
    )

    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(
        x=avg_corr.index, y=avg_corr.values,
        name="Avg |Corr|",
        line=dict(color=PALETTE[1], width=1.8),
        fill="tozeroy", fillcolor="rgba(207,185,145,0.12)",
    ))

    # Simple threshold lines
    fig_corr.add_hline(y=0.15, line=dict(color="#2e7d32", width=1, dash="dot"),
                       annotation_text="Low", annotation_font_size=9)
    fig_corr.add_hline(y=0.45, line=dict(color="#c0392b", width=1, dash="dot"),
                       annotation_text="High", annotation_font_size=9)

    _add_event_bands(fig_corr)
    _chart(_style_fig(fig_corr, height=380))

    _takeaway_block(
        f"Current average cross-asset correlation is <b>{current_avg_corr:.3f}</b> "
        f"({regime_name} regime). "
        "Correlation spikes during GFC (2008), COVID (2020), and Ukraine War (2022) "
        "reflect simultaneous risk-off selling across equities and commodities."
    )

    # ── Current correlation heatmap ────────────────────────────────────────
    st.subheader("Cross-Asset Correlation Heatmap (Full Sample)")

    window_opt = st.select_slider(
        "Sample window",
        options=[63, 126, 252, 504, 0],
        value=252,
        format_func=lambda x: "Full" if x == 0 else f"{x}d",
    )

    corr_mat = cross_asset_corr(eq_r, cmd_r, window=window_opt or None)
    if not corr_mat.empty:
        fig_heat = go.Figure(go.Heatmap(
            z=corr_mat.values,
            x=corr_mat.columns.tolist(),
            y=corr_mat.index.tolist(),
            colorscale=[
                [0.0,  "#2980b9"],
                [0.5,  "#ffffff"],
                [1.0,  "#c0392b"],
            ],
            zmid=0, zmin=-1, zmax=1,
            text=corr_mat.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=9, family="JetBrains Mono, monospace"),
            colorbar=dict(
                title="Corr", thickness=12, len=0.8,
                tickfont=dict(size=9, family="JetBrains Mono, monospace"),
            ),
            hoverongaps=False,
        ))
        fig_heat.update_layout(
            template="purdue",
            height=480,
            xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
            margin=dict(l=120, r=40, t=40, b=120),
        )
        _chart(fig_heat)

    _section_note(
        "🔴 Red = positive correlation (equities and commodities move together — risk-off or inflation). "
        "🔵 Blue = negative correlation (flight-to-safety or supply shock divergence). "
        "White = decorrelated."
    )

    # ── Active geopolitical events ─────────────────────────────────────────
    from datetime import date as _date
    today = _date.today()
    active = [e for e in GEOPOLITICAL_EVENTS if e["end"] >= today]
    if active:
        st.subheader("Active Geopolitical Risk Events")
        for ev in active:
            st.markdown(
                f"""<div style="border-left:3px solid {ev['color']};
                padding:0.5rem 0.8rem;margin:0.4rem 0;background:#fafaf8">
                <span style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.08em;color:{ev['color']}">{ev['category']} · {ev['label']}</span>
                <span style="font-size:0.72rem;color:#000;font-weight:600;
                margin-left:0.5rem">{ev['name']}</span>
                <p style="font-size:0.7rem;color:#555960;margin:0.2rem 0 0;
                line-height:1.55">{ev['description']}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    _page_conclusion(
        "Cross-Asset Spillover Dashboard",
        "This dashboard quantifies how geopolitical shocks, supply disruptions, and monetary "
        "policy shifts transmit across equity and commodity markets. Navigate the pages to "
        "explore event-driven correlation shifts, Granger causality flows, and trade ideas "
        "triggered by regime changes."
    )
    _page_footer()

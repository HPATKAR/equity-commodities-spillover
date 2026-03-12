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

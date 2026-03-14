"""
Page 2 - Geopolitical Triggers
Event timeline, pre/during/post performance, correlation shifts, vol comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.data.loader import load_all_prices, load_returns
from src.data.config import GEOPOLITICAL_EVENTS, EQUITY_TICKERS, COMMODITY_TICKERS, PALETTE
from src.analysis.events import (
    event_window_returns, event_normalised_prices,
    pre_post_volatility, correlation_shift,
)
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_footer,
)


def page_geopolitical(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.3rem">Geopolitical Trigger Analysis</h1>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "How do commodity shocks born from war, sanctions, and supply disruptions transmit "
        "into global equity markets? This page isolates each event window and measures "
        "cross-asset performance, volatility shifts, and correlation regime changes across "
        "pre, during, and post-event periods."
    )

    # ── Load ───────────────────────────────────────────────────────────────
    with st.spinner("Loading market data…"):
        eq_p, cmd_p = load_all_prices(start, end)
        eq_r, cmd_r = load_returns(start, end)

    if eq_p.empty or cmd_p.empty:
        st.error("Market data unavailable.")
        return

    all_prices = pd.concat([eq_p, cmd_p], axis=1)
    all_returns = pd.concat([eq_r, cmd_r], axis=1)

    # ── Event selector ─────────────────────────────────────────────────────
    st.markdown("---")
    event_names = [f"{e['label']} - {e['name']}" for e in GEOPOLITICAL_EVENTS]
    selected = st.selectbox("Select event", event_names, index=6)  # Default: Ukraine War
    ev = GEOPOLITICAL_EVENTS[event_names.index(selected)]

    c_info1, c_info2, c_info3 = st.columns(3)
    c_info1.markdown(
        f"""<div style="border-left:3px solid {ev['color']};padding:0.4rem 0.8rem;
        background:#fafaf8">
        <div style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.12em;
        color:{ev['color']};font-weight:700">{ev['category']}</div>
        <div style="font-size:0.88rem;font-weight:700;color:#000">{ev['name']}</div>
        </div>""", unsafe_allow_html=True,
    )
    c_info2.markdown(
        f"""<div style="border-left:3px solid #E8E5E0;padding:0.4rem 0.8rem">
        <div style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.12em;
        color:#666666;font-weight:600">Period</div>
        <div style="font-size:0.82rem;font-weight:600;color:#000;
        font-family:'JetBrains Mono',monospace">
        {ev['start'].strftime('%d %b %Y')} → {ev['end'].strftime('%d %b %Y')}</div>
        </div>""", unsafe_allow_html=True,
    )
    c_info3.markdown(
        f"""<div style="border-left:3px solid #E8E5E0;padding:0.4rem 0.8rem">
        <div style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.12em;
        color:#666666;font-weight:600">Description</div>
        <div style="font-size:0.74rem;color:#333333;line-height:1.65">
        {ev['description'][:180]}…</div>
        </div>""", unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    pre_days  = st.slider("Pre-event window (days)", 15, 90, 30)
    post_days = st.slider("Post-event window (days)", 15, 180, 60)

    # ── Asset selector ─────────────────────────────────────────────────────
    all_asset_names = list(eq_p.columns) + list(cmd_p.columns)
    default_assets = [
        "S&P 500", "Eurostoxx 50", "Nikkei 225",
        "WTI Crude Oil", "Gold", "Wheat", "Copper",
    ]
    default_assets = [a for a in default_assets if a in all_asset_names]

    selected_assets = st.multiselect(
        "Assets to compare",
        options=all_asset_names,
        default=default_assets,
    )
    if not selected_assets:
        st.info("Select at least one asset.")
        return

    assets_in_prices = [a for a in selected_assets if a in all_prices.columns]

    # ── 1. Indexed price chart ─────────────────────────────────────────────
    st.subheader("Indexed Performance Around Event (Base = Event Start)")
    _definition_block(
        "How to Read This Chart",
        "Prices are re-indexed to 100 at the event start date. "
        "Values above 100 indicate appreciation; below 100 indicate loss. "
        "The shaded region marks the event window.",
    )

    normed = event_normalised_prices(
        all_prices[assets_in_prices],
        ev["start"], event_end=ev["end"],
        pre_days=pre_days, post_days=post_days,
    )

    if not normed.empty:
        fig_idx = go.Figure()
        for i, col in enumerate(normed.columns):
            fig_idx.add_trace(go.Scatter(
                x=normed.index, y=normed[col],
                name=col,
                line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            ))
        fig_idx.add_vrect(
            x0=str(ev["start"]), x1=str(ev["end"]),
            fillcolor=ev["color"], opacity=0.08,
            layer="below", line_width=0,
            annotation_text="Event Window",
            annotation_font=dict(size=9, color=ev["color"]),
        )
        fig_idx.add_hline(y=100, line=dict(color="#ABABAB", width=1, dash="dot"))
        _chart(_style_fig(fig_idx, height=420))

    # ── 2. Pre / During / Post bar chart ──────────────────────────────────
    st.subheader("Cumulative Return: Pre / During / Post")

    ew = event_window_returns(
        all_prices[assets_in_prices],
        ev["start"], ev["end"],
        pre_days=pre_days, post_days=post_days,
    )

    perf_df = pd.DataFrame({
        ew["labels"]["pre"]:    ew["pre"],
        ew["labels"]["during"]: ew["during"],
        ew["labels"]["post"]:   ew["post"],
    }).T.fillna(0)

    if not perf_df.empty:
        fig_bar = go.Figure()
        bar_colors = [PALETTE[5], ev["color"], PALETTE[4]]
        for i, period in enumerate(perf_df.index):
            fig_bar.add_trace(go.Bar(
                name=period,
                x=perf_df.columns.tolist(),
                y=perf_df.loc[period].values,
                marker_color=bar_colors[i % 3],
            ))
        fig_bar.update_layout(
            template="purdue",
            barmode="group",
            height=400,
            yaxis=dict(title="Cumulative Return (%)", ticksuffix="%"),
            xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
        )
        _chart(fig_bar)

    # ── 3. Volatility shift ────────────────────────────────────────────────
    st.subheader("Pre vs Post-Event Annualised Volatility")

    eq_cols_sel  = [a for a in selected_assets if a in eq_r.columns]
    cmd_cols_sel = [a for a in selected_assets if a in cmd_r.columns]
    sel_returns  = all_returns[[a for a in selected_assets if a in all_returns.columns]]

    vol_df = pre_post_volatility(sel_returns, ev["start"], ev["end"], window=pre_days)
    if not vol_df.empty:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            name="Pre-Event Vol %",
            x=vol_df.index, y=vol_df["Pre-Event Vol %"],
            marker_color=PALETTE[5],
        ))
        fig_vol.add_trace(go.Bar(
            name="Post-Event Vol %",
            x=vol_df.index, y=vol_df["Post-Event Vol %"],
            marker_color=ev["color"],
        ))
        fig_vol.update_layout(
            template="purdue",
            barmode="group",
            height=350,
            yaxis=dict(title="Annualised Vol (%)", ticksuffix="%"),
            xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
        )
        _chart(fig_vol)

    _section_note(
        f"Volatility is annualised (daily std × √252). "
        f"Events like {ev['label']} typically cause vol to spike 30–80% "
        "across correlated assets in the post-event window."
    )

    # ── 4. Correlation shift ───────────────────────────────────────────────
    if eq_cols_sel and cmd_cols_sel:
        st.subheader("Correlation Shift: Pre → During → Post Event")
        _definition_block(
            "Correlation Regime Shift",
            "When geopolitical crises hit, equities and commodities often become "
            "more correlated (risk-off panic selling) or sharply diverge (safe-haven flows). "
            "This table shows whether correlations increased or decreased around the event.",
        )

        eq_sel_r   = eq_r[[c for c in eq_cols_sel if c in eq_r.columns]]
        cmd_sel_r  = cmd_r[[c for c in cmd_cols_sel if c in cmd_r.columns]]
        corr_shift = correlation_shift(
            eq_sel_r, cmd_sel_r,
            ev["start"], ev["end"],
            pre_days=pre_days, post_days=post_days,
        )

        if not corr_shift.empty:
            # Colour shift column
            def _colour_shift(val):
                if pd.isna(val): return ""
                if val > 0.1:  return "background-color:#fde8e8;color:#c0392b"
                if val < -0.1: return "background-color:#e8f5e9;color:#2e7d32"
                return ""

            styled = corr_shift.style.applymap(_colour_shift, subset=["Shift"])
            st.dataframe(styled, use_container_width=True, height=280)

    _takeaway_block(
        f"<b>{ev['name']}</b>: Cross-asset correlation typically "
        "spikes during the first 2–4 weeks of a geopolitical shock as markets "
        "simultaneously de-risk. Safe-haven assets (gold, natural gas) then decouple "
        "as regime-specific dynamics reassert themselves."
    )

    _page_conclusion(
        ev["label"],
        ev["description"],
    )
    _page_footer()

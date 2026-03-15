"""
Page 2 - Geopolitical Triggers
Event timeline, pre/during/post performance, correlation shifts, vol comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    _insight_note, _line_style, _EQUITY_REGIONS,
)

_F = "font-family:'DM Sans',sans-serif;"


def _label(txt: str) -> None:
    st.markdown(
        f'<p style="{_F}font-size:0.58rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.14em;color:#8E6F3E;margin:0 0 5px 0">{txt}</p>',
        unsafe_allow_html=True,
    )


def page_geopolitical(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.1rem">Geopolitical Trigger Analysis</h1>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#555;'
        'margin:0 0 0.7rem">Event windows · Pre/During/Post performance · Vol shifts · Correlation regime change</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading market data…"):
        eq_p, cmd_p = load_all_prices(start, end)
        eq_r, cmd_r = load_returns(start, end)

    if eq_p.empty or cmd_p.empty:
        st.error("Market data unavailable.")
        return

    all_prices  = pd.concat([eq_p, cmd_p], axis=1)
    all_returns = pd.concat([eq_r, cmd_r], axis=1)

    # ── Control strip ───────────────────────────────────────────────────────
    ev_names = [f"{e['label']}: {e['name']}" for e in GEOPOLITICAL_EVENTS]
    ctrl_l, ctrl_r = st.columns([2.2, 1])

    with ctrl_l:
        selected = st.selectbox("Event", ev_names, index=6, label_visibility="collapsed")
    ev = GEOPOLITICAL_EVENTS[ev_names.index(selected)]

    # Event info bar
    st.markdown(
        f'<div style="display:flex;gap:16px;align-items:stretch;margin-bottom:8px">'
        f'<div style="border-left:3px solid {ev["color"]};padding:4px 10px;background:#fafaf8;flex:1">'
        f'<div style="{_F}font-size:0.56rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.10em;color:{ev["color"]}">{ev["category"]}</div>'
        f'<div style="{_F}font-size:0.85rem;font-weight:700;color:#000">{ev["name"]}</div>'
        f'</div>'
        f'<div style="border-left:1px solid #E8E5E0;padding:4px 10px">'
        f'<div style="{_F}font-size:0.56rem;font-weight:600;text-transform:uppercase;letter-spacing:0.10em;color:#888">Period</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;font-weight:600">'
        f'{ev["start"].strftime("%d %b %Y")} → {ev["end"].strftime("%d %b %Y")}</div>'
        f'</div>'
        f'<div style="border-left:1px solid #E8E5E0;padding:4px 10px;flex:2">'
        f'<div style="{_F}font-size:0.56rem;font-weight:600;text-transform:uppercase;letter-spacing:0.10em;color:#888">Context</div>'
        f'<div style="{_F}font-size:0.64rem;color:#333;line-height:1.5">{ev["description"][:180]}{"…" if len(ev["description"])>180 else ""}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Compact controls row
    ctl1, ctl2, ctl3 = st.columns([0.8, 0.8, 2.4])
    pre_days  = ctl1.slider("Pre-event (days)",  15, 90,  30, label_visibility="visible")
    post_days = ctl2.slider("Post-event (days)", 15, 180, 60, label_visibility="visible")

    all_asset_names = list(eq_p.columns) + list(cmd_p.columns)
    default_assets  = [a for a in ["S&P 500","Eurostoxx 50","Nikkei 225",
                                   "WTI Crude Oil","Gold","Wheat","Copper"]
                       if a in all_asset_names]
    selected_assets = ctl3.multiselect("Assets", all_asset_names, default=default_assets)

    if not selected_assets:
        st.info("Select at least one asset.")
        return

    assets_in_prices = [a for a in selected_assets if a in all_prices.columns]

    st.markdown('<div style="margin:0.3rem 0 0.5rem;border-top:1px solid #E8E5E0"></div>',
                unsafe_allow_html=True)

    # Compute
    normed = event_normalised_prices(
        all_prices[assets_in_prices],
        ev["start"], event_end=ev["end"],
        pre_days=pre_days, post_days=post_days,
    )
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

    # ── ROW 1: Indexed chart (wider) | Pre/During/Post bar (narrower) ───────
    row1_l, row1_r = st.columns([1.7, 1])

    with row1_l:
        _label("Indexed Performance: Base = Event Start (100)")
        if not normed.empty:
            fig_idx = go.Figure()
            eq_i = cmd_i = 0
            for col in normed.columns:
                ls = _line_style(col, eq_i, cmd_i)
                if col in _EQUITY_REGIONS:
                    eq_i += 1
                else:
                    cmd_i += 1
                fig_idx.add_trace(go.Scatter(
                    x=normed.index, y=normed[col], name=col,
                    line=ls,
                ))
            fig_idx.add_vrect(
                x0=str(ev["start"]), x1=str(ev["end"]),
                fillcolor=ev["color"], opacity=0.08, layer="below", line_width=0,
                annotation_text="Event Window", annotation_font=dict(size=8, color=ev["color"]),
            )
            fig_idx.add_hline(y=100, line=dict(color="#ABABAB", width=1, dash="dot"))
            _chart(_style_fig(fig_idx, height=360))
            _insight_note(
                "Tracks how each asset performed around the event, indexed to 100 at the event "
                "start date. Lines rising above 100 gained value; lines falling below 100 lost it. "
                "The shaded band marks the active event window - look for sharp divergences "
                "between commodities and equities as the event unfolds."
            )

    with row1_r:
        _label("Cumulative Return by Period")
        if not perf_df.empty:
            fig_bar = go.Figure()
            bar_colors = [PALETTE[5], ev["color"], PALETTE[4]]
            for i, period in enumerate(perf_df.index):
                fig_bar.add_trace(go.Bar(
                    name=period, x=perf_df.columns.tolist(),
                    y=perf_df.loc[period].values,
                    marker_color=bar_colors[i % 3],
                ))
            fig_bar.update_layout(
                template="purdue", barmode="group", height=360,
                yaxis=dict(title="Return (%)", ticksuffix="%"),
                xaxis=dict(tickangle=-40, tickfont=dict(size=8)),
                margin=dict(l=40, r=10, t=20, b=80),
                legend=dict(orientation="h", y=1.05),
            )
            _chart(fig_bar)
            _insight_note(
                "Compares total cumulative returns across three distinct phases: before, during, "
                "and after the event. Assets with large positive post-event bars recovered quickly; "
                "those remaining negative post-event suggest lasting structural damage from the shock."
            )

    st.markdown('<div style="margin:0.5rem 0;border-top:1px solid #E8E5E0"></div>',
                unsafe_allow_html=True)

    # ── ROW 2: Volatility shift | Correlation shift table ───────────────────
    eq_cols_sel  = [a for a in selected_assets if a in eq_r.columns]
    cmd_cols_sel = [a for a in selected_assets if a in cmd_r.columns]
    sel_returns  = all_returns[[a for a in selected_assets if a in all_returns.columns]]
    vol_df = pre_post_volatility(sel_returns, ev["start"], ev["end"], window=pre_days)

    row2_l, row2_r = st.columns([1.2, 1])

    with row2_l:
        _label("Pre vs Post-Event Annualised Volatility")
        if not vol_df.empty:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                name="Pre-Event Vol", x=vol_df.index, y=vol_df["Pre-Event Vol %"],
                marker_color=PALETTE[5],
            ))
            fig_vol.add_trace(go.Bar(
                name="Post-Event Vol", x=vol_df.index, y=vol_df["Post-Event Vol %"],
                marker_color=ev["color"],
            ))
            fig_vol.update_layout(
                template="purdue", barmode="group", height=320,
                yaxis=dict(title="Annualised Vol (%)", ticksuffix="%"),
                xaxis=dict(tickangle=-40, tickfont=dict(size=8)),
                margin=dict(l=40, r=10, t=20, b=80),
                legend=dict(orientation="h", y=1.05),
            )
            _chart(fig_vol)
            _insight_note(
                "Compares annualised price volatility in the period before versus after the event. "
                "Taller post-event bars indicate the shock left markets in a more uncertain state "
                "than before. Commodities with persistent post-event volatility elevation often "
                "signal ongoing supply disruption rather than a one-day price spike."
            )

    with row2_r:
        _label("Correlation Shift: Pre → During → Post")
        if eq_cols_sel and cmd_cols_sel:
            eq_sel_r  = eq_r[[c for c in eq_cols_sel if c in eq_r.columns]]
            cmd_sel_r = cmd_r[[c for c in cmd_cols_sel if c in cmd_r.columns]]
            corr_shift = correlation_shift(
                eq_sel_r, cmd_sel_r, ev["start"], ev["end"],
                pre_days=pre_days, post_days=post_days,
            )
            if not corr_shift.empty:
                def _colour_shift(val):
                    if pd.isna(val): return ""
                    if val > 0.1:  return "background-color:#fde8e8;color:#c0392b"
                    if val < -0.1: return "background-color:#e8f5e9;color:#2e7d32"
                    return ""
                styled = corr_shift.style.applymap(_colour_shift, subset=["Shift"])
                st.dataframe(styled, use_container_width=True, height=320)
                _insight_note(
                    "Shows whether the event changed how closely equities and commodities moved "
                    "together. A large positive Shift value (red) means two assets that used to "
                    "move independently started moving in tandem - a contagion signal. "
                    "A large negative Shift (green) signals the event broke an existing relationship, "
                    "which can create short-term diversification opportunities."
                )
        else:
            st.markdown(
                f'<p style="{_F}font-size:0.70rem;color:#888;margin-top:1rem">'
                f'Select both equity and commodity assets to see correlation shifts.</p>',
                unsafe_allow_html=True,
            )

    _page_footer()

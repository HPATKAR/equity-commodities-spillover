"""
Page 5 — Commodities to Watch
Live snapshot, intraday (hourly) price & vol, daily historical charts, commodity cards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta

from src.data.loader import (
    load_commodity_prices, load_returns, load_live_snapshot,
    load_hourly_commodity_prices, load_hourly_returns,
)
from src.analysis.cot import load_cot_data, plot_cot_overlay, cot_extremes_table
from src.data.config import (
    WATCHLIST, COMMODITY_TICKERS, COMMODITY_GROUPS,
    GEOPOLITICAL_EVENTS, PALETTE,
)
from src.analysis.correlations import rolling_correlation
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _section_note,
    _definition_block, _takeaway_block, _page_footer,
)

_ALERT_MAP = {name: alert for (_, name, _, alert) in WATCHLIST}
_GROUP_MAP  = {name: grp  for (_, name, grp,  _)   in WATCHLIST}

# Commodity futures trade ~23 h/day on CME Globex
_HOURLY_ANNUAL = np.sqrt(252 * 23)   # ≈ 76.1


def _vol_regime(vol_pct: float) -> tuple[str, str]:
    if vol_pct < 20: return "Low Vol",  "#2e7d32"
    if vol_pct < 35: return "Normal",   "#555960"
    if vol_pct < 55: return "Elevated", "#e67e22"
    return "High Vol", "#c0392b"


def page_watchlist(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.3rem">Commodities to Watch</h1>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "A curated watchlist of key commodity futures with live price data, "
        "intraday (hourly) price and volatility, YTD performance, and geopolitical "
        "risk context. Hourly data covers up to 730 days via yfinance."
    )

    watch_tickers = {name: tk for (tk, name, _, _) in WATCHLIST}

    # ── Live snapshot ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Live Snapshot")

    with st.spinner("Fetching live prices…"):
        snapshot = load_live_snapshot(watch_tickers)

    if not snapshot.empty:
        def _colour_cell(val):
            if isinstance(val, (int, float)):
                if val > 0: return "color: #2e7d32; font-weight: 600"
                if val < 0: return "color: #c0392b; font-weight: 600"
            return ""

        styled = (
            snapshot.style
            .applymap(_colour_cell, subset=["1D %", "5D %", "YTD %"])
            .format({"Last": "{:.2f}", "1D %": "{:+.2f}%",
                     "5D %": "{:+.2f}%", "YTD %": "{:+.2f}%"})
        )
        st.dataframe(styled, use_container_width=True, height=340)
    else:
        st.warning("Live snapshot unavailable. Check yfinance connectivity.")

    # ── Intraday view (hourly) ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Intraday View (Hourly)")
    _definition_block(
        "Hourly Data",
        "yfinance provides hourly OHLCV for up to 730 days. "
        "Volatility is annualised using σ_hourly × √(252 × 23) "
        "— commodity futures trade ~23 hours/day on CME Globex.",
    )

    with st.spinner("Loading hourly prices…"):
        h_prices = load_hourly_commodity_prices(start, end)
        h_eq_r, h_cmd_r = load_hourly_returns(start, end)

    watch_names_h = [n for (_, n, _, _) in WATCHLIST if n in h_prices.columns]

    # ── Controls ──
    c1, c2 = st.columns([1, 2])
    lookback_days = c1.select_slider(
        "Lookback window",
        options=[1, 5, 10, 30, 60, 90, 180, 365],
        value=30,
        format_func=lambda x: f"{x}d",
        key="wl_h_lookback",
    )
    h_group_filter = c2.selectbox(
        "Filter by group",
        ["All"] + list(COMMODITY_GROUPS.keys()),
        key="wl_h_group",
    )

    h_names_filtered = [
        n for n in watch_names_h
        if h_group_filter == "All" or _GROUP_MAP.get(n) == h_group_filter
    ]
    h_selected = st.multiselect(
        "Select commodities",
        h_names_filtered,
        default=h_names_filtered[:4],
        key="wl_h_select",
    )

    if h_selected and not h_prices.empty:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        h_slice = h_prices[h_selected].loc[cutoff:].dropna(how="all")

        # Indexed hourly price chart
        st.markdown("##### Hourly Price (Indexed, Base = 100)")
        if not h_slice.empty:
            normed_h = (h_slice / h_slice.iloc[0]) * 100
            fig_h = go.Figure()
            for i, col in enumerate(normed_h.columns):
                fig_h.add_trace(go.Scatter(
                    x=normed_h.index, y=normed_h[col],
                    name=col,
                    line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
                ))
            fig_h.add_hline(y=100, line=dict(color="#ABABAB", width=1, dash="dot"))
            fig_h.update_layout(
                template="purdue", height=360,
                xaxis=dict(rangeslider=dict(visible=True, thickness=0.04), type="date"),
            )
            _chart(fig_h)

        # Hourly rolling 24-bar (24h) annualised vol
        st.markdown("##### Rolling 24h Annualised Volatility (%)")
        if not h_cmd_r.empty:
            h_ret_slice = h_cmd_r[
                [c for c in h_selected if c in h_cmd_r.columns]
            ].loc[cutoff:]
            fig_hv = go.Figure()
            for i, col in enumerate(h_selected):
                if col not in h_cmd_r.columns:
                    continue
                hv = h_cmd_r[col].loc[cutoff:].rolling(24).std() * _HOURLY_ANNUAL * 100
                fig_hv.add_trace(go.Scatter(
                    x=hv.index, y=hv.values,
                    name=col,
                    line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
                ))
            fig_hv.update_layout(
                template="purdue", height=300,
                xaxis=dict(rangeslider=dict(visible=False), type="date"),
                yaxis=dict(ticksuffix="%"),
            )
            _chart(fig_hv)

    elif not h_prices.empty and not h_selected:
        st.info("Select at least one commodity above.")

    # ── Historical daily charts ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Historical Price & Daily Volatility")

    with st.spinner("Loading daily prices…"):
        cmd_p  = load_commodity_prices(start, end)
        _, cmd_r = load_returns(start, end)

    watch_names = [n for (_, n, _, _) in WATCHLIST if n in cmd_p.columns]

    group_filter = st.selectbox(
        "Filter by group",
        ["All"] + list(COMMODITY_GROUPS.keys()),
        key="wl_d_group",
    )
    if group_filter != "All":
        watch_names = [n for n in watch_names if _GROUP_MAP.get(n) == group_filter]

    selected = st.multiselect(
        "Select commodities",
        watch_names,
        default=watch_names[:5],
        key="wl_d_select",
    )

    if not selected:
        st.info("Select at least one commodity.")
    else:
        # Indexed daily price chart
        st.markdown("##### Indexed Price (Base = 100 at Start Date)")
        prices_sel = cmd_p[selected].dropna(how="all")
        if not prices_sel.empty:
            normed = (prices_sel / prices_sel.iloc[0]) * 100
            fig_norm = go.Figure()
            for i, col in enumerate(normed.columns):
                fig_norm.add_trace(go.Scatter(
                    x=normed.index, y=normed[col],
                    name=col,
                    line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
                ))
            for ev in GEOPOLITICAL_EVENTS[-5:]:
                fig_norm.add_vrect(
                    x0=str(ev["start"]), x1=str(ev["end"]),
                    fillcolor=ev["color"], opacity=0.06,
                    layer="below", line_width=0,
                )
            fig_norm.add_hline(y=100, line=dict(color="#ABABAB", width=1, dash="dot"))
            _chart(_style_fig(fig_norm, height=400))

        # Rolling 30d daily vol
        st.markdown("##### Rolling 30d Annualised Volatility (%)")
        if not cmd_r.empty:
            fig_vol = go.Figure()
            for i, col in enumerate(selected):
                if col in cmd_r.columns:
                    rv = cmd_r[col].rolling(30).std() * np.sqrt(252) * 100
                    fig_vol.add_trace(go.Scatter(
                        x=rv.index, y=rv.values,
                        name=col,
                        line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
                    ))
            for ev in GEOPOLITICAL_EVENTS[-5:]:
                fig_vol.add_vrect(
                    x0=str(ev["start"]), x1=str(ev["end"]),
                    fillcolor=ev["color"], opacity=0.06,
                    layer="below", line_width=0,
                )
            fig_vol.update_layout(yaxis=dict(ticksuffix="%"))
            _chart(_style_fig(fig_vol, height=340))

    # ── Commodity cards ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Commodity Cards")

    # Pre-compute 24h hourly vol for all watch names
    _h_vol: dict[str, float] = {}
    if not h_cmd_r.empty:
        for n in [nm for (_, nm, _, _) in WATCHLIST]:
            if n in h_cmd_r.columns:
                s = h_cmd_r[n].dropna()
                _h_vol[n] = s.iloc[-24:].std() * _HOURLY_ANNUAL * 100 if len(s) >= 24 else 0.0

    all_selected = list(dict.fromkeys((h_selected or []) + (selected or [])))
    card_names = [n for (_, n, _, _) in WATCHLIST
                  if n in cmd_p.columns and (not all_selected or n in all_selected)]

    cols = st.columns(2)
    for idx, name in enumerate(card_names):
        with cols[idx % 2]:
            price_series = cmd_p[name].dropna()
            ret_series   = cmd_r[name].dropna() if name in cmd_r.columns else pd.Series()

            last_price  = price_series.iloc[-1] if not price_series.empty else 0
            d1_pct      = (price_series.iloc[-1] / price_series.iloc[-2] - 1) * 100 \
                          if len(price_series) >= 2 else 0
            ytd_slice   = price_series.loc[str(date.today().year):]
            ytd_pct     = (ytd_slice.iloc[-1] / ytd_slice.iloc[0] - 1) * 100 \
                          if len(ytd_slice) >= 2 else 0
            vol_30d     = ret_series.iloc[-30:].std() * np.sqrt(252) * 100 \
                          if len(ret_series) >= 30 else 0
            vol_24h     = _h_vol.get(name, 0.0)
            regime_lbl, regime_col = _vol_regime(vol_30d)
            alert_ctx   = _ALERT_MAP.get(name, "")
            group       = _GROUP_MAP.get(name, "")

            d1_color  = "#2e7d32" if d1_pct  >= 0 else "#c0392b"
            ytd_color = "#2e7d32" if ytd_pct >= 0 else "#c0392b"

            vol_24h_html = (
                f'<div style="font-size:0.68rem;color:#555960">'
                f'24h Vol <span style="font-weight:600;color:{regime_col}">'
                f'{vol_24h:.1f}%</span></div>'
            ) if vol_24h > 0 else ""

            st.markdown(
                f"""<div style="border:1px solid #E8E5E0;border-radius:5px;
                padding:0.9rem 1rem;margin-bottom:0.8rem;background:#fff">
                <div style="display:flex;justify-content:space-between;align-items:center">
                  <div>
                    <span style="font-size:0.58rem;text-transform:uppercase;letter-spacing:0.12em;
                    color:#9D9795;font-weight:600">{group}</span>
                    <div style="font-size:0.9rem;font-weight:700;color:#000;
                    font-family:'DM Sans',sans-serif">{name}</div>
                  </div>
                  <div style="text-align:right">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.95rem;
                    font-weight:700">{last_price:,.2f}</div>
                    <div style="font-size:0.68rem;color:{d1_color};
                    font-family:'JetBrains Mono',monospace">{d1_pct:+.2f}% 1d</div>
                  </div>
                </div>
                <div style="display:flex;gap:1rem;margin-top:0.5rem">
                  <div style="font-size:0.68rem;color:#555960">
                    YTD <span style="color:{ytd_color};font-weight:600">{ytd_pct:+.1f}%</span>
                  </div>
                  <div style="font-size:0.68rem;color:#555960">
                    30d Vol <span style="color:{regime_col};font-weight:600">{vol_30d:.1f}% ({regime_lbl})</span>
                  </div>
                  {vol_24h_html}
                </div>
                <div style="margin-top:0.5rem;padding-top:0.5rem;border-top:1px solid #F0EDEA">
                  <span style="font-size:0.64rem;color:#8E6F3E">⚑ {alert_ctx}</span>
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Hourly correlation vs S&P 500 (recent, hourly) ─────────────────────
    if "S&P 500" in h_eq_r.columns and not h_cmd_r.empty:
        st.markdown("---")
        st.subheader("Rolling 24h Hourly Correlation vs S&P 500 (Recent)")
        _section_note(
            "Computed on hourly log returns with a 24-bar rolling window. "
            "Shows intraday co-movement — spikes reveal short-term contagion "
            "faster than daily data."
        )
        sp_h = h_eq_r["S&P 500"]
        cutoff_corr = pd.Timestamp.now() - pd.Timedelta(days=90)

        h_corr_names = [n for (_, n, _, _) in WATCHLIST if n in h_cmd_r.columns]
        corr_sel = st.multiselect(
            "Commodities for hourly correlation",
            h_corr_names,
            default=h_corr_names[:4],
            key="wl_h_corr_sel",
        )
        if corr_sel:
            fig_hcorr = go.Figure()
            for i, name in enumerate(corr_sel):
                rc = rolling_correlation(sp_h, h_cmd_r[name], 24).loc[cutoff_corr:]
                fig_hcorr.add_trace(go.Scatter(
                    x=rc.index, y=rc.values,
                    name=name,
                    line=dict(color=PALETTE[i % len(PALETTE)], width=1.4),
                ))
            fig_hcorr.add_hline(y=0, line=dict(color="#ABABAB", width=1, dash="dot"))
            fig_hcorr.update_layout(
                template="purdue", height=340,
                xaxis=dict(rangeslider=dict(visible=False), type="date"),
            )
            _chart(fig_hcorr)

    # ── Daily correlation vs S&P 500 ──────────────────────────────────────
    from src.data.loader import load_equity_prices
    eq_p = load_equity_prices(start, end)
    if "S&P 500" in eq_p.columns and not cmd_r.empty:
        from src.data.loader import load_returns as _lr
        eq_r2, _ = _lr(start, end)
        sp_r = eq_r2.get("S&P 500")
        if sp_r is not None and selected:
            st.markdown("---")
            st.subheader("Rolling 60d Daily Correlation vs S&P 500")
            fig_spx = go.Figure()
            for i, name in enumerate(selected):
                if name in cmd_r.columns:
                    rc = rolling_correlation(sp_r, cmd_r[name], 60)
                    fig_spx.add_trace(go.Scatter(
                        x=rc.index, y=rc.values,
                        name=name,
                        line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
                    ))
            fig_spx.add_hline(y=0, line=dict(color="#ABABAB", width=1, dash="dot"))
            for ev in GEOPOLITICAL_EVENTS:
                fig_spx.add_vrect(
                    x0=str(ev["start"]), x1=str(ev["end"]),
                    fillcolor=ev["color"], opacity=0.05, layer="below", line_width=0,
                )
            _chart(_style_fig(fig_spx, height=380))

            _takeaway_block(
                "Gold's correlation with S&P 500 turns sharply negative during panic selling "
                "as capital flees to safe havens. WTI and copper track equities during normal "
                "conditions (growth proxy) but decouple sharply during supply shocks."
            )

    # ── CFTC COT Positioning ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("CFTC Commitments of Traders — Speculative Positioning")
    _definition_block(
        "COT Report (CFTC)",
        "The CFTC publishes weekly Commitments of Traders reports showing how "
        "non-commercial (speculative) traders are positioned in commodity futures. "
        "<b>Net Spec % OI</b> = (Long − Short) ÷ Open Interest × 100. "
        "Readings above +25% signal crowded longs (contrarian sell); "
        "below −25% signal crowded shorts (contrarian buy). "
        "Data sourced from CFTC's public disaggregated futures files.",
    )

    with st.spinner("Loading COT data from CFTC… (first load may take ~10s)"):
        cot_df = load_cot_data(years=3)

    if cot_df.empty:
        st.warning("COT data unavailable. Check internet connectivity to www.cftc.gov.")
    else:
        # Extremes summary table
        st.markdown("##### Current Positioning Extremes")
        ext_tbl = cot_extremes_table(cot_df)
        if not ext_tbl.empty:
            def _sig_col(val):
                if "Crowded Long"  in str(val): return "color:#c0392b;font-weight:700"
                if "Crowded Short" in str(val): return "color:#2e7d32;font-weight:700"
                return ""
            styled_cot = ext_tbl.style.applymap(_sig_col, subset=["Signal"])
            st.dataframe(styled_cot, use_container_width=True, hide_index=True)

        # Per-commodity overlay chart
        st.markdown("##### Positioning Detail + Price Overlay")
        cot_markets = sorted(cot_df["market"].unique().tolist())
        cot_sel = st.selectbox(
            "Select commodity",
            cot_markets,
            index=cot_markets.index("WTI Crude Oil") if "WTI Crude Oil" in cot_markets else 0,
            key="wl_cot_sel",
        )

        price_s = cmd_p[cot_sel].dropna() if cot_sel in cmd_p.columns else None
        _chart(plot_cot_overlay(cot_df, cot_sel, price_s, height=420))

        _takeaway_block(
            "Extreme speculative positioning is a powerful contrarian indicator. "
            "When non-commercial traders are maximally long (>+25% of OI) — typically near "
            "commodity price peaks — a reversal often follows as longs are forced to unwind. "
            "The reverse applies for crowded shorts. Use alongside the correlation regime "
            "to distinguish supply-driven moves from speculative crowding."
        )

    _section_note(
        "Daily prices from Yahoo Finance. Daily vol: σ × √252. "
        "Hourly vol: σ_hourly × √(252 × 23) — commodity futures trade ~23h/day. "
        "Hourly lookback capped at 730 days by yfinance. "
        "COT data: CFTC disaggregated futures-only, updated weekly (Tuesday releases)."
    )
    _page_footer()

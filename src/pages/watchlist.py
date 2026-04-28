"""
Page 5 - Commodities to Watch
Bloomberg-style grid: Live snapshot | Intraday + daily charts | COT positioning.
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
    _style_fig, _chart, _page_intro, _thread, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_header, _page_footer, _insight_note,
    _line_style, _EQUITY_REGIONS,
)

_F = "font-family:'DM Sans',sans-serif;"
_ALERT_MAP = {name: alert for (_, name, _, alert) in WATCHLIST}
_GROUP_MAP  = {name: grp  for (_, name, grp,  _)   in WATCHLIST}
_HOURLY_ANNUAL = np.sqrt(252 * 23)   # ≈ 76.1


_M = "font-family:'JetBrains Mono',monospace;"
_G = "#CFB991"


def _label(txt: str) -> None:
    st.markdown(
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:0.58rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.14em;color:#8890a1;margin:0.8rem 0 0.4rem 0">{txt}</p>',
        unsafe_allow_html=True,
    )


def _panel_note(txt: str) -> None:
    st.markdown(
        f'<p style="{_F}font-size:0.64rem;color:#666;line-height:1.5;margin:2px 0 6px 0">{txt}</p>',
        unsafe_allow_html=True,
    )


def _vol_regime(vol_pct: float) -> tuple[str, str]:
    if vol_pct < 20: return "Low Vol",  "#2e7d32"
    if vol_pct < 35: return "Normal",   "#555960"
    if vol_pct < 55: return "Elevated", "#e67e22"
    return "High Vol", "#c0392b"


def page_watchlist(start: str, end: str, fred_key: str = "") -> None:
    _page_header("Commodity Watchlist",
                 "Live Snapshot · Intraday Prices · Daily Historical · CFTC COT Positioning")
    _page_intro(
        "The commodities on this page are the primary spillover conduits into equity markets. "
        "Crude oil, gold, copper, and agricultural futures have historically led equity market "
        "repricing during macro stress - price moves here often precede equity effects by days. "
        "<strong>Use this page as an early warning monitor.</strong> "
        "The COT positioning data adds a second layer: when speculative positioning in a commodity "
        "reaches a historical extreme, mean-reversion is the base case - and that reversal typically "
        "generates its own downstream spillover into correlated equity sectors."
    )

    # ── Conflict commodity pressure banner ────────────────────────────────
    try:
        from src.analysis.conflict_model import score_all_conflicts, aggregate_portfolio_scores
        _wl_cr  = score_all_conflicts()
        _wl_agg = aggregate_portfolio_scores(_wl_cr)
        _wl_cis = _wl_agg.get("portfolio_cis", 50.0)
        _wl_tps = _wl_agg.get("portfolio_tps", 50.0)

        # Find conflicts with commodity transmission channels
        _wl_commodity_conflicts = []
        for _wl_r in _wl_cr.values():
            if _wl_r.get("state") != "active":
                continue
            _tx = _wl_r.get("transmission", {})
            _commodity_ch = {k: v for k, v in _tx.items()
                             if any(x in k for x in ["oil", "commodity", "food", "metal", "energy"])}
            if _commodity_ch:
                _top_ch = max(_commodity_ch, key=_commodity_ch.get)
                _wl_commodity_conflicts.append(
                    f'{_wl_r["label"]} [{_top_ch.replace("_"," ").upper()}]'
                )

        _wl_col = "#c0392b" if _wl_tps >= 65 else "#e67e22" if _wl_tps >= 45 else "#CFB991"
        if _wl_commodity_conflicts:
            _wl_conf_str = " · ".join(_wl_commodity_conflicts[:3])
            st.markdown(
                f'<div style="background:#080808;border:1px solid #1e1e1e;'
                f'border-left:3px solid {_wl_col};padding:.4rem .9rem;'
                f'margin-bottom:.6rem;display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
                f'font-weight:700;color:{_wl_col};white-space:nowrap">COMMODITY WAR PRESSURE</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;color:#8E9AAA">'
                f'{_wl_conf_str}</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                f'color:#8E9AAA;margin-left:auto">'
                f'CIS&nbsp;<b style="color:{_wl_col}">{_wl_cis:.0f}</b>&nbsp;·&nbsp;'
                f'TPS&nbsp;<b style="color:#CFB991">{_wl_tps:.0f}</b>&nbsp;·&nbsp;'
                f'Conflict-driven moves may diverge from fundamentals</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    # ── Load all data upfront ──────────────────────────────────────────────
    with st.spinner("Loading commodity data…"):
        cmd_p  = load_commodity_prices(start, end)
        _, cmd_r = load_returns(start, end)
        h_prices = load_hourly_commodity_prices(start, end)
        h_eq_r, h_cmd_r = load_hourly_returns(start, end)
        try:
            from src.analysis.freshness import record_fetch
            record_fetch("yfinance_prices")
        except Exception:
            pass

    watch_tickers  = {name: tk for (tk, name, _, _) in WATCHLIST}
    watch_names    = [n for (_, n, _, _) in WATCHLIST if n in cmd_p.columns]
    watch_names_h  = [n for (_, n, _, _) in WATCHLIST if n in h_prices.columns]

    # ── Controls strip ────────────────────────────────────────────────────
    with st.expander("Asset & window controls", expanded=False):
        sc1, sc2, sc3, sc4 = st.columns([1, 1, 1.5, 1.5])
        lookback_days = sc1.select_slider(
            "Intraday lookback",
            options=[1, 5, 10, 30, 60, 90, 180, 365], value=30,
            format_func=lambda x: f"{x}d", key="wl_h_lookback",
        )
        h_group_filter = sc2.selectbox(
            "Filter by group", ["All"] + list(COMMODITY_GROUPS.keys()), key="wl_h_group",
        )
        h_names_filtered = [
            n for n in watch_names_h
            if h_group_filter == "All" or _GROUP_MAP.get(n) == h_group_filter
        ]
        h_selected = sc3.multiselect(
            "Intraday assets", h_names_filtered,
            default=h_names_filtered[:4], key="wl_h_select",
        )
        d_selected = sc4.multiselect(
            "Daily assets", watch_names,
            default=watch_names[:5], key="wl_d_select",
        )

    h_selected = h_selected or h_names_filtered[:4]
    d_selected = d_selected or watch_names[:5]

    st.markdown('<div style="margin:0.4rem 0 0.5rem;border-top:1px solid #1a1a1a"></div>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 1: Live Snapshot (full-width compact table)
    # ══════════════════════════════════════════════════════════════════════
    _label("Live Market Snapshot")
    with st.spinner("Fetching live prices…"):
        snapshot = load_live_snapshot(watch_tickers)

    if not snapshot.empty:
        _TBL_CSS_SNAP = """
<style>
.ec-table{width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:0.78rem}
.ec-table th{background:#080808;color:#CFB991;padding:7px 10px;text-align:left;
    border-bottom:1px solid rgba(207,185,145,0.3);font-weight:600;
    letter-spacing:0.06em;text-transform:uppercase;font-size:0.68rem}
.ec-table td{padding:5px 10px;border-bottom:1px solid #1e1e1e;color:#e8e9ed}
.ec-table tr:nth-child(even) td{background:#171717}
.ec-table tr:nth-child(odd) td{background:#111111}
.ec-table tr:hover td{background:#202020}
</style>"""
        snap_pct_cols = ["1D %", "5D %", "YTD %"]
        snap_cols = list(snapshot.columns)
        snap_header = "<th>Asset</th>" + "".join(f"<th>{c}</th>" for c in snap_cols)
        snap_rows = ""
        for asset_name, row in snapshot.iterrows():
            cells = f"<td style='color:#b8b8b8'>{asset_name}</td>"
            for col in snap_cols:
                v = row[col]
                is_nan = pd.isna(v) if not isinstance(v, str) else False
                if is_nan:
                    cells += "<td style='color:#8890a1'>-</td>"
                elif col == "Last":
                    cells += f"<td style='color:#e8e9ed'>{v:.2f}</td>"
                elif col in snap_pct_cols:
                    if v > 0:
                        cells += f"<td style='color:#4ade80;font-weight:600'>{v:+.2f}%</td>"
                    elif v < 0:
                        cells += f"<td style='color:#f87171;font-weight:600'>{v:+.2f}%</td>"
                    else:
                        cells += f"<td style='color:#8890a1'>{v:+.2f}%</td>"
                else:
                    cells += f"<td style='color:#e8e9ed'>{v}</td>"
            snap_rows += f"<tr>{cells}</tr>"
        html_snap = (
            _TBL_CSS_SNAP
            + "<table class='ec-table'>"
            + f"<thead><tr>{snap_header}</tr></thead>"
            + f"<tbody>{snap_rows}</tbody>"
            + "</table>"
        )
        st.markdown(html_snap, unsafe_allow_html=True)
        _insight_note(
            "Real-time price snapshot across all watched commodities. "
            "The 1D % column shows today's price move - the quickest read on which "
            "commodity is under immediate pressure. YTD % gives the year-to-date context: "
            "a commodity down strongly YTD despite a positive 1D reading may just be bouncing "
            "within a longer downtrend."
        )
    else:
        st.warning("Live snapshot unavailable. Check yfinance connectivity.")

    _thread(
        "The snapshot above is a single moment. The intraday charts below add the dimension of time - "
        "showing how each commodity has moved through recent trading sessions and whether current "
        "volatility is normal or elevated."
    )

    st.markdown('<div style="margin:0.5rem 0;border-top:1px solid #1a1a1a"></div>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 2: Intraday (hourly price indexed) | Hourly 24h vol
    # ══════════════════════════════════════════════════════════════════════
    col_hp, col_hv = st.columns(2, gap="medium")

    if h_selected and not h_prices.empty:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        h_slice = h_prices[h_selected].loc[cutoff:].dropna(how="all")

        with col_hp:
            _label(f"Intraday Price: Indexed (last {lookback_days}d)")
            if not h_slice.empty:
                normed_h = (h_slice / h_slice.iloc[0]) * 100
                fig_h = go.Figure()
                eq_i = cmd_i = 0
                for col in normed_h.columns:
                    ls = _line_style(col, eq_i, cmd_i)
                    if col in _EQUITY_REGIONS: eq_i += 1
                    else: cmd_i += 1
                    fig_h.add_trace(go.Scatter(
                        x=normed_h.index, y=normed_h[col], name=col, line=ls,
                    ))
                fig_h.add_hline(y=100, line=dict(color="#ABABAB", width=1, dash="dot"))
                fig_h.update_layout(
                    template="purdue", height=340,
                    xaxis=dict(rangeslider=dict(visible=False), type="date"),
                    margin=dict(l=44, r=20, t=20, b=60),
                )
                _chart(fig_h)
                _insight_note(
                    "Intraday price movement indexed so all assets start at the same "
                    "base level of 100 - allowing fair relative comparison regardless "
                    "of absolute price differences. Lines diverging sharply above or "
                    "below 100 signal a supply or demand shock in that specific market."
                )

        with col_hv:
            _label(f"Rolling 24h Annualised Volatility (last {lookback_days}d)")
            if not h_cmd_r.empty:
                fig_hv = go.Figure()
                cmd_i2 = 0
                for name in h_selected:
                    if name not in h_cmd_r.columns:
                        continue
                    hv = h_cmd_r[name].loc[cutoff:].rolling(24).std() * _HOURLY_ANNUAL * 100
                    ls = _line_style(name, 0, cmd_i2); cmd_i2 += 1
                    fig_hv.add_trace(go.Scatter(
                        x=hv.index, y=hv.values, name=name, line=ls,
                    ))
                fig_hv.add_hline(y=40, line=dict(color="#e67e22", width=1, dash="dot"),
                                 annotation_text="Elevated (40%)", annotation_font_size=8,
                                 annotation_font_color="#e67e22")
                fig_hv.update_layout(
                    template="purdue", height=340,
                    xaxis=dict(type="date"),
                    yaxis=dict(ticksuffix="%"),
                    margin=dict(l=44, r=20, t=20, b=60),
                )
                _chart(fig_hv)
                _insight_note(
                    "How volatile each commodity has been over the rolling 24-hour window, "
                    "expressed as an annualised rate. Readings above 40% indicate elevated "
                    "intraday stress - energy tends to spike here first during geopolitical "
                    "events, often before daily-bar charts show any movement."
                )

    _thread(
        "Intraday patterns reveal short-term stress. The daily charts below zoom out to reveal whether "
        "that stress is noise or part of a sustained directional move."
    )

    st.markdown('<div style="margin:0.5rem 0;border-top:1px solid #1a1a1a"></div>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 3: Daily price indexed (wider) | Daily rolling 30d vol (narrower)
    # ══════════════════════════════════════════════════════════════════════
    col_dp, col_dv = st.columns([1.6, 1], gap="medium")

    with col_dp:
        _label("Daily Indexed Price: Base = 100 at Start Date")
        if d_selected:
            prices_sel = cmd_p[d_selected].dropna(how="all")
            if not prices_sel.empty:
                normed = (prices_sel / prices_sel.iloc[0]) * 100
                fig_norm = go.Figure()
                eq_i = cmd_i = 0
                for col in normed.columns:
                    ls = _line_style(col, eq_i, cmd_i)
                    if col in _EQUITY_REGIONS: eq_i += 1
                    else: cmd_i += 1
                    fig_norm.add_trace(go.Scatter(
                        x=normed.index, y=normed[col], name=col, line=ls,
                    ))
                for ev in GEOPOLITICAL_EVENTS[-5:]:
                    fig_norm.add_vrect(
                        x0=str(ev["start"]), x1=str(ev["end"]),
                        fillcolor=ev["color"], opacity=0.06,
                        layer="below", line_width=0,
                    )
                fig_norm.add_hline(y=100, line=dict(color="#ABABAB", width=1, dash="dot"))
                _chart(_style_fig(fig_norm, height=380))
                _insight_note(
                    "Long-run performance on a level playing field - all assets start at 100 "
                    "on the analysis start date. Annotated shading marks major geopolitical "
                    "events; look for price dislocations that coincide with conflict or "
                    "sanction dates. Persistent deviation from 100 indicates a structural "
                    "shift in supply/demand rather than temporary noise."
                )

    with col_dv:
        _label("Rolling 30d Annualised Volatility (%)")
        if d_selected and not cmd_r.empty:
            fig_vol = go.Figure()
            cmd_iv = 0
            for col in d_selected:
                if col in cmd_r.columns:
                    rv = cmd_r[col].rolling(30).std() * np.sqrt(252) * 100
                    ls = _line_style(col, 0, cmd_iv); cmd_iv += 1
                    fig_vol.add_trace(go.Scatter(
                        x=rv.index, y=rv.values, name=col, line=ls,
                    ))
            for ev in GEOPOLITICAL_EVENTS[-5:]:
                fig_vol.add_vrect(
                    x0=str(ev["start"]), x1=str(ev["end"]),
                    fillcolor=ev["color"], opacity=0.06, layer="below", line_width=0,
                )
            fig_vol.add_hline(y=35, line=dict(color="#e67e22", width=1, dash="dot"),
                              annotation_text="Elevated (35%)", annotation_font_size=8,
                              annotation_font_color="#e67e22")
            fig_vol.update_layout(yaxis=dict(ticksuffix="%"))
            _chart(_style_fig(fig_vol, height=380))
            _insight_note(
                "30-day realised volatility measures how much price risk is present "
                "right now. Sustained readings above 35% signal persistent market "
                "stress rather than a one-day spike - and historically precede trend "
                "reversals as investors adjust positions."
            )

    st.markdown('<div style="margin:0.5rem 0;border-top:1px solid #1a1a1a"></div>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 4: Commodity cards (2-column grid)
    # ══════════════════════════════════════════════════════════════════════
    _label("Commodity Snapshot Cards")
    _h_vol: dict[str, float] = {}
    if not h_cmd_r.empty:
        for n in [nm for (_, nm, _, _) in WATCHLIST]:
            if n in h_cmd_r.columns:
                s = h_cmd_r[n].dropna()
                _h_vol[n] = s.iloc[-24:].std() * _HOURLY_ANNUAL * 100 if len(s) >= 24 else 0.0

    all_selected = list(dict.fromkeys((h_selected or []) + (d_selected or [])))
    card_names = [n for (_, n, _, _) in WATCHLIST
                  if n in cmd_p.columns and (not all_selected or n in all_selected)]

    for row_start in range(0, len(card_names), 2):
        row_names = card_names[row_start:row_start + 2]
        card_cols = st.columns(len(row_names), gap="medium")
        for ccol, name in zip(card_cols, row_names):
            with ccol:
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
                d1_color    = "#2e7d32" if d1_pct  >= 0 else "#c0392b"
                ytd_color   = "#2e7d32" if ytd_pct >= 0 else "#c0392b"

                vol_24h_html = (
                    f'<div style="font-size:0.68rem;color:#333333">'
                    f'24h Vol&nbsp;<span style="font-weight:600;color:{regime_col}">'
                    f'{vol_24h:.1f}%</span></div>'
                ) if vol_24h > 0 else ""

                st.markdown(
                    f'<div style="border:1px solid #1e1e1e;border-left:2px solid rgba(207,185,145,0.22);'
                    f'border-radius:0;padding:0.75rem 1rem;margin-bottom:0.6rem;background:#141414">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                    f'<div><span style="font-size:0.52rem;text-transform:uppercase;'
                    f'letter-spacing:0.12em;color:#8890a1;font-weight:600">{group}</span>'
                    f'<div style="font-size:0.84rem;font-weight:700;color:#e8e9ed;'
                    f'font-family:\'DM Sans\',sans-serif;margin-top:1px">{name}</div></div>'
                    f'<div style="text-align:right">'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1.0rem;'
                    f'font-weight:700;color:#CFB991">{last_price:,.2f}</div>'
                    f'<div style="font-size:0.62rem;color:{d1_color};'
                    f'font-family:\'JetBrains Mono\',monospace;font-weight:600">{d1_pct:+.2f}% 1d</div>'
                    f'</div></div>'
                    f'<div style="display:flex;gap:1.1rem;margin-top:0.45rem;flex-wrap:wrap">'
                    f'<div style="font-size:0.66rem;color:#8890a1">'
                    f'YTD&nbsp;<span style="color:{ytd_color};font-weight:600">{ytd_pct:+.1f}%</span></div>'
                    f'<div style="font-size:0.66rem;color:#8890a1">'
                    f'30d Vol&nbsp;<span style="color:{regime_col};font-weight:600">'
                    f'{vol_30d:.1f}% <span style="font-size:0.58rem">({regime_lbl})</span></span></div>'
                    f'{vol_24h_html}</div>'
                    f'<div style="margin-top:0.45rem;padding-top:0.4rem;border-top:1px solid #1e1e1e">'
                    f'<span style="font-size:0.62rem;color:#8890a1">&#9872;&nbsp;{alert_ctx}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div style="margin:0.5rem 0;border-top:1px solid #1a1a1a"></div>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 5: Hourly correlation vs S&P 500 | Daily correlation vs S&P 500
    # ══════════════════════════════════════════════════════════════════════
    from src.data.loader import load_equity_prices
    eq_p_full = load_equity_prices(start, end)
    from src.data.loader import load_returns as _lr
    eq_r2, _ = _lr(start, end)
    sp_r_daily = eq_r2.get("S&P 500") if "S&P 500" in eq_r2.columns else None

    has_hourly_corr = "S&P 500" in h_eq_r.columns and not h_cmd_r.empty
    has_daily_corr  = sp_r_daily is not None and not cmd_r.empty and d_selected

    if has_hourly_corr or has_daily_corr:
        col_hcorr, col_dcorr = st.columns(2, gap="medium")

        if has_hourly_corr:
            with col_hcorr:
                _label("Rolling 24h Hourly Correlation vs S&P 500")
                h_corr_names = [n for (_, n, _, _) in WATCHLIST if n in h_cmd_r.columns]
                corr_sel = st.multiselect(
                    "Commodities (hourly correlation)",
                    h_corr_names, default=h_corr_names[:4], key="wl_h_corr_sel",
                )
                sp_h = h_eq_r["S&P 500"]
                cutoff_corr = pd.Timestamp.now() - pd.Timedelta(days=90)
                if corr_sel:
                    fig_hcorr = go.Figure()
                    for i, name in enumerate(corr_sel):
                        fig_hcorr.add_trace(go.Scatter(
                            x=rolling_correlation(sp_h, h_cmd_r[name], 24).loc[cutoff_corr:].index,
                            y=rolling_correlation(sp_h, h_cmd_r[name], 24).loc[cutoff_corr:].values,
                            name=name,
                            line=_line_style(name, 0, i),
                        ))
                    fig_hcorr.add_hline(y=0, line=dict(color="#ABABAB", width=1, dash="dot"))
                    fig_hcorr.update_layout(
                        template="purdue", height=340,
                        xaxis=dict(type="date"),
                        yaxis=dict(range=[-1, 1]),
                        margin=dict(l=44, r=20, t=20, b=60),
                    )
                    _chart(fig_hcorr)
                    _insight_note(
                        "How closely each commodity co-moved with US equities on an intraday "
                        "basis over the past 90 days. Near-zero or negative readings indicate "
                        "diversification - the commodity moves independently of stocks. "
                        "Spikes toward +1 are contagion alerts: everything is moving together, "
                        "which typically signals broad-based panic selling."
                    )

        if has_daily_corr:
            with col_dcorr:
                _label("Rolling 60d Daily Correlation vs S&P 500")
                fig_spx = go.Figure()
                for i, name in enumerate(d_selected):
                    if name in cmd_r.columns:
                        rc = rolling_correlation(sp_r_daily, cmd_r[name], 60)
                        fig_spx.add_trace(go.Scatter(
                            x=rc.index, y=rc.values, name=name,
                            line=_line_style(name, 0, i),
                        ))
                fig_spx.add_hline(y=0, line=dict(color="#ABABAB", width=1, dash="dot"))
                for ev in GEOPOLITICAL_EVENTS:
                    fig_spx.add_vrect(
                        x0=str(ev["start"]), x1=str(ev["end"]),
                        fillcolor=ev["color"], opacity=0.05, layer="below", line_width=0,
                    )
                fig_spx.update_layout(yaxis=dict(range=[-1, 1]))
                _chart(_style_fig(fig_spx, height=340))
                _insight_note(
                    "Long-run 60-day rolling correlation with the S&P 500. Gold characteristically "
                    "turns negative during risk-off periods as capital flees to safety - its "
                    "correlation is a reliable crisis barometer. Copper and oil tend to track "
                    "equities during growth cycles (both move on demand expectations) but "
                    "decouple sharply when supply shocks dominate."
                )

        st.markdown('<div style="margin:0.5rem 0;border-top:1px solid #1a1a1a"></div>',
                    unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 6: COT extremes table | COT overlay chart
    # ══════════════════════════════════════════════════════════════════════
    _thread(
        "Price charts show what has happened. The CFTC Commitments of Traders data below shows what "
        "large speculative traders expect to happen next - and when their positioning becomes extreme, "
        "it often marks the turning point."
    )
    _label("CFTC Commitments of Traders: Speculative Positioning")
    _definition_block(
        "COT Signal Logic",
        "The CFTC publishes weekly data on how non-commercial (speculative) traders are positioned "
        "in commodity futures. <b>Net Spec % OI</b> = (Long − Short) ÷ Open Interest × 100. "
        "Readings above <b>+25%</b> signal a crowded long (contrarian sell signal); "
        "below <b>−25%</b> signal a crowded short (contrarian buy signal). "
        "These extremes have historically preceded price reversals of 10–20% within 4–8 weeks.",
    )

    with st.spinner("Loading COT data from CFTC…"):
        cot_df = load_cot_data(years=3)

    if cot_df.empty:
        st.warning("COT data unavailable. Check internet connectivity to www.cftc.gov.")
    else:
        col_cot_l, col_cot_r = st.columns([1, 1.6], gap="medium")

        with col_cot_l:
            _label("Current Positioning Extremes")
            ext_tbl = cot_extremes_table(cot_df)
            if not ext_tbl.empty:
                _TBL_CSS_COT = """
<style>
.ec-table{width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:0.78rem}
.ec-table th{background:#080808;color:#CFB991;padding:7px 10px;text-align:left;
    border-bottom:1px solid rgba(207,185,145,0.3);font-weight:600;
    letter-spacing:0.06em;text-transform:uppercase;font-size:0.68rem}
.ec-table td{padding:5px 10px;border-bottom:1px solid #1e1e1e;color:#e8e9ed}
.ec-table tr:nth-child(even) td{background:#171717}
.ec-table tr:nth-child(odd) td{background:#111111}
.ec-table tr:hover td{background:#202020}
</style>"""
                cot_cols = list(ext_tbl.columns)
                cot_header = "".join(f"<th>{c}</th>" for c in cot_cols)
                cot_rows = ""
                for _, row in ext_tbl.iterrows():
                    cells = ""
                    for col in cot_cols:
                        v = row[col]
                        if col == "Signal":
                            if "Crowded Long" in str(v):
                                cells += f"<td style='color:#f87171;font-weight:700'>{v}</td>"
                            elif "Crowded Short" in str(v):
                                cells += f"<td style='color:#4ade80;font-weight:700'>{v}</td>"
                            else:
                                cells += f"<td style='color:#8890a1'>{v}</td>"
                        else:
                            is_nan = pd.isna(v) if not isinstance(v, str) else False
                            if is_nan:
                                cells += "<td style='color:#8890a1'>-</td>"
                            else:
                                cells += f"<td style='color:#b8b8b8'>{v}</td>"
                    cot_rows += f"<tr>{cells}</tr>"
                html_cot = (
                    _TBL_CSS_COT
                    + "<table class='ec-table'>"
                    + f"<thead><tr>{cot_header}</tr></thead>"
                    + f"<tbody>{cot_rows}</tbody>"
                    + "</table>"
                )
                st.markdown(html_cot, unsafe_allow_html=True)
                _insight_note(
                    "Flags when speculative traders are all-in on one side of a commodity. "
                    "A 'Crowded Long' means too many people are betting on price rises - "
                    "historically, these positions unwind and the price falls. "
                    "A 'Crowded Short' is the opposite: a contrarian buy signal."
                )

        with col_cot_r:
            _label("Positioning Detail + Price Overlay")
            cot_markets = sorted(cot_df["market"].unique().tolist())
            cot_sel = st.selectbox(
                "Select commodity",
                cot_markets,
                index=cot_markets.index("WTI Crude Oil") if "WTI Crude Oil" in cot_markets else 0,
                key="wl_cot_sel",
            )
            price_s = cmd_p[cot_sel].dropna() if cot_sel in cmd_p.columns else None
            _chart(plot_cot_overlay(cot_df, cot_sel, price_s, height=360))
            _insight_note(
                "Top panel: price history. Bottom panel: net speculative positioning as a "
                "percentage of open interest. Readings crossing ±25% (dashed lines) "
                "have historically been reliable reversal signals over a 4–8 week horizon. "
                "Extreme speculative crowding is a classic contrarian indicator."
            )

    _section_note(
        "Daily prices sourced from Yahoo Finance. Daily volatility: σ × √252. "
        "Hourly volatility: σ_hourly × √(252 × 23) - commodity futures trade approximately "
        "23 hours per day on CME Globex. Hourly data lookback capped at 730 days by yfinance. "
        "COT data from CFTC disaggregated futures-only reports, updated weekly (Tuesday releases)."
    )
    _page_conclusion(
        "Commodity Positioning Summary",
        "Commodities with elevated post-event volatility, a sustained price trend, and a crowded "
        "speculative positioning reading (above 85th percentile long or short) are the highest-conviction "
        "setups for a reversal. Cross-reference with the Spillover page to confirm the commodity is also "
        "a price transmitter into equity markets.",
    )
    # ── AI Commodities Specialist ──────────────────────────────────────────
    try:
        from src.agents.commodities_specialist import run as _cs_run
        from src.ui.agent_panel import render_agent_output_block
        from src.analysis.agent_state import is_enabled

        if is_enabled("commodities_specialist"):
            _anthropic_key = _openai_key = ""
            try:
                _keys = st.secrets.get("keys", {})
                _anthropic_key = _keys.get("anthropic_api_key", "") or ""
                _openai_key    = _keys.get("openai_api_key",    "") or ""
            except Exception:
                pass
            _provider = "anthropic" if _anthropic_key else ("openai" if _openai_key else None)
            _api_key  = _anthropic_key or _openai_key

            # Build context
            _cs_ctx: dict = {}
            try:
                if not cmd_r.empty and len(cmd_r) >= 5:
                    _w5 = cmd_r.iloc[-5:].sum() * 100
                    _top3  = sorted(zip(_w5.index, _w5.values), key=lambda x: -x[1])[:3]
                    _bot3  = sorted(zip(_w5.index, _w5.values), key=lambda x: x[1])[:3]
                    _cs_ctx["top_performers"]   = [(n, v) for n, v in _top3 if v > 0]
                    _cs_ctx["worst_performers"]  = [(n, v) for n, v in _bot3 if v < 0]
                from src.analysis.correlations import average_cross_corr_series
                from src.data.loader import load_returns as _lr2
                _eq_r2, _ = _lr2(start, end)
                if not _eq_r2.empty and not cmd_r.empty:
                    _avg_c = average_cross_corr_series(_eq_r2, cmd_r, window=60)
                    if not _avg_c.empty:
                        _cs_ctx["avg_corr"] = float(_avg_c.iloc[-1])
                # COT crowded positions
                if "cot_df" in dir() and cot_df is not None and not cot_df.empty:
                    _longs  = [(m, float(cot_df[cot_df["market"]==m]["net_spec_pct"].iloc[-1]))
                               for m in cot_df["market"].unique()
                               if not cot_df[cot_df["market"]==m].empty
                               and cot_df[cot_df["market"]==m]["net_spec_pct"].iloc[-1] >= 25]
                    _shorts = [(m, float(cot_df[cot_df["market"]==m]["net_spec_pct"].iloc[-1]))
                               for m in cot_df["market"].unique()
                               if not cot_df[cot_df["market"]==m].empty
                               and cot_df[cot_df["market"]==m]["net_spec_pct"].iloc[-1] <= -25]
                    if _longs:
                        _cs_ctx["crowded_longs"] = _longs
                    if _shorts:
                        _cs_ctx["crowded_shorts"] = _shorts
            except Exception:
                pass

            with st.spinner("AI Commodities Specialist analysing…"):
                _cs_result = _cs_run(_cs_ctx, _provider, _api_key)

            if _cs_result.get("narrative"):
                st.markdown("---")
                render_agent_output_block("commodities_specialist", _cs_result)
    except Exception:
        pass

    _page_footer()

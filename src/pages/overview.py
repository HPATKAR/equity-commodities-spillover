"""
Page 1 - Overview
KPIs, equity-commodities correlation heatmap snapshot, regime status, recent events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_returns, load_all_prices, load_fixed_income_returns, load_fx_returns
from src.data.config import GEOPOLITICAL_EVENTS, EQUITY_REGIONS, COMMODITY_GROUPS, PALETTE
from src.analysis.correlations import (
    cross_asset_corr, average_cross_corr_series, detect_correlation_regime,
    regime_transition_matrix, early_warning_signals,
)
from src.analysis.risk_score import (
    compute_risk_score, risk_score_history, plot_risk_gauge, plot_risk_history,
)
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _thread, _section_note,
    _definition_block, _takeaway_block, _page_conclusion,
    _page_footer, _add_event_bands, _insight_note,
)

_F = "font-family:'DM Sans',sans-serif;"


def _label(txt: str) -> None:
    st.markdown(
        f'<p style="{_F}font-size:0.58rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.14em;color:#8E6F3E;margin:0 0 6px 0">{txt}</p>',
        unsafe_allow_html=True,
    )


def page_overview(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.1rem">Equity-Commodities Spillover Monitor</h1>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#8890a1;'
        'margin:0 0 0.8rem 0">15 equity indices · 17 commodity futures · '
        'Correlation regimes · Geopolitical risk · Spillover signals</p>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "The central research question of this dashboard: <strong>do equity market shocks spill into "
        "commodities - and in which direction?</strong> This page is your live answer. The regime badge "
        "tells you whether equity-commodity co-movement is currently amplifying or absorbing risk. "
        "The KPIs quantify how tight the spillover channel is right now. The heatmap shows which "
        "specific equity-commodity pairs are most coupled. Start here before reading any other page."
    )

    with st.spinner("Loading market data…"):
        eq_r, cmd_r = load_returns(start, end)

    if eq_r.empty or cmd_r.empty:
        st.error("Could not load market data. Check your internet connection.")
        return

    # ── Regime & KPIs ──────────────────────────────────────────────────────
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

    recent_eq  = eq_r.iloc[-22:].sum()
    recent_cmd = cmd_r.iloc[-22:].sum()
    best_eq    = recent_eq.idxmax();  worst_eq  = recent_eq.idxmin()
    best_cmd   = recent_cmd.idxmax(); worst_cmd = recent_cmd.idxmin()

    # KPI strip
    k1, k2, k3, k4, k5 = st.columns(5)

    def _kpi(col, label, value, delta="", dcolor=""):
        col.markdown(
            f'<div style="border:1px solid #E8E5E0;border-radius:4px;'
            f'padding:0.55rem 0.75rem;background:#1a1d27">'
            f'<div style="{_F}font-size:0.58rem;font-weight:600;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:#8890a1;margin-bottom:3px">{label}</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.98rem;'
            f'font-weight:700;color:#e8e9ed;line-height:1.2">{value}</div>'
            + (f'<div style="{_F}font-size:0.62rem;color:{dcolor};margin-top:2px">{delta}</div>' if delta else "")
            + '</div>',
            unsafe_allow_html=True,
        )

    k1.markdown(
        f'<div style="border:1px solid #E8E5E0;border-radius:4px;'
        f'padding:0.55rem 0.75rem;background:#1a1d27;border-left:3px solid {regime_color}">'
        f'<div style="{_F}font-size:0.58rem;font-weight:600;letter-spacing:0.14em;'
        f'text-transform:uppercase;color:#8890a1;margin-bottom:3px">Correlation Regime</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.98rem;'
        f'font-weight:700;color:{regime_color}">{regime_name}</div></div>',
        unsafe_allow_html=True,
    )
    _kpi(k2, "Avg Cross-Asset Corr (60d)", f"{current_avg_corr:.3f}",
         f"{corr_delta:+.3f} vs 1M ago", "#2e7d32" if corr_delta < 0 else "#c0392b")
    _kpi(k3, "Best Equity (1M)", best_eq,
         f"{recent_eq[best_eq]*100:+.1f}%", "#2e7d32")
    _kpi(k4, "Worst Equity (1M)", worst_eq,
         f"{recent_eq[worst_eq]*100:+.1f}%", "#c0392b")
    _kpi(k5, "Best Commodity (1M)", best_cmd,
         f"{recent_cmd[best_cmd]*100:+.1f}%", "#2e7d32")

    # ── Load FI and FX data ────────────────────────────────────────────────
    try:
        fi_r = load_fixed_income_returns(start, end)
    except Exception:
        fi_r = __import__("pandas").DataFrame()
    try:
        fx_r = load_fx_returns(start, end)
    except Exception:
        fx_r = __import__("pandas").DataFrame()

    # ── FI / FX KPI strip ────────────────────────────────────────────────
    st.markdown(
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:0.52rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.14em;color:#8E6F3E;margin:0.7rem 0 0.3rem">'
        f'Fixed Income & FX Context</p>',
        unsafe_allow_html=True,
    )

    def _dark_kpi(col, label, value, delta="", delta_up=None):
        if delta_up is True:
            d_color = "#2e7d32"
        elif delta_up is False:
            d_color = "#c0392b"
        else:
            d_color = "#8890a1"
        col.markdown(
            f'<div style="border:1px solid #E8E5E0;border-radius:4px;'
            f'padding:0.55rem 0.75rem;background:#1a1d27">'
            f'<div style="{_F}font-size:0.58rem;font-weight:600;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:#8890a1;margin-bottom:3px">{label}</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.98rem;'
            f'font-weight:700;color:#e8e9ed;line-height:1.2">{value}</div>'
            + (f'<div style="{_F}font-size:0.62rem;color:{d_color};margin-top:2px">{delta}</div>' if delta else "")
            + '</div>',
            unsafe_allow_html=True,
        )

    _fi_k1, _fi_k2, _fi_k3, _fi_k4, _fi_k5 = st.columns(5)

    # TLT 30d return
    _tlt_30d = None
    try:
        if not fi_r.empty and "US 20Y+ Treasury (TLT)" in fi_r.columns:
            _tlt = fi_r["US 20Y+ Treasury (TLT)"].dropna()
            if len(_tlt) >= 21:
                _tlt_30d = float(_tlt.iloc[-21:].sum() * 100)
    except Exception:
        pass
    _dark_kpi(_fi_k1, "TLT 30d Return", f"{_tlt_30d:+.1f}%" if _tlt_30d is not None else "-",
              delta="Long duration" if _tlt_30d and _tlt_30d > 0 else "Duration pressure" if _tlt_30d is not None else "",
              delta_up=True if _tlt_30d and _tlt_30d > 0 else False if _tlt_30d is not None else None)

    # HYG vs LQD spread (credit stress)
    _hyg_30d = _lqd_30d = None
    try:
        if not fi_r.empty:
            if "HY Corporate (HYG)" in fi_r.columns:
                _s = fi_r["HY Corporate (HYG)"].dropna()
                if len(_s) >= 21:
                    _hyg_30d = float(_s.iloc[-21:].sum() * 100)
            if "IG Corporate (LQD)" in fi_r.columns:
                _s = fi_r["IG Corporate (LQD)"].dropna()
                if len(_s) >= 21:
                    _lqd_30d = float(_s.iloc[-21:].sum() * 100)
    except Exception:
        pass
    _credit_spread_signal = None
    if _hyg_30d is not None and _lqd_30d is not None:
        _credit_spread_signal = float(_hyg_30d - _lqd_30d)
    _dark_kpi(_fi_k2, "HYG vs LQD (30d alpha)",
              f"{_credit_spread_signal:+.1f}%" if _credit_spread_signal is not None else "-",
              delta="HY outperforming" if _credit_spread_signal is not None and _credit_spread_signal > 0 else "HY stress signal" if _credit_spread_signal is not None else "",
              delta_up=True if _credit_spread_signal is not None and _credit_spread_signal > 0 else False if _credit_spread_signal is not None else None)

    # DXY trend
    _dxy_30d = None
    try:
        if not fx_r.empty and "DXY (Dollar Index)" in fx_r.columns:
            _dxy = fx_r["DXY (Dollar Index)"].dropna()
            if len(_dxy) >= 21:
                _dxy_30d = float(_dxy.iloc[-21:].sum() * 100)
    except Exception:
        pass
    _dark_kpi(_fi_k3, "DXY 30d Return", f"{_dxy_30d:+.1f}%" if _dxy_30d is not None else "-",
              delta="Dollar strengthening" if _dxy_30d is not None and _dxy_30d > 0 else "Dollar weakening" if _dxy_30d is not None else "",
              delta_up=None)

    # TLT vs SPY divergence (equity-bond correlation signal)
    _spx_30d = None
    try:
        _spx_col = next((c for c in ["S&P 500"] if c in eq_r.columns), None)
        if _spx_col:
            _spx_s = eq_r[_spx_col].dropna()
            if len(_spx_s) >= 21:
                _spx_30d = float(_spx_s.iloc[-21:].sum() * 100)
    except Exception:
        pass
    _tlt_spx_div = None
    if _tlt_30d is not None and _spx_30d is not None:
        _tlt_spx_div = _tlt_30d - _spx_30d
    _dark_kpi(_fi_k4, "TLT vs SPX Divergence",
              f"{_tlt_spx_div:+.1f}pp" if _tlt_spx_div is not None else "-",
              delta="Bonds/equities decoupling" if _tlt_spx_div is not None and abs(_tlt_spx_div) > 5 else "Normal co-movement" if _tlt_spx_div is not None else "",
              delta_up=None)

    # EMB 30d return
    _emb_30d = None
    try:
        if not fi_r.empty and "EM USD Bonds (EMB)" in fi_r.columns:
            _emb = fi_r["EM USD Bonds (EMB)"].dropna()
            if len(_emb) >= 21:
                _emb_30d = float(_emb.iloc[-21:].sum() * 100)
    except Exception:
        pass
    _dark_kpi(_fi_k5, "EMB 30d Return",
              f"{_emb_30d:+.1f}%" if _emb_30d is not None else "-",
              delta="EM credit bid" if _emb_30d is not None and _emb_30d > 0 else "EM credit stress" if _emb_30d is not None else "",
              delta_up=True if _emb_30d is not None and _emb_30d > 0 else False if _emb_30d is not None else None)

    st.markdown('<div style="margin:0.7rem 0 0.5rem;border-top:1px solid #2a2d3a"></div>',
                unsafe_allow_html=True)
    _thread(
        "The numbers above give the level of stress. The section below breaks down its source - "
        "which geopolitical conflicts are active and how severely each is scoring."
    )

    # ── ROW: Risk gauge | Risk history ─────────────────────────────────────
    with st.spinner("Computing risk score…"):
        risk_result = compute_risk_score(avg_corr, cmd_r)
        score_hist  = risk_score_history(avg_corr, cmd_r, eq_r=eq_r)

    _r_colors = {0: "#2e7d32", 1: "#555960", 2: "#e67e22", 3: "#c0392b"}

    gc1, gc2 = st.columns([1, 2.2])
    with gc1:
        _label("Geopolitical Risk Score")
        _chart(plot_risk_gauge(risk_result, height=220))
        _insight_note(
            "Displays the current market stress level on a gauge from 0 (calm) to 100 (crisis). "
            "Readings above 60 indicate equity-commodities turbulence is building. "
            "This is the single most important number on the dashboard."
        )
        comp = risk_result["components"]
        for name, val in comp.items():
            col_c = "#c0392b" if val > 70 else "#e67e22" if val > 45 else "#2e7d32"
            pct = min(val, 100)
            st.markdown(
                f'<div style="margin-bottom:5px">'
                f'<div style="display:flex;justify-content:space-between;{_F}font-size:0.66rem;margin-bottom:2px">'
                f'<span style="color:#b8bec8">{name}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-weight:700;color:{col_c}">{val:.0f}</span>'
                f'</div>'
                f'<div style="height:3px;background:#2a2d3a;border-radius:2px">'
                f'<div style="width:{pct:.0f}%;height:3px;background:{col_c};border-radius:2px"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── Dynamic weights today ─────────────────────────────────────────
        dw = risk_result.get("weights", {})
        if dw:
            _key_map = {"corr": "Correlation", "vol": "Cmd Vol",
                        "vix": "VIX", "og": "Oil-Gold", "events": "Events"}
            wt_rows = "".join(
                f'<td style="text-align:center;padding:2px 4px">'
                f'<div style="font-size:0.60rem;color:#6b7280">{_key_map.get(k,k)}</div>'
                f'<div style="font-family:JetBrains Mono,monospace;font-weight:700;'
                f'font-size:0.68rem;color:#8E6F3E">{v*100:.0f}%</div>'
                f'</td>'
                for k, v in dw.items()
            )
            st.markdown(
                f'<div style="margin-top:8px;{_F}font-size:0.57rem;color:#8E6F3E;'
                f'text-transform:uppercase;letter-spacing:.12em;font-weight:700;'
                f'margin-bottom:3px">Today\'s dynamic weights</div>'
                f'<table style="width:100%;border-collapse:collapse"><tr>{wt_rows}</tr></table>',
                unsafe_allow_html=True,
            )

        # ── Oil-Gold geo signal detail ─────────────────────────────────────
        og = risk_result.get("oil_gold", {})
        if og:
            g_z, o_z = og.get("gold_z", 0), og.get("oil_z", 0)
            g_r, o_r = og.get("gold_ret", 0), og.get("oil_ret", 0)
            st.markdown(
                f'<div style="margin-top:8px;padding:6px 8px;background:#1a1d27;'
                f'border-left:2px solid #CFB991;{_F}font-size:0.63rem;color:#b8bec8;line-height:1.6">'
                f'<b style="color:#8E6F3E;text-transform:uppercase;font-size:0.57rem;'
                f'letter-spacing:.12em">Oil-Gold Signal</b><br>'
                f'Gold 20d: <b>{g_r:+.1f}%</b> (z={g_z:+.2f}) &nbsp;·&nbsp; '
                f'Oil 20d: <b>{o_r:+.1f}%</b> (z={o_z:+.2f})'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Top active/tail events ─────────────────────────────────────────
        evs = risk_result.get("events", [])[:3]
        if evs:
            rows = "".join(
                f'<tr><td style="padding:1px 6px 1px 0;color:#8890a1">{e["label"]}</td>'
                f'<td style="color:#6b7280;font-size:0.60rem">{e["status"]}</td>'
                f'<td style="font-family:JetBrains Mono,monospace;font-weight:700;'
                f'color:#c0392b;text-align:right">{e["score"]:.0f}</td></tr>'
                for e in evs
            )
            st.markdown(
                f'<div style="margin-top:6px;{_F}font-size:0.63rem">'
                f'<b style="color:#8E6F3E;text-transform:uppercase;font-size:0.57rem;'
                f'letter-spacing:.12em">Event Severity Drivers</b>'
                f'<table style="width:100%;margin-top:3px;border-collapse:collapse">{rows}</table>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with gc2:
        _label("Risk Score History")
        if not score_hist.empty:
            _chart(plot_risk_history(score_hist, height=310))
            _insight_note(
                "Historical trace of the composite stress index over the full analysis period. "
                "Peaks align with known crises - COVID (March 2020), Ukraine invasion (Feb 2022), and banking stress (Mar 2023). "
                "Sustained readings above 60 have historically preceded equity drawdowns of 10% or more."
            )

    st.markdown('<div style="margin:0.6rem 0;border-top:1px solid #2a2d3a"></div>',
                unsafe_allow_html=True)
    _thread(
        "Knowing the score is useful; knowing whether it is triggering actionable signals is more "
        "useful. The early warning system below converts the raw score into discrete alerts."
    )

    # ── ROW: EWS composite (left) | 5 signal cards (right) ────────────────
    with st.spinner("Computing early warning signals…"):
        trans_matrix = regime_transition_matrix(regimes)
        ews = early_warning_signals(avg_corr, cmd_r, eq_r, regimes, trans_matrix)

    def _ews_score_color(s: float) -> str:
        if s >= 70: return "#c0392b"
        if s >= 40: return "#e67e22"
        return "#2e7d32"

    comp_ews   = ews["composite"]
    comp_color = _ews_score_color(comp_ews)
    comp_label = (
        "Imminent shift risk" if comp_ews >= 70
        else "Monitor closely" if comp_ews >= 55
        else "Signals mixed"   if comp_ews >= 40
        else "Benign trajectory"
    )

    ews_l, ews_r = st.columns([1, 4])
    with ews_l:
        _label("Early Warning System")
        st.markdown(
            f'<div style="border:1px solid #E8E5E0;border-radius:4px;padding:0.75rem;'
            f'background:#1a1d27;border-top:3px solid {comp_color}">'
            f'<div style="{_F}font-size:0.56rem;font-weight:700;letter-spacing:0.12em;'
            f'text-transform:uppercase;color:#6b7280;margin-bottom:4px">Composite Score</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:2.2rem;'
            f'font-weight:700;color:{comp_color};line-height:1">{comp_ews:.0f}'
            f'<span style="font-size:0.75rem;color:#6b7280">/100</span></div>'
            f'<div style="background:#2a2d3a;border-radius:3px;height:6px;margin:6px 0">'
            f'<div style="width:{comp_ews:.0f}%;background:{comp_color};height:6px;border-radius:3px"></div>'
            f'</div>'
            f'<div style="{_F}font-size:0.65rem;font-weight:600;color:{comp_color}">'
            f'{comp_label}</div></div>',
            unsafe_allow_html=True,
        )
    with ews_r:
        _label("Signal Components")
        sig_items = list(ews["signals"].items())
        sig_cols  = st.columns(5)
        for col, (name, data) in zip(sig_cols, sig_items):
            s = data["score"]; c = _ews_score_color(s)
            col.markdown(
                f'<div style="border:1px solid #E8E5E0;border-radius:4px;padding:0.6rem 0.55rem;'
                f'background:#1a1d27;border-top:2px solid {c}">'
                f'<div style="{_F}font-size:0.54rem;font-weight:700;letter-spacing:0.10em;'
                f'text-transform:uppercase;color:#6b7280;margin-bottom:3px">{name}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1.1rem;'
                f'font-weight:700;color:{c}">{s:.0f}'
                f'<span style="{_F}font-size:0.60rem;color:#6b7280">/100</span></div>'
                f'<div style="background:#2a2d3a;border-radius:2px;height:3px;margin:4px 0">'
                f'<div style="width:{s:.0f}%;background:{c};height:3px;border-radius:2px"></div>'
                f'</div>'
                f'<div style="{_F}font-size:0.64rem;color:#8890a1;line-height:1.45;margin-top:3px">'
                f'{data["desc"]}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="margin:0.6rem 0;border-top:1px solid #2a2d3a"></div>',
                unsafe_allow_html=True)
    _thread(
        "Alerts tell you something is wrong. The correlation timeline below tells you how that "
        "stress is transmitting - when equities and commodities start moving together, a shock "
        "in one will hit the other."
    )

    # ── ROW: Correlation timeline | Analogues table ───────────────────────
    ct_col, an_col = st.columns([1.6, 1])

    with ct_col:
        _label("Average Equity-Commodity Correlation: 60d Rolling")
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(
            x=avg_corr.index, y=avg_corr.values,
            name="Avg |Corr|",
            line=dict(color=PALETTE[1], width=1.6),
            fill="tozeroy", fillcolor="rgba(207,185,145,0.12)",
        ))
        fig_corr.add_hline(y=0.15, line=dict(color="#2e7d32", width=1, dash="dot"),
                           annotation_text="Low", annotation_font_size=8)
        fig_corr.add_hline(y=0.45, line=dict(color="#c0392b", width=1, dash="dot"),
                           annotation_text="High", annotation_font_size=8)
        _add_event_bands(fig_corr)
        _chart(_style_fig(fig_corr, height=280))
        _insight_note(
            "Shows the rolling pairwise correlation between every equity and commodity in the dashboard. "
            "Dark warm colours mean assets are moving together (risk-on or systemic stress). "
            "Dark cool colours mean they are moving in opposite directions (safe-haven flows or diverging fundamentals)."
        )
        _thread(
            "Correlation shows the mechanism. The performance chart below shows the outcome - "
            "which assets have actually moved and by how much."
        )

        # ── 1-Month Performance Bar Chart ─────────────────────────────────
        _label("1-Month Asset Performance: Equities & Commodities")
        perf_all = pd.concat([
            recent_eq.rename("return") * 100,
            recent_cmd.rename("return") * 100,
        ]).sort_values()
        bar_cols = ["#c0392b" if v < 0 else "#2e7d32" for v in perf_all.values]
        fig_perf = go.Figure(go.Bar(
            x=perf_all.values,
            y=perf_all.index,
            orientation="h",
            marker_color=bar_cols,
            marker_line_width=0,
            hovertemplate="%{y}: %{x:+.1f}%<extra></extra>",
        ))
        fig_perf.add_vline(x=0, line=dict(color="#ABABAB", width=1))
        fig_perf.update_layout(
            template="purdue",
            height=max(260, len(perf_all) * 16),
            margin=dict(l=10, r=30, t=10, b=20),
            xaxis=dict(ticksuffix="%", tickfont=dict(size=8), zeroline=False),
            yaxis=dict(tickfont=dict(size=8)),
            showlegend=False,
        )
        _chart(fig_perf)
        _insight_note(
            "What each equity index and commodity returned over the past month. "
            "Green bars gained value; red bars lost it. "
            "This gives an instant snapshot of which markets are under pressure and which are holding up."
        )

    with an_col:
        _thread(
            "To put those returns in context, the table below finds the historical periods that "
            "most closely resemble today's conditions. History does not repeat, but it rhymes."
        )
        _label("Historical Analogues: Most Similar Conditions")
        if ews["analogues"]:
            rows_html = ""
            for a in ews["analogues"]:
                sim_pct = a["sim"]
                rows_html += (
                    f'<tr style="border-bottom:1px solid #2a2d3a">'
                    f'<td style="padding:4px 6px;font-family:JetBrains Mono,monospace;font-size:0.66rem;font-weight:600;color:#e8e9ed">{a["date"]}</td>'
                    f'<td style="padding:4px 6px;font-size:0.66rem;color:{_r_colors.get(list({0:"Decorrelated",1:"Normal",2:"Elevated",3:"Crisis"}.values()).index(a["regime"]) if a["regime"] in list({0:"Decorrelated",1:"Normal",2:"Elevated",3:"Crisis"}.values()) else 1,_r_colors[1])};font-weight:600">{a["regime"]}</td>'
                    f'<td style="padding:4px 6px;font-size:0.66rem;color:{_r_colors.get(a["r30_int"],_r_colors[1])};font-weight:600">{a["r30"]}</td>'
                    f'<td style="padding:4px 6px;font-size:0.66rem;color:{_r_colors.get(a["r90_int"],_r_colors[1])};font-weight:600">{a["r90"]}</td>'
                    f'<td style="padding:4px 6px">'
                    f'<div style="background:#2a2d3a;border-radius:2px;height:4px;width:50px">'
                    f'<div style="width:{sim_pct:.0f}%;background:#CFB991;height:4px;border-radius:2px"></div>'
                    f'</div>'
                    f'<div style="{_F}font-size:0.60rem;color:#6b7280;margin-top:1px">{sim_pct:.0f}%</div>'
                    f'</td></tr>'
                )
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;{_F}">'
                f'<thead><tr style="background:#1a1d27">'
                f'<th style="padding:4px 6px;font-size:0.56rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#CFB991;text-align:left">Date</th>'
                f'<th style="padding:4px 6px;font-size:0.56rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#CFB991;text-align:left">Then</th>'
                f'<th style="padding:4px 6px;font-size:0.56rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#CFB991;text-align:left">+30d</th>'
                f'<th style="padding:4px 6px;font-size:0.56rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#CFB991;text-align:left">+90d</th>'
                f'<th style="padding:4px 6px;font-size:0.56rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#CFB991;text-align:left">Sim</th>'
                f'</tr></thead>'
                f'<tbody>{rows_html}</tbody>'
                f'</table>',
                unsafe_allow_html=True,
            )
            best = ews["analogues"][0]
            st.markdown(
                f'<p style="{_F}font-size:0.65rem;color:#8890a1;line-height:1.55;margin-top:8px">'
                f'Closest match: <b>{best["date"]}</b> ({best["sim"]:.0f}% similar). '
                f'Regime moved to <b>{best["r30"]}</b> within 30d, '
                f'<b>{best["r90"]}</b> within 90d.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<p style="{_F}font-size:0.70rem;color:#6b7280;margin-top:1rem">'
                f'Historical analogue matching requires ≥ 200 days of regime history.</p>',
                unsafe_allow_html=True,
            )

        # Active events stacked below analogues
        st.markdown('<div style="margin:0.6rem 0;border-top:1px solid #2a2d3a"></div>',
                    unsafe_allow_html=True)
        _label("Active Geopolitical Events")
        from datetime import date as _date
        today  = _date.today()
        active = [e for e in GEOPOLITICAL_EVENTS if e["end"] >= today]
        if active:
            for ev in active:
                st.markdown(
                    f'<div style="border-left:2px solid {ev["color"]};'
                    f'padding:0.35rem 0.6rem;margin-bottom:5px;background:#1a1d27">'
                    f'<div style="{_F}font-size:0.56rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.08em;color:{ev["color"]}">{ev["category"]} · {ev["label"]}</div>'
                    f'<div style="{_F}font-size:0.70rem;color:#e8e9ed;font-weight:600;margin:1px 0">{ev["name"]}</div>'
                    f'<div style="{_F}font-size:0.64rem;color:#b8bec8;line-height:1.5">{ev["description"][:120]}{"…" if len(ev["description"])>120 else ""}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div style="margin:0.6rem 0;border-top:1px solid #2a2d3a"></div>',
                unsafe_allow_html=True)

    # ── ROW: Heatmap | Window control ─────────────────────────────────────
    hm_l, hm_r = st.columns([3, 1])
    with hm_r:
        _label("Heatmap Window")
        window_opt = st.select_slider(
            "Sample window",
            options=[63, 126, 252, 504, 0],
            value=252,
            format_func=lambda x: "Full" if x == 0 else f"{x}d",
            label_visibility="collapsed",
        )
        st.markdown(
            f'<p style="{_F}font-size:0.64rem;color:#8890a1;line-height:1.6;margin-top:8px">'
            f'<b style="color:#c0392b">Red</b> = positive correlation (risk-off, inflation). '
            f'<b style="color:#2980b9">Blue</b> = negative (safe-haven divergence). '
            f'White = decorrelated.</p>',
            unsafe_allow_html=True,
        )
        # Mini regime legend
        for r_id, r_name in {0:"Decorrelated",1:"Normal",2:"Elevated",3:"Crisis"}.items():
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">'
                f'<div style="width:10px;height:10px;border-radius:2px;background:{regime_colors[r_id]};flex-shrink:0"></div>'
                f'<span style="{_F}font-size:0.64rem;color:#b8bec8">{r_name}</span></div>',
                unsafe_allow_html=True,
            )

    with hm_l:
        _label("Cross-Asset Correlation Heatmap")
        corr_mat = cross_asset_corr(eq_r, cmd_r, window=window_opt or None)
        if not corr_mat.empty:
            fig_heat = go.Figure(go.Heatmap(
                z=corr_mat.values,
                x=corr_mat.columns.tolist(),
                y=corr_mat.index.tolist(),
                colorscale=[[0.0,"#2980b9"],[0.5,"#1e2130"],[1.0,"#c0392b"]],
                zmid=0, zmin=-1, zmax=1,
                text=corr_mat.round(2).values,
                texttemplate="%{text}",
                textfont=dict(size=8, family="JetBrains Mono, monospace", color="#e8e9ed"),
                colorbar=dict(title="Corr", thickness=10, len=0.8,
                              tickfont=dict(size=8, family="JetBrains Mono, monospace")),
                hoverongaps=False,
            ))
            fig_heat.update_layout(
                template="purdue",
                height=420,
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                font=dict(color="#e8e9ed"),
                xaxis=dict(tickangle=-40, tickfont=dict(size=8, color="#8890a1"), rangeslider=dict(visible=False)),
                yaxis=dict(tickfont=dict(size=8, color="#8890a1")),
                margin=dict(l=110, r=20, t=20, b=110),
            )
            _chart(fig_heat)
            _insight_note(
                "Shows the rolling pairwise correlation between every equity and commodity in the dashboard. "
                "Dark warm colours mean assets are moving together (risk-on or systemic stress). "
                "Dark cool colours mean they are moving in opposite directions (safe-haven flows or diverging fundamentals)."
            )

    st.markdown('<div style="margin:0.6rem 0;border-top:1px solid #2a2d3a"></div>',
                unsafe_allow_html=True)

    # ── AI Narrative (always visible, auto-generated) ──────────────────────
    _narrative_key = f"ai_narrative_{start}_{end}_{regime_name}_{risk_result['score']}"
    if _narrative_key not in st.session_state:
        _ai_key = ""
        _ai_provider = ""
        try:
            _sec = st.secrets.get("keys", {})
            if _sec.get("anthropic_api_key", ""):
                _ai_key = _sec["anthropic_api_key"]
                _ai_provider = "anthropic"
            elif _sec.get("openai_api_key", ""):
                _ai_key = _sec["openai_api_key"]
                _ai_provider = "openai"
        except Exception:
            pass

        if _ai_key:
            import datetime as _dt
            _active_events = [e["label"] for e in GEOPOLITICAL_EVENTS
                              if e["end"] >= _dt.date.today()]
            _prompt = (
                f"You are a quantitative equity-commodities analyst at a macro hedge fund. "
                f"Write a concise 2-paragraph market commentary (150-200 words).\n\n"
                f"Correlation Regime: {regime_name} (60d avg corr: {current_avg_corr:.3f})\n"
                f"Risk Score: {risk_result['score']}/100 ({risk_result['label']})\n"
                f"Best/Worst equity 1M: {best_eq} / {worst_eq}\n"
                f"Best commodity 1M: {best_cmd}\n"
                f"Active events: {', '.join(_active_events) or 'None'}\n\n"
                "Paragraph 1: current regime and equity-commodity spillover implications.\n"
                "Paragraph 2: key risk factor and 1-2 trade implications."
            )
            try:
                with st.spinner("Generating AI market narrative…"):
                    if _ai_provider == "anthropic":
                        import anthropic as _ant
                        _client = _ant.Anthropic(api_key=_ai_key)
                        _resp = _client.messages.create(
                            model="claude-haiku-4-5-20251001", max_tokens=400,
                            messages=[{"role": "user", "content": _prompt}],
                        )
                        st.session_state[_narrative_key] = _resp.content[0].text
                    else:
                        from openai import OpenAI as _OpenAI
                        _client = _OpenAI(api_key=_ai_key)
                        _resp = _client.chat.completions.create(
                            model="gpt-4o", max_tokens=400,
                            messages=[{"role": "user", "content": _prompt}],
                        )
                        st.session_state[_narrative_key] = _resp.choices[0].message.content
            except Exception as _e:
                st.session_state[_narrative_key] = f"__error__{_e}"

    _narrative_val = st.session_state.get(_narrative_key, "")
    st.markdown(
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:0.56rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.14em;color:#8E6F3E;margin:0.5rem 0 0.3rem">AI Market Narrative</p>',
        unsafe_allow_html=True,
    )
    if not _narrative_val:
        st.info("Add `anthropic_api_key` or `openai_api_key` to `.streamlit/secrets.toml` to enable AI narratives.")
    elif _narrative_val.startswith("__error__"):
        st.error(f"Narrative generation failed: {_narrative_val[9:]}")
    else:
        st.markdown(
            f'<div style="border-left:4px solid {regime_color};padding:0.8rem 1rem;'
            f'background:#1a1d27;border-radius:0 4px 4px 0;margin-bottom:0.8rem">'
            f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.78rem;color:#e8e9ed;line-height:1.75">'
            f'{_narrative_val.replace(chr(10), "<br>")}</div></div>',
            unsafe_allow_html=True,
        )

    _page_conclusion(
        "Current Market Regime",
        "Use this page to form a view before exploring deeper analysis. A rising geopolitical "
        "score, tightening equity-commodity correlations, and underperformance concentrated in "
        "a specific region or commodity group is the classic early-warning pattern. Drill into "
        "the Analysis pages for the quantitative breakdown."
    )
    _page_footer()

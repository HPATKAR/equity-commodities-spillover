"""
Page 1 - Overview
KPIs, equity-commodities correlation heatmap snapshot, regime status, recent events.
"""

from __future__ import annotations

import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"


@st.cache_data(ttl=86400, show_spinner=False)
def _logo_b64() -> str:
    p = _ASSETS / "logo.png"
    if p.exists():
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    return ""

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
    _page_header, _page_footer, _add_event_bands, _insight_note,
    _data_status_bar,
)
from src.analysis.proactive_alerts import compute_alerts
from src.ui.alert_banner import render_alert_banner

_F = "font-family:'DM Sans',sans-serif;"


def _label(txt: str) -> None:
    st.markdown(
        f'<p style="{_F}font-size:0.58rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.14em;color:#8E6F3E;margin:0 0 6px 0">{txt}</p>',
        unsafe_allow_html=True,
    )


def page_overview(start: str, end: str, fred_key: str = "") -> None:
    import datetime as _dt
    _now_str = _dt.datetime.now().strftime("%H:%M:%S UTC")

    # Auto-refresh: track last load time; offer manual refresh button
    if "overview_last_loaded" not in st.session_state:
        st.session_state["overview_last_loaded"] = _dt.datetime.now()
    _age = int((_dt.datetime.now() - st.session_state["overview_last_loaded"]).total_seconds())
    _age_label = f"{_age // 60}m {_age % 60}s ago" if _age >= 60 else f"{_age}s ago"
    _stale = _age > 300    # >5 min = stale

    _hdr_col, _btn_col = st.columns([5, 1])
    with _hdr_col:
        _page_header("Market Spillover Command Center",
                     "15 equity indices · 17 commodity futures · Correlation regimes · Geopolitical risk · Spillover signals")
    with _btn_col:
        _stale_color = "#c0392b" if _stale else "#27ae60"
        st.markdown(
            f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:0;'
            f'padding:0.4rem 0.6rem;margin-bottom:0.4rem;text-align:center">'
            f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.50rem;font-weight:700;'
            f'letter-spacing:0.14em;text-transform:uppercase;color:#3a3a3a;margin-bottom:2px">LAST UPDATED</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;color:{_stale_color}">'
            f'{_age_label}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Refresh", key="overview_refresh_btn", use_container_width=True):
            st.session_state["overview_last_loaded"] = _dt.datetime.now()
            st.cache_data.clear()
            st.rerun()
        if st.button("Open Analyst", key="hdr_ai_analyst_btn", type="primary", use_container_width=True):
            st.session_state["current_page"] = "ai_chat"
            st.rerun()
    _page_intro(
        "The central research question of this dashboard: <strong>do equity market shocks spill into "
        "commodities - and in which direction?</strong> This page is your live answer. The regime badge "
        "tells you whether equity-commodity co-movement is currently amplifying or absorbing risk. "
        "The KPIs quantify how tight the spillover channel is right now. The heatmap shows which "
        "specific equity-commodity pairs are most coupled. Start here before reading any other page."
    )

    with st.spinner("Loading market data…"):
        eq_r, cmd_r = load_returns(start, end)
        try:
            from src.data.loader import load_iv_snapshot
            _iv_snap = load_iv_snapshot()
        except Exception:
            _iv_snap = {}

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

    recent_eq  = (1 + eq_r.iloc[-22:]).prod() - 1
    recent_cmd = (1 + cmd_r.iloc[-22:]).prod() - 1
    best_eq    = recent_eq.idxmax();  worst_eq  = recent_eq.idxmin()
    best_cmd   = recent_cmd.idxmax(); worst_cmd = recent_cmd.idxmin()

    # KPI strip
    k1, k2, k3, k4, k5 = st.columns(5)

    def _kpi(col, label, value, delta="", dcolor=""):
        col.markdown(
            f'<div style="border:1px solid #E8E5E0;border-radius:0;'
            f'padding:0.55rem 0.75rem;background:#1c1c1c">'
            f'<div style="{_F}font-size:0.58rem;font-weight:600;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:#8890a1;margin-bottom:3px">{label}</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.98rem;'
            f'font-weight:700;color:#e8e9ed;line-height:1.2">{value}</div>'
            + (f'<div style="{_F}font-size:0.62rem;color:{dcolor};margin-top:2px">{delta}</div>' if delta else "")
            + '</div>',
            unsafe_allow_html=True,
        )

    k1.markdown(
        f'<div style="border-top:2px solid {regime_color};border-bottom:1px solid #E8E5E0;border-radius:0;'
        f'padding:0.55rem 0.75rem;background:#1c1c1c">'
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

    # ── Data freshness & implied vol strip ────────────────────────────────
    import time as _time
    _now_ts = _time.time()
    _status_items: list[tuple[str, str, int | None]] = [
        ("Regime", regime_name, None),
        ("Corr", f"{current_avg_corr:.3f}", 60),     # refreshed with page data
    ]
    for _iv_name in ("VIX", "OVX", "GVZ", "VVIX"):
        _v = _iv_snap.get(_iv_name)
        if _v is not None:
            _status_items.append((_iv_name, f"{_v:.1f}", 300))   # 5-min TTL
    _data_status_bar(_status_items)

    # ── Implied Vol History (OVX / GVZ / VIX / VVIX) ─────────────────────
    try:
        from src.data.loader import load_implied_vol as _load_iv_hist
        with st.expander("Implied Volatility History - OVX · GVZ · VIX · VVIX", expanded=False):
            _iv_hist = _load_iv_hist(start, end)
            if not _iv_hist.empty:
                _iv_cols = st.columns(2)
                _iv_pairs = [
                    (["OVX", "GVZ"], "Oil & Gold Implied Vol (OVX / GVZ)", "#e67e22", "#CFB991"),
                    (["VIX", "VVIX"], "Equity Implied Vol (VIX / VVIX)", "#c0392b", "#2980b9"),
                ]
                for _ic, (_keys, _title, _c1, _c2) in zip(_iv_cols, _iv_pairs):
                    with _ic:
                        _fig_iv = go.Figure()
                        _colors_iv = [_c1, _c2]
                        for _ki, _k in enumerate(_keys):
                            if _k in _iv_hist.columns:
                                _fig_iv.add_trace(go.Scatter(
                                    x=_iv_hist.index, y=_iv_hist[_k],
                                    name=_k, line=dict(color=_colors_iv[_ki], width=1.4),
                                ))
                        _fig_iv.update_layout(
                            template="purdue", height=180,
                            margin=dict(l=0, r=10, t=18, b=10),
                            title=dict(text=_title, font=dict(size=9, color="#8890a1"), x=0),
                            legend=dict(orientation="h", y=1.15, x=1, xanchor="right",
                                        font=dict(size=8)),
                            yaxis=dict(tickfont=dict(size=8)),
                            xaxis=dict(tickfont=dict(size=8)),
                            paper_bgcolor="#111111", plot_bgcolor="#111111",
                        )
                        _chart(_fig_iv)
                st.markdown(
                    '<p style="font-size:0.60rem;color:#6b7280;margin:0">'
                    'OVX = CBOE Oil Volatility Index · GVZ = CBOE Gold Volatility Index · '
                    'VVIX = Volatility of VIX · All sourced from CBOE via Yahoo Finance.</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Implied vol data unavailable. Requires ^OVX, ^GVZ, ^VIX, ^VVIX from Yahoo Finance.")
    except Exception:
        pass

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
            f'<div style="border:1px solid #E8E5E0;border-radius:0;'
            f'padding:0.55rem 0.75rem;background:#1c1c1c">'
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
                _tlt_30d = float(((1 + _tlt.iloc[-21:]).prod() - 1) * 100)
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
                    _hyg_30d = float(((1 + _s.iloc[-21:]).prod() - 1) * 100)
            if "IG Corporate (LQD)" in fi_r.columns:
                _s = fi_r["IG Corporate (LQD)"].dropna()
                if len(_s) >= 21:
                    _lqd_30d = float(((1 + _s.iloc[-21:]).prod() - 1) * 100)
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
                _dxy_30d = float(((1 + _dxy.iloc[-21:]).prod() - 1) * 100)
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
                _spx_30d = float(((1 + _spx_s.iloc[-21:]).prod() - 1) * 100)
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
                _emb_30d = float(((1 + _emb.iloc[-21:]).prod() - 1) * 100)
    except Exception:
        pass
    _dark_kpi(_fi_k5, "EMB 30d Return",
              f"{_emb_30d:+.1f}%" if _emb_30d is not None else "-",
              delta="EM credit bid" if _emb_30d is not None and _emb_30d > 0 else "EM credit stress" if _emb_30d is not None else "",
              delta_up=True if _emb_30d is not None and _emb_30d > 0 else False if _emb_30d is not None else None)

    st.markdown('<div style="margin:0.7rem 0 0.5rem;border-top:1px solid #2a2a2a"></div>',
                unsafe_allow_html=True)
    _thread(
        "The numbers above give the level of stress. The section below breaks down its source - "
        "which geopolitical conflicts are active and how severely each is scoring."
    )

    # ── ROW: Risk gauge | Risk history ─────────────────────────────────────
    with st.spinner("Computing risk score…"):
        risk_result = compute_risk_score(avg_corr, cmd_r)
        score_hist  = risk_score_history(avg_corr, cmd_r, eq_r=eq_r)

    # ── Proactive AI alert feed ─────────────────────────────────────────────
    try:
        from src.analysis.cot import load_cot_data
        _cot_df = load_cot_data(years=2)
    except Exception:
        _cot_df = None
    _alerts = compute_alerts(
        eq_r=eq_r, cmd_r=cmd_r, avg_corr=avg_corr, regimes=regimes,
        risk_score=float(risk_result["score"]),
        risk_history=score_hist if isinstance(score_hist, pd.Series) else pd.Series(dtype=float),
        cot_df=_cot_df,
    )
    # Build a minimal context string for the AI briefing
    _ctx_brief = (
        f"Regime: {regime_name} (level {int(current_regime)}/3). "
        f"Risk score: {risk_result['score']:.0f}/100. "
        f"60d avg |corr|: {float(current_avg_corr):.3f}. "
        f"Best equity 1M: {best_eq}. Worst equity 1M: {worst_eq}. "
        f"Best commodity 1M: {best_cmd}. Worst commodity 1M: {worst_cmd}."
    )
    render_alert_banner(_alerts, market_context=_ctx_brief)

    # ── Morning Briefing Agent Chain ───────────────────────────────────────
    # Auto-expands when risk score is Elevated or above — visible deliberation.
    _briefing_risk = float(risk_result["score"])
    _briefing_expanded = _briefing_risk >= 50
    try:
        from src.ui.agent_panel import render_morning_briefing_panel
        _top_alert_texts = [a.get("title", "") for a in _alerts[:3] if a.get("title")]
        _top_conflict    = risk_result.get("top_conflict")
        _briefing_label = (
            f"⚡ AI Analyst Team — Morning Briefing (Risk {_briefing_risk:.0f}/100)"
            if _briefing_expanded
            else f"AI Analyst Team — Morning Briefing (Risk {_briefing_risk:.0f}/100)"
        )
        with st.expander(_briefing_label, expanded=_briefing_expanded):
            render_morning_briefing_panel(
                risk_score=_briefing_risk,
                top_alerts=_top_alert_texts,
                top_conflict=_top_conflict,
                auto_run=True,
            )
    except Exception:
        pass

    _r_colors = {0: "#2e7d32", 1: "#555960", 2: "#e67e22", 3: "#c0392b"}


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
            f'<div style="border:1px solid #E8E5E0;border-radius:0;padding:0.75rem;'
            f'background:#1c1c1c;border-top:3px solid {comp_color}">'
            f'<div style="{_F}font-size:0.56rem;font-weight:700;letter-spacing:0.12em;'
            f'text-transform:uppercase;color:#6b7280;margin-bottom:4px">Composite Score</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:2.2rem;'
            f'font-weight:700;color:{comp_color};line-height:1">{comp_ews:.0f}'
            f'<span style="font-size:0.75rem;color:#6b7280">/100</span></div>'
            f'<div style="background:#2a2a2a;border-radius:0;height:6px;margin:6px 0">'
            f'<div style="width:{comp_ews:.0f}%;background:{comp_color};height:6px;border-radius:0"></div>'
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
                f'<div style="border:1px solid #E8E5E0;border-radius:0;padding:0.6rem 0.55rem;'
                f'background:#1c1c1c;border-top:2px solid {c}">'
                f'<div style="{_F}font-size:0.54rem;font-weight:700;letter-spacing:0.10em;'
                f'text-transform:uppercase;color:#6b7280;margin-bottom:3px">{name}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1.1rem;'
                f'font-weight:700;color:{c}">{s:.0f}'
                f'<span style="{_F}font-size:0.60rem;color:#6b7280">/100</span></div>'
                f'<div style="background:#2a2a2a;border-radius:2px;height:3px;margin:4px 0">'
                f'<div style="width:{s:.0f}%;background:{c};height:3px;border-radius:2px"></div>'
                f'</div>'
                f'<div style="{_F}font-size:0.64rem;color:#8890a1;line-height:1.45;margin-top:3px">'
                f'{data["desc"]}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="margin:0.6rem 0;border-top:1px solid #2a2a2a"></div>',
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
                    f'<tr style="border-bottom:1px solid #2a2a2a">'
                    f'<td style="padding:4px 6px;font-family:JetBrains Mono,monospace;font-size:0.66rem;font-weight:600;color:#e8e9ed">{a["date"]}</td>'
                    f'<td style="padding:4px 6px;font-size:0.66rem;color:{_r_colors.get(list({0:"Decorrelated",1:"Normal",2:"Elevated",3:"Crisis"}.values()).index(a["regime"]) if a["regime"] in list({0:"Decorrelated",1:"Normal",2:"Elevated",3:"Crisis"}.values()) else 1,_r_colors[1])};font-weight:600">{a["regime"]}</td>'
                    f'<td style="padding:4px 6px;font-size:0.66rem;color:{_r_colors.get(a["r30_int"],_r_colors[1])};font-weight:600">{a["r30"]}</td>'
                    f'<td style="padding:4px 6px;font-size:0.66rem;color:{_r_colors.get(a["r90_int"],_r_colors[1])};font-weight:600">{a["r90"]}</td>'
                    f'<td style="padding:4px 6px">'
                    f'<div style="background:#2a2a2a;border-radius:2px;height:4px;width:50px">'
                    f'<div style="width:{sim_pct:.0f}%;background:#CFB991;height:4px;border-radius:2px"></div>'
                    f'</div>'
                    f'<div style="{_F}font-size:0.60rem;color:#6b7280;margin-top:1px">{sim_pct:.0f}%</div>'
                    f'</td></tr>'
                )
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;{_F}">'
                f'<thead><tr style="background:#1c1c1c">'
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
        st.markdown('<div style="margin:0.6rem 0;border-top:1px solid #2a2a2a"></div>',
                    unsafe_allow_html=True)
        _label("Active Geopolitical Events")
        from datetime import date as _date
        today  = _date.today()
        active = [e for e in GEOPOLITICAL_EVENTS if e["end"] >= today]
        if active:
            for ev in active:
                st.markdown(
                    f'<div style="border-left:2px solid {ev["color"]};'
                    f'padding:0.35rem 0.6rem;margin-bottom:5px;background:#1c1c1c">'
                    f'<div style="{_F}font-size:0.56rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.08em;color:{ev["color"]}">{ev["category"]} · {ev["label"]}</div>'
                    f'<div style="{_F}font-size:0.70rem;color:#e8e9ed;font-weight:600;margin:1px 0">{ev["name"]}</div>'
                    f'<div style="{_F}font-size:0.64rem;color:#b8b8b8;line-height:1.5">{ev["description"][:120]}{"…" if len(ev["description"])>120 else ""}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div style="margin:0.6rem 0;border-top:1px solid #2a2a2a"></div>',
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
                f'<span style="{_F}font-size:0.64rem;color:#b8b8b8">{r_name}</span></div>',
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
                colorscale=[[0.0,"#2980b9"],[0.5,"#1e1e1e"],[1.0,"#c0392b"]],
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
                paper_bgcolor="#111111", plot_bgcolor="#111111",
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

    st.markdown('<div style="margin:0.6rem 0;border-top:1px solid #2a2a2a"></div>',
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
            f'<div style="padding:0.8rem 1rem;'
            f'background:#1c1c1c;border-radius:0;margin-bottom:0.8rem">'
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

    # ── AI Workforce - Orchestrated Pipeline ─────────────────────────────────
    try:
        from src.analysis.agent_orchestrator import get_orchestrator
        from src.ui.agent_panel import (
            render_agent_output_block, render_activity_feed, render_pending_review,
        )
        from src.analysis.agent_state import pending_count, init_agents

        init_agents()

        _anthropic_key = _openai_key = ""
        try:
            _keys = st.secrets.get("keys", {})
            _anthropic_key = _keys.get("anthropic_api_key", "") or ""
            _openai_key    = _keys.get("openai_api_key",    "") or ""
        except Exception:
            pass
        _provider = "anthropic" if _anthropic_key else ("openai" if _openai_key else None)
        _api_key  = _anthropic_key or _openai_key

        # Full market context - scalars only (no DataFrames; agents don't need raw returns)
        _market_ctx = {
            "regime_name":     regime_name,
            "regime_level":    int(current_regime),
            "risk_score":      float(risk_result["score"]),
            "avg_corr":        float(current_avg_corr),
            "corr_delta":      float(corr_delta),
            "best_equity":     best_eq,
            "worst_equity":    worst_eq,
            "best_commodity":  best_cmd,
            "worst_commodity": worst_cmd,
            "n_alerts":        len(_alerts),
            "alert_categories": [str(a.category) for a in _alerts],
            "alert_summaries": [a.title for a in _alerts[:4]],
            # Implied vol context
            "vix":   _iv_snap.get("VIX"),
            "ovx":   _iv_snap.get("OVX"),
            "gvz":   _iv_snap.get("GVZ"),
            "vvix":  _iv_snap.get("VVIX"),
        }

        # Cache market context for AI Workforce "Run All Agents" button
        st.session_state["overview_market_context"] = _market_ctx

        orch = get_orchestrator(_provider, _api_key)

        # Run only Round 1 agents on Overview (fast path) - heavier rounds
        # run on their respective pages when user navigates there.
        # Round 1: signal_auditor, macro_strategist, geopolitical_analyst
        # They inform the Risk Officer which is in Round 2.
        # We run the full pipeline here on Overview since it's the hub.
        if _provider and _api_key:
            with st.spinner("AI Workforce analysing markets…"):
                _pipeline_results = orch.run(_market_ctx)
        else:
            # No API key - still run orchestrator for calibration (no LLM calls)
            _pipeline_results = {}

        # ── Pipeline status strip ────────────────────────────────────────
        _orch_status = orch.status()
        _div_flags   = orch.divergence_flags()

        if _orch_status.get("agents_fresh") or _div_flags:
            _fresh_agents = _orch_status.get("agents_fresh", [])
            _stale_agents = _orch_status.get("agents_stale", [])
            _flag_html = ""
            if _div_flags:
                _flag_html = "".join([
                    f'<span style="font-size:0.58rem;color:#e67e22;margin-left:0.6rem">'
                    f'&#9651; {f["topic"].upper()}: {f["agent_a"]} ↔ {f["agent_b"]}</span>'
                    for f in _div_flags[:3]
                ])
            _fresh_str = ", ".join(_fresh_agents[:5]) if _fresh_agents else "-"
            st.markdown(
                f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:0.4rem;'
                f'background:#0d0d0d;border:1px solid #1e1e1e;border-radius:0;'
                f'padding:0.35rem 0.8rem;margin:0.6rem 0">'
                f'<span style="font-size:0.50rem;font-weight:700;letter-spacing:0.16em;'
                f'text-transform:uppercase;color:#3a3a3a;margin-right:0.5rem">PIPELINE</span>'
                f'<span style="font-size:0.58rem;color:#27ae60">&#9679; Active: {_fresh_str}</span>'
                f'{_flag_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Risk Officer output (primary briefing) ────────────────────────
        _ro_result = _pipeline_results.get("risk_officer", {})
        if not _ro_result:
            # Fallback: try direct call if orchestrator didn't run it
            try:
                from src.agents.risk_officer import run as _ro_run
                _ro_result = _ro_run(_market_ctx, _provider, _api_key)
            except Exception:
                _ro_result = {}

        if _ro_result.get("narrative"):
            st.markdown("---")
            render_agent_output_block("risk_officer", _ro_result)

        # ── Secondary agent outputs (round 1 inline snippets) ─────────────
        _secondary = {
            k: v for k, v in _pipeline_results.items()
            if k != "risk_officer" and v.get("narrative")
        }
        if _secondary:
            with st.expander(f"Full AI Workforce Output ({len(_secondary)} agents)", expanded=False):
                for _aid, _res in _secondary.items():
                    render_agent_output_block(_aid, _res)

        # ── Divergence flags callout ──────────────────────────────────────
        if _div_flags:
            _flag_rows = "\n".join([
                f"• **{f['agent_a']}** vs **{f['agent_b']}** - divergence on `{f['topic']}`"
                for f in _div_flags
            ])
            st.warning(f"**Agent Divergence Detected**\n\n{_flag_rows}")

        # Activity feed - full view on Overview
        st.markdown("---")
        render_activity_feed(max_entries=20, collapsible=False)

        # Pending review summary
        _n_pend = pending_count()
        if _n_pend > 0:
            st.markdown("---")
            render_pending_review()
    except Exception as _e:
        import traceback as _tb
        st.error(f"AI Workforce error: {_e}")
        with st.expander("Error details", expanded=False):
            st.code(_tb.format_exc(), language="text")

    _page_footer()

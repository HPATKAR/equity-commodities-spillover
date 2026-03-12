"""
Page 7 — Portfolio Stress Tester
Build a custom equity/commodity allocation, then stress-test it against
every historical geopolitical event in the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

from src.data.loader import load_all_prices, load_returns
from src.data.config import GEOPOLITICAL_EVENTS, PALETTE, EQUITY_REGIONS, COMMODITY_GROUPS
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_footer,
)


def _portfolio_path(prices: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Compute daily portfolio value (base=100) from price DataFrame and weights dict."""
    assets = [a for a in weights if a in prices.columns]
    if not assets:
        return pd.Series(dtype=float)
    w = np.array([weights[a] for a in assets])
    w = w / w.sum()  # normalise
    p = prices[assets].dropna(how="all").ffill()
    p_norm = p / p.iloc[0]
    port = (p_norm * w).sum(axis=1) * 100
    return port


def _max_drawdown(series: pd.Series) -> float:
    """Maximum drawdown (%) of an indexed series."""
    if series.empty:
        return np.nan
    roll_max = series.cummax()
    dd = (series - roll_max) / roll_max * 100
    return float(dd.min())


def _sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio from daily log returns."""
    if returns.empty or returns.std() < 1e-9:
        return np.nan
    return float((returns.mean() - rf / 252) / returns.std() * np.sqrt(252))


def _event_stats(
    prices: pd.DataFrame,
    weights: dict[str, float],
    event: dict,
    pre_days: int = 30,
    post_days: int = 60,
) -> dict:
    """Compute portfolio stats for pre / during / post windows of an event."""
    today = date.today()
    t0 = pd.Timestamp(event["start"])
    t1 = pd.Timestamp(min(event["end"], today))

    full_slice = prices.loc[
        t0 - pd.Timedelta(days=pre_days + 5):
        t1 + pd.Timedelta(days=post_days + 5)
    ]
    if full_slice.empty:
        return {}

    port = _portfolio_path(full_slice, weights)
    if port.empty:
        return {}

    log_r = np.log(port / port.shift(1)).dropna()

    def _window_ret(a, b) -> float:
        sl = port.loc[a:b]
        if len(sl) < 2:
            return np.nan
        return float((sl.iloc[-1] / sl.iloc[0] - 1) * 100)

    def _window_r(a, b) -> pd.Series:
        return log_r.loc[a:b]

    return {
        "event":       event["label"],
        "name":        event["name"],
        "color":       event["color"],
        "pre_ret":     _window_ret(t0 - pd.Timedelta(days=pre_days), t0),
        "during_ret":  _window_ret(t0, t1),
        "post_ret":    _window_ret(t1, t1 + pd.Timedelta(days=post_days)),
        "max_dd":      _max_drawdown(port.loc[t0:t1]),
        "sharpe":      _sharpe(_window_r(t0, t1)),
        "port":        port,
        "t0":          t0,
        "t1":          t1,
    }


def page_stress_test(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.3rem">Portfolio Stress Tester</h1>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "Build a custom multi-asset portfolio and stress-test it against every "
        "geopolitical and macro event in the dashboard. Metrics include cumulative "
        "return (pre/during/post), maximum drawdown, and Sharpe ratio during each event. "
        "Not investment advice — for educational purposes only."
    )

    with st.spinner("Loading price data…"):
        eq_p, cmd_p = load_all_prices(start, end)

    if eq_p.empty or cmd_p.empty:
        st.error("Market data unavailable.")
        return

    all_prices = pd.concat([eq_p, cmd_p], axis=1)

    # ── Portfolio builder ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Build Your Portfolio")
    _definition_block(
        "How to Use",
        "Select assets and enter portfolio weights (%). "
        "Weights are automatically normalised to sum to 100%. "
        "Mix equities and commodities to test diversification effects.",
    )

    all_assets = list(eq_p.columns) + list(cmd_p.columns)

    # Preset portfolios
    presets = {
        "60/40 (S&P / Gold)":          {"S&P 500": 60, "Gold": 40},
        "Energy Macro":                {"S&P 500": 40, "WTI Crude Oil": 30, "Gold": 20, "Natural Gas": 10},
        "EM + Commodities":            {"Sensex": 25, "CSI 300": 25, "Copper": 25, "Wheat": 25},
        "Global Diversified":          {"S&P 500": 25, "Eurostoxx 50": 20, "Nikkei 225": 15,
                                        "Gold": 20, "WTI Crude Oil": 10, "Wheat": 10},
        "Custom (clear preset)":       {},
    }

    c1, c2 = st.columns([1, 2])
    preset_name = c1.selectbox("Start from preset", list(presets.keys()), key="st_preset")
    preset_assets = list(presets[preset_name].keys()) if presets[preset_name] else []

    selected_assets = c2.multiselect(
        "Select assets",
        all_assets,
        default=[a for a in preset_assets if a in all_assets] or ["S&P 500", "Gold", "WTI Crude Oil"],
        key="st_assets",
    )

    if not selected_assets:
        st.info("Select at least one asset.")
        return

    # Weight inputs
    st.markdown("##### Portfolio Weights (%)")
    default_weights = presets.get(preset_name, {})
    weight_cols = st.columns(min(len(selected_assets), 5))
    weights: dict[str, float] = {}

    for i, asset in enumerate(selected_assets):
        default_w = default_weights.get(asset, round(100 / len(selected_assets), 1))
        w = weight_cols[i % 5].number_input(
            asset, min_value=0.0, max_value=100.0,
            value=float(default_w), step=5.0, key=f"w_{asset}",
        )
        weights[asset] = w

    total_w = sum(weights.values())
    if total_w <= 0:
        st.warning("Total weight must be > 0.")
        return

    # Normalise display
    norm_weights = {a: round(w / total_w * 100, 1) for a, w in weights.items()}
    w_html = " &nbsp;|&nbsp; ".join(
        f'<b>{a}</b>: {norm_weights[a]:.1f}%' for a in selected_assets
    )
    st.markdown(
        f'<p style="font-size:0.72rem;color:#555960;margin:0.4rem 0">'
        f'Normalised allocation: {w_html}</p>',
        unsafe_allow_html=True,
    )

    # ── Event selector & run ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Select Events to Stress-Test")

    c3, c4 = st.columns(2)
    pre_days  = c3.slider("Pre-event window (days)", 10, 60, 30, key="st_pre")
    post_days = c4.slider("Post-event window (days)", 10, 90, 45, key="st_post")

    event_labels = [f"{e['label']} — {e['name']}" for e in GEOPOLITICAL_EVENTS]
    sel_events = st.multiselect(
        "Events to include",
        event_labels,
        default=event_labels,
        key="st_events",
    )
    chosen_events = [
        GEOPOLITICAL_EVENTS[event_labels.index(s)]
        for s in sel_events
    ]

    if not st.button("Run Stress Test", type="primary", key="st_run"):
        st.info("Configure your portfolio above and click **Run Stress Test**.")
        _page_footer()
        return

    # ── Compute ────────────────────────────────────────────────────────────
    with st.spinner("Running stress tests…"):
        results = [
            _event_stats(all_prices, weights, ev, pre_days, post_days)
            for ev in chosen_events
        ]
        results = [r for r in results if r]

    if not results:
        st.warning("No data available for selected events and date range.")
        _page_footer()
        return

    # ── Summary table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Stress Test Results")

    summary = pd.DataFrame([{
        "Event":         r["event"],
        "Name":          r["name"],
        "Pre (%)":       round(r["pre_ret"],   2) if not np.isnan(r["pre_ret"])   else None,
        "During (%)":    round(r["during_ret"], 2) if not np.isnan(r["during_ret"]) else None,
        "Post (%)":      round(r["post_ret"],  2) if not np.isnan(r["post_ret"])  else None,
        "Max DD (%)":    round(r["max_dd"],    2) if not np.isnan(r["max_dd"])    else None,
        "Sharpe":        round(r["sharpe"],    2) if r["sharpe"] and not np.isnan(r["sharpe"]) else None,
    } for r in results])

    def _col_pct(val):
        if pd.isna(val): return ""
        if val > 0:  return "color:#2e7d32;font-weight:600"
        if val < 0:  return "color:#c0392b;font-weight:600"
        return ""

    styled = (
        summary.style
        .applymap(_col_pct, subset=["Pre (%)", "During (%)", "Post (%)"])
        .applymap(lambda v: "color:#c0392b;font-weight:600" if isinstance(v, float) and v < -10 else "",
                  subset=["Max DD (%)"])
        .format({
            "Pre (%)":    "{:+.2f}%", "During (%)": "{:+.2f}%",
            "Post (%)":   "{:+.2f}%", "Max DD (%)":  "{:.2f}%",
            "Sharpe":     "{:.2f}",
        }, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=340)

    # KPI row
    valid_during = [r["during_ret"] for r in results if not np.isnan(r["during_ret"])]
    valid_dd     = [r["max_dd"]     for r in results if not np.isnan(r["max_dd"])]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Worst Event Return",  f"{min(valid_during):+.1f}%" if valid_during else "N/A")
    k2.metric("Best Event Return",   f"{max(valid_during):+.1f}%" if valid_during else "N/A")
    k3.metric("Worst Max Drawdown",  f"{min(valid_dd):.1f}%"      if valid_dd else "N/A")
    k4.metric("Avg During Return",   f"{np.mean(valid_during):+.1f}%" if valid_during else "N/A")

    # ── During-event bar chart ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Portfolio Return During Each Event")

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=[r["event"] for r in results],
        x=[r["during_ret"] for r in results],
        orientation="h",
        marker_color=[
            "#2e7d32" if r["during_ret"] >= 0 else "#c0392b"
            for r in results
        ],
        text=[f"{r['during_ret']:+.1f}%" for r in results],
        textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono, monospace"),
        hovertemplate="%{y}: %{x:+.1f}%<extra></extra>",
    ))
    fig_bar.add_vline(x=0, line=dict(color="#ABABAB", width=1, dash="dot"))
    fig_bar.update_layout(
        template="purdue",
        height=max(300, len(results) * 30),
        xaxis=dict(title="Portfolio Return (%)", ticksuffix="%"),
        yaxis=dict(tickfont=dict(size=9, family="JetBrains Mono, monospace")),
        margin=dict(l=130, r=80, t=30, b=30),
    )
    _chart(fig_bar)

    # ── Portfolio path for each event ─────────────────────────────────────
    st.markdown("---")
    st.subheader("Portfolio Value Path: Event Windows (Indexed to 100)")
    _section_note(
        "Each line shows portfolio value indexed to 100 at the event start. "
        "Dashed vertical lines mark event start (left) and end (right)."
    )

    fig_path = go.Figure()
    for i, r in enumerate(results):
        port = r["port"]
        t0, t1 = r["t0"], r["t1"]
        # Slice around event
        window_start = t0 - pd.Timedelta(days=pre_days)
        window_end   = t1 + pd.Timedelta(days=post_days)
        sl = port.loc[window_start:window_end]
        if sl.empty:
            continue
        base_at_start = sl.loc[t0:].iloc[0] if not sl.loc[t0:].empty else sl.iloc[0]
        sl_norm = sl / base_at_start * 100

        fig_path.add_trace(go.Scatter(
            x=sl_norm.index, y=sl_norm.values,
            name=r["event"],
            line=dict(color=r["color"], width=1.5),
            hovertemplate="%{x|%d %b %Y}: %{y:.1f}<extra>" + r["event"] + "</extra>",
        ))

    fig_path.add_hline(y=100, line=dict(color="#ABABAB", width=1, dash="dot"))
    fig_path.update_layout(
        template="purdue",
        height=420,
        yaxis=dict(title="Portfolio Value (Base=100)"),
        xaxis=dict(type="date"),
        margin=dict(l=50, r=20, t=30, b=30),
    )
    _chart(fig_path)

    _takeaway_block(
        "A portfolio's worst stress-test score reveals its most vulnerable scenario. "
        "High commodity weights amplify energy-supply-shock returns; high equity weights "
        "amplify correlation-spike drawdowns. Gold typically reduces max drawdown during "
        "crisis regimes."
    )

    _page_conclusion(
        "Stress Test Complete",
        f"Portfolio tested across {len(results)} historical events. "
        "The analysis shows how your allocation performs under real market stress. "
        "Combine with the Spillover Network to understand which assets drive the most "
        "contagion risk in your portfolio.",
    )
    _page_footer()

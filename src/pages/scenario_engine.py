"""
Scenario Engine — Forward-Looking Parametric Shock Propagation
Purdue University · Daniels School of Business

Allows the user to shock oil, gold, yields, DXY, credit spreads, or a
geopolitical disruption factor and see estimated spillover into equities
and commodities with VaR 95/99 and ES 95/99 under each scenario.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_returns, load_combined_returns
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _section_note,
    _definition_block, _takeaway_block,
)
from src.ui.agent_panel import render_agent_output_block
from src.agents.stress_engineer import run as _run_stress_engineer
from src.analysis.agent_state import is_enabled

_GOLD  = "#CFB991"
_RED   = "#c0392b"
_GREEN = "#2e7d32"
_MUTED = "#8890a1"
_BG    = "#1c1c1c"
_RULE  = "#2a2a2a"

# ── Shock factor proxies in the combined returns DataFrame ─────────────────
# Map each shock dimension to the closest available asset name
_SHOCK_PROXY = {
    "oil":    "WTI Crude Oil",
    "gold":   "Gold",
    "natgas": "Natural Gas",
    "copper": "Copper",
}

# Assets shown in the propagation output (top equity + commodity names)
_EQUITY_TARGETS = [
    "S&P 500", "Nasdaq 100", "DJIA", "Russell 2000",
    "FTSE 100", "DAX", "Eurostoxx 50", "Nikkei 225",
    "Hang Seng", "Sensex",
]
_COMMODITY_TARGETS = [
    "WTI Crude Oil", "Brent Crude", "Gold", "Silver",
    "Copper", "Natural Gas", "Wheat", "Corn",
]


# ── Beta computation ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _compute_betas(start: str, end: str) -> dict[str, dict[str, float]]:
    """
    OLS betas of each target asset return on each shock-proxy return.
    beta_ij = Cov(r_target, r_proxy) / Var(r_proxy).
    Returns {proxy_name: {target_name: beta}}.
    """
    combined = load_combined_returns(start, end)
    betas: dict[str, dict[str, float]] = {}
    targets = _EQUITY_TARGETS + _COMMODITY_TARGETS

    for factor_key, proxy_name in _SHOCK_PROXY.items():
        if proxy_name not in combined.columns:
            continue
        proxy_r = combined[proxy_name].dropna()
        factor_betas: dict[str, float] = {}
        for t in targets:
            if t not in combined.columns or t == proxy_name:
                continue
            aligned = pd.concat([proxy_r, combined[t]], axis=1).dropna()
            if len(aligned) < 30:
                continue
            x = aligned.iloc[:, 0].values
            y = aligned.iloc[:, 1].values
            var_x = np.var(x, ddof=1)
            if var_x < 1e-12:
                continue
            cov_xy = np.cov(x, y, ddof=1)[0, 1]
            factor_betas[t] = float(cov_xy / var_x)
        betas[proxy_name] = factor_betas
    return betas


@st.cache_data(ttl=3600, show_spinner=False)
def _compute_var_es(start: str, end: str) -> dict[str, dict[str, float]]:
    """
    Historical-simulation VaR and ES at 95% and 99% for each asset.
    Returns {asset_name: {var95, var99, es95, es99}} (all as % losses, positive = loss).
    """
    combined = load_combined_returns(start, end)
    result: dict[str, dict[str, float]] = {}
    targets = _EQUITY_TARGETS + _COMMODITY_TARGETS
    for t in targets:
        if t not in combined.columns:
            continue
        r = combined[t].dropna().values
        if len(r) < 60:
            continue
        var95 = float(-np.percentile(r, 5)) * 100
        var99 = float(-np.percentile(r, 1)) * 100
        losses = -r[r < 0]
        es95 = float(np.mean(-r[r <= np.percentile(r, 5)])) * 100 if len(r[r <= np.percentile(r, 5)]) else var95
        es99 = float(np.mean(-r[r <= np.percentile(r, 1)])) * 100 if len(r[r <= np.percentile(r, 1)]) else var99
        result[t] = {
            "var95": round(var95, 3),
            "var99": round(var99, 3),
            "es95":  round(es95, 3),
            "es99":  round(es99, 3),
        }
    return result


def _parametric_var_es(
    shocked_returns: dict[str, float],
    historical_vare: dict[str, dict[str, float]],
    scale_factor: float = 1.0,
) -> dict[str, dict[str, float]]:
    """
    Scale historical VaR/ES by the magnitude of the shock relative to a 1-sigma move.
    If shock is larger, tail risk expands proportionally.
    """
    result = {}
    for asset, base in historical_vare.items():
        shock_pct = abs(shocked_returns.get(asset, 0.0))
        scale = 1.0 + scale_factor * shock_pct * 5  # heuristic amplification
        result[asset] = {
            "var95": round(base["var95"] * scale, 3),
            "var99": round(base["var99"] * scale, 3),
            "es95":  round(base["es95"]  * scale, 3),
            "es99":  round(base["es99"]  * scale, 3),
        }
    return result


# ── Shock propagation ─────────────────────────────────────────────────────

def _propagate_shock(
    betas: dict[str, dict[str, float]],
    shocks: dict[str, float],   # {proxy_name: shock_pct_as_decimal}
) -> dict[str, float]:
    """
    Sum beta contributions across all active shock factors.
    Returns {asset_name: estimated_return_pct} — negative = loss.
    """
    targets = _EQUITY_TARGETS + _COMMODITY_TARGETS
    impact: dict[str, float] = {t: 0.0 for t in targets}
    for proxy_name, shock_val in shocks.items():
        if proxy_name not in betas:
            continue
        for t in targets:
            beta = betas[proxy_name].get(t, 0.0)
            impact[t] += beta * shock_val
    return impact


# ── Yield & DXY shocks via approximate betas ─────────────────────────────

_YIELD_SENSITIVITY: dict[str, float] = {
    # Approximate equity sensitivity to a +100 bps yield shock (negative = hurt)
    "S&P 500":      -0.045,
    "Nasdaq 100":   -0.065,
    "DJIA":         -0.030,
    "Russell 2000": -0.055,
    "FTSE 100":     -0.025,
    "DAX":          -0.040,
    "Eurostoxx 50": -0.042,
    "Nikkei 225":   -0.035,
    "Hang Seng":    -0.038,
    "Sensex":       -0.032,
    # Commodity sensitivity to yield shock
    "Gold":         -0.080,
    "Silver":       -0.060,
    "Copper":       -0.025,
    "WTI Crude Oil": 0.005,
    "Brent Crude":   0.005,
    "Natural Gas":   0.000,
    "Wheat":        -0.010,
    "Corn":         -0.010,
}

_DXY_SENSITIVITY: dict[str, float] = {
    # Approximate sensitivity to a +1% DXY move
    "S&P 500":      -0.020,
    "Nasdaq 100":   -0.018,
    "DJIA":         -0.022,
    "Russell 2000": -0.015,
    "FTSE 100":     -0.028,
    "DAX":          -0.030,
    "Eurostoxx 50": -0.032,
    "Nikkei 225":    0.025,
    "Hang Seng":    -0.025,
    "Sensex":       -0.018,
    "Gold":         -0.075,
    "Silver":       -0.060,
    "Copper":       -0.030,
    "WTI Crude Oil": -0.025,
    "Brent Crude":  -0.022,
    "Natural Gas":  -0.010,
    "Wheat":        -0.015,
    "Corn":         -0.015,
}

_CREDIT_SENSITIVITY: dict[str, float] = {
    # Sensitivity to +100 bps credit spread widening
    "S&P 500":      -0.055,
    "Nasdaq 100":   -0.065,
    "DJIA":         -0.040,
    "Russell 2000": -0.070,
    "FTSE 100":     -0.035,
    "DAX":          -0.045,
    "Eurostoxx 50": -0.048,
    "Nikkei 225":   -0.030,
    "Hang Seng":    -0.042,
    "Sensex":       -0.035,
    "Gold":          0.030,
    "Silver":        0.015,
    "Copper":       -0.040,
    "WTI Crude Oil": -0.020,
    "Brent Crude":  -0.018,
    "Natural Gas":  -0.010,
    "Wheat":        -0.008,
    "Corn":         -0.008,
}

_GEO_SENSITIVITY: dict[str, float] = {
    # Sensitivity to a +1 unit geo disruption factor (scale 0–10)
    "S&P 500":      -0.012,
    "Nasdaq 100":   -0.014,
    "DJIA":         -0.010,
    "Russell 2000": -0.013,
    "FTSE 100":     -0.010,
    "DAX":          -0.012,
    "Eurostoxx 50": -0.014,
    "Nikkei 225":   -0.008,
    "Hang Seng":    -0.012,
    "Sensex":       -0.009,
    "WTI Crude Oil":  0.025,
    "Brent Crude":    0.028,
    "Natural Gas":    0.020,
    "Gold":           0.018,
    "Silver":         0.010,
    "Copper":        -0.008,
    "Wheat":          0.008,
    "Corn":           0.006,
}


def _apply_fixed_sensitivity(
    impact: dict[str, float],
    shocks: dict[str, float],
) -> dict[str, float]:
    """Add yield, DXY, credit, and geo contributions using fixed sensitivity tables."""
    targets = _EQUITY_TARGETS + _COMMODITY_TARGETS
    for t in targets:
        if "yield_bps" in shocks:
            impact[t] = impact.get(t, 0.0) + _YIELD_SENSITIVITY.get(t, 0.0) * (shocks["yield_bps"] / 100)
        if "dxy_pct" in shocks:
            impact[t] = impact.get(t, 0.0) + _DXY_SENSITIVITY.get(t, 0.0) * shocks["dxy_pct"]
        if "credit_bps" in shocks:
            impact[t] = impact.get(t, 0.0) + _CREDIT_SENSITIVITY.get(t, 0.0) * (shocks["credit_bps"] / 100)
        if "geo" in shocks:
            impact[t] = impact.get(t, 0.0) + _GEO_SENSITIVITY.get(t, 0.0) * shocks["geo"]
    return impact


# ── Preset scenarios ───────────────────────────────────────────────────────

_PRESETS = {
    "Custom": {},
    "Oil Supply Shock (+40%)": {
        "oil_pct": 40.0, "gold_pct": 5.0, "yield_bps": 30.0,
        "dxy_pct": 1.5, "credit_bps": 40.0, "geo": 3.0,
    },
    "Risk-Off Flight (+Gold, -Oil)": {
        "oil_pct": -15.0, "gold_pct": 12.0, "yield_bps": -40.0,
        "dxy_pct": 2.0, "credit_bps": 80.0, "geo": 5.0,
    },
    "Rate Shock (+150 bps)": {
        "oil_pct": 0.0, "gold_pct": -8.0, "yield_bps": 150.0,
        "dxy_pct": 3.0, "credit_bps": 60.0, "geo": 0.0,
    },
    "Strait Closure (Hormuz)": {
        "oil_pct": 35.0, "gold_pct": 8.0, "yield_bps": 20.0,
        "dxy_pct": 0.5, "credit_bps": 90.0, "geo": 8.0,
    },
    "China Hard Landing": {
        "oil_pct": -18.0, "gold_pct": -5.0, "yield_bps": -30.0,
        "dxy_pct": 4.0, "credit_bps": 120.0, "geo": 4.0,
    },
    "Stagflation Scenario": {
        "oil_pct": 25.0, "gold_pct": 15.0, "yield_bps": 80.0,
        "dxy_pct": -2.0, "credit_bps": 100.0, "geo": 3.0,
    },
}


# ── Chart helpers ─────────────────────────────────────────────────────────

def _waterfall_chart(impact: dict[str, float], height: int = 420) -> go.Figure:
    assets_sorted = sorted(impact.items(), key=lambda x: x[1])
    names  = [a[0] for a in assets_sorted]
    values = [a[1] * 100 for a in assets_sorted]

    colors = [_RED if v < 0 else _GREEN for v in values]
    texts  = [f"{v:+.2f}%" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=texts,
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=10),
        hovertemplate="%{y}: %{x:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font=dict(family="DM Sans, sans-serif", color="#c8c8c8", size=11),
        xaxis=dict(
            title="Estimated Return Impact (%)",
            showgrid=True, gridcolor="#1e1e1e",
            zeroline=True, zerolinecolor="#2a2a2a", zerolinewidth=1.5,
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        margin=dict(l=140, r=80, t=24, b=40),
        bargap=0.25,
    )
    return fig


def _tail_risk_chart(
    var_es: dict[str, dict[str, float]],
    assets: list[str],
    height: int = 380,
) -> go.Figure:
    filtered = {a: var_es[a] for a in assets if a in var_es}
    if not filtered:
        return go.Figure()

    names  = list(filtered.keys())
    var95  = [filtered[a]["var95"] for a in names]
    var99  = [filtered[a]["var99"] for a in names]
    es95   = [filtered[a]["es95"]  for a in names]
    es99   = [filtered[a]["es99"]  for a in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="VaR 95%", x=names, y=var95,
                         marker_color="rgba(192,57,43,0.55)",
                         hovertemplate="%{x}<br>VaR 95%: %{y:.3f}%<extra></extra>"))
    fig.add_trace(go.Bar(name="VaR 99%", x=names, y=var99,
                         marker_color="rgba(192,57,43,0.85)",
                         hovertemplate="%{x}<br>VaR 99%: %{y:.3f}%<extra></extra>"))
    fig.add_trace(go.Scatter(name="ES 95%", x=names, y=es95, mode="lines+markers",
                             line=dict(color="#e67e22", width=2, dash="dot"),
                             marker=dict(size=5),
                             hovertemplate="%{x}<br>ES 95%: %{y:.3f}%<extra></extra>"))
    fig.add_trace(go.Scatter(name="ES 99%", x=names, y=es99, mode="lines+markers",
                             line=dict(color="#f39c12", width=2, dash="dash"),
                             marker=dict(size=5),
                             hovertemplate="%{x}<br>ES 99%: %{y:.3f}%<extra></extra>"))
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font=dict(family="DM Sans, sans-serif", color="#c8c8c8", size=11),
        xaxis=dict(tickangle=-35, showgrid=False),
        yaxis=dict(title="Loss (% of position)", showgrid=True, gridcolor="#1e1e1e"),
        legend=dict(orientation="h", y=-0.25, x=0,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=48, r=24, t=24, b=80),
        barmode="group",
        bargap=0.18,
        bargroupgap=0.05,
    )
    return fig


# ── Section label ─────────────────────────────────────────────────────────

def _section_label(text: str) -> None:
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:0.6rem;margin:1.6rem 0 0.6rem">
        <div style="width:3px;height:14px;background:{_GOLD};border-radius:2px;flex-shrink:0"></div>
        <span style="font-family:'DM Sans',sans-serif;font-size:0.6rem;font-weight:700;
        letter-spacing:0.15em;text-transform:uppercase;color:{_MUTED}">{text}</span>
        <div style="flex:1;height:1px;background:{_RULE}"></div></div>""",
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, delta: str = "", color: str = _GOLD) -> None:
    delta_html = (
        f'<span style="font-size:0.65rem;color:{color};margin-left:0.4rem">{delta}</span>'
        if delta else ""
    )
    st.markdown(
        f"""<div style="background:#1c1c1c;border:1px solid #2a2a2a;border-radius:4px;
        padding:0.7rem 0.9rem;margin-bottom:0.5rem">
        <div style="font-family:'DM Sans',sans-serif;font-size:0.55rem;font-weight:700;
        letter-spacing:0.14em;text-transform:uppercase;color:{_MUTED};margin-bottom:3px">{label}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.05rem;
        font-weight:700;color:{_GOLD}">{value}{delta_html}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _impact_table(impact: dict[str, float]) -> None:
    """Render a styled HTML table for the shock impact output."""
    rows = sorted(impact.items(), key=lambda x: x[1])
    rows_html = ""
    for asset, val in rows:
        pct = val * 100
        color = _RED if pct < 0 else _GREEN
        sign  = "+" if pct >= 0 else ""
        rows_html += (
            f'<tr>'
            f'<td style="padding:0.28rem 0.7rem;font-size:0.72rem;color:#c8c8c8">{asset}</td>'
            f'<td style="padding:0.28rem 0.7rem;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:0.72rem;color:{color};text-align:right">{sign}{pct:.3f}%</td>'
            f'</tr>'
        )
    st.markdown(
        f"""<div style="overflow:auto;border:1px solid #2a2a2a;border-radius:4px;margin-bottom:1rem">
        <table style="width:100%;border-collapse:collapse;background:#1c1c1c">
        <thead><tr style="border-bottom:1px solid #2a2a2a">
          <th style="padding:0.35rem 0.7rem;font-size:0.55rem;letter-spacing:0.12em;
          text-transform:uppercase;color:{_MUTED};text-align:left;font-weight:600">Asset</th>
          <th style="padding:0.35rem 0.7rem;font-size:0.55rem;letter-spacing:0.12em;
          text-transform:uppercase;color:{_MUTED};text-align:right;font-weight:600">Estimated Impact</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
        </table></div>""",
        unsafe_allow_html=True,
    )


def _var_table(var_es: dict[str, dict[str, float]], assets: list[str]) -> None:
    """Render VaR/ES table for selected assets."""
    rows_html = ""
    for a in assets:
        if a not in var_es:
            continue
        d = var_es[a]
        rows_html += (
            f'<tr>'
            f'<td style="padding:0.28rem 0.7rem;font-size:0.72rem;color:#c8c8c8">{a}</td>'
            f'<td style="padding:0.28rem 0.7rem;font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:{_RED};text-align:right">{d["var95"]:.3f}%</td>'
            f'<td style="padding:0.28rem 0.7rem;font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:{_RED};text-align:right">{d["var99"]:.3f}%</td>'
            f'<td style="padding:0.28rem 0.7rem;font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:#e67e22;text-align:right">{d["es95"]:.3f}%</td>'
            f'<td style="padding:0.28rem 0.7rem;font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:#f39c12;text-align:right">{d["es99"]:.3f}%</td>'
            f'</tr>'
        )
    st.markdown(
        f"""<div style="overflow:auto;border:1px solid #2a2a2a;border-radius:4px;margin-bottom:1rem">
        <table style="width:100%;border-collapse:collapse;background:#1c1c1c">
        <thead><tr style="border-bottom:1px solid #2a2a2a">
          <th style="padding:0.35rem 0.7rem;font-size:0.55rem;letter-spacing:0.12em;text-transform:uppercase;color:{_MUTED};text-align:left;font-weight:600">Asset</th>
          <th style="padding:0.35rem 0.7rem;font-size:0.55rem;letter-spacing:0.12em;text-transform:uppercase;color:{_RED};text-align:right;font-weight:600">VaR 95%</th>
          <th style="padding:0.35rem 0.7rem;font-size:0.55rem;letter-spacing:0.12em;text-transform:uppercase;color:{_RED};text-align:right;font-weight:600">VaR 99%</th>
          <th style="padding:0.35rem 0.7rem;font-size:0.55rem;letter-spacing:0.12em;text-transform:uppercase;color:#e67e22;text-align:right;font-weight:600">ES 95%</th>
          <th style="padding:0.35rem 0.7rem;font-size:0.55rem;letter-spacing:0.12em;text-transform:uppercase;color:#f39c12;text-align:right;font-weight:600">ES 99%</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
        </table></div>""",
        unsafe_allow_html=True,
    )


def _shock_badge(label: str, value: str, active: bool) -> str:
    color  = _GOLD if active else _MUTED
    border = f"1px solid {_GOLD}" if active else f"1px solid #2a2a2a"
    bg     = "#1e1a12" if active else "#1c1c1c"
    return (
        f'<span style="display:inline-block;background:{bg};border:{border};'
        f'border-radius:3px;padding:2px 8px;font-family:\'JetBrains Mono\',monospace;'
        f'font-size:0.68rem;color:{color};margin:2px 3px 2px 0">'
        f'{label}: {value}</span>'
    )


# ── Main page function ─────────────────────────────────────────────────────

def page_scenario_engine(
    start: str,
    end: str,
    fred_key: str | None = None,
) -> None:
    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        f'<h1 style="font-size:1.15rem;font-weight:700;color:{_GOLD};'
        f'letter-spacing:-0.01em;margin:0 0 0.2rem">Scenario Engine</h1>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "Forward-looking parametric shock propagation. Shock any combination of oil, gold, "
        "interest rates, the US dollar, credit spreads, or a geopolitical disruption factor "
        "and observe the estimated cross-asset spillover. VaR and Expected Shortfall at 95% "
        "and 99% are computed via historical simulation on the selected date window."
    )

    # ── Load data ─────────────────────────────────────────────────────────
    with st.spinner("Computing betas and historical tail risk…"):
        betas   = _compute_betas(start, end)
        var_es  = _compute_var_es(start, end)

    data_ok = bool(betas) and bool(var_es)
    if not data_ok:
        st.warning("Insufficient return data to compute betas. Expand the date range.")
        return

    # ── Scenario selector + controls ──────────────────────────────────────
    _section_label("Scenario Configuration")

    preset_name = st.selectbox(
        "Load preset scenario",
        list(_PRESETS.keys()),
        index=0,
        key="se_preset",
    )
    preset = _PRESETS.get(preset_name, {})

    col1, col2 = st.columns(2)
    with col1:
        oil_pct = st.slider(
            "Oil shock (%)",
            min_value=-60.0, max_value=80.0,
            value=float(preset.get("oil_pct", 0.0)),
            step=1.0, key="se_oil",
            help="Percentage change in WTI Crude Oil price.",
        )
        gold_pct = st.slider(
            "Gold shock (%)",
            min_value=-30.0, max_value=40.0,
            value=float(preset.get("gold_pct", 0.0)),
            step=0.5, key="se_gold",
            help="Percentage change in Gold price.",
        )
        yield_bps = st.slider(
            "Yield shock (bps)",
            min_value=-200, max_value=300,
            value=int(preset.get("yield_bps", 0)),
            step=5, key="se_yield",
            help="Parallel shift in 10-year yield (basis points).",
        )
    with col2:
        dxy_pct = st.slider(
            "DXY shock (%)",
            min_value=-10.0, max_value=12.0,
            value=float(preset.get("dxy_pct", 0.0)),
            step=0.5, key="se_dxy",
            help="Percentage change in the US Dollar Index.",
        )
        credit_bps = st.slider(
            "Credit spread shock (bps)",
            min_value=-100, max_value=400,
            value=int(preset.get("credit_bps", 0)),
            step=5, key="se_credit",
            help="Change in investment-grade credit spreads (basis points).",
        )
        geo = st.slider(
            "Geopolitical disruption factor",
            min_value=0.0, max_value=10.0,
            value=float(preset.get("geo", 0.0)),
            step=0.5, key="se_geo",
            help="Abstract disruption score (0 = neutral, 10 = severe crisis).",
        )

    # Active shocks summary badges
    badges = ""
    if oil_pct    != 0: badges += _shock_badge("Oil",    f"{oil_pct:+.0f}%",    True)
    if gold_pct   != 0: badges += _shock_badge("Gold",   f"{gold_pct:+.0f}%",   True)
    if yield_bps  != 0: badges += _shock_badge("Yields", f"{yield_bps:+d}bps",  True)
    if dxy_pct    != 0: badges += _shock_badge("DXY",    f"{dxy_pct:+.1f}%",    True)
    if credit_bps != 0: badges += _shock_badge("Credit", f"{credit_bps:+d}bps", True)
    if geo        != 0: badges += _shock_badge("Geo",    f"{geo:.1f}/10",        True)
    if not badges:
        badges = f'<span style="color:{_MUTED};font-size:0.70rem">No active shocks — adjust sliders above.</span>'

    st.markdown(
        f'<div style="margin:0.6rem 0 1.2rem;line-height:2">{badges}</div>',
        unsafe_allow_html=True,
    )

    # Zero-shock guard
    all_zero = (oil_pct == 0 and gold_pct == 0 and yield_bps == 0
                and dxy_pct == 0 and credit_bps == 0 and geo == 0)

    if all_zero:
        st.info("Set at least one shock parameter to run the propagation engine.")
        # Still show baseline VaR/ES
        _section_label("Baseline Tail Risk — Historical Simulation")
        _definition_block(
            "VaR & ES Methodology",
            "Historical simulation VaR: the loss not exceeded with probability p over one day, "
            "computed as the p-th percentile of the empirical return distribution. "
            "Expected Shortfall (ES): the mean loss in the tail beyond the VaR threshold. "
            "Both are reported as percentage of position. No distributional assumption is imposed."
        )
        tab_eq, tab_cmd = st.tabs(["Equities", "Commodities"])
        with tab_eq:
            _var_table(var_es, _EQUITY_TARGETS)
            _chart(_tail_risk_chart(var_es, _EQUITY_TARGETS, height=360))
        with tab_cmd:
            _var_table(var_es, _COMMODITY_TARGETS)
            _chart(_tail_risk_chart(var_es, _COMMODITY_TARGETS, height=360))
        return

    # ── Run propagation ───────────────────────────────────────────────────
    # Build shock dictionary for OLS-beta channel
    raw_shocks: dict[str, float] = {}
    if oil_pct  != 0: raw_shocks[_SHOCK_PROXY["oil"]]  = oil_pct  / 100
    if gold_pct != 0: raw_shocks[_SHOCK_PROXY["gold"]] = gold_pct / 100

    impact = _propagate_shock(betas, raw_shocks)

    # Add yield, DXY, credit, geo via fixed sensitivity tables
    fixed_shocks: dict[str, float] = {}
    if yield_bps  != 0: fixed_shocks["yield_bps"]  = float(yield_bps)
    if dxy_pct    != 0: fixed_shocks["dxy_pct"]    = dxy_pct
    if credit_bps != 0: fixed_shocks["credit_bps"] = float(credit_bps)
    if geo        != 0: fixed_shocks["geo"]         = geo

    impact = _apply_fixed_sensitivity(impact, fixed_shocks)

    # Scenario-adjusted VaR/ES
    total_shock_magnitude = (
        abs(oil_pct / 100) + abs(gold_pct / 100)
        + abs(yield_bps / 10000) * 50
        + abs(dxy_pct / 100)
        + abs(credit_bps / 10000) * 50
        + geo * 0.05
    )
    shocked_var_es = _parametric_var_es(impact, var_es, scale_factor=total_shock_magnitude)

    # ── Summary metrics ───────────────────────────────────────────────────
    _section_label("Scenario Impact Summary")

    all_impacts = list(impact.values())
    worst_asset  = min(impact, key=impact.get)
    best_asset   = max(impact, key=impact.get)
    avg_eq_impact = np.mean([impact.get(a, 0.0) for a in _EQUITY_TARGETS]) * 100
    avg_cm_impact = np.mean([impact.get(a, 0.0) for a in _COMMODITY_TARGETS]) * 100

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        color = _RED if avg_eq_impact < 0 else _GREEN
        _metric_card("Avg Equity Impact", f"{avg_eq_impact:+.2f}%", color=color)
    with m2:
        color = _RED if avg_cm_impact < 0 else _GREEN
        _metric_card("Avg Commodity Impact", f"{avg_cm_impact:+.2f}%", color=color)
    with m3:
        pct = impact[worst_asset] * 100
        _metric_card("Worst Hit Asset", f"{worst_asset}", f"{pct:+.2f}%", color=_RED)
    with m4:
        pct = impact[best_asset] * 100
        _metric_card("Best Positioned Asset", f"{best_asset}", f"{pct:+.2f}%", color=_GREEN)

    # ── Waterfall chart ───────────────────────────────────────────────────
    _section_label("Cross-Asset Shock Propagation — Estimated Returns")
    _chart(_waterfall_chart(impact, height=480))

    st.markdown(
        f'<p style="font-size:0.65rem;color:{_MUTED};margin:0.2rem 0 1.2rem">'
        f'Equity betas estimated via OLS regression on historical log-returns. '
        f'Yield, DXY, credit, and geopolitical channels use calibrated sensitivity factors '
        f'drawn from the empirical cross-asset literature. All estimates are per-period approximations '
        f'and assume no second-round feedback.'
        f'</p>',
        unsafe_allow_html=True,
    )

    # ── Detailed impact table ─────────────────────────────────────────────
    _section_label("Asset-Level Propagation Detail")
    tab_eq, tab_cmd = st.tabs(["Equities", "Commodities"])
    with tab_eq:
        _impact_table({a: impact[a] for a in _EQUITY_TARGETS if a in impact})
    with tab_cmd:
        _impact_table({a: impact[a] for a in _COMMODITY_TARGETS if a in impact})

    # ── Tail risk under shock ─────────────────────────────────────────────
    _section_label("Tail Risk Under Scenario — VaR & Expected Shortfall")
    _definition_block(
        "Shocked VaR & ES",
        "Baseline VaR/ES is from historical simulation on the selected window. Under a scenario, "
        "tail risk is amplified proportionally to the total shock magnitude across all channels. "
        "This is a parametric scaling approach — not a full Monte Carlo reprice. "
        "For severe dislocations (total shock > 3 sigma), treat as an order-of-magnitude estimate."
    )

    tab_eq2, tab_cmd2 = st.tabs(["Equities", "Commodities"])
    with tab_eq2:
        _var_table(shocked_var_es, _EQUITY_TARGETS)
        _chart(_tail_risk_chart(shocked_var_es, _EQUITY_TARGETS, height=360))
    with tab_cmd2:
        _var_table(shocked_var_es, _COMMODITY_TARGETS)
        _chart(_tail_risk_chart(shocked_var_es, _COMMODITY_TARGETS, height=360))

    # ── Baseline comparison ───────────────────────────────────────────────
    with st.expander("Baseline VaR/ES (no shock)", expanded=False):
        _section_note(
            "Historical simulation on the full selected date window with no scenario applied. "
            "Use this as the pre-shock reference to understand incremental tail risk from the scenario."
        )
        tab_b_eq, tab_b_cmd = st.tabs(["Equities", "Commodities"])
        with tab_b_eq:
            _var_table(var_es, _EQUITY_TARGETS)
        with tab_b_cmd:
            _var_table(var_es, _COMMODITY_TARGETS)

    # ── Model guardrails ──────────────────────────────────────────────────
    _section_label("Model Assumptions & Guardrails")
    st.markdown(
        f"""<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-bottom:1rem">
        {"".join([
            f'<div style="background:#1c1c1c;border:1px solid #2a2a2a;border-radius:4px;'
            f'padding:0.6rem 0.85rem">'
            f'<div style="font-size:0.55rem;font-weight:700;letter-spacing:0.12em;'
            f'text-transform:uppercase;color:{_MUTED};margin-bottom:4px">{title}</div>'
            f'<div style="font-size:0.70rem;color:#b8b8b8;line-height:1.65">{body}</div>'
            f'</div>'
            for title, body in [
                ("Linearity", "Beta propagation assumes linear relationships. Non-linearity and feedback loops are not modelled — impacts can be materially larger in tail events."),
                ("Stationarity", "Betas are estimated on the full historical window. Regime shifts (e.g., post-2022 inflation) may have altered sensitivities significantly."),
                ("Single-period", "All estimates are single-day or single-period. Cumulative path effects and multi-day drift are not captured."),
                ("Correlation stability", "Cross-asset correlations are assumed constant. In stress periods, correlations typically spike toward +1, amplifying portfolio losses."),
                ("No liquidity cost", "Market impact, bid-ask widening, and forced liquidation costs are excluded. Actual realised losses in stress are likely larger."),
                ("Geo factor", "The geopolitical disruption score uses calibrated sensitivities from the 2022 Ukraine/energy crisis analog. Other crises may have different transmission paths."),
            ]
        ])}
        </div>""",
        unsafe_allow_html=True,
    )

    # ── AI Stress Engineer ────────────────────────────────────────────────
    _section_label("AI Stress Engineer Assessment")

    provider = st.session_state.get("ai_provider")
    api_key  = st.session_state.get("api_key", "")

    if provider and api_key and is_enabled("stress_engineer"):
        scenario_desc = preset_name if preset_name != "Custom" else "custom scenario"
        shock_parts = []
        if oil_pct    != 0: shock_parts.append(f"oil {oil_pct:+.0f}%")
        if gold_pct   != 0: shock_parts.append(f"gold {gold_pct:+.0f}%")
        if yield_bps  != 0: shock_parts.append(f"yields {yield_bps:+d}bps")
        if dxy_pct    != 0: shock_parts.append(f"DXY {dxy_pct:+.1f}%")
        if credit_bps != 0: shock_parts.append(f"credit {credit_bps:+d}bps")
        if geo        != 0: shock_parts.append(f"geo-factor {geo:.1f}/10")
        transmission = ", ".join(shock_parts) or "no active shocks"

        _se_ctx = {
            "scenarios": [{
                "name":        scenario_desc,
                "shock_type":  transmission,
                "magnitude":   total_shock_magnitude,
                "impact_pct":  round(avg_eq_impact, 2),
                "transmission": transmission,
            }],
            "worst_scenario": f"{worst_asset} ({scenario_desc})",
            "worst_impact":   round(impact[worst_asset] * 100, 2),
            "avg_impact":     round((avg_eq_impact + avg_cm_impact) / 2, 2),
            "n_scenarios":    1,
            "regime_name":    st.session_state.get("current_regime", "unknown"),
            "risk_score":     None,
        }
        with st.spinner("Stress Engineer analysing scenario…"):
            _se_result = _run_stress_engineer(_se_ctx, provider, api_key)
        render_agent_output_block("stress_engineer", _se_result)
    elif not provider:
        st.markdown(
            f'<p style="font-size:0.70rem;color:{_MUTED};margin:0.4rem 0">'
            f'Add an API key in Controls to activate AI scenario commentary.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<p style="font-size:0.70rem;color:{_MUTED};margin:0.4rem 0">'
            f'Stress Engineer agent is disabled. Enable it in the AI Workforce panel.</p>',
            unsafe_allow_html=True,
        )

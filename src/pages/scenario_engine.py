"""
Scenario Engine - Forward-Looking Parametric Shock Propagation
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
    _definition_block, _takeaway_block, _page_header, _page_footer,
)
from src.ui.agent_panel import render_agent_output_block
from src.agents.stress_engineer import run as _run_stress_engineer
from src.analysis.agent_state import is_enabled

_GOLD  = "#CFB991"
_RED   = "#c0392b"
_GREEN = "#2e7d32"
_MUTED = "#8890a1"
_BG    = "#080808"
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
def _compute_regime_betas(start: str, end: str) -> dict[int, dict[str, dict[str, float]]]:
    """
    Regime-conditional OLS betas: separate beta estimates for each correlation regime.

    Returns {regime_id: {proxy_name: {target_name: beta}}}
    where regime_id in {0=Decorrelated, 1=Normal, 2=Elevated, 3=Crisis}.

    Key insight (GAP 19): beta(oil→S&P) in a supply-shock regime is materially
    different from the full-sample unconditional beta.
    - In Elevated/Crisis regime: commodity shocks transmit harder into equities
      (forced liquidations, risk-off amplification, correlated margin calls).
    - In Decorrelated/Normal regime: commodity shocks are absorbed by rotation,
      not amplified across asset classes.
    Using regime-appropriate betas gives materially different shock impacts.
    """
    from src.analysis.correlations import average_cross_corr_series, detect_correlation_regime
    try:
        eq_r, cmd_r = load_returns(start, end)
        combined = load_combined_returns(start, end)
        if combined.empty or eq_r.empty or cmd_r.empty:
            return {}

        avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
        regimes  = detect_correlation_regime(avg_corr)

        result: dict[int, dict[str, dict[str, float]]] = {}
        targets = _EQUITY_TARGETS + _COMMODITY_TARGETS

        for regime_id in range(4):
            regime_dates = regimes[regimes == regime_id].index
            subset = combined.loc[combined.index.intersection(regime_dates)]
            if len(subset) < 30:
                continue

            regime_betas: dict[str, dict[str, float]] = {}
            for factor_key, proxy_name in _SHOCK_PROXY.items():
                if proxy_name not in subset.columns:
                    continue
                proxy_r = subset[proxy_name].dropna()
                factor_betas: dict[str, float] = {}
                for t in targets:
                    if t not in subset.columns or t == proxy_name:
                        continue
                    aligned = pd.concat([proxy_r, subset[t]], axis=1).dropna()
                    if len(aligned) < 20:
                        continue
                    x = aligned.iloc[:, 0].values
                    y = aligned.iloc[:, 1].values
                    var_x = np.var(x, ddof=1)
                    if var_x < 1e-12:
                        continue
                    cov_xy = np.cov(x, y, ddof=1)[0, 1]
                    factor_betas[t] = float(cov_xy / var_x)
                regime_betas[proxy_name] = factor_betas
            result[regime_id] = regime_betas

        return result
    except Exception:
        return {}


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
    Returns {asset_name: estimated_return_pct} - negative = loss.
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


_GDP_SENSITIVITY: dict[str, float] = {
    # Sensitivity to a +1% GDP growth shock (positive = beneficiary of growth)
    "S&P 500":       0.040,
    "Nasdaq 100":    0.050,
    "DJIA":          0.030,
    "Russell 2000":  0.045,
    "FTSE 100":      0.028,
    "DAX":           0.032,
    "Eurostoxx 50":  0.030,
    "Nikkei 225":    0.025,
    "Hang Seng":     0.022,
    "Sensex":        0.020,
    "WTI Crude Oil": 0.022,
    "Brent Crude":   0.020,
    "Natural Gas":   0.010,
    "Gold":         -0.015,
    "Silver":       -0.008,
    "Copper":        0.035,
    "Wheat":         0.006,
    "Corn":          0.006,
}

_CPI_SENSITIVITY: dict[str, float] = {
    # Sensitivity to a +1% unexpected CPI surprise (negative = inflation hurts)
    "S&P 500":      -0.025,
    "Nasdaq 100":   -0.040,
    "DJIA":         -0.015,
    "Russell 2000": -0.020,
    "FTSE 100":     -0.015,
    "DAX":          -0.018,
    "Eurostoxx 50": -0.018,
    "Nikkei 225":   -0.010,
    "Hang Seng":    -0.012,
    "Sensex":       -0.015,
    "Gold":          0.020,
    "Silver":        0.015,
    "Copper":        0.012,
    "WTI Crude Oil": 0.025,
    "Brent Crude":   0.025,
    "Natural Gas":   0.010,
    "Wheat":         0.015,
    "Corn":          0.015,
}

_UNEMP_SENSITIVITY: dict[str, float] = {
    # Sensitivity to +1pp unemployment rate shock (negative = hurt by rising unemployment)
    "S&P 500":      -0.030,
    "Nasdaq 100":   -0.025,
    "DJIA":         -0.035,
    "Russell 2000": -0.045,
    "FTSE 100":     -0.020,
    "DAX":          -0.025,
    "Eurostoxx 50": -0.022,
    "Nikkei 225":   -0.018,
    "Hang Seng":    -0.015,
    "Sensex":       -0.012,
    "Gold":          0.015,
    "Silver":        0.005,
    "Copper":       -0.030,
    "WTI Crude Oil": -0.020,
    "Brent Crude":  -0.018,
    "Natural Gas":  -0.008,
    "Wheat":        -0.005,
    "Corn":         -0.005,
}


def _apply_fixed_sensitivity(
    impact: dict[str, float],
    shocks: dict[str, float],
) -> dict[str, float]:
    """Add yield, DXY, credit, geo, and macro contributions using fixed sensitivity tables."""
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
        if "gdp_pct" in shocks:
            impact[t] = impact.get(t, 0.0) + _GDP_SENSITIVITY.get(t, 0.0) * shocks["gdp_pct"]
        if "cpi_pct" in shocks:
            impact[t] = impact.get(t, 0.0) + _CPI_SENSITIVITY.get(t, 0.0) * shocks["cpi_pct"]
        if "unemp_pct" in shocks:
            impact[t] = impact.get(t, 0.0) + _UNEMP_SENSITIVITY.get(t, 0.0) * shocks["unemp_pct"]
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
        <div style="width:3px;height:14px;background:{_GOLD};;flex-shrink:0"></div>
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
        f"""<div style="background:#080808;border:1px solid #2a2a2a;border-radius:0;
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
        f"""<div style="overflow:auto;border:1px solid #2a2a2a;border-radius:0;margin-bottom:1rem">
        <table style="width:100%;border-collapse:collapse;background:#080808">
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
        f"""<div style="overflow:auto;border:1px solid #2a2a2a;border-radius:0;margin-bottom:1rem">
        <table style="width:100%;border-collapse:collapse;background:#080808">
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


def _scenario_comparison_chart(
    impacts: dict[str, dict[str, float]],
    asset_group: list[str],
    height: int = 420,
) -> go.Figure:
    """Grouped bar chart comparing impact across multiple scenarios for a set of assets."""
    scenario_names = list(impacts.keys())
    # palette cycling for up to 8 scenarios
    palette = ["#CFB991", "#c0392b", "#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#16a085", "#e74c3c"]
    fig = go.Figure()
    for i, name in enumerate(scenario_names):
        impact_row = impacts[name]
        vals = [impact_row.get(a, 0.0) * 100 for a in asset_group]
        fig.add_trace(go.Bar(
            name=name,
            x=asset_group,
            y=vals,
            marker_color=palette[i % len(palette)],
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.2f}}%<extra></extra>",
            opacity=0.85,
        ))
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font=dict(family="DM Sans, sans-serif", color="#c8c8c8", size=11),
        xaxis=dict(tickangle=-35, showgrid=False),
        yaxis=dict(title="Estimated Return Impact (%)", showgrid=True, gridcolor="#1e1e1e",
                   zeroline=True, zerolinecolor="#2a2a2a"),
        legend=dict(orientation="h", y=-0.30, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=48, r=24, t=24, b=100),
        barmode="group",
        bargap=0.15,
        bargroupgap=0.04,
    )
    return fig


def _comparison_table(impacts: dict[str, dict[str, float]], assets: list[str]) -> None:
    """Render multi-scenario comparison as an HTML table."""
    scenario_names = list(impacts.keys())
    palette = ["#CFB991", "#c0392b", "#2980b9", "#27ae60", "#e67e22", "#8e44ad"]

    header_cells = (
        '<th style="padding:0.35rem 0.6rem;font-size:0.55rem;letter-spacing:.12em;'
        'text-transform:uppercase;color:#8890a1;text-align:left;font-weight:600">Asset</th>'
    )
    for i, sname in enumerate(scenario_names):
        col_c = palette[i % len(palette)]
        header_cells += (
            f'<th style="padding:0.35rem 0.6rem;font-size:0.55rem;letter-spacing:.10em;'
            f'text-transform:uppercase;color:{col_c};text-align:right;font-weight:600">'
            f'{sname[:22]}{"…" if len(sname) > 22 else ""}</th>'
        )

    rows_html = ""
    for a in assets:
        row = f'<td style="padding:0.28rem 0.6rem;font-size:0.70rem;color:#c8c8c8">{a}</td>'
        for i, sname in enumerate(scenario_names):
            pct = impacts[sname].get(a, 0.0) * 100
            col_c = _RED if pct < 0 else _GREEN
            sign  = "+" if pct >= 0 else ""
            row += (
                f'<td style="padding:0.28rem 0.6rem;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:0.70rem;color:{col_c};text-align:right">{sign}{pct:.2f}%</td>'
            )
        rows_html += f"<tr>{row}</tr>"

    st.markdown(
        f"""<div style="overflow:auto;border:1px solid #2a2a2a;border-radius:0;margin-bottom:1rem">
        <table style="width:100%;border-collapse:collapse;background:#080808">
        <thead><tr style="border-bottom:1px solid #2a2a2a">{header_cells}</tr></thead>
        <tbody>{rows_html}</tbody>
        </table></div>""",
        unsafe_allow_html=True,
    )


def _shock_badge(label: str, value: str, active: bool) -> str:
    color  = _GOLD if active else _MUTED
    border = f"1px solid {_GOLD}" if active else f"1px solid #2a2a2a"
    bg     = "#1e1a12" if active else "#080808"
    return (
        f'<span style="display:inline-block;background:{bg};border:{border};'
        f'border-radius:0;padding:2px 8px;font-family:\'JetBrains Mono\',monospace;'
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
    _page_header("Geopolitical Scenario Simulator",
                 "Geopolitical scenario lens · Geo multiplier · Vol multiplier · Safe-haven / short-bias overlays")
    _page_intro(
        "Forward-looking parametric shock propagation. Shock any combination of oil, gold, "
        "interest rates, the US dollar, credit spreads, or a geopolitical disruption factor "
        "and observe the estimated cross-asset spillover. VaR and Expected Shortfall at 95% "
        "and 99% are computed via historical simulation on the selected date window."
    )

    # ── Geo risk context banner ────────────────────────────────────────────
    try:
        from src.analysis.conflict_model import score_all_conflicts, aggregate_portfolio_scores
        from src.analysis.scenario_state import get_scenario
        _se_cr  = score_all_conflicts()
        _se_agg = aggregate_portfolio_scores(_se_cr)
        _se_cis = _se_agg.get("portfolio_cis", 50.0)
        _se_tps = _se_agg.get("portfolio_tps", 50.0)
        _se_ss  = get_scenario()
        _se_scenario = _se_ss.get("label", "Base Case")
        _se_geo_mult = _se_ss.get("geo_mult", 1.0)
        _se_active = [r for r in _se_cr.values() if r.get("state") == "active"]
        _se_col = "#c0392b" if _se_cis >= 65 else "#e67e22" if _se_cis >= 45 else "#CFB991"
        _se_conflict_tags = "".join(
            f'<span style="background:#0a0a1a;color:#8E9AAA;'
            f'font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'padding:2px 6px;margin-right:4px;border:1px solid #2a2a2a">'
            f'{r["label"].upper()}</span>'
            for r in sorted(_se_active, key=lambda x: x["cis"], reverse=True)[:3]
        )
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;'
            f'border-left:3px solid {_se_col};padding:.4rem .9rem;'
            f'margin-bottom:.6rem;display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'font-weight:700;color:{_se_col};white-space:nowrap">LIVE GEO INPUT</span>'
            f'{_se_conflict_tags}'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'color:#8E9AAA;margin-left:auto">'
            f'Scenario&nbsp;<b style="color:#CFB991">{_se_scenario}</b>&nbsp;·&nbsp;'
            f'Geo×<b style="color:{_se_col}">{_se_geo_mult:.2f}</b>&nbsp;·&nbsp;'
            f'CIS&nbsp;<b style="color:{_se_col}">{_se_cis:.0f}</b>&nbsp;·&nbsp;'
            f'TPS&nbsp;<b style="color:#CFB991">{_se_tps:.0f}</b>&nbsp;·&nbsp;'
            f'Use Geo Disruption slider to simulate current conflict intensity</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    # ── Load data ─────────────────────────────────────────────────────────
    with st.spinner("Computing betas and historical tail risk…"):
        betas        = _compute_betas(start, end)
        regime_betas = _compute_regime_betas(start, end)
        var_es       = _compute_var_es(start, end)

    data_ok = bool(betas) and bool(var_es)
    if not data_ok:
        st.warning("Insufficient return data to compute betas. Expand the date range.")
        return

    # Detect current correlation regime and pick regime-conditional betas
    _current_regime = st.session_state.get("current_regime", 1)
    try:
        _current_regime = int(_current_regime)
    except (TypeError, ValueError):
        _current_regime = 1

    _REGIME_NAMES = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
    _regime_label = _REGIME_NAMES.get(_current_regime, "Normal")
    _active_betas = regime_betas.get(_current_regime, betas)  # fall back to unconditional
    _using_regime_betas = _current_regime in regime_betas and bool(_active_betas)

    # ── Scenario selector + controls ──────────────────────────────────────
    _section_label("Scenario Configuration")

    preset_name = st.selectbox(
        "Load preset scenario",
        list(_PRESETS.keys()),
        index=0,
        key="se_preset",
    )
    preset = _PRESETS.get(preset_name, {})
    is_custom = preset_name == "Custom"

    if is_custom:
        custom_name = st.text_input(
            "Custom scenario name",
            value=st.session_state.get("se_custom_name", "My Custom Scenario"),
            key="se_custom_name",
            help="Label for your custom scenario — shown in comparison views.",
        )

    col1, col2 = st.columns(2)
    with col1:
        oil_pct = st.slider(
            "Oil shock (%)",
            min_value=-60.0, max_value=80.0,
            value=float(preset.get("oil_pct", 0.0)) if not is_custom
                  else float(st.session_state.get("_cust_oil", 0.0)),
            step=1.0, key="se_oil",
            help="Percentage change in WTI Crude Oil price.",
        )
        gold_pct = st.slider(
            "Gold shock (%)",
            min_value=-30.0, max_value=40.0,
            value=float(preset.get("gold_pct", 0.0)) if not is_custom
                  else float(st.session_state.get("_cust_gold", 0.0)),
            step=0.5, key="se_gold",
            help="Percentage change in Gold price.",
        )
        yield_bps = st.slider(
            "Yield shock (bps)",
            min_value=-200, max_value=300,
            value=int(preset.get("yield_bps", 0)) if not is_custom
                  else int(st.session_state.get("_cust_yield", 0)),
            step=5, key="se_yield",
            help="Parallel shift in 10-year yield (basis points).",
        )
    with col2:
        dxy_pct = st.slider(
            "DXY shock (%)",
            min_value=-10.0, max_value=12.0,
            value=float(preset.get("dxy_pct", 0.0)) if not is_custom
                  else float(st.session_state.get("_cust_dxy", 0.0)),
            step=0.5, key="se_dxy",
            help="Percentage change in the US Dollar Index.",
        )
        credit_bps = st.slider(
            "Credit spread shock (bps)",
            min_value=-100, max_value=400,
            value=int(preset.get("credit_bps", 0)) if not is_custom
                  else int(st.session_state.get("_cust_credit", 0)),
            step=5, key="se_credit",
            help="Change in investment-grade credit spreads (basis points).",
        )
        geo = st.slider(
            "Geopolitical disruption factor",
            min_value=0.0, max_value=10.0,
            value=float(preset.get("geo", 0.0)) if not is_custom
                  else float(st.session_state.get("_cust_geo", 0.0)),
            step=0.5, key="se_geo",
            help="Abstract disruption score (0 = neutral, 10 = severe crisis).",
        )

    # ── Custom macro variables (only in Custom mode) ──────────────────────
    natgas_pct = 0.0
    copper_pct = 0.0
    gdp_pct    = 0.0
    cpi_pct    = 0.0
    unemp_pct  = 0.0

    if is_custom:
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;border-left:3px solid {_GOLD};'
            f'padding:.5rem .9rem;margin:.8rem 0 .4rem">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.60rem;font-weight:700;'
            f'letter-spacing:.14em;text-transform:uppercase;color:{_GOLD}">Custom Macro Variables</span>'
            f'<p style="font-size:0.68rem;color:#8890a1;margin:.3rem 0 0;line-height:1.5">'
            f'Extend the scenario with commodity and macroeconomic shocks unavailable in presets. '
            f'GDP, CPI, and unemployment sensitivities are calibrated to cross-asset empirical literature.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        cm1, cm2 = st.columns(2)
        with cm1:
            natgas_pct = st.slider(
                "Natural Gas shock (%)",
                min_value=-60.0, max_value=100.0,
                value=float(st.session_state.get("_cust_natgas", 0.0)),
                step=1.0, key="se_natgas",
                help="Percentage change in Henry Hub Natural Gas price.",
            )
            gdp_pct = st.slider(
                "GDP growth shock (%)",
                min_value=-5.0, max_value=3.0,
                value=float(st.session_state.get("_cust_gdp", 0.0)),
                step=0.1, key="se_gdp",
                help="Annualized GDP growth surprise relative to consensus (negative = recession shock).",
            )
            cpi_pct = st.slider(
                "CPI surprise (%)",
                min_value=-3.0, max_value=6.0,
                value=float(st.session_state.get("_cust_cpi", 0.0)),
                step=0.1, key="se_cpi",
                help="Unexpected CPI deviation from consensus (positive = inflation surprise).",
            )
        with cm2:
            copper_pct = st.slider(
                "Copper shock (%)",
                min_value=-40.0, max_value=50.0,
                value=float(st.session_state.get("_cust_copper", 0.0)),
                step=1.0, key="se_copper",
                help="Percentage change in Copper price (industrial demand proxy).",
            )
            unemp_pct = st.slider(
                "Unemployment shock (pp)",
                min_value=-2.0, max_value=5.0,
                value=float(st.session_state.get("_cust_unemp", 0.0)),
                step=0.1, key="se_unemp",
                help="Change in unemployment rate in percentage points (positive = rising unemployment).",
            )

        # Persist custom values to session_state
        for k, v in [
            ("_cust_oil", oil_pct), ("_cust_gold", gold_pct), ("_cust_yield", yield_bps),
            ("_cust_dxy", dxy_pct), ("_cust_credit", credit_bps), ("_cust_geo", geo),
            ("_cust_natgas", natgas_pct), ("_cust_copper", copper_pct),
            ("_cust_gdp", gdp_pct), ("_cust_cpi", cpi_pct), ("_cust_unemp", unemp_pct),
        ]:
            st.session_state[k] = v

    # Active shocks summary badges
    badges = ""
    if oil_pct    != 0: badges += _shock_badge("Oil",       f"{oil_pct:+.0f}%",    True)
    if gold_pct   != 0: badges += _shock_badge("Gold",      f"{gold_pct:+.0f}%",   True)
    if yield_bps  != 0: badges += _shock_badge("Yields",    f"{yield_bps:+d}bps",  True)
    if dxy_pct    != 0: badges += _shock_badge("DXY",       f"{dxy_pct:+.1f}%",    True)
    if credit_bps != 0: badges += _shock_badge("Credit",    f"{credit_bps:+d}bps", True)
    if geo        != 0: badges += _shock_badge("Geo",       f"{geo:.1f}/10",        True)
    if natgas_pct != 0: badges += _shock_badge("Nat Gas",   f"{natgas_pct:+.0f}%", True)
    if copper_pct != 0: badges += _shock_badge("Copper",    f"{copper_pct:+.0f}%", True)
    if gdp_pct    != 0: badges += _shock_badge("GDP",       f"{gdp_pct:+.1f}%",    True)
    if cpi_pct    != 0: badges += _shock_badge("CPI",       f"{cpi_pct:+.1f}%",    True)
    if unemp_pct  != 0: badges += _shock_badge("Unemp",     f"{unemp_pct:+.1f}pp", True)
    if not badges:
        badges = f'<span style="color:{_MUTED};font-size:0.70rem">No active shocks — adjust sliders above.</span>'

    st.markdown(
        f'<div style="margin:0.6rem 0 1.2rem;line-height:2">{badges}</div>',
        unsafe_allow_html=True,
    )

    # Zero-shock guard
    all_zero = (oil_pct == 0 and gold_pct == 0 and yield_bps == 0
                and dxy_pct == 0 and credit_bps == 0 and geo == 0
                and natgas_pct == 0 and copper_pct == 0
                and gdp_pct == 0 and cpi_pct == 0 and unemp_pct == 0)

    if all_zero:
        st.info("Set at least one shock parameter to run the propagation engine.")
        # Still show baseline VaR/ES
        _section_label("Baseline Tail Risk - Historical Simulation")
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
    if oil_pct    != 0: raw_shocks[_SHOCK_PROXY["oil"]]    = oil_pct    / 100
    if gold_pct   != 0: raw_shocks[_SHOCK_PROXY["gold"]]   = gold_pct   / 100
    if natgas_pct != 0: raw_shocks[_SHOCK_PROXY["natgas"]] = natgas_pct / 100
    if copper_pct != 0: raw_shocks[_SHOCK_PROXY["copper"]] = copper_pct / 100

    # GAP 19: use regime-conditional betas instead of unconditional OLS
    # In Elevated/Crisis regimes, betas are empirically higher (amplified transmission)
    impact = _propagate_shock(_active_betas, raw_shocks)

    # Show which beta set is being applied
    if _using_regime_betas:
        _beta_note_color = {"Decorrelated": "#27ae60", "Normal": "#CFB991",
                            "Elevated": "#e67e22", "Crisis": "#e74c3c"}.get(_regime_label, "#CFB991")
        st.markdown(
            f'<div style="background:#0a0a0a;border-left:2px solid {_beta_note_color};'
            f'border-radius:3px;padding:6px 10px;margin:4px 0;">'
            f'<span style="color:{_beta_note_color};font-family:\'JetBrains Mono\',monospace;'
            f'font-size:10px;font-weight:700;">REGIME BETAS ACTIVE — {_regime_label.upper()}</span>'
            f'<span style="color:#777;font-family:\'JetBrains Mono\',monospace;font-size:10px;"> · '
            f'Betas estimated on {_regime_label} regime periods only (not full-sample OLS). '
            f'Unconditional betas are available in tail risk tab.</span></div>',
            unsafe_allow_html=True,
        )

    # Add yield, DXY, credit, geo, and macro via fixed sensitivity tables
    fixed_shocks: dict[str, float] = {}
    if yield_bps  != 0: fixed_shocks["yield_bps"]  = float(yield_bps)
    if dxy_pct    != 0: fixed_shocks["dxy_pct"]    = dxy_pct
    if credit_bps != 0: fixed_shocks["credit_bps"] = float(credit_bps)
    if geo        != 0: fixed_shocks["geo"]         = geo
    if gdp_pct    != 0: fixed_shocks["gdp_pct"]    = gdp_pct
    if cpi_pct    != 0: fixed_shocks["cpi_pct"]    = cpi_pct
    if unemp_pct  != 0: fixed_shocks["unemp_pct"]  = unemp_pct

    impact = _apply_fixed_sensitivity(impact, fixed_shocks)

    # Scenario-adjusted VaR/ES
    total_shock_magnitude = (
        abs(oil_pct / 100) + abs(gold_pct / 100)
        + abs(yield_bps / 10000) * 50
        + abs(dxy_pct / 100)
        + abs(credit_bps / 10000) * 50
        + geo * 0.05
        + abs(natgas_pct / 100) * 0.5
        + abs(copper_pct / 100) * 0.5
        + abs(gdp_pct) * 0.3
        + abs(cpi_pct) * 0.2
        + abs(unemp_pct) * 0.15
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
    _section_label("Cross-Asset Shock Propagation - Estimated Returns")
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
    _section_label("Tail Risk Under Scenario - VaR & Expected Shortfall")
    _definition_block(
        "Shocked VaR & ES",
        "Baseline VaR/ES is from historical simulation on the selected window. Under a scenario, "
        "tail risk is amplified proportionally to the total shock magnitude across all channels. "
        "This is a parametric scaling approach - not a full Monte Carlo reprice. "
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
            f'<div style="background:#080808;border:1px solid #2a2a2a;border-radius:0;'
            f'padding:0.6rem 0.85rem">'
            f'<div style="font-size:0.55rem;font-weight:700;letter-spacing:0.12em;'
            f'text-transform:uppercase;color:{_MUTED};margin-bottom:4px">{title}</div>'
            f'<div style="font-size:0.70rem;color:#b8b8b8;line-height:1.65">{body}</div>'
            f'</div>'
            for title, body in [
                ("Linearity", "Beta propagation assumes linear relationships. Non-linearity and feedback loops are not modelled - impacts can be materially larger in tail events."),
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
        scenario_desc = (
            st.session_state.get("se_custom_name", "Custom Scenario")
            if preset_name == "Custom" else preset_name
        )
        shock_parts = []
        if oil_pct    != 0: shock_parts.append(f"oil {oil_pct:+.0f}%")
        if gold_pct   != 0: shock_parts.append(f"gold {gold_pct:+.0f}%")
        if yield_bps  != 0: shock_parts.append(f"yields {yield_bps:+d}bps")
        if dxy_pct    != 0: shock_parts.append(f"DXY {dxy_pct:+.1f}%")
        if credit_bps != 0: shock_parts.append(f"credit {credit_bps:+d}bps")
        if geo        != 0: shock_parts.append(f"geo-factor {geo:.1f}/10")
        if natgas_pct != 0: shock_parts.append(f"nat-gas {natgas_pct:+.0f}%")
        if copper_pct != 0: shock_parts.append(f"copper {copper_pct:+.0f}%")
        if gdp_pct    != 0: shock_parts.append(f"GDP {gdp_pct:+.1f}%")
        if cpi_pct    != 0: shock_parts.append(f"CPI {cpi_pct:+.1f}%")
        if unemp_pct  != 0: shock_parts.append(f"unemployment {unemp_pct:+.1f}pp")
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
    # ── Multi-Scenario Comparison ─────────────────────────────────────────
    _section_label("Multi-Scenario Comparison")
    st.markdown(
        f'<p style="font-size:0.68rem;color:{_MUTED};margin:0.1rem 0 0.7rem;line-height:1.55">'
        f'Select 2–6 preset scenarios to compare estimated cross-asset impacts side-by-side. '
        f'Betas are shared across scenarios; only shock inputs differ.</p>',
        unsafe_allow_html=True,
    )

    preset_options = [k for k in _PRESETS.keys() if k != "Custom"]
    compare_selected = st.multiselect(
        "Scenarios to compare",
        preset_options,
        default=preset_options[:3],
        key="se_compare",
        help="Select multiple preset scenarios to view side-by-side impact on equities and commodities.",
    )

    if len(compare_selected) >= 2:
        # Compute impact for each selected preset
        compare_impacts: dict[str, dict[str, float]] = {}
        for sc_name in compare_selected:
            sc = _PRESETS[sc_name]
            sc_raw: dict[str, float] = {}
            if sc.get("oil_pct", 0)  != 0: sc_raw[_SHOCK_PROXY["oil"]]  = sc["oil_pct"]  / 100
            if sc.get("gold_pct", 0) != 0: sc_raw[_SHOCK_PROXY["gold"]] = sc["gold_pct"] / 100
            sc_imp = _propagate_shock(betas, sc_raw)
            sc_fixed: dict[str, float] = {}
            if sc.get("yield_bps",  0) != 0: sc_fixed["yield_bps"]  = float(sc["yield_bps"])
            if sc.get("dxy_pct",    0) != 0: sc_fixed["dxy_pct"]    = sc["dxy_pct"]
            if sc.get("credit_bps", 0) != 0: sc_fixed["credit_bps"] = float(sc["credit_bps"])
            if sc.get("geo",        0) != 0: sc_fixed["geo"]         = sc["geo"]
            sc_imp = _apply_fixed_sensitivity(sc_imp, sc_fixed)
            compare_impacts[sc_name] = sc_imp

        # Also include current custom scenario if active
        if is_custom and not all_zero:
            cust_label = st.session_state.get("se_custom_name", "Custom Scenario")
            compare_impacts[cust_label] = impact

        tab_cmp_eq, tab_cmp_cm = st.tabs(["Equities", "Commodities"])
        with tab_cmp_eq:
            _comparison_table(compare_impacts, _EQUITY_TARGETS)
            _chart(_scenario_comparison_chart(compare_impacts, _EQUITY_TARGETS, height=400))
        with tab_cmp_cm:
            _comparison_table(compare_impacts, _COMMODITY_TARGETS)
            _chart(_scenario_comparison_chart(compare_impacts, _COMMODITY_TARGETS, height=400))
    elif len(compare_selected) == 1:
        st.info("Select at least 2 scenarios to run the comparison.")
    else:
        st.info("Select 2–6 preset scenarios above to compare cross-asset impacts.")

    # ── GAP 4: Sector-Level Equity Decomposition via SPDR ETFs ───────────────
    _section_label("Sector Exposure — Which Equity Sectors Are Most Vulnerable to This Shock?")
    st.markdown(
        f'<p style="font-size:0.68rem;color:{_MUTED};margin:0.1rem 0 0.7rem;line-height:1.55">'
        f'SPDR sector ETFs mapped to commodity shock channels. '
        f'Commodity shocks do not hit all equity sectors equally: energy shock → XLE up, XAL down. '
        f'Betas estimated via OLS on {_regime_label} regime data.</p>',
        unsafe_allow_html=True,
    )

    _SECTOR_ETFS: dict[str, str] = {
        "XLE": "Energy",
        "XLB": "Materials",
        "XLI": "Industrials",
        "XLF": "Financials",
        "XLK": "Technology",
        "XLU": "Utilities",
        "XLP": "Consumer Staples",
        "XLY": "Consumer Discretionary",
        "XLV": "Health Care",
        "XLC": "Communication Svcs",
    }

    try:
        import yfinance as _yfsec
        _sec_data = _yfsec.download(
            list(_SECTOR_ETFS.keys()), period="2y",
            auto_adjust=True, progress=False, show_errors=False,
        )["Close"].pct_change().dropna()

        if not _sec_data.empty and len(raw_shocks) > 0:
            # Compute sector betas to the active commodity shocks
            _combined_for_sec = load_combined_returns(start, end)
            _sec_betas: dict[str, dict[str, float]] = {}

            for _proxy_name, _shock_val in raw_shocks.items():
                if _proxy_name not in _combined_for_sec.columns:
                    continue
                _proxy_r = _combined_for_sec[_proxy_name].dropna()
                _sb_row: dict[str, float] = {}
                for _etf, _sec_name in _SECTOR_ETFS.items():
                    if _etf not in _sec_data.columns:
                        continue
                    _aligned = pd.concat([_proxy_r, _sec_data[_etf]], axis=1).dropna()
                    if len(_aligned) < 60:
                        continue
                    _x = _aligned.iloc[:, 0].values
                    _y = _aligned.iloc[:, 1].values
                    _var_x = np.var(_x, ddof=1)
                    if _var_x < 1e-12:
                        continue
                    _sb_row[_etf] = float(np.cov(_x, _y, ddof=1)[0, 1] / _var_x)
                if _sb_row:
                    _sec_betas[_proxy_name] = _sb_row

            if _sec_betas:
                # Compute estimated sector impact from current shocks
                _sec_impact: dict[str, float] = {etf: 0.0 for etf in _SECTOR_ETFS}
                for _proxy_name, _shock_val in raw_shocks.items():
                    if _proxy_name not in _sec_betas:
                        continue
                    for _etf in _SECTOR_ETFS:
                        _beta = _sec_betas[_proxy_name].get(_etf, 0.0)
                        _sec_impact[_etf] += _beta * _shock_val

                # Sort sectors by impact
                _sec_sorted = sorted(
                    [(etf, _SECTOR_ETFS[etf], imp * 100)
                     for etf, imp in _sec_impact.items() if etf in _SECTOR_ETFS],
                    key=lambda x: x[2],
                )
                _sec_names   = [f"{x[1]} ({x[0]})" for x in _sec_sorted]
                _sec_impacts = [x[2] for x in _sec_sorted]
                _sec_colors  = ["#4ade80" if v > 0 else "#f87171" for v in _sec_impacts]

                _sec_fig = go.Figure(go.Bar(
                    x=_sec_impacts,
                    y=_sec_names,
                    orientation="h",
                    marker=dict(color=_sec_colors),
                    text=[f"{v:+.2f}%" for v in _sec_impacts],
                    textfont=dict(size=9, family="JetBrains Mono, monospace"),
                ))
                _sec_fig.update_layout(
                    template="purdue", height=350,
                    paper_bgcolor="#111111", plot_bgcolor="#111111",
                    font=dict(color="#e8e9ed"),
                    xaxis=dict(title="Estimated 1-day return %",
                               zeroline=True, zerolinecolor="#555",
                               tickfont=dict(size=9, color="#8890a1")),
                    yaxis=dict(tickfont=dict(size=9, color="#c8c8c8")),
                    margin=dict(l=160, r=60, t=20, b=40),
                )
                _chart(_sec_fig)

                # Trade idea: most positive vs most negative sector
                _top_sec  = _sec_sorted[-1]
                _bot_sec  = _sec_sorted[0]
                if abs(_top_sec[2]) > 0.05 or abs(_bot_sec[2]) > 0.05:
                    st.markdown(
                        f'<div style="background:#080808;border-left:3px solid #CFB991;'
                        f'border-radius:4px;padding:8px 14px;margin:6px 0">'
                        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                        f'font-weight:700;color:#CFB991;letter-spacing:.1em">SECTOR TRADE IMPLICATION</span><br>'
                        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:11px;color:#b0b0b0">'
                        f'Long <b style="color:#4ade80">{_top_sec[1]} ({_top_sec[0]})</b> '
                        f'({_top_sec[2]:+.2f}% est.) vs '
                        f'Short <b style="color:#f87171">{_bot_sec[1]} ({_bot_sec[0]})</b> '
                        f'({_bot_sec[2]:+.2f}% est.) under this shock configuration. '
                        f'Regime: {_regime_label}.</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Insufficient data to compute sector betas.")
        elif len(raw_shocks) == 0:
            st.info("Configure a commodity shock above to see sector impact decomposition.")
        else:
            st.info("Sector ETF data unavailable. Check internet connectivity.")
    except Exception as _sec_err:
        st.caption(f"Sector decomposition unavailable: {type(_sec_err).__name__}")

    _page_footer()

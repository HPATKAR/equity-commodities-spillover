"""
Geopolitical Risk Score — composite 0-100 index.

Components (weights):
  1. Cross-asset correlation percentile  40%
  2. Commodity volatility z-score        30%
  3. VIX level score                     20%
  4. Active event count                  10%

Bands:  0-25 Low · 25-50 Moderate · 50-75 Elevated · 75-100 High/Crisis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import date

from src.data.config import GEOPOLITICAL_EVENTS, PALETTE


# ── Component scorers ──────────────────────────────────────────────────────

def _corr_percentile_score(avg_corr: pd.Series) -> float:
    """Percentile of current correlation within its own history → 0-100."""
    if avg_corr.empty or len(avg_corr) < 60:
        return 50.0
    current = float(avg_corr.iloc[-1])
    return float((avg_corr < current).mean() * 100)


def _vol_zscore_score(cmd_r: pd.DataFrame, window: int = 30) -> float:
    """
    Average annualised vol of energy + metals vs 1Y rolling mean.
    Convert z-score to 0-100 (z=0→50, z=+2→80, z=-2→20).
    """
    energy = ["WTI Crude Oil", "Brent Crude", "Natural Gas"]
    metals = ["Gold", "Silver", "Copper"]
    cols = [c for c in energy + metals if c in cmd_r.columns]
    if not cols:
        return 50.0

    rv = cmd_r[cols].rolling(window).std() * np.sqrt(252) * 100
    cur_vol   = float(rv.iloc[-1].mean())
    hist_mean = float(rv.iloc[-252:].stack().mean()) if len(rv) >= 252 else float(rv.stack().mean())
    hist_std  = float(rv.iloc[-252:].stack().std())  if len(rv) >= 252 else float(rv.stack().std())

    if hist_std < 1e-6:
        return 50.0
    z = (cur_vol - hist_mean) / hist_std
    return float(np.clip(50 + np.clip(z, -3, 3) * 15, 0, 100))


def _vix_score() -> float:
    """Piecewise-linear VIX → 0-100 score."""
    try:
        vix = yf.Ticker("^VIX").history(period="5d")
        if vix.empty:
            return 50.0
        level = float(vix["Close"].iloc[-1])
        if level < 12:  return 10.0
        if level < 20:  return 10  + (level - 12) / 8  * 30
        if level < 30:  return 40  + (level - 20) / 10 * 30
        if level < 45:  return 70  + (level - 30) / 15 * 20
        return 95.0
    except Exception:
        return 50.0


def _active_event_score() -> float:
    """Score from number of currently active geopolitical events (25 pts each, max 100)."""
    today = date.today()
    active = [e for e in GEOPOLITICAL_EVENTS if e["end"] >= today]
    return float(min(len(active) * 25, 100))


# ── Main scorer ────────────────────────────────────────────────────────────

def compute_risk_score(
    avg_corr: pd.Series,
    cmd_r: pd.DataFrame,
) -> dict:
    """
    Compute composite geopolitical risk score (0-100).

    Returns dict:
      score, label, color,
      components: {name: score},
      vix: float (raw VIX level if available)
    """
    c1 = _corr_percentile_score(avg_corr)
    c2 = _vol_zscore_score(cmd_r)
    c3 = _vix_score()
    c4 = _active_event_score()

    total = float(np.clip(0.40 * c1 + 0.30 * c2 + 0.20 * c3 + 0.10 * c4, 0, 100))

    if total < 25:   label, color = "Low",      "#2e7d32"
    elif total < 50: label, color = "Moderate", "#555960"
    elif total < 75: label, color = "Elevated", "#e67e22"
    else:            label, color = "High",     "#c0392b"

    return {
        "score":      round(total, 1),
        "label":      label,
        "color":      color,
        "components": {
            "Correlation Percentile (40%)": round(c1, 1),
            "Commodity Vol Z-Score (30%)":  round(c2, 1),
            "VIX Level Score (20%)":        round(c3, 1),
            "Active Events (10%)":          round(c4, 1),
        },
    }


# ── Historical score series ────────────────────────────────────────────────

def risk_score_history(
    avg_corr: pd.Series,
    cmd_r: pd.DataFrame,
    eq_r: pd.DataFrame | None = None,
    window: int = 252,
) -> pd.Series:
    """
    Rolling daily risk score with up to 3 components:
      40% cross-asset correlation percentile
      35% commodity vol z-score  (energy + metals)
      25% equity realised vol    (requires eq_r; strong VIX proxy)

    If eq_r is not supplied, falls back to 57% corr / 43% vol (legacy).
    """
    if avg_corr.empty or cmd_r.empty:
        return pd.Series(dtype=float)

    corr_pct = avg_corr.rolling(window, min_periods=60).apply(
        lambda x: float((x[:-1] < x[-1]).mean() * 100), raw=True
    )

    energy = ["WTI Crude Oil", "Brent Crude", "Natural Gas"]
    metals = ["Gold", "Silver", "Copper"]
    cols   = [c for c in energy + metals if c in cmd_r.columns]

    if cols:
        rv        = cmd_r[cols].rolling(30).std() * np.sqrt(252) * 100
        avg_vol   = rv.mean(axis=1)
        v_mean    = avg_vol.rolling(window, min_periods=60).mean()
        v_std     = avg_vol.rolling(window, min_periods=60).std().replace(0, np.nan)
        z         = (avg_vol - v_mean) / v_std
        vol_score = (50 + z.clip(-3, 3) * 15).clip(0, 100)
    else:
        vol_score = pd.Series(50.0, index=avg_corr.index)

    # Equity vol component — realized vol of equal-weight equity universe
    if eq_r is not None and not eq_r.empty:
        eq_vol_raw  = eq_r.rolling(20).std().mean(axis=1) * np.sqrt(252) * 100
        ev_mean     = eq_vol_raw.rolling(window, min_periods=60).mean()
        ev_std      = eq_vol_raw.rolling(window, min_periods=60).std().replace(0, np.nan)
        ev_z        = (eq_vol_raw - ev_mean) / ev_std
        eq_vol_score = (50 + ev_z.clip(-3, 3) * 15).clip(0, 100)

        aligned = pd.concat([corr_pct, vol_score, eq_vol_score], axis=1).dropna()
        if not aligned.empty:
            aligned.columns = ["corr", "vol", "eq"]
            return (
                0.40 * aligned["corr"]
                + 0.35 * aligned["vol"]
                + 0.25 * aligned["eq"]
            ).clip(0, 100).round(1)

    # Fallback: 2-component version
    aligned = pd.concat([corr_pct, vol_score], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    aligned.columns = ["corr", "vol"]
    return (0.57 * aligned["corr"] + 0.43 * aligned["vol"]).clip(0, 100).round(1)


# ── Plotly charts ──────────────────────────────────────────────────────────

def plot_risk_gauge(result: dict, height: int = 260) -> go.Figure:
    """Plotly gauge chart for current risk score."""
    score = result["score"]
    color = result["color"]
    label = result["label"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(
            suffix="/100",
            font=dict(size=32, family="JetBrains Mono, monospace", color=color),
        ),
        title=dict(
            text=f"Geopolitical Risk Score<br>"
                 f"<span style='font-size:0.82em;color:{color}'>{label}</span>",
            font=dict(size=13, family="DM Sans, sans-serif"),
        ),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1,
                      tickcolor="#ABABAB", tickfont=dict(size=9)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="white",
            borderwidth=0,
            steps=[
                dict(range=[0,  25], color="#e8f5e9"),
                dict(range=[25, 50], color="#f5f5f5"),
                dict(range=[50, 75], color="#fff3e0"),
                dict(range=[75, 100], color="#ffebee"),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.8, value=score,
            ),
        ),
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=30, t=60, b=10),
        paper_bgcolor="#ffffff",
        font=dict(family="DM Sans, sans-serif"),
    )
    return fig


def plot_risk_history(
    score_series: pd.Series,
    events: list[dict] | None = None,
    height: int = 300,
) -> go.Figure:
    """Line chart of historical risk score with regime bands."""
    from src.data.config import GEOPOLITICAL_EVENTS as _EVENTS
    if events is None:
        events = _EVENTS

    fig = go.Figure()

    # Coloured regime bands
    for y0, y1, col, lbl in [
        (0,  25,  "#e8f5e9", "Low"),
        (25, 50,  "#f5f5f5", "Moderate"),
        (50, 75,  "#fff3e0", "Elevated"),
        (75, 100, "#ffebee", "High"),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=col, opacity=0.4,
                      layer="below", line_width=0)
        fig.add_hline(y=y0, line=dict(color="#DEDAD5", width=0.5, dash="dot"))

    # Event bands (last 5)
    for ev in events[-5:]:
        fig.add_vrect(
            x0=str(ev["start"]), x1=str(ev["end"]),
            fillcolor=ev["color"], opacity=0.05, layer="below", line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=score_series.index, y=score_series.values,
        name="Risk Score",
        line=dict(color="#c0392b", width=1.8),
        fill="tozeroy", fillcolor="rgba(192,57,43,0.08)",
    ))

    fig.update_layout(
        template="purdue",
        height=height,
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.04),
            type="date",
        ),
        yaxis=dict(range=[0, 100], title="Score",
                   tickvals=[0, 25, 50, 75, 100],
                   ticktext=["0", "25<br>Low", "50<br>Mod", "75<br>Elev", "100"]),
        title=dict(text="Historical Geopolitical Risk Score", font=dict(size=11)),
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig

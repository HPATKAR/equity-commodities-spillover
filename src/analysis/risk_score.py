"""
Geopolitical Risk Score — composite 0-100 index.

Components (weights):
  1. Cross-asset correlation percentile   25%  — market coupling signal
  2. Commodity volatility z-score         20%  — energy/metals stress
  3. VIX level score                      15%  — equity fear gauge
  4. Oil-Gold geopolitical signal         25%  — strongest direct geo-risk market signal
  5. Event severity score                 15%  — severity-weighted, recency-decayed catalogue

The oil-gold signal is the core geopolitical indicator:
  - Gold rising + Oil rising simultaneously = supply-shock/conflict premium
  - Gold rising + Oil falling              = pure safe-haven flight
  - Both elevated vs history               = maximum geopolitical signal

Event severity uses category weights (War=5, Sanctions/Conflict=4,
Financial=3, Trade=2) decayed exponentially by days since event start,
with a partial tail for events that ended within 90 days.

Bands:  0-25 Low · 25-50 Moderate · 50-75 Elevated · 75-100 High/Crisis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

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


def _vix_score() -> tuple[float, float]:
    """Piecewise-linear VIX → 0-100 score. Returns (score, raw_level)."""
    try:
        vix = yf.Ticker("^VIX").history(period="5d")
        if vix.empty:
            return 50.0, float("nan")
        level = float(vix["Close"].iloc[-1])
        if level < 12:  s = 10.0
        elif level < 20: s = 10  + (level - 12) / 8  * 30
        elif level < 30: s = 40  + (level - 20) / 10 * 30
        elif level < 45: s = 70  + (level - 30) / 15 * 20
        else:            s = 95.0
        return float(s), level
    except Exception:
        return 50.0, float("nan")


def _oil_gold_signal(cmd_r: pd.DataFrame, window: int = 20) -> tuple[float, dict]:
    """
    Oil-Gold geopolitical signal — the most direct market-based measure of
    geopolitical stress.

    Logic:
      • Compute 20-day cumulative return for Gold and Oil (WTI or Brent).
      • Compute Gold z-score and Oil z-score vs 1-year rolling history.
      • Gold z > 1 AND Oil z > 1  → simultaneous spike = conflict/supply-shock premium → HIGH
      • Gold z > 1 AND Oil z < 0  → safe-haven flight, equities selling off → ELEVATED
      • Gold z < 0 AND Oil z > 2  → pure supply shock (less geo-political) → MODERATE-HIGH
      • Gold z < 0 AND Oil z < 0  → risk-on, both falling → LOW
      • Weighted composite score 0-100.

    Returns (score, detail_dict).
    """
    gold_cols = [c for c in ["Gold"] if c in cmd_r.columns]
    oil_cols  = [c for c in ["WTI Crude Oil", "Brent Crude"] if c in cmd_r.columns]

    if not gold_cols or not oil_cols:
        return 50.0, {"gold_z": 0.0, "oil_z": 0.0, "gold_ret": 0.0, "oil_ret": 0.0}

    gold_r = cmd_r[gold_cols[0]].dropna()
    oil_r  = cmd_r[oil_cols[0]].dropna()

    if len(gold_r) < 60 or len(oil_r) < 60:
        return 50.0, {"gold_z": 0.0, "oil_z": 0.0, "gold_ret": 0.0, "oil_ret": 0.0}

    # 20-day cumulative return
    gold_cum = float(gold_r.iloc[-window:].sum()) * 100
    oil_cum  = float(oil_r.iloc[-window:].sum()) * 100

    # Z-score vs 1Y rolling 20d cumulative returns
    hist = 252
    gold_roll = gold_r.rolling(window).sum() * 100
    oil_roll  = oil_r.rolling(window).sum() * 100

    g_hist = gold_roll.iloc[-hist:].dropna()
    o_hist = oil_roll.iloc[-hist:].dropna()

    g_z = float((gold_cum - g_hist.mean()) / (g_hist.std() + 1e-8)) if len(g_hist) > 20 else 0.0
    o_z = float((oil_cum  - o_hist.mean()) / (o_hist.std()  + 1e-8)) if len(o_hist) > 20 else 0.0

    # Score logic
    # Base: 50. Gold z drives geo-premium (fear signal). Oil z amplifies if concurrent.
    gold_contribution = float(np.clip(g_z * 15, -30, 35))   # gold fear premium
    oil_amplifier     = float(np.clip(o_z * 8,  -15, 20))   # oil amplifies/dampens

    # Simultaneous gold+oil spike = conflict premium: extra boost
    conflict_bonus = 0.0
    if g_z > 1.0 and o_z > 1.0:
        conflict_bonus = min(g_z * o_z * 5, 20)  # up to +20 for simultaneous spikes

    score = float(np.clip(50 + gold_contribution + oil_amplifier + conflict_bonus, 0, 100))
    return score, {
        "gold_z":   round(g_z, 2),
        "oil_z":    round(o_z, 2),
        "gold_ret": round(gold_cum, 2),
        "oil_ret":  round(oil_cum, 2),
    }


# ── Category severity weights ──────────────────────────────────────────────
_SEVERITY: dict[str, float] = {
    "War":        5.0,
    "Conflict":   4.5,
    "Sanctions":  4.0,
    "Financial":  3.5,
    "Pandemic":   4.0,
    "Energy":     3.5,
    "Trade":      2.5,
    "Political":  2.0,
    "Geopolitical": 4.0,
}

def _event_severity_score() -> tuple[float, list[dict]]:
    """
    Severity-weighted, recency-decayed geopolitical event score.

    For active events:   full severity × recency multiplier (decays over 180d since start)
    For ended events:    tail score, decays to zero at 90 days post-end
    Score normalised to 0-100 with cap at 100.

    Returns (score, list of scored events for display).
    """
    today = date.today()
    scored = []
    total  = 0.0

    for ev in GEOPOLITICAL_EVENTS:
        cat      = ev.get("category", "Geopolitical")
        severity = _SEVERITY.get(cat, 3.0)
        ev_start = ev["start"]
        ev_end   = ev.get("end", today)

        if ev_start <= today <= ev_end:
            # Active: recency multiplier — freshest events score highest
            days_active = max((today - ev_start).days, 1)
            # Recency: peaks at start (1.0), decays to 0.5 at 180 days
            recency = max(0.5, 1.0 - days_active / 360)
            pts = severity * recency * 20   # max 100 for severity=5, recency=1
            scored.append({"label": ev["label"], "status": "active", "score": round(pts, 1),
                           "severity": severity, "recency": round(recency, 2)})
            total += pts

        elif ev_end < today:
            # Recently ended: tail that decays to 0 at 90 days post-end
            days_since_end = (today - ev_end).days
            if days_since_end <= 90:
                tail = (1 - days_since_end / 90) * severity * 8  # max 40 for tail
                scored.append({"label": ev["label"], "status": "tail",
                               "score": round(tail, 1), "severity": severity,
                               "recency": round(1 - days_since_end / 90, 2)})
                total += tail

    score = float(np.clip(total, 0, 100))
    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
    return score, scored_sorted


# ── Main scorer ────────────────────────────────────────────────────────────

def compute_risk_score(
    avg_corr: pd.Series,
    cmd_r: pd.DataFrame,
) -> dict:
    """
    Compute composite geopolitical risk score (0-100).

    Returns dict with score, label, color, components, raw values.
    """
    c1 = _corr_percentile_score(avg_corr)
    c2 = _vol_zscore_score(cmd_r)
    c3, vix_level = _vix_score()
    c4, oil_gold_detail = _oil_gold_signal(cmd_r)
    c5, event_detail    = _event_severity_score()

    total = float(np.clip(
        0.25 * c1
        + 0.20 * c2
        + 0.15 * c3
        + 0.25 * c4
        + 0.15 * c5,
        0, 100,
    ))

    if total < 25:   label, color = "Low",      "#2e7d32"
    elif total < 50: label, color = "Moderate", "#555960"
    elif total < 75: label, color = "Elevated", "#e67e22"
    else:            label, color = "High",     "#c0392b"

    return {
        "score":        round(total, 1),
        "label":        label,
        "color":        color,
        "vix_level":    round(vix_level, 1) if not np.isnan(vix_level) else None,
        "oil_gold":     oil_gold_detail,
        "events":       event_detail,
        "corr_pct":     round(c1, 1),
        "cmd_vol_z":    round((c2 - 50) / 15, 2),   # back to z-score for display
        "eq_vol_z":     0.0,                          # populated by callers if available
        "components": {
            "Cross-Asset Correlation (25%)":  round(c1, 1),
            "Commodity Vol Z-Score (20%)":    round(c2, 1),
            "VIX Level Score (15%)":          round(c3, 1),
            "Oil-Gold Geo Signal (25%)":      round(c4, 1),
            "Event Severity Score (15%)":     round(c5, 1),
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
    Rolling daily risk score — same 5-component structure as compute_risk_score.

    Components (weights):
      25% cross-asset correlation percentile
      20% commodity vol z-score
      25% oil-gold geopolitical signal (rolling 20d)
      15% equity realised vol (VIX proxy when eq_r supplied)
      15% constant component = 50 (event severity not rolling; injected as neutral)
    """
    if cmd_r.empty:
        return pd.Series(dtype=float)

    # Rebuild dense avg_corr if the supplied series is too sparse
    if eq_r is not None and not eq_r.empty and (
        avg_corr.empty or len(avg_corr) < 0.5 * len(cmd_r)
    ):
        common = cmd_r.index.intersection(eq_r.index)
        if len(common) > 120:
            eq_idx  = eq_r.reindex(common).mean(axis=1)
            cmd_idx = cmd_r.reindex(common).mean(axis=1)
            avg_corr = eq_idx.rolling(60, min_periods=30).corr(cmd_idx).abs().dropna()

    if avg_corr.empty:
        return pd.Series(dtype=float)

    # ── 1. Correlation percentile ──────────────────────────────────────────
    corr_pct = avg_corr.rolling(window, min_periods=60).apply(
        lambda x: float((x[:-1] < x[-1]).mean() * 100), raw=True
    )

    # ── 2. Commodity vol z-score ───────────────────────────────────────────
    energy = ["WTI Crude Oil", "Brent Crude", "Natural Gas"]
    metals = ["Gold", "Silver", "Copper"]
    vol_cols = [c for c in energy + metals if c in cmd_r.columns]

    if vol_cols:
        rv        = cmd_r[vol_cols].rolling(30).std() * np.sqrt(252) * 100
        avg_vol   = rv.mean(axis=1)
        v_mean    = avg_vol.rolling(window, min_periods=60).mean()
        v_std     = avg_vol.rolling(window, min_periods=60).std().replace(0, np.nan)
        vol_score = (50 + ((avg_vol - v_mean) / v_std).clip(-3, 3) * 15).clip(0, 100)
    else:
        vol_score = pd.Series(50.0, index=avg_corr.index)

    # ── 3. Oil-Gold geopolitical signal (rolling 20d cumulative returns) ───
    gold_col = next((c for c in ["Gold"] if c in cmd_r.columns), None)
    oil_col  = next((c for c in ["WTI Crude Oil", "Brent Crude"] if c in cmd_r.columns), None)

    if gold_col and oil_col:
        g_cum = cmd_r[gold_col].rolling(20).sum() * 100
        o_cum = cmd_r[oil_col].rolling(20).sum()  * 100

        g_mean = g_cum.rolling(window, min_periods=60).mean()
        g_std  = g_cum.rolling(window, min_periods=60).std().replace(0, np.nan)
        o_mean = o_cum.rolling(window, min_periods=60).mean()
        o_std  = o_cum.rolling(window, min_periods=60).std().replace(0, np.nan)

        g_z = ((g_cum - g_mean) / g_std).clip(-4, 4)
        o_z = ((o_cum - o_mean) / o_std).clip(-4, 4)

        gold_contrib   = (g_z * 15).clip(-30, 35)
        oil_amp        = (o_z *  8).clip(-15, 20)
        conflict_bonus = ((g_z > 1) & (o_z > 1)) * (g_z * o_z * 5).clip(0, 20)
        og_score       = (50 + gold_contrib + oil_amp + conflict_bonus).clip(0, 100)
    else:
        og_score = pd.Series(50.0, index=avg_corr.index)

    # ── 4. Equity vol (VIX proxy) ──────────────────────────────────────────
    if eq_r is not None and not eq_r.empty:
        eq_vol_raw   = eq_r.rolling(20).std().mean(axis=1) * np.sqrt(252) * 100
        ev_mean      = eq_vol_raw.rolling(window, min_periods=60).mean()
        ev_std       = eq_vol_raw.rolling(window, min_periods=60).std().replace(0, np.nan)
        eq_vol_score = (50 + ((eq_vol_raw - ev_mean) / ev_std).clip(-3, 3) * 15).clip(0, 100)
    else:
        eq_vol_score = pd.Series(50.0, index=avg_corr.index)

    # ── Combine ────────────────────────────────────────────────────────────
    aligned = pd.concat([corr_pct, vol_score, og_score, eq_vol_score], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    aligned.columns = ["corr", "vol", "og", "eq"]

    return (
        0.25 * aligned["corr"]
        + 0.20 * aligned["vol"]
        + 0.25 * aligned["og"]
        + 0.15 * aligned["eq"]
        + 0.15 * 50          # neutral placeholder for event severity (not rolling)
    ).clip(0, 100).round(1)


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

    # Event markers — vertical dashed line + label annotation at each event start
    idx = score_series.index
    for ev in events:
        ev_date = str(ev["start"])
        # Only draw if the event start falls within the data range
        if not score_series.empty and (
            pd.Timestamp(ev["start"]) < idx[0] or pd.Timestamp(ev["start"]) > idx[-1]
        ):
            continue
        col = ev.get("color", "#8E6F3E")
        fig.add_vline(
            x=ev_date,
            line=dict(color=col, width=1, dash="dot"),
        )
        fig.add_annotation(
            x=ev_date,
            y=97,
            text=ev["label"],
            showarrow=False,
            textangle=-90,
            font=dict(size=7.5, color=col, family="DM Sans, sans-serif"),
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.65)",
            borderpad=2,
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

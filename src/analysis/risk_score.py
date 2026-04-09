"""
Geopolitical Risk Score — 3-Layer Architecture.

Top-level score:
  Global Geo Risk = 40% Conflict Intensity + 35% Transmission Pressure + 25% Market Confirmation

Layer 1 — Conflict Intensity (40%)
  Delegated to conflict_model.py (CIS/TPS per-conflict, then portfolio aggregate).

Layer 2 — Transmission Pressure (35%)
  Portfolio TPS from conflict_model.py.

Layer 3 — Market Confirmation (25%)
  Orthogonalized market signals: equity vol, rates vol, commodity vol residual,
  safe-haven behavior, cross-asset spillover, oil-gold signal.
  EWM z-scores (span=60) to avoid procyclicality.
  Signals are residualized to remove double-counting of shared factor.

Confidence overlay:
  Blends conflict model confidence, market signal agreement, and data freshness.

Bands:  0–25 Low · 25–50 Moderate · 50–75 Elevated · 75–100 High/Crisis

Backward-compatible API: compute_risk_score(), risk_score_history(),
plot_risk_gauge(), plot_risk_history() all preserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

from src.data.config import GEOPOLITICAL_EVENTS, PALETTE


# ── EWM z-score helper ────────────────────────────────────────────────────────

def _ewm_zscore(series: pd.Series, span: int = 60) -> pd.Series:
    """Exponentially weighted z-score. Slower to react than simple rolling."""
    mu    = series.ewm(span=span, min_periods=20).mean()
    sigma = series.ewm(span=span, min_periods=20).std()
    return (series - mu) / sigma.replace(0, np.nan)


def _zscore_to_score(z: float, scale: float = 12.0) -> float:
    """Map EWM z-score → [0, 100]. z=0 → 50, z=+2 → ~74, z=−2 → ~26."""
    return float(np.clip(50.0 + float(z) * scale, 0.0, 100.0))


# ── Market Confirmation Layer ─────────────────────────────────────────────────

def _equity_vol_score(eq_r: pd.DataFrame | None) -> float:
    """20d realized equity vol z-scored via EWM. Returns 0–100."""
    if eq_r is None or eq_r.empty:
        return 50.0
    spx_cols = [c for c in ["S&P 500", "Eurostoxx 50", "Nikkei 225"] if c in eq_r.columns]
    if not spx_cols:
        spx_cols = list(eq_r.columns[:3])
    rv = eq_r[spx_cols].rolling(20).std().mean(axis=1) * np.sqrt(252) * 100
    rv = rv.dropna()
    if len(rv) < 40:
        return 50.0
    z = float(_ewm_zscore(rv, span=60).iloc[-1])
    return _zscore_to_score(z)


def _rates_vol_score(cmd_r: pd.DataFrame) -> float:
    """
    MOVE-proxy: 20d realized vol of TLT (long Treasury ETF) as rates vol measure.
    More orthogonal to equity vol than VIX.
    """
    try:
        tlt = yf.Ticker("TLT").history(period="200d")["Close"]
        if tlt.empty or len(tlt) < 40:
            return 50.0
        tlt_r = tlt.pct_change().dropna()
        rv    = tlt_r.rolling(20).std() * np.sqrt(252) * 100
        rv    = rv.dropna()
        z = float(_ewm_zscore(rv, span=60).iloc[-1])
        return _zscore_to_score(z, scale=10.0)
    except Exception:
        return 50.0


def _commodity_vol_residual_score(
    cmd_r: pd.DataFrame, eq_r: pd.DataFrame | None
) -> float:
    """
    Commodity vol residualized on equity vol — extracts the commodity-specific
    stress component that is NOT explained by broad equity fear.
    Prevents double-counting VIX-driven commodity vol.
    """
    energy_metals = ["WTI Crude Oil", "Brent Crude", "Natural Gas",
                     "Gold", "Silver", "Copper"]
    cols = [c for c in energy_metals if c in cmd_r.columns]
    if not cols:
        return 50.0

    rv_cmd = cmd_r[cols].rolling(20).std().mean(axis=1) * np.sqrt(252) * 100
    rv_cmd = rv_cmd.dropna()

    if eq_r is not None and not eq_r.empty:
        # Residualize: regress cmd_vol on eq_vol, use residual
        eq_cols  = list(eq_r.columns[:5])
        rv_eq    = eq_r[eq_cols].rolling(20).std().mean(axis=1) * np.sqrt(252) * 100
        common   = rv_cmd.index.intersection(rv_eq.index)
        if len(common) > 40:
            rv_c = rv_cmd.reindex(common)
            rv_e = rv_eq.reindex(common)
            # OLS over trailing 252 days
            tail = min(len(common), 252)
            x    = rv_e.iloc[-tail:].values
            y    = rv_c.iloc[-tail:].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 30:
                beta = np.polyfit(x[mask], y[mask], 1)[0]
                rv_cmd = rv_c - beta * rv_e

    if len(rv_cmd.dropna()) < 30:
        return 50.0
    z = float(_ewm_zscore(rv_cmd.dropna(), span=60).iloc[-1])
    return _zscore_to_score(z, scale=11.0)


def _safe_haven_score(cmd_r: pd.DataFrame, eq_r: pd.DataFrame | None) -> float:
    """
    Safe-haven signal: Gold 20d cumulative return z-score.
    Boosted if TLT is also positive (corroborating flight-to-safety).
    Gold z > 1 = safe-haven bid = geopolitical stress signal.
    """
    gold_col = next((c for c in ["Gold"] if c in cmd_r.columns), None)
    if not gold_col:
        return 50.0

    gold_r = cmd_r[gold_col].dropna()
    if len(gold_r) < 60:
        return 50.0

    g_cum  = gold_r.rolling(20).sum() * 100
    g_cum  = g_cum.dropna()
    if len(g_cum) < 40:
        return 50.0

    g_z = float(_ewm_zscore(g_cum, span=60).iloc[-1])

    # TLT corroboration (if equity data has TLT-like proxy)
    tlt_boost = 0.0
    try:
        tlt = yf.Ticker("TLT").history(period="60d")["Close"]
        if not tlt.empty:
            tlt_ret = float(tlt.pct_change().dropna().iloc[-20:].sum() * 100)
            tlt_z   = tlt_ret / max(float(tlt.pct_change().dropna().std() * np.sqrt(20) * 100), 1)
            if g_z > 0.5 and tlt_z > 0.3:
                tlt_boost = min(tlt_z * 4, 10.0)
    except Exception:
        pass

    raw_score = _zscore_to_score(g_z, scale=14.0)
    return float(np.clip(raw_score + tlt_boost, 0.0, 100.0))


def _oil_gold_signal(cmd_r: pd.DataFrame, window: int = 20) -> tuple[float, dict]:
    """
    Oil-Gold geopolitical signal.
    Gold z > 1 AND Oil z > 1  → conflict/supply-shock premium → HIGH
    Gold z > 1 AND Oil z < 0  → safe-haven flight               → ELEVATED
    Gold z < 0 AND Oil z > 2  → pure supply shock               → MODERATE-HIGH
    Gold z < 0 AND Oil z < 0  → risk-on                         → LOW
    """
    gold_cols = [c for c in ["Gold"] if c in cmd_r.columns]
    oil_cols  = [c for c in ["WTI Crude Oil", "Brent Crude"] if c in cmd_r.columns]

    if not gold_cols or not oil_cols:
        return 50.0, {"gold_z": 0.0, "oil_z": 0.0, "gold_ret": 0.0, "oil_ret": 0.0}

    gold_r = cmd_r[gold_cols[0]].dropna()
    oil_r  = cmd_r[oil_cols[0]].dropna()

    if len(gold_r) < 60 or len(oil_r) < 60:
        return 50.0, {"gold_z": 0.0, "oil_z": 0.0, "gold_ret": 0.0, "oil_ret": 0.0}

    gold_cum = float(gold_r.iloc[-window:].sum()) * 100
    oil_cum  = float(oil_r.iloc[-window:].sum()) * 100

    # EWM z-scores vs trailing history
    g_roll = gold_r.rolling(window).sum() * 100
    o_roll = oil_r.rolling(window).sum() * 100

    g_z = float(_ewm_zscore(g_roll.dropna(), span=60).iloc[-1]) if len(g_roll.dropna()) > 30 else 0.0
    o_z = float(_ewm_zscore(o_roll.dropna(), span=60).iloc[-1]) if len(o_roll.dropna()) > 30 else 0.0

    gold_contribution = float(np.clip(g_z * 15, -30, 35))
    oil_amplifier     = float(np.clip(o_z * 8,  -15, 20))
    conflict_bonus    = 0.0
    if g_z > 1.0 and o_z > 1.0:
        conflict_bonus = min(g_z * o_z * 5, 20)

    score = float(np.clip(50 + gold_contribution + oil_amplifier + conflict_bonus, 0, 100))
    return score, {
        "gold_z":   round(g_z, 2),
        "oil_z":    round(o_z, 2),
        "gold_ret": round(gold_cum, 2),
        "oil_ret":  round(oil_cum, 2),
    }


def _spillover_score(avg_corr: pd.Series) -> float:
    """Cross-asset spillover signal: percentile of current correlation in history."""
    if avg_corr.empty or len(avg_corr) < 60:
        return 50.0
    current = float(avg_corr.iloc[-1])
    return float((avg_corr < current).mean() * 100)


def _market_confirmation_score(
    avg_corr: pd.Series,
    cmd_r: pd.DataFrame,
    eq_r: pd.DataFrame | None = None,
) -> tuple[float, dict]:
    """
    Market Confirmation Score (MCS) — 5 orthogonalized signals.
    Weights sum to 1.0. Returns (score, component_dict).
    """
    eq_vol   = _equity_vol_score(eq_r)
    rates_vol = _rates_vol_score(cmd_r)
    cmd_vol  = _commodity_vol_residual_score(cmd_r, eq_r)
    safe_hav = _safe_haven_score(cmd_r, eq_r)
    og_score, og_detail = _oil_gold_signal(cmd_r)
    spill    = _spillover_score(avg_corr)

    # Weights — oil-gold and safe-haven carry most geopolitical signal
    w = {"eq_vol": 0.15, "rates_vol": 0.15, "cmd_vol": 0.15,
         "safe_haven": 0.22, "oil_gold": 0.18, "spillover": 0.15}

    mcs = (
        w["eq_vol"]    * eq_vol
        + w["rates_vol"] * rates_vol
        + w["cmd_vol"]   * cmd_vol
        + w["safe_haven"]* safe_hav
        + w["oil_gold"]  * og_score
        + w["spillover"] * spill
    )

    components = {
        "Equity Vol (orthog)":     round(eq_vol,    1),
        "Rates Vol (TLT proxy)":   round(rates_vol, 1),
        "Commodity Vol (residual)":round(cmd_vol,   1),
        "Safe-Haven Bid":          round(safe_hav,  1),
        "Oil-Gold Signal":         round(og_score,  1),
        "Cross-Asset Spillover":   round(spill,     1),
    }
    return float(np.clip(mcs, 0, 100)), components


# ── Main scorer ────────────────────────────────────────────────────────────────

def compute_risk_score(
    avg_corr: pd.Series,
    cmd_r: pd.DataFrame,
    eq_r: pd.DataFrame | None = None,
) -> dict:
    """
    Compute composite geopolitical risk score (0–100).

    Architecture:
      40% Conflict Intensity Score (conflict_model.py)
      35% Transmission Pressure Score (conflict_model.py)
      25% Market Confirmation Score (orthogonalized market signals)

    Scenario multiplier from session_state["scenario"] applied after assembly.
    Returns dict with score, label, color, components, confidence, and detail.
    """
    from src.analysis.conflict_model import aggregate_portfolio_scores
    from src.analysis.scenario_state import get_scenario

    # Layer 1 + 2: Conflict Intensity and Transmission Pressure
    conflict_agg = aggregate_portfolio_scores()
    cis_portfolio = conflict_agg["cis"]
    tps_portfolio = conflict_agg["tps"]
    conflict_conf = conflict_agg["confidence"]

    # Layer 3: Market Confirmation
    mcs, mcs_components = _market_confirmation_score(avg_corr, cmd_r, eq_r)

    # Assembly
    raw = (
        0.40 * cis_portfolio
        + 0.35 * tps_portfolio
        + 0.25 * mcs
    )

    # Scenario multiplier
    scenario = get_scenario()
    geo_mult = scenario.get("geo_mult", 1.0)
    total    = float(np.clip(raw * geo_mult, 0.0, 100.0))

    # Confidence overlay
    mcs_signal_agreement = float(np.std(list(mcs_components.values()))) / 50.0
    mcs_agreement_score  = float(np.clip(1.0 - mcs_signal_agreement, 0.0, 1.0))
    overall_confidence   = float(
        0.50 * conflict_conf
        + 0.30 * mcs_agreement_score
        + 0.20 * (1.0 if not avg_corr.empty else 0.5)
    )

    if total < 25:   label, color = "Low",      "#2e7d32"
    elif total < 50: label, color = "Moderate", "#8E9AAA"
    elif total < 75: label, color = "Elevated", "#e67e22"
    else:            label, color = "High",     "#c0392b"

    # News GPR layer — diagnostic output only (does not affect the 3-layer score)
    try:
        from src.analysis.gpr_news import get_news_gpr_layer
        news = get_news_gpr_layer()
        news_gpr_out = {
            "news_gpr":     news["news_gpr"],
            "threat_score": news["threat_score"],
            "act_score":    news["act_score"],
            "alpha":        news["alpha"],
            "n_threat":     news["n_threat"],
            "n_act":        news["n_act"],
            "news_per_conflict": news["per_conflict"],
        }
    except Exception:
        news_gpr_out = {
            "news_gpr": None, "threat_score": None, "act_score": None,
            "alpha": None, "n_threat": 0, "n_act": 0, "news_per_conflict": {},
        }

    return {
        "score":          round(total, 1),
        "label":          label,
        "color":          color,
        "confidence":     round(overall_confidence, 2),
        "scenario":       scenario.get("label", "Base"),
        # Layer breakdown
        "cis":            round(cis_portfolio, 1),
        "tps":            round(tps_portfolio, 1),
        "mcs":            round(mcs, 1),
        # News GPR (diagnostic — not in score)
        **news_gpr_out,
        # Detail
        "conflict_detail": conflict_agg.get("conflict_detail", {}),
        "top_conflict":    conflict_agg.get("top_conflict"),
        "mcs_components":  mcs_components,
        # Legacy fields (kept for backward compat with existing pages)
        "weights": {"conflict_intensity": 0.40, "transmission": 0.35, "market_conf": 0.25},
        "corr_pct": round(_spillover_score(avg_corr), 1),
        "components": {
            f"Conflict Intensity (40%)":    round(cis_portfolio, 1),
            f"Transmission Pressure (35%)": round(tps_portfolio, 1),
            f"Market Confirmation (25%)":   round(mcs, 1),
            **{f"  {k}": v for k, v in mcs_components.items()},
        },
    }


# ── Historical score series (used by risk history chart) ─────────────────────

def risk_score_history(
    avg_corr: pd.Series,
    cmd_r: pd.DataFrame,
    eq_r: pd.DataFrame | None = None,
    window: int = 252,
) -> pd.Series:
    """
    Rolling daily risk score using market-confirmation signals only
    (conflict model scores are not available historically at daily frequency).

    Weights: 30% correlation pct + 25% oil-gold + 20% commodity vol +
             15% equity vol + 10% neutral placeholder (conflict)
    """
    if cmd_r.empty:
        return pd.Series(dtype=float)

    # Rebuild avg_corr if sparse
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

    # 1. Correlation percentile
    corr_pct = avg_corr.rolling(window, min_periods=60).apply(
        lambda x: float((x[:-1] < x[-1]).mean() * 100), raw=True
    )

    # 2. Commodity vol z-score (EWM)
    energy_metals = ["WTI Crude Oil", "Brent Crude", "Natural Gas", "Gold", "Silver", "Copper"]
    vol_cols = [c for c in energy_metals if c in cmd_r.columns]
    if vol_cols:
        rv        = cmd_r[vol_cols].rolling(30).std() * np.sqrt(252) * 100
        avg_vol   = rv.mean(axis=1)
        v_mean    = avg_vol.ewm(span=60).mean()
        v_std     = avg_vol.ewm(span=60).std().replace(0, np.nan)
        vol_score = (50 + ((avg_vol - v_mean) / v_std).clip(-3, 3) * 11).clip(0, 100)
    else:
        vol_score = pd.Series(50.0, index=avg_corr.index)

    # 3. Oil-Gold signal (rolling EWM z-scores)
    gold_col = next((c for c in ["Gold"] if c in cmd_r.columns), None)
    oil_col  = next((c for c in ["WTI Crude Oil", "Brent Crude"] if c in cmd_r.columns), None)
    if gold_col and oil_col:
        g_cum = cmd_r[gold_col].rolling(20).sum() * 100
        o_cum = cmd_r[oil_col].rolling(20).sum()  * 100
        g_mean = g_cum.ewm(span=60).mean()
        g_std  = g_cum.ewm(span=60).std().replace(0, np.nan)
        o_mean = o_cum.ewm(span=60).mean()
        o_std  = o_cum.ewm(span=60).std().replace(0, np.nan)
        g_z = ((g_cum - g_mean) / g_std).clip(-4, 4)
        o_z = ((o_cum - o_mean) / o_std).clip(-4, 4)
        conflict_bonus = ((g_z > 1) & (o_z > 1)) * (g_z * o_z * 5).clip(0, 20)
        og_score = (50 + g_z * 15 + o_z * 8 + conflict_bonus).clip(0, 100)
    else:
        og_score = pd.Series(50.0, index=avg_corr.index)

    # 4. Equity vol proxy
    if eq_r is not None and not eq_r.empty:
        eq_vol_raw   = eq_r.rolling(20).std().mean(axis=1) * np.sqrt(252) * 100
        ev_mean      = eq_vol_raw.ewm(span=60).mean()
        ev_std       = eq_vol_raw.ewm(span=60).std().replace(0, np.nan)
        eq_vol_score = (50 + ((eq_vol_raw - ev_mean) / ev_std).clip(-3, 3) * 11).clip(0, 100)
    else:
        eq_vol_score = pd.Series(50.0, index=avg_corr.index)

    aligned = pd.concat([corr_pct, vol_score, og_score, eq_vol_score], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    aligned.columns = ["corr", "vol", "og", "eq"]

    return (
        0.30 * aligned["corr"]
        + 0.25 * aligned["og"]
        + 0.20 * aligned["vol"]
        + 0.15 * aligned["eq"]
        + 0.10 * 50
    ).clip(0, 100).round(1)


# ── Plotly charts (backward compatible) ──────────────────────────────────────

def plot_risk_gauge(result: dict, height: int = 260) -> go.Figure:
    """Plotly gauge chart for current risk score."""
    score = result["score"]
    color = result["color"]
    label = result["label"]

    conf_text = f" · {result.get('confidence', 0):.0%} conf" if "confidence" in result else ""

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(
            suffix="/100",
            font=dict(size=32, family="JetBrains Mono, monospace", color=color),
        ),
        title=dict(
            text=(f"Geopolitical Risk Score<br>"
                  f"<span style='font-size:0.82em;color:{color}'>{label}{conf_text}</span>"),
            font=dict(size=13, family="DM Sans, sans-serif"),
        ),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1,
                      tickcolor="#ABABAB", tickfont=dict(size=9)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(30,33,45,0.6)",
            borderwidth=0,
            steps=[
                dict(range=[0,  25], color="rgba(46,125,50,0.18)"),
                dict(range=[25, 50], color="rgba(120,120,130,0.12)"),
                dict(range=[50, 75], color="rgba(230,126,34,0.18)"),
                dict(range=[75, 100], color="rgba(192,57,43,0.22)"),
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif"),
    )
    return fig


def plot_risk_history(
    score_series: pd.Series,
    events: list[dict] | None = None,
    height: int = 300,
) -> go.Figure:
    """Line chart of historical risk score with regime bands and event markers."""
    if events is None:
        events = GEOPOLITICAL_EVENTS

    fig = go.Figure()

    for y0, y1, col, _ in [
        (0,  25,  "rgba(46,125,50,0.12)",   "Low"),
        (25, 50,  "rgba(100,100,100,0.06)", "Moderate"),
        (50, 75,  "rgba(230,126,34,0.12)",  "Elevated"),
        (75, 100, "rgba(192,57,43,0.18)",   "High"),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=col, opacity=1.0,
                      layer="below", line_width=0)
        fig.add_hline(y=y0, line=dict(color="rgba(150,150,150,0.25)", width=0.5, dash="dot"))

    idx = score_series.index
    for ev in events:
        if not score_series.empty and (
            pd.Timestamp(ev["start"]) < idx[0] or pd.Timestamp(ev["start"]) > idx[-1]
        ):
            continue
        col = ev.get("color", "#8E6F3E")
        fig.add_vline(x=str(ev["start"]),
                      line=dict(color=col, width=1, dash="dot"))
        fig.add_annotation(
            x=str(ev["start"]), y=97,
            text=ev["label"], showarrow=False,
            textangle=-90,
            font=dict(size=7.5, color=col, family="DM Sans, sans-serif"),
            xanchor="right", yanchor="top",
            bgcolor="rgba(10,12,20,0.70)", borderpad=2,
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
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
        yaxis=dict(range=[0, 100], title="Score",
                   tickvals=[0, 25, 50, 75, 100],
                   ticktext=["0", "25<br>Low", "50<br>Mod", "75<br>Elev", "100"]),
        title_text="",
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig

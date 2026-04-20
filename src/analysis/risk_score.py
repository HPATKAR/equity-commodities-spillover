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

import streamlit as st
from src.data.config import GEOPOLITICAL_EVENTS, PALETTE


# ── Model configuration (single source of truth) ──────────────────────────────

_MODEL_CONFIG: dict = {
    # Top-level layer weights (must sum to 1.0)
    # MCS raised to 0.35: live market signals (price, vol, safe-haven) are
    # the most responsive real-time stress indicators.  CIS/TPS are driven by
    # the CONFLICTS registry which updates manually and slowly.
    "weights": {
        "cis":  0.35,   # Conflict Intensity Score  (conflict_model.py)
        "tps":  0.30,   # Transmission Pressure Score (conflict_model.py)
        "mcs":  0.35,   # Market Confirmation Score   (this file)
    },
    # MCS sub-signal weights (must sum to 1.0)
    # - rates_vol reduced: TLT falls during geo stress (not a reliable boost)
    # - cmd_vol raised: raw vol now (not residualized); captures oil spikes directly
    # - safe_haven raised: gold + oil joint elevation is primary geo-stress signal
    # - corr_accel reduced: velocity-based, less lag, but narrower scope
    "mcs_weights": {
        "eq_vol":     0.15,
        "rates_vol":  0.07,
        "cmd_vol":    0.22,
        "safe_haven": 0.28,
        "oil_gold":   0.23,
        "corr_accel": 0.05,
    },
    # risk_score_history() weights — approximates live model using mkt-only signals
    "history_weights": {
        "eq_vol":     0.35,
        "oil_gold":   0.35,
        "cmd_vol":    0.20,
        "corr_accel": 0.10,
    },
}


# ── Cached auxiliary fetches ──────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_tlt() -> pd.Series:
    """
    Single cached TLT fetch (200d) shared by _rates_vol_score and _safe_haven_score.
    Avoids two separate uncached network calls on every risk score computation.
    Returns Close price series; empty Series on failure.
    """
    try:
        s = yf.Ticker("TLT").history(period="200d")["Close"]
        return s if not s.empty else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


# ── EWM z-score helper ────────────────────────────────────────────────────────

def _ewm_zscore(series: pd.Series, span: int = 252) -> pd.Series:
    """
    Exponentially weighted z-score. Default span=252 (one trading year).
    span=60 caused the 'new normal' effect: after 2+ months of elevated stress,
    the EWM mean rose to match the crisis level, producing z≈0 even during
    active wars.  A 252-day window keeps the pre-crisis baseline visible
    throughout the stress period.
    """
    mu    = series.ewm(span=span, min_periods=60).mean()
    sigma = series.ewm(span=span, min_periods=60).std()
    return (series - mu) / sigma.replace(0, np.nan)


def _zscore_to_score(z: float, scale: float = 15.0) -> float:
    """Map EWM z-score → [0, 100]. z=0 → 50, z=+2 → ~80, z=−2 → ~20."""
    return float(np.clip(50.0 + float(z) * scale, 0.0, 100.0))


# ── Market Confirmation Layer ─────────────────────────────────────────────────

def _equity_vol_score(eq_r: pd.DataFrame | None) -> float:
    """20d realized equity vol z-scored via EWM (span=252). Returns 0–100."""
    if eq_r is None or eq_r.empty:
        return 50.0
    spx_cols = [c for c in ["S&P 500", "Eurostoxx 50", "Nikkei 225"] if c in eq_r.columns]
    if not spx_cols:
        spx_cols = list(eq_r.columns[:3])
    rv = eq_r[spx_cols].rolling(20).std().mean(axis=1) * np.sqrt(252) * 100
    rv = rv.dropna()
    if len(rv) < 80:
        return 50.0
    z = float(_ewm_zscore(rv).iloc[-1])
    return _zscore_to_score(z)


def _rates_vol_score(cmd_r: pd.DataFrame) -> float:
    """
    MOVE-proxy: 20d realized vol of TLT (long Treasury ETF) as rates vol measure.
    More orthogonal to equity vol than VIX. span=252 baseline.
    """
    try:
        tlt = _fetch_tlt()
        if tlt.empty or len(tlt) < 80:
            return 50.0
        tlt_r = tlt.pct_change().dropna()
        rv    = tlt_r.rolling(20).std() * np.sqrt(252) * 100
        rv    = rv.dropna()
        z = float(_ewm_zscore(rv).iloc[-1])
        return _zscore_to_score(z, scale=10.0)
    except Exception:
        return 50.0


def _commodity_vol_score(cmd_r: pd.DataFrame) -> float:
    """
    Raw commodity vol z-score (EWM span=252). No residualization.

    The previous version residualized commodity vol against equity vol to
    prevent double-counting.  This was wrong for geopolitical supply shocks:
    during war/blockade, oil vol rises AND equities fall in tandem.  The OLS
    regression finds a positive beta and subtracts it, producing a near-zero
    residual precisely when the oil signal should be highest.

    Raw vol z-scored against a 252-day baseline correctly captures elevated
    commodity stress without removing the legitimate co-movement.
    """
    energy_metals = ["WTI Crude Oil", "Brent Crude", "Natural Gas",
                     "Gold", "Silver", "Copper"]
    cols = [c for c in energy_metals if c in cmd_r.columns]
    if not cols:
        return 50.0
    rv = cmd_r[cols].rolling(20).std().mean(axis=1) * np.sqrt(252) * 100
    rv = rv.dropna()
    if len(rv) < 80:
        return 50.0
    z = float(_ewm_zscore(rv).iloc[-1])
    return _zscore_to_score(z, scale=14.0)


def _safe_haven_score(cmd_r: pd.DataFrame, eq_r: pd.DataFrame | None) -> float:
    """
    Safe-haven signal: Gold 20d cumulative return z-score (span=252).

    Geopolitical premium: when both gold AND oil are elevated simultaneously,
    this is the market signature of a war/supply-shock — distinct from a pure
    risk-off growth scare (gold up, oil down).  Boost is a smooth ramp
    proportional to the joint magnitude, no binary threshold.

    Previous TLT-corroboration was anti-signal: TLT FALLS during geopolitical
    stress (rates up, inflation fear), so the boost condition was never met
    during the exact regimes we need to score highest.
    """
    gold_col = next((c for c in ["Gold"] if c in cmd_r.columns), None)
    if not gold_col:
        return 50.0

    gold_r = cmd_r[gold_col].dropna()
    if len(gold_r) < 80:
        return 50.0

    g_cum = gold_r.rolling(20).sum() * 100
    g_cum = g_cum.dropna()
    if len(g_cum) < 80:
        return 50.0

    g_z       = float(_ewm_zscore(g_cum).iloc[-1])
    raw_score = _zscore_to_score(g_z, scale=16.0)

    # Geopolitical premium: gold + oil joint elevation (war/supply-shock signature)
    # Proportional ramp — no hardcoded on/off threshold
    geo_boost = 0.0
    oil_col = next((c for c in ["WTI Crude Oil", "Brent Crude"] if c in cmd_r.columns), None)
    if oil_col:
        try:
            oil_r = cmd_r[oil_col].dropna()
            if len(oil_r) >= 80:
                o_roll = oil_r.rolling(20).sum() * 100
                o_roll = o_roll.dropna()
                if len(o_roll) > 60:
                    o_z = float(_ewm_zscore(o_roll).iloc[-1])
                    # Both positive → geopolitical premium, scaled to joint strength
                    if g_z > 0 and o_z > 0:
                        geo_boost = float(np.clip((g_z + o_z) * 5.0, 0.0, 20.0))
        except Exception:
            pass

    return float(np.clip(raw_score + geo_boost, 0.0, 100.0))


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

    if len(gold_r) < 80 or len(oil_r) < 80:
        return 50.0, {"gold_z": 0.0, "oil_z": 0.0, "gold_ret": 0.0, "oil_ret": 0.0}

    gold_cum = float(gold_r.iloc[-window:].sum()) * 100
    oil_cum  = float(oil_r.iloc[-window:].sum()) * 100

    # EWM z-scores vs 252-day trailing baseline (prevents new-normal adaptation)
    g_roll = gold_r.rolling(window).sum() * 100
    o_roll = oil_r.rolling(window).sum() * 100

    g_z = float(_ewm_zscore(g_roll.dropna()).iloc[-1]) if len(g_roll.dropna()) > 60 else 0.0
    o_z = float(_ewm_zscore(o_roll.dropna()).iloc[-1]) if len(o_roll.dropna()) > 60 else 0.0

    gold_contribution = float(np.clip(g_z * 17, -34, 42))
    oil_amplifier     = float(np.clip(o_z * 11, -20, 28))
    # Smooth conflict bonus: proportional to how far above neutral each signal is.
    # Previous flat +5 created a discontinuity at g_z=1, o_z=1 (±5 pts at threshold).
    # Now: ramp begins as soon as either signal is positive, scales with joint strength.
    g_excess       = max(0.0, g_z - 0.7)
    o_excess       = max(0.0, o_z - 0.7)
    conflict_bonus = float(np.clip((g_excess + o_excess) * 7.5, 0.0, 22.0))

    score = float(np.clip(50 + gold_contribution + oil_amplifier + conflict_bonus, 0, 100))
    return score, {
        "gold_z":   round(g_z, 2),
        "oil_z":    round(o_z, 2),
        "gold_ret": round(gold_cum, 2),
        "oil_ret":  round(oil_cum, 2),
    }


def _corr_accel_score(avg_corr: pd.Series, span: int = 20) -> float:
    """
    Correlation velocity score — first derivative of rolling correlation.

    Changed from 2nd derivative (acceleration) to 1st derivative (velocity):
    triple EWM smoothing created ~30d of lag, meaning the signal fired AFTER
    the correlation regime had already stabilised.  Velocity (1st deriv) has
    ~10d lag and is still orthogonal to the correlation level already captured
    by CIS geographic_diffusion.

    Returns 0–100 where 100 = correlation rising faster than any point in sample.
    """
    if avg_corr.empty or len(avg_corr) < 60:
        return 50.0
    smooth   = avg_corr.ewm(span=span).mean()
    velocity = smooth.diff().dropna()
    if len(velocity) < 40:
        return 50.0
    current_vel = float(velocity.iloc[-1])
    return float(np.clip((velocity < current_vel).mean() * 100, 0, 100))


def _market_confirmation_score(
    avg_corr: pd.Series,
    cmd_r: pd.DataFrame,
    eq_r: pd.DataFrame | None = None,
) -> tuple[float, dict]:
    """
    Market Confirmation Score (MCS) — 5 orthogonalized signals.
    Weights sum to 1.0. Returns (score, component_dict).
    """
    eq_vol     = _equity_vol_score(eq_r)
    rates_vol  = _rates_vol_score(cmd_r)
    cmd_vol    = _commodity_vol_score(cmd_r)
    safe_hav   = _safe_haven_score(cmd_r, eq_r)
    og_score, og_detail = _oil_gold_signal(cmd_r)
    corr_vel   = _corr_accel_score(avg_corr)

    w = _MODEL_CONFIG["mcs_weights"]

    mcs = (
        w["eq_vol"]     * eq_vol
        + w["rates_vol"]  * rates_vol
        + w["cmd_vol"]    * cmd_vol
        + w["safe_haven"] * safe_hav
        + w["oil_gold"]   * og_score
        + w["corr_accel"] * corr_vel
    )

    components = {
        "Equity Vol":        round(eq_vol,    1),
        "Rates Vol (TLT)":   round(rates_vol, 1),
        "Commodity Vol":     round(cmd_vol,   1),
        "Safe-Haven Bid":    round(safe_hav,  1),
        "Oil-Gold Signal":   round(og_score,  1),
        "Corr Velocity":     round(corr_vel,  1),
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
    from src.analysis.conflict_model import aggregate_portfolio_scores, build_market_signals
    from src.analysis.scenario_state import get_scenario

    # Inject live market signals into session_state so aggregate_portfolio_scores()
    # can rank conflicts by current market impact, not just static intensity.
    # This runs every time compute_risk_score() is called (TTL=300), ensuring
    # the "Lead Conflict" on the command center reflects today's market moves.
    try:
        import streamlit as _st
        _mf_signals = build_market_signals(cmd_r)
        if _mf_signals:
            _st.session_state["_mf_signals"] = _mf_signals
    except Exception:
        pass

    # Layer 1 + 2: Conflict Intensity and Transmission Pressure
    conflict_agg = aggregate_portfolio_scores()
    cis_portfolio = conflict_agg["cis"]
    tps_portfolio = conflict_agg["tps"]
    conflict_conf = conflict_agg["confidence"]

    # Layer 3: Market Confirmation
    mcs, mcs_components = _market_confirmation_score(avg_corr, cmd_r, eq_r)

    # Assembly
    _w = _MODEL_CONFIG["weights"]
    raw = (
        _w["cis"] * cis_portfolio
        + _w["tps"] * tps_portfolio
        + _w["mcs"] * mcs
    )

    # Scenario multiplier — hard clamp [0, 100] prevents score runaway.
    # geo_mult is user-controlled from scenario_state; baseline = 1.0.
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

    # Confidence intervals (quadrature combination of per-layer uncertainty)
    # Uncertainty grows as confidence falls; at 100% confidence → ±0 pts.
    # At 0% confidence → ±(weight * score) — i.e., layer is effectively unknown.
    _w = _MODEL_CONFIG["weights"]
    cis_err = (1.0 - conflict_conf) * cis_portfolio * _w["cis"]
    tps_err = (1.0 - conflict_conf) * tps_portfolio * _w["tps"]
    mcs_err = (1.0 - mcs_agreement_score) * mcs       * _w["mcs"]
    score_uncertainty = float(np.sqrt(cis_err**2 + tps_err**2 + mcs_err**2))
    score_low  = round(max(0.0,   total - score_uncertainty), 1)
    score_high = round(min(100.0, total + score_uncertainty), 1)

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
        "score_low":      score_low,
        "score_high":     score_high,
        "uncertainty":    round(score_uncertainty, 1),
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
        "corr_pct": round(_corr_accel_score(avg_corr), 1),
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

    Weights (from _MODEL_CONFIG["history_weights"]):
        40% equity vol  +  35% oil-gold  +  13% commodity vol  +  12% corr accel

    Aligned to match the live model's top-two drivers (equity vol and oil-gold).
    Replaced legacy correlation-percentile with correlation-acceleration to avoid
    double-counting the correlation-level signal already present in CIS.
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

    hw = _MODEL_CONFIG["history_weights"]

    # 1. Correlation velocity (1st derivative — matches live _corr_accel_score update)
    # Switched from acceleration (2nd deriv, ~30d lag) to velocity (1st deriv, ~10d lag)
    smooth   = avg_corr.ewm(span=20).mean()
    velocity = smooth.diff()
    v_mean   = velocity.ewm(span=252).mean()
    v_std    = velocity.ewm(span=252).std().replace(0, np.nan)
    corr_accel_score = (50 + ((velocity - v_mean) / v_std.replace(0, np.nan)).clip(-3, 3) * 16.67).clip(0, 100)

    # 2. Raw commodity vol z-score (span=252, no residualization — matches live)
    energy_metals = ["WTI Crude Oil", "Brent Crude", "Natural Gas", "Gold", "Silver", "Copper"]
    vol_cols = [c for c in energy_metals if c in cmd_r.columns]
    if vol_cols:
        rv        = cmd_r[vol_cols].rolling(20).std() * np.sqrt(252) * 100
        avg_vol   = rv.mean(axis=1)
        v_mean2   = avg_vol.ewm(span=252).mean()
        v_std2    = avg_vol.ewm(span=252).std().replace(0, np.nan)
        vol_score = (50 + ((avg_vol - v_mean2) / v_std2).clip(-3, 3) * 14).clip(0, 100)
    else:
        vol_score = pd.Series(50.0, index=avg_corr.index)

    # 3. Oil-Gold signal (span=252, smooth conflict bonus — matches live)
    gold_col = next((c for c in ["Gold"] if c in cmd_r.columns), None)
    oil_col  = next((c for c in ["WTI Crude Oil", "Brent Crude"] if c in cmd_r.columns), None)
    if gold_col and oil_col:
        g_cum  = cmd_r[gold_col].rolling(20).sum() * 100
        o_cum  = cmd_r[oil_col].rolling(20).sum()  * 100
        g_mean = g_cum.ewm(span=252).mean()
        g_std  = g_cum.ewm(span=252).std().replace(0, np.nan)
        o_mean = o_cum.ewm(span=252).mean()
        o_std  = o_cum.ewm(span=252).std().replace(0, np.nan)
        g_z    = ((g_cum - g_mean) / g_std).clip(-3.5, 3.5)
        o_z    = ((o_cum - o_mean) / o_std).clip(-3.5, 3.5)
        # Smooth ramp bonus (matches live fix — no binary threshold)
        g_excess = (g_z - 0.7).clip(lower=0)
        o_excess = (o_z - 0.7).clip(lower=0)
        conflict_bonus = ((g_excess + o_excess) * 5.0).clip(upper=15.0)
        og_score = (50 + g_z * 17 + o_z * 11 + conflict_bonus).clip(0, 100)
    else:
        og_score = pd.Series(50.0, index=avg_corr.index)

    # 4. Equity vol proxy (span=252)
    if eq_r is not None and not eq_r.empty:
        eq_vol_raw   = eq_r.rolling(20).std().mean(axis=1) * np.sqrt(252) * 100
        ev_mean      = eq_vol_raw.ewm(span=252).mean()
        ev_std       = eq_vol_raw.ewm(span=252).std().replace(0, np.nan)
        eq_vol_score = (50 + ((eq_vol_raw - ev_mean) / ev_std).clip(-3, 3) * 15).clip(0, 100)
    else:
        eq_vol_score = pd.Series(50.0, index=avg_corr.index)

    aligned = pd.concat(
        [corr_accel_score, vol_score, og_score, eq_vol_score], axis=1
    ).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    aligned.columns = ["corr_accel", "vol", "og", "eq"]

    return (
        hw["eq_vol"]     * aligned["eq"]
        + hw["oil_gold"]   * aligned["og"]
        + hw["cmd_vol"]    * aligned["vol"]
        + hw["corr_accel"] * aligned["corr_accel"]
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
            valueformat=".1f",
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

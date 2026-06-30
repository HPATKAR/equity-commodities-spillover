"""
Geopolitical Risk Score — 3-Layer Architecture.

Top-level score:
  Global Geo Risk = 35% Conflict Intensity + 30% Transmission Pressure + 35% Market Confirmation

Layer 1 — Conflict Intensity (35%)
  Delegated to conflict_model.py (CIS/TPS per-conflict, then portfolio aggregate).

Layer 2 — Transmission Pressure (30%)
  Portfolio TPS from conflict_model.py.

Layer 3 — Market Confirmation (35%)
  Market signals: equity vol, rates vol, commodity vol, safe-haven behavior,
  oil-gold joint signal, correlation velocity.
  EWM z-scores (span=252) against a one-year baseline.

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
    # Architecture follows Caldara-Iacoviello (2022) GPR intuition:
    #   Primary: news-flow (most responsive, least market-contaminated)
    #   Secondary: conflict events intensity (ACLED-calibrated, not manual judgment)
    #   Tertiary: physical transmission via chokepoints (PortWatch throughput)
    #   Confirmation: oil-gold only — geo supply-shock signature, not generic vol
    "weights": {
        "news_gpr":   0.40,  # News GPR (C&I-style threat+act classification)
        "cis":        0.30,  # Conflict Events Intensity (ACLED-calibrated CIS)
        "chokepoint": 0.20,  # Chokepoint/Shipping Stress (PortWatch + transmission)
        "mcs":        0.10,  # Market Confirmation: oil-gold + commodity vol ONLY
    },
    # MCS sub-signal weights (must sum to 1.0)
    # Equity vol, rates vol, safe-haven, corr-accel removed: they fire on any
    # market stress (rate cycles, earnings, growth scares) — not geo-specific.
    "mcs_weights": {
        "oil_gold": 0.70,   # war/supply-shock signature: gold+oil dual elevation
        "cmd_vol":  0.30,   # geo-linked commodity vol: captures supply disruptions
    },
    # risk_score_history() weights — MCS-proxy only (CIS/news not available historically)
    # eq_vol removed: fires on any market stress, not geo-specific.
    "history_weights": {
        "oil_gold":   0.60,  # level-blended: momentum + price-history rank
        "cmd_vol":    0.30,  # geo-linked supply disruptions
        "corr_accel": 0.10,  # co-movement velocity
    },
}


# ── Cached auxiliary fetches ──────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False, max_entries=3)
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
    """
    Map EWM z-score → [0, 100] via logistic sigmoid.

    Sigmoid replaces the prior linear + hard-clip: the logistic naturally
    bounds the output in (0, 100) with no clipping artefacts at extremes.
    Steepness k = scale/25 matches the slope of the old linear function at
    z=0 (the most common operating range), so on-centre behaviour is unchanged.

    z=0 → 50.0 (by construction)
    z=+2 → ~77 (scale=15), ~69 (scale=10)  [linear was 80, 70]
    z=+5 → ~95+ — saturates smoothly instead of hard-capping at 100
    """
    k = scale / 25.0
    return float(100.0 / (1.0 + np.exp(-k * float(z))))


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
        tlt_r = np.log(tlt / tlt.shift(1)).dropna()
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


# ── Chokepoint/Shipping Stress Layer ──────────────────────────────────────────

def _chokepoint_stress_score() -> float:
    """
    Chokepoint/Shipping Stress Score (0–100).

    Combines live PortWatch disruption (from session_state._hormuz_disruption)
    with structural transmission pressure from active CONFLICTS registry.

    This is the explicit physical-transmission layer of the GRS — representing
    how much actual throughput disruption exists at critical chokepoints right now,
    not a judgment about transmission probabilities.

    No disruption + no active chokepoint conflicts → ~15 (low baseline)
    Active Red Sea rerouting (partial disruption) → ~55
    Active Hormuz blockade (near-complete) → ~90+
    """
    hormuz_d = 0.0
    try:
        hormuz_d = float(np.clip(
            st.session_state.get("_hormuz_disruption", 0.0), 0.0, 1.0
        ))
    except Exception:
        pass

    from src.data.config import CONFLICTS
    _STATE_MULT_CP = {"active": 1.0, "latent": 0.35, "frozen": 0.15}
    max_cp = 0.0
    for c in CONFLICTS:
        sm = _STATE_MULT_CP.get(c.get("state", "active"), 1.0)
        tx = c.get("transmission", {})
        cp = max(float(tx.get("chokepoint", 0.0)), float(tx.get("shipping", 0.0)) * 0.75)
        max_cp = max(max_cp, cp * sm)

    base = max_cp * 65.0  # max active conflict's chokepoint strength → base score

    if hormuz_d > 0.05:
        live = 20.0 + hormuz_d * 78.0  # 5% disruption→24, full blockade→98
        return float(np.clip(max(base, live), 0.0, 100.0))
    return float(np.clip(base, 0.0, 100.0))


# ── News GPR Fallback (when RSS feed unavailable) ─────────────────────────────

def _news_gpr_fallback(conflict_detail: dict) -> float:
    """
    Proxy News GPR from conflict escalation signals when live feed is unavailable.
    Escalating active conflicts → higher score; de-escalating → lower.
    Returns 0–100 calibrated to approximate the live News GPR range.
    """
    if not conflict_detail:
        return 35.0  # neutral prior — neither high nor low
    esc_map = {"escalating": 80.0, "stable": 42.0, "de-escalating": 18.0}
    w_sum = tot = 0.0
    for detail in conflict_detail.values():
        cis = float(detail.get("cis", 0.0))
        esc = detail.get("escalation", "stable")
        w_sum += cis * esc_map.get(esc, 42.0)
        tot   += cis
    return float(np.clip(w_sum / max(tot, 1e-9), 0.0, 100.0)) if tot > 0 else 35.0


# ── Market Confirmation Layer ─────────────────────────────────────────────────

def _market_confirmation_score(
    avg_corr: pd.Series,
    cmd_r: pd.DataFrame,
    eq_r: pd.DataFrame | None = None,
) -> tuple[float, dict]:
    """
    Market Confirmation Score (MCS) — geo-specific signals ONLY.

    70% Oil-Gold joint signal: supply-shock / safe-haven dual elevation,
        the cleanest market signature of a geopolitical war/blockade.
    30% Commodity Vol: captures geo supply disruptions via vol spike
        (symmetric — fires on both price spike and price crash from disruption).

    Equity vol, rates vol, safe-haven bid, and correlation velocity removed:
    all four fire on generic market stress (rate cycles, earnings, COVID
    demand crashes) and are not specific to geopolitical events.
    """
    og_score, og_detail = _oil_gold_signal(cmd_r)
    cmd_vol              = _commodity_vol_score(cmd_r)

    w = _MODEL_CONFIG["mcs_weights"]
    mcs = w["oil_gold"] * og_score + w["cmd_vol"] * cmd_vol

    components = {
        "Oil-Gold Signal": round(og_score, 1),
        "Commodity Vol":   round(cmd_vol,  1),
    }
    return float(np.clip(mcs, 0, 100)), components


# ── Main scorer ────────────────────────────────────────────────────────────────

def compute_risk_score(
    avg_corr: pd.Series,
    cmd_r: pd.DataFrame,
    eq_r: pd.DataFrame | None = None,
) -> dict:
    """
    Compute composite Geopolitical Risk Score (GRS), 0–100.

    4-Layer Architecture (Caldara-Iacoviello + ACLED-style):
      40% News GPR      — RSS threat+act classification (C&I-style)
      30% Conflict CIS  — ACLED-calibrated event intensity (conflict_model.py)
      20% Chokepoint    — PortWatch throughput disruption + transmission pressure
      10% Market Conf   — oil-gold + commodity vol (geo-specific signals only)

    The previous architecture used 35% generic market vol (eq_vol, rates_vol),
    which inflated scores during rate cycles and earnings events unrelated to
    geopolitical stress.  News GPR now leads as the most responsive and least
    market-contaminated primary signal.

    Scenario multiplier applied after assembly.
    Returns dict with score, label, color, components, confidence, and detail.
    """
    from src.analysis.conflict_model import aggregate_portfolio_scores, build_market_signals
    from src.analysis.scenario_state import get_scenario

    # Inject live market signals into session_state for conflict market-freshness ranking
    try:
        _mf_signals = build_market_signals(cmd_r)
        if _mf_signals:
            st.session_state["_mf_signals"] = _mf_signals
    except Exception:
        pass

    # ── Layer 1: News GPR (Caldara-Iacoviello style) ───────────────────────────
    news_live = False
    news_gpr_out: dict = {
        "news_gpr": None, "threat_score": None, "act_score": None,
        "alpha": None, "n_threat": 0, "n_act": 0, "news_per_conflict": {},
    }
    try:
        from src.analysis.gpr_news import get_news_gpr_layer
        news = get_news_gpr_layer()
        news_gpr_out = {
            "news_gpr":          news["news_gpr"],
            "threat_score":      news["threat_score"],
            "act_score":         news["act_score"],
            "alpha":             news["alpha"],
            "n_threat":          news["n_threat"],
            "n_act":             news["n_act"],
            "news_per_conflict": news["per_conflict"],
        }
        if news["data_status"] == "live":
            news_layer = float(news["news_gpr"])   # may be 0 if feed returned no classified headlines
            news_live  = news_layer > 0
        else:
            news_layer = None  # compute fallback after conflict_agg
    except Exception:
        news_layer = None

    # ── Layer 2: Conflict Events Intensity ────────────────────────────────────
    conflict_agg  = aggregate_portfolio_scores()
    cis_portfolio = conflict_agg["cis"]
    tps_portfolio = conflict_agg["tps"]   # kept for backward compat / display
    conflict_conf = conflict_agg["confidence"]

    # Conflict-state floor: the news layer cannot fall below 60% of what the
    # conflict registry implies.  A weak RSS feed (few classified headlines) should
    # not override the structural signal of 3 active escalating wars.
    # e.g. 3 escalating conflicts → fallback ≈ 67 → floor = 40.
    # Strong live feed (news_gpr > floor) passes through unchanged.
    _conflict_floor = _news_gpr_fallback(conflict_agg.get("conflict_detail", {})) * 0.60

    if news_layer is None:
        news_layer = _news_gpr_fallback(conflict_agg.get("conflict_detail", {}))
    else:
        news_layer = max(news_layer, _conflict_floor)

    # ── Layer 3: Chokepoint / Shipping Stress ─────────────────────────────────
    chokepoint = _chokepoint_stress_score()

    # ── Layer 4: Market Confirmation (oil-gold + commodity vol only) ──────────
    mcs, mcs_components = _market_confirmation_score(avg_corr, cmd_r, eq_r)

    # ── Assembly ──────────────────────────────────────────────────────────────
    _w = _MODEL_CONFIG["weights"]
    raw = (
        _w["news_gpr"]   * news_layer
        + _w["cis"]        * cis_portfolio
        + _w["chokepoint"] * chokepoint
        + _w["mcs"]        * mcs
    )

    scenario = get_scenario()
    geo_mult = scenario.get("geo_mult", 1.0)
    total    = float(np.clip(raw * geo_mult, 0.0, 100.0))

    # ── Confidence overlay ────────────────────────────────────────────────────
    news_conf = 0.90 if news_live else 0.45   # live feed → high conf; fallback → lower
    mcs_signal_agreement = float(np.std(list(mcs_components.values()))) / 50.0
    mcs_agreement_score  = float(np.clip(1.0 - mcs_signal_agreement, 0.0, 1.0))
    overall_confidence   = float(
        0.35 * conflict_conf
        + 0.35 * news_conf
        + 0.20 * mcs_agreement_score
        + 0.10 * (1.0 if not avg_corr.empty else 0.5)
    )

    # Confidence intervals via quadrature per-layer uncertainty
    cis_err  = (1.0 - conflict_conf)     * cis_portfolio * _w["cis"]
    news_err = (1.0 - news_conf)         * news_layer    * _w["news_gpr"]
    mcs_err  = (1.0 - mcs_agreement_score) * mcs         * _w["mcs"]
    cp_err   = 0.0   # PortWatch data either present or fallback to static
    score_uncertainty = float(np.sqrt(cis_err**2 + news_err**2 + mcs_err**2 + cp_err**2))
    score_low  = round(max(0.0,   total - score_uncertainty), 1)
    score_high = round(min(100.0, total + score_uncertainty), 1)

    if total < 25:   label, color = "Low",      "#2e7d32"
    elif total < 50: label, color = "Moderate", "#8E9AAA"
    elif total < 75: label, color = "Elevated", "#e67e22"
    else:            label, color = "High",     "#c0392b"

    return {
        "score":          round(total, 1),
        "score_low":      score_low,
        "score_high":     score_high,
        "uncertainty":    round(score_uncertainty, 1),
        "label":          label,
        "color":          color,
        "confidence":     round(overall_confidence, 2),
        "scenario":       scenario.get("label", "Base"),
        # Layer breakdown (new 4-layer architecture)
        "news_gpr_layer": round(news_layer,    1),
        "cis":            round(cis_portfolio, 1),
        "chokepoint":     round(chokepoint,    1),
        "mcs":            round(mcs,           1),
        # Backward compat aliases (pages that reference "tps" keep working)
        "tps":            round(chokepoint,    1),   # semantic: chokepoint IS the transmission layer
        # News GPR sub-fields
        **news_gpr_out,
        # Detail
        "conflict_detail": conflict_agg.get("conflict_detail", {}),
        "top_conflict":    conflict_agg.get("top_conflict"),
        "mcs_components":  mcs_components,
        # Legacy fields
        "weights": {
            "news_gpr":    _w["news_gpr"],
            "conflict_intensity": _w["cis"],
            "chokepoint":  _w["chokepoint"],
            "market_conf": _w["mcs"],
            # old keys kept for compat
            "transmission": _w["chokepoint"],
        },
        "corr_pct": round(_corr_accel_score(avg_corr), 1),
        "components": {
            f"News GPR ({int(_w['news_gpr']*100)}%)":      round(news_layer,    1),
            f"Conflict Events ({int(_w['cis']*100)}%)":    round(cis_portfolio, 1),
            f"Chokepoint Stress ({int(_w['chokepoint']*100)}%)": round(chokepoint, 1),
            f"Market Conf ({int(_w['mcs']*100)}%)":        round(mcs,           1),
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
    VIX-mimic cross-asset realized volatility index (0–100).

    Methodology: 20-day annualized realized volatility across equity indices
    (55%) and commodities (45%), combined into one composite vol series,
    then EWM z-scored against a 3-year (756-day) baseline and mapped through
    a sigmoid to [0, 100].

    Why this works:
    - 2008 GFC, 2020 COVID, 2022 Ukraine → massive realized vol spikes → HIGH  ✓
    - 2012-13 QE era → VIX was 13-18 (calm) → LOW                             ✓
    - 2011 Euro crisis → equity vol spike → ELEVATED                           ✓
    - Peacetime 2015-19 → low realized vol → LOW-MODERATE                      ✓

    3-year EWM baseline (vs. old span=252) prevents the "war normal" effect:
    sustained high vol in a multi-year conflict keeps the z-score elevated
    for longer before the baseline adapts.
    """
    VOL_WIN = 20    # 20-day realized vol window (matches VIX convention)
    SPAN    = 756   # 3-year EWM baseline

    if cmd_r.empty:
        return pd.Series(dtype=float)

    series = []

    # Equity realized vol (55% weight) — the core VIX signal
    if eq_r is not None and not eq_r.empty:
        eq_rv = (
            eq_r.rolling(VOL_WIN, min_periods=10).std()
            .mean(axis=1) * np.sqrt(252) * 100
        ).dropna()
        if not eq_rv.empty:
            series.append((eq_rv, 0.55))

    # Commodity realized vol (45% weight) — geo supply shocks, energy crises
    cmd_cols = [c for c in ["WTI Crude Oil", "Brent Crude", "Natural Gas",
                             "Gold", "Silver", "Copper"] if c in cmd_r.columns]
    if cmd_cols:
        cmd_rv = (
            cmd_r[cmd_cols].rolling(VOL_WIN, min_periods=5).std()
            .mean(axis=1) * np.sqrt(252) * 100
        ).dropna()
        if not cmd_rv.empty:
            series.append((cmd_rv, 0.45))

    if not series:
        return pd.Series(dtype=float)

    # Weighted composite on shared index
    total_w  = sum(w for _, w in series)
    combined = sum(
        rv.reindex(series[0][0].index).ffill() * (w / total_w)
        for rv, w in series
    ).dropna()

    if len(combined) < SPAN // 4:
        return pd.Series(dtype=float)

    # EWM z-score against 3-year baseline
    mu    = combined.ewm(span=SPAN, min_periods=120).mean()
    sigma = combined.ewm(span=SPAN, min_periods=120).std().replace(0, np.nan)
    z     = ((combined - mu) / sigma).clip(-3.5, 3.5)

    # Sigmoid → [0, 100]  (z=0 → 50, z=+2 → ~77, z=+3 → ~88)
    score = (100.0 / (1.0 + np.exp(-0.56 * z))).clip(0, 100)

    return score.round(1)


# ── Market Fear Index ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
def market_fear_index(period: str = "3y") -> pd.Series:
    """
    Market Fear Index (MFI) — CBOE implied volatility composite, 0–100.

    Weights:
        80%  VIX  (^VIX)  — S&P 500 implied vol; the canonical fear gauge
        10%  OVX  (^OVX)  — Crude oil implied vol; geo/energy supply shocks
        10%  GVZ  (^GVZ)  — Gold implied vol; safe-haven demand and monetary stress

    Construction:
        1. Fetch each index from Yahoo Finance (daily Close).
        2. EWM z-score each independently against a 3-year (756-day) baseline.
           3-year span: slow enough that a sustained multi-year crisis stays elevated;
           span=252 adapted in ~12 months (Ukraine 2023 read z≈0 while war ongoing).
        3. Weighted composite z = 0.80·z_VIX + 0.10·z_OVX + 0.10·z_GVZ.
           Where OVX/GVZ are unavailable (pre-2007/2008), their weight redistributes
           to VIX so the composite z is always on the same scale.
        4. Sigmoid → [0, 100]:  score = 100 / (1 + exp(−0.56·z))
           z=0 → 50  ·  z=+2 → 77  ·  z=+3 → 88  ·  z=−2 → 23
    """
    SPAN     = 756
    CONFIGS  = [("VIX", "^VIX", 0.70), ("OVX", "^OVX", 0.20), ("GVZ", "^GVZ", 0.10)]

    z_series: dict[str, pd.Series] = {}
    for name, ticker, _ in CONFIGS:
        try:
            s = yf.Ticker(ticker).history(period=period)["Close"].dropna()
            if len(s) < 120:
                continue
            mu    = s.ewm(span=SPAN, min_periods=120).mean()
            sigma = s.ewm(span=SPAN, min_periods=120).std().replace(0, np.nan)
            z_series[name] = ((s - mu) / sigma).clip(-3.5, 3.5)
        except Exception:
            pass

    if "VIX" not in z_series:
        return pd.Series(dtype=float)

    vix_idx = z_series["VIX"].index

    # Align OVX / GVZ to VIX index; fill gaps with VIX z-score so weights stay valid
    aligned: dict[str, pd.Series] = {"VIX": z_series["VIX"]}
    for name in ("OVX", "GVZ"):
        if name in z_series:
            aligned[name] = z_series[name].reindex(vix_idx, method="ffill")
        else:
            aligned[name] = z_series["VIX"]   # full fallback to VIX

    weights = dict(zip(["VIX", "OVX", "GVZ"], [0.80, 0.10, 0.10]))
    composite_z = sum(aligned[n].fillna(aligned["VIX"]) * w for n, w in weights.items())

    score = (100.0 / (1.0 + np.exp(-0.56 * composite_z))).clip(0, 100)
    score.index = score.index.tz_localize(None) if score.index.tz is not None else score.index
    return score.round(1)


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
    conflict_start: str | None = None,
) -> go.Figure:
    """
    Line chart of historical risk score with regime bands and event markers.

    Parameters
    ----------
    conflict_start : str | None
        ISO date string (YYYY-MM-DD) from which the full 3-layer GRS (CIS + TPS + MCS)
        is included.  Everything before this date is MCS-proxy only.  When provided, a
        vertical boundary line and shaded region are added to annotate the distinction.
        None (default) = entire series labelled as proxy with a chart annotation.
    """
    if events is None:
        events = GEOPOLITICAL_EVENTS

    # Normalise index to tz-naive so event timestamp comparisons always work
    if score_series.index.tz is not None:
        score_series = score_series.copy()
        score_series.index = score_series.index.tz_localize(None)

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

    # ── Proxy boundary annotation ──────────────────────────────────────────
    # risk_score_history() uses only the MCS (market confirmation) layer —
    # CIS and TPS cannot be reconstructed at daily frequency from public data.
    # Make this explicit so a reviewer is not misled into thinking the full
    # 3-layer model applies to the entire historical series.
    idx = score_series.index
    if conflict_start and len(idx) > 0:
        cs_ts = pd.Timestamp(conflict_start)
        if idx[0] < cs_ts < idx[-1]:
            # Left region: MCS proxy only; right region: full GRS
            fig.add_vrect(
                x0=str(idx[0]), x1=conflict_start,
                fillcolor="rgba(100,100,100,0.06)", opacity=1.0,
                layer="below", line_width=0,
            )
            fig.add_vline(
                x=conflict_start,
                line=dict(color="#CFB991", width=1.2, dash="dashdot"),
            )
            fig.add_annotation(
                x=conflict_start, y=92,
                text="← MCS proxy only · Full model →",
                showarrow=False,
                font=dict(size=7.5, color="#CFB991", family="JetBrains Mono, monospace"),
                xanchor="center", yanchor="top",
                bgcolor="rgba(10,12,20,0.75)", borderpad=2,
            )
        else:
            # Entire series is proxy — add a single corner label
            fig.add_annotation(
                x=0.01, y=0.97, xref="paper", yref="paper",
                text="Historical: MCS proxy only (market signals — conflict layer excluded)",
                showarrow=False,
                font=dict(size=7, color="#8E9AAA", family="JetBrains Mono, monospace"),
                xanchor="left", yanchor="top",
                bgcolor="rgba(10,12,20,0.65)", borderpad=2,
            )
    else:
        # No boundary provided — label the whole chart as a proxy
        fig.add_annotation(
            x=0.01, y=0.97, xref="paper", yref="paper",
            text="Historical: MCS proxy only (market signals — conflict layer excluded)",
            showarrow=False,
            font=dict(size=7, color="#8E9AAA", family="JetBrains Mono, monospace"),
            xanchor="left", yanchor="top",
            bgcolor="rgba(10,12,20,0.65)", borderpad=2,
        )

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

    # 1-year rolling high / low bands (~252 trading days)
    window = 252
    roll_high = score_series.rolling(window, min_periods=5).max()
    roll_low  = score_series.rolling(window, min_periods=5).min()

    roll_ma = score_series.rolling(window, min_periods=20).mean()
    fig.add_trace(go.Scatter(
        x=roll_ma.index, y=roll_ma.values,
        name=f"{window}d MA",
        line=dict(color="rgba(207,185,145,0.90)", width=1.4),
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=roll_high.index, y=roll_high.values,
        name=f"{window}d High",
        line=dict(color="rgba(207,185,145,0.55)", width=1.0, dash="dash"),
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=roll_low.index, y=roll_low.values,
        name=f"{window}d Low",
        line=dict(color="rgba(142,154,170,0.55)", width=1.0, dash="dash"),
        fill="tonexty",
        fillcolor="rgba(207,185,145,0.05)",
        showlegend=True,
    ))

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

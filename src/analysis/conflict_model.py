"""
Conflict Intensity Model.

Computes per-conflict Conflict Intensity Score (CIS) and Transmission
Pressure Score (TPS) for every entry in CONFLICTS registry.

Architecture:
  CIS(w)  = weighted blend of 6 intensity dimensions     → 0–100
  TPS(w)  = weighted sum of 12 transmission channels     → 0–100
  Confidence(w) = data coverage + freshness + agreement  → 0–1

Portfolio aggregates:
  CIS_portfolio  = intensity-weighted average of all active conflict CIS scores
  TPS_portfolio  = intensity-weighted average of all active conflict TPS scores

Called by risk_score.py as the Conflict Intensity Layer (40%) and
Transmission Pressure Layer (35%) of the top-level geopolitical risk score.
"""

from __future__ import annotations

import datetime
import math
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.data.config import CONFLICTS


# ── CIS dimension weights ─────────────────────────────────────────────────────

_CIS_WEIGHTS: dict[str, float] = {
    "deadliness":           0.22,
    "civilian_danger":      0.15,
    "geographic_diffusion": 0.12,
    "fragmentation":        0.08,
    "escalation_trend":     0.20,
    "recency":              0.13,
    "source_coverage":      0.10,
}

_ESCALATION_MAP: dict[str, float] = {
    "escalating":    1.00,
    "stable":        0.50,
    "de-escalating": 0.00,
}

# ── TPS channel weights ───────────────────────────────────────────────────────

_TPS_WEIGHTS: dict[str, float] = {
    "oil_gas":       0.18,
    "metals":        0.10,
    "agriculture":   0.08,
    "shipping":      0.12,
    "chokepoint":    0.10,
    "sanctions":     0.12,
    "equity_sector": 0.08,
    "fx":            0.06,
    "inflation":     0.07,
    "supply_chain":  0.05,
    "credit":        0.02,
    "energy_infra":  0.02,
}


# ── State multipliers ─────────────────────────────────────────────────────────

_STATE_MULT: dict[str, float] = {
    "active":  1.00,
    "latent":  0.35,
    "frozen":  0.15,
}

# ── Model configuration (single source of truth for conflict_model) ───────────

_CM_CONFIG: dict = {
    # Confidence score sub-weights (must sum to 1.0)
    "confidence_weights": {
        "source_coverage": 0.30,
        "data_confidence": 0.25,
        "freshness":       0.25,
        "completeness":    0.20,
    },
    # Staleness cap: conflicts not updated within this many days get a
    # confidence penalty and a CIS soft-cap to prevent stale data inflating rank
    "staleness_warn_days":  90,   # yellow flag
    "staleness_cap_days":   180,  # hard cap applied
    "staleness_cis_cap":    65.0, # CIS capped at this value when stale
}


# ── Recency decay ─────────────────────────────────────────────────────────────

def _recency_score(conflict: dict) -> float:
    """
    For active conflicts: peaks at 1.0 when new, decays to 0.3 at 365 days.
    For latent conflicts: flat 0.35.
    For ended conflicts: returns 0.0 (shouldn't be in active CONFLICTS list).
    """
    state = conflict.get("state", "active")
    if state == "latent":
        return 0.35
    if state == "frozen":
        return 0.15

    today  = datetime.date.today()
    start  = conflict.get("start", today)
    days   = max((today - start).days, 1)
    # Decay: 1.0 at day 1 → 0.3 at day 365 (exponential)
    return max(0.30, math.exp(-days / 730))   # half-life ~2 years


def _freshness_score(conflict: dict) -> float:
    """
    How fresh is the last manual update?
    Returns 0–1 based on days since last_updated.
    """
    last = conflict.get("last_updated")
    if last is None:
        return 0.20   # unknown freshness → penalize
    today  = datetime.date.today()
    days   = (today - last).days
    if days <= 7:
        return 1.00
    elif days <= 30:
        return 0.80
    elif days <= 90:
        return 0.55
    elif days <= 180:
        return 0.35
    else:
        return 0.15


# ── Staleness check ───────────────────────────────────────────────────────────

def _check_conflict_freshness(conflict: dict) -> tuple[bool, bool]:
    """
    Returns (is_stale_warn, is_stale_cap).

    is_stale_warn — last_updated > staleness_warn_days (90d): yellow flag.
    is_stale_cap  — last_updated > staleness_cap_days (180d): hard cap on CIS.

    Used by compute_cis() and compute_confidence() to prevent stale manual
    data from inflating a conflict's rank above fresher, market-confirmed ones.
    """
    last = conflict.get("last_updated")
    if last is None:
        return True, True
    days = (datetime.date.today() - last).days
    return (
        days > _CM_CONFIG["staleness_warn_days"],
        days > _CM_CONFIG["staleness_cap_days"],
    )


# ── Per-conflict scorers ──────────────────────────────────────────────────────


def compute_cis(conflict: dict) -> float:
    """
    Conflict Intensity Score for a single conflict.
    Returns value in [0, 100].

    When ACLED API is configured, live event counts (last 30 days) REPLACE the
    hardcoded deadliness, geographic_diffusion, and escalation_trend dimensions
    entirely. civilian_danger and fragmentation remain structural (no live proxy).
    This is the full live replacement path (not just a multiplicative nudge).
    """
    state_mult = _STATE_MULT.get(conflict.get("state", "active"), 1.0)

    # ── Start from hardcoded registry values (structural baseline) ────────────
    dims = {
        "deadliness":           float(conflict.get("deadliness",           0.5)),
        "civilian_danger":      float(conflict.get("civilian_danger",      0.5)),
        "geographic_diffusion": float(conflict.get("geographic_diffusion", 0.3)),
        "fragmentation":        float(conflict.get("fragmentation",        0.2)),
        "escalation_trend":     _ESCALATION_MAP.get(
                                    conflict.get("escalation_trend", "stable"), 0.5),
        "recency":              _recency_score(conflict),
        "source_coverage":      float(conflict.get("source_coverage",      0.7)),
    }

    # ── ACLED live dimension replacement (when API is configured) ─────────────
    # acled_to_cis_dimensions() returns only the keys ACLED can replace:
    #   deadliness, geographic_diffusion, escalation_trend
    # The remaining keys (civilian_danger, fragmentation) stay structural.
    acled_nudge = 1.0
    try:
        from src.data.acled import fetch_acled_intensity, acled_to_cis_dimensions, acled_configured
        if acled_configured():
            acled_id = conflict.get("acled_id", "")
            if acled_id:
                result = fetch_acled_intensity(acled_id, days=30)
                live_dims = acled_to_cis_dimensions(result, conflict)
                # Replace hardcoded values with live ACLED measurements
                if "deadliness" in live_dims:
                    dims["deadliness"] = live_dims["deadliness"]
                if "geographic_diffusion" in live_dims:
                    dims["geographic_diffusion"] = live_dims["geographic_diffusion"]
                if "escalation_trend" in live_dims:
                    dims["escalation_trend"] = _ESCALATION_MAP.get(
                        live_dims["escalation_trend"], 0.5)
    except Exception:
        pass  # fall through to hardcoded baseline

    raw = sum(_CIS_WEIGHTS[k] * dims[k] for k in _CIS_WEIGHTS)
    cis = float(np.clip(raw * state_mult * acled_nudge * 100, 0, 100))

    # Staleness cap: prevent stale manual data from outranking fresh, market-
    # confirmed conflicts. Cap only kicks in after staleness_cap_days (180d).
    _, is_cap = _check_conflict_freshness(conflict)
    if is_cap:
        cis = min(cis, _CM_CONFIG["staleness_cis_cap"])

    return cis


def compute_tps(conflict: dict) -> float:
    """
    Transmission Pressure Score for a single conflict.
    Returns value in [0, 100].
    """
    state_mult = _STATE_MULT.get(conflict.get("state", "active"), 1.0)
    tx = conflict.get("transmission", {})
    raw = sum(_TPS_WEIGHTS[ch] * float(tx.get(ch, 0.0)) for ch in _TPS_WEIGHTS)
    return float(np.clip(raw * state_mult * 100, 0, 100))


def compute_confidence(conflict: dict) -> float:
    """
    Confidence score for a single conflict's scoring.
    Returns value in [0, 1].

    Weights from _CM_CONFIG["confidence_weights"].
    Applies additional staleness penalties at 90d (warn) and 180d (cap) thresholds.
    """
    cw = _CM_CONFIG["confidence_weights"]

    source_cov    = float(conflict.get("source_coverage",   0.70))
    data_conf     = float(conflict.get("data_confidence",   0.70))
    freshness     = _freshness_score(conflict)

    # Dimension completeness — penalize if key dimensions are at exact defaults
    _defaults = {"deadliness": 0.5, "civilian_danger": 0.5,
                 "geographic_diffusion": 0.3, "fragmentation": 0.2}
    missing = sum(1 for k, v in _defaults.items()
                  if abs(conflict.get(k, v) - v) < 1e-6)
    completeness = max(0.0, 1.0 - missing * 0.15)

    confidence = (
        cw["source_coverage"] * source_cov
        + cw["data_confidence"] * data_conf
        + cw["freshness"]       * freshness
        + cw["completeness"]    * completeness
    )

    # Staleness penalty (additive on top of freshness already captured above)
    is_warn, is_cap = _check_conflict_freshness(conflict)
    if is_cap:
        confidence *= 0.70   # hard reduction: stale data → less reliable
    elif is_warn:
        confidence *= 0.90   # soft reduction: getting stale

    return float(np.clip(confidence, 0.0, 1.0))


def compute_trend(conflict: dict) -> str:
    """Human-readable trend label from escalation_trend field."""
    t = conflict.get("escalation_trend", "stable")
    return {"escalating": "rising", "stable": "stable",
            "de-escalating": "falling"}.get(t, "stable")


def compute_market_freshness(conflict: dict, market_data: dict) -> float:
    """
    Market-freshness multiplier [0.7, 1.5] for a single conflict.

    Measures how actively this conflict is moving live markets RIGHT NOW.
    Conflicts whose primary transmission channels show large live moves get
    a boost; conflicts whose channels are quiet get a mild discount.

    market_data keys (all optional, float):
        brent_pct_1d    — Brent crude 1-day % change
        wti_pct_1d      — WTI crude 1-day % change
        natgas_pct_1d   — Natural gas 1-day % change
        wheat_pct_1d    — Wheat 1-day % change
        tanker_disruption — 0–1 (1 = complete blockade, from PortWatch)
        vix_1d          — VIX 1-day point change

    Channel-to-signal mapping (per conflict's transmission weights):
        oil_gas / chokepoint  → brent, wti, tanker_disruption
        energy_infra          → natgas
        agriculture           → wheat
        equity_sector         → vix
    """
    if not market_data:
        return 1.0

    tx = conflict.get("transmission", {})

    # Collect signal magnitudes, weighted by this conflict's transmission strength
    signals: list[float] = []

    # Oil / chokepoint — dominant channel for Hormuz/Iran
    oil_weight = max(float(tx.get("oil_gas", 0)), float(tx.get("chokepoint", 0)))
    if oil_weight > 0.3:
        brent_abs = abs(float(market_data.get("brent_pct_1d", 0.0)))
        wti_abs   = abs(float(market_data.get("wti_pct_1d",   0.0)))
        tanker    = float(market_data.get("tanker_disruption", 0.0))
        oil_sig   = max(brent_abs / 3.0, wti_abs / 3.0, tanker) * oil_weight
        signals.append(min(oil_sig, 1.0))

    # Natural gas / energy infra — dominant for Russia-Ukraine
    eg_weight = max(float(tx.get("energy_infra", 0)), float(tx.get("oil_gas", 0)) * 0.5)
    if eg_weight > 0.3:
        ng_abs  = abs(float(market_data.get("natgas_pct_1d", 0.0)))
        ng_sig  = (ng_abs / 5.0) * eg_weight
        signals.append(min(ng_sig, 1.0))

    # Agriculture — Ukraine/Russia primary
    ag_weight = float(tx.get("agriculture", 0))
    if ag_weight > 0.3:
        wheat_abs = abs(float(market_data.get("wheat_pct_1d", 0.0)))
        ag_sig    = (wheat_abs / 4.0) * ag_weight
        signals.append(min(ag_sig, 1.0))

    # Equity volatility — broad signal
    eq_weight = float(tx.get("equity_sector", 0))
    if eq_weight > 0.3:
        vix_abs = abs(float(market_data.get("vix_1d", 0.0)))
        eq_sig  = (vix_abs / 5.0) * eq_weight
        signals.append(min(eq_sig, 1.0))

    if not signals:
        return 1.0

    avg_signal = float(np.mean(signals))
    # Map [0, 1] signal → [0.7, 1.5] multiplier
    return float(np.clip(0.7 + avg_signal * 0.8, 0.7, 1.5))


# ── Portfolio aggregation ─────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def score_all_conflicts() -> dict[str, dict]:
    """
    Compute CIS, TPS, and confidence for every conflict in CONFLICTS.
    Returns dict keyed by conflict["id"].

    Cached for 30 minutes — structural data changes slowly; news GPR
    layer is separate and not cached here.

    market_freshness is initialised to 1.0 here; callers with live market
    data should call apply_market_freshness() to update it before ranking.
    """
    results: dict[str, dict] = {}
    for c in CONFLICTS:
        cid = c["id"]
        cis  = compute_cis(c)
        tps  = compute_tps(c)
        conf = compute_confidence(c)
        results[cid] = {
            "id":           cid,
            "name":         c["name"],
            "label":        c["label"],
            "region":       c["region"],
            "color":        c.get("color", "#CFB991"),
            "state":        c.get("state", "active"),
            "cis":          round(cis,  1),
            "tps":          round(tps,  1),
            "confidence":   round(conf, 3),
            "trend":        compute_trend(c),
            "freshness":    _freshness_label(c),
            "escalation":   c.get("escalation_trend", "stable"),
            "market_freshness": 1.0,   # updated by apply_market_freshness()
            # Pass through for display
            "transmission": c.get("transmission", {}),
            "affected_commodities": c.get("affected_commodities", []),
            "affected_equities":    c.get("affected_equities",    []),
            "hedge_assets":         c.get("hedge_assets",         []),
            "last_updated":         str(c.get("last_updated", "")),
        }
    # GAP 17 fix: freshness of conflict_manual should reflect when the CONFLICTS dict
    # was last edited, not when this function ran. We check the most recent
    # last_updated field across all conflicts rather than calling record_fetch()
    # unconditionally (which would always show "Live" even on months-old data).
    try:
        from src.analysis.freshness import record_fetch, record_failure
        today = datetime.date.today()
        most_recent = None
        for c in CONFLICTS:
            lu = c.get("last_updated")
            if isinstance(lu, datetime.date):
                d = lu
            elif isinstance(lu, str):
                try:
                    d = datetime.date.fromisoformat(lu)
                except ValueError:
                    continue
            else:
                continue
            if most_recent is None or d > most_recent:
                most_recent = d

        if most_recent is not None:
            # Record freshness as of the conflict's own last_updated date
            record_fetch(
                "conflict_manual",
                ts=datetime.datetime.combine(most_recent, datetime.time(0, 0)),
            )
            days_since = (today - most_recent).days
            if days_since > 90:
                record_failure(
                    "conflict_manual",
                    f"Conflict data last updated {days_since}d ago ({most_recent}) — manual refresh needed",
                )
        else:
            record_failure("conflict_manual", "No last_updated dates found in CONFLICTS registry")
    except Exception:
        pass
    return results


def apply_market_freshness(
    conflict_results: dict[str, dict],
    market_data: dict,
) -> dict[str, dict]:
    """
    Enrich per-conflict results with live market_freshness multipliers.

    Call this after score_all_conflicts() when live price/vol data is available.
    Updates the "market_freshness" field in-place on a shallow copy.

    market_data keys (all float, all optional):
        brent_pct_1d, wti_pct_1d, natgas_pct_1d, wheat_pct_1d,
        tanker_disruption (0–1), vix_1d

    Returns the enriched dict (same reference as input for convenience).
    """
    conflict_by_id = {c["id"]: c for c in CONFLICTS}
    for cid, result in conflict_results.items():
        conf = conflict_by_id.get(cid)
        if conf is None:
            continue
        mf = compute_market_freshness(conf, market_data)
        result["market_freshness"] = round(mf, 3)
    return conflict_results


def build_market_signals(cmd_r: "pd.DataFrame") -> dict:
    """
    Extract live market-freshness signals from a commodity log-returns DataFrame.

    Returns a dict compatible with compute_market_freshness() and
    apply_market_freshness(). Safe to call with None or empty DataFrame —
    returns {} which leaves all market_freshness values at 1.0.

    Signal keys (all floats, approximate 1-day % move):
        brent_pct_1d, wti_pct_1d, natgas_pct_1d, wheat_pct_1d

    Also reads tanker_disruption (0–1) from session_state["_hormuz_disruption"]
    if available (set by strait_watch.py from live PortWatch data).
    """
    signals: dict = {}
    try:
        if cmd_r is None or cmd_r.empty:
            return signals
        last = cmd_r.iloc[-1]
        _col_map = {
            "Brent Crude":   "brent_pct_1d",
            "WTI Crude Oil": "wti_pct_1d",
            "Natural Gas":   "natgas_pct_1d",
            "Wheat":         "wheat_pct_1d",
        }
        for col, key in _col_map.items():
            if col in last.index and not np.isnan(float(last[col])):
                # log return × 100 ≈ % change (exact for small moves)
                signals[key] = float(last[col]) * 100.0
    except Exception:
        pass

    # Hormuz tanker disruption — set by strait_watch.py from live PortWatch data
    # tanker_disruption = 1 − (live_ships / baseline_ships), clamped [0, 1]
    try:
        disruption = st.session_state.get("_hormuz_disruption")
        if disruption is not None:
            signals["tanker_disruption"] = float(np.clip(disruption, 0.0, 1.0))
    except Exception:
        pass

    return signals


def _freshness_label(conflict: dict) -> str:
    score = _freshness_score(conflict)
    if score >= 0.80:
        return "live"
    elif score >= 0.50:
        return "recent"
    elif score >= 0.30:
        return "aging"
    else:
        return "stale"


def aggregate_portfolio_scores(
    conflict_results: Optional[dict] = None,
) -> dict:
    """
    Aggregate per-conflict CIS and TPS into portfolio-level scores.

    CIS_portfolio  = intensity-weighted mean of per-conflict CIS
    TPS_portfolio  = intensity-weighted mean of per-conflict TPS
    Confidence     = mean confidence across active conflicts

    Returns:
        {cis, tps, confidence, conflict_weights, top_conflict}
    """
    if conflict_results is None:
        conflict_results = score_all_conflicts()

    if not conflict_results:
        return {"cis": 50.0, "tps": 50.0, "confidence": 0.50,
                "conflict_weights": {}, "top_conflict": None}

    # Apply live market-freshness multipliers from session_state.
    # Signals are injected by compute_risk_score() (risk_score.py) and by
    # home.py, both of which have access to live commodity returns.
    # Falls back silently — all market_freshness values remain 1.0 (no ranking change).
    try:
        mf_signals = st.session_state.get("_mf_signals", {})
        if mf_signals:
            apply_market_freshness(conflict_results, mf_signals)
    except Exception:
        pass

    # Weight by CIS (higher-intensity conflicts get more weight in TPS aggregate)
    cis_vals = {cid: r["cis"] for cid, r in conflict_results.items()}
    total_cis = sum(cis_vals.values()) + 1e-9

    weights = {cid: v / total_cis for cid, v in cis_vals.items()}

    portfolio_cis = float(np.clip(
        sum(cis_vals[cid] * weights[cid] * len(conflict_results)
            for cid in conflict_results),
        0, 100
    ))

    portfolio_tps = float(np.clip(
        sum(conflict_results[cid]["tps"] * weights[cid] * len(conflict_results)
            for cid in conflict_results),
        0, 100
    ))

    avg_conf = float(np.mean([r["confidence"] for r in conflict_results.values()]))

    # Rank by CIS × market_freshness so conflicts actively moving markets
    # surface above equally intense but priced-in conflicts.
    # market_freshness defaults to 1.0 (no change to ranking) unless
    # apply_market_freshness() was called with live price data.
    top = max(
        conflict_results.items(),
        key=lambda x: x[1]["cis"] * x[1].get("market_freshness", 1.0),
    )

    return {
        "cis":              round(portfolio_cis, 1),
        "tps":              round(portfolio_tps, 1),
        "confidence":       round(avg_conf, 3),
        "conflict_weights": {cid: round(w, 3) for cid, w in weights.items()},
        "top_conflict":     top[1]["name"] if top else None,
        "conflict_detail":  conflict_results,
    }


# ── Conflict-to-commodity relevance matrix ────────────────────────────────────

def conflict_commodity_matrix() -> dict[str, dict[str, float]]:
    """
    Returns {conflict_id: {commodity: relevance}} matrix.
    Uses the transmission channels as a proxy for commodity relevance.
    """
    channel_to_commodity = {
        "oil_gas":     ["WTI Crude Oil", "Brent Crude", "Natural Gas",
                        "Gasoline (RBOB)", "Heating Oil"],
        "metals":      ["Nickel", "Aluminum", "Copper", "Platinum", "Silver"],
        "agriculture": ["Wheat", "Corn", "Soybeans", "Sugar #11", "Coffee"],
        "shipping":    ["WTI Crude Oil", "Brent Crude", "Wheat", "Corn"],
        "energy_infra":["Natural Gas", "WTI Crude Oil", "Brent Crude"],
    }

    matrix: dict[str, dict[str, float]] = {}
    for c in CONFLICTS:
        cid = c["id"]
        tx  = c.get("transmission", {})
        commodity_scores: dict[str, float] = {}
        for channel, commodities in channel_to_commodity.items():
            ch_score = float(tx.get(channel, 0.0))
            for commodity in commodities:
                commodity_scores[commodity] = max(
                    commodity_scores.get(commodity, 0.0), ch_score
                )
        matrix[cid] = commodity_scores
    return matrix


# ── Top exposed assets per conflict ──────────────────────────────────────────

def top_affected_assets(conflict_id: str, n: int = 5) -> list[dict]:
    """
    Return the top-n most-affected assets for a given conflict,
    using SECURITY_EXPOSURE structural values.
    """
    from src.data.config import SECURITY_EXPOSURE
    ranked = []
    for asset, data in SECURITY_EXPOSURE.items():
        score = data.get("structural", {}).get(conflict_id, 0.0)
        if score > 0:
            ranked.append({"asset": asset, "exposure": score})
    ranked.sort(key=lambda x: x["exposure"], reverse=True)
    return ranked[:n]


# ── Transmission lag signal ───────────────────────────────────────────────────

def transmission_lag_signal(
    cmd_r: "pd.DataFrame",
    equity_r: "pd.DataFrame",
    lookback: int = 5,
    z_threshold: float = 1.5,
) -> dict:
    """
    Detect commodity→equity transmission lag: when commodity stress spiked
    N days ago but equities have not yet fully responded.

    Geopolitical shocks transmit to equities with a 1-5 day lag (sanctions
    announcements, supply route confirmations, refinery pass-through).
    This function surfaces those windows so analysts can anticipate delayed
    equity moves.

    Parameters
    ----------
    cmd_r       : daily commodity log-return DataFrame
    equity_r    : daily equity log-return DataFrame
    lookback    : max lag to check (default 5 trading days)
    z_threshold : minimum commodity z-score to flag as a stress event (1.5σ)

    Returns
    -------
    dict with keys:
        active          : bool — True if a pending transmission is detected
        peak_day_ago    : int  — days since peak commodity stress (1=yesterday)
        commodity_z     : float — commodity stress z-score at the peak day
        equity_z        : float — equity z-score on the SAME peak day
        equity_lag_z    : float — equity z-score TODAY (lagged response expected)
        lag_signal      : str  — "Pending" / "In progress" / "Absorbed" / "No stress"
        dominant_cmd    : str  — commodity with strongest stress at peak day
        detail          : str  — human-readable summary for display
    """
    _empty = {
        "active": False,
        "peak_day_ago": 0,
        "commodity_z": 0.0,
        "equity_z": 0.0,
        "equity_lag_z": 0.0,
        "lag_signal": "No stress",
        "dominant_cmd": "",
        "detail": "Insufficient data",
    }

    try:
        if cmd_r is None or cmd_r.empty or equity_r is None or equity_r.empty:
            return _empty
        if len(cmd_r) < 30 or len(equity_r) < 30:
            return _empty

        # Align on common index, last 60 days for z-score context
        common_idx = cmd_r.index.intersection(equity_r.index)
        if len(common_idx) < 30:
            return _empty

        cmd_aligned = cmd_r.loc[common_idx].tail(60)
        eq_aligned  = equity_r.loc[common_idx].tail(60)

        # Commodity stress composite: mean of |daily returns| × 100 across all cols
        cmd_composite = cmd_aligned.mean(axis=1) * 100.0  # daily avg return %
        eq_composite  = eq_aligned.mean(axis=1)  * 100.0

        # Rolling z-score using first 30 observations as baseline
        baseline_cmd = cmd_composite.iloc[:30]
        baseline_eq  = eq_composite.iloc[:30]
        mu_cmd, sd_cmd = float(baseline_cmd.mean()), float(baseline_cmd.std())
        mu_eq,  sd_eq  = float(baseline_eq.mean()),  float(baseline_eq.std())

        if sd_cmd < 1e-8 or sd_eq < 1e-8:
            return _empty

        cmd_z = (cmd_composite - mu_cmd) / sd_cmd
        eq_z  = (eq_composite  - mu_eq)  / sd_eq

        # Look at the last `lookback` days for a commodity stress spike
        recent_cmd_z = cmd_z.iloc[-(lookback + 1):-1]  # exclude today
        today_eq_z   = float(eq_z.iloc[-1])

        if len(recent_cmd_z) == 0:
            return _empty

        # Find peak commodity stress day in the lookback window
        peak_idx  = int(recent_cmd_z.abs().argmax())
        peak_z    = float(recent_cmd_z.iloc[peak_idx])
        peak_date = recent_cmd_z.index[peak_idx]
        days_ago  = len(recent_cmd_z) - peak_idx  # 1 = yesterday

        if abs(peak_z) < z_threshold:
            return {**_empty, "lag_signal": "No stress",
                    "detail": f"No significant commodity stress in past {lookback} days (peak z={peak_z:.2f})"}

        # Equity response on the same peak day
        eq_z_at_peak = float(eq_z.loc[peak_date]) if peak_date in eq_z.index else 0.0

        # Determine dominant commodity at peak day
        cmd_daily = cmd_aligned.loc[peak_date] if peak_date in cmd_aligned.index else pd.Series()
        dominant_cmd = str(cmd_daily.abs().idxmax()) if not cmd_daily.empty else "—"

        # Classify lag signal
        same_sign  = (peak_z * today_eq_z) > 0
        equity_absorbed = abs(today_eq_z) >= abs(eq_z_at_peak) * 0.7

        if days_ago == 1 and not same_sign:
            lag_signal = "Pending"
            detail = (f"Commodity stress spiked yesterday (z={peak_z:+.2f}, {dominant_cmd}). "
                      f"Equity has NOT yet responded (eq z={today_eq_z:+.2f}). "
                      f"Watch for delayed equity transmission today.")
        elif days_ago <= 3 and not equity_absorbed:
            lag_signal = "In progress"
            detail = (f"{dominant_cmd} stress peaked {days_ago}d ago (z={peak_z:+.2f}). "
                      f"Equity absorption partial (eq today z={today_eq_z:+.2f}). "
                      f"Transmission still propagating — expect continued equity move.")
        elif days_ago <= lookback and equity_absorbed:
            lag_signal = "Absorbed"
            detail = (f"{dominant_cmd} stress peaked {days_ago}d ago (z={peak_z:+.2f}). "
                      f"Equity has absorbed the shock (eq today z={today_eq_z:+.2f}). "
                      f"Transmission complete.")
        else:
            lag_signal = "No stress"
            detail = f"No significant commodity→equity lag detected in past {lookback} days."

        return {
            "active":        lag_signal in ("Pending", "In progress"),
            "peak_day_ago":  days_ago,
            "commodity_z":   round(peak_z, 2),
            "equity_z":      round(eq_z_at_peak, 2),
            "equity_lag_z":  round(today_eq_z, 2),
            "lag_signal":    lag_signal,
            "dominant_cmd":  dominant_cmd,
            "detail":        detail,
        }

    except Exception:
        return _empty

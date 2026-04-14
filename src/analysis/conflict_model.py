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


# ── Per-conflict scorers ──────────────────────────────────────────────────────

def compute_cis(conflict: dict) -> float:
    """
    Conflict Intensity Score for a single conflict.
    Returns value in [0, 100].
    """
    state_mult = _STATE_MULT.get(conflict.get("state", "active"), 1.0)

    # Dimension values
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

    raw = sum(_CIS_WEIGHTS[k] * dims[k] for k in _CIS_WEIGHTS)
    return float(np.clip(raw * state_mult * 100, 0, 100))


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
    """
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
        0.30 * source_cov
        + 0.25 * data_conf
        + 0.25 * freshness
        + 0.20 * completeness
    )
    return float(np.clip(confidence, 0.0, 1.0))


def compute_trend(conflict: dict) -> str:
    """Human-readable trend label from escalation_trend field."""
    t = conflict.get("escalation_trend", "stable")
    return {"escalating": "rising", "stable": "stable",
            "de-escalating": "falling"}.get(t, "stable")


# ── Portfolio aggregation ─────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def score_all_conflicts() -> dict[str, dict]:
    """
    Compute CIS, TPS, and confidence for every conflict in CONFLICTS.
    Returns dict keyed by conflict["id"].

    Cached for 30 minutes — structural data changes slowly; news GPR
    layer is separate and not cached here.
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
            # Pass through for display
            "transmission": c.get("transmission", {}),
            "affected_commodities": c.get("affected_commodities", []),
            "affected_equities":    c.get("affected_equities",    []),
            "hedge_assets":         c.get("hedge_assets",         []),
            "last_updated":         str(c.get("last_updated", "")),
        }
    try:
        from src.analysis.freshness import record_fetch
        record_fetch("conflict_manual")
    except Exception:
        pass
    return results


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

    top = max(conflict_results.items(), key=lambda x: x[1]["cis"])

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

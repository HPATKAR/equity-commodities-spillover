"""
Conflict Exposure Framework.

Per-asset geopolitical exposure scoring for every security in SECURITY_EXPOSURE.

Methodology
-----------
Three complementary scores per asset:

  Structural Exposure Score (SES)  0–1
    CIS-weighted average of the asset's structural exposure values across all
    active conflicts. Answers: "which assets are most structurally tied to the
    current conflict portfolio?"
      SES(a) = Σ_c [ (CIS_c / Σ CIS) × structural[a][c] ]

  Transmission-Adjusted Exposure (TAE)  0–1
    Structural exposure further scaled by each conflict's TPS — reflects how
    much of the structural link is currently being transmitted through markets.
      TAE(a) = Σ_c [ (CIS_c / Σ CIS) × structural[a][c] × (TPS_c / 100) ]

  Scenario-Adjusted Score (SAS)  0–100
    TAE scaled by the current scenario's geo_mult, then expressed as 0–100.
      SAS(a) = clip(TAE(a) × geo_mult × 100, 0, 100)

Per-conflict beta:
    beta(a, c) = structural[a][c] × (TPS_c / 100)
    Sensitivity of asset `a` specifically to conflict `c`'s transmission.

Hedge score:
    For assets listed as hedge_assets in high-CIS conflicts.
    hedge_score(a) = Σ_c [ w_c × hedge_presence(a, c) ] × 100
    where hedge_presence = 1 if asset in conflict.hedge_assets, else 0.

All results from score_all_assets() are usable downstream by:
  - exposure_scoring.py (display)
  - trade_generator.py (trade idea generation)
  - trade_filter.py (conflict-beta filter)
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from typing import Optional

from src.data.config import CONFLICTS, SECURITY_EXPOSURE


# ── Sector / tag metadata ─────────────────────────────────────────────────────

_SECTOR_LABELS: dict[str, str] = {
    "energy":         "Energy",
    "safe_haven":     "Safe Haven",
    "precious_metals":"Precious Metals",
    "industrial":     "Industrial",
    "agriculture":    "Agriculture",
    "broad_equity":   "Broad Equity",
    "tech":           "Technology",
    "semiconductors": "Semiconductors",
    "defense":        "Defense",
    "ev_supply":      "EV Supply",
    "textiles":       "Textiles",
    "fx":             "FX",
    "rates":          "Fixed Income",
}

# Assets that behave as safe havens (inverse exposure in stress scenarios)
_SAFE_HAVEN_ASSETS = {
    "Gold", "Silver", "US 20Y+ Treasury (TLT)", "JPY/USD", "CHF/USD",
    "US Dollar Index (DXY)",
}

# Assets that typically benefit from geopolitical stress (long geo-risk plays)
_GEO_RISK_BENEFICIARIES = {
    "Gold", "Silver", "WTI Crude Oil", "Brent Crude",
    "Natural Gas", "Heating Oil", "Gasoline (RBOB)",
    "Wheat", "Corn",
}


# ── Core scorers ──────────────────────────────────────────────────────────────

def _structural_exposure_score(
    asset: str,
    conflict_weights: dict[str, float],  # {conflict_id: weight (sum=1)}
) -> float:
    """CIS-weighted average structural exposure. Returns 0–1."""
    s_data = SECURITY_EXPOSURE.get(asset, {})
    structural = s_data.get("structural", {})
    if not structural or not conflict_weights:
        return 0.0
    return float(np.clip(
        sum(conflict_weights.get(cid, 0.0) * float(v)
            for cid, v in structural.items()),
        0.0, 1.0
    ))


def _transmission_adjusted_exposure(
    asset: str,
    conflict_weights: dict[str, float],
    conflict_tps: dict[str, float],        # {conflict_id: TPS 0–100}
) -> float:
    """Structural exposure × TPS fraction. Returns 0–1."""
    s_data = SECURITY_EXPOSURE.get(asset, {})
    structural = s_data.get("structural", {})
    if not structural:
        return 0.0
    total = 0.0
    for cid, sv in structural.items():
        w   = conflict_weights.get(cid, 0.0)
        tps = conflict_tps.get(cid, 0.0) / 100.0
        total += w * float(sv) * tps
    return float(np.clip(total, 0.0, 1.0))


def _conflict_beta(
    asset: str,
    conflict_tps: dict[str, float],
) -> dict[str, float]:
    """
    Per-conflict beta: structural[a][c] × TPS_c / 100.
    Returns {conflict_id: beta_value 0–1}.
    """
    s_data   = SECURITY_EXPOSURE.get(asset, {})
    structural = s_data.get("structural", {})
    return {
        cid: round(float(sv) * conflict_tps.get(cid, 0.0) / 100.0, 3)
        for cid, sv in structural.items()
    }


def _hedge_score(
    asset: str,
    conflict_weights: dict[str, float],
) -> float:
    """
    CIS-weighted presence in conflict.hedge_assets lists.
    Returns 0–100.
    """
    total = 0.0
    for c in CONFLICTS:
        cid = c["id"]
        w = conflict_weights.get(cid, 0.0)
        if asset in c.get("hedge_assets", []):
            total += w
    return float(np.clip(total * 100, 0.0, 100.0))


def _top_conflict(
    asset: str,
    conflict_tps: dict[str, float],
) -> Optional[str]:
    """Return the conflict_id with highest beta for this asset."""
    betas = _conflict_beta(asset, conflict_tps)
    if not betas:
        return None
    return max(betas, key=lambda k: betas[k])


# ── Main scorer ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def score_all_assets(
    conflict_results: Optional[dict] = None,
    scenario_id: Optional[str] = None,
) -> dict[str, dict]:
    """
    Compute structural exposure, TAE, conflict beta, hedge score, and
    scenario-adjusted score for every asset in SECURITY_EXPOSURE.

    Returns {asset_name: {ses, tae, sas, beta, hedge_score, top_conflict,
                          sector_tags, route_tags, direction}}

    direction: "long_geo_risk" | "safe_haven" | "neutral"
    """
    from src.analysis.conflict_model import (
        score_all_conflicts, aggregate_portfolio_scores,
    )
    from src.analysis.scenario_state import get_scenario

    if conflict_results is None:
        conflict_results = score_all_conflicts()

    agg = aggregate_portfolio_scores(conflict_results)

    # Build conflict-level inputs
    conflict_weights: dict[str, float] = agg.get("conflict_weights", {})
    conflict_tps: dict[str, float]     = {
        cid: r["tps"] for cid, r in conflict_results.items()
    }

    # Scenario multiplier
    scenario   = get_scenario()
    geo_mult   = scenario.get("geo_mult", 1.0)

    results: dict[str, dict] = {}
    for asset in SECURITY_EXPOSURE:
        ses   = _structural_exposure_score(asset, conflict_weights)
        tae   = _transmission_adjusted_exposure(asset, conflict_weights, conflict_tps)
        sas   = float(np.clip(tae * geo_mult * 100, 0.0, 100.0))
        beta  = _conflict_beta(asset, conflict_tps)
        hs    = _hedge_score(asset, conflict_weights)
        top_c = _top_conflict(asset, conflict_tps)

        s_data = SECURITY_EXPOSURE.get(asset, {})
        sectors = s_data.get("sector_tags", [])
        routes  = s_data.get("route_tags",  [])
        sanctions = s_data.get("sanction_tags", [])

        if asset in _SAFE_HAVEN_ASSETS:
            direction = "safe_haven"
        elif asset in _GEO_RISK_BENEFICIARIES:
            direction = "long_geo_risk"
        else:
            direction = "neutral"

        results[asset] = {
            "asset":       asset,
            "ses":         round(ses,  3),    # 0–1 structural
            "tae":         round(tae,  3),    # 0–1 transmission-adjusted
            "sas":         round(sas,  1),    # 0–100 scenario-adjusted
            "beta":        beta,              # {conflict_id: 0–1}
            "hedge_score": round(hs,   1),    # 0–100
            "top_conflict": top_c,
            "sector_tags": sectors,
            "route_tags":  routes,
            "sanction_tags": sanctions,
            "direction":   direction,
            "scenario_mult": round(geo_mult, 2),
        }

    return results


# ── Filtered views (used by trade_generator and exposure page) ────────────────

def ranked_by_exposure(
    n: int = 20,
    direction: Optional[str] = None,  # "long_geo_risk" | "safe_haven" | "neutral" | None
    sector: Optional[str] = None,
    conflict_id: Optional[str] = None,
) -> list[dict]:
    """
    Return top-n assets by SAS (scenario-adjusted score), optionally filtered.
    conflict_id filter: only assets with beta[conflict_id] > 0.
    """
    all_assets = score_all_assets()
    items = list(all_assets.values())

    if direction:
        items = [a for a in items if a["direction"] == direction]
    if sector:
        items = [a for a in items if sector in a["sector_tags"]]
    if conflict_id:
        items = [a for a in items if a["beta"].get(conflict_id, 0) > 0]

    items.sort(key=lambda a: a["sas"], reverse=True)
    return items[:n]


def ranked_hedges(n: int = 10) -> list[dict]:
    """Return top-n hedge assets by hedge_score."""
    all_assets = score_all_assets()
    items = [a for a in all_assets.values() if a["direction"] == "safe_haven"
             or a["hedge_score"] > 20]
    items.sort(key=lambda a: a["hedge_score"], reverse=True)
    return items[:n]


def conflict_affected_universe(conflict_id: str, min_beta: float = 0.1) -> list[dict]:
    """
    Return all assets with meaningful beta to a specific conflict.
    Sorted by beta[conflict_id] descending.
    """
    all_assets = score_all_assets()
    items = []
    for a in all_assets.values():
        b = a["beta"].get(conflict_id, 0.0)
        if b >= min_beta:
            items.append({**a, "_sort_beta": b})
    items.sort(key=lambda a: a["_sort_beta"], reverse=True)
    return items


def exposure_summary_stats(all_assets: Optional[dict] = None) -> dict:
    """
    Portfolio-level summary statistics for the exposure panel header.
    """
    if all_assets is None:
        all_assets = score_all_assets()

    sas_vals = [a["sas"] for a in all_assets.values()]
    if not sas_vals:
        return {}

    high_exp  = sum(1 for v in sas_vals if v >= 60)
    med_exp   = sum(1 for v in sas_vals if 30 <= v < 60)
    low_exp   = sum(1 for v in sas_vals if v < 30)
    top_asset = max(all_assets.items(), key=lambda x: x[1]["sas"])
    top_hedge = max(
        ((k, v) for k, v in all_assets.items() if v["hedge_score"] > 0),
        key=lambda x: x[1]["hedge_score"],
        default=(None, {"hedge_score": 0}),
    )

    return {
        "n_assets":    len(sas_vals),
        "mean_sas":    round(float(np.mean(sas_vals)), 1),
        "max_sas":     round(float(np.max(sas_vals)),  1),
        "high_exp":    high_exp,
        "med_exp":     med_exp,
        "low_exp":     low_exp,
        "top_asset":   top_asset[0],
        "top_asset_sas": top_asset[1]["sas"],
        "top_hedge":   top_hedge[0],
        "top_hedge_score": top_hedge[1]["hedge_score"],
    }

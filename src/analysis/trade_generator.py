"""
Conflict-Driven Trade Generator.

Generates trade candidates programmatically from:
  - Current exposure scores (exposure.py score_all_assets)
  - Active conflict scores (conflict_model.py score_all_conflicts)
  - Current scenario (scenario_state.py get_scenario)
  - Current correlation regime (passed in as regime int 0–3)

Generated candidates follow the same trade dict schema as the static
_TRADE_LIBRARY in trade_ideas.py, extended with:
  - confidence: float 0–1  (computed from CIS, TPS, SAS, scenario alignment)
  - conflict_id: str        (primary conflict driving the idea)
  - scenarios: list[str]   (scenario IDs where idea is valid)
  - qc_flags: list[str]    (auto-populated pre-generation flags)

Usage:
    from src.analysis.trade_generator import generate_conflict_trades
    candidates = generate_conflict_trades(regime=2)
    # pass to apply_filters() from trade_filter.py
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np

from src.data.config import (
    COMMODITY_TICKERS as _COMMODITY_TICKERS,
    EQUITY_TICKERS as _EQUITY_TICKERS,
    FIXED_INCOME_TICKERS as _FI_TICKERS,
)

# ── Ticker resolution ─────────────────────────────────────────────────────────

# Preferred liquid ETF proxy per asset (overrides raw futures/index ticker for trade ideas)
_ETF_PROXY: dict[str, str] = {
    "WTI Crude Oil":    "USO",
    "Brent Crude":      "BNO",
    "Natural Gas":      "UNG",
    "Gold":             "GLD",
    "Silver":           "SLV",
    "Copper":           "CPER",
    "Wheat":            "WEAT",
    "Corn":             "CORN",
    "Soybeans":         "SOYB",
    "S&P 500":          "SPY",
    "Nasdaq 100":       "QQQ",
    "Russell 2000":     "IWM",
    "Eurostoxx 50":     "FEZ",
    "DAX":              "EWG",
    "FTSE 100":         "EWU",
    "Nikkei 225":       "EWJ",
    "Hang Seng":        "EWH",
    "Shanghai Comp":    "MCHI",
    "CSI 300":          "MCHI",
}

# Combined lookup: ETF proxy first, then config tickers as fallback
_ALL_TICKERS: dict[str, str] = {
    **_COMMODITY_TICKERS,
    **_EQUITY_TICKERS,
    **_FI_TICKERS,
    **_ETF_PROXY,  # ETF proxy overrides raw futures for cleaner display
}


def _resolve_tickers(assets: list[str]) -> dict[str, str]:
    """Map asset display names to their preferred tradeable ticker."""
    return {a: _ALL_TICKERS[a] for a in assets if a in _ALL_TICKERS}


# ── Scenario-to-category affinity ────────────────────────────────────────────

_SCENARIO_TRADE_TYPES: dict[str, list[str]] = {
    "base":            ["Macro", "Growth"],
    "escalation":      ["Geopolitical", "Crisis Hedge"],
    "de_escalation":   ["Macro", "FX"],
    "supply_shock":    ["Geopolitical", "Commodity", "Macro"],
    "sanctions_shock": ["Geopolitical", "FX", "Crisis Hedge"],
    "shipping_shock":  ["Geopolitical", "Commodity"],
    "risk_off":        ["Crisis Hedge", "Fixed Income"],
    "recovery":        ["Macro", "Growth"],
}

# Regime labels for confidence calculation
_REGIME_NAMES = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}

# Minimum SAS threshold to qualify an asset for trade inclusion
_MIN_SAS_LONG = 30.0     # long geo-risk ideas: lead asset SAS >= this
_MIN_SAS_HEDGE = 20.0    # hedge ideas: hedge_score >= this
_MIN_CIS = 45.0          # conflict must have CIS >= this to generate ideas
_MIN_CONFIDENCE = 0.55   # suppress generated trades below this floor


# ── Template generators ───────────────────────────────────────────────────────

def _safe_haven_pair(
    lead_asset: str,
    lead_data: dict,
    hedge_asset: str,
    hedge_data: dict,
    conflict: dict,
    scenario: dict,
    scenario_id: str,
    cis: float,
    tps: float,
    regime: int,
) -> dict:
    """Long safe-haven / Short geo-risk-exposed asset."""
    lead_beta  = lead_data["beta"].get(conflict["id"], 0.0)
    confidence = float(np.clip(
        0.35
        + 0.20 * (cis / 100)
        + 0.20 * (tps / 100)
        + 0.10 * (lead_data["sas"] / 100)
        + 0.10 * (regime / 3)
        + 0.05 * scenario.get("safe_haven", False),
        0.35, 0.92,
    ))
    qc_flags: list[str] = []
    if confidence < 0.55:
        qc_flags.append("Low confidence (<55%)")
    if tps < 40:
        qc_flags.append("Low transmission pressure — signal may be premature")

    return {
        "name":      f"Long {hedge_asset} / Short {lead_asset}",
        "trigger":   f"{conflict['name']} — safe-haven rotation (SAS={lead_data['sas']:.0f})",
        "rationale": (
            f"{lead_asset} carries structural geo exposure of {lead_data['ses']:.2f} "
            f"to {conflict['name']} (beta={lead_beta:.2f}). "
            f"Under the current {scenario['label']} scenario, transmission pressure "
            f"({tps:.0f}/100) is elevating downside risk. "
            f"{hedge_asset} provides inverse flight-to-safety exposure with "
            f"a hedge score of {hedge_data['hedge_score']:.0f}/100."
        ),
        "entry":     (
            f"CIS ≥ {cis:.0f} AND TPS ≥ {tps:.0f}; "
            f"{lead_asset} SAS above {_MIN_SAS_LONG:.0f}; "
            f"regime = Elevated or Crisis"
        ),
        "exit":      (
            f"Conflict de-escalation signal; {lead_asset} SAS drops below 20; "
            f"TPS falls >15 points in 7 days"
        ),
        "risk":      (
            f"Rapid ceasefire or diplomatic breakthrough compresses geo premium; "
            f"central bank intervention can override safe-haven bid"
        ),
        "tickers":   _resolve_tickers([hedge_asset, lead_asset]),
        "assets":    [hedge_asset, lead_asset],
        "direction": ["Long", "Short"],
        "regime":    [2, 3] if regime >= 2 else [1, 2],
        "category":  "Crisis Hedge",
        "conflict_id": conflict["id"],
        "scenarios": ["escalation", "risk_off", "supply_shock", "sanctions_shock"],
        "confidence": round(confidence, 3),
        "upside_pct": round(confidence * 14, 1),
        "qc_flags":  qc_flags,
        "generated": True,
        "generated_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "stop":           f"{lead_asset} SAS drops below 15 or CIS falls below 30",
        "target":         f"{hedge_asset} outperforms by 8–12% OR CIS peaks and TPS rolls over",
        "invalidation":   f"Ceasefire / peace agreement; {lead_asset} SAS < 15 for 3+ days",
        "holding_period": "2–6 weeks depending on conflict trajectory",
        "reward_risk":    f"{round(confidence * 2.5, 1)}:1 estimated",
    }


def _supply_shock_long(
    asset: str,
    asset_data: dict,
    conflict: dict,
    scenario: dict,
    scenario_id: str,
    cis: float,
    tps: float,
    regime: int,
) -> Optional[dict]:
    """Long a commodity disrupted by supply shock from this conflict."""
    s_data = asset_data
    if s_data["sas"] < _MIN_SAS_LONG:
        return None
    beta = s_data["beta"].get(conflict["id"], 0.0)

    # Only generate if the conflict has meaningful oil_gas or agriculture transmission
    tx = conflict.get("transmission", {})
    channel_score = max(
        tx.get("oil_gas", 0),
        tx.get("agriculture", 0),
        tx.get("metals", 0),
    )
    if channel_score < 0.5:
        return None

    confidence = float(np.clip(
        0.30
        + 0.25 * (tps / 100)
        + 0.20 * (cis / 100)
        + 0.15 * beta
        + 0.10 * (regime / 3),
        0.30, 0.90,
    ))

    qc_flags: list[str] = []
    if confidence < 0.55:
        qc_flags.append("Low confidence (<55%)")
    if beta < 0.10:
        qc_flags.append("Low conflict beta — check structural exposure")

    return {
        "name":      f"Long {asset} — Supply Shock ({conflict['label']})",
        "trigger":   f"{conflict['name']} disrupting {asset} supply chains",
        "rationale": (
            f"{asset} has structural exposure {s_data['ses']:.2f} to {conflict['name']}, "
            f"with conflict beta {beta:.2f}. "
            f"Current TPS of {tps:.0f}/100 indicates active transmission through "
            f"supply routes. Under {scenario['label']} scenario (geo_mult={scenario['geo_mult']:.2f}x), "
            f"scenario-adjusted exposure SAS={s_data['sas']:.0f}/100. "
            f"Physical supply disruption supports commodity outperformance."
        ),
        "entry":     (
            f"TPS ≥ {max(tps - 5, 40):.0f}; {asset} 20d momentum positive; "
            f"regime Elevated or Crisis; CIS ≥ {max(cis - 5, 30):.0f}"
        ),
        "exit":      (
            f"Supply route restored; TPS drops below 40; "
            f"conflict transitions to de-escalating state"
        ),
        "risk":      (
            f"Demand destruction from recession overwhelms supply premium; "
            f"OPEC+ supply response; strategic reserve releases"
        ),
        "tickers":   _resolve_tickers([asset]),
        "assets":    [asset],
        "direction": ["Long"],
        "regime":    [1, 2, 3],
        "category":  "Geopolitical" if scenario.get("safe_haven") else "Commodity",
        "conflict_id": conflict["id"],
        "scenarios": ["supply_shock", "escalation", "shipping_shock"],
        "confidence": round(confidence, 3),
        "upside_pct": round(confidence * 13, 1),
        "qc_flags":  qc_flags,
        "generated": True,
        "generated_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "stop":           f"{asset} 15d momentum turns negative OR TPS drops below 35",
        "target":         f"{asset} +10–15% from entry OR supply route confirmed restored",
        "invalidation":   f"Supply disruption resolves; conflict TPS drops below 30",
        "holding_period": "1–4 weeks (conflict-driven move; take profit on news)",
        "reward_risk":    f"{round(confidence * 2.2, 1)}:1 estimated",
    }


def _pair_long_short(
    long_asset: str,
    long_data: dict,
    short_asset: str,
    short_data: dict,
    conflict: dict,
    scenario: dict,
    scenario_id: str,
    cis: float,
    tps: float,
    regime: int,
) -> dict:
    """
    Long high-exposure geo-risk beneficiary / Short high-exposure equity victim.
    Used when both assets are exposed but in opposite directions
    (e.g. Long energy commodity / Short import-dependent equity).
    """
    long_beta  = long_data["beta"].get(conflict["id"], 0.0)
    short_beta = short_data["beta"].get(conflict["id"], 0.0)
    spread_beta = long_beta - short_beta

    confidence = float(np.clip(
        0.30
        + 0.20 * (cis / 100)
        + 0.15 * (tps / 100)
        + 0.15 * abs(spread_beta)
        + 0.10 * (long_data["sas"] / 100)
        + 0.10 * (regime / 3),
        0.30, 0.90,
    ))

    qc_flags: list[str] = []
    if confidence < 0.55:
        qc_flags.append("Low confidence (<55%)")
    if spread_beta < 0.05:
        qc_flags.append("Low beta spread — pair differentiation is weak")
    if long_data["sas"] < 20 or short_data["sas"] < 20:
        qc_flags.append("One leg has low scenario-adjusted exposure")

    return {
        "name":      f"Long {long_asset} / Short {short_asset}",
        "trigger":   f"{conflict['name']} — geo-risk beneficiary vs. victim pair",
        "rationale": (
            f"Both assets exposed to {conflict['name']} but in opposing directions. "
            f"{long_asset} (beta={long_beta:.2f}, SAS={long_data['sas']:.0f}) benefits from "
            f"supply disruption premium. "
            f"{short_asset} (beta={short_beta:.2f}, SAS={short_data['sas']:.0f}) faces "
            f"margin compression under the same shock. "
            f"Pair trade isolates the beta spread ({spread_beta:+.2f}) with reduced directional risk."
        ),
        "entry":     (
            f"CIS ≥ {cis:.0f}; TPS ≥ {tps:.0f}; "
            f"beta spread ({long_asset} − {short_asset}) ≥ 0.05; "
            f"regime Elevated or Crisis"
        ),
        "exit":      (
            f"Beta spread compresses to <0.02; conflict de-escalation; "
            f"scenario shifts to recovery or de-escalation"
        ),
        "risk":      (
            f"Macro demand shock can flip the pair (both fall together); "
            f"correlation snap-back in crisis compresses spread"
        ),
        "tickers":   _resolve_tickers([long_asset, short_asset]),
        "assets":    [long_asset, short_asset],
        "direction": ["Long", "Short"],
        "regime":    [2, 3],
        "category":  "Geopolitical",
        "conflict_id": conflict["id"],
        "scenarios": ["escalation", "supply_shock", "shipping_shock", "sanctions_shock"],
        "confidence": round(confidence, 3),
        "upside_pct": round(confidence * 11, 1),
        "qc_flags":  qc_flags,
        "generated": True,
        "generated_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "stop":           f"Beta spread ({long_asset}−{short_asset}) collapses to <0; macro demand shock",
        "target":         f"Pair P&L +10% OR conflict resolves; rebalance if beta spread exceeds 0.30",
        "invalidation":   f"Both assets move in same direction for 5+ days; conflict CIS drops below 30",
        "holding_period": "1–3 weeks (pair trade; rebalance on spread mean-reversion)",
        "reward_risk":    f"{round(confidence * 2.0, 1)}:1 estimated",
    }


def _deescalation_reversal(
    asset: str,
    asset_data: dict,
    conflict: dict,
    scenario: dict,
    scenario_id: str,
    cis: float,
    tps: float,
    regime: int,
) -> Optional[dict]:
    """Short geo-risk premium in de-escalation / recovery scenario."""
    if scenario_id not in ("de_escalation", "recovery"):
        return None
    beta = asset_data["beta"].get(conflict["id"], 0.0)
    if beta < 0.08:
        return None

    confidence = float(np.clip(
        0.30
        + 0.20 * (1 - cis / 100)    # lower CIS → higher confidence in reversal
        + 0.20 * (1 - tps / 100)
        + 0.15 * beta
        + 0.10 * (regime / 3),
        0.30, 0.85,
    ))

    qc_flags: list[str] = []
    if confidence < 0.55:
        qc_flags.append("Low confidence (<55%)")
    if conflict.get("escalation_trend") == "escalating":
        qc_flags.append("Conflict trend still escalating — de-escalation thesis fragile")

    return {
        "name":      f"Short {asset} — Geo Premium Reversal ({conflict['label']})",
        "trigger":   f"De-escalation in {conflict['name']} compresses {asset} geo premium",
        "rationale": (
            f"{asset} carries a structural geo-risk premium tied to {conflict['name']} "
            f"(beta={beta:.2f}). Under the {scenario['label']} scenario "
            f"(geo_mult={scenario['geo_mult']:.2f}x), the conflict contribution "
            f"to this asset's price is expected to compress. "
            f"Short position captures reversal of elevated CIS ({cis:.0f}) and TPS ({tps:.0f}) "
            f"toward normalised levels."
        ),
        "entry":     (
            f"Scenario = de_escalation or recovery; "
            f"CIS trending down; TPS below {tps:.0f} and falling; "
            f"{asset} still pricing elevated geo premium"
        ),
        "exit":      (
            f"Ceasefire fails; CIS re-accelerates; "
            f"asset beta compresses to <0.05"
        ),
        "risk":      (
            f"False de-escalation signal; new front opens; "
            f"supply disruption independently re-prices commodity"
        ),
        "tickers":   _resolve_tickers([asset]),
        "assets":    [asset],
        "direction": ["Short"],
        "regime":    [1, 2],
        "category":  "Macro",
        "conflict_id": conflict["id"],
        "scenarios": ["de_escalation", "recovery"],
        "confidence": round(confidence, 3),
        "upside_pct": round(confidence * 10, 1),
        "qc_flags":  qc_flags,
        "generated": True,
        "generated_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "stop":           f"CIS re-accelerates above {min(cis + 10, 85):.0f} or new escalation headline",
        "target":         f"{asset} geo premium mean-reverts −8–12% OR conflict TPS normalises to 25",
        "invalidation":   f"Peace talks collapse; CIS rises above {cis:.0f} for 3+ days",
        "holding_period": "1–3 weeks (news-driven; exit on conflict re-escalation)",
        "reward_risk":    f"{round(confidence * 2.0, 1)}:1 estimated",
    }


# ── Conflict–asset pairing logic ──────────────────────────────────────────────

def _top_n_exposed(
    all_assets: dict[str, dict],
    conflict_id: str,
    n: int = 3,
    exclude: Optional[set] = None,
) -> list[tuple[str, dict]]:
    """Top-n assets ranked by beta to a specific conflict."""
    exclude = exclude or set()
    ranked = sorted(
        [(name, d) for name, d in all_assets.items()
         if name not in exclude and d["beta"].get(conflict_id, 0.0) > 0],
        key=lambda x: x[1]["beta"].get(conflict_id, 0.0),
        reverse=True,
    )
    return ranked[:n]


def _top_n_hedges(
    all_assets: dict[str, dict],
    conflict_id: str,
    n: int = 2,
) -> list[tuple[str, dict]]:
    """Top-n hedge assets for a conflict by hedge_score."""
    from src.analysis.exposure import _SAFE_HAVEN_ASSETS
    candidates = [
        (name, d) for name, d in all_assets.items()
        if name in _SAFE_HAVEN_ASSETS and d["hedge_score"] >= _MIN_SAS_HEDGE
    ]
    return sorted(candidates, key=lambda x: x[1]["hedge_score"], reverse=True)[:n]


def _geo_risk_beneficiaries(
    all_assets: dict[str, dict],
    conflict_id: str,
    n: int = 3,
) -> list[tuple[str, dict]]:
    """Assets that are geo-risk beneficiaries (long_geo_risk direction, SAS >= threshold)."""
    from src.analysis.exposure import _GEO_RISK_BENEFICIARIES
    candidates = [
        (name, d) for name, d in all_assets.items()
        if name in _GEO_RISK_BENEFICIARIES
        and d["sas"] >= _MIN_SAS_LONG
        and d["beta"].get(conflict_id, 0.0) > 0.05
    ]
    return sorted(candidates, key=lambda x: x[1]["sas"], reverse=True)[:n]


def _equity_victims(
    all_assets: dict[str, dict],
    conflict_id: str,
    n: int = 2,
) -> list[tuple[str, dict]]:
    """Equity assets with high SAS and 'neutral' or negative direction (import-dependent)."""
    candidates = [
        (name, d) for name, d in all_assets.items()
        if any(tag in d.get("sector_tags", []) for tag in
               ["broad_equity", "semiconductors", "tech", "textiles", "ev_supply"])
        and d["sas"] >= _MIN_SAS_LONG
        and d["beta"].get(conflict_id, 0.0) > 0.05
    ]
    return sorted(candidates, key=lambda x: x[1]["sas"], reverse=True)[:n]


# ── Main generator ────────────────────────────────────────────────────────────

def generate_conflict_trades(
    regime: int = 1,
    conflict_results: Optional[dict] = None,
    all_assets: Optional[dict] = None,
    scenario_id: Optional[str] = None,
    max_per_conflict: int = 4,
) -> list[dict]:
    """
    Generate conflict-driven trade candidates from live scoring data.

    Parameters
    ----------
    regime          : int 0–3. Current correlation regime.
    conflict_results: output of score_all_conflicts(). Fetched if None.
    all_assets      : output of score_all_assets(). Fetched if None.
    scenario_id     : str scenario key. Uses session_state if None.
    max_per_conflict: max candidates to generate per active conflict.

    Returns
    -------
    list[dict]  — trade dicts ready for apply_filters()
    """
    from src.analysis.conflict_model import score_all_conflicts
    from src.analysis.exposure import score_all_assets
    from src.analysis.scenario_state import get_scenario, get_scenario_id

    if conflict_results is None:
        conflict_results = score_all_conflicts()
    if all_assets is None:
        all_assets = score_all_assets()
    if scenario_id is None:
        scenario_id = get_scenario_id()

    scenario = get_scenario()
    candidates: list[dict] = []

    from src.data.config import CONFLICTS
    conflict_map = {c["id"]: c for c in CONFLICTS}

    for cid, cresult in conflict_results.items():
        cis = float(cresult.get("cis", 0.0))
        tps = float(cresult.get("tps", 0.0))

        if cis < _MIN_CIS:
            continue  # not significant enough to drive trades

        conflict = conflict_map.get(cid)
        if conflict is None or conflict.get("state") == "frozen":
            continue

        generated_for_conflict = 0

        # ── 1. Safe-haven pairs ────────────────────────────────────────────
        hedges   = _top_n_hedges(all_assets, cid, n=1)
        exposed  = _top_n_exposed(all_assets, cid, n=2,
                                   exclude={h for h, _ in hedges})

        for hedge_name, hedge_data in hedges:
            for exp_name, exp_data in exposed:
                if exp_data["sas"] >= _MIN_SAS_LONG and cis >= 45:
                    t = _safe_haven_pair(
                        exp_name, exp_data,
                        hedge_name, hedge_data,
                        conflict, scenario, scenario_id,
                        cis, tps, regime,
                    )
                    candidates.append(t)
                    generated_for_conflict += 1
                    if generated_for_conflict >= max_per_conflict:
                        break
            if generated_for_conflict >= max_per_conflict:
                break

        # ── 2. Supply shock longs ──────────────────────────────────────────
        if generated_for_conflict < max_per_conflict and scenario_id in (
            "supply_shock", "escalation", "shipping_shock", "base",
        ):
            beneficiaries = _geo_risk_beneficiaries(all_assets, cid, n=3)
            for bname, bdata in beneficiaries:
                t = _supply_shock_long(
                    bname, bdata, conflict, scenario, scenario_id,
                    cis, tps, regime,
                )
                if t:
                    candidates.append(t)
                    generated_for_conflict += 1
                    if generated_for_conflict >= max_per_conflict:
                        break

        # ── 3. Pair trades (geo beneficiary vs. equity victim) ────────────
        if generated_for_conflict < max_per_conflict and tps >= 40:
            bens    = _geo_risk_beneficiaries(all_assets, cid, n=2)
            victims = _equity_victims(all_assets, cid, n=2)
            for bname, bdata in bens:
                for vname, vdata in victims:
                    if bname == vname:
                        continue
                    t = _pair_long_short(
                        bname, bdata, vname, vdata,
                        conflict, scenario, scenario_id,
                        cis, tps, regime,
                    )
                    candidates.append(t)
                    generated_for_conflict += 1
                    if generated_for_conflict >= max_per_conflict:
                        break
                if generated_for_conflict >= max_per_conflict:
                    break

        # ── 4. De-escalation reversals ─────────────────────────────────────
        if scenario_id in ("de_escalation", "recovery"):
            top_exp = _top_n_exposed(all_assets, cid, n=2)
            for aname, adata in top_exp:
                t = _deescalation_reversal(
                    aname, adata, conflict, scenario, scenario_id,
                    cis, tps, regime,
                )
                if t:
                    candidates.append(t)
                    generated_for_conflict += 1
                    if generated_for_conflict >= max_per_conflict:
                        break

    # Deduplicate by name (same pair can be generated from multiple conflicts)
    seen: set[str] = set()
    unique: list[dict] = []
    for t in candidates:
        key = t["name"]
        if key not in seen and float(t.get("confidence", 0.0)) >= _MIN_CONFIDENCE:
            seen.add(key)
            unique.append(t)

    unique.sort(key=lambda t: float(t.get("confidence", 0.5)), reverse=True)
    return unique


def merge_with_library(
    generated: list[dict],
    library: list[dict],
    regime: int,
) -> list[dict]:
    """
    Merge generated candidates with static _TRADE_LIBRARY entries.
    Static entries are regime-filtered and stamped with defaults for fields
    that trade_filter.py expects (confidence, qc_flags, scenarios).
    """
    # Backfill missing fields on static library entries
    augmented_library: list[dict] = []
    for t in library:
        t = dict(t)  # shallow copy — don't mutate original
        t.setdefault("confidence",   0.60)
        t.setdefault("qc_flags",     [])
        t.setdefault("scenarios",    ["base", "escalation", "supply_shock"])
        t.setdefault("conflict_id",  None)
        t.setdefault("generated",    False)

        if regime in t.get("regime", []):
            augmented_library.append(t)

    # Merge: generated first (conflict-aware, fresher), then static
    combined = generated + [
        t for t in augmented_library
        if t["name"] not in {g["name"] for g in generated}
    ]
    combined.sort(key=lambda t: float(t.get("confidence", 0.5)), reverse=True)
    return combined


# ── Individual signal-ranked candidate universe ──────────────────────────────
# Individual (single-name / single-asset) directional trades — NOT pairs —
# across a LIQUID universe people actually trade: liquid macro (energy, metals,
# major indices, safe havens; niche softs deprioritised), plus liquid single
# stocks (S&P 500 + top India NSE + top China HK), each mapped to the macro
# signal driving its sector. Screened candidates only: the eligibility → DSR
# gate still decides deployment, and the wider search raises the DSR penalty.

# Niche soft-ag commodities kept "lite" (excluded from generation)
_NICHE_AG = {"Cotton", "Coffee", "Sugar #11", "Soybeans"}

# Top India large-caps (NSE) — ticker → (name, sector)
_INDIA_UNIVERSE: dict[str, tuple[str, str]] = {
    "RELIANCE.NS": ("Reliance Industries", "Energy"),
    "TCS.NS": ("Tata Consultancy", "Tech"),
    "HDFCBANK.NS": ("HDFC Bank", "Financials"),
    "ICICIBANK.NS": ("ICICI Bank", "Financials"),
    "INFY.NS": ("Infosys", "Tech"),
    "HINDUNILVR.NS": ("Hindustan Unilever", "Consumer Staples"),
    "ITC.NS": ("ITC", "Consumer Staples"),
    "SBIN.NS": ("State Bank of India", "Financials"),
    "BHARTIARTL.NS": ("Bharti Airtel", "Communications"),
    "BAJFINANCE.NS": ("Bajaj Finance", "Financials"),
    "KOTAKBANK.NS": ("Kotak Mahindra Bank", "Financials"),
    "LT.NS": ("Larsen & Toubro", "Industrials"),
    "AXISBANK.NS": ("Axis Bank", "Financials"),
    "ASIANPAINT.NS": ("Asian Paints", "Materials"),
    "MARUTI.NS": ("Maruti Suzuki", "Consumer Discretionary"),
    "SUNPHARMA.NS": ("Sun Pharma", "Healthcare"),
    "TITAN.NS": ("Titan", "Consumer Discretionary"),
    "ONGC.NS": ("ONGC", "Energy"),
    "NTPC.NS": ("NTPC", "Utilities"),
}

# Top China large-caps (HK) — ticker → (name, sector)
_CHINA_UNIVERSE: dict[str, tuple[str, str]] = {
    "0700.HK": ("Tencent", "Tech"),
    "9988.HK": ("Alibaba", "Tech"),
    "3690.HK": ("Meituan", "Tech"),
    "0941.HK": ("China Mobile", "Communications"),
    "1299.HK": ("AIA Group", "Financials"),
    "0939.HK": ("China Construction Bank", "Financials"),
    "1810.HK": ("Xiaomi", "Tech"),
    "2318.HK": ("Ping An", "Financials"),
    "0883.HK": ("CNOOC", "Energy"),
    "1211.HK": ("BYD", "Consumer Discretionary"),
    "9618.HK": ("JD.com", "Consumer Discretionary"),
    "0388.HK": ("HKEX", "Financials"),
    "9999.HK": ("NetEase", "Tech"),
    "0857.HK": ("PetroChina", "Energy"),
    "2628.HK": ("China Life", "Financials"),
    "3968.HK": ("China Merchants Bank", "Financials"),
    "1088.HK": ("China Shenhua Energy", "Energy"),
    "0386.HK": ("Sinopec", "Energy"),
    "2382.HK": ("Sunny Optical", "Tech"),
    "1024.HK": ("Kuaishou", "Tech"),
}

# Stock sector → (macro signal proxy, direction sign, thesis angle).
# Proxy "__equity__" uses the region's index SAS; "__conflict__" uses conflict.
_STOCK_SECTOR_SIGNAL: dict[str, tuple[str, int, str]] = {
    "Energy":                 ("WTI Crude Oil", +1, "crude-leveraged earnings"),
    "Defense":                ("__conflict__",  +1, "elevated conflict intensity / defense spend"),
    "Gold Mining":            ("Gold",          +1, "gold-price leverage + safe-haven bid"),
    "Airlines":               ("WTI Crude Oil", -1, "jet-fuel cost squeeze when crude is bid"),
    "Industrial Metals":      ("Copper",        +1, "industrial-metals demand"),
    "Materials":              ("Copper",        +1, "commodity / materials beta"),
    "Utilities":              ("Natural Gas",   +1, "power generation, gas input pass-through"),
    "Agriculture":            ("Wheat",         +1, "ag supply-shock leverage"),
    "Financials":             ("__equity__",    +1, "rate / credit cycle"),
    "Tech":                   ("__equity__",    +1, "growth / risk appetite"),
    "Healthcare":             ("__equity__",    +1, "defensive quality"),
    "Consumer Staples":       ("__equity__",    +1, "defensive staples"),
    "Consumer Discretionary": ("__equity__",    +1, "consumer cycle"),
    "Industrials":            ("__equity__",    +1, "capex / cycle"),
    "Real Estate":            ("__equity__",    +1, "rate-sensitive property"),
    "Communications":         ("__equity__",    +1, "growth / connectivity"),
}

_REGION_INDEX = {"US": "S&P 500", "India": "Nifty 50", "China": "Hang Seng"}


def all_stock_universe() -> dict[str, tuple[str, str, str]]:
    """Combined liquid single-stock universe → {display_name: (ticker, sector, region)}.
    Lazy-imports the S&P 500 set from the page to avoid a circular import."""
    out: dict[str, tuple[str, str, str]] = {}
    try:
        from src.pages.trade_ideas import _SP500_UNIVERSE
        for tk, (nm, sec) in _SP500_UNIVERSE.items():
            out[nm] = (tk, sec, "US")
    except Exception:
        pass
    for tk, (nm, sec) in _INDIA_UNIVERSE.items():
        out[nm] = (tk, sec, "India")
    for tk, (nm, sec) in _CHINA_UNIVERSE.items():
        out[nm] = (tk, sec, "China")
    return out


def _confidence(sas: float, lo: float = 0.28, hi: float = 0.70, base: float = 0.30, div: float = 100.0) -> float:
    return float(np.clip(base + max(sas, 0.0) / div, lo, hi))


def _macro_directional(name, d, side, conf, regime, now) -> dict:
    sect = (d.get("sector_tags") or ["cross-asset"])[0]
    verb = "beneficiary" if side == "Long" else "underweight"
    return {
        "name": f"{side} {name}",
        "trigger": f"{name} SAS {d['sas']:.0f} — {sect} {verb} on the geo-risk / spillover signal",
        "rationale": (
            f"Individual directional signal: {name} ({sect}) scores SAS {d['sas']:.0f} on the current "
            f"geo-risk / spillover signal, a {'top' if side == 'Long' else 'bottom'}-ranked {sect} read. "
            f"Screened candidate, not discretionary — gated by backtest + deflated Sharpe."
        ),
        "entry": f"{name} SAS { 'above' if side=='Long' else 'below' } {max(12, d['sas'] + (-8 if side=='Long' else 8)):.0f}; regime {regime}",
        "exit": f"Signal rank reverts to the middle of the universe",
        "risk": "Single-leg directional — carries full asset beta, no relative hedge.",
        "tickers": _resolve_tickers([name]),
        "assets": [name], "direction": [side],
        "regime": [1, 2, 3] if regime >= 2 else [0, 1, 2],
        "category": "Commodity" if sect not in ("broad_equity", "tech", "small_cap") else "Equity Index",
        "scenarios": ["base", "escalation", "supply_shock"],
        "confidence": round(conf, 3), "upside_pct": round(conf * 11, 1),
        "qc_flags": [], "generated": True, "screened": True, "generated_at": now,
        "stop": "Signal rank falls out of the actionable band",
        "target": "Signal rank holds through the regime",
        "invalidation": "SAS rank mean-reverts to neutral",
        "holding_period": "2–6 weeks", "reward_risk": f"{round(1.1 + conf * 1.4, 1)}:1 est.",
    }


def _haven(name, d, conf, regime, now) -> dict:
    hs = d.get("hedge_score", 0)
    return {
        "name": f"Long {name} — safe-haven hedge",
        "trigger": f"{name} hedge score {hs:.0f}/100 — portfolio crisis ballast",
        "rationale": (
            f"Safe-haven hedge: {name} scores {hs:.0f}/100 on the crisis-ballast signal "
            f"(SAS {d.get('sas', 0):.0f}). Held as a hedge overlay, not an alpha bet."
        ),
        "entry": f"Regime ≥ Elevated OR rising cross-asset correlation; {name} hedge score ≥ 40",
        "exit": "Regime normalises / cross-asset correlation falls back",
        "risk": "Carry cost in calm regimes; hedge decays without a crisis.",
        "tickers": _resolve_tickers([name]), "assets": [name], "direction": ["Long"],
        "regime": [2, 3], "category": "Safe-Haven Hedge",
        "scenarios": ["escalation", "risk_off", "supply_shock"],
        "confidence": round(conf, 3), "upside_pct": round(conf * 9, 1),
        "qc_flags": [], "generated": True, "screened": True, "generated_at": now,
        "stop": "Regime falls to Normal for 5+ days",
        "target": "Outperforms risk assets during a correlation spike",
        "invalidation": "Sustained risk-on regime",
        "holding_period": "Hedge overlay — regime-conditional", "reward_risk": "Convex / hedge",
    }


def _stock_trade(disp, ticker, sector, region, side, sig_sas, angle, conf, regime, now) -> dict:
    return {
        "name": f"{side} {disp} ({region})",
        "trigger": f"{disp} · {sector} — {angle} (signal {sig_sas:.0f})",
        "rationale": (
            f"Individual single-name {region} equity: {disp} is a liquid {sector} name whose earnings track "
            f"the {angle} macro signal (strength {sig_sas:.0f} on the current read). Expressed {side.lower()} "
            f"the name directly. Screened candidate — gated by backtest + deflated Sharpe."
        ),
        "entry": f"{sector} signal { 'elevated' if side=='Long' else 'stretched'} (≈{sig_sas:.0f}); regime {regime}",
        "exit": f"{sector} macro signal normalises",
        "risk": f"Single-name idiosyncratic risk on top of the {sector} macro signal; liquidity/FX for non-US.",
        "tickers": {disp: ticker}, "assets": [disp], "direction": [side],
        "regime": [1, 2, 3] if regime >= 2 else [0, 1, 2],
        "category": f"{region} Equity",
        "scenarios": ["base", "escalation"],
        "confidence": round(conf, 3), "upside_pct": round(conf * 12, 1),
        "qc_flags": [], "generated": True, "screened": True, "generated_at": now,
        "stop": f"{sector} macro signal reverses",
        "target": f"Name re-rates with the {sector} signal",
        "invalidation": f"{sector} signal mean-reverts to neutral",
        "holding_period": "3–8 weeks", "reward_risk": f"{round(1.1 + conf * 1.5, 1)}:1 est.",
    }


def generate_signal_trades(
    regime: int = 1,
    all_assets: Optional[dict] = None,
    max_trades: int = 135,
) -> list[dict]:
    """Individual (single-name) directional candidates across the liquid macro +
    liquid stock universe (US / India / China), each mapped to the macro signal
    driving it. Ranked by signal strength; all India + China + macro always
    included, remaining slots filled with top-signal US names."""
    from src.analysis.exposure import score_all_assets
    if all_assets is None:
        try:
            all_assets = score_all_assets()
        except Exception:
            return []
    if not all_assets:
        return []
    now = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def sas(asset):  # macro SAS lookup
        return float((all_assets.get(asset) or {}).get("sas", 0.0))

    # conflict intensity proxy for Defense
    try:
        from src.analysis.conflict_model import score_all_conflicts, aggregate_portfolio_scores
        _conf_sig = float(aggregate_portfolio_scores(score_all_conflicts()).get("portfolio_cis", 45.0))
    except Exception:
        _conf_sig = 45.0

    def sector_signal(sector, region):
        proxy, sign, angle = _STOCK_SECTOR_SIGNAL.get(sector, ("__equity__", +1, "regional growth"))
        if proxy == "__equity__":
            s = sas(_REGION_INDEX.get(region, "S&P 500"))
        elif proxy == "__conflict__":
            s = _conf_sig
        else:
            s = sas(proxy)
        return s, ("Long" if sign > 0 else "Short"), angle

    macro, stocks_us, stocks_other = [], [], []

    # ── Macro individual (liquid; niche ag excluded) ──
    for n, d in sorted(all_assets.items(), key=lambda kv: -kv[1].get("sas", 0)):
        if n in _NICHE_AG:
            continue
        if d.get("direction") == "safe_haven":
            macro.append((d.get("hedge_score", 0), _haven(n, d, _confidence(d.get("hedge_score", 0), base=0.35, div=200), regime, now)))
        elif d.get("sas", 0) >= 22:
            macro.append((d["sas"] + 30, _macro_directional(n, d, "Long", _confidence(d["sas"], base=0.32, div=90), regime, now)))

    # ── Single stocks (US / India / China) ──
    for disp, (ticker, sector, region) in all_stock_universe().items():
        s, side, angle = sector_signal(sector, region)
        conf = _confidence(s, base=0.30, div=100)
        tr = _stock_trade(disp, ticker, sector, region, side, s, angle, conf, regime, now)
        (stocks_us if region == "US" else stocks_other).append((s, tr))

    # rank US by signal; always keep macro + all non-US (India/China)
    macro.sort(key=lambda x: -x[0])
    stocks_us.sort(key=lambda x: -x[0])
    keep = [t for _, t in macro] + [t for _, t in stocks_other]
    room = max(0, max_trades - len(keep))
    keep += [t for _, t in stocks_us[:room]]

    out, seen = [], set()
    for t in keep:
        if t["name"] in seen:
            continue
        seen.add(t["name"])
        out.append(t)
    return out

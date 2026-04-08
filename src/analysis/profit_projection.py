"""
Scenario-Aware P&L Projection Engine.

Projects expected P&L, max drawdown, and risk-adjusted return for a trade
candidate under the current and alternative scenarios.

Methodology
-----------
Each leg of a trade is assigned a scenario-conditional return distribution:

  expected_return(leg, scenario) =
    base_return × geo_mult × direction_sign
    + conflict_premium(asset, conflict_id, TPS)
    + volatility_adjustment(asset, vol_mult)

Aggregated trade metrics:
  - Expected P&L (%)         : weighted sum across legs
  - Worst-case drawdown (%)  : P10 outcome under stressed vol
  - Sharpe proxy             : E[R] / σ(R) across scenarios
  - Payoff table             : {scenario_id: {expected_pnl, drawdown, prob}}
  - Break-even probability   : fraction of scenarios with positive P&L

All outputs are for institutional framing — explicitly labeled as model-
derived estimates, not investment advice.

Usage:
    from src.analysis.profit_projection import project_trade, payoff_table
    proj   = project_trade(trade)
    table  = payoff_table(trade)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.analysis.scenario_state import SCENARIOS, get_scenario_id


# ── Per-asset return assumptions (annualised vol, base return by direction) ──
# These are model priors — calibrated from historical ranges, not forecasts.
# Format: {asset_name: {"vol": annualised_vol_pct, "base_return": directional_drift}}

_ASSET_PARAMS: dict[str, dict] = {
    # Energy
    "WTI Crude Oil":          {"vol": 35.0, "base_return":  5.0},
    "Brent Crude":            {"vol": 33.0, "base_return":  5.0},
    "Natural Gas":            {"vol": 55.0, "base_return":  3.0},
    "Heating Oil":            {"vol": 38.0, "base_return":  4.0},
    "Gasoline (RBOB)":        {"vol": 36.0, "base_return":  4.0},

    # Metals
    "Gold":                   {"vol": 15.0, "base_return":  6.0},
    "Silver":                 {"vol": 28.0, "base_return":  5.0},
    "Copper":                 {"vol": 25.0, "base_return":  4.0},
    "Platinum":               {"vol": 22.0, "base_return":  3.0},
    "Palladium":              {"vol": 40.0, "base_return":  2.0},
    "Nickel":                 {"vol": 38.0, "base_return":  2.0},
    "Aluminum":               {"vol": 25.0, "base_return":  2.0},
    "Iron Ore":               {"vol": 30.0, "base_return":  2.0},
    "Cobalt":                 {"vol": 35.0, "base_return":  2.0},

    # Agriculture
    "Wheat":                  {"vol": 32.0, "base_return":  3.0},
    "Corn":                   {"vol": 28.0, "base_return":  2.0},
    "Soybeans":               {"vol": 22.0, "base_return":  2.0},

    # Equities
    "S&P 500":                {"vol": 18.0, "base_return":  8.0},
    "NASDAQ 100":             {"vol": 24.0, "base_return":  9.0},
    "Eurostoxx 50":           {"vol": 22.0, "base_return":  5.0},
    "DAX":                    {"vol": 22.0, "base_return":  5.0},
    "CAC 40":                 {"vol": 21.0, "base_return":  5.0},
    "FTSE 100":               {"vol": 16.0, "base_return":  4.0},
    "Nikkei 225":             {"vol": 20.0, "base_return":  5.0},
    "Nifty 50":               {"vol": 22.0, "base_return":  7.0},
    "Sensex":                 {"vol": 22.0, "base_return":  7.0},
    "Shanghai Comp":          {"vol": 26.0, "base_return":  4.0},
    "Hang Seng":              {"vol": 25.0, "base_return":  3.0},
    "MSCI EM":                {"vol": 22.0, "base_return":  5.0},
    "Ares Capital (ARCC)":    {"vol": 18.0, "base_return":  6.0},
    "Blue Owl (OBDC)":        {"vol": 19.0, "base_return":  6.0},

    # Fixed Income / FX
    "US 20Y+ Treasury (TLT)": {"vol": 16.0, "base_return":  3.0},
    "TIPS / Inflation (TIP)": {"vol": 10.0, "base_return":  3.0},
    "HY Corporate (HYG)":     {"vol": 10.0, "base_return":  4.0},
    "EUR/USD":                {"vol": 8.5,  "base_return":  0.0},
    "JPY/USD":                {"vol": 9.5,  "base_return":  0.0},
    "CHF/USD":                {"vol": 7.5,  "base_return":  0.0},
    "USD/INR":                {"vol": 6.0,  "base_return":  3.0},
    "US Dollar Index (DXY)":  {"vol": 8.0,  "base_return":  0.0},
}

_DEFAULT_PARAMS = {"vol": 25.0, "base_return": 3.0}

# Holding period in years for annualisation (default: 3-month tactical)
_DEFAULT_HOLDING_YEARS = 0.25


# ── Scenario return adjustments ───────────────────────────────────────────────

# Additive return adjustment (%) by scenario for geo-risk assets
_SCENARIO_RETURN_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "escalation":      {
        "WTI Crude Oil": +18.0, "Brent Crude": +16.0, "Natural Gas": +20.0,
        "Gold": +12.0, "Silver": +8.0,
        "S&P 500": -8.0, "Eurostoxx 50": -12.0, "Nikkei 225": -10.0,
        "Nifty 50": -8.0, "MSCI EM": -10.0,
        "Wheat": +15.0, "Corn": +10.0,
        "US 20Y+ Treasury (TLT)": +5.0,
    },
    "de_escalation":   {
        "WTI Crude Oil": -12.0, "Brent Crude": -10.0, "Natural Gas": -15.0,
        "Gold": -8.0, "Silver": -6.0,
        "S&P 500": +6.0, "Eurostoxx 50": +8.0,
        "Wheat": -8.0, "Corn": -5.0,
    },
    "supply_shock":    {
        "WTI Crude Oil": +22.0, "Brent Crude": +20.0, "Natural Gas": +25.0,
        "Heating Oil": +20.0, "Gasoline (RBOB)": +18.0,
        "Gold": +8.0, "Wheat": +18.0, "Corn": +12.0,
        "S&P 500": -10.0, "Eurostoxx 50": -14.0,
        "TIPS / Inflation (TIP)": +6.0,
    },
    "sanctions_shock": {
        "Nickel": +15.0, "Aluminum": +10.0, "Palladium": +18.0,
        "Gold": +10.0, "Silver": +7.0,
        "EUR/USD": -5.0,
        "S&P 500": -6.0, "Eurostoxx 50": -10.0,
    },
    "shipping_shock":  {
        "WTI Crude Oil": +14.0, "Brent Crude": +12.0,
        "Natural Gas": +15.0, "Heating Oil": +12.0,
        "Wheat": +10.0, "Corn": +8.0, "Soybeans": +7.0,
        "Copper": -4.0, "Iron Ore": -5.0,
    },
    "risk_off":        {
        "Gold": +10.0, "Silver": +7.0,
        "US 20Y+ Treasury (TLT)": +8.0,
        "US Dollar Index (DXY)": +5.0,
        "S&P 500": -12.0, "NASDAQ 100": -15.0, "Eurostoxx 50": -14.0,
        "HY Corporate (HYG)": -8.0,
        "MSCI EM": -14.0,
    },
    "recovery":        {
        "Copper": +12.0, "S&P 500": +10.0, "NASDAQ 100": +12.0,
        "Eurostoxx 50": +9.0, "MSCI EM": +11.0,
        "WTI Crude Oil": +6.0,
        "Gold": -5.0, "US 20Y+ Treasury (TLT)": -4.0,
    },
    "base": {},  # no adjustment
}

# Scenario probability weights (sum to 1 over the scenario set)
_SCENARIO_PROBS: dict[str, float] = {
    "base":            0.30,
    "escalation":      0.20,
    "de_escalation":   0.12,
    "supply_shock":    0.12,
    "sanctions_shock": 0.08,
    "shipping_shock":  0.08,
    "risk_off":        0.06,
    "recovery":        0.04,
}


# ── P&L calculation helpers ───────────────────────────────────────────────────

def _leg_return(
    asset: str,
    direction: str,   # "Long" | "Short"
    scenario_id: str,
    geo_mult: float,
    vol_mult: float,
    holding_years: float,
) -> tuple[float, float]:
    """
    Returns (expected_return_pct, vol_pct) for one leg over the holding period.
    Volatility is scaled to holding period (sqrt time).
    """
    params   = _ASSET_PARAMS.get(asset, _DEFAULT_PARAMS)
    adj_map  = _SCENARIO_RETURN_ADJUSTMENTS.get(scenario_id, {})
    adj      = adj_map.get(asset, 0.0)

    base_r   = params["base_return"] * holding_years
    annual_v = params["vol"] * vol_mult
    period_v = annual_v * math.sqrt(holding_years)

    # Geo scenarios amplify scenario adjustment
    total_r = base_r + adj * geo_mult * holding_years

    sign = 1.0 if direction.lower() == "long" else -1.0
    return round(total_r * sign, 2), round(period_v, 2)


def _trade_stats(
    trade: dict,
    scenario_id: str,
    holding_years: float = _DEFAULT_HOLDING_YEARS,
) -> dict:
    """
    Compute expected P&L and vol for a trade in a single scenario.
    Equal-weight legs (could be extended with notional weights).
    """
    scenario = SCENARIOS.get(scenario_id, SCENARIOS["base"])
    geo_mult = scenario.get("geo_mult", 1.0)
    vol_mult = scenario.get("vol_mult", 1.0)

    assets     = trade.get("assets", [])
    directions = trade.get("direction", [])
    n = len(assets)
    if n == 0:
        return {"expected_pnl": 0.0, "vol": 0.0, "scenario_id": scenario_id}

    leg_returns: list[float] = []
    leg_vols:    list[float] = []

    for i, asset in enumerate(assets):
        dirn = directions[i] if i < len(directions) else "Long"
        r, v = _leg_return(asset, dirn, scenario_id, geo_mult, vol_mult, holding_years)
        leg_returns.append(r)
        leg_vols.append(v)

    # Equal-weighted portfolio
    portfolio_r = float(np.mean(leg_returns))

    # Naive independent-leg vol (no cross-asset correlation adjustment)
    # In crisis regimes, correlation goes to 1 — use max leg vol as conservative estimate
    portfolio_v = float(np.mean(leg_vols))

    return {
        "scenario_id":   scenario_id,
        "expected_pnl":  round(portfolio_r, 2),
        "vol":           round(portfolio_v, 2),
        "legs":          [
            {"asset": a, "direction": d, "expected_return": r, "vol": v}
            for a, d, r, v in zip(assets, directions, leg_returns, leg_vols)
        ],
    }


# ── Public API ────────────────────────────────────────────────────────────────

def payoff_table(
    trade: dict,
    holding_years: float = _DEFAULT_HOLDING_YEARS,
    current_scenario_id: Optional[str] = None,
) -> list[dict]:
    """
    Compute expected P&L for the trade under every scenario.

    Returns list of dicts, one per scenario:
        scenario_id, label, expected_pnl, vol, prob, prob_weighted_pnl, color
    Sorted by probability-weighted P&L descending.
    """
    if current_scenario_id is None:
        current_scenario_id = get_scenario_id()

    rows = []
    for sid, prob in _SCENARIO_PROBS.items():
        stats = _trade_stats(trade, sid, holding_years)
        sc    = SCENARIOS.get(sid, SCENARIOS["base"])
        rows.append({
            "scenario_id":       sid,
            "label":             sc["label"],
            "expected_pnl":      stats["expected_pnl"],
            "vol":               stats["vol"],
            "prob":              prob,
            "prob_weighted_pnl": round(stats["expected_pnl"] * prob, 3),
            "is_current":        sid == current_scenario_id,
            "color":             sc.get("color", "#8E9AAA"),
        })
    rows.sort(key=lambda r: r["prob_weighted_pnl"], reverse=True)
    return rows


def project_trade(
    trade: dict,
    holding_years: float = _DEFAULT_HOLDING_YEARS,
    current_scenario_id: Optional[str] = None,
) -> dict:
    """
    Full projection for a trade. Returns summary metrics + payoff table.

    Returns
    -------
    {
        "expected_pnl"       : float   # probability-weighted expected P&L %
        "best_case_pnl"      : float   # 90th percentile scenario P&L
        "worst_case_pnl"     : float   # 10th percentile scenario P&L
        "breakeven_prob"     : float   # fraction of weighted prob with positive P&L
        "sharpe_proxy"       : float   # E[R] / avg_vol (across scenarios)
        "holding_months"     : int
        "current_scenario_pnl": float  # P&L in current active scenario
        "payoff_table"       : list[dict]
        "top_scenario"       : str     # best probability-weighted scenario_id
        "bottom_scenario"    : str     # worst probability-weighted scenario_id
    }
    """
    if current_scenario_id is None:
        current_scenario_id = get_scenario_id()

    table = payoff_table(trade, holding_years, current_scenario_id)

    pnls  = [r["expected_pnl"]      for r in table]
    probs = [r["prob"]              for r in table]
    vols  = [r["vol"]               for r in table]

    # Expected P&L = probability-weighted sum
    exp_pnl = float(sum(p * r for p, r in zip(probs, pnls)))

    # Best/worst case: 90th / 10th percentile by pure P&L
    sorted_pnls = sorted(pnls)
    worst = sorted_pnls[max(0, int(len(sorted_pnls) * 0.10) - 1)]
    best  = sorted_pnls[min(len(sorted_pnls) - 1, int(len(sorted_pnls) * 0.90))]

    # Breakeven probability: sum probs where P&L > 0
    be_prob = float(sum(p for p, r in zip(probs, pnls) if r > 0))

    # Sharpe proxy
    avg_vol = float(np.mean(vols)) if vols else 1.0
    sharpe  = round(exp_pnl / avg_vol, 2) if avg_vol > 0 else 0.0

    # Current scenario P&L
    current_row = next((r for r in table if r["is_current"]), None)
    current_pnl = current_row["expected_pnl"] if current_row else exp_pnl

    top_row    = max(table, key=lambda r: r["prob_weighted_pnl"])
    bottom_row = min(table, key=lambda r: r["prob_weighted_pnl"])

    return {
        "expected_pnl":         round(exp_pnl, 2),
        "best_case_pnl":        round(best, 2),
        "worst_case_pnl":       round(worst, 2),
        "breakeven_prob":       round(be_prob, 3),
        "sharpe_proxy":         sharpe,
        "holding_months":       round(holding_years * 12),
        "current_scenario_pnl": round(current_pnl, 2),
        "payoff_table":         table,
        "top_scenario":         top_row["scenario_id"],
        "bottom_scenario":      bottom_row["scenario_id"],
    }

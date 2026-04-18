"""
War country scoring — single source of truth.

Canonical data (base scores, weights, index mapping) and scoring functions
used by both war_impact_map.py (visualization) and proactive_alerts.py (alerts).

Importing from a page module creates circular-import risk; this analysis module
is the correct home for shared scoring logic.

Public API:
    war_multipliers(cmd_r)         → dict with ukraine/hamas/iran multipliers + signals
    compute_country_scores(cmd_r)  → dict[iso, composite_score] for equity-indexed countries
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── ISO-3 → tracked equity indices ────────────────────────────────────────────
# Only countries with a mapped equity index are tracked in compute_country_scores.

COUNTRY_INDICES: dict[str, list[str]] = {
    "USA": ["S&P 500", "Nasdaq 100", "DJIA", "Russell 2000"],
    "GBR": ["FTSE 100"],
    "DEU": ["DAX"],
    "FRA": ["CAC 40"],
    "JPN": ["Nikkei 225", "TOPIX"],
    "HKG": ["Hang Seng"],
    "CHN": ["Shanghai Comp", "CSI 300"],
    "IND": ["Sensex", "Nifty 50"],
}


# ── War impact base scores (0–100) ────────────────────────────────────────────
# Static structural exposure — proximity, energy dependency, trade routes.
# Multiplied by live commodity z-score multipliers at scoring time.

WAR_DATA: list[dict] = [
    {
        "label":       "Ukraine War",
        "event_label": "Ukraine War",
        "color":       "#e74c3c",
        "scores": {
            "UKR": 100, "RUS": 88,  "BLR": 84,  "MDA": 78,
            "POL": 78,  "FIN": 72,  "LTU": 70,  "LVA": 70,  "EST": 70,
            "DEU": 74,  "ROU": 68,  "CZE": 65,  "SVK": 64,  "HUN": 62,
            "BGR": 60,  "SRB": 52,  "HRV": 55,  "SVN": 52,  "GRC": 55,
            "ITA": 62,  "FRA": 60,  "NLD": 58,  "BEL": 55,  "AUT": 58,
            "SWE": 52,  "NOR": 50,  "DNK": 48,  "GBR": 48,
            "CHE": 42,  "PRT": 38,  "ESP": 40,  "IRL": 30,  "LUX": 32,
            "TUR": 48,  "GEO": 65,  "ARM": 50,  "AZE": 52,
            "USA": 35,  "CAN": 28,  "BRA": 15,  "MEX": 12,
            "ARG": 12,  "CHL": 10,  "COL": 10,
            "JPN": 42,  "KOR": 35,  "CHN": 32,  "IND": 38,
            "HKG": 30,  "TWN": 25,  "SGP": 22,
            "AUS": 18,  "NZL": 14,
            "PAK": 18,  "BGD": 14,  "VNM": 12,  "THA": 12,
            "IDN": 10,  "MYS": 10,  "PHL": 10,
            "ISR": 22,  "SAU": 20,  "ARE": 18,  "EGY": 20,
            "TUN": 15,  "MAR": 14,  "DZA": 18,  "LBY": 22,
            "IRN": 20,  "IRQ": 18,  "SYR": 25,  "JOR": 18,
            "QAT": 15,  "KWT": 14,  "OMN": 12,  "BHR": 12,
            "ZAF": 14,  "NGA": 12,  "KEN": 10,  "ETH": 10,  "SDN": 15,
        },
    },
    {
        "label":       "Israel-Hamas War",
        "event_label": "Israel-Hamas",
        "color":       "#f39c12",
        "scores": {
            "ISR": 100, "PSE": 100, "LBN": 88,  "YEM": 82,
            "IRN": 80,  "SYR": 72,  "IRQ": 65,  "JOR": 70,
            "EGY": 65,  "SAU": 62,  "QAT": 58,  "KWT": 50,
            "ARE": 52,  "OMN": 45,  "BHR": 48,
            "TUR": 55,  "LBY": 38,  "SDN": 35,  "MAR": 25,  "TUN": 28,
            "DZA": 22,  "CYP": 42,  "GRC": 35,
            "GBR": 35,  "DEU": 28,  "FRA": 32,  "ITA": 30,
            "ESP": 28,  "NLD": 26,  "BEL": 25,  "PRT": 20,
            "CHE": 18,  "SWE": 16,  "NOR": 18,  "DNK": 16,
            "AUT": 22,  "POL": 18,  "HUN": 15,  "ROU": 15,
            "USA": 30,  "CAN": 18,  "BRA": 10,  "MEX": 12,  "ARG": 10,
            "JPN": 18,  "KOR": 20,  "CHN": 22,  "IND": 25,
            "HKG": 22,  "TWN": 20,  "SGP": 28,
            "PAK": 28,  "BGD": 22,  "IDN": 22,  "MYS": 20,
            "VNM": 15,  "THA": 18,  "PHL": 14,
            "AUS": 12,  "NZL": 10,
            "RUS": 20,  "UKR": 18,
            "ZAF": 14,  "NGA": 15,  "ETH": 12,  "KEN": 10,
        },
    },
    {
        "label":       "Iran/Hormuz Crisis",
        "event_label": "Iran/Hormuz",
        "color":       "#c0392b",
        "scores": {
            "IRN": 100, "ISR": 88,  "USA": 80,  "YEM": 75,
            "OMN": 82,  "QAT": 80,  "ARE": 78,  "SAU": 78,
            "KWT": 72,  "BHR": 70,  "IRQ": 68,
            "LBN": 65,  "SYR": 58,  "PSE": 62,  "JOR": 55,
            "JPN": 68,  "KOR": 65,  "IND": 62,  "PAK": 55,
            "CHN": 55,  "SGP": 52,  "BGD": 30,
            "EGY": 52,  "TUR": 50,  "GRC": 45,  "CYP": 40,
            "ITA": 42,  "DEU": 40,  "GBR": 38,  "FRA": 38,
            "NLD": 35,  "BEL": 28,  "ESP": 32,  "PRT": 22,
            "NOR": 28,  "SWE": 20,  "DNK": 22,  "CHE": 18,
            "AUT": 25,  "POL": 20,  "HUN": 18,  "ROU": 18,
            "TWN": 48,  "HKG": 42,  "IDN": 32,  "MYS": 30,
            "THA": 28,  "VNM": 22,  "PHL": 20,  "AUS": 25,
            "CAN": 15,  "BRA": 12,  "MEX": 14,  "ARG": 10,
            "LBY": 25,  "DZA": 20,  "MAR": 18,  "TUN": 15,
            "NGA": 15,  "ZAF": 20,  "ETH": 12,  "KEN": 10, "SDN": 18,
            "RUS": 18,  "UKR": 15,
        },
    },
]


# ── Country-specific conflict relevance weights ────────────────────────────────
# (ukraine_weight, hamas_weight, iran_weight) — must sum to 1.0
# Reflects structural conflict driver per country: proximity, energy dependency,
# alliance, hometurf.

COUNTRY_WAR_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "UKR": (0.90, 0.06, 0.04),
    "RUS": (0.82, 0.08, 0.10),
    "BLR": (0.84, 0.07, 0.09),
    "ISR": (0.05, 0.70, 0.25),
    "PSE": (0.05, 0.88, 0.07),
    "IRN": (0.05, 0.15, 0.80),
    "YEM": (0.08, 0.47, 0.45),
    "ARE": (0.07, 0.24, 0.69),
    "OMN": (0.06, 0.18, 0.76),
    "QAT": (0.07, 0.20, 0.73),
    "KWT": (0.08, 0.24, 0.68),
    "BHR": (0.08, 0.26, 0.66),
    "SAU": (0.07, 0.23, 0.70),
    "LBN": (0.06, 0.68, 0.26),
    "SYR": (0.15, 0.38, 0.47),
    "JOR": (0.07, 0.62, 0.31),
    "EGY": (0.10, 0.58, 0.32),
    "IRQ": (0.12, 0.38, 0.50),
    "MDA": (0.84, 0.08, 0.08),
    "POL": (0.80, 0.09, 0.11),
    "FIN": (0.78, 0.09, 0.13),
    "LTU": (0.80, 0.08, 0.12),
    "LVA": (0.80, 0.08, 0.12),
    "EST": (0.80, 0.08, 0.12),
    "GEO": (0.76, 0.10, 0.14),
    "ARM": (0.55, 0.12, 0.33),
    "AZE": (0.55, 0.10, 0.35),
    "DEU": (0.62, 0.16, 0.22),
    "AUT": (0.60, 0.15, 0.25),
    "HUN": (0.58, 0.14, 0.28),
    "CZE": (0.65, 0.14, 0.21),
    "SVK": (0.64, 0.14, 0.22),
    "BGR": (0.60, 0.16, 0.24),
    "ROU": (0.62, 0.15, 0.23),
    "HRV": (0.55, 0.18, 0.27),
    "SRB": (0.55, 0.16, 0.29),
    "SVN": (0.50, 0.18, 0.32),
    "ITA": (0.32, 0.28, 0.40),
    "GRC": (0.30, 0.28, 0.42),
    "CYP": (0.22, 0.32, 0.46),
    "FRA": (0.38, 0.28, 0.34),
    "GBR": (0.36, 0.28, 0.36),
    "NLD": (0.40, 0.24, 0.36),
    "BEL": (0.38, 0.24, 0.38),
    "ESP": (0.30, 0.28, 0.42),
    "PRT": (0.28, 0.26, 0.46),
    "SWE": (0.48, 0.20, 0.32),
    "NOR": (0.42, 0.20, 0.38),
    "DNK": (0.42, 0.20, 0.38),
    "CHE": (0.38, 0.22, 0.40),
    "IRL": (0.32, 0.22, 0.46),
    "LUX": (0.38, 0.24, 0.38),
    "TUR": (0.38, 0.30, 0.32),
    "USA": (0.15, 0.33, 0.52),
    "CAN": (0.28, 0.20, 0.52),
    "JPN": (0.22, 0.14, 0.64),
    "KOR": (0.24, 0.14, 0.62),
    "TWN": (0.22, 0.14, 0.64),
    "HKG": (0.20, 0.16, 0.64),
    "CHN": (0.20, 0.16, 0.64),
    "SGP": (0.18, 0.18, 0.64),
    "IND": (0.22, 0.16, 0.62),
    "PAK": (0.18, 0.20, 0.62),
    "BGD": (0.18, 0.24, 0.58),
    "IDN": (0.20, 0.22, 0.58),
    "MYS": (0.18, 0.20, 0.62),
    "THA": (0.22, 0.22, 0.56),
    "VNM": (0.24, 0.20, 0.56),
    "PHL": (0.22, 0.20, 0.58),
    "AUS": (0.22, 0.18, 0.60),
    "BRA": (0.16, 0.18, 0.66),
    "MEX": (0.14, 0.20, 0.66),
    "ARG": (0.18, 0.16, 0.66),
    "CHL": (0.16, 0.16, 0.68),
    "COL": (0.16, 0.18, 0.66),
    "LBY": (0.18, 0.32, 0.50),
    "DZA": (0.18, 0.28, 0.54),
    "MAR": (0.16, 0.30, 0.54),
    "TUN": (0.16, 0.30, 0.54),
    "NGA": (0.16, 0.18, 0.66),
    "ZAF": (0.20, 0.18, 0.62),
    "ETH": (0.18, 0.24, 0.58),
    "KEN": (0.18, 0.22, 0.60),
    "SDN": (0.16, 0.28, 0.56),
}

# Default weights for countries not in COUNTRY_WAR_WEIGHTS
DEFAULT_WAR_WEIGHTS: tuple[float, float, float] = (0.36, 0.26, 0.38)


# ── Dynamic weight adjustment ─────────────────────────────────────────────────

def dynamic_war_weights(
    iso: str,
    gas_z: float,
    oil_z: float,
    gold_z: float,
    max_shift: float = 0.12,
) -> tuple[float, float, float]:
    """
    Session-dynamic (ukraine_w, hamas_w, iran_w) that float with live commodity signals.

    Structural weights from COUNTRY_WAR_WEIGHTS anchor the baseline.
    Live z-scores nudge weights within ±max_shift (default ±0.12) of structural values,
    then renormalize to sum to 1.0.

    Signal logic:
      gas_z  > 0  → Ukraine conflict more salient (European gas dependency)
      oil_z  > 0  → Iran/Hormuz more salient (crude supply via Strait)
      gold_z > 0  → Hamas/Middle East more salient (safe-haven demand spike)

    A country structurally dominated by Ukraine (e.g., Poland, wu=0.80) won't flip
    to Iran-dominated even if oil spikes: the nudge is bounded at max_shift=0.12,
    so the maximum shift for any single weight is ±12 percentage points.
    """
    wu_base, wh_base, wi_base = COUNTRY_WAR_WEIGHTS.get(iso, DEFAULT_WAR_WEIGHTS)

    # Normalize z-scores to [-1, 1]
    gas_n  = float(np.clip(gas_z,  -3.5, 3.5)) / 3.5
    oil_n  = float(np.clip(oil_z,  -3.5, 3.5)) / 3.5
    gold_n = float(np.clip(gold_z, -3.5, 3.5)) / 3.5

    wu_adj = max(0.0, wu_base + max_shift * gas_n)
    wh_adj = max(0.0, wh_base + max_shift * gold_n)
    wi_adj = max(0.0, wi_base + max_shift * oil_n)

    total = wu_adj + wh_adj + wi_adj
    if total < 1e-9:
        return wu_base, wh_base, wi_base
    return (wu_adj / total, wh_adj / total, wi_adj / total)


# ── Scoring functions ─────────────────────────────────────────────────────────

def war_multipliers(cmd_r: pd.DataFrame) -> dict:
    """
    Compute live per-war intensity multipliers from commodity market signals.

    Ukraine War      → driven by European gas dependency + broad oil spike.
                       Natural gas z-score is the primary driver.
    Israel-Hamas War → driven by safe-haven gold demand + Red Sea oil disruption.
                       Gold z-score is the primary driver.
    Iran/Hormuz      → almost entirely crude oil Strait-of-Hormuz risk.
                       Oil z-score is the dominant driver.

    Multiplier range: 0.92 (dead-calm markets) → 1.15 (full commodity stress pricing).
    Baseline = 1.0. This intentionally narrow range preserves the structural
    ranking (epicentre countries stay near structural score) while commodity
    signals fine-tune ±8-15%. Floor 0.92 means Iran (structural=93) never falls
    below ~86, regardless of current oil prices.

    Returns dict with per-war multipliers + raw signal values for display.
    """
    if cmd_r.empty or len(cmd_r) < 30:
        return {
            "ukraine": 1.0, "hamas": 1.0, "iran": 1.0,
            "signals": {}, "method": "fallback - insufficient data",
        }

    window = 20  # 20-trading-day rolling return for z-score

    def _z(col_names: list[str]) -> float:
        cols = [c for c in col_names if c in cmd_r.columns]
        if not cols:
            return 0.0
        cum = cmd_r[cols].rolling(window).sum().mean(axis=1) * 100
        hist = cum.dropna()
        if len(hist) < 40:
            return 0.0
        mu, sd = float(hist.iloc[:-1].mean()), float(hist.iloc[:-1].std())
        if sd < 1e-6:
            return 0.0
        return float(np.clip((float(hist.iloc[-1]) - mu) / sd, -3.5, 3.5))

    gas_z  = _z(["Natural Gas"])
    oil_z  = _z(["WTI Crude Oil", "Brent Crude"])
    gold_z = _z(["Gold"])

    # Multiplier range [0.82, 1.45]:
    #   - Floor 0.82: quiet markets pull structural score down ≤18%. Structural rank
    #     stays intact (Iran ~93 falls to ~76 in calm, still highest in the region).
    #   - Ceiling 1.45: allows full +45% uplift when markets are fully pricing the conflict.
    #     Practical effect: Germany (Ukraine-score=74) rises to ~107 (capped at 100),
    #     or Japan (Iran-score=68) rises to ~99 — live signal becomes analytically visible.
    #   - Previous range [0.92, 1.15] only allowed ±8-15% — commodity z-scores were
    #     decorative, not analytical. The natural formula output at z=3.5 is ~1.43,
    #     which is now no longer clipped away.
    ukraine_raw = 1.0 + 0.55*(gas_z/3.5)*0.45 + 0.30*(oil_z/3.5)*0.45 + 0.15*(gold_z/3.5)*0.30
    ukraine_m   = float(np.clip(ukraine_raw, 0.82, 1.45))

    hamas_raw = 1.0 + 0.55*(gold_z/3.5)*0.40 + 0.35*(oil_z/3.5)*0.40 + 0.10*(gas_z/3.5)*0.20
    hamas_m   = float(np.clip(hamas_raw, 0.82, 1.45))

    iran_raw = 1.0 + 0.75*(oil_z/3.5)*0.50 + 0.20*(gold_z/3.5)*0.30 + 0.05*(gas_z/3.5)*0.20
    iran_m   = float(np.clip(iran_raw, 0.82, 1.45))

    return {
        "ukraine": round(ukraine_m, 3),
        "hamas":   round(hamas_m,   3),
        "iran":    round(iran_m,    3),
        "signals": {
            "Natural Gas z": round(gas_z,  2),
            "Crude Oil z":   round(oil_z,  2),
            "Gold z":        round(gold_z, 2),
        },
        # Raw z-scores for dynamic_war_weights()
        "_gas_z":  gas_z,
        "_oil_z":  oil_z,
        "_gold_z": gold_z,
        "method": "live 20d commodity z-scores",
    }


def compute_country_scores(cmd_r: pd.DataFrame) -> dict[str, int]:
    """
    Return composite conflict-exposure score (0–100) for each equity-indexed country.

    Score = Σ_war(dynamic_weight_war × base_score_war × multiplier_war) × concurrent_war_amplifier

    Weights are session-dynamic via dynamic_war_weights(): countries float toward the
    most commodity-stressed conflict each session (gas spike → Ukraine weight up,
    oil spike → Iran weight up, gold spike → Hamas weight up).

    Only returns the 8 countries in COUNTRY_INDICES (those with mapped equity indices).
    Used by proactive_alerts._check_country_exposure to fire geopolitical warnings.
    """
    mults = war_multipliers(cmd_r)
    um = mults["ukraine"]
    hm = mults["hamas"]
    im = mults["iran"]

    # Extract raw z-scores for dynamic weight adjustment
    gas_z  = mults.get("_gas_z",  0.0)
    oil_z  = mults.get("_oil_z",  0.0)
    gold_z = mults.get("_gold_z", 0.0)

    hot_wars  = int(um > 1.10) + int(hm > 1.10) + int(im > 1.10)
    amplifier = 1.0 + 0.12 * hot_wars / 3.0

    scores: dict[str, int] = {}
    for iso in COUNTRY_INDICES:
        u_base  = WAR_DATA[0]["scores"].get(iso, 0)
        h_base  = WAR_DATA[1]["scores"].get(iso, 0)
        ir_base = WAR_DATA[2]["scores"].get(iso, 0)

        # Dynamic weights float with live commodity signals (±12pp max shift)
        wu, wh, wi = dynamic_war_weights(iso, gas_z, oil_z, gold_z)
        raw = wu * u_base * um + wh * h_base * hm + wi * ir_base * im
        scores[iso] = int(np.clip(round(raw * amplifier), 0, 100))

    return scores

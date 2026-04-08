"""
Scenario State — global scenario switch and helpers.

The scenario switch is persisted in st.session_state["scenario"] and read
by every page that adjusts its framing, multipliers, or trade generation.
Call init_scenario() once per session (idempotent).
"""

from __future__ import annotations

# ── Scenario registry ─────────────────────────────────────────────────────────

SCENARIOS: dict[str, dict] = {
    "base": {
        "label":       "Base",
        "desc":        "Current macro and geopolitical conditions persist without material change.",
        "geo_mult":    1.00,
        "vol_mult":    1.00,
        "color":       "#8E9AAA",
        "cis_mult":    1.00,
        "tps_mult":    1.00,
        "safe_haven":  False,
        "short_bias":  False,
    },
    "escalation": {
        "label":       "Escalation",
        "desc":        "Active conflicts intensify; new fronts open; diplomatic channels break down.",
        "geo_mult":    1.45,
        "vol_mult":    1.30,
        "color":       "#c0392b",
        "cis_mult":    1.40,
        "tps_mult":    1.35,
        "safe_haven":  True,
        "short_bias":  False,
    },
    "de_escalation": {
        "label":       "De-escalation",
        "desc":        "Ceasefire talks progress; sanctions eased; supply routes reopen.",
        "geo_mult":    0.60,
        "vol_mult":    0.80,
        "color":       "#27ae60",
        "cis_mult":    0.60,
        "tps_mult":    0.65,
        "safe_haven":  False,
        "short_bias":  True,
    },
    "supply_shock": {
        "label":       "Supply Shock",
        "desc":        "Major commodity supply disruption — production cuts, infrastructure damage.",
        "geo_mult":    1.20,
        "vol_mult":    1.40,
        "color":       "#e67e22",
        "cis_mult":    1.10,
        "tps_mult":    1.50,
        "safe_haven":  False,
        "short_bias":  False,
    },
    "sanctions_shock": {
        "label":       "Sanctions Shock",
        "desc":        "New sweeping sanctions imposed; financial system fragmentation accelerates.",
        "geo_mult":    1.35,
        "vol_mult":    1.20,
        "color":       "#8e44ad",
        "cis_mult":    1.30,
        "tps_mult":    1.25,
        "safe_haven":  True,
        "short_bias":  False,
    },
    "shipping_shock": {
        "label":       "Shipping Shock",
        "desc":        "Major chokepoint disruption — Hormuz, Suez, or Malacca closure/blockade.",
        "geo_mult":    1.25,
        "vol_mult":    1.15,
        "color":       "#2980b9",
        "cis_mult":    1.05,
        "tps_mult":    1.60,
        "safe_haven":  False,
        "short_bias":  False,
    },
    "risk_off": {
        "label":       "Risk-Off",
        "desc":        "Broad market risk aversion — equities sell off; safe havens bid.",
        "geo_mult":    1.10,
        "vol_mult":    1.50,
        "color":       "#c0392b",
        "cis_mult":    1.10,
        "tps_mult":    1.10,
        "safe_haven":  True,
        "short_bias":  True,
    },
    "recovery": {
        "label":       "Recovery",
        "desc":        "Conflict resolution and macro improvement; risk-on rotation.",
        "geo_mult":    0.70,
        "vol_mult":    0.75,
        "color":       "#27ae60",
        "cis_mult":    0.65,
        "tps_mult":    0.70,
        "safe_haven":  False,
        "short_bias":  False,
    },
}

SCENARIO_ORDER = [
    "base", "escalation", "de_escalation",
    "supply_shock", "sanctions_shock", "shipping_shock",
    "risk_off", "recovery",
]


# ── Session state helpers ─────────────────────────────────────────────────────

def init_scenario() -> None:
    """Idempotently initialize scenario state."""
    import streamlit as st
    if "scenario" not in st.session_state:
        st.session_state["scenario"] = "base"
    # Validate — reset if stored value is no longer in registry
    if st.session_state["scenario"] not in SCENARIOS:
        st.session_state["scenario"] = "base"


def get_scenario() -> dict:
    """Return the current scenario definition dict."""
    import streamlit as st
    init_scenario()
    sid = st.session_state.get("scenario", "base")
    return SCENARIOS.get(sid, SCENARIOS["base"])


def get_scenario_id() -> str:
    import streamlit as st
    init_scenario()
    return st.session_state.get("scenario", "base")


def set_scenario(scenario_id: str) -> None:
    import streamlit as st
    if scenario_id in SCENARIOS:
        st.session_state["scenario"] = scenario_id


def apply_geo_mult(score: float) -> float:
    """Multiply a score by current scenario's geo_mult, clipped to [0, 100]."""
    mult = get_scenario().get("geo_mult", 1.0)
    return float(min(max(score * mult, 0.0), 100.0))


def apply_vol_mult(vol: float) -> float:
    mult = get_scenario().get("vol_mult", 1.0)
    return float(vol * mult)

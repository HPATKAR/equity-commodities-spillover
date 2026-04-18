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

# Compatible scenario combinations — these can be compounded
# e.g., "escalation AND shipping_shock" = Iran blockade + military escalation
SCENARIO_COMPOUNDS: list[dict] = [
    {
        "label":    "Escalation + Shipping Shock",
        "desc":     "Military escalation AND major chokepoint blockade — realistic Iran/Hormuz scenario.",
        "scenarios": ["escalation", "shipping_shock"],
        "color":    "#c0392b",
    },
    {
        "label":    "Supply Shock + Sanctions",
        "desc":     "Supply disruption compounded by financial sanctions — energy + financial channel hit.",
        "scenarios": ["supply_shock", "sanctions_shock"],
        "color":    "#8e44ad",
    },
    {
        "label":    "Risk-Off + Supply Shock",
        "desc":     "Risk aversion AND supply tightness — stagflationary shock pattern (2022 analog).",
        "scenarios": ["risk_off", "supply_shock"],
        "color":    "#e67e22",
    },
    {
        "label":    "Escalation + Sanctions + Shipping",
        "desc":     "Full-spectrum conflict: military escalation, new sanctions, AND chokepoint closure.",
        "scenarios": ["escalation", "sanctions_shock", "shipping_shock"],
        "color":    "#c0392b",
    },
]


def compound_scenarios(scenario_ids: list[str]) -> dict:
    """
    Compound multiple scenarios by multiplying their multipliers.

    This is the fix for GAP 14: simultaneous scenario stacking.
    Example: escalation (geo_mult=1.45) × shipping_shock (tps_mult=1.60)
    gives a combined geo_mult=1.45, tps_mult=1.60×1.35=2.16.

    The compound is computed as the element-wise product of each multiplier,
    capped at 3.0 to prevent unrealistic values.

    Returns a combined scenario dict.
    """
    valid_ids = [sid for sid in scenario_ids if sid in SCENARIOS and sid != "base"]
    if not valid_ids:
        return SCENARIOS["base"]
    if len(valid_ids) == 1:
        return SCENARIOS[valid_ids[0]]

    # Start from the first scenario's non-mult fields
    base = SCENARIOS[valid_ids[0]].copy()
    labels = [SCENARIOS[sid]["label"] for sid in valid_ids]
    base["label"] = " + ".join(labels)
    base["desc"]  = f"Compound: {' AND '.join(labels)}"
    base["color"] = SCENARIOS[valid_ids[0]]["color"]

    # Compound the multiplier fields
    _MULT_KEYS = ["geo_mult", "vol_mult", "cis_mult", "tps_mult"]
    for key in _MULT_KEYS:
        compound = 1.0
        for sid in valid_ids:
            compound *= SCENARIOS[sid].get(key, 1.0)
        base[key] = round(min(compound, 3.0), 3)  # cap at 3×

    # Boolean flags: OR across scenarios
    base["safe_haven"] = any(SCENARIOS[sid].get("safe_haven", False) for sid in valid_ids)
    base["short_bias"] = any(SCENARIOS[sid].get("short_bias", False) for sid in valid_ids)

    return base


# ── Session state helpers ─────────────────────────────────────────────────────

def init_scenario() -> None:
    """Idempotently initialize scenario state."""
    import streamlit as st
    if "scenario" not in st.session_state:
        st.session_state["scenario"] = "base"
    if "scenario_compound" not in st.session_state:
        st.session_state["scenario_compound"] = []
    # Validate — reset if stored value is no longer in registry
    if st.session_state["scenario"] not in SCENARIOS:
        st.session_state["scenario"] = "base"


def get_scenario() -> dict:
    """
    Return the current scenario definition dict.

    If compound scenarios are active (scenario_compound in session_state has >1 entry),
    returns the compounded scenario. Otherwise returns the single selected scenario.
    """
    import streamlit as st
    init_scenario()
    compound_ids = st.session_state.get("scenario_compound", [])
    if len(compound_ids) >= 2:
        return compound_scenarios(compound_ids)
    sid = st.session_state.get("scenario", "base")
    return SCENARIOS.get(sid, SCENARIOS["base"])


def get_scenario_id() -> str:
    import streamlit as st
    init_scenario()
    compound_ids = st.session_state.get("scenario_compound", [])
    if len(compound_ids) >= 2:
        return "+".join(compound_ids)
    return st.session_state.get("scenario", "base")


def set_scenario(scenario_id: str) -> None:
    import streamlit as st
    if scenario_id in SCENARIOS:
        st.session_state["scenario"] = scenario_id
        st.session_state["scenario_compound"] = []  # clear compound when single selected


def set_compound_scenario(scenario_ids: list[str]) -> None:
    """Activate a compound (multi-scenario) state."""
    import streamlit as st
    valid = [sid for sid in scenario_ids if sid in SCENARIOS]
    st.session_state["scenario_compound"] = valid
    if valid:
        st.session_state["scenario"] = valid[0]


def apply_geo_mult(score: float) -> float:
    """Multiply a score by current scenario's geo_mult, clipped to [0, 100]."""
    mult = get_scenario().get("geo_mult", 1.0)
    return float(min(max(score * mult, 0.0), 100.0))


def apply_vol_mult(vol: float) -> float:
    mult = get_scenario().get("vol_mult", 1.0)
    return float(vol * mult)

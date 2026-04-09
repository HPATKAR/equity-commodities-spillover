"""
Trade Filter Engine.

Filters trade candidates before generation and before display.
All filters operate on the trade dict schema used by trade_generator.py.

Filters:
  - min_confidence: reject ideas below confidence floor
  - conflict_beta: only ideas whose lead asset has beta >= threshold to an active conflict
  - direction: "long", "short", "all"
  - category: "Geopolitical", "Crisis Hedge", "Macro", "FX", etc.
  - regime_match: only show ideas that fire in the current regime
  - scenario_match: only show ideas whose scenario tag matches current scenario
  - max_qc_flags: reject ideas with too many quality flags
  - conflict_ids: only show ideas linked to selected conflict IDs (or "all")
  - country_perspective: filter by user's home market context (Global/US/India/Europe/Japan/UK)

Usage:
    from src.analysis.trade_filter import apply_filters, build_filter_ui
    filtered = apply_filters(candidates, filters)
"""

from __future__ import annotations

from typing import Optional


# ── Default filter thresholds ─────────────────────────────────────────────────

DEFAULTS = {
    "min_confidence":    0.45,
    "min_beta":          0.0,       # 0 = no beta filter
    "min_sas":           0.0,       # 0 = no SAS filter
    "direction":         "all",
    "category":          "all",
    "regime_match":      True,
    "scenario_match":    False,     # loose by default
    "max_qc_flags":      2,
    "conflict_ids":      [],        # empty = all conflicts pass
    "country_perspective": "Global",
}

# ── Country perspective asset relevance map ───────────────────────────────────
# For each country perspective, defines asset subsets that are most relevant.
# Trades where at least ONE asset matches the perspective's relevant set pass.
# "Global" perspective has no restriction (all assets pass).

_COUNTRY_ASSET_RELEVANCE: dict[str, set[str]] = {
    "Global": set(),   # empty = no restriction
    "US": {
        "S&P 500", "Nasdaq 100", "Russell 2K",
        "WTI Crude Oil", "Natural Gas", "Gold", "Silver", "Copper",
        "US 20Y+ Treasury (TLT)", "HY Corporate (HYG)", "IG Corporate (LQD)",
        "USD/INR", "DXY (Dollar Index)", "TIPS / Inflation (TIP)",
        "Ares Capital (ARCC)", "Blue Owl (OBDC)",
    },
    "India": {
        "Sensex", "Nifty 50",
        "Brent Crude", "WTI Crude Oil", "Natural Gas",
        "Gold", "Silver",
        "USD/INR",
        "Wheat", "Corn",
        "EM USD Bonds (EMB)",
    },
    "Europe": {
        "Eurostoxx 50", "DAX", "FTSE 100",
        "Brent Crude", "Natural Gas",
        "Gold", "Silver", "Copper",
        "EUR/USD",
        "Wheat",
    },
    "Japan": {
        "Nikkei 225",
        "Natural Gas", "Brent Crude", "WTI Crude Oil",
        "JPY/USD",
        "Gold",
        "Copper",
    },
    "UK": {
        "FTSE 100",
        "Brent Crude", "Natural Gas",
        "Gold", "Silver",
        "GBP/USD",
    },
    "China": {
        "Shanghai Comp",
        "Copper", "Iron Ore", "Nickel", "Aluminium",
        "Brent Crude", "Natural Gas",
        "Gold",
    },
}

# ── Conflict ID registry for filter UI ───────────────────────────────────────
# Maps conflict_id → display label

_CONFLICT_DISPLAY: dict[str, str] = {
    "ukraine_russia":   "Ukraine / Russia",
    "red_sea_houthi":   "Red Sea / Houthi",
    "israel_gaza":      "Israel / Gaza",
    "iran_conflict":    "Iran / Hormuz",
    "india_pakistan":   "India / Pakistan",
    "taiwan_strait":    "Taiwan Strait",
}


# ── Filter logic ──────────────────────────────────────────────────────────────

def _passes_confidence(trade: dict, min_conf: float) -> bool:
    return float(trade.get("confidence", 0.5)) >= min_conf


def _passes_direction(trade: dict, direction: str) -> bool:
    if direction == "all":
        return True
    dirs = [d.lower() for d in trade.get("direction", [])]
    if direction == "long":
        return "long" in dirs
    if direction == "short":
        return "short" in dirs
    return True


def _passes_category(trade: dict, category: str) -> bool:
    if category == "all":
        return True
    return trade.get("category", "").lower() == category.lower()


def _passes_regime(trade: dict, current_regime: int) -> bool:
    regimes = trade.get("regime", [])
    if not regimes:
        return True
    return current_regime in regimes


def _passes_scenario(trade: dict, current_scenario_id: str) -> bool:
    scenarios = trade.get("scenarios", [])
    if not scenarios:
        return True   # no scenario constraint = always show
    return current_scenario_id in scenarios


def _passes_conflict_beta(
    trade: dict,
    min_beta: float,
    conflict_betas: Optional[dict] = None,   # {asset: {conflict_id: beta}}
) -> bool:
    """
    Check if any asset in the trade has conflict beta >= min_beta.
    conflict_betas: pre-computed from exposure.py score_all_assets()
    """
    if min_beta <= 0 or conflict_betas is None:
        return True
    assets = trade.get("assets", [])
    for asset in assets:
        betas = conflict_betas.get(asset, {}).get("beta", {})
        if any(v >= min_beta for v in betas.values()):
            return True
    return False


def _passes_qc_flags(trade: dict, max_flags: int) -> bool:
    flags = trade.get("qc_flags", [])
    return len(flags) <= max_flags


def _passes_conflict_id_filter(trade: dict, conflict_ids: list[str]) -> bool:
    """
    If conflict_ids is non-empty, only allow trades linked to those conflicts.
    Static library trades (no conflict_id) pass when conflict_ids includes
    a sentinel "static" OR when conflict_ids is empty.
    """
    if not conflict_ids:
        return True   # no filter active
    trade_conflict = trade.get("conflict_id")
    if not trade_conflict:
        # Static trade — passes unless the user explicitly excluded them
        return "static" in conflict_ids or len(conflict_ids) == len(_CONFLICT_DISPLAY)
    return trade_conflict in conflict_ids


def _passes_min_sas(
    trade: dict,
    min_sas: float,
    asset_exposure: Optional[dict] = None,
) -> bool:
    """
    Require at least one asset in the trade to have SAS >= min_sas.
    Passes if min_sas == 0 or no exposure data available.
    Static trades (no conflict_id) are not filtered by SAS.
    """
    if min_sas <= 0 or asset_exposure is None:
        return True
    if not trade.get("conflict_id"):
        return True   # static library trades bypass SAS filter
    assets = trade.get("assets", [])
    for a in assets:
        ed = asset_exposure.get(a)
        if ed and ed.get("sas", 0.0) >= min_sas:
            return True
    return False


def _passes_country_perspective(trade: dict, perspective: str) -> bool:
    """
    Allow trades where at least one asset is in the country's relevant asset universe.
    "Global" perspective has no restriction.
    Static trades without conflict_id always pass (they're already regime-filtered).
    """
    if perspective == "Global":
        return True
    relevant = _COUNTRY_ASSET_RELEVANCE.get(perspective, set())
    if not relevant:
        return True
    assets = trade.get("assets", [])
    return any(a in relevant for a in assets)


def apply_filters(
    candidates: list[dict],
    filters: Optional[dict] = None,
    current_regime: int = 1,
    current_scenario_id: str = "base",
    conflict_betas: Optional[dict] = None,
    asset_exposure: Optional[dict] = None,
) -> list[dict]:
    """
    Apply all active filters to a list of trade candidates.
    Returns filtered list, sorted by confidence desc.

    filters dict keys (all optional, defaults from DEFAULTS):
        min_confidence, min_beta, min_sas, direction, category,
        regime_match, scenario_match, max_qc_flags,
        conflict_ids, country_perspective
    """
    if filters is None:
        filters = {}

    min_conf         = float(filters.get("min_confidence",    DEFAULTS["min_confidence"]))
    min_beta         = float(filters.get("min_beta",          DEFAULTS["min_beta"]))
    min_sas          = float(filters.get("min_sas",           DEFAULTS["min_sas"]))
    direction        = filters.get("direction",               DEFAULTS["direction"])
    category         = filters.get("category",                DEFAULTS["category"])
    regime_match     = bool(filters.get("regime_match",       DEFAULTS["regime_match"]))
    scenario_match   = bool(filters.get("scenario_match",     DEFAULTS["scenario_match"]))
    max_qc_flags     = int(filters.get("max_qc_flags",        DEFAULTS["max_qc_flags"]))
    conflict_ids     = list(filters.get("conflict_ids",       DEFAULTS["conflict_ids"]))
    country_persp    = filters.get("country_perspective",     DEFAULTS["country_perspective"])

    result = []
    for t in candidates:
        if not _passes_confidence(t, min_conf):
            continue
        if not _passes_direction(t, direction):
            continue
        if not _passes_category(t, category):
            continue
        if regime_match and not _passes_regime(t, current_regime):
            continue
        if scenario_match and not _passes_scenario(t, current_scenario_id):
            continue
        if not _passes_conflict_beta(t, min_beta, conflict_betas):
            continue
        if not _passes_qc_flags(t, max_qc_flags):
            continue
        if not _passes_conflict_id_filter(t, conflict_ids):
            continue
        if not _passes_country_perspective(t, country_persp):
            continue
        if not _passes_min_sas(t, min_sas, asset_exposure):
            continue
        result.append(t)

    result.sort(key=lambda t: float(t.get("confidence", 0.5)), reverse=True)
    return result


def score_trade_quality(trade: dict) -> dict:
    """
    QC scoring for a trade. Returns {score: 0–100, flags: [str], grade: str}.
    Used by challenge_trade() protocol and display.
    """
    flags: list[str] = list(trade.get("qc_flags", []))
    score = 100

    if float(trade.get("confidence", 0.5)) < 0.55:
        flags.append("Low confidence (<55%)")
        score -= 20

    assets = trade.get("assets", [])
    if len(assets) < 2:
        flags.append("Single-asset trade — no pair structure")
        score -= 10

    if not trade.get("entry"):
        flags.append("Missing entry trigger")
        score -= 15

    if not trade.get("exit"):
        flags.append("Missing exit criteria")
        score -= 15

    if not trade.get("rationale"):
        flags.append("No rationale documented")
        score -= 20

    if trade.get("category") == "Geopolitical" and not trade.get("conflict_id"):
        flags.append("Geo trade missing conflict_id link")
        score -= 5

    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 50 else "D"
    return {"score": max(0, score), "flags": flags, "grade": grade}


def build_filter_ui(key_prefix: str = "ti") -> dict:
    """
    Render Streamlit filter widgets and return a filters dict.
    Call from within a Streamlit page.

    Includes:
      - War/conflict selector: filter by specific active conflicts
      - Country perspective: surface trades relevant to user's market context
      - Standard filters: confidence, direction, category, regime, beta, QC flags
    """
    import streamlit as st

    with st.expander("Set Filters — War, Country Perspective & Quality Criteria",
                     expanded=True):
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            'color:#555960;letter-spacing:.18em;text-transform:uppercase;margin-bottom:6px">'
            'FILTER CRITERIA — SET BEFORE REVIEWING TRADES</p>',
            unsafe_allow_html=True,
        )

        # ── Row 1: War filter + Country perspective ────────────────────────
        r1c1, r1c2 = st.columns([2, 1])

        with r1c1:
            _conflict_options = list(_CONFLICT_DISPLAY.items())  # [(id, label), ...]
            _conflict_labels  = [label for _, label in _conflict_options]
            _conflict_ids_map = {label: cid for cid, label in _conflict_options}

            selected_war_labels = st.multiselect(
                "Wars / Conflicts to include",
                options=_conflict_labels,
                default=[],
                placeholder="All conflicts (leave empty for no restriction)",
                key=f"{key_prefix}_conflict_ids",
                help=(
                    "Restrict trade generation to ideas linked to specific conflicts. "
                    "Leave empty to see all. Static library trades always show unless "
                    "a conflict filter is set."
                ),
            )
            selected_conflict_ids = [_conflict_ids_map[lbl] for lbl in selected_war_labels]

        with r1c2:
            country_perspective = st.selectbox(
                "Country perspective",
                options=["Global", "US", "India", "Europe", "Japan", "UK", "China"],
                index=0,
                key=f"{key_prefix}_country",
                help=(
                    "Filters trades to those relevant to your home market. "
                    "'India' surfaces Nifty, Brent, INR, Gold trades. "
                    "'Japan' surfaces Nikkei, LNG, JPY trades. "
                    "'Global' shows all ideas."
                ),
            )

        # ── Row 2: Quality criteria ────────────────────────────────────────
        st.markdown(
            '<div style="border-top:1px solid #1e1e1e;margin:8px 0 6px"></div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        min_confidence = c1.slider(
            "Min confidence", 0.30, 0.90, 0.45, 0.05,
            key=f"{key_prefix}_min_conf",
        )
        direction = c2.selectbox(
            "Direction", ["all", "long", "short"],
            key=f"{key_prefix}_direction",
        )
        category = c3.selectbox(
            "Category",
            ["all", "Geopolitical", "Crisis Hedge", "Macro", "FX",
             "Commodity", "Fixed Income", "India/EM", "Private Credit"],
            key=f"{key_prefix}_category",
        )
        c4, c5, c6, c7 = st.columns(4)
        regime_match = c4.checkbox("Regime-matched only", value=True,
                                   key=f"{key_prefix}_regime")
        min_beta = c5.slider(
            "Min conflict beta", 0.0, 0.8, 0.0, 0.05,
            key=f"{key_prefix}_min_beta",
            help="Only show ideas where lead asset has ≥ this beta to an active conflict",
        )
        min_sas = c6.slider(
            "Min asset SAS", 0, 80, 0, 5,
            key=f"{key_prefix}_min_sas",
            help="Require at least one trade asset to have Scenario-Adjusted Score ≥ this value (0 = no filter). Static library trades are exempt.",
        )
        max_flags = c7.number_input(
            "Max QC flags", 0, 5, 2, key=f"{key_prefix}_max_flags",
        )

        # ── Active filter summary ──────────────────────────────────────────
        _active_summary = []
        if selected_conflict_ids:
            _active_summary.append(
                f"Wars: {', '.join(selected_war_labels[:3])}"
                + (f" +{len(selected_war_labels)-3} more" if len(selected_war_labels) > 3 else "")
            )
        if country_perspective != "Global":
            _active_summary.append(f"Perspective: {country_perspective}")
        if min_confidence > 0.45:
            _active_summary.append(f"Confidence ≥ {min_confidence:.0%}")
        if min_beta > 0:
            _active_summary.append(f"Beta ≥ {min_beta:.2f}")
        if min_sas > 0:
            _active_summary.append(f"SAS ≥ {min_sas}")
        if direction != "all":
            _active_summary.append(f"Direction: {direction.title()}")
        if category != "all":
            _active_summary.append(f"Category: {category}")

        if _active_summary:
            st.markdown(
                '<p style="font-family:\'JetBrains Mono\',monospace;font-size:7.5px;'
                'color:#CFB991;margin-top:4px">'
                'Active: ' + " · ".join(_active_summary) + '</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p style="font-family:\'JetBrains Mono\',monospace;font-size:7.5px;'
                'color:#555960;margin-top:4px">'
                'No active restrictions — showing all eligible trades</p>',
                unsafe_allow_html=True,
            )

    return {
        "min_confidence":      min_confidence,
        "direction":           direction,
        "category":            category,
        "regime_match":        regime_match,
        "min_beta":            min_beta,
        "min_sas":             float(min_sas),
        "max_qc_flags":        int(max_flags),
        "conflict_ids":        selected_conflict_ids,
        "country_perspective": country_perspective,
    }

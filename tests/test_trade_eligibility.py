"""
Step 1 of 4 — trade eligibility gate.

A trade is ELIGIBLE only if every leg exists in the return data AND its
thesis passed Stage-3 confirmation. Phantom-leg and unconfirmed trades are
NON-ALLOCATABLE: annotated with a specific reason and provably locked to
zero weight through enforce_weight(), the choke point all downstream
allocation (steps 2–4) must route through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.trade_filter import annotate_eligibility, enforce_weight

COLS = ["Gold", "Eurostoxx 50", "WTI Crude Oil", "S&P 500", "Natural Gas"]

PASSED = {"stage_passed": True, "confirmation_score": 0.7}
FAILED = {"stage_passed": False, "rejection_reason": "sign check failed on Gold"}


def _trade(name, assets, direction=None):
    return {"name": name, "assets": assets,
            "direction": direction or ["Long"] * len(assets)}


# ── Verdicts and reasons ─────────────────────────────────────────────────────

def test_clean_trade_is_eligible():
    t = _trade("A", ["Gold", "Eurostoxx 50"])
    annotate_eligibility([t], COLS, {"A": PASSED})
    assert t["is_eligible"] is True
    assert t["eligibility_reason"] == "eligible"
    assert t["max_weight"] == 1.0


def test_phantom_leg_is_locked_and_names_the_legs():
    t = _trade("B", ["Gold", "LQD", "HYG"])       # bond ETFs not in the frame
    annotate_eligibility([t], COLS, {"B": PASSED})
    assert t["is_eligible"] is False
    assert "missing legs: LQD, HYG" in t["eligibility_reason"]
    assert t["max_weight"] == 0.0


def test_failed_thesis_is_locked_with_rejection_text():
    t = _trade("C", ["Gold", "S&P 500"])
    annotate_eligibility([t], COLS, {"C": FAILED})
    assert t["is_eligible"] is False
    assert "thesis unconfirmed: sign check failed on Gold" in t["eligibility_reason"]


def test_missing_thesis_result_is_locked():
    t = _trade("D", ["Gold"])
    annotate_eligibility([t], COLS, {})            # never confirmed
    assert t["is_eligible"] is False
    assert "no Stage-3 result" in t["eligibility_reason"]


def test_both_failures_report_both_reasons():
    t = _trade("E", ["Gold", "ARCC"])
    annotate_eligibility([t], COLS, {"E": FAILED})
    r = t["eligibility_reason"]
    assert "missing legs: ARCC" in r
    assert "thesis unconfirmed" in r


def test_no_legs_defined_is_locked():
    t = {"name": "F", "assets": []}
    annotate_eligibility([t], COLS, {"F": PASSED})
    assert t["is_eligible"] is False
    assert "no legs defined" in t["eligibility_reason"]


# ── The hard zero-weight lock ────────────────────────────────────────────────

def test_lock_is_provable_for_any_proposed_weight():
    t = _trade("G", ["Gold", "NOPE"])
    annotate_eligibility([t], COLS, {"G": PASSED})
    assert t["is_eligible"] is False
    for proposed in (0.8, 1.0, -0.5, 1e9, 0.0):
        assert enforce_weight(t, proposed) == 0.0, (
            f"ineligible trade received weight {proposed} — lock breached"
        )


def test_eligible_trade_weight_passes_through():
    t = _trade("H", ["Gold", "S&P 500"])
    annotate_eligibility([t], COLS, {"H": PASSED})
    assert enforce_weight(t, 0.35) == 0.35
    assert enforce_weight(t, 0.0) == 0.0


def test_unannotated_trade_defaults_to_locked():
    # A trade that never passed through the gate must not be allocatable
    assert enforce_weight({"name": "raw"}, 0.5) == 0.0


# ── Structurally dead vs merely missing ──────────────────────────────────────
# A leg with NO loader mapping at all is structurally dead — permanently
# untradeable, counted separately. A mapped leg absent from today's frame is
# a live "missing legs" lock that heals when the data loads.

UNIVERSE = set(COLS) | {"LQD", "HYG"}   # LQD/HYG mapped but not loaded today


def test_dead_leg_marks_trade_structurally_dead():
    t = _trade("X", ["Gold", "Imaginary Asset"])
    annotate_eligibility([t], COLS, {"X": PASSED}, loadable_universe=UNIVERSE)
    assert t["structurally_dead"] is True
    assert t["is_eligible"] is False
    assert "structurally dead — no data source for: Imaginary Asset" \
        in t["eligibility_reason"]
    assert enforce_weight(t, 0.99) == 0.0          # still zero-locked


def test_mapped_but_unloaded_leg_is_missing_not_dead():
    t = _trade("Y", ["Gold", "LQD"])               # LQD mapped, not loaded
    annotate_eligibility([t], COLS, {"Y": PASSED}, loadable_universe=UNIVERSE)
    assert t["structurally_dead"] is False
    assert "missing legs: LQD" in t["eligibility_reason"]


def test_dead_and_missing_legs_both_reported():
    t = _trade("Z", ["Gold", "LQD", "Imaginary Asset"])
    annotate_eligibility([t], COLS, {"Z": PASSED}, loadable_universe=UNIVERSE)
    assert t["structurally_dead"] is True
    assert "no data source for: Imaginary Asset" in t["eligibility_reason"]
    assert "missing legs: LQD" in t["eligibility_reason"]


def test_no_universe_disables_dead_marking():
    t = _trade("W", ["Gold", "Imaginary Asset"])
    annotate_eligibility([t], COLS, {"W": PASSED})     # legacy call
    assert t["structurally_dead"] is False
    assert "missing legs: Imaginary Asset" in t["eligibility_reason"]


def test_eligible_trade_is_never_dead():
    t = _trade("V", ["Gold", "S&P 500"])
    annotate_eligibility([t], COLS, {"V": PASSED}, loadable_universe=UNIVERSE)
    assert t["is_eligible"] is True
    assert t["structurally_dead"] is False

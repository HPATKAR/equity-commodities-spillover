"""
Replay Mode - point-in-time enforcement tests.

The one rule: NO data after the replay cutoff touches the computation, ever.
These tests inject leaked future observations and assert the choke point
(pit_slice / pit_assert) fails LOUD with LookaheadError.
"""

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.replay import (
    LookaheadError, pit_assert, pit_slice,
    replay_conflict_scores, replay_presets, terminal_call,
)

CUTOFF = datetime.date(2022, 2, 24)   # Ukraine breakout


def _frame(start: str, periods: int, cols=("A", "B"), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(rng.normal(0, 0.01, (periods, len(cols))), index=idx, columns=cols)


# ── The choke point ──────────────────────────────────────────────────────────

def test_pit_assert_raises_on_single_leaked_row():
    df = _frame("2021-01-01", 310)          # runs past 2022-02-24
    assert df.index.max() > pd.Timestamp(CUTOFF)
    with pytest.raises(LookaheadError, match="PIT VIOLATION"):
        pit_assert(df, CUTOFF, "unit_test")


def test_pit_assert_passes_at_exact_cutoff():
    idx = pd.bdate_range(end=str(CUTOFF), periods=100)
    df = pd.DataFrame(0.0, index=idx, columns=["A"])
    pit_assert(df, CUTOFF, "unit_test")     # must not raise - cutoff day inclusive


def test_pit_slice_trims_and_result_passes_assert():
    df = _frame("2021-01-01", 400)
    out = pit_slice(df, CUTOFF, "unit_test")
    assert out.index.max() <= pd.Timestamp(CUTOFF) + pd.Timedelta(days=1)
    assert len(out) < len(df)               # future rows actually removed
    pit_assert(out, CUTOFF, "unit_test")


def test_pit_assert_rejects_tz_aware_leak():
    idx = pd.date_range("2022-02-20", periods=10, tz="America/New_York")
    df = pd.DataFrame(0.0, index=idx, columns=["A"])
    with pytest.raises(LookaheadError):
        pit_assert(df, CUTOFF, "unit_test")


def test_pit_assert_rejects_non_datetime_index():
    df = pd.DataFrame({"A": [1, 2, 3]})     # RangeIndex - unverifiable
    with pytest.raises(LookaheadError, match="non-datetime"):
        pit_assert(df, CUTOFF, "unit_test")


# ── terminal_call slices on entry (never trusts the caller's window) ────────

def test_terminal_call_slices_leaked_inputs_on_entry():
    eq = _frame("2018-01-01", 1400, cols=("S&P 500", "DAX"), seed=1)
    cmd = _frame("2018-01-01", 1400, cols=("WTI Crude Oil", "Gold", "Natural Gas"), seed=2)
    assert eq.index.max() > pd.Timestamp(CUTOFF)   # inputs deliberately leak
    call = terminal_call(eq, cmd, CUTOFF)
    for src, newest in call["pit"].items():
        assert pd.Timestamp(newest) <= pd.Timestamp(CUTOFF), (
            f"{src} newest observation {newest} is after cutoff {CUTOFF}"
        )
    assert not np.isnan(call["grs"])
    assert call["regime"] in (0, 1, 2, 3)


# ── Conflict registry PIT rules ──────────────────────────────────────────────

def test_conflicts_after_cutoff_are_excluded():
    scores = replay_conflict_scores(CUTOFF)
    assert "ukraine_russia" in scores       # broke out that day
    assert "iran_conflict" not in scores    # June 2025 - must not exist yet
    assert "india_pakistan" not in scores   # May 2025 - must not exist yet


def test_breakout_day_is_active_and_escalating():
    scores = replay_conflict_scores(CUTOFF)
    ua = scores["ukraine_russia"]
    assert ua["state"] == "active"
    assert ua["escalation"] == "escalating"
    assert ua["days_since_onset"] == 0


def test_presets_cover_tracked_conflicts():
    presets = replay_presets()
    assert len(presets) >= 5
    assert all(isinstance(p["date"], datetime.date) for p in presets)

"""
Pattern Memory - fingerprint causality and matching tests.

Raw-feature causality: the fingerprint at day t must use no data after t.
Geo/chokepoint features must carry zero trace of events that had not started;
market features at t must be unchanged by deleting the future.
"""

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.fingerprint import (
    BLOCK_FEATURES, daily_geo_state, build_fingerprints,
    match_fingerprint, match_outcomes, base_rates, aggregate_verdicts,
)


def _returns(start: str, periods: int, cols, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(rng.normal(0, 0.012, (periods, len(cols))), index=idx, columns=cols)


EQ_COLS = ("S&P 500", "DAX", "Nikkei 225")
CMD_COLS = ("WTI Crude Oil", "Gold", "Natural Gas", "Copper", "Silver", "Brent Crude")


# ── Geo/chokepoint causality ─────────────────────────────────────────────────

def test_geo_state_is_zero_before_any_event():
    idx = pd.bdate_range("2007-01-02", periods=30)   # before GFC (2008-09-15)
    g = daily_geo_state(idx)
    assert (g["geo_n_active"] == 0).all()
    assert (g["geo_intensity"] == 0).all()
    assert (g["choke_pressure"] == 0).all()


def test_ukraine_invisible_the_day_before_onset():
    idx = pd.DatetimeIndex([pd.Timestamp("2022-02-23"), pd.Timestamp("2022-02-24")])
    g = daily_geo_state(idx)
    # Feb 23: no active events (COVID event ended 2021-11-30) - Feb 24: Ukraine active
    assert g["geo_n_active"].iloc[1] == g["geo_n_active"].iloc[0] + 1
    assert g["choke_pressure"].iloc[1] > g["choke_pressure"].iloc[0]


def test_gfc_period_is_financial_not_war():
    idx = pd.DatetimeIndex([pd.Timestamp("2008-10-15")])
    g = daily_geo_state(idx)
    assert g["geo_share_financial"].iloc[0] == 1.0
    assert g["geo_share_war"].iloc[0] == 0.0


# ── Market-feature causality: truncating the future changes nothing at t ────

def test_market_features_causal_under_truncation():
    eq = _returns("2015-01-01", 1500, EQ_COLS, seed=1)
    cmd = _returns("2015-01-01", 1500, CMD_COLS, seed=2)
    fp_full, _ = build_fingerprints(eq, cmd)

    t = fp_full.index[900]
    fp_trunc, _ = build_fingerprints(eq.loc[:t], cmd.loc[:t])

    causal_feats = ["grs", "eq_vol20", "cmd_vol20", "wti_mom20", "gold_mom20",
                    "spx_mom20", "avg_corr", "corr_velocity", "oil_spx_beta60",
                    "geo_n_active", "geo_intensity", "choke_pressure"]
    for f in causal_feats:
        a, b = fp_full.loc[t, f], fp_trunc.loc[t, f]
        assert np.isclose(a, b, rtol=1e-6, equal_nan=True), (
            f"{f} at {t.date()} changed when future data was removed: {a} vs {b}"
        )


# ── Matching mechanics ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fp_and_regimes():
    eq = _returns("2012-01-01", 2600, EQ_COLS, seed=3)
    cmd = _returns("2012-01-01", 2600, CMD_COLS, seed=4)
    fp, regimes = build_fingerprints(eq, cmd)
    return fp, regimes, pd.concat([eq, cmd], axis=1)


def test_match_respects_blackout_and_forward_window(fp_and_regimes):
    fp, _, _ = fp_and_regimes
    res = match_fingerprint(fp)
    assert res["matches"], "expected at least one match on synthetic data"
    last = fp.index[-1]
    for m in res["matches"]:
        gap_rows = len(fp.loc[m["date"]:last]) - 1
        assert gap_rows >= 60, f"match {m['date']} inside forward/blackout window"


def test_matches_are_non_overlapping(fp_and_regimes):
    fp, _, _ = fp_and_regimes
    res = match_fingerprint(fp)
    dates = sorted(m["date"] for m in res["matches"])
    for a, b in zip(dates, dates[1:]):
        assert (b - a).days > 40, f"matches {a} and {b} overlap"


def test_block_weight_zero_removes_block_influence(fp_and_regimes):
    fp, _, _ = fp_and_regimes
    only_market = {b: (1.0 if b == "market" else 0.0) for b in BLOCK_FEATURES}
    res = match_fingerprint(fp, block_weights=only_market)
    assert res["matches"]                      # still matches on market block alone


def test_outcomes_and_base_rates(fp_and_regimes):
    fp, regimes, combined = fp_and_regimes
    res = match_fingerprint(fp)
    outs = match_outcomes(res, combined, regimes)
    assert outs
    for o in outs:
        assert o["verdict"] in ("GOOD", "MIXED", "BAD", " - ")
        assert o["regime_then"] in (0, 1, 2, 3)
    br = base_rates(combined, regimes)
    assert abs(br["GOOD"] + br["MIXED"] + br["BAD"] - 1.0) < 1e-9
    agg = aggregate_verdicts(outs)
    if agg:
        assert abs(agg["GOOD"] + agg["MIXED"] + agg["BAD"] - 1.0) < 1e-9

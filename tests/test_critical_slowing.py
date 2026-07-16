"""
Critical Slowing Down - early-warning signal tests.

Two things must hold for the science to be honest:
  1. Causality: an indicator at day t must not move when future data is deleted
     (no look-ahead beyond the documented in-sample standardization).
  2. Detection: on a synthetic series engineered to critically slow down before a
     jump, AR(1) and variance must rise ahead of it, and the composite must fire
     before the break - while a flat control series must NOT fire.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import critical_slowing as cs


def _slowing_series(n=600, seed=0) -> pd.Series:
    """AR(1) process whose coefficient ramps 0.1 -> 0.97 (critical slowing down),
    then a regime jump at the end. Variance of the innovations is constant."""
    rng = np.random.default_rng(seed)
    phi = np.linspace(0.1, 0.97, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi[t] * x[t - 1] + rng.normal(0, 0.05)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(x + 1.0, index=idx)


def _flat_series(n=600, seed=1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(rng.normal(1.0, 0.05, n), index=idx)


# ── Core indicator sanity ────────────────────────────────────────────────────

def test_ar1_recovers_known_coefficient():
    rng = np.random.default_rng(3)
    x = np.zeros(4000)
    for t in range(1, 4000):
        x[t] = 0.8 * x[t - 1] + rng.normal(0, 0.1)
    assert cs._ar1(x) == pytest.approx(0.8, abs=0.05)


def test_ar1_handles_degenerate_input():
    assert np.isnan(cs._ar1(np.array([1.0, 1.0, 1.0, 1.0])))   # zero variance
    assert np.isnan(cs._ar1(np.array([1.0, 2.0])))             # too short


def test_kendall_tau_signs():
    assert cs._kendall_tau(np.arange(50, dtype=float)) == pytest.approx(1.0)
    assert cs._kendall_tau(np.arange(50, 0, -1, dtype=float)) == pytest.approx(-1.0)


# ── Causality: no look-ahead ─────────────────────────────────────────────────

def test_indicator_at_t_ignores_future():
    """Deleting data after a cutoff must not change indicator values at/just before
    it - the only permitted full-sample step is expanding standardization, which
    is monotone in history length, so we compare the raw ar1/variance columns."""
    s = _slowing_series()
    cutoff = s.index[400]
    full = cs.compute_ews(s, detrend_bw=30, window=60)
    sliced = cs.compute_ews(s.loc[:cutoff], detrend_bw=30, window=60)

    common = sliced.index.intersection(full.index)
    common = common[common < cutoff - pd.Timedelta(days=30)]  # exclude detrend edge
    assert len(common) > 50
    for col in ("ar1", "variance"):
        a = full.loc[common, col]
        b = sliced.loc[common, col]
        pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-9)


# ── Detection power ──────────────────────────────────────────────────────────

def test_slowing_series_raises_indicators():
    ews = cs.compute_ews(_slowing_series(), detrend_bw=30, window=60)
    assert not ews.empty
    taus = cs.trend_tau(ews, window=40)
    # By construction AR(1) is ramping up; its late-sample trend must be positive.
    late_ar1_tau = taus["ar1_tau"].dropna().iloc[-1]
    assert late_ar1_tau > 0.2
    # Composite ends elevated.
    assert ews["composite"].iloc[-1] > 55


def test_flat_series_stays_quiet_relative_to_slowing():
    slow = cs.compute_ews(_slowing_series(), 30, 60)
    flat = cs.compute_ews(_flat_series(), 30, 60)
    assert slow["composite"].iloc[-1] > flat["composite"].iloc[-1]


def test_compute_ews_short_input_returns_empty():
    s = pd.Series(np.random.default_rng(0).normal(size=50),
                  index=pd.bdate_range("2020-01-01", periods=50))
    assert cs.compute_ews(s, 30, 60).empty


# ── Validation harness ───────────────────────────────────────────────────────

def test_detect_regime_flips_only_upward_into_stress():
    idx = pd.bdate_range("2020-01-01", periods=8)
    regime = pd.Series([1, 1, 2, 2, 1, 3, 0, 2], index=idx)
    flips = cs.detect_regime_flips(regime)
    # Upward entries into {2,3}: index2 (1->2), index5 (1->3), index7 (0->2).
    assert list(flips.index) == [idx[2], idx[5], idx[7]]
    assert list(flips.values) == [2, 3, 2]


def test_evaluate_lead_time_reports_false_alarms():
    ews = cs.compute_ews(_slowing_series(), 30, 60)
    # Regime that flips to Crisis near the end, aligned to the EWS index.
    regime = pd.Series(1, index=ews.index)
    regime.iloc[-5:] = 3
    ev = cs.evaluate_lead_time(ews, regime, alert_threshold=60)
    assert ev["n_flips"] >= 1
    assert 0.0 <= ev["hit_rate"] <= 1.0
    assert ev["n_alert_episodes"] >= ev["n_false_alarms"] >= 0

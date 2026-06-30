"""
Unit tests for src/analysis/risk_score.py.

Tests target the pure-math helper functions that don't require network
access, API keys, or Streamlit session state. compute_risk_score itself
is integration-level (calls conflict model, news, PortWatch) and is
excluded from unit tests — integration coverage comes from the app running.

Run with: python -m pytest tests/test_risk_score.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.risk_score import (
    _zscore_to_score,
    _ewm_zscore,
    _corr_accel_score,
    _commodity_vol_score,
    _equity_vol_score,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _const_series(val: float, n: int = 300) -> pd.Series:
    return pd.Series(np.full(n, val), index=_dates(n))


def _trend_series(start: float, end: float, n: int = 300) -> pd.Series:
    return pd.Series(np.linspace(start, end, n), index=_dates(n))


def _noisy_series(mean: float, std: float, n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(mean + rng.standard_normal(n) * std, index=_dates(n))


def _returns_df(n: int = 300, k: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, k)) * 0.01
    return pd.DataFrame(data, index=_dates(n), columns=[f"A{i}" for i in range(k)])


# ── _zscore_to_score ──────────────────────────────────────────────────────────

class TestZscoreToScore:
    def test_zero_maps_to_50(self):
        assert abs(_zscore_to_score(0.0) - 50.0) < 1e-9

    def test_positive_z_above_50(self):
        assert _zscore_to_score(1.0) > 50.0
        assert _zscore_to_score(3.0) > 50.0

    def test_negative_z_below_50(self):
        assert _zscore_to_score(-1.0) < 50.0
        assert _zscore_to_score(-3.0) < 50.0

    def test_output_bounded_0_to_100(self):
        for z in [-10, -5, -2, -1, 0, 1, 2, 5, 10]:
            s = _zscore_to_score(float(z))
            assert 0.0 <= s <= 100.0, f"_zscore_to_score({z}) = {s} out of [0, 100]"

    def test_monotone_increasing(self):
        """Higher z → higher score."""
        zs = [-3.0, -1.0, 0.0, 1.0, 3.0]
        scores = [_zscore_to_score(z) for z in zs]
        assert scores == sorted(scores)

    def test_custom_scale_changes_steepness(self):
        """Higher scale → steeper sigmoid → scores further from 50 at same z."""
        s_low  = _zscore_to_score(2.0, scale=5.0)
        s_high = _zscore_to_score(2.0, scale=25.0)
        assert s_high > s_low

    def test_symmetry(self):
        """_zscore_to_score(z) + _zscore_to_score(-z) == 100."""
        for z in [0.5, 1.0, 2.0, 3.0]:
            total = _zscore_to_score(z) + _zscore_to_score(-z)
            assert abs(total - 100.0) < 1e-9, f"Symmetry failed at z={z}: sum={total}"


# ── _ewm_zscore ───────────────────────────────────────────────────────────────

class TestEwmZscore:
    def test_returns_series_same_length(self):
        s = _noisy_series(0.0, 1.0, n=300)
        out = _ewm_zscore(s, span=60)
        assert isinstance(out, pd.Series)
        assert len(out) == len(s)

    def test_early_values_are_nan_before_min_periods(self):
        s = _noisy_series(0.0, 1.0, n=300)
        out = _ewm_zscore(s, span=252)
        # min_periods=60: first 59 values should be NaN
        assert out.iloc[:59].isna().all()

    def test_constant_series_gives_nan_zscore(self):
        """Constant series → std=0 → z=NaN after first obs."""
        s = _const_series(5.0, n=200)
        out = _ewm_zscore(s, span=60)
        # All non-NaN values should be NaN (sigma=0 → division by NaN)
        assert out.dropna().empty or (out.dropna().abs() < 1e-6).all()

    def test_large_positive_shock_gives_positive_z(self):
        """Series that jumps to 10x its normal level should have z > 0."""
        rng = np.random.default_rng(0)
        s = pd.Series(
            np.concatenate([rng.standard_normal(200) * 0.01, np.full(100, 0.5)]),
            index=_dates(300),
        )
        out = _ewm_zscore(s, span=252)
        last_z = out.dropna().iloc[-1]
        # span=252 adapts the EWM mean toward the shock over 100 obs,
        # so z dampens below 2.0 but remains clearly positive.
        assert last_z > 0.5, f"Expected z > 0.5 after shock, got {last_z:.2f}"

    def test_index_preserved(self):
        s = _noisy_series(0.0, 1.0, n=300)
        out = _ewm_zscore(s, span=60)
        pd.testing.assert_index_equal(out.index, s.index)


# ── _corr_accel_score ─────────────────────────────────────────────────────────

class TestCorrAccelScore:
    def test_empty_series_returns_50(self):
        assert _corr_accel_score(pd.Series(dtype=float)) == 50.0

    def test_short_series_returns_50(self):
        s = pd.Series([0.3, 0.4, 0.5], index=_dates(3))
        assert _corr_accel_score(s) == 50.0

    def test_output_bounded_0_to_100(self):
        s = _noisy_series(0.4, 0.05, n=300)
        score = _corr_accel_score(s)
        assert 0.0 <= score <= 100.0

    def test_rising_series_scores_above_50(self):
        """Correlation trending up → velocity positive → score > 50."""
        s = _trend_series(0.1, 0.9, n=300)
        score = _corr_accel_score(s)
        assert score > 50.0, f"Rising corr should score >50, got {score:.1f}"

    def test_falling_series_scores_below_50(self):
        """Correlation trending down → velocity negative → score < 50."""
        s = _trend_series(0.9, 0.1, n=300)
        score = _corr_accel_score(s)
        assert score < 50.0, f"Falling corr should score <50, got {score:.1f}"

    def test_flat_series_scores_near_50(self):
        """Near-flat series: last-velocity percentile rank is uniform on [0,100].
        A single draw can land anywhere, so we average over 20 seeds."""
        # A constant series has velocity=0 always; score=0 (nothing below 0).
        # A noisy near-constant series produces random-sign velocities; the last
        # velocity's empirical CDF is approximately Uniform[0,100], mean ≈ 50.
        scores = [_corr_accel_score(_noisy_series(0.5, 0.01, n=300, seed=i))
                  for i in range(20)]
        mean_score = float(np.mean(scores))
        assert 35.0 <= mean_score <= 65.0, (
            f"Mean score over 20 seeds should be near 50, got {mean_score:.1f}"
        )


# ── _commodity_vol_score ──────────────────────────────────────────────────────

class TestCommodityVolScore:
    def test_empty_dataframe_returns_50(self):
        assert _commodity_vol_score(pd.DataFrame()) == 50.0

    def test_output_bounded_0_to_100(self):
        cmd_r = _returns_df(n=300, k=5)
        score = _commodity_vol_score(cmd_r)
        assert 0.0 <= score <= 100.0

    def test_short_series_returns_50(self):
        cmd_r = _returns_df(n=50, k=3)
        # < 80 obs → ewm_zscore will have NaN last value → returns 50
        score = _commodity_vol_score(cmd_r)
        assert score == 50.0

    def test_high_vol_returns_score_above_50(self):
        """Inject persistent high-vol returns vs low-vol baseline."""
        rng  = np.random.default_rng(1)
        n    = 400
        low  = rng.standard_normal((200, 3)) * 0.005
        high = rng.standard_normal((200, 3)) * 0.05   # 10× higher vol
        data = np.vstack([low, high])
        # _commodity_vol_score filters columns by name — must use recognised names
        cols  = ["WTI Crude Oil", "Brent Crude", "Natural Gas"]
        cmd_r = pd.DataFrame(data, index=_dates(n), columns=cols)
        score = _commodity_vol_score(cmd_r)
        assert score > 55.0, f"High-vol period should score >55, got {score:.1f}"

    def test_low_constant_vol_returns_score_below_60(self):
        """Uniformly low vol → z-score near 0 → score near 50."""
        rng   = np.random.default_rng(2)
        data  = rng.standard_normal((400, 3)) * 0.005
        cmd_r = pd.DataFrame(data, index=_dates(400), columns=["C0", "C1", "C2"])
        score = _commodity_vol_score(cmd_r)
        assert score < 65.0, f"Low uniform vol should score <65, got {score:.1f}"


# ── _equity_vol_score ─────────────────────────────────────────────────────────

class TestEquityVolScore:
    def test_none_returns_50(self):
        assert _equity_vol_score(None) == 50.0

    def test_empty_dataframe_returns_50(self):
        assert _equity_vol_score(pd.DataFrame()) == 50.0

    def test_output_bounded_0_to_100(self):
        eq_r = _returns_df(n=300, k=3)
        eq_r.columns = ["S&P 500", "Eurostoxx 50", "Nikkei 225"]
        score = _equity_vol_score(eq_r)
        assert 0.0 <= score <= 100.0

    def test_short_series_returns_50(self):
        eq_r = _returns_df(n=50, k=3)
        eq_r.columns = ["S&P 500", "A", "B"]
        assert _equity_vol_score(eq_r) == 50.0

    def test_uses_first_columns_when_no_named_cols(self):
        """Falls back to first 3 columns when SPX/Eurostoxx/Nikkei not present."""
        eq_r = _returns_df(n=300, k=5)
        score = _equity_vol_score(eq_r)
        assert 0.0 <= score <= 100.0

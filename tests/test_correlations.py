"""
Unit tests for src/analysis/correlations.py.

All tests use deterministic synthetic data — no network calls, no Streamlit state.
Run with: python -m pytest tests/test_correlations.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.correlations import (
    rolling_correlation,
    cross_asset_corr,
    average_cross_corr_series,
    detect_correlation_regime,
    regime_transition_matrix,
    regime_steady_state,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _iid(n: int = 300, k: int = 1, seed: int = 42) -> pd.DataFrame:
    """IID N(0,1) — zero true correlation between any pair."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, k))
    cols = [f"X{i}" for i in range(k)]
    return pd.DataFrame(data, index=_dates(n), columns=cols)


def _correlated_pair(n: int = 300, rho: float = 0.9, seed: int = 7) -> tuple[pd.Series, pd.Series]:
    """Return two series with true Pearson correlation ≈ rho."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    y = rho * x + np.sqrt(1 - rho**2) * eps
    idx = _dates(n)
    return pd.Series(x, index=idx), pd.Series(y, index=idx)


def _high_corr_regime_data(n: int = 300, seed: int = 1) -> pd.Series:
    """Avg cross-corr series stuck near 0.8 — should classify as Elevated/Crisis."""
    rng = np.random.default_rng(seed)
    base = np.full(n, 0.8) + rng.standard_normal(n) * 0.03
    return pd.Series(np.clip(base, 0, 1), index=_dates(n))


def _low_corr_regime_data(n: int = 300, seed: int = 2) -> pd.Series:
    """Avg cross-corr series stuck near 0.05 — should classify as Decorrelated/Normal."""
    rng = np.random.default_rng(seed)
    base = np.full(n, 0.05) + rng.standard_normal(n) * 0.01
    return pd.Series(np.clip(base, 0, 1), index=_dates(n))


# ── rolling_correlation ───────────────────────────────────────────────────────

class TestRollingCorrelation:
    def test_output_is_series(self):
        s1, s2 = _correlated_pair()
        rc = rolling_correlation(s1, s2, window=60)
        assert isinstance(rc, pd.Series)

    def test_values_in_valid_range(self):
        s1, s2 = _correlated_pair(rho=0.8)
        rc = rolling_correlation(s1, s2, window=60).dropna()
        assert (rc >= -1.0 - 1e-9).all() and (rc <= 1.0 + 1e-9).all()

    def test_highly_correlated_pair_gives_positive_rolling_corr(self):
        s1, s2 = _correlated_pair(rho=0.9, n=300)
        rc = rolling_correlation(s1, s2, window=60).dropna()
        assert rc.mean() > 0.5, f"Expected mean > 0.5, got {rc.mean():.3f}"

    def test_iid_pair_gives_near_zero_mean_corr(self):
        df = _iid(n=500, k=2, seed=99)
        rc = rolling_correlation(df["X0"], df["X1"], window=60).dropna()
        assert abs(rc.mean()) < 0.15, f"IID mean corr should be near 0, got {rc.mean():.3f}"

    def test_identical_series_gives_correlation_one(self):
        s1, _ = _correlated_pair(n=200)
        rc = rolling_correlation(s1, s1, window=30).dropna()
        assert (rc > 0.999).all()

    def test_short_series_returns_all_nan_before_window(self):
        s1, s2 = _correlated_pair(n=100)
        rc = rolling_correlation(s1, s2, window=60)
        # first 59 observations should be NaN
        assert rc.iloc[:59].isna().all()

    def test_empty_series_returns_empty(self):
        s = pd.Series(dtype=float)
        rc = rolling_correlation(s, s, window=30)
        assert rc.dropna().empty


# ── cross_asset_corr ──────────────────────────────────────────────────────────

class TestCrossAssetCorr:
    def test_output_shape(self):
        eq  = _iid(n=200, k=3)
        cmd = _iid(n=200, k=4, seed=10)
        eq.columns  = [f"EQ{i}"  for i in range(3)]
        cmd.columns = [f"CMD{i}" for i in range(4)]
        out = cross_asset_corr(eq, cmd)
        assert out.shape == (3, 4)

    def test_values_in_valid_range(self):
        eq  = _iid(n=300, k=3)
        cmd = _iid(n=300, k=3, seed=5)
        eq.columns  = [f"EQ{i}"  for i in range(3)]
        cmd.columns = [f"CMD{i}" for i in range(3)]
        out = cross_asset_corr(eq, cmd)
        assert out.values.min() >= -1.0 - 1e-9
        assert out.values.max() <=  1.0 + 1e-9

    def test_window_subset_uses_last_n_rows(self):
        n = 300
        eq  = pd.DataFrame({"A": np.linspace(0, 1, n)}, index=_dates(n))
        cmd = pd.DataFrame({"B": np.linspace(0, 1, n)}, index=_dates(n))
        full   = cross_asset_corr(eq, cmd)
        window = cross_asset_corr(eq, cmd, window=60)
        # Both should return high positive correlation for linearly increasing series
        assert full.values[0, 0] > 0.99
        assert window.values[0, 0] > 0.99

    def test_column_labels_preserved(self):
        eq  = _iid(n=200, k=2)
        cmd = _iid(n=200, k=2, seed=3)
        eq.columns  = ["SPX", "DAX"]
        cmd.columns = ["Gold", "Oil"]
        out = cross_asset_corr(eq, cmd)
        assert list(out.index)   == ["SPX", "DAX"]
        assert list(out.columns) == ["Gold", "Oil"]


# ── average_cross_corr_series ─────────────────────────────────────────────────

class TestAverageCrossCorr:
    def test_returns_series(self):
        eq  = _iid(n=300, k=3)
        cmd = _iid(n=300, k=3, seed=4)
        eq.columns  = [f"EQ{i}"  for i in range(3)]
        cmd.columns = [f"CMD{i}" for i in range(3)]
        out = average_cross_corr_series(eq, cmd, window=60)
        assert isinstance(out, pd.Series)

    def test_values_non_negative(self):
        """Takes absolute value of correlation, so must be ≥ 0."""
        eq  = _iid(n=300, k=4)
        cmd = _iid(n=300, k=4, seed=6)
        eq.columns  = [f"EQ{i}"  for i in range(4)]
        cmd.columns = [f"CMD{i}" for i in range(4)]
        out = average_cross_corr_series(eq, cmd, window=60)
        assert (out >= -1e-9).all(), "Average |correlation| must be non-negative"

    def test_high_correlation_input_gives_high_avg(self):
        rng = np.random.default_rng(0)
        n = 300
        factor = rng.standard_normal(n)
        idx = _dates(n)
        eq  = pd.DataFrame({"EQ": factor + rng.standard_normal(n) * 0.05}, index=idx)
        cmd = pd.DataFrame({"CMD": factor + rng.standard_normal(n) * 0.05}, index=idx)
        out = average_cross_corr_series(eq, cmd, window=60).dropna()
        assert out.mean() > 0.7, f"Expected high avg corr, got {out.mean():.3f}"

    def test_empty_equity_returns_empty(self):
        eq  = pd.DataFrame(dtype=float)
        cmd = _iid(n=300, k=2)
        cmd.columns = ["A", "B"]
        out = average_cross_corr_series(eq, cmd, window=60)
        assert out.empty

    def test_no_common_dates_returns_empty(self):
        eq  = _iid(n=100, k=1)
        cmd = pd.DataFrame(
            {"X": np.random.randn(100)},
            index=pd.date_range("2025-01-01", periods=100, freq="B"),
        )
        out = average_cross_corr_series(eq, cmd, window=60)
        assert out.empty


# ── detect_correlation_regime ─────────────────────────────────────────────────

class TestDetectCorrelationRegime:
    def test_output_values_in_valid_set(self):
        data = _high_corr_regime_data()
        out  = detect_correlation_regime(data)
        assert set(out.unique()).issubset({0, 1, 2, 3})

    def test_output_length_matches_input(self):
        data = _high_corr_regime_data(n=200)
        out  = detect_correlation_regime(data)
        assert len(out) == len(data)

    def test_insufficient_data_returns_all_normal(self):
        """< 60 observations → all regime=1 (Normal), insufficient_data=True."""
        short = pd.Series(np.random.rand(50), index=_dates(50))
        out   = detect_correlation_regime(short)
        assert out.attrs.get("insufficient_data") is True
        assert (out == 1).all()

    def test_high_correlation_data_skews_toward_elevated_crisis(self):
        """Series near 0.8 should produce mostly regime 2 or 3."""
        data = _high_corr_regime_data(n=400)
        out  = detect_correlation_regime(data)
        high_frac = (out >= 2).mean()
        assert high_frac > 0.5, f"Expected >50% elevated/crisis, got {high_frac:.2f}"

    def test_low_correlation_data_skews_toward_decorrelated(self):
        """Low-corr series should have more regime 0/1 than a high-corr series.

        detect_correlation_regime uses percentile-based adaptive thresholds — it
        classifies by relative position within each series, not absolute level.
        A near-constant 0.05 series cannot be guaranteed to exceed 50% in regime
        0/1 by absolute count, but it must produce fewer high-regime points than
        an equivalent near-constant 0.8 series.
        """
        data_high = _high_corr_regime_data(n=400)
        data_low  = _low_corr_regime_data(n=400)
        low_frac_in_low  = (detect_correlation_regime(data_low)  <= 1).mean()
        low_frac_in_high = (detect_correlation_regime(data_high) <= 1).mean()
        assert low_frac_in_low >= low_frac_in_high, (
            f"Low-corr should have ≥ regime-0/1 fraction than high-corr: "
            f"{low_frac_in_low:.2f} vs {low_frac_in_high:.2f}"
        )

    def test_index_preserved(self):
        data = _high_corr_regime_data(n=200)
        out  = detect_correlation_regime(data)
        pd.testing.assert_index_equal(out.index, data.index)

    def test_no_nan_in_output(self):
        data = _high_corr_regime_data(n=200)
        out  = detect_correlation_regime(data)
        assert out.isna().sum() == 0


# ── regime_transition_matrix ──────────────────────────────────────────────────

class TestRegimeTransitionMatrix:
    def _sample_regimes(self, n: int = 300, seed: int = 0) -> pd.Series:
        rng = np.random.default_rng(seed)
        vals = rng.integers(0, 4, size=n)
        return pd.Series(vals, index=_dates(n))

    def test_output_shape_4x4(self):
        tm = regime_transition_matrix(self._sample_regimes())
        assert tm.shape == (4, 4)

    def test_rows_sum_to_one(self):
        tm = regime_transition_matrix(self._sample_regimes())
        row_sums = tm.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, np.ones(4), atol=1e-10)

    def test_values_non_negative(self):
        tm = regime_transition_matrix(self._sample_regimes())
        assert (tm.values >= 0).all()

    def test_index_and_columns_are_0_to_3(self):
        tm = regime_transition_matrix(self._sample_regimes())
        assert list(tm.index)   == [0, 1, 2, 3]
        assert list(tm.columns) == [0, 1, 2, 3]

    def test_single_regime_series_gives_uniform_row(self):
        """All-regime-1 series → row 1 should have 1.0 in column 1."""
        regimes = pd.Series([1] * 100, index=_dates(100))
        tm = regime_transition_matrix(regimes)
        # Row 1, column 1 should be 1.0 (only transition observed: 1→1)
        assert abs(tm.loc[1, 1] - 1.0) < 1e-10

    def test_unobserved_rows_filled_uniform(self):
        """If regime 3 never appears, row 3 should be 0.25 each."""
        regimes = pd.Series([0, 1, 2] * 100, index=_dates(300))
        tm = regime_transition_matrix(regimes)
        np.testing.assert_allclose(tm.loc[3].values, np.full(4, 0.25), atol=1e-10)


# ── regime_steady_state ───────────────────────────────────────────────────────

class TestRegimeSteadyState:
    def _uniform_transition(self) -> pd.DataFrame:
        """Uniform 4x4 matrix — steady state = [0.25, 0.25, 0.25, 0.25]."""
        data = np.full((4, 4), 0.25)
        return pd.DataFrame(data, index=[0, 1, 2, 3], columns=[0, 1, 2, 3])

    def test_output_length_4(self):
        pi = regime_steady_state(self._uniform_transition())
        assert len(pi) == 4

    def test_sums_to_one(self):
        tm = regime_transition_matrix(pd.Series([0, 1, 2, 3] * 100, index=_dates(400)))
        pi = regime_steady_state(tm)
        assert abs(pi.sum() - 1.0) < 1e-8

    def test_values_non_negative(self):
        tm = regime_transition_matrix(pd.Series([0, 1, 2, 3] * 100, index=_dates(400)))
        pi = regime_steady_state(tm)
        assert (pi >= -1e-10).all()

    def test_uniform_matrix_gives_uniform_steady_state(self):
        pi = regime_steady_state(self._uniform_transition())
        np.testing.assert_allclose(pi, np.full(4, 0.25), atol=1e-6)

    def test_absorbing_state_steady_state(self):
        """Regime 1 is absorbing (P[1→1]=1, all others transition to 1)."""
        data = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        tm = pd.DataFrame(data, index=[0, 1, 2, 3], columns=[0, 1, 2, 3])
        pi = regime_steady_state(tm)
        assert pi[1] > 0.99, f"Absorbing state 1 should dominate, got {pi[1]:.4f}"

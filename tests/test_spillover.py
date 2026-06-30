"""
Unit tests for src/analysis/spillover.py.

Uses deterministic synthetic AR(1) data so results are reproducible
without network access.  Run with:  python -m pytest tests/test_spillover.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.spillover import (
    check_stationarity,
    diebold_yilmaz,
    granger_test,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_ar1(n: int = 300, k: int = 4, phi: float = 0.3, seed: int = 42) -> pd.DataFrame:
    """
    Stationary AR(1) panel: x_t = phi * x_{t-1} + e_t.
    phi < 1 guarantees stationarity; cross-series independence by construction.
    """
    rng = np.random.default_rng(seed)
    e   = rng.standard_normal((n, k))
    data = np.zeros((n, k))
    data[0] = e[0]
    for t in range(1, n):
        data[t] = phi * data[t - 1] + e[t]
    idx  = pd.date_range("2020-01-01", periods=n, freq="B")
    cols = [f"A{i}" for i in range(k)]
    return pd.DataFrame(data, index=idx, columns=cols)


def _make_random_walk(n: int = 300, k: int = 2, seed: int = 7) -> pd.DataFrame:
    """I(1) random walk: x_t = x_{t-1} + e_t (unit root, non-stationary)."""
    rng  = np.random.default_rng(seed)
    data = np.cumsum(rng.standard_normal((n, k)), axis=0)
    idx  = pd.date_range("2020-01-01", periods=n, freq="B")
    cols = [f"X{i}" for i in range(k)]
    return pd.DataFrame(data, index=idx, columns=cols)


# ── check_stationarity ───────────────────────────────────────────────────────

class TestCheckStationarity:
    def test_stationary_ar1_returns_empty_nonstationary(self):
        df = _make_ar1()
        result = check_stationarity(df)
        assert isinstance(result["non_stationary"], list)
        assert len(result["non_stationary"]) == 0, (
            f"AR(1) with phi=0.3 should be stationary; got non_stationary={result['non_stationary']}"
        )

    def test_random_walk_flagged_as_nonstationary(self):
        # ADF has limited power on short series; test that at least one I(1)
        # column is detected rather than requiring all (avoids flaky assertions
        # due to finite-sample ADF variability).
        df = _make_random_walk(n=600)  # longer series → more ADF power
        result = check_stationarity(df)
        assert len(result["non_stationary"]) >= 1, (
            "At least one random-walk column should be flagged as non-stationary"
        )

    def test_stationary_keys_present(self):
        df = _make_ar1(k=2)
        result = check_stationarity(df)
        assert "stationary" in result
        assert "non_stationary" in result
        assert set(result["stationary"].keys()) == set(df.columns)

    def test_too_short_series_does_not_raise(self):
        df = pd.DataFrame({"A": [0.1, -0.1, 0.2]})
        result = check_stationarity(df)
        assert "A" in result["stationary"]

    def test_all_nan_column_does_not_raise(self):
        df = pd.DataFrame({"A": [float("nan")] * 50, "B": np.random.randn(50)})
        result = check_stationarity(df)
        assert "A" in result["stationary"]


# ── diebold_yilmaz ──────────────────────────────────────────────────────────

class TestDieboldYilmaz:
    def test_required_keys_present(self):
        df = _make_ar1()
        result = diebold_yilmaz(df)
        required = {
            "spillover_table", "from_spillover", "to_spillover",
            "net_spillover", "total_spillover", "top_transmitter",
            "top_receiver", "direction_label", "assets_used",
            "non_stationary_assets",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_net_equals_to_minus_from(self):
        df = _make_ar1()
        result = diebold_yilmaz(df)
        if result["spillover_table"].empty:
            pytest.skip("VAR did not converge on this dataset")
        net  = result["net_spillover"]
        diff = result["to_spillover"] - result["from_spillover"]
        # Tolerance is 0.02 because FROM, TO, NET are each independently
        # rounded to 2 decimal places, so round(TO) - round(FROM) can
        # differ from round(NET) by up to ±0.01 per element.
        pd.testing.assert_series_equal(net, diff, check_names=False, atol=0.02)

    def test_total_spillover_in_valid_range(self):
        df = _make_ar1()
        result = diebold_yilmaz(df)
        if result["spillover_table"].empty:
            pytest.skip("VAR did not converge on this dataset")
        assert 0.0 <= result["total_spillover"] <= 100.0

    def test_assets_used_matches_input_columns(self):
        df = _make_ar1(k=4)
        result = diebold_yilmaz(df)
        if result["spillover_table"].empty:
            pytest.skip("VAR did not converge on this dataset")
        assert set(result["assets_used"]) == set(df.columns)

    def test_top_n_selection_reduces_assets(self):
        df = _make_ar1(k=4)
        result = diebold_yilmaz(df, top_n=2)
        if result["spillover_table"].empty:
            pytest.skip("VAR did not converge on this dataset")
        assert len(result["assets_used"]) == 2

    def test_non_stationary_assets_is_list(self):
        df = _make_ar1()
        result = diebold_yilmaz(df)
        assert isinstance(result["non_stationary_assets"], list)

    def test_non_stationary_assets_empty_for_stationary_input(self):
        df = _make_ar1()
        result = diebold_yilmaz(df)
        assert result["non_stationary_assets"] == [], (
            "AR(1) returns should produce no stationarity warnings"
        )

    def test_insufficient_data_returns_empty_dict(self):
        df = _make_ar1(n=5)
        result = diebold_yilmaz(df)
        assert result["spillover_table"].empty
        assert result["total_spillover"] != result["total_spillover"]  # NaN check

    def test_empty_dataframe_returns_empty_dict(self):
        result = diebold_yilmaz(pd.DataFrame())
        assert result["spillover_table"].empty


# ── granger_test ─────────────────────────────────────────────────────────────

class TestGrangerTest:
    def test_returns_required_keys(self):
        df = _make_ar1(k=2)
        result = granger_test(df.iloc[:, 0], df.iloc[:, 1])
        assert "min_p" in result
        assert "significant" in result
        assert "bic_lag" in result

    def test_significant_is_bool(self):
        df = _make_ar1(k=2)
        result = granger_test(df.iloc[:, 0], df.iloc[:, 1])
        # Accept both Python bool and numpy bool_ (granger_test uses numpy comparison)
        assert isinstance(result["significant"], (bool, np.bool_))

    def test_too_short_returns_safely(self):
        s = pd.Series(np.random.randn(10))
        result = granger_test(s, s)
        assert result["significant"] is False
        assert np.isnan(result["min_p"])

"""
Step 2 of 4 — weight allocator.

Pins: the zero-weight lock survives allocation; sizing is DSR-aware and never
raw-Sharpe; low-DSR and stale trades size to zero; inverse-vol and liquidity
scale as declared; weights land only on eligible trades and sum to 1.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.trade_allocator import (
    allocate_weights, build_allocation_inputs, trade_leg_vol, MAX_SINGLE_WEIGHT,
)


def _t(name, eligible=True, assets=("Gold", "S&P 500"), regime=(2, 3)):
    return {"name": name, "is_eligible": eligible,
            "eligibility_reason": "eligible" if eligible else "missing legs: X",
            "assets": list(assets), "direction": ["Long", "Short"],
            "regime": list(regime)}


def _m(conviction=0.7, dsr=0.8, vol=0.15, liquidity=1.0, sharpe_raw=1.0, n=10):
    return {"conviction": conviction, "dsr": dsr,
            "dsr_factor": float(np.clip((dsr - 0.5) / 0.5, 0, 1)),
            "vol": vol, "liquidity": liquidity,
            "sharpe_raw": sharpe_raw, "n_trades": n}


# ── The lock survives allocation ─────────────────────────────────────────────

def test_ineligible_trade_gets_zero_even_with_perfect_metrics():
    trades = [_t("A"), _t("LOCKED", eligible=False)]
    metrics = {"A": _m(), "LOCKED": _m(conviction=1.0, dsr=0.99, vol=0.05)}
    allocate_weights(trades, metrics)
    locked = next(t for t in trades if t["name"] == "LOCKED")
    assert locked["alloc_weight"] == 0.0
    assert locked["alloc_detail"]["locked"] is True


def test_weights_land_only_on_eligible_and_sum_to_one():
    trades = [_t("A"), _t("B"), _t("C", eligible=False), _t("D")]
    metrics = {n: _m() for n in ("A", "B", "C", "D")}
    allocate_weights(trades, metrics)
    elig_sum = sum(t["alloc_weight"] for t in trades if t["is_eligible"])
    inelig = [t["alloc_weight"] for t in trades if not t["is_eligible"]]
    assert abs(elig_sum - 1.0) < 1e-5   # 6dp rounding on stamps
    assert all(w == 0.0 for w in inelig)


# ── DSR-aware, never raw Sharpe ──────────────────────────────────────────────

def test_dsr_beats_raw_sharpe():
    # X: spectacular raw Sharpe but DSR below the luck benchmark.
    # Y: modest Sharpe with genuine deflated edge. Y must out-weight X.
    trades = [_t("X"), _t("Y")]
    metrics = {"X": _m(dsr=0.45, sharpe_raw=3.5),
               "Y": _m(dsr=0.85, sharpe_raw=0.9)}
    allocate_weights(trades, metrics)
    wx = next(t["alloc_weight"] for t in trades if t["name"] == "X")
    wy = next(t["alloc_weight"] for t in trades if t["name"] == "Y")
    assert wx == 0.0                      # DSR ≤ 0.5 sizes to zero
    assert wy > 0.0


def test_low_dsr_sizes_to_zero():
    trades = [_t("A"), _t("B")]
    metrics = {"A": _m(dsr=0.50), "B": _m(dsr=0.75)}
    allocate_weights(trades, metrics)
    assert next(t["alloc_weight"] for t in trades if t["name"] == "A") == 0.0


# ── Factor mechanics ─────────────────────────────────────────────────────────

def test_inverse_vol_halves_weight_at_double_vol():
    trades = [_t("LO"), _t("HI")]
    metrics = {"LO": _m(vol=0.10), "HI": _m(vol=0.20)}
    allocate_weights(trades, metrics, cap=1.0)   # cap off: testing the factor
    w_lo = next(t["alloc_weight"] for t in trades if t["name"] == "LO")
    w_hi = next(t["alloc_weight"] for t in trades if t["name"] == "HI")
    assert w_lo / w_hi == pytest.approx(2.0, rel=1e-4)  # 6dp stamps


def test_liquidity_scales_weight():
    trades = [_t("DEEP"), _t("THIN")]
    metrics = {"DEEP": _m(liquidity=1.0), "THIN": _m(liquidity=0.5)}
    allocate_weights(trades, metrics, cap=1.0)   # cap off: testing the factor
    w_d = next(t["alloc_weight"] for t in trades if t["name"] == "DEEP")
    w_t = next(t["alloc_weight"] for t in trades if t["name"] == "THIN")
    assert w_d / w_t == pytest.approx(2.0, rel=1e-4)   # 6dp stamps


def test_cap_redistributes_and_preserves_total():
    trades = [_t("BIG"), _t("S1"), _t("S2")]
    metrics = {"BIG": _m(conviction=1.0, dsr=0.99, vol=0.02),
               "S1": _m(vol=0.30), "S2": _m(vol=0.30)}
    allocate_weights(trades, metrics)
    ws = {t["name"]: t["alloc_weight"] for t in trades}
    assert ws["BIG"] <= MAX_SINGLE_WEIGHT + 1e-6
    assert abs(sum(ws.values()) - 1.0) < 1e-6


def test_cap_binding_everywhere_leaves_cash_remainder():
    # Concentration cap is a hard risk limit: with only two eligible trades,
    # 2 × 35% = 70% gross and the remainder is cash — never force-fill to 100%.
    trades = [_t("A"), _t("B")]
    metrics = {"A": _m(), "B": _m()}
    allocate_weights(trades, metrics)
    ws = [t["alloc_weight"] for t in trades]
    assert all(abs(w - MAX_SINGLE_WEIGHT) < 1e-6 for w in ws)
    assert sum(ws) == pytest.approx(2 * MAX_SINGLE_WEIGHT, abs=1e-6)


def test_all_zero_factors_yield_all_zero_no_nan():
    trades = [_t("A"), _t("B")]
    metrics = {"A": _m(dsr=0.2), "B": _m(dsr=0.3)}
    allocate_weights(trades, metrics)
    for t in trades:
        assert t["alloc_weight"] == 0.0
        assert not np.isnan(t["alloc_weight"])


# ── Input builder on synthetic data ──────────────────────────────────────────

def test_builder_skips_ineligible_and_stale_gets_zero_dsr():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=900)
    all_r = pd.DataFrame(rng.normal(0, 0.01, (900, 2)), index=idx,
                         columns=["Gold", "S&P 500"])
    regimes = pd.Series(1, index=idx)
    regimes.iloc[500:520] = 2             # exactly ONE regime-2 episode → 1 trade
    trades = [_t("ELIG", regime=(2,)), _t("LOCKED", eligible=False)]
    thesis = {"ELIG": {"stage_passed": True, "confirmation_score": 0.6}}
    m = build_allocation_inputs(trades, all_r, regimes, thesis)
    assert "LOCKED" not in m              # never computed for ineligible
    assert m["ELIG"]["n_trades"] < 3
    assert m["ELIG"]["dsr"] == 0.0        # stale → zero DSR → zero weight
    allocate_weights(trades, m)
    assert all(t["alloc_weight"] == 0.0 for t in trades)


def test_trade_leg_vol_signed_combination():
    idx = pd.bdate_range("2020-01-01", periods=300)
    base = pd.Series(np.random.default_rng(1).normal(0, 0.01, 300), index=idx)
    all_r = pd.DataFrame({"Gold": base, "S&P 500": base})   # identical series
    # Long/Short of identical series is flat → vol ≈ 0; Long/Long is not
    v_ls = trade_leg_vol(all_r, ["Gold", "S&P 500"], ["Long", "Short"])
    v_ll = trade_leg_vol(all_r, ["Gold", "S&P 500"], ["Long", "Long"])
    assert v_ls < 1e-10
    assert v_ll > 0.05

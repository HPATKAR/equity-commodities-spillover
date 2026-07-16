"""
Step 3 of 4 - portfolio constraints.

Pins: no conflict and no correlation-cluster exceeds its cap in the FINAL
(re-normalized) weights; duplicated bets share one cluster budget via the
same union-find compute_effective_n uses; locked trades stay at zero; when
every group binds the residual is cash, never a cap breach.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.trade_allocator import (
    apply_portfolio_constraints, build_correlation_clusters,
    CONFLICT_CAP, CLUSTER_CAP, MAX_SINGLE_WEIGHT,
)
from src.analysis.backtest import correlation_clusters


def _frame(seed=0, periods=400):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=periods)
    base = rng.normal(0, 0.01, periods)
    return pd.DataFrame({
        "Gold": base + rng.normal(0, 0.001, periods),          # ~identical to GoldB
        "GoldB": base + rng.normal(0, 0.001, periods),
        "S&P 500": rng.normal(0, 0.012, periods),
        "WTI Crude Oil": rng.normal(0, 0.02, periods),
        "Nikkei 225": rng.normal(0, 0.011, periods),
    }, index=idx)


def _t(name, assets, weight, conflict=None, eligible=True):
    return {"name": name, "is_eligible": eligible,
            "assets": list(assets), "direction": ["Long"] * len(assets),
            "alloc_weight": weight, "conflict_id": conflict}


def _w(trades, name):
    return next(t["alloc_weight"] for t in trades if t["name"] == name)


# ── Conflict cap ─────────────────────────────────────────────────────────────

def test_conflict_cap_holds_after_renormalization():
    all_r = _frame()
    trades = [
        _t("U1", ["Gold"], 0.30, conflict="ukraine_russia"),
        _t("U2", ["WTI Crude Oil"], 0.30, conflict="ukraine_russia"),
        _t("M1", ["S&P 500"], 0.20),
        _t("M2", ["Nikkei 225"], 0.20),
    ]
    apply_portfolio_constraints(trades, all_r)
    gross = sum(t["alloc_weight"] for t in trades)
    ua = _w(trades, "U1") + _w(trades, "U2")
    assert ua <= CONFLICT_CAP * 1.0 + 1e-6          # target gross was 1.0
    assert gross == pytest.approx(1.0, abs=1e-4)    # excess re-normalized out
    # FORBIDDEN: renorm must not have pushed the conflict back over its cap
    assert ua <= CONFLICT_CAP + 1e-6


def test_unattributed_trades_exempt_from_conflict_cap():
    all_r = _frame()
    trades = [
        _t("M1", ["S&P 500"], 0.55),                 # no conflict tag
        _t("M2", ["Nikkei 225"], 0.45),
    ]
    apply_portfolio_constraints(trades, all_r, single_cap=1.0)
    # No conflict groups exist - weights untouched by the conflict cap
    assert _w(trades, "M1") == pytest.approx(0.55, abs=1e-4)


# ── Cluster cap (shared union-find) ──────────────────────────────────────────

def test_duplicate_trades_cluster_and_share_one_budget():
    all_r = _frame()
    trades = [
        _t("GoldPairA", ["Gold"], 0.40),             # Gold ≈ GoldB: one bet
        _t("GoldPairB", ["GoldB"], 0.40),
        _t("Equity", ["S&P 500"], 0.20),
    ]
    clusters = build_correlation_clusters(trades, all_r)
    assert clusters["GoldPairA"] == clusters["GoldPairB"]
    assert clusters["GoldPairA"] != clusters["Equity"]

    apply_portfolio_constraints(trades, all_r, single_cap=1.0)
    dup = _w(trades, "GoldPairA") + _w(trades, "GoldPairB")
    assert dup <= CLUSTER_CAP + 1e-6                 # 0.80 → capped at 0.45


def test_uncorrelated_trades_do_not_cluster():
    all_r = _frame()
    series = {
        "A": all_r["S&P 500"], "B": all_r["WTI Crude Oil"],
        "C": all_r["Nikkei 225"],
    }
    cl = correlation_clusters(series)
    assert len(set(cl.values())) == 3


# ── Interplay, cash remainder, and the lock ──────────────────────────────────

def test_all_groups_binding_leaves_cash_not_breach():
    all_r = _frame()
    trades = [                                       # one cluster holds it all
        _t("GA", ["Gold"], 0.50),
        _t("GB", ["GoldB"], 0.50),
    ]
    apply_portfolio_constraints(trades, all_r, single_cap=1.0)
    gross = sum(t["alloc_weight"] for t in trades)
    assert gross <= CLUSTER_CAP + 1e-6               # rest is cash - no breach


def test_locked_trades_stay_zero_through_constraints():
    all_r = _frame()
    trades = [
        _t("OK", ["S&P 500"], 1.0),
        _t("LOCKED", ["Gold"], 0.0, eligible=False),
    ]
    trades[1]["alloc_weight"] = 0.0
    apply_portfolio_constraints(trades, all_r)
    assert _w(trades, "LOCKED") == 0.0
    assert trades[1]["constraint_detail"]["cluster_id"] is None


def test_single_cap_still_enforced_with_groups():
    all_r = _frame()
    trades = [
        _t("BIG", ["S&P 500"], 0.70),
        _t("M1", ["WTI Crude Oil"], 0.15),
        _t("M2", ["Nikkei 225"], 0.15),
    ]
    apply_portfolio_constraints(trades, all_r)
    assert _w(trades, "BIG") <= MAX_SINGLE_WEIGHT + 1e-6
    assert sum(t["alloc_weight"] for t in trades) == pytest.approx(1.0, abs=1e-3)


def test_final_state_never_exceeds_any_cap():
    # Adversarial mix: conflict concentration + duplicates + one giant trade
    all_r = _frame()
    trades = [
        _t("U1", ["Gold"], 0.35, conflict="iran_conflict"),
        _t("U2", ["GoldB"], 0.35, conflict="iran_conflict"),
        _t("BIG", ["S&P 500"], 0.25),
        _t("M", ["Nikkei 225"], 0.05),
    ]
    apply_portfolio_constraints(trades, all_r)
    ws = {t["name"]: t["alloc_weight"] for t in trades}
    assert all(w <= MAX_SINGLE_WEIGHT + 1e-6 for w in ws.values())
    assert ws["U1"] + ws["U2"] <= CONFLICT_CAP + 1e-6
    assert ws["U1"] + ws["U2"] <= CLUSTER_CAP + 1e-6   # also one cluster

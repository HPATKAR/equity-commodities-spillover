"""
Spillover analytics:
  - Granger causality (commodity → equity and vice versa)
  - Transfer entropy (directional information flow)
  - Diebold-Yilmaz spillover index (VAR-based)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from typing import Optional


# ── Granger causality ──────────────────────────────────────────────────────

def granger_test(
    cause: pd.Series,
    effect: pd.Series,
    max_lag: int = 5,
    significance: float = 0.05,
) -> dict:
    """
    Test whether `cause` Granger-causes `effect`.
    Returns dict: {lag: p_value, ..., 'min_p': ..., 'significant': bool}
    """
    combined = pd.concat([effect, cause], axis=1).dropna()
    if len(combined) < max_lag * 10:
        return {"min_p": np.nan, "significant": False, "results": {}}
    try:
        res = grangercausalitytests(combined.values, maxlag=max_lag, verbose=False)
        p_values = {
            lag: min(r[0][test][1] for test in ["ssr_ftest", "ssr_chi2test"])
            for lag, r in res.items()
        }
        min_p = min(p_values.values())
        return {
            "min_p": round(min_p, 4),
            "significant": min_p < significance,
            "results": p_values,
            "best_lag": min(p_values, key=p_values.get),
        }
    except Exception:
        return {"min_p": np.nan, "significant": False, "results": {}}


def granger_grid(
    equity_returns: pd.DataFrame,
    commodity_returns: pd.DataFrame,
    max_lag: int = 5,
    significance: float = 0.05,
) -> pd.DataFrame:
    """
    Full grid of Granger tests: commodity → equity AND equity → commodity.
    Returns tidy DataFrame: [cause, effect, direction, min_p, significant, best_lag].
    """
    rows = []
    eq_cols  = equity_returns.columns.tolist()
    cmd_cols = commodity_returns.columns.tolist()

    for eq in eq_cols:
        for cmd in cmd_cols:
            r_eq  = equity_returns[eq].dropna()
            r_cmd = commodity_returns[cmd].dropna()
            idx   = r_eq.index.intersection(r_cmd.index)
            if len(idx) < 60:
                continue
            r_eq  = r_eq.loc[idx]
            r_cmd = r_cmd.loc[idx]

            # commodity → equity
            res1 = granger_test(r_cmd, r_eq, max_lag, significance)
            rows.append({
                "cause": cmd, "effect": eq,
                "direction": "Commodity → Equity",
                **res1,
            })
            # equity → commodity
            res2 = granger_test(r_eq, r_cmd, max_lag, significance)
            rows.append({
                "cause": eq, "effect": cmd,
                "direction": "Equity → Commodity",
                **res2,
            })

    df = pd.DataFrame(rows)
    return df.drop(columns=["results"], errors="ignore")


# ── Transfer entropy ───────────────────────────────────────────────────────

def _discretize(x: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Bin continuous series into integer labels."""
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    return np.digitize(x, edges[:-1]) - 1


def transfer_entropy(
    source: pd.Series,
    target: pd.Series,
    lag: int = 1,
    n_bins: int = 5,
) -> float:
    """
    Transfer entropy: TE(source → target).
    Measures how much the past of `source` reduces uncertainty in `target`
    beyond `target`'s own past.
    """
    combined = pd.concat([source, target], axis=1).dropna()
    if len(combined) < lag + 20:
        return np.nan

    x = _discretize(combined.iloc[:, 0].values, n_bins)
    y = _discretize(combined.iloc[:, 1].values, n_bins)

    T = len(y) - lag
    y_fut  = y[lag:]
    y_past = y[:T]
    x_past = x[:T]

    def entropy(*arrays) -> float:
        combined_arr = np.column_stack(arrays)
        _, counts = np.unique(combined_arr, axis=0, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))

    h_yfut_given_ypast   = entropy(y_fut, y_past) - entropy(y_past)
    h_yfut_given_both    = entropy(y_fut, y_past, x_past) - entropy(y_past, x_past)
    te = h_yfut_given_ypast - h_yfut_given_both
    return float(max(te, 0.0))


def transfer_entropy_matrix(
    equity_returns: pd.DataFrame,
    commodity_returns: pd.DataFrame,
    lag: int = 1,
    n_bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two DataFrames:
      te_cmd_to_eq[equity, commodity] = TE(commodity → equity)
      te_eq_to_cmd[equity, commodity] = TE(equity → commodity)
    """
    eq_cols  = equity_returns.columns.tolist()
    cmd_cols = commodity_returns.columns.tolist()

    te_c2e = pd.DataFrame(index=eq_cols, columns=cmd_cols, dtype=float)
    te_e2c = pd.DataFrame(index=eq_cols, columns=cmd_cols, dtype=float)

    for eq in eq_cols:
        for cmd in cmd_cols:
            r_eq  = equity_returns[eq]
            r_cmd = commodity_returns[cmd]
            te_c2e.loc[eq, cmd] = transfer_entropy(r_cmd, r_eq,  lag, n_bins)
            te_e2c.loc[eq, cmd] = transfer_entropy(r_eq,  r_cmd, lag, n_bins)

    return te_c2e.astype(float), te_e2c.astype(float)


def net_flow_matrix(
    te_c2e: pd.DataFrame,
    te_e2c: pd.DataFrame,
) -> pd.DataFrame:
    """Net transfer entropy: TE(cmd→eq) − TE(eq→cmd). Positive = commodity leads."""
    return te_c2e - te_e2c


# ── Diebold-Yilmaz spillover index ────────────────────────────────────────

def diebold_yilmaz(
    returns: pd.DataFrame,
    lag_order: int = 4,
    horizon: int = 10,
    top_n: int = 6,
) -> dict:
    """
    Diebold-Yilmaz (2012) forecast error variance decomposition spillover index.
    Uses VAR(lag_order) fitted on the first `top_n` columns of `returns`.

    Returns:
      spillover_table: pd.DataFrame - from/to variance decomposition
      total_spillover: float - aggregate spillover index (%)
    """
    data = returns.dropna(how="all").iloc[:, :top_n]
    data = data.dropna()
    if len(data) < lag_order * 10:
        return {"spillover_table": pd.DataFrame(), "total_spillover": np.nan}

    try:
        model  = VAR(data)
        result = model.fit(lag_order)
        fevd   = result.fevd(horizon)

        n = len(data.columns)
        table = pd.DataFrame(
            fevd.decomp[-1].T,
            index=data.columns,
            columns=data.columns,
        )
        # Each row sums to 1; express as %
        table = table * 100
        own_share     = np.diag(table.values).sum()
        total_spillover = 100 - own_share / n

        return {
            "spillover_table": table.round(2),
            "total_spillover": round(total_spillover, 2),
        }
    except Exception:
        return {"spillover_table": pd.DataFrame(), "total_spillover": np.nan}

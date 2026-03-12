"""
Correlation analytics:
  - Rolling Pearson correlations
  - Correlation regime detection (high / low / crisis)
  - DCC-GARCH dynamic conditional correlations
  - Correlation heatmap matrix
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Optional


# ── Rolling correlation ────────────────────────────────────────────────────

def rolling_correlation(
    s1: pd.Series,
    s2: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Pearson rolling correlation between two return series."""
    combined = pd.concat([s1, s2], axis=1).dropna()
    return combined.iloc[:, 0].rolling(window).corr(combined.iloc[:, 1])


def rolling_correlation_matrix(
    returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Rolling pairwise correlations — returns a tidy DataFrame:
    columns [date, asset_a, asset_b, correlation].
    """
    rows = []
    cols = returns.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rc = rolling_correlation(returns[a], returns[b], window)
            for dt, val in rc.dropna().items():
                rows.append({"date": dt, "asset_a": a, "asset_b": b, "correlation": val})
    return pd.DataFrame(rows)


# ── Static correlation matrix ──────────────────────────────────────────────

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Full-sample Pearson correlation matrix."""
    return returns.corr()


def cross_asset_corr(
    equity_returns: pd.DataFrame,
    commodity_returns: pd.DataFrame,
    window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Cross-asset correlation matrix: equities (rows) vs commodities (cols).
    If window is provided, uses the last `window` observations.
    """
    eq  = equity_returns.iloc[-window:] if window else equity_returns
    cmd = commodity_returns.iloc[-window:] if window else commodity_returns
    combined = pd.concat([eq, cmd], axis=1).dropna(how="all")
    corr = combined.corr()
    return corr.loc[equity_returns.columns, commodity_returns.columns]


# ── Correlation regime detection ───────────────────────────────────────────

def detect_correlation_regime(
    avg_corr_series: pd.Series,
    low_thresh: float = 0.15,
    high_thresh: float = 0.45,
) -> pd.Series:
    """
    Classify daily average cross-asset correlation into regime:
      0 = Decorrelated (< low_thresh)
      1 = Normal
      2 = Elevated
      3 = Crisis (> high_thresh)
    """
    regimes = pd.Series(1, index=avg_corr_series.index, name="regime")
    regimes[avg_corr_series < low_thresh]  = 0
    regimes[avg_corr_series >= high_thresh] = 2
    # Rolling 5d persistence: if 3+ days above high_thresh → crisis
    crisis_mask = (avg_corr_series >= high_thresh).rolling(5).sum() >= 3
    regimes[crisis_mask] = 3
    return regimes


def average_cross_corr_series(
    equity_returns: pd.DataFrame,
    commodity_returns: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    """
    Daily time series of mean |rolling correlation| across all equity-commodity pairs.
    """
    combined = pd.concat([equity_returns, commodity_returns], axis=1).dropna(how="all")
    eq_cols  = [c for c in equity_returns.columns    if c in combined.columns]
    cmd_cols = [c for c in commodity_returns.columns if c in combined.columns]

    corr_series = []
    for eq in eq_cols:
        for cmd in cmd_cols:
            rc = combined[eq].rolling(window).corr(combined[cmd])
            corr_series.append(rc.abs())

    if not corr_series:
        return pd.Series(dtype=float)
    avg = pd.concat(corr_series, axis=1).mean(axis=1)
    return avg.dropna()


# ── DCC-GARCH (simplified) ─────────────────────────────────────────────────

def dcc_correlation(
    r1: pd.Series,
    r2: pd.Series,
    a: float = 0.05,
    b: float = 0.90,
) -> pd.Series:
    """
    Simplified DCC-GARCH dynamic conditional correlation between two series.
    Uses Engle (2002) DCC(1,1) with given a, b parameters.
    a + b < 1 required for stationarity.
    """
    combined = pd.concat([r1, r2], axis=1).dropna()
    if len(combined) < 30:
        return pd.Series(dtype=float)

    x = combined.values.astype(float)
    T, K = x.shape

    # Step 1: demean
    x = x - x.mean(axis=0)

    # Step 2: unconditional covariance Q-bar
    Q_bar = (x.T @ x) / T

    # Step 3: iterate DCC
    Q = Q_bar.copy()
    dcc = np.zeros(T)

    for t in range(1, T):
        eps = x[t - 1]
        Q = (1 - a - b) * Q_bar + a * np.outer(eps, eps) + b * Q
        D_inv = np.diag(1.0 / np.sqrt(np.diag(Q)))
        R = D_inv @ Q @ D_inv
        dcc[t] = R[0, 1]

    dcc_series = pd.Series(dcc, index=combined.index, name="DCC")
    dcc_series.iloc[0] = np.nan
    return dcc_series


def dcc_matrix(
    equity_returns: pd.DataFrame,
    commodity_returns: pd.DataFrame,
    pairs: Optional[list[tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Compute DCC correlations for selected (equity, commodity) pairs.
    Returns wide DataFrame: one column per pair.
    """
    if pairs is None:
        eq_cols  = equity_returns.columns[:3].tolist()
        cmd_cols = commodity_returns.columns[:4].tolist()
        pairs = [(e, c) for e in eq_cols for c in cmd_cols]

    result = {}
    for eq, cmd in pairs:
        if eq in equity_returns.columns and cmd in commodity_returns.columns:
            r1 = equity_returns[eq]
            r2 = commodity_returns[cmd]
            label = f"{eq} / {cmd}"
            result[label] = dcc_correlation(r1, r2)

    if not result:
        return pd.DataFrame()
    return pd.DataFrame(result).sort_index()

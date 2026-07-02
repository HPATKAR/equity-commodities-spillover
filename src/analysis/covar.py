"""
ΔCoVaR — Adrian and Brunnermeier (2016) systemic risk measure.

ONE quantile regression of the system return on each asset's return at τ=0.05.
The regression is then evaluated at two asset states:
  distress: asset at its own VaR(5%)     →  CoVaR_{5%}^{system|i distressed}
  median:   asset at its own median       →  CoVaR_{5%}^{system|i normal}

ΔCoVaR_i = CoVaR(5% | asset distressed) − CoVaR(5% | asset at median)
           = β_{5%}^{system|i} × (VaR^i_{5%} − Median^i)

Both CoVaR terms are the 5th percentile of the SYSTEM distribution —
only the conditioning asset state differs. This is the correct AB convention:
a single τ=5% regression evaluated at two points on the asset's return axis.

The previous (wrong) formulation used two separate regressions (τ=5% and τ=50%)
which compared system 5th-percentile against system median, inflating every
ΔCoVaR by the full distributional spread of the system (~2–3%) and washing out
the cross-sectional signal.

Sign:  β > 0 and VaR(5%) < Median  ⟹  ΔCoVaR < 0  always.
Rank:  more-negative ΔCoVaR = wider asset tail × stronger system coupling
       = higher systemic risk amplification.

Reference:
  Adrian, T. & Brunnermeier, M. K. (2016). CoVaR.
  American Economic Review, 106(7), 1705–1741.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import streamlit as st

try:
    from statsmodels.regression.quantile_regression import QuantReg, IterationLimitWarning
    _HAS_QR = True
except ImportError:
    try:
        from statsmodels.regression.quantile_regression import QuantReg
        from statsmodels.tools.sm_exceptions import IterationLimitWarning
        _HAS_QR = True
    except ImportError:
        _HAS_QR = False

_MAX_ITER = 2000


# ── Core regression helper ────────────────────────────────────────────────────

def _fit_quantreg(
    x: np.ndarray,
    y: np.ndarray,
    q: float,
) -> tuple[float, float, bool]:
    """
    Fit QuantReg(y ~ 1 + x) at quantile q.
    Returns (alpha, beta, converged).
    converged=False means solver hit _MAX_ITER without meeting p_tol; the
    estimate is still returned but should be flagged.
    OLS fallback (converged=False) if solver raises.
    """
    X = np.column_stack([np.ones(len(x)), x])
    try:
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter("always", IterationLimitWarning)
            res = QuantReg(y, X).fit(q=q, max_iter=_MAX_ITER, p_tol=1e-6)
        converged = not (
            any(issubclass(w.category, IterationLimitWarning) for w in _w)
            or res.iterations >= _MAX_ITER
        )
        return float(res.params[0]), float(res.params[1]), converged
    except Exception:
        cov  = np.cov(x, y, ddof=1)
        beta = cov[0, 1] / (cov[0, 0] + 1e-12)
        return float(np.mean(y)) - beta * float(np.mean(x)), beta, False


# ── Full-sample ΔCoVaR ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False, max_entries=4)
def compute_covar(
    _all_r: pd.DataFrame,
    q_sys: float = 0.05,
    q_median: float = 0.50,
    min_obs: int = 252,
) -> pd.DataFrame:
    """
    Full-sample ΔCoVaR for every asset in _all_r.

    ONE QuantReg at τ=q_sys (default 5%) of system_r on each asset_r.
    Evaluated at the asset's own q_sys-percentile (distress) and median:
      CoVaR_distress = α + β · VaR^i_{5%}
      CoVaR_median   = α + β · Median^i
      ΔCoVaR         = CoVaR_distress − CoVaR_median  (always ≤ 0)

    System return: equal-weighted mean of all columns in _all_r.

    Returns DataFrame indexed by asset with columns:
      var5         — asset VaR at 5th percentile (%, negative)
      median_r     — asset median return (%, near zero)
      covar5       — system 5th-pctile when asset is distressed (%, negative)
      covar_med    — system 5th-pctile when asset is at median  (%, negative)
      delta_covar  — covar5 − covar_med  (%, always negative; more-negative = more systemic)
      beta         — QuantReg slope at τ=q_sys
      n_obs        — overlapping observations used
    Sorted ascending by delta_covar (most systemic first).
    """
    if not _HAS_QR:
        return pd.DataFrame()

    df = _all_r.dropna(how="all")
    if df.shape[1] < 2 or len(df) < min_obs:
        return pd.DataFrame()

    sys_r = df.mean(axis=1)
    rows: list[dict] = []

    for asset in df.columns:
        xi = df[asset].dropna()
        yi = sys_r.reindex(xi.index).dropna()
        xi = xi.reindex(yi.index)
        n  = len(xi)
        if n < min_obs:
            continue

        xa = xi.values.astype(float)
        ya = yi.values.astype(float)

        alpha, beta, converged = _fit_quantreg(xa, ya, q_sys)

        var5      = float(np.percentile(xa, q_sys    * 100))
        median_r  = float(np.percentile(xa, q_median * 100))
        covar5    = alpha + beta * var5
        covar_med = alpha + beta * median_r

        rows.append({
            "asset":       asset,
            "var5":        round(var5      * 100, 3),
            "median_r":    round(median_r  * 100, 3),
            "covar5":      round(covar5    * 100, 3),
            "covar_med":   round(covar_med * 100, 3),
            "delta_covar": round((covar5 - covar_med) * 100, 3),
            "beta":        round(beta, 4),
            "n_obs":       n,
            "converged":   converged,
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .set_index("asset")
        .sort_values("delta_covar")
    )


# ── Rolling ΔCoVaR ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False, max_entries=2)
def rolling_covar(
    _all_r: pd.DataFrame,
    window: int = 126,
    step: int = 5,
    q_sys: float = 0.05,
    q_median: float = 0.50,
    min_obs: int = 60,
    top_n: int = 5,
    _n_rows: int = 0,
) -> pd.DataFrame:
    """
    Rolling ΔCoVaR for the top_n most systemic assets (by full-sample rank).
    One QuantReg at τ=q_sys per asset per window, stepped every `step` days.

    Returns DataFrame: index=date, columns=asset names, values=ΔCoVaR (%).
    """
    if not _HAS_QR:
        return pd.DataFrame()

    df = _all_r.dropna(how="all")
    if df.shape[1] < 2 or len(df) < window + min_obs:
        return pd.DataFrame()

    full = compute_covar(_all_r, q_sys=q_sys, q_median=q_median, min_obs=min_obs)
    if full.empty:
        return pd.DataFrame()
    top_assets = [a for a in full.head(top_n).index if a in df.columns]
    if not top_assets:
        return pd.DataFrame()

    records: list[dict] = []
    idx = df.index

    for i in range(window - 1, len(df), step):
        chunk = df.iloc[max(0, i - window + 1): i + 1]
        if len(chunk) < min_obs:
            continue
        sys_r = chunk.mean(axis=1)
        row: dict = {"date": idx[i]}

        for asset in top_assets:
            xi = chunk[asset].dropna()
            yi = sys_r.reindex(xi.index).dropna()
            xi = xi.reindex(yi.index)
            if len(xi) < min_obs:
                row[asset] = np.nan
                continue
            xa = xi.values.astype(float)
            ya = yi.values.astype(float)
            alpha, beta, converged = _fit_quantreg(xa, ya, q_sys)
            if not converged:
                row[asset] = np.nan
                continue
            var5     = float(np.percentile(xa, q_sys    * 100))
            median_r = float(np.percentile(xa, q_median * 100))
            row[asset] = round((alpha + beta * var5 - alpha - beta * median_r) * 100, 3)

        records.append(row)

    if not records:
        return pd.DataFrame()

    roll = pd.DataFrame(records).set_index("date")
    roll.index = pd.to_datetime(roll.index)
    return roll

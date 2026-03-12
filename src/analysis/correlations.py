"""
Correlation analytics:
  - Rolling Pearson correlations
  - Correlation regime detection (high / low / crisis)
  - DCC-GARCH dynamic conditional correlations
  - Correlation heatmap matrix
  - Multi-feature composite stress index (improved regime input)
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
    p_decorr: float = 20.0,
    p_normal: float = 55.0,
    p_elevated: float = 80.0,
    smooth_window: int = 5,
    persist_window: int = 10,
) -> pd.Series:
    """
    Adaptive, percentile-based regime classification with smoothing and hysteresis.

    Improvements over fixed-threshold version:
      1. Percentile thresholds — calibrated to the actual data distribution, not
         hardcoded values that can miss regime shifts when the vol environment changes.
      2. Median smoothing (smooth_window days) — removes single-day flickers.
      3. Hysteresis — exit thresholds are 5pp lower than entry thresholds, preventing
         rapid oscillation near the boundary.
      4. Persistence filter — Crisis (3) requires the smoothed series to exceed the
         elevated threshold for the majority of a rolling window.

    Regimes:
      0 = Decorrelated  (< p_decorr percentile)
      1 = Normal        (p_decorr – p_normal percentile)
      2 = Elevated      (p_normal – p_elevated percentile)
      3 = Crisis        (> p_elevated percentile + persistence confirmed)
    """
    s = avg_corr_series.dropna()
    if len(s) < 60:
        return pd.Series(1, index=avg_corr_series.index, name="regime")

    # 1. Smooth with rolling median to remove single-day noise
    smoothed = s.rolling(smooth_window, min_periods=1, center=True).median()

    # 2. Compute adaptive thresholds from full-sample percentiles
    t_decorr   = float(np.percentile(smoothed, p_decorr))
    t_normal   = float(np.percentile(smoothed, p_normal))
    t_elevated = float(np.percentile(smoothed, p_elevated))

    # Hysteresis bands (exit 5pp below entry)
    t_decorr_exit   = t_decorr   * 1.05
    t_normal_exit   = t_normal   * 0.95
    t_elevated_exit = t_elevated * 0.95

    # 3. Initial classification on smoothed series
    raw = pd.Series(1, index=smoothed.index)
    raw[smoothed < t_decorr]   = 0
    raw[smoothed >= t_normal]  = 2
    raw[smoothed >= t_elevated] = 3

    # 4. Apply hysteresis: walk forward, only change regime when clear of exit band
    regime_out = raw.copy()
    current = int(raw.iloc[0])
    for i in range(1, len(raw)):
        v = smoothed.iloc[i]
        if current == 0:
            if v >= t_decorr_exit:   current = 1
        elif current == 1:
            if v < t_decorr:         current = 0
            elif v >= t_normal:      current = 2
        elif current == 2:
            if v < t_normal_exit:   current = 1
            elif v >= t_elevated:   current = 3
        elif current == 3:
            if v < t_elevated_exit: current = 2
        regime_out.iloc[i] = current

    # 5. Persistence gate: Crisis (3) only confirmed when 60%+ of rolling window ≥ elevated
    persist_frac = (smoothed >= t_elevated).rolling(persist_window, min_periods=3).mean()
    # Downgrade crisis to elevated where persistence is insufficient
    regime_out[(regime_out == 3) & (persist_frac < 0.5)] = 2

    # Reindex back to original (may include NaN-leading rows)
    result = regime_out.reindex(avg_corr_series.index).ffill().fillna(1).astype(int)
    result.name = "regime"
    return result


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


# ── Multi-feature composite stress index ──────────────────────────────────

_ENERGY_METALS = [
    "WTI Crude Oil", "Brent Crude", "Natural Gas",
    "Gold", "Silver", "Copper",
]


def compute_regime_features(
    equity_returns: pd.DataFrame,
    commodity_returns: pd.DataFrame,
    avg_corr_slow: Optional[pd.Series] = None,
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Build a 4-column feature matrix for ML-based regime detection.
    All features are backward-looking (no look-ahead).

    Columns:
      avg_corr_slow  — precomputed 60d avg |cross-asset corr| (pass to avoid recompute)
      avg_corr_fast  — 20d rolling |corr| of equal-weight composites (fast stress signal)
      equity_vol     — 20d realised vol of equal-weight equity universe (ann. %)
      cmd_vol        — 20d realised vol of energy+metals basket (ann. %)
    """
    combined = pd.concat([equity_returns, commodity_returns], axis=1).dropna(how="all")
    eq_cols  = [c for c in equity_returns.columns  if c in combined.columns]
    cmd_cols = [c for c in commodity_returns.columns if c in combined.columns]
    if not eq_cols or not cmd_cols:
        return pd.DataFrame()

    # Equal-weight composite returns → fast, single-series signals
    eq_idx  = combined[eq_cols].mean(axis=1)
    cmd_idx = combined[cmd_cols].mean(axis=1)

    # Fast cross-asset correlation on composite (one pair, 20d window)
    fast_corr = eq_idx.rolling(vol_window, min_periods=vol_window // 2).corr(cmd_idx).abs()

    # Equity realised vol (equal-weight; strong proxy for VIX)
    eq_vol = combined[eq_cols].rolling(vol_window, min_periods=vol_window // 2).std().mean(axis=1) * np.sqrt(252) * 100

    # Commodity vol — energy + metals
    em_cols = [c for c in _ENERGY_METALS if c in combined.columns]
    if not em_cols:
        em_cols = cmd_cols[:6]
    cmd_vol = combined[em_cols].rolling(vol_window, min_periods=vol_window // 2).std().mean(axis=1) * np.sqrt(252) * 100

    # Slow correlation — reuse precomputed series if supplied
    if avg_corr_slow is not None:
        slow_corr = avg_corr_slow.reindex(combined.index)
    else:
        # Cheap proxy: 60d rolling corr of composite indices
        slow_corr = eq_idx.rolling(60, min_periods=30).corr(cmd_idx).abs()

    out = pd.DataFrame({
        "avg_corr_slow": slow_corr,
        "avg_corr_fast": fast_corr,
        "equity_vol":    eq_vol,
        "cmd_vol":       cmd_vol,
    }, index=combined.index)
    return out.dropna()


def composite_stress_index(
    equity_returns: pd.DataFrame,
    commodity_returns: pd.DataFrame,
    avg_corr: Optional[pd.Series] = None,
    lookback: int = 252,
) -> pd.Series:
    """
    0–100 composite stress index with research-calibrated blending.

    Weights:
      45%  equity_vol     (z-score mapped — direct VIX proxy, strongest single predictor)
      35%  avg_corr_slow  (rolling percentile — relative stress in cross-asset linkage)
      15%  cmd_vol        (z-score mapped — energy/metals risk premium)
       5%  avg_corr_fast  (rolling percentile — short-term correlation acceleration)

    Vol components use z-score → 0-100 mapping (z=0→50, z=+2→80, z=-2→20) rather
    than empirical percentiles so that absolute vol level information is preserved.
    This is critical: equity_vol ≈ VIX (r~0.85), so absolute vol levels predict
    VIX stress threshold breaches far better than percentile ranks which are always
    uniform by construction and lose regime information.

    Returns a 0–100 series with the same index as the inputs.
    Higher values = more stress.
    """
    feats = compute_regime_features(
        equity_returns, commodity_returns,
        avg_corr_slow=avg_corr,
    )
    if feats.empty:
        return pd.Series(dtype=float)

    min_periods = max(60, lookback // 4)

    def _pct(s: pd.Series) -> pd.Series:
        """Empirical rolling percentile: fraction of past obs below current."""
        return s.rolling(lookback, min_periods=min_periods).apply(
            lambda x: float((x[:-1] < x[-1]).mean() * 100) if len(x) > 1 else 50.0,
            raw=True,
        )

    def _zscore_score(s: pd.Series) -> pd.Series:
        """z-score → 0-100: z=0→50, z=+2→80, z=-2→20 (z clipped ±3).
        Preserves absolute level information — equates to VIX level, not just rank."""
        mean = s.rolling(lookback, min_periods=min_periods).mean()
        std  = s.rolling(lookback, min_periods=min_periods).std().replace(0, np.nan)
        z    = (s - mean) / std
        return (50 + z.clip(-3, 3) * 15).clip(0, 100)

    # Correlation signals: relative position is the right measure
    p_slow = _pct(feats["avg_corr_slow"])
    p_fast = _pct(feats["avg_corr_fast"])

    # Vol signals: z-score preserves absolute level (equity_vol ≈ VIX)
    p_eq  = _zscore_score(feats["equity_vol"])
    p_cmd = _zscore_score(feats["cmd_vol"])

    composite = (
        0.45 * p_eq      # dominant: equity vol tracks VIX directly
        + 0.35 * p_slow  # broad correlation regime
        + 0.15 * p_cmd   # commodity stress premium
        + 0.05 * p_fast  # short-term correlation acceleration
    ).clip(0, 100)

    composite.name = "stress_index"
    return composite.dropna()


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

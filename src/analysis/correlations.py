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
    Rolling pairwise correlations - returns a tidy DataFrame:
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
      1. Percentile thresholds - calibrated to the actual data distribution, not
         hardcoded values that can miss regime shifts when the vol environment changes.
      2. Median smoothing (smooth_window days) - removes single-day flickers.
      3. Hysteresis - exit thresholds are 5pp lower than entry thresholds, preventing
         rapid oscillation near the boundary.
      4. Persistence filter - Crisis (3) requires the smoothed series to exceed the
         elevated threshold for the majority of a rolling window.

    Regimes:
      0 = Decorrelated  (< p_decorr percentile)
      1 = Normal        (p_decorr – p_normal percentile)
      2 = Elevated      (p_normal – p_elevated percentile)
      3 = Crisis        (> p_elevated percentile + persistence confirmed)
    """
    s = avg_corr_series.dropna()
    if len(s) < 60:
        _insuf = pd.Series(1, index=avg_corr_series.index, name="regime")
        _insuf.attrs["insufficient_data"] = True
        _insuf.attrs["n_obs"] = len(s)
        return _insuf

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
    result.attrs["insufficient_data"] = False
    result.attrs["n_obs"] = len(s)
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
      avg_corr_slow  - precomputed 60d avg |cross-asset corr| (pass to avoid recompute)
      avg_corr_fast  - 20d rolling |corr| of equal-weight composites (fast stress signal)
      equity_vol     - 20d realised vol of equal-weight equity universe (ann. %)
      cmd_vol        - 20d realised vol of energy+metals basket (ann. %)
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

    # Commodity vol - energy + metals
    em_cols = [c for c in _ENERGY_METALS if c in combined.columns]
    if not em_cols:
        em_cols = cmd_cols[:6]
    cmd_vol = combined[em_cols].rolling(vol_window, min_periods=vol_window // 2).std().mean(axis=1) * np.sqrt(252) * 100

    # Slow correlation - reuse precomputed series if supplied and sufficiently dense
    if avg_corr_slow is not None:
        slow_reindexed = avg_corr_slow.reindex(combined.index)
        # Fall back to dense proxy when the supplied series is too sparse (< 50% coverage).
        # average_cross_corr_series uses a union-index approach that can produce only ~32
        # non-NaN rows for a 14-year dataset; using it directly would make dropna() below
        # discard virtually all rows and the ML classifier would fail with "insufficient obs".
        if slow_reindexed.notna().mean() >= 0.50:
            slow_corr = slow_reindexed
        else:
            slow_corr = eq_idx.rolling(60, min_periods=30).corr(cmd_idx).abs()
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
      45%  equity_vol     (z-score mapped - direct VIX proxy, strongest single predictor)
      35%  avg_corr_slow  (rolling percentile - relative stress in cross-asset linkage)
      15%  cmd_vol        (z-score mapped - energy/metals risk premium)
       5%  avg_corr_fast  (rolling percentile - short-term correlation acceleration)

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
        Preserves absolute level information - equates to VIX level, not just rank."""
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
    ewma_lambda: float = 0.94,
) -> pd.Series:
    """
    DCC-GARCH(1,1) dynamic conditional correlation (Engle 2002).

    Implementation follows the two-step procedure:
      Step 1  — EWMA volatility pre-whitening (RiskMetrics, λ=0.94).
                Raw returns are standardised by their EWMA conditional
                standard deviation before the DCC recursion.  Skipping
                this step causes the quasi-correlation matrix Q to be
                contaminated by heteroskedasticity, inflating correlations
                during high-vol episodes even when the true ρ is unchanged.
      Step 2  — DCC(1,1) recursion on the standardised residuals ε̃_t:
                Q_t = (1−a−b)Q̄ + a ε̃_{t−1}ε̃'_{t−1} + b Q_{t−1}
                R_t = diag(Q_t)^{−½} Q_t diag(Q_t)^{−½}

    Parameters
    ----------
    a, b        : DCC(1,1) parameters; a+b < 1 for covariance stationarity.
    ewma_lambda : EWMA decay factor for volatility pre-whitening (0 < λ < 1).
                  λ=0.94 is the J.P. Morgan RiskMetrics daily standard.
    """
    combined = pd.concat([r1, r2], axis=1).dropna()
    if len(combined) < 30:
        return pd.Series(dtype=float)

    x = combined.values.astype(float)
    T, K = x.shape

    # Step 1: demean
    x = x - x.mean(axis=0)

    # EWMA volatility pre-whitening (RiskMetrics λ=0.94)
    # h[t] = λ·h[t-1] + (1-λ)·x[t-1]²  — initialised at sample variance
    h = np.var(x, axis=0, ddof=1).copy()
    eps = np.zeros_like(x)
    for t in range(T):
        eps[t] = x[t] / np.sqrt(h + 1e-12)
        h = ewma_lambda * h + (1.0 - ewma_lambda) * x[t] ** 2

    # Step 2: unconditional covariance of standardised residuals
    Q_bar = (eps.T @ eps) / T

    # Step 3: DCC(1,1) recursion on standardised residuals
    Q   = Q_bar.copy()
    dcc = np.zeros(T)

    for t in range(1, T):
        e = eps[t - 1]
        Q = (1 - a - b) * Q_bar + a * np.outer(e, e) + b * Q
        D_inv = np.diag(1.0 / np.sqrt(np.maximum(np.diag(Q), 1e-12)))
        R = D_inv @ Q @ D_inv
        # Clip to valid correlation range to guard against numerical drift
        dcc[t] = float(np.clip(R[0, 1], -1.0, 1.0))

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


# ── Regime Markov Chain ────────────────────────────────────────────────────

def regime_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """
    4x4 first-order Markov transition probability matrix.

    Counts day-to-day transitions i -> j across the full regime history,
    then normalises each row so that probabilities sum to 1.

    Returns a DataFrame indexed and columned by [0, 1, 2, 3].
    Rows with no observed transitions are left as uniform (0.25 each).
    """
    labels = [0, 1, 2, 3]
    counts = pd.DataFrame(0, index=labels, columns=labels, dtype=float)

    r = regimes.dropna().astype(int).values
    for t in range(len(r) - 1):
        counts.loc[r[t], r[t + 1]] += 1

    row_sums = counts.sum(axis=1)
    # Avoid division by zero - fill uniform for rows with zero observations
    prob = counts.div(row_sums.replace(0, np.nan), axis=0).fillna(0.25)
    return prob


def regime_steady_state(trans_matrix: pd.DataFrame) -> np.ndarray:
    """
    Stationary distribution pi such that pi @ P = pi, sum(pi) = 1.

    Solved via: (P^T - I)^T * pi = 0  with last row replaced by sum = 1.
    Returns length-4 array, values in [0, 1].
    """
    P = trans_matrix.values.astype(float)
    n = P.shape[0]

    A = (P.T - np.eye(n))
    A[-1, :] = 1.0          # normalization constraint
    b = np.zeros(n)
    b[-1] = 1.0

    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        pi = np.ones(n) / n  # fallback: uniform

    return np.clip(pi, 0, 1)


def regime_mean_first_passage(
    trans_matrix: pd.DataFrame,
    target: int = 3,
) -> dict[int, float]:
    """
    Mean First Passage Time (MFPT) to reach `target` regime from each other regime.

    Solves: for each i != target:
        MFPT[i] = 1 + sum_{k != target} P[i,k] * MFPT[k]
    i.e.  (I - P_sub) * m = 1_vector

    Returns {0: days, 1: days, 2: days, 3: 0.0}.
    Values > 5000 are capped to indicate "very unlikely" rather than infinity.
    """
    P = trans_matrix.values.astype(float)
    n = P.shape[0]
    states = [i for i in range(n) if i != target]

    P_sub = P[np.ix_(states, states)]
    rhs   = np.ones(len(states))
    A     = np.eye(len(states)) - P_sub

    try:
        mfpt_vals = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        mfpt_vals = np.full(len(states), np.nan)

    result: dict[int, float] = {target: 0.0}
    for i, s in enumerate(states):
        v = float(mfpt_vals[i])
        result[s] = min(v, 5000.0) if np.isfinite(v) and v > 0 else 5000.0

    return result


def regime_run_statistics(regimes: pd.Series) -> pd.DataFrame:
    """
    Consecutive run-length statistics per regime.

    Walks the regime series and records each unbroken run.
    Returns DataFrame with columns [mean, median, max, count] per regime (0-3).
    """
    runs: list[tuple[int, int]] = []
    r = regimes.dropna().astype(int)
    current = None
    count   = 0
    for val in r:
        if val == current:
            count += 1
        else:
            if current is not None:
                runs.append((current, count))
            current = int(val)
            count   = 1
    if current is not None:
        runs.append((current, count))

    if not runs:
        return pd.DataFrame(
            columns=["mean", "median", "max", "count"],
            index=pd.Index([0, 1, 2, 3], name="regime"),
        )

    df = pd.DataFrame(runs, columns=["regime", "duration"])
    stats = (
        df.groupby("regime")["duration"]
        .agg(mean="mean", median="median", max="max", count="count")
        .round(1)
        .reindex([0, 1, 2, 3])
        .fillna(0)
    )
    return stats


# ── Early Warning System ───────────────────────────────────────────────────

def early_warning_signals(
    avg_corr:    pd.Series,
    cmd_r:       pd.DataFrame,
    eq_r:        pd.DataFrame,
    regimes:     pd.Series,
    trans_matrix: pd.DataFrame,
) -> dict:
    """
    Five forward-looking early-warning signal scores (0-100) plus composite.

    Components:
    1. Correlation Velocity    (25%) - 30d OLS slope of avg_corr, normalised
    2. Vol Acceleration        (20%) - 10d change in 30d commodity vol z-score
    3. Regime Duration Pressure(20%) - current run length vs historical avg
    4. Equity Vol Trend        (15%) - 20d slope of equity realised vol
    5. Markov Crisis Probability(20%)- 1-step P(Crisis | current regime)

    Also returns top-5 historical analogue dates (Euclidean NN on 4-dim signature).
    """

    # ── Helper: OLS slope score ──────────────────────────────────────────────
    def _slope_score(series: pd.Series, window: int) -> float:
        s = series.dropna().iloc[-window:]
        if len(s) < window // 2:
            return 50.0
        x = np.arange(len(s), dtype=float)
        slope = np.polyfit(x, s.values, 1)[0]
        # Normalise: clip slope to ±3 historical std, map to 0-100
        hist_slopes = series.dropna().rolling(window).apply(
            lambda v: np.polyfit(np.arange(len(v)), v, 1)[0], raw=True,
        ).dropna()
        if hist_slopes.empty or hist_slopes.std() < 1e-10:
            return 50.0
        z = slope / (hist_slopes.std() * 3 + 1e-10)
        return float(np.clip(50 + z * 50, 0, 100))

    # ── Component 1: Correlation Velocity ───────────────────────────────────
    try:
        c1 = _slope_score(avg_corr, 30)
    except Exception:
        c1 = 50.0

    # ── Component 2: Vol Acceleration ───────────────────────────────────────
    try:
        energy_metals = ["WTI Crude Oil", "Brent Crude", "Natural Gas", "Gold", "Silver", "Copper"]
        cols = [c for c in energy_metals if c in cmd_r.columns]
        if cols:
            rv    = cmd_r[cols].rolling(30).std().mean(axis=1) * np.sqrt(252) * 100
            rv_mn = rv.rolling(252, min_periods=60).mean()
            rv_sd = rv.rolling(252, min_periods=60).std().replace(0, np.nan)
            z_vol = ((rv - rv_mn) / rv_sd).dropna()
            if len(z_vol) >= 10:
                delta = float(z_vol.iloc[-1] - z_vol.iloc[-10])
                # delta typically in [-3, 3]; map rising vol to high score
                c2 = float(np.clip(50 + delta / 0.12, 0, 100))
            else:
                c2 = 50.0
        else:
            c2 = 50.0
    except Exception:
        c2 = 50.0

    # ── Component 3: Regime Duration Pressure ───────────────────────────────
    try:
        run_stats = regime_run_statistics(regimes)
        current_r = int(regimes.dropna().iloc[-1])
        # Count current consecutive run
        r_vals = regimes.dropna().astype(int).values
        run_len = 0
        for v in reversed(r_vals):
            if v == current_r:
                run_len += 1
            else:
                break
        avg_run = float(run_stats.loc[current_r, "mean"]) if current_r in run_stats.index else 20.0
        # Pressure: 100 when run_len = 2x avg; 50 at avg; 0 at 0
        ratio = run_len / max(avg_run, 1.0)
        c3 = float(np.clip(50 * ratio, 0, 100))
    except Exception:
        c3 = 50.0

    # ── Component 4: Equity Vol Trend ───────────────────────────────────────
    try:
        eq_vol = eq_r.rolling(20).std().mean(axis=1) * np.sqrt(252) * 100
        c4 = _slope_score(eq_vol.dropna(), 20)
    except Exception:
        c4 = 50.0

    # ── Component 5: Markov Crisis Probability ──────────────────────────────
    try:
        current_r = int(regimes.dropna().iloc[-1])
        c5 = float(trans_matrix.loc[current_r, 3]) * 100
    except Exception:
        c5 = 20.0

    composite = float(np.clip(
        0.25 * c1 + 0.20 * c2 + 0.20 * c3 + 0.15 * c4 + 0.20 * c5,
        0, 100,
    ))

    # ── Signal labels and descriptions ──────────────────────────────────────
    _REGIME_NAMES = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
    current_regime = int(regimes.dropna().iloc[-1]) if not regimes.dropna().empty else 1
    signals = {
        "Correlation Velocity":     {"score": round(c1, 1),
            "desc": "Rate of change in cross-asset correlation (30d trend)"},
        "Vol Acceleration":         {"score": round(c2, 1),
            "desc": "10d change in commodity vol z-score (energy + metals)"},
        "Regime Duration Pressure": {"score": round(c3, 1),
            "desc": f"Current run length vs. historical avg ({_REGIME_NAMES.get(current_regime, '')} regime)"},
        "Equity Vol Trend":         {"score": round(c4, 1),
            "desc": "20d slope of equal-weight equity realised volatility"},
        "Markov Crisis P":          {"score": round(c5, 1),
            "desc": f"1-step Markov probability of entering Crisis from {_REGIME_NAMES.get(current_regime, '')}"},
    }

    # ── Historical analogues (Euclidean NN on 4-dim signature) ──────────────
    # Use composite-index rolling correlations (avoids sparse union-index issue)
    analogues: list[dict] = []
    try:
        energy_metals = ["WTI Crude Oil", "Brent Crude", "Natural Gas", "Gold", "Silver", "Copper"]
        cols = [c for c in energy_metals if c in cmd_r.columns]

        eq_vol_s  = eq_r.rolling(20).std().mean(axis=1) * np.sqrt(252) * 100
        cmd_vol_s = (cmd_r[cols].rolling(30).std().mean(axis=1) * np.sqrt(252) * 100
                     if cols else eq_vol_s.copy())

        # Composite-index cross-asset correlation (pairwise, avoids union-index NaN issue)
        eq_idx  = eq_r.mean(axis=1)
        cmd_idx = cmd_r[cols].mean(axis=1) if cols else cmd_r.mean(axis=1)
        pair_df = pd.concat([eq_idx.rename("eq"), cmd_idx.rename("cmd")], axis=1).dropna()
        fast_corr = pair_df["eq"].rolling(60, min_periods=30).corr(pair_df["cmd"]).abs()

        # Build daily signature on the intersection of all series
        sig_df = pd.concat([
            eq_vol_s.rename("eq_vol"),
            cmd_vol_s.rename("cmd_vol"),
            fast_corr.rename("corr_proxy"),
        ], axis=1).dropna()

        # Attach regime column via nearest-date join (reindex + ffill)
        reg_daily = regimes.reindex(sig_df.index, method="ffill")
        sig_df["regime"] = reg_daily

        if len(sig_df) >= 60:
            # Distance on the 3 vol/corr features only (no regime in metric)
            feat_cols = ["eq_vol", "cmd_vol", "corr_proxy"]
            feat_df   = sig_df[feat_cols]
            norm = (feat_df - feat_df.min()) / (feat_df.max() - feat_df.min() + 1e-10)
            current_vec = norm.iloc[-1].values

            dists = np.sqrt(((norm.values - current_vec) ** 2).sum(axis=1))
            dist_s = pd.Series(dists, index=sig_df.index)
            # Exclude recent 90 days to avoid trivial self-matches
            dist_s = dist_s.iloc[:-90] if len(dist_s) > 90 else dist_s.iloc[:-1]
            dist_s = dist_s.dropna()

            # Regime lookup – fill NaN with Normal (1) for pre-history dates
            regime_at = reg_daily.fillna(1).astype(int)

            picked: list[pd.Timestamp] = []
            for dt in dist_s.sort_values().index:
                if all(abs((dt - p).days) >= 90 for p in picked):
                    picked.append(dt)
                if len(picked) >= 5:
                    break

            for dt in picked:
                # Get regime at dt + 30/60/90 via nearest-available in regime_at
                def _reg_at_offset(base_dt: pd.Timestamp, offset_days: int) -> int:
                    target = base_dt + pd.Timedelta(days=offset_days)
                    idx = regime_at.index
                    pos = idx.searchsorted(target)
                    pos = min(pos, len(idx) - 1)
                    return int(regime_at.iloc[pos])

                r_now  = _reg_at_offset(dt, 0)
                r_30   = _reg_at_offset(dt, 30)
                r_60   = _reg_at_offset(dt, 60)
                r_90   = _reg_at_offset(dt, 90)
                similarity = max(0.0, 100.0 - dist_s[dt] * 100)
                analogues.append({
                    "date":    dt.strftime("%b %Y"),
                    "regime":  _REGIME_NAMES.get(r_now, str(r_now)),
                    "r30":     _REGIME_NAMES.get(r_30,  str(r_30)),
                    "r60":     _REGIME_NAMES.get(r_60,  str(r_60)),
                    "r90":     _REGIME_NAMES.get(r_90,  str(r_90)),
                    "sim":     round(similarity, 1),
                    "r30_int": r_30,
                    "r60_int": r_60,
                    "r90_int": r_90,
                })
    except Exception:
        pass

    return {
        "signals":       signals,
        "composite":     round(composite, 1),
        "current_regime": current_regime,
        "current_regime_name": _REGIME_NAMES.get(current_regime, "Unknown"),
        "analogues":     analogues,
    }

"""
Market-model event study engine.

Methodology:
  1. Estimation window: [-est_before, -est_gap] trading days relative to event start.
     OLS: R_asset = alpha + beta * R_benchmark + epsilon.
  2. Event window: [-pre, +post] trading days.
     AR_t = R_asset_t - (alpha + beta * R_benchmark_t)
     CAR = cumsum(AR_t) over the event window.
  3. CAAR across N events: cross-sectional mean of per-event CARs.
     t-stat  = CAAR / (std(CAR_i) / sqrt(N))
     Sign test: proportion of positive CARs vs. H0 = 0.5
     Bootstrap CI: resample {CAR_i} with replacement, B=1000.

All index-based operations use integer positional slicing on a pre-sorted
trading-day index so holiday calendars and weekend gaps are handled correctly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats


def _trading_day_offsets(
    idx: pd.DatetimeIndex,
    anchor: pd.Timestamp,
    pre: int,
    post: int,
) -> tuple[int, int, int]:
    """
    Returns (anchor_pos, window_start_pos, window_end_pos) in the DatetimeIndex.
    anchor_pos is the first position >= anchor date.
    Window: [anchor_pos - pre, anchor_pos + post + 1)
    """
    pos = idx.searchsorted(anchor)
    start = max(0, pos - pre)
    end = min(len(idx), pos + post + 1)
    return int(pos), int(start), int(end)


def market_model_params(
    asset_r: pd.Series,
    bench_r: pd.Series,
    event_date: pd.Timestamp,
    est_before: int = 120,
    est_gap: int = 20,
) -> dict:
    """
    OLS alpha/beta over estimation window ending `est_gap` days before event_date.
    Returns: {"alpha": float, "beta": float, "sigma": float, "n_obs": int}
    """
    idx = bench_r.index
    pos = idx.searchsorted(event_date)
    est_end = max(0, pos - est_gap)
    est_start = max(0, est_end - est_before)
    if est_end - est_start < 30:
        return {"alpha": 0.0, "beta": 1.0, "sigma": 0.01, "n_obs": 0}

    bench_est = bench_r.iloc[est_start:est_end].dropna()
    asset_est = asset_r.reindex(bench_est.index).dropna()
    common = bench_est.reindex(asset_est.index).dropna()
    asset_common = asset_est.reindex(common.index).dropna()
    common = common.reindex(asset_common.index)

    n = len(common)
    if n < 20:
        return {"alpha": 0.0, "beta": 1.0, "sigma": 0.01, "n_obs": n}

    X = np.column_stack([np.ones(n), common.values])
    y = asset_common.values
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coeffs
        sigma = float(np.std(residuals, ddof=2))
        return {"alpha": float(coeffs[0]), "beta": float(coeffs[1]),
                "sigma": sigma, "n_obs": n}
    except Exception:
        return {"alpha": 0.0, "beta": 1.0, "sigma": 0.01, "n_obs": 0}


def compute_car(
    asset_r: pd.Series,
    bench_r: pd.Series,
    params: dict,
    event_date: pd.Timestamp,
    pre: int = 5,
    post: int = 20,
) -> pd.Series:
    """
    Cumulative abnormal returns for one event.
    Returns a Series with integer index [-pre, ..., +post] and values in decimal
    (multiply by 100 for percent).  Missing days → 0 AR.
    """
    alpha, beta = params["alpha"], params["beta"]
    idx = bench_r.index
    pos, start, end = _trading_day_offsets(idx, event_date, pre, post)
    anchor_in_window = pos - start  # position of day 0 within the slice

    bench_window = bench_r.iloc[start:end]
    asset_window = asset_r.reindex(bench_window.index).fillna(0.0)

    ar = asset_window.values - (alpha + beta * bench_window.values)
    car = np.cumsum(ar)

    n_actual = len(car)
    full_length = pre + post + 1
    if n_actual < full_length:
        car = np.concatenate([car, np.full(full_length - n_actual, np.nan)])
    else:
        car = car[:full_length]

    day_idx = np.arange(-pre, post + 1)
    return pd.Series(car, index=day_idx, name=asset_r.name)


def caar_stats(
    cars: list[pd.Series],
    n_boot: int = 1000,
    ci: float = 0.95,
) -> dict:
    """
    Aggregate CAR series to CAAR with t-stat, sign test, and bootstrap CI.

    cars: list of pd.Series, each indexed [-pre..+post], decimal returns.
    Returns dict with:
      caar     : pd.Series (decimal)
      lower/upper: pd.Series (bootstrap CI, decimal)
      tstat    : float
      pval_t   : float (two-sided t-test)
      pval_sign: float (sign test, one-sided: more positives than chance)
      n        : int (events with valid data)
    """
    if not cars:
        empty = pd.Series(dtype=float)
        return {"caar": empty, "lower": empty, "upper": empty,
                "tstat": np.nan, "pval_t": np.nan, "pval_sign": np.nan, "n": 0}

    # Align to common day index
    combined = pd.concat(cars, axis=1)
    n = combined.shape[1]
    caar = combined.mean(axis=1)
    cross_std = combined.std(axis=1, ddof=1)
    se = cross_std / np.sqrt(n)

    # t-stat at terminal day (+post)
    terminal = caar.iloc[-1]
    se_terminal = float(se.iloc[-1]) if se.iloc[-1] > 0 else np.nan
    tstat = float(terminal / se_terminal) if se_terminal and not np.isnan(se_terminal) else np.nan
    pval_t = float(2 * (1 - _scipy_stats.t.cdf(abs(tstat), df=n - 1))) if not np.isnan(tstat) else np.nan

    # Sign test: how many CARs at terminal are positive?
    terminal_cars = combined.iloc[-1].dropna()
    n_pos = int((terminal_cars > 0).sum())
    n_valid = len(terminal_cars)
    pval_sign = float(_scipy_stats.binomtest(n_pos, n_valid, 0.5, alternative="greater").pvalue) if n_valid > 0 else np.nan

    # Bootstrap CI over the full time series
    rng = np.random.default_rng(42)
    boot_means = np.zeros((n_boot, len(caar)))
    vals = combined.values.T  # shape (n_events, n_days)
    for b in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot_means[b] = vals[sample_idx].mean(axis=0)
    alpha_half = (1 - ci) / 2
    lower = pd.Series(np.nanpercentile(boot_means, alpha_half * 100, axis=0),
                      index=caar.index)
    upper = pd.Series(np.nanpercentile(boot_means, (1 - alpha_half) * 100, axis=0),
                      index=caar.index)

    return {
        "caar": caar, "lower": lower, "upper": upper,
        "tstat": tstat, "pval_t": pval_t, "pval_sign": pval_sign, "n": n,
    }


def run_event_study(
    assets: list[str],
    all_returns: pd.DataFrame,
    events: list[dict],
    benchmark: str = "S&P 500",
    est_before: int = 120,
    est_gap: int = 20,
    pre: int = 5,
    post: int = 20,
    n_boot: int = 500,
) -> dict:
    """
    Run a full market-model event study.

    Returns:
      per_event:  {event_name: {asset_name: pd.Series(car, index=-pre..+post)}}
      caar:       {asset_name: {"caar": Series, "lower": Series, "upper": Series,
                                "tstat": float, "pval_t": float, "pval_sign": float,
                                "n": int}}
      params:     {event_name: {asset_name: {"alpha": float, "beta": float, ...}}}
    """
    if benchmark not in all_returns.columns:
        benchmark = all_returns.columns[0]

    bench_r = all_returns[benchmark]
    per_event: dict[str, dict[str, pd.Series]] = {}
    cars_by_asset: dict[str, list[pd.Series]] = {a: [] for a in assets if a in all_returns.columns}
    params_out: dict[str, dict] = {}

    valid_assets = [a for a in assets if a in all_returns.columns and a != benchmark]

    for ev in events:
        ev_name = ev["name"]
        ev_date = pd.Timestamp(ev["start"])
        per_event[ev_name] = {}
        params_out[ev_name] = {}

        for asset in valid_assets:
            asset_r = all_returns[asset]
            p = market_model_params(asset_r, bench_r, ev_date, est_before, est_gap)
            params_out[ev_name][asset] = p
            if p["n_obs"] < 20:
                continue
            car = compute_car(asset_r, bench_r, p, ev_date, pre, post)
            per_event[ev_name][asset] = car
            cars_by_asset[asset].append(car)

    caar_out: dict[str, dict] = {}
    for asset in valid_assets:
        if not cars_by_asset.get(asset):
            continue
        caar_out[asset] = caar_stats(cars_by_asset[asset], n_boot=n_boot)

    return {"per_event": per_event, "caar": caar_out, "params": params_out,
            "pre": pre, "post": post, "benchmark": benchmark}

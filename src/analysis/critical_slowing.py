"""
Critical Slowing Down — regime-transition early-warning signals.

Tipping-point theory (Scheffer et al. 2009, Nature; Dakos et al. 2012, PLoS ONE)
observes that as a dynamical system approaches a critical transition, it recovers
more and more slowly from small perturbations. Two statistical fingerprints of
that "critical slowing down" (CSD) appear BEFORE the system actually flips:

  1. Rising lag-1 autocorrelation — AR(1) coefficient climbs toward 1
     (the system's memory lengthens; shocks decay slower).
  2. Rising variance — fluctuations grow as the basin of attraction flattens.

This module applies that machinery — imported from ecology/climate tipping-point
research — to a market *driver* series (the Diebold-Yilmaz connectedness index or
the average cross-asset correlation). The goal is to flag an impending correlation
regime transition (Normal → Elevated → Crisis) some days BEFORE the terminal's
Markov regime classifier (src/analysis/correlations.detect_correlation_regime)
actually switches state.

Method, per Dakos et al. (2012):
  · Detrend the driver with a rolling mean (bandwidth `detrend_bw`) to isolate
    fluctuations from the slow-moving level.
  · On the residuals, compute rolling-window AR(1), variance and skewness.
  · The *early-warning signal* is not the level of an indicator but its upward
    TREND — quantified with Kendall's rank correlation (tau) over a trailing
    window. tau → +1 means a strong, monotonic rise = strong warning.
  · A composite blends the standardized AR(1) and variance indicators into a
    single 0–100 warning level.

Nothing here uses look-ahead beyond the standard in-sample standardization used
throughout the terminal's analog engines; every indicator at day t is a function
of data ≤ t. Validation against realized regime flips is provided by
`evaluate_lead_time`, which is honest about false alarms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Regime codes emitted by detect_correlation_regime.
DECORRELATED, NORMAL, ELEVATED, CRISIS = 0, 1, 2, 3


# ── Core indicators ──────────────────────────────────────────────────────────

def _ar1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D window (AR(1) coefficient proxy)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 4:
        return np.nan
    a, b = x[:-1], x[1:]
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _kendall_tau(y: np.ndarray) -> float:
    """Kendall rank correlation of a series against time — trend strength in [-1, 1]."""
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    y = y[mask]
    if y.size < 5 or np.allclose(y, y[0]):
        return np.nan
    tau, _ = stats.kendalltau(np.arange(y.size), y)
    return float(tau) if tau is not None and not np.isnan(tau) else np.nan


def detrend(series: pd.Series, bandwidth: int = 30) -> pd.Series:
    """Residuals after removing a centred rolling-mean trend of width `bandwidth`.

    Isolates the fast fluctuations CSD acts on from the slow level of the driver,
    so a rising indicator reflects changing dynamics, not a drifting mean.
    """
    s = series.astype(float)
    trend = s.rolling(bandwidth, min_periods=max(5, bandwidth // 3), center=True).mean()
    return (s - trend).dropna()


def compute_ews(
    driver: pd.Series,
    detrend_bw: int = 30,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling early-warning indicators on a driver series.

    Returns a DataFrame indexed like the (detrended) driver with columns:
      · driver     — the raw input, aligned
      · residual   — detrended driver
      · ar1        — rolling lag-1 autocorrelation (rising = warning)
      · variance   — rolling variance of residuals (rising = warning)
      · skew       — rolling skewness (asymmetry grows near a transition)
      · ar1_z      — expanding z-score of ar1
      · var_z      — expanding z-score of variance
      · composite  — 0–100 warning level: logistic blend of ar1_z and var_z
    """
    driver = driver.dropna()
    out = pd.DataFrame(index=driver.index)
    if len(driver) < window + detrend_bw:
        return pd.DataFrame(
            columns=["driver", "residual", "ar1", "variance", "skew",
                     "ar1_z", "var_z", "composite"]
        )

    resid = detrend(driver, detrend_bw)
    mp = max(10, window // 2)
    ar1 = resid.rolling(window, min_periods=mp).apply(_ar1, raw=True)
    variance = resid.rolling(window, min_periods=mp).var()
    skew = resid.rolling(window, min_periods=mp).skew()

    df = pd.DataFrame({
        "driver": driver.reindex(resid.index),
        "residual": resid,
        "ar1": ar1,
        "variance": variance,
        "skew": skew,
    }).dropna(subset=["ar1", "variance"])

    # Expanding z-scores — each day standardized against its own history only
    # (no forward leakage). min_periods keeps early estimates from exploding.
    def _exp_z(s: pd.Series) -> pd.Series:
        mu = s.expanding(min_periods=mp).mean()
        sd = s.expanding(min_periods=mp).std().replace(0, np.nan)
        return ((s - mu) / sd).clip(-4, 4)

    df["ar1_z"] = _exp_z(df["ar1"])
    df["var_z"] = _exp_z(df["variance"])
    # Composite: average the two CSD z-scores, squash to 0–100 with a logistic.
    blend = df[["ar1_z", "var_z"]].mean(axis=1)
    df["composite"] = (100.0 / (1.0 + np.exp(-1.3 * blend))).round(1)
    return df.dropna(subset=["composite"])


def trend_tau(ews: pd.DataFrame, window: int = 40) -> pd.DataFrame:
    """Trailing Kendall-tau trend of each CSD indicator — the actual warning.

    A positive tau on ar1 AND variance is the classic CSD signature. Returns
    columns ar1_tau, var_tau, composite_tau aligned to `ews`.
    """
    mp = max(8, window // 2)
    out = pd.DataFrame(index=ews.index)
    for src, dst in (("ar1", "ar1_tau"), ("variance", "var_tau"),
                     ("composite", "composite_tau")):
        out[dst] = ews[src].rolling(window, min_periods=mp).apply(_kendall_tau, raw=True)
    return out


# ── Validation against realized regime flips ─────────────────────────────────

def detect_regime_flips(regime: pd.Series, into=(ELEVATED, CRISIS)) -> pd.Series:
    """Dates where the regime steps UP into a stressed state.

    Returns a Series indexed by flip date whose value is the regime entered.
    A flip counts only when the new regime is in `into` and strictly higher than
    the previous regime (entering, not sitting in, stress).
    """
    r = regime.dropna().astype(int)
    prev = r.shift(1)
    flips = r[(r > prev) & (r.isin(into))]
    return flips


def evaluate_lead_time(
    ews: pd.DataFrame,
    regime: pd.Series,
    alert_threshold: float = 62.0,
    max_lookback: int = 60,
    into=(ELEVATED, CRISIS),
) -> dict:
    """How much warning did the composite give before each realized regime flip?

    For every flip, scan back up to `max_lookback` CALENDAR days for the first day
    the composite crossed `alert_threshold` in the run leading up to it. That gap
    is the lead time. Also counts alert episodes that were NOT followed by a flip
    (false alarms) so the signal is judged honestly, not cherry-picked.

    Returns a dict with per-flip lead times, hit rate, median lead, and a false-
    alarm rate over independent alert episodes. All day counts are calendar days.
    """
    empty = {
        "flips": [], "n_flips": 0, "n_caught": 0, "hit_rate": np.nan,
        "median_lead": np.nan, "mean_lead": np.nan,
        "n_alert_episodes": 0, "n_false_alarms": 0, "false_alarm_rate": np.nan,
    }
    if ews.empty or "composite" not in ews:
        return empty

    comp = ews["composite"].dropna()
    if comp.empty:
        return empty
    flips = detect_regime_flips(regime, into=into)
    flips = flips[flips.index.isin(comp.index) | (flips.index > comp.index.min())]

    idx = comp.index
    above = comp >= alert_threshold

    per_flip = []
    caught = 0
    leads = []
    for fdate, target in flips.items():
        # window of composite readings within max_lookback calendar days before flip
        win = comp[(idx < fdate) & (idx >= fdate - pd.Timedelta(days=max_lookback))]
        if win.empty:
            per_flip.append({"date": fdate, "regime": int(target),
                             "lead_days": None, "caught": False})
            continue
        alert_days = win.index[win.values >= alert_threshold]
        if len(alert_days):
            first = alert_days[0]
            lead = (fdate - first).days
            caught += 1
            leads.append(lead)
            per_flip.append({"date": fdate, "regime": int(target),
                             "lead_days": int(lead), "caught": True})
        else:
            per_flip.append({"date": fdate, "regime": int(target),
                             "lead_days": None, "caught": False})

    # Independent alert episodes: contiguous runs above threshold. An episode is
    # a "hit" if any flip occurs within max_lookback days after it starts.
    episodes = []
    run_start = None
    for t, hot in above.items():
        if hot and run_start is None:
            run_start = t
        elif not hot and run_start is not None:
            episodes.append(run_start)
            run_start = None
    if run_start is not None:
        episodes.append(run_start)

    flip_dates = list(flips.index)
    false_alarms = 0
    for ep in episodes:
        hit = any(0 <= (fd - ep).days <= max_lookback for fd in flip_dates)
        if not hit:
            false_alarms += 1

    n_flips = len(per_flip)
    return {
        "flips": per_flip,
        "n_flips": n_flips,
        "n_caught": caught,
        "hit_rate": (caught / n_flips) if n_flips else np.nan,
        "median_lead": float(np.median(leads)) if leads else np.nan,
        "mean_lead": float(np.mean(leads)) if leads else np.nan,
        "n_alert_episodes": len(episodes),
        "n_false_alarms": false_alarms,
        "false_alarm_rate": (false_alarms / len(episodes)) if episodes else np.nan,
    }


def latest_reading(ews: pd.DataFrame, taus: pd.DataFrame,
                   alert_threshold: float = 62.0) -> dict:
    """Headline snapshot for the radar: current composite, indicator state, verdict."""
    if ews.empty:
        return {"status": "INSUFFICIENT DATA"}
    last = ews.iloc[-1]
    tau_last = taus.iloc[-1] if not taus.empty else pd.Series(dtype=float)
    ar1_rising = tau_last.get("ar1_tau", np.nan)
    var_rising = tau_last.get("var_tau", np.nan)
    comp = float(last["composite"])

    # Both CSD indicators trending up + composite above bar = building transition.
    both_rising = (np.nan_to_num(ar1_rising) > 0.1) and (np.nan_to_num(var_rising) > 0.1)
    if comp >= alert_threshold and both_rising:
        status = "TRANSITION RISK BUILDING"
    elif comp >= alert_threshold or both_rising:
        status = "WATCH — MIXED SIGNAL"
    else:
        status = "STABLE"

    return {
        "status": status,
        "date": ews.index[-1],
        "composite": comp,
        "ar1": float(last["ar1"]),
        "ar1_z": float(last["ar1_z"]),
        "variance": float(last["variance"]),
        "var_z": float(last["var_z"]),
        "ar1_tau": float(ar1_rising) if not np.isnan(ar1_rising) else None,
        "var_tau": float(var_rising) if not np.isnan(var_rising) else None,
        "both_rising": bool(both_rising),
    }

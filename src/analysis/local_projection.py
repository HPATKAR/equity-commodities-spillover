"""
Jordà (2005) local-projection impulse-response functions.

Shock identification: AR(p) residual on daily GDELT news-volume series for each
conflict (log1p-transformed, BIC lag selection p=1..5, standardised to unit
variance). The shock is orthogonal to conflict's own past — it captures
unexpected escalation, not anticipated intensity level.

Regression at horizon h (h = 0..20 trading days):

  cum_ret_{a,t→t+h} = α_h + β_h · shock_t
                     + Σ_{j=1}^{p_s} γ_{h,j} · shock_{t-j}
                     + Σ_{j=1}^{p_r} δ_{h,j} · ret_{a,t-j}  +  ε_{h,t}

β_h is the h-day cumulative IRF coefficient.
SEs: Newey-West HAC, bandwidth = max(h, 2) — covers the MA(h-1) structure
introduced by using overlapping h-step cumulative returns as the dependent
variable (Jordà 2005, footnote 4; also Montiel Olea & Plagborg-Møller 2021).

IRF units: log-return % per 1-σ shock (shock series standardised before entry).

References:
  Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local
    Projections. AER, 95(1), 161–182.
  Montiel Olea, J.L. & Plagborg-Møller, M. (2021). Local Projection Inference
    is Simpler and More Robust Than You Think. Econometrica, 89(4), 1789–1823.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

_log = logging.getLogger(__name__)

_MAX_H    = 20   # maximum IRF horizon (trading days)
_N_LAG_S  = 4   # lags of shock included as controls
_N_LAG_R  = 2   # lags of asset return included as controls
_MIN_OBS  = 80  # minimum obs at any horizon to trust the OLS
_AR_MAX_P = 5   # BIC search ceiling for shock AR pre-filter

# CONFLICTS registry IDs → GDELT query map IDs (the two systems use different keys)
_CONFLICT_GDELT_ID: dict[str, str] = {
    "israel_gaza":   "israel_hamas",
    "iran_conflict": "iran_regional",
}


def _gdelt_id(conflict_id: str) -> str:
    """Translate CONFLICTS registry ID to GDELT query map key."""
    return _CONFLICT_GDELT_ID.get(conflict_id, conflict_id)


# ── GDELT volume fetch (cached) ────────────────────────────────────────────────

@st.cache_data(ttl=21600, show_spinner=False, max_entries=24)
def _fetch_gdelt_volume(conflict_id: str, timespan: str) -> pd.Series:
    """
    Fetch daily GDELT article-volume time-series for one conflict.

    `conflict_id` is already translated to the GDELT key before calling here.
    Returns a pd.Series indexed by calendar date; empty Series on failure.
    Cached 6 h to match GDELT update cadence.
    Retries once with 5-s backoff on 429.
    """
    _empty = pd.Series(dtype=float, name="volume")
    try:
        from src.data.gdelt import _GDELT_CONFLICT_QUERIES, _GDELT_DOC_API, _HEADERS
        import requests

        if conflict_id not in _GDELT_CONFLICT_QUERIES:
            return _empty

        cfg = _GDELT_CONFLICT_QUERIES[conflict_id]
        params = {
            "query":      f'({cfg["query"]}) theme:{cfg["theme"]}',
            "mode":       "TimelineVol",
            "format":     "json",
            "TIMESPAN":   timespan,
            "MAXRECORDS": 1000,
        }

        for _attempt in range(2):
            try:
                resp = requests.get(_GDELT_DOC_API, params=params,
                                    timeout=30, headers=_HEADERS)
                if resp.status_code == 429:
                    if _attempt == 0:
                        time.sleep(6.0)   # GDELT hard rate limit: 1 req/5s
                        continue
                    return _empty
                resp.raise_for_status()
                data = resp.json()
                break
            except (ValueError, requests.exceptions.JSONDecodeError):
                # GDELT returned empty/HTML body — timespan not supported
                return _empty
        else:
            return _empty

        timeline = data.get("timeline", [{}])[0].get("data", [])
        if not timeline:
            return _empty

        rows = []
        for pt in timeline:
            try:
                raw = str(pt.get("date", "")).rstrip("Z")   # strip trailing Z
                dt  = pd.to_datetime(raw, format="%Y%m%dT%H%M%S")
                vol = float(pt.get("value", 0) or 0)
                rows.append({"dt": dt, "volume": vol})
            except Exception:
                continue

        if not rows:
            return _empty

        df    = pd.DataFrame(rows).set_index("dt").sort_index()
        daily = df["volume"].resample("1D").sum()
        return daily

    except Exception as exc:
        _log.warning("GDELT LP volume fetch failed (%s, %s): %s", conflict_id, timespan, exc)
        return pd.Series(dtype=float, name="volume")


# ── Shock construction ─────────────────────────────────────────────────────────

def _build_shock_series(
    conflict_id: str,
    trading_idx: pd.DatetimeIndex,
    ar_max_p: int = _AR_MAX_P,
) -> Optional[pd.Series]:
    """
    Build standardised AR-residual shock series aligned to trading_idx.

    1. GDELT daily volumes — cascade: 2y → 1y → 3m until ≥ 90 days
    2. Forward-fill gaps (weekends, low-news days)
    3. Reindex to trading days
    4. log1p-transform (stabilises count-data variance)
    5. BIC AR(p), p = 1..ar_max_p; extract residuals
    6. Standardise to zero mean, unit std

    Returns None if < 120 trading-day observations remain after AR trimming.
    """
    gdelt_id = _gdelt_id(conflict_id)   # translate registry ID → GDELT key
    # Single TIMESPAN=1y fetch — GDELT enforces 5-s/request; no cascade here.
    # Inter-conflict sleep is enforced at the lp_irf_all_conflicts call site.
    daily = _fetch_gdelt_volume(gdelt_id, "1y")

    if daily.empty or daily.sum() == 0 or len(daily) < 60:
        return None

    # Fill calendar gaps then align to trading-day index
    daily_filled = daily.resample("D").sum().fillna(0.0)
    aligned = daily_filled.reindex(trading_idx, method="ffill").dropna()

    if len(aligned) < 120:
        return None

    s = np.log1p(aligned.values.astype(float))
    if np.std(s, ddof=1) < 1e-8:
        return None   # degenerate (all zeros)

    n = len(s)

    # BIC AR(p) selection ──────────────────────────────────────────────────────
    best_bic, best_p = np.inf, 1
    for p in range(1, ar_max_p + 1):
        if n - p < 40:
            break
        y   = s[p:]
        Xp  = np.column_stack([np.ones(len(y))] + [s[p - j - 1: n - j - 1] for j in range(p)])
        try:
            coef, res, _, _ = np.linalg.lstsq(Xp, y, rcond=None)
            sigma2 = float(res[0] / len(y)) if len(res) else float(np.mean((y - Xp @ coef) ** 2))
            if sigma2 <= 0:
                continue
            bic = len(y) * np.log(sigma2) + (p + 1) * np.log(len(y))
            if bic < best_bic:
                best_bic, best_p = bic, p
        except Exception:
            continue

    p  = best_p
    y  = s[p:]
    Xp = np.column_stack([np.ones(len(y))] + [s[p - j - 1: n - j - 1] for j in range(p)])
    try:
        coef, _, _, _ = np.linalg.lstsq(Xp, y, rcond=None)
        resid = y - Xp @ coef
    except Exception:
        return None

    resid_std = float(np.std(resid, ddof=1))
    if resid_std < 1e-8:
        return None

    return pd.Series(
        resid / resid_std,
        index=aligned.index[p:],
        name=f"{conflict_id}_shock",
    )


# ── Core LP-IRF estimator ──────────────────────────────────────────────────────

@st.cache_data(ttl=21600, show_spinner=False, max_entries=24)
def compute_lp_irf(
    _shock: pd.Series,
    _returns: pd.DataFrame,
    conflict_id: str,
    max_h: int = _MAX_H,
    n_lags_shock: int = _N_LAG_S,
    n_lags_ret: int = _N_LAG_R,
    _n_shock: int = 0,   # cache-invalidation proxy: len(_shock)
    _n_rows: int = 0,    # cache-invalidation proxy: len(_returns)
) -> pd.DataFrame:
    """
    Jordà LP-IRF for every asset column in _returns.

    At each horizon h:
      y_{t,h} = Σ_{k=0}^{h} ret_{a,t+k}  (h-day cumulative log-return, %)
      X_{t}   = [1, shock_t, shock_{t-1..p_s}, ret_{t-1..p_r}]
      Fit OLS with Newey-West HAC (bandwidth = max(h, 2))

    Returns DataFrame with columns:
        horizon  : int     (0..max_h)
        asset    : str
        coef     : float   (IRF in %, β_h × 100)
        ci_lo    : float   (90% NW lower bound)
        ci_hi    : float   (90% NW upper bound)
        pval     : float   (two-sided p-value for β_h = 0)
        nobs     : int
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        _log.error("statsmodels not available for LP-IRF")
        return pd.DataFrame()

    records: list[dict] = []
    shock   = _shock.dropna()
    max_lag = max(n_lags_shock, n_lags_ret)

    for asset in _returns.columns:
        ret    = _returns[asset].dropna()
        common = shock.index.intersection(ret.index)
        if len(common) < 120:
            continue

        s = shock.reindex(common).values.astype(float)
        r = ret.reindex(common).values.astype(float)
        n = len(s)

        for h in range(max_h + 1):
            t_start = max_lag
            t_end   = n - h - 1
            n_t     = t_end - t_start + 1
            if n_t < _MIN_OBS:
                continue

            t_idx = np.arange(t_start, t_end + 1)

            # h-day cumulative log-return (×100 → %)
            if h == 0:
                y = r[t_idx] * 100.0
            else:
                y = np.array([r[t: t + h + 1].sum() for t in t_idx]) * 100.0

            # Regressors: 1 + shock_t + p_s shock lags + p_r return lags
            cols: list[np.ndarray] = [np.ones(n_t), s[t_idx]]
            for j in range(1, n_lags_shock + 1):
                cols.append(s[t_idx - j])
            for j in range(1, n_lags_ret + 1):
                cols.append(r[t_idx - j] * 100.0)
            X = np.column_stack(cols)

            try:
                # Bandwidth = h (canonical Jordà 2005 / MOP 2021 choice).
                # At h=0 there is no window overlap so no MA structure in
                # residuals — use HC3 (heteroscedasticity-only). At h≥1 the
                # dependent variable is a sum of h+1 overlapping returns,
                # inducing MA(h-1) serial correlation → NW with maxlags=h.
                if h == 0:
                    res = sm.OLS(y, X).fit(cov_type="HC3")
                else:
                    res = sm.OLS(y, X).fit(
                        cov_type="HAC",
                        cov_kwds={"maxlags": h, "use_correction": True},
                    )
                # conf_int returns ndarray (k, 2) or DataFrame depending on sm version
                ci = np.asarray(res.conf_int(alpha=0.10))
                records.append({
                    "horizon": h,
                    "asset":   asset,
                    "coef":    round(float(res.params[1]),   4),
                    "ci_lo":   round(float(ci[1, 0]),        4),
                    "ci_hi":   round(float(ci[1, 1]),        4),
                    "se":      round(float(res.bse[1]),      5),
                    "pval":    round(float(res.pvalues[1]),   4),
                    "nobs":    n_t,
                })
            except Exception:
                continue

    return pd.DataFrame(records) if records else pd.DataFrame()


# ── Portfolio-level entry point ────────────────────────────────────────────────

@st.cache_data(ttl=21600, show_spinner=False, max_entries=2)
def lp_irf_all_conflicts(
    _eq_r: pd.DataFrame,
    _cmd_r: pd.DataFrame,
    _n_rows: int = 0,
    max_h: int = _MAX_H,
) -> dict[str, dict]:
    """
    Compute LP-IRFs for all 6 tracked conflicts.

    For each conflict the affected_equities + affected_commodities from the
    CONFLICTS registry are used as response assets (filtered to columns that
    are present in the return data).

    Returns:
        {conflict_id: {
            "irf":         pd.DataFrame,  # columns: horizon, asset, coef, ci_lo, ci_hi, pval, nobs
            "conflict":    dict,           # raw conflict entry from CONFLICTS registry
            "n_obs_shock": int,
            "ar_ok":       bool,
            "error":       str | None,
        }}
    """
    from src.data.config import CONFLICTS

    all_r       = pd.concat([_eq_r, _cmd_r], axis=1).sort_index()
    trading_idx = all_r.dropna(how="all").index

    out: dict[str, dict] = {}
    for _ci, conflict in enumerate(CONFLICTS):
        cid      = conflict["id"]
        ae       = [a for a in conflict.get("affected_equities",    []) if a in all_r.columns]
        ac       = [a for a in conflict.get("affected_commodities", []) if a in all_r.columns]
        affected = ae + ac

        base = {"conflict": conflict, "irf": pd.DataFrame(),
                "n_obs_shock": 0, "ar_ok": False, "error": None}

        if not affected:
            out[cid] = {**base, "error": "No affected assets in loaded return data"}
            continue

        # 6-s gap between conflicts — GDELT enforces 1 request/5 s.
        # Cached results return instantly so this only costs time on cold start.
        if _ci > 0:
            time.sleep(6.0)

        shock = _build_shock_series(cid, trading_idx)

        if shock is None or len(shock) < 100:
            out[cid] = {**base,
                        "n_obs_shock": len(shock) if shock is not None else 0,
                        "error": "Insufficient GDELT history (<100 trading-day obs)"}
            continue

        irf_df = compute_lp_irf(
            shock, all_r[affected],
            conflict_id=cid, max_h=max_h,
            _n_shock=len(shock), _n_rows=len(all_r),
        )
        out[cid] = {
            **base,
            "irf":         irf_df,
            "n_obs_shock": len(shock),
            "ar_ok":       True,
            "error":       None if not irf_df.empty else "LP regression returned no results",
        }

    return out

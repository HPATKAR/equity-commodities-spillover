"""
Walk-forward vectorized backtest engine for regime-triggered trade ideas.

Design
------
* walk_forward_regimes : generates OOS regime labels with no look-ahead bias.
  At each fold boundary (every `fold_days` trading days after the first
  `min_train_days`), percentile thresholds are fit on the training window only,
  then the same hysteresis walk-forward from detect_correlation_regime is
  applied to that OOS fold.

* vectorized_backtest : single-period engine.
  - Position array is built in one O(T) pass (not O(n_trades × T)).
  - Daily P&L is computed with numpy vector ops.
  - TC + slippage deducted at every entry *and* exit event.

* walk_forward_backtest : driver that calls walk_forward_regimes, aligns
  returns to the OOS window, then calls vectorized_backtest once on the
  stitched OOS period.

* qc_grade_backtest : A–D grade based on Sharpe, hit rate, max drawdown,
  trade count, and win/loss ratio (100-point rubric).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations as _itr_combinations
from scipy.stats import norm as _st_norm
from scipy.stats import skew as _st_skew, kurtosis as _st_kurt

__all__ = [
    "walk_forward_regimes",
    "vectorized_backtest",
    "walk_forward_backtest",
    "qc_grade_backtest",
    "deflated_sharpe_probability",
    "cscv_pbo_single",
    "hlz_tstat_threshold",
    "compute_effective_n",
]

_DEFAULT_TC_BPS        = 5
_DEFAULT_SLIP_BPS      = 2
_EULER_MASCHERONI      = 0.5772156649015328
# Raw count of strategies with all declared legs in return data.
# Used as fallback when effective N cannot be computed from live pairwise correlations.
_N_LIBRARY_STRATEGIES  = 9    # 18 declared − 9 with missing legs (bond ETFs, BDC stocks,
                               # FX pairs, DXY). Confirmed by data-integrity audit.


# ── Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) ──────────────────────

def deflated_sharpe_probability(
    sr_hat: float,
    n_obs: int,
    skew: float = 0.0,
    excess_kurt: float = 0.0,
    n_strategies: int = 1,
) -> tuple[float, float]:
    """
    P(true SR > SR*) after multiple-testing deflation and non-normality correction.

    SR* is the expected maximum Sharpe an N-strategy search produces by luck alone
    (Bailey & Lopez de Prado 2014, Equations 1–4).

    Parameters
    ----------
    sr_hat        : per-period (per-trade) Sharpe, NOT annualized.
    n_obs         : number of completed trades.
    skew          : skewness of trade-level returns.
    excess_kurt   : excess kurtosis (kurtosis − 3) of trade-level returns.
    n_strategies  : number of strategies tried (sets the multiple-testing bar).

    Returns
    -------
    (dsr_prob, sr_star)
        dsr_prob : float in [0, 1] — probability the strategy has genuine edge.
        sr_star  : the multiple-testing benchmark Sharpe under H₀.
    """
    if n_obs < 2:
        return 0.0, 0.0
    T = max(n_obs, 2)
    N = max(n_strategies, 1)

    # Expected max SR from N IID trials under H₀ (SR = 0)
    # Uses Euler-Mascheroni approximation of the expected extreme value.
    z1 = float(_st_norm.ppf(1.0 - 1.0 / N))
    z2 = float(_st_norm.ppf(1.0 - 1.0 / (N * np.e)))
    sr_star = ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2) / np.sqrt(T - 1)

    # Guard against degenerate SR (e.g. zero-variance returns)
    if not np.isfinite(sr_hat):
        return (1.0 if sr_hat > 0 else 0.0), float(sr_star)

    # Variance of SR_hat corrected for non-normality (Lo 2002 / Bailey 2014)
    # Var(SR) ≈ (1 − skew·SR + (excess_kurt/4 + 0.5)·SR²) / (T−1)
    var_sr = (1.0 - skew * sr_hat + (excess_kurt / 4.0 + 0.5) * sr_hat ** 2) / max(T - 1, 1)
    var_sr = max(var_sr, 1e-10)

    z = (sr_hat - sr_star) / np.sqrt(var_sr)
    prob = float(_st_norm.cdf(z))
    return (prob if np.isfinite(prob) else 0.0), float(sr_star)


# ── Harvey, Liu & Zhu (2016) multiple-testing cross-check ────────────────────
# This is a DIAGNOSTIC ONLY — not a second gate on top of DSR.
# DSR already deflates each Sharpe for trial count N, so stacking an HLZ
# haircut on top would double-count multiplicity and over-penalise borderline
# genuine strategies.  HLZ is reported alongside DSR so they can be compared;
# disagreements are flagged for manual review.
#
# Procedure: Benjamini-Hochberg-Yekutieli (BHY) at FDR 10%.
# BHY (not Bonferroni) is recommended for finance because strategy t-stats are
# positively correlated — Bonferroni overestimates the required adjustment.
# Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns",
# RFS 29(1) pp 5–68, recommend BHY with c(N) = Σ(1/i) harmonic correction.
#
# Economic-prior carve-out:
#   theory-motivated strategies (static library): use N_theory (smaller search
#     space → lower SR* and lower t-hurdle, reflecting genuine ex-ante motivation).
#   grid-mined / conflict-generated strategies: use N_all (raised N reflects the
#     fact that many combinations were implicitly tested to surface these ideas).

def hlz_tstat_threshold(
    n_strategies: int,
    fdr: float = 0.10,
    is_economic_prior: bool = True,
) -> float:
    """
    BHY-adjusted minimum t-statistic for a single new discovery.

    Uses the rank-1 BHY threshold — i.e., the hurdle for the single best-looking
    strategy in a pool of N.  This is conservative for mid-ranked strategies but
    gives a clean scalar hurdle comparable to DSR.

    Parameters
    ----------
    n_strategies      : number of strategies in the tested pool.
    fdr               : target false-discovery rate (default 10%).
    is_economic_prior : if True, applies a 50%-wider FDR budget reflecting
                        that theory-motivated strategies had a lower prior
                        probability of being pure noise.
    """
    n = max(n_strategies, 1)
    # BHY harmonic correction for positive dependence
    harmonic_n = sum(1.0 / i for i in range(1, n + 1))
    # Theory-motivated strategies earn a looser FDR (fewer implicit tests)
    effective_fdr = fdr * 1.5 if is_economic_prior else fdr
    # Rank-1 (most stringent) BHY p-value threshold
    p_threshold = effective_fdr / (n * harmonic_n)
    p_threshold = min(max(p_threshold, 1e-10), 0.5)
    return float(_st_norm.ppf(1.0 - p_threshold))


def strategy_tstat(result: dict) -> float | None:
    """
    Per-trade t-statistic under H₀: mean trade return = 0.

    t = SR_per_trade × √n_trades

    This is the canonical t-stat used in Harvey et al. (2016) to compare against
    their adjusted hurdles.  It is distinct from DSR, which tests against SR* > 0
    (the expected max Sharpe by luck) rather than against 0.
    """
    n  = result.get("n_trades", 0)
    sh = result.get("sharpe")   # annualized OOS Sharpe
    if n < 2 or sh is None:
        return None
    h = result.get("_holding_days", 30)
    ann_factor   = np.sqrt(252.0 / max(h, 1))
    sr_per_trade = sh / ann_factor if ann_factor > 0 else sh
    return float(sr_per_trade * np.sqrt(n))


def compute_effective_n(
    wf_results: dict[str, dict],
    corr_threshold: float = 0.90,
) -> tuple[int, list[tuple[str, str, float]]]:
    """
    Compute effective strategy count after collapsing r > corr_threshold duplicates.

    Builds a pairwise correlation matrix of daily equity-curve returns across all
    strategies that produced a usable OOS equity curve (≥ 3 trades).  Two strategies
    are treated as a single distinct bet if their return-series correlation exceeds
    corr_threshold.  Uses union-find to collapse all clusters.

    Parameters
    ----------
    wf_results      : {strategy_name: walk_forward_backtest result dict}
    corr_threshold  : r-value above which two strategies count as one bet.

    Returns
    -------
    (effective_n, cluster_pairs)
        effective_n   : number of distinct bets.
        cluster_pairs : list of (name_a, name_b, r) for all pairs above threshold.
    """
    # Collect usable equity curves
    curves: dict[str, pd.Series] = {}
    for name, res in wf_results.items():
        ec = res.get("equity_curve")
        if ec is not None and res.get("n_trades", 0) >= 3:
            daily = ec.pct_change().dropna()
            if len(daily) > 20:
                curves[name] = daily

    if len(curves) < 2:
        return len(curves), []

    # Align on common index and compute pairwise correlations
    ec_df  = pd.DataFrame(curves).dropna(how="all")
    corr_m = ec_df.corr()

    names = list(corr_m.columns)
    n = len(names)

    # Union-find: collapse strategies with r > threshold into the same cluster
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cluster_pairs: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr_m.iloc[i, j])
            if np.isfinite(r) and r > corr_threshold:
                cluster_pairs.append((names[i], names[j], round(r, 4)))
                pi, pj = _find(i), _find(j)
                if pi != pj:
                    parent[pi] = pj

    distinct_roots = len({_find(i) for i in range(n)})
    return distinct_roots, cluster_pairs


# ── CSCV Probability of Backtest Overfitting ─────────────────────────────────

def cscv_pbo_single(
    trade_returns: list[float],
    n_blocks: int = 16,
    max_combinations: int = 12870,  # C(16,8) — full enumeration
    rng_seed: int = 42,
) -> tuple[float, int]:
    """
    Per-strategy CSCV Probability of Backtest Overfitting (Bailey & Lopez de Prado 2014).

    Splits the OOS trade-return series into `n_blocks` equal blocks.
    For each of the C(n_blocks, n_blocks/2) combinatorial train/test partitions:
      IS  Sharpe  = Sharpe on training blocks
      OOS Sharpe  = Sharpe on test blocks
    PBO = fraction of partitions where IS Sharpe > 0 AND OOS Sharpe ≤ 0.

    A PBO > 0.5 means that in more than half of all time-block splits the
    strategy appeared profitable in-sample but failed out-of-sample — strong
    evidence of overfitting to a specific market regime.

    Implementation uses fully vectorised numpy operations: all C(s, s/2)
    partition Sharpes are computed simultaneously via a (P × s) mask matrix.

    Parameters
    ----------
    trade_returns   : per-trade compound OOS returns (%).
    n_blocks        : number of equal blocks (must be even; default 16 → C(16,8)=12870).
    max_combinations: cap on total partitions evaluated (random subsample if exceeded).

    Returns
    -------
    (pbo, n_partitions)
        pbo           : float in [0,1] — probability of backtest overfitting.
        n_partitions  : number of valid partitions evaluated.
    """
    arr = np.array(trade_returns, dtype=float)
    n   = len(arr)

    if n < 6:
        return 1.0, 0

    # Adaptive block count: ensure ≥3 trades per block; enforce even.
    s = min(n_blocks, (n // 3) // 2 * 2)
    s = max(4, s)
    half = s // 2

    # Split into s equal blocks and pre-compute per-block sufficient statistics.
    blocks    = np.array_split(arr, s)
    block_n   = np.array([len(b) for b in blocks], dtype=float)    # count
    block_sum = np.array([b.sum() for b in blocks], dtype=float)   # Σx
    block_ssq = np.array([(b ** 2).sum() for b in blocks], dtype=float)  # Σx²

    # Enumerate all C(s, half) combinatorial partitions.
    all_combos = list(_itr_combinations(range(s), half))
    if len(all_combos) > max_combinations:
        rng = np.random.default_rng(rng_seed)
        sel = rng.choice(len(all_combos), size=max_combinations, replace=False)
        all_combos = [all_combos[i] for i in sorted(sel)]

    P = len(all_combos)

    # Build a (P × s) boolean mask: mask[k, j] = True → block j in training set k.
    mask = np.zeros((P, s), dtype=bool)
    for k, train_idx in enumerate(all_combos):
        mask[k, list(train_idx)] = True

    # Vectorised pooled statistics for train and test halves across all P partitions.
    # Pooled mean  = Σ(block_sum) / Σ(block_n)
    # Pooled var   = Σ(block_ssq)/Σ(block_n) − pooled_mean²
    def _pool(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (n_total, mean, variance) for the blocks selected by boolean mask m."""
        tot_n   = (m * block_n  ).sum(axis=1)                # (P,)
        tot_sum = (m * block_sum).sum(axis=1)                # (P,)
        tot_ssq = (m * block_ssq).sum(axis=1)                # (P,)
        safe_n  = np.maximum(tot_n, 1.0)
        mean    = tot_sum / safe_n
        var     = np.maximum(tot_ssq / safe_n - mean ** 2, 0.0)
        return tot_n, mean, var

    tr_n, tr_mean, tr_var = _pool(mask)
    te_n, te_mean, te_var = _pool(~mask)

    # Valid partitions: both halves have ≥2 trades and positive variance.
    valid = (tr_n >= 2) & (te_n >= 2) & (tr_var > 1e-12) & (te_var > 1e-12)

    is_sh  = np.where(valid, tr_mean / np.sqrt(np.where(valid, tr_var, 1.0)), 0.0)
    oos_sh = np.where(valid, te_mean / np.sqrt(np.where(valid, te_var, 1.0)), 0.0)

    n_valid   = int(valid.sum())
    n_overfit = int((valid & (is_sh > 0.0) & (oos_sh <= 0.0)).sum())

    pbo = float(n_overfit / n_valid) if n_valid > 0 else 0.0
    return pbo, n_valid


# ── Walk-forward regime labels ────────────────────────────────────────────────

def walk_forward_regimes(
    avg_corr: pd.Series,
    min_train_days: int = 252,
    fold_days: int = 63,
    smooth_window: int = 5,
    p_decorr: int = 20,
    p_normal: int = 55,
    p_elevated: int = 80,
    purge_days: int = 0,
) -> pd.Series:
    """
    Return an OOS regime Series aligned to avg_corr's index.

    Dates in the first `min_train_days` rows are NaN — the backtest only runs
    on dates that have a valid OOS label.

    Regime codes: 0 Decorrelated, 1 Normal, 2 Elevated, 3 Crisis.
    """
    s = avg_corr.dropna()
    if len(s) < min_train_days + fold_days:
        return pd.Series(np.nan, index=avg_corr.index, name="regime_oos")

    # Causal smoothing (no center=True, which would leak future data)
    smoothed = s.rolling(smooth_window, min_periods=1).median()
    out = pd.Series(np.nan, index=avg_corr.index, name="regime_oos")
    n = len(smoothed)

    for fold_start in range(min_train_days, n, fold_days):
        fold_end = min(fold_start + fold_days, n)
        # Purge: exclude the last purge_days from the training window so that
        # training observations whose label windows overlap the test fold are removed.
        # (Lopez de Prado 2018, Chapter 7 — Purged Walk-Forward CV)
        train_s = smoothed.iloc[:max(1, fold_start - purge_days)]

        td = float(np.percentile(train_s, p_decorr))
        tn = float(np.percentile(train_s, p_normal))
        te = float(np.percentile(train_s, p_elevated))
        td_exit = td * 1.05
        tn_exit = tn * 0.95
        te_exit = te * 0.95

        # Seed the hysteresis state from the last training day
        v_seed = float(smoothed.iloc[fold_start - 1])
        curr = 3 if v_seed >= te else 2 if v_seed >= tn else 1 if v_seed >= td else 0

        for i in range(fold_start, fold_end):
            v = float(smoothed.iloc[i])
            if curr == 0:
                if v >= td_exit:
                    curr = 1
            elif curr == 1:
                if v < td:
                    curr = 0
                elif v >= tn:
                    curr = 2
            elif curr == 2:
                if v < tn_exit:
                    curr = 1
                elif v >= te:
                    curr = 3
            elif curr == 3:
                if v < te_exit:
                    curr = 2
            out[smoothed.index[i]] = curr

    return out


# ── Single-period vectorized engine ──────────────────────────────────────────

def vectorized_backtest(
    returns: pd.DataFrame,
    regimes: pd.Series,
    assets: list[str],
    directions: list[str],
    trigger_regimes: list[int],
    holding_days: int = 30,
    leg_weights: list[float] | None = None,
    tc_bps: float = _DEFAULT_TC_BPS,
    slippage_bps: float = _DEFAULT_SLIP_BPS,
    entry_blackout: "np.ndarray | None" = None,
) -> dict:
    """
    Vectorized backtest over a single returns/regimes window.

    Entry logic
    -----------
    Rising edge into any `trigger_regimes` regime triggers a new entry
    (regime[t-1] not in trigger → regime[t] in trigger).
    Only one position at a time — a new entry while already in a trade is
    skipped.

    Exit logic
    ----------
    Position is closed on the *first* of:
      (a) `holding_days` calendar days since entry, or
      (b) regime exits the trigger set.

    Transaction costs
    -----------------
    One-way cost = (tc_bps + slippage_bps) / 10 000 per leg.
    Applied at entry (opening) and at exit (closing). Total round-trip per
    trade = 2 × n_legs × one_way_cost.

    Returns
    -------
    dict with n_trades, hit_rate (%), avg_win (%), avg_loss (%),
    sharpe (annualized), max_drawdown (%), equity_curve (pd.Series, base 100),
    trade_returns (list[float]), tc_bps, slippage_bps.
    Returns {"n_trades": 0, "error": "..."} on failure.
    """
    avail = [(a, d) for a, d in zip(assets, directions) if a in returns.columns]
    if not avail or returns.empty or regimes.empty:
        return {"n_trades": 0, "error": "Insufficient data"}

    n_legs = len(avail)

    if leg_weights is not None and len(leg_weights) == len(assets):
        avail_w: list[float] = []
        for a, _ in avail:
            avail_w.append(leg_weights[assets.index(a)])
        total_w = sum(avail_w) or 1.0
        avail_w = [w / total_w for w in avail_w]
    else:
        avail_w = [1.0 / n_legs] * n_legs

    reg = regimes.reindex(returns.index, method="ffill", limit=20).fillna(-1).astype(int)
    in_regime_arr = np.array(reg.isin(trigger_regimes), dtype=bool)

    prev_in          = np.concatenate([[False], in_regime_arr[:-1]])
    entry_signal_raw = (in_regime_arr & ~prev_in)
    # Shift 1 bar: signal on day t fires entry on day t+1 (regime known only at close of t)
    entry_signal     = np.concatenate([[False], entry_signal_raw[:-1]])

    T = len(returns)

    # Embargo / purge mask: suppress new entries during blackout windows.
    # Existing positions are held through blackout — only new entries are blocked.
    if entry_blackout is not None and len(entry_blackout) == T:
        entry_signal = entry_signal & ~entry_blackout.astype(bool)

    position = np.zeros(T, dtype=float)

    # O(T) position builder — one trade at a time, no overlapping positions
    in_trade = False
    entry_day = -1
    for i in range(T):
        if in_trade:
            if (i - entry_day) >= holding_days or not in_regime_arr[i]:
                in_trade = False
        if not in_trade and entry_signal[i]:
            in_trade = True
            entry_day = i
        position[i] = 1.0 if in_trade else 0.0

    # Vectorized daily portfolio return (signed, conviction-weighted)
    daily_r = np.zeros(T, dtype=float)
    for (asset, direction), w in zip(avail, avail_w):
        sign = 1.0 if direction.lower() == "long" else -1.0
        daily_r += returns[asset].reindex(returns.index).fillna(0.0).values * sign * w

    # TC deducted at every position open/close event
    pos_diff = np.diff(position, prepend=0.0)
    entry_events = (pos_diff > 0.5).astype(float)
    exit_events  = (pos_diff < -0.5).astype(float)
    one_way = (tc_bps + slippage_bps) / 10_000
    cost = (entry_events + exit_events) * n_legs * one_way

    net_daily = daily_r * position - cost

    # Equity curve (starts at 100)
    equity = np.cumprod(1 + net_daily) * 100.0
    equity_s = pd.Series(equity, index=returns.index, name="equity")

    # Per-trade compound returns
    entry_indices = np.where(entry_events > 0.5)[0]
    exit_indices  = np.where(exit_events  > 0.5)[0]

    trade_returns: list[float] = []
    for eidx in entry_indices:
        next_exits = exit_indices[exit_indices > eidx]
        xidx = int(next_exits[0]) if len(next_exits) > 0 else T - 1
        window = net_daily[eidx: xidx + 1]
        tr = float((np.prod(1 + window) - 1) * 100)
        trade_returns.append(tr)

    if len(trade_returns) < 3:
        return {
            "n_trades": len(trade_returns),
            "error": "Too few trades",
            "equity_curve": equity_s,
        }

    tr_arr = np.array(trade_returns)
    wins   = tr_arr[tr_arr > 0]
    losses = tr_arr[tr_arr <= 0]

    hit_rate = float(len(wins) / len(tr_arr) * 100)
    avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    # Annualized Sharpe from per-trade compound returns (already net of all costs,
    # including exit-day costs that land when position = 0 and would be missed by
    # filtering on position > 0.5).
    if len(tr_arr) > 1 and tr_arr.std() > 1e-10:
        ann_factor = np.sqrt(252 / max(holding_days, 1))
        sharpe = float(tr_arr.mean() / tr_arr.std() * ann_factor)
    else:
        sharpe = 0.0

    peak  = np.maximum.accumulate(equity)
    dd    = (equity - peak) / (peak + 1e-8) * 100
    max_dd = float(dd.min())

    wl = avg_win / (abs(avg_loss) + 1e-8)

    return {
        "n_trades":     len(tr_arr),
        "hit_rate":     round(hit_rate, 1),
        "avg_win":      round(avg_win, 2),
        "avg_loss":     round(avg_loss, 2),
        "win_loss_ratio": round(wl, 2),
        "sharpe":       round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "equity_curve": equity_s,
        "trade_returns": tr_arr.tolist(),
        "tc_bps":       tc_bps,
        "slippage_bps": slippage_bps,
    }


# ── Walk-forward driver ───────────────────────────────────────────────────────

def walk_forward_backtest(
    returns: pd.DataFrame,
    avg_corr: pd.Series,
    trade: dict,
    min_train_days: int = 252,
    fold_days: int = 63,
    embargo_days: int = 5,
    tc_bps: float = _DEFAULT_TC_BPS,
    slippage_bps: float = _DEFAULT_SLIP_BPS,
    leg_weights: list[float] | None = None,
    n_strategies: int = _N_LIBRARY_STRATEGIES,
    is_economic_prior: bool = True,
) -> dict:
    """
    Purged, embargoed walk-forward backtest (Lopez de Prado 2018, Chapter 7).

    Label-overlap leakage is eliminated at two levels:

    1. Threshold purging (walk_forward_regimes)
       Each fold's training window is shrunk by `holding_days` at the right
       edge, so no training observation whose label window [t, t+holding_days]
       overlaps the test fold influences the regime thresholds.

    2. OOS entry embargo (vectorized_backtest entry_blackout)
       New entries are suppressed for `embargo_days` trading days after:
         (a) the IS→OOS transition, and
         (b) each within-OOS fold boundary (where thresholds update).
       Existing open positions are held through blackout windows.

    IS Sharpe is computed on the training slice using full-sample regime
    labels (definitionally IS, no purging needed — it is diagnostic only,
    not a training signal) and is used solely for IS→OOS decay reporting.

    Parameters
    ----------
    trade        : dict with keys assets, direction, regime, holding_period.
    embargo_days : trading-day gap suppressing new entries after each fold
                   boundary (default 5 = 1 week).
    """
    assets     = trade.get("assets", [])
    directions = trade.get("direction", [])
    trigger    = trade.get("regime", [2, 3])
    holding    = _parse_holding_days(trade)

    # Fail loud on missing legs — silent subset-trading produces mislabeled results
    # (e.g. bond ETFs absent from return data turned Gold/TLT into pure Long Gold).
    missing = [a for a in assets if a not in returns.columns]
    if missing:
        return {
            "n_trades":     0,
            "error":        f"Missing declared legs: {missing}",
            "missing_legs": missing,
            "mode":         "walk_forward",
        }

    # Auto-scale if the avg_corr series is shorter than the default parameters.
    # avg_corr may be sparse (rolling corr with strict min_periods knocks out most days).
    T_corr = len(avg_corr.dropna())
    if T_corr < min_train_days + fold_days:
        min_train_days = max(int(T_corr * 0.55), 20)
        fold_days      = max(int(T_corr * 0.10), 5)

    oos_regimes = walk_forward_regimes(
        avg_corr,
        min_train_days=min_train_days,
        fold_days=fold_days,
        purge_days=holding,    # remove last holding_days from each training window
    )
    oos_valid = oos_regimes.dropna()

    if len(oos_valid) < 5:
        return {"n_trades": 0, "error": "Insufficient OOS data", "mode": "walk_forward"}

    # Forward-fill sparse OOS regime labels to the daily returns index so that
    # holding_days always means real trading days, not correlation-update intervals.
    # A regime determined on day T stays in effect until the next computation date.
    oos_daily = oos_valid.reindex(returns.index, method="ffill").dropna()
    oos_returns = returns.loc[oos_daily.index].dropna(how="all")
    oos_regimes_aligned = oos_daily.reindex(oos_returns.index)

    if oos_returns.empty or len(oos_returns) < holding + 5:
        return {"n_trades": 0, "error": "Insufficient OOS data", "mode": "walk_forward"}

    # ── Embargo blackout for OOS period ──────────────────────────────────────
    # Suppress new entries for embargo_days after:
    #   (a) the IS→OOS transition point, and
    #   (b) each within-OOS fold boundary (where training data is updated).
    # This prevents any return information that spans the boundary from being
    # traded — the embargo gap exceeds a normal settlement/look-back period.
    # (Lopez de Prado 2018, Chapter 7 — embargo after each test fold)
    n_oos = len(oos_returns)
    oos_blackout = np.zeros(n_oos, dtype=bool)

    # (a) IS→OOS transition embargo
    oos_blackout[:min(embargo_days, n_oos)] = True

    # (b) Within-OOS fold boundaries (mapped from avg_corr fold schedule to daily index)
    oos_avg_dates = oos_valid.index
    for fb_row in range(fold_days, len(oos_avg_dates), fold_days):
        if fb_row < len(oos_avg_dates):
            fb_date = oos_avg_dates[fb_row]
            pos_arr = np.where(oos_returns.index >= fb_date)[0]
            if len(pos_arr) > 0:
                start_pos = int(pos_arr[0])
                oos_blackout[start_pos: min(start_pos + embargo_days, n_oos)] = True

    result = vectorized_backtest(
        returns=oos_returns,
        regimes=oos_regimes_aligned,
        assets=assets,
        directions=directions,
        trigger_regimes=trigger,
        holding_days=holding,
        leg_weights=leg_weights,
        tc_bps=tc_bps,
        slippage_bps=slippage_bps,
        entry_blackout=oos_blackout,
    )

    result["mode"]            = "walk_forward"
    result["n_folds"]         = max(1, len(oos_valid) // fold_days)
    result["oos_days"]        = len(oos_returns)
    result["_holding_days"]   = holding
    result["purge_days"]      = holding
    result["embargo_days"]    = embargo_days
    result["oos_blackout_bars"] = int(oos_blackout.sum())

    # ── IS Sharpe: retrospective IS backtest (no IS purge) ───────────────────────
    # IS Sharpe is a DIAGNOSTIC metric used only to compute IS→OOS decay; it is
    # not a training signal, so Lopez de Prado purging does not apply here.
    # Purging is applied only where leakage matters:
    #   (a) threshold fitting in walk_forward_regimes (purge_days=holding above), and
    #   (b) OOS entry signals (oos_blackout above).
    # The IS period labels use full-sample percentile thresholds (detect_correlation_regime),
    # which is definitionally in-sample and contains no forward return information.
    try:
        from src.analysis.correlations import detect_correlation_regime as _dcr
        _is_reg_full = _dcr(avg_corr).reindex(returns.index, method="ffill").dropna()
        _oos_start   = oos_valid.index[0]
        _is_ret      = returns.loc[returns.index < _oos_start].dropna(how="all")
        _is_reg      = _is_reg_full.reindex(_is_ret.index).dropna()
        _is_ret      = _is_ret.loc[_is_reg.index]
        if len(_is_ret) >= holding + 5:
            _is_bt = vectorized_backtest(
                returns=_is_ret, regimes=_is_reg,
                assets=assets, directions=directions,
                trigger_regimes=trigger, holding_days=holding,
                leg_weights=leg_weights, tc_bps=tc_bps, slippage_bps=slippage_bps,
            )
            result["is_sharpe"]      = _is_bt.get("sharpe")
            result["is_n_trades"]    = _is_bt.get("n_trades", 0)
        else:
            result["is_sharpe"]      = None
            result["is_n_trades"]    = 0
        result["is_purged_rows"]  = 0          # IS not purged by design
        result["purge_validated"] = True       # OOS purge guaranteed by construction
    except Exception:
        result["is_sharpe"]       = None
        result["is_n_trades"]     = 0
        result["is_purged_rows"]  = 0
        result["purge_validated"] = False

    # ── CSCV Probability of Backtest Overfitting ─────────────────────────────
    _oos_tr = result.get("trade_returns", [])
    if _oos_tr:
        _pbo, _n_cscv = cscv_pbo_single(_oos_tr)
    else:
        _pbo, _n_cscv = 1.0, 0
    result["pbo"]    = _pbo
    result["n_cscv"] = _n_cscv

    if "error" not in result:
        result["qc"] = qc_grade_backtest(
            result,
            n_strategies=n_strategies,
            is_economic_prior=is_economic_prior,
        )

    return result


# ── QC grading ────────────────────────────────────────────────────────────────

_MIN_TRADES_CONFIDENT = 20   # below this, grade capped at C (Sharpe SE too wide)
_MIN_TRADES_GRADED    = 6    # below this, grade is F regardless


def qc_grade_backtest(
    result: dict,
    n_strategies: int = _N_LIBRARY_STRATEGIES,
    is_economic_prior: bool = True,
) -> dict:
    """
    Grade a walk-forward backtest on OOS robustness — not raw IS performance.

    Three axes drive the grade:

    1. Deflated Sharpe Ratio (DSR) probability
       P(true OOS SR > SR*) after adjusting for:
         • multiple testing across N strategies (SR* = expected max SR by luck)
         • sample size (T = n_trades)
         • return non-normality (skew, excess kurtosis)
       Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio".

    2. IS-to-OOS Sharpe decay
       decay = 1 − OOS_Sharpe / IS_Sharpe   (when IS_Sharpe > 0)
       Measures how much the edge shrinks out-of-sample.
       High decay caps the grade even if DSR looks good.

    3. Trade count adequacy
       n < 6  → F (no statistical content).
       n < 20 → capped at C (Sharpe SE too wide to support A or B).

    Grade mapping (base, before caps):
       A: DSR ≥ 0.95
       B: DSR ≥ 0.75
       C: DSR ≥ 0.50
       D: DSR ≥ 0.25
       F: DSR < 0.25

    Hard caps (applied after base grade):
       decay > 0.90 → max D  (strategy nearly dies OOS)
       decay > 0.70 → max C  (significant OOS degradation)
       n < 20       → max C  (low-N statistical limit)
       n < 6        → F      (no meaningful sample)

    The `score` field = round(dsr_prob × 100), directly interpretable as the
    probability (%) that this strategy has genuine edge beyond multiple-testing luck.
    """
    flags: list[str] = []
    n  = result.get("n_trades", 0)
    sh = result.get("sharpe", 0.0)
    tr = result.get("trade_returns", [])

    low_confidence = n < _MIN_TRADES_CONFIDENT

    # ── Absolute floor: too few trades to say anything ────────────────────────
    if n < _MIN_TRADES_GRADED:
        flags.append(f"Only {n} trades — cannot compute meaningful DSR (need ≥{_MIN_TRADES_GRADED})")
        return {
            "grade": "F", "score": 0,
            "dsr_prob": 0.0, "sr_star": 0.0,
            "decay": None, "is_sharpe": result.get("is_sharpe"),
            "flags": flags, "low_confidence": True,
        }

    # ── Convert annualized Sharpe → per-trade Sharpe for DSR formula ──────────
    holding_days = result.get("_holding_days", 30)
    ann_factor   = np.sqrt(252.0 / max(holding_days, 1))
    sr_per_trade = sh / ann_factor if ann_factor > 0 else sh

    # ── Moments of trade-level returns (for non-normality correction) ─────────
    if len(tr) >= 4:
        tr_arr   = np.array(tr, dtype=float)
        skewness = float(_st_skew(tr_arr))
        ex_kurt  = float(_st_kurt(tr_arr, fisher=True))   # excess kurtosis (normal → 0)
    else:
        skewness, ex_kurt = 0.0, 0.0

    # ── DSR probability ───────────────────────────────────────────────────────
    dsr_prob, sr_star = deflated_sharpe_probability(
        sr_hat=sr_per_trade,
        n_obs=n,
        skew=skewness,
        excess_kurt=ex_kurt,
        n_strategies=n_strategies,
    )

    # ── IS-to-OOS decay ───────────────────────────────────────────────────────
    is_sharpe = result.get("is_sharpe")
    if is_sharpe is not None and is_sharpe > 0.0:
        decay: float | None = max(0.0, 1.0 - sh / is_sharpe)
    elif is_sharpe is not None and is_sharpe <= 0.0:
        decay = 1.0   # already negative or zero in-sample
    else:
        decay = None  # IS not computed (data unavailable)

    # ── Base grade from DSR probability ──────────────────────────────────────
    if dsr_prob >= 0.95:
        grade = "A"
    elif dsr_prob >= 0.75:
        grade = "B"
    elif dsr_prob >= 0.50:
        grade = "C"
    elif dsr_prob >= 0.25:
        grade = "D"
    else:
        grade = "F"

    # ── Hard caps (applied in priority order) ─────────────────────────────────
    # n-cap: sample too small for A or B
    if low_confidence and grade in ("A", "B"):
        grade = "C"
        flags.append(
            f"LOW N: {n} trades (need ≥{_MIN_TRADES_CONFIDENT}) — "
            f"Sharpe SE ≈ {np.sqrt((1 + sr_per_trade**2/2)/max(n-1,1)):.2f} — capped at C"
        )

    # decay cap
    if decay is not None:
        if decay > 0.90 and grade not in ("D", "F"):
            flags.append(f"IS→OOS decay {decay:.0%} (>{90}%) — capped at D")
            grade = "D"
        elif decay > 0.70 and grade in ("A", "B"):
            flags.append(f"IS→OOS decay {decay:.0%} (>{70}%) — capped at C")
            grade = "C"
        if decay < 0.0:
            flags.append(f"OOS beats IS (decay {decay:.0%}) — confirm no regime look-ahead")

    # PBO cap: CSCV shows IS/OOS performance uncorrelated in majority of block splits.
    pbo = result.get("pbo")
    if pbo is not None and np.isfinite(pbo) and pbo > 0.5 and grade not in ("D", "F"):
        flags.append(
            f"CSCV PBO {pbo:.0%} — IS/OOS Sharpe sign flips in majority of block partitions → capped at D"
        )
        grade = "D"

    # ── HLZ cross-check (diagnostic only — NOT a second gate) ───────────────────
    # t-stat under H₀: mean trade return = 0.  Compared against BHY-adjusted
    # minimum t-statistic for the given N and economic-prior status.
    # A disagreement between DSR and HLZ is flagged for review; neither overrides
    # the other.  DSR is the binding grade criterion.
    t_stat = strategy_tstat(result)                               # may be None
    hlz_threshold = hlz_tstat_threshold(
        n_strategies=n_strategies,
        fdr=0.10,
        is_economic_prior=is_economic_prior,
    )
    if t_stat is not None and np.isfinite(t_stat):
        hlz_pass = t_stat >= hlz_threshold
    else:
        hlz_pass = None   # not computable

    # DSR pass defined as dsr_prob ≥ 0.50 (strategy more likely to be genuine than not)
    dsr_pass = dsr_prob >= 0.50
    if hlz_pass is not None:
        hlz_agree_dsr = (hlz_pass == dsr_pass)
        if not hlz_agree_dsr:
            which = "DSR passes, HLZ fails" if dsr_pass else "HLZ passes, DSR fails"
            flags.append(
                f"HLZ DISAGREES WITH DSR — {which} "
                f"(t={t_stat:.2f} vs HLZ threshold {hlz_threshold:.2f}, "
                f"DSR={dsr_prob:.0%}) — review manually"
            )
    else:
        hlz_agree_dsr = None

    # informational flags (don't change grade)
    if dsr_prob < 0.25:
        flags.append(f"DSR {dsr_prob:.1%} — below 25%, likely statistical noise from {n_strategies}-strategy search")
    if sh < 0.0:
        flags.append(f"Negative OOS Sharpe ({sh:.2f})")
    if pbo is not None and np.isfinite(pbo) and pbo > 0.3:
        flags.append(f"CSCV PBO {pbo:.0%} — IS profitability predicts OOS loss in {pbo:.0%} of block splits")

    dsr_prob = dsr_prob if np.isfinite(dsr_prob) else 0.0
    score = round(dsr_prob * 100)

    return {
        "grade":          grade,
        "score":          score,           # = DSR prob × 100 (auditable)
        "dsr_prob":       round(dsr_prob, 4),
        "sr_star":        round(sr_star, 4),
        "decay":          round(decay, 4) if decay is not None else None,
        "is_sharpe":      is_sharpe,
        "pbo":            round(pbo, 4) if pbo is not None and np.isfinite(pbo) else None,
        # ── HLZ cross-check fields ──────────────────────────────────────────
        "hlz_tstat":      round(t_stat, 3) if t_stat is not None else None,
        "hlz_threshold":  round(hlz_threshold, 3),
        "hlz_pass":       hlz_pass,        # bool or None
        "hlz_agree_dsr":  hlz_agree_dsr,   # bool or None — False triggers review flag
        "is_economic_prior": is_economic_prior,
        "n_strategies_used": n_strategies,
        # ───────────────────────────────────────────────────────────────────
        "flags":          flags,
        "low_confidence": low_confidence,
    }


# ── Utility ───────────────────────────────────────────────────────────────────

def _parse_holding_days(trade: dict, default: int = 30) -> int:
    raw = trade.get("holding_period", default)
    if isinstance(raw, int):
        return max(1, raw)
    try:
        return max(1, int(str(raw).replace("d", "").strip()))
    except (ValueError, TypeError):
        return default

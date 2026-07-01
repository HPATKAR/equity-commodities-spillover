"""
Walk-forward validation of the five-stage thesis-first pipeline as a decision rule.

The unit of analysis is a (thesis, time-window) pair — NOT a trade's P&L.

At each window the pipeline classifies every thesis as:
  admit   → Stage 3 confirmed + DSR ≥ 0.50 (the gate says "tradeable edge exists")
  mt      → Stage 3 confirmed + DSR < 0.50 ("mechanism real, not tradeable")
  ie      → sign matched but < 20 OOS trades ("insufficient evidence")
  reject  → Stage 3 not confirmed or wrong OOS sign

The pipeline passes only if all three hold out of sample:
  1. admitted_vs_rejected_gap > 0  — the gates discriminate
  2. admitted_vs_random_gap > 0    — the gates beat random selection of the same size
  3. MT and IE bucket means behave as labeled:
       MT  ≈ small positive (mechanism real, edge arbitraged)
       IE  ≈ high variance, no reliable sign (not enough signal, not a verdict)

Nothing in this module selects or ranks theses by their own performance. The
classifier is the pipeline's five gates, evaluated strictly on past data at each
window endpoint. Sharpe and P&L appear only as outcome variables in the test window.
"""

from __future__ import annotations

import logging
from copy import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

_log = logging.getLogger(__name__)

# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ThesisWindowDecision:
    """
    Single (thesis, window) classification record.

    train_* fields were computed using ONLY data up to train_end.
    oos_* fields are the realised outcome in the following test window.
    """
    thesis_name: str
    window_idx: int
    train_end: pd.Timestamp
    test_end: pd.Timestamp

    # Stage 3 on training data
    stage3_track: str           # 'lp_irf' | 'regime_conditional' | 'error'
    stage3_sign_matched: bool   # direction confirmed (regardless of significance)
    stage3_confirmed: bool      # direction + statistical significance

    # Stage 5 on training data
    train_n_trades: int
    train_sharpe: float
    dsr_prob: float
    pipeline_decision: str      # 'admit' | 'mt' | 'ie' | 'reject'

    # OOS outcome (test window, unknown at decision time)
    oos_mean_return: float | None   # mean signed holding-period return, %
    oos_n_signals: int              # regime-entry signals in test window


@dataclass
class BucketStats:
    mean: float | None
    std: float | None
    n_obs: int              # thesis-window pairs with at least one OOS signal
    n_windows: int          # windows where this bucket had at least one thesis

    def fmt(self) -> str:
        if self.mean is None:
            return f"n={self.n_obs} (no OOS signals)"
        sign = "+" if self.mean >= 0 else ""
        return f"mean={sign}{self.mean:.2f}%  std={self.std:.2f}%  n={self.n_obs}"


@dataclass
class PipelineValidationResult:
    """
    Full output of walk_forward_pipeline_validation().

    Primary decision metric: admitted_vs_rejected_gap and admitted_vs_random_gap.
    Bucket behavior: confirms MT and IE labels are calibrated.
    """
    # ── Per-bucket OOS statistics ─────────────────────────────────────────────
    buckets: dict[str, BucketStats]   # keys: 'admit', 'mt', 'ie', 'reject'

    # ── Primary gap metrics ───────────────────────────────────────────────────
    admitted_vs_rejected_gap: float | None      # mean(admit) − mean(reject), %
    admitted_vs_rejected_pval: float | None     # two-sample t-test

    admitted_vs_random_gap: float | None        # mean(admit) − mean(random), %
    random_p_value: float | None                # P(random mean ≥ admit mean)
    random_distribution: list[float]            # 1000 random-draw means for plotting

    # ── Metadata ──────────────────────────────────────────────────────────────
    n_windows: int
    n_theses: int
    train_days: int
    test_days: int
    n_random_trials: int
    decisions: list[ThesisWindowDecision]

    def passed(self) -> bool:
        """
        Pipeline passes iff both gaps are positive.
        The MT/IE check is a labeling audit, not a pass/fail gate.
        """
        return (
            self.admitted_vs_rejected_gap is not None and
            self.admitted_vs_rejected_gap > 0 and
            self.admitted_vs_random_gap is not None and
            self.admitted_vs_random_gap > 0
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _run_stage3_on_window(
    thesis,                  # ThesisStrategy (stage1+2 pre-verified)
    train_returns: pd.DataFrame,
    train_regimes: pd.Series | None,
) -> tuple[str, bool, bool]:
    """
    Run Stage 3 on training data only.
    Returns (track, sign_matched, stage_confirmed).
    """
    from src.analysis.thesis_engine import ThesisStrategy, run_stage3

    # Build a clean copy — do not mutate the original
    tmp = ThesisStrategy(
        name=thesis.name,
        category=thesis.category,
        thesis=thesis.thesis,
        signal=thesis.signal,
        stage1_passed=True,
        stage2_passed=True,
    )
    tmp = run_stage3(tmp, train_returns, train_regimes)

    conf = tmp.confirmation
    if conf is None:
        return "error", False, False

    return conf.track, conf.sign_matched, conf.stage_passed


def _run_stage5_decision(
    thesis,
    train_returns: pd.DataFrame,
    train_regimes: pd.Series | None,
    stage3_confirmed: bool,
    stage3_sign_matched: bool,
    n_strategies: int,
) -> tuple[str, int, float, float]:
    """
    Estimate training-window Sharpe, compute DSR, apply gate logic.
    Returns (decision, n_trades, sharpe, dsr_prob).

    The Sharpe here is an in-sample estimate over the training window —
    slightly optimistic relative to a nested walk-forward, but sufficient
    to test whether the gate classification predicts OOS direction.
    """
    from src.analysis.backtest import vectorized_backtest, deflated_sharpe_probability

    sig = thesis.signal
    if sig is None or train_regimes is None:
        return "reject", 0, 0.0, 0.0

    bt = vectorized_backtest(
        returns=train_returns,
        regimes=train_regimes,
        assets=sig.assets,
        directions=sig.direction,
        trigger_regimes=sig.regime,
        holding_days=sig.holding_period,
        leg_weights=sig.leg_weights,
    )

    n_trades = bt.get("n_trades", 0)
    sharpe   = bt.get("sharpe", 0.0) or 0.0

    if n_trades < 20:
        if stage3_sign_matched:
            return "ie", n_trades, sharpe, 0.0
        else:
            return "reject", n_trades, sharpe, 0.0

    # Convert annualized Sharpe → per-trade for DSR (same formula as qc_grade_backtest)
    ann_factor   = np.sqrt(252.0 / max(sig.holding_period, 1))
    sr_per_trade = sharpe / ann_factor if ann_factor > 0 else sharpe

    trade_rets = bt.get("trade_returns", [])
    if len(trade_rets) >= 4:
        from scipy.stats import skew as _skew, kurtosis as _kurt
        tr_arr = np.array(trade_rets)
        skewness = float(_skew(tr_arr))
        ex_kurt  = float(_kurt(tr_arr, fisher=True))
    else:
        skewness, ex_kurt = 0.0, 0.0

    dsr_prob, _ = deflated_sharpe_probability(
        sr_hat=sr_per_trade,
        n_obs=n_trades,
        skew=skewness,
        excess_kurt=ex_kurt,
        n_strategies=n_strategies,
    )

    # Gate logic mirrors run_stage5 exactly
    if stage3_confirmed and dsr_prob < 0.50:
        return "mt", n_trades, sharpe, dsr_prob

    if not stage3_confirmed:
        return "reject", n_trades, sharpe, dsr_prob

    # stage3_confirmed and dsr_prob >= 0.50 → admit
    return "admit", n_trades, sharpe, dsr_prob


def _compute_oos_return(
    thesis,
    test_returns: pd.DataFrame,
    test_regimes: pd.Series | None,
) -> tuple[float | None, int]:
    """
    Realised mean signed holding-period return in the test window.
    Uses only the thesis's declared regime trigger and leg directions.
    Returns (mean_pct, n_signals). Returns (None, 0) if no signals fired.
    """
    from src.analysis.backtest import vectorized_backtest

    sig = thesis.signal
    if sig is None or test_regimes is None:
        return None, 0
    if test_returns.empty:
        return None, 0

    # Require all legs present in test window
    if any(a not in test_returns.columns for a in sig.assets):
        return None, 0

    bt = vectorized_backtest(
        returns=test_returns,
        regimes=test_regimes,
        assets=sig.assets,
        directions=sig.direction,
        trigger_regimes=sig.regime,
        holding_days=sig.holding_period,
        leg_weights=sig.leg_weights,
    )

    trade_rets = bt.get("trade_returns", [])
    n = len(trade_rets)
    if n == 0:
        return None, 0

    return float(np.mean(trade_rets)), n


def _bucket_stats(returns_list: list[float]) -> BucketStats:
    arr = np.array(returns_list)
    if len(arr) == 0:
        return BucketStats(mean=None, std=None, n_obs=0, n_windows=0)
    return BucketStats(
        mean=float(arr.mean()),
        std=float(arr.std()) if len(arr) > 1 else 0.0,
        n_obs=len(arr),
        n_windows=0,  # filled in by the caller
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def walk_forward_pipeline_validation(
    theses: list,                       # list[ThesisStrategy], stage1+2 pre-verified
    returns: pd.DataFrame,
    regimes: pd.Series,
    train_days: int = 756,              # 3 years minimum training (≈756 trading days)
    test_days: int = 63,               # 1 quarter OOS per step
    n_strategies: int = 9,
    n_random_trials: int = 1000,
    rng_seed: int = 42,
) -> PipelineValidationResult:
    """
    Walk the five-stage pipeline forward through history.

    Gate sequence at each window (strictly past data only):
      Stage 1+2 : pre-verified once (static — thesis completeness and leg presence)
      Stage 3   : sign confirmation on training slice (time-varying)
      Stage 5   : DSR on training-window Sharpe (time-varying)

    OOS outcome: mean signed holding-period return in the following test window,
    computed under the thesis's own regime-trigger rule. No optimization of any kind.

    The classifier is the pipeline's binary admit/reject/ie/mt decision.
    The outcome variable is the test-window return. The test: do the gates
    predict direction?
    """
    rng = np.random.default_rng(rng_seed)
    T   = len(returns)

    # Screen Stage 1 + 2 once
    from src.analysis.thesis_engine import run_stage1, run_stage2
    gradeable: list = []
    for t in theses:
        tmp = run_stage1(copy(t))
        tmp = run_stage2(tmp, set(returns.columns))
        if tmp.stage1_passed and tmp.stage2_passed:
            gradeable.append(t)
        else:
            _log.info("Thesis excluded from validation: %s — %s",
                      t.name, tmp.stage2_reason or tmp.stage1_reason)

    if not gradeable:
        _log.warning("No theses passed Stage 1+2 — validation cannot run")
        return PipelineValidationResult(
            buckets={}, admitted_vs_rejected_gap=None, admitted_vs_rejected_pval=None,
            admitted_vs_random_gap=None, random_p_value=None, random_distribution=[],
            n_windows=0, n_theses=0, train_days=train_days, test_days=test_days,
            n_random_trials=n_random_trials, decisions=[],
        )

    # Build window indices
    windows: list[tuple[int, int, int]] = []
    t = train_days
    while t + test_days <= T:
        windows.append((t - train_days, t, t + test_days))
        t += test_days

    if not windows:
        _log.warning("Not enough history for even one walk-forward window "
                     "(need %d + %d = %d days, have %d)",
                     train_days, test_days, train_days + test_days, T)
        return PipelineValidationResult(
            buckets={}, admitted_vs_rejected_gap=None, admitted_vs_rejected_pval=None,
            admitted_vs_random_gap=None, random_p_value=None, random_distribution=[],
            n_windows=0, n_theses=len(gradeable), train_days=train_days,
            test_days=test_days, n_random_trials=n_random_trials, decisions=[],
        )

    decisions: list[ThesisWindowDecision] = []

    for w_idx, (tr_s, tr_e, te_e) in enumerate(windows):
        train_r  = returns.iloc[tr_s:tr_e]
        test_r   = returns.iloc[tr_e:te_e]
        train_rg = regimes.iloc[tr_s:tr_e]
        test_rg  = regimes.iloc[tr_e:te_e]

        for thesis in gradeable:
            # ── Stage 3 on training data only ────────────────────────────────
            s3_track, s3_sign, s3_conf = _run_stage3_on_window(
                thesis, train_r, train_rg,
            )

            # ── Stage 5 on training data only ────────────────────────────────
            decision, n_tr, sh, dsr = _run_stage5_decision(
                thesis, train_r, train_rg,
                stage3_confirmed=s3_conf,
                stage3_sign_matched=s3_sign,
                n_strategies=n_strategies,
            )

            # ── OOS outcome (invisible at decision time) ──────────────────────
            oos_ret, oos_n = _compute_oos_return(thesis, test_r, test_rg)

            decisions.append(ThesisWindowDecision(
                thesis_name=thesis.name,
                window_idx=w_idx,
                train_end=returns.index[tr_e - 1],
                test_end=returns.index[te_e - 1],
                stage3_track=s3_track,
                stage3_sign_matched=s3_sign,
                stage3_confirmed=s3_conf,
                train_n_trades=n_tr,
                train_sharpe=round(sh, 4),
                dsr_prob=round(dsr, 4),
                pipeline_decision=decision,
                oos_mean_return=oos_ret,
                oos_n_signals=oos_n,
            ))

    # ── Aggregate by bucket ───────────────────────────────────────────────────
    # Only include thesis-window pairs where at least one OOS signal fired.
    # A thesis that never entered in the test window has no outcome to measure.
    bucket_raw: dict[str, list[float]] = {
        "admit": [], "mt": [], "ie": [], "reject": [],
    }
    bucket_windows: dict[str, set] = {k: set() for k in bucket_raw}

    for d in decisions:
        if d.oos_mean_return is None:
            continue
        key = d.pipeline_decision
        if key in bucket_raw:
            bucket_raw[key].append(d.oos_mean_return)
            bucket_windows[key].add(d.window_idx)

    buckets: dict[str, BucketStats] = {}
    for key in ("admit", "mt", "ie", "reject"):
        bs = _bucket_stats(bucket_raw[key])
        bs.n_windows = len(bucket_windows[key])
        buckets[key] = bs

    # ── Gap 1: admitted vs rejected ───────────────────────────────────────────
    adm = bucket_raw["admit"]
    rej = bucket_raw["reject"]

    if len(adm) > 0 and len(rej) > 0:
        gap1 = float(np.mean(adm) - np.mean(rej))
        # Welch's t-test (unequal variance)
        t_stat, pval = _scipy_stats.ttest_ind(adm, rej, equal_var=False)
        # One-sided p-value: H1 = admit > reject
        p1 = float(pval / 2 if t_stat > 0 else 1.0 - pval / 2)
    else:
        gap1, p1 = None, None

    # ── Gap 2: admitted vs random ─────────────────────────────────────────────
    # At each window, pool all theses with OOS signals (regardless of decision).
    # Randomly select n_admitted of them, compute mean OOS return.
    # Repeat across windows, average the per-window random means.
    random_means: list[float] = []
    adm_mean_oos = float(np.mean(adm)) if adm else None

    # Build per-window pools and admission counts
    per_window_pool: dict[int, list[float]] = {}
    per_window_n_adm: dict[int, int] = {}

    for d in decisions:
        if d.oos_mean_return is None:
            continue
        per_window_pool.setdefault(d.window_idx, []).append(d.oos_mean_return)
        if d.pipeline_decision == "admit":
            per_window_n_adm[d.window_idx] = (
                per_window_n_adm.get(d.window_idx, 0) + 1
            )

    # 1000 trials: each trial draws the same N per window as the pipeline admitted
    for _ in range(n_random_trials):
        trial_oos: list[float] = []
        for w_idx, pool in per_window_pool.items():
            n_adm = per_window_n_adm.get(w_idx, 0)
            if n_adm == 0 or n_adm > len(pool):
                continue
            sample = rng.choice(pool, size=n_adm, replace=False)
            trial_oos.extend(sample.tolist())
        if trial_oos:
            random_means.append(float(np.mean(trial_oos)))

    if adm_mean_oos is not None and random_means:
        gap2 = float(adm_mean_oos - np.mean(random_means))
        # P(random >= admitted) — fraction of random draws that beat the pipeline
        p2   = float(np.mean([m >= adm_mean_oos for m in random_means]))
    else:
        gap2, p2 = None, None

    return PipelineValidationResult(
        buckets=buckets,
        admitted_vs_rejected_gap=round(gap1, 4) if gap1 is not None else None,
        admitted_vs_rejected_pval=round(p1, 4)  if p1  is not None else None,
        admitted_vs_random_gap=round(gap2, 4)   if gap2 is not None else None,
        random_p_value=round(p2, 4)             if p2  is not None else None,
        random_distribution=random_means,
        n_windows=len(windows),
        n_theses=len(gradeable),
        train_days=train_days,
        test_days=test_days,
        n_random_trials=n_random_trials,
        decisions=decisions,
    )

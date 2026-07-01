"""
Thesis-first strategy pipeline.

Five ordered stages. No candidate advances without passing the one before it.

Stage 1 – THESIS       : completeness gate on the written mechanism.
Stage 2 – SIGNAL       : leg validation at construction; hard-fail on missing data.
Stage 3 – CONFIRMATION : prior-aligned test (LP-IRF for conflict theses,
                          regime-conditional for macro theses). Sign check against
                          the thesis's predicted direction is a binary gate.
Stage 4 – SIZING       : vol-target position scaled by IRF confirmation strength
                          and capped per conflict source.
Stage 5 – GRADE        : thesis-consistent rubric — high Sharpe without a confirmed
                          mechanism is capped, not rewarded.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm as _st_norm

_log = logging.getLogger(__name__)

# ── The 12 transmission channels that appear in CONFLICTS[*]["transmission"] ─
TPS_CHANNELS = (
    "oil_gas",
    "metals",
    "agriculture",
    "shipping",
    "chokepoint",
    "sanctions",
    "equity_sector",
    "fx",
    "inflation",
    "supply_chain",
    "credit",
    "energy_infra",
)

ConfirmationTrack = Literal["lp_irf", "regime_conditional"]
ConfirmationState = Literal["confirm", "reject", "insufficient_evidence", "mechanism_not_tradeable"]


# ── Stage 1: Thesis ───────────────────────────────────────────────────────────

@dataclass
class ThesisBlock:
    """
    Economic mechanism that must exist before any signal is constructed.

    Fields mirror the four required elements from the pipeline spec:
    (a) shock, (b) channel, (c) predicted_sign + horizon, (d) persistence.
    """
    shock: str                      # (a) what shock or regime state this exploits
    tps_channels: list[str]         # (b) which of the 12 TPS channels carry the transmission
    conflict_id: str | None         # which CONFLICTS entry drives this, or None for pure-macro
    chokepoint: str | None          # named physical chokepoint if applicable
    predicted_sign: dict[str, int]  # (c) {asset_name: +1 | -1}
    horizon_days: int               # (c) expected transmission horizon
    persistence: str                # (d) why the edge won't be immediately arbitraged

    def validate(self) -> tuple[bool, str]:
        """Check all required fields are non-empty. Returns (ok, reason)."""
        if not self.shock.strip():
            return False, "thesis.shock is empty"
        if not self.tps_channels:
            return False, "thesis.tps_channels is empty — name at least one channel"
        bad = [c for c in self.tps_channels if c not in TPS_CHANNELS]
        if bad:
            return False, f"unknown TPS channels: {bad}"
        if not self.predicted_sign:
            return False, "thesis.predicted_sign is empty"
        if self.horizon_days < 1:
            return False, "thesis.horizon_days must be ≥ 1"
        if not self.persistence.strip():
            return False, "thesis.persistence is empty"
        return True, ""


# ── Stage 2: Signal ───────────────────────────────────────────────────────────

@dataclass
class SignalSpec:
    """
    Entry/exit rule expressed only in terms of variables the thesis named.
    Built from the thesis; never adds predictors the thesis didn't justify.
    """
    assets: list[str]               # legs (must all exist in return data)
    direction: list[str]            # "Long" | "Short" per leg
    regime: list[int]               # trigger regime codes
    holding_period: int             # must equal thesis.horizon_days
    signal_vars: list[str]          # TPS channel names or asset names actually used in signal
    leg_weights: list[float] | None = None


# ── Stage 3: Confirmation ─────────────────────────────────────────────────────

@dataclass
class IRFPoint:
    horizon: int
    coef: float
    ci_lo: float
    ci_hi: float
    pval: float


@dataclass
class IRFConfirmation:
    """
    Result of prior-aligned validation for one strategy.

    For conflict theses: LP-IRF from GDELT shock.
    For regime/macro theses: regime-conditional return test.

    sign_matched is set independently of significance: it is True whenever the
    majority of legs point in the predicted direction, regardless of p-values.
    This lets Stage 5 distinguish INSUFFICIENT_EVIDENCE (sign OK, trade count
    too low to grade) from REJECT (sign wrong, mechanism contradicted).
    """
    track: ConfirmationTrack
    per_leg: dict[str, dict]        # {asset: {matched_sign, significant, irf_coef, pval}}
    irf_df: pd.DataFrame | None     # LP-IRF result (horizons 0–20), None for regime track
    regime_stats: dict | None       # regime-conditional mean returns, None for LP track
    confirmation_score: float       # fraction of legs confirming (sign + significance)
    sign_matched: bool              # majority of legs point in the predicted direction
    stage_passed: bool              # True iff sign_matched AND sufficient significance
    rejection_reason: str | None


# ── Stage 4: Sizing ───────────────────────────────────────────────────────────

@dataclass
class SizingRule:
    """
    Explicit sizing rule derived from thesis conviction and empirical risk.
    """
    target_annual_vol: float        # portfolio vol target for this strategy
    estimated_strat_vol: float      # annualized std of OOS trade returns
    base_weight_pct: float          # vol-target implied weight before scaling
    irf_scale_factor: float         # confirmation_score — scales weight up/down
    conflict_cap_pct: float         # hard ceiling per conflict source
    final_weight_pct: float         # min(base × irf_scale, cap)
    rule_text: str                  # human-readable description


# ── Stage 5: Thesis-consistent grade ─────────────────────────────────────────

@dataclass
class ThesisGrade:
    """
    Grade that audits thesis-consistency, not just Sharpe.

    Four mutually exclusive outcomes:
      confirmation_state = "confirm"                 → grade A/B/C/D (normal rubric)
      confirmation_state = "mechanism_not_tradeable" → grade "MT": Stage 3 confirmed the
                                                        transmission but DSR < 50% — no
                                                        edge after costs + trial deflation.
                                                        Thesis correct; market already priced
                                                        it. Not a failure — do not discard.
      confirmation_state = "reject"                  → grade F (Stage 3 NOT confirmed,
                                                        mechanism contradicted or inverted)
      confirmation_state = "insufficient_evidence"   → grade "IE" (sign OK, < 20 trades)

    The fix for MT     is not "discard" — find the faster signal that beats the market's
                        pricing speed, or use the thesis as a risk factor, not a trade.
    The fix for REJECT is "discard the thesis."
    The fix for IE     is "get more trades."
    Conflating MT with REJECT throws away correct theses.
    """
    grade: str                      # A/B/C/D/F · "MT" (real, not tradeable) · "IE" (insufficient evidence)
    confirmation_state: ConfirmationState
    score: int                      # DSR prob × 100  (0 for IE)
    has_thesis: bool
    stage3_confirmed: bool          # Stage 3 passed (sign + significance)
    sign_matches_oos: bool | None   # OOS return sign consistent with thesis
    dsr_prob: float
    pbo: float | None
    rationale: str                  # one sentence explaining the grade
    flags: list[str]
    capped_reason: str | None       # set when Sharpe is high but mechanism unconfirmed


# ── Top-level strategy object ─────────────────────────────────────────────────

@dataclass
class ThesisStrategy:
    """
    A strategy that must pass all five stages in order.
    Fields are populated progressively as each stage runs.
    """
    # ── Identity ─────────────────────────────────────────────────────────────
    name: str
    category: str

    # ── Stage 1: Thesis ──────────────────────────────────────────────────────
    thesis: ThesisBlock
    stage1_passed: bool = False
    stage1_reason: str = ""

    # ── Stage 2: Signal ──────────────────────────────────────────────────────
    signal: SignalSpec | None = None
    stage2_passed: bool = False
    stage2_reason: str = ""

    # ── Stage 3: Confirmation ─────────────────────────────────────────────────
    confirmation: IRFConfirmation | None = None
    stage3_passed: bool = False
    stage3_reason: str = ""

    # ── Stage 4: Sizing ──────────────────────────────────────────────────────
    sizing: SizingRule | None = None

    # ── Stage 5: Grade ───────────────────────────────────────────────────────
    wf_result: dict = field(default_factory=dict)
    thesis_grade: ThesisGrade | None = None

    @property
    def final_stage(self) -> int:
        """Highest stage this strategy has reached."""
        if self.thesis_grade is not None:
            return 5
        if self.sizing is not None:
            return 4
        if self.confirmation is not None:
            return 3
        if self.stage2_passed:
            return 2
        if self.stage1_passed:
            return 1
        return 0


# ── Pipeline stage functions ──────────────────────────────────────────────────

def run_stage1(strategy: ThesisStrategy) -> ThesisStrategy:
    """Validate thesis completeness. Gate: all four fields non-empty."""
    ok, reason = strategy.thesis.validate()
    strategy.stage1_passed = ok
    strategy.stage1_reason = reason or "Thesis complete"
    return strategy


def run_stage2(
    strategy: ThesisStrategy,
    available_columns: set[str],
) -> ThesisStrategy:
    """
    Leg validation at construction time.

    Hard-fail (do not degrade silently) if any declared leg is absent from the
    return data — this is the bug that caused two Gold strategies to appear
    identical by collapsing to a single-leg Long Gold.
    """
    if not strategy.stage1_passed:
        strategy.stage2_reason = "Stage 1 not passed"
        return strategy

    sig = strategy.signal
    if sig is None:
        strategy.stage2_reason = "No signal spec attached"
        return strategy

    missing = [a for a in sig.assets if a not in available_columns]
    if missing:
        strategy.stage2_passed = False
        strategy.stage2_reason = f"Missing legs: {missing} — rejected at construction"
        return strategy

    # Holding period must match thesis horizon
    if sig.holding_period != strategy.thesis.horizon_days:
        strategy.stage2_passed = False
        strategy.stage2_reason = (
            f"Signal holding_period={sig.holding_period}d ≠ "
            f"thesis.horizon_days={strategy.thesis.horizon_days}d — "
            "the backtest horizon must match the thesis horizon"
        )
        return strategy

    strategy.stage2_passed = True
    strategy.stage2_reason = (
        f"All {len(sig.assets)} legs present; "
        f"signal uses only thesis-named vars: {sig.signal_vars}"
    )
    return strategy


def _lp_irf_confirmation(
    strategy: ThesisStrategy,
    returns: pd.DataFrame,
    conflict_id: str,
) -> IRFConfirmation:
    """
    Run LP-IRF using GDELT conflict shock and compare against thesis prediction.
    Falls back to regime-conditional if GDELT data is unavailable.
    """
    thesis = strategy.thesis
    sig    = strategy.signal
    assert sig is not None

    assets_to_test = [a for a in sig.assets if a in returns.columns]

    try:
        from src.analysis.local_projection import _build_shock_series, compute_lp_irf
        trading_idx = returns.dropna(how="all").index
        shock = _build_shock_series(conflict_id, trading_idx)

        if shock is None or len(shock) < 100:
            raise ValueError("Insufficient GDELT history — falling back to regime track")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            irf_df = compute_lp_irf(
                shock, returns[assets_to_test],
                conflict_id=conflict_id,
                _n_shock=len(shock),
                _n_rows=len(returns),
            )

        if irf_df.empty:
            raise ValueError("LP-IRF returned empty — falling back to regime track")

        # Check sign + significance at thesis horizon (±5d window)
        h_target = thesis.horizon_days
        per_leg: dict[str, dict] = {}
        for asset, pred_sign in thesis.predicted_sign.items():
            if asset not in assets_to_test:
                per_leg[asset] = {"matched_sign": False, "significant": False,
                                   "irf_coef": None, "pval": None, "note": "not in return data"}
                continue
            asset_irf = irf_df[irf_df["asset"] == asset].copy()
            if asset_irf.empty:
                per_leg[asset] = {"matched_sign": False, "significant": False,
                                   "irf_coef": None, "pval": None, "note": "no IRF computed"}
                continue
            # Row closest to thesis horizon
            idx_closest = (asset_irf["horizon"] - h_target).abs().idxmin()
            row = asset_irf.loc[idx_closest]
            coef, pval = float(row["coef"]), float(row["pval"])
            matched = (coef > 0) == (pred_sign == 1)
            sig_flag = pval < 0.10
            per_leg[asset] = {
                "matched_sign": matched,
                "significant":  sig_flag,
                "irf_coef":     round(coef, 4),
                "pval":         round(pval, 4),
            }

        n_match  = sum(v["matched_sign"] for v in per_leg.values())
        n_sig    = sum(v["matched_sign"] and v["significant"] for v in per_leg.values())
        n_total  = len(per_leg)
        conf_score   = (n_sig / n_total) if n_total > 0 else 0.0
        sign_matched = n_match > (n_total / 2)
        passed       = sign_matched and conf_score > 0.0

        return IRFConfirmation(
            track="lp_irf",
            per_leg=per_leg,
            irf_df=irf_df,
            regime_stats=None,
            confirmation_score=round(conf_score, 4),
            sign_matched=sign_matched,
            stage_passed=passed,
            rejection_reason=None if passed else (
                f"Only {n_match}/{n_total} legs match predicted sign "
                f"({n_sig} significant at p<0.10)"
            ),
        )

    except Exception as exc:
        _log.info("LP-IRF fell back to regime-conditional for %s: %s", strategy.name, exc)
        return _regime_conditional_confirmation(strategy, returns)


def _regime_conditional_confirmation(
    strategy: ThesisStrategy,
    returns: pd.DataFrame,
    regimes: pd.Series | None = None,
) -> IRFConfirmation:
    """
    Regime-conditional return test for non-conflict (macro/regime) strategies.

    In the trigger regime, each leg's mean daily return must match the predicted sign.
    We also compute the t-statistic for the mean to assess significance.
    """
    thesis = strategy.thesis
    sig    = strategy.signal
    assert sig is not None

    # Build a synthetic regime series if not provided
    if regimes is None:
        try:
            from src.analysis.correlations import (
                average_cross_corr_series, detect_correlation_regime
            )
            # Use the equity columns only for cross-corr
            eq_cols = [c for c in returns.columns
                       if c in {"S&P 500", "Eurostoxx 50", "Nikkei 225",
                                "Shanghai Comp", "Sensex", "Nifty 50", "DAX",
                                "CAC 40", "FTSE 100", "Hang Seng"}]
            cmd_cols = [c for c in returns.columns if c not in eq_cols]
            if eq_cols and cmd_cols:
                eq_r  = returns[eq_cols]
                cmd_r = returns[cmd_cols]
                avg_c = average_cross_corr_series(eq_r, cmd_r, window=60)
                regimes = detect_correlation_regime(avg_c).reindex(
                    returns.index, method="ffill"
                )
        except Exception:
            pass

    per_leg: dict[str, dict] = {}
    regime_stats: dict = {}

    for asset, direction, pred_sign in zip(
        sig.assets, sig.direction, [thesis.predicted_sign.get(a, 0) for a in sig.assets]
    ):
        if asset not in returns.columns:
            per_leg[asset] = {"matched_sign": False, "significant": False,
                               "mean_ret": None, "t_stat": None}
            continue

        col = returns[asset].dropna()

        if regimes is not None:
            r_aligned = regimes.reindex(col.index, method="ffill")
            in_trigger = r_aligned.isin(sig.regime)
            col_regime = col[in_trigger]
        else:
            col_regime = col   # fallback: use all observations

        if len(col_regime) < 20:
            per_leg[asset] = {"matched_sign": False, "significant": False,
                               "mean_ret": None, "t_stat": None,
                               "note": "too few regime observations"}
            continue

        mu  = float(col_regime.mean())
        se  = float(col_regime.std() / np.sqrt(len(col_regime)))
        t   = mu / se if se > 1e-10 else 0.0
        # Expected positive if direction==Long and pred_sign==+1
        expected_positive = (direction.lower() == "long" and pred_sign == 1) or \
                            (direction.lower() == "short" and pred_sign == -1)
        matched = (mu > 0) == expected_positive
        sig_10  = abs(t) > _st_norm.ppf(0.90)   # one-sided 10%

        per_leg[asset]    = {
            "matched_sign": matched,
            "significant":  sig_10,
            "mean_ret":     round(mu * 100, 4),   # in %
            "t_stat":       round(t, 3),
        }
        regime_stats[asset] = {"mean_ret_pct": round(mu * 100, 4), "t_stat": round(t, 3)}

    n_total      = len(per_leg)
    n_match      = sum(v["matched_sign"] for v in per_leg.values())
    n_sig        = sum(v["matched_sign"] and v["significant"] for v in per_leg.values())
    conf_score   = (n_sig / n_total) if n_total > 0 else 0.0
    sign_matched = n_match > (n_total / 2)
    passed       = sign_matched

    return IRFConfirmation(
        track="regime_conditional",
        per_leg=per_leg,
        irf_df=None,
        regime_stats=regime_stats,
        confirmation_score=round(conf_score, 4),
        sign_matched=sign_matched,
        stage_passed=passed,
        rejection_reason=None if passed else (
            f"Only {n_match}/{n_total} legs match predicted sign in trigger regime"
        ),
    )


def run_stage3(
    strategy: ThesisStrategy,
    returns: pd.DataFrame,
    regimes: pd.Series | None = None,
) -> ThesisStrategy:
    """
    Prior-aligned validation. The test must match the thesis:
    - Conflict thesis with LP-IRF track → use GDELT shock + compute_lp_irf
    - Macro/regime thesis → use regime-conditional return test

    A correct-magnitude, wrong-sign result is a REJECT.
    """
    if not strategy.stage2_passed:
        strategy.stage3_reason = "Stage 2 not passed"
        return strategy

    thesis = strategy.thesis
    conflict_id = thesis.conflict_id

    if conflict_id is not None:
        conf = _lp_irf_confirmation(strategy, returns, conflict_id)
    else:
        conf = _regime_conditional_confirmation(strategy, returns, regimes)

    strategy.confirmation = conf
    strategy.stage3_passed = conf.stage_passed
    strategy.stage3_reason = (
        conf.rejection_reason or
        f"{conf.track} — {conf.confirmation_score:.0%} of legs confirm predicted sign"
    )
    return strategy


def run_stage4(
    strategy: ThesisStrategy,
    wf_result: dict,
    target_annual_vol: float = 0.10,
    conflict_cap_pct: float = 0.20,
) -> ThesisStrategy:
    """
    Compute explicit position sizing rule.

    size = min(target_vol / strat_vol × confirmation_score, conflict_cap)

    Output is the sizing rule stored on the strategy and a human-readable
    description that travels with every card.
    """
    strategy.wf_result = wf_result

    if not strategy.stage3_passed:
        # Sizing is not meaningful for unconfirmed strategies, but compute it
        # defensively so Stage 5 can still run (grade will be D/F regardless).
        pass

    trade_returns = wf_result.get("trade_returns", [])
    if trade_returns and len(trade_returns) >= 3:
        h = wf_result.get("_holding_days", strategy.signal.holding_period if strategy.signal else 30)
        ann_factor  = np.sqrt(252.0 / max(h, 1))
        tr_arr      = np.array(trade_returns, dtype=float)
        strat_vol   = float(tr_arr.std()) * ann_factor / 100.0   # annualized, decimal
    else:
        strat_vol = 0.20   # fallback 20% vol assumption

    strat_vol = max(strat_vol, 0.01)   # floor to avoid division by zero

    conf_score   = (strategy.confirmation.confirmation_score
                    if strategy.confirmation else 0.0)
    base_weight  = target_annual_vol / strat_vol
    final_weight = min(base_weight * conf_score, conflict_cap_pct)

    rule_text = (
        f"Vol-target: {target_annual_vol:.0%} annual / {strat_vol:.0%} strat vol "
        f"= {base_weight:.0%} gross weight. "
        f"Scale by IRF confirmation {conf_score:.0%} → {base_weight * conf_score:.0%}. "
        f"Cap at {conflict_cap_pct:.0%} per conflict source → "
        f"final weight {final_weight:.0%}."
    )

    strategy.sizing = SizingRule(
        target_annual_vol=round(target_annual_vol, 4),
        estimated_strat_vol=round(strat_vol, 4),
        base_weight_pct=round(base_weight * 100, 2),
        irf_scale_factor=round(conf_score, 4),
        conflict_cap_pct=round(conflict_cap_pct * 100, 2),
        final_weight_pct=round(final_weight * 100, 2),
        rule_text=rule_text,
    )
    return strategy


def run_stage5(
    strategy: ThesisStrategy,
    n_strategies: int,
    is_economic_prior: bool = True,
) -> ThesisStrategy:
    """
    Thesis-consistent grade.

    Four mutually exclusive outcomes (in routing order):
      IE  (grade="IE") — n_trades < 20 and sign matched; thesis parked, not discarded.
      MT  (grade="MT") — Stage 3 confirmed + DSR < 50%; mechanism real, edge gone.
                         Exits before caps — neither the base grade nor any cap can
                         override an explicit S3-confirm / DSR-fail split.
      CONFIRM (A/B/C/D) — Stage 3 confirmed or unexplained Sharpe; graded by DSR rubric.
      REJECT (F)         — Stage 3 NOT confirmed and OOS inverted or DSR < 0.25.

    The fix for MT     is "find a faster signal or use as a hedge."
    The fix for REJECT is "discard the thesis."
    The fix for IE     is "get more trades."
    """
    wf = strategy.wf_result
    flags: list[str] = []
    capped_reason: str | None = None

    has_thesis    = strategy.stage1_passed
    stage3_ok     = strategy.stage3_passed
    sign_matched  = (strategy.confirmation.sign_matched
                     if strategy.confirmation else False)
    n_trades      = wf.get("n_trades", 0)
    pbo           = wf.get("pbo")
    trade_returns = wf.get("trade_returns", [])

    # ── Low trade count: route to IE or REJECT before grading ────────────────
    if n_trades < 20:
        if sign_matched:
            # Thesis direction was not contradicted — park it
            strategy.thesis_grade = ThesisGrade(
                grade="IE",
                confirmation_state="insufficient_evidence",
                score=0,
                has_thesis=has_thesis,
                stage3_confirmed=stage3_ok,
                sign_matches_oos=None,
                dsr_prob=0.0, pbo=pbo,
                rationale=(
                    f"Only {n_trades} OOS trades (< 20) — not enough history to grade. "
                    "Thesis sign not contradicted; mechanism is plausible but untested. "
                    "Resolution: reduce holding period to increase trade frequency, "
                    "or wait for more regime history."
                ),
                flags=[
                    f"Only {n_trades} OOS trades — DSR requires ≥ 20",
                    "Stage 3 sign direction matched — mechanism not contradicted",
                ],
                capped_reason=None,
            )
        else:
            # Sign was wrong AND trade count is low → still a reject
            strategy.thesis_grade = ThesisGrade(
                grade="F",
                confirmation_state="reject",
                score=0,
                has_thesis=has_thesis,
                stage3_confirmed=False,
                sign_matches_oos=None,
                dsr_prob=0.0, pbo=pbo,
                rationale=(
                    f"Only {n_trades} OOS trades and Stage 3 sign did not match — "
                    "mechanism contradicted even in the limited available data. REJECT."
                ),
                flags=[
                    f"Only {n_trades} OOS trades",
                    "Stage 3 sign direction WRONG — mechanism contradicted",
                ],
                capped_reason=None,
            )
        return strategy

    # ── DSR from wf result (already computed with correct N in walk_forward_backtest) ─
    qc       = wf.get("qc", {})
    dsr_prob = float(qc.get("dsr_prob", 0.0))
    flags.extend(qc.get("flags", []))

    # ── OOS sign consistency with thesis ────────────────────────────────────
    sign_matches_oos: bool | None = None
    if trade_returns and strategy.signal and strategy.thesis:
        tr_arr   = np.array(trade_returns, dtype=float)
        oos_sign = 1 if tr_arr.mean() > 0 else -1
        # Thesis predicts the aggregate trade direction: sum of signed leg contributions
        thesis_net = sum(strategy.thesis.predicted_sign.get(a, 0) * (1 if d.lower() == "long" else -1)
                         for a, d in zip(strategy.signal.assets, strategy.signal.direction))
        thesis_sign = 1 if thesis_net > 0 else -1
        sign_matches_oos = (oos_sign == thesis_sign)
        if not sign_matches_oos:
            flags.append(
                "OOS net return sign OPPOSES thesis prediction — "
                "strategy is directionally inverted relative to mechanism; REJECT"
            )

    # ── MT: mechanism real, not tradeable ───────────────────────────────────
    # This block must fire BEFORE the base-grade assignment and caps so that
    # neither can silently override an explicit mechanism/edge disagreement.
    #
    # Condition: Stage 3 confirmed (transmission IS real) + DSR < 50% (edge
    # does not survive costs and N-trial deflation). These two facts can and
    # do coexist. They mean the thesis is correct and the market has already
    # arbitraged the edge. The honest label is MT, not a grade and not a reject.
    #
    # Note on OOS sign: if Stage 3 is confirmed but OOS returns are negative
    # (sign_matches_oos is False), the most likely explanation is that the
    # market prices the signal before our entry fires — "already priced in."
    # This is a sub-type of MT, not REJECT, because Stage 3 used full-history
    # LP-IRF / regime data, which is more statistically reliable than the OOS
    # subsample sign. We label it MT and surface the sub-reason explicitly.
    if stage3_ok and dsr_prob < 0.50:
        _mt_flags: list[str] = []
        if sign_matches_oos is False:
            _mt_sub = (
                "OOS returns negative despite confirmed mechanism — "
                "market likely prices the signal before entry fires (already-priced). "
                "Find a faster signal or use as a risk-factor hedge, not a directional trade."
            )
        elif dsr_prob < 0.25:
            _mt_sub = (
                f"DSR {dsr_prob:.0%} — net of transaction costs and N={n_strategies} "
                f"trial deflation, expected return is negative. "
                "Mechanism confirmed but fully arbitraged."
            )
        else:
            _mt_sub = (
                f"DSR {dsr_prob:.0%} — transmission is real but the edge is too small "
                f"to survive costs ({n_strategies}-trial SR* deflation). "
                "Mechanism is correct; monetisation requires faster entry or cheaper execution."
            )
        _mt_flags.append(_mt_sub)
        _mt_flags.extend(qc.get("flags", []))

        strategy.thesis_grade = ThesisGrade(
            grade="MT",
            confirmation_state="mechanism_not_tradeable",
            score=round(dsr_prob * 100),
            has_thesis=has_thesis,
            stage3_confirmed=True,
            sign_matches_oos=sign_matches_oos,
            dsr_prob=round(dsr_prob, 4),
            pbo=round(pbo, 4) if pbo is not None and np.isfinite(pbo) else None,
            rationale=(
                "Stage 3 confirmed: transmission mechanism is real. "
                f"DSR {dsr_prob:.0%} < 50% — no tradeable edge after deflation and costs. "
                + _mt_sub
            ),
            flags=_mt_flags,
            capped_reason=None,
        )
        return strategy

    # ── Base grade from DSR ──────────────────────────────────────────────────
    # (reached only when Stage 3 is NOT confirmed, OR when Stage 3 IS confirmed
    # and DSR ≥ 50% — i.e., the edge survived. MT has already returned above.)
    if dsr_prob >= 0.95:   grade = "A"
    elif dsr_prob >= 0.75: grade = "B"
    elif dsr_prob >= 0.50: grade = "C"
    elif dsr_prob >= 0.25: grade = "D"
    else:                  grade = "F"

    # ── Thesis-consistency caps (applied after DSR base grade) ───────────────
    # Cap 1: Wrong OOS sign → F.
    # At this point Stage 3 is NOT confirmed (MT with confirmed S3 exited above),
    # so an inverted OOS sign genuinely contradicts the mechanism — REJECT.
    if sign_matches_oos is False:
        grade = "F"
        flags.append("Grade forced to F: OOS direction contradicts thesis mechanism")

    # Cap 2: Stage 3 not confirmed → max C
    # A strategy may have a valid Sharpe purely by accident in the test window.
    if not stage3_ok and grade in ("A", "B"):
        grade = "C"
        capped_reason = (
            "Stage 3 not confirmed: "
            + (strategy.stage3_reason or "mechanism not validated by IRF or regime test")
            + " — unexplained Sharpe capped at C"
        )
        flags.append(capped_reason)

    # Cap 3: No thesis → max C (shouldn't reach here, but defensive)
    if not has_thesis and grade in ("A", "B"):
        grade = "C"
        capped_reason = "No thesis provided — unexplained Sharpe capped at C"
        flags.append(capped_reason)

    # Cap 4: PBO > 0.5 → max D
    if pbo is not None and np.isfinite(pbo) and pbo > 0.5 and grade not in ("D", "F"):
        grade = "D"
        flags.append(f"CSCV PBO {pbo:.0%} — majority of block splits show IS→OOS sign flip")

    # Cap 5: Low N (< 20 handled above, but 20-30 warrants a flag)
    if n_trades < 30 and grade in ("A", "B"):
        grade = "C"
        flags.append(f"LOW N ({n_trades} trades) — Sharpe SE too wide for A/B with thesis-first rigor")

    # ── Rationale ────────────────────────────────────────────────────────────
    if grade == "A":
        rationale = (
            "Stated mechanism · Stage 3 confirmed · DSR ≥ 95% · "
            "OOS direction consistent with thesis"
        )
    elif grade == "B":
        rationale = f"Stated mechanism · Stage 3 confirmed · DSR {dsr_prob:.0%}"
    elif grade == "C":
        rationale = (
            capped_reason if capped_reason else
            f"Partial confirmation or DSR {dsr_prob:.0%} — mechanism plausible but not decisive"
        )
    elif grade == "D":
        rationale = f"Mechanism stated but Stage 3 failed or DSR {dsr_prob:.0%} — weak empirical support"
    else:
        rationale = f"DSR {dsr_prob:.0%} or mechanism contradiction — REJECT"

    # confirmation_state: reject when sign inverted OR grade F; confirm otherwise
    if sign_matches_oos is False or grade == "F":
        conf_state: ConfirmationState = "reject"
    else:
        conf_state = "confirm"

    strategy.thesis_grade = ThesisGrade(
        grade=grade,
        confirmation_state=conf_state,
        score=round(dsr_prob * 100),
        has_thesis=has_thesis,
        stage3_confirmed=stage3_ok,
        sign_matches_oos=sign_matches_oos,
        dsr_prob=round(dsr_prob, 4),
        pbo=round(pbo, 4) if pbo is not None and np.isfinite(pbo) else None,
        rationale=rationale,
        flags=flags,
        capped_reason=capped_reason,
    )
    return strategy


def run_full_pipeline(
    strategy: ThesisStrategy,
    returns: pd.DataFrame,
    avg_corr: pd.Series,
    regimes: pd.Series | None = None,
    n_strategies: int = 9,
    is_economic_prior: bool = True,
    target_annual_vol: float = 0.10,
    conflict_cap_pct: float = 0.20,
) -> ThesisStrategy:
    """
    Execute all five stages in order. Each stage mutates and returns the strategy.
    A failed stage still allows the next to run (so we can report what failed),
    but the final grade reflects all failures.
    """
    strategy = run_stage1(strategy)
    if not strategy.stage1_passed:
        _log.warning("Stage 1 FAIL [%s]: %s", strategy.name, strategy.stage1_reason)
        # Stage 1 is a hard gate — no further processing without a thesis
        return strategy

    strategy = run_stage2(strategy, available_columns=set(returns.columns))
    if not strategy.stage2_passed:
        _log.warning("Stage 2 FAIL [%s]: %s", strategy.name, strategy.stage2_reason)
        return strategy

    # Stage 3: confirmation (runs even if we'll reject later — we want the IRF curve)
    strategy = run_stage3(strategy, returns, regimes)

    # Walk-forward backtest (call from UI with caching; here we run directly)
    from src.analysis.backtest import walk_forward_backtest
    sig = strategy.signal
    assert sig is not None
    trade_stub = {
        "name":           strategy.name,
        "assets":         sig.assets,
        "direction":      sig.direction,
        "regime":         sig.regime,
        "holding_period": sig.holding_period,
    }
    wf = walk_forward_backtest(
        returns=returns,
        avg_corr=avg_corr,
        trade=trade_stub,
        leg_weights=sig.leg_weights,
        n_strategies=n_strategies,
        is_economic_prior=is_economic_prior,
    )

    strategy = run_stage4(strategy, wf, target_annual_vol=target_annual_vol,
                          conflict_cap_pct=conflict_cap_pct)
    strategy = run_stage5(strategy, n_strategies=n_strategies,
                          is_economic_prior=is_economic_prior)
    return strategy

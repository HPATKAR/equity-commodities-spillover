"""
STEP 2 of 4 — weight allocator over Step-1-eligible trades.

Each eligible trade is sized from four multiplicative factors:

    raw_w = conviction × dsr_factor × (1 / leg_vol) × liquidity

  conviction  — Stage-3 thesis confirmation_score (0–1). Measured, not
                self-reported.
  dsr_factor  — clip((DSR − 0.5) / 0.5, 0, 1) where DSR is the deflated
                Sharpe probability (Bailey & López de Prado 2014) computed on
                the trade's own backtest at n_strategies = 9. Raw Sharpe NEVER
                sizes a weight: a strategy that cannot beat the best-of-9 luck
                benchmark (DSR ≤ 0.5) sizes to exactly zero, as do stale
                strategies with < 3 historical signals.
  1/leg_vol   — inverse annualized vol of the trade's signed, equal-weight
                leg return series (vol-parity sizing).
  liquidity   — min across legs of a STATED static tier map (no volume data
                is loaded anywhere in the terminal; tiers are a registry-style
                calibration, same disclosure class as the transmission
                weights). Majors 1.0, softs/regionals ~0.75.

Raw weights are normalized to 100% gross across eligible trades, capped at
35% per trade (excess redistributed pro-rata), and EVERY final assignment —
eligible or not — is routed through Step 1's enforce_weight() choke point.
A non-allocatable trade receives literal 0.0 regardless of its metrics.

Ranking and display are Step 3+ — this module computes and stamps only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.trade_filter import enforce_weight

MAX_SINGLE_WEIGHT = 0.35
MIN_TRADES_FOR_DSR = 3
_N_STRATEGIES = 9          # honest trial count — see backtest._N_LIBRARY_STRATEGIES

# ── Deploy bar & risk appetite ────────────────────────────────────────────────
# DSR_DEPLOY_BAR is the deflated-Sharpe confidence threshold (Bailey & López de
# Prado): below it a strategy cannot be distinguished from the best-of-9 luck
# benchmark. It is the DEFAULT (Defensive) bar and the labelled reference — it
# is never silently lowered. A user-set risk appetite may lower the EFFECTIVE
# bar toward DEPLOY_BAR_FLOOR, deploying the strongest relative ideas below the
# confidence bar. That is an explicit, disclosed choice (see the desk-call
# disclosure), not a presentation tweak.
DSR_DEPLOY_BAR = 0.50       # Defensive / reference (deflated-Sharpe threshold)
DEPLOY_BAR_FLOOR = 0.05     # most-aggressive effective bar; below this = no edge

# Named appetite stops → risk appetite fraction in [0, 1].
APPETITE_STOPS: dict[str, float] = {
    "Defensive":    0.0,
    "Cautious":     0.25,
    "Balanced":     0.50,
    "Constructive": 0.75,
    "Aggressive":   1.0,
}


def effective_deploy_bar(appetite: float) -> float:
    """Map risk appetite [0,1] to the effective DSR deploy bar. Appetite 0
    keeps the strict deflated-Sharpe bar (Defensive); appetite 1 lowers it to
    DEPLOY_BAR_FLOOR (best-available). The ramp is concave (sqrt) so the upper
    half of the slider deploys progressively — a linear ramp would leave every
    mid-setting stranded above a weak-edge book's DSR mass and only the extreme
    would ever leave cash."""
    a = float(np.clip(appetite, 0.0, 1.0))
    return DSR_DEPLOY_BAR - (DSR_DEPLOY_BAR - DEPLOY_BAR_FLOOR) * (a ** 0.5)

# ── Stated liquidity tiers (calibration, not measurement) ────────────────────
LIQUIDITY_TIERS: dict[str, float] = {
    # deepest futures / index complexes
    "S&P 500": 1.0, "Nasdaq 100": 1.0, "DJIA": 1.0, "Gold": 1.0,
    "WTI Crude Oil": 1.0, "Brent Crude": 1.0, "Eurostoxx 50": 0.95,
    "DAX": 0.95, "Nikkei 225": 0.95, "FTSE 100": 0.95, "Silver": 0.9,
    "Copper": 0.9, "Natural Gas": 0.9, "Russell 2000": 0.9,
    # regional / agricultural
    "Hang Seng": 0.85, "CAC 40": 0.85, "Wheat": 0.8, "Corn": 0.8,
    "Soybeans": 0.8, "Shanghai Comp": 0.75, "CSI 300": 0.75,
    "Sensex": 0.75, "Nifty 50": 0.75, "TOPIX": 0.8,
    "Sugar #11": 0.7, "Coffee": 0.7, "Cotton": 0.7,
    "Platinum": 0.7, "Aluminum": 0.7, "Heating Oil": 0.75,
    "Gasoline (RBOB)": 0.75,
}
_DEFAULT_LIQUIDITY = 0.7


def _liquidity(trade: dict) -> float:
    assets = trade.get("assets") or []
    if not assets:
        return 0.0
    return min(LIQUIDITY_TIERS.get(a, _DEFAULT_LIQUIDITY) for a in assets)


def trade_leg_series(all_r: pd.DataFrame, assets: list[str],
                     directions: list[str]) -> pd.Series | None:
    """Signed, equal-weight leg daily return series (None if a leg is absent)."""
    legs = []
    for a, d in zip(assets, directions):
        if a not in all_r.columns:
            return None
        legs.append(all_r[a] * (1.0 if str(d).lower() == "long" else -1.0))
    if not legs:
        return None
    return pd.concat(legs, axis=1).mean(axis=1).dropna()


def trade_leg_vol(all_r: pd.DataFrame, assets: list[str],
                  directions: list[str]) -> float:
    """Annualized vol of the signed, equal-weight leg return series."""
    combo = trade_leg_series(all_r, assets, directions)
    if combo is None or len(combo) < 60:
        return float("nan")
    return float(combo.std() * np.sqrt(252))


def build_allocation_inputs(
    trades: list[dict],
    all_r: pd.DataFrame,
    regimes: pd.Series,
    thesis_results: dict[str, dict],
    n_strategies: int = _N_STRATEGIES,
    deploy_bar: float = DSR_DEPLOY_BAR,
) -> dict[str, dict]:
    """
    Assemble the four sizing factors per ELIGIBLE trade using the existing
    backtest machinery. Ineligible trades are skipped — they can never carry
    weight, so computing metrics for them would only invite misuse.

    deploy_bar is the EFFECTIVE DSR threshold sizing scales from (Defensive =
    DSR_DEPLOY_BAR = 0.50; a higher risk appetite lowers it toward
    DEPLOY_BAR_FLOOR). The raw `dsr` is stored unchanged so ranking stays
    evidence-based — only `dsr_factor` (and thus weight) responds to appetite.

    Returns {trade_name: {conviction, dsr, dsr_factor, vol, liquidity,
                          sharpe_raw, n_trades, deploy_bar}}.
    """
    bar = float(np.clip(deploy_bar, 0.0, 0.99))
    from src.analysis.backtest import (
        vectorized_backtest, deflated_sharpe_probability, _parse_holding_days,
    )

    metrics: dict[str, dict] = {}
    for t in trades:
        if not t.get("is_eligible", False):
            continue
        name = t.get("name", "")
        th = thesis_results.get(name) or {}
        conviction = float(th.get("confirmation_score", 0.0))

        assets = t.get("assets") or []
        directions = t.get("direction") or []
        bt = vectorized_backtest(
            all_r, regimes, assets, directions,
            trigger_regimes=list(t.get("regime", [])) or [1],
            holding_days=_parse_holding_days(t, default=20),
        )
        tr = [float(x) for x in (bt.get("trade_returns") or [])]
        n_tr = len(tr)
        sharpe_raw = float(bt.get("sharpe") or 0.0)
        hold_days = _parse_holding_days(t, default=20)

        # Direction-aware, regime-conditional P&L stats — the honest per-name
        # expected return + dispersion for this signal over ONE holding window.
        # These drive the scenario-free profit projection (a Short's negative
        # realised edge shows as negative, not a flat zero).
        arr = np.asarray(tr, dtype=float) if n_tr else np.asarray([0.0])
        avg_return = float(arr.mean()) if n_tr else 0.0
        ret_std = float(arr.std(ddof=1)) if n_tr >= 2 else 0.0

        if n_tr < MIN_TRADES_FOR_DSR:
            dsr = 0.0                      # stale: too few signals to trust
        else:
            sd = ret_std
            sr_trade = float(avg_return / sd) if sd > 1e-12 else 0.0
            skew = float(pd.Series(arr).skew()) if n_tr >= 3 else 0.0
            ekurt = float(pd.Series(arr).kurt()) if n_tr >= 4 else 0.0
            dsr, _sr_star = deflated_sharpe_probability(
                sr_trade, n_obs=n_tr, skew=skew, excess_kurt=ekurt,
                n_strategies=n_strategies,
            )

        vol = trade_leg_vol(all_r, assets, directions)
        metrics[name] = {
            "conviction": conviction,
            "dsr": float(dsr),
            # sizing ramp above the EFFECTIVE bar (appetite-adjusted); raw dsr
            # above is untouched so ranking stays evidence-based
            "dsr_factor": float(np.clip((dsr - bar) / (1.0 - bar), 0.0, 1.0)),
            "vol": vol,
            "liquidity": _liquidity(t),
            "sharpe_raw": sharpe_raw,      # reported for transparency, never sized on
            "n_trades": n_tr,
            "deploy_bar": bar,
            # backtest projection inputs (per holding window, %)
            "avg_return": avg_return,
            "ret_std": ret_std,
            "holding_days": hold_days,
        }
    return metrics


def allocate_weights(
    trades: list[dict],
    metrics: dict[str, dict],
    cap: float = MAX_SINGLE_WEIGHT,
    concentration: float = 1.0,
) -> list[dict]:
    """
    Pure allocator. Stamps every trade with:
      alloc_weight : float — final portfolio weight (0.0 for non-allocatable)
      alloc_detail : dict  — the factor breakdown, for Step-3 transparency

    Every assignment routes through enforce_weight(); a non-allocatable trade
    receives 0.0 no matter what its metrics say.

    EQUITY-SLEEVE MODEL. This book is the EQUITY COMPONENT of a larger
    portfolio, so it stays FULLY INVESTED — the cash / hedge overlay is the
    parent portfolio's decision, not this sleeve's. Capital is sized by each
    trade's RISK-ADJUSTED EXPECTED EDGE: the direction-aware realised Sharpe
    (backtest mean ÷ σ per holding window), shrunk by deflated-Sharpe
    confidence. A historically money-LOSING signal earns ~no capital — that
    capital is reallocated to the winners, never parked in cash — so the book's
    expected return actually reflects its best ideas rather than washing out
    across every eligible name. If literally nothing has a positive edge, the
    sleeve falls back to an inverse-vol spread (still fully invested, never
    cash). Raw weights normalize to 100% gross.

    `concentration` (driven by risk appetite) sharpens the edge tilt: 1.0
    spreads capital broadly across positive-edge ideas; higher values
    concentrate into the strongest. It never changes gross deployment — only
    the shape.
    """
    conc = float(np.clip(concentration, 0.5, 6.0))
    raw: dict[str, float] = {}
    inv_vol_fallback: dict[str, float] = {}
    for t in trades:
        name = t.get("name", "")
        m = metrics.get(name)
        if not t.get("is_eligible", False) or m is None:
            continue
        vol = m.get("vol", float("nan"))
        inv_vol = (1.0 / vol) if (vol and np.isfinite(vol) and vol > 1e-9) else 0.0
        inv_vol_fallback[name] = inv_vol
        mu   = float(m.get("avg_return", 0.0))        # realised mean P&L / window
        sd   = max(float(m.get("ret_std", 0.0)), 1e-6)
        dsr  = max(m.get("dsr", 0.0), 0.0)            # deflated-Sharpe prob
        liq  = max(m.get("liquidity", 0.0), 0.0)
        # Risk-adjusted expected edge (direction-aware realised Sharpe), shrunk
        # by deflated-Sharpe confidence. max(...,0) starves historically
        # losing signals so capital flows to the winners, not to cash.
        edge = max(mu / sd, 0.0) * (0.35 + 0.65 * dsr)
        raw[name] = (edge ** conc) * (0.40 + 0.60 * liq)

    total = sum(raw.values())
    if total <= 1e-12:
        # No positive-edge trade in this regime — keep the sleeve fully invested
        # by spreading over eligible names inverse to volatility (never cash).
        raw = dict(inv_vol_fallback)
    total = sum(raw.values())
    weights = {n: (w / total if total > 1e-12 else 0.0) for n, w in raw.items()}

    # Single-trade cap with pro-rata redistribution of the excess
    for _ in range(10):
        over = {n: w for n, w in weights.items() if w > cap + 1e-9}
        if not over:
            break
        excess = sum(w - cap for w in over.values())
        for n in over:
            weights[n] = cap
        under = {n: w for n, w in weights.items() if w < cap - 1e-9}
        u_tot = sum(under.values())
        if u_tot <= 1e-12:
            break
        for n, w in under.items():
            weights[n] = w + excess * (w / u_tot)

    for t in trades:
        name = t.get("name", "")
        proposed = weights.get(name, 0.0)
        final = enforce_weight(t, proposed)          # THE choke point
        t["alloc_weight"] = round(final, 6)
        m = metrics.get(name, {})
        t["alloc_detail"] = {
            "conviction": round(m.get("conviction", 0.0), 4),
            "dsr": round(m.get("dsr", 0.0), 4),
            "dsr_factor": round(m.get("dsr_factor", 0.0), 4),
            "vol": (round(m["vol"], 4) if m.get("vol") and np.isfinite(m["vol"])
                    else None),
            "liquidity": round(m.get("liquidity", 0.0), 3),
            "sharpe_raw": round(m.get("sharpe_raw", 0.0), 3),
            "n_trades": m.get("n_trades", 0),
            # backtest projection inputs (per holding window, %)
            "avg_return": round(m.get("avg_return", 0.0), 4),
            "ret_std": round(m.get("ret_std", 0.0), 4),
            "holding_days": m.get("holding_days", 0),
            "locked": not t.get("is_eligible", False),
        }
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 of 4 — PORTFOLIO CONSTRAINTS on top of Step-2 weights.
#
# Two group caps, applied over alloc_weight and re-normalized to the original
# gross without ever letting a group re-breach (iterative water-fill; the
# residual goes to cash when every group binds — same semantics as Step 2's
# single-trade cap):
#   • conflict cap  — trades grouped by conflict_id; no event may carry more
#     than CONFLICT_CAP of target gross. Untagged trades form an
#     "unattributed" macro bucket EXEMPT from this cap (capping "no event" as
#     if it were one event would be fake rigor) but fully cluster-capped.
#   • cluster cap   — groups from backtest.correlation_clusters(), the SAME
#     union-find compute_effective_n uses (r > 0.90 on signed leg-series
#     returns): duplicated bets share one budget.
# A runtime assertion — not just a test — fails loudly if any group exceeds
# its cap after re-normalization.
# ─────────────────────────────────────────────────────────────────────────────

CONFLICT_CAP = 0.40
CLUSTER_CAP = 0.45
_UNATTRIBUTED = "__unattributed__"


def build_correlation_clusters(
    trades: list[dict],
    all_r: pd.DataFrame,
    corr_threshold: float = 0.90,
) -> dict[str, int]:
    """Cluster id per eligible trade via the shared union-find."""
    from src.analysis.backtest import correlation_clusters
    series = {
        t["name"]: trade_leg_series(all_r, t.get("assets") or [],
                                    t.get("direction") or [])
        for t in trades if t.get("is_eligible", False)
    }
    return correlation_clusters(series, corr_threshold=corr_threshold)


def apply_portfolio_constraints(
    trades: list[dict],
    all_r: pd.DataFrame,
    conflict_cap: float = CONFLICT_CAP,
    cluster_cap: float = CLUSTER_CAP,
    single_cap: float = MAX_SINGLE_WEIGHT,
    corr_threshold: float = 0.90,
    clusters: dict[str, int] | None = None,
) -> list[dict]:
    """
    Re-stamps alloc_weight subject to single/conflict/cluster caps and
    stamps constraint_detail per trade. Caps are fractions of the TARGET
    gross (Step-2's invested total), which keeps the constraint set linear;
    unfillable weight becomes cash.
    """
    elig = [t for t in trades if t.get("is_eligible", False)]
    if clusters is None:
        clusters = build_correlation_clusters(trades, all_r, corr_threshold)

    w = {t["name"]: float(t.get("alloc_weight", 0.0)) for t in elig}
    target_gross = sum(w.values())

    conflict_of = {
        t["name"]: (t.get("conflict_id") or _UNATTRIBUTED) for t in elig
    }
    cluster_of = {t["name"]: clusters.get(t["name"], hash(t["name"]) % 10 ** 9)
                  for t in elig}

    def _groups(mapping):
        g: dict = {}
        for n_ in w:
            g.setdefault(mapping[n_], []).append(n_)
        return g

    conflict_groups = {k: v for k, v in _groups(conflict_of).items()
                       if k != _UNATTRIBUTED}
    # Singleton clusters are exempt: the cluster cap exists to stop DUPLICATE
    # bets double-counting — a lone bet is concentration, which the single
    # and conflict caps already govern.
    cluster_groups = {k: v for k, v in _groups(cluster_of).items() if len(v) > 1}

    conflict_lim = conflict_cap * target_gross
    cluster_lim = cluster_cap * target_gross

    clipped: set[str] = set()
    if target_gross > 1e-12:
        for _ in range(60):
            changed = False
            for n_ in w:                                        # single cap
                if w[n_] > single_cap + 1e-12:
                    w[n_] = single_cap
                    clipped.add(n_)
                    changed = True
            for members in conflict_groups.values():            # conflict cap
                gs = sum(w[m] for m in members)
                if gs > conflict_lim + 1e-12:
                    f = conflict_lim / gs
                    for m in members:
                        w[m] *= f
                        clipped.add(m)
                    changed = True
            for members in cluster_groups.values():             # cluster cap
                gs = sum(w[m] for m in members)
                if gs > cluster_lim + 1e-12:
                    f = cluster_lim / gs
                    for m in members:
                        w[m] *= f
                        clipped.add(m)
                    changed = True

            deficit = target_gross - sum(w.values())            # re-normalize
            if deficit > 1e-9:
                free = [n_ for n_ in w if n_ not in clipped and w[n_] > 1e-12]
                f_tot = sum(w[n_] for n_ in free)
                if f_tot > 1e-12:
                    for n_ in free:
                        w[n_] += deficit * (w[n_] / f_tot)
                    changed = True
            if not changed:
                break

        # FORBIDDEN clause, enforced at runtime — not only in tests
        assert all(w[n_] <= single_cap + 1e-6 for n_ in w), "single cap breached"
        for members in conflict_groups.values():
            assert sum(w[m] for m in members) <= conflict_lim + 1e-6, \
                "conflict cap breached after re-normalization"
        for members in cluster_groups.values():
            assert sum(w[m] for m in members) <= cluster_lim + 1e-6, \
                "cluster cap breached after re-normalization"

    for t in trades:
        n_ = t.get("name", "")
        final = enforce_weight(t, w.get(n_, 0.0))               # THE choke point
        t["alloc_weight"] = round(final, 6)
        t["constraint_detail"] = {
            "conflict_group": conflict_of.get(n_, None),
            "cluster_id": cluster_of.get(n_, None),
            "clipped": n_ in clipped,
            "conflict_cap": conflict_cap,
            "cluster_cap": cluster_cap,
        }
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 of 4 — RANKING. Attractiveness, never raw P&L:
#
#     attractiveness = 0.50·DSR + 0.35·conviction + 0.15·constraint_room
#
# DSR is the deflated Sharpe probability (Step 2); sharpe_raw exists in the
# stamps for transparency and MUST NOT enter this score. constraint_room is
# the trade's minimum remaining headroom across its single / conflict /
# cluster caps at final weights — trades with room to grow outrank cap-pinned
# ones at equal evidence. With zero gross (the honest all-cash book) room is
# 1 everywhere and ranking reduces to evidence strength.
# ─────────────────────────────────────────────────────────────────────────────

ATTR_W_DSR = 0.50
ATTR_W_CONVICTION = 0.35
ATTR_W_ROOM = 0.15


def rank_trades(
    trades: list[dict],
    conflict_cap: float = CONFLICT_CAP,
    cluster_cap: float = CLUSTER_CAP,
    single_cap: float = MAX_SINGLE_WEIGHT,
) -> list[dict]:
    """
    Stamp eligible trades with rank / attractiveness / attr_detail and return
    them sorted best-first. Ineligible trades are excluded and any stale rank
    field is stripped from them.
    """
    elig = [t for t in trades if t.get("is_eligible", False)]
    G = sum(float(t.get("alloc_weight", 0.0)) for t in elig)

    conf_sums: dict = {}
    clus_sums: dict = {}
    clus_members: dict = {}
    for t in elig:
        cd = t.get("constraint_detail") or {}
        w = float(t.get("alloc_weight", 0.0))
        cg = cd.get("conflict_group")
        if cg and cg != _UNATTRIBUTED:
            conf_sums[cg] = conf_sums.get(cg, 0.0) + w
        cl = cd.get("cluster_id")
        if cl is not None:
            clus_sums[cl] = clus_sums.get(cl, 0.0) + w
            clus_members[cl] = clus_members.get(cl, 0) + 1

    for t in elig:
        d = t.get("alloc_detail") or {}
        dsr = float(d.get("dsr", 0.0))
        conv = float(d.get("conviction", 0.0))
        w = float(t.get("alloc_weight", 0.0))
        cd = t.get("constraint_detail") or {}

        rooms = [(single_cap - w) / single_cap]
        if G > 1e-9:
            cg = cd.get("conflict_group")
            if cg and cg != _UNATTRIBUTED:
                rooms.append((conflict_cap * G - conf_sums.get(cg, 0.0))
                             / (conflict_cap * G))
            cl = cd.get("cluster_id")
            if cl is not None and clus_members.get(cl, 0) > 1:
                rooms.append((cluster_cap * G - clus_sums.get(cl, 0.0))
                             / (cluster_cap * G))
        room = float(np.clip(min(rooms), 0.0, 1.0))

        attr = ATTR_W_DSR * dsr + ATTR_W_CONVICTION * conv + ATTR_W_ROOM * room
        t["attractiveness"] = round(attr, 4)
        t["attr_detail"] = {"dsr": round(dsr, 4), "conviction": round(conv, 4),
                            "room": round(room, 3)}

    ranked = sorted(elig, key=lambda x: -x.get("attractiveness", 0.0))
    for i, t in enumerate(ranked, 1):
        t["rank"] = i
    for t in trades:
        if not t.get("is_eligible", False):
            t.pop("rank", None)
            t.pop("attractiveness", None)
    return ranked


def desk_report_feed(ranked: list[dict]) -> list[dict]:
    """
    Adapt the ranked book to the EXISTING generate_report() trade-card schema
    — the adaptation lives in the fed data, never in the generator. Rank and
    weight ride in `name`; attractiveness components ride in `trigger`; the
    invalidation condition is prepended to `risk`.

    The feed mirrors the screen's cash state: a zero-gross book prepends a
    synthetic lead card — "DESK CALL — 0% DEPLOYED · 100% CASH" plus the
    one-line reason — and labels every trade card WATCHLIST / NOT YET
    ALLOCATABLE. A deployed book feeds exactly as before.
    """
    gross = sum(float(t.get("alloc_weight", 0.0)) for t in ranked)
    cash_book = bool(ranked) and gross <= 1e-9

    feed = []
    if cash_book:
        best_dsr = max(float((t.get("alloc_detail") or {}).get("dsr", 0.0))
                       for t in ranked)
        # effective bar in force (appetite-adjusted); falls back to the strict
        # reference bar if unstamped
        bar = min((float((t.get("alloc_detail") or {}).get("deploy_bar",
                          DSR_DEPLOY_BAR)) for t in ranked),
                  default=DSR_DEPLOY_BAR)
        _appetite_note = ("" if bar >= DSR_DEPLOY_BAR - 1e-9 else
                          f" even at your lowered {bar:.2f} risk bar")
        feed.append({
            "name": "DESK CALL — 0% DEPLOYED · 100% CASH",
            "category": "Desk Call",
            "trigger": "CASH IS THE POSITION — HELD, NOT DEFAULTED",
            "rationale": (
                f"All {len(ranked)} eligible theses sit below the "
                f"{bar:.2f} deploy bar{_appetite_note} (best {best_dsr:.2f}) "
                "— no thesis clears it, so nothing earns weight. The ranked "
                "ideas that follow are a watchlist, not an allocation."
            ),
            "regime": [], "assets": [], "direction": [],
            "entry": "—", "exit": "—",
            "risk": (f"INVALIDATED IF: any eligible thesis clears the "
                     f"{bar:.2f} deploy bar — the book redeploys on reload"),
        })

    for t in ranked:
        c = dict(t)
        ad = t.get("attr_detail") or {}
        inval = (t.get("invalidation") or t.get("invalidation_condition")
                 or t.get("exit") or t.get("risk") or "—")
        c["name"] = ((f"WATCHLIST #{t.get('rank', '?')} — NOT YET ALLOCATABLE"
                      if cash_book else f"#{t.get('rank', '?')}") +
                     f" · {t.get('alloc_weight', 0.0) * 100:.1f}% wt · "
                     f"{t.get('name', '')}")
        c["trigger"] = (f"ATTR {t.get('attractiveness', 0.0):.2f} · "
                        f"DSR {ad.get('dsr', 0.0):.2f} · "
                        f"CONV {ad.get('conviction', 0.0):.2f} · "
                        f"ROOM {ad.get('room', 0.0):.2f} — "
                        f"{t.get('trigger', '')}")
        c["risk"] = f"INVALIDATED IF: {inval}"
        feed.append(c)
    return feed

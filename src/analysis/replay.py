"""
Replay Mode - strict point-in-time (PIT) reconstruction of the terminal's call.

Points the terminal at a past date and recomputes what it WOULD have said using
ONLY data available up to that timestamp, then fast-forwards to compare against
the actual outcome. Each tracked conflict breakout becomes a worked case study.

════════════════════════════════════════════════════════════════════════════
THE ONE RULE - CUTOFF ENFORCEMENT
════════════════════════════════════════════════════════════════════════════
No data after the replay cutoff may touch the terminal-call computation. Ever.
A single leaked future point makes the terminal look brilliant and it's a lie.

Enforcement is three-layered:

1. `pit_slice()` / `pit_assert()` - THE single choke point. Every DataFrame or
   Series entering the computation passes through `pit_slice`, which trims to
   index ≤ cutoff and then re-asserts. `pit_assert` raises `LookaheadError`
   naming the offending source and timestamp - loud, never silent. The fetch
   layer is asked for end=cutoff but NEVER trusted: yfinance end-exclusivity,
   the loader's disk cache, and tz-aware indices are all leak vectors, so we
   slice-and-assert regardless of what the loader claims it returned.

2. Structural separation - `terminal_call()` computes and returns a frozen
   dict; `actual_outcome()` / `trade_leg_outcomes()` are SEPARATE functions
   that load post-cutoff data. Their outputs render next to the call but can
   never flow into it: no shared frames, no shared namespace.

3. `tests/test_replay_pit.py` - injects a leaked future row and asserts
   `LookaheadError` fires.

What honestly CANNOT be replayed (shown in the UI, never silently substituted
with today's values): GDELT, ACLED, PortWatch, EIA, COT, RSS - no historical
API snapshots exist. The replay GRS is therefore the market-confirmation proxy
layer (`risk_score_history`), whose own docstring states CIS/news are not
available historically. Additionally, the CONFLICTS registry's transmission
weights were calibrated TODAY with knowledge of how these conflicts played
out - that structural hindsight is disclosed on-screen, not hidden.
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd

from src.data.config import CONFLICTS

# ═══════════════════════════════════════════════════════════════════════════
# Layer 1 - the choke point
# ═══════════════════════════════════════════════════════════════════════════

class LookaheadError(RuntimeError):
    """Raised when any data point after the replay cutoff reaches a PIT computation."""


def _naive_ts(cutoff) -> pd.Timestamp:
    """Cutoff as a tz-naive end-of-day Timestamp (dates compare inclusively)."""
    ts = pd.Timestamp(cutoff)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    # date-level cutoff means "through the close of that day"
    return ts.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)


def _naive_index(obj: pd.DataFrame | pd.Series) -> pd.Index:
    idx = obj.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def pit_assert(obj: pd.DataFrame | pd.Series, cutoff, source: str) -> None:
    """
    Fail LOUD if `obj` contains any observation after `cutoff`.
    This is the assertion every replay computation runs on entry.
    """
    if obj is None or len(obj) == 0:
        return
    idx = _naive_index(obj)
    if not isinstance(idx, pd.DatetimeIndex):
        raise LookaheadError(
            f"PIT VIOLATION [{source}]: non-datetime index - cannot verify cutoff."
        )
    mx = idx.max()
    limit = _naive_ts(cutoff)
    if mx > limit:
        n_bad = int((idx > limit).sum())
        raise LookaheadError(
            f"PIT VIOLATION [{source}]: {n_bad} observation(s) after cutoff "
            f"{limit.date()} - latest is {mx}. Future data must never reach "
            f"the replay computation."
        )


def pit_slice(
    obj: pd.DataFrame | pd.Series,
    cutoff,
    source: str,
) -> pd.DataFrame | pd.Series:
    """
    THE single choke point. Trim to index ≤ cutoff, then re-assert.
    All data entering `terminal_call` passes through here - the fetch layer
    is never trusted to have honoured its end date.
    """
    if obj is None or len(obj) == 0:
        return obj
    idx = _naive_index(obj)
    if not isinstance(idx, pd.DatetimeIndex):
        raise LookaheadError(
            f"PIT VIOLATION [{source}]: non-datetime index - cannot slice to cutoff."
        )
    out = obj.loc[idx <= _naive_ts(cutoff)]
    pit_assert(out, cutoff, source)   # belt and braces: slicing bug = loud failure
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Replay presets - the tracked conflict breakouts
# ═══════════════════════════════════════════════════════════════════════════

def replay_presets() -> list[dict]:
    """Breakout dates of the tracked conflicts, oldest first."""
    out = []
    for c in CONFLICTS:
        start = c.get("start")
        if isinstance(start, datetime.date):
            out.append({
                "id": c["id"], "label": c["label"], "name": c["name"],
                "date": start, "region": c.get("region", ""),
            })
    return sorted(out, key=lambda x: x["date"])


# ═══════════════════════════════════════════════════════════════════════════
# Static (no-live-API) conflict scoring as-of the cutoff
# ═══════════════════════════════════════════════════════════════════════════

# Stated replay rules - not leaked data:
#   * Conflicts whose start date is AFTER the cutoff are EXCLUDED entirely
#     (replaying Ukraine 2022 must not know Iran/Hormuz 2025 exists).
#   * recency is computed against the CUTOFF, not today.
#   * Within 90 days of onset a conflict is treated as "active" and within
#     30 days its escalation_trend is "escalating" - the breakout IS the
#     observable escalation on that day. Otherwise today's registry statics
#     apply, with the hindsight caveat disclosed in the UI.
_ONSET_ACTIVE_DAYS = 90
_ONSET_ESCALATING_DAYS = 30


def _recency_at(conflict: dict, cutoff: datetime.date, state: str) -> float:
    if state == "latent":
        return 0.35
    if state == "frozen":
        return 0.15
    start = conflict.get("start", cutoff)
    days = max((cutoff - start).days, 1)
    return max(0.30, math.exp(-days / 303))


def replay_conflict_scores(cutoff: datetime.date) -> dict[str, dict]:
    """
    CIS/TPS per conflict as-of cutoff - ZERO live API calls (GDELT/ACLED have
    no historical snapshots; attempting them would return TODAY's state, which
    is lookahead). Mirrors compute_cis()'s static path with cutoff-aware
    recency/state/escalation per the stated rules above.
    """
    from src.analysis.conflict_model import (
        _CIS_WEIGHTS, _ESCALATION_MAP, _STATE_MULT, compute_tps,
    )

    results: dict[str, dict] = {}
    for c in CONFLICTS:
        start = c.get("start")
        if not isinstance(start, datetime.date) or start > cutoff:
            continue    # did not exist yet - excluding it IS the PIT rule

        days_since = (cutoff - start).days
        if days_since <= _ONSET_ACTIVE_DAYS:
            state = "active"
        else:
            state = c.get("state", "active")

        if days_since <= _ONSET_ESCALATING_DAYS:
            escalation = "escalating"
        else:
            escalation = c.get("escalation_trend", "stable")

        dims = {
            "deadliness":           float(c.get("deadliness", 0.5)),
            "civilian_danger":      float(c.get("civilian_danger", 0.5)),
            "geographic_diffusion": float(c.get("geographic_diffusion", 0.3)),
            "fragmentation":        float(c.get("fragmentation", 0.2)),
            "escalation_trend":     _ESCALATION_MAP.get(escalation, 0.5),
            "recency":              _recency_at(c, cutoff, state),
            "source_coverage":      float(c.get("source_coverage", 0.7)),
        }
        state_mult = _STATE_MULT.get(state, 1.0)
        cis = float(np.clip(
            sum(_CIS_WEIGHTS[k] * dims[k] for k in _CIS_WEIGHTS) * state_mult * 100,
            0, 100,
        ))
        # compute_tps reads structural transmission channels only - no live I/O.
        # Recompute with the as-of state multiplier.
        c_asof = {**c, "state": state}
        tps = compute_tps(c_asof)

        results[c["id"]] = {
            "id": c["id"], "name": c["name"], "label": c["label"],
            "region": c.get("region", ""), "color": c.get("color", "#CFB991"),
            "state": state, "cis": round(cis, 1), "cis_source": "static-replay",
            "tps": round(tps, 1), "confidence": 0.5,
            "trend": {"escalating": "rising", "stable": "stable",
                      "de-escalating": "falling"}[escalation],
            "escalation": escalation, "market_freshness": 1.0,
            "days_since_onset": days_since,
            "transmission": c.get("transmission", {}),
            "affected_commodities": c.get("affected_commodities", []),
            "affected_equities": c.get("affected_equities", []),
            "hedge_assets": c.get("hedge_assets", []),
        }
    return results


def replay_asset_scores(conflict_results: dict[str, dict]) -> dict[str, dict]:
    """
    Structural exposure scores fed by the REPLAY conflict scores (base scenario,
    geo_mult = 1.0). Mirrors exposure.score_all_assets() aggregation but with
    injected conflict results and no session/scenario/live dependencies.
    """
    from src.analysis.conflict_model import aggregate_portfolio_scores
    from src.analysis.exposure import (
        _structural_exposure_score, _transmission_adjusted_exposure,
        _conflict_beta, _hedge_score, _top_conflict,
        _SAFE_HAVEN_ASSETS, _GEO_RISK_BENEFICIARIES,
    )
    from src.data.config import SECURITY_EXPOSURE

    agg = aggregate_portfolio_scores(conflict_results)
    weights = agg.get("conflict_weights", {})
    tps_map = {cid: r["tps"] for cid, r in conflict_results.items()}

    results: dict[str, dict] = {}
    for asset in SECURITY_EXPOSURE:
        ses = _structural_exposure_score(asset, weights)
        tae = _transmission_adjusted_exposure(asset, weights, tps_map)
        sas = float(np.clip(round(tae * 100, 1), 0.0, 100.0))
        s_data = SECURITY_EXPOSURE.get(asset, {})
        if asset in _SAFE_HAVEN_ASSETS:
            direction = "safe_haven"
        elif asset in _GEO_RISK_BENEFICIARIES:
            direction = "long_geo_risk"
        else:
            direction = "neutral"
        results[asset] = {
            "asset": asset,
            "ses": round(ses, 3), "tae": round(tae, 3),
            "sas": sas, "sas_raw": sas, "sas_capped": False,
            "beta": _conflict_beta(asset, tps_map),
            "hedge_score": round(_hedge_score(asset, weights), 1),
            "top_conflict": _top_conflict(asset, tps_map),
            "sector_tags": s_data.get("sector_tags", []),
            "route_tags": s_data.get("route_tags", []),
            "sanction_tags": s_data.get("sanction_tags", []),
            "direction": direction, "scenario_mult": 1.0,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2a - the frozen terminal call (PIT side)
# ═══════════════════════════════════════════════════════════════════════════

# Transmission channel pairs surfaced in the call
_CHANNEL_PAIRS = [
    ("WTI Crude Oil", "S&P 500"), ("WTI Crude Oil", "DAX"),
    ("Gold", "S&P 500"), ("Natural Gas", "DAX"),
    ("Gold", "DAX"), ("WTI Crude Oil", "Eurostoxx 50"),
]
_DY_ASSETS = ["S&P 500", "DAX", "Eurostoxx 50", "Nikkei 225",
              "WTI Crude Oil", "Brent Crude", "Gold", "Natural Gas", "Copper"]
_REGIME_NAMES = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}


def terminal_call(
    eq_r: pd.DataFrame,
    cmd_r: pd.DataFrame,
    cutoff: datetime.date,
    focus_conflict: str | None = None,
) -> dict:
    """
    Everything the terminal would have said at end-of-day `cutoff`, computed
    from return frames that are pit-sliced ON ENTRY - the caller's fetch
    window is never trusted. Returns a frozen dict; nothing here may receive
    post-cutoff data (enforced by pit_slice/pit_assert on every input).
    """
    # ── THE CHOKE POINT: slice + assert every input before any computation ──
    eq_r  = pit_slice(eq_r,  cutoff, "equity_returns")
    cmd_r = pit_slice(cmd_r, cutoff, "commodity_returns")

    from src.analysis.correlations import (
        average_cross_corr_series, detect_correlation_regime,
    )
    from src.analysis.risk_score import risk_score_history
    from src.analysis.spillover import diebold_yilmaz

    out: dict = {"cutoff": str(cutoff), "pit": {}}

    # Record the newest observation per source - displayed as proof of cutoff
    for name, df in (("equity_returns", eq_r), ("commodity_returns", cmd_r)):
        pit_assert(df, cutoff, name)          # re-assert before use
        out["pit"][name] = str(_naive_index(df).max().date()) if len(df) else "empty"

    # ── GRS (market-confirmation proxy layer - the replayable GRS) ─────────
    grs_series = risk_score_history(pd.Series(dtype=float), cmd_r, eq_r)
    grs_series = pit_slice(grs_series, cutoff, "grs_history")
    out["grs"] = float(grs_series.iloc[-1]) if len(grs_series) else float("nan")
    out["grs_series"] = grs_series.tail(252)

    # ── Correlation regime ──────────────────────────────────────────────────
    avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
    avg_corr = pit_slice(avg_corr, cutoff, "avg_cross_corr")
    regimes = detect_correlation_regime(avg_corr)
    regimes = pit_slice(regimes, cutoff, "regime_series")
    regime = int(regimes.iloc[-1]) if len(regimes) else 1
    out["regime"] = regime
    out["regime_name"] = _REGIME_NAMES[regime]
    out["avg_corr"] = float(avg_corr.iloc[-1]) if len(avg_corr) else float("nan")

    # ── Transmission channels: 60d rolling corr + 252d OLS beta ────────────
    combined = pd.concat([eq_r, cmd_r], axis=1)
    combined = pit_slice(combined, cutoff, "combined_returns")
    channels = []
    for cmd, eq in _CHANNEL_PAIRS:
        if cmd not in combined.columns or eq not in combined.columns:
            continue
        pair = combined[[cmd, eq]].dropna().tail(252)
        if len(pair) < 60:
            continue
        corr60 = float(pair[cmd].tail(60).corr(pair[eq].tail(60)))
        var_x = float(pair[cmd].var())
        beta = float(pair[cmd].cov(pair[eq]) / var_x) if var_x > 1e-12 else 0.0
        channels.append({
            "commodity": cmd, "equity": eq,
            "corr_60d": round(corr60, 3), "beta_252d": round(beta, 3),
        })
    out["channels"] = sorted(channels, key=lambda c: abs(c["corr_60d"]), reverse=True)

    # ── Diebold-Yilmaz on the trailing 252d window ──────────────────────────
    try:
        dy_cols = [a for a in _DY_ASSETS if a in combined.columns]
        dy_win = combined[dy_cols].dropna().tail(252)
        pit_assert(dy_win, cutoff, "dy_window")
        dy = diebold_yilmaz(dy_win, lag_order=4, horizon=10)
        net = dy.get("net_spillover")
        out["dy_total"] = round(float(dy.get("total_spillover", float("nan"))), 1)
        out["dy_top_transmitter"] = str(dy.get("top_transmitter", ""))
        out["dy_net"] = ({k: round(float(v), 1) for k, v in net.items()}
                         if net is not None else {})
    except Exception as e:
        out["dy_total"], out["dy_top_transmitter"], out["dy_net"] = float("nan"), "", {}
        out["dy_error"] = f"{type(e).__name__}: {e}"

    # ── Conflict scoring + trade theses (static, zero live calls) ──────────
    conflicts = replay_conflict_scores(cutoff)
    out["conflicts"] = conflicts
    out["focus_conflict"] = focus_conflict if focus_conflict in conflicts else None

    assets = replay_asset_scores(conflicts)
    from src.analysis.trade_generator import generate_conflict_trades
    trades = generate_conflict_trades(
        regime=regime,
        conflict_results=conflicts,
        all_assets=assets,
        scenario_id="base",
    )
    if focus_conflict:
        focused = [t for t in trades if t.get("conflict_id") == focus_conflict]
        trades = focused or trades
    out["trades"] = trades[:6]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2b - the actual outcome (post-cutoff side, structurally separate)
# ═══════════════════════════════════════════════════════════════════════════
# These functions load data AFTER the cutoff on purpose. Their outputs are
# rendered NEXT TO the frozen call and must never be passed back into
# terminal_call or anything it consumes.

OUTCOME_HORIZONS = (5, 30, 60)          # trading days
_FREIGHT_TICKER = "BDRY"                 # Breakwave dry-bulk freight ETF (2018+)
_REGIONAL_INDEX = {
    "ukraine_russia": "DAX", "red_sea_houthi": "Eurostoxx 50",
    "israel_gaza": "Eurostoxx 50", "iran_conflict": "Eurostoxx 50",
    "india_pakistan": "Sensex", "taiwan_strait": "Hang Seng",
}


def _fwd_cum(returns: pd.Series, cutoff, horizons) -> dict[int, float]:
    """Cumulative return over the first h trading days strictly after cutoff."""
    idx = _naive_index(returns)
    fwd = returns.loc[idx > _naive_ts(cutoff)].dropna()
    return {
        h: float(np.expm1(np.log1p(fwd.iloc[:h]).sum())) if len(fwd) >= min(h, 3) else float("nan")
        for h in horizons
    }


def actual_outcome(
    cutoff: datetime.date,
    conflict_id: str | None = None,
    horizons: tuple[int, ...] = OUTCOME_HORIZONS,
) -> dict:
    """
    What actually happened after the cutoff: crude / gold / freight / equities.
    Deliberately loads post-cutoff data - never feed this into terminal_call.
    """
    from src.data.loader import load_returns

    start = str(cutoff - datetime.timedelta(days=10))
    end = str(cutoff + datetime.timedelta(days=max(horizons) * 2 + 30))
    eq_r, cmd_r = load_returns(start, end)
    combined = pd.concat([eq_r, cmd_r], axis=1)

    assets = ["WTI Crude Oil", "Gold", "S&P 500"]
    regional = _REGIONAL_INDEX.get(conflict_id or "", "")
    if regional and regional not in assets:
        assets.append(regional)

    out: dict = {"horizons": list(horizons), "assets": {}, "paths": {}}
    for a in assets:
        if a not in combined.columns:
            continue
        out["assets"][a] = _fwd_cum(combined[a], cutoff, horizons)
        idx = _naive_index(combined[a])
        fwd = combined[a].loc[idx > _naive_ts(cutoff)].dropna().iloc[:max(horizons)]
        out["paths"][a] = (np.exp(np.log1p(fwd).cumsum()) - 1.0) * 100

    # Freight proxy - BDRY only trades from 2018; marked unavailable otherwise
    try:
        import yfinance as yf
        px = yf.download(_FREIGHT_TICKER, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        r = px.pct_change().dropna()
        if len(r) > 5:
            out["assets"]["Freight (BDRY)"] = _fwd_cum(r, cutoff, horizons)
            idx = _naive_index(r)
            fwd = r.loc[idx > _naive_ts(cutoff)].dropna().iloc[:max(horizons)]
            out["paths"]["Freight (BDRY)"] = (np.exp(np.log1p(fwd).cumsum()) - 1.0) * 100
    except Exception:
        pass
    return out


def trade_leg_outcomes(
    trades: list[dict],
    cutoff: datetime.date,
    horizon: int = 30,
) -> list[dict]:
    """
    Forward P&L of each generated trade thesis: equal-weight legs, Long legs
    positive / Short legs inverted, over `horizon` trading days after cutoff.
    Post-cutoff by design - outcome side only.
    """
    import yfinance as yf

    tickers = sorted({tk for t in trades for tk in (t.get("tickers") or {}).values() if tk})
    if not tickers:
        return []
    start = str(cutoff - datetime.timedelta(days=10))
    end = str(cutoff + datetime.timedelta(days=horizon * 2 + 30))
    try:
        px = yf.download(tickers, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
        if isinstance(px, pd.Series):
            px = px.to_frame(tickers[0])
    except Exception:
        return []

    results = []
    for t in trades:
        tk_map = t.get("tickers") or {}
        legs, leg_rows = [], []
        for asset, direction in zip(t.get("assets", []), t.get("direction", [])):
            tk = tk_map.get(asset, "")
            if tk not in px.columns:
                leg_rows.append({"asset": asset, "ticker": tk,
                                 "direction": direction, "ret": float("nan")})
                continue
            r = px[tk].pct_change().dropna()
            cum = _fwd_cum(r, cutoff, (horizon,))[horizon]
            signed = cum if direction.lower() == "long" else -cum
            legs.append(signed)
            leg_rows.append({"asset": asset, "ticker": tk,
                             "direction": direction, "ret": cum})
        pnl = float(np.nanmean(legs)) if legs else float("nan")
        results.append({
            "name": t.get("name", ""), "conflict_id": t.get("conflict_id", ""),
            "confidence": t.get("confidence", 0.0),
            "legs": leg_rows, "pnl": pnl, "horizon": horizon,
            "verdict": (" - " if np.isnan(pnl)
                        else "PAID" if pnl > 0.01
                        else "FLAT" if pnl > -0.01 else "LOST"),
        })
    return results

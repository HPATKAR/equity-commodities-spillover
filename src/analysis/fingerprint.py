"""
Pattern Memory — "the market has seen this before" engine.

Compresses each historical day into a fingerprint of the terminal's FULL state
— geopolitical configuration + chokepoint pressure + spillover structure +
correlation regime + market state — then finds which past periods most
resemble RIGHT NOW across the whole configuration, not one variable.
This is pattern-memory over the terminal's entire state space, not a price
correlation: a day only matches if the geopolitical shape AND the market
shape both rhyme.

════════════════════════════════════════════════════════════════════════════
FINGERPRINT BLOCKS (block weights perturbable in the UI; features z-scored,
per-feature weight = block_weight / n_features so no block dominates by count)
════════════════════════════════════════════════════════════════════════════
GEO        — from GEOPOLITICAL_EVENTS (13 events, 2008–present): active-event
             count, recency-decayed intensity (same exp(-days/303) decay CIS
             uses), days since latest onset, crisis-type composition.
CHOKEPOINT — daily chokepoint+shipping pressure: CONFLICTS registry
             transmission weights for conflicts active that day (2022+),
             stated per-event mapping for earlier events. Structural proxy —
             PortWatch has no history. Disclosed, never presented as live.
SPILLOVER  — rolling Diebold-Yilmaz total spillover (optional, precomputed),
             oil→S&P 60d rolling beta, avg cross-asset correlation level,
             correlation velocity (10d).
REGIME     — correlation regime id + days spent in the current regime.
MARKET     — GRS proxy (risk_score_history), 20d realized vols, 20d momentum
             for WTI / Gold / S&P.

CAUSALITY: every raw feature at day t is computed from data ≤ t (rolling /
EWM / registry state as-of t). Feature STANDARDIZATION and the regime
percentile thresholds are full-sample — standard in-sample normalization for
analog recall, disclosed on-screen. tests/test_fingerprint.py asserts the raw
causality (a day before an event's onset carries zero contribution from it;
market features at t are unchanged by truncating the future).

HONESTY: GEO and CHOKEPOINT are structural reconstructions from a registry
calibrated recently — same hindsight caveat as Replay Mode. SPILLOVER /
REGIME / MARKET are genuinely historical. Match verdicts are always shown
against the unconditional base rate: "5/7 matches went bad" means nothing
without knowing bad's base rate is 22%.
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd

from src.data.config import GEOPOLITICAL_EVENTS, CONFLICTS

# ── Stated structural mappings ───────────────────────────────────────────────

# Crisis-type composition buckets (one-hot-ish shares of active events)
_CATEGORY_BUCKET = {
    "War": "geo", "Conflict": "geo", "Geopolitical": "geo",
    "Financial": "financial", "Monetary": "financial",
    "Commodity": "commodity", "Pandemic": "pandemic", "Trade": "trade",
}

# Chokepoint pressure for pre-registry events (CONFLICTS cover 2022+).
# Stated mapping — Arab Spring priced a Suez risk premium, Aramco threatened
# Hormuz throughput. Everything else: no maritime chokepoint channel.
_EVENT_CHOKEPOINT = {
    "Arab Spring / Libya": 0.35,
    "Aramco Attack":       0.50,
}
_STATE_MULT = {"active": 1.0, "latent": 0.35, "frozen": 0.15}

_RECENCY_TAU = 303       # same decay constant as CIS recency

BLOCKS = ("geo", "chokepoint", "spillover", "regime", "market")
DEFAULT_BLOCK_WEIGHTS = {b: 1.0 for b in BLOCKS}

BLOCK_FEATURES = {
    "geo":        ["geo_n_active", "geo_intensity", "geo_days_since_onset",
                   "geo_share_war", "geo_share_financial", "geo_share_commodity"],
    "chokepoint": ["choke_pressure"],
    "spillover":  ["dy_total", "oil_spx_beta60", "avg_corr", "corr_velocity"],
    "regime":     ["regime", "days_in_regime"],
    "market":     ["grs", "eq_vol20", "cmd_vol20",
                   "wti_mom20", "gold_mom20", "spx_mom20"],
}

MATCH_K = 7
EXCLUSION_ROWS = 40          # ± trading days between selected matches
RECENT_BLACKOUT = 63         # trailing days excluded (today ≠ yesterday)
FWD_HORIZONS = (20, 60)      # trading days
_BAD_SPX, _GOOD_SPX = -0.05, 0.02


# ═══════════════════════════════════════════════════════════════════════════
# GEO + CHOKEPOINT blocks — day-by-day registry reconstruction
# ═══════════════════════════════════════════════════════════════════════════

def daily_geo_state(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Structural geopolitical state per day. Causal by construction: an event
    contributes ONLY between its start and end dates; days before onset carry
    zero trace of it.
    """
    events = [
        {"start": e["start"], "end": e.get("end"),
         "bucket": _CATEGORY_BUCKET.get(e.get("category", ""), "geo"),
         "choke": _EVENT_CHOKEPOINT.get(e.get("label", ""), 0.0)}
        for e in GEOPOLITICAL_EVENTS
        if isinstance(e.get("start"), datetime.date)
    ]
    conflicts = [
        {"start": c["start"],
         "choke": (float(c.get("transmission", {}).get("chokepoint", 0.0))
                   + float(c.get("transmission", {}).get("shipping", 0.0))) / 2.0
                  * _STATE_MULT.get(c.get("state", "active"), 1.0)}
        for c in CONFLICTS
        if isinstance(c.get("start"), datetime.date)
    ]

    rows = []
    for ts in index:
        d = ts.date()
        active = [e for e in events
                  if e["start"] <= d and (e["end"] is None or d <= e["end"])]
        n = len(active)
        intensity = sum(
            max(0.30, math.exp(-max((d - e["start"]).days, 1) / _RECENCY_TAU))
            for e in active
        )
        onsets = [e["start"] for e in events if e["start"] <= d]
        days_since = (d - max(onsets)).days if onsets else 3650
        buckets = [e["bucket"] for e in active]
        share = lambda b: (buckets.count(b) / n) if n else 0.0

        choke = sum(e["choke"] for e in active)
        choke += sum(
            c["choke"] * max(0.30, math.exp(-max((d - c["start"]).days, 1) / _RECENCY_TAU))
            for c in conflicts if c["start"] <= d
        )
        rows.append({
            "geo_n_active": float(n),
            "geo_intensity": intensity,
            "geo_days_since_onset": math.log1p(min(days_since, 1500)),
            "geo_share_war": share("geo"),
            "geo_share_financial": share("financial"),
            "geo_share_commodity": share("commodity") + share("pandemic") + share("trade"),
            "choke_pressure": choke,
        })
    return pd.DataFrame(rows, index=index)


# ═══════════════════════════════════════════════════════════════════════════
# Full fingerprint assembly
# ═══════════════════════════════════════════════════════════════════════════

def active_event_labels(d: datetime.date) -> list[str]:
    """Event labels active on a given date — for display next to each match."""
    out = []
    for e in GEOPOLITICAL_EVENTS:
        start, end = e.get("start"), e.get("end")
        if isinstance(start, datetime.date) and start <= d and (end is None or d <= end):
            out.append(e.get("label", "?"))
    return out


def build_fingerprints(
    eq_r: pd.DataFrame,
    cmd_r: pd.DataFrame,
    dy_series: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    One fingerprint row per trading day. All raw features at day t use data
    ≤ t only (rolling/EWM/registry-as-of-t).

    Returns (fingerprints, regime_series) on a shared index.
    """
    from src.analysis.correlations import (
        average_cross_corr_series, detect_correlation_regime,
    )
    from src.analysis.risk_score import risk_score_history

    combined = pd.concat([eq_r, cmd_r], axis=1)

    # ── SPILLOVER block ─────────────────────────────────────────────────────
    avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
    corr_vel = avg_corr.diff(10)
    oil_beta = pd.Series(dtype=float)
    if "WTI Crude Oil" in combined.columns and "S&P 500" in combined.columns:
        pair = combined[["WTI Crude Oil", "S&P 500"]].dropna()
        cov = pair["WTI Crude Oil"].rolling(60).cov(pair["S&P 500"])
        var = pair["WTI Crude Oil"].rolling(60).var()
        oil_beta = (cov / var.replace(0, np.nan)).rename("oil_spx_beta60")

    # ── REGIME block ────────────────────────────────────────────────────────
    regimes = detect_correlation_regime(avg_corr)
    changed = regimes.ne(regimes.shift())
    run_id = changed.cumsum()
    days_in_regime = regimes.groupby(run_id).cumcount().astype(float)

    # ── MARKET block ────────────────────────────────────────────────────────
    grs = risk_score_history(pd.Series(dtype=float), cmd_r, eq_r)
    eq_vol = eq_r.rolling(20, min_periods=10).std().mean(axis=1) * np.sqrt(252) * 100
    cmd_vol = cmd_r.rolling(20, min_periods=10).std().mean(axis=1) * np.sqrt(252) * 100

    def mom(name: str) -> pd.Series:
        if name not in combined.columns:
            return pd.Series(dtype=float)
        return combined[name].rolling(20, min_periods=10).sum()

    fp = pd.DataFrame({
        "avg_corr": avg_corr,
        "corr_velocity": corr_vel,
        "oil_spx_beta60": oil_beta,
        "regime": regimes.astype(float),
        "days_in_regime": days_in_regime,
        "grs": grs,
        "eq_vol20": eq_vol,
        "cmd_vol20": cmd_vol,
        "wti_mom20": mom("WTI Crude Oil"),
        "gold_mom20": mom("Gold"),
        "spx_mom20": mom("S&P 500"),
    })
    if dy_series is not None and len(dy_series):
        fp["dy_total"] = dy_series.reindex(fp.index).ffill()
    else:
        fp["dy_total"] = np.nan

    fp = fp.dropna(subset=["avg_corr", "grs", "eq_vol20"])
    geo = daily_geo_state(fp.index)
    fp = pd.concat([fp, geo], axis=1)

    regime_series = regimes.reindex(fp.index)
    return fp, regime_series


# ═══════════════════════════════════════════════════════════════════════════
# Matching — weighted distance in z-space across the whole configuration
# ═══════════════════════════════════════════════════════════════════════════

def _feature_weights(block_weights: dict[str, float], available: list[str]) -> dict[str, float]:
    """Per-feature weight = block weight / n available features in that block."""
    w: dict[str, float] = {}
    for block, feats in BLOCK_FEATURES.items():
        avail = [f for f in feats if f in available]
        if not avail:
            continue
        bw = float(block_weights.get(block, 1.0))
        for f in avail:
            w[f] = bw / len(avail)
    return w


def match_fingerprint(
    fp: pd.DataFrame,
    block_weights: dict[str, float] | None = None,
    k: int = MATCH_K,
    exclusion: int = EXCLUSION_ROWS,
    recent_blackout: int = RECENT_BLACKOUT,
    min_forward: int = max(FWD_HORIZONS),
) -> dict:
    """
    Compare today (last row) against all history. Weighted Euclidean distance
    in z-score space over the full feature vector; greedy non-overlapping
    selection (± exclusion rows); trailing recent_blackout rows excluded so
    today cannot trivially match last week.

    Returns {"today": {...}, "matches": [...], "n_candidates": int}
    """
    block_weights = block_weights or DEFAULT_BLOCK_WEIGHTS
    cols = [f for feats in BLOCK_FEATURES.values() for f in feats if f in fp.columns]
    F = fp[cols].copy()
    # dy_total may be all-NaN (DY unavailable) — drop unusable columns
    cols = [c for c in cols if F[c].notna().sum() > len(F) * 0.5]
    F = F[cols].ffill().dropna()
    if len(F) < 300:
        return {"today": {}, "matches": [], "n_candidates": 0}

    mu, sd = F.mean(), F.std().replace(0, 1.0)
    Z = (F - mu) / sd
    today_z = Z.iloc[-1]

    w = _feature_weights(block_weights, cols)
    wv = np.array([w[c] for c in cols])
    wv = wv / wv.sum()

    cand_end = len(Z) - max(min_forward, 1) - 1
    blackout_start = len(Z) - recent_blackout
    Z_cand = Z.iloc[:min(cand_end, blackout_start)]
    if len(Z_cand) < 100:
        return {"today": {}, "matches": [], "n_candidates": 0}

    diffs = Z_cand.values - today_z.values
    dists = np.sqrt((wv * diffs ** 2).sum(axis=1))
    order = np.argsort(dists)

    selected, blocked = [], set()
    for idx in order:
        if idx in blocked:
            continue
        selected.append(int(idx))
        blocked.update(range(max(0, idx - exclusion), min(len(Z_cand), idx + exclusion + 1)))
        if len(selected) >= k:
            break

    # Similarity = percentile rank of closeness: "closer to today than X% of
    # all candidate days". Robust to the greedy exclusion (a ratio-to-ceiling
    # metric collapses to 0 once nearby candidates are blocked out).
    n_cand = len(dists)
    matches = [{
        "date": F.index[i],
        "pos": int(np.where(fp.index == F.index[i])[0][0]),
        "distance": round(float(dists[i]), 3),
        "similarity": round(float((dists > dists[i]).sum()) / n_cand * 100, 1),
        "snapshot": {c: round(float(F.iloc[i][c]), 3) for c in cols},
        "z": {c: round(float(Z_cand.iloc[i][c]), 2) for c in cols},
    } for i in selected]

    return {
        "today": {
            "date": F.index[-1],
            "snapshot": {c: round(float(F.iloc[-1][c]), 3) for c in cols},
            "z": {c: round(float(today_z[c]), 2) for c in cols},
        },
        "matches": matches,
        "n_candidates": int(len(Z_cand)),
        "features": cols,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Outcomes — what followed each match, and the base rate that gives it meaning
# ═══════════════════════════════════════════════════════════════════════════

def _verdict(spx_fwd: float, crisis_hit: bool) -> str:
    if np.isnan(spx_fwd):
        return "—"
    if spx_fwd <= _BAD_SPX or crisis_hit:
        return "BAD"
    if spx_fwd >= _GOOD_SPX and not crisis_hit:
        return "GOOD"
    return "MIXED"


def _fwd_return(r: pd.Series, pos: int, h: int) -> float:
    fwd = r.iloc[pos + 1: pos + 1 + h].dropna()
    if len(fwd) < min(h, 5):
        return float("nan")
    return float(np.expm1(np.log1p(fwd).sum()))


def match_outcomes(
    result: dict,
    combined: pd.DataFrame,
    regimes: pd.Series,
    horizons: tuple[int, ...] = FWD_HORIZONS,
) -> list[dict]:
    """Forward outcomes per match: asset returns, regime path, S&P drawdown, verdict."""
    assets = [a for a in ("S&P 500", "WTI Crude Oil", "Gold") if a in combined.columns]
    out = []
    h_max = max(horizons)
    for m in result.get("matches", []):
        pos_arr = np.where(combined.index == m["date"])[0]
        if not len(pos_arr):
            continue
        pos = int(pos_arr[0])
        rets = {a: {h: _fwd_return(combined[a], pos, h) for h in horizons} for a in assets}

        spx_path = combined.get("S&P 500", pd.Series(dtype=float)).iloc[pos + 1: pos + 1 + h_max]
        cum = np.exp(np.log1p(spx_path.fillna(0)).cumsum()) - 1.0
        max_dd = float(cum.min()) if len(cum) else float("nan")

        rpos_arr = np.where(regimes.index == m["date"])[0]
        regime_then = regime_20 = regime_60 = None
        crisis_hit = False
        if len(rpos_arr):
            rp = int(rpos_arr[0])
            regime_then = int(regimes.iloc[rp])
            if rp + 20 < len(regimes):
                regime_20 = int(regimes.iloc[rp + 20])
            if rp + 60 < len(regimes):
                regime_60 = int(regimes.iloc[rp + 60])
            crisis_hit = bool((regimes.iloc[rp + 1: rp + 1 + 60] == 3).any())

        spx60 = rets.get("S&P 500", {}).get(60, float("nan"))
        out.append({
            **m,
            "returns": rets,
            "spx_max_dd_60": max_dd,
            "regime_then": regime_then, "regime_20": regime_20, "regime_60": regime_60,
            "crisis_within_60": crisis_hit,
            "verdict": _verdict(spx60, crisis_hit),
        })
    return out


def base_rates(
    combined: pd.DataFrame,
    regimes: pd.Series,
    horizon: int = 60,
) -> dict:
    """
    Unconditional GOOD/MIXED/BAD frequencies over ALL history — the yardstick
    that makes the match verdicts meaningful.
    """
    if "S&P 500" not in combined.columns:
        return {}
    r = combined["S&P 500"]
    log1p = np.log1p(r.fillna(0))
    fwd = (np.exp(log1p[::-1].rolling(horizon).sum()[::-1].shift(-1)) - 1.0)
    crisis = (regimes == 3).astype(float)
    crisis_fwd = crisis[::-1].rolling(horizon).max()[::-1].shift(-1)
    crisis_fwd = crisis_fwd.reindex(fwd.index).fillna(0) > 0

    valid = fwd.dropna().index
    counts = {"GOOD": 0, "MIXED": 0, "BAD": 0}
    for d in valid:
        counts[_verdict(float(fwd.loc[d]), bool(crisis_fwd.loc[d]))] += 1
    n = sum(counts.values()) or 1
    return {k: v / n for k, v in counts.items()} | {"n": n}


def aggregate_verdicts(outcomes: list[dict]) -> dict:
    """Similarity-weighted GOOD/MIXED/BAD shares over the matched analogs."""
    w_tot, acc = 0.0, {"GOOD": 0.0, "MIXED": 0.0, "BAD": 0.0}
    counts = {"GOOD": 0, "MIXED": 0, "BAD": 0}
    for o in outcomes:
        if o["verdict"] == "—":
            continue
        w = max(o["similarity"], 1.0)
        acc[o["verdict"]] += w
        counts[o["verdict"]] += 1
        w_tot += w
    if w_tot == 0:
        return {}
    return {k: v / w_tot for k, v in acc.items()} | {"counts": counts, "n": sum(counts.values())}

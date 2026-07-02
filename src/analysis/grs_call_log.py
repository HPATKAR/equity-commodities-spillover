"""
GRS Live Call Log — forward-looking accuracy tracker.

Every week, logs the live GRS score as a directional call (STRESS / CALM).
Four weeks later, grades the call against what realized vol actually did.

One JSON entry per ISO week. Unresolved entries carry "resolved": false.
Resolved entries carry outcome_vol, outcome_stressed, and is_hit.

Hit definition:
  - Build 20d realized vol composite (60% equity, 40% commodity, annualized %)
  - Rolling 252d median of that series = the "normal" baseline for each call date
  - Outcome: mean vol over the 20 trading days AFTER call_date vs. that baseline
  - STRESSED if forward vol > baseline median; CALM otherwise
  - Hit = (GRS called STRESS and outcome STRESSED) OR (GRS called CALM and outcome CALM)

Why rolling median instead of fixed threshold:
  The baseline shifts with the vol regime, so a 15% annualized vol week in the
  Covid era reads as calm while the same in 2018 reads as elevated. By construction
  ~50% of outcomes are STRESSED and ~50% CALM, making 50% the no-skill baseline —
  any hit rate above 50% is genuine edge from the model.

File: logs/grs_call_log.json (project-relative, gitignored)
"""

from __future__ import annotations

import datetime
import json
import pathlib

import numpy as np
import pandas as pd

_LOG_PATH       = pathlib.Path(__file__).resolve().parents[2] / "logs" / "grs_call_log.json"
_HORIZON_WEEKS  = 4      # forward window length
_HORIZON_DAYS   = 20     # ~4 trading weeks
_STRESS_THRESH  = 50.0   # GRS ≥ 50 → "STRESS" call
_VOL_LOOKBACK   = 252    # rolling window for baseline median


# ── I/O ───────────────────────────────────────────────────────────────────────

def _read_log() -> list[dict]:
    try:
        if _LOG_PATH.exists():
            data = json.loads(_LOG_PATH.read_text())
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def _write_log(entries: list[dict]) -> None:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LOG_PATH.write_text(json.dumps(entries, indent=2, default=str))
    except Exception:
        pass


def _iso_week(dt: datetime.date) -> str:
    """ISO year-week, e.g. '2025-W03'. One entry per week."""
    return dt.strftime("%G-W%V")


# ── Realized vol builder ──────────────────────────────────────────────────────

def _build_realized_vol(eq_r: pd.DataFrame, cmd_r: pd.DataFrame) -> pd.Series:
    """
    20d realized vol composite (annualized %) used for both baseline and outcomes.
    Weights: 60% equity universe, 40% commodity universe.
    """
    parts: list[tuple[pd.Series, float]] = []
    if not eq_r.empty:
        ev = (eq_r.rolling(20, min_periods=10).std()
                  .mean(axis=1) * np.sqrt(252) * 100)
        parts.append((ev, 0.60))
    if not cmd_r.empty:
        cv = (cmd_r.rolling(20, min_periods=10).std()
                   .mean(axis=1) * np.sqrt(252) * 100)
        parts.append((cv, 0.40))
    if not parts:
        return pd.Series(dtype=float)

    base_idx = parts[0][0].dropna().index
    rv = sum(s.reindex(base_idx).ffill() * w for s, w in parts)
    return rv.dropna().rename("realized_vol")


# ── Logging ───────────────────────────────────────────────────────────────────

def log_weekly_call(
    grs_score: float,
    grs_label: str,
    cis: float = 0.0,
    tps: float = 0.0,
    mcs: float = 0.0,
    news_gpr: float = 0.0,
) -> None:
    """
    Append this week's GRS call if no entry exists for the current ISO week.
    Same-week reloads update the score (model may have fresher data) but
    never create a duplicate row.
    """
    today    = datetime.date.today()
    week_key = _iso_week(today)
    entries  = _read_log()
    call     = "STRESS" if grs_score >= _STRESS_THRESH else "CALM"
    out_dt   = (today + datetime.timedelta(weeks=_HORIZON_WEEKS)).isoformat()

    existing = next((e for e in entries if e.get("week") == week_key), None)
    if existing is None:
        entries.append({
            "week":             week_key,
            "call_date":        today.isoformat(),
            "grs_score":        round(float(grs_score), 1),
            "grs_label":        grs_label,
            "call":             call,
            "cis":              round(float(cis),      1),
            "tps":              round(float(tps),      1),
            "mcs":              round(float(mcs),      1),
            "news_gpr":         round(float(news_gpr), 1),
            "outcome_date":     out_dt,
            "resolved":         False,
            "outcome_vol":      None,
            "outcome_vol_base": None,
            "outcome_stressed": None,
            "is_hit":           None,
        })
    else:
        existing["grs_score"] = round(float(grs_score), 1)
        existing["grs_label"] = grs_label
        existing["call"]      = call

    _write_log(entries)


# ── Resolution ────────────────────────────────────────────────────────────────

def resolve_outcomes(eq_r: pd.DataFrame, cmd_r: pd.DataFrame) -> list[dict]:
    """
    Grade unresolved entries whose outcome window has closed.

    For each entry with outcome_date ≤ today and resolved=False:
      1. Compute mean realized vol over the 20 trading days after call_date
      2. Compare to the 252d rolling median at call_date (the baseline)
      3. Mark STRESSED if forward vol > baseline; CALM otherwise
      4. is_hit = (call == outcome)

    Returns the full updated log (resolved + pending).
    """
    today   = datetime.date.today()
    entries = _read_log()
    if not entries:
        return entries

    rv      = _build_realized_vol(eq_r, cmd_r)
    if rv.empty:
        return entries

    rv_median = rv.rolling(_VOL_LOOKBACK, min_periods=_VOL_LOOKBACK // 4).median()

    changed = False
    for entry in entries:
        if entry.get("resolved"):
            continue
        outcome_dt = datetime.date.fromisoformat(entry["outcome_date"])
        if outcome_dt > today:
            continue

        call_ts = pd.Timestamp(entry["call_date"])

        # Forward vol: 20 trading days after call_date
        fwd_idx = rv.index[rv.index > call_ts][:_HORIZON_DAYS]
        if len(fwd_idx) < _HORIZON_DAYS // 2:
            continue  # not enough forward data in loaded range

        fwd_vol = float(rv.reindex(fwd_idx).mean())

        # Baseline: rolling median at call_date (last available value ≤ call_ts)
        baseline_slice = rv_median[rv_median.index <= call_ts].dropna()
        if baseline_slice.empty:
            continue
        baseline_med = float(baseline_slice.iloc[-1])

        outcome_stressed = bool(fwd_vol > baseline_med)
        call_stressed    = entry["call"] == "STRESS"

        entry["resolved"]          = True
        entry["outcome_vol"]       = round(fwd_vol,      2)
        entry["outcome_vol_base"]  = round(baseline_med, 2)
        entry["outcome_stressed"]  = outcome_stressed
        entry["is_hit"]            = bool(call_stressed == outcome_stressed)
        changed = True

    if changed:
        _write_log(entries)

    return entries


# ── Aggregation ───────────────────────────────────────────────────────────────

def compute_hit_stats(entries: list[dict]) -> dict:
    """
    Aggregate accuracy stats from the call log.

    Returns dict with:
      n_total, n_resolved, n_pending, n_hits, hit_rate (0-100 | None),
      n_stress_calls, n_calm_calls,
      stress_hit_rate, calm_hit_rate,
      rolling_hr: list of {date, rolling_hit_rate, n} in chronological order,
      resolved: list of resolved entries (for charting),
      pending:  list of pending entries,
    """
    resolved = sorted(
        [e for e in entries if e.get("resolved") and e.get("is_hit") is not None],
        key=lambda e: e["call_date"],
    )
    pending = [e for e in entries if not e.get("resolved")]

    n_resolved = len(resolved)
    n_hits     = sum(1 for e in resolved if e["is_hit"])

    stress_res = [e for e in resolved if e["call"] == "STRESS"]
    calm_res   = [e for e in resolved if e["call"] == "CALM"]

    def _rate(lst):
        hits = sum(1 for e in lst if e["is_hit"])
        return round(hits / len(lst) * 100, 1) if lst else None

    # Cumulative hit rate series (chronological)
    rolling_hr: list[dict] = []
    cum_hits = 0
    for i, e in enumerate(resolved):
        cum_hits += int(e["is_hit"])
        rolling_hr.append({
            "date":             e["call_date"],
            "rolling_hit_rate": round(cum_hits / (i + 1) * 100, 1),
            "n":                i + 1,
        })

    return {
        "n_total":         len(entries),
        "n_resolved":      n_resolved,
        "n_pending":       len(pending),
        "n_hits":          n_hits,
        "hit_rate":        _rate(resolved),
        "n_stress_calls":  len(stress_res),
        "n_calm_calls":    len(calm_res),
        "stress_hit_rate": _rate(stress_res),
        "calm_hit_rate":   _rate(calm_res),
        "rolling_hr":      rolling_hr,
        "resolved":        resolved,
        "pending":         pending,
    }

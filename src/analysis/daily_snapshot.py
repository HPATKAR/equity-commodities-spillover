"""
Daily snapshot for the Home "what changed" delta panel.

Stores three slots:
  snapshot_today       — latest capture, rewritten on every load
  snapshot_yesterday   — previous calendar day's last capture, rolled over
  snapshot_first_today — the FIRST capture of the current calendar day

Baseline selection (what "today" is compared against), best available first:
  1. snapshot_yesterday    → true day-over-day move (preferred)
  2. snapshot_first_today  → intraday move "since first capture today", used
                             until a prior calendar day exists. This is what
                             keeps the panel useful on day one and on an
                             ephemeral filesystem (e.g. Render), where a
                             prior-day file rarely survives a restart.
  3. None                  → only on the very first capture ever seen, when
                             there is genuinely nothing to diff against.

File: <SNAPSHOT_DIR or logs>/delta_snapshot.json. Set SPILLOVER_SNAPSHOT_DIR to
a persistent path (a mounted disk) to survive restarts on ephemeral hosts.

Day-rollover logic (called on every page load):
  - No file                     → write today + first_today, baseline None
  - snapshot_today.date < today → roll today → yesterday, reset first_today
  - snapshot_today.date == today→ keep yesterday + first_today, update today
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib


def _snapshot_path() -> pathlib.Path:
    """Resolve the snapshot file path. Honours SPILLOVER_SNAPSHOT_DIR so a
    persistent disk can back it on ephemeral hosts without a code change."""
    env_dir = os.environ.get("SPILLOVER_SNAPSHOT_DIR")
    base = (pathlib.Path(env_dir) if env_dir
            else pathlib.Path(__file__).resolve().parents[2] / "logs")
    return base / "delta_snapshot.json"


_SNAPSHOT_PATH = _snapshot_path()
_MIN_DELTA     = 0.3   # suppress noise below this magnitude
_TOP_N         = 8     # maximum rows shown in the panel


# ── Snapshot I/O ─────────────────────────────────────────────────────────────

def _read_file() -> dict:
    try:
        path = _snapshot_path()
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _write_file(data: dict) -> None:
    try:
        path = _snapshot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        pass


def _build_payload(
    conflict_results: dict,
    portfolio_cis: float,
    portfolio_tps: float,
    geo_risk_score: float,
) -> dict:
    from src.utils.timeutil import now_ct, today_ct
    today_str = today_ct().isoformat()
    now_str   = now_ct().strftime("%H:%M")
    conflicts: dict[str, dict] = {}
    for cid, v in conflict_results.items():
        conflicts[cid] = {
            "label": v.get("label", cid),
            "cis":   round(float(v.get("cis",  0.0)), 2),
            "tps":   round(float(v.get("tps",  0.0)), 2),
        }
    return {
        "date":             today_str,
        "captured_at":      now_str,
        "portfolio_cis":    round(float(portfolio_cis),    2),
        "portfolio_tps":    round(float(portfolio_tps),    2),
        "geo_risk_score":   round(float(geo_risk_score),   2),
        "conflicts":        conflicts,
    }


def update_snapshot(
    conflict_results: dict,
    portfolio_cis: float,
    portfolio_tps: float,
    geo_risk_score: float,
) -> tuple[dict | None, dict]:
    """
    Persist today's snapshot and return (baseline, today) for delta computation.
    Call once per page load after conflict data is available.

    Returns
    -------
    baseline : dict | None  — what today is compared against. Prior calendar
                              day if one exists; else the first capture of the
                              current day (intraday); else None only on the
                              very first capture ever, when nothing can be
                              compared. The caller labels it by inspecting the
                              baseline's own ``date`` vs today.
    today    : dict         — current payload
    """
    from src.utils.timeutil import today_ct
    today_str = today_ct().isoformat()
    file_data = _read_file()
    today_slot       = file_data.get("snapshot_today")
    yesterday_slot   = file_data.get("snapshot_yesterday")
    first_today_slot = file_data.get("snapshot_first_today")

    new_today = _build_payload(conflict_results, portfolio_cis, portfolio_tps, geo_risk_score)

    if today_slot is None:
        # First capture ever — seed first_today so an intraday baseline can
        # form on the next load; nothing to compare against yet.
        _write_file({"snapshot_today": new_today, "snapshot_yesterday": None,
                     "snapshot_first_today": new_today})
        return None, new_today

    if today_slot.get("date", "") < today_str:
        # New calendar day: roll today → yesterday, reset the intraday anchor.
        _write_file({"snapshot_today": new_today, "snapshot_yesterday": today_slot,
                     "snapshot_first_today": new_today})
        return today_slot, new_today

    # Same day: keep yesterday, preserve the day's first capture as the
    # intraday anchor (heal older two-slot files that predate first_today).
    if first_today_slot is None or first_today_slot.get("date", "") != today_str:
        first_today_slot = today_slot
    _write_file({"snapshot_today": new_today, "snapshot_yesterday": yesterday_slot,
                 "snapshot_first_today": first_today_slot})

    # Prefer a true prior-day baseline; else fall back to the intraday anchor,
    # but only once it is an EARLIER capture than now (otherwise there is no
    # move to show).
    if yesterday_slot is not None:
        return yesterday_slot, new_today
    if (first_today_slot is not None
            and first_today_slot.get("captured_at") != new_today.get("captured_at")):
        return first_today_slot, new_today
    return None, new_today


# ── Delta computation ─────────────────────────────────────────────────────────

def compute_deltas(yesterday: dict, today: dict) -> list[dict]:
    """
    Compute ranked list of metric moves between yesterday and today.

    Returns list of dicts sorted by |delta| descending:
      {key, label, delta, current, previous, category}
    Only includes items where |delta| >= _MIN_DELTA.
    """
    rows: list[dict] = []

    def _add(key, label, prev, curr, category):
        d = round(float(curr) - float(prev), 2)
        if abs(d) >= _MIN_DELTA:
            rows.append({
                "key":      key,
                "label":    label,
                "delta":    d,
                "current":  round(float(curr), 1),
                "previous": round(float(prev), 1),
                "category": category,
            })

    # Portfolio-level scores
    _add("geo_risk_score",  "Geo Risk Score",    yesterday["geo_risk_score"],  today["geo_risk_score"],  "composite")
    _add("portfolio_cis",   "Portfolio CIS",     yesterday["portfolio_cis"],   today["portfolio_cis"],   "aggregate")
    _add("portfolio_tps",   "Portfolio TPS",     yesterday["portfolio_tps"],   today["portfolio_tps"],   "aggregate")

    # Per-conflict
    yc = yesterday.get("conflicts", {})
    tc = today.get("conflicts", {})
    for cid, tc_val in tc.items():
        if cid not in yc:
            continue
        yc_val = yc[cid]
        label  = tc_val.get("label", cid)
        _add(f"{cid}_cis", f"{label} · CIS", yc_val["cis"], tc_val["cis"], "conflict")
        _add(f"{cid}_tps", f"{label} · TPS", yc_val["tps"], tc_val["tps"], "conflict")

    # Sort by absolute magnitude descending, cap at _TOP_N
    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return rows[:_TOP_N]

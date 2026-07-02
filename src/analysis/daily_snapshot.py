"""
Daily snapshot for the Home "what changed since yesterday" delta panel.

Stores two slots:
  snapshot_today     — written on the first load of each calendar day
  snapshot_yesterday — previous day's snapshot_today, rolled over automatically

File: logs/delta_snapshot.json (project-relative, gitignored or irrelevant to CI).

Day-rollover logic (called on every page load):
  - If snapshot_today.date < today   → copy today → yesterday, write fresh today
  - If snapshot_today.date == today  → keep yesterday unchanged, update today
  - If no file                       → write today only, yesterday = None
"""

from __future__ import annotations

import datetime
import json
import pathlib

_SNAPSHOT_PATH = pathlib.Path(__file__).resolve().parents[2] / "logs" / "delta_snapshot.json"
_MIN_DELTA     = 0.3   # suppress noise below this magnitude
_TOP_N         = 8     # maximum rows shown in the panel


# ── Snapshot I/O ─────────────────────────────────────────────────────────────

def _read_file() -> dict:
    try:
        if _SNAPSHOT_PATH.exists():
            return json.loads(_SNAPSHOT_PATH.read_text())
    except Exception:
        pass
    return {}


def _write_file(data: dict) -> None:
    try:
        _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT_PATH.write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        pass


def _build_payload(
    conflict_results: dict,
    portfolio_cis: float,
    portfolio_tps: float,
    geo_risk_score: float,
) -> dict:
    today_str = datetime.date.today().isoformat()
    now_str   = datetime.datetime.now().strftime("%H:%M")
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
    Persist today's snapshot and return (yesterday, today) for delta computation.
    Call once per page load after conflict data is available.

    Returns
    -------
    yesterday : dict | None  — previous day's payload, or None if first run
    today     : dict         — current payload
    """
    today_str = datetime.date.today().isoformat()
    file_data = _read_file()
    today_slot     = file_data.get("snapshot_today")
    yesterday_slot = file_data.get("snapshot_yesterday")

    new_today = _build_payload(conflict_results, portfolio_cis, portfolio_tps, geo_risk_score)

    if today_slot is None:
        # First ever run
        new_file = {"snapshot_today": new_today, "snapshot_yesterday": None}
        _write_file(new_file)
        return None, new_today

    if today_slot.get("date", "") < today_str:
        # New day: roll today → yesterday, fresh today
        new_file = {"snapshot_today": new_today, "snapshot_yesterday": today_slot}
        _write_file(new_file)
        return today_slot, new_today
    else:
        # Same day: keep yesterday unchanged, update today's values
        new_file = {"snapshot_today": new_today, "snapshot_yesterday": yesterday_slot}
        _write_file(new_file)
        return yesterday_slot, new_today


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

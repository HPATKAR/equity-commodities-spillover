"""
Daily-snapshot baseline selection for the Home "what changed" panel.

Pins: the first capture ever has no baseline; once a day has more than one
capture an INTRADAY baseline forms (fixing the perpetual "no prior baseline
yet"); a true prior-day snapshot is preferred over the intraday anchor; and
the snapshot file location honours SPILLOVER_SNAPSHOT_DIR so a persistent disk
can back it on an ephemeral host.
"""

import datetime
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import daily_snapshot as ds

_CR = {"iran": {"label": "Iran", "cis": 60, "tps": 40}}


def _payload(date, cap, cis=50, tps=50, geo=55):
    return {"date": date, "captured_at": cap,
            "portfolio_cis": cis, "portfolio_tps": tps, "geo_risk_score": geo,
            "conflicts": {"iran": {"label": "Iran", "cis": 60, "tps": 40}}}


@pytest.fixture
def snap_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SPILLOVER_SNAPSHOT_DIR", str(tmp_path))
    return tmp_path


def _read(snap_dir):
    return json.loads((snap_dir / "delta_snapshot.json").read_text())


def _seed(snap_dir, data):
    (snap_dir / "delta_snapshot.json").write_text(json.dumps(data))


def test_first_capture_ever_has_no_baseline(snap_dir):
    base, today = ds.update_snapshot(_CR, 50, 50, 55)
    assert base is None                                  # nothing to diff yet
    f = _read(snap_dir)
    assert f["snapshot_yesterday"] is None
    assert f["snapshot_first_today"]["date"] == today["date"]   # anchor seeded


def test_intraday_baseline_forms_without_a_prior_day(snap_dir):
    from src.utils.timeutil import today_ct
    today = today_ct().isoformat()
    _seed(snap_dir, {
        "snapshot_today":       _payload(today, "07:00", cis=50),
        "snapshot_yesterday":   None,
        "snapshot_first_today": _payload(today, "07:00", cis=50),
    })
    base, cur = ds.update_snapshot(_CR, 55, 50, 55)      # CIS 50 → 55 intraday
    assert base is not None
    assert base["date"] == today and base["captured_at"] == "07:00"
    deltas = ds.compute_deltas(base, cur)
    assert any(d["key"] == "portfolio_cis" and d["delta"] == 5.0 for d in deltas)


def test_prior_day_is_preferred_over_intraday_anchor(snap_dir):
    from src.utils.timeutil import today_ct
    today = today_ct().isoformat()
    yday = (today_ct() - datetime.timedelta(days=1)).isoformat()
    _seed(snap_dir, {
        "snapshot_today":       _payload(yday, "16:00", cis=40),
        "snapshot_yesterday":   None,
        "snapshot_first_today": _payload(yday, "16:00", cis=40),
    })
    base, _cur = ds.update_snapshot(_CR, 50, 50, 55)     # crosses the day line
    assert base["date"] == yday                          # day-over-day wins
    f = _read(snap_dir)
    assert f["snapshot_yesterday"]["date"] == yday
    assert f["snapshot_first_today"]["date"] == today    # anchor reset for today


def test_equal_capture_time_yields_no_intraday_move(snap_dir):
    from src.utils.timeutil import today_ct
    today = today_ct().isoformat()
    _seed(snap_dir, {
        "snapshot_today":       _payload(today, "07:00"),
        "snapshot_yesterday":   None,
        "snapshot_first_today": _payload(today, "07:00"),
    })
    # Force the new capture to land at the same minute as the anchor. The
    # module builds captured_at via timeutil.now_ct(), so patch that.
    from unittest.mock import patch
    from src.utils import timeutil
    fixed = timeutil.now_ct().replace(hour=7, minute=0, second=0, microsecond=0)
    with patch("src.utils.timeutil.now_ct", return_value=fixed):
        base, _cur = ds.update_snapshot(_CR, 99, 99, 99)
    assert base is None                                  # same minute, no move


def test_snapshot_path_honours_env_var(snap_dir):
    ds.update_snapshot(_CR, 50, 50, 55)
    assert (snap_dir / "delta_snapshot.json").exists()   # wrote to the env dir

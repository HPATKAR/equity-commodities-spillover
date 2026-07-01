"""
Thin disk cache for expensive page computations.

Persists serialisable dicts to JSON under cache/ in the project root so that
results survive Streamlit server restarts. On next page load the stale result
is shown immediately while the user decides whether to refresh.

Usage:
    from src.utils.page_cache import save_cache, load_cache, age_str

    # Save after a successful computation:
    save_cache("pipeline_validation", result_dict)

    # Load at page start (returns None if no file exists):
    data, saved_at = load_cache("pipeline_validation")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

_log = logging.getLogger(__name__)

# Cache directory relative to the project root (two levels up from this file:
# src/utils/page_cache.py → src/ → project_root/)
_CACHE_DIR = Path(__file__).parent.parent.parent / "cache"


def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{key}.json"


def save_cache(key: str, data: dict) -> None:
    """
    Persist data dict to disk under cache/<key>.json.
    Adds a __saved_at__ ISO timestamp so load_cache can report staleness.
    """
    payload = {"__saved_at__": datetime.now(timezone.utc).isoformat(), **data}
    path = _cache_path(key)
    try:
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        _log.debug("page_cache: saved %s (%d bytes)", key, path.stat().st_size)
    except Exception as exc:
        _log.warning("page_cache: could not save %s — %s", key, exc)


def load_cache(key: str) -> tuple[dict | None, datetime | None]:
    """
    Load cache/<key>.json from disk.
    Returns (data_dict, saved_at_utc) or (None, None) if no file exists.
    """
    path = _cache_path(key)
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        saved_at_str = payload.pop("__saved_at__", None)
        saved_at = (
            datetime.fromisoformat(saved_at_str)
            if saved_at_str else None
        )
        return payload, saved_at
    except Exception as exc:
        _log.warning("page_cache: could not load %s — %s", key, exc)
        return None, None


def age_str(saved_at: datetime | None) -> str:
    """Human-readable staleness: '2h 15m ago', '3d ago', etc."""
    if saved_at is None:
        return "unknown age"
    now = datetime.now(timezone.utc)
    # Make saved_at tz-aware if naive
    if saved_at.tzinfo is None:
        saved_at = saved_at.replace(tzinfo=timezone.utc)
    delta = now - saved_at
    total_seconds = int(delta.total_seconds())
    if total_seconds < 60:
        return "just now"
    if total_seconds < 3600:
        return f"{total_seconds // 60}m ago"
    if total_seconds < 86400:
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        return f"{h}h {m}m ago" if m else f"{h}h ago"
    d = total_seconds // 86400
    return f"{d}d ago"

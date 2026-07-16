"""
Disk-persistent cache for expensive SHARED artifacts (conflict scores, asset
scores, single-stock returns, ...).

Why this exists
---------------
The app already warms an in-memory ``@st.cache_data`` store, but that cache is
per-process and evaporates on every deploy, scale-up, or TTL race - so a COLD
process has to recompute ~20s of network-bound work before the first visitor
sees anything. That is the difference between a 30-80s cold load and a fast one.

This module lets a cold process read the LAST computed value from disk in
milliseconds instead. The in-memory cache still serves warm hits; this only
backs the cold (cache-miss) path - a classic stale-while-revalidate layer.

Store location is env-configurable (``ARTIFACT_CACHE_DIR``) so production can
point it at a persistent disk / mounted volume that survives redeploys; locally
it defaults to ``<repo>/cache/artifacts``. Writes are atomic and every function
is failure-safe (never raises) - a bad cache must never break a page.
"""
from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)

_DIR = Path(os.environ.get(
    "ARTIFACT_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "cache" / "artifacts"),
))


def _path(key: str) -> Path:
    _DIR.mkdir(parents=True, exist_ok=True)
    return _DIR / f"{key}.pkl"


def read_artifact(key: str, max_age_s: float) -> Optional[Any]:
    """Return the cached object if it exists and is younger than ``max_age_s``;
    otherwise ``None`` (missing, stale, or unreadable). Never raises."""
    try:
        p = _path(key)
        if not p.exists():
            return None
        if (time.time() - p.stat().st_mtime) > max_age_s:
            return None
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as exc:  # corrupt / partial / version mismatch - treat as miss
        _log.debug("artifact_cache read '%s' failed: %s", key, exc)
        return None


def write_artifact(key: str, data: Any) -> None:
    """Persist ``data`` atomically (write-temp-then-rename). Never raises."""
    try:
        p = _path(key)
        tmp = p.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)  # atomic on POSIX - readers never see a partial file
    except Exception as exc:
        _log.debug("artifact_cache write '%s' failed: %s", key, exc)


def artifact_age_s(key: str) -> Optional[float]:
    """Age of the cached artifact in seconds, or ``None`` if absent."""
    try:
        p = _path(key)
        return (time.time() - p.stat().st_mtime) if p.exists() else None
    except Exception:
        return None


def cached_call(key: str, max_age_s: float, compute, *args, **kwargs) -> Any:
    """Stale-while-revalidate helper: return a fresh disk artifact if present,
    else run ``compute(*args, **kwargs)``, persist it, and return it. A falsy
    computed result is NOT cached (so a transient empty fetch doesn't stick)."""
    hit = read_artifact(key, max_age_s)
    if hit is not None:
        return hit
    result = compute(*args, **kwargs)
    if result is not None and not (hasattr(result, "empty") and result.empty) and result != {}:
        write_artifact(key, result)
    return result

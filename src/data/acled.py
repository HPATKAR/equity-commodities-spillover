"""
ACLED (Armed Conflict Location & Event Data) live feed.

ACLED provides free public API access for academic use (registration required).
API documentation: https://acleddata.com/acleddatanerd/

Registration: https://developer.acleddata.com/

Returns event counts, fatalities, and event types for active conflict zones
used to dynamically update CIS (Conflict Intensity Score) inputs — replacing
the static manual data that was previously the only source (GAP 17).

Environment variable: ACLED_API_KEY and ACLED_EMAIL must be set.
Falls back gracefully to zero if not configured.

Usage in conflict_model.py:
    from src.data.acled import fetch_acled_intensity
    acled = fetch_acled_intensity(conflict_id="ukraine_russia", days=30)
    # Returns: {events_30d, fatalities_30d, events_trend, escalation_signal}
"""

from __future__ import annotations

import datetime
import os
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# ── ACLED country/region mapping for each tracked conflict ──────────────────

_ACLED_CONFLICT_MAP: dict[str, dict] = {
    "ukraine_russia": {
        "country": "Ukraine",
        "actor_filter": None,   # all events in country
        "event_types": ["Battles", "Explosions/Remote violence", "Violence against civilians"],
        "description": "Ukraine–Russia War",
    },
    "israel_hamas": {
        "country": "Palestine",  # ACLED uses "Palestine" for Gaza/West Bank
        "actor_filter": None,
        "event_types": ["Battles", "Explosions/Remote violence", "Violence against civilians"],
        "description": "Israel-Hamas War",
    },
    "iran_regional": {
        "country": "Yemen",      # Houthi operations most trackable via ACLED
        "actor_filter": None,
        "event_types": ["Battles", "Explosions/Remote violence"],
        "description": "Iran/Hormuz Proxy",
    },
    "red_sea_houthi": {
        "country": "Yemen",
        "actor_filter": "Houthi",
        "event_types": ["Explosions/Remote violence"],
        "description": "Red Sea Houthi Operations",
    },
    "russia_ukraine_border": {
        "country": "Russia",
        "actor_filter": None,
        "event_types": ["Battles", "Explosions/Remote violence"],
        "description": "Russia border violence",
    },
}

# Base URL for ACLED API
_ACLED_BASE = "https://api.acleddata.com/acled/read"


@st.cache_data(ttl=21600, show_spinner=False)  # 6-hour cache
def fetch_acled_intensity(
    conflict_id: str,
    days: int = 30,
    api_key: Optional[str] = None,
    email: Optional[str] = None,
) -> dict:
    """
    Fetch ACLED event data for a conflict zone and compute live intensity metrics.

    Parameters
    ----------
    conflict_id : str — key from _ACLED_CONFLICT_MAP
    days        : int — lookback window in days
    api_key     : str — ACLED API key (or set ACLED_API_KEY env var)
    email       : str — registered ACLED email (or set ACLED_EMAIL env var)

    Returns
    -------
    dict with keys:
        events_nd       : int   — total events in past `days` days
        fatalities_nd   : int   — total fatalities in past `days` days
        events_prior    : int   — events in prior `days` window (for trend)
        events_trend    : float — events_nd / events_prior − 1 (+ = escalating)
        escalation_signal: str — "escalating" / "stable" / "de-escalating"
        event_types     : dict  — {type: count}
        data_available  : bool  — False if ACLED key not set or fetch failed
        source          : str   — "ACLED live" or "unavailable"
        as_of           : str   — ISO date string
    """
    _empty = {
        "events_nd":           0,
        "fatalities_nd":       0,
        "events_prior":        0,
        "events_trend":        0.0,
        "escalation_signal":   "stable",
        "event_types":         {},
        "data_available":      False,
        "source":              "ACLED key not configured",
        "as_of":               str(datetime.date.today()),
    }

    if conflict_id not in _ACLED_CONFLICT_MAP:
        return {**_empty, "source": f"Unknown conflict_id: {conflict_id}"}

    _api_key = api_key or os.environ.get("ACLED_API_KEY", "")
    _email   = email   or os.environ.get("ACLED_EMAIL",   "")

    if not _api_key or not _email:
        return _empty

    cfg = _ACLED_CONFLICT_MAP[conflict_id]
    today    = datetime.date.today()
    date_end = today.isoformat()
    date_start       = (today - datetime.timedelta(days=days)).isoformat()
    date_prior_start = (today - datetime.timedelta(days=days * 2)).isoformat()
    date_prior_end   = (today - datetime.timedelta(days=days + 1)).isoformat()

    try:
        import requests

        def _fetch(start: str, end: str) -> list[dict]:
            params = {
                "key":         _api_key,
                "email":       _email,
                "country":     cfg["country"],
                "event_date":  f"{start}|{end}",
                "event_date_where": "BETWEEN",
                "fields":      "event_date|event_type|fatalities",
                "limit":       5000,
            }
            if cfg.get("actor_filter"):
                params["actor1"] = cfg["actor_filter"]
            r = requests.get(_ACLED_BASE, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                return data.get("data", [])
            return []

        current_events = _fetch(date_start, date_end)
        prior_events   = _fetch(date_prior_start, date_prior_end)

        # Filter by event types if specified
        _etypes = cfg.get("event_types")
        if _etypes:
            current_events = [e for e in current_events if e.get("event_type") in _etypes]
            prior_events   = [e for e in prior_events   if e.get("event_type") in _etypes]

        events_nd    = len(current_events)
        fatalities_nd = sum(int(e.get("fatalities", 0) or 0) for e in current_events)
        events_prior  = max(len(prior_events), 1)  # avoid div-by-zero

        events_trend = float(events_nd / events_prior - 1.0)

        if events_trend > 0.15:
            escalation = "escalating"
        elif events_trend < -0.15:
            escalation = "de-escalating"
        else:
            escalation = "stable"

        # Event type breakdown
        etype_counts: dict[str, int] = {}
        for e in current_events:
            et = str(e.get("event_type", "Unknown"))
            etype_counts[et] = etype_counts.get(et, 0) + 1

        return {
            "events_nd":           events_nd,
            "fatalities_nd":       fatalities_nd,
            "events_prior":        events_prior,
            "events_trend":        round(events_trend, 3),
            "escalation_signal":   escalation,
            "event_types":         etype_counts,
            "data_available":      True,
            "source":              "ACLED live",
            "as_of":               date_end,
        }

    except Exception as e:
        return {**_empty, "source": f"ACLED fetch error: {type(e).__name__}"}


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_all_conflict_intensities(
    days: int = 30,
    api_key: Optional[str] = None,
    email: Optional[str] = None,
) -> dict[str, dict]:
    """Fetch ACLED intensity for all tracked conflicts. Returns {conflict_id: result}."""
    return {
        cid: fetch_acled_intensity(cid, days=days, api_key=api_key, email=email)
        for cid in _ACLED_CONFLICT_MAP
    }


def acled_to_cis_dimensions(acled_result: dict, conflict_baseline: dict) -> dict:
    """
    Convert ACLED live event data into CIS dimension values that REPLACE the
    hardcoded registry values when ACLED is configured and data is available.

    This is the full live replacement (not just a nudge factor) for:
      - deadliness           ← normalised fatalities vs conflict-type baseline
      - geographic_diffusion ← event count spread (proxy for multi-region ops)
      - escalation_trend     ← direct from ACLED escalation_signal

    Parameters
    ----------
    acled_result      : dict returned by fetch_acled_intensity()
    conflict_baseline : conflict dict from CONFLICTS registry (for fallback values
                        and conflict-type calibration)

    Returns
    -------
    dict with keys matching CIS dimension names. Only contains keys that ACLED
    can confidently replace. Caller merges with hardcoded baseline using:
        dims.update(acled_to_cis_dimensions(result, conflict))
    """
    if not acled_result.get("data_available"):
        return {}

    out: dict = {}

    # ── deadliness: normalised fatalities ─────────────────────────────────────
    # Scale: 0 fatalities → 0.0, 200+ fatalities/30d → 1.0 (linear, capped)
    # 200/month ≈ Ukraine-scale intense combat; 20/month ≈ low-intensity conflict
    fat = int(acled_result.get("fatalities_nd", 0))
    out["deadliness"] = float(np.clip(fat / 200.0, 0.0, 1.0))

    # ── geographic_diffusion: event count as proxy for operational spread ─────
    # More events spread across locations → higher diffusion.
    # Scale: 0 events → 0.0, 300+ events/30d → 1.0
    events = int(acled_result.get("events_nd", 0))
    out["geographic_diffusion"] = float(np.clip(events / 300.0, 0.0, 1.0))

    # ── escalation_trend: direct from ACLED signal ────────────────────────────
    # ACLED escalation_signal is already "escalating"/"stable"/"de-escalating"
    out["escalation_trend"] = str(acled_result.get("escalation_signal", "stable"))

    return out


def acled_configured() -> bool:
    """True if ACLED credentials are set in environment."""
    return bool(os.environ.get("ACLED_API_KEY")) and bool(os.environ.get("ACLED_EMAIL"))


def acled_setup_instructions() -> str:
    """Return setup instructions for ACLED API."""
    return (
        "**ACLED Live Conflict Data Setup**\n\n"
        "1. Register at https://developer.acleddata.com/ (free for academic use)\n"
        "2. Set environment variables:\n"
        "   ```\n"
        "   export ACLED_API_KEY=your_key_here\n"
        "   export ACLED_EMAIL=your_email@example.com\n"
        "   ```\n"
        "3. Or add to `.env` file in the project root.\n\n"
        "ACLED updates daily with verified conflict event data from 1997–present. "
        "This replaces the static CONFLICTS registry for CIS computation."
    )

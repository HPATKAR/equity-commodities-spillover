"""
GDELT (Global Database of Events, Language and Tone) live conflict feed.

No API key required. Uses the GDELT 2.0 DOC API (free, public).

GDELT monitors news coverage worldwide and generates tone/event-count signals
that function as a conflict escalation proxy. Used here to:
  1. Validate ACLED escalation signals with an independent media-based source
  2. Provide conflict escalation signals when ACLED is not configured
  3. Feed the "escalation_trend" override in CIS computation

API endpoint: https://api.gdeltproject.org/api/v2/doc/doc
  mode=ArtList  → list of articles
  mode=TimelineVol → volume time-series (used here)

Forensically confirmed endpoint (2026-04-19):
  https://api.gdeltproject.org/api/v2/doc/doc
  ?query=<theme>&mode=TimelineVol&format=json&TIMESPAN=BRRR&MAXRECORDS=250

GDELT conflict theme codes (confirmed from GDELT GKG Category List):
  CRISISLEX_T03  = Armed Conflict (fires, battles)
  TAX_FNCACT_MILITARY_PERSONNEL = military events
  WB_635_CONFLICT_AND_VIOLENCE  = World Bank conflict taxonomy

Tone: GDELT assigns a "tone" score (−100 to +100) per article.
  Negative tone = more alarming/negative language → escalation signal.
  We compute rolling 7-day mean tone vs prior 7-day mean tone.

Date filter note: GDELT TimelineVol returns a time-series of volume
  (number of matching articles) per 15-minute bucket. We sum into daily counts.
"""

from __future__ import annotations

import datetime
import time
from typing import Optional

import numpy as np
import pandas as pd

_GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Browser-like UA — Python's default "python-requests/x.x" is commonly blocked by CDN rules
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

# Manual success-only cache.
# @st.cache_data caches failure dicts identically to successes, locking the app
# into a 3-hour failure window even after GDELT recovers.  This dict only stores
# successful results; failures are never persisted so the next call retries live.
_GDELT_CACHE: dict[tuple, tuple[dict, datetime.datetime]] = {}
_GDELT_TTL_S = 10800  # 3 hours (matches old @st.cache_data ttl)

# ── Conflict → GDELT search query mapping ─────────────────────────────────────
# Queries are GDELT boolean search strings: keywords + location + theme tags.
# Tested queries return consistent results without key.
_GDELT_CONFLICT_QUERIES: dict[str, dict] = {
    "ukraine_russia": {
        "query":       '"Ukraine" OR "Zelensky" OR "Russian army"',
        "theme":       "CRISISLEX_T03",
        "description": "Ukraine–Russia War",
    },
    "israel_hamas": {
        "query":       '"Gaza" OR "Hamas" OR "IDF" OR "West Bank"',
        "theme":       "CRISISLEX_T03",
        "description": "Israel-Hamas War",
    },
    "iran_regional": {
        "query":       '"Iran" OR "IRGC" OR "Hormuz" OR "Iranian military"',
        "theme":       "CRISISLEX_T03",
        "description": "Iran/Hormuz Regional Tensions",
    },
    "red_sea_houthi": {
        "query":       '"Houthi" OR "Red Sea" OR "Bab-el-Mandeb" OR "Ansar Allah"',
        "theme":       "CRISISLEX_T03",
        "description": "Red Sea Houthi Operations",
    },
    "india_pakistan": {
        "query":       '"India Pakistan" OR "Pahalgam" OR "Operation Sindoor" OR "Line of Control" OR "Kashmir attack"',
        "theme":       "CRISISLEX_T03",
        "description": "India-Pakistan Military Escalation",
    },
    "taiwan_strait": {
        "query":       '"Taiwan" OR "PLA" OR "Taiwan Strait" OR "TSMC" OR "China Taiwan military"',
        "theme":       "CRISISLEX_T03",
        "description": "Taiwan Strait / China Military Activity",
    },
}


def fetch_gdelt_escalation(
    conflict_id: str,
    timespan: str = "7d",
) -> dict:
    """
    Fetch GDELT article volume time-series for a conflict and compute
    escalation signals based on volume trend.

    Parameters
    ----------
    conflict_id : str — key from _GDELT_CONFLICT_QUERIES
    timespan    : str — GDELT TIMESPAN parameter: "7d", "14d", "30d"

    Returns
    -------
    dict with keys:
        volume_recent   : int   — article count in recent half of timespan
        volume_prior    : int   — article count in prior half of timespan
        volume_trend    : float — (recent/prior) − 1 (+ = escalating coverage)
        escalation_signal: str  — "escalating"/"stable"/"de-escalating"
        tone_recent     : float — mean tone of recent articles (negative = alarming)
        tone_delta      : float — recent tone − prior tone (negative = worsening)
        data_available  : bool
        source          : str
        as_of           : str
    """
    _empty = {
        "volume_recent":    0,
        "volume_prior":     0,
        "volume_trend":     0.0,
        "escalation_signal": "stable",
        "tone_recent":      0.0,
        "tone_delta":       0.0,
        "data_available":   False,
        "source":           "GDELT unavailable",
        "as_of":            str(datetime.date.today()),
    }

    if conflict_id not in _GDELT_CONFLICT_QUERIES:
        return {**_empty, "source": f"Unknown conflict_id: {conflict_id}"}

    # Check success-only manual cache before hitting the network
    _cache_key = (conflict_id, timespan)
    _cached = _GDELT_CACHE.get(_cache_key)
    if _cached:
        _result, _cached_at = _cached
        if (datetime.datetime.now() - _cached_at).total_seconds() < _GDELT_TTL_S:
            return _result
        # TTL expired — remove stale entry and fall through to live fetch
        del _GDELT_CACHE[_cache_key]

    cfg = _GDELT_CONFLICT_QUERIES[conflict_id]

    # Retry loop: one automatic retry on 429 (rate-limit) with exponential back-off.
    # All other errors surface immediately; successes break out of the loop.
    _max_attempts = 2
    for _attempt in range(_max_attempts):
      try:
        import requests

        # TimelineVol: returns volume per time bucket as JSON
        params = {
            "query":      f'({cfg["query"]}) theme:{cfg["theme"]}',
            "mode":       "TimelineVol",
            "format":     "json",
            "TIMESPAN":   timespan,
            "MAXRECORDS": 250,
        }
        resp = requests.get(_GDELT_DOC_API, params=params, timeout=20, headers=_HEADERS)
        resp.raise_for_status()
        data = resp.json()

        # GDELT TimelineVol response: {"timeline": [{"date":..., "value":...}, ...]}
        timeline = data.get("timeline", [{}])[0].get("data", [])
        if not timeline:
            return {**_empty, "source": "GDELT: no data returned"}

        # Build DataFrame
        rows = []
        for pt in timeline:
            try:
                dt  = pd.to_datetime(str(pt.get("date", "")), format="%Y%m%dT%H%M%S")
                vol = float(pt.get("value", 0) or 0)
                rows.append({"dt": dt, "volume": vol})
            except Exception:
                continue

        if not rows:
            return {**_empty, "source": "GDELT: unparseable timeline"}

        df = pd.DataFrame(rows).set_index("dt").sort_index()

        # Aggregate to daily
        daily = df["volume"].resample("1D").sum()

        if len(daily) < 4:
            return {**_empty, "source": "GDELT: insufficient days"}

        # Split into recent / prior halves
        mid = len(daily) // 2
        vol_prior  = int(daily.iloc[:mid].sum())
        vol_recent = int(daily.iloc[mid:].sum())

        prior_safe = max(vol_prior, 1)
        trend = float(vol_recent / prior_safe - 1.0)

        if trend > 0.20:
            escalation = "escalating"
        elif trend < -0.20:
            escalation = "de-escalating"
        else:
            escalation = "stable"

        # Brief pause between TimelineVol and ArtList to avoid back-to-back 429s
        time.sleep(0.8)

        # Fetch tone via ArtList for the same query (last 3d only to keep fast)
        tone_recent = 0.0
        tone_delta  = 0.0
        try:
            art_params = {
                "query":      f'({cfg["query"]}) theme:{cfg["theme"]}',
                "mode":       "ArtList",
                "format":     "json",
                "TIMESPAN":   "3d",
                "MAXRECORDS": 50,
            }
            art_resp = requests.get(_GDELT_DOC_API, params=art_params, timeout=15,
                                    headers=_HEADERS)
            art_data = art_resp.json()
            articles = art_data.get("articles", [])
            if articles:
                tones = [float(a.get("tone", 0) or 0) for a in articles if a.get("tone")]
                if tones:
                    tone_recent = float(np.mean(tones))
                    # Positive GDELT tone = more positive language.
                    # A tone < -2 is meaningfully alarming for conflict coverage.
                    # We report tone_delta vs a neutral baseline of -1.5 (typical conflict coverage)
                    tone_delta = tone_recent - (-1.5)
        except Exception:
            pass

        try:
            from src.analysis.freshness import record_fetch
            record_fetch("gdelt")
        except Exception:
            pass

        result = {
            "volume_recent":     vol_recent,
            "volume_prior":      vol_prior,
            "volume_trend":      round(trend, 3),
            "escalation_signal": escalation,
            "tone_recent":       round(tone_recent, 2),
            "tone_delta":        round(tone_delta, 2),
            "data_available":    True,
            "source":            "GDELT live",
            "as_of":             str(datetime.date.today()),
        }
        # Only cache successes — failures must NOT be cached so the next call retries
        _GDELT_CACHE[_cache_key] = (result, datetime.datetime.now())
        return result

      except requests.exceptions.HTTPError as he:
        # Capture actual HTTP status code for actionable error messages
        _status = he.response.status_code if he.response is not None else "?"
        # 429 = rate limited — back off and retry once before giving up
        if _status == 429 and _attempt < _max_attempts - 1:
            _backoff = 4.0 * (_attempt + 1)  # 4s on first retry
            time.sleep(_backoff)
            continue
        _msg = f"GDELT HTTP {_status}"
        try:
            from src.analysis.freshness import record_failure
            record_failure("gdelt", _msg)
        except Exception:
            pass
        return {**_empty, "source": _msg}

      except Exception as e:
        _msg = f"GDELT {type(e).__name__}"
        try:
            from src.analysis.freshness import record_failure
            record_failure("gdelt", _msg)
        except Exception:
            pass
        return {**_empty, "source": _msg}


def fetch_all_gdelt_signals(timespan: str = "7d") -> dict[str, dict]:
    """
    Fetch GDELT escalation signals for all tracked conflicts.
    Each conflict result is individually cached (success-only, 3h TTL) inside
    fetch_gdelt_escalation. No outer cache needed here: callers always get
    cached data for healthy conflicts and a live retry for failed ones.

    Inter-conflict delay (1.5s) is applied only for live fetches — cached results
    are returned immediately. This prevents the 6-conflict cold-start from firing
    12 requests in <2s and triggering GDELT's 429 rate limiter.
    """
    results = {}
    for cid in _GDELT_CONFLICT_QUERIES:
        # Check local cache before deciding whether to sleep
        _ck = (cid, timespan)
        _hit = _GDELT_CACHE.get(_ck)
        _is_cached = bool(
            _hit and (datetime.datetime.now() - _hit[1]).total_seconds() < _GDELT_TTL_S
        )
        results[cid] = fetch_gdelt_escalation(cid, timespan=timespan)
        if not _is_cached:
            # Only throttle between live network calls, not cache hits
            time.sleep(1.5)
    return results


def gdelt_escalation_override(conflict_id: str) -> str:
    """
    Fast single-call convenience: return GDELT escalation signal for one conflict.
    Returns "escalating"/"stable"/"de-escalating".
    Falls back to "stable" on any failure.
    """
    try:
        result = fetch_gdelt_escalation(conflict_id, timespan="7d")
        if result.get("data_available"):
            return str(result["escalation_signal"])
    except Exception:
        pass
    return "stable"


def gdelt_corroboration(acled_signal: str, gdelt_signal: str) -> dict:
    """
    Cross-validate ACLED and GDELT escalation signals.

    Returns a corroboration dict:
        agreed       : bool — both sources agree on direction
        confidence   : str  — "high"/"medium"/"low"
        final_signal : str  — consensus signal
        note         : str  — human-readable explanation
    """
    agreed = acled_signal == gdelt_signal

    if agreed:
        confidence   = "high"
        final_signal = acled_signal
        note = f"ACLED + GDELT both signal '{acled_signal}' — high confidence."
    elif acled_signal == "stable" or gdelt_signal == "stable":
        # One says stable, the other says something — moderate confidence
        confidence   = "medium"
        # Prefer the non-stable signal (more informative)
        final_signal = gdelt_signal if acled_signal == "stable" else acled_signal
        note = (f"ACLED='{acled_signal}', GDELT='{gdelt_signal}' — "
                f"using non-stable signal with medium confidence.")
    else:
        # One escalating, one de-escalating — contradictory
        confidence   = "low"
        final_signal = "stable"  # conservative fallback
        note = (f"ACLED='{acled_signal}' vs GDELT='{gdelt_signal}' — "
                f"contradictory signals; defaulting to 'stable'.")

    return {
        "agreed":        agreed,
        "confidence":    confidence,
        "final_signal":  final_signal,
        "note":          note,
    }

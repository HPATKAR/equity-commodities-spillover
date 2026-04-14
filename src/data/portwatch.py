"""
IMF PortWatch chokepoint transit data loader.

Data source: IMF PortWatch (portwatch.imf.org) — ArcGIS Feature Service.
No API key required for public access; optional token raises rate limits.

Usage:
    from src.data.portwatch import load_hormuz_tankers, brent_sensitivity_table

secrets.toml keys (optional):
    [keys]
    portwatch_endpoint = "https://..."   # override the default ArcGIS endpoint
    portwatch_token    = "..."           # ArcGIS token for higher rate limits

Chokepoint portid mapping (confirmed/estimated):
    chokepoint1 — Suez Canal  (confirmed from data)
    Others auto-discovered via _discover_portids() — do not hardcode.
"""

from __future__ import annotations

import datetime
import streamlit as st
import pandas as pd

# ── ArcGIS endpoint ────────────────────────────────────────────────────────────
# Find the real URL: portwatch.imf.org → DevTools → Network → filter "query"
# → copy the FeatureServer URL from the request.
_DEFAULT_ENDPOINT = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
    "IMF_PortWatch_Chokepoint_Transits_Daily_v2/FeatureServer/0/query"
)

_OIL_TANKER_FRAC = 0.60   # crude + product carriers ≈ 60% of n_tanker


def _get_endpoint_and_token() -> tuple[str, str]:
    try:
        keys  = st.secrets.get("keys", {})
        url   = keys.get("portwatch_endpoint", _DEFAULT_ENDPOINT) or _DEFAULT_ENDPOINT
        token = keys.get("portwatch_token",    "") or ""
    except Exception:
        url, token = _DEFAULT_ENDPOINT, ""
    return url, token


@st.cache_data(ttl=86400, show_spinner=False)
def _discover_portids() -> dict[str, str]:
    """
    Query the API without a portid filter to discover portname→portid mappings.
    Cached 24 h. Returns {} on failure.
    """
    import requests
    url, token = _get_endpoint_and_token()
    params: dict = {
        "where":               "1=1",
        "outFields":           "portid,portname",
        "returnDistinctValues": True,
        "resultRecordCount":   50,
        "f":                   "json",
    }
    if token:
        params["token"] = token
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        mapping = {}
        for feat in data.get("features", []):
            a = feat.get("attributes", {})
            name = (a.get("portname") or "").strip()
            pid  = (a.get("portid")   or "").strip()
            if name and pid:
                mapping[name.lower()] = pid
        return mapping
    except Exception:
        return {}


def _hormuz_portid() -> str:
    """Return the live portid for Strait of Hormuz, auto-discovered from the API."""
    mapping = _discover_portids()
    for name, pid in mapping.items():
        if "hormuz" in name:
            return pid
    # fallback: try all chokepoint IDs 1-10; caller will get empty df and try next
    return "chokepoint3"


@st.cache_data(ttl=3600, show_spinner=False)
def load_hormuz_tankers(days: int = 365) -> pd.DataFrame:
    """
    Fetch daily tanker transit counts for Strait of Hormuz from IMF PortWatch.

    Auto-discovers the correct portid via _discover_portids().
    If discovery fails, tries chokepoint2 through chokepoint7 sequentially.

    Returns a DataFrame with columns:
        date, n_tanker, oil_tanker, n_total, capacity_tanker
    Returns empty DataFrame on total failure (caller must handle gracefully).
    """
    import requests

    url, token = _get_endpoint_and_token()
    cutoff_ms  = int(
        (datetime.datetime.utcnow() - datetime.timedelta(days=days)).timestamp() * 1000
    )

    def _fetch(portid: str) -> pd.DataFrame:
        params: dict = {
            "where":           f"portid='{portid}' AND date>={cutoff_ms}",
            "outFields":       "date,n_tanker,n_total,capacity_tanker",
            "orderByFields":   "date ASC",
            "resultRecordCount": 2000,
            "f":               "json",
        }
        if token:
            params["token"] = token
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return pd.DataFrame()

        features = data.get("features", [])
        if not features:
            return pd.DataFrame()

        rows = [f["attributes"] for f in features]
        df = pd.DataFrame(rows)
        if "date" not in df.columns or "n_tanker" not in df.columns:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True).dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)
        df["oil_tanker"] = (df["n_tanker"] * _OIL_TANKER_FRAC).round().astype(int)
        return df[["date", "n_tanker", "oil_tanker", "n_total", "capacity_tanker"]]

    # 1. Try auto-discovered portid
    primary_pid = _hormuz_portid()
    df = _fetch(primary_pid)
    if not df.empty:
        return df

    # 2. Discovery failed or returned wrong ID — try all candidates
    for pid in ["chokepoint2", "chokepoint3", "chokepoint4", "chokepoint5", "chokepoint6"]:
        if pid == primary_pid:
            continue
        df = _fetch(pid)
        if not df.empty:
            return df

    return pd.DataFrame()


def brent_sensitivity_table(base_price: float) -> pd.DataFrame:
    """
    Brent crude price disruption sensitivity table.

    Formula: new_price = base × (1 + ε × (−disruption_fraction))

    Elasticity sign convention:
        Positive (0.004, 0.014): empirical OLS — counterintuitive, captures
          demand co-movement rather than supply shock channel.
        Negative (−0.25, −0.35, −0.50): structural EIA/IEA forecaster range
          for supply-shock scenario analysis.
    """
    disruptions  = [0.10, 0.25, 0.50, 0.75, 1.00]
    elasticities = [0.004, 0.014, -0.25, -0.35, -0.50]

    rows = {}
    for eps in elasticities:
        rows[eps] = {
            f"-{int(d*100)}%": round(base_price * (1 + eps * (-d)), 2)
            for d in disruptions
        }

    df = pd.DataFrame(rows).T
    df.index.name = "Elasticity"
    return df

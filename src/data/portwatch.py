"""
IMF PortWatch chokepoint transit data loader.

Data source: IMF PortWatch (portwatch.imf.org) — ArcGIS Feature Service.
Public endpoint, no API key required.

Forensically confirmed endpoints (2026-04-14):
  org:     weJ1QsnbMYJlCHdG
  service: Daily_Chokepoints_Data / FeatureServer / layer 0
  full:    https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/
             Daily_Chokepoints_Data/FeatureServer/0/query

Confirmed portid mapping (from live _discover_portids() call):
  chokepoint1  — Suez Canal
  chokepoint2  — Panama Canal
  chokepoint3  — Bosporus Strait
  chokepoint4  — Bab el-Mandeb Strait
  chokepoint5  — Malacca Strait
  chokepoint6  — Strait of Hormuz   ← used here
  chokepoint7  — Cape of Good Hope
  chokepoint8  — Gibraltar Strait
  chokepoint9  — Dover Strait
  chokepoint10 — Oresund Strait
  ... (14 more global chokepoints)

Date filter note: the 'date' field rejects epoch-ms WHERE clauses (400 error).
  Use year/month/day integer fields instead.

secrets.toml (optional override):
  [keys]
  portwatch_endpoint = "https://..."
  portwatch_token    = ""
"""

from __future__ import annotations

import datetime
import streamlit as st
import pandas as pd

_DEFAULT_ENDPOINT = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Chokepoints_Data/FeatureServer/0/query"
)
_HORMUZ_PORTID   = "chokepoint6"   # forensically confirmed
_OIL_TANKER_FRAC = 0.60            # crude+product ≈ 60% of n_tanker


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
    """Map portname.lower() → portid. Cached 24 h. Returns {} on failure."""
    import requests
    url, token = _get_endpoint_and_token()
    params: dict = {
        "where":                "1=1",
        "outFields":            "portid,portname",
        "returnDistinctValues": True,
        "resultRecordCount":    50,
        "f":                    "json",
    }
    if token:
        params["token"] = token
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        mapping = {}
        for feat in resp.json().get("features", []):
            a = feat.get("attributes", {})
            name = (a.get("portname") or "").strip().lower()
            pid  = (a.get("portid")   or "").strip()
            if name and pid:
                mapping[name] = pid
        return mapping
    except Exception:
        return {}


def _hormuz_portid() -> str:
    """Return live portid for Strait of Hormuz from discovery, or confirmed default."""
    mapping = _discover_portids()
    for name, pid in mapping.items():
        if "hormuz" in name:
            return pid
    return _HORMUZ_PORTID


@st.cache_data(ttl=3600, show_spinner=False)
def load_hormuz_tankers(days: int = 365) -> pd.DataFrame:
    """
    Fetch daily tanker transit counts for Strait of Hormuz from IMF PortWatch.

    Uses year/month/day integer fields for date filtering (epoch-ms WHERE fails on this service).

    Returns DataFrame columns: date, n_tanker, oil_tanker, n_total, capacity_tanker
    Returns empty DataFrame on failure (caller must degrade gracefully).
    """
    import requests

    url, token   = _get_endpoint_and_token()
    portid       = _hormuz_portid()
    cutoff       = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    cutoff_year  = cutoff.year
    cutoff_month = cutoff.month

    # Date filter via year/month integers — epoch-ms on 'date' field returns 400
    where = (
        f"portid='{portid}' AND ("
        f"year>{cutoff_year} OR "
        f"(year={cutoff_year} AND month>={cutoff_month})"
        f")"
    )

    params: dict = {
        "where":             where,
        "outFields":         "date,year,month,day,n_tanker,n_total,capacity_tanker",
        "orderByFields":     "date ASC",
        "resultRecordCount": 2000,
        "f":                 "json",
    }
    if token:
        params["token"] = token

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return pd.DataFrame()

    if "error" in data:
        return pd.DataFrame()

    features = data.get("features", [])
    if not features:
        return pd.DataFrame()

    rows = [f["attributes"] for f in features]
    df   = pd.DataFrame(rows)

    if "date" not in df.columns or "n_tanker" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    df["oil_tanker"] = (df["n_tanker"] * _OIL_TANKER_FRAC).round().astype(int)

    return df[["date", "n_tanker", "oil_tanker", "n_total", "capacity_tanker"]]


def brent_sensitivity_table(base_price: float) -> pd.DataFrame:
    """
    Brent crude disruption sensitivity.
    Formula: new_price = base × (1 + ε × (−disruption_fraction))

    Positive ε (0.004, 0.014): empirical OLS — captures demand co-movement,
      not supply shock. Price falls with disruption — counterintuitive but correct
      for the regression context (COVID/demand collapses dominate the sample).
    Negative ε (−0.25/−0.35/−0.50): structural EIA/IEA range for supply shocks.
    """
    disruptions  = [0.10, 0.25, 0.50, 0.75, 1.00]
    elasticities = [0.004, 0.014, -0.25, -0.35, -0.50]

    rows = {
        eps: {f"-{int(d*100)}%": round(base_price * (1 + eps * (-d)), 2) for d in disruptions}
        for eps in elasticities
    }
    df = pd.DataFrame(rows).T
    df.index.name = "Elasticity"
    return df

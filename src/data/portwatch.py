"""
IMF PortWatch chokepoint transit data loader.

Data source: IMF PortWatch (portwatch.imf.org) — ArcGIS Feature Service.
No API key required for public access; optional token raises rate limits.

Usage:
    from src.data.portwatch import load_hormuz_tankers, brent_sensitivity_table

Chokepoint portid mapping (confirmed/estimated from PortWatch data):
    chokepoint1 — Suez Canal          (confirmed from data)
    chokepoint2 — Panama Canal
    chokepoint3 — Strait of Hormuz    (used below as default)
    chokepoint4 — Strait of Malacca
    chokepoint5 — Turkish Straits
    chokepoint6 — Bab-el-Mandeb
    chokepoint7 — Danish Straits

secrets.toml keys (optional):
    [keys]
    portwatch_endpoint = "https://..."   # override the default ArcGIS endpoint
    portwatch_token    = "..."           # ArcGIS token for higher rate limits
"""

from __future__ import annotations

import datetime
import streamlit as st
import pandas as pd

# ── ArcGIS endpoint ────────────────────────────────────────────────────────────
# The IMF PortWatch portal serves data via ArcGIS Online.
# To find the exact URL: open portwatch.imf.org → browser DevTools → Network tab
# → filter for "query" → copy the FeatureServer URL.
_DEFAULT_ENDPOINT = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
    "IMF_PortWatch_Chokepoint_Transits_Daily_v2/FeatureServer/0/query"
)

_HORMUZ_PORTID   = "chokepoint3"
_OIL_TANKER_FRAC = 0.60   # ~60% of n_tanker are crude/product carriers (friend's proxy)


def _get_endpoint_and_token() -> tuple[str, str]:
    try:
        keys    = st.secrets.get("keys", {})
        url     = keys.get("portwatch_endpoint", _DEFAULT_ENDPOINT) or _DEFAULT_ENDPOINT
        token   = keys.get("portwatch_token",    "") or ""
    except Exception:
        url, token = _DEFAULT_ENDPOINT, ""
    return url, token


@st.cache_data(ttl=3600, show_spinner=False)
def load_hormuz_tankers(days: int = 180, portid: str = _HORMUZ_PORTID) -> pd.DataFrame:
    """
    Fetch daily tanker transit counts for the Strait of Hormuz from IMF PortWatch.

    Returns a DataFrame with columns:
        date         — datetime64[ns]
        n_tanker     — total tanker count (crude + product + LNG + LPG)
        n_total      — all vessel types
        oil_tanker   — estimated oil tankers (n_tanker × 0.60)
        capacity_tanker — DWT capacity of tankers

    Returns an empty DataFrame on any failure (caller must handle gracefully).
    """
    import requests

    url, token = _get_endpoint_and_token()

    # Date filter: last `days` days
    cutoff_ms = int(
        (datetime.datetime.utcnow() - datetime.timedelta(days=days)).timestamp() * 1000
    )

    params: dict = {
        "where":      f"portid='{portid}' AND date>={cutoff_ms}",
        "outFields":  "date,n_tanker,n_total,capacity_tanker",
        "orderByFields": "date ASC",
        "resultRecordCount": 2000,
        "f":          "json",
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

    # Convert Unix ms timestamp → date
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)

    # Derived column: estimated oil tankers
    df["oil_tanker"] = (df["n_tanker"] * _OIL_TANKER_FRAC).round().astype(int)

    return df[["date", "n_tanker", "oil_tanker", "n_total", "capacity_tanker"]]


def brent_sensitivity_table(base_price: float) -> pd.DataFrame:
    """
    Compute the Brent crude price disruption sensitivity table.

    Formula (from IMF PortWatch / FRED regression):
        new_price = base × (1 + elasticity × (−disruption_fraction))

    Where disruption_fraction is the share of supply lost [0, 1].

    Elasticity sign convention:
        Positive (0.004, 0.014): empirical OLS — counterintuitive because the
          regression captures demand-driven co-movement, not supply shocks.
        Negative (−0.25, −0.35, −0.50): structural supply-demand elasticities
          used by EIA, IEA, and energy-economics researchers for scenario analysis.

    Returns a DataFrame indexed by elasticity, with disruption % as columns.
    """
    disruptions = [0.10, 0.25, 0.50, 0.75, 1.00]
    elasticities = [0.004, 0.014, -0.25, -0.35, -0.50]

    rows = {}
    for eps in elasticities:
        row = {}
        for d in disruptions:
            row[f"-{int(d*100)}%"] = round(base_price * (1 + eps * (-d)), 2)
        rows[eps] = row

    df = pd.DataFrame(rows).T
    df.index.name = "Elasticity"
    return df

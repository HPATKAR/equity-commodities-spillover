"""
CFTC Commitments of Traders (COT) — disaggregated futures loader.

Downloads annual ZIP files from CFTC's public repository.
Falls back to the CFTC Socrata public API when ZIP downloads are blocked (cloud deployments).
Maps our commodity names to CFTC report names.
Computes net speculative positioning and normalises to % of open interest.
"""

from __future__ import annotations

import io
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

# ── Market name mapping ─────────────────────────────────────────────────────
# Our name → substring to match in CFTC "Market_and_Exchange_Names" column

COT_MARKETS: dict[str, str] = {
    "WTI Crude Oil": "CRUDE OIL, LIGHT SWEET",
    "Natural Gas":   "NATURAL GAS - NEW YORK",
    "Gold":          "GOLD - COMMODITY EXCHANGE",
    "Silver":        "SILVER - COMMODITY EXCHANGE",
    "Copper":        "COPPER-GRADE #1",
    "Wheat":         "WHEAT-SRW",
    "Corn":          "CORN - CHICAGO BOARD",
    "Soybeans":      "SOYBEANS - CHICAGO BOARD",
}

_BASE_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
_ALT_URL  = "https://www.cftc.gov/sites/default/files/files/dea/history/fut_disagg_txt_{year}.zip"

# CFTC public Socrata API — works from cloud IPs where direct ZIP downloads are blocked
_SOCRATA_URL = (
    "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
    "?$where=report_date_as_yyyy_mm_dd >= '{since}'&$limit=50000&$order=report_date_as_yyyy_mm_dd ASC"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/zip, application/octet-stream, */*",
}


def _download_cot_year(session, year: int) -> "pd.DataFrame | None":
    """Try both ZIP URL patterns for a given year; return parsed DataFrame or None."""
    for url_tmpl in (_BASE_URL, _ALT_URL):
        url = url_tmpl.format(year=year)
        try:
            resp = session.get(url, timeout=60, headers=_HEADERS)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                names = zf.namelist()
                data_file = next(
                    (n for n in names if n.lower().endswith((".txt", ".csv"))),
                    names[0],
                )
                with zf.open(data_file) as f:
                    return pd.read_csv(f, low_memory=False)
        except Exception:
            continue
    return None


def _load_via_socrata(session, years: int = 3) -> "pd.DataFrame | None":
    """
    Fallback: fetch COT data via CFTC's public Socrata API.
    Works from cloud IPs (Hugging Face, AWS, GCP) where direct ZIP downloads are blocked.
    The disaggregated dataset uses 'm_money' (Managed Money) as the speculative category.
    Returns a DataFrame with columns normalised to match the rest of the pipeline.
    """
    since = f"{date.today().year - years}-01-01"
    url = _SOCRATA_URL.format(since=since)
    try:
        resp = session.get(
            url, timeout=90,
            headers={"Accept": "application/json", "User-Agent": _HEADERS["User-Agent"]},
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        # In the disaggregated report, Managed Money = speculative non-commercial traders
        # Socrata column names → normalised names used by the rest of the pipeline
        rename = {
            "report_date_as_yyyy_mm_dd":  "Report_Date_as_YYYY-MM-DD",
            "market_and_exchange_names":  "Market_and_Exchange_Names",
            "m_money_positions_long_all": "NonComm_Positions_Long_All",
            "m_money_positions_short_all":"NonComm_Positions_Short_All",
            "open_interest_all":          "Open_Interest_All",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        # Keep only futures-only rows to match ZIP behaviour
        if "futonly_or_combined" in df.columns:
            df = df[df["futonly_or_combined"] == "FutOnly"]
        return df
    except Exception:
        return None


# ── Downloader ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def load_cot_data(years: int = 3) -> pd.DataFrame:
    """
    Download CFTC disaggregated COT data for the last `years` annual files.
    Tries direct ZIP downloads first; falls back to Socrata API for cloud deployments.
    Returns tidy DataFrame:
      date | market | net_speculative | net_spec_pct | oi
    """
    import requests

    current_year = date.today().year
    raw_frames: list[pd.DataFrame] = []

    session = requests.Session()
    session.headers.update(_HEADERS)

    # Primary: try ZIP downloads
    for yr in range(current_year - years + 1, current_year + 1):
        df = _download_cot_year(session, yr)
        if df is not None:
            raw_frames.append(df)

    # Fallback: Socrata API (works from cloud IPs)
    if not raw_frames:
        df = _load_via_socrata(session, years=years)
        if df is not None:
            raw_frames.append(df)

    if not raw_frames:
        return pd.DataFrame()

    raw = pd.concat(raw_frames, ignore_index=True)

    # ── Parse & compute ──────────────────────────────────────────────────
    date_col = next(
        (c for c in raw.columns if "date" in c.lower() and "report" in c.lower()),
        None,
    )
    if date_col is None:
        return pd.DataFrame()

    raw["date"] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=["date"])

    long_col  = "NonComm_Positions_Long_All"
    short_col = "NonComm_Positions_Short_All"
    oi_col    = "Open_Interest_All"
    name_col  = "Market_and_Exchange_Names"

    for col in [long_col, short_col, oi_col, name_col]:
        if col not in raw.columns:
            return pd.DataFrame()

    raw["net_speculative"] = (
        pd.to_numeric(raw[long_col], errors="coerce") -
        pd.to_numeric(raw[short_col], errors="coerce")
    )
    raw["oi"] = pd.to_numeric(raw[oi_col], errors="coerce")
    raw["net_spec_pct"] = (raw["net_speculative"] / raw["oi"].replace(0, np.nan) * 100).round(2)

    # ── Filter to our markets ─────────────────────────────────────────
    result_rows: list[pd.DataFrame] = []
    for our_name, pattern in COT_MARKETS.items():
        mask = raw[name_col].str.contains(pattern, case=False, na=False)
        subset = raw.loc[mask, ["date", "net_speculative", "net_spec_pct", "oi"]].copy()
        subset["market"] = our_name
        result_rows.append(subset)

    if not result_rows:
        return pd.DataFrame()

    out = pd.concat(result_rows, ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    return out


# ── Chart ───────────────────────────────────────────────────────────────────

def plot_cot_overlay(
    cot_df: pd.DataFrame,
    market: str,
    price_series: pd.Series | None = None,
    height: int = 420,
) -> go.Figure:
    """
    Dual-axis chart:
      Left  — commodity price (line, if provided)
      Right — net speculative positioning as bars + % OI line

    Contrarian signal bands at ±25% net_spec_pct.
    """
    subset = cot_df[cot_df["market"] == market].sort_values("date").dropna(subset=["net_spec_pct"])
    if subset.empty:
        return go.Figure()

    fig = go.Figure()

    # ── Extreme positioning bands ─────────────────────────────────────
    fig.add_hrect(y0=25,  y1=100, fillcolor="#ffebee", opacity=0.35,
                  layer="below", line_width=0)
    fig.add_hrect(y0=-100, y1=-25, fillcolor="#e8f5e9", opacity=0.35,
                  layer="below", line_width=0)
    fig.add_hline(y=0, line=dict(color="#ABABAB", width=1, dash="dot"))
    fig.add_hline(y=25,  line=dict(color="#c0392b", width=0.8, dash="dot"),
                  annotation_text="Crowded Long", annotation_font_size=8,
                  annotation_font_color="#c0392b")
    fig.add_hline(y=-25, line=dict(color="#2e7d32", width=0.8, dash="dot"),
                  annotation_text="Crowded Short", annotation_font_size=8,
                  annotation_font_color="#2e7d32")

    # ── Net speculative % bars ────────────────────────────────────────
    bar_colors = [
        "#c0392b" if v > 0 else "#2e7d32"
        for v in subset["net_spec_pct"]
    ]
    fig.add_trace(go.Bar(
        x=subset["date"],
        y=subset["net_spec_pct"],
        name="Net Spec % OI",
        marker_color=bar_colors,
        marker_line_width=0,
        opacity=0.7,
        hovertemplate="%{x|%d %b %Y}: %{y:+.1f}%<extra>Net Spec % OI</extra>",
    ))

    # ── Price overlay (secondary y-axis) ──────────────────────────────
    if price_series is not None and not price_series.empty:
        fig.add_trace(go.Scatter(
            x=price_series.index,
            y=price_series.values,
            name=f"{market} Price",
            line=dict(color="#CFB991", width=1.8),
            yaxis="y2",
            hovertemplate="%{x|%d %b %Y}: %{y:.2f}<extra>Price</extra>",
        ))

    fig.update_layout(
        template="purdue",
        height=height,
        barmode="relative",
        title=dict(
            text=f"CFTC COT: {market} - Net Speculative Positioning (% Open Interest)",
            font=dict(size=11),
        ),
        yaxis=dict(
            title="Net Spec % OI",
            ticksuffix="%",
            zeroline=False,
        ),
        yaxis2=dict(
            title="Price",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        ),
        legend=dict(orientation="h", y=1.08),
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.04),
            type="date",
        ),
        margin=dict(l=50, r=60, t=55, b=40),
    )
    return fig


def cot_extremes_table(cot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table: for each market, show latest net_spec_pct,
    its percentile vs history, and a contrarian signal.
    """
    rows = []
    for market in cot_df["market"].unique():
        s = cot_df[cot_df["market"] == market].sort_values("date")
        if s.empty:
            continue
        latest = float(s["net_spec_pct"].iloc[-1])
        pct    = float((s["net_spec_pct"] < latest).mean() * 100)
        if pct > 85:   signal, sig_col = "Crowded Long - Contrarian Sell",  "#c0392b"
        elif pct < 15: signal, sig_col = "Crowded Short - Contrarian Buy",   "#2e7d32"
        else:          signal, sig_col = "Neutral",                             "#555960"
        rows.append({
            "Commodity":        market,
            "Net Spec % OI":    round(latest, 1),
            "Hist. Percentile": round(pct, 0),
            "Signal":           signal,
        })
    return pd.DataFrame(rows).sort_values("Hist. Percentile", ascending=False)

"""
War Impact Map - Geopolitical Equity Risk
Choropleth map (flat or 3-D globe) showing equity-market exposure to active wars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

from src.data.config import GEOPOLITICAL_EVENTS, PALETTE
from src.data.loader import load_returns
from src.ui.shared import (
    _chart, _page_intro, _section_note,
    _page_conclusion, _page_footer,
)


# ── Full world ISO-3 list (unscored countries get 0 → cream, choropleth covers all land) ──

_ALL_COUNTRIES: list[str] = [
    "AFG","AGO","ALB","AND","ARE","ARG","ARM","AUS","AUT","AZE",
    "BDI","BEL","BEN","BFA","BGD","BGR","BHR","BHS","BIH","BLR",
    "BLZ","BOL","BRA","BRN","BTN","BWA","CAF","CAN","CHE","CHL",
    "CHN","CIV","CMR","COD","COG","COL","CRI","CUB","CYP","CZE",
    "DEU","DJI","DNK","DOM","DZA","ECU","EGY","ERI","ESP","EST",
    "ETH","FIN","FJI","FRA","GAB","GBR","GEO","GHA","GIN","GNB",
    "GNQ","GRC","GTM","GUY","HND","HRV","HTI","HUN","IDN","IND",
    "IRL","IRN","IRQ","ISL","ISR","ITA","JAM","JOR","JPN","KAZ",
    "KEN","KGZ","KHM","KOR","KWT","LAO","LBN","LBR","LBY","LKA",
    "LSO","LTU","LUX","LVA","MAR","MDA","MDG","MEX","MKD","MLI",
    "MLT","MMR","MNG","MOZ","MRT","MWI","MYS","NAM","NER","NGA",
    "NIC","NLD","NOR","NPL","NZL","OMN","PAK","PAN","PER","PHL",
    "POL","PRT","PRY","PSE","QAT","ROU","RUS","RWA","SAU","SDN",
    "SEN","SGP","SLE","SLV","SOM","SRB","SSD","SWE","SWZ","SYR",
    "TCD","TGO","THA","TJK","TKM","TTO","TUN","TUR","TWN","TZA",
    "UGA","UKR","URY","USA","UZB","VEN","VNM","YEM","ZAF","ZMB","ZWE",
    "SVK","SVN","MDA","GEO","ARM","AZE","BLR","LTU","LVA","EST",
    "HRV","SVN","SRB","BGR","ROU","HUN","CZE","POL","LUX","IRL",
    "BEL","NLD","DNK","SWE","FIN","NOR","CHE","AUT","PRT","GRC","CYP",
]


# ── Country names ─────────────────────────────────────────────────────────────

_NAMES: dict[str, str] = {
    "USA": "United States",   "CAN": "Canada",          "MEX": "Mexico",
    "BRA": "Brazil",          "ARG": "Argentina",       "CHL": "Chile",
    "COL": "Colombia",        "PER": "Peru",            "VEN": "Venezuela",
    "GBR": "United Kingdom",  "DEU": "Germany",         "FRA": "France",
    "ITA": "Italy",           "ESP": "Spain",           "PRT": "Portugal",
    "NLD": "Netherlands",     "BEL": "Belgium",         "AUT": "Austria",
    "CHE": "Switzerland",     "SWE": "Sweden",          "NOR": "Norway",
    "DNK": "Denmark",         "FIN": "Finland",         "IRL": "Ireland",
    "LUX": "Luxembourg",      "GRC": "Greece",          "CYP": "Cyprus",
    "POL": "Poland",          "CZE": "Czech Republic",  "SVK": "Slovakia",
    "HUN": "Hungary",         "ROU": "Romania",         "BGR": "Bulgaria",
    "HRV": "Croatia",         "SVN": "Slovenia",        "SRB": "Serbia",
    "LTU": "Lithuania",       "LVA": "Latvia",          "EST": "Estonia",
    "UKR": "Ukraine",         "RUS": "Russia",          "BLR": "Belarus",
    "MDA": "Moldova",         "TUR": "Turkey",          "GEO": "Georgia",
    "ARM": "Armenia",         "AZE": "Azerbaijan",
    "JPN": "Japan",           "KOR": "South Korea",     "CHN": "China",
    "HKG": "Hong Kong",       "TWN": "Taiwan",          "SGP": "Singapore",
    "IND": "India",           "PAK": "Pakistan",        "BGD": "Bangladesh",
    "VNM": "Vietnam",         "THA": "Thailand",        "MYS": "Malaysia",
    "IDN": "Indonesia",       "PHL": "Philippines",     "AUS": "Australia",
    "NZL": "New Zealand",
    "ZAF": "South Africa",    "NGA": "Nigeria",         "ETH": "Ethiopia",
    "KEN": "Kenya",           "EGY": "Egypt",           "MAR": "Morocco",
    "DZA": "Algeria",         "TUN": "Tunisia",         "LBY": "Libya",
    "SDN": "Sudan",
    "SAU": "Saudi Arabia",    "IRN": "Iran",            "IRQ": "Iraq",
    "SYR": "Syria",           "JOR": "Jordan",          "LBN": "Lebanon",
    "ISR": "Israel",          "PSE": "Palestine",       "YEM": "Yemen",
    "ARE": "UAE",             "QAT": "Qatar",           "KWT": "Kuwait",
    "OMN": "Oman",            "BHR": "Bahrain",
}

# ISO-3 → tracked equity indices
_COUNTRY_INDICES: dict[str, list[str]] = {
    "USA": ["S&P 500", "Nasdaq 100", "DJIA", "Russell 2000"],
    "GBR": ["FTSE 100"],
    "DEU": ["DAX"],
    "FRA": ["CAC 40"],
    "JPN": ["Nikkei 225", "TOPIX"],
    "HKG": ["Hang Seng"],
    "CHN": ["Shanghai Comp", "CSI 300"],
    "IND": ["Sensex", "Nifty 50"],
}


# ── War impact scores (0–100) ─────────────────────────────────────────────────

_WAR_DATA: list[dict] = [
    {
        "label":       "Ukraine War",
        "event_label": "Ukraine War",
        "color":       "#e74c3c",
        "scores": {
            "UKR": 100, "RUS": 88,  "BLR": 84,  "MDA": 78,
            "POL": 78,  "FIN": 72,  "LTU": 70,  "LVA": 70,  "EST": 70,
            "DEU": 74,  "ROU": 68,  "CZE": 65,  "SVK": 64,  "HUN": 62,
            "BGR": 60,  "SRB": 52,  "HRV": 55,  "SVN": 52,  "GRC": 55,
            "ITA": 62,  "FRA": 60,  "NLD": 58,  "BEL": 55,  "AUT": 58,
            "SWE": 52,  "NOR": 50,  "DNK": 48,  "GBR": 48,
            "CHE": 42,  "PRT": 38,  "ESP": 40,  "IRL": 30,  "LUX": 32,
            "TUR": 48,  "GEO": 65,  "ARM": 50,  "AZE": 52,
            "USA": 35,  "CAN": 28,  "BRA": 15,  "MEX": 12,
            "ARG": 12,  "CHL": 10,  "COL": 10,
            "JPN": 42,  "KOR": 35,  "CHN": 32,  "IND": 38,
            "HKG": 30,  "TWN": 25,  "SGP": 22,
            "AUS": 18,  "NZL": 14,
            "PAK": 18,  "BGD": 14,  "VNM": 12,  "THA": 12,
            "IDN": 10,  "MYS": 10,  "PHL": 10,
            "ISR": 22,  "SAU": 20,  "ARE": 18,  "EGY": 20,
            "TUN": 15,  "MAR": 14,  "DZA": 18,  "LBY": 22,
            "IRN": 20,  "IRQ": 18,  "SYR": 25,  "JOR": 18,
            "QAT": 15,  "KWT": 14,  "OMN": 12,  "BHR": 12,
            "ZAF": 14,  "NGA": 12,  "KEN": 10,  "ETH": 10,  "SDN": 15,
        },
    },
    {
        "label":       "Israel-Hamas War",
        "event_label": "Israel-Hamas",
        "color":       "#f39c12",
        "scores": {
            "ISR": 100, "PSE": 100, "LBN": 88,  "YEM": 82,
            "IRN": 80,  "SYR": 72,  "IRQ": 65,  "JOR": 70,
            "EGY": 65,  "SAU": 62,  "QAT": 58,  "KWT": 50,
            "ARE": 52,  "OMN": 45,  "BHR": 48,
            "TUR": 55,  "LBY": 38,  "SDN": 35,  "MAR": 25,  "TUN": 28,
            "DZA": 22,  "CYP": 42,  "GRC": 35,
            "GBR": 35,  "DEU": 28,  "FRA": 32,  "ITA": 30,
            "ESP": 28,  "NLD": 26,  "BEL": 25,  "PRT": 20,
            "CHE": 18,  "SWE": 16,  "NOR": 18,  "DNK": 16,
            "AUT": 22,  "POL": 18,  "HUN": 15,  "ROU": 15,
            "USA": 30,  "CAN": 18,  "BRA": 10,  "MEX": 12,  "ARG": 10,
            "JPN": 18,  "KOR": 20,  "CHN": 22,  "IND": 25,
            "HKG": 22,  "TWN": 20,  "SGP": 28,
            "PAK": 28,  "BGD": 22,  "IDN": 22,  "MYS": 20,
            "VNM": 15,  "THA": 18,  "PHL": 14,
            "AUS": 12,  "NZL": 10,
            "RUS": 20,  "UKR": 18,
            "ZAF": 14,  "NGA": 15,  "ETH": 12,  "KEN": 10,
        },
    },
]


# ── Indian administered disputed territories - centroid hover markers ────────
# These regions appear blank on Plotly's Natural Earth base map (not part of IND ISO polygon).
# We drop invisible centroid markers so hovering over the blank shows the correct attribution.

_INDIA_DISPUTED: list[dict] = [
    {
        "name": "Jammu & Kashmir + Ladakh",
        "lat": 34.2, "lon": 76.5,
        "note": "Disputed territories, administered by the Indian Republic.",
    },
    {
        "name": "Arunachal Pradesh",
        "lat": 27.8, "lon": 94.5,
        "note": "Disputed territories, administered by the Indian Republic.",
    },
]


# ── Active-war hometurf markers ───────────────────────────────────────────────

_HOMETURF_WARS: list[dict] = [
    {"name": "Ukraine",       "lat": 49.0, "lon": 31.5, "war": "Russia-Ukraine War",
     "note": "Active front line. Full-scale invasion since Feb 2022."},
    {"name": "Israel",        "lat": 31.5, "lon": 34.8, "war": "Israel-Hamas War",
     "note": "Multiple-front war: Gaza, Lebanon border, Iranian strikes."},
    {"name": "Gaza / W.Bank", "lat": 31.9, "lon": 35.2, "war": "Israel-Hamas War",
     "note": "Ground combat in Gaza; West Bank tensions."},
    {"name": "Lebanon",       "lat": 33.9, "lon": 35.5, "war": "Israel-Hamas War",
     "note": "Hezbollah-Israel exchanges; southern Lebanon bombed."},
    {"name": "Yemen",         "lat": 15.6, "lon": 48.5, "war": "Israel-Hamas War",
     "note": "Houthi missile/drone campaign; Red Sea shipping blocked."},
    {"name": "Syria",         "lat": 34.8, "lon": 38.5, "war": "Multi-conflict",
     "note": "Ongoing proxy war; Israeli strikes on Iranian positions."},
]


# ── Colorscale (cream → deep-crimson) ────────────────────────────────────────

_COLORSCALE = [
    [0.00, "#f5f2ee"],
    [0.15, "#fde0c8"],
    [0.35, "#f5a870"],
    [0.55, "#e05c3a"],
    [0.75, "#b82020"],
    [1.00, "#7a0e0e"],
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _impact_label(s: int) -> str:
    if s >= 75: return "High"
    if s >= 50: return "Elevated"
    if s >= 25: return "Moderate"
    if s >  0:  return "Low"
    return "Minimal"


def _war_period_returns(eq_r: pd.DataFrame) -> dict[str, dict[str, float]]:
    ev_map = {ev["label"]: ev["start"] for ev in GEOPOLITICAL_EVENTS}
    out: dict[str, dict[str, float]] = {}
    for war in _WAR_DATA:
        war_start = None
        for key, dt in ev_map.items():
            if war["event_label"] in key or key in war["event_label"]:
                war_start = pd.Timestamp(dt)
                break
        if war_start is None:
            continue
        for indices in _COUNTRY_INDICES.values():
            for idx in indices:
                if idx not in eq_r.columns:
                    continue
                s = eq_r[idx].loc[war_start:]
                if len(s) < 5:
                    continue
                cum = float((1 + s).prod() - 1) * 100
                out.setdefault(idx, {})[war["label"]] = round(cum, 1)
    return out


def _build_df(war_rets: dict[str, dict[str, float]]) -> pd.DataFrame:
    # Start with all scored countries
    scored_iso: set[str] = set()
    for w in _WAR_DATA:
        scored_iso.update(w["scores"].keys())

    rows = []
    # All countries in the world - unscored get 0 (renders as cream)
    all_iso = set(_ALL_COUNTRIES) | scored_iso
    for iso in all_iso:
        u = _WAR_DATA[0]["scores"].get(iso, 0)
        h = _WAR_DATA[1]["scores"].get(iso, 0)
        composite = max(u, h)
        indices = _COUNTRY_INDICES.get(iso, [])
        ret_lines = []
        for idx in indices:
            for wlbl, ret in war_rets.get(idx, {}).items():
                sign = "+" if ret >= 0 else ""
                ret_lines.append(f"{idx}: {sign}{ret:.1f}% ({wlbl})")
        rows.append({
            "iso3":          iso,
            "country":       _NAMES.get(iso, iso),
            "score":         composite,
            "ukraine_score": u,
            "hamas_score":   h,
            "primary_war":   _WAR_DATA[0]["label"] if u >= h else _WAR_DATA[1]["label"],
            "indices":       ", ".join(indices) if indices else "-",
            "ret_text":      " · ".join(ret_lines),
        })
    return pd.DataFrame(rows)


# ── Page ──────────────────────────────────────────────────────────────────────

def page_war_impact_map(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.3rem">Geopolitical War Impact Map</h1>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "Countries shaded cream → deep-red by composite war-impact score (0–100), "
        "combining geographic proximity, energy/gas dependency, trade-route exposure, "
        "and equity-market correlation to conflict assets. "
        "Hover any country for the breakdown and war-period equity returns. "
        "Active conflicts: Russia-Ukraine War (Feb 2022) · "
        "Israel-Hamas & Iran Escalation (Oct 2023)."
    )

    with st.spinner("Loading equity data…"):
        eq_r, _ = load_returns(start, end)

    war_rets = _war_period_returns(eq_r) if not eq_r.empty else {}
    df = _build_df(war_rets)

    # ── Active conflict cards ─────────────────────────────────────────────────
    st.markdown("---")
    today = date.today()
    active = [ev for ev in GEOPOLITICAL_EVENTS
              if ev["category"] == "Geopolitical" and ev["end"] >= today]
    if active:
        cols = st.columns(len(active))
        for col, ev in zip(cols, active):
            days = (today - ev["start"]).days
            col.markdown(
                f'<div style="border-left:4px solid {ev["color"]};padding:0.6rem 0.9rem;'
                f'background:#fafaf8;border-radius:0 4px 4px 0">'
                f'<div style="font-size:0.54rem;font-weight:700;letter-spacing:0.12em;'
                f'text-transform:uppercase;color:{ev["color"]}">Active Conflict</div>'
                f'<div style="font-size:0.82rem;font-weight:700;color:#1a1a1a;margin-top:2px">'
                f'{ev["name"]}</div>'
                f'<div style="font-size:0.62rem;color:#666;margin-top:3px">'
                f'Since {ev["start"].strftime("%b %d, %Y")} &nbsp;·&nbsp; <b>{days:,} days</b></div>'
                f'<div style="font-size:0.62rem;color:#444;margin-top:4px;line-height:1.5">'
                f'{ev["description"][:120]}…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, _ = st.columns([2, 2, 4])
    war_filter = c1.radio(
        "Show impact for",
        ["Combined (max)", "Ukraine War only", "Israel-Hamas only"],
        key="war_filter_map",
    )
    view_mode = c2.radio(
        "Map style",
        ["Flat Map", "3D Globe"],
        key="map_view_mode",
    )

    if war_filter == "Ukraine War only":
        score_col, title_sfx = "ukraine_score", "- Russia-Ukraine War"
    elif war_filter == "Israel-Hamas only":
        score_col, title_sfx = "hamas_score", "- Israel-Hamas War"
    else:
        score_col, title_sfx = "score", "- Combined (worst of both conflicts)"

    is_globe = (view_mode == "3D Globe")

    # ── Hover text ────────────────────────────────────────────────────────────
    hometurf_iso = {"UKR", "ISR", "PSE", "LBN", "YEM", "SYR"}
    hover_texts = []
    for _, row in df.iterrows():
        s = int(row[score_col])
        if s == 0:
            # Unscored country - minimal hover
            hover_texts.append(f"<b>{row['country']}</b><br>No significant direct exposure")
            continue
        lv = _impact_label(s)
        lv_color = (
            "#7a0e0e" if s >= 75 else
            "#b82020" if s >= 50 else
            "#e05c3a" if s >= 25 else
            "#888888"
        )
        hw = ("<br><b style='color:#c0392b'>⚔ Active war on home soil</b>"
              if row["iso3"] in hometurf_iso else "")
        tip = (
            f"<b>{row['country']}</b>{hw}<br>"
            f"<span style='color:{lv_color}'><b>{s}/100 - {lv}</b></span><br><br>"
            f"Ukraine War: {int(row['ukraine_score'])}/100<br>"
            f"Israel-Hamas: {int(row['hamas_score'])}/100"
        )
        if row["indices"] != "-":
            tip += f"<br><br><b>Tracked index:</b> {row['indices']}"
        if row["ret_text"]:
            tip += f"<br><b>War-period returns:</b><br>{row['ret_text']}"
        hover_texts.append(tip)
    df["hover"] = hover_texts

    # ── Figure ────────────────────────────────────────────────────────────────
    # Strategy: populate ALL world countries in the choropleth.
    # Unscored countries → z=0 → cream. This means showland is irrelevant;
    # the choropleth itself covers every land area without triggering the
    # showland-overwrites-choropleth bug in orthographic projection.
    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        locations=df["iso3"],
        locationmode="ISO-3",
        z=df[score_col].astype(float),
        text=df["hover"],
        hovertemplate="%{text}<extra></extra>",
        colorscale=_COLORSCALE,
        zmin=0, zmax=100,
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Impact",
                font=dict(size=9, family="DM Sans, sans-serif", color="#333"),
            ),
            thickness=12,
            len=0.70,
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0", "25", "50", "75", "100"],
            tickfont=dict(size=8, family="JetBrains Mono, monospace", color="#444"),
            outlinewidth=0,
            bgcolor="rgba(0,0,0,0)",
            x=1.01,
        ),
        marker_line_color="rgba(255,255,255,0.55)",
        marker_line_width=0.4,
    ))

    # India disputed-territory hover markers (invisible dots, hover only)
    fig.add_trace(go.Scattergeo(
        lat=[t["lat"] for t in _INDIA_DISPUTED],
        lon=[t["lon"] for t in _INDIA_DISPUTED],
        mode="markers",
        marker=dict(size=18, color="rgba(255,255,255,0.0)",
                    line=dict(color="rgba(255,255,255,0.0)", width=0)),
        customdata=[[t["name"], t["note"]] for t in _INDIA_DISPUTED],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "<b>India</b><br>"
            "<i>%{customdata[1]}</i><extra></extra>"
        ),
        showlegend=False,
        hoverinfo="text",
    ))

    # Active war-zone markers (white X, red outline)
    fig.add_trace(go.Scattergeo(
        lat=[h["lat"] for h in _HOMETURF_WARS],
        lon=[h["lon"] for h in _HOMETURF_WARS],
        mode="markers",
        marker=dict(
            size=11,
            color="#ffffff",
            symbol="x",
            line=dict(color="#c0392b", width=2.2),
        ),
        customdata=[[h["name"], h["war"], h["note"]] for h in _HOMETURF_WARS],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "⚔ %{customdata[1]}<br>"
            "<i>%{customdata[2]}</i><extra></extra>"
        ),
        name="⚔ Active War Zone",
        showlegend=True,
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    # showland=False everywhere - choropleth covers all land.
    # bgcolor=ocean colour so the globe "sphere" exterior looks like water.
    if is_globe:
        geo_cfg = dict(
            projection=dict(
                type="orthographic",
                rotation=dict(lon=20, lat=20, roll=0),
            ),
            showland=False,
            showocean=True,
            oceancolor="#1a3f5c",
            showcoastlines=True,
            coastlinecolor="rgba(255,255,255,0.60)",
            coastlinewidth=0.6,
            showcountries=True,
            countrycolor="rgba(255,255,255,0.25)",
            countrywidth=0.3,
            showlakes=True,
            lakecolor="#1a3f5c",
            showframe=False,
            bgcolor="#000000",   # space black - fills area outside the globe sphere
            lataxis=dict(showgrid=True,  gridcolor="rgba(255,255,255,0.12)", gridwidth=0.5),
            lonaxis=dict(showgrid=True,  gridcolor="rgba(255,255,255,0.12)", gridwidth=0.5),
            resolution=50,
        )
        paper_bg = "#000000"
        height   = 580
    else:
        geo_cfg = dict(
            projection=dict(type="natural earth"),
            showland=False,
            showocean=True,
            oceancolor="#c4dcea",
            showcoastlines=True,
            coastlinecolor="rgba(100,90,80,0.40)",
            coastlinewidth=0.5,
            showcountries=True,
            countrycolor="rgba(120,110,100,0.35)",
            countrywidth=0.3,
            showlakes=True,
            lakecolor="#c4dcea",
            showframe=False,
            bgcolor="rgba(0,0,0,0)",
            resolution=110,
        )
        paper_bg = "#ffffff"
        height   = 530

    legend_cfg = dict(
        x=0.01, y=0.05,
        bgcolor="rgba(0,0,0,0.55)" if is_globe else "rgba(255,255,255,0.88)",
        bordercolor="rgba(255,255,255,0.25)" if is_globe else "#E8E5E0",
        borderwidth=1,
        font=dict(size=9, family="DM Sans, sans-serif",
                  color="#dddddd" if is_globe else "#333"),
    )
    title_color = "#dddddd" if is_globe else "#1a1a1a"
    cb_font_color = "#cccccc" if is_globe else "#444"

    fig.update_traces(
        selector=dict(type="choropleth"),
        colorbar=dict(
            title=dict(text="Impact",
                       font=dict(size=9, family="DM Sans, sans-serif", color=cb_font_color)),
            thickness=12, len=0.70,
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0", "25", "50", "75", "100"],
            tickfont=dict(size=8, family="JetBrains Mono, monospace", color=cb_font_color),
            outlinewidth=0,
            bgcolor="rgba(0,0,0,0)",
            x=1.01,
        ),
    )

    fig.update_layout(
        uirevision="war_map_v3",
        geo=geo_cfg,
        height=height,
        margin=dict(l=0, r=80, t=40, b=0),
        paper_bgcolor=paper_bg,
        legend=legend_cfg,
        title=dict(
            text=f"Equity Market War Impact  {title_sfx}",
            font=dict(size=11, family="DM Sans, sans-serif", color=title_color),
            x=0.01, y=0.98,
        ),
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color="#aaaaaa",
            activecolor="#CFB991",
            remove=["select2d", "lasso2d", "autoScale2d"],
        ),
    )

    _chart(fig)

    # Colour legend strip
    st.markdown(
        '<div style="display:flex;gap:1rem;flex-wrap:wrap;margin:0.2rem 0 0.8rem">'
        + "".join(
            f'<span style="font-size:0.62rem;color:#444">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{c};border-radius:2px;vertical-align:middle;margin-right:4px"></span>'
            f'{lbl}</span>'
            for c, lbl in [
                ("#f5f2ee", "No exposure"),
                ("#fde0c8", "Low (10–25)"),
                ("#f5a870", "Moderate (25–50)"),
                ("#e05c3a", "Elevated (50–75)"),
                ("#b82020", "High (75–90)"),
                ("#7a0e0e", "Crisis (90–100)"),
            ]
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Tracked equity performance table ─────────────────────────────────────
    st.markdown("---")
    st.subheader("Tracked Equity Markets - War-Period Performance")
    _section_note(
        "Cumulative returns for each tracked equity index since each conflict's start date."
    )

    tracked_rows = []
    for iso, indices in _COUNTRY_INDICES.items():
        u = _WAR_DATA[0]["scores"].get(iso, 0)
        h = _WAR_DATA[1]["scores"].get(iso, 0)
        for idx in indices:
            w = war_rets.get(idx, {})
            tracked_rows.append({
                "Country":         _NAMES.get(iso, iso),
                "Equity Index":    idx,
                "Impact Score":    max(u, h),
                "Ukraine War (%)": w.get("Ukraine War", None),
                "Israel War (%)":  w.get("Israel-Hamas War", None),
            })

    if tracked_rows:
        tdf = (
            pd.DataFrame(tracked_rows)
            .sort_values("Impact Score", ascending=False)
            .reset_index(drop=True)
        )

        def _fmt(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "-"
            return f"{v:+.1f}%"

        def _sty(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ""
            if v > 5:   return "color:#2e7d32;font-weight:600"
            if v > 0:   return "color:#4a8a4e;font-weight:500"
            if v > -10: return "color:#e67e22;font-weight:600"
            return "color:#c0392b;font-weight:700"

        st.dataframe(
            tdf.style
            .applymap(_sty, subset=["Ukraine War (%)", "Israel War (%)"])
            .format({"Impact Score": "{:.0f}", "Ukraine War (%)": _fmt, "Israel War (%)": _fmt})
            .background_gradient(subset=["Impact Score"], cmap="RdYlGn_r", vmin=0, vmax=100),
            use_container_width=True,
            hide_index=True,
        )

    # ── Top 25 table ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Top 25 Most Affected Countries")

    top25 = (
        df[df[score_col] > 0]
        .sort_values(score_col, ascending=False)
        .head(25)
        [["country", score_col, "ukraine_score", "hamas_score", "indices"]]
        .rename(columns={
            "country":       "Country",
            score_col:       "Impact Score",
            "ukraine_score": "Ukraine War",
            "hamas_score":   "Israel-Hamas War",
            "indices":       "Tracked Index",
        })
        .reset_index(drop=True)
    )
    top25.index += 1

    st.dataframe(
        top25.style
        .background_gradient(subset=["Impact Score"], cmap="RdYlGn_r", vmin=0, vmax=100)
        .format({"Impact Score": "{:.0f}", "Ukraine War": "{:.0f}", "Israel-Hamas War": "{:.0f}"}),
        use_container_width=True,
    )

    _page_conclusion(
        "War Impact Map",
        "This map synthesises geographic proximity, energy dependence, trade linkages, and equity "
        "market correlation to quantify how active conflicts propagate across global capital markets. "
        "High-impact zones face disruptions via energy supply chains, shipping routes "
        "(Suez, Black Sea), refugee/fiscal spillovers, or direct military involvement.",
    )
    _page_footer()

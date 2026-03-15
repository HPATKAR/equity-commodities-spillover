"""
War Impact Map - Geopolitical Equity Risk
Choropleth map (flat or 3-D globe) showing equity-market exposure to active wars.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from datetime import date

from src.data.config import GEOPOLITICAL_EVENTS, PALETTE
from src.data.loader import load_returns
from src.ui.shared import (
    _chart, _page_intro, _section_note, _definition_block,
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
    {
        "label":       "Iran/Hormuz Crisis",
        "event_label": "Iran/Hormuz",
        "color":       "#c0392b",
        "scores": {
            # Direct conflict parties
            "IRN": 100, "ISR": 88,  "USA": 80,  "YEM": 75,
            # Strait of Hormuz - critical chokepoint states
            "OMN": 82,  "QAT": 80,  "ARE": 78,  "SAU": 78,
            "KWT": 72,  "BHR": 70,  "IRQ": 68,
            # Iran proxy / linked conflicts
            "LBN": 65,  "SYR": 58,  "PSE": 62,  "JOR": 55,
            # High oil/LNG import dependency via Hormuz
            "JPN": 68,  "KOR": 65,  "IND": 62,  "PAK": 55,
            "CHN": 55,  "SGP": 52,  "BGD": 30,
            # Regional / shipping exposure
            "EGY": 52,  "TUR": 50,  "GRC": 45,  "CYP": 40,
            "ITA": 42,  "DEU": 40,  "GBR": 38,  "FRA": 38,
            "NLD": 35,  "BEL": 28,  "ESP": 32,  "PRT": 22,
            "NOR": 28,  "SWE": 20,  "DNK": 22,  "CHE": 18,
            "AUT": 25,  "POL": 20,  "HUN": 18,  "ROU": 18,
            # Asia-Pacific energy importers
            "TWN": 48,  "HKG": 42,  "IDN": 32,  "MYS": 30,
            "THA": 28,  "VNM": 22,  "PHL": 20,  "AUS": 25,
            # Americas - indirect via oil price
            "CAN": 15,  "BRA": 12,  "MEX": 14,  "ARG": 10,
            # Africa - competing exporters / importers
            "LBY": 25,  "DZA": 20,  "MAR": 18,  "TUN": 15,
            "NGA": 15,  "ZAF": 20,  "ETH": 12,  "KEN": 10, "SDN": 18,
            # Russia / Ukraine - indirect oil price benefit/exposure
            "RUS": 18,  "UKR": 15,
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
    {"name": "Iran",          "lat": 32.4, "lon": 53.7, "war": "Iran/Hormuz Crisis",
     "note": "US-Israel strikes on nuclear/military facilities; Hormuz closure threat."},
    {"name": "Strait of Hormuz", "lat": 26.6, "lon": 56.3, "war": "Iran/Hormuz Crisis",
     "note": "~20% of global oil + 25% of global LNG transits this chokepoint."},
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
        u  = _WAR_DATA[0]["scores"].get(iso, 0)
        h  = _WAR_DATA[1]["scores"].get(iso, 0)
        ir = _WAR_DATA[2]["scores"].get(iso, 0)
        composite = max(u, h, ir)
        primary = max(
            (_WAR_DATA[0]["label"], u),
            (_WAR_DATA[1]["label"], h),
            (_WAR_DATA[2]["label"], ir),
            key=lambda x: x[1],
        )[0]
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
            "iran_score":    ir,
            "primary_war":   primary,
            "indices":       ", ".join(indices) if indices else "-",
            "ret_text":      " · ".join(ret_lines),
        })
    return pd.DataFrame(rows)


# ── Local GeoJSON loader ──────────────────────────────────────────────────────
# Reads the pre-processed compact GeoJSON bundled with the project.
# No network request — 100% reliable in any deployment context.

_ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"
_GEO_FILE = _ASSETS / "ne_110m_countries_compact.geojson"


@st.cache_data(show_spinner=False)
def _load_globe_geojson() -> list[dict]:
    """Load [{iso, name, g}] from the bundled compact GeoJSON. Cached for the session."""
    try:
        return json.loads(_GEO_FILE.read_bytes())
    except Exception:
        return []


# ── D3 Orthographic Globe template ───────────────────────────────────────────
# Engine: D3 v7 geoOrthographic + canvas.
# Drag: setPointerCapture — works inside Streamlit iframes.
# Hover: proj.invert(mouse) → d3.geoContains(feature, point) — 100% reliable.
# GeoJSON: injected server-side — no client fetch, no CORS issues.

_GLOBE_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#050d1a;overflow:hidden;user-select:none;-webkit-user-select:none}
#wrap{position:relative;width:100%;height:580px}
#gc{display:block;touch-action:none}
#panel{
  position:absolute;bottom:14px;left:14px;width:222px;
  background:rgba(5,11,26,0.95);
  border:1px solid rgba(207,185,145,0.25);
  border-radius:8px;padding:13px 15px;
  font-family:'DM Sans',sans-serif;font-size:11px;line-height:1.62;
  color:#dde0e8;transition:border-color .2s;
  pointer-events:none;
}
#panel.lit{border-color:rgba(207,185,145,0.60)}
#panel .plbl{font-size:7.5px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:#CFB991;margin-bottom:6px}
#panel .ph{color:#4a5568;font-size:10px;line-height:1.5;font-style:italic}
#panel b{color:#fff}
#panel hr{border:none;border-top:1px solid rgba(255,255,255,0.07);margin:7px 0}
#legend{
  position:absolute;bottom:14px;right:14px;
  background:rgba(5,11,26,0.90);
  border:1px solid rgba(100,140,220,0.14);
  border-radius:7px;padding:9px 12px;
  font-family:'DM Sans',sans-serif;font-size:9px;color:#aaa;
  pointer-events:none;
}
.lt{font-weight:700;color:#CFB991;letter-spacing:.1em;font-size:8px;text-transform:uppercase;margin-bottom:5px}
.lr{display:flex;align-items:center;gap:6px;margin:2px 0}
.lc{width:11px;height:11px;border-radius:2px;flex-shrink:0}
</style></head><body>
<div id="wrap">
  <canvas id="gc"></canvas>
  <div id="panel">
    <div class="plbl">Country Analysis</div>
    <p class="ph">Drag to rotate &middot; Scroll to zoom<br>Hover any country for war-impact data</p>
  </div>
  <div id="legend">
    <div class="lt">Impact Scale</div>
    <div class="lr"><div class="lc" style="background:#f5f2ee;border:1px solid #bbb"></div>No exposure</div>
    <div class="lr"><div class="lc" style="background:#fde0c8"></div>Low (10&ndash;25)</div>
    <div class="lr"><div class="lc" style="background:#f5a870"></div>Moderate (25&ndash;50)</div>
    <div class="lr"><div class="lc" style="background:#e05c3a"></div>Elevated (50&ndash;75)</div>
    <div class="lr"><div class="lc" style="background:#b82020"></div>High (75&ndash;90)</div>
    <div class="lr"><div class="lc" style="background:#7a0e0e"></div>Crisis (90&ndash;100)</div>
    <hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:5px 0">
    <div class="lr"><div style="width:9px;height:9px;border-radius:50%;background:#ff3333;flex-shrink:0"></div>Active War Zone</div>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script>
/* ── injected from Python (no client fetch — data embedded server-side) ── */
const CDATA    = __COUNTRY_DATA__;     // {ISO3:{score}}
const HOVER    = __HOVER_DATA__;       // {ISO3:html}
const HT       = __HOMETURF__;         // [{lat,lng,name,war,note}]
const GEO_RAW  = __GEO_FEATURES__;     // [{iso,name,g:{type,coordinates}}]

/* ── canvas setup — size after layout ── */
const wrap   = document.getElementById('wrap');
const canvas = document.getElementById('gc');
const panel  = document.getElementById('panel');

const W = Math.max(wrap.getBoundingClientRect().width || wrap.offsetWidth, 300);
const H = 580;
canvas.width  = W;
canvas.height = H;
const ctx = canvas.getContext('2d');
const cx = W / 2, cy = H / 2;
const R0 = Math.min(W, H * 0.88) * 0.44;

/* ── D3 orthographic projection ── */
const proj = d3.geoOrthographic()
  .scale(R0).translate([cx, cy]).clipAngle(90).precision(0.1);
const pathGen = d3.geoPath(proj, ctx);
const sphere  = {type: 'Sphere'};
const gratGen = d3.geoGraticule10();

let featData = [];
let hoverIso = null;

/* ── colour scale ── */
const CS = [[0,[245,242,238]],[15,[253,224,200]],[35,[245,168,112]],
            [55,[224,92,58]],[75,[184,32,32]],[100,[122,14,14]]];
function sc(v) {
  if (!v || v <= 0) return 'rgba(245,242,238,0.88)';
  const s = Math.max(0, Math.min(100, v));
  let lo = CS[0], hi = CS[CS.length-1];
  for (let i = 0; i < CS.length-1; i++)
    if (s >= CS[i][0] && s <= CS[i+1][0]) { lo=CS[i]; hi=CS[i+1]; break; }
  const t = hi[0]===lo[0] ? 0 : (s-lo[0])/(hi[0]-lo[0]);
  return 'rgba('+[0,1,2].map(j=>Math.round(lo[1][j]+t*(hi[1][j]-lo[1][j]))).join(',')+',0.88)';
}

/* ── draw ── */
let rot = [20, -20, 0];
function draw() {
  ctx.clearRect(0, 0, W, H);
  proj.rotate(rot);

  /* atmosphere */
  const ag = ctx.createRadialGradient(cx,cy,R0*0.94,cx,cy,R0*1.16);
  ag.addColorStop(0,'rgba(40,100,220,0.20)');
  ag.addColorStop(0.6,'rgba(20,60,200,0.06)');
  ag.addColorStop(1,'rgba(0,0,0,0)');
  ctx.beginPath(); ctx.arc(cx,cy,R0*1.16,0,Math.PI*2); ctx.fillStyle=ag; ctx.fill();

  /* ocean */
  ctx.beginPath(); pathGen(sphere);
  const og = ctx.createRadialGradient(cx-R0*.28,cy-R0*.25,R0*.04,cx,cy,R0);
  og.addColorStop(0,'#1a4b8e'); og.addColorStop(0.55,'#0d2f6e'); og.addColorStop(1,'#071840');
  ctx.fillStyle=og; ctx.fill();

  /* country fills */
  for (const {f,iso} of featData) {
    ctx.beginPath(); pathGen(f);
    ctx.fillStyle = iso===hoverIso ? 'rgba(255,210,72,0.96)' : sc((CDATA[iso]||{}).score||0);
    ctx.fill();
  }
  /* country borders (separate pass — avoids fill bleeding over stroke) */
  for (const {f,iso} of featData) {
    ctx.beginPath(); pathGen(f);
    if (iso===hoverIso) { ctx.strokeStyle='rgba(255,210,72,1.0)'; ctx.lineWidth=1.6; }
    else                { ctx.strokeStyle='rgba(255,255,255,0.18)'; ctx.lineWidth=0.35; }
    ctx.stroke();
  }

  /* graticule */
  ctx.beginPath(); pathGen(gratGen);
  ctx.strokeStyle='rgba(255,255,255,0.04)'; ctx.lineWidth=0.4; ctx.stroke();

  /* sphere rim */
  ctx.beginPath(); pathGen(sphere);
  ctx.strokeStyle='rgba(80,140,255,0.20)'; ctx.lineWidth=0.9; ctx.stroke();

  /* war-zone dots (front hemisphere only) */
  const front = [-rot[0], -rot[1]];
  for (const ht of HT) {
    if (d3.geoDistance([ht.lng,ht.lat], front) >= Math.PI/2-0.02) continue;
    const p = proj([ht.lng,ht.lat]); if (!p) continue;
    const [px,py] = p;
    const rg = ctx.createRadialGradient(px,py,0,px,py,13);
    rg.addColorStop(0,'rgba(255,50,50,0.55)'); rg.addColorStop(1,'rgba(255,50,50,0)');
    ctx.beginPath(); ctx.arc(px,py,13,0,Math.PI*2); ctx.fillStyle=rg; ctx.fill();
    ctx.beginPath(); ctx.arc(px,py,4.3,0,Math.PI*2);
    ctx.fillStyle='#ff3333'; ctx.fill(); ctx.strokeStyle='#fff'; ctx.lineWidth=1.1; ctx.stroke();
  }
}

/* ── hover ── */
function setHover(iso, name) {
  if (iso === hoverIso) return;
  hoverIso = iso; draw();
  const lbl = '<div class="plbl">Country Analysis</div>';
  if (!iso) {
    panel.classList.remove('lit');
    panel.innerHTML = lbl+'<p class="ph">Drag to rotate &middot; Scroll to zoom<br>Hover any country for war-impact data</p>';
    return;
  }
  panel.classList.add('lit');
  const h = HOVER[iso];
  panel.innerHTML = lbl + (h
    ? '<div style="font-size:11px;line-height:1.62">'+h+'</div>'
    : '<b>'+(name||iso)+'</b><hr><span style="color:#4a5568;font-size:10px">No war-impact data tracked.</span>');
}

function handleHover(mx, my) {
  const dx=mx-cx, dy=my-cy;
  if (dx*dx+dy*dy > R0*R0) { setHover(null,null); canvas.style.cursor='grab'; return; }
  const ll = proj.invert([mx,my]); if (!ll) { setHover(null,null); return; }
  /* reject back-hemisphere (proj.invert is purely mathematical — no clipping) */
  if (d3.geoDistance(ll, [-rot[0],-rot[1]]) >= Math.PI/2) { setHover(null,null); return; }
  for (const {f,iso,name} of featData) {
    if (d3.geoContains(f, ll)) { setHover(iso,name); canvas.style.cursor='crosshair'; return; }
  }
  setHover(null,null); canvas.style.cursor='grab';
}

/* ── pointer-capture drag (works in Streamlit iframes — no window listener needed) ── */
let velX=0, velY=0, lastX=0, lastY=0, lastT=0, inertiaId=null, dragging=false;

function clampPhi() { rot[1]=Math.max(-89,Math.min(89,rot[1])); }

function startInertia() {
  if (inertiaId) cancelAnimationFrame(inertiaId);
  let vx=velX, vy=velY;
  function tick() {
    vx*=0.91; vy*=0.91;
    if (Math.hypot(vx,vy) < 0.008) return;
    rot[0]+=vx; rot[1]+=vy; clampPhi(); draw();
    inertiaId = requestAnimationFrame(tick);
  }
  inertiaId = requestAnimationFrame(tick);
}

canvas.addEventListener('pointerdown', e => {
  canvas.setPointerCapture(e.pointerId);   /* ← keeps events flowing even outside iframe */
  dragging=true; lastX=e.clientX; lastY=e.clientY; lastT=e.timeStamp;
  velX=velY=0;
  if (inertiaId) { cancelAnimationFrame(inertiaId); inertiaId=null; }
  canvas.style.cursor='grabbing';
  e.preventDefault();
});

canvas.addEventListener('pointermove', e => {
  const r = canvas.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  if (dragging) {
    const dx=e.clientX-lastX, dy=e.clientY-lastY;
    const dt=Math.max(1, e.timeStamp-lastT);
    velX=dx/dt*16; velY=dy/dt*16;
    rot[0]+=dx*0.28; rot[1]-=dy*0.28; clampPhi();
    lastX=e.clientX; lastY=e.clientY; lastT=e.timeStamp;
    draw();
  } else {
    handleHover(mx, my);
  }
});

canvas.addEventListener('pointerup', e => {
  canvas.releasePointerCapture(e.pointerId);
  dragging=false; canvas.style.cursor='grab'; startInertia();
});

canvas.addEventListener('pointerleave', e => {
  if (!dragging) setHover(null,null);
});

/* ── scroll zoom ── */
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  proj.scale(Math.max(R0*.44, Math.min(R0*3.8, proj.scale()-e.deltaY*0.4)));
  draw();
}, {passive:false});

/* ── build feature index from injected data (synchronous — no fetch needed) ── */
featData = GEO_RAW.map(d => ({
  f:    {type: 'Feature', geometry: d.g, properties: {ISO_A3: d.iso}},
  iso:  d.iso,
  name: d.name,
}));
draw();
</script></body></html>"""


def _render_globe_component(df: pd.DataFrame, score_col: str) -> None:
    """Fetch GeoJSON server-side, inject all data into D3 globe template, render."""
    country_data = {str(row["iso3"]): {"score": int(row[score_col])}
                    for _, row in df.iterrows()}
    hover_data   = {str(row["iso3"]): str(row.get("hover", ""))
                    for _, row in df.iterrows()}
    hometurf_data = [
        {"lat": h["lat"], "lng": h["lon"],
         "name": h["name"], "war": h["war"], "note": h["note"]}
        for h in _HOMETURF_WARS
    ]
    # Server-side GeoJSON — injected directly, no browser fetch required
    geo_features = _load_globe_geojson()
    html = (
        _GLOBE_HTML
        .replace("__COUNTRY_DATA__", json.dumps(country_data))
        .replace("__HOVER_DATA__",   json.dumps(hover_data))
        .replace("__HOMETURF__",     json.dumps(hometurf_data))
        .replace("__GEO_FEATURES__", json.dumps(geo_features))
    )
    components.html(html, height=600, scrolling=False)


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
        "Israel-Hamas & Iran Escalation (Oct 2023) · "
        "Iran/Hormuz Crisis (Jun 2025) - U.S.-Israel strikes on Iranian facilities; "
        "Strait of Hormuz closure threat disrupts ~20% of global oil supply."
    )

    _definition_block(
        "How the Impact Score is Determined",
        "Each country receives an independent score (0–100) for each of the three tracked conflicts. "
        "The <b>composite score</b> shown on the map is the worst-case value across all three - "
        "i.e. <code>max(Ukraine score, Israel-Hamas score, Iran/Hormuz score)</code>. "
        "Scores are constructed from five factors:<br><br>"

        "<b>1. Geographic proximity</b> - Direct border exposure, contiguous-state risk, "
        "and displacement or refugee-flow pressure on neighbouring economies.<br>"

        "<b>2. Energy dependency</b> - Share of oil, gas, and LNG imports that transit "
        "disrupted supply corridors or originate from conflict-adjacent exporters. "
        "Countries with high Russian-gas or Middle-East-oil reliance score higher.<br>"

        "<b>3. Trade route exposure</b> - Vulnerability to blocked or threatened shipping lanes. "
        "Ukraine War elevates Black Sea exposure; Israel-Hamas escalation threatens Suez/Red Sea; "
        "Iran/Hormuz Crisis puts ~20 % of global oil and 25 % of global LNG at risk.<br>"

        "<b>4. Equity-market correlation</b> - Historical co-movement of the domestic index "
        "with conflict-sensitive commodities (crude oil, natural gas, wheat). "
        "Markets that historically reprice sharply when these commodities spike score higher.<br>"

        "<b>5. Alliance and sanctions exposure</b> - NATO/EU commitments requiring fiscal "
        "or military contributions, active sanctions regimes affecting trade flows, "
        "and diplomatic alignment with conflict parties.<br><br>"

        "<b>Conflict-specific emphasis:</b> Ukraine War scores weight energy dependency and "
        "geographic proximity most heavily (especially EU states reliant on Russian gas). "
        "Israel-Hamas scores weight regional spillover, Red Sea shipping disruption, and "
        "Muslim-world political risk premiums. Iran/Hormuz scores weight Strait of Hormuz "
        "oil-transit dependency - Japan, South Korea, India, and the Gulf states score highest "
        "because the strait carries the bulk of their crude imports.",
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
        ["Combined (max)", "Ukraine War only", "Israel-Hamas only", "Iran/Hormuz only"],
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
    elif war_filter == "Iran/Hormuz only":
        score_col, title_sfx = "iran_score", "- Iran/Hormuz Crisis"
    else:
        score_col, title_sfx = "score", "- Combined (worst of all conflicts)"

    is_globe = (view_mode == "3D Globe")

    # ── Hover text ────────────────────────────────────────────────────────────
    hometurf_iso = {"UKR", "ISR", "PSE", "LBN", "YEM", "SYR", "IRN"}
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
            f"Israel-Hamas: {int(row['hamas_score'])}/100<br>"
            f"Iran/Hormuz: {int(row['iran_score'])}/100"
        )
        if row["indices"] != "-":
            tip += f"<br><br><b>Tracked index:</b> {row['indices']}"
        if row["ret_text"]:
            tip += f"<br><b>War-period returns:</b><br>{row['ret_text']}"
        hover_texts.append(tip)
    df["hover"] = hover_texts

    # ── Render map ────────────────────────────────────────────────────────────
    if is_globe:
        # WebGL Globe.GL component — smooth inertial physics
        _render_globe_component(df, score_col)
    else:
        # Plotly flat map (Natural Earth projection)
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
                thickness=12, len=0.70,
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

        fig.update_layout(
            uirevision="war_map_v3",
            geo=dict(
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
            ),
            height=530,
            margin=dict(l=0, r=80, t=40, b=0),
            paper_bgcolor="#ffffff",
            legend=dict(
                x=0.01, y=0.05,
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#E8E5E0",
                borderwidth=1,
                font=dict(size=9, family="DM Sans, sans-serif", color="#333"),
            ),
            title=dict(
                text=f"Equity Market War Impact  {title_sfx}",
                font=dict(size=11, family="DM Sans, sans-serif", color="#1a1a1a"),
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
        u  = _WAR_DATA[0]["scores"].get(iso, 0)
        h  = _WAR_DATA[1]["scores"].get(iso, 0)
        ir = _WAR_DATA[2]["scores"].get(iso, 0)
        for idx in indices:
            w = war_rets.get(idx, {})
            tracked_rows.append({
                "Country":           _NAMES.get(iso, iso),
                "Equity Index":      idx,
                "Impact Score":      max(u, h, ir),
                "Ukraine War (%)":   w.get("Ukraine War", None),
                "Israel War (%)":    w.get("Israel-Hamas War", None),
                "Iran/Hormuz (%)":   w.get("Iran/Hormuz Crisis", None),
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
            .applymap(_sty, subset=["Ukraine War (%)", "Israel War (%)", "Iran/Hormuz (%)"])
            .format({"Impact Score": "{:.0f}", "Ukraine War (%)": _fmt,
                     "Israel War (%)": _fmt, "Iran/Hormuz (%)": _fmt})
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
        [["country", score_col, "ukraine_score", "hamas_score", "iran_score", "indices"]]
        .rename(columns={
            "country":       "Country",
            score_col:       "Impact Score",
            "ukraine_score": "Ukraine War",
            "hamas_score":   "Israel-Hamas War",
            "iran_score":    "Iran/Hormuz",
            "indices":       "Tracked Index",
        })
        .reset_index(drop=True)
    )
    top25.index += 1

    st.dataframe(
        top25.style
        .background_gradient(subset=["Impact Score"], cmap="RdYlGn_r", vmin=0, vmax=100)
        .format({"Impact Score": "{:.0f}", "Ukraine War": "{:.0f}",
                 "Israel-Hamas War": "{:.0f}", "Iran/Hormuz": "{:.0f}"}),
        use_container_width=True,
    )

    _page_conclusion(
        "War Impact Map",
        "This map synthesises geographic proximity, energy dependence, trade linkages, and equity "
        "market correlation to quantify how active conflicts propagate across global capital markets. "
        "Three simultaneous active conflicts - Russia-Ukraine War, Israel-Hamas & Iran Escalation, "
        "and the Iran/Hormuz Crisis - create compounding risk vectors. High-impact zones face "
        "disruptions via energy supply chains, shipping routes (Suez, Black Sea, Strait of Hormuz), "
        "refugee/fiscal spillovers, or direct military involvement.",
    )
    _page_footer()

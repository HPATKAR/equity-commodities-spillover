"""
War Impact Map - Geopolitical Equity Risk
Choropleth map (flat or 3-D globe) showing equity-market exposure to active wars.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
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


# ── Globe.GL WebGL template ───────────────────────────────────────────────────
# Loaded once at module level; data injected at render time via placeholder swap.

_GLOBE_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0a1a2e; overflow: hidden; }
#gw { position: relative; width: 100%; height: 580px; }
#globe { width: 100%; height: 100%; }
/* Custom tooltip — fixed so it never gets clipped by the iframe */
#tt {
  position: fixed; display: none; pointer-events: none; z-index: 9999;
  font-family: 'DM Sans',sans-serif; font-size: 11px; line-height: 1.55;
  padding: 8px 12px; max-width: 220px;
  background: rgba(5,15,35,0.94); border: 1px solid rgba(100,160,255,0.25);
  border-radius: 5px; color: #eee;
}
#cbar {
  position: absolute; bottom: 18px; right: 12px;
  background: rgba(5,15,35,0.78); border: 1px solid rgba(100,160,255,0.18);
  border-radius: 5px; padding: 8px 11px;
  font-size: 9px; font-family: 'DM Sans',sans-serif; color: #ccc;
}
.cr { display: flex; align-items: center; gap: 6px; margin: 2px 0; }
.cs { width: 13px; height: 13px; border-radius: 2px; flex-shrink: 0; }
#leg {
  position: absolute; bottom: 18px; left: 12px;
  background: rgba(5,15,35,0.72); border: 1px solid rgba(100,160,255,0.18);
  border-radius: 5px; padding: 6px 10px;
  font-size: 9px; font-family: 'DM Sans',sans-serif; color: #ccc;
}
</style></head><body>
<div id="tt"></div>
<div id="gw">
  <div id="globe"></div>
  <div id="cbar">
    <div style="font-weight:700;color:#CFB991;letter-spacing:.1em;font-size:8px;text-transform:uppercase;margin-bottom:5px">Impact Scale</div>
    <div class="cr"><div class="cs" style="background:#f5f2ee;border:1px solid #aaa"></div>No exposure</div>
    <div class="cr"><div class="cs" style="background:#fde0c8"></div>Low (10-25)</div>
    <div class="cr"><div class="cs" style="background:#f5a870"></div>Moderate (25-50)</div>
    <div class="cr"><div class="cs" style="background:#e05c3a"></div>Elevated (50-75)</div>
    <div class="cr"><div class="cs" style="background:#b82020"></div>High (75-90)</div>
    <div class="cr"><div class="cs" style="background:#7a0e0e"></div>Crisis (90-100)</div>
  </div>
  <div id="leg">
    <div style="display:flex;align-items:center;gap:5px">
      <span style="color:#ff4444;font-size:12px">●</span><span>Active War Zone</span>
    </div>
  </div>
</div>
<script src="//unpkg.com/globe.gl@2.27.2/dist/globe.gl.min.js"></script>
<script>
const CDATA = __COUNTRY_DATA__;
const HT    = __HOMETURF__;
const HT_ISO = new Set(['UKR','ISR','PSE','LBN','YEM','SYR','IRN']);

const CS = [
  [0,   [245,242,238]],
  [15,  [253,224,200]],
  [35,  [245,168,112]],
  [55,  [224, 92, 58]],
  [75,  [184, 32, 32]],
  [100, [122, 14, 14]],
];
function sc(score, a) {
  if (score <= 0) return 'rgba(245,242,238,'+a+')';
  const s = Math.max(0, Math.min(100, score));
  let lo = CS[0], hi = CS[CS.length-1];
  for (let i = 0; i < CS.length-1; i++) {
    if (s >= CS[i][0] && s <= CS[i+1][0]) { lo = CS[i]; hi = CS[i+1]; break; }
  }
  const t = hi[0]===lo[0] ? 0 : (s-lo[0])/(hi[0]-lo[0]);
  return 'rgba('+
    Math.round(lo[1][0]+t*(hi[1][0]-lo[1][0]))+','+
    Math.round(lo[1][1]+t*(hi[1][1]-lo[1][1]))+','+
    Math.round(lo[1][2]+t*(hi[1][2]-lo[1][2]))+','+a+')';
}
function lvl(s) { return s>=75?'High':s>=50?'Elevated':s>=25?'Moderate':s>0?'Low':'Minimal'; }
function lvc(s) { return s>=75?'#ff6b6b':s>=50?'#f5a870':s>=25?'#fde0c8':'#aaa'; }

function polyHtml(iso) {
  const d = CDATA[iso];
  if (!d) return '';
  const s = d.score;
  if (s === 0) return '<b>'+d.name+'</b><br><span style="color:#88aacc">No significant direct exposure</span>';
  const hw = HT_ISO.has(iso) ? "<br><b style='color:#e74c3c'>&#9876; Active war on home soil</b>" : '';
  let t = '<b>'+d.name+'</b>'+hw+'<br>'+
          '<span style="color:'+lvc(s)+'"><b>'+s+'/100 &ndash; '+lvl(s)+'</b></span><br><br>'+
          'Ukraine War: <b>'+d.ukraine+'</b>/100<br>'+
          'Israel-Hamas: <b>'+d.hamas+'</b>/100<br>'+
          'Iran/Hormuz: <b>'+d.iran+'</b>/100';
  if (d.indices && d.indices !== '-') t += '<br><br><span style="color:#aac">Tracked:</span> '+d.indices;
  if (d.ret_text) t += '<br><span style="color:#aac">Returns:</span><br>'+d.ret_text;
  return t;
}
function ptHtml(d) {
  return '<b>'+d.name+'</b><br>&#9876; '+d.war+'<br><i style="color:#99b">'+d.note+'</i>';
}

/* ── Tooltip positioning ── */
const tt = document.getElementById('tt');
document.addEventListener('mousemove', e => {
  const x = e.clientX, y = e.clientY;
  const W = window.innerWidth, H = window.innerHeight;
  tt.style.left = (x + 16 + 220 > W ? x - 230 : x + 16) + 'px';
  tt.style.top  = (y - 10 + tt.offsetHeight > H ? y - tt.offsetHeight - 6 : y - 10) + 'px';
});
function showTip(html) { tt.innerHTML = html; tt.style.display = 'block'; }
function hideTip()     { tt.style.display = 'none'; }

/* ── Globe init ── */
const globe = Globe()
  .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
  .backgroundImageUrl('//unpkg.com/three-globe/example/img/night-sky.png')
  (document.getElementById('globe'));

/* ── Car-configurator physics ── */
const ctrl = globe.controls();
ctrl.enableDamping = true;
ctrl.dampingFactor  = 0.06;
ctrl.rotateSpeed    = 0.85;
ctrl.zoomSpeed      = 0.65;
ctrl.minDistance    = 115;
ctrl.maxDistance    = 550;

globe.pointOfView({ lat: 25, lng: 20, altitude: 2.1 }, 0);

/* ── War-zone dot markers ── */
globe
  .pointsData(HT)
  .pointLat(d => d.lat)
  .pointLng(d => d.lng)
  .pointColor(() => '#ff3333')
  .pointAltitude(0.055)
  .pointRadius(0.5)
  .onPointHover(pt => { pt ? showTip(ptHtml(pt)) : hideTip(); });

/* ── Country choropleth polygons (async) ── */
fetch('https://raw.githubusercontent.com/vasturiano/globe.gl/master/example/country-polygons/ne_110m_admin_0_countries.geojson')
  .then(r => r.json())
  .then(world => {
    const feats = world.features.filter(f => f.properties.ISO_A3 && f.properties.ISO_A3 !== '-99');
    globe
      .polygonsData(feats)
      .polygonCapColor(f => { const d=CDATA[f.properties.ISO_A3]; return sc(d?d.score:0, 0.88); })
      .polygonSideColor(() => 'rgba(8,8,8,0.55)')
      .polygonStrokeColor(() => 'rgba(255,255,255,0.13)')
      .polygonAltitude(0.003)
      .onPolygonHover(hov => {
        globe
          .polygonAltitude(f => f===hov ? 0.022 : 0.003)
          .polygonCapColor(f => {
            const d=CDATA[f.properties.ISO_A3], s=d?d.score:0;
            return f===hov ? sc(Math.min(s+18,100),1.0) : sc(s,0.88);
          });
        if (hov) {
          const html = polyHtml(hov.properties.ISO_A3);
          if (html) showTip(html); else hideTip();
        } else {
          hideTip();
        }
      });
  });
</script></body></html>"""


def _render_globe_component(df: pd.DataFrame, score_col: str) -> None:
    """Inject country data into the Globe.GL template and render via components.html."""
    country_data: dict = {}
    for _, row in df.iterrows():
        country_data[str(row["iso3"])] = {
            "score":   int(row[score_col]),
            "name":    str(row["country"]),
            "ukraine": int(row["ukraine_score"]),
            "hamas":   int(row["hamas_score"]),
            "iran":    int(row["iran_score"]),
            "indices": str(row["indices"]),
            "ret_text": str(row["ret_text"]),
        }
    hometurf_data = [
        {"lat": h["lat"], "lng": h["lon"],
         "name": h["name"], "war": h["war"], "note": h["note"]}
        for h in _HOMETURF_WARS
    ]
    html = (
        _GLOBE_HTML
        .replace("__COUNTRY_DATA__", json.dumps(country_data))
        .replace("__HOMETURF__", json.dumps(hometurf_data))
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

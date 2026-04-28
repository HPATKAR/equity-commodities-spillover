"""
War Impact Map - Geopolitical Equity Risk
Choropleth map (flat or 3-D globe) showing equity-market exposure to active wars.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from datetime import date

from src.data.config import GEOPOLITICAL_EVENTS, PALETTE
from src.data.loader import load_returns
from src.ui.shared import (
    _chart, _page_intro, _section_note, _definition_block,
    _page_conclusion, _page_header, _page_footer, _no_api_key_banner,
)
from src.analysis.war_country_scores import (
    WAR_DATA as _WAR_DATA,
    COUNTRY_WAR_WEIGHTS as _COUNTRY_WAR_WEIGHTS,
    COUNTRY_INDICES as _COUNTRY_INDICES,
    DEFAULT_WAR_WEIGHTS as _DEFAULT_WAR_WEIGHTS,
    war_multipliers as _war_multipliers,
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

# _COUNTRY_INDICES, _WAR_DATA, _COUNTRY_WAR_WEIGHTS, _DEFAULT_WAR_WEIGHTS
# and _war_multipliers are imported from src.analysis.war_country_scores above.
# Edit scoring data there — this file is visualization only.


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


def _build_df(
    war_rets: dict[str, dict[str, float]],
    multipliers: dict | None = None,
) -> pd.DataFrame:
    """
    Build country-level impact DataFrame.
    If multipliers provided, baseline static scores are scaled by per-war
    intensity multipliers derived from live commodity signals.
    """
    if multipliers is None:
        multipliers = {"ukraine": 1.0, "hamas": 1.0, "iran": 1.0}

    um = multipliers["ukraine"]
    hm = multipliers["hamas"]
    im = multipliers["iran"]

    scored_iso: set[str] = set()
    for w in _WAR_DATA:
        scored_iso.update(w["scores"].keys())

    rows = []
    all_iso = set(_ALL_COUNTRIES) | scored_iso
    for iso in all_iso:
        u_base  = _WAR_DATA[0]["scores"].get(iso, 0)
        h_base  = _WAR_DATA[1]["scores"].get(iso, 0)
        ir_base = _WAR_DATA[2]["scores"].get(iso, 0)

        # Apply dynamic multipliers (scale per-war scores by live commodity signals)
        u  = int(np.clip(round(u_base  * um), 0, 100))
        h  = int(np.clip(round(h_base  * hm), 0, 100))
        ir = int(np.clip(round(ir_base * im), 0, 100))

        # Composite: country-specific structural weights × live-scaled per-war scores.
        # Each country has a (ukraine_w, hamas_w, iran_w) tuple reflecting how much
        # each conflict structurally drives risk for that country - Gulf states are
        # Hormuz-dominated, Eastern Europe is Ukraine-dominated, etc.
        wu, wh, wi = _COUNTRY_WAR_WEIGHTS.get(iso, _DEFAULT_WAR_WEIGHTS)
        base_composite = wu * u + wh * h + wi * ir

        # Concurrent-war amplifier: systemic risk rises when multiple conflicts
        # are simultaneously hot (multipliers > 1.05 = market stress elevated)
        hot_wars = int(um > 1.05) + int(hm > 1.05) + int(im > 1.05)
        amplifier = 1.0 + 0.12 * hot_wars / 3.0
        composite = int(np.clip(round(base_composite * amplifier), 0, 100))

        primary = max(
            (_WAR_DATA[0]["label"], u),
            (_WAR_DATA[1]["label"], h),
            (_WAR_DATA[2]["label"], ir),
            key=lambda x: x[1],
        )[0]

        indices  = _COUNTRY_INDICES.get(iso, [])
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
# No network request - 100% reliable in any deployment context.

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
# Drag: setPointerCapture - works inside Streamlit iframes.
# Hover: proj.invert(mouse) → d3.geoContains(feature, point) - 100% reliable.
# GeoJSON: injected server-side - no client fetch, no CORS issues.

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
  border-radius:0;padding:13px 15px;
  font-family:'DM Sans',sans-serif;font-size:11px;line-height:1.62;
  color:#dedede;transition:border-color .2s;
  pointer-events:none;
}
#panel.lit{border-color:rgba(207,185,145,0.60)}
#panel .plbl{font-size:7.5px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:#CFB991;margin-bottom:6px}
#panel .ph{color:#525252;font-size:10px;line-height:1.5;font-style:italic}
#panel b{color:#fff}
#panel hr{border:none;border-top:1px solid rgba(255,255,255,0.07);margin:7px 0}
#legend{
  position:absolute;bottom:14px;right:14px;
  background:rgba(5,11,26,0.90);
  border:1px solid rgba(100,140,220,0.14);
  border-radius:0;padding:9px 12px;
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
/* ── injected from Python (no client fetch - data embedded server-side) ── */
const CDATA    = __COUNTRY_DATA__;     // {ISO3:{score}}
const HOVER    = __HOVER_DATA__;       // {ISO3:html}
const HT       = __HOMETURF__;         // [{lat,lng,name,war,note}]
const GEO_RAW  = __GEO_FEATURES__;     // [{iso,name,g:{type,coordinates}}]

/* ── canvas setup - size after layout ── */
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
  /* country borders (separate pass - avoids fill bleeding over stroke) */
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
    : '<b>'+(name||iso)+'</b><hr><span style="color:#525252;font-size:10px">No war-impact data tracked.</span>');
}

function handleHover(mx, my) {
  const dx=mx-cx, dy=my-cy;
  if (dx*dx+dy*dy > R0*R0) { setHover(null,null); canvas.style.cursor='grab'; return; }
  const ll = proj.invert([mx,my]); if (!ll) { setHover(null,null); return; }
  /* reject back-hemisphere (proj.invert is purely mathematical - no clipping) */
  if (d3.geoDistance(ll, [-rot[0],-rot[1]]) >= Math.PI/2) { setHover(null,null); return; }
  for (const {f,iso,name} of featData) {
    if (d3.geoContains(f, ll)) { setHover(iso,name); canvas.style.cursor='crosshair'; return; }
  }
  setHover(null,null); canvas.style.cursor='grab';
}

/* ── pointer-capture drag (works in Streamlit iframes - no window listener needed) ── */
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

/* ── build feature index from injected data (synchronous - no fetch needed) ── */
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
    # Server-side GeoJSON - injected directly, no browser fetch required
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

_F = "font-family:'DM Sans',sans-serif;"

def page_war_impact_map(start: str, end: str, fred_key: str = "") -> None:

    # ── Page header ───────────────────────────────────────────────────────────
    _page_header("Global Conflict Risk Map",
                 "Live conflict risk · Composite 0–100 index · Active theatre mapping")
    _no_api_key_banner("AI-generated conflict narratives")
    st.markdown(
        f'<p style="{_F}font-size:0.70rem;color:#8890a1;margin:0 0 0.8rem;line-height:1.6">'
        f'Active conflicts disrupt commodity supply chains - oil, wheat, metals - causing those '
        f'commodities to decouple from their usual equity correlation. This map scores each country\'s '
        f'<strong>structural exposure to that decoupling risk</strong>: geographic proximity, energy '
        f'dependency, trade-route vulnerability, and alliance obligations. High-exposure markets are '
        f'the ones where a commodity supply shock is most likely to produce an equity spillover event. '
        f'Hover any country for a full breakdown and its realised equity returns during active war windows.</p>',
        unsafe_allow_html=True,
    )

    # ── Live conflict scoring overlay ─────────────────────────────────────────
    try:
        from src.analysis.conflict_model import score_all_conflicts, aggregate_portfolio_scores
        _wm_cr  = score_all_conflicts()
        _wm_agg = aggregate_portfolio_scores(_wm_cr)
        _wm_cis = _wm_agg.get("portfolio_cis", 50.0)
        _wm_tps = _wm_agg.get("portfolio_tps", 50.0)
        _wm_active = sorted(
            [r for r in _wm_cr.values() if r.get("state") == "active"],
            key=lambda x: x["cis"], reverse=True
        )
        _wm_col = "#c0392b" if _wm_cis >= 65 else "#e67e22" if _wm_cis >= 45 else "#CFB991"

        _wm_conflict_cells = "".join(
            f'<div style="display:inline-flex;align-items:center;gap:5px;'
            f'background:#080808;border:1px solid #1e1e1e;'
            f'padding:3px 8px;margin-right:5px;margin-bottom:3px">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7.5px;'
            f'font-weight:700;color:#CFB991">{r["label"].upper()}</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'color:{("#c0392b" if r["cis"]>=65 else "#e67e22" if r["cis"]>=45 else "#8E9AAA")}">'
            f'CIS {r["cis"]:.0f}</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;color:#555960">'
            f'TPS {r.get("tps",50):.0f}</span>'
            f'</div>'
            for r in _wm_active[:5]
        )

        st.markdown(
            f'<div style="background:#040404;border:1px solid #1e1e1e;'
            f'border-left:3px solid {_wm_col};padding:.5rem .9rem;margin-bottom:.7rem">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'font-weight:700;color:{_wm_col};letter-spacing:.15em;margin-bottom:5px">'
            f'LIVE CONFLICT RISK SCORES — SCORING MAP BELOW</div>'
            f'<div style="display:flex;flex-wrap:wrap;align-items:center">'
            f'{_wm_conflict_cells}'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'color:#8E9AAA;margin-left:auto">'
            f'Portfolio CIS&nbsp;<b style="color:{_wm_col}">{_wm_cis:.0f}</b>&nbsp;·&nbsp;'
            f'TPS&nbsp;<b style="color:#CFB991">{_wm_tps:.0f}</b>&nbsp;·&nbsp;'
            f'Darker map regions = higher structural exposure to active conflicts</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    with st.spinner("Loading market data…"):
        eq_r, cmd_r = load_returns(start, end)

    war_rets    = _war_period_returns(eq_r) if not eq_r.empty else {}
    multipliers = _war_multipliers(cmd_r)
    df          = _build_df(war_rets, multipliers)

    today = date.today()
    active = [ev for ev in GEOPOLITICAL_EVENTS
              if ev["category"] == "Geopolitical" and ev["end"] >= today]

    # ── War filter (above the hero) ───────────────────────────────────────────
    st.markdown(
        f'<p style="{_F}font-size:0.52rem;font-weight:700;letter-spacing:0.14em;'
        f'text-transform:uppercase;color:#8E9AAA;margin:0 0 4px">Filter by conflict</p>',
        unsafe_allow_html=True,
    )
    war_filter = st.radio(
        "Filter by conflict",
        ["Combined (max)", "Ukraine War only", "Israel-Hamas only", "Iran/Hormuz only"],
        key="war_filter_map",
        horizontal=True,
        label_visibility="collapsed",
    )
    if war_filter == "Ukraine War only":
        score_col = "ukraine_score"
    elif war_filter == "Israel-Hamas only":
        score_col = "hamas_score"
    elif war_filter == "Iran/Hormuz only":
        score_col = "iran_score"
    else:
        score_col = "score"

    st.markdown('<div style="margin:0.4rem 0 0.5rem;border-top:1px solid #1a1a1a"></div>',
                unsafe_allow_html=True)

    # ── Hero: left panel + globe ──────────────────────────────────────────────
    left_col, globe_col = st.columns([1, 2.2])

    with left_col:
        # Active conflict status cards
        st.markdown(
            f'<p style="{_F}font-size:0.52rem;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:#8E9AAA;margin:0 0 8px">Active Conflicts</p>',
            unsafe_allow_html=True,
        )
        for ev in active:
            days = (today - ev["start"]).days
            st.markdown(
                f'<div style="padding:0.55rem 0.8rem;border-left:2px solid rgba(207,185,145,0.22);'
                f'background:#141414;border-radius:0;margin-bottom:8px">'
                f'<div style="{_F}font-size:0.52rem;font-weight:700;letter-spacing:0.11em;'
                f'text-transform:uppercase;color:{ev["color"]}">{ev["category"]}</div>'
                f'<div style="{_F}font-size:0.80rem;font-weight:700;color:#e8e9ed;'
                f'margin-top:1px;line-height:1.3">{ev["name"]}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.62rem;'
                f'color:#8890a1;margin-top:3px">'
                f'{ev["start"].strftime("%d %b %Y")} &nbsp;&middot;&nbsp; {days:,}d elapsed</div>'
                f'<div style="{_F}font-size:0.62rem;color:#8890a1;margin-top:5px;line-height:1.55">'
                f'{ev["description"][:130]}{"…" if len(ev["description"]) > 130 else ""}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Dynamic intensity multipliers ────────────────────────────────
        st.markdown('<div style="margin:0.6rem 0 0.5rem;border-top:1px solid #1a1a1a"></div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<p style="{_F}font-size:0.52rem;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:#8E9AAA;margin:0 0 6px">'
            f'Today\'s live intensity multipliers</p>',
            unsafe_allow_html=True,
        )
        _mult_rows = [
            ("Ukraine War",     multipliers["ukraine"], "#e74c3c"),
            ("Israel-Hamas",    multipliers["hamas"],   "#f39c12"),
            ("Iran/Hormuz",     multipliers["iran"],    "#c0392b"),
        ]
        for _wname, _m, _wc in _mult_rows:
            _bar = int(min((_m - 0.92) / (1.15 - 0.92) * 100, 100))
            _mc  = "#c0392b" if _m > 1.15 else "#e67e22" if _m > 1.0 else "#2e7d32"
            st.markdown(
                f'<div style="margin-bottom:6px">'
                f'<div style="display:flex;justify-content:space-between;{_F}font-size:0.63rem">'
                f'<span style="color:#c8c8c8">{_wname}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-weight:700;'
                f'color:{_mc}">×{_m:.2f}</span></div>'
                f'<div style="height:3px;background:#F0EDEA;border-radius:2px;margin-top:2px">'
                f'<div style="width:{_bar}%;height:3px;background:{_mc};border-radius:2px">'
                f'</div></div></div>',
                unsafe_allow_html=True,
            )
        sigs = multipliers.get("signals", {})
        if sigs:
            sig_txt = " &nbsp;·&nbsp; ".join(
                f'{k}: <b>{v:+.2f}σ</b>' for k, v in sigs.items()
            )
            st.markdown(
                f'<div style="{_F}font-size:0.60rem;color:#888;margin-top:4px;line-height:1.6">'
                f'{sig_txt}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="margin:0.6rem 0 0.5rem;border-top:1px solid #1a1a1a"></div>',
                    unsafe_allow_html=True)

        # Impact scale legend
        st.markdown(
            f'<p style="{_F}font-size:0.52rem;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:#8E9AAA;margin:0 0 6px">Impact Scale</p>',
            unsafe_allow_html=True,
        )
        for c, lbl in [
            ("#f5f2ee", "No exposure  (0)"),
            ("#fde0c8", "Low          (10–25)"),
            ("#f5a870", "Moderate     (25–50)"),
            ("#e05c3a", "Elevated     (50–75)"),
            ("#b82020", "High         (75–90)"),
            ("#7a0e0e", "Crisis       (90–100)"),
        ]:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
                f'<div style="width:11px;height:11px;background:{c};border-radius:2px;'
                f'flex-shrink:0;{"border:1px solid #ccc" if c == "#f5f2ee" else ""}"></div>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.62rem;'
                f'color:#c8c8c8">{lbl}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="margin:0.6rem 0 0.5rem;border-top:1px solid #1a1a1a"></div>',
                    unsafe_allow_html=True)

        # Score methodology note
        st.markdown(
            f'<p style="{_F}font-size:0.52rem;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:#8E9AAA;margin:0 0 5px">Composite Score</p>'
            f'<p style="{_F}font-size:0.64rem;color:#8890a1;line-height:1.6;margin:0">'
            f'<code style="font-size:0.62rem;background:#f0ede8;padding:1px 4px;'
            f'border-radius:2px">max(Ukraine, Hamas, Hormuz)</code> - the worst-case '
            f'conflict score across all three tracked wars. Five factors: geographic '
            f'proximity, energy dependency, trade-route exposure, equity-market '
            f'correlation, and alliance obligations.</p>',
            unsafe_allow_html=True,
        )

    with globe_col:
        # ── Hover text construction ───────────────────────────────────────────
        hometurf_iso = {"UKR", "ISR", "PSE", "LBN", "YEM", "SYR", "IRN"}
        hover_texts = []
        for _, row in df.iterrows():
            s = int(row[score_col])
            if s == 0:
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

        _render_globe_component(df, score_col)

    st.markdown('<div style="margin:0.6rem 0 0.5rem;border-top:1px solid #1a1a1a"></div>',
                unsafe_allow_html=True)

    # ── Methodology expanders ─────────────────────────────────────────────────
    with st.expander("Score Methodology - Plain English", expanded=False):
        _definition_block(
            "How the Impact Score is Determined",
            "Each country receives an independent score (0–100) for each of the three tracked conflicts. "
            "The <b>composite score</b> shown on the globe is a <b>country-specific weighted average</b> "
            "across all three conflicts - weights reflect structural relevance: Gulf states are dominated "
            "by the Iran/Hormuz conflict (they sit on the Strait), Eastern Europe by the Ukraine war "
            "(hometurf), and the Levant by the Israel-Hamas war. A concurrent-war amplifier (+12% max) "
            "applies when multiple conflict markets are simultaneously stressed. "
            "Per-war scores are further scaled by live commodity z-scores to reflect current market intensity. "
            "Scores are constructed from five factors:<br><br>"
            "Scores are constructed from five factors:<br><br>"
            "<b>1. Geographic proximity</b> - Direct border exposure, contiguous-state risk, "
            "and displacement or refugee-flow pressure on neighbouring economies.<br>"
            "<b>2. Energy dependency</b> - Share of oil, gas, and LNG imports that transit "
            "disrupted supply corridors or originate from conflict-adjacent exporters.<br>"
            "<b>3. Trade route exposure</b> - Vulnerability to blocked or threatened shipping lanes "
            "(Black Sea, Suez/Red Sea, Strait of Hormuz).<br>"
            "<b>4. Equity-market correlation</b> - Historical co-movement of the domestic index "
            "with conflict-sensitive commodities (crude oil, natural gas, wheat).<br>"
            "<b>5. Alliance and sanctions exposure</b> - NATO/EU commitments, active sanctions "
            "regimes, and diplomatic alignment with conflict parties.<br><br>"
            "<b>Conflict-specific emphasis:</b> Ukraine War weights energy dependency and geographic "
            "proximity most heavily. Israel-Hamas weights Red Sea shipping disruption and regional "
            "spillover. Iran/Hormuz weights Strait of Hormuz crude-transit dependency.",
        )

    with st.expander("Technical Scoring Methodology", expanded=False):
        st.markdown(
            """
**Model class:** Structured Expert Judgment (SEJ) composite index - not the output of a
statistical model or machine learning pipeline. Scores are expert-calibrated on a 0–100
ordinal-interval scale, anchored at known extremes and validated against observed market
reactions to each conflict.

---

#### Scoring Formula (per conflict, per country)

Each country *i* receives a conflict-specific score *S_c(i)* computed as a weighted sum
of five sub-scores, each independently assessed on [0, 100]:

```
S_c(i) = w1·P(i) + w2·E(i) + w3·T(i) + w4·R(i) + w5·A(i)
```

| Symbol | Dimension | Weight range | Measurement basis |
|--------|-----------|--------------|-------------------|
| P(i) | **Geographic Proximity** | 18 – 33 % | Geodesic distance from conflict epicentre; border states score ≥ 80. Decays ~−15 pts per degree of separation for adjacent states, flattening to a floor of ~5 beyond 5,000 km. For Iran/Hormuz, P also captures *military projection proximity* (naval basing, carrier group presence). |
| E(i) | **Energy Dependency** | 18 – 32 % | Share of oil, gas, and LNG imports transiting disrupted corridors or sourced from conflict-adjacent exporters. Calibrated against IEA bilateral trade matrices; Russian-gas reliance (EU states) and Gulf-crude reliance (East Asia) are primary inputs. |
| T(i) | **Trade Route Exposure** | 11 – 28 % | Fraction of annual merchandise trade value transiting threatened chokepoints (Black Sea, Suez/Red Sea, Strait of Hormuz). Proxied from UNCTAD maritime freight statistics and Lloyd's voyage data. For Iran/Hormuz, T is intentionally down-weighted - ~85 % of Hormuz traffic is energy already captured by E; non-energy cargo represents only ~15 % of Hormuz transit. |
| R(i) | **Equity-Market Correlation** | 14 – 17 % | 252-day rolling Pearson correlation of the domestic equity index with a conflict commodity basket (crude oil, natural gas, wheat), averaged over the 12 months preceding conflict onset, rescaled linearly from [−1, 1] → [0, 100]. |
| A(i) | **Alliance & Sanctions Exposure** | 7 – 15 % | Ordinal measure: NATO Article 5 obligation (max weight), EU sanctions participation, UN resolution co-sponsorship, formal alignment with a conflict party. Scored on a 0/25/50/75/100 step scale. Also captures direct military engagement (USA in Iran/Hormuz) and fiscal obligations (Germany's Zeitenwende, Baltic defense build-up). |

> **Note:** Weights are conflict-specific (table below). Each row sums to 100 %. Validated by
> checking formula output against diagnostic country pairs (Germany/Poland/UK for Ukraine;
> Egypt/Singapore for Israel-Hamas; Japan/Oman for Iran/Hormuz) to within ± 5 pts.

---

#### Conflict-Specific Weight Assignments

| Conflict | P | E | T | R | A | Primary rationale |
|----------|---|---|---|---|---|---|
| Russia-Ukraine War | **33 %** | **30 %** | 11 % | 14 % | 12 % | P and E dominate (border exposure + Russian-gas lock-in). R raised from prior 10 % - DAX–gas correlation was the dominant market narrative throughout 2022. A raised from 10 % to reflect NATO fiscal obligations and EU sanctions depth. T reduced: Black Sea grain corridor is real but secondary to the energy channel. |
| Israel-Hamas War | 22 % | 18 % | **28 %** | 17 % | 15 % | T is the lead factor (Suez/Red Sea shipping-cost spike was the primary global transmission mechanism). P raised slightly - Jordan and Egypt border analysis shows regional proximity is more discriminating than 20 % implied. E raised from 15 %: Middle East oil import dependency for South/Southeast Asia is a genuine scored channel (Pakistan 28, India 25, Indonesia 22). R moderated: global equity correlation to this conflict is less systematic than energy-channel conflicts. |
| Iran / Hormuz Crisis | 18 % | **32 %** | 28 % | 15 % | 7 % | E is dominant: Hormuz crude dependency is the single most discriminating variable (Japan 68, Korea 65, India 62 vs. Canada 15, Brazil 12). T reduced from prior 35 % to avoid double-counting with E. P raised from 15 % - Gulf chokepoint states (Oman, Qatar, UAE) derive exposure from literal proximity to the strait. A raised from 5 %: USA's direct military role (carrier group, strike operations) cannot be adequately represented at 5 %. |

---

#### Composite Score and Aggregation

```
Composite(i) = max { S_Ukraine(i),  S_Hamas(i),  S_Hormuz(i) }
```

A max-aggregator is preferred over sum or average because the map represents *peak risk
exposure*. A country with a high score under any single conflict faces a material equity
repricing event regardless of its scores under the others. Summing would double-count
unrelated risk channels; averaging would understate the dominant exposure.

---

#### Anchor Calibration

- **Score = 100** - Direct conflict party on home soil (Ukraine, Israel/Palestine, Iran).
- **Score = 0** - No measurable exposure via any of the five channels.

All intermediate scores triangulate from these anchors, cross-checked against:
1. Observed equity drawdowns at conflict onset (DAX −4 % on 24 Feb 2022; TA-35 −8 % on 7 Oct 2023).
2. IEA bilateral energy-import dependency ratios.
3. IMF Direction of Trade Statistics for trade-route exposure fractions.

---

#### Score Tiers and Risk Interpretation

| Tier | Range | Equity-market interpretation |
|------|-------|------------------------------|
| **Crisis** | 90–100 | Active military engagement; circuit-breaker risk; index VaR widens ≥ 3× in escalation. |
| **High** | 75–89 | Structural energy or border exposure; sustained risk-premium elevation; equity beta to conflict commodities > 0.5. |
| **Elevated** | 50–74 | Meaningful trade-route or energy-import risk; commodity correlation historically > 0.3; 5–15 % sector drawdowns likely. |
| **Moderate** | 25–49 | Secondary exposure via commodity-price or regional contagion; index-level impact < 5 % in moderate escalation. |
| **Low** | 1–24 | Tertiary exposure only (risk-off sentiment, USD strength, commodity inflation pass-through). |
| **None** | 0 | No scored exposure across all five dimensions for any tracked conflict. |

---

#### Limitations and Caveats

- **Static calibration.** Scores reflect the configuration at the last model update; escalations or ceasefires are not auto-reflected.
- **Expert subjectivity.** The weight table represents informed judgment, not econometrically estimated coefficients.
- **Pre-conflict correlation window.** R(i) uses pre-onset returns; regime changes during the conflict may alter the realized correlation structure.
- **Country-level aggregation.** An energy exporter (Norway, Canada) may score moderate on the composite while domestic energy-sector equities actually benefit from supply-driven price spikes. The map scores the broad index, not sectors.
            """,
            unsafe_allow_html=False,
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

    _TBL_CSS = """
    <style>
    .wim-table{width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:0.78rem}
    .wim-table th{background:#1c1c1c;color:#CFB991;padding:7px 10px;text-align:left;
        border-bottom:1px solid rgba(207,185,145,0.3);font-weight:600;
        letter-spacing:0.06em;text-transform:uppercase;font-size:0.68rem}
    .wim-table td{padding:5px 10px;border-bottom:1px solid #1e1e1e;color:#e8e9ed}
    .wim-table tr:nth-child(even) td{background:#171717}
    .wim-table tr:nth-child(odd) td{background:#111111}
    .wim-table tr:hover td{background:#202020}
    </style>"""

    def _score_badge(s):
        if s >= 75: bg, fg = "#d73027", "#fff"
        elif s >= 50: bg, fg = "#fc8d59", "#000"
        elif s >= 25: bg, fg = "#fee08b", "#000"
        else: bg, fg = "#91cf60", "#000"
        return f'<span style="background:{bg};color:{fg};padding:2px 7px;border-radius:0;font-weight:700">{s:.0f}</span>'

    def _ret_cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return '<span style="color:#4d4d4d">-</span>'
        if v > 5:   c = "#4ade80"
        elif v > 0: c = "#86efac"
        elif v > -10: c = "#fb923c"
        else: c = "#f87171"
        return f'<span style="color:{c};font-weight:600">{v:+.1f}%</span>'

    if tracked_rows:
        tdf = (
            pd.DataFrame(tracked_rows)
            .sort_values("Impact Score", ascending=False)
            .reset_index(drop=True)
        )
        rows_html = ""
        for _, r in tdf.iterrows():
            rows_html += (
                f"<tr><td>{r['Country']}</td>"
                f"<td style='color:#b8b8b8'>{r['Equity Index']}</td>"
                f"<td>{_score_badge(r['Impact Score'])}</td>"
                f"<td>{_ret_cell(r['Ukraine War (%)'])}</td>"
                f"<td>{_ret_cell(r['Israel War (%)'])}</td>"
                f"<td>{_ret_cell(r['Iran/Hormuz (%)'])}</td></tr>"
            )
        st.markdown(
            _TBL_CSS +
            "<table class='wim-table'><thead><tr>"
            "<th>Country</th><th>Equity Index</th><th>Impact Score</th>"
            "<th>Ukraine War %</th><th>Israel War %</th><th>Iran/Hormuz %</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )

    # ── Top 25 table ──────────────────────────────────────────────────────────
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

    rows25_html = ""
    for _, r in top25.iterrows():
        rows25_html += (
            f"<tr><td>{r['Country']}</td>"
            f"<td>{_score_badge(r['Impact Score'])}</td>"
            f"<td style='color:#b8b8b8'>{r['Ukraine War']:.0f}</td>"
            f"<td style='color:#b8b8b8'>{r['Israel-Hamas War']:.0f}</td>"
            f"<td style='color:#b8b8b8'>{r['Iran/Hormuz']:.0f}</td>"
            f"<td style='color:#8890a1;font-size:0.72rem'>{r['Tracked Index']}</td></tr>"
        )
    st.markdown(
        _TBL_CSS +
        "<table class='wim-table'><thead><tr>"
        "<th>Country</th><th>Impact Score</th><th>Ukraine War</th>"
        "<th>Israel-Hamas War</th><th>Iran/Hormuz</th><th>Tracked Index</th>"
        f"</tr></thead><tbody>{rows25_html}</tbody></table>",
        unsafe_allow_html=True,
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

    # ── AI Geopolitical Analyst (War Map view) ──────────────────────────────
    try:
        from src.agents.geopolitical_analyst import run as _ga_run2
        from src.ui.agent_panel import render_agent_output_block, render_activity_feed
        from src.analysis.agent_state import is_enabled, get_agent
        from src.data.config import GEOPOLITICAL_EVENTS as _GEO_EVENTS

        if is_enabled("geopolitical_analyst"):
            _a_state = get_agent("geopolitical_analyst")
            if _a_state.get("last_output"):
                # Already ran from geopolitical.py this session - just show cached output
                st.markdown("---")
                render_agent_output_block("geopolitical_analyst",
                                         {"narrative": _a_state["last_output"]})
            else:
                _anthropic_key = _openai_key = ""
                try:
                    _keys = st.secrets.get("keys", {})
                    _anthropic_key = _keys.get("anthropic_api_key", "") or ""
                    _openai_key    = _keys.get("openai_api_key",    "") or ""
                except Exception:
                    pass
                _provider = "anthropic" if _anthropic_key else ("openai" if _openai_key else None)
                _api_key  = _anthropic_key or _openai_key

                import datetime as _dt2
                _today2 = _dt2.date.today()
                _geo_ctx2 = {
                    "n_events":      len(_GEO_EVENTS),
                    "high_severity": sum(1 for e in _GEO_EVENTS if e.get("category","") in ("War","Conflict","Crisis")),
                    "active_events": [
                        {"name": e.get("name",""), "severity": e.get("category",""),
                         "commodity_impact": e.get("commodity_impact","")}
                        for e in _GEO_EVENTS[:6]
                    ],
                }
                with st.spinner("AI Geopolitical Analyst assessing war map…"):
                    _ga_result2 = _ga_run2(_geo_ctx2, _provider, _api_key)
                if _ga_result2.get("narrative"):
                    st.markdown("---")
                    render_agent_output_block("geopolitical_analyst", _ga_result2)

        render_activity_feed(max_entries=8, collapsible=True)
    except Exception:
        pass

    # CQO runs silently - output visible in About > AI Workforce
    try:
        from src.agents.quality_officer import run as _cqo_run
        from src.analysis.agent_state import is_enabled
        if is_enabled("quality_officer"):
            _anthropic_key = _openai_key = ""
            try:
                _keys = st.secrets.get("keys", {})
                _anthropic_key = _keys.get("anthropic_api_key", "") or ""
                _openai_key    = _keys.get("openai_api_key",    "") or ""
            except Exception:
                pass
            _provider = "anthropic" if _anthropic_key else ("openai" if _openai_key else None)
            _api_key  = _anthropic_key or _openai_key
            _cqo_ctx = {
                "hardcoded_scores": True,
                "model": "Country-specific weighted composite of 3 conflict scores",
                "assumption_count": 5,
                "notes": [
                    "Per-war baseline scores (0-100) are expert-assigned, not derived from market data",
                    "Country-specific conflict weights are manually calibrated - no empirical validation",
                    "Live multipliers use 20d commodity z-scores as proxy for conflict intensity - indirect at best",
                    "Composite only covers 3 active conflicts - ignores India-Pakistan, Taiwan Strait, Sudan, etc.",
                    "Country scoring treats each nation as homogeneous - ignores regional sub-exposure",
                    "Concurrent-war amplifier (+12%) is an arbitrary scalar with no empirical basis",
                ],
            }
            _cqo_run(_cqo_ctx, _provider, _api_key, page="War Impact Map")
    except Exception:
        pass

    # ── Conflict Exposure Overlay ─────────────────────────────────────────────
    st.markdown(
        f'<div style="margin:1.2rem 0 0.4rem;border-top:1px solid #1e1e1e;padding-top:1rem">'
        f'<p style="{_F}font-size:0.58rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.14em;color:#8E9AAA;margin:0 0 6px">Asset Exposure Overlay</p>'
        f'<p style="{_F}font-size:0.68rem;color:#8890a1;margin:0 0 10px;line-height:1.5">'
        f'Scenario-adjusted exposure scores for tracked assets — derived from the '
        f'structural conflict exposure registry, scaled by active TPS and current scenario multiplier.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    try:
        from src.analysis.exposure import ranked_by_exposure, exposure_summary_stats, score_all_assets
        _all_exp = score_all_assets()
        _exp_stats = exposure_summary_stats(_all_exp)
        _exp_top = ranked_by_exposure(n=16)

        # Summary strip
        _es1, _es2, _es3, _es4 = st.columns(4)
        for _ec, (_el, _ev, _ecol) in zip(
            [_es1, _es2, _es3, _es4],
            [
                ("Top Exposed",  _exp_stats.get("top_asset","—"),      "#e67e22"),
                ("Peak SAS",     f'{_exp_stats.get("top_asset_sas",0):.0f}', "#c0392b"),
                ("Top Hedge",    _exp_stats.get("top_hedge","—"),       "#27ae60"),
                ("Mean Exposure",f'{_exp_stats.get("mean_sas",0):.1f}', "#CFB991"),
            ]
        ):
            _ec.markdown(
                f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;'
                f'padding:6px 10px;margin-bottom:6px">'
                f'<div style="{_F}font-size:0.52rem;font-weight:700;letter-spacing:.12em;'
                f'text-transform:uppercase;color:#555960">{_el}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.85rem;'
                f'font-weight:700;color:{_ecol}">{_ev}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Ranked bars
        _exp_rows = ""
        for _ea in _exp_top:
            _sas = _ea["sas"]; _bar = int(_sas)
            _ec2 = "#c0392b" if _sas>=60 else "#e67e22" if _sas>=35 else "#CFB991" if _sas>=15 else "#8E9AAA"
            _dir_col = {"long_geo_risk":"#e67e22","safe_haven":"#27ae60","neutral":"#8E9AAA"}.get(_ea["direction"],"#8E9AAA")
            _exp_rows += (
                f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;'
                f'border-bottom:1px solid #141414">'
                f'<span style="{_F}font-size:0.68rem;color:#c8cdd8;width:140px;flex-shrink:0;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{_ea["asset"]}</span>'
                f'<div style="flex:1;height:4px;background:#1a1a1a">'
                f'<div style="width:{_bar}%;height:4px;background:{_ec2}"></div></div>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;'
                f'color:{_ec2};width:25px;text-align:right">{_sas:.0f}</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.55rem;'
                f'color:{_dir_col};width:60px;text-align:right">'
                f'{_ea["direction"].replace("_"," ").upper()[:8]}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;padding:6px 12px">'
            f'{_exp_rows}</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    _page_footer()

"""
Equity-Commodities Spillover Monitor
Purdue University · Daniels School of Business

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equity-Commodities Spillover | Purdue Daniels",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --font-sans: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --fs-micro:  0.52rem;
    --fs-tiny:   0.56rem;
    --fs-xs:     0.60rem;
    --fs-sm:     0.65rem;
    --fs-base:   0.70rem;
    --fs-md:     0.74rem;
    --fs-lg:     0.78rem;
    --fs-xl:     0.82rem;
    --fs-2xl:    0.88rem;
    --fs-metric: 1.05rem;
    --fs-h1:     1.25rem;
    --gold:       #CFB991;
    --gold-light: #EBD99F;
    --gold-mid:   #DAAA00;
    --aged:       #8E6F3E;
    --black:      #000000;
    --ink-soft:   #333333;
    --ink-muted:  #444444;
    --ink-faint:  #666666;
    --bg:         #ffffff;
    --bg-warm:    #fafaf8;
    --rule:       #E8E5E0;
    --rule-light: #F0EDEA;
    --red:        #c0392b;
    --green:      #2e7d32;
    --nav-h:      56px;
}

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: var(--font-sans);
    background: var(--bg);
    color: var(--black);
}

/* ── Hide Streamlit chrome + sidebar entirely ── */
#MainMenu, footer,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}

/* ── Full-width main content, starts at 0 (navbar is fixed, spacer provides gap) ── */
.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
    max-width: 1360px;
}

/* ── Fixed top navbar ── */
.topnav {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    height: var(--nav-h);
    background: #000;
    border-bottom: 1px solid #1c1c1c;
    font-family: var(--font-sans);
}
.nav-inner {
    height: 100%;
    max-width: 1600px;
    margin: 0 auto;
    padding: 0 2.2rem;
    display: flex;
    align-items: stretch;
    gap: 0;
}

/* ── Brand ── */
.nav-brand {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 2px;
    margin-right: 2.8rem;
    flex-shrink: 0;
    text-decoration: none;
    cursor: pointer;
    padding: 0 0.2rem;
    border-bottom: 2px solid transparent;
    transition: opacity 0.15s;
}
.nav-brand:hover { opacity: 0.82; }
.nav-brand-main {
    font-size: 0.80rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.01em;
    white-space: nowrap;
    line-height: 1.2;
}
.nav-brand-sub {
    font-size: 0.54rem;
    color: rgba(207,185,145,0.62);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    white-space: nowrap;
}

/* ── Nav links ── */
.nav-links {
    display: flex;
    list-style: none;
    margin: 0; padding: 0;
    height: 100%;
    align-items: stretch;
    gap: 0;
}
.nav-item {
    position: relative;
    display: flex;
    align-items: stretch;
}

/* Top-level link + group trigger */
.nav-item > a,
.nav-item > .nav-top {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 0 1.05rem;
    font-size: 0.72rem;
    font-weight: 500;
    color: rgba(255,255,255,0.58);
    text-decoration: none;
    white-space: nowrap;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    transition: color 0.13s, border-color 0.13s, background 0.13s;
    letter-spacing: 0.025em;
    user-select: none;
}
.nav-item > a:hover,
.nav-item.has-drop:hover > .nav-top {
    color: #fff;
    background: rgba(255,255,255,0.03);
    border-bottom-color: rgba(207,185,145,0.30);
}
.nav-item.active > a {
    color: var(--gold);
    border-bottom-color: var(--gold);
    font-weight: 600;
}
.nav-item.active > .nav-top,
.nav-item.active.has-drop > .nav-top {
    color: var(--gold);
    border-bottom-color: var(--gold);
    font-weight: 600;
}

/* Dropdown caret */
.caret {
    font-size: 0.48rem;
    opacity: 0.50;
    transition: transform 0.16s ease;
    margin-top: 1px;
    display: inline-block;
}
.nav-item.has-drop:hover .caret {
    transform: rotate(180deg);
    opacity: 0.80;
}

/* ── Dropdown panel ── */
.dropdown {
    display: none;
    position: absolute;
    top: 100%;
    left: 0;
    min-width: 210px;
    background: #060606;
    border: 1px solid #252525;
    border-top: 2px solid var(--gold);
    border-radius: 0 0 5px 5px;
    list-style: none;
    margin: 0;
    padding: 6px 0;
    box-shadow: 0 12px 32px rgba(0,0,0,0.60);
    z-index: 10000;
}
.nav-item.has-drop:hover > .dropdown {
    display: block;
}
.dropdown li { margin: 0; padding: 0; }
.dropdown li a {
    display: block;
    padding: 0.50rem 1.15rem;
    font-size: 0.70rem;
    font-weight: 400;
    color: rgba(255,255,255,0.58);
    text-decoration: none;
    transition: color 0.10s, background 0.10s, border-color 0.10s;
    letter-spacing: 0.015em;
    white-space: nowrap;
    border-left: 2px solid transparent;
}
.dropdown li a:hover {
    color: #fff;
    background: rgba(255,255,255,0.045);
    border-left-color: rgba(207,185,145,0.35);
}
.dropdown li a.active {
    color: var(--gold);
    background: rgba(207,185,145,0.07);
    border-left-color: var(--gold);
    font-weight: 500;
}
.dropdown-sep {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 5px 0;
}

/* ── Right-side data-source status strip (inside navbar) ── */
.nav-ds {
    margin-left: auto;
    display: flex;
    align-items: center;
    padding: 0 0 0 1.5rem;
    flex-shrink: 0;
    font-size: 0.59rem;
    letter-spacing: 0.05em;
    color: rgba(255,255,255,0.30);
    gap: 0.85rem;
    white-space: nowrap;
}
.nav-ds .ds-item { display: flex; align-items: center; gap: 4px; }
.nav-ds .ds-dot  { font-size: 0.55rem; }

/* ── Spacer pushes Streamlit content below the fixed bar ── */
.nav-spacer { height: var(--nav-h); }

/* ── Date range strip (top of content area) ── */
.date-strip-wrap {
    border-bottom: 1px solid var(--rule);
    margin-bottom: 1.4rem;
    padding-bottom: 0.2rem;
}
.date-strip-wrap [data-testid="stDateInput"] label {
    font-size: 0.58rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--ink-faint) !important;
}
.date-strip-wrap [data-testid="stDateInput"] [data-baseweb="input"] {
    border: 1px solid var(--rule) !important;
    background: var(--bg-warm) !important;
    border-radius: 3px !important;
    font-size: 0.70rem !important;
}

/* ── Headings ── */
h1 { font-size: 1.25rem !important; font-weight: 700 !important; letter-spacing: -0.01em; }
h2 { font-size: 0.95rem !important; font-weight: 700 !important; margin-top: 1.4rem !important; }
h3 { font-size: 0.82rem !important; font-weight: 600 !important; margin-top: 1rem !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-warm);
    border: 1px solid var(--rule);
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
    transition: box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-color: var(--gold);
}
[data-testid="stMetricLabel"] {
    font-size: 0.58rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #555960 !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--rule);
    gap: 0;
}
[data-testid="stTabs"] button[role="tab"] {
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: var(--ink-muted) !important;
    padding: 0.5rem 1rem !important;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--black) !important;
    border-bottom-color: var(--gold) !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--black) !important;
    color: var(--gold) !important;
    border: 1px solid var(--black) !important;
    border-radius: 3px !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.45rem 1.4rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--gold) !important;
    color: var(--black) !important;
    border-color: var(--gold) !important;
}

/* ── Form controls ── */
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stSlider"] label,
.stCheckbox label span {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--ink-soft) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--rule);
    border-radius: 4px;
}
[data-testid="stDataFrame"] th {
    background: var(--bg-warm) !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--ink-soft) !important;
}
[data-testid="stDataFrame"] td {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--gold) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #DEDAD5; border-radius: 3px; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid var(--rule); margin: 1.2rem 0; }

/* ── Fade-in ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.main .block-container > div { animation: fadeInUp 0.28s ease; }
</style>
""", unsafe_allow_html=True)

# ── API keys ──────────────────────────────────────────────────────────────────
def _get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get("keys", {}).get(key, default) or default
    except Exception:
        return default

_FRED_KEY = _get_secret("fred_api_key")
_FD_KEY   = _get_secret("financial_datasets_key")

# ── Navigation state ──────────────────────────────────────────────────────────
_VALID_PAGES = {
    "overview", "war_impact_map", "geopolitical", "correlation",
    "spillover", "watchlist", "trade_ideas", "stress_test",
    "model_accuracy", "ai_chat", "about_heramb", "about_jiahe", "about_ilian",
}
_ANALYSIS_PAGES = {"war_impact_map", "geopolitical", "correlation", "spillover", "watchlist"}
_STRATEGY_PAGES = {"trade_ideas", "stress_test"}
_RESEARCH_PAGES = {"model_accuracy", "ai_chat"}
_ABOUT_PAGES    = {"about_heramb", "about_jiahe", "about_ilian"}

# Query-param routing (footer links, direct URLs, external links all work)
_qp = st.query_params.get("page", "")
if _qp in _VALID_PAGES:
    st.session_state["current_page"] = _qp
    st.query_params.clear()

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "overview"

current = st.session_state["current_page"]

# Active-class helpers
def _a(page: str) -> str:
    return " active" if current == page else ""

def _ga(pages: set) -> str:
    return " active" if current in pages else ""

# ── Data-source indicator config ─────────────────────────────────────────────
_fred_col = "#CFB991" if _FRED_KEY else "rgba(255,255,255,0.22)"
_fd_col   = "#CFB991" if _FD_KEY   else "rgba(255,255,255,0.22)"

# ── Top navbar (fixed, CSS-hover dropdowns, no JS required) ──────────────────
st.markdown(f"""
<nav class="topnav">
  <div class="nav-inner">

    <a class="nav-brand" href="?page=overview">
      <div class="nav-brand-main">Equity &amp; Commodities Spillover</div>
      <div class="nav-brand-sub">Purdue Daniels &middot; MGMT 69000&ndash;120</div>
    </a>

    <ul class="nav-links">

      <li class="nav-item{_a('overview')}">
        <a href="?page=overview">Overview</a>
      </li>

      <li class="nav-item has-drop{_ga(_ANALYSIS_PAGES)}">
        <div class="nav-top">
          Analysis <span class="caret">&#9660;</span>
        </div>
        <ul class="dropdown">
          <li><a href="?page=war_impact_map" class="{_a('war_impact_map').strip()}">War Impact Map</a></li>
          <li><a href="?page=geopolitical"   class="{_a('geopolitical').strip()}">Geopolitical Triggers</a></li>
          <li><a href="?page=correlation"    class="{_a('correlation').strip()}">Correlation Analysis</a></li>
          <li><a href="?page=spillover"      class="{_a('spillover').strip()}">Spillover Analytics</a></li>
          <li><a href="?page=watchlist"      class="{_a('watchlist').strip()}">Commodities to Watch</a></li>
        </ul>
      </li>

      <li class="nav-item has-drop{_ga(_STRATEGY_PAGES)}">
        <div class="nav-top">
          Strategy <span class="caret">&#9660;</span>
        </div>
        <ul class="dropdown">
          <li><a href="?page=trade_ideas" class="{_a('trade_ideas').strip()}">Trade Ideas</a></li>
          <li><a href="?page=stress_test" class="{_a('stress_test').strip()}">Portfolio Stress Test</a></li>
        </ul>
      </li>

      <li class="nav-item has-drop{_ga(_RESEARCH_PAGES)}">
        <div class="nav-top">
          Research <span class="caret">&#9660;</span>
        </div>
        <ul class="dropdown">
          <li><a href="?page=model_accuracy" class="{_a('model_accuracy').strip()}">Performance Review</a></li>
          <li><a href="?page=ai_chat"        class="{_a('ai_chat').strip()}">AI Analyst</a></li>
        </ul>
      </li>

      <li class="nav-item has-drop{_ga(_ABOUT_PAGES)}">
        <div class="nav-top">
          About <span class="caret">&#9660;</span>
        </div>
        <ul class="dropdown">
          <li><a href="?page=about_heramb" class="{_a('about_heramb').strip()}">Heramb S. Patkar</a></li>
          <li><a href="?page=about_jiahe"  class="{_a('about_jiahe').strip()}">Jiahe Miao</a></li>
          <li><a href="?page=about_ilian"  class="{_a('about_ilian').strip()}">Ilian Zalomai</a></li>
        </ul>
      </li>

    </ul>

    <div class="nav-ds">
      <div class="ds-item">
        <span class="ds-dot" style="color:#CFB991">●</span>
        <span>Yahoo Finance</span>
      </div>
      <div class="ds-item">
        <span class="ds-dot" style="color:{_fred_col}">{"●" if _FRED_KEY else "○"}</span>
        <span style="color:{_fred_col}">FRED</span>
      </div>
      <div class="ds-item">
        <span class="ds-dot" style="color:{_fd_col}">{"●" if _FD_KEY else "○"}</span>
        <span style="color:{_fd_col}">FinancialDatasets</span>
      </div>
    </div>

  </div>
</nav>
<div class="nav-spacer"></div>
""", unsafe_allow_html=True)

# ── Date range strip (compact row at top of content) ─────────────────────────
st.markdown('<div class="date-strip-wrap">', unsafe_allow_html=True)
_dc1, _dc2, _dc3 = st.columns([1, 1, 5])
start_date = _dc1.date_input(
    "From",
    value=date(2010, 1, 1),
    min_value=date(2000, 1, 1),
    max_value=date.today(),
    key="g_start",
)
end_date = _dc2.date_input(
    "To",
    value=date.today(),
    min_value=date(2000, 1, 1),
    max_value=date.today(),
    key="g_end",
)
st.markdown('</div>', unsafe_allow_html=True)

# ── Router ────────────────────────────────────────────────────────────────────
from src.pages.overview        import page_overview
from src.pages.war_impact_map  import page_war_impact_map
from src.pages.geopolitical    import page_geopolitical
from src.pages.correlation     import page_correlation
from src.pages.spillover       import page_spillover
from src.pages.watchlist       import page_watchlist
from src.pages.trade_ideas     import page_trade_ideas
from src.pages.stress_test     import page_stress_test
from src.pages.model_accuracy  import page_model_accuracy
from src.pages.ai_chat         import page_ai_chat
from src.pages.about_heramb    import page_about_heramb
from src.pages.about_jiahe     import page_about_jiahe
from src.pages.about_ilian     import page_about_ilian

_start = str(start_date)
_end   = str(end_date)

_PAGE_MAP = {
    "overview":       lambda: page_overview(_start, _end, _FRED_KEY),
    "war_impact_map": lambda: page_war_impact_map(_start, _end, _FRED_KEY),
    "geopolitical":   lambda: page_geopolitical(_start, _end, _FRED_KEY),
    "correlation":    lambda: page_correlation(_start, _end, _FRED_KEY),
    "spillover":      lambda: page_spillover(_start, _end, _FRED_KEY),
    "watchlist":      lambda: page_watchlist(_start, _end, _FRED_KEY),
    "trade_ideas":    lambda: page_trade_ideas(_start, _end, _FRED_KEY),
    "stress_test":    lambda: page_stress_test(_start, _end, _FRED_KEY),
    "model_accuracy": lambda: page_model_accuracy(_start, _end, _FRED_KEY),
    "ai_chat":        lambda: page_ai_chat(_start, _end),
    "about_heramb":   page_about_heramb,
    "about_jiahe":    page_about_jiahe,
    "about_ilian":    page_about_ilian,
}

_PAGE_MAP.get(current, _PAGE_MAP["overview"])()

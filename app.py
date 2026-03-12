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

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equity-Commodities Spillover | Purdue Daniels",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Purdue institutional CSS ───────────────────────────────────────────────
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
    --ls-tight:   -0.01em;
    --ls-normal:   0em;
    --ls-wide:     0.04em;
    --ls-wider:    0.08em;
    --ls-widest:   0.14em;
}

:root {
    --gold:       #CFB991;
    --gold-light: #EBD99F;
    --gold-mid:   #DAAA00;
    --aged:       #8E6F3E;
    --black:      #000000;
    --ink-soft:   #555960;
    --ink-muted:  #6F727B;
    --ink-faint:  #9D9795;
    --bg:         #ffffff;
    --bg-warm:    #fafaf8;
    --rule:       #E8E5E0;
    --rule-light: #F0EDEA;
    --red:        #c0392b;
    --green:      #2e7d32;
}

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--black);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--black) !important;
    border-right: 1px solid #1a1a1a;
}
[data-testid="stSidebar"] * {
    color: #E8E5E0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stDateInput label,
[data-testid="stSidebar"] .stTextInput label {
    font-size: 0.62rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--gold) !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select {
    background: #111 !important;
    border-color: #333 !important;
    color: #E8E5E0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.7rem !important;
    line-height: 1.6;
    color: var(--ink-faint) !important;
}

/* ── Sidebar nav links ── */
.nav-section-label {
    font-size: 0.56rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold) !important;
    margin: 1.2rem 0 0.4rem;
    padding: 0 0.2rem;
}
.nav-link {
    display: block;
    font-size: 0.72rem;
    font-weight: 500;
    color: #c0bdb8 !important;
    text-decoration: none;
    padding: 0.32rem 0.6rem;
    border-left: 2px solid transparent;
    border-radius: 0 3px 3px 0;
    margin-bottom: 1px;
    cursor: pointer;
    transition: all 0.15s;
}
.nav-link:hover, .nav-link.active {
    color: var(--gold) !important;
    border-left-color: var(--gold);
    background: rgba(207,185,145,0.08);
}

/* ── Main content ── */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
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
    color: var(--ink-faint) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
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

/* ── Selectbox, multiselect, slider ── */
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
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--gold) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Force sidebar always open ── */
section[data-testid="stSidebar"] {
    min-width: 260px !important;
    width: 260px !important;
    transform: none !important;
    visibility: visible !important;
}
[data-testid="collapsedControl"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #DEDAD5; border-radius: 3px; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid var(--rule); margin: 1.2rem 0; }

/* ── Fade-in ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.main .block-container > div { animation: fadeInUp 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94); }
</style>
""", unsafe_allow_html=True)

# ── Read API keys from secrets.toml ───────────────────────────────────────
def _get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get("keys", {}).get(key, default) or default
    except Exception:
        return default

_FRED_KEY = _get_secret("fred_api_key")
_FD_KEY   = _get_secret("financial_datasets_key")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown("""
    <div style="padding:1.2rem 0.2rem 0.8rem">
      <div style="font-size:0.58rem;font-weight:700;letter-spacing:0.2em;
      text-transform:uppercase;color:#CFB991;margin-bottom:0.3rem">
        Purdue University · Daniels School of Business
      </div>
      <div style="font-size:1.3rem;font-weight:800;color:#fff;line-height:1.1;
      letter-spacing:-0.01em">
        Equity &amp; Commodities
      </div>
      <div style="font-size:0.72rem;font-weight:400;color:#9D9795;margin-top:0.2rem">
        Spillover Monitor
      </div>
    </div>
    <div style="border-top:1px solid #1a1a1a;margin-bottom:0.8rem"></div>
    """, unsafe_allow_html=True)

    # Date range
    start_date = st.date_input(
        "Start date",
        value=date(2010, 1, 1),
        min_value=date(2000, 1, 1),
        max_value=date.today(),
    )
    end_date = st.date_input(
        "End date",
        value=date.today(),
        min_value=date(2000, 1, 1),
        max_value=date.today(),
    )

    st.markdown("<div style='border-top:1px solid #1a1a1a;margin:0.8rem 0'></div>",
                unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="nav-section-label">Dashboard</div>', unsafe_allow_html=True)

    _PAGES = [
        ("overview",       "🏠  Overview"),
        ("geopolitical",   "⚑   Geopolitical Triggers"),
        ("correlation",    "⟳   Correlation Analysis"),
        ("spillover",      "→   Spillover Analytics"),
        ("watchlist",      "👁  Commodities to Watch"),
        ("stress_test",    "⚡  Portfolio Stress Test"),
        ("trade_ideas",    "◈   Trade Ideas"),
    ]

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "overview"

    for page_key, page_label in _PAGES:
        is_active = st.session_state["current_page"] == page_key
        if st.button(
            page_label,
            key=f"nav_{page_key}",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state["current_page"] = page_key
            st.rerun()

    st.markdown("<div style='border-top:1px solid #1a1a1a;margin:0.8rem 0'></div>",
                unsafe_allow_html=True)

    # API key status
    st.markdown('<div class="nav-section-label">Data Sources</div>', unsafe_allow_html=True)
    fred_status = "✓ FRED connected" if _FRED_KEY else "○ FRED (not set)"
    fd_status   = "✓ FinancialDatasets" if _FD_KEY   else "○ FinancialDatasets (not set)"
    st.markdown(
        f"""<div style="font-size:0.64rem;color:#9D9795;line-height:1.9">
        ✓ Yahoo Finance (active)<br>
        <span style="color:{'#CFB991' if _FRED_KEY else '#555960'}">{fred_status}</span><br>
        <span style="color:{'#CFB991' if _FD_KEY else '#555960'}">{fd_status}</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.58rem;color:#555960;margin-top:0.4rem">'
        'Edit <code>.streamlit/secrets.toml</code> to add API keys.</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div style="padding:0.7rem 0;margin-top:1rem;border-top:1px solid #1a1a1a">
        <p style="font-size:0.58rem;color:#555960;margin:0;line-height:1.5">
        For educational purposes only.<br>Not investment advice.</p>
        </div>""",
        unsafe_allow_html=True,
    )

# ── Router ─────────────────────────────────────────────────────────────────
from src.pages.overview     import page_overview
from src.pages.geopolitical import page_geopolitical
from src.pages.correlation  import page_correlation
from src.pages.spillover    import page_spillover
from src.pages.watchlist    import page_watchlist
from src.pages.stress_test  import page_stress_test
from src.pages.trade_ideas  import page_trade_ideas

_start = str(start_date)
_end   = str(end_date)

_PAGE_MAP = {
    "overview":     lambda: page_overview(_start, _end, _FRED_KEY),
    "geopolitical": lambda: page_geopolitical(_start, _end, _FRED_KEY),
    "correlation":  lambda: page_correlation(_start, _end, _FRED_KEY),
    "spillover":    lambda: page_spillover(_start, _end, _FRED_KEY),
    "watchlist":    lambda: page_watchlist(_start, _end, _FRED_KEY),
    "stress_test":  lambda: page_stress_test(_start, _end, _FRED_KEY),
    "trade_ideas":  lambda: page_trade_ideas(_start, _end, _FRED_KEY),
}

current = st.session_state.get("current_page", "overview")
_PAGE_MAP.get(current, _PAGE_MAP["overview"])()

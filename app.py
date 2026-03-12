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
    page_icon=None,
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
    --ink-soft:   #333333;
    --ink-muted:  #444444;
    --ink-faint:  #666666;
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

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background: #000000 !important;
    border-right: 1px solid #1e1e1e !important;
    min-width: 272px !important;
    width: 272px !important;
    transform: none !important;
    visibility: visible !important;
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.73rem !important;
    line-height: 1.6 !important;
    color: rgba(255,255,255,0.52) !important;
}
[data-testid="stSidebar"] code {
    background: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.52) !important;
    font-size: 0.63rem !important;
}

/* ── Date inputs ── */
[data-testid="stSidebar"] .stDateInput label {
    font-size: 0.60rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: rgba(207,185,145,0.65) !important;
    margin-bottom: 3px !important;
}
[data-testid="stSidebar"] .stDateInput [data-baseweb="input"] {
    background: #0d0d0d !important;
    border: 1px solid #222 !important;
    border-radius: 2px !important;
    box-shadow: none !important;
    transition: border-color 0.12s ease !important;
}
[data-testid="stSidebar"] .stDateInput [data-baseweb="input"]:hover,
[data-testid="stSidebar"] .stDateInput [data-baseweb="input"]:focus-within {
    border-color: var(--gold) !important;
    background: #0d0d0d !important;
}
[data-testid="stSidebar"] .stDateInput input {
    color: rgba(255,255,255,0.72) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.74rem !important;
    font-weight: 400 !important;
    padding: 0.28rem 0.5rem !important;
    background: transparent !important;
    caret-color: var(--gold) !important;
}
[data-testid="stSidebar"] .stDateInput button {
    color: rgba(255,255,255,0.28) !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stDateInput button:hover {
    color: var(--gold) !important;
    background: transparent !important;
}

/* ── Nav section labels ── */
.nav-section-label {
    display: block !important;
    font-size: 0.60rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.20em !important;
    text-transform: uppercase !important;
    color: var(--gold) !important;
    padding: 0.75rem 1.0rem 0.22rem !important;
    line-height: 1 !important;
}

/* ── Nav buttons ── */
[data-testid="stSidebar"] .stButton {
    margin-top: 0 !important;
    margin-bottom: 1px !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    color: rgba(255,255,255,0.72) !important;
    background: transparent !important;
    border: none !important;
    border-left: 2px solid transparent !important;
    border-radius: 0 2px 2px 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    text-align: left !important;
    font-size: 0.77rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.01em !important;
    padding: 0.42rem 1.0rem !important;
    text-transform: none !important;
    width: 100% !important;
    box-shadow: none !important;
    transition: color 0.12s ease, border-color 0.12s ease,
                background 0.12s ease !important;
}
[data-testid="stSidebar"] .stButton > button > div,
[data-testid="stSidebar"] .stButton > button > span,
[data-testid="stSidebar"] .stButton > button p {
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    color: inherit !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    color: rgba(255,255,255,0.94) !important;
    border-left-color: rgba(207,185,145,0.35) !important;
    background: rgba(255,255,255,0.03) !important;
}
[data-testid="stSidebar"] .stButton > button:hover * { color: rgba(255,255,255,0.94) !important; }

/* Active nav item (type="primary") */
[data-testid="stSidebar"] [data-testid="baseButton-primary"] {
    color: var(--gold) !important;
    border-left-color: var(--gold) !important;
    border-left-width: 2px !important;
    background: rgba(207,185,145,0.06) !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="baseButton-primary"] * {
    color: var(--gold) !important;
}
[data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover {
    background: rgba(207,185,145,0.09) !important;
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
    color: #555960 !important;
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

/* ── Sidebar: always open, no collapse controls ── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarNavLink"],
button[aria-label="Close sidebar"],
button[aria-label="Collapse sidebar"],
button[aria-label="Open sidebar"],
button[aria-label="Expand sidebar"],
[data-testid="stSidebar"] [data-testid="stBaseButton-header"],
[data-testid="stSidebar"] button[kind="header"] {
    display: none !important;
    pointer-events: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}

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

    # ── Brand header ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:1.0rem 0.2rem 0.7rem">
      <div style="font-size:0.60rem;font-weight:700;letter-spacing:0.22em;
                  text-transform:uppercase;color:#CFB991;margin-bottom:0.30rem">
        Purdue University &middot; Daniels School of Business
      </div>
      <div style="font-size:1.28rem;font-weight:800;color:#ffffff;
                  line-height:1.15;letter-spacing:-0.01em">
        Equity &amp; Commodities
      </div>
      <div style="font-size:0.74rem;font-weight:400;color:rgba(255,255,255,0.52);
                  margin-top:0.18rem;letter-spacing:0.01em">
        Spillover Monitor
      </div>
    </div>
    <div style="border-top:1px solid #1c1c1c"></div>
    """, unsafe_allow_html=True)

    # ── Date range ────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.60rem;font-weight:700;letter-spacing:0.20em;'
        'text-transform:uppercase;color:rgba(255,255,255,0.38);padding:0.75rem 1.0rem 0.22rem;'
        'display:block;line-height:1">Date Range</div>',
        unsafe_allow_html=True,
    )
    _d1, _d2 = st.columns(2)
    start_date = _d1.date_input(
        "Start",
        value=date(2010, 1, 1),
        min_value=date(2000, 1, 1),
        max_value=date.today(),
    )
    end_date = _d2.date_input(
        "End",
        value=date.today(),
        min_value=date(2000, 1, 1),
        max_value=date.today(),
    )

    st.markdown(
        "<div style='border-top:1px solid #1c1c1c;margin:0.7rem 0 0'></div>",
        unsafe_allow_html=True,
    )

    # ── Navigation ────────────────────────────────────────────────────────
    # Groups: Overview (unlabelled) → Analysis → Strategy → Research
    _PAGE_GROUPS = [
        (None, [
            ("overview", "Overview"),
        ]),
        ("Analysis", [
            ("war_impact_map", "War Impact Map"),
            ("geopolitical",   "Geopolitical Triggers"),
            ("correlation",    "Correlation Analysis"),
            ("spillover",      "Spillover Analytics"),
            ("watchlist",      "Commodities to Watch"),
        ]),
        ("Strategy", [
            ("trade_ideas", "Trade Ideas"),
            ("stress_test", "Portfolio Stress Test"),
        ]),
        ("Research", [
            ("model_accuracy", "Performance Review"),
            ("ai_chat",        "AI Analyst"),
        ]),
    ]

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "overview"

    for group_label, pages in _PAGE_GROUPS:
        if group_label is not None:
            st.markdown(
                f'<div class="nav-section-label">{group_label}</div>',
                unsafe_allow_html=True,
            )
        for page_key, page_label in pages:
            is_active = st.session_state["current_page"] == page_key
            if st.button(
                page_label,
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["current_page"] = page_key
                st.rerun()

    # ── Data sources ──────────────────────────────────────────────────────
    st.markdown(
        "<div style='border-top:1px solid #1c1c1c;margin:0.7rem 0 0'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.60rem;font-weight:700;letter-spacing:0.20em;'
        'text-transform:uppercase;color:rgba(255,255,255,0.38);padding:0.75rem 1.0rem 0.22rem;'
        'display:block;line-height:1">Data Sources</div>',
        unsafe_allow_html=True,
    )

    _fred_col = "#CFB991" if _FRED_KEY else "rgba(255,255,255,0.28)"
    _fd_col   = "#CFB991" if _FD_KEY   else "rgba(255,255,255,0.28)"
    _fred_ic  = "✓" if _FRED_KEY else "○"
    _fd_ic    = "✓" if _FD_KEY   else "○"

    st.markdown(
        f"""<div style="font-size:0.69rem;line-height:2.0;padding:0 0.2rem">
        <span style="color:#CFB991">✓</span>
        <span style="color:rgba(255,255,255,0.65)"> Yahoo Finance</span><br>
        <span style="color:{_fred_col}">{_fred_ic} FRED &middot; Federal Reserve</span><br>
        <span style="color:{_fd_col}">{_fd_ic} FinancialDatasets</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.63rem;color:rgba(255,255,255,0.30);margin-top:0.35rem;'
        'padding:0 0.2rem;line-height:1.5">'
        'Add keys in <code>.streamlit/secrets.toml</code></p>',
        unsafe_allow_html=True,
    )

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(
        """<div style="border-top:1px solid #1c1c1c;margin-top:1.0rem;
                       padding:0.7rem 0.2rem 0.4rem">
           <p style="font-size:0.63rem;color:rgba(255,255,255,0.30);margin:0;line-height:1.6">
           For educational purposes only.<br>Not investment advice.</p>
           </div>""",
        unsafe_allow_html=True,
    )

# ── Router ─────────────────────────────────────────────────────────────────
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

_start = str(start_date)
_end   = str(end_date)

_PAGE_MAP = {
    "overview":        lambda: page_overview(_start, _end, _FRED_KEY),
    "war_impact_map":  lambda: page_war_impact_map(_start, _end, _FRED_KEY),
    "geopolitical":    lambda: page_geopolitical(_start, _end, _FRED_KEY),
    "correlation":     lambda: page_correlation(_start, _end, _FRED_KEY),
    "spillover":       lambda: page_spillover(_start, _end, _FRED_KEY),
    "watchlist":       lambda: page_watchlist(_start, _end, _FRED_KEY),
    "trade_ideas":     lambda: page_trade_ideas(_start, _end, _FRED_KEY),
    "stress_test":     lambda: page_stress_test(_start, _end, _FRED_KEY),
    "model_accuracy":  lambda: page_model_accuracy(_start, _end, _FRED_KEY),
    "ai_chat":         lambda: page_ai_chat(_start, _end),
}

current = st.session_state.get("current_page", "overview")
_PAGE_MAP.get(current, _PAGE_MAP["overview"])()

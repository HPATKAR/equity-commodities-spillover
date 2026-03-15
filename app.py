"""
Equity-Commodities Spillover Monitor
Purdue University · Daniels School of Business

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import date

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equity-Commodities Spillover | Purdue Daniels",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
# (Sidebar fully hidden; content pushed below the fixed 56 px navbar iframe)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --font-sans: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --gold:       #CFB991;
    --black:      #000000;
    --ink-soft:   #333333;
    --ink-muted:  #444444;
    --ink-faint:  #666666;
    --bg:         #ffffff;
    --bg-warm:    #fafaf8;
    --rule:       #E8E5E0;
    --red:        #c0392b;
    --green:      #2e7d32;
}

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: var(--font-sans);
    background: var(--bg);
    color: var(--black);
}

/* ── Hide Streamlit chrome + sidebar ── */
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

/* ── Content area — padded below the 72 px fixed navbar ── */
.main .block-container {
    padding-top:    86px !important;
    padding-bottom: 3rem !important;
    padding-left:   2.2rem !important;
    padding-right:  2.2rem !important;
    max-width: 1360px;
}

/* ── Headings ── */
h1 { font-size: 1.25rem !important; font-weight: 700 !important; letter-spacing: -0.01em; }
h2 { font-size: 0.95rem !important; font-weight: 700 !important; margin-top: 1.4rem !important; }
h3 { font-size: 0.82rem !important; font-weight: 600 !important; margin-top: 1rem !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-warm); border: 1px solid var(--rule);
    border-radius: 4px; padding: 0.6rem 0.8rem; transition: box-shadow 0.2s;
}
[data-testid="metric-container"]:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.06); border-color: var(--gold); }
[data-testid="stMetricLabel"] {
    font-size: 0.58rem !important; font-weight: 600 !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important; color: #555960 !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important; font-size: 1.05rem !important; font-weight: 700 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] { border-bottom: 1px solid var(--rule); gap: 0; }
[data-testid="stTabs"] button[role="tab"] {
    font-size: 0.7rem !important; font-weight: 500 !important;
    letter-spacing: 0.04em !important; text-transform: uppercase !important;
    color: var(--ink-muted) !important; padding: 0.5rem 1rem !important;
    border-bottom: 2px solid transparent; transition: all 0.2s;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--black) !important; border-bottom-color: var(--gold) !important; font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--black) !important; color: var(--gold) !important;
    border: 1px solid var(--black) !important; border-radius: 3px !important;
    font-size: 0.68rem !important; font-weight: 600 !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    padding: 0.45rem 1.4rem !important; transition: all 0.2s !important;
}
.stButton > button:hover { background: var(--gold) !important; color: var(--black) !important; border-color: var(--gold) !important; }

/* ── Form controls ── */
[data-testid="stSelectbox"] label, [data-testid="stMultiSelect"] label,
[data-testid="stSlider"] label, .stCheckbox label span {
    font-size: 0.65rem !important; font-weight: 600 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important; color: var(--ink-soft) !important;
}

/* ── Date inputs in the control strip ── */
[data-testid="stDateInput"] label {
    font-size: 0.58rem !important; font-weight: 700 !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important; color: var(--ink-faint) !important;
}
[data-testid="stDateInput"] [data-baseweb="input"] {
    background: var(--bg-warm) !important; border: 1px solid var(--rule) !important;
    border-radius: 3px !important; font-size: 0.70rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid var(--rule); border-radius: 4px; }
[data-testid="stDataFrame"] th {
    background: var(--bg-warm) !important; font-size: 0.65rem !important;
    font-weight: 600 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; color: var(--ink-soft) !important;
}
[data-testid="stDataFrame"] td { font-family: var(--font-mono) !important; font-size: 0.7rem !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--gold) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #DEDAD5; border-radius: 3px; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid var(--rule); margin: 1.2rem 0; }

/* ── Fade-in ── */
@keyframes fadeInUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
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
_ABOUT_PAGES = {"about_heramb", "about_jiahe", "about_ilian"}

_qp = st.query_params.get("page", "")
if _qp in _VALID_PAGES:
    st.session_state["current_page"] = _qp
    st.query_params.clear()
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "overview"

current = st.session_state["current_page"]

# ── Navbar (rendered via components.html — no sanitisation; JS fixes it to top) ─
_fred_dot = "●" if _FRED_KEY else "○"
_fred_col = "#CFB991" if _FRED_KEY else "rgba(255,255,255,0.22)"
_fd_dot   = "●" if _FD_KEY   else "○"
_fd_col   = "#CFB991" if _FD_KEY   else "rgba(255,255,255,0.22)"

_NAVBAR = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:72px;overflow:visible;background:#000;
  font-family:'DM Sans',-apple-system,BlinkMacSystemFont,sans-serif;
  -webkit-font-smoothing:antialiased}}
#nav{{
  display:flex;align-items:stretch;height:72px;
  padding:0 2rem;background:#000;
  border-bottom:1px solid #1e1e1e;
}}

/* ── Logotype ── */
.brand{{
  display:flex;align-items:center;gap:13px;
  margin-right:2.8rem;flex-shrink:0;
  text-decoration:none;padding:0 2px;
  transition:opacity .15s;
}}
.brand:hover{{opacity:.82}}
/* Gold monogram tile */
.brand-mark{{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  width:36px;height:36px;flex-shrink:0;
  background:#CFB991;border-radius:5px;
  font-size:.60rem;font-weight:800;
  color:#000;letter-spacing:.06em;line-height:1.15;
  user-select:none;
}}
.brand-mark .mk-top{{font-size:.58rem;font-weight:800;letter-spacing:.10em}}
.brand-mark .mk-bot{{font-size:.44rem;font-weight:600;letter-spacing:.14em;opacity:.70}}
/* Divider */
.brand-div{{width:1px;height:30px;background:rgba(207,185,145,.18);flex-shrink:0}}
/* Text stack */
.brand-text{{display:flex;flex-direction:column;gap:3px}}
.bm{{
  font-size:.82rem;font-weight:700;
  color:#fff;letter-spacing:.005em;
  white-space:nowrap;line-height:1.1;
}}
/* Italic serif-like styling via font-style + lighter weight for the "&" */
.bm em{{font-style:italic;font-weight:400;color:rgba(207,185,145,.85);margin:0 1px}}
.bs{{
  font-size:.50rem;font-weight:500;
  color:rgba(207,185,145,.55);
  letter-spacing:.16em;text-transform:uppercase;
  white-space:nowrap;
}}

/* ── Nav list ── */
ul.links{{display:flex;list-style:none;height:100%;align-items:stretch;margin:0;padding:0;gap:0}}
li.ni{{position:relative;display:flex;align-items:stretch}}

/* Top-level items */
.ni > a.lnk,.ni > span.lnk{{
  display:flex;align-items:center;gap:5px;padding:0 1.05rem;
  font-size:.72rem;font-weight:500;
  color:rgba(255,255,255,.78);
  text-decoration:none;white-space:nowrap;
  border-bottom:2px solid transparent;
  cursor:pointer;user-select:none;
  transition:color .13s,border-color .13s,background .13s;
  letter-spacing:.022em;
}}
.ni > a.lnk:hover,.ni > span.lnk:hover{{
  color:#fff;background:rgba(255,255,255,.04);
  border-bottom-color:rgba(207,185,145,.35);
}}
.ni > a.lnk.active{{color:#CFB991;border-bottom-color:#CFB991;font-weight:600}}
.ni.hd-active > span.lnk{{color:#CFB991;border-bottom-color:#CFB991;font-weight:600}}

/* Caret */
.ct{{font-size:.45rem;opacity:.45;transition:transform .16s;display:inline-block;margin-top:1px}}
.ni:hover .ct{{transform:rotate(180deg);opacity:.75}}

/* ── Dropdown ── */
ul.drop{{
  display:none;position:absolute;top:72px;left:0;
  min-width:214px;background:#070707;
  border:1px solid #282828;border-top:2px solid #CFB991;
  border-radius:0 0 6px 6px;list-style:none;
  margin:0;padding:7px 0;
  box-shadow:0 14px 36px rgba(0,0,0,.70);z-index:99999;
}}
ul.drop li a{{
  display:block;padding:.52rem 1.15rem;
  font-size:.70rem;font-weight:400;
  color:rgba(255,255,255,.55);text-decoration:none;
  border-left:2px solid transparent;white-space:nowrap;
  transition:color .10s,background .10s,border-color .10s;
  letter-spacing:.012em;
}}
ul.drop li a:hover{{color:#fff;background:rgba(255,255,255,.04);border-left-color:rgba(207,185,145,.32)}}
ul.drop li a.active{{color:#CFB991;background:rgba(207,185,145,.07);border-left-color:#CFB991;font-weight:500}}

/* ── Right data-source strip ── */
.ds{{margin-left:auto;display:flex;align-items:center;gap:.85rem;flex-shrink:0;
     font-size:.57rem;letter-spacing:.05em;padding-left:1.4rem}}
.dsi{{display:flex;align-items:center;gap:4px}}
.dsi .dot{{font-size:.50rem}}
</style>
</head>
<body>
<script>
(function(){{
  var NAV_H = 72;
  var f = window.frameElement;
  if(!f) return;

  /* ── Hoist iframe to fixed top bar ── */
  function fixNav(){{
    var hs = NAV_H+'px';
    f.style.cssText='position:fixed!important;top:0!important;left:0!important;'+
      'width:100%!important;height:'+hs+'!important;border:none!important;'+
      'z-index:9999!important;margin:0!important;padding:0!important;background:#000!important;';
    var p=f.parentElement;
    if(p) p.style.cssText='position:fixed!important;top:0!important;left:0!important;'+
      'width:100%!important;height:'+hs+'!important;z-index:9999!important;'+
      'margin:0!important;padding:0!important;';
  }}
  fixNav();
  try{{
    new window.parent.MutationObserver(fixNav)
      .observe(window.parent.document.body,{{childList:true,subtree:false,attributes:false}});
  }}catch(e){{}}

  /* ── Parent-document dropdown (bypasses iframe clipping entirely) ── */
  var _timer=null, _pdoc=window.parent.document;

  function cancelTimer(){{ if(_timer){{ clearTimeout(_timer); _timer=null; }} }}

  function removeDropdown(){{
    var el=_pdoc.getElementById('ec-nav-drop');
    if(el) el.remove();
  }}

  function scheduleHide(){{
    cancelTimer();
    _timer=setTimeout(removeDropdown,130);
  }}

  function buildDropdown(li){{
    cancelTimer();
    removeDropdown();

    var items=li.querySelectorAll('ul.drop > li > a');
    if(!items.length) return;

    var ifrR=f.getBoundingClientRect();
    var liR=li.getBoundingClientRect();

    var ul=_pdoc.createElement('ul');
    ul.id='ec-nav-drop';
    ul.style.cssText=
      'position:fixed;top:'+(ifrR.top+NAV_H)+'px;left:'+(ifrR.left+liR.left)+'px;'+
      'min-width:220px;background:#070707;'+
      'border:1px solid #282828;border-top:2px solid #CFB991;'+
      'border-radius:0 0 6px 6px;list-style:none;margin:0;padding:7px 0;'+
      'box-shadow:0 14px 36px rgba(0,0,0,.75);z-index:99999;'+
      'font-family:DM Sans,-apple-system,sans-serif;';

    items.forEach(function(origA){{
      var pg=origA.getAttribute('data-pg');
      var isActive=origA.classList.contains('active');
      var li2=_pdoc.createElement('li');
      var a=_pdoc.createElement('a');
      a.textContent=origA.textContent;
      a.href='?page='+pg;
      a.style.cssText=
        'display:block;padding:.52rem 1.15rem;font-size:.72rem;font-weight:400;'+
        'color:'+(isActive?'#CFB991':'rgba(255,255,255,.72)')+';text-decoration:none;'+
        'border-left:2px solid '+(isActive?'#CFB991':'transparent')+';'+
        'white-space:nowrap;letter-spacing:.012em;'+
        'background:'+(isActive?'rgba(207,185,145,.07)':'')+';'+
        'cursor:pointer;';
      a.addEventListener('mouseover',function(){{
        a.style.color='#fff';
        a.style.background='rgba(255,255,255,.04)';
        a.style.borderLeftColor='rgba(207,185,145,.35)';
      }});
      a.addEventListener('mouseout',function(){{
        a.style.color=isActive?'#CFB991':'rgba(255,255,255,.72)';
        a.style.background=isActive?'rgba(207,185,145,.07)':'';
        a.style.borderLeftColor=isActive?'#CFB991':'transparent';
      }});
      a.addEventListener('click',function(e){{
        e.preventDefault();
        removeDropdown();
        window.parent.location.href='?page='+pg;
      }});
      li2.appendChild(a);
      ul.appendChild(li2);
    }});

    ul.addEventListener('mouseenter',cancelTimer);
    ul.addEventListener('mouseleave',scheduleHide);
    _pdoc.body.appendChild(ul);
  }}

  document.addEventListener('DOMContentLoaded',function(){{
    /* Wire hover items */
    document.querySelectorAll('li.ni').forEach(function(li){{
      if(!li.querySelector('ul.drop')) return;
      li.addEventListener('mouseenter',function(){{ buildDropdown(li); }});
      li.addEventListener('mouseleave',scheduleHide);
    }});

    /* Mark active page + group */
    var cur={json.dumps(current)};
    var ANALYSIS=['war_impact_map','geopolitical','correlation','spillover','watchlist'];
    var STRATEGY=['trade_ideas','stress_test'];
    var RESEARCH=['model_accuracy','ai_chat'];
    var ABOUT   =['about_heramb','about_jiahe','about_ilian'];
    document.querySelectorAll('[data-pg]').forEach(function(a){{
      if(a.dataset.pg===cur) a.classList.add('active');
    }});
    var MAP={{'ga':ANALYSIS,'gs':STRATEGY,'gr':RESEARCH,'gab':ABOUT}};
    Object.keys(MAP).forEach(function(id){{
      if(MAP[id].indexOf(cur)>-1){{
        var el=document.getElementById(id);
        if(el) el.classList.add('hd-active');
      }}
    }});
  }});
}})();
</script>

<div id="nav">
  <!-- Logotype -->
  <a class="brand" href="?page=overview" target="_parent">
    <div class="brand-mark">
      <span class="mk-top">E&amp;C</span>
      <span class="mk-bot">MON</span>
    </div>
    <div class="brand-div"></div>
    <div class="brand-text">
      <span class="bm">Equity <em>&amp;</em> Commodities Spillover</span>
      <span class="bs">Purdue Daniels &middot; MGMT&nbsp;69000&ndash;120</span>
    </div>
  </a>

  <ul class="links">
    <li class="ni">
      <a class="lnk {'active' if current == 'overview' else ''}" data-pg="overview"
         href="?page=overview" target="_parent">Overview</a>
    </li>

    <li class="ni" id="ga">
      <span class="lnk">Analysis <span class="ct">&#9660;</span></span>
      <ul class="drop">
        <li><a data-pg="war_impact_map" href="?page=war_impact_map" target="_parent" class="{'active' if current=='war_impact_map' else ''}">War Impact Map</a></li>
        <li><a data-pg="geopolitical"   href="?page=geopolitical"   target="_parent" class="{'active' if current=='geopolitical' else ''}">Geopolitical Triggers</a></li>
        <li><a data-pg="correlation"    href="?page=correlation"    target="_parent" class="{'active' if current=='correlation' else ''}">Correlation Analysis</a></li>
        <li><a data-pg="spillover"      href="?page=spillover"      target="_parent" class="{'active' if current=='spillover' else ''}">Spillover Analytics</a></li>
        <li><a data-pg="watchlist"      href="?page=watchlist"      target="_parent" class="{'active' if current=='watchlist' else ''}">Commodities to Watch</a></li>
      </ul>
    </li>

    <li class="ni" id="gs">
      <span class="lnk">Strategy <span class="ct">&#9660;</span></span>
      <ul class="drop">
        <li><a data-pg="trade_ideas" href="?page=trade_ideas" target="_parent" class="{'active' if current=='trade_ideas' else ''}">Trade Ideas</a></li>
        <li><a data-pg="stress_test" href="?page=stress_test" target="_parent" class="{'active' if current=='stress_test' else ''}">Portfolio Stress Test</a></li>
      </ul>
    </li>

    <li class="ni" id="gr">
      <span class="lnk">Research <span class="ct">&#9660;</span></span>
      <ul class="drop">
        <li><a data-pg="model_accuracy" href="?page=model_accuracy" target="_parent" class="{'active' if current=='model_accuracy' else ''}">Performance Review</a></li>
        <li><a data-pg="ai_chat"        href="?page=ai_chat"        target="_parent" class="{'active' if current=='ai_chat' else ''}">AI Analyst</a></li>
      </ul>
    </li>

    <li class="ni" id="gab">
      <span class="lnk">About <span class="ct">&#9660;</span></span>
      <ul class="drop">
        <li><a data-pg="about_heramb" href="?page=about_heramb" target="_parent" class="{'active' if current=='about_heramb' else ''}">Heramb S. Patkar</a></li>
        <li><a data-pg="about_jiahe"  href="?page=about_jiahe"  target="_parent" class="{'active' if current=='about_jiahe' else ''}">Jiahe Miao</a></li>
        <li><a data-pg="about_ilian"  href="?page=about_ilian"  target="_parent" class="{'active' if current=='about_ilian' else ''}">Ilian Zalomai</a></li>
      </ul>
    </li>
  </ul>

  <div class="ds">
    <div class="dsi"><span class="dot" style="color:#CFB991">&#9679;</span><span style="color:rgba(255,255,255,.40)">Yahoo Finance</span></div>
    <div class="dsi"><span class="dot" style="color:{_fred_col}">{_fred_dot}</span><span style="color:{_fred_col}">FRED</span></div>
    <div class="dsi"><span class="dot" style="color:{_fd_col}">{_fd_dot}</span><span style="color:{_fd_col}">FinancialDatasets</span></div>
  </div>
</div>
</body></html>"""

components.html(_NAVBAR, height=1, scrolling=False)

# ── Date range strip (hidden on About pages — they need no data range) ────────
_is_about = current in _ABOUT_PAGES

if not _is_about:
    _dc1, _dc2, _dc3 = st.columns([1, 1, 5])
    start_date = _dc1.date_input(
        "From", value=date(2010, 1, 1),
        min_value=date(2000, 1, 1), max_value=date.today(), key="g_start",
    )
    end_date = _dc2.date_input(
        "To", value=date.today(),
        min_value=date(2000, 1, 1), max_value=date.today(), key="g_end",
    )
    st.markdown('<div style="border-top:1px solid #E8E5E0;margin:0.4rem 0 1.4rem"></div>',
                unsafe_allow_html=True)
else:
    # Provide dummy dates so the router lambdas below don't fail
    start_date = st.session_state.get("g_start", date(2010, 1, 1))
    end_date   = st.session_state.get("g_end",   date.today())

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

"""
Equity-Commodities Spillover Dashboard - Data Config
Tickers, geopolitical event windows, FRED series, display metadata.
"""

from datetime import date

# ── Date range ─────────────────────────────────────────────────────────────
DEFAULT_START = date(2005, 1, 1)
DEFAULT_END   = date.today()

# ── Equity tickers (yfinance) ──────────────────────────────────────────────
EQUITY_TICKERS = {
    # United States
    "S&P 500":       "^GSPC",
    "Nasdaq 100":    "^NDX",
    "DJIA":          "^DJI",
    "Russell 2000":  "^RUT",
    # Europe
    "Eurostoxx 50":  "^STOXX50E",
    "DAX":           "^GDAXI",
    "CAC 40":        "^FCHI",
    "FTSE 100":      "^FTSE",
    # Asia-Pacific
    "Nikkei 225":    "^N225",
    "TOPIX":         "1306.T",   # NEXT FUNDS TOPIX ETF as TOPIX proxy
    "Hang Seng":     "^HSI",
    "Shanghai Comp": "000001.SS",
    "CSI 300":       "000300.SS",
    # India
    "Sensex":        "^BSESN",
    "Nifty 50":      "^NSEI",
}

# ── Equity regions (for grouping in UI) ───────────────────────────────────
EQUITY_REGIONS = {
    "USA":    ["S&P 500", "Nasdaq 100", "DJIA", "Russell 2000"],
    "Europe": ["Eurostoxx 50", "DAX", "CAC 40", "FTSE 100"],
    "Japan":  ["Nikkei 225", "TOPIX"],
    "China":  ["Hang Seng", "Shanghai Comp", "CSI 300"],
    "India":  ["Sensex", "Nifty 50"],
}

# ── Commodity tickers (yfinance) ───────────────────────────────────────────
COMMODITY_TICKERS = {
    # Energy
    "WTI Crude Oil":    "CL=F",
    "Brent Crude":      "BZ=F",
    "Natural Gas":      "NG=F",
    "Gasoline (RBOB)":  "RB=F",
    "Heating Oil":      "HO=F",
    # Metals - precious
    "Gold":             "GC=F",
    "Silver":           "SI=F",
    "Platinum":         "PL=F",
    # Metals - industrial
    "Copper":           "HG=F",
    "Aluminum":         "ALI=F",
    "Nickel":           "NILSY",
    # Agriculture
    "Wheat":            "ZW=F",
    "Corn":             "ZC=F",
    "Soybeans":         "ZS=F",
    "Sugar #11":        "SB=F",
    "Coffee":           "KC=F",
    "Cotton":           "CT=F",
}

# ── Commodity groups ───────────────────────────────────────────────────────
COMMODITY_GROUPS = {
    "Energy":          ["WTI Crude Oil", "Brent Crude", "Natural Gas", "Gasoline (RBOB)", "Heating Oil"],
    "Precious Metals": ["Gold", "Silver", "Platinum"],
    "Industrial Metals": ["Copper", "Aluminum", "Nickel"],
    "Agriculture":     ["Wheat", "Corn", "Soybeans", "Sugar #11", "Coffee", "Cotton"],
}

# ── Fixed Income tickers (yfinance ETFs) ───────────────────────────────────
FIXED_INCOME_TICKERS: dict[str, str] = {
    "US 20Y+ Treasury (TLT)":  "TLT",
    "US 7-10Y Treasury (IEF)": "IEF",
    "US 1-3Y Treasury (SHY)":  "SHY",
    "IG Corporate (LQD)":      "LQD",
    "HY Corporate (HYG)":      "HYG",
    "EM USD Bonds (EMB)":      "EMB",
    "TIPS / Inflation (TIP)":  "TIP",
}

FIXED_INCOME_GROUPS: dict[str, list[str]] = {
    "Government":    ["US 20Y+ Treasury (TLT)", "US 7-10Y Treasury (IEF)", "US 1-3Y Treasury (SHY)"],
    "Credit":        ["IG Corporate (LQD)", "HY Corporate (HYG)"],
    "International": ["EM USD Bonds (EMB)"],
    "Inflation":     ["TIPS / Inflation (TIP)"],
}

# ── FX tickers (yfinance) ───────────────────────────────────────────────────
FX_TICKERS: dict[str, str] = {
    "DXY (Dollar Index)": "UUP",
    "EUR/USD":            "EURUSD=X",
    "USD/JPY":            "USDJPY=X",
    "GBP/USD":            "GBPUSD=X",
    "USD/CNY":            "USDCNY=X",
    "USD/BRL":            "USDBRL=X",
    "USD/INR":            "USDINR=X",
}

FX_GROUPS: dict[str, list[str]] = {
    "Dollar Index": ["DXY (Dollar Index)"],
    "G3":           ["EUR/USD", "USD/JPY", "GBP/USD"],
    "Emerging":     ["USD/CNY", "USD/BRL", "USD/INR"],
}

# ── FRED series ────────────────────────────────────────────────────────────
FRED_SERIES = {
    "VIX":         "VIXCLS",
    "US_10Y":      "DGS10",
    "US_2Y":       "DGS2",
    "US_CPI_YOY":  "CPIAUCSL",
    "US_GDP":      "GDPC1",
    "DXY_proxy":   "DTWEXBGS",   # Broad dollar index
    "OIL_PRICE":   "DCOILWTICO", # WTI daily (FRED backup)
    "GOLD_PRICE":  "GOLDAMGBD228NLBM",
    "WHEAT_PPI":   "WPU012",
    "TIPS_BREAKEVEN_5Y":  "T5YIE",
    "TIPS_BREAKEVEN_10Y": "T10YIE",
    "REAL_RATE_10Y":      "DFII10",
    "HY_OAS":             "BAMLH0A0HYM2",
    "IG_OAS":             "BAMLC0A0CM",
}

# ── Private Credit Proxy Tickers (yfinance) ────────────────────────────────
# Used for private credit bubble risk scoring in Insights page.
# BKLN: Invesco Senior Loan ETF - best liquid proxy for leveraged loan market
# ARCC: Ares Capital - largest BDC by AUM (~$22B); direct lending benchmark
# OBDC: Blue Owl Capital - large mid-market direct lender
# FSK:  FS KKR Capital - more aggressive credit, higher default sensitivity
# JBBB: Janus Henderson CLO BBB - CLO mezzanine tranche stress signal
PC_PROXY_TICKERS: dict[str, str] = {
    "BKLN":  "BKLN",   # Invesco Senior Loan ETF
    "ARCC":  "ARCC",   # Ares Capital (BDC)
    "OBDC":  "OBDC",   # Blue Owl (BDC)
    "FSK":   "FSK",    # FS KKR Capital (BDC)
    "JBBB":  "JBBB",   # Janus CLO BBB
}

# ── Geopolitical / macro event windows ─────────────────────────────────────
# Each event: label, start, end, color (hex), category, description
GEOPOLITICAL_EVENTS = [
    {
        "label":       "GFC",
        "name":        "Global Financial Crisis",
        "start":       date(2008, 9, 15),   # Lehman collapse
        "end":         date(2009, 3, 9),    # S&P trough
        "color":       "#c0392b",
        "category":    "Financial",
        "description": "Lehman Brothers collapse triggers global credit freeze. "
                       "VIX hits 89.5. S&P 500 loses 56% peak-to-trough. "
                       "Commodities collapse: WTI drops from $147 to $32.",
    },
    {
        "label":       "Arab Spring / Libya",
        "name":        "Arab Spring & Libya Oil Shock",
        "start":       date(2011, 1, 14),
        "end":         date(2011, 10, 23),
        "color":       "#e67e22",
        "category":    "Geopolitical",
        "description": "Uprisings across MENA disrupt Libyan oil output (-1.6 mb/d). "
                       "Brent spikes above $125. Arab Spring premium embedded in oil.",
    },
    {
        "label":       "US-China Trade War",
        "name":        "US-China Trade War",
        "start":       date(2018, 3, 1),
        "end":         date(2020, 1, 15),   # Phase 1 deal
        "color":       "#8e44ad",
        "category":    "Trade",
        "description": "Section 232/301 tariffs disrupt global supply chains. "
                       "Copper and soybeans hit hardest. "
                       "Safe havens (gold, JPY) surge on escalation.",
    },
    {
        "label":       "Aramco Attack",
        "name":        "Abqaiq-Khurais Drone Attack",
        "start":       date(2019, 9, 14),
        "end":         date(2019, 10, 15),
        "color":       "#d35400",
        "category":    "Geopolitical",
        "description": "Drone strikes on Saudi Aramco facilities cut ~5% of global oil supply. "
                       "Brent spikes +15% in a single session - largest intraday surge on record.",
    },
    {
        "label":       "COVID-19",
        "name":        "COVID-19 Pandemic",
        "start":       date(2020, 2, 19),   # S&P peak before crash
        "end":         date(2021, 11, 30),  # Pre-Omicron
        "color":       "#2980b9",
        "category":    "Pandemic",
        "description": "Global demand shock + supply disruptions. "
                       "WTI briefly goes negative (-$37, Apr 2020). "
                       "Gold hits ATH $2,075. Wheat surges on hoarding.",
    },
    {
        "label":       "WTI Negative",
        "name":        "WTI Goes Negative",
        "start":       date(2020, 4, 20),
        "end":         date(2020, 5, 5),
        "color":       "#16a085",
        "category":    "Commodity",
        "description": "May 2020 WTI futures close at -$37.63/bbl. "
                       "Storage capacity exhausted. Negative price unprecedented in history.",
    },
    {
        "label":       "Ukraine War",
        "name":        "Russia-Ukraine War",
        "start":       date(2022, 2, 24),   # Invasion
        "end":         date.today(),
        "color":       "#c0392b",
        "category":    "Geopolitical",
        "description": "Russia's invasion disrupts global energy and food supply. "
                       "Europe loses ~40% of Russian gas. "
                       "Wheat spikes +40%, Nickel +250% in days (LME halt). "
                       "Brent hits $139. Gold surges as safe haven.",
    },
    {
        "label":       "LME Nickel Squeeze",
        "name":        "LME Nickel Short Squeeze",
        "start":       date(2022, 3, 7),
        "end":         date(2022, 3, 25),
        "color":       "#7f8c8d",
        "category":    "Commodity",
        "description": "Nickel prices double in 24 hours to >$100,000/tonne. "
                       "LME suspends trading and cancels trades. "
                       "China's Tsingshan group faces massive margin calls.",
    },
    {
        "label":       "Fed Hiking Cycle",
        "name":        "Fed Rate Hiking Cycle",
        "start":       date(2022, 3, 16),
        "end":         date(2023, 7, 26),
        "color":       "#2c3e50",
        "category":    "Monetary",
        "description": "Most aggressive Fed tightening cycle since 1980. "
                       "+525 bps in 16 months. Dollar surges (DXY +18%). "
                       "Gold paradoxically holds. EM commodities under pressure.",
    },
    {
        "label":       "SVB Crisis",
        "name":        "SVB / Banking Crisis",
        "start":       date(2023, 3, 8),
        "end":         date(2023, 5, 1),
        "color":       "#e74c3c",
        "category":    "Financial",
        "description": "Silicon Valley Bank collapses. "
                       "Gold spikes +8% as banking safe haven. "
                       "Oil drops on recession fears. "
                       "Credit Suisse emergency sale to UBS.",
    },
    {
        "label":       "Israel-Hamas",
        "name":        "Israel-Hamas War & Iran Escalation",
        "start":       date(2023, 10, 7),
        "end":         date.today(),
        "color":       "#f39c12",
        "category":    "Geopolitical",
        "description": "Hamas attack on Israel triggers war in Gaza. "
                       "Iran proxies attack Red Sea shipping (Houthis). "
                       "Suez Canal disruptions push shipping costs +300%. "
                       "Oil risk premium embedded; gold hits ATH $2,400+.",
    },
    {
        "label":       "India-Pakistan",
        "name":        "India-Pakistan Military Escalation",
        "start":       date(2025, 5, 7),    # Operation Sindoor - Indian strikes on Pakistan
        "end":         date.today(),
        "color":       "#f39c12",
        "category":    "Geopolitical",
        "description": "India launched Operation Sindoor (May 7, 2025) striking Pakistani militant infrastructure "
                       "following a terrorist attack in Pahalgam, Kashmir that killed 26 civilians. "
                       "Pakistan responded with artillery fire and airspace closures. "
                       "Both nations are nuclear-armed. Elevated cross-border tensions spike Nifty 50 volatility, "
                       "weaken the rupee (INR -1 to -3% during escalation), and raise India's geopolitical risk premium. "
                       "South Asia conflict historically boosts gold (safe-haven) and pressures EM equities.",
    },
    {
        "label":       "Iran/Hormuz",
        "name":        "Iran Military Conflict & Strait of Hormuz Crisis",
        "start":       date(2025, 6, 13),
        "end":         date.today(),
        "color":       "#c0392b",
        "category":    "Geopolitical",
        "description": "Direct U.S.-Israel strikes on Iranian nuclear and military facilities trigger "
                       "escalating retaliation. Iran threatens Strait of Hormuz closure - through which "
                       "~20% of global oil and 25% of global LNG transits. Brent spikes on supply "
                       "disruption fears; gold and USD surge as safe havens. Energy-importing Asia "
                       "bears disproportionate risk.",
    },
]

# ── Commodities watchlist - curated display metadata ──────────────────────
WATCHLIST = [
    # (ticker, display_name, group, alert_reason)
    ("CL=F",  "WTI Crude Oil",    "Energy",           "Iran & Ukraine war premium"),
    ("BZ=F",  "Brent Crude",      "Energy",           "Geopolitical risk benchmark"),
    ("NG=F",  "Natural Gas",      "Energy",           "Europe energy crisis proxy"),
    ("GC=F",  "Gold",             "Precious Metals",  "Safe haven & inflation hedge"),
    ("HG=F",  "Copper",           "Industrial Metals","Global growth bellwether"),
    ("NILSY", "Nickel",           "Industrial Metals","EV battery supply risk"),
    ("ZW=F",  "Wheat",            "Agriculture",      "Ukraine/Russia supply shock"),
    ("ZC=F",  "Corn",             "Agriculture",      "Ethanol & food inflation"),
    ("ZS=F",  "Soybeans",         "Agriculture",      "China-US trade war proxy"),
    ("SI=F",  "Silver",           "Precious Metals",  "Industrial + safe haven hybrid"),
]

# ── Colour palette ─────────────────────────────────────────────────────────
PALETTE = [
    "#000000",  # Black      - strong anchor (equities lead)
    "#c0392b",  # Red        - crisis / energy
    "#2980b9",  # Blue       - liquid / financials
    "#2e7d32",  # Green      - growth / metals
    "#e67e22",  # Orange     - grain / soft commodities
    "#8e44ad",  # Purple     - diversification
    "#16a085",  # Teal       - EM / Asia
    "#DAAA00",  # Rush Gold  - Purdue brand accent
    "#555960",  # Steel      - additional series
    "#CFB991",  # Boilermaker Gold - lighter accent
    "#8E6F3E",  # Aged Brown - tertiary
    "#737373",  # Cool Gray  - fallback
]

# High-contrast palette explicitly for multi-line charts (equity + commodity mixed).
# Equities use solid lines; commodities use dash patterns - see _EQUITY_DASH / _CMD_DASH.
CHART_PALETTE = PALETTE  # alias - update PALETTE to change all charts

CATEGORY_COLORS = {
    "Geopolitical": "#c0392b",
    "Pandemic":     "#2980b9",
    "Financial":    "#8e44ad",
    "Monetary":     "#2c3e50",
    "Trade":        "#8e44ad",
    "Commodity":    "#16a085",
}

"""
Equity-Commodities Spillover Dashboard - Data Config
Tickers, geopolitical event windows, FRED series, display metadata.
"""

from datetime import date

# ── Conflict registry ───────────────────────────────────────────────────────
# CONFLICTS: rich per-conflict metadata used by conflict_model.py for CIS/TPS.
# Separate from GEOPOLITICAL_EVENTS (event-window analysis) - this is the live
# conflict tracking layer for the scoring architecture.
#
# Transmission channel values: 0–1 (how much this conflict affects this channel).
# Intensity dimensions: 0–1 (current state; update when situation changes).
# escalation_trend: "escalating" | "stable" | "de-escalating"
# state: "active" | "latent" | "frozen"
#
# NOTE: Update last_updated and dimension values when the situation changes.
# All other scoring is computed dynamically by conflict_model.py.

CONFLICTS: list[dict] = [
    {
        "id":      "ukraine_russia",
        "name":    "Russia-Ukraine War",
        "label":   "UA/RU",
        "region":  "Russia/Ukraine",
        "start":   date(2022, 2, 24),
        "end":     None,
        "state":   "active",
        "category": "War",
        "color":   "#c0392b",
        "last_updated": date(2026, 4, 19),

        # Intensity dimensions (0–1)
        "deadliness":           0.90,
        "civilian_danger":      0.80,
        "geographic_diffusion": 0.60,   # widened front lines, cross-border drone campaigns
        "fragmentation":        0.35,
        "escalation_trend":     "escalating",  # 2026: continued missile/drone escalation
        "source_coverage":      0.95,

        # Transmission channels (0–1 relevance per market)
        "transmission": {
            "oil_gas":       0.80,
            "metals":        0.75,   # nickel, aluminum, palladium
            "agriculture":   0.90,   # wheat, corn, sunflower
            "shipping":      0.55,   # Black Sea
            "chokepoint":    0.25,   # Bosphorus partial
            "sanctions":     0.95,
            "equity_sector": 0.65,   # defense, energy, fertilizers
            "fx":            0.55,   # EUR, PLN
            "inflation":     0.80,
            "supply_chain":  0.70,
            "credit":        0.40,
            "energy_infra":  0.85,
        },

        # RSS keyword boosters
        "keywords": ["russia", "ukraine", "putin", "kyiv", "donetsk",
                     "zaporizhzhia", "kherson", "nato ukraine", "gazprom",
                     "nord stream", "russian oil", "grain deal"],

        # Affected assets (for exposure scoring)
        "affected_equities":    ["DAX", "CAC 40", "FTSE 100", "Eurostoxx 50"],
        "affected_commodities": ["WTI Crude Oil", "Brent Crude", "Natural Gas",
                                 "Wheat", "Corn", "Nickel", "Aluminum"],
        "affected_fx":          ["EUR/USD"],
        "hedge_assets":         ["Gold", "Silver"],

        "data_confidence": 0.88,
        "acled_id":        "ukraine_russia",   # maps to src/data/acled.py _ACLED_CONFLICT_MAP
    },

    {
        "id":      "red_sea_houthi",
        "name":    "Red Sea Shipping Crisis (Houthi)",
        "label":   "Red Sea",
        "region":  "Middle East",
        "start":   date(2023, 11, 19),
        "end":     None,
        "state":   "active",
        "category": "Conflict",
        "color":   "#e67e22",
        "last_updated": date(2026, 4, 19),

        "deadliness":           0.45,
        "civilian_danger":      0.30,
        "geographic_diffusion": 0.40,
        "fragmentation":        0.50,
        "escalation_trend":     "stable",
        "source_coverage":      0.85,

        "transmission": {
            "oil_gas":       0.55,
            "metals":        0.30,
            "agriculture":   0.35,
            "shipping":      0.95,
            "chokepoint":    0.90,   # Bab-el-Mandeb / Suez
            "sanctions":     0.20,
            "equity_sector": 0.50,   # shipping, energy, retail
            "fx":            0.20,
            "inflation":     0.60,   # freight costs
            "supply_chain":  0.80,
            "credit":        0.20,
            "energy_infra":  0.15,
        },

        "keywords": ["houthi", "red sea", "bab el mandeb", "suez canal",
                     "shipping attack", "vessel seized", "tanker attack",
                     "yemen", "maersk red sea", "shipping reroute"],

        "affected_equities":    ["S&P 500", "FTSE 100", "DAX"],
        "affected_commodities": ["Brent Crude", "WTI Crude Oil", "Natural Gas",
                                 "Wheat", "Corn"],
        "affected_fx":          [],
        "hedge_assets":         ["Gold"],

        "data_confidence": 0.80,
        "acled_id":        "red_sea_houthi",
    },

    {
        "id":      "israel_gaza",
        "name":    "Israel-Gaza War",
        "label":   "Israel/Gaza",
        "region":  "Middle East",
        "start":   date(2023, 10, 7),
        "end":     None,
        "state":   "active",
        "category": "War",
        "color":   "#f39c12",
        "last_updated": date(2026, 4, 19),

        "deadliness":           0.75,
        "civilian_danger":      0.90,
        "geographic_diffusion": 0.35,
        "fragmentation":        0.40,
        "escalation_trend":     "stable",
        "source_coverage":      0.92,

        "transmission": {
            "oil_gas":       0.40,
            "metals":        0.15,
            "agriculture":   0.15,
            "shipping":      0.30,
            "chokepoint":    0.35,
            "sanctions":     0.30,
            "equity_sector": 0.35,
            "fx":            0.25,
            "inflation":     0.30,
            "supply_chain":  0.20,
            "credit":        0.25,
            "energy_infra":  0.20,
        },

        "keywords": ["israel", "gaza", "hamas", "idf", "hezbollah",
                     "iran israel", "middle east war", "ceasefire",
                     "west bank", "rafah", "beirut"],

        "affected_equities":    ["S&P 500", "DAX", "Eurostoxx 50"],
        "affected_commodities": ["Gold", "Brent Crude", "WTI Crude Oil"],
        "affected_fx":          ["EUR/USD"],
        "hedge_assets":         ["Gold", "US 20Y+ Treasury (TLT)"],

        "data_confidence": 0.87,
        "acled_id":        "israel_hamas",
    },

    {
        "id":      "iran_conflict",
        "name":    "Iran Military Conflict & Hormuz Risk",
        "label":   "Iran/Hormuz",
        "region":  "Middle East",
        "start":   date(2025, 6, 13),
        "end":     None,
        "state":   "active",
        "category": "War",
        "color":   "#c0392b",

        # 2026-04-14: Active Hormuz blockade confirmed (PortWatch: 2–3 oil
        # tankers/day vs 71 pre-crisis). US carrier confrontation escalating.
        # ~20% of global oil supply disrupted. Updated from "stable" to reflect
        # live market conditions - this is the primary geopolitical risk driver.
        "deadliness":           0.82,   # active military blockade, US carrier confrontation
        "civilian_danger":      0.72,   # global oil supply crisis, humanitarian downstream
        "geographic_diffusion": 0.68,   # Persian Gulf + global supply chain disruption
        "fragmentation":        0.30,
        "escalation_trend":     "escalating",
        "source_coverage":      0.92,
        "last_updated":         date(2026, 4, 14),

        "transmission": {
            "oil_gas":       0.97,   # ~20% of global crude flows blocked
            "metals":        0.15,
            "agriculture":   0.10,
            "shipping":      0.90,   # near-complete tanker halt
            "chokepoint":    0.97,   # Strait of Hormuz - live PortWatch: 2–3 tankers/day
            "sanctions":     0.90,
            "equity_sector": 0.65,   # energy, defense, refining
            "fx":            0.45,   # USD strengthens on safe-haven; EM oil importers hit
            "inflation":     0.92,   # direct fuel cost transmission globally
            "supply_chain":  0.72,   # Asian manufacturing (India, China, Japan, Korea)
            "credit":        0.40,
            "energy_infra":  0.85,
        },

        "keywords": ["iran", "hormuz", "strait of hormuz", "tehran",
                     "iranian nuclear", "irgc", "us iran", "israel iran",
                     "iranian oil", "persian gulf", "oil tanker iran",
                     "hormuz blockade", "tanker halt", "persian gulf crisis"],

        "affected_equities":    ["S&P 500", "Nikkei 225", "Sensex", "Nifty 50",
                                 "Hang Seng", "Kospi", "Shanghai Comp"],
        "affected_commodities": ["WTI Crude Oil", "Brent Crude", "Natural Gas",
                                 "Gasoline (RBOB)", "Heating Oil"],
        "affected_fx":          ["USD/JPY", "USD/INR", "USD/KRW"],
        "hedge_assets":         ["Gold", "Silver"],

        "acled_id":        "iran_regional",    # maps to src/data/acled.py + gdelt.py
        "data_confidence": 0.88,
    },

    {
        "id":      "india_pakistan",
        "name":    "India-Pakistan Military Escalation",
        "label":   "India/Pak",
        "region":  "South Asia",
        "start":   date(2025, 5, 7),
        "end":     None,
        "state":   "active",
        "category": "Conflict",
        "color":   "#f39c12",
        "last_updated": date(2026, 4, 19),

        "deadliness":           0.62,   # elevated post-Op Sindoor, cross-border fire ongoing
        "civilian_danger":      0.55,
        "geographic_diffusion": 0.38,   # LoC + airspace violations
        "fragmentation":        0.20,
        "escalation_trend":     "escalating",  # Op Sindoor (May 2025) standoff not resolved
        "source_coverage":      0.82,

        "transmission": {
            "oil_gas":       0.15,
            "metals":        0.10,
            "agriculture":   0.20,
            "shipping":      0.15,
            "chokepoint":    0.10,
            "sanctions":     0.25,
            "equity_sector": 0.60,   # Nifty, defense
            "fx":            0.70,   # INR primary
            "inflation":     0.45,
            "supply_chain":  0.35,
            "credit":        0.30,
            "energy_infra":  0.10,
        },

        "keywords": ["india pakistan", "operation sindoor", "kashmir",
                     "line of control", "nuclear india", "nifty conflict",
                     "indo-pak", "pahalgam", "pakistan military"],

        "affected_equities":    ["Sensex", "Nifty 50"],
        "affected_commodities": ["Gold"],
        "affected_fx":          ["USD/INR"],
        "hedge_assets":         ["Gold", "US 20Y+ Treasury (TLT)"],

        "acled_id":        "india_pakistan",    # maps to src/data/acled.py + gdelt.py
        "data_confidence": 0.75,
    },

    {
        "id":      "taiwan_strait",
        "name":    "Taiwan Strait Tensions",
        "label":   "Taiwan",
        "region":  "Asia Pacific",
        "start":   date(2022, 8, 2),   # Pelosi visit triggered sharp escalation
        "end":     None,
        "state":   "latent",           # Latent but persistent structural risk
        "category": "Geopolitical",
        "color":   "#2980b9",
        "last_updated": date(2026, 4, 19),

        "deadliness":           0.10,   # No active hostilities
        "civilian_danger":      0.10,
        "geographic_diffusion": 0.25,
        "fragmentation":        0.15,
        "escalation_trend":     "stable",
        "source_coverage":      0.75,

        "transmission": {
            "oil_gas":       0.25,
            "metals":        0.60,   # rare earths, semiconductors
            "agriculture":   0.10,
            "shipping":      0.70,
            "chokepoint":    0.75,   # South China Sea / Taiwan Strait
            "sanctions":     0.80,
            "equity_sector": 0.90,   # semiconductors, tech globally
            "fx":            0.70,   # USD/CNY, JPY, KRW
            "inflation":     0.50,
            "supply_chain":  0.95,   # TSMC, semiconductor
            "credit":        0.55,
            "energy_infra":  0.20,
        },

        "keywords": ["taiwan", "pla", "strait of taiwan", "tsmc",
                     "china taiwan", "taipei", "south china sea",
                     "xi jinping taiwan", "taiwan strait exercise",
                     "semiconductor taiwan"],

        "affected_equities":    ["Nasdaq 100", "S&P 500", "Hang Seng",
                                 "Shanghai Comp", "Nikkei 225", "TOPIX"],
        "affected_commodities": ["Copper"],
        "affected_fx":          ["USD/CNY", "USD/JPY"],
        "hedge_assets":         ["Gold", "US 20Y+ Treasury (TLT)"],

        "acled_id":        "taiwan_strait",     # maps to src/data/acled.py + gdelt.py
        "data_confidence": 0.72,
    },
]

# ── Security exposure registry ──────────────────────────────────────────────
# Structural exposure of each tracked asset to each conflict.
# Values: 0–1 relevance weight.
# Used by exposure.py for Structural Exposure Score computation.

SECURITY_EXPOSURE: dict[str, dict] = {
    # ── Commodities ───────────────────────────────────────────────────────
    "WTI Crude Oil": {
        "structural": {
            "ukraine_russia": 0.80, "iran_conflict": 0.95,
            "red_sea_houthi": 0.60, "israel_gaza":   0.40,
            "india_pakistan": 0.10, "taiwan_strait":  0.20,
        },
        "route_tags":    ["hormuz", "suez", "black_sea"],
        "sector_tags":   ["energy"],
        "sanction_tags": ["russia", "iran"],
    },
    "Brent Crude": {
        "structural": {
            "ukraine_russia": 0.80, "iran_conflict": 0.90,
            "red_sea_houthi": 0.65, "israel_gaza":   0.40,
            "india_pakistan": 0.10, "taiwan_strait":  0.20,
        },
        "route_tags":    ["hormuz", "suez"],
        "sector_tags":   ["energy"],
        "sanction_tags": ["russia", "iran"],
    },
    "Natural Gas": {
        "structural": {
            "ukraine_russia": 0.90, "iran_conflict": 0.55,
            "red_sea_houthi": 0.30, "israel_gaza":   0.20,
            "india_pakistan": 0.05, "taiwan_strait":  0.15,
        },
        "route_tags":    ["black_sea", "hormuz"],
        "sector_tags":   ["energy"],
        "sanction_tags": ["russia"],
    },
    "Gasoline (RBOB)": {
        "structural": {
            "ukraine_russia": 0.60, "iran_conflict": 0.85,
            "red_sea_houthi": 0.55, "israel_gaza":   0.30,
            "india_pakistan": 0.05, "taiwan_strait":  0.10,
        },
        "route_tags":    ["hormuz"],
        "sector_tags":   ["energy"],
        "sanction_tags": ["iran"],
    },
    "Heating Oil": {
        "structural": {
            "ukraine_russia": 0.70, "iran_conflict": 0.80,
            "red_sea_houthi": 0.50, "israel_gaza":   0.25,
            "india_pakistan": 0.05, "taiwan_strait":  0.10,
        },
        "route_tags":    ["hormuz"],
        "sector_tags":   ["energy"],
        "sanction_tags": ["russia", "iran"],
    },
    "Gold": {
        "structural": {
            "ukraine_russia": 0.70, "iran_conflict": 0.80,
            "red_sea_houthi": 0.35, "israel_gaza":   0.65,
            "india_pakistan": 0.60, "taiwan_strait":  0.70,
        },
        "route_tags":    [],
        "sector_tags":   ["safe_haven", "precious_metals"],
        "sanction_tags": [],
    },
    "Silver": {
        "structural": {
            "ukraine_russia": 0.50, "iran_conflict": 0.55,
            "red_sea_houthi": 0.25, "israel_gaza":   0.40,
            "india_pakistan": 0.35, "taiwan_strait":  0.50,
        },
        "route_tags":    [],
        "sector_tags":   ["safe_haven", "industrial"],
        "sanction_tags": [],
    },
    "Platinum": {
        "structural": {
            "ukraine_russia": 0.65, "iran_conflict": 0.20,
            "red_sea_houthi": 0.15, "israel_gaza":   0.10,
            "india_pakistan": 0.05, "taiwan_strait":  0.15,
        },
        "route_tags":    [],
        "sector_tags":   ["precious_metals", "industrial"],
        "sanction_tags": ["russia"],
    },
    "Copper": {
        "structural": {
            "ukraine_russia": 0.45, "iran_conflict": 0.20,
            "red_sea_houthi": 0.35, "israel_gaza":   0.15,
            "india_pakistan": 0.10, "taiwan_strait":  0.60,
        },
        "route_tags":    ["panama_canal", "suez"],
        "sector_tags":   ["industrial", "ev_supply"],
        "sanction_tags": [],
    },
    "Aluminum": {
        "structural": {
            "ukraine_russia": 0.75, "iran_conflict": 0.15,
            "red_sea_houthi": 0.30, "israel_gaza":   0.10,
            "india_pakistan": 0.05, "taiwan_strait":  0.20,
        },
        "route_tags":    [],
        "sector_tags":   ["industrial"],
        "sanction_tags": ["russia"],
    },
    "Nickel": {
        "structural": {
            "ukraine_russia": 0.85, "iran_conflict": 0.10,
            "red_sea_houthi": 0.25, "israel_gaza":   0.05,
            "india_pakistan": 0.05, "taiwan_strait":  0.15,
        },
        "route_tags":    [],
        "sector_tags":   ["industrial", "ev_supply"],
        "sanction_tags": ["russia"],
    },
    "Wheat": {
        "structural": {
            "ukraine_russia": 0.95, "iran_conflict": 0.15,
            "red_sea_houthi": 0.40, "israel_gaza":   0.20,
            "india_pakistan": 0.25, "taiwan_strait":  0.10,
        },
        "route_tags":    ["black_sea", "suez"],
        "sector_tags":   ["agriculture"],
        "sanction_tags": ["russia"],
    },
    "Corn": {
        "structural": {
            "ukraine_russia": 0.80, "iran_conflict": 0.10,
            "red_sea_houthi": 0.35, "israel_gaza":   0.10,
            "india_pakistan": 0.15, "taiwan_strait":  0.10,
        },
        "route_tags":    ["black_sea"],
        "sector_tags":   ["agriculture"],
        "sanction_tags": [],
    },
    "Soybeans": {
        "structural": {
            "ukraine_russia": 0.30, "iran_conflict": 0.10,
            "red_sea_houthi": 0.30, "israel_gaza":   0.05,
            "india_pakistan": 0.10, "taiwan_strait":  0.50,
        },
        "route_tags":    ["panama_canal"],
        "sector_tags":   ["agriculture"],
        "sanction_tags": [],
    },
    "Sugar #11": {
        "structural": {
            "ukraine_russia": 0.20, "iran_conflict": 0.15,
            "red_sea_houthi": 0.30, "israel_gaza":   0.10,
            "india_pakistan": 0.20, "taiwan_strait":  0.10,
        },
        "route_tags":    ["suez"],
        "sector_tags":   ["agriculture"],
        "sanction_tags": [],
    },
    "Coffee": {
        "structural": {
            "ukraine_russia": 0.10, "iran_conflict": 0.10,
            "red_sea_houthi": 0.35, "israel_gaza":   0.05,
            "india_pakistan": 0.05, "taiwan_strait":  0.05,
        },
        "route_tags":    ["suez"],
        "sector_tags":   ["agriculture"],
        "sanction_tags": [],
    },
    "Cotton": {
        "structural": {
            "ukraine_russia": 0.15, "iran_conflict": 0.10,
            "red_sea_houthi": 0.30, "israel_gaza":   0.05,
            "india_pakistan": 0.15, "taiwan_strait":  0.15,
        },
        "route_tags":    ["suez"],
        "sector_tags":   ["agriculture", "textiles"],
        "sanction_tags": [],
    },

    # ── Equity indices ────────────────────────────────────────────────────
    "S&P 500": {
        "structural": {
            "ukraine_russia": 0.40, "iran_conflict": 0.55,
            "red_sea_houthi": 0.35, "israel_gaza":   0.25,
            "india_pakistan": 0.10, "taiwan_strait":  0.60,
        },
        "route_tags":    [],
        "sector_tags":   ["broad_equity", "defense", "energy", "tech"],
        "sanction_tags": [],
    },
    "Nasdaq 100": {
        "structural": {
            "ukraine_russia": 0.30, "iran_conflict": 0.35,
            "red_sea_houthi": 0.25, "israel_gaza":   0.20,
            "india_pakistan": 0.10, "taiwan_strait":  0.85,
        },
        "route_tags":    [],
        "sector_tags":   ["tech", "semiconductors"],
        "sanction_tags": [],
    },
    "DJIA": {
        "structural": {
            "ukraine_russia": 0.40, "iran_conflict": 0.50,
            "red_sea_houthi": 0.35, "israel_gaza":   0.25,
            "india_pakistan": 0.10, "taiwan_strait":  0.50,
        },
        "route_tags":    [],
        "sector_tags":   ["broad_equity", "industrial", "energy"],
        "sanction_tags": [],
    },
    "Russell 2000": {
        "structural": {
            "ukraine_russia": 0.30, "iran_conflict": 0.40,
            "red_sea_houthi": 0.25, "israel_gaza":   0.20,
            "india_pakistan": 0.05, "taiwan_strait":  0.35,
        },
        "route_tags":    [],
        "sector_tags":   ["small_cap"],
        "sanction_tags": [],
    },
    "Eurostoxx 50": {
        "structural": {
            "ukraine_russia": 0.75, "iran_conflict": 0.55,
            "red_sea_houthi": 0.50, "israel_gaza":   0.35,
            "india_pakistan": 0.05, "taiwan_strait":  0.40,
        },
        "route_tags":    ["suez"],
        "sector_tags":   ["broad_equity", "energy", "industrial"],
        "sanction_tags": ["russia"],
    },
    "DAX": {
        "structural": {
            "ukraine_russia": 0.80, "iran_conflict": 0.55,
            "red_sea_houthi": 0.50, "israel_gaza":   0.30,
            "india_pakistan": 0.05, "taiwan_strait":  0.50,
        },
        "route_tags":    [],
        "sector_tags":   ["industrial", "auto", "chemicals"],
        "sanction_tags": ["russia"],
    },
    "CAC 40": {
        "structural": {
            "ukraine_russia": 0.70, "iran_conflict": 0.50,
            "red_sea_houthi": 0.45, "israel_gaza":   0.30,
            "india_pakistan": 0.05, "taiwan_strait":  0.40,
        },
        "route_tags":    ["suez"],
        "sector_tags":   ["luxury", "energy", "industrial"],
        "sanction_tags": ["russia"],
    },
    "FTSE 100": {
        "structural": {
            "ukraine_russia": 0.65, "iran_conflict": 0.55,
            "red_sea_houthi": 0.55, "israel_gaza":   0.30,
            "india_pakistan": 0.10, "taiwan_strait":  0.35,
        },
        "route_tags":    ["suez"],
        "sector_tags":   ["energy", "mining", "financials"],
        "sanction_tags": ["russia"],
    },
    "Nikkei 225": {
        "structural": {
            "ukraine_russia": 0.40, "iran_conflict": 0.75,
            "red_sea_houthi": 0.45, "israel_gaza":   0.25,
            "india_pakistan": 0.15, "taiwan_strait":  0.70,
        },
        "route_tags":    ["hormuz", "malacca"],
        "sector_tags":   ["auto", "tech", "industrial"],
        "sanction_tags": [],
    },
    "TOPIX": {
        "structural": {
            "ukraine_russia": 0.40, "iran_conflict": 0.75,
            "red_sea_houthi": 0.45, "israel_gaza":   0.25,
            "india_pakistan": 0.15, "taiwan_strait":  0.70,
        },
        "route_tags":    ["hormuz", "malacca"],
        "sector_tags":   ["broad_equity", "industrial"],
        "sanction_tags": [],
    },
    "Hang Seng": {
        "structural": {
            "ukraine_russia": 0.35, "iran_conflict": 0.40,
            "red_sea_houthi": 0.40, "israel_gaza":   0.15,
            "india_pakistan": 0.10, "taiwan_strait":  0.85,
        },
        "route_tags":    ["malacca", "taiwan_strait"],
        "sector_tags":   ["tech", "financials", "real_estate"],
        "sanction_tags": [],
    },
    "Shanghai Comp": {
        "structural": {
            "ukraine_russia": 0.30, "iran_conflict": 0.35,
            "red_sea_houthi": 0.35, "israel_gaza":   0.10,
            "india_pakistan": 0.10, "taiwan_strait":  0.90,
        },
        "route_tags":    ["malacca"],
        "sector_tags":   ["broad_equity", "industrial", "tech"],
        "sanction_tags": [],
    },
    "CSI 300": {
        "structural": {
            "ukraine_russia": 0.30, "iran_conflict": 0.35,
            "red_sea_houthi": 0.35, "israel_gaza":   0.10,
            "india_pakistan": 0.10, "taiwan_strait":  0.90,
        },
        "route_tags":    ["malacca"],
        "sector_tags":   ["broad_equity"],
        "sanction_tags": [],
    },
    "Sensex": {
        "structural": {
            "ukraine_russia": 0.35, "iran_conflict": 0.55,
            "red_sea_houthi": 0.30, "israel_gaza":   0.15,
            "india_pakistan": 0.90, "taiwan_strait":  0.20,
        },
        "route_tags":    ["hormuz"],
        "sector_tags":   ["broad_equity", "it", "financials"],
        "sanction_tags": [],
    },
    "Nifty 50": {
        "structural": {
            "ukraine_russia": 0.35, "iran_conflict": 0.55,
            "red_sea_houthi": 0.30, "israel_gaza":   0.15,
            "india_pakistan": 0.90, "taiwan_strait":  0.20,
        },
        "route_tags":    ["hormuz"],
        "sector_tags":   ["broad_equity", "it", "financials"],
        "sanction_tags": [],
    },
}

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

# ── Sector ETF universe (Benjamin: sector SPDRs for narrower analysis) ─────
# These complement the broad-index EQUITY_REGIONS with sector-level granularity.
# Each maps to a yfinance-compatible ticker.
SECTOR_ETFS: dict[str, str] = {
    "XLE":  "Energy Select SPDR",
    "XLF":  "Financial Select SPDR",
    "XLI":  "Industrial Select SPDR",
    "XLB":  "Materials Select SPDR",
    "XLP":  "Consumer Staples SPDR",
    "XLU":  "Utilities Select SPDR",
    "XLV":  "Health Care SPDR",
    "XLK":  "Technology Select SPDR",
    "XLY":  "Consumer Discr SPDR",
    "IYT":  "iShares US Transportation",   # Oil/Transport pair Benjamin suggested
    "USO":  "US Oil Fund (WTI proxy)",
    "UNG":  "US Nat Gas Fund",
    "PDBC": "Invesco Optimum Yield Commodities",
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

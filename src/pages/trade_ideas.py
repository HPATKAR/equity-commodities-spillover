"""
Page 6 - Trade Ideas
Regime-triggered + conflict-driven cross-asset trade cards.
Integrates conflict exposure scoring, scenario-aware payoff tables, QC grading,
filter sidebar, and agent debate threads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

from src.data.loader import load_returns, load_commodity_prices
from src.data.config import GEOPOLITICAL_EVENTS, PALETTE
from src.analysis.correlations import (
    average_cross_corr_series, detect_correlation_regime, rolling_correlation,
    composite_stress_index,
)
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _thread, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_header, _page_footer,
    _insight_note,
)

_REGIME_NAMES  = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
_REGIME_COLORS = {0: "#2e7d32",      1: "#555960", 2: "#e67e22",  3: "#c0392b"}


# ── Master Investor Lens ───────────────────────────────────────────────────
# Maps trade category → [manager, archetype, insight] aligned with
# fundamental/macro-driven style: Druckenmiller, Templeton, Marks, Naren, Sharma.
_MASTER_INVESTOR_LENS: dict[str, list[dict]] = {
    "Crisis Hedge": [
        {"manager": "Howard Marks",         "archetype": "CYCLE AWARE",
         "insight": "Risk control over return maximisation. Permanent capital loss is the only real risk. Being right eventually is worthless if leverage wipes you out before the thesis plays."},
        {"manager": "Stanley Druckenmiller","archetype": "MACRO TACTICIAN",
         "insight": "Ruthless capital preservation: prioritise zero-loss years above all else. Cut immediately the moment the macro thesis cracks."},
        {"manager": "Seth Klarman",         "archetype": "ABSOLUTE RETURN",
         "insight": "Cash is an asset class and an option. Liquidity evaporation during crisis is precisely when forced selling creates the asymmetric entry."},
    ],
    "Geopolitical": [
        {"manager": "John Templeton",       "archetype": "MAX PESSIMISM",
         "insight": "Buy at the point of maximum pessimism. Geopolitical panic is the entry point, not the exit. The crowd abandons assets at precisely the wrong moment."},
        {"manager": "Shankar Sharma",       "archetype": "MACRO INFLECTION",
         "insight": "Spotting inflection points early: geopolitical catalysts create the entry, but it is the fundamental macro story that drives the multi-year return."},
        {"manager": "Prem Watsa",           "archetype": "CONTRARIAN VALUE",
         "insight": "Macro-economic hedging with derivatives. Geo events create the asymmetric tail protection the Fairfax playbook demands. Never fight the geopolitical cycle."},
    ],
    "Macro": [
        {"manager": "Stanley Druckenmiller","archetype": "MACRO LIQUIDITY",
         "insight": "Liquidity drives markets above near-term earnings by an order of magnitude. Follow the money supply and central bank balance sheet — they are the real alpha signal."},
        {"manager": "S Naren",             "archetype": "DYNAMIC ALLOCATOR",
         "insight": "Dynamic asset allocation based on macro valuation metrics like Market Cap-to-GDP. Shift systematically; mean reversion is the law of financial gravity."},
        {"manager": "Shankar Sharma",       "archetype": "MACRO INFLECTION",
         "insight": "Global macro allocation based on structural data, not narrative. The largest forces — fiscal, monetary, demographic — move asset prices over years, not quarters."},
    ],
    "Growth": [
        {"manager": "Peter Lynch",          "archetype": "GARP",
         "insight": "Copper rising with equities confirms genuine demand expansion, not just liquidity fiction. Industrial metals are the purest economic truth-teller."},
        {"manager": "Rakesh Jhunjhunwala",  "archetype": "INDIA GARP",
         "insight": "When the fundamental thesis matches reality, back it with massive capital and hold through multi-year volatility. Growth plus momentum is the optimal combination."},
        {"manager": "Ramdeo Agrawal",       "archetype": "QGLP",
         "insight": "Buy right, sit tight. Low-correlation growth regimes with commodity confirmation are the multi-year compounders the QGLP framework was built to own."},
    ],
    "Private Credit": [
        {"manager": "Howard Marks",         "archetype": "CREDIT CYCLE",
         "insight": "Private credit is marked-to-model at cycle peaks. When HY OAS widens, the BDC NAV fiction unravels with a 1–2 quarter lag. Permanent capital loss is the only risk that matters."},
        {"manager": "Seth Klarman",         "archetype": "DISTRESSED",
         "insight": "Forced institutional selling driven by credit rating downgrades is the Baupost entry point. The psychology of institutional ignorance creates the asymmetric mispricing."},
        {"manager": "David Tepper",         "archetype": "DISTRESSED DEBT",
         "insight": "Invest at the absolute bottom of the capital structure. Distressed debt converts to dominant post-restructuring equity — the Appaloosa playbook from the 2009 financials trade."},
    ],
    "Fixed Income": [
        {"manager": "Howard Marks",         "archetype": "CYCLE AWARE",
         "insight": "Asymmetry of returns: capture the quality safe-haven bid while shorting the credit spread blow-out. Know where you are in the credit cycle."},
        {"manager": "S Naren",             "archetype": "DYNAMIC ALLOCATOR",
         "insight": "Shift systematically from equity to fixed income when macro valuation metrics signal cycle-peak complacency. Patience in contrariness."},
        {"manager": "Prem Watsa",           "archetype": "MACRO HEDGE",
         "insight": "Prohibiting leverage and using macro derivatives to protect against systemic credit collapse is the Fairfax doctrine. Duration safety hedges deflation scenarios."},
    ],
    "India/EM": [
        {"manager": "S Naren",             "archetype": "INDIA MACRO",
         "insight": "Patience in contrariness: invest in deeply undervalued sectors facing cyclical headwinds. India's oil import shock is a cyclical headwind — not a structural ruin."},
        {"manager": "Prashant Jain",        "archetype": "CONTRA-CYCLICAL",
         "insight": "Contra-cyclical value discipline: buy out-of-favour sectors at peak cyclical pain. Ask whether the business survives over a 10-year horizon — not 10 weeks."},
        {"manager": "John Templeton",       "archetype": "MAX PESSIMISM",
         "insight": "Maximum pessimism in EM is the entry point. INR stress and crude shock at extremes is precisely the Templeton setup. Buy when blood is in the streets."},
    ],
    "Asia Divergence": [
        {"manager": "Shankar Sharma",       "archetype": "MACRO INFLECTION",
         "insight": "Spotting inflection points early across cap ranges. China property is structural, not cyclical. Japan is the mirror image — BOJ policy normalisation is a decade-long re-rating."},
        {"manager": "Kerr Neilson",         "archetype": "GLOBAL CONTRARIAN",
         "insight": "True global sourcing agility: allocate into regions when valuations become compelling. Japan at Shiller CAPE 22x vs China property distress is a screaming divergence trade."},
        {"manager": "S Naren",             "archetype": "DYNAMIC ALLOCATOR",
         "insight": "Macro risk tracking: China leverage cycle is in systemic unwind. Mean reversion for Japan is supported by BOJ policy normalisation and structural Yen tailwind."},
    ],
    "Dollar Cycle": [
        {"manager": "Stanley Druckenmiller","archetype": "MACRO LIQUIDITY",
         "insight": "The dollar cycle is the single most powerful force for EM asset re-rating. Liquidity flows globally — when DXY peaks, EM assets inflect sharply."},
        {"manager": "John Templeton",       "archetype": "MAX PESSIMISM",
         "insight": "Buy at maximum pessimism. When EM is universally abandoned at a dollar peak, it is universally mispriced. The fundamental equation reverses."},
        {"manager": "Kerr Neilson",         "archetype": "GLOBAL CONTRARIAN",
         "insight": "True global sourcing agility: allocate freely across geographies when dollar-driven valuation dislocations create compelling entry points in EM."},
    ],
}

# ── Specific tradeable instruments for each trade ──────────────────────────
# {trade_name: {asset_name: "TICKER — description"}}
# Used for display only — backtest uses the asset name against return data columns.
_TRADE_TICKERS: dict[str, dict[str, str]] = {
    "Long Gold / Short Eurostoxx 50": {
        "Gold":          "GLD  — SPDR Gold Shares (NYSE)",
        "Eurostoxx 50":  "FEZ  — SPDR Euro Stoxx 50 ETF (NYSE)",
    },
    "Long Natural Gas / Short Nikkei 225": {
        "Natural Gas":   "UNG  — United States Natural Gas Fund (NYSE)",
        "Nikkei 225":    "EWJ  — iShares MSCI Japan ETF (NYSE)",
    },
    "Long Wheat / Long Gold / Short Emerging Markets": {
        "Wheat":         "WEAT — Teucrium Wheat Fund (NYSE)",
        "Gold":          "GLD  — SPDR Gold Shares (NYSE)",
        "Sensex":        "EEM  — iShares MSCI Emerging Markets ETF (NYSE)",
    },
    "Long Copper / Long S&P 500": {
        "Copper":        "CPER — United States Copper Index Fund (NYSE)",
        "S&P 500":       "SPY  — SPDR S&P 500 ETF Trust (NYSE)",
    },
    "Long WTI Crude / Short S&P 500 Energy-Heavy Sectors": {
        "WTI Crude Oil": "USO  — United States Oil Fund (NYSE) | XLE short for sector precision",
        "S&P 500":       "SPY  — SPDR S&P 500 ETF Trust (NYSE)",
    },
    "Long Gold, Long Silver / Short Copper, Short Shanghai Comp": {
        "Gold":          "GLD  — SPDR Gold Shares (NYSE)",
        "Silver":        "SLV  — iShares Silver Trust (NYSE)",
        "Copper":        "CPER — United States Copper Index Fund (NYSE)",
        "Shanghai Comp": "MCHI — iShares MSCI China ETF (NYSE)",
    },
    "Short BDC Basket / Long HY Credit Protection": {
        "Ares Capital (ARCC)": "ARCC — Ares Capital Corp (NASDAQ) — short",
        "Blue Owl (OBDC)":     "OBDC — Blue Owl Capital Corp (NYSE) — short",
        "Gold":                "GLD  — SPDR Gold Shares (NYSE) — long",
    },
    "Long TLT / Short HYG (Flight to Quality)": {
        "US 20Y+ Treasury (TLT)": "TLT  — iShares 20+ Year Treasury Bond ETF (NYSE)",
        "HY Corporate (HYG)":     "HYG  — iShares iBoxx $ High Yield Corporate Bond ETF (NYSE) — short",
    },
    "Long TIP / Short TLT (Inflation Breakeven Trade)": {
        "TIPS / Inflation (TIP)":  "TIP  — iShares TIPS Bond ETF (NYSE)",
        "US 20Y+ Treasury (TLT)":  "TLT  — iShares 20+ Year Treasury Bond ETF (NYSE) — short",
    },
    "Long Brent Crude / Short Nifty 50 (India Import Shock)": {
        "Brent Crude":  "BNO  — United States Brent Oil Fund (NYSE)",
        "Nifty 50":     "INDY — iShares India 50 ETF (NYSE) | NIFTYBEES.NS (NSE)",
    },
    "Long Gold / Short INR (India Geopolitical Hedge)": {
        "Gold":    "GLD  — SPDR Gold Shares (NYSE)",
        "USD/INR": "USDINR=X — Forex spot | USDINR futures on NSE",
    },
    "Long EMB / Short DXY (Dollar Debasement - EM Relief)": {
        "EM USD Bonds (EMB)":  "EMB  — iShares J.P. Morgan USD EM Bond ETF (NYSE)",
        "DXY (Dollar Index)":  "UUP  — Invesco DB US Dollar Index Bullish Fund (NYSE) — short",
        "Gold":                "GLD  — SPDR Gold Shares (NYSE)",
    },
    "Long Gold / Short TLT (Fiscal Dominance / Dollar Debasement)": {
        "Gold":                    "GLD  — SPDR Gold Shares (NYSE) | GDX for miners leverage",
        "US 20Y+ Treasury (TLT)": "TLT  — iShares 20+ Year Treasury Bond ETF (NYSE) — short",
    },
    "Long EM Asia / Short DXY (Max Pessimism EM Reversal)": {
        "Shanghai Comp":      "MCHI — iShares MSCI China ETF (NYSE) | 2800.HK Tracker Fund",
        "Sensex":             "INDA — iShares MSCI India ETF (NYSE) | NIFTYBEES.NS",
        "DXY (Dollar Index)": "UUP  — Invesco DB US Dollar Index Bullish Fund (NYSE) — short",
    },
    "Long LQD / Short HYG (Credit Cycle Peak — Quality Flight)": {
        "IG Corporate (LQD)": "LQD  — iShares iBoxx $ IG Corporate Bond ETF (NYSE)",
        "HY Corporate (HYG)": "HYG  — iShares iBoxx $ HY Corporate Bond ETF (NYSE) — short",
    },
    "Long SHY / Long Gold (Fed Pivot Front-Run)": {
        "US 1-3Y Treasury (SHY)": "SHY  — iShares 1-3 Year Treasury Bond ETF (NYSE)",
        "Gold":                    "GLD  — SPDR Gold Shares (NYSE) | GDX for leveraged exposure",
    },
    "Long Nifty 50 / Short Brent (India Rate Cut + Oil Tailwind)": {
        "Nifty 50":     "INDA — iShares MSCI India ETF (NYSE) | NIFTYBEES.NS (NSE)",
        "Brent Crude":  "BNO  — United States Brent Oil Fund (NYSE) — short",
    },
    "Short Shanghai Comp / Long Nikkei 225 (China Deflation vs Japan Reflation)": {
        "Shanghai Comp": "FXI  — iShares China Large-Cap ETF (NYSE) — short via puts",
        "Nikkei 225":    "EWJ  — iShares MSCI Japan ETF (NYSE)",
    },
}


# ── Trade idea library ─────────────────────────────────────────────────────
_TRADE_LIBRARY = [
    {
        "regime":    [2, 3],
        "trigger":   "Elevated/Crisis correlation",
        "name":      "Long Gold / Short Eurostoxx 50",
        "rationale": "When cross-asset correlation spikes, equities and commodities sell off together. "
                     "Gold decouples as safe-haven demand absorbs panic flows. "
                     "Short European equities amplifies energy-cost transmission risk.",
        "entry":     "Enter when 60d avg cross-asset corr > 0.45 and DCC(Gold/SPX) < −0.1",
        "exit":      "Close when correlation regime drops back to Normal for 5+ days",
        "risk":      "Correlation snap-back; Central bank intervention can flip gold",
        "assets":    ["Gold", "Eurostoxx 50"],
        "direction": ["Long", "Short"],
        "category":  "Crisis Hedge",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Energy supply shock",
        "name":      "Long Natural Gas / Short Nikkei 225",
        "rationale": "Japan is the world's largest LNG importer. "
                     "Natural gas supply shocks directly impair Japanese manufacturing margins "
                     "and current account. Energy spikes translate to yen weakness and equity underperformance.",
        "entry":     "Enter on Ukraine escalation OR Strait of Hormuz closure signal + NG vol spike",
        "exit":      "Peace signal, supply restoration, or NG price mean-reversion (−20% from entry)",
        "risk":      "BOJ FX intervention; domestic LNG stockpile release",
        "assets":    ["Natural Gas", "Nikkei 225"],
        "direction": ["Long", "Short"],
        "category":  "Geopolitical",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Wheat/food supply disruption (Ukraine War)",
        "name":      "Long Wheat / Long Gold / Short Emerging Markets",
        "rationale": "Food price spikes trigger inflation in EM countries with high cereal import ratios. "
                     "Combined with USD strength (gold hedge), EM equities face dual pressure from "
                     "import inflation and capital outflows.",
        "entry":     "Wheat 30d return > +15% + political instability signals in MENA/SSA",
        "exit":      "Wheat normalises to 12M average; EM carry recovery",
        "risk":      "IMF/World Bank intervention; US export restrictions",
        "assets":    ["Wheat", "Gold", "Sensex"],
        "direction": ["Long", "Long", "Short"],
        "category":  "Macro",
    },
    {
        "regime":    [0, 1],
        "trigger":   "Global growth recovery (low correlation regime)",
        "name":      "Long Copper / Long S&P 500",
        "rationale": "Copper is the premier global growth bellwether. "
                     "When correlation is low, commodities and equities price "
                     "independent fundamentals - copper rising with equities signals "
                     "genuine demand expansion, not just liquidity.",
        "entry":     "Copper 60d momentum > 0, ISM Manufacturing > 50, cross-asset corr < 0.20",
        "exit":      "Copper momentum reversal; ISM contraction; Fed pause signals",
        "risk":      "China property market collapse; USD spike from geopolitical safe-haven",
        "assets":    ["Copper", "S&P 500"],
        "direction": ["Long", "Long"],
        "category":  "Growth",
    },
    {
        "regime":    [1, 2],
        "trigger":   "Oil-equity divergence (supply shock premium)",
        "name":      "Long WTI Crude / Short S&P 500 Energy-Heavy Sectors",
        "rationale": "When oil spikes from a supply shock (not demand), "
                     "energy futures gain while broader equities face margin compression. "
                     "The long/short captures the spread between commodity producer "
                     "and equity consumer dynamics.",
        "entry":     "Brent-WTI spread widens + OPEC+ surprise cut + SPX P/E compression",
        "exit":      "Supply restoration event; recession pricing dominates oil",
        "risk":      "Demand destruction flips correlation; tech-led equity rally decouples",
        "assets":    ["WTI Crude Oil", "S&P 500"],
        "direction": ["Long", "Short"],
        "category":  "Macro",
    },
    {
        "regime":    [3],
        "trigger":   "Full crisis - all correlations elevated",
        "name":      "Long Gold, Long Silver / Short Copper, Short Shanghai Comp",
        "rationale": "Full crisis regime: precious metals outperform as industrial metals "
                     "and EM equities (particularly China) collapse under dollar strength "
                     "and risk-off flows. Gold/Silver spread also captures "
                     "the industrial-vs-monetary metals divergence.",
        "entry":     "Crisis regime active > 3 days; VIX > 35; DXY trending up",
        "exit":      "Regime drops below Elevated; Fed emergency action; VIX < 25",
        "risk":      "Chinese stimulus; commodity demand front-running recovery",
        "assets":    ["Gold", "Silver", "Copper", "Shanghai Comp"],
        "direction": ["Long", "Long", "Short", "Short"],
        "category":  "Crisis Hedge",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Private credit stress / HY spread widening + BDC underperformance",
        "name":      "Short BDC Basket / Long HY Credit Protection",
        "rationale": "Private credit ($2T+ AUM) is illiquid and marked-to-model quarterly. "
                     "When HY OAS widens >35bps in 90 days and BKLN underperforms SPY, "
                     "BDC equity (ARCC, OBDC, FSK) reprices before private marks surface - "
                     "a lagged NAV writedown is typically confirmed 1–2 quarters later. "
                     "The CDX HY 5Y index provides liquid short exposure to the same credit universe. "
                     "Gold long captures contagion into safe-haven flows as financial sector sells off.",
        "entry":     "HY OAS >350bps and rising >30bps/month; BKLN below 200d MA; "
                     "BDC basket -5% vs SPY on 60d basis; elevated/crisis equity-commodity regime",
        "exit":      "HY OAS stabilises <300bps; Fed emergency rate cut signal; "
                     "BDC premiums re-compress; credit facilities extended at par",
        "risk":      "Fed emergency cuts collapse floating-rate costs rapidly; "
                     "sponsor rescue financing delays visible defaults; "
                     "interval fund gates mask redemption pressure for 12–18 months",
        "assets":    ["Ares Capital (ARCC)", "Blue Owl (OBDC)", "Gold"],
        "direction": ["Short", "Short", "Long"],
        "category":  "Private Credit",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Flight to quality - elevated/crisis correlation regime",
        "name":      "Long TLT / Short HYG (Flight to Quality)",
        "rationale": "In elevated and crisis correlation regimes, investors rotate from credit risk to duration safety. "
                     "TLT captures the safe-haven Treasury bid while HYG shorts the credit spread widening. "
                     "The trade isolates the quality spread compression that accompanies every risk-off episode.",
        "entry":     "Crisis/Elevated regime active; HY OAS rising >30bps in 30 days; VIX >25; TLT above 200d MA",
        "exit":      "Regime drops to Normal; HY OAS stabilises; Fed pivot signal",
        "risk":      "Bear steepening (long rates rise with HY spreads simultaneously); fiscal dominance narrative",
        "assets":    ["US 20Y+ Treasury (TLT)", "HY Corporate (HYG)"],
        "direction": ["Long", "Short"],
        "category":  "Fixed Income",
    },
    {
        "regime":    [1, 2],
        "trigger":   "Inflation breakeven expansion / stagflation risk",
        "name":      "Long TIP / Short TLT (Inflation Breakeven Trade)",
        "rationale": "When commodity prices spike (energy, food) and the Fed is behind the curve, "
                     "real yields compress while nominal yields stay elevated. "
                     "TIPS outperform nominal Treasuries as breakeven inflation widens. "
                     "This trade directly monetises the commodity-to-bond inflation transmission channel.",
        "entry":     "5Y breakeven inflation rising >20bps in 60 days; WTI or wheat up >15% in 30 days; CPI surprise positive",
        "exit":      "Breakeven inflation peaks; commodity prices mean-revert; Fed delivers credible inflation response",
        "risk":      "Demand destruction flips inflation to deflation; recession pricing overrides inflation premium",
        "assets":    ["TIPS / Inflation (TIP)", "US 20Y+ Treasury (TLT)"],
        "direction": ["Long", "Short"],
        "category":  "Fixed Income",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Oil spike + INR depreciation (India crude import stress)",
        "name":      "Long Brent Crude / Short Nifty 50 (India Import Shock)",
        "rationale": "India imports ~85% of its crude oil needs (~5 mb/d). "
                     "When Brent spikes >15% in 60 days and USD/INR is depreciating simultaneously, "
                     "India's current account deficit widens sharply - historically correlating with "
                     "Nifty 50 underperformance of -6% to -12% vs global peers. "
                     "The long Brent / short Nifty trade monetises the commodity-to-EM-equity transmission channel specific to India.",
        "entry":     "Brent up >15% in 60 days AND USD/INR rising >3% in 30 days; elevated/crisis regime active",
        "exit":      "Oil supply restored; INR stabilises / RBI intervention; India CAD narrows",
        "risk":      "RBI forex reserve intervention caps INR weakness; India domestic demand surprises to upside; OPEC+ production cut reversal",
        "assets":    ["Brent Crude", "Nifty 50"],
        "direction": ["Long", "Short"],
        "category":  "India/EM",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Geopolitical stress + dollar strength - India gold import sensitivity",
        "name":      "Long Gold / Short INR (India Geopolitical Hedge)",
        "rationale": "India is the world's #2 gold consumer (~800-900 tonnes/year). "
                     "During geopolitical stress (Middle East escalation, South Asia tensions), "
                     "gold demand surges from Indian households as a safe-haven AND currency hedge. "
                     "Simultaneously, USD/INR typically rises (INR weakens) under global risk-off conditions. "
                     "The long gold / short INR (via USD/INR long) trade captures both legs of this transmission.",
        "entry":     "Crisis/Elevated regime; VIX >25; USD/INR rising >2% in 20 days; Gold 20d momentum positive",
        "exit":      "Geopolitical de-escalation; gold mean-reverts -8% from peak; INR stabilises on RBI action",
        "risk":      "RBI aggressively defends INR using $620B forex reserves; Gold sell-off on Fed hawkish surprise",
        "assets":    ["Gold", "USD/INR"],
        "direction": ["Long", "Long"],
        "category":  "India/EM",
    },
    {
        "regime":    [0, 1],
        "trigger":   "Dollar weakness cycle / EM credit rally",
        "name":      "Long EMB / Short DXY (Dollar Debasement - EM Relief)",
        "rationale": "A weakening dollar reduces dollar-denominated debt service costs for EM sovereigns, "
                     "compresses EM credit spreads, and attracts capital inflows into EM assets. "
                     "EMB captures the bond price appreciation; short DXY amplifies the currency leg. "
                     "Gold is the commodity expression of the same dollar debasement theme.",
        "entry":     "DXY down >3% in 60 days; Fed on hold or cutting; EM current accounts improving; commodity prices rising",
        "exit":      "Dollar reversal; Fed hawkish pivot; EM-specific credit event; commodity demand collapse",
        "risk":      "EM-specific credit events (sovereign default, political crisis); commodity demand collapse flips EM outlook",
        "assets":    ["EM USD Bonds (EMB)", "DXY (Dollar Index)", "Gold"],
        "direction": ["Long", "Short", "Long"],
        "category":  "Fixed Income",
    },

    # ── NEW: Macro-fundamental / largest-forces style trades ─────────────────
    {
        "regime":         [1, 2, 3],
        "trigger":        "US fiscal deficit >6% GDP + Fed balance sheet monetisation + real rates collapsing",
        "name":           "Long Gold / Short TLT (Fiscal Dominance / Dollar Debasement)",
        "rationale":      (
            "When the US fiscal deficit exceeds 6% of GDP and the Fed is monetising debt (balance sheet expanding), "
            "real rates collapse and fiat currency credibility erodes. Gold — the monetary metal — outperforms nominal "
            "Treasuries structurally. Druckenmiller: 'Liquidity drives markets above all else. When the Fed prints and "
            "fiscal expands, the answer is always gold over bonds.' Watsa: macro-hedging against systemic dollar "
            "debasement. Templeton: real returns focus — nominal treasury yields minus realized inflation = deeply "
            "negative real returns. The math is unambiguous. "
            "ETFs: Long GLD (SPDR Gold Shares), Short TLT (iShares 20+ Year Treasury Bond ETF). "
            "Leverage play: Long GDX (VanEck Gold Miners) for 2–3× operating leverage to gold price."
        ),
        "entry":          "US fiscal deficit >6% GDP AND WALCL expanding >5% in 90d AND gold 60d momentum positive AND 10Y real yield (DFII10) below −0.5%",
        "exit":           "Fed credibly tightens / balance sheet shrinks; gold −12% from entry peak; deficit narrows below 4%",
        "stop":           "GLD below 200d MA for 5+ days; TLT rallies >8% in 30d (fear-driven flight)",
        "target":         "GLD +20–35% over 12 months; TLT −15–20% as nominal yields re-price fiscal risk",
        "invalidation":   "Fed delivers credible quantitative tightening >$100B/month; Congress passes credible deficit reduction",
        "holding_period": "6–18 months (structural macro thesis)",
        "assets":         ["Gold", "US 20Y+ Treasury (TLT)"],
        "direction":      ["Long", "Short"],
        "category":       "Macro",
        "investor_lens":  ["Stanley Druckenmiller", "Prem Watsa", "John Templeton"],
    },
    {
        "regime":         [0, 1],
        "trigger":        "DXY at 3-year high + EM PMIs recovering + US current account deficit widening",
        "name":           "Long EM Asia / Short DXY (Max Pessimism EM Reversal)",
        "rationale":      (
            "Templeton's core principle: buy at the point of maximum pessimism. When the dollar peaks at 3-year highs "
            "and EM assets are universally abandoned, the fundamental equation reverses. EM GDP growth premium vs DM "
            "is widening; current accounts improving; dollar debasement is structural as US fiscal position deteriorates. "
            "Kerr Neilson: 'True global sourcing agility — allocate into regions when valuations become compelling.' "
            "Druckenmiller: the dollar cycle is the most powerful single force for EM asset re-rating. "
            "Instruments: Long MCHI (iShares MSCI China ETF) + INDA (iShares MSCI India ETF); Short UUP (Invesco DB "
            "USD Index Bullish Fund). Add EEM for broad EM exposure."
        ),
        "entry":          "DXY (DX-Y.NYB) at 3Y high and declining; EM Manufacturing PMI >50 for 3 consecutive months; FII outflows reversing; US 10Y real yield peaking",
        "exit":           "DXY reversal back above prior 3Y high; EM credit event; commodity demand collapse reversal",
        "stop":           "DXY breaks above prior 3Y high with momentum; Shanghai Comp −10% from entry on fresh stimulus failure",
        "target":         "EM Asia +25–40% over 18–24 months; DXY −8–12% as US fiscal dominance weakens the dollar",
        "invalidation":   "Fed re-accelerates rate hikes; EM sovereign credit event; China hard landing confirmed",
        "holding_period": "12–24 months",
        "assets":         ["Shanghai Comp", "Sensex", "DXY (Dollar Index)"],
        "direction":      ["Long", "Long", "Short"],
        "category":       "Dollar Cycle",
        "investor_lens":  ["John Templeton", "Kerr Neilson", "Stanley Druckenmiller"],
    },
    {
        "regime":         [1, 2],
        "trigger":        "HY–IG spread compression to cycle lows + leveraged loan issuance at 3Y high + Fed at terminal rate",
        "name":           "Long LQD / Short HYG (Credit Cycle Peak — Quality Flight)",
        "rationale":      (
            "Howard Marks: 'Gauge risk as permanent loss of capital, not volatility. Calibrate aggressiveness based on "
            "where the market stands in the credit cycle.' When HY–IG spreads compress to cycle lows (<200 bps), "
            "leveraged loan issuance is at records, and the Fed has reached terminal rate — credit cycle is at peak. "
            "Investment-grade bonds offer duration safety as HY reprices default risk. Marks: 'Asymmetry of returns — "
            "capture upside while protecting downside.' Klarman: forced institutional selling driven by credit rating "
            "downgrades is the Baupost entry point. "
            "ETFs: Long LQD (iShares iBoxx $ Investment Grade Corporate Bond ETF); "
            "Short HYG (iShares iBoxx $ High Yield Corporate Bond ETF). "
            "Alternative: CDX IG index long vs CDX HY index short for pure credit spread pair."
        ),
        "entry":          "HY OAS (BAMLH0A0HYM2) − IG OAS (BAMLC0A0CM) <200 bps AND leveraged loan issuance at 3Y high AND Fed funds held >6 months at cycle high",
        "exit":           "HY spreads widen to 400 bps; credit cycle turns; recession confirmed by 2Q negative GDP",
        "stop":           "LQD −5% from entry (bear-steepening scenario); HYG −8% in risk-off panic (correlations spike — exit both legs)",
        "target":         "LQD +8–12% as rates rally; HYG −15–25% as spreads blow out on first credit event",
        "invalidation":   "Fed pivots to cuts immediately; fiscal stimulus backstops credit markets; HY default rate stays below 2%",
        "holding_period": "6–18 months",
        "assets":         ["IG Corporate (LQD)", "HY Corporate (HYG)"],
        "direction":      ["Long", "Short"],
        "category":       "Fixed Income",
        "investor_lens":  ["Howard Marks", "Seth Klarman", "Prem Watsa"],
    },
    {
        "regime":         [1, 2, 3],
        "trigger":        "Fed funds at cycle peak + unemployment +0.5% from trough + 2Y–10Y curve bull-steepening",
        "name":           "Long SHY / Long Gold (Fed Pivot Front-Run)",
        "rationale":      (
            "Druckenmiller: 'Liquidity drives markets vastly above near-term earnings. Never fight the Fed.' When the "
            "Fed is at terminal rate, unemployment has risen 0.5% from trough, and the yield curve begins "
            "bull-steepening — the next macro move is rate cuts. 2Y Treasuries front-run the cut cycle with "
            "mathematical certainty (SHY +4–6% per 100 bps cut). Gold front-runs the real yield collapse and "
            "dollar weakness that follows easing. S Naren: 'Dynamic asset allocation — shift systematically to "
            "duration and gold at the cycle peak. Market Cap-to-GDP >1.0 confirms equity risk is too high.' "
            "ETFs: Long SHY (iShares 1-3 Year Treasury Bond ETF); Long GLD (SPDR Gold Shares). "
            "Tactical overlay: GDX (VanEck Gold Miners) for leveraged gold exposure on confirmed cut signals."
        ),
        "entry":          "Fed funds held >6 months at cycle high AND unemployment +0.5% from cycle trough AND 2Y–10Y curve bull-steepening AND gold above 200d MA",
        "exit":           "Fed delivers 100 bps+ of cuts (fully priced in); gold −10% from cycle high; 2Y yields stop declining",
        "stop":           "CPI re-accelerates above 3.5%; 2Y yields break back above Fed funds rate; unemployment stops rising",
        "target":         "SHY +4–6% on 100–150 bps cut cycle; GLD +20–40% over 12 months on real yield collapse",
        "invalidation":   "No recession; unemployment reverses before Fed cuts; inflation re-accelerates to >3.5%",
        "holding_period": "6–18 months",
        "assets":         ["US 1-3Y Treasury (SHY)", "Gold"],
        "direction":      ["Long", "Long"],
        "category":       "Macro",
        "investor_lens":  ["Stanley Druckenmiller", "S Naren", "Howard Marks"],
    },
    {
        "regime":         [0, 1],
        "trigger":        "RBI rate cut cycle + Brent <$80 + India CAD improving + USD/INR stabilising",
        "name":           "Long Nifty 50 / Short Brent (India Rate Cut + Oil Tailwind)",
        "rationale":      (
            "S Naren: 'Contra-cyclical value: accumulate quality Indian equities when the macro headwinds peak.' "
            "India imports ~85% of crude oil needs. When Brent falls below $80 AND the RBI begins cutting, "
            "the India macro equation flips structurally: CAD narrows, INR stabilises, real household incomes rise, "
            "and corporate margins expand. This is the inverse of the India Import Shock trade — it fires when oil "
            "stress reverses. Prashant Jain: 'Business longevity — ask whether the company grows cash flows over 10 "
            "years. India's structural consumption story is intact below $80 oil.' "
            "ETFs: Long INDA (iShares MSCI India ETF) or NIFTYBEES.NS (NSE); "
            "Short BNO (United States Brent Oil Fund) or BZ=F futures."
        ),
        "entry":          "Brent below $80 AND declining; RBI cuts >25 bps; USD/INR <84.5 and stable; Nifty 50 P/E <22x",
        "exit":           "Oil reversal above $90; RBI pauses; India CAD widens above 3% GDP; Nifty P/E >28x",
        "stop":           "Brent spikes above $90 on new supply shock; RBI reverses cuts; INR depreciates >5% in 30d",
        "target":         "Nifty 50 +20–30% over 12 months; short Brent +10–15% as oil normalises",
        "invalidation":   "India-Pakistan escalation resumes; US recession triggers global EM selloff",
        "holding_period": "6–12 months",
        "assets":         ["Nifty 50", "Brent Crude"],
        "direction":      ["Long", "Short"],
        "category":       "India/EM",
        "investor_lens":  ["S Naren", "Prashant Jain", "Ramdeo Agrawal"],
    },
    {
        "regime":         [2, 3],
        "trigger":        "China property developer debt crisis deepening + Japan BOJ policy normalisation + Yen below ¥148",
        "name":           "Short Shanghai Comp / Long Nikkei 225 (China Deflation vs Japan Reflation)",
        "rationale":      (
            "Shankar Sharma: 'Spotting inflection points early across cap ranges. China property is structural, not "
            "cyclical — $300B+ in offshore dollar debt with demographic reversal and structural oversupply.' Japan is "
            "the mirror image: Yen weakness (USD/JPY >148) boosts export profits, domestic reflation is accelerating, "
            "and the Nikkei Shiller CAPE at ~22x is cheap vs its own 30-year history. Kerr Neilson: 'True global "
            "sourcing agility — Japan valuations are compelling. China property is a regulatory and demographic trap. "
            "Rotate with conviction.' S Naren: 'Macro risk tracking: China leverage cycle is in systemic unwind. "
            "Mean reversion for Japan is supported by BOJ policy normalisation.' "
            "ETFs: Short FXI (iShares China Large-Cap ETF) via puts or inverse ETF; Long EWJ (iShares MSCI Japan ETF). "
            "Leverage: Short MCHI for broader China exposure; add DXJ (WisdomTree Japan Hedged Equity) if hedging Yen."
        ),
        "entry":          "FXI below 200d MA AND China property PMI <45 AND USD/JPY >148 AND Japan PMI >52 AND BOJ holds rates",
        "exit":           "China stimulus package >$500B announced; BOJ hikes >50 bps driving Yen to <130; FXI rallies >15%",
        "stop":           "FXI rallies >12% on China stimulus surprise; BOJ pivot pauses",
        "target":         "FXI −20–35% on debt restructuring events; EWJ +20–30% on Yen carry + reflation trade",
        "invalidation":   "China PBOC delivers massive credit stimulus; BOJ reverses course; global recession kills Japan exports",
        "holding_period": "6–24 months",
        "assets":         ["Shanghai Comp", "Nikkei 225"],
        "direction":      ["Short", "Long"],
        "category":       "Asia Divergence",
        "investor_lens":  ["Shankar Sharma", "Kerr Neilson", "S Naren"],
    },
]

_CATEGORY_COLORS = {
    "Crisis Hedge":    "#c0392b",
    "Geopolitical":    "#e67e22",
    "Macro":           "#2980b9",
    "Growth":          "#2e7d32",
    "Private Credit":  "#8e44ad",
    "Fixed Income":    "#2980b9",
    "India/EM":        "#16a085",
    "Dollar Cycle":    "#1abc9c",
    "Asia Divergence": "#9b59b6",
}

# ── S&P 500 stock universe for AI Trade Structurer ────────────────────────────
# ~150 S&P 500 members organised by macro-relevant sector.
# At call time, _select_sectors_for_signal() picks the 3-5 most relevant sectors
# for the current regime/scenario so the AI context stays lean (~40 stocks max).
# Format: ticker → (display_name, sector)
_SP500_UNIVERSE: dict[str, tuple[str, str]] = {
    # ── Energy: Integrated, E&P, Refining, OFS ──────────────────────────────
    "XOM":  ("ExxonMobil",            "Energy"),
    "CVX":  ("Chevron",               "Energy"),
    "COP":  ("ConocoPhillips",        "Energy"),
    "EOG":  ("EOG Resources",         "Energy"),
    "OXY":  ("Occidental Petroleum",  "Energy"),
    "DVN":  ("Devon Energy",          "Energy"),
    "HES":  ("Hess",                  "Energy"),
    "MRO":  ("Marathon Oil",          "Energy"),
    "APA":  ("APA Corp",              "Energy"),
    "FANG": ("Diamondback Energy",    "Energy"),
    "PSX":  ("Phillips 66",           "Energy"),
    "VLO":  ("Valero Energy",         "Energy"),
    "MPC":  ("Marathon Petroleum",    "Energy"),
    "SLB":  ("SLB (Schlumberger)",    "Energy"),
    "HAL":  ("Halliburton",           "Energy"),
    "BKR":  ("Baker Hughes",          "Energy"),
    "KMI":  ("Kinder Morgan",         "Energy"),
    "WMB":  ("Williams Companies",    "Energy"),
    "OKE":  ("ONEOK",                 "Energy"),
    # ── Defense & Aerospace ─────────────────────────────────────────────────
    "LMT":  ("Lockheed Martin",       "Defense"),
    "RTX":  ("RTX (Raytheon)",        "Defense"),
    "NOC":  ("Northrop Grumman",      "Defense"),
    "GD":   ("General Dynamics",      "Defense"),
    "BA":   ("Boeing",                "Defense"),
    "LHX":  ("L3Harris Technologies", "Defense"),
    "HII":  ("Huntington Ingalls",    "Defense"),
    "LDOS": ("Leidos",                "Defense"),
    "TDG":  ("TransDigm Group",       "Defense"),
    "AXON": ("Axon Enterprise",       "Defense"),
    # ── Airlines ────────────────────────────────────────────────────────────
    "DAL":  ("Delta Air Lines",       "Airlines"),
    "UAL":  ("United Airlines",       "Airlines"),
    "AAL":  ("American Airlines",     "Airlines"),
    "LUV":  ("Southwest Airlines",    "Airlines"),
    "ALK":  ("Alaska Air Group",      "Airlines"),
    # ── Gold & Precious Metals Mining ───────────────────────────────────────
    "NEM":  ("Newmont",               "Gold Mining"),
    "GOLD": ("Barrick Gold",          "Gold Mining"),
    "AEM":  ("Agnico Eagle",          "Gold Mining"),
    "WPM":  ("Wheaton Precious Metals","Gold Mining"),
    # ── Industrial Metals & Mining ──────────────────────────────────────────
    "FCX":  ("Freeport-McMoRan",      "Industrial Metals"),
    "NUE":  ("Nucor Steel",           "Industrial Metals"),
    "CLF":  ("Cleveland-Cliffs",      "Industrial Metals"),
    "X":    ("US Steel",              "Industrial Metals"),
    "AA":   ("Alcoa",                 "Industrial Metals"),
    "MP":   ("MP Materials",          "Industrial Metals"),
    "SCCO": ("Southern Copper",       "Industrial Metals"),
    # ── Agriculture & Fertilizers ───────────────────────────────────────────
    "MOS":  ("Mosaic (Fertilizers)",  "Agriculture"),
    "ADM":  ("Archer-Daniels-Midland","Agriculture"),
    "BG":   ("Bunge Global",          "Agriculture"),
    "CF":   ("CF Industries",         "Agriculture"),
    "CTVA": ("Corteva Agriscience",   "Agriculture"),
    "FMC":  ("FMC Corp",              "Agriculture"),
    # ── Technology ──────────────────────────────────────────────────────────
    "AAPL": ("Apple",                 "Tech"),
    "MSFT": ("Microsoft",             "Tech"),
    "NVDA": ("NVIDIA",                "Tech"),
    "GOOGL":("Alphabet (A)",          "Tech"),
    "META": ("Meta Platforms",        "Tech"),
    "AMZN": ("Amazon",                "Tech"),
    "TSLA": ("Tesla",                 "Tech"),
    "AMD":  ("Advanced Micro Devices","Tech"),
    "INTC": ("Intel",                 "Tech"),
    "QCOM": ("Qualcomm",              "Tech"),
    "AVGO": ("Broadcom",              "Tech"),
    "CRM":  ("Salesforce",            "Tech"),
    "ORCL": ("Oracle",                "Tech"),
    "NOW":  ("ServiceNow",            "Tech"),
    "ADBE": ("Adobe",                 "Tech"),
    # ── Financials ──────────────────────────────────────────────────────────
    "JPM":  ("JPMorgan Chase",        "Financials"),
    "BAC":  ("Bank of America",       "Financials"),
    "GS":   ("Goldman Sachs",         "Financials"),
    "MS":   ("Morgan Stanley",        "Financials"),
    "WFC":  ("Wells Fargo",           "Financials"),
    "C":    ("Citigroup",             "Financials"),
    "BLK":  ("BlackRock",             "Financials"),
    "SCHW": ("Charles Schwab",        "Financials"),
    "COF":  ("Capital One",           "Financials"),
    "AXP":  ("American Express",      "Financials"),
    "BX":   ("Blackstone",            "Financials"),
    "KKR":  ("KKR & Co",              "Financials"),
    # ── Healthcare ──────────────────────────────────────────────────────────
    "UNH":  ("UnitedHealth Group",    "Healthcare"),
    "LLY":  ("Eli Lilly",             "Healthcare"),
    "JNJ":  ("Johnson & Johnson",     "Healthcare"),
    "ABBV": ("AbbVie",                "Healthcare"),
    "MRK":  ("Merck",                 "Healthcare"),
    "PFE":  ("Pfizer",                "Healthcare"),
    "TMO":  ("Thermo Fisher",         "Healthcare"),
    "ABT":  ("Abbott Laboratories",   "Healthcare"),
    "BMY":  ("Bristol-Myers Squibb",  "Healthcare"),
    "AMGN": ("Amgen",                 "Healthcare"),
    "ISRG": ("Intuitive Surgical",    "Healthcare"),
    "VRTX": ("Vertex Pharmaceuticals","Healthcare"),
    # ── Consumer Staples (safe-haven) ───────────────────────────────────────
    "PG":   ("Procter & Gamble",      "Consumer Staples"),
    "KO":   ("Coca-Cola",             "Consumer Staples"),
    "PEP":  ("PepsiCo",               "Consumer Staples"),
    "WMT":  ("Walmart",               "Consumer Staples"),
    "COST": ("Costco",                "Consumer Staples"),
    "MO":   ("Altria Group",          "Consumer Staples"),
    "PM":   ("Philip Morris",         "Consumer Staples"),
    "CL":   ("Colgate-Palmolive",     "Consumer Staples"),
    "GIS":  ("General Mills",         "Consumer Staples"),
    "KR":   ("Kroger",                "Consumer Staples"),
    # ── Consumer Discretionary ──────────────────────────────────────────────
    "MCD":  ("McDonald's",            "Consumer Discretionary"),
    "SBUX": ("Starbucks",             "Consumer Discretionary"),
    "NKE":  ("Nike",                  "Consumer Discretionary"),
    "HD":   ("Home Depot",            "Consumer Discretionary"),
    "TGT":  ("Target",                "Consumer Discretionary"),
    "F":    ("Ford Motor",            "Consumer Discretionary"),
    "GM":   ("General Motors",        "Consumer Discretionary"),
    "TJX":  ("TJX Companies",         "Consumer Discretionary"),
    "LOW":  ("Lowe's",                "Consumer Discretionary"),
    "BKNG": ("Booking Holdings",      "Consumer Discretionary"),
    "RCL":  ("Royal Caribbean",       "Consumer Discretionary"),
    "CCL":  ("Carnival Corp",         "Consumer Discretionary"),
    # ── Industrials ─────────────────────────────────────────────────────────
    "CAT":  ("Caterpillar",           "Industrials"),
    "DE":   ("Deere & Company",       "Industrials"),
    "HON":  ("Honeywell",             "Industrials"),
    "GE":   ("GE Aerospace",          "Industrials"),
    "UPS":  ("United Parcel Service", "Industrials"),
    "FDX":  ("FedEx",                 "Industrials"),
    "ETN":  ("Eaton Corp",            "Industrials"),
    "EMR":  ("Emerson Electric",      "Industrials"),
    "PCAR": ("PACCAR",                "Industrials"),
    "MMM":  ("3M",                    "Industrials"),
    # ── Utilities (safe-haven, rate-sensitive) ───────────────────────────────
    "NEE":  ("NextEra Energy",        "Utilities"),
    "DUK":  ("Duke Energy",           "Utilities"),
    "SO":   ("Southern Company",      "Utilities"),
    "D":    ("Dominion Energy",       "Utilities"),
    "EXC":  ("Exelon",                "Utilities"),
    "AEP":  ("AEP",                   "Utilities"),
    "SRE":  ("Sempra Energy",         "Utilities"),
    # ── Materials ────────────────────────────────────────────────────────────
    "LIN":  ("Linde",                 "Materials"),
    "APD":  ("Air Products",          "Materials"),
    "SHW":  ("Sherwin-Williams",      "Materials"),
    "ECL":  ("Ecolab",                "Materials"),
    "DOW":  ("Dow Inc",               "Materials"),
    "DD":   ("DuPont",                "Materials"),
    "PPG":  ("PPG Industries",        "Materials"),
    # ── Real Estate ──────────────────────────────────────────────────────────
    "AMT":  ("American Tower",        "Real Estate"),
    "PLD":  ("Prologis",              "Real Estate"),
    "EQIX": ("Equinix",               "Real Estate"),
    "SPG":  ("Simon Property Group",  "Real Estate"),
    # ── Communications ───────────────────────────────────────────────────────
    "NFLX": ("Netflix",               "Communications"),
    "DIS":  ("Walt Disney",           "Communications"),
    "T":    ("AT&T",                  "Communications"),
    "VZ":   ("Verizon",               "Communications"),
    "CMCSA":("Comcast",               "Communications"),
}

# Maps signal context → which sectors to pull for the AI (keeps prompt lean)
_SECTOR_SIGNAL_MAP: dict[str, list[str]] = {
    "supply_shock":    ["Energy", "Agriculture", "Industrial Metals", "Defense"],
    "escalation":      ["Energy", "Defense", "Gold Mining", "Airlines"],
    "sanctions_shock": ["Energy", "Defense", "Financials", "Industrial Metals"],
    "shipping_shock":  ["Energy", "Industrials", "Airlines", "Consumer Discretionary"],
    "risk_off":        ["Gold Mining", "Consumer Staples", "Utilities", "Defense"],
    "crisis":          ["Gold Mining", "Defense", "Consumer Staples", "Utilities"],
    "de_escalation":   ["Airlines", "Consumer Discretionary", "Tech", "Industrials"],
    "recovery":        ["Tech", "Consumer Discretionary", "Financials", "Industrials"],
    "base":            ["Tech", "Financials", "Healthcare", "Consumer Discretionary"],
    "default":         ["Energy", "Tech", "Financials", "Defense", "Healthcare"],
}

# Always included alongside signal-driven sectors (core reference)
_ANCHOR_SECTORS = ["Energy", "Tech", "Financials"]


def _select_sectors_for_signal(regime_level: int, scenario_id: str | None) -> list[str]:
    """Pick 4-6 relevant sectors based on regime and active scenario."""
    signal_sectors = _SECTOR_SIGNAL_MAP.get(scenario_id or "default",
                                             _SECTOR_SIGNAL_MAP["default"])
    # Crisis/Elevated regime always adds safe-haven and defense
    if regime_level >= 3:
        signal_sectors = list(dict.fromkeys(
            signal_sectors + ["Gold Mining", "Defense", "Consumer Staples", "Utilities"]
        ))
    elif regime_level >= 2:
        signal_sectors = list(dict.fromkeys(signal_sectors + ["Gold Mining", "Defense"]))

    # Merge with anchor sectors (no duplicates, preserve order)
    combined = list(dict.fromkeys(signal_sectors + _ANCHOR_SECTORS))
    return combined[:7]  # cap at 7 sectors ≈ 40-50 stocks


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_stock_prices(sectors: tuple[str, ...] = ()) -> dict[str, float]:
    """
    Fetch latest closing prices for S&P 500 universe stocks.
    If sectors is provided, fetches only those sectors; otherwise fetches all.
    Cached 15 min.
    """
    try:
        import yfinance as yf
        sector_set = set(sectors)
        tickers = [
            t for t, (_, s) in _SP500_UNIVERSE.items()
            if not sector_set or s in sector_set
        ]
        if not tickers:
            return {}
        raw = yf.download(tickers, period="5d", progress=False, auto_adjust=True)["Close"]
        if raw.empty:
            return {}
        latest = raw.ffill().iloc[-1]
        return {str(t): round(float(v), 2) for t, v in latest.items() if not np.isnan(v)}
    except Exception:
        return {}


def _format_stock_context(prices: dict[str, float], sectors: list[str]) -> str:
    """Format stock prices compactly by sector for the AI context block."""
    if not prices:
        return ""
    sector_set = set(sectors)
    lines: list[str] = [
        "S&P 500 STOCK REFERENCE PRICES (live — use these for specific entry/target/stop):"
    ]
    by_sector: dict[str, list[str]] = {}
    for ticker, (name, sector) in _SP500_UNIVERSE.items():
        if sector_set and sector not in sector_set:
            continue
        price = prices.get(ticker)
        if price is None:
            continue
        by_sector.setdefault(sector, []).append(f"{ticker} ${price:.2f}")
    for sector in sectors:
        items = by_sector.get(sector)
        if items:
            lines.append(f"  {sector}: {', '.join(items)}")
    return "\n".join(lines)


_TI_STYLE = """<style>
/* ── Trade Ideas Page Design System ────────────────────────── */
.ti-card{border:1px solid #1e1e1e;background:#0d0d0d;margin-bottom:.6rem;overflow:hidden}
.ti-hdr{background:#111;border-bottom:1px solid #1e1e1e;padding:.45rem .9rem;
  display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap}
.ti-hdr-lbl{font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase}
.ti-badges{display:flex;gap:4px;align-items:center;flex-wrap:wrap}
.ti-pill{font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;
  padding:2px 6px;color:#fff;border-radius:1px}
.ti-badge{font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;
  padding:1px 6px;letter-spacing:.08em;border-radius:1px}
.ti-body{padding:.65rem .9rem}
.ti-name{font-family:'DM Sans',sans-serif;font-size:13px;font-weight:700;
  color:#f0f0f0;line-height:1.3;margin-bottom:5px}
.ti-dir{font-family:'JetBrains Mono',monospace;font-size:9px;
  color:#8890a1;margin-bottom:9px;line-height:1.6}
.ti-meta{display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
.ti-lbl{font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;
  letter-spacing:.10em;text-transform:uppercase;color:#555960}
.ti-conf{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700}
.ti-qc{font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;padding:2px 7px;color:#fff}
.ti-rationale{font-family:'DM Sans',sans-serif;font-size:11px;color:#cccccc;
  line-height:1.65;margin-bottom:10px}
.ti-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-bottom:5px}
.ti-ext-grid{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:5px;margin-top:5px}
.ti-cell{background:#0a0a0a;border:1px solid #1a1a1a;padding:.35rem .55rem}
.ti-cell-lbl{font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;
  letter-spacing:.10em;text-transform:uppercase;margin-bottom:3px}
.ti-cell-val{font-family:'DM Sans',sans-serif;font-size:10px;color:#ddd;line-height:1.45}
.ti-strip{display:flex;gap:14px;padding:5px .9rem;align-items:center;
  flex-wrap:wrap;border-top:1px solid #1a1a1a}
.ti-strip-tag{font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;
  letter-spacing:.10em;text-transform:uppercase;min-width:56px}
.ti-strip-val{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700}
.ti-strip-dim{font-family:'JetBrains Mono',monospace;font-size:8px;color:#444c5c;margin-left:auto}
.ti-why{display:flex;gap:4px;flex-wrap:wrap;padding:4px .9rem .45rem}
.ti-why-chip{font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
  padding:2px 7px;border:1px solid #1a3a1a;background:#0a1a0a;color:#27ae60}
.ti-exp{display:flex;gap:5px;flex-wrap:wrap;padding:4px .9rem .5rem}
.ti-exp-cell{background:#0a0a0a;border:1px solid #1a1a1a;padding:4px 8px;min-width:80px}
.ti-exp-name{font-family:'JetBrains Mono',monospace;font-size:8px;color:#555960;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:110px;margin-bottom:2px}
.ti-exp-sas{font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700}
.ti-kpi{border:1px solid #1e1e1e;padding:.6rem .85rem;background:#0d0d0d}
.ti-kpi-lbl{font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
  text-transform:uppercase;letter-spacing:.12em;color:#CFB991;margin-bottom:4px}
.ti-kpi-val{font-family:'JetBrains Mono',monospace;font-size:1.0rem;font-weight:700}
.ti-geo-bar{background:#080808;border:1px solid #1e1e1e;padding:.5rem 1rem;
  margin-bottom:.65rem;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
/* Entrance animations */
@keyframes ti-card-in{0%{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.ti-card-anim{animation:ti-card-in .45s ease-out both}
</style>"""


def _render_investor_lens_strip(col, trade: dict) -> None:
    """Slim strip showing which money managers' philosophy validates this trade and why."""
    category     = trade.get("category", "Macro")
    custom_names = trade.get("investor_lens", [])

    # Build lens list: prefer trade-specific managers, else fall back to category map
    if custom_names:
        category_pool = _MASTER_INVESTOR_LENS.get(category, []) + sum(_MASTER_INVESTOR_LENS.values(), [])
        lens_data = [l for l in category_pool if l["manager"] in custom_names][:3]
        # pad with category defaults if we got fewer than requested
        if len(lens_data) < len(custom_names):
            seen = {l["manager"] for l in lens_data}
            for l in _MASTER_INVESTOR_LENS.get(category, []):
                if l["manager"] not in seen:
                    lens_data.append(l)
                    seen.add(l["manager"])
                    if len(lens_data) >= 3:
                        break
    else:
        lens_data = _MASTER_INVESTOR_LENS.get(category, [])[:3]

    if not lens_data:
        return

    items_html = ""
    for lens in lens_data:
        mgr     = lens.get("manager", "")
        arch    = lens.get("archetype", "")
        insight = lens.get("insight", "")
        items_html += (
            f'<div style="display:flex;gap:10px;align-items:flex-start;'
            f'padding:4px 0;border-bottom:1px solid #111">'
            f'<div style="min-width:160px;flex-shrink:0">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'font-weight:700;color:#CFB991">{mgr.upper()}</span><br>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'color:#555960;letter-spacing:.07em">{arch}</span>'
            f'</div>'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:10px;'
            f'color:#888;line-height:1.55">&ldquo;{insight}&rdquo;</span>'
            f'</div>'
        )

    col.markdown(
        f'<div style="background:#060606;border-top:1px solid #1a1a0a;'
        f'padding:.5rem .9rem .4rem .9rem">'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        f'font-weight:700;letter-spacing:.12em;color:#5a4a20;'
        f'text-transform:uppercase;margin-bottom:6px">MASTER INVESTOR LENS</div>'
        + items_html
        + '</div>',
        unsafe_allow_html=True,
    )


def _parse_holding_days(trade: dict, default: int = 30) -> int:
    """
    Derive backtest holding window from the trade's stated holding_period field.
    Uses the lower bound of the range as the conservative test horizon.
    Caps at 252 days (1 year) to keep backtest samples meaningful.
    """
    import re
    hp = trade.get("holding_period", "")
    if not hp:
        return default
    nums = re.findall(r"\d+", hp.lower())
    if not nums:
        return default
    lo = int(nums[0])
    if "month" in hp.lower():
        return max(default, min(lo * 21, 252))
    if "week" in hp.lower():
        return max(default, lo * 5)
    return default


def _compute_leg_weights(
    trade: dict,
    asset_exposure: dict | None,
) -> list[float]:
    """
    Compute conviction-weighted allocation per trade leg.
    Weights incorporate:
      - Trade-level confidence score
      - Per-asset SAS (Structural Asset Score) from exposure data
      - Direction modifier: safe-haven assets weighted higher in Crisis regime
    Returns normalised list of floats that sum to 1.0.
    """
    assets     = trade.get("assets", [])
    directions = trade.get("direction", [])
    n          = len(assets)
    if n == 0:
        return []

    confidence = float(trade.get("confidence", 0.60))
    weights: list[float] = []

    for asset, direction in zip(assets, directions):
        base = 1.0 / n   # equal-weight base
        # SAS modifier: higher exposure → more conviction on the right directional side
        if asset_exposure and asset in asset_exposure:
            sas = float(asset_exposure[asset].get("sas", 50))
            asset_dir = asset_exposure[asset].get("direction", "neutral")
            # Align SAS with trade direction: long geo-risk assets get SAS boost on long legs
            if direction.lower() == "long" and asset_dir == "long_geo_risk":
                base *= (1 + sas / 200)
            elif direction.lower() == "short" and asset_dir == "safe_haven":
                base *= (1 + sas / 200)
            elif direction.lower() == "long" and asset_dir == "safe_haven":
                base *= (1 + sas / 300)  # smaller boost — safe haven long is defensive
        # Confidence modifier
        base *= (0.5 + confidence)
        weights.append(max(base, 1e-6))

    total = sum(weights)
    return [w / total for w in weights]


@st.cache_data(show_spinner=False, ttl=3600)
def _backtest_trade(
    _all_r: pd.DataFrame,
    _regimes: pd.Series,
    trade_name: str,
    trigger_regimes: list[int],
    assets: list[str],
    directions: list[str],
    holding_days: int = 30,
    leg_weights: tuple[float, ...] | None = None,   # conviction-weighted allocation per leg
) -> dict:
    """
    Historical backtest for a single trade idea.

    Signal: every time the regime enters one of `trigger_regimes`
    (first day of that regime), enter the trade.
    Hold for `holding_days` business days.

    Portfolio P&L:
      - Equal-weight if leg_weights is None.
      - Conviction-weighted (from _compute_leg_weights) if provided.
        Weights capture confidence score × SAS modifier per leg.
        Weighted backtest shows the approach as deployed, not a naive basket.

    Returns:
      n_signals, win_rate (%), avg_return (%), sharpe, max_drawdown (%),
      available_assets, missing_assets, weighted (bool flag)
    """
    # Only backtest assets available in the returns DataFrame
    avail = [(a, d) for a, d in zip(assets, directions) if a in _all_r.columns]
    if not avail or _all_r.empty or _regimes.empty:
        return {"n_signals": 0, "error": "Insufficient data"}

    # Build leg weights aligned to available assets
    if leg_weights is not None and len(leg_weights) == len(assets):
        # Re-index weights to only available assets and renormalise
        avail_w: list[float] = []
        for a, _ in avail:
            idx = assets.index(a)
            avail_w.append(leg_weights[idx])
        total_w = sum(avail_w) or 1.0
        avail_w = [w / total_w for w in avail_w]
        is_weighted = True
    else:
        avail_w = [1.0 / len(avail)] * len(avail)
        is_weighted = False

    # Align regime to returns index.
    # limit=20: forward-fill only up to 20 trading days (intra-week alignment).
    # Dates before the regime series begins, or gaps > 20 days, receive sentinel -1
    # so they are excluded from signal generation rather than mislabelled Normal.
    reg = _regimes.reindex(_all_r.index, method="ffill", limit=20).fillna(-1).astype(int)

    # Entry signal: first day of a qualifying regime
    in_regime    = reg.isin(trigger_regimes).astype(int)
    entry_signal = (in_regime.diff() == 1)       # rising edge = regime just entered
    entry_dates  = _all_r.index[entry_signal].tolist()

    if not entry_dates:
        return {"n_signals": 0, "error": "No entry signals in history"}

    trade_returns = []
    for entry in entry_dates:
        try:
            i_start = _all_r.index.get_loc(entry)
        except KeyError:
            continue
        i_end = min(i_start + holding_days, len(_all_r) - 1)
        if i_end <= i_start:
            continue
        window = _all_r.iloc[i_start: i_end]
        # Compute conviction-weighted portfolio return
        leg_rets = []
        for (asset, direction), w in zip(avail, avail_w):
            leg_ret = (1 + window[asset]).prod() - 1   # compound return over window
            signed  = leg_ret if direction.lower() == "long" else -leg_ret
            leg_rets.append(signed * w)
        if leg_rets:
            trade_returns.append(sum(leg_rets) * 100)  # in %

    if len(trade_returns) < 3:
        return {"n_signals": len(trade_returns), "error": "Too few signals to backtest"}

    tr = np.array(trade_returns)
    wins     = (tr > 0).sum()
    win_rate = wins / len(tr) * 100
    avg_ret  = float(tr.mean())
    std_ret  = float(tr.std()) if len(tr) > 1 else 1.0
    sharpe   = avg_ret / (std_ret + 1e-8) * np.sqrt(252 / holding_days)

    # Max drawdown: waterfall of cumulative returns
    cum  = np.cumprod(1 + tr / 100)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / (peak + 1e-8) * 100
    max_dd = float(dd.min())

    return {
        "n_signals":         len(tr),
        "win_rate":          round(win_rate, 1),
        "avg_return":        round(avg_ret, 2),
        "sharpe":            round(sharpe, 2),
        "max_drawdown":      round(max_dd, 2),
        "available_assets":  [a for a, _ in avail],
        "missing_assets":    [a for a, _ in zip(assets, directions) if a not in _all_r.columns],
        "weighted":          is_weighted,
        "leg_weights":       {a: round(w, 3) for (a, _), w in zip(avail, avail_w)},
    }


def _render_trade_card(
    col,
    trade: dict,
    all_r_concat: pd.DataFrame,
    current: int,
    trade_idx: int,
    asset_exposure: dict | None = None,
    regimes: "pd.Series | None" = None,
) -> None:
    """Render a single trade card with QC grade, confidence, payoff table, and debate thread."""
    # Suppress low-confidence generated ideas to keep the page signal-to-noise high
    _conf_raw = float(trade.get("confidence", 0.60))
    if trade.get("generated") and _conf_raw < 0.55:
        return

    cat_col = _CATEGORY_COLORS.get(trade["category"], "#CFB991")

    # ── QC scoring ─────────────────────────────────────────────────────────
    try:
        from src.analysis.trade_filter import score_trade_quality
        qc = score_trade_quality(trade)
    except Exception:
        qc = {"score": 60, "grade": "B", "flags": []}

    grade       = qc["grade"]
    qc_score    = qc["score"]
    grade_color = {"A": "#27ae60", "B": "#2980b9", "C": "#e67e22", "D": "#c0392b"}.get(grade, "#8890a1")
    confidence  = float(trade.get("confidence", 0.60))
    conf_pct    = f"{confidence * 100:.0f}%"
    conf_color  = "#27ae60" if confidence >= 0.70 else "#e67e22" if confidence >= 0.55 else "#c0392b"

    # ── Badges ─────────────────────────────────────────────────────────────
    is_generated  = trade.get("generated", False)
    conflict_id   = trade.get("conflict_id")
    source_badge  = (
        '<span class="ti-badge" style="background:#1a1a2e;color:#CFB991">LIVE GEO</span>'
        if is_generated else
        '<span class="ti-badge" style="background:#1e1e1e;color:#555960">STATIC</span>'
    )
    conflict_badge = (
        f'<span class="ti-badge" style="background:#3d1a00;color:#e67e22">'
        f'{conflict_id.upper().replace("_", " ")}</span>'
        if conflict_id else ""
    )

    dir_html = " &nbsp;·&nbsp; ".join(
        f'<span style="color:{"#27ae60" if d == "Long" else "#c0392b"};font-weight:700">'
        f'{d}</span>&nbsp;{a}'
        for a, d in zip(trade["assets"], trade["direction"])
    )

    # ── Specific tradeable instruments ─────────────────────────────────────
    # Priority: static _TRADE_TICKERS lookup → tickers dict on trade → asset name fallback
    _ticker_map_static = _TRADE_TICKERS.get(trade.get("name", ""), {})
    _ticker_map_gen    = trade.get("tickers") or {}
    # For AI-structured trades tickers is a freeform string; handle both str and dict
    if isinstance(_ticker_map_gen, str):
        _ticker_map_gen = {}
    _ticker_parts = []
    for _a, _d in zip(trade.get("assets", []), trade.get("direction", [])):
        _t = _ticker_map_static.get(_a) or _ticker_map_gen.get(_a)
        if _t:
            _clr = "#27ae60" if _d.lower() == "long" else "#c0392b"
            _ticker_parts.append(
                f'<span style="color:{_clr};font-weight:700">{_d[0]}</span>'
                f'&nbsp;<span style="color:#8890a1;font-weight:600">{_t}</span>'
            )
    # AI-structured trades carry tickers as a single descriptive string — surface it directly
    _ai_tickers_str = trade.get("tickers", "") if isinstance(trade.get("tickers"), str) else ""
    if not _ticker_parts and _ai_tickers_str:
        _ticker_parts = [f'<span style="color:#8890a1">{_ai_tickers_str}</span>']
    ticker_html = (
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        'color:#8890a1;margin-bottom:8px;line-height:2;display:flex;flex-wrap:wrap;gap:10px">'
        + "  ".join(_ticker_parts)
        + '</div>'
    ) if _ticker_parts else ""

    regime_pills = "".join(
        f'<span class="ti-pill" style="background:{_REGIME_COLORS[r]}">{_REGIME_NAMES[r]}</span>'
        for r in trade.get("regime", [])
    )

    # ── Entry / Exit / Risk cells ───────────────────────────────────────────
    def _cell(lbl: str, val: str, lbl_col: str, extra: str = "") -> str:
        return (
            f'<div class="ti-cell"{(" " + extra) if extra else ""}>'
            f'<div class="ti-cell-lbl" style="color:{lbl_col}">{lbl}</div>'
            f'<div class="ti-cell-val">{val or "—"}</div>'
            f'</div>'
        )

    grid_html = (
        '<div class="ti-grid">'
        + _cell("Entry",  trade.get("entry", "—"), "#CFB991")
        + _cell("Exit",   trade.get("exit",  "—"), "#8890a1")
        + _cell("Risk",   trade.get("risk",  "—"), "#c0392b", 'style="border-left:2px solid #220000"')
        + '</div>'
    )

    # ── AI-structured specific fields (entry_price_ref, upside_pct, stop_loss, options_structure)
    ai_price_html = ""
    _ai_entry   = trade.get("entry_price_ref", "")
    _ai_upside  = trade.get("upside_pct", "")
    _ai_stop    = trade.get("stop_loss", "")
    _ai_opts    = trade.get("options_structure", "")
    _gen_upside = trade.get("upside_pct")  # numeric from generator
    if any([_ai_entry, _ai_upside, _ai_stop, _ai_opts]):
        ai_price_html = (
            '<div class="ti-ext-grid" style="border-top:1px solid #1a1a1a;margin-top:4px">'
            + (_cell("Entry Ref",    _ai_entry,  "#CFB991") if _ai_entry  else "")
            + (_cell("Upside",       _ai_upside, "#27ae60") if _ai_upside else "")
            + (_cell("Stop-Loss",    _ai_stop,   "#c0392b") if _ai_stop   else "")
            + (_cell("Options Alt.", _ai_opts,   "#2980b9") if _ai_opts   else "")
            + '</div>'
        )
    elif isinstance(_gen_upside, (int, float)):
        ai_price_html = (
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            'color:#8890a1;margin-top:4px">'
            f'Est. upside: <span style="color:#27ae60;font-weight:700">~{_gen_upside:.1f}%</span>'
            '</div>'
        )

    ext_html = ""
    if any(trade.get(k) for k in ["stop", "target", "invalidation", "holding_period"]):
        ext_html = (
            '<div class="ti-ext-grid">'
            + _cell("Stop",         trade.get("stop",           "—"), "#27ae60")
            + _cell("Target",       trade.get("target",         "—"), "#27ae60")
            + _cell("Invalidation", trade.get("invalidation",   "—"), "#2980b9")
            + _cell("Hold Period",  trade.get("holding_period", "—"), "#8890a1")
            + '</div>'
        )

    card_delay = f"{trade_idx * 0.06:.2f}s"

    with col:
        st.markdown(
            f'<div class="ti-card ti-card-anim" '
            f'style="border-left:3px solid {cat_col};animation-delay:{card_delay}">'
            # ── Header ──────────────────────────────────────────────────────
            f'<div class="ti-hdr">'
            f'<span class="ti-hdr-lbl" style="color:{cat_col}">'
            f'{trade["category"]} &nbsp;·&nbsp; {trade["trigger"]}</span>'
            f'<div class="ti-badges">{regime_pills} {source_badge} {conflict_badge}</div>'
            f'</div>'
            # ── Body ────────────────────────────────────────────────────────
            f'<div class="ti-body">'
            f'<div class="ti-name">{trade["name"]}</div>'
            f'<div class="ti-dir">{dir_html}</div>'
            + ticker_html
            + f'<div class="ti-meta">'
            f'<span class="ti-lbl">Confidence</span>'
            f'<span class="ti-conf" style="color:{conf_color}">{conf_pct}</span>'
            f'<span class="ti-lbl" style="margin-left:6px">QC</span>'
            f'<span class="ti-qc" style="background:{grade_color}">{grade} &middot; {qc_score}</span>'
            f'</div>'
            f'<p class="ti-rationale">{trade["rationale"]}</p>'
            + grid_html
            + ai_price_html
            + ext_html
            + '</div>'  # ti-body
            + '</div>',  # ti-card
            unsafe_allow_html=True,
        )

        # ── Projected P&L strip ───────────────────────────────────────────
        try:
            from src.analysis.profit_projection import project_trade
            _proj   = project_trade(trade, current_regime=current)
            _epnl   = _proj["expected_pnl"]
            _wpnl   = _proj["worst_case_pnl"]
            _bprob  = _proj["breakeven_prob"]
            _sharpe = _proj["sharpe_proxy"]
            _ec = "#27ae60" if _epnl >= 0 else "#c0392b"
            _wc = "#c0392b" if _wpnl < -5  else "#e67e22"
            with col:
                st.markdown(
                    f'<div class="ti-strip" style="background:#080808">'
                    f'<span class="ti-strip-tag" style="color:#CFB991">Projected</span>'
                    f'<span class="ti-strip-val" style="color:{_ec}">E[P&L]&nbsp;{_epnl:+.1f}%</span>'
                    f'<span class="ti-strip-val" style="color:{_wc}">Worst&nbsp;{_wpnl:+.1f}%</span>'
                    f'<span class="ti-strip-val" style="color:#8E9AAA">BE&nbsp;{_bprob * 100:.0f}%</span>'
                    f'<span class="ti-strip-val" style="color:#CFB991">Sharpe&nbsp;{_sharpe:.2f}</span>'
                    f'<span class="ti-strip-dim">model estimate</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

        # ── Historical backtest strip (conviction-weighted) ───────────────
        try:
            if regimes is not None and not all_r_concat.empty:
                # Compute conviction weights for this trade
                _leg_w = _compute_leg_weights(trade, asset_exposure)
                _leg_w_tuple = tuple(_leg_w) if _leg_w else None
                _bt = _backtest_trade(
                    all_r_concat,
                    regimes,
                    trade_name=trade["name"],
                    trigger_regimes=trade.get("regime", [2, 3]),
                    assets=trade.get("assets", []),
                    directions=trade.get("direction", []),
                    holding_days=_parse_holding_days(trade),
                    leg_weights=_leg_w_tuple,
                )
                if "error" not in _bt and _bt.get("n_signals", 0) >= 3:
                    _bt_wr = _bt["win_rate"]
                    _bt_ar = _bt["avg_return"]
                    _bt_sh = _bt["sharpe"]
                    _bt_dd = _bt["max_drawdown"]
                    _bt_n  = _bt["n_signals"]
                    _bt_wt = _bt.get("weighted", False)
                    _bt_lw = _bt.get("leg_weights", {})
                    _wr_c = "#27ae60" if _bt_wr >= 55 else "#e67e22" if _bt_wr >= 45 else "#c0392b"
                    _ar_c = "#27ae60" if _bt_ar >= 0 else "#c0392b"
                    _sh_c = "#27ae60" if _bt_sh >= 0.5 else "#e67e22" if _bt_sh >= 0 else "#c0392b"
                    _wt_label = "conviction-wtd" if _bt_wt else "equal-wtd"
                    # Leg weight tooltip string
                    _wt_str = " · ".join(
                        f"{a[:8]}:{w:.0%}" for a, w in list(_bt_lw.items())[:3]
                    ) if _bt_lw else ""
                    with col:
                        st.markdown(
                            f'<div class="ti-strip" style="background:#050a05;'
                            f'border-top:1px solid #1a2a1a">'
                            f'<span class="ti-strip-tag" style="color:#27ae60">Backtest</span>'
                            f'<span class="ti-strip-val" style="color:{_wr_c}">Win&nbsp;{_bt_wr:.0f}%</span>'
                            f'<span class="ti-strip-val" style="color:{_ar_c}">Avg&nbsp;{_bt_ar:+.1f}%</span>'
                            f'<span class="ti-strip-val" style="color:{_sh_c}">Sharpe&nbsp;{_bt_sh:.2f}</span>'
                            f'<span class="ti-strip-val" style="color:#c0392b">MaxDD&nbsp;{_bt_dd:.1f}%</span>'
                            f'<span class="ti-strip-dim">{_bt_n} signals · {_parse_holding_days(trade)}d · {_wt_label}'
                            + (f' · {_wt_str}' if _wt_str else '')
                            + '</span></div>',
                            unsafe_allow_html=True,
                        )
        except Exception:
            pass

        # ── QC flags ─────────────────────────────────────────────────────
        if qc["flags"]:
            with col:
                st.markdown(
                    '<div style="display:flex;gap:4px;flex-wrap:wrap;padding:4px .9rem">'
                    + "".join(
                        f'<span style="background:#1a0e00;color:#e67e22;'
                        f'font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                        f'padding:2px 7px;border:1px solid #3a2000">⚠ {f}</span>'
                        for f in qc["flags"]
                    )
                    + '</div>',
                    unsafe_allow_html=True,
                )

        # ── Why this trade passed filters ─────────────────────────────────
        try:
            _pass_reasons: list[str] = []
            if trade.get("generated"):
                _pass_reasons.append("Live GEO signal")
            if trade.get("conflict_id"):
                _cid = trade["conflict_id"].replace("_", " ").title()
                _pass_reasons.append(f"Conflict: {_cid}")
            _pass_reasons.append(f"Regime {_REGIME_NAMES.get(current, current)}")
            if confidence >= 0.70:
                _pass_reasons.append(f"Conf {confidence * 100:.0f}%")
            _cat = trade.get("category", "")
            if _cat and _cat != "all":
                _pass_reasons.append(_cat)
            if asset_exposure:
                _trade_assets = trade.get("assets", [])
                _sas_vals = [asset_exposure[a]["sas"] for a in _trade_assets if a in asset_exposure]
                if _sas_vals:
                    _avg_sas = sum(_sas_vals) / len(_sas_vals)
                    if _avg_sas >= 60:
                        _pass_reasons.append(f"SAS {_avg_sas:.0f} — high exposure")
                    elif _avg_sas >= 35:
                        _pass_reasons.append(f"SAS {_avg_sas:.0f}")
                _hedge_scores = [asset_exposure[a]["hedge_score"] for a in _trade_assets if a in asset_exposure]
                if any(h >= 40 for h in _hedge_scores):
                    _pass_reasons.append("Hedge signal active")
                _directions = [asset_exposure[a]["direction"] for a in _trade_assets if a in asset_exposure]
                if "safe_haven" in _directions and any(d.lower() == "long" for d in trade.get("direction", [])):
                    _pass_reasons.append("Safe-haven demand")
            if _pass_reasons:
                with col:
                    st.markdown(
                        '<div class="ti-why">'
                        + "".join(
                            f'<span class="ti-why-chip">✓ {r}</span>'
                            for r in _pass_reasons
                        )
                        + '</div>',
                        unsafe_allow_html=True,
                    )
        except Exception:
            pass

        # ── Asset exposure strip ──────────────────────────────────────────
        if asset_exposure:
            try:
                _exp_items = []
                for _a, _d in zip(trade.get("assets", []), trade.get("direction", [])):
                    _ed = asset_exposure.get(_a)
                    if not _ed:
                        continue
                    _sas      = _ed["sas"]
                    _dir      = _ed["direction"]
                    _top_c    = _ed.get("top_conflict") or ""
                    _top_beta = _ed["beta"].get(_top_c, 0.0) if _top_c else 0.0
                    _dir_icon = "↑" if _dir == "long_geo_risk" else "↓" if _dir == "safe_haven" else "→"
                    _sas_col  = "#e67e22" if _sas >= 60 else "#CFB991" if _sas >= 30 else "#555960"
                    _beta_str = f"β {_top_beta:.2f}" if _top_c else ""
                    _exp_items.append(
                        f'<div class="ti-exp-cell">'
                        f'<div class="ti-exp-name">{_a[:18]}</div>'
                        f'<div style="display:flex;gap:6px;align-items:center">'
                        f'<span class="ti-exp-sas" style="color:{_sas_col}">SAS {_sas:.0f}</span>'
                        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                        f'color:#8890a1">{_dir_icon}</span>'
                        + (f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                           f'color:#555960">{_beta_str}</span>' if _beta_str else "")
                        + '</div></div>'
                    )
                if _exp_items:
                    with col:
                        st.markdown(
                            '<div class="ti-exp">' + "".join(_exp_items) + '</div>',
                            unsafe_allow_html=True,
                        )
            except Exception:
                pass

        # ── Master Investor Lens ──────────────────────────────────────────
        _render_investor_lens_strip(col, trade)

        # ── Payoff table expander ─────────────────────────────────────────
        with col:
            with st.expander(f"Scenario Payoff — {trade['name'][:40]}", expanded=False):
                try:
                    from src.analysis.profit_projection import project_trade
                    proj    = project_trade(trade)
                    p_table = proj["payoff_table"]

                    pm1, pm2, pm3, pm4 = st.columns(4)
                    pm1.metric("Exp. P&L",       f"{proj['expected_pnl']:+.1f}%")
                    pm2.metric("Worst Case",      f"{proj['worst_case_pnl']:+.1f}%")
                    pm3.metric("Breakeven Prob",  f"{proj['breakeven_prob'] * 100:.0f}%")
                    pm4.metric("Sharpe Proxy",    f"{proj['sharpe_proxy']:.2f}")

                    sc_labels = [r["label"] for r in p_table]
                    sc_pnls   = [r["expected_pnl"] for r in p_table]
                    sc_colors = ["#27ae60" if pnl >= 0 else "#c0392b" for pnl in sc_pnls]
                    fig_pf = go.Figure(go.Bar(
                        x=sc_labels, y=sc_pnls,
                        marker_color=sc_colors,
                        text=[f"{v:+.1f}%" for v in sc_pnls],
                        textposition="outside",
                    ))
                    fig_pf.update_layout(
                        template="plotly_dark", height=220,
                        title=dict(text="Expected P&L by Scenario (%)", font=dict(size=11)),
                        margin=dict(l=40, r=20, t=36, b=40),
                        yaxis=dict(title="P&L %", zeroline=True, zerolinecolor="#333", zerolinewidth=1),
                        showlegend=False,
                        plot_bgcolor="#0d0d0d", paper_bgcolor="#0d0d0d",
                    )
                    _chart(fig_pf)

                    pt_df = pd.DataFrame([{
                        "Scenario": r["label"],
                        "Prob":     f"{r['prob'] * 100:.0f}%",
                        "Exp. P&L": f"{r['expected_pnl']:+.1f}%",
                        "Vol":      f"{r['vol']:.1f}%",
                        "Wtd P&L":  f"{r['prob_weighted_pnl']:+.2f}%",
                        "Active":   "★" if r["is_current"] else "",
                    } for r in p_table])
                    st.dataframe(pt_df, width="stretch", hide_index=True)

                except Exception as exc:
                    st.caption(f"Payoff projection unavailable: {exc}")

        # ── Agent debate thread ───────────────────────────────────────────
        _is_geo_trade = trade.get("generated", False)
        _debate_open  = _is_geo_trade
        _debate_label = (
            f"⚡ Agent Debate — {trade['name'][:40]}"
            if _is_geo_trade
            else f"Agent Debate — {trade['name'][:40]}"
        )
        with col:
            with st.expander(_debate_label, expanded=_debate_open):
                try:
                    from src.ui.agent_panel import render_deliberation_panel
                    from src.analysis.agent_dialogue import challenge_trade, get_subject_threads
                    _trade_subject_id = trade.get("name", f"trade_{trade_idx}")
                    _stored_key = f"_debate_tid_{trade_idx}"
                    msgs = get_subject_threads(_trade_subject_id)

                    if not msgs:
                        if _is_geo_trade:
                            try:
                                _new_tid = challenge_trade(
                                    trade_id=_trade_subject_id,
                                    trade_title=trade.get("name", "Unknown Trade"),
                                    confidence=float(trade.get("confidence", 0.60)),
                                    qc_flags=list(trade.get("qc_flags", [])),
                                )
                                st.session_state[_stored_key] = _new_tid
                                msgs = get_subject_threads(_trade_subject_id)
                            except Exception:
                                pass
                        if not msgs:
                            if st.button("Run Agent Debate", key=f"debate_{trade_idx}"):
                                try:
                                    _new_tid = challenge_trade(
                                        trade_id=_trade_subject_id,
                                        trade_title=trade.get("name", "Unknown Trade"),
                                        confidence=float(trade.get("confidence", 0.60)),
                                        qc_flags=list(trade.get("qc_flags", [])),
                                    )
                                    st.session_state[_stored_key] = _new_tid
                                    st.rerun()
                                except Exception as exc:
                                    st.caption(f"Debate unavailable: {exc}")

                    if msgs:
                        _render_tid = st.session_state.get(_stored_key) or msgs[0]["thread_id"]
                        render_deliberation_panel(
                            thread_id=_render_tid,
                            subject_id=_trade_subject_id,
                            title="Agent Deliberation",
                            max_msgs=8,
                            show_consensus=True,
                        )
                except Exception as exc:
                    st.caption(f"Debate panel unavailable: {exc}")

        # ── Mini correlation chart ────────────────────────────────────────
        if len(trade["assets"]) >= 2:
            a1, a2 = trade["assets"][0], trade["assets"][1]
            if a1 in all_r_concat.columns and a2 in all_r_concat.columns:
                rc = rolling_correlation(all_r_concat[a1], all_r_concat[a2], 60)
                r_hex = int(cat_col[1:3], 16)
                g_hex = int(cat_col[3:5], 16)
                b_hex = int(cat_col[5:7], 16)
                fig_mini = go.Figure()
                fig_mini.add_trace(go.Scatter(
                    x=rc.index, y=rc.values,
                    name=f"{a1}/{a2}",
                    line=dict(color=cat_col, width=1.4),
                    fill="tozeroy",
                    fillcolor=f"rgba({r_hex},{g_hex},{b_hex},0.12)",
                ))
                fig_mini.add_hline(y=0, line=dict(color="#444", width=1, dash="dot"))
                for ev in GEOPOLITICAL_EVENTS:
                    fig_mini.add_vrect(
                        x0=str(ev["start"]), x1=str(ev["end"]),
                        fillcolor=ev["color"], opacity=0.04, layer="below", line_width=0,
                    )
                fig_mini.update_layout(
                    template="plotly_dark", height=180,
                    title=dict(text=f"60d Corr: {a1} / {a2}", font=dict(size=10)),
                    showlegend=False,
                    margin=dict(l=36, r=12, t=28, b=24),
                    xaxis=dict(rangeslider=dict(visible=False)),
                    plot_bgcolor="#0d0d0d", paper_bgcolor="#0d0d0d",
                )
                with col:
                    _chart(fig_mini)
                    _insight_note(
                        "Rolling 60-day correlation between the commodity driver and equity target. "
                        "Rising correlation = thesis strengthening. Declining = causal link breaking down."
                    )


def page_trade_ideas(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(_TI_STYLE, unsafe_allow_html=True)
    _page_header("Structured Trade Ideas",
                 "Step 6 of 7 · AI Conclusions · Regime-driven · Conflict-linked · Exposure-ranked · QC-graded")
    _page_intro(
        "Spillover analysis is most useful when it connects to positioning hypotheses. "
        "<strong>Each structure here is a research-oriented translation of a spillover or regime signal into an illustrative trade idea.</strong> "
        "Conflict-driven candidates are generated from current CIS/TPS scores and exposure data. "
        "Static library ideas fire when the current regime matches their structural trigger. "
        "All ideas are QC-graded (A–D), scenario-payoff-projected, and debatable via agent threads."
    )
    st.markdown(
        '<div style="display:flex;gap:1rem;align-items:center;margin-bottom:.6rem;flex-wrap:wrap">'
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:.58rem;font-weight:700;'
        'letter-spacing:.12em;text-transform:uppercase;color:#8890a1">'
        'Static Library Last Reviewed</span>'
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:.58rem;font-weight:700;'
        'color:#CFB991">June 2025</span>'
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:.58rem;'
        'color:rgba(255,255,255,.2)">|</span>'
        '<span style="font-family:\'DM Sans\',sans-serif;font-size:.72rem;color:#8890a1">'
        'Structural triggers and entry/exit levels reflect research-period market conditions. '
        'Conflict-driven ideas are generated live from current session data.</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Geopolitical context & filter gate (shown BEFORE any trades) ──────────
    try:
        from src.analysis.conflict_model import score_all_conflicts, aggregate_portfolio_scores
        from src.analysis.scenario_state import get_scenario, get_scenario_id
        _ti_cr   = score_all_conflicts()
        _ti_agg  = aggregate_portfolio_scores(_ti_cr)
        _ti_sc   = get_scenario()
        _ti_sid  = get_scenario_id()
        _ti_cis  = _ti_agg.get("portfolio_cis", _ti_agg.get("cis", 50.0))
        _ti_tps  = _ti_agg.get("portfolio_tps", _ti_agg.get("tps", 50.0))
        _ti_top  = (_ti_agg.get("top_conflict", "-") or "-").replace("_", " ").title()
        _ti_mult = _ti_sc.get("geo_mult", 1.0)
        _ti_sc_color = _ti_sc.get("color", "#CFB991")

        if _ti_cis >= 70:    _ti_risk_color, _ti_risk_lbl = "#c0392b", "HIGH CONFLICT"
        elif _ti_cis >= 50:  _ti_risk_color, _ti_risk_lbl = "#e67e22", "ELEVATED"
        else:                _ti_risk_color, _ti_risk_lbl = "#CFB991", "MODERATE"

        st.markdown(
            f'<div class="ti-geo-bar" style="border-left:3px solid {_ti_risk_color}">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
            f'font-weight:700;color:{_ti_risk_color};letter-spacing:.14em">'
            f'■ GEO CONTEXT</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
            f'color:#e67e22">CIS&nbsp;<b>{_ti_cis:.0f}</b></span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
            f'color:#CFB991">TPS&nbsp;<b>{_ti_tps:.0f}</b></span>'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:10px;color:#8E9AAA">'
            f'Lead:&nbsp;<b style="color:{_ti_risk_color}">{_ti_top}</b></span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
            f'color:{_ti_sc_color};font-weight:700">'
            f'{_ti_sc.get("label", "Base").upper()}&nbsp;×{_ti_mult:.2f}</span>'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:10px;'
            f'color:#555960;margin-left:auto">'
            f'Conflict-driven ideas reflect live CIS/TPS. Set filters before reviewing.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception as _geo_err:
        st.caption(f"Geo context unavailable - conflict model load failed: {_geo_err}")

    with st.spinner("Loading data…"):
        eq_r, cmd_r = load_returns(start, end)

    if eq_r.empty or cmd_r.empty:
        st.error("Market data unavailable.")
        return

    # ── Current regime ─────────────────────────────────────────────────────
    avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
    regimes  = detect_correlation_regime(avg_corr)
    # Read attrs BEFORE any pandas operation (reindex/ffill lose attrs)
    _regime_insuf = bool(regimes.attrs.get("insufficient_data", False))
    _regime_n_obs = int(regimes.attrs.get("n_obs", 0))
    current  = int(regimes.iloc[-1]) if not regimes.empty else 1
    r_name   = _REGIME_NAMES[current]
    r_color  = _REGIME_COLORS[current]

    # ── Conflict-driven candidates ─────────────────────────────────────────
    generated: list[dict] = []
    conflict_betas: dict  = {}
    asset_exposure: dict  = {}
    try:
        from src.analysis.trade_generator import generate_conflict_trades, merge_with_library
        from src.analysis.exposure import score_all_assets
        from src.analysis.conflict_model import score_all_conflicts
        _cr  = score_all_conflicts()
        _aa  = score_all_assets()
        generated = generate_conflict_trades(regime=current, conflict_results=_cr, all_assets=_aa)
        # Retain full exposure data for ranking and card display
        asset_exposure = dict(_aa)
        conflict_betas = {
            name: {"beta": d["beta"]}
            for name, d in _aa.items()
        }
    except Exception:
        pass  # silently degrade if exposure/conflict data unavailable

    # ── Merge with static library ──────────────────────────────────────────
    try:
        from src.analysis.trade_generator import merge_with_library as _merge
        all_candidates = _merge(generated, _TRADE_LIBRARY, current)
    except Exception:
        all_candidates = list(_TRADE_LIBRARY)
        for t in all_candidates:
            t.setdefault("confidence", 0.60)
            t.setdefault("qc_flags",  [])

    # ── Apply filters ──────────────────────────────────────────────────────
    try:
        from src.analysis.trade_filter import apply_filters, build_filter_ui
        from src.analysis.scenario_state import get_scenario_id
        _filters      = build_filter_ui(key_prefix="ti")
        _scenario_id  = get_scenario_id()
        active_trades = apply_filters(
            all_candidates,
            filters=_filters,
            current_regime=current,
            current_scenario_id=_scenario_id,
            conflict_betas=conflict_betas,
            asset_exposure=asset_exposure,
        )
    except Exception:
        # Fallback: basic regime filter
        active_trades = [t for t in all_candidates if current in t.get("regime", [current])]

    # ── Re-rank by exposure × confidence composite ─────────────────────────
    if asset_exposure:
        def _exposure_rank_key(t: dict) -> float:
            conf = float(t.get("confidence", 0.5))
            sas_vals = [asset_exposure[a]["sas"] for a in t.get("assets", []) if a in asset_exposure]
            avg_sas = (sum(sas_vals) / len(sas_vals)) if sas_vals else 0.0
            return conf * (1 + avg_sas / 200)
        active_trades.sort(key=_exposure_rank_key, reverse=True)

    # ── KPI strip ──────────────────────────────────────────────────────────
    _n_geo  = sum(1 for t in active_trades if t.get("generated"))
    _n_stat = len(active_trades) - _n_geo
    _n_bull = sum(1 for t in active_trades if t.get("direction", [""])[0] == "Long")
    _n_bear = sum(1 for t in active_trades if t.get("direction", [""])[0] == "Short")

    def _ti_kpi(col, label, value, value_color="#e8e8e8", border_color="#1e1e1e"):
        col.markdown(
            f'<div class="ti-kpi" style="border-color:{border_color}">'
            f'<div class="ti-kpi-lbl">{label}</div>'
            f'<div class="ti-kpi-val" style="color:{value_color}">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    _insuf_note = (
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
        f'color:#e67e22;margin-top:3px">INSUF DATA ({_regime_n_obs}&lt;60 obs)</div>'
        if _regime_insuf else ""
    )
    k1.markdown(
        f'<div class="ti-kpi" style="border-color:{r_color}">'
        f'<div class="ti-kpi-lbl">Regime</div>'
        f'<div class="ti-kpi-val" style="color:{r_color}">{r_name}</div>'
        + _insuf_note
        + '</div>',
        unsafe_allow_html=True,
    )
    _ti_kpi(k2, "Avg Corr (60d)",  f"{avg_corr.iloc[-1]:.3f}")
    _ti_kpi(k3, "Active Ideas",    str(len(active_trades)), "#CFB991")
    _ti_kpi(k4, "Conflict-Driven", str(_n_geo),  "#e67e22")
    _ti_kpi(k5, "Long-First",      str(_n_bull), "#27ae60")
    _ti_kpi(k6, "Short-First",     str(_n_bear), "#c0392b")

    if not active_trades:
        st.info(f"No trade ideas pass current filters for {r_name} regime.")
        st.stop()

    st.markdown(
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:11px;'
        f'color:#8890a1;margin-bottom:.6rem">'
        f'<b style="color:#e8e8e8">{len(active_trades)}</b> ideas — '
        f'<b style="color:#e67e22">{_n_geo}</b> conflict-driven · '
        f'<b style="color:#8890a1">{_n_stat}</b> static · '
        f'sorted by exposure × confidence · regime: '
        f'<b style="color:{r_color}">{r_name}</b></p>',
        unsafe_allow_html=True,
    )

    # Expand backtest universe to include Fixed Income + FX so new macro trades are backtested
    _extra_frames: list[pd.DataFrame] = []
    try:
        from src.data.loader import load_fixed_income_returns, load_fx_returns
        _fi_r  = load_fixed_income_returns(start, end)
        _fx_r  = load_fx_returns(start, end)
        if not _fi_r.empty:
            _extra_frames.append(_fi_r)
        if not _fx_r.empty:
            _extra_frames.append(_fx_r)
    except Exception:
        pass
    all_r_concat = pd.concat([eq_r, cmd_r] + _extra_frames, axis=1)

    _thread(
        "Conflict-driven ideas are generated live from current CIS/TPS/SAS scores. "
        "Static ideas fire when the regime matches their structural trigger. "
        "Both sets are QC-graded and filtered by the active filter settings above."
    )

    # ── 2-column card grid ──────────────────────────────────────────────────
    for row_start in range(0, len(active_trades), 2):
        pair = active_trades[row_start:row_start + 2]
        card_cols = st.columns(len(pair), gap="medium")
        for col, (trade, idx) in zip(card_cols, [(t, row_start + i) for i, t in enumerate(pair)]):
            _render_trade_card(col, trade, all_r_concat, current, idx, asset_exposure or None, regimes=regimes)

    # ── Download report ─────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:10px;font-weight:700;'
        'letter-spacing:.10em;text-transform:uppercase;color:#CFB991;margin-bottom:.5rem">'
        'Institution Report Download</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:11px;'
        'color:#8890a1;line-height:1.7;margin-bottom:.8rem">'
        'Generate a professionally formatted A4 research report covering the current regime, '
        'all illustrative trade ideas, geopolitical context, and methodology — suitable for '
        'academic submission or instructor review.</p>',
        unsafe_allow_html=True,
    )

    if st.button("Generate PDF Report", key="gen_report", type="primary"):
        try:
            from src.reports.report_generator import generate_report
            with st.spinner("Building report - generating charts…"):
                stress = composite_stress_index(eq_r, cmd_r, avg_corr=avg_corr)
                pdf_bytes = generate_report(
                    start=start,
                    end=end,
                    avg_corr_series=avg_corr,
                    current_regime=current,
                    regimes=regimes,
                    active_trades=active_trades,
                    all_trades=_TRADE_LIBRARY,
                    eq_r=eq_r,
                    cmd_r=cmd_r,
                    stress_series=stress,
                    geopolitical_events=GEOPOLITICAL_EVENTS,
                )
            filename = (
                f"purdue_spillover_report_"
                f"{start.replace('-','')}_to_{end.replace('-','')}.pdf"
            )
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                key="download_report",
            )
        except ImportError:
            st.error(
                "reportlab is required for PDF generation. "
                "Run: `pip install reportlab>=4.2.0`"
            )
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")

    _section_note(
        "Trade ideas are generated from historical cross-asset patterns and regime signals. "
        "All ideas are illustrative and must be validated against current market structure, "
        "liquidity, and position sizing constraints before implementation."
    )

    _page_conclusion(
        "Framework",
        "The regime-based trade generation framework matches historical spillover patterns "
        "to current correlation regimes. Crisis regimes activate hedging and divergence plays; "
        "normal regimes favour growth-correlated long positioning. "
        "Use Granger and transfer entropy results from the Spillover page to validate lead-lag direction."
    )

    # ── AI Trade Structurer + Pending Review Panel ─────────────────────────
    try:
        from src.agents.trade_structurer import run as _ts_run
        from src.ui.agent_panel import render_agent_output_block, render_pending_review
        from src.analysis.agent_state import is_enabled, get_agent

        if is_enabled("trade_structurer"):
            _anthropic_key = _openai_key = ""
            try:
                _keys = st.secrets.get("keys", {})
                _anthropic_key = _keys.get("anthropic_api_key", "") or ""
                _openai_key    = _keys.get("openai_api_key",    "") or ""
            except Exception:
                pass
            _provider = "anthropic" if _anthropic_key else ("openai" if _openai_key else None)
            _api_key  = _anthropic_key or _openai_key

            # Build context from what's been computed on this page
            _ts_ctx: dict = {
                "regime_name":   r_name,
                "regime_level":  current,
                "avg_corr":      float(avg_corr.iloc[-1]) if not avg_corr.empty else 0.0,
            }
            try:
                from src.analysis.risk_score import compute_risk_score
                from src.data.loader import load_returns as _lr3
                _eq_r3, _cmd_r3 = _lr3(start, end)
                from src.analysis.correlations import average_cross_corr_series as _acs
                _avg_c3 = _acs(_eq_r3, _cmd_r3, window=60)
                _rs = compute_risk_score(_avg_c3, _cmd_r3, _eq_r3)
                _ts_ctx["risk_score"] = float(_rs.get("score", 0))
                if len(_cmd_r3) >= 5:
                    _w5c = _cmd_r3.iloc[-5:].sum()
                    _ts_ctx["top_commodity"]     = str(_w5c.idxmax())
                    _ts_ctx["top_commodity_ret"] = float(_w5c.max()) * 100
                if len(_eq_r3) >= 5:
                    _w5e = _eq_r3.iloc[-5:].sum()
                    _ts_ctx["worst_equity"]      = str(_w5e.idxmin())
                    _ts_ctx["worst_equity_ret"]  = float(_w5e.min()) * 100
            except Exception:
                pass

            # Live S&P 500 stock prices — sector-filtered by current regime/scenario
            try:
                _scenario_id: str | None = None
                try:
                    from src.analysis.scenario_state import get_scenario_id as _gsi
                    _scenario_id = _gsi()
                except Exception:
                    pass
                _sel_sectors = _select_sectors_for_signal(current, _scenario_id)
                _stock_px    = _fetch_stock_prices(tuple(_sel_sectors))
                if _stock_px:
                    _ts_ctx["stock_prices_text"] = _format_stock_context(_stock_px, _sel_sectors)
            except Exception:
                pass

            # Peer context from orchestrator (structured, freshness-aware)
            _peer_signals = {}
            try:
                from src.analysis.agent_orchestrator import get_orchestrator as _get_orch_ts
                _peer_signals = _get_orch_ts(_provider, _api_key).get_peer_context("trade_structurer")
            except Exception:
                # Fallback: read directly from agent state
                for _pid in ("macro_strategist", "geopolitical_analyst",
                             "stress_engineer", "commodities_specialist"):
                    _pa = get_agent(_pid)
                    if _pa.get("last_output"):
                        _raw = _pa["last_output"]
                        _lines = [l for l in _raw.split("\n") if "confidence:" not in l.lower()]
                        _peer_signals[_pid] = " ".join(_lines).strip()[:250]
            if _peer_signals:
                _ts_ctx["peer_signals"] = _peer_signals

            with st.spinner("AI Trade Structurer generating idea…"):
                _ts_result = _ts_run(_ts_ctx, _provider, _api_key)

            if _ts_result.get("narrative"):
                st.markdown("---")
                render_agent_output_block("trade_structurer", _ts_result)

        # Always show the review queue on this page
        st.markdown("---")
        render_pending_review()

    except Exception as _ts_err:
        st.caption(f"AI Trade Structurer unavailable: {_ts_err}")

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
                "n_obs": len(eq_r.dropna(how="all")), "date_range": f"{start} to {end}",
                "model": "Conflict-driven + regime-filtered illustrative trade structures", "regime": r_name,
                "assumption_count": 5, "trade_has_stop": True,
                "notes": [
                    f"Current regime index: {current}/3 - {len(active_trades)} ideas active after filters",
                    f"{_n_geo} conflict-generated candidates, {_n_stat} static library ideas",
                    "Trade entry/exit levels are illustrative ranges, not live-calibrated prices",
                    "Correlation-based regime uses 60d rolling window - whipsaws in trending vol regimes",
                    "No walk-forward backtest - ideas are not validated against historical win rates",
                    "All ideas assume liquid markets - slippage and execution risk not modelled",
                ],
            }
            _cqo_run(_cqo_ctx, _provider, _api_key, page="Structured Trade Ideas")
    except Exception:
        pass

    _page_footer()

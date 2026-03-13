# Equity-Commodities Spillover Monitor

A multi-page analytical dashboard for tracking cross-asset spillover dynamics between global equity markets and commodity futures — with a focus on geopolitical risk, correlation regimes, and portfolio stress testing.

---

## Overview

The Spillover Monitor brings together 15 global equity indices and 17 commodity futures into a single interactive workspace. It tracks how stress transmits across asset classes — not just what is happening, but *why*, *where it came from*, and *what comes next*.

Built for analysts, portfolio managers, and researchers who need more than price charts: directional spillover networks, Markov-regime forecasts, conflict impact scoring, and AI-assisted narrative — all in one terminal.

---

## Coverage

### Equity Markets — 15 Indices, 5 Regions
| Region | Indices |
|--------|---------|
| **USA** | S&P 500, Nasdaq 100, DJIA, Russell 2000 |
| **Europe** | Eurostoxx 50, DAX, CAC 40, FTSE 100 |
| **Japan** | Nikkei 225, TOPIX |
| **China** | Hang Seng, Shanghai Composite, CSI 300 |
| **India** | Sensex, Nifty 50 |

### Commodities — 17 Futures, 4 Groups
| Group | Contracts |
|-------|-----------|
| **Energy** | WTI Crude, Brent Crude, Natural Gas, Gasoline (RBOB), Heating Oil |
| **Precious Metals** | Gold, Silver, Platinum |
| **Industrial Metals** | Copper, Aluminum, Nickel |
| **Agriculture** | Wheat, Corn, Soybeans, Sugar #11, Coffee, Cotton |

### Geopolitical Events — 12 Tracked Shocks (2008–2025)
| ID | Event | Period |
|----|-------|--------|
| GFC | Global Financial Crisis | Sep 2008 – Mar 2009 |
| Arab Spring | Arab Spring & Libya Oil Shock | Jan – Oct 2011 |
| Trade War | US-China Trade War | Mar 2018 – Jan 2020 |
| Aramco | Abqaiq-Khurais Drone Attack | Sep – Oct 2019 |
| COVID-19 | COVID-19 Pandemic | Feb 2020 – Nov 2021 |
| WTI Neg | WTI Goes Negative | Apr – May 2020 |
| Ukraine | Russia-Ukraine War | Feb 2022 – present |
| Nickel | LME Nickel Short Squeeze | Mar 7–25, 2022 |
| Fed Hike | Fed Rate Hiking Cycle | Mar 2022 – Jul 2023 |
| SVB | SVB / Banking Crisis | Mar – May 2023 |
| Hamas | Israel-Hamas War & Iran Escalation | Oct 2023 – present |
| Hormuz | Iran Military Conflict & Strait of Hormuz Crisis | Jun 2025 – present |

---

## Key Attributes

- **Real-time cross-asset correlation regimes** — four-state classification (Decorrelated → Normal → Elevated → Crisis) updated on each load
- **Markov chain regime forecasting** — forward-looking transition probabilities, mean days to next Crisis, and steady-state distribution
- **Early Warning System** — five-component composite score (0–100) flagging conditions that historically precede regime transitions
- **Three-conflict war risk scoring** — country-level equity exposure mapped for Ukraine, Israel-Hamas, and Iran/Hormuz simultaneously
- **Directional spillover network** — Granger causality, transfer entropy, and Diebold-Yilmaz index identify who leads and who follows
- **Portfolio stress tester** — combine any mix of indices, commodities, and S&P 500 stocks; test against every tracked geopolitical event
- **Regime-triggered trade ideas** — curated cross-asset trade cards with entry logic, rationale, and risk anchored to current regime
- **AI analyst** — GPT-4o chatbot with live dashboard context injected into the system prompt for institutional-quality narrative
- **Adjustable date range** — all analytics recompute for any start/end window from the sidebar
- **No investment advice** — research and educational tool only

---

## Use Cases

### 1. Geopolitical Risk Assessment
Track how an active conflict (Ukraine, Gaza, Hormuz) is transmitting into equity and commodity prices. The War Impact Map scores 195+ countries by market exposure, updated for all three active theaters simultaneously.

### 2. Correlation Regime Monitoring
Identify whether markets are in a stress or calm phase. The four-regime model signals when correlations are "elevated" or in full "crisis" — the point at which diversification breaks down and hedges matter most.

### 3. Crisis Early Warning
The Early Warning System synthesizes five leading indicators into a composite score before a regime transition occurs. Use it as a pre-emptive signal, not a reactive label.

### 4. Portfolio Stress Testing
Build a custom multi-asset portfolio (indices + commodities + any S&P 500 stock), and see exactly how it would have performed during every major shock since 2008 — with pre/during/post return breakdowns, max drawdown, and Sharpe ratio.

### 5. Spillover & Contagion Analysis
Understand transmission direction: does oil drive equities, or do equity selloffs lead commodity liquidation? Granger causality grids and transfer entropy matrices answer this quantitatively.

### 6. Trade Ideation
Use the Regime-triggered Trade Ideas page to find cross-asset setups anchored to the current correlation regime — each card includes rationale, directional bias, and risk considerations.

### 7. Historical Event Forensics
The Geopolitical Triggers page isolates every tracked event and shows normalized price performance, volatility shifts, and correlation regime changes in a single clean view — useful for pattern-matching to current conditions.

---

## Page-by-Page Manual

### Overview
**Starting point for every session.** Load this page first.

- **KPI strip** — five headline metrics: current regime, average cross-asset correlation (with 22-day delta), best and worst equity region, best and worst commodity
- **Risk Score** — composite 0–100 gauge combining regime state, correlation level, recent volatility, and geopolitical event proximity
- **Early Warning System** — five signal cards (Correlation Velocity, Vol Acceleration, Regime Duration Pressure, Equity Vol Trend, Markov Crisis Probability) plus a composite banner. Below this: a Historical Analogues table showing the five most similar past market environments and how they resolved
- **AI Narrative** — one-paragraph contextual summary generated from live market data

**How to use:** Check the regime and EWS score at the start of each session. If EWS composite exceeds 60/100, investigate further using Correlation Analysis and Spillover Analytics.

---

### War Impact Map
**Global choropleth showing equity-market exposure to active geopolitical conflicts.**

- **Filter bar** — switch between Combined (worst of all conflicts), Ukraine War only, Israel-Hamas only, or Iran/Hormuz only
- **Map view** — flat world map with countries colored green (low risk) to red (high risk). Home-turf countries (Russia, Ukraine, Israel, Iran) marked with a distinct outline
- **Top 25 exposed markets** — ranked table of highest-risk countries with per-conflict breakdown scores
- **Tracked equity performance** — returns for each high-risk market since conflict onset, with per-conflict score columns
- **Conflict summaries** — background on each active conflict's commodity channels and market transmission mechanisms

**How to use:** Select a specific conflict to isolate its exposure. Cross-reference the Top 25 table with your equity positions to identify geographic concentration.

---

### Geopolitical Triggers
**Event-window forensics for every tracked shock since 2008.**

- **Event selector** — choose one or multiple events from the full 12-event library
- **Normalized price performance** — all assets rebased to 100 at event start; pre/during/post windows overlaid
- **Pre vs. post volatility shift** — annualized realized volatility before and after each event, with a % change column
- **Correlation regime change** — regime state at event onset vs. peak stress vs. 90-day recovery

**How to use:** Select the event most analogous to current conditions (use the Historical Analogues section on Overview for guidance). Compare its commodity/equity performance patterns against current positioning.

---

### Correlation Analysis
**Four-tab deep-dive into the cross-asset correlation structure.**

#### Tab 1: Rolling Correlation
Pearson rolling correlations between selected equity indices and commodities. Adjust the window (30d–252d) to see short-term spikes vs. structural trends. Use the pair-picker to focus on specific cross-asset relationships.

#### Tab 2: DCC-GARCH
Dynamic Conditional Correlations with GARCH(1,1) volatility filtering. Removes the mechanical effect of changing volatility on correlation estimates — what remains is the "true" co-movement signal. Particularly useful for identifying whether a recent correlation spike is volatility-driven or structural.

#### Tab 3: Regime Detection
Historical timeline of the four correlation regimes with color bands. Summary statistics: average correlation per regime, typical duration, frequency. Use the date slider to zoom into specific crisis episodes and compare regime transitions.

#### Tab 4: Regime Forecast
Markov chain output anchored to the current regime:
- **Transition matrix heatmap** — 4×4 probability matrix; each cell shows the probability of moving from regime (row) to regime (column)
- **Steady-state distribution** — where the system spends its time in the long run
- **Mean days to Crisis** — expected number of trading days to reach the Crisis regime from each starting state
- **Run statistics** — historical average, median, and maximum duration in each regime

**How to use:** Check the current-regime row of the transition matrix. If P(Crisis | current) > 25%, pair with the EWS composite on Overview to assess whether pre-emptive hedging is warranted.

---

### Spillover Analytics
**Directional spillover network: who leads, who follows.**

- **Granger causality grid** — pairwise test results at 5% significance; a checkmark means past values of asset A improve forecasts of asset B
- **Transfer entropy matrix** — nonlinear information flow between all asset pairs; heat-mapped by bit score
- **Net flow matrix** — difference between outbound and inbound transfer entropy; net transmitters vs. net receivers
- **Diebold-Yilmaz spillover index** — aggregate system-level connectedness score (0–100%); rising DY index = rising contagion risk
- **Network graph** — visual node-link diagram; node size = net spillover power; edge thickness = flow magnitude
- **Net transmitter bar chart** — ranked view of which assets export the most risk to the rest of the system

**How to use:** Identify the dominant transmitters (typically energy futures or S&P 500 during stress). If your portfolio is heavily loaded on net receivers, it is more vulnerable to spillover from those transmitter assets.

---

### Commodities to Watch
**Live commodity watchlist with volatility regime classification.**

- **Live snapshot** — current price, 1-day and 5-day return, annualized realized volatility, and regime label (Low Vol / Normal / Elevated / High Vol) for all 17 futures
- **Intraday view** — hourly price and volume data for a selected commodity; useful for same-day signal monitoring
- **Daily chart** — configurable rolling window; overlaid with volatility regime color bands
- **COT overlay** — Commitment of Traders positioning data where available (net speculative positioning vs. commercial hedgers)
- **Rolling correlation** — selected commodity vs. all equity regions; rolling 60-day Pearson

**How to use:** Sort by volatility regime to surface the most stressed commodities. Use the COT overlay to assess whether speculative positioning is extended — crowded longs in a High Vol regime typically mean higher reversal risk.

---

### Trade Ideas
**Regime-triggered cross-asset trade cards.**

Each card contains:
- **Trade title and direction** (long/short, asset pair)
- **Regime trigger** — which correlation regime activates this trade
- **Rationale** — macro logic behind the setup
- **Entry signal** — indicator or price-action condition to watch
- **Risk consideration** — what would invalidate the trade

Cards are filtered by the current live regime automatically. You can also manually select a regime to review trades for other scenarios.

**Examples of available trades:**
- Long Gold / Short Eurostoxx 50 (Crisis regime hedge)
- Long Natural Gas / Short Nikkei 225 (Energy supply shock)
- Long Copper / Long EM equities (Reflation regime)
- Long USD / Short commodity basket (DXY regime divergence)

**How to use:** When the current regime is Elevated or Crisis, filter to those cards. Cross-reference with the Spillover Analytics page to confirm that the transmitter/receiver direction aligns with the trade thesis.

---

### Portfolio Stress Test
**Build any portfolio, test it against every tracked shock.**

#### Step 1: Build your portfolio (Section A — Indices & Commodities)
- Choose a preset (60/40, Energy Macro, EM + Commodities, Global Diversified) or start custom
- Select indices and commodity futures from the multiselect
- Enter any ticker directly (e.g., AAPL, NVDA, JPM) for individual stocks
- Adjust weights using the number inputs (0–100%, increments of 1%)
- The allocation bar chart updates in real-time showing your portfolio breakdown

#### Step 2: Add individual stocks (Section B — S&P 500)
- Filter by sector to narrow the list
- Search any of the ~500 constituents by ticker or company name
- Edit weights in the interactive table (supports CSV upload: columns `ticker`, `weight`)
- Use the Equal Weight button for a quick equal-allocation starting point

#### Step 3: Review the weight summary
- The status bar shows total allocated weight and color-codes it (green = 100%, neutral = under, orange = over)
- Normalize to 100% with a single click (or let the engine normalize automatically at run time)
- The portfolio allocation chart shows proportional bar visualization

#### Step 4: Select events and run
- Choose which of the 12 geopolitical events to include
- Set the pre-event and post-event windows (10–90 days)
- Click **Run Stress Test**

#### Results
- **Summary table** — Pre / During / Post return (%), Max Drawdown (%), Sharpe Ratio for each event; green/red color coding
- **Key metrics strip** — worst event return, best event return, worst drawdown, average during-event return
- **Event Returns Heatmap** — pre/during/post returns for all events in a single diverging color grid (green = gain, red = loss)
- **Max Drawdown bar chart** — drawdown depth per event; red threshold line at -10%
- **Portfolio path chart** — normalized to 100 at event start, x-axis in days from event start (making all events directly comparable); vertical marker at day 0

**How to use:** Run the default preset first to establish a baseline. Then adjust weights toward commodities or cash equivalents and re-run to see how much drawdown protection you gain. Pay attention to which events consistently generate the worst results for your specific allocation — those are your tail-risk scenarios.

---

### AI Analyst
**GPT-4o chatbot with live dashboard context.**

The AI Analyst automatically loads the current market state into its system prompt before you type anything:
- Current correlation regime and risk score
- Recent equity and commodity returns (22-day window)
- Active geopolitical events
- EWS composite and component scores

Ask anything market-related in plain language:
- "What is driving the elevated regime right now?"
- "Which commodities look most vulnerable to the current geopolitical environment?"
- "Explain the Diebold-Yilmaz spillover score in simple terms."
- "What does a Markov transition probability of 38% to Crisis regime mean for my portfolio?"

The model responds in institutional prose — no emojis, no hedging clichés. Treat it as a senior analyst who has just read the full dashboard.

**Note:** An OpenAI API key is required. Responses are generated by GPT-4o and reflect the model's training data plus whatever live context is injected — not a licensed financial advisory service.

---

## Sidebar Controls

Every page respects two global controls in the sidebar:

| Control | Effect |
|---------|--------|
| **Start Date** | Sets the historical lookback for all data and analytics |
| **End Date** | Caps the analysis window (defaults to today) |
| **FRED API Key** (optional) | Enables additional macro overlays where available |

Short windows (< 2 years) may produce sparse correlation matrices and fewer historical analogues. For regime analysis, a minimum 5-year window is recommended. For the full 12-event coverage, use a start date of 2008 or earlier.

---

## Navigation

```
Overview
├── Analysis
│   ├── War Impact Map
│   ├── Geopolitical Triggers
│   ├── Correlation Analysis
│   ├── Spillover Analytics
│   └── Commodities to Watch
├── Strategy
│   ├── Trade Ideas
│   └── Portfolio Stress Test
└── Research
    ├── Performance Review
    └── AI Analyst
```

---

## Disclaimers

This dashboard is built for **educational and research purposes only**. Nothing on this platform constitutes investment advice. All analytics are based on historical data and statistical models — past relationships do not guarantee future behavior. Commodity and equity prices are sourced from public data providers (Yahoo Finance, FRED) and may have delays or gaps. The AI Analyst responses are generated by a third-party language model and should not be treated as financial guidance.

---

*Equity-Commodities Spillover Monitor — Purdue University | AI for Finance*

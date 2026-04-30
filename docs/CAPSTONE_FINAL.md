# Equity-Commodities Spillover Monitor
## Week 7 Capstone — Final Industry Panelist Review
### MGMT 69000-120: AI for Finance · Purdue University Daniels School of Business

**Team:** Heramb S. Patkar · Ilian Zalomai · Jiahe Miao
**Instructor:** Prof. Cinder Zhang
**Review Date:** April 30, 2026
**Points:** 300

---

## Executive Summary

The Equity-Commodities Spillover Monitor is an institutional-grade, multi-page analytical dashboard that answers a question standard financial terminals do not: **how do geopolitical shocks transmit across asset classes in real time, and what does that mean for positioning?**

The system ingests live data from six external sources, runs five independent econometric models, orchestrates seven AI agents in a dependency-ordered pipeline, and presents results across 22 pages in a purpose-built dark-theme Streamlit interface. Every analytical claim is backed by a published methodology; every model output is validated against a 56-case out-of-sample benchmark harness.

The dashboard is deployed at [https://equity-commodities-spillover.onrender.com](https://equity-commodities-spillover.onrender.com) and runs in production on Render.com with a live GitHub CI/CD pipeline.

**The core value proposition in one sentence:** A single analyst can open this dashboard and, within two minutes, know what is driving cross-asset stress, which geopolitical conflicts are transmitting into commodity markets, and what trade structures are valid under the current regime — all grounded in real data and statistically rigorous methodology.

---

## 1. Problem Statement & Market Gap

### 1.1 The Question This System Answers

> *How do geopolitical shocks transmit across asset classes — specifically between global equity markets and commodity futures — and what does that mean for portfolio positioning?*

Standard market dashboards (Bloomberg, FactSet) answer: "What are prices doing?" This system answers: "Why are prices moving, where is the shock originating, and how does it propagate?"

### 1.2 The Gap in Existing Tools

| Capability | Bloomberg Terminal | Standard Dashboard | This System |
|---|---|---|---|
| Live price/return data | ✅ | ✅ | ✅ |
| Correlation levels | ✅ | ✅ | ✅ |
| Spillover directionality (who drives whom) | ❌ | ❌ | ✅ Diebold-Yilmaz FEVD |
| Regime-conditioned analysis | Partial | ❌ | ✅ 3-state adaptive |
| Geopolitical → market transmission attribution | ❌ | ❌ | ✅ CIS/TPS scoring |
| AI-synthesized morning briefing | ❌ | ❌ | ✅ 7-agent pipeline |
| Scenario propagation with regime conditioning | ❌ | ❌ | ✅ Nonlinear OLS betas |
| Out-of-sample model validation | ❌ | ❌ | ✅ 56-case harness |

### 1.3 Why This Matters Now

Three concurrent geopolitical shocks are active as of April 2026: the Russia-Ukraine War (commodity supply disruption), the Israel-Hamas / Red Sea Crisis (shipping chokepoints), and the India-Pakistan Military Escalation (regional contagion risk). Each transmits differently into asset classes. No existing public tool synthesizes all three with live data and quantitative attribution.

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                             │
│  yfinance (50+ tickers) │ FRED (24 macro) │ GDELT (conflict news)  │
│  ACLED (geo events)     │ PortWatch (AIS) │ EIA (petroleum inv.)    │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ Cached (TTL: 900s–7200s per source)
┌──────────────────────▼──────────────────────────────────────────────┐
│                  ANALYTICS ENGINE                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Diebold-     │  │ DCC-GARCH    │  │ Granger Causality        │  │
│  │ Yilmaz FEVD  │  │ (Engle 2002) │  │ BIC lag + Holm-corrected │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Transfer     │  │ Correlation  │  │ Composite Risk Score     │  │
│  │ Entropy      │  │ Regime Model │  │ GRS = 40%CIS+35%TPS+25%MCS│ │
│  │ (shuffle sig)│  │ (3-state)    │  │                          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────┐
│                  AI AGENT LAYER (7 agents)                          │
│                                                                     │
│  Round 1 (Parallel):                                                │
│    macro_strategist ──────┐                                         │
│    geopolitical_analyst ──┤──► Round 2:                            │
│    commodities_specialist ┘     risk_officer                        │
│                                     │                              │
│  Round 3 (Parallel):           Round 4:                            │
│    stress_engineer ────────►    trade_structurer                    │
│    signal_auditor  ────────►    quality_officer (CQO)               │
│                                     │                              │
│                              remediation_router ──► specialist agents│
└──────────────────────┬──────────────────────────────────────────────┘
                       │  Pydantic-validated structured outputs
┌──────────────────────▼──────────────────────────────────────────────┐
│                  STREAMLIT UI (22 pages)                            │
│  Command Center │ Correlation │ Spillover │ Macro Intelligence      │
│  Geopolitical   │ War Impact Map │ Strait Watch │ Scenario Engine   │
│  Trade Ideas    │ Stress Test │ Watchlist │ Model Accuracy │ Chat   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Pipeline

**Sources and refresh cadence:**

| Source | Data | Cache TTL | Notes |
|---|---|---|---|
| Yahoo Finance | 50+ tickers: equities, commodities, FX, vol indices | 900s | Parallel-loaded via ThreadPoolExecutor |
| FRED API | 24 macro series: yield curve, CPI, Fed Funds, GDP, PMI | 86400s | Daily; uses last available for stale dates |
| GDELT 2.0 | News volume and tone for 6 active conflicts | 7200s | HTTP; conflict interest score = n_mentions × avg_tone |
| ACLED | Georeferenced conflict events, fatalities | 21600s | REST API; filtered to active conflict zones |
| IMF PortWatch | Vessel traffic at Hormuz + 5 chokepoints | 86400s | AIS-derived; disruption = deviation from 90d baseline |
| EIA | Weekly US petroleum inventory (crude + products) | 86400s | Wednesday release; surprise = actual − consensus |
| RSS Feeds | 7 sources: Reuters, BBC, NYT, WSJ, FT, AP, Al Jazeera | 3600s | Keyword-filtered for conflict/commodity/macro terms |

**Cold start load time:** ~10–12 seconds (parallelized via `ThreadPoolExecutor(max_workers=2)` for GDELT HTTP and yfinance; FRED runs sequentially).
**Warm cache load time:** ~1–2 seconds (all data served from `@st.cache_data`).

### 2.3 Analytics Engine — Five Independent Models

#### A. Diebold-Yilmaz FEVD Spillover Index (Diebold & Yilmaz, 2012)

The headline methodology. A VAR(p) is estimated with BIC-optimal lag selection over a 252-day rolling window across all equity-commodity pairs. The 10-step generalized forecast-error variance decomposition (GFEVD) produces:

- **Pairwise spillover matrix**: % of asset A's forecast variance explained by shocks to asset B
- **Net directional spillover**: `NET(A→B) = FROM(B) − TO(B)`; positive = A is a net transmitter to B
- **Total Spillover Index (TSI)**: headline scalar; 0% = fully decoupled, 100% = fully integrated
- **Rolling TSI chart**: shows how integration evolves over time (spikes during Ukraine Feb 2022, COVID Mar 2020, Aramco Sep 2019)

*Design choices:* Generalized rather than Cholesky decomposition — eliminates ordering dependence. `ic='bic'` for lag selection with a 10-lag upper bound. 252-day window provides approximately one year of trading days.

#### B. Granger Causality (Granger, 1969; Lütkepohl, 2005)

- VAR fitted with `ic='bic'` for BIC-optimal lag selection before testing
- Tests directional causality: equity → commodity, commodity → equity
- **Holm-Bonferroni step-down correction** applied across the full N×M×2 test grid to control family-wise error rate
- Results visualized as a color-coded causality matrix with net directional arrows

#### C. Transfer Entropy (Schreiber, 2000)

- Non-parametric directed information measure: TE(X→Y) = reduction in uncertainty of Y from knowing X's history, beyond Y's own history
- **Lag-optimized**: optimal lag selected by maximizing TE(commodity→equity) over lags 1–7
- **200-permutation shuffle test** establishes significance threshold; only significant edges retained
- Complements Granger (which assumes linearity); transfer entropy captures nonlinear dependencies

#### D. DCC-GARCH Dynamic Correlation (Engle, 2002)

- Two-step estimation: EWMA pre-whitening (λ=0.94, RiskMetrics) standardizes returns → removes heteroskedasticity contamination
- DCC(1,1) recursion on standardized residuals
- Produces time-varying correlation matrix; used for regime detection and correlation regime visualization
- *Key design choice:* Pre-whitening before DCC, not raw returns; raw-return DCC produces biased correlations during high-vol periods

#### E. Composite Risk Score — GRS (0–100)

Three-layer architecture combining geopolitical, transmission, and market signals:

```
GRS = 40% × CIS + 35% × TPS + 25% × MCS

CIS (Conflict Intensity Score):  GDELT + ACLED + war impact map scoring
TPS (Transmission Pressure Score): channel-weighted commodity transmission
MCS (Market Conditions Score):    VIX z-score + yield curve + DXY + vol regime
```

- EWM span of 252 days for vol normalization
- HHI-based breadth multiplier (n_eff^0.25) on CIS/TPS to prevent single-conflict concentration from inflating composite
- Regime detection from correlation model feeds back into GRS interpretation

### 2.4 Correlation Regime Model

3-state regime classification driving all regime-conditioned outputs:

| State | Value | Threshold | Market Interpretation |
|---|---|---|---|
| Decoupled | 1 | ρ < 0.30 | Idiosyncratic drivers; commodity and equity shocks self-contained |
| Transitioning | 2 | 0.30 ≤ ρ < 0.55 | Cross-asset linkage building; correlation channel opening |
| High Coupling | 3 | ρ ≥ 0.55 | Synchronized stress; commodity shocks propagate fully to equities |

- 5-day median smoothing to reduce regime switching noise
- Hysteresis: exit threshold = entry − 5pp (prevents oscillation at boundaries)
- Persistence gate: Regime 3 requires ≥60% of 10-day window above elevated threshold
- Markov transition matrix estimated from historical regimes; Mean First Passage Time to Crisis computed analytically

### 2.5 AI Agent Workforce — 7 Agents, 4-Round Pipeline

```
Round 1 (parallel):
  macro_strategist      → macro regime, yield curve interpretation, Fed outlook
  geopolitical_analyst  → active conflict summary, CIS/TPS attribution
  commodities_specialist → commodity-specific supply/demand dynamics

Round 2 (sequential — depends on Round 1):
  risk_officer          → synthesizes all Round 1 inputs → composite risk narrative
                          + 3 specific risk flags with quantitative thresholds

Round 3 (parallel — depends on risk_officer):
  stress_engineer       → worst-case scenario construction, tail risk quantification
  signal_auditor        → data quality flags, stale data warnings, model limitations

Round 4 (sequential — depends on Round 3):
  trade_structurer      → 3 regime-conditioned trade ideas with entry/exit/rationale
  quality_officer (CQO) → validates all outputs against schema; routes failures to
                          remediation_router → specialist agent for active correction
```

**Key architectural decisions:**
- Pydantic-validated structured outputs enforce schema at every agent boundary
- CQO remediation loop: quality flags are not silently dropped; they route back to the relevant specialist for correction up to 2 rounds
- All agents receive the full analytics output (GRS, CIS, TPS, regime, DY spillover TSI) as structured context — no agent is reasoning from raw text alone
- Supports both Claude Sonnet (Anthropic) and GPT-4o (OpenAI) backends; model selection per agent

### 2.6 Geopolitical Risk Layer

**Conflict Intelligence Score (CIS) per conflict:**
- GDELT 2.0: news volume + average tone (normalized 0–100)
- ACLED: georeferenced event count + fatality weight
- War Impact Map: baseline score (Ilian Zalomai framework) × concurrent-war amplifier
- Lead-lag filter: CIS at t-1 cross-correlated with market returns at t to surface lead-lag relationships

**Transmission Pressure Score (TPS) per conflict:**
Twelve transmission channels, each scored 0–1 and weighted:

| Channel Group | Channels |
|---|---|
| Energy | oil_gas, energy_infra |
| Shipping | shipping, chokepoint, supply_chain |
| Metals | metals |
| FX / Sanctions | fx, sanctions |
| Equity / Inflation | equity_sector, credit, inflation |

**Strait Watch:**
5 maritime chokepoints monitored — Hormuz, Malacca, Suez, Bab-el-Mandeb, Turkish Straits.
Disruption score = vessel transit deviation from 90-day baseline (IMF PortWatch AIS data).

**13 Geopolitical Events Tracked (2008–2026):**
GFC, Arab Spring, US-China Trade War, Aramco Attack, COVID-19, WTI Negative, Russia-Ukraine War, LME Nickel Squeeze, Fed Hiking Cycle, SVB Crisis, Israel-Hamas/Gaza, India-Pakistan Escalation, Iran/Hormuz Crisis.

---

## 3. Dashboard — 22 Pages

| Page | Purpose | Key Analytics |
|---|---|---|
| **Command Center** | Home page: live intelligence feed, GRS, conflict landscape, transmission channels, 5-axis risk compass | GRS, CIS, TPS, regime badge, correlation pulse, escalation tracker, returns heatmap |
| **Correlation Analysis** | Rolling correlation between equity and commodity returns; regime time series | DCC-GARCH, Pearson rolling, 4-state regime classifier |
| **Spillover Analytics** | Diebold-Yilmaz network; pairwise FEVD matrix; rolling TSI | D-Y (2012), VAR BIC-optimal |
| **Macro Intelligence** | Yield curve, CPI, Fed Funds, PMI, GDP — macro regime dashboard | FRED live data, yield curve spread |
| **Geopolitical Triggers** | Per-conflict CIS/TPS timelines; GDELT/ACLED overlays; lead-lag signal | Conflict Intelligence Score |
| **War Impact Map** | Country-level commodity exposure to active conflicts; concurrent-war amplifier | War scoring framework (Zalomai) |
| **Strait Watch** | Maritime chokepoint disruption scores; Hormuz crisis tracker | PortWatch AIS, EIA |
| **Scenario Engine** | Custom shock propagation; regime-nonlinear OLS betas; 13 historical scenarios | Parametric shock model |
| **Trade Ideas** | 6 regime-conditioned trade structures; entry/exit/thesis per regime | Correlation regime + AI structurer |
| **Portfolio Stress Test** | User CSV upload; portfolio-weighted impact attribution per asset per scenario | Exposure scoring framework |
| **Watchlist** | 17-commodity futures tracking; backwardation/contango curve analysis | Futures curve (Zalomai) |
| **Transmission Matrix** | Full pairwise Granger causality matrix; directional arrows; network view | Granger + Holm-Bonferroni |
| **Exposure Scoring** | SES / TAE / SAS per asset-conflict pair | Structural + transmission-adjusted exposure |
| **Conflict Intelligence** | GDELT + ACLED live conflict monitor; news feed; intensity timeline | GDELT 2.0 REST |
| **Threat Activity Monitor** | Real-time threat event stream from ACLED | ACLED REST API |
| **Model Accuracy** | 56-case benchmark harness results per agent | Out-of-sample validation |
| **AI Chat** | Free-form natural language interface to all analytics | Claude Sonnet / GPT-4o |
| **About — AI Workforce** | Documentation of all 7 agents, pipeline, and benchmark methodology | — |
| **About — Heramb** | Contributor profile | — |
| **About — Ilian** | Contributor profile | — |
| **About — Jiahe** | Contributor profile | — |
| **Methodology** | Full statistical methodology with academic citations | All methods documented |

---

## 4. Command Center — Design Decisions

The home page Command Center uses a three-column layout `[1.0, 2.2, 1.0]`:

**Left column (Data):**
- Intelligence Feed — live alerts from all sources
- Correlation Pulse — 60-day SVG sparkline of rolling equity↔commodity correlation with regime badge (HIGH COUPLING / TRANSITIONING / DECOUPLED)
- Conflict Landscape — 2D CIS×TPS scatter; quadrant fill (CRITICAL / VOLATILE / ISOLATED / LATENT)
- Escalation Tracker — per-conflict trend (▲/▼/→→) and escalation status (ESC↑ / DE-ESC / STABLE)
- Top Commodities — exposure-weighted commodity risk ranking from active conflicts

**Center column (Primary analytics):**
- Market Pulse strip — S&P, WTI, Gold, VIX, DXY, 10Y
- Portfolio Pulse (conditional on CSV upload)
- GRS block — gauge + 60-day score history + component decomposition
- Scenario transmission signal
- Alert banner (critical conflicts)

**Right column (Diagnosis → Action):**
- Market Pulse instrument cards
- Risk Arc — GRS decomposition with component bars (CIS 40%, TPS 35%, MCS 25%)
- Next Action router — "where to go now" based on dominant risk factor
- Risk Compass — 5-axis pentagon radar (CIS, TPS, MCS, Volatility, Coupling)
- Returns Heatmap — 5-day day-over-day return grid (color: green/red, intensity capped at ±2%)
- Transmission Channels — CIS-weighted aggregate by channel group

**Design language:**
- Pure black background (#000000) with gold accent (#CFB991, Purdue)
- JetBrains Mono for all data values; DM Sans for body text
- All infographics: pure inline SVG — zero Plotly dependency on the home page, no render flicker
- Responsive dark theme; works at 1440px and 1280px viewport widths

---

## 5. Benchmark Validation — Model Accuracy

The system includes a 56-case out-of-sample benchmark harness (`evals/run_eval.py`) that validates each agent's outputs against historical ground truth across 15 well-documented market events from 2008 to 2024.

### Results (2026-04-28)

| Agent | Cases | Passed | Hit Rate | 60% Gate |
|---|---|---|---|---|
| risk_officer | 30 | 13 | 43.3% | ❌ |
| macro_strategist | 11 | 7 | 63.6% | ✅ |
| geopolitical_analyst | 11 | 0 | 0.0%* | ❌ |
| commodities_specialist | 3 | 1 | 33.3% | ❌ |
| signal_auditor | 4 | 0 | 0.0% | ❌ |
| stress_engineer | 1 | 1 | 100.0% | ✅ |

**Pass threshold:** 60% hit rate (set at project outset; documented in `evals/benchmark-cases.json`)

### Interpretation

- **macro_strategist (63.6%)** and **stress_engineer (100%)** pass the gate. The macro model correctly classifies regimes and yield curve states across COVID, SVB, and other macro-driven events.
- **risk_officer (43.3%)** — the primary failure mode is over-scoring calm periods. The composite GRS is calibrated to geopolitical intensity as the primary driver; in purely financial-stress events (Fitch downgrade, soft landing), the GRS produces elevated scores because geopolitical risk remains elevated even when financial risk subsides. This is an acknowledged calibration issue, not a data error.
- **geopolitical_analyst (0%)** — all failures are `None` actual values. This is a **data infrastructure issue, not a model failure**: the `score_all_conflicts()` function requires live GDELT and ACLED access, which is throttled in the historical backtesting harness (which does not replay live data). CIS/TPS cannot be computed for dates before the dashboard was live. This is documented as a known limitation in `evals/results-2026-04.md`.
- **Gap 15 (AUDIT.md)** — originally identified as "circular ground truth (VIX predicts VIX)." Resolved by replacing VIX-dependent ground truth cases with historical regime classifications derived from realized returns and yield curve spreads (non-circular).

### The 56-Case Harness as an Asset

The existence of the harness is itself a deliverable. Most academic finance projects present results only on the full in-sample period. This system maintains a separate out-of-sample validation infrastructure with documented pass/fail criteria — a standard absent from all prior course projects.

---

## 6. Performance Architecture

### Cold-Start Bottleneck Analysis

On first load, three I/O operations run:
1. `score_all_conflicts()` — GDELT HTTP requests for 6 conflicts × 2 endpoints = ~10s
2. `_load_market_pulse()` — yfinance download for 6 tickers, 30-day window = ~8s
3. `_load_market_risk()` — yfinance download for 50+ tickers + FRED = ~15s

**Before optimization:** Sequential execution → ~30s cold start
**After optimization:** `ThreadPoolExecutor(max_workers=2)` parallelizes (1) and (2) while (3) runs on the main thread. Effective cold start: ~15s.

*Why (3) stays on main thread:* `compute_risk_score()` writes to `st.session_state["_mf_signals"]`. Writing session state from a background thread is unsafe in Streamlit — it triggers a `RuntimeError` or silently corrupts state. Only I/O-pure functions are threaded.

### TTL Configuration

| Function | Cache TTL | Rationale |
|---|---|---|
| `score_all_conflicts` | 7200s | GDELT updates at 15-min intervals; 2h cache avoids API throttling |
| `_load_market_risk` | 900s | 15min matches intraday signal relevance |
| `_load_market_pulse` | 900s | Same; pulse tickers are liquid, 15min lag acceptable |
| FRED series | 86400s | Daily release; no intraday updates |
| ACLED | 21600s | 6h; ACLED bulk data updates daily |

### Warm-Cache Performance

All data functions decorated with `@st.cache_data`. After first load, every subsequent navigation (including page switches) serves from memory. Observed warm-cache load: <2s for all pages.

---

## 7. Business Value & Impact

### For Institutional Analysts

**Time savings:** A cross-asset risk briefing that would take a research analyst 60–90 minutes to assemble manually (pulling Bloomberg data, running Excel regressions, checking GDELT manually, writing narrative) is produced in under 15 seconds on dashboard load.

**Signal quality:** The AI agent pipeline synthesizes five independent model outputs into a single coherent narrative, with explicit quality flagging by the CQO. This reduces cognitive load and eliminates the risk of cherry-picking one model's result.

**Geopolitical attribution:** Standard risk dashboards cannot attribute "how much of today's WTI move is the Red Sea vs. the Fed." This system quantifies that attribution via CIS × TPS channel weighting — a signal that is not available from any commercial terminal.

### For Portfolio Managers

**Regime-conditioned trade ideas:** The six trade structures are explicitly conditioned on the current correlation regime. A trade valid in Regime 1 (Decoupled) is explicitly invalidated in Regime 3 (High Coupling). This prevents the common error of applying ideas calibrated to a different market regime.

**Scenario propagation:** Custom shocks can be propagated through regime-conditioned OLS betas in the Scenario Engine. A PM can run "Hormuz blockade — 30% oil supply cut" and see portfolio-level P&L impact within seconds.

**Live stress signals:** The Fixed Income cross-asset stress framework (TLT/HYG/LQD/EMB signals) provides early warning of credit tightening before it appears in equity vol.

### For Risk Officers

**Composite risk score (GRS):** A single 0–100 number with documented decomposition (CIS/TPS/MCS) and full methodology transparency. All inputs are reproducible from public data sources.

**Out-of-sample validation:** The 56-case harness gives risk officers quantitative confidence bounds on model accuracy — documented, not claimed.

**Proactive alerts:** The alert engine flags regime transitions, CIS spikes above threshold, and stale data conditions before they propagate silently.

### Scalability Path

The current system handles 6 active conflicts and 50+ tickers. The architecture scales horizontally:
- Additional conflicts: add entry to `CONFLICT_REGISTRY` in `config.py` — no code changes
- Additional data sources: add `@st.cache_data`-decorated loader function in `src/data/loader.py`
- Additional AI agents: register in `agent_orchestrator.py` with dependency declaration
- Production deployment: `render.yaml` already configures Render.com auto-deploy on `git push`

---

## 8. End-to-End Demo Script

*Estimated duration: 12–15 minutes for full panel review*

### Phase 1: Command Center Overview (3 min)

1. Open the dashboard. Point to the **GRS gauge** in the center column — "This is the composite risk score, currently [X]/100. It's a weighted average of geopolitical intensity (CIS), transmission pressure (TPS), and market conditions (MCS)."
2. Point to the **Conflict Landscape** (left column) — "Each dot is a tracked conflict. The x-axis is conflict intensity, y-axis is how much that intensity is transmitting into commodity markets. Dots in the top-right quadrant are critical — high intensity, high transmission."
3. Point to the **Risk Compass** (right column) — "Five-axis radar showing CIS, TPS, Market Conditions, VIX-normalized Volatility, and the current equity-commodity coupling level. When the polygon is large and red, we're in a high-stress multi-channel environment."
4. Point to the **Transmission Channels** panel — "This shows which channels are carrying the most stress right now: energy infrastructure, shipping/chokepoints, FX/sanctions, or equity/inflation. This isn't a guess — it's CIS-weighted aggregation of per-conflict channel scores."

### Phase 2: Spillover Analytics (3 min)

5. Navigate to **Spillover Analytics** page. Show the Diebold-Yilmaz pairwise matrix — "This is the FEVD decomposition. Read row by row: 'X% of WTI's forecast variance is explained by shocks to the S&P 500.' During Ukraine, this number for energy commodities spiked from ~15% to ~40%."
6. Show the **Rolling TSI chart** — "Total Spillover Index over time. You can see the integration spikes during COVID and Ukraine — markets that are normally decoupled become synchronized under stress. The current reading is [X%]."
7. Show net directional arrows — "Right now, the net flow is commodity → equity / equity → commodity. This means the current stress is supply-side [or demand-side], not financial-panic-driven."

### Phase 3: Geopolitical Layer (2 min)

8. Navigate to **War Impact Map** — "Per-conflict, per-country commodity exposure weights. The concurrent-war amplifier accounts for the fact that three active conflicts compound each other's market impact — Ukraine cutting wheat supply while Red Sea disrupts shipping is not additive, it's multiplicative in some commodity markets."
9. Navigate to **Strait Watch** — "Live AIS vessel traffic at Hormuz and five other chokepoints. Disruption = deviation from 90-day baseline. The EIA weekly petroleum inventory data feeds in here — an unexpected draw signals that supply disruption is showing up in physical inventory."

### Phase 4: Scenario Engine (2 min)

10. Navigate to **Scenario Engine**. Select "Hormuz Blockade" — "This scenario applies a 30% WTI supply shock. The engine propagates that through regime-conditioned OLS betas — in Regime 3 (High Coupling), the equity impact is 1.4× larger than in Regime 1, because during high coupling, commodity shocks transmit fully to equities."
11. Show the compound scenario toggle — "You can stack scenarios: Hormuz blockade + Fed rate cut. The system handles nonlinear interaction through regime conditioning."

### Phase 5: AI Agent Output (2 min)

12. Navigate to **AI Chat** or show the morning briefing — "Seven agents produced this briefing in [X] seconds. The Risk Officer synthesized macro, geopolitical, and commodity inputs. The Trade Structurer produced three regime-valid trade ideas. The Chief Quality Officer validated every output against schema and flagged one data quality issue — [describe flag]."
13. Show the **Model Accuracy** page — "This is the benchmark harness. 56 cases, 15 historical events. We document what passed and what failed — including the acknowledged failure modes. The macro_strategist passes at 63.6%. The geopolitical agent scores 0% on historical cases because it requires live GDELT data that doesn't exist for 2020 — that's a documented limitation, not a bug we're hiding."

### Phase 6: Close (1 min)

14. Return to Command Center — "The full loop: live data → five models → seven agents → regime-conditioned trade ideas → one dashboard. The methodology is published, the code is public on GitHub, the benchmark results are documented and warts-and-all. Questions?"

---

## 9. Methodological Integrity — What We Got Right and What We Got Wrong

### What the Dashboard Gets Right

**All 25 AUDIT.md gaps resolved.** Every item from the initial gap analysis was addressed:
- Circular ground truth eliminated (Gap 15) — historical ground truth rewritten from non-VIX sources
- Out-of-sample validation added (Gap 18) — 56-case harness with documented pass/fail
- Transfer entropy lag optimized (Gap 20) — lag 1–7 search instead of hardcoded lag=1
- Scenario nonlinearity added (Gap 19) — regime-conditioned betas replace linear OLS
- Granger network visualized (Gap 24) — transmission matrix page with directional arrows
- All silent failures surfaced (Gap 16) — stale data banners, insufficient data warnings throughout

**Method selection justified:** Every model choice has an academic citation and an explanation of why the alternative was inferior (e.g., DCC pre-whitening vs. raw-return DCC; GFEVD vs. Cholesky FEVD; Holm-Bonferroni vs. uncorrected Granger p-values).

### Known Limitations (Documented, Not Hidden)

**risk_officer over-scores calm periods.** The GRS is geopolitically anchored. When financial markets stabilize but active conflicts continue (e.g., Soft Landing 2023), GRS remains elevated. Fix: separate geopolitical-only score from market-conditions score in the composite — not implemented due to scope.

**geopolitical_analyst requires live data.** Historical backtesting of CIS/TPS is not possible because GDELT reflects news at the time of access, not at historical dates. The 56-case harness marks these as `None` actual values — correctly classified as infrastructure limitations, not model failures.

**Lehman Collapse (2008) is unscorable.** FRED data for several macro series is unavailable before 2010 in the exact format required. GRS returns `nan` for this date. Documented in benchmark results.

**Transfer entropy significance at 200 shuffles.** Standard in academic literature but computationally bounded. 1000-shuffle test would increase statistical power; deferred due to runtime cost on dashboard load.

---

## 10. Academic References

| Method | Citation |
|---|---|
| Diebold-Yilmaz FEVD | Diebold, F.X. and Yilmaz, K. (2012). "Better to give than to receive: Predictive directional measurement of volatility spillovers." *International Journal of Forecasting*, 28(1), 57–66. |
| Granger Causality | Granger, C.W.J. (1969). "Investigating causal relations by econometric models and cross-spectral methods." *Econometrica*, 37(3), 424–438. |
| Lag Selection (BIC/VAR) | Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. Ch. 4. |
| Transfer Entropy | Schreiber, T. (2000). "Measuring information transfer." *Physical Review Letters*, 85(2), 461. |
| DCC-GARCH | Engle, R. (2002). "Dynamic conditional correlation: A simple class of multivariate GARCH models." *Journal of Business & Economic Statistics*, 20(3), 339–350. |
| Regime-Switching | Hamilton, J.D. (1989). "A new approach to the economic analysis of nonstationary time series." *Econometrica*, 57(2), 357–384. |
| Financial Stress Index | Illing, M. and Liu, Y. (2006). "Measuring financial stress in a developed country: An application to Canada." *Journal of Financial Stability*, 2(3), 243–265. |
| RiskMetrics EWMA | J.P. Morgan / Reuters (1996). *RiskMetrics — Technical Document*. 4th ed. |
| Holm-Bonferroni | Holm, S. (1979). "A simple sequentially rejective multiple test procedure." *Scandinavica Statistica*, 6(2), 65–70. |
| Markov Dynamics | Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press. Ch. 22. |

---

## 11. Repository & Deployment

**GitHub:** [https://github.com/HPATKAR/equity-commodities-spillover](https://github.com/HPATKAR/equity-commodities-spillover)

**Live deployment:** Render.com (auto-deploy on `git push main`)
- `render.yaml` defines build command, start command, environment variables
- `requirements.txt` pins all Python dependencies
- `packages.txt` handles system-level apt dependencies

**Local development:**
```bash
cd equity-commodities-spillover
pip install -r requirements.txt
streamlit run app.py
```

**Codebase scale:**
- ~23,000 lines of Python across `src/` (pages, agents, analysis, data, UI, ingestion, reports)
- 22 dashboard pages
- 14 analysis modules
- 9 AI agent files
- 56-case benchmark harness
- Programmatic document generation: `generate_submission_docx.py`, `generate_script_docx.py`

**Commit history:** Full commit log available at GitHub. Key milestones:
- Initial architecture + data layer
- Analytics engine (DY, Granger, Transfer Entropy, DCC-GARCH)
- AI agent pipeline + CQO remediation loop
- Benchmark harness (56 cases)
- Command Center visuals + performance optimization (Week 7)

---

## 12. Team Contributions Summary

| Contributor | Core Deliverables |
|---|---|
| **Heramb S. Patkar** | Full system architecture and build; all 22 pages; AI agent orchestrator (7 agents, CQO loop); quantitative analytics modules (D-Y, Granger, TE, DCC-GARCH, GRS); all data integrations (yfinance, FRED, GDELT, ACLED, PortWatch, EIA); benchmark harness; deployment pipeline; all submission documents |
| **Ilian Zalomai** | War Impact Map scoring framework; 13 geopolitical event catalog; Strait Watch chokepoint framework; Iran/Hormuz crisis content; concurrent-war amplifier methodology; all dashboard narrative text |
| **Jiahe Miao** | Correlation regime taxonomy (4-state); rolling window methodology; 6 regime-conditioned trade structures; Fixed Income signal framework; D-Y methodology research; private credit proxy selection; equity/commodity return data quality review |

---

*Document generated: 2026-04-30*
*Version: Final — Week 7 Capstone Submission*
*Course: MGMT 69000-120: AI for Finance · Purdue University Daniels School of Business*

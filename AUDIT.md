# Dashboard Audit — Gaps vs Core Problem Statement

**Project:** Equity-Commodities Spillover Monitor
**Course:** MGMT 69000-120 · AI for Finance · Purdue
**Maintained by:** Heramb Patkar
**Last updated:** 2026-04-18

---

## Core Problem Statement

> **How do geopolitical shocks transmit across asset classes — specifically between global equity markets and commodity futures — and what does that mean for portfolio positioning?**

The dashboard should give a single analyst everything needed to:
1. **Measure** the magnitude and direction of current equity-commodity spillover
2. **Attribute** spillover to specific geopolitical drivers
3. **Forecast** how spillover evolves under different scenarios
4. **Act** — position or hedge based on the above

Every gap below is scored against this standard.

---

## Progress Tracker

| # | Gap | Severity | Status | Fixed On |
|---|-----|----------|--------|----------|
| 1 | No Diebold-Yilmaz spillover index | Critical | ✅ Fixed | 2026-04-18 |
| 2 | Spillover directionality not shown | Critical | ✅ Fixed | 2026-04-18 |
| 3 | Commodity futures curve not analyzed | High | ✅ Fixed | 2026-04-18 |
| 4 | Sector-level equity decomposition missing | High | ✅ Fixed | 2026-04-18 |
| 5 | No regime-conditional spillover magnitude | High | ✅ Fixed | 2026-04-18 |
| 6 | GRS decimal truncated in command center | Low | ✅ Fixed | 2026-04-18 |
| 7 | Strait Watch disruption scores were static | High | ✅ Fixed | prev session |
| 8 | Commodity vol window mismatch (live 20d vs history 30d) | High | ✅ Fixed | prev session |
| 9 | Z-score clip mismatch (war scores 3.5 vs history 4.0) | High | ✅ Fixed | prev session |
| 10 | Country war weights hardcoded, not session-dynamic | High | ✅ Fixed | 2026-04-18 |
| 11 | War Impact Map multiplier range too narrow [0.92,1.15] — live signal barely moves scores | Medium | ✅ Fixed | 2026-04-18 |
| 12 | Conflict model: no lead-lag between CIS and market impact | High | ✅ Fixed | 2026-04-18 |
| 13 | Trade ideas have no backtested Sharpe / win rate | High | ✅ Fixed | 2026-04-18 |
| 14 | Scenario multipliers not compoundable | Medium | ✅ Fixed | 2026-04-18 |
| 15 | Model accuracy: circular ground truth (VIX predicts VIX) | Critical | ✅ Fixed | 2026-04-18 |
| 16 | Silent failures throughout (data absent, dashboard renders as complete) | High | ✅ Fixed | 2026-04-18 |
| 17 | CONFLICT registry: no lead-lag; stale manual data appears live | High | ✅ Fixed | 2026-04-18 |
| 18 | No out-of-sample test / cross-validation on regime classifier | Critical | ✅ Fixed | 2026-04-18 |
| 19 | Scenario engine uses linear betas (nonlinear regime not captured) | Medium | ✅ Fixed | 2026-04-18 |
| 20 | Transfer entropy uses lag=1 only; commodity→equity lag is 2–5 days | Medium | ✅ Fixed | 2026-04-18 |
| 21 | About pages UI overhaul | Low | ✅ Fixed | 2026-04-18 |
| 22 | GRS vs Map shared logic duplication (proactive_alerts) | Medium | ✅ Fixed | prev session |
| 23 | No war-risk insurance premium feed (Lloyd's surcharge) | Medium | ✅ Fixed | 2026-04-18 |
| 24 | Granger causality not network-visualized across all pairs | Medium | ✅ Fixed | 2026-04-18 |
| 25 | No commodity futures curve (backwardation/contango) analysis | High | ✅ Fixed | 2026-04-18 |

---

## Section 1 — What the Dashboard Is Missing Against the Problem Statement

These are not technical bugs. These are gaps in the core analytical answer the dashboard is supposed to provide.

---

### GAP 1 — No Diebold-Yilmaz Spillover Index `[Critical]`

**What it is:** The Diebold-Yilmaz (2012) framework decomposes a forecast-error variance of a VAR model into "from" and "to" contributions across assets. It answers: "What % of equity market uncertainty originates from commodity markets, and vice versa?"

**What we have instead:** Pearson correlation regimes and Granger causality p-values. These tell you *whether* markets are related, not *how much* of the variance is explained by cross-asset transmission.

**Why it matters for the problem statement:** The core question is spillover magnitude. We're answering correlation regime instead. An equity analyst looking at this dashboard cannot answer "how much of DAX variance last week was driven by commodity markets?" — which is the literal thesis.

**What needs building:**
- Rolling VAR(p) estimation (p=5 lags, 252-day window) across equity-commodity pairs
- Generalized Forecast Error Variance Decomposition (GFEVD)
- Net spillover index per asset (receiver vs transmitter)
- Total spillover index as single headline number
- Rolling chart: how spillover intensity evolves over time (should spike during Ukraine war, COVID, Aramco)

**Files to modify:** `src/analysis/spillover.py` (add `diebold_yilmaz()` function), new `src/pages/spillover.py` page

---

### GAP 2 — Spillover Directionality Not Shown `[Critical]`

**What it is:** Even with Granger causality, we show p-values in a table. We do not show *net directional flow*: is commodity → equity dominant right now, or equity → commodity?

**What we have:** A Granger table with green/red p-values. A transfer entropy heuristic with lag=1.

**Why it matters:** During a supply shock (e.g., Hormuz blockade), commodity → equity direction dominates. During financial panic, equity selloff → commodity selloff (demand destruction). These are opposite causal structures. The dashboard currently cannot distinguish them.

**What needs building:**
- Net pairwise spillover matrix (from DY framework or asymmetric Granger)
- Direction arrows on the correlation/spillover visualization
- "Who's leading today" text block updated each session

---

### GAP 3 — Commodity Futures Curve Not Analyzed `[High]`

**What it is:** Commodity futures markets have a term structure — spot vs 1M vs 6M vs 12M contracts. Backwardation (spot > future) signals supply shock or near-term demand spike. Contango (spot < future) signals oversupply.

**What we have:** Spot price returns for WTI, Brent, Natural Gas, etc. No curve analysis.

**Why it matters:** A geopolitical shock affecting Hormuz should push near-term WTI into backwardation. If the curve is NOT moving into backwardation despite the risk score being 89, that's a contradictory signal — and the dashboard currently cannot surface this contradiction.

**What needs building:**
- Fetch CME/ICE front-month vs deferred contracts (USO vs USL, or individual contract months via yfinance)
- Backwardation/contango indicator per energy commodity
- Integration with GRS: "Market structure corroborates geopolitical signal? Y/N"

---

### GAP 4 — No Sector-Level Equity Decomposition `[High]`

**What it is:** Geopolitical shocks don't hit all equities uniformly. An oil spike hits energy stocks up and airlines down. A gold rally hits miners. A wheat shock hits food processors. The core spillover thesis is sector-specific.

**What we have:** Index-level equity returns (S&P 500, DAX, Nikkei, etc.). No sector breakdown.

**Why it matters:** When we say "Ukraine War impacts DAX at 73/100," we're aggregating over BASF (chemicals, gas-exposed), BMW (autos), SAP (tech). The claim is imprecise. The transmission channel is sector-mediated and we don't model it.

**What needs building:**
- Fetch SPDR sector ETFs (XLE, XLF, XLU, XLB, etc.) or equivalent for European/Asian markets
- Sector beta to commodity shocks
- "Which sectors are most exposed to current conflict?" visualization
- This would make trade ideas far more actionable (Long XLE vs Short XLU for energy disruption)

---

### GAP 5 — No Regime-Conditional Spillover Magnitude `[High]`

**What it is:** The question "how does spillover change across regimes?" is the regime-specific version of the core thesis. In a low-correlation regime, commodity moves have little effect on equities. In an elevated/crisis regime, the spillover is amplified.

**What we have:** Correlation regime detection (decorrelated / normal / elevated / crisis). Separate from spillover analysis. They never talk to each other.

**Why it matters:** The correlation regime page and the spillover page could together answer: "Right now we are in elevated correlation regime, and historical DY spillover in this regime is 42% (vs 18% in normal)." That's the analytical punchline. Currently we show two separate numbers with no connection.

**What needs building:**
- Regime-stratified DY spillover computation (compute DY within each regime's time periods)
- Expected spillover conditional on current regime
- Show on command center: "In elevated regime, expected commodity→equity spillover: 38-45%"

---

### GAP 12 — Conflict Model: No Lead-Lag Between CIS and Market Impact `[High]`

**What it is:** When conflict intensity (CIS) spikes, markets don't react instantly — there's a 1–5 day lag as news is priced, sanctions are announced, and supply routes are disrupted. Our model applies CIS to TPS×MCS simultaneously.

**Why it matters:** The transmission delay is where alpha lives. If we could show "CIS spiked 3 days ago → oil should be reacting now → watch WTI for delayed transmission," that's a predictive edge the dashboard currently doesn't have.

---

### GAP 13 — Trade Ideas Have No Backtested Sharpe / Win Rate `[High]`

**What it is:** Every trade card shows entry/exit criteria but no quantitative evidence it works. A professor reviewing this would immediately ask: "What's the historical win rate of going long Natural Gas / short Nikkei in an elevated correlation regime?"

**Why it matters:** Trade ideas without backtested stats are editorial opinions. This is an AI for Finance project — we should be computing, not opining.

**What needs building:**
- For each trade idea: run the entry/exit rules over the 2008–2025 historical data
- Report: number of signals, win rate, avg return, Sharpe, max drawdown
- Show on trade card as a metrics strip

---

### GAP 15 — Model Accuracy: Circular Ground Truth `[Critical]`

**What it is:** The model accuracy page validates the regime detector using VIX > 25 as the "true crisis" label. But the regime detector itself uses VIX-derived equity vol as an input. This is circular: we're predicting our own input.

**Why it matters for the course:** Prof. Zhang would catch this immediately. Methodological circularity is a common error in ML for finance — it inflates accuracy metrics.

**What needs building:**
- Independent ground truth: NBER recession dates, FRED financial stress index (STLFSI), or external geopolitical event dates
- Out-of-sample hold-out period (train on 2008–2022, validate on 2023–2025)

---

### GAP 18 — No Out-of-Sample Test on Regime Classifier `[Critical]`

**What it is:** The correlation regime classifier (4-regime K-means or percentile-based) is evaluated on the same data it was trained on. No hold-out, no walk-forward validation.

**Why it matters:** This is a graded academic project. Any quantitative model without out-of-sample validation is academically indefensible.

---

## Section 2 — Technical Gaps (from code audit)

These are real weaknesses surfaced during code review. Severity is relative to the problem statement.

### HIGH Priority Technical Gaps

**GAP 10 — Country war weights are hardcoded, not dynamic**
File: `src/analysis/war_country_scores.py`
`COUNTRY_WAR_WEIGHTS` (e.g., USA: 5% Ukraine, 30% Hamas, 65% Iran) are hardcoded tuples. They should float based on which commodity signals are most elevated in the current session. Requested by user (session N-1) but not yet implemented.

**GAP 11 — Multiplier range [0.92, 1.15] kills live signal**
File: `src/analysis/war_country_scores.py:271`
The structural baseline scores (e.g., Iran=93) dominate because the commodity multiplier can only move scores ±8-15%. In a real crisis (z=+3.5), oil still only bumps Iran from 93 to 107 (capped at 100). The live signal is decorative, not analytical.

**GAP 16 — Silent failures throughout**
Files: `risk_score.py`, `conflict_model.py`, `strait_watch.py`, `proactive_alerts.py`
Broad `except: pass` or `except Exception: return default_value` throughout. Dashboard renders as complete even when 3+ data sources are down. Users cannot trust freshness indicators.

**GAP 17 — Conflict registry: stale manual data appears live**
File: `src/analysis/conflict_model.py:336-362`
`record_fetch("conflict_manual")` is called after every compute, so freshness timestamp is always current — even if the underlying `CONFLICTS` dict hasn't been manually updated in months. The freshness badge is meaningless for this source.

**GAP 20 — Transfer entropy lag fixed at 1 day**
File: `src/analysis/spillover.py`
Commodity-to-equity transmission takes 2–5 days empirically (sanctions announcements, supply route confirmations, refinery pass-through). Using lag=1 misses the dominant transmission window.

**GAP 24 — Granger causality not network-visualized**
File: `src/pages/` (missing feature)
We compute 15×17 = 255 Granger p-values but display them in a table. A directed network graph would immediately show which commodity is the "hub" transmitter to equities. Currently buried in a grid.

**GAP 25 — No commodity futures curve**
Files: `src/analysis/`, `src/pages/`
We use spot returns only. The term structure of commodity futures is a leading indicator of supply shock severity that we don't use anywhere.

### MEDIUM Priority Technical Gaps

**GAP 14 — Scenario multipliers can't compound**
File: `src/analysis/scenario_state.py`
"Escalation" and "Shipping Shock" are separate mutually exclusive scenarios. No way to model "escalation AND shipping shock simultaneously" — which is the realistic Iran/Hormuz scenario.

**GAP 19 — Scenario engine uses linear betas**
File: `src/pages/scenario_engine.py`
OLS beta of S&P 500 on oil is computed over 252 days and applied linearly. Oil +15% in a supply-shock regime has a different equity beta than oil +15% in a demand-recovery regime. The direction of the shock matters.

**GAP 23 — No war-risk insurance premium**
File: `src/pages/strait_watch.py`
Lloyd's of London war-risk surcharges for Hormuz/Red Sea routes are publicly available and directly price disruption risk. This is a hard number we could display but don't fetch.

---

## Section 3 — What Good Looks Like (Target State)

When this dashboard is at its best, an analyst visiting the command center should see:

1. **Headline:** "Current equity-commodity spillover: 41% (↑ from 28% last week). Regime: Elevated. Driver: Energy supply shock via Hormuz."
2. **Decomposition:** "Of that 41%, 28% is oil→equity, 9% is gold→equity (safe haven bid), 4% is agricultural (secondary)."
3. **Direction:** "Net transmitter today: Brent Crude (+18% net outward spillover). Net receiver: European equities (-22% net inward)."
4. **Forward view:** "Under Escalation scenario (1.45×), spillover projected to reach 58%. Historical precedent: Aramco attack 2019 peaked at 54%."
5. **Trade:** "In this regime, Long Brent / Short DAX has historically returned 14% over 30 days (Sharpe 1.2, win rate 68%). Entry: today."

**We currently deliver items 4 (partially) and 5 (without backtests).** Items 1–3 are the core thesis and are absent.

---

*Update this file by changing `⬜ Open` → `✅ Fixed` and filling in `Fixed On` date whenever a gap is resolved.*

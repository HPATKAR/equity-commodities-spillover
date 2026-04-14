# Data Quality Report — Cross-Asset Spillover Monitor

**As of:** 2026-04-14
**Standard:** SR 11-7 model risk management (conceptual soundness + ongoing monitoring)

---

## 1. Validation Rules by Data Source

### 1.1 Equity & Commodity Returns (`loader.load_returns`)

| Rule | Check | Action on Failure |
|------|-------|-------------------|
| No all-NaN columns | `df.isna().all(axis=0)` | `st.warning` listing bad columns |
| Return series finite | `np.isfinite(df).all()` after ffill | Warning with NaN/Inf count |
| Minimum row count | ≥ 30 rows required | Warning; downstream agents degrade gracefully |
| Column types | All numeric (float64) | Validated by pandas dtype check |

Implemented in: `src/data/loader._validate_returns()`

### 1.2 FRED Macro Series (`loader.load_fred_series`)

| Rule | Check | Action on Failure |
|------|-------|-------------------|
| Non-empty result | `prices.empty` | Warning; agent context omits missing fields |
| Price series finite | `np.isfinite(prices)` | Warning with row/column counts |
| Column types | Numeric | Pandera-style dtype check |

Implemented in: `src/data/loader._validate_prices()`

### 1.3 RSS Headlines (`geo_rss.ingest_headlines`)

| Rule | Check | Action on Failure |
|------|-------|-------------------|
| ID format | `len(h.id) == 12` (SHA-1 hex) | Row dropped |
| Title non-empty | `h.title.strip() != ""` | Row dropped |
| Relevance in range | `0.0 ≤ h.relevance ≤ 100.0` | Row dropped |
| Regions / commodities | `isinstance(list)` | Row dropped |
| Drop count reported | `st.warning(f"{n} malformed rows dropped")` | Warning displayed |

Implemented in: `src/ingestion/geo_rss.ingest_headlines()` — validation loop after scoring.

### 1.4 Fixed Income / Credit Spreads (`loader.load_fixed_income_prices`)

| Rule | Check | Action on Failure |
|------|-------|-------------------|
| Non-empty | `result.empty` | Warning |
| Return series finite | `_validate_returns()` applied | Warning |

Source: HYG / LQD via yfinance (proxy for credit spreads). Documented in freshness registry as `YF/HYG·LQD`.

---

## 2. Freshness Thresholds (Staleness SLAs)

| Source Key | Label | Warn After | Stale After | How Triggered |
|------------|-------|-----------|-------------|---------------|
| `equity_returns` | Equity Returns | 6 h | 26 h | `load_returns()` on data load |
| `commodity_returns` | Commodity Returns | 6 h | 26 h | `load_returns()` on data load |
| `fred_macro` | FRED Macro | 24 h | 72 h | `load_fred_series()` on data load |
| `fred_spreads` | Credit Spreads (HYG/LQD) | 4 h | 26 h | `load_fixed_income_prices()` |
| `cot_data` | COT Positioning | 48 h | 120 h | `load_cot_data()` at startup |
| `rss_headlines` | RSS Headlines | 1 h | 4 h | `ingest_headlines()` |
| `conflict_manual` | Conflict Intelligence | 24 h | 72 h | `score_all_conflicts()` |

Displayed on: **Model Accuracy → Data Source Integrity** panel.

---

## 3. Outlier Flags

| Signal | Flag Condition | Where Applied |
|--------|---------------|---------------|
| Extreme single-day equity return | `|ret| > 0.10` (10%) | `_validate_returns()` warning |
| NaN share > 10% | `df.isna().mean() > 0.10` per column | `_validate_returns()` warning |
| Correlation outside [−1, 1] | Pairwise corr validity | `average_cross_corr_series()` |
| RSS relevance clamp | `relevance > 100` | Row dropped in geo_rss |
| Confidence out of [0, 1] | `calibrate_confidence()` clamps to [0, 1] | `agent_state.calibrate_confidence()` |

---

## 4. Agent Output Schema Validation

| Agent | Output Type | Validation Method |
|-------|------------|-------------------|
| `trade_structurer` | `TradeOutput` (Pydantic) | `model_validate_json()` + retry ×2 |
| All other agents | Free narrative `str` | Checked non-empty before `set_output()` |
| Inter-agent handoffs | `AgentHandoff` TypedDict | Typed field access; numeric comparison in divergence detection |

---

## 5. Known Limitations

1. **CIS/TPS not validated against historical sources.** Conflict intensity scores are live-only; benchmark assertions proxy from `risk_score`.
2. **Granger hit rates not checkable from prices alone.** Require VAR model estimation per pair per window; evaluated via the live Model Accuracy page, not the standalone eval script.
3. **FRED macro series dependent on API key.** Without a valid key, macro freshness shows stale; agents degrade gracefully to monitoring mode.
4. **yfinance may return stale or missing data for distant dates.** The eval script handles this with `_computed=False` flag; affected snapshots are reported as PARTIAL.

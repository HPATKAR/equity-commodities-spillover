# Benchmark Results — 2026-04-28

Pass threshold: **60%** hit rate

## Summary

| Agent | Cases | Passed | Hit Rate | Gate |
|-------|------:|-------:|---------:|------|
| risk_officer | 30 | 13 | 43.3% | ❌ FAIL |
| macro_strategist | 11 | 7 | 63.6% | ✅ PASS |
| geopolitical_analyst | 11 | 0 | 0.0% | ❌ FAIL |
| commodities_specialist | 3 | 1 | 33.3% | ❌ FAIL |
| signal_auditor | 4 | 0 | 0.0% | ❌ FAIL |
| stress_engineer | 1 | 1 | 100.0% | ✅ PASS |

## Per-Agent Detail

### risk_officer

| Date | Label | Field | Expected | Actual | Pass |
|------|-------|-------|----------|--------|------|
| 2020-03-16 | COVID Crash | `risk_score` | gte 75 | 82.6 | ✅ |
| 2020-03-16 | COVID Crash | `regime` | eq 3 | 3 | ✅ |
| 2020-03-23 | COVID Trough | `risk_score` | gte 70 | 83.2 | ✅ |
| 2020-03-23 | COVID Trough | `regime` | eq 3 | 3 | ✅ |
| 2022-02-24 | Russia-Ukraine Invasion | `risk_score` | gte 70 | 64.4 | ❌ |
| 2022-02-24 | Russia-Ukraine Invasion | `regime` | eq 3 | 1 | ❌ |
| 2022-03-08 | Ukraine War — Peak Commodity Spike | `risk_score` | gte 75 | 59.6 | ❌ |
| 2023-03-10 | SVB Collapse | `risk_score` | gte 55 | 67.3 | ✅ |
| 2023-03-10 | SVB Collapse | `regime` | gte 2 | 1 | ❌ |
| 2022-06-13 | CPI 9.1% — Inflation Peak | `risk_score` | between [55, 80] | 65.0 | ✅ |
| 2018-12-24 | Fed Tightening Tantrum | `risk_score` | gte 55 | 64.9 | ✅ |
| 2015-08-24 | China Devaluation / Black Monday | `risk_score` | gte 70 | 65.7 | ❌ |
| 2015-08-24 | China Devaluation / Black Monday | `regime` | eq 3 | 1 | ❌ |
| 2023-10-07 | Hamas Attack / Gaza Escalation | `risk_score` | between [50, 75] | 75.1 | ❌ |
| 2018-10-11 | Fed-Driven Equity Drawdown | `risk_score` | between [50, 70] | 74.7 | ❌ |
| 2022-09-28 | UK Gilt Crisis / LDI Unwind | `risk_score` | gte 58 | 77.2 | ✅ |
| 2008-09-15 | Lehman Collapse | `risk_score` | gte 85 | nan | ❌ |
| 2008-09-15 | Lehman Collapse | `regime` | eq 3 | 1 | ❌ |
| 2023-08-01 | Fitch US Downgrade | `risk_score` | between [45, 65] | 73.8 | ❌ |
| 2024-04-15 | Iran Drone Attack on Israel | `risk_score` | between [50, 72] | 56.2 | ✅ |
| 2021-11-26 | Omicron Discovery | `risk_score` | between [55, 75] | 82.8 | ❌ |
| 2021-06-01 | Post-COVID Recovery Peak | `risk_score` | lte 45 | 70.6 | ❌ |
| 2021-06-01 | Post-COVID Recovery Peak | `regime` | eq 1 | 1 | ✅ |
| 2023-11-01 | Soft Landing Optimism | `risk_score` | lte 50 | 62.3 | ❌ |
| 2023-11-01 | Soft Landing Optimism | `regime` | lte 2 | 1 | ✅ |
| 2024-01-02 | 2024 Start — Risk-On | `risk_score` | lte 40 | 62.3 | ❌ |
| 2024-01-02 | 2024 Start — Risk-On | `regime` | eq 1 | 1 | ✅ |
| 2019-09-01 | Trade War Pause | `risk_score` | between [30, 55] | 68.6 | ❌ |
| 2017-01-01 | Low-Vol Calm | `risk_score` | lte 35 | 55.7 | ❌ |
| 2017-01-01 | Low-Vol Calm | `regime` | eq 1 | 1 | ✅ |

### macro_strategist

| Date | Label | Field | Expected | Actual | Pass |
|------|-------|-------|----------|--------|------|
| 2020-03-16 | COVID Crash | `regime` | eq 3 | 3 | ✅ |
| 2020-03-23 | COVID Trough | `regime` | eq 3 | 3 | ✅ |
| 2023-03-10 | SVB Collapse | `yield_curve_spread` | lt 0 | -0.37 | ✅ |
| 2022-06-13 | CPI 9.1% — Inflation Peak | `cpi_yoy` | gte 8.0 | None | ❌ |
| 2022-06-13 | CPI 9.1% — Inflation Peak | `regime` | gte 2 | 1 | ❌ |
| 2018-12-24 | Fed Tightening Tantrum | `regime` | gte 2 | 1 | ❌ |
| 2018-10-11 | Fed-Driven Equity Drawdown | `regime` | gte 2 | 2 | ✅ |
| 2022-09-28 | UK Gilt Crisis / LDI Unwind | `yield_curve_spread` | lt 0 | -8.65 | ✅ |
| 2008-09-15 | Lehman Collapse | `regime` | eq 3 | 1 | ❌ |
| 2023-08-01 | Fitch US Downgrade | `regime` | gte 1 | 1 | ✅ |
| 2023-11-01 | Soft Landing Optimism | `regime` | lte 2 | 1 | ✅ |

### geopolitical_analyst

| Date | Label | Field | Expected | Actual | Pass |
|------|-------|-------|----------|--------|------|
| 2020-03-16 | COVID Crash | `cis` | lte 40 | None | ❌ |
| 2022-02-24 | Russia-Ukraine Invasion | `cis` | gte 75 | None | ❌ |
| 2022-02-24 | Russia-Ukraine Invasion | `tps` | gte 65 | None | ❌ |
| 2022-03-08 | Ukraine War — Peak Commodity Spike | `cis` | gte 80 | None | ❌ |
| 2022-03-08 | Ukraine War — Peak Commodity Spike | `tps` | gte 75 | None | ❌ |
| 2023-03-10 | SVB Collapse | `cis` | lte 50 | None | ❌ |
| 2015-08-24 | China Devaluation / Black Monday | `cis` | between [40, 65] | None | ❌ |
| 2023-10-07 | Hamas Attack / Gaza Escalation | `cis` | gte 65 | None | ❌ |
| 2023-10-07 | Hamas Attack / Gaza Escalation | `tps` | gte 55 | None | ❌ |
| 2024-04-15 | Iran Drone Attack on Israel | `cis` | gte 70 | None | ❌ |
| 2024-04-15 | Iran Drone Attack on Israel | `tps` | gte 60 | None | ❌ |

### commodities_specialist

| Date | Label | Field | Expected | Actual | Pass |
|------|-------|-------|----------|--------|------|
| 2022-02-24 | Russia-Ukraine Invasion | `cmd_vol_z` | gte 1.5 | -1.22 | ❌ |
| 2022-03-08 | Ukraine War — Peak Commodity Spike | `cmd_vol_z` | gte 2.0 | 1.72 | ❌ |
| 2021-11-26 | Omicron Discovery | `cmd_vol_z` | gte 1.0 | 1.64 | ✅ |

### signal_auditor

| Date | Label | Field | Expected | Actual | Pass |
|------|-------|-------|----------|--------|------|
| 2020-03-16 | COVID Crash | `granger_hit_rate` | lte 0.45 | None | ❌ |
| 2008-09-15 | Lehman Collapse | `granger_hit_rate` | lte 0.4 | None | ❌ |
| 2021-06-01 | Post-COVID Recovery Peak | `granger_hit_rate` | gte 0.55 | None | ❌ |
| 2024-01-02 | 2024 Start — Risk-On | `granger_hit_rate` | gte 0.55 | None | ❌ |

### stress_engineer

| Date | Label | Field | Expected | Actual | Pass |
|------|-------|-------|----------|--------|------|
| 2020-03-16 | COVID Crash | `regime` | eq 3 | 3 | ✅ |

# Post-Submission Extensions

### Equity-Commodities Spillover Monitor · continued work after the Week 7 capstone
**Author:** Heramb S. Patkar · **Period:** July 2026 · **Status:** post-submission, individual work

---

## Why this document exists

The Week 7 capstone ([`CAPSTONE_FINAL.md`](CAPSTONE_FINAL.md)) was submitted on 2026-04-30 and is a point-in-time record. I did not stop building after the grade was in. This document covers what I added in **July 2026**, so the timeline stays honest: here is what was submitted, and here is what I kept building afterward, dated as such. Everything below is verifiable in the commit history and lives in the terminal today; the [README](../README.md#book-risk-character-prosecuting-the-book) is the canonical description.

I am writing it because continuing past the deadline is the point. The capstone showed a working system. This work asks a harder question of it, and answers honestly.

---

## The catalyst: reviewing my own book the way a desk would

After submission I put the Trade Ideas book through the review a buy-side veteran would actually run, and built to close every gap I found. The uncomfortable question was: **is my trade book alpha, or is it just factor beta wearing a costume?** Most student projects, and plenty of professional ones, never ask. I decided the honest move was to build the machine that could indict my own book, and then report whatever it found.

It found that the book is beta. So I said so, on the page and in the report, and rebuilt the product identity around what the evidence supports.

---

## What I added

### 1. Book Risk Character: a six-audit self-prosecution of the deployed book

All six reuse one weight-normalised book-return series and run live under the ranked book on the Trade Ideas page.

| Audit | Method | Honest finding (current book) |
|-------|--------|-------------------------------|
| **Factor Attribution** | Book return on market (S&P 500) + market-orthogonalised thematic factors, HAC/Newey-West errors; Jensen alpha, tracking error, effective number of bets | ~0.68 market beta, Jensen alpha ~0 and insignificant, ~6 effective bets of 11 -> **SECTOR-TILT** |
| **Factor-Neutral Skill Test** | Excess book return on Fama-French 5 + Momentum (Ken French daily factors); Deflated Sharpe Ratio on the residual (Bailey and Lopez de Prado) | Raw Sharpe collapses to ~0 after stripping the academic factors; DSR ~1% -> **NO SKILL AFTER FACTORS** |
| **Rolling Exposure** | Univariate rolling 126-day factor betas | Market beta ranged ~0.45 to ~1.23 over the sample -> **DRIFTING**, not a stable number |
| **Cost & Capacity** | Turnover-driven cost drag; per-position days-to-exit at a share of dollar ADV, with a data-quality guard on implausibly thin volume | Modest net-of-cost drag; capacity set by the least-liquid name; one name auto-flagged `?data` |
| **Hedge Overlay** | Sequential hedge (market beta on SPY, residual sector tilts on ITA/GLD/XLE/TLT/UUP) | The tradeable ETF basket that neutralises the book |
| **Out-of-Sample Validation** | Walk-forward: hedge ratios re-fit on a trailing 252-day window, applied only to the next unseen month | Removes variance and pulls market beta toward zero forward -> **HOLDS OUT OF SAMPLE** |

### 2. The product: a tradeable, out-of-sample-validated hedge overlay

The diagnosis (the book is beta) implies a product: the exact ETF basket that strips the systematic risk, so a parent portfolio can hold the intended thematic view without the market beta it already owns. I did not just propose it. The walk-forward backtest confirms it is not an in-sample artifact, and I keep it honest about its limits (the sector legs over-hedge as exposures drift; the residual has no alpha, so it is exposure control inside a portfolio, not a standalone strategy).

### 3. Reframed identity plus one falsifiable call

Because all four diagnostics agree, I stopped implying alpha and reframed the Trade Ideas page as what it is: a **risk-monitoring and hedge-overlay tool**. The page states that plainly, discloses that the static thesis library carries selection look-ahead, and puts one falsifiable call on the record.

### 4. Supporting rigor fixes

- **Diebold-Yilmaz generalized FEVD.** An econometrics review found the headline direction call was using the order-dependent Cholesky FEVD, which had biased it toward "equity-led." Replaced with the order-invariant generalized FEVD (Pesaran-Shin), cross-checked against an independent Baruník-Křehlík spectral GFEVD.
- **Spillover page reframed** as risk-monitoring, not forecasting, after showing that transmission direction does not predict forward returns (it validates against forward volatility instead).
- **CIS disclosure** (analyst-assessed composite, a monitoring input, not a return forecast) and a **thesis-selection look-ahead** disclosure.

### 5. The deliverable carries all of it

*Generate Desk Report (PDF)* now exports a ~20-page institutional document with an Executive-Summary Mandate and all six audits. The screen and the document tell the identical story.

---

## Commit trail (for traceability)

The July work is a continuous, dated arc in the history. Representative commits:

- Book Factor & Alpha Decomposition (screen + PDF)
- Factor-Neutral Skill Test on FF5 + Momentum (screen + PDF)
- Rolling Factor Exposure
- Cost & Capacity with the `?data` guard
- Product-identity reframe + Executive-Summary Mandate
- Hedge Overlay (screen + PDF)
- Walk-forward out-of-sample validation of the overlay (screen + PDF)

---

## An honest note

None of this was in the April submission, and it should not be read as if it were. It is post-deadline work I did because the project deserved a harder standard than the one it passed. The capstone document stands as the record of what was submitted; this stands as the record of what I did next.

*Written in my voice as lead architect and author of this extension. See [`CAPSTONE_FINAL.md`](CAPSTONE_FINAL.md) section 12 for full team contributions on the submitted system.*

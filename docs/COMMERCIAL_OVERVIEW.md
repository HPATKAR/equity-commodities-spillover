# Cross-Asset Spillover Monitor
## Commercial Overview

**One line.** Priced cross-asset risk intelligence that connects the geopolitics you read about to the book you actually run.

**What it is.** A research terminal that links geopolitical risk (conflict intensity, maritime chokepoints, transmission channels) to cross-asset market risk (equity-commodity spillover, correlation regimes, Diebold-Yilmaz connectedness), and turns that link into concrete, priced actions on a user's own portfolio: audits, hedges, alerts, and client-ready outputs.

**What it is not, and why that matters.** It is not an alpha engine. The terminal audits its own model book and reports, on screen and in every generated PDF, that the book is market and factor beta with no statistically significant factor-neutral alpha. The honest edge is risk-monitoring and hedge sizing: connectedness carries a modest but real signal on forward volatility, while direction-of-return is not predicted. Selling the tool for exactly what it is, is the point. It is also why the outputs indict their own book rather than flatter it, which is what earns trust from a buyer who has seen through black-box "signal" products before.

---

### Who it is for (ranked by willingness to pay)

1. **Commodity-exposed risk seats.** Trading houses, energy and agriculture desks, corporate treasuries hedging fuel, grains and metals. The sharpest wedge, and the least served by incumbents.
2. **Boutique multi-asset and macro managers ($100M to $5B).** The largest addressable count. They cannot justify Bloomberg PORT plus Barra plus a geopolitical service plus a quant team.
3. **Family offices and sophisticated RIAs.** They want a plain-language geopolitical narrative and outputs they can hand to an end client.
4. **Geopolitical and political-risk shops.** A feed and API partnership, not a terminal seat.

Not for: mega-institutions that already own the incumbent stack, alpha hedge-fund pods (there is no alpha here), or retail.

---

### What it does today (six capabilities, built and demonstrable)

| Capability | For | What it delivers |
|---|---|---|
| **Portfolio X-Ray** | 1 to 3 | Audits any book: factor attribution, a Fama-French factor-neutral skill test, rolling exposures, cost and capacity, a tradeable hedge overlay, and its out-of-sample validation. Deliberately discriminating: a book the thematic model calls "alpha present" the factor test will overrule as beta if that is what it is. |
| **Commodity Hedge Desk** | 1 | A physical exposure in native units (barrels, bushels, tonnes) becomes a futures hedge ticket, with roll carry read off the live forward curve and a hedge ratio scaled by the terminal's own geopolitical stress engine. |
| **White-label Tearsheet** | 2 to 3 | The X-Ray audit rendered as a client-, IC- or LP-ready PDF on the manager's own letterhead. |
| **Alert Center** | 1, 3 | The terminal's live signals priced as an estimated dollar impact on the user's own exposure, ranked by cost, with a downloadable briefing. |
| **Client Brief** | 3 | The same intelligence translated into a plain-language, jargon-free client brief and white-label PDF. Rule-based, so it needs no AI key and reads the same way every time. |
| **Signals Export** | 4 | Conflict intensity and transmission scores, Diebold-Yilmaz connectedness, and a per-commodity Geopolitical Stress Index as a versioned JSON and CSV feed against a documented schema. |

---

### Why not just Bloomberg plus Barra plus a geopolitical service

No incumbent connects geopolitical transmission, cross-asset risk, the user's own book, and a priced action in one place. A boutique cannot afford four vendors and a quant team, and this is one honest tool that does the join. Because it audits its own book down to factor beta, it earns a credibility that a product which only ever flatters its signal cannot.

---

### Honest limits, and what going live requires

- **Data.** Public feeds (Yahoo Finance, FRED, IMF PortWatch, GDELT, CFTC) power it today. Licensed data is the precondition for institutional due diligence. This is the hard gate, named plainly.
- **Infrastructure.** Multi-tenant accounts, authentication and entitlements, a live HTTP API, and push delivery (email, Slack, SMS) are not built. The code is structured so they drop in: the signals feed is already one pure function sitting behind the export page, and the alert briefing is one delivery channel away from being pushed.
- **The edge.** Modest and real: risk-monitoring and hedge sizing, not alpha, priced accordingly.

---

### Status and roadmap

- **Built and demonstrable:** the six capabilities above.
- **Pilot-gated next:** licensed data, accounts and entitlements, a live API, and push delivery. Their shape should be set by a first pilot's actual requirements, not guessed in advance.

**Close.** Cross-asset risk intelligence that connects the geopolitics you read about to the book you actually run: priced, honest about its edge, and sold for exactly what it is.

---

*Cross-Asset Spillover Monitor · Heramb S. Patkar, Jiahe Miao, Ilian Zalomai · Purdue Daniels School of Business. For research and educational purposes. Not investment advice.*

---
version: 1
slug: "web-index-html"
primary_target: "web/index.html"
related_targets: []
---

# Farmer hub

Scope: the S.E.D.O.S.S. farmer hub (web/), laptop first, English. Mode: Operate.
Audience: durian growers running a unit; also shown to IGC judges as the working product.
Job: pick the orchard's trained site, understand today's AI decision, watch the unit, and act (irrigate now, water only, skip, pause, regenerate, refill).
Data: virtual Pi API (ml/run_bench.py --serve): state, events, daily log, hardware, sites. Everything shown from the bench is simulated and labelled so.
Constraints: only trained sites are selectable; distance to the nearest trained site is always shown; never invent field results.

## Direction contract

THESIS: Today's decision is a tear-off almanac leaf: the date as a huge red numeral with Do and Avoid columns in plain words. It refuses the KPI-tile agritech dashboard where the decision is one card among many.

OWN-WORLD: Near-white sheet (#fbfbf8) and white leaf, ink black #1a1a1a, almanac red #c62828 for date and avoid, leaf green #2e7d32 for do. Heavy condensed Archivo numerals, Archivo text, tabular figures. Two-pixel ink rules, square corners, bordered tiles.

STORY: The farmer sees the date and what to do, reads why (soil, forecast, crop water use) in one line, checks the bench is healthy, then acts with standard buttons.

FIRST VIEWPORT: Left rail (brand, nav, current site). Centre: the day leaf at full height, numeral around 300px, Do and Avoid below. Right: bench-now tiles in two rows of three, the alert, then controls with Irrigate now primary.

FORM: Tear-off Almanac, my rank 1 of 7. seed:b5e7e1781cf0175b. Signature move: when the simulated day changes, the old leaf tears off and falls away to reveal the next day.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

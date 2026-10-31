---
name: S.E.D.O.S.S. Hub
description: Farmer hub for the S.E.D.O.S.S. unit, built as a tear-off almanac page.
colors:
  sheet: "#fbfbf8"
  paper: "#ffffff"
  ink: "#1a1a1a"
  ink-2: "#4a4945"
  ink-3: "#62615c"
  rule: "#d9d7cf"
  red: "#c62828"
  red-tint: "#fbeaea"
  green: "#2e7d32"
  green-tint: "#e8f2e8"
  amber: "#8a5a00"
  amber-tint: "#fdf3dc"
  blue: "#1f5f8b"
typography:
  display:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(170px, 31vh, 300px)"
    fontWeight: 900
    lineHeight: 0.86
    letterSpacing: "-0.02em"
    fontVariation: "'wdth' 70"
  headline:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "40px"
    fontWeight: 900
    lineHeight: 1.15
    fontVariation: "'wdth' 80"
  title:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "32px"
    fontWeight: 900
    lineHeight: 1.15
    fontVariation: "'wdth' 85"
  numeral:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "38px"
    fontWeight: 800
    lineHeight: 1.1
    fontFeature: "'tnum'"
    fontVariation: "'wdth' 78"
  body-lg:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.35
  body:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "'tnum'"
  label:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "14px"
    fontWeight: 700
    letterSpacing: "0.1em"
  label-sm:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "12px"
    fontWeight: 700
    letterSpacing: "0.06em"
rounded:
  none: "0px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "24px"
  xl: "34px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.none}"
    padding: "11px 16px"
  button-secondary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "11px 16px"
  button-secondary-hover:
    backgroundColor: "{colors.sheet}"
  button-active:
    backgroundColor: "{colors.red-tint}"
    textColor: "{colors.red}"
  button-ghost:
    textColor: "{colors.ink}"
    padding: "11px 8px"
  chip:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "6px 12px"
  chip-on:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  input:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "7px 8px"
  reading-tile:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.numeral}"
    padding: "14px 16px 12px"
  notice-warning:
    backgroundColor: "{colors.amber-tint}"
    textColor: "{colors.amber}"
    padding: "11px 14px"
  notice-alarm:
    backgroundColor: "{colors.red-tint}"
    textColor: "{colors.red}"
    padding: "11px 14px"
---

# Design System: S.E.D.O.S.S. Hub

## Overview

**Creative North Star: "The Tear-off Almanac"**

The hub is a wall almanac, not a dashboard. Today's decision is a full-height white leaf: the day of the month as a huge condensed red numeral, the month and weekday ruled in red above it, and two plain-language columns, Do in leaf green and Avoid in almanac red, below. Everything else (unit readings, controls, logs, maps, history) sits beside or behind that leaf on a near-white sheet, organised by ink rules and square bordered tiles the way a printed form or ledger would be.

Density is calm and printed. One heavy variable sans (Archivo) does all the work, pushed narrow and black for numerals and kept at normal width for reading. Tabular figures run throughout so measurements line up like a ledger. Colour is functional and scarce: ink on paper by default, red for the date and anything to avoid or fix, green for what to do and what is healthy, amber for warnings, blue only for water.

The one moving gesture belongs to the world: when the simulated day changes, the old leaf tears off its perforated top edge and falls away to reveal the next day. Every other state change is a short colour transition.

**Key Characteristics:**
- A single dominant leaf per screen carries today's decision; it is never one card among many.
- Condensed black Archivo numerals for figures, normal-width Archivo for words, tabular figures everywhere.
- Square corners everywhere, structure drawn with 2px ink rules and 1px hairlines.
- Colour carries meaning, never decoration.
- Flat paper; the only shadow is on the leaf while it falls.

## Colors

Ink on near-white paper, with four signal colours that each mean one thing.

### Primary
- **Almanac Red** (red): the date numeral, the leaf header and its 2px underline, the Avoid heading, alarms, the active nav item, the alert count badge, the current site, and the focus ring. It is the almanac's printed red, and it marks "look here" or "don't".

### Secondary
- **Leaf Green** (green): the Do heading, healthy readings, and running-stage stamps. Its tint (green-tint) backs the "on" stamp only.

### Tertiary
- **Warning Amber** (amber): warning notices, regenerating stamps, brine levels, and warning log lines. It is always text on its own tint (amber-tint) with an amber border.
- **Water Blue** (blue): water only: the watering progress bar, tank level fills, the live arrow on the water path, the soil-water line in the history chart, and the map click pin.

### Neutral
- **Sheet** (sheet): the page behind the leaf, the hover fill on secondary buttons, the offline backdrop, and code blocks.
- **Paper** (paper): the leaf, the rail, and every tile, block, form, and table surface.
- **Ink** (ink): body text, 2px structural rules, tile borders, the primary button fill, and the selected chip.
- **Ink 2** (ink-2): secondary text such as reading labels, the "why" line, table headers, and view intros.
- **Ink 3** (ink-3): captions, units, empty states, and stale readings.
- **Rule** (rule): 1px hairlines inside tiles and lists, quiet block borders, and the perforation dots on the leaf edge.

### Named Rules
**The One Meaning Rule.** Each signal colour has one job: red = date, avoid, alarm; green = do, healthy; amber = warning; blue = water. Never use a signal colour as decoration or brand colour.

**The Tint-With-Its-Ink Rule.** A tint (red-tint, green-tint, amber-tint) appears only behind text and a border in its own full colour, never on its own as a panel fill.

## Typography

**Display Font:** Archivo variable (self-hosted, weight 400–900, width 62–125%), with Helvetica Neue, Arial fallback
**Body Font:** Archivo, same file

**Character:** One family, two widths. Narrow black Archivo makes figures loud and printed; normal-width Archivo keeps the plain-language advice easy to read.

### Hierarchy
- **Display** (900, width 70%, clamp(170px, 31vh, 300px), line-height 0.86): the day numeral on the leaf, in red. 150px under 640px. The offline "No signal" uses the same treatment at clamp(64px, 12vw, 120px).
- **Headline** (900, width 80%, 40px; 32px on mobile): view titles (Unit, Sites, History, Alerts and log).
- **Title** (900, width 85%, 32px): the Do and Avoid headings. Roster site names use 800 at 20px.
- **Numeral** (800, width 78%, 38px; 30px on mobile and in water-path stages): live readings in the bench tiles and flow stages.
- **Body Large** (400, 17px, line-height 1.35): Do/Avoid list items, with key quantities in 800.
- **Body** (400, 15px, line-height 1.45, tabular figures): everything else; the "why" line is capped at 62ch.
- **Label** (700, 14px, 0.1em, uppercase): section heads above a 1–2px ink rule (Unit now, Today's log, block titles). The leaf header uses 800 at 15px, 0.08em, in red.
- **Label Small** (700, 11–13px, 0.06em, uppercase): table column heads, flow stage names, the Simulated unit tag, and roster status.

### Named Rules
**The Ledger Figures Rule.** All figures use tabular numerals, so columns of readings, times, and litres line up.

**The Two Widths Rule.** Condense Archivo (70–85%) only for numerals and big headings. Running text stays at normal width.

## Layout

A fixed 220px rail (brand, nav, current site, simulation tag and speed) sits beside the main area. On Today, the main area splits into the leaf column (clamp(400px, 34vw, 540px), full viewport height, 30px 34px padding) and a side column (28px 32px padding, 22px vertical gap) with the Unit now heading, a 3×2 readings grid, a notice, the control row, and the log. Other views use 30px 34px padding with a 26px stack gap, starting with a view head (title left, intro or filter right, 2px ink rule below). Content blocks sit in grids with a 24px gutter (the gap token).

Breakpoints: at 1180px the bench grid drops to two columns, the water path wraps to 3×2, and Sites stacks map over roster. At 980px the rail becomes a top bar with an inline nav and a 2px ink bottom rule, and the leaf stacks above the side column. At 640px gutters tighten to 16px, readings go 2×3, the flow and grids go to one column, and the numeral fixes at 150px.

## Elevation & Depth

Flat, like printed paper. Depth comes from paper on sheet, ink rules, and borders, not shadows. The one shadow belongs to the tear-off: while the old leaf falls it carries a soft lift so it reads as a separate sheet, and the shadow leaves with the leaf.

### Shadow Vocabulary
- **Falling leaf** (`box-shadow: 0 18px 40px rgba(26, 26, 26, .18)`): only on the leaf during the tear animation.

### Named Rules
**The Flat Paper Rule.** Resting surfaces have no shadow. Separate surfaces with a border or a rule, never a lift.

## Shapes

Every corner is square (0px): buttons, inputs, chips, tiles, notices, map pins, and status stamps. Borders follow a two-weight system: 2px ink for structure and commitment (rail edge, view and side-head underlines, the Do/Avoid top rule, buttons, alarm and current-item emphasis), and 1px for containment (ink around data tiles and forms, rule colour for hairlines between cells and around quiet blocks). The leaf's top edge is a perforation: a repeating row of 3px dots in rule colour, 12px apart. Direction on the water path is drawn as a square rotated 45° and bordered on two sides, which makes a printed chevron.

## Components

### Buttons
Plain printed controls with heavy borders.
- **Shape:** square (0px), 2px ink border, 600 weight at 15px, 11px 16px padding.
- **Primary:** ink fill with white text; one per control group (Irrigate now, Start watering, Confirm). Hover lifts slightly to charcoal (#3b3c3d, an untokenized value in the build).
- **Secondary:** paper fill with ink text; hover fills with sheet. Transitions are 150ms ease-out on background and colour.
- **Active (toggled):** red-tint fill, red border, red text (for example, a paused or skipped state).
- **Ghost:** transparent border, underlined text, 8px inline padding; used for Cancel and Keep plan.
- **Disabled:** 45% opacity, not-allowed cursor.
- **Focus:** 2px red outline, 2px offset (global).

### Chips
- **Style:** 1px ink border, paper fill, 14px text, 6px 12px padding, square.
- **State:** the selected chip inverts to ink fill with white text. Used for the alert filter.

### Cards / Containers
- **Corner Style:** square.
- **Background:** paper on sheet.
- **Shadow Strategy:** none (see Flat Paper).
- **Border:** data tiles and forms use 1px ink; blocks, roster items, and event lists use 1px rule; the current or far item is promoted to 2px red.
- **Internal Padding:** 14–16px for tiles, 18px 20px for blocks.

### Inputs / Fields
- **Style:** 1px ink border, paper fill, square, 7px 8px padding, inherited type. Checkboxes use ink as the accent colour.
- **Focus:** the global 2px red outline.
- **Error:** feedback text turns red below the controls.

### Navigation
- **Style:** a vertical list in the rail, 16px, separated by 1px rule hairlines. Hover and the current page turn red; the current page is also 700 weight. The alert count is a square red badge with white 12px bold text. Under 980px the nav goes inline with no hairlines.

### The Leaf (signature)
A full-height paper column with a perforated top edge. From top to bottom: a red uppercase month/weekday header over a 2px red rule; the red display numeral; a bold one-line summary; a two-column Do/Avoid grid under a 2px ink rule, split by a 1px rule; the "why" line in ink-2 under a hairline; and the watering plan pinned to the bottom with a 10px ink-bordered blue progress bar. When the day changes, the old leaf tears off: it rotates from its top-left corner and falls (1.05s, cubic-bezier(.55, .05, .7, .2)). With reduced motion it fades out over 300ms instead.

### Readings Grid
A 1px ink-bordered paper grid of 3×2 cells separated by rule hairlines. Each cell has an ink-2 label, a condensed numeral, and an ink-3 unit line. Healthy units turn green; a bad reading turns the numeral and unit red; stale readings grey to ink-3.

### Notices
A 2px amber border on amber-tint with amber text, with a bold ink lead-in. The alarm variant swaps to red on red-tint.

### Status Stamps
Small 1px-bordered square tags (12px bold, 2px 7px padding) on water-path stages: neutral (rule/ink-2), on (green), regenerating (amber), warning (red). The Simulated unit tag is the same form in uppercase ink.

### Level Bars
Inline 9–10px tall bars with a 1px ink border and a solid fill: blue for water, amber for brine, red when below 15%.

### Map
Leaflet tiles are filtered to greyscale on a sheet background. Site pins are 18px paper squares with a 3px ink border; the current site is solid red. The clicked-orchard pin is a 12px blue square.

## Do's and Don'ts

### Do:
- **Do** keep today's decision on one full-height leaf with the red numeral, Do in green, and Avoid in red.
- **Do** use 2px ink rules for structure and 1px rule hairlines inside tiles and lists.
- **Do** keep every corner square (0px), including pins and stamps.
- **Do** set figures in condensed black Archivo with tabular numerals.
- **Do** give each signal colour exactly one meaning: red date/avoid/alarm, green do/healthy, amber warning, blue water.
- **Do** put tints only behind text and borders in their own colour.
- **Do** offer a 300ms fade instead of the tear for reduced motion.

### Don't:
- **Don't** break the decision into KPI tiles in a dashboard grid; the leaf is the decision.
- **Don't** add resting shadows, gradient surface fills, or rounded corners to tiles, buttons, or inputs. The perforation dots are a pattern, not a fill, and stay.
- **Don't** use red, green, amber, or blue as decoration or brand colour.
- **Don't** condense running text; condensed widths are for numerals and big headings only.

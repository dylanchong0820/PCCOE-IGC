# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Delegated: static HTML/CSS/JS in `web/`, no build step, for the first version. It reads static pipeline output (`ml/sedoss_outputs/`, including the per-site `index.json`) and hosts anywhere (e.g. GitHub Pages). Live daily predictions and unit control will need a small backend (e.g. a Python API wrapping `predict_today.py`); that choice is open. Deploy target not yet decided.

## Users

Two primary audiences, one website:

- **Durian farmers / orchard operators** (e.g. Musang King growers) who own or run a S.E.D.O.S.S. unit. They use the website as the product's hub: pick their orchard's location on a map, see the daily irrigation and fertigation recommendation for that location, and control their unit.
- **PCCOE International Grand Challenge 2026 judges** (theme: AI for Climate Change; track: AgriTech and Water Resilience). They are evaluating many entries and need to see quickly what problem S.E.D.O.S.S. solves, how it works, and whether the evidence holds up. For judges the hub itself is the demonstration that this is a product.

## Product Purpose

S.E.D.O.S.S. (Smart E-Desalination & On-Site Salt Segregation) is an off-grid, containerized system that responds to climate-driven salinity intrusion in irrigation water, which threatens high-value perennial crops such as Musang King durian. It:

1. **Protects** crops by turning brackish water into crop-safe irrigation water with selective Faradaic deionization (FDI) at low voltage.
2. **Recovers** the removed salts on site through zero liquid discharge (ZLD) evaporation and fractional crystallization.
3. **Optimizes** water and nutrient delivery with an AI decision engine that forecasts crop water demand and doses fertilizer through a Venturi injector.

The website is the product's front end, not only a project showcase. A farmer chooses their orchard location on a map. The map offers only locations the ML models have been trained on (NASA POWER weather per site), and the system uses that site's models to predict. The website shows the prediction and the recommended action, and acts as the control hub for the farmer's unit.

Success: a farmer can find their site, understand today's recommendation, and act on it. A judge leaves understanding the problem, the three-part system, the AI loop, and the measured results with their limitations.

## Positioning

Desalination, salt recovery and AI-driven irrigation in one off-grid unit: treated water is matched to forecast crop demand, and the removed salt becomes a recovered product rather than brine waste.

## Operating Context

- The AI "brain" follows a Sense → Predict → Decide → Act → Verify → Adapt loop.
- Weather comes from NASA POWER daily data (2015–2024, 3,653 days); default site lat 1.85, lon 102.93 (Johor, Malaysia).
- `predict_today.py` is designed to run once a day (e.g. 05:00) on a Raspberry Pi / PLC, taking sensor readings plus weather and returning the irrigation/fertigation recipe.
- Judges likely also see the project slide deck; the site complements it.

## Capabilities and Constraints

- Stage 1: XGBoost forecasts tomorrow's ET₀ (FAO-56 Penman-Monteith labels).
- Stage 2: XGBoost forecasts next-day soil water change and a 24-hour drying curve, on **simulated** soil.
- Decide stage is rule-based (irrigation volume, timing, Venturi dose, rain delay, heatwave mitigation), not ML.
- Adapt stage is a PID loop on a **simulated** plant holding outlet EC on target.
- **Multiple trained sites.** Models are trained per site; each site has its own config in `ml/config/sites/` and its own models and metrics under `ml/sedoss_outputs/sites/<id>/`. `ml/sedoss_outputs/sites/index.json` lists the trained sites (name, coordinates, metrics); the website's map reads it. A location the models were not trained on is mapped to the nearest trained site, with its distance shown.
- NASA POWER weather is on a coarse grid (about 0.5° × 0.625°, roughly 50–70 km), so trained sites should be spread out. Two orchards inside the same grid cell get the same weather.
- The site shows **real pipeline outputs** (results.json, figures, test predictions, per-site metrics), not invented demo values. When the pipeline reruns, the site should reflect the regenerated files.
- **Prototype hardware** follows the team's parts budget (`SEDOSS_Small_Scale_Budget_MYR_USD.xlsx`, Stages 0–8): Raspberry Pi 5 (AI brain), two ESP32s (treatment node; delivery/dosing node), three EC sensors (feed, outlet, final), pH and water-temperature sensors, float switches, flow meters, a 3-way solenoid valve, feed and delivery pumps, A/B peristaltic dosing pumps, air temperature/humidity and solar-radiation sensors, a hydroponic/growing-medium test bed, and brine evaporation trays. Nutrients are dosed by the **peristaltic pumps, not a Venturi**; the ML code's "Venturi dose factor" maps to the dosing target. A hobby capacitive soil-moisture probe was added (not yet in the budget).
- **Control hub.** Farmers control their unit from the website. The virtual Pi (`ml/run_bench.py --serve`) exposes the API the website uses: live state, event log, daily log, hardware list, and commands (pause, resume, skip today, water only today, irrigate now, regenerate FDI, refill feed/nutrients). The real Pi should serve the same API. Accounts/authentication are not yet designed.
- **Safety interlock:** the Pi caps the dosing EC target at the crop-safe limit (0.5 ppt ≈ 0.78 dS/m). With the current config the AI's full dose always exceeds it, so the fertilizer EC value needs review with the team.
- Terminology: ET₀ (reference evapotranspiration), EC (electrical conductivity, dS/m), FDI, ZLD, Kc, TAW.
- Undecided: deploy target; hosting needs once live predictions and control are added (a static site cannot send commands); whether hardware photos or renders exist; which orchards are the final trained sites (current list is a starting set to confirm).

## Brand Commitments

- Name: **S.E.D.O.S.S.** — Smart E-Desalination & On-Site Salt Segregation.
- Event framing: PCCOE International Grand Challenge 2026 · AI for Climate Change · AgriTech and Water Resilience.
- Team (from README): Hansen Lee Ming Haw (Leader, Universiti Teknologi PETRONAS), Lum Jia Ying (Universiti Teknologi PETRONAS), Jordan Voo Yee Fung (Universiti Malaysia Sabah), Ivan Liew Chun Fui (Universiti Malaya), Chong Jia Ying Dylan (University of Nottingham Malaysia).
- No logo or visual identity exists in the repository.

## Evidence on Hand

- `ml/sedoss_outputs/results.json`: all metrics, assumptions and example decisions.
- `ml/sedoss_outputs/et0_test_predictions.csv`: actual vs predicted ET₀ for every test day (2023–2024).
- Figures: `fig2_feature_importance.png`, `fig4_pid_response.png` committed; `fig1_et0_forecast.png` and `fig3_soil_drying.png` are produced by the pipeline but not committed.
- Headline results (test 2023–2024):
  - ET₀ forecast: XGBoost RMSE 0.701 mm/day vs 0.893 persistence (~21% lower error); R² 0.123.
  - Soil water change (simulated): XGBoost R² 0.826, RMSE 1.391 mm/day.
  - PID (simulated): ~12% overshoot, 39 s settle, 18 s recovery after FDI drift, 34 s settle after heatwave setpoint change (±0.04 dS/m).
- Limitations documented in README (simulated soil and plant, placeholder crop/soil parameters, equation-based ET₀ labels).
- Absent and must not be fabricated: field trial results, real-sensor validation, hardware performance data (FDI removal rates, energy use, salt yield), costs, customers, testimonials, partners.

## Product Principles

1. **Today's action first, for the farmer.** Irrigate or not, how much, when, and the fertilizer dose come before charts and metrics.
2. **Honest evidence over impressive claims.** Show real metrics next to their baselines, and state what is simulated or assumed in the same place as the number, including how far the farmer's orchard is from the trained site.
3. **Only offer what was trained.** The map and predictions are limited to sites with trained models; never imply a prediction exists for an untrained location.
4. **One system, three jobs.** Present protect, recover and optimize as one connected unit, and make the Sense → Adapt loop legible, including which stages are ML and which are rules or control.
5. **Fast to evaluate.** Judges are time-poor; the problem, the product and the results should be reachable in a minute.

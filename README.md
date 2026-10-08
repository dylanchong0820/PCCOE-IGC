# PCCOE-IGC
# S.E.D.O.S.S. — Smart E-Desalination & On-Site Salt Segregation

**PCCOE International Grand Challenge 2026** · Theme: AI for Climate Change · Track: AgriTech and Water Resilience

Climate-driven salinity intrusion is turning freshwater irrigation sources brackish and threatening high-value perennial crops such as Musang King durian. S.E.D.O.S.S. is an off-grid, containerized system that:

1. **Protects** crops by converting brackish water into crop-safe irrigation water using selective Faradaic deionization (FDI) at low voltage.
2. **Recovers** the removed salts on site through zero liquid discharge (ZLD) evaporation and fractional crystallization.
3. **Optimizes** water and nutrient delivery with an AI decision engine that forecasts crop water demand and doses fertilizer through a Venturi injector.

## Repository structure

| Folder | Contents |
|---|---|
| `ml/` | Machine-learning pipeline: ET₀ forecasting, soil-drying forecasting, rule-based decisions, PID simulation |
| `web/` | [Describe the website, e.g. "Project website / interactive dashboard"] |

---

## 1. ML pipeline (`ml/`)

The pipeline follows the "AI Brain" loop from our slides: **Sense → Predict → Decide → Act → Verify → Adapt**.

| Stage | What it does | Method |
|---|---|---|
| Sense | Daily weather: temperature, humidity, wind, solar radiation, rainfall | NASA POWER (2015–2024, 3,653 days) |
| Label | Reference evapotranspiration (ET₀) | FAO-56 Penman-Monteith equation |
| Predict (model 1) | Forecast **tomorrow's ET₀** | XGBoost |
| Predict (model 2) | Forecast **next-day soil water change** and a 24-hour drying curve | XGBoost |
| Decide | Irrigation volume, timing and Venturi fertilizer dose (rain delay, heatwave mitigation) | Rule-based logic (not ML) |
| Adapt | Hold outlet EC on target after disturbances | PID loop on a simulated plant (not ML) |

### Run it

```bash
cd ml
pip install -r requirements.txt
python3 sedoss_ml.py                      # downloads NASA POWER data
python3 sedoss_ml.py --lat 2.05 --lon 102.57   # different site
python3 sedoss_ml.py --offline            # synthetic weather, pipeline test only
```

On macOS, XGBoost needs the OpenMP runtime: `brew install libomp`.

Outputs are written to `ml/sedoss_outputs/`:

| File | Description |
|---|---|
| `fig1_et0_forecast.png` | Predicted vs actual ET₀ on the test set |
| `fig2_feature_importance.png` | Which inputs the ET₀ model relies on |
| `fig3_soil_drying.png` | Soil-drying forecast and 24-hour curve (simulated soil) |
| `fig4_pid_response.png` | EC control response to disturbances |
| `results.json` | All metrics, assumptions and example decisions |
| `et0_test_predictions.csv` | Actual vs predicted ET₀ for every test day |
| `models/` | Trained XGBoost models |

### Results

Data is split by date (no shuffling): model 1 trains on 2015-01 to 2021-01, model 2 on 2021-01 to 2022-12, and everything is tested on 2023-01 to 2024-12.

**Stage 1: forecasting tomorrow's ET₀ (test set)**

| Model | R² | RMSE (mm/day) | MAE (mm/day) |
|---|---|---|---|
| **XGBoost** | 0.123 | **0.701** | 0.573 |
| Ridge (linear baseline) | 0.033 | 0.736 | 0.580 |
| Persistence (tomorrow = today) | −0.423 | 0.893 | 0.669 |

XGBoost reduces forecast error by about 21% compared with the "tomorrow = today" baseline (mean ET₀ over the period is 3.62 mm/day). The R² is low because tomorrow's ET₀ depends on cloud cover and rainfall that today's observations cannot see.

**Stage 2: next-day soil water change (simulated soil, test set)**

| Model | R² | RMSE (mm/day) | MAE (mm/day) |
|---|---|---|---|
| **XGBoost** | 0.826 | **1.391** | 0.766 |
| Ridge (linear baseline) | 0.067 | 3.226 | 1.999 |
| Persistence (yesterday's change) | −0.789 | 4.467 | 2.347 |

**PID loop (simulated plant):** about 12% overshoot on the first setpoint step, 39 s to settle, 18 s to recover after FDI output drift, and 34 s to settle after the heatwave setpoint change (±0.04 dS/m band).

### Limitations

- **Soil moisture is simulated** by a water-balance model, so the Stage 2 score shows the pipeline works, not that it works on real soil. Real-sensor validation is future work.
- ET₀ labels come from the Penman-Monteith equation, not from lysimeter measurements.
- Crop and soil parameters (Kc = 0.9, 100 mm root-zone water capacity, 50% depletion threshold, 10 mm rain-skip threshold, 34 °C heatwave trigger) are placeholder assumptions and are listed in `results.json`.
- Model 1 uses only observed weather up to today. Adding forecast inputs from a weather API is the planned next step.
- The PID gains are tuned for a simulated plant and would need retuning on hardware.

---

## 2. Website (`web/`)

[Describe what the website shows.]

**Run locally**

```bash
cd web
[install command, e.g. npm install]
[start command, e.g. npm run dev]
```

**Built with:** [framework / tools]

**How it uses the ML results:** [e.g. "displays the figures and results.json from ml/sedoss_outputs" or "demo values only, not live model output"]

---

## Team

| Name | Institution |
|---|---|
| Hansen Lee Ming Haw (Leader) | Universiti Teknologi PETRONAS |
| Lum Jia Ying | Universiti Teknologi PETRONAS |
| Jordan Voo Yee Fung | Universiti Malaysia Sabah |
| Ivan Liew Chun Fui | Universiti Malaya |
| Chong Jia Ying Dylan | University of Nottingham Malaysia |

## Data and references

- NASA POWER daily meteorological data: https://power.larc.nasa.gov/
- Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop evapotranspiration: Guidelines for computing crop water requirements.* FAO Irrigation and Drainage Paper No. 56.
- Full reference list: see the project slide deck.

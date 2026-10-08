#!/usr/bin/env python3
"""
S.E.D.O.S.S. - "what should the system do today?"
==================================================
Takes today's SENSOR READINGS + a short weather history + today's weather FORECAST and
returns the irrigation / fertigation recipe, using the models trained by sedoss_ml.py.

This is the piece that runs on the Raspberry Pi / PLC once a day (e.g. 05:00).
For now the inputs are two small files; later the sensor-reading code can write the same
JSON, or import and call `recommend()` directly.

Usage
-----
  python3 predict_today.py \
      --models sedoss_outputs/models \
      --history examples/recent_weather_example.csv \
      --readings examples/readings_example.json

INPUT 1 - history CSV: observed DAILY weather for the last >= 10 days, ending YESTERDAY
  columns: date,tmax,tmin,rh,u2,rs,rain
  units  : degC, degC, % relative humidity, m/s wind at 2 m, MJ/m2/day solar radiation, mm rain

INPUT 2 - readings JSON (today, morning):
  {
    "date": "2026-10-08",                 # today
    "soil_water_mm": 62.0,                # root-zone available water (mm)  -- OR --
    "soil_vwc": 0.24,                     # volumetric water content from the probe (0-1)
    "outlet_ec_dS_m": 0.42,               # optional: measured EC of treated water (overrides config)
    "forecast_today": {"tmax": 33.5, "tmin": 24.0, "rh": 82, "u2": 1.2, "rs": 17.5, "rain": 3.0},
    "fertigation_due": true               # optional (default true)
  }
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sedoss_ml import (FC_COLS, SOIL_COLS, Params, decide, et0_pm, fc_next_features,
                       obs_features, vwc_to_mm)

REQUIRED = ["tmax", "tmin", "rh", "u2", "rs", "rain"]


def load_models(model_dir: Path):
    meta = json.loads((model_dir / "meta.json").read_text())
    m_fc, m2 = xgb.XGBRegressor(), xgb.XGBRegressor()
    m_fc.load_model(str(model_dir / "et0_forecast_xgb.json"))
    m2.load_model(str(model_dir / "soil_drying_xgb.json"))
    return meta, m_fc, m2


def recommend(history: pd.DataFrame, readings: dict, meta: dict, m_fc, m2):
    P = Params(**meta["params"])
    today = pd.to_datetime(readings["date"])

    h = history.copy()
    missing = [c for c in REQUIRED if c not in h.columns]
    if missing:
        raise ValueError(f"history is missing columns: {missing}")
    h = h.sort_index()
    if len(h) < 10:
        raise ValueError("need at least 10 days of history (lags and 7-day averages)")
    if h.index[-1] != today - pd.Timedelta(days=1):
        print(f"[warn] history ends {h.index[-1].date()}, expected {(today - pd.Timedelta(days=1)).date()} (yesterday)")
    h["tmean"] = (h["tmax"] + h["tmin"]) / 2
    h["doy"] = h.index.dayofyear
    h["et0"] = et0_pm(h["tmean"], h["tmax"], h["tmin"], h["rh"], h["u2"], h["rs"], h["doy"], P.lat, P.elev)

    # ---- weather forecast for today (what the API gives) ----------------------
    f = readings["forecast_today"]
    fc = pd.DataFrame(np.nan, index=h.index.append(pd.DatetimeIndex([today])), columns=FC_COLS)
    for c in FC_COLS:
        fc.loc[today, c] = float(f[c])

    # ---- model 1: today's crop water demand (ET0) -----------------------------
    X = obs_features(h).join(fc_next_features(fc))
    row = X.iloc[[-1]][meta["et0_forecast_cols"]]
    if row.isna().any(axis=None):
        raise ValueError("could not build features (NaN) - check the history file")
    et0_hat = float(m_fc.predict(row)[0])

    # ---- soil sensor --------------------------------------------------------------
    if "soil_water_mm" in readings:
        theta = float(readings["soil_water_mm"])
    elif "soil_vwc" in readings:
        theta = vwc_to_mm(float(readings["soil_vwc"]), P)
    else:
        raise ValueError("readings need soil_water_mm or soil_vwc")

    # ---- model 2: how much will the soil dry today? -------------------------------
    rain_hist = h["rain"].values
    dry_days = 0
    for r in rain_hist[::-1]:
        if r > 1:
            break
        dry_days += 1
    srow = {"theta": theta, "et0_hat": et0_hat, "rain_fc": float(f["rain"]), "tmax_fc": float(f["tmax"]),
            "rh_prev": float(h["rh"].iloc[-1]), "rs_prev": float(h["rs"].iloc[-1]), "dry_days": dry_days,
            "doy_sin": np.sin(2 * np.pi * today.dayofyear / 365), "doy_cos": np.cos(2 * np.pi * today.dayofyear / 365)}
    delta = float(m2.predict(pd.DataFrame([srow], columns=meta["soil_cols"]))[0])

    # ---- rules ------------------------------------------------------------------------
    if "outlet_ec_dS_m" in readings:
        P = dataclasses.replace(P, base_ec=float(readings["outlet_ec_dS_m"]))
    recipe = decide(theta, delta, float(f["rain"]), float(f["tmax"]), P,
                    fert_due=bool(readings.get("fertigation_due", True)))
    return {"date": str(today.date()),
            "sensor_and_forecast_inputs": {"soil_water_mm": round(theta, 1), "rain_forecast_mm": f["rain"],
                                           "tmax_forecast_C": f["tmax"], "outlet_ec_dS_m_used": P.base_ec},
            "model_outputs": {"forecast_et0_mm_per_day": round(et0_hat, 2),
                              "predicted_soil_water_change_mm": round(delta, 2)},
            "recipe": recipe}


def main():
    ap = argparse.ArgumentParser(description="S.E.D.O.S.S. daily recommendation")
    ap.add_argument("--models", default="sedoss_outputs/models")
    ap.add_argument("--history", required=True)
    ap.add_argument("--readings", required=True)
    ap.add_argument("--json", action="store_true", help="print JSON only")
    args = ap.parse_args()

    meta, m_fc, m2 = load_models(Path(args.models))
    history = pd.read_csv(args.history, parse_dates=["date"]).set_index("date")
    readings = json.loads(Path(args.readings).read_text())
    out = recommend(history, readings, meta, m_fc, m2)

    if args.json:
        print(json.dumps(out, indent=2))
        return
    r, mo, si = out["recipe"], out["model_outputs"], out["sensor_and_forecast_inputs"]
    print(f"\nS.E.D.O.S.S. recommendation for {out['date']}")
    print(f"  Soil water now           : {si['soil_water_mm']} mm")
    print(f"  Forecast rain / Tmax     : {si['rain_forecast_mm']} mm / {si['tmax_forecast_C']} C")
    print(f"  Model: crop water demand : {mo['forecast_et0_mm_per_day']} mm/day (ET0)")
    print(f"  Model: soil water change : {mo['predicted_soil_water_change_mm']} mm over the next day")
    print(f"  DECISION                 : {r['reason']}")
    if r["irrigate"]:
        print(f"    irrigate {r['volume_mm']} mm in the {r['timing']}")
        print(f"    Venturi dose factor {r['venturi_dose_factor']}  ->  EC setpoint {r['ec_setpoint_dS_m']} dS/m "
              f"(~{r['final_ec_ppt_est']} ppt)")
        if not r["within_crop_ec_limit"]:
            print("    WARNING: this EC exceeds the crop-safe limit set in the config")
    print()


if __name__ == "__main__":
    main()

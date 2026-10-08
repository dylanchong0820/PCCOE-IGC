#!/usr/bin/env python3
"""
S.E.D.O.S.S. "AI Brain" - training, experiments and demo (config-driven)
=========================================================================
Loop from the slides:  Sense -> Predict -> Decide -> Act -> Verify -> Adapt

  Sense   : daily weather (NASA POWER) + soil / EC sensor readings (see predict_today.py)
  Predict : XGBoost forecasts tomorrow's crop water demand (ET0) and next-day soil drying
  Decide  : rule engine -> irrigation volume, fertiliser dose, timing   (NOT machine learning)
  Adapt   : PID loop holds the delivered EC on target                    (NOT machine learning)

Everything site-specific lives in a YAML config (see config/site_example.yaml).

Usage
-----
  pip install -r requirements.txt
  python3 sedoss_ml.py                                   # default config, NASA POWER data
  python3 sedoss_ml.py --config config/site_example.yaml
  python3 sedoss_ml.py --offline                         # synthetic weather (pipeline test only)
  python3 sedoss_ml.py --quick                           # skip sweeps and extra seeds (fast)

HONESTY NOTES
-------------
  * ET0 labels come from FAO-56 Penman-Monteith, not from a lysimeter.
  * Soil moisture is SIMULATED (water-balance bucket). Stage-2 and policy-comparison results
    show the pipeline and logic work; they are NOT field results.
  * Weather-forecast inputs are SIMULATED (actual weather + configurable error) because no
    historical forecast archive is used here. Real forecast data is the next step.
  * If the script prints "SYNTHETIC DATA", do not quote any numbers.
"""

import argparse
import copy
import dataclasses
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 0. CONFIG
# =============================================================================
DEFAULT_CONFIG = {
    "site": {"name": "Example durian orchard (PLACEHOLDER site)",
             "latitude": 1.85, "longitude": 102.93, "elevation_m": 50.0},
    "data": {"start": "20150101", "end": "20241231"},
    "soil_crop": {"taw_mm": 100.0, "p_depletion": 0.5, "kc": 0.9, "rain_eff": 0.8,
                  "vwc_field_capacity": 0.30, "vwc_wilting_point": 0.15},
    "water_quality": {"base_ec_dS_m": 0.4, "ec_to_ppt": 0.64, "crop_safe_limit_ppt": 0.5},
    "fertigation": {"fert_ec_increase_dS_m": 0.8, "fertigate_every_n_irrigations": 1},
    "irrigation": {"max_irrig_mm": 15.0, "fixed_timer_every_n_days": 2,
                   "fixed_timer_amount_mm": 7.0},
    "policy": {"rain_skip_mm": 10.0, "heat_tmax_c": "auto", "heat_percentile": 0.90},
    "forecast_error": {"tmax_sd_c": 1.0, "tmin_sd_c": 0.8, "rh_sd_pct": 5.0,
                       "u2_rel_sd": 0.25, "rs_rel_sd": 0.15,
                       "rain_lognorm_sigma": 0.3, "rain_false_alarm_prob": 0.05},
    "simulator": {"initial_fraction": 0.7, "irrigation_trigger_range": [0.25, 0.8],
                  "irrigation_amount_range_mm": [8.0, 20.0], "sensor_noise_mm": 1.0},
    "pid": {"plant_gain_dS_m": 2.0, "tau_s": 15.0, "delay_s": 5, "kp": 0.8, "ki": 0.04,
            "kd": 0.0, "fdi_drift_dS_m": 0.2},
    "experiments": {"extra_seeds": [1, 2, 3, 4], "driest_window_days": 90,
                    "sensitivity_world": {"kc": [0.7, 0.8, 0.9, 1.0, 1.1],
                                          "taw_mm": [60, 80, 100, 120, 140],
                                          "rain_eff": [0.6, 0.8, 1.0]},
                    "sensitivity_policy": {"rain_skip_mm": [5, 10, 20],
                                           "heat_percentile": [0.8, 0.9, 0.95]},
                    "forecast_error_scales": [0.0, 0.5, 1.0, 2.0, 4.0]},
}


def deep_update(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        import yaml
        with open(path) as f:
            deep_update(cfg, yaml.safe_load(f) or {})
    return cfg


@dataclass(frozen=True)
class Params:
    """Everything the soil model and the rule engine need (flat, easy to vary in sweeps)."""
    lat: float
    lon: float
    elev: float
    taw: float
    p_dep: float
    kc: float
    rain_eff: float
    vwc_fc: float
    vwc_wp: float
    base_ec: float
    ec_to_ppt: float
    ec_limit_ppt: float
    fert_inc: float
    fert_every_n: int
    max_irrig: float
    rain_skip: float
    heat_tmax: float

    @property
    def raw_thr(self):          # soil water (mm) below which the crop is stressed
        return (1 - self.p_dep) * self.taw


def make_params(cfg, heat_tmax):
    s, c, w = cfg["site"], cfg["soil_crop"], cfg["water_quality"]
    return Params(
        lat=s["latitude"], lon=s["longitude"], elev=s["elevation_m"],
        taw=c["taw_mm"], p_dep=c["p_depletion"], kc=c["kc"], rain_eff=c["rain_eff"],
        vwc_fc=c["vwc_field_capacity"], vwc_wp=c["vwc_wilting_point"],
        base_ec=w["base_ec_dS_m"], ec_to_ppt=w["ec_to_ppt"], ec_limit_ppt=w["crop_safe_limit_ppt"],
        fert_inc=cfg["fertigation"]["fert_ec_increase_dS_m"],
        fert_every_n=int(cfg["fertigation"]["fertigate_every_n_irrigations"]),
        max_irrig=cfg["irrigation"]["max_irrig_mm"], rain_skip=cfg["policy"]["rain_skip_mm"],
        heat_tmax=float(heat_tmax))


def validate_config(P, cfg):
    warns = []
    final_ec = P.base_ec + P.fert_inc
    final_ppt = final_ec * P.ec_to_ppt
    if final_ppt > P.ec_limit_ppt:
        warns.append(f"EC SAFETY: full-dose EC {final_ec:.2f} dS/m = ~{final_ppt:.2f} ppt exceeds the "
                     f"crop-safe limit {P.ec_limit_ppt} ppt. Lower fert_ec_increase_dS_m or base EC, "
                     "or justify why fertiliser ions are less harmful than NaCl (with a source).")
    if P.fert_inc > cfg["pid"]["plant_gain_dS_m"]:
        warns.append("PID: the EC increase asked for is larger than the Venturi valve can deliver "
                     "(plant_gain_dS_m). The PID cannot reach the setpoint.")
    for w in warns:
        print(f"[CONFIG WARNING] {w}")
    return warns


# =============================================================================
# 1. DATA
# =============================================================================
def fetch_nasa_power(lat, lon, start, end, cache: Path) -> pd.DataFrame:
    if cache.exists():
        print(f"[data] using cached download: {cache}")
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    import requests
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point"
           "?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR"
           f"&community=AG&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON")
    print("[data] downloading NASA POWER ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    df = pd.DataFrame(p)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.rename(columns={"T2M": "tmean", "T2M_MAX": "tmax", "T2M_MIN": "tmin", "RH2M": "rh",
                            "WS2M": "u2", "ALLSKY_SFC_SW_DWN": "rs", "PRECTOTCORR": "rain"})
    df = df.replace(-999, np.nan).interpolate(limit=3).dropna()
    if df["rs"].median() < 10:   # kWh/m2/day -> MJ/m2/day
        print("[data] Rs looks like kWh/m2/day -> converting to MJ/m2/day (x3.6)")
        df["rs"] = df["rs"] * 3.6
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def synthetic_weather(start, end, seed) -> pd.DataFrame:
    """Plausible tropical weather. ONLY for testing the pipeline offline."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.to_datetime(start, format="%Y%m%d"),
                        pd.to_datetime(end, format="%Y%m%d"), freq="D")
    n = len(idx)
    doy = idx.dayofyear.values
    season = 1 + 0.25 * np.cos(2 * np.pi * (doy - 340) / 365)
    wet = np.zeros(n, dtype=bool)
    for i in range(1, n):
        p = (0.65 if wet[i - 1] else 0.35) * season[i]
        wet[i] = rng.random() < min(p, 0.95)
    rain = np.where(wet, rng.gamma(0.8, 14.0, n), 0.0)
    rs = np.clip(23 - 9 * wet + 1.5 * np.sin(2 * np.pi * (doy - 80) / 365) + rng.normal(0, 1.8, n), 5, 28)
    tmax = 33 + 0.15 * (rs - 18) - 2.0 * wet + rng.normal(0, 0.8, n)
    tmin = 23.5 + 0.5 * wet + rng.normal(0, 0.7, n)
    rh = np.clip(80 - 2 * (tmax - 32) + 8 * wet + rng.normal(0, 4, n), 50, 100)
    u2 = np.clip(rng.gamma(3, 0.4, n), 0.3, 6)
    return pd.DataFrame({"tmean": (tmax + tmin) / 2, "tmax": tmax, "tmin": tmin, "rh": rh,
                         "u2": u2, "rs": rs, "rain": rain}, index=idx)


def get_weather(args, cfg, out: Path):
    s, d = cfg["site"], cfg["data"]
    if not args.offline:
        try:
            cache = out / "cache" / f"nasa_{s['latitude']}_{s['longitude']}_{d['start']}_{d['end']}.csv"
            return fetch_nasa_power(s["latitude"], s["longitude"], d["start"], d["end"], cache), False
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] NASA POWER download failed ({e}).")
    print("[WARN] *** USING SYNTHETIC DATA - metrics are for pipeline testing only ***")
    return synthetic_weather(d["start"], d["end"], args.seed), True


# =============================================================================
# 2. PHYSICS: ET0 (FAO-56 Penman-Monteith) and Hargreaves baseline
# =============================================================================
def extraterrestrial_radiation(doy, lat_deg):
    """Ra in MJ/m2/day (FAO-56 eq. 21)."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))


def et0_pm(tmean, tmax, tmin, rh, u2, rs, doy, lat_deg, elev):
    """FAO-56 Penman-Monteith reference evapotranspiration (mm/day)."""
    P = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26
    gamma = 0.000665 * P

    def e(t):
        return 0.6108 * np.exp(17.27 * t / (t + 237.3))

    es = (e(tmax) + e(tmin)) / 2
    ea = es * rh / 100
    delta = 4098 * e(tmean) / (tmean + 237.3) ** 2
    ra = extraterrestrial_radiation(doy, lat_deg)
    rso = (0.75 + 2e-5 * elev) * ra
    rnl = (4.903e-9 * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2)
           * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * np.minimum(rs / rso, 1) - 0.35))
    rn = 0.77 * rs - rnl
    et0 = (0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (es - ea)) \
        / (delta + gamma * (1 + 0.34 * u2))
    return np.clip(et0, 0, None)


def hargreaves(tmax, tmin, doy, lat_deg):
    """Classic temperature-only ET0 estimate (uncalibrated baseline)."""
    ra = extraterrestrial_radiation(doy, lat_deg)
    tmean = (tmax + tmin) / 2
    return 0.0023 * (tmean + 17.8) * np.sqrt(np.clip(tmax - tmin, 0, None)) * 0.408 * ra


def reg_metrics(y, p):
    return {"R2": float(r2_score(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p))}


# =============================================================================
# 3. FEATURES AND SIMULATED FORECAST
# =============================================================================
OBS_COLS = ["tmax", "tmin", "rh", "u2", "rs", "rain"]
FC_COLS = ["tmax", "tmin", "rh", "u2", "rs", "rain"]
SOIL_COLS = ["theta", "et0_hat", "rain_fc", "tmax_fc", "rh_prev", "rs_prev",
             "dry_days", "doy_sin", "doy_cos"]


def obs_features(df):
    """Features known at the end of day t from OBSERVATIONS (df needs OBS_COLS, et0, doy)."""
    f = pd.DataFrame(index=df.index)
    for c in OBS_COLS + ["et0"]:
        f[c] = df[c]
        for k in (1, 2, 3):
            f[f"{c}_l{k}"] = df[c].shift(k)
    f["et0_7d"] = df["et0"].rolling(7).mean()
    f["rain_7d"] = df["rain"].rolling(7).sum()
    f["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    f["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
    return f


def fc_next_features(fc):
    """Forecast (made on day t) for day t+1. fc is indexed by the day being forecast."""
    nxt = fc[FC_COLS].shift(-1)
    nxt.columns = [f"fc_{c}" for c in FC_COLS]
    return nxt


def make_noise(n, rng):
    z = {k: rng.standard_normal(n) for k in ["tmax", "tmin", "rh", "u2", "rs", "rain"]}
    z["fa_u"] = rng.random(n)
    z["fa_g"] = rng.gamma(1, 3, n)
    return z


def apply_forecast_error(df, z, fe, scale=1.0):
    """SIMULATED weather forecast = actual weather + error. scale=0 -> perfect forecast."""
    fc = pd.DataFrame(index=df.index)
    fc["tmax"] = df["tmax"] + z["tmax"] * fe["tmax_sd_c"] * scale
    fc["tmin"] = df["tmin"] + z["tmin"] * fe["tmin_sd_c"] * scale
    fc["rh"] = np.clip(df["rh"] + z["rh"] * fe["rh_sd_pct"] * scale, 10, 100)
    fc["u2"] = np.clip(df["u2"] * np.exp(z["u2"] * fe["u2_rel_sd"] * scale), 0.2, 10)
    fc["rs"] = np.clip(df["rs"] * (1 + z["rs"] * fe["rs_rel_sd"] * scale), 1, None)
    false_alarm = (z["fa_u"] < min(1.0, fe["rain_false_alarm_prob"] * scale)) * z["fa_g"]
    fc["rain"] = df["rain"] * np.exp(z["rain"] * fe["rain_lognorm_sigma"] * scale) + false_alarm
    return fc


def mk_xgb(seed, n=500, lr=0.03, depth=4):
    return xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=3, random_state=seed, n_jobs=-1)


# =============================================================================
# 4. SOIL WATER-BALANCE SIMULATOR (SIMULATED soil)
# =============================================================================
def soil_step(theta, rain, irrig, et0, P):
    """One-day bucket update. theta = root-zone available water (mm). Returns (new, drained, etc)."""
    ks = float(np.clip(theta / P.raw_thr, 0, 1)) if P.raw_thr > 0 else 1.0
    etc = P.kc * ks * et0
    raw = theta + P.rain_eff * rain + irrig - etc
    drained = max(0.0, raw - P.taw)           # excess above capacity drains below the roots
    return float(np.clip(raw, 0, P.taw)), float(drained), float(etc)


def simulate_soil(df, P, sim, rng):
    n = len(df)
    theta = np.zeros(n + 1)
    theta[0] = sim["initial_fraction"] * P.taw
    irrig = np.zeros(n)
    delta_nat = np.zeros(n)           # change over day t if NO irrigation
    lo, hi = sim["irrigation_trigger_range"]
    a_lo, a_hi = sim["irrigation_amount_range_mm"]
    thr = rng.uniform(lo, hi, size=n // 30 + 1)
    for t in range(n):
        th = theta[t]
        if th < thr[t // 30] * P.taw and rng.random() < 0.8:
            irrig[t] = rng.uniform(a_lo, a_hi)
        delta_nat[t] = soil_step(th, df["rain"].iloc[t], 0.0, df["et0"].iloc[t], P)[0] - th
        theta[t + 1] = soil_step(th, df["rain"].iloc[t], irrig[t], df["et0"].iloc[t], P)[0]
    out = pd.DataFrame({"theta": theta[:-1], "irrig": irrig, "delta_nat": delta_nat}, index=df.index)
    out["theta_meas"] = out["theta"] + rng.normal(0, sim["sensor_noise_mm"], n)
    return out


def hourly_drying_curve(theta0, delta_day):
    w = np.zeros(24)
    w[6:18] = np.sin(np.pi * (np.arange(12) + 0.5) / 12)
    w /= w.sum()
    return theta0 + delta_day * np.concatenate([[0.0], np.cumsum(w)])


def vwc_to_mm(vwc, P):
    """Convert a volumetric-water-content sensor reading to available root-zone water (mm)."""
    frac = (vwc - P.vwc_wp) / max(P.vwc_fc - P.vwc_wp, 1e-6)
    return float(np.clip(frac, 0, 1) * P.taw)


# =============================================================================
# 5. RULE-BASED DECISION ENGINE (not ML)
# =============================================================================
def decide(theta_meas, delta_pred, rain_fc, tmax_fc, P, fert_due=True):
    """Turn sensor reading + ML forecast into an irrigation / fertigation recipe."""
    projected = theta_meas + delta_pred
    r = {"scenario": "no_irrigation_needed", "irrigate": False, "volume_mm": 0.0,
         "fertigate": False, "venturi_dose_factor": 0.0, "timing": "n/a", "reason": ""}
    if rain_fc >= P.rain_skip:
        r["scenario"] = "rain_delay"
        r["reason"] = (f"RAIN DELAY: forecast {rain_fc:.1f} mm >= {P.rain_skip} mm -> suppress "
                       "irrigation and fertiliser (no N-P-K runoff)")
    elif projected < P.raw_thr:
        r.update(scenario="irrigate", irrigate=True, timing="morning",
                 volume_mm=round(min(P.max_irrig, P.taw - theta_meas), 1))
        r["reason"] = (f"Projected soil water {projected:.0f} mm < stress threshold {P.raw_thr:.0f} mm "
                       "-> irrigate")
        if fert_due:
            r.update(fertigate=True, venturi_dose_factor=1.0)
        else:
            r["reason"] += "; fertigation skipped this time (every-N-irrigations rule)"
        if tmax_fc >= P.heat_tmax:
            r["scenario"] = "irrigate_heatwave"
            r["timing"] = "evening"
            r["reason"] += f"; HEATWAVE (Tmax {tmax_fc:.1f} C >= {P.heat_tmax:.1f}): evening irrigation"
            if r["fertigate"]:
                r["venturi_dose_factor"] = 0.5
                r["reason"] += ", halve Venturi dose"
    else:
        r["reason"] = (f"Projected soil water {projected:.0f} mm >= stress threshold "
                       f"{P.raw_thr:.0f} mm -> no irrigation needed")
    r["ec_setpoint_dS_m"] = round(P.base_ec + r["venturi_dose_factor"] * P.fert_inc, 2)
    r["final_ec_ppt_est"] = round(r["ec_setpoint_dS_m"] * P.ec_to_ppt, 2)
    r["within_crop_ec_limit"] = bool(r["final_ec_ppt_est"] <= P.ec_limit_ppt)
    return r


# =============================================================================
# 6. POLICY SIMULATION: fixed timer vs soil-sensor rule vs AI (all SIMULATED)
# =============================================================================
def run_policy(policy, df_te, s2_te, world, ctrl, m2, cfg, seed):
    """Run one policy over the test period. `world` = true physics, `ctrl` = what the
    controller believes. policy in {'fixed','reactive','ai'}."""
    rng = np.random.default_rng(seed + 1000)
    n = len(df_te)
    sim = cfg["simulator"]
    every = cfg["irrigation"]["fixed_timer_every_n_days"]
    amount = cfg["irrigation"]["fixed_timer_amount_mm"]
    theta = sim["initial_fraction"] * world.taw
    keys = ["irrig", "drained", "stress", "fert_units", "fert_risk", "rain_delay", "heat_dilute"]
    rec = {k: np.zeros(n) for k in keys}
    static = s2_te[SOIL_COLS].to_dict("records") if policy == "ai" else None
    events = 0
    for i in range(n):
        th_meas = theta + rng.normal(0, sim["sensor_noise_mm"])
        rain, et0 = df_te["rain"].iloc[i], df_te["et0"].iloc[i]
        rain_fc, tmax_fc = s2_te["rain_fc"].iloc[i], s2_te["tmax_fc"].iloc[i]
        rec["stress"][i] = float(theta < world.raw_thr)
        rd = hd = False
        if policy == "fixed":
            vol = amount if i % every == 0 else 0.0
            dose = 1.0 if vol > 0 else 0.0
        else:
            if policy == "ai":
                row = dict(static[i])
                row["theta"] = th_meas
                delta = float(m2.predict(pd.DataFrame([row], columns=SOIL_COLS))[0])
            else:
                delta = 0.0
            rcp = decide(th_meas, delta, rain_fc, tmax_fc, ctrl, fert_due=(events % ctrl.fert_every_n == 0))
            vol, dose = rcp["volume_mm"], rcp["venturi_dose_factor"]
            rd, hd = rcp["scenario"] == "rain_delay", rcp["scenario"] == "irrigate_heatwave"
            if vol > 0:
                events += 1
        theta, drained, _ = soil_step(theta, rain, vol, et0, world)
        rec["irrig"][i], rec["drained"][i] = vol, drained
        rec["fert_units"][i] = vol * dose
        rec["fert_risk"][i] = vol * dose if rain >= ctrl.rain_skip else 0.0
        rec["rain_delay"][i], rec["heat_dilute"][i] = float(rd), float(hd)
    return pd.DataFrame(rec, index=df_te.index)


def summarize(rec, mask=None):
    r = rec if mask is None else rec[mask]
    return {"irrigation_mm": float(r["irrig"].sum()), "irrigation_events": int((r["irrig"] > 0).sum()),
            "stress_days": int(r["stress"].sum()), "drainage_mm": float(r["drained"].sum()),
            "fertiliser_units": float(r["fert_units"].sum()),
            "fertiliser_on_heavy_rain_days": float(r["fert_risk"].sum()),
            "rain_delay_days": int(r["rain_delay"].sum()),
            "heat_dilution_days": int(r["heat_dilute"].sum()), "days": int(len(r))}


def driest_window_mask(df_te, days):
    s = df_te["rain"].rolling(days).sum()
    end = s.idxmin()
    start = end - pd.Timedelta(days=days - 1)
    return (df_te.index >= start) & (df_te.index <= end), start, end


# =============================================================================
# 7. PID LOOP SIMULATION (not ML)
# =============================================================================
class PID:
    def __init__(self, kp, ki, kd, dt, umin=0.0, umax=1.0):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.umin, self.umax = umin, umax
        self.i, self.prev = 0.0, None

    def step(self, sp, pv):
        e = sp - pv
        d = 0.0 if self.prev is None else (e - self.prev) / self.dt
        self.prev = e
        i_new = self.i + e * self.dt
        u_unsat = self.kp * e + self.ki * i_new + self.kd * d
        u = float(np.clip(u_unsat, self.umin, self.umax))
        if u == u_unsat:       # anti-windup
            self.i = i_new
        return u


def settle_time(t, ec, sp, t0, t1, band):
    mask = (t >= t0) & (t < t1)
    bad = np.where(np.abs(ec[mask] - sp[mask]) > band)[0]
    return 0 if len(bad) == 0 else int(bad[-1] + 1)


def simulate_pid(P, pc, seed):
    """Venturi opening u in [0,1] -> outlet EC (first-order lag + transport delay).
    t=0: setpoint = base + full fertiliser dose;  t=450: FDI output drifts up;
    t=700: heatwave -> setpoint drops to the half-dose value."""
    rng = np.random.default_rng(seed)
    dt, T = 1.0, 1000
    K, tau, delay = pc["plant_gain_dS_m"], pc["tau_s"], int(pc["delay_s"])
    pid = PID(pc["kp"], pc["ki"], pc["kd"], dt)
    buf = deque([0.0] * delay, maxlen=max(delay, 1))
    t = np.arange(0, T, dt)
    base = np.where(t < 450, P.base_ec, P.base_ec + pc["fdi_drift_dS_m"])
    sp_full, sp_half = P.base_ec + P.fert_inc, P.base_ec + 0.5 * P.fert_inc
    sp = np.where(t < 700, sp_full, sp_half)
    ec, u_hist = np.zeros(T), np.zeros(T)
    ec[0] = base[0]
    for k in range(1, T):
        meas = ec[k - 1] + rng.normal(0, 0.01)
        u = pid.step(sp[k], meas)
        u_hist[k] = u
        u_del = buf[0] if delay > 0 else u
        if delay > 0:
            buf.append(u)
        ec[k] = ec[k - 1] + dt / tau * (base[k] + K * u_del - ec[k - 1])
    band = 0.05 * P.fert_inc
    step = max(sp_full - P.base_ec, 1e-6)
    return t, ec, sp, u_hist, {
        "overshoot_pct_first_step": float(max(0, (ec[:450].max() - sp_full) / step * 100)),
        "settle_s_first_step": settle_time(t, ec, sp, 0, 450, band),
        "recovery_s_after_FDI_drift": settle_time(t, ec, sp, 450, 700, band),
        "settle_s_heatwave_setpoint": settle_time(t, ec, sp, 700, T, band),
        "band_dS_m": float(band)}


# =============================================================================
# 8. THE PIPELINE (one seed)
# =============================================================================
def build_et0_dataset(df, cfg, rng):
    """Observation + simulated-forecast features, target = tomorrow's ET0."""
    z = make_noise(len(df), rng)
    fc = apply_forecast_error(df, z, cfg["forecast_error"], 1.0)
    X = obs_features(df).join(fc_next_features(fc))
    data = X.join(df["et0"].shift(-1).rename("target")).dropna()
    return data, fc, z


def run_pipeline(df, cfg, P, seed, out: Path, full: bool):
    rng = np.random.default_rng(seed)
    exp = cfg["experiments"]
    res = {"seed": seed}

    # ---- ET0 dataset and split --------------------------------------------
    data, fc, z = build_et0_dataset(df, cfg, rng)
    n = len(data)
    i1, i2 = int(0.6 * n), int(0.8 * n)
    tr, te = data.iloc[:i1], data.iloc[i2:]
    d1, d2 = data.index[i1], data.index[i2]
    all_cols = [c for c in data.columns if c != "target"]
    obs_only = [c for c in all_cols if not c.startswith("fc_")]
    if full:
        print(f"[split] ET0 model train {tr.index[0].date()}..{tr.index[-1].date()} | soil-model train "
              f"{d1.date()}..{(d2 - pd.Timedelta(days=1)).date()} | TEST {te.index[0].date()}..{te.index[-1].date()}")

    # ---- Task A: nowcast with reduced sensors (today's ET0, no forecast) ----
    now_T = ["tmax", "tmin", "doy_sin", "doy_cos"]
    now_TR = ["tmax", "tmin", "rs", "doy_sin", "doy_cos"]
    y_now_tr, y_now_te = tr["et0"], te["et0"]
    nowcast, nowcast_models, nowcast_pred = {}, {}, {}
    for name, cols in (("T only (Tmax,Tmin)", now_T), ("T + radiation", now_TR)):
        m = mk_xgb(seed).fit(tr[cols], y_now_tr)
        nowcast_models[name] = m
        nowcast_pred[name] = m.predict(te[cols])
        nowcast[f"XGBoost - {name}"] = reg_metrics(y_now_te, nowcast_pred[name])
        rg = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(tr[cols], y_now_tr)
        nowcast[f"Ridge - {name}"] = reg_metrics(y_now_te, rg.predict(te[cols]))
    nowcast["Hargreaves (uncalibrated, T only)"] = reg_metrics(
        y_now_te, hargreaves(te["tmax"], te["tmin"], te.index.dayofyear, P.lat))
    res["task_A_nowcast_reduced_sensors"] = nowcast

    # ---- Task B: tomorrow's ET0 forecast ------------------------------------
    y_tr, y_te = tr["target"], te["target"]
    m_obs = mk_xgb(seed).fit(tr[obs_only], y_tr)
    m_fc = mk_xgb(seed).fit(tr[all_cols], y_tr)          # deployed model (uses forecast inputs)
    p_obs, p_fc = m_obs.predict(te[obs_only]), m_fc.predict(te[all_cols])
    rg_fc = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(tr[all_cols], y_tr)
    forecast = {"XGBoost - observations only": reg_metrics(y_te, p_obs),
                "XGBoost - with (simulated) weather forecast": reg_metrics(y_te, p_fc),
                "Ridge - with (simulated) weather forecast": reg_metrics(y_te, rg_fc.predict(te[all_cols])),
                "Persistence (tomorrow = today)": reg_metrics(y_te, te["et0"].values)}
    res["task_B_forecast_tomorrow_et0"] = forecast

    if full:
        curve = {}
        for sc in exp["forecast_error_scales"]:
            fcs = apply_forecast_error(df, z, cfg["forecast_error"], sc)
            Xs = obs_features(df).join(fc_next_features(fcs)).join(df["et0"].shift(-1).rename("target")).dropna()
            ms = mk_xgb(seed).fit(Xs.iloc[:i1][all_cols], Xs.iloc[:i1]["target"])
            curve[str(sc)] = reg_metrics(Xs.iloc[i2:]["target"], ms.predict(Xs.iloc[i2:][all_cols]))
        res["task_B_forecast_error_curve"] = curve

    # ---- Stage 2: soil drying (SIMULATED soil) --------------------------------
    soil = simulate_soil(df, P, cfg["simulator"], rng)
    s2 = pd.DataFrame(index=df.index)
    s2["theta"] = soil["theta_meas"]
    s2["et0_hat"] = pd.Series(m_fc.predict(data[all_cols]), index=data.index).reindex(df.index).shift(1)
    s2["rain_fc"], s2["tmax_fc"] = fc["rain"], fc["tmax"]
    s2["rh_prev"], s2["rs_prev"] = df["rh"].shift(1), df["rs"].shift(1)
    wet = (df["rain"] > 1).astype(int)
    s2["dry_days"] = (1 - wet).groupby(wet.cumsum()).cumsum().shift(1)
    s2["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    s2["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
    s2["target"] = soil["delta_nat"]
    s2 = s2.dropna()
    s2_tr = s2[(s2.index >= d1) & (s2.index < d2)]
    s2_te = s2[s2.index >= d2]
    m2 = mk_xgb(seed, n=400, lr=0.04).fit(s2_tr[SOIL_COLS], s2_tr["target"])
    rg2 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(s2_tr[SOIL_COLS], s2_tr["target"])
    q = m2.predict(s2_te[SOIL_COLS])
    res["stage2_soil_drying_SIMULATED"] = {
        "XGBoost": reg_metrics(s2_te["target"], q),
        "Ridge": reg_metrics(s2_te["target"], rg2.predict(s2_te[SOIL_COLS])),
        "Persistence (yesterday's change)": reg_metrics(
            s2_te["target"], soil["delta_nat"].shift(1).reindex(s2_te.index).fillna(0).values)}

    # ---- Policy comparison (SIMULATED) ------------------------------------------
    df_te = df.loc[s2_te.index]
    pol = {p: run_policy(p, df_te, s2_te, P, P, m2, cfg, seed) for p in ("fixed", "reactive", "ai")}
    dmask, dstart, dend = driest_window_mask(df_te, exp["driest_window_days"])
    res["policy_comparison_SIMULATED"] = {
        "full_test_period": {p: summarize(r) for p, r in pol.items()},
        f"driest_{exp['driest_window_days']}_days": {p: summarize(r, dmask) for p, r in pol.items()},
        "driest_window": [str(dstart.date()), str(dend.date())]}

    if not full:
        return res

    # =========================== full-run extras ===============================
    # decision demo: one example per scenario
    pred_delta = pd.Series(q, index=s2_te.index)
    cats = {}
    for day, r in s2_te.iterrows():
        rcp = decide(r["theta"], pred_delta[day], r["rain_fc"], r["tmax_fc"], P)
        cats.setdefault(rcp["scenario"], []).append((day, rcp, r))
    demo = {}
    for sc in ("irrigate", "irrigate_heatwave", "rain_delay", "no_irrigation_needed"):
        if sc in cats:
            day, rcp, r = cats[sc][len(cats[sc]) // 2]
            demo[sc] = {"date": str(day.date()),
                        "inputs": {"soil_water_mm": round(float(r["theta"]), 1),
                                   "pred_change_mm": round(float(pred_delta[day]), 2),
                                   "rain_forecast_mm": round(float(r["rain_fc"]), 1),
                                   "tmax_forecast_C": round(float(r["tmax_fc"]), 1)},
                        "recipe": rcp}
        else:
            demo[sc] = None
    res["decision_demo"] = demo
    res["scenario_day_counts_test"] = {k: len(v) for k, v in cats.items()}

    # PID
    t, ec, sp, u_hist, pid_m = simulate_pid(P, cfg["pid"], seed)
    res["pid_SIMULATED"] = pid_m

    # sensitivity: wrong world assumptions (controller keeps its configured beliefs)
    sens_world = []
    for name, key in (("kc", "kc"), ("taw_mm", "taw"), ("rain_eff", "rain_eff")):
        for v in exp["sensitivity_world"][name]:
            world = dataclasses.replace(P, **{key: float(v)})
            for p in ("fixed", "reactive", "ai"):
                s = summarize(run_policy(p, df_te, s2_te, world, P, m2, cfg, seed))
                sens_world.append({"param": name, "true_value": v, "policy": p, **s})
    res["sensitivity_world_SIMULATED"] = sens_world

    # sensitivity: controller thresholds
    sens_pol = []
    for rs_mm in exp["sensitivity_policy"]["rain_skip_mm"]:
        c = dataclasses.replace(P, rain_skip=float(rs_mm))
        sens_pol.append({"varied": "rain_skip_mm", "value": rs_mm, "heat_tmax_c": round(P.heat_tmax, 1),
                         **summarize(run_policy("ai", df_te, s2_te, P, c, m2, cfg, seed))})
    for pc_ in exp["sensitivity_policy"]["heat_percentile"]:
        ht = float(df["tmax"].quantile(pc_))
        c = dataclasses.replace(P, heat_tmax=ht)
        sens_pol.append({"varied": "heat_percentile", "value": pc_, "heat_tmax_c": round(ht, 1),
                         **summarize(run_policy("ai", df_te, s2_te, P, c, m2, cfg, seed))})
    res["sensitivity_policy_thresholds_SIMULATED"] = sens_pol

    # ---- save models + meta ---------------------------------------------------
    mdir = out / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    m_fc.save_model(str(mdir / "et0_forecast_xgb.json"))
    m_obs.save_model(str(mdir / "et0_forecast_obs_only_xgb.json"))
    nowcast_models["T only (Tmax,Tmin)"].save_model(str(mdir / "et0_nowcast_T_xgb.json"))
    nowcast_models["T + radiation"].save_model(str(mdir / "et0_nowcast_T_radiation_xgb.json"))
    m2.save_model(str(mdir / "soil_drying_xgb.json"))
    (mdir / "meta.json").write_text(json.dumps({
        "params": dataclasses.asdict(P), "et0_forecast_cols": all_cols, "soil_cols": SOIL_COLS,
        "heat_tmax_c": P.heat_tmax, "trained_on": f"{df.index[0].date()}..{df.index[-1].date()}"}, indent=2))

    make_plots(out, df, te, y_te, p_obs, p_fc, nowcast_pred, y_now_te, res, m_fc, all_cols,
               s2_te, q, soil, pred_delta, P, pol, dmask, exp, t, ec, sp, u_hist, sens_world)
    pd.DataFrame({"actual": y_te, "xgb_obs_only": p_obs, "xgb_with_forecast": p_fc,
                  "persistence": te["et0"].values}).to_csv(out / "et0_test_predictions.csv")
    pd.concat({k: v for k, v in pol.items()}, axis=1).to_csv(out / "policy_daily_records.csv")
    pd.DataFrame(sens_world).to_csv(out / "sensitivity_world.csv", index=False)
    pd.DataFrame(sens_pol).to_csv(out / "sensitivity_policy.csv", index=False)
    return res


# =============================================================================
# 9. PLOTS
# =============================================================================
def make_plots(out, df, te, y_te, p_obs, p_fc, nowcast_pred, y_now_te, res, m_fc, all_cols,
               s2_te, q, soil, pred_delta, P, pol, dmask, exp, t, ec, sp, u_hist, sens_world):
    # fig1: ET0 models
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    k = min(100, len(te))
    ax[0, 0].plot(te.index[:k], y_te.iloc[:k], "k-", lw=2, label="Actual (Penman-Monteith)")
    ax[0, 0].plot(te.index[:k], p_fc[:k], color="#1f77b4", label="XGBoost + simulated forecast")
    ax[0, 0].plot(te.index[:k], p_obs[:k], "--", color="#d62728", alpha=0.7, label="XGBoost observations only")
    ax[0, 0].set(title="Task B: tomorrow's ET0 (first 100 test days)", ylabel="ET0 (mm/day)")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].tick_params(axis="x", rotation=30)
    best = "T + radiation"
    ax[0, 1].scatter(y_now_te, nowcast_pred[best], s=6, alpha=0.5)
    lim = [y_now_te.min(), y_now_te.max()]
    ax[0, 1].plot(lim, lim, "k--")
    r2 = res["task_A_nowcast_reduced_sensors"][f"XGBoost - {best}"]["R2"]
    ax[0, 1].set(title=f"Task A: today's ET0 from Tmax, Tmin, radiation (R2={r2:.2f})",
                 xlabel="Actual ET0", ylabel="Predicted ET0")
    ax[1, 0].scatter(y_te, p_fc, s=6, alpha=0.5)
    lim = [y_te.min(), y_te.max()]
    ax[1, 0].plot(lim, lim, "k--")
    r2b = res["task_B_forecast_tomorrow_et0"]["XGBoost - with (simulated) weather forecast"]["R2"]
    ax[1, 0].set(title=f"Task B scatter (R2={r2b:.2f})", xlabel="Actual ET0", ylabel="Predicted ET0")
    cv = res["task_B_forecast_error_curve"]
    xs = [float(s) for s in cv]
    ax[1, 1].plot(xs, [cv[s]["RMSE"] for s in cv], "o-")
    ax[1, 1].set(title="Forecast skill vs (simulated) weather-forecast error",
                 xlabel="Forecast error scale (0 = perfect forecast, 1 = config value)",
                 ylabel="Test RMSE (mm/day)")
    fig.tight_layout()
    fig.savefig(out / "fig1_et0_models.png", dpi=150)
    plt.close(fig)

    # fig2: importance
    imp = pd.Series(m_fc.get_booster().get_score(importance_type="gain")).sort_values()[-12:]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color="#1f77b4")
    ax.set(title="Tomorrow's-ET0 model: feature importance (gain)", xlabel="Gain")
    fig.tight_layout()
    fig.savefig(out / "fig2_feature_importance.png", dpi=150)
    plt.close(fig)

    # fig3: soil
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(s2_te["target"], q, s=6, alpha=0.5)
    lim = [s2_te["target"].min(), s2_te["target"].max()]
    ax[0].plot(lim, lim, "k--")
    ax[0].set(title=f"Next-day soil water change - SIMULATED soil (R2={res['stage2_soil_drying_SIMULATED']['XGBoost']['R2']:.2f})",
              xlabel="Actual change (mm/day)", ylabel="Predicted change (mm/day)")
    day = s2_te["tmax_fc"].idxmax()
    th0 = float(s2_te.loc[day, "theta"])
    ax[1].plot(range(25), hourly_drying_curve(th0, float(s2_te.loc[day, "target"])), "k-", lw=2, label="Actual (simulated)")
    ax[1].plot(range(25), hourly_drying_curve(th0, float(pred_delta[day])), color="#1f77b4", lw=2, label="XGBoost forecast")
    ax[1].axhline(P.raw_thr, color="#d62728", ls="--", label="Stress threshold")
    ax[1].set(title=f"24-h drying curve {day.date()} (SIMULATED, illustrative hourly split)",
              xlabel="Hour of day", ylabel="Root-zone water (mm)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "fig3_soil_drying_SIMULATED.png", dpi=150)
    plt.close(fig)

    # fig4: PID
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(t, sp, "k--", label="EC setpoint")
    ax[0].plot(t, ec, color="#1f77b4", label="Outlet EC")
    for x, lab in ((450, "FDI output drifts"), (700, "Heatwave: dilute")):
        ax[0].axvline(x, color="gray", ls=":")
        ax[0].text(x + 5, ec.min() + 0.05, lab, fontsize=8)
    ax[0].set(ylabel="EC (dS/m)", title="PID control of Venturi-dosed EC (SIMULATED plant)")
    ax[0].legend()
    ax[1].plot(t, u_hist, color="#2ca02c")
    ax[1].set(ylabel="Venturi valve opening", xlabel="Time (s)")
    fig.tight_layout()
    fig.savefig(out / "fig4_pid_response_SIMULATED.png", dpi=150)
    plt.close(fig)

    # fig5: policy comparison
    pc = res["policy_comparison_SIMULATED"]
    wins = [("full_test_period", "Full test period"),
            (f"driest_{exp['driest_window_days']}_days", f"Driest {exp['driest_window_days']} days")]
    metrics = [("irrigation_mm", "Irrigation water (mm)"), ("stress_days", "Days under water stress"),
               ("fertiliser_on_heavy_rain_days", "Fertiliser applied on heavy-rain days")]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    labels = {"fixed": "Fixed timer", "reactive": "Sensor rule\n(no ML)", "ai": "AI\n(XGBoost)"}
    colors = {"fixed": "#999999", "reactive": "#ff7f0e", "ai": "#1f77b4"}
    for r_i, (wk, wt) in enumerate(wins):
        for c_i, (mk, mt) in enumerate(metrics):
            vals = [pc[wk][p][mk] for p in labels]
            ax[r_i, c_i].bar([labels[p] for p in labels], vals, color=[colors[p] for p in labels])
            ax[r_i, c_i].set_title(f"{wt}\n{mt}", fontsize=10)
    fig.suptitle("Policy comparison - SIMULATED soil, weather-forecast inputs simulated", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig5_policy_comparison_SIMULATED.png", dpi=150)
    plt.close(fig)

    # fig6: sensitivity to wrong world assumptions
    sw = pd.DataFrame(sens_world)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for c_i, name in enumerate(["kc", "taw_mm", "rain_eff"]):
        for r_i, (mk, mt) in enumerate((("stress_days", "Stress days"), ("irrigation_mm", "Irrigation (mm)"))):
            for p in ("fixed", "reactive", "ai"):
                d = sw[(sw.param == name) & (sw.policy == p)].sort_values("true_value")
                ax[r_i, c_i].plot(d["true_value"], d[mk], "o-", color=colors[p], label=labels[p].replace("\n", " "))
            ax[r_i, c_i].set(xlabel=f"TRUE {name} (controller keeps its configured value)", ylabel=mt)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Robustness: what if our assumed crop/soil values are wrong? (SIMULATED)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig6_sensitivity_SIMULATED.png", dpi=150)
    plt.close(fig)


# =============================================================================
# 10. PRINTING + MAIN
# =============================================================================
def print_table(title, d):
    print(f"\n=== {title} ===")
    for k, v in d.items():
        print(f"  {k:46s} R2={v['R2']:7.3f}  RMSE={v['RMSE']:6.3f}  MAE={v['MAE']:6.3f}")


def to_jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def main():
    ap = argparse.ArgumentParser(description="S.E.D.O.S.S. AI-brain training script")
    ap.add_argument("--config", default=None, help="YAML site config (default: built-in example)")
    ap.add_argument("--out", default="sedoss_outputs")
    ap.add_argument("--offline", action="store_true", help="synthetic weather (pipeline test only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="skip extra seeds (and keep sweeps off the table)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    print(f"[config] site: {cfg['site']['name']}  ({cfg['site']['latitude']}, {cfg['site']['longitude']})")

    df, synthetic = get_weather(args, cfg, out)
    df = df.copy()
    df["doy"] = df.index.dayofyear
    df["et0"] = et0_pm(df["tmean"], df["tmax"], df["tmin"], df["rh"], df["u2"], df["rs"],
                       df["doy"], cfg["site"]["latitude"], cfg["site"]["elevation_m"])
    print(f"[data] {len(df)} days  {df.index[0].date()} -> {df.index[-1].date()}  mean ET0 = {df['et0'].mean():.2f} mm/day")

    heat = cfg["policy"]["heat_tmax_c"]
    if heat == "auto":
        heat = float(df["tmax"].quantile(cfg["policy"]["heat_percentile"]))
        print(f"[config] heatwave threshold = {heat:.1f} C (the {int(cfg['policy']['heat_percentile'] * 100)}th "
              "percentile of daily Tmax in this dataset; NASA grid values run slightly cooler than a farm thermometer)")
    P = make_params(cfg, heat)
    warnings = validate_config(P, cfg)
    print(f"[rain] days with rain >= {P.rain_skip} mm: {(df['rain'] >= P.rain_skip).mean() * 100:.1f}%")

    res = run_pipeline(df, cfg, P, args.seed, out, full=True)

    print_table("TASK A - today's ET0 from FEWER SENSORS (test set, mm/day)", res["task_A_nowcast_reduced_sensors"])
    print_table("TASK B - tomorrow's ET0 (test set, mm/day)", res["task_B_forecast_tomorrow_et0"])
    print("\n  Forecast skill vs simulated forecast error (RMSE mm/day; scale 0 = PERFECT forecast, upper bound only):")
    for s, v in res["task_B_forecast_error_curve"].items():
        print(f"    scale {s:>4}: RMSE={v['RMSE']:.3f}  R2={v['R2']:.3f}")
    print_table("STAGE 2 - next-day soil water change (SIMULATED soil, mm/day)", res["stage2_soil_drying_SIMULATED"])

    print("\n=== POLICY COMPARISON (all SIMULATED) ===")
    for window, block in res["policy_comparison_SIMULATED"].items():
        if window == "driest_window":
            continue
        print(f"  -- {window} --")
        print(f"  {'policy':10s} {'irrig mm':>9s} {'events':>7s} {'stress d':>9s} {'drain mm':>9s} {'fert@rain':>10s}")
        for p, s in block.items():
            print(f"  {p:10s} {s['irrigation_mm']:9.0f} {s['irrigation_events']:7d} {s['stress_days']:9d} "
                  f"{s['drainage_mm']:9.0f} {s['fertiliser_on_heavy_rain_days']:10.0f}")
    print(f"  (driest window: {res['policy_comparison_SIMULATED']['driest_window']})")

    print("\n=== DECISION ENGINE: one example per scenario ===")
    for sc, v in res["decision_demo"].items():
        print(f"  [{sc}] " + ("not present in the test period" if v is None else f"{v['date']}: {v['recipe']['reason']}"))
    print(f"  scenario day counts in test period: {res['scenario_day_counts_test']}")

    print("\n=== PID loop (SIMULATED plant) ===")
    for k, v in res["pid_SIMULATED"].items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    # ---- extra seeds ------------------------------------------------------------
    seed_summary = None
    if not args.quick and cfg["experiments"]["extra_seeds"]:
        print("\n[seeds] re-running with extra seeds for stability ...")
        runs = [res] + [run_pipeline(df, cfg, P, s, out, full=False) for s in cfg["experiments"]["extra_seeds"]]

        def pick(r):
            pc = r["policy_comparison_SIMULATED"]["full_test_period"]
            return {"taskA_T_only_R2": r["task_A_nowcast_reduced_sensors"]["XGBoost - T only (Tmax,Tmin)"]["R2"],
                    "taskA_T_radiation_R2": r["task_A_nowcast_reduced_sensors"]["XGBoost - T + radiation"]["R2"],
                    "taskB_obs_only_RMSE": r["task_B_forecast_tomorrow_et0"]["XGBoost - observations only"]["RMSE"],
                    "taskB_with_forecast_RMSE": r["task_B_forecast_tomorrow_et0"]["XGBoost - with (simulated) weather forecast"]["RMSE"],
                    "stage2_R2_simulated": r["stage2_soil_drying_SIMULATED"]["XGBoost"]["R2"],
                    "ai_irrigation_mm": pc["ai"]["irrigation_mm"], "fixed_irrigation_mm": pc["fixed"]["irrigation_mm"],
                    "ai_stress_days": pc["ai"]["stress_days"], "fixed_stress_days": pc["fixed"]["stress_days"]}
        tbl = pd.DataFrame([pick(r) for r in runs])
        seed_summary = {c: {"mean": float(tbl[c].mean()), "sd": float(tbl[c].std()),
                            "min": float(tbl[c].min()), "max": float(tbl[c].max())} for c in tbl}
        print(f"\n=== STABILITY across {len(runs)} random seeds (mean +/- sd) ===")
        for c, v in seed_summary.items():
            print(f"  {c:28s} {v['mean']:9.3f} +/- {v['sd']:.3f}   [{v['min']:.3f}, {v['max']:.3f}]")

    summary = {"synthetic_data": synthetic, "config_used": cfg, "heat_tmax_c_used": heat,
               "config_warnings": warnings, "data_range": [str(df.index[0].date()), str(df.index[-1].date())],
               "results_primary_seed": res, "stability_across_seeds": seed_summary}
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=to_jsonable))
    print(f"\nDone. Everything is in ./{out}/  (results.json, figures fig1-fig6, CSVs, models/)")
    print("Stage-2, the policy comparison, sensitivity tests and the PID are SIMULATED - say so on slides.")
    print("Weather-forecast inputs are simulated (actual weather + configured error).")
    if synthetic:
        print("*** REMINDER: data was SYNTHETIC. Re-run on real NASA data before quoting numbers. ***")


if __name__ == "__main__":
    main()

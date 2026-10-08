#!/usr/bin/env python3
"""
S.E.D.O.S.S. "AI Brain" - full training + demo script
=======================================================
Pipeline (matches the slide: Sense -> Predict -> Decide -> Act -> Verify -> Adapt)

  1. Sense    : daily weather from NASA POWER (or synthetic fallback for testing)
  2. Label    : FAO-56 Penman-Monteith reference evapotranspiration (ET0) = physics label
  3. Predict  : XGBoost model #1 -> forecasts TOMORROW's ET0
                XGBoost model #2 -> forecasts the next-day soil drying (delta soil water, mm)
  4. Decide   : rule-based irrigation / fertigation recipe (NOT machine learning)
  5. Adapt    : PID loop simulation holding Venturi-dosed EC on target (NOT machine learning)

Install:   pip install xgboost scikit-learn pandas numpy matplotlib requests
Run:       python sedoss_ml.py                 # downloads NASA POWER data
           python sedoss_ml.py --offline       # synthetic weather (pipeline test only)
           python sedoss_ml.py --lat 2.05 --lon 102.57 --start 20150101 --end 20241231

IMPORTANT HONESTY NOTES (put these on a slide / say them in Q&A):
  * ET0 labels come from the Penman-Monteith equation, not from a lysimeter.
    The model learns to FORECAST tomorrow's ET0; that is the ML contribution.
  * Soil moisture is SIMULATED (water-balance bucket). Model #2 therefore shows the
    pipeline works, NOT that it works on real soil. Real-sensor validation = future work.
  * If the script says "SYNTHETIC DATA", do NOT quote any of its metrics.
"""

import argparse
import json
from collections import deque
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

# ----------------------------------------------------------------------------
# ASSUMPTIONS - EDIT THESE AND CITE A SOURCE FOR EACH ONE BEFORE PRESENTING
# ----------------------------------------------------------------------------
ELEVATION_M = 50.0        # site elevation for the ET0 equation
TAW_MM = 100.0            # total available water in the root zone (mm)  [assumption]
P_DEPLETION = 0.5         # fraction of TAW usable before stress starts   [assumption]
KC = 0.9                  # crop coefficient (placeholder, not durian-verified)
RAIN_EFF = 0.8            # fraction of rainfall that infiltrates         [assumption]
MAX_IRRIG_MM = 15.0       # max irrigation per event (mm)
RAIN_SKIP_MM = 10.0       # forecast rain above this -> rain delay (skip irrigation + fertigation)
HEAT_TMAX_C = 34.0        # forecast Tmax above this -> heatwave mitigation (dilute Venturi dose)
FERT_EC_INC = 0.8         # dS/m added by the Venturi nutrient injection at full dose
SEED = 42


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def fetch_nasa_power(lat, lon, start, end, cache: Path) -> pd.DataFrame:
    """Download daily weather from NASA POWER (free, no API key)."""
    if cache.exists():
        print(f"[data] using cached download: {cache}")
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    import requests

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR"
        f"&community=AG&longitude={lon}&latitude={lat}"
        f"&start={start}&end={end}&format=JSON"
    )
    print("[data] downloading NASA POWER ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    df = pd.DataFrame(p)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.rename(columns={
        "T2M": "tmean", "T2M_MAX": "tmax", "T2M_MIN": "tmin", "RH2M": "rh",
        "WS2M": "u2", "ALLSKY_SFC_SW_DWN": "rs", "PRECTOTCORR": "rain",
    })
    df = df.replace(-999, np.nan).interpolate(limit=3).dropna()
    # Penman-Monteith below needs Rs in MJ/m2/day. Tropical values are ~15-22 MJ.
    # If the download is in kWh/m2/day (~4-6) convert it.
    if df["rs"].median() < 10:
        print("[data] Rs looks like kWh/m2/day -> converting to MJ/m2/day (x3.6)")
        df["rs"] = df["rs"] * 3.6
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
    rs = np.clip(23 - 9 * wet + 1.5 * np.sin(2 * np.pi * (doy - 80) / 365)
                 + rng.normal(0, 1.8, n), 5, 28)
    tmax = 33 + 0.15 * (rs - 18) - 2.0 * wet + rng.normal(0, 0.8, n)
    tmin = 23.5 + 0.5 * wet + rng.normal(0, 0.7, n)
    rh = np.clip(80 - 2 * (tmax - 32) + 8 * wet + rng.normal(0, 4, n), 50, 100)
    u2 = np.clip(rng.gamma(3, 0.4, n), 0.3, 6)
    return pd.DataFrame({"tmean": (tmax + tmin) / 2, "tmax": tmax, "tmin": tmin,
                         "rh": rh, "u2": u2, "rs": rs, "rain": rain}, index=idx)


def get_weather(args, out: Path):
    if not args.offline:
        try:
            return fetch_nasa_power(args.lat, args.lon, args.start, args.end,
                                    out / "nasa_power_cache.csv"), False
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] NASA POWER download failed ({e}).")
    print("[WARN] *** USING SYNTHETIC DATA - metrics are for pipeline testing only ***")
    return synthetic_weather(args.start, args.end, args.seed), True


# ----------------------------------------------------------------------------
# 2. PHYSICS: FAO-56 Penman-Monteith ET0  (mm/day)
# ----------------------------------------------------------------------------
def et0_pm(tmean, tmax, tmin, rh, u2, rs, doy, lat_deg, elev=ELEVATION_M):
    P = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26
    gamma = 0.000665 * P

    def e(t):
        return 0.6108 * np.exp(17.27 * t / (t + 237.3))

    es = (e(tmax) + e(tmin)) / 2
    ea = es * rh / 100
    delta = 4098 * e(tmean) / (tmean + 237.3) ** 2

    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    rso = (0.75 + 2e-5 * elev) * ra

    rns = 0.77 * rs
    rnl = (4.903e-9 * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2)
           * (0.34 - 0.14 * np.sqrt(ea))
           * (1.35 * np.minimum(rs / rso, 1) - 0.35))
    rn = rns - rnl
    G = 0.0  # daily soil heat flux ~ 0
    et0 = (0.408 * delta * (rn - G) + gamma * 900 / (tmean + 273) * u2 * (es - ea)) \
        / (delta + gamma * (1 + 0.34 * u2))
    return np.clip(et0, 0, None)


# ----------------------------------------------------------------------------
# 3. STAGE 1 - XGBoost forecasts TOMORROW'S ET0
# ----------------------------------------------------------------------------
BASE_COLS = ["tmax", "tmin", "rh", "u2", "rs", "rain"]


def build_et0_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    for c in BASE_COLS + ["et0"]:
        f[c] = df[c]
        for k in (1, 2, 3):
            f[f"{c}_l{k}"] = df[c].shift(k)
    f["et0_7d"] = df["et0"].rolling(7).mean()
    f["rain_7d"] = df["rain"].rolling(7).sum()
    f["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    f["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
    return f


def reg_metrics(y, p):
    return {"R2": float(r2_score(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p))}


# ----------------------------------------------------------------------------
# 4. SOIL WATER-BALANCE SIMULATOR (SIMULATED soil data)
# ----------------------------------------------------------------------------
def soil_step(theta, rain, irrig, et0):
    """One-day bucket update. theta = root-zone available water (mm)."""
    ks = np.clip(theta / ((1 - P_DEPLETION) * TAW_MM), 0, 1)  # water-stress factor
    etc = KC * ks * et0
    new = theta + RAIN_EFF * rain + irrig - etc
    return float(np.clip(new, 0, TAW_MM))  # excess above TAW drains away


def simulate_soil(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Run the bucket with varied irrigation habits so all moisture levels appear."""
    n = len(df)
    theta = np.zeros(n + 1)
    theta[0] = 0.7 * TAW_MM
    irrig = np.zeros(n)
    delta_nat = np.zeros(n)  # change over day t if NO irrigation is applied
    thr_blocks = rng.uniform(0.25, 0.8, size=n // 30 + 1)
    for t in range(n):
        th = theta[t]
        if th < thr_blocks[t // 30] * TAW_MM and rng.random() < 0.8:
            irrig[t] = rng.uniform(8, 20)
        delta_nat[t] = soil_step(th, df["rain"].iloc[t], 0.0, df["et0"].iloc[t]) - th
        theta[t + 1] = soil_step(th, df["rain"].iloc[t], irrig[t], df["et0"].iloc[t])
    out = pd.DataFrame({"theta": theta[:-1], "irrig": irrig, "delta_nat": delta_nat},
                       index=df.index)
    out["theta_meas"] = out["theta"] + rng.normal(0, 1.0, n)  # sensor noise (mm)
    return out


def hourly_drying_curve(theta0, delta_day):
    """Spread a daily change over 24 h (daylight-weighted). Illustrative curve."""
    w = np.zeros(24)
    w[6:18] = np.sin(np.pi * (np.arange(12) + 0.5) / 12)
    w /= w.sum()
    return theta0 + delta_day * np.concatenate([[0.0], np.cumsum(w)])


# ----------------------------------------------------------------------------
# 5. RULE-BASED DECISION ENGINE ("Decide" box on the slide - not ML)
# ----------------------------------------------------------------------------
def decide(theta_meas, delta_pred, rain_fc, tmax_fc, base_ec):
    raw_threshold = (1 - P_DEPLETION) * TAW_MM
    projected = theta_meas + delta_pred
    recipe = {"irrigate": False, "volume_mm": 0.0, "fertigate": False,
              "venturi_dose_factor": 0.0, "ec_setpoint_dS_m": round(base_ec, 2),
              "timing": "n/a", "reason": ""}
    if rain_fc >= RAIN_SKIP_MM:
        recipe["reason"] = (f"RAIN DELAY: forecast {rain_fc:.1f} mm >= {RAIN_SKIP_MM} mm "
                            "-> suppress irrigation + fertilizer (no N-P-K runoff)")
        return recipe
    if projected < raw_threshold:
        recipe["irrigate"] = True
        recipe["volume_mm"] = round(min(MAX_IRRIG_MM, TAW_MM - theta_meas), 1)
        recipe["fertigate"] = True
        recipe["venturi_dose_factor"] = 1.0
        recipe["timing"] = "morning"
        recipe["reason"] = (f"Projected soil water {projected:.0f} mm < stress threshold "
                            f"{raw_threshold:.0f} mm -> irrigate")
        if tmax_fc >= HEAT_TMAX_C:
            recipe["venturi_dose_factor"] = 0.5
            recipe["timing"] = "evening"
            recipe["reason"] += (f"; HEATWAVE (Tmax {tmax_fc:.1f} C): halve Venturi dose, "
                                 "dilute stream, irrigate in the evening")
    else:
        recipe["reason"] = (f"Projected soil water {projected:.0f} mm is above the stress "
                            f"threshold {raw_threshold:.0f} mm -> no irrigation needed")
    recipe["ec_setpoint_dS_m"] = round(base_ec + recipe["venturi_dose_factor"] * FERT_EC_INC, 2)
    return recipe


# ----------------------------------------------------------------------------
# 6. PID LOOP SIMULATION ("Adapt" box on the slide - not ML)
# ----------------------------------------------------------------------------
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
        if u == u_unsat:          # anti-windup: only integrate when not saturated
            self.i = i_new
        return u


def settle_time(t, ec, sp, t0, t1, band):
    """Seconds after t0 until |ec - sp| stays within +/-band until t1."""
    mask = (t >= t0) & (t < t1)
    err = np.abs(ec[mask] - sp[mask])
    bad = np.where(err > band)[0]
    return 0 if len(bad) == 0 else int(bad[-1] + 1)


def simulate_pid(kp=1.0, ki=0.06, kd=0.0, seed=SEED):
    """Venturi valve opening u in [0,1] -> outlet EC. First-order lag + transport delay.
    Events: t=0 setpoint 1.2 dS/m; t=450 FDI output drifts (base EC 0.4 -> 0.6);
            t=700 heatwave -> setpoint dropped to 0.9 dS/m."""
    rng = np.random.default_rng(seed)
    dt, T = 1.0, 1000
    K, tau, delay = 2.0, 15.0, 5          # plant gain (dS/m per unit), lag (s), delay (s)
    pid = PID(kp, ki, kd, dt)
    buf = deque([0.0] * delay, maxlen=delay)
    t = np.arange(0, T, dt)
    base = np.where(t < 450, 0.4, 0.6)
    sp = np.where(t < 700, 1.2, 0.9)
    ec = np.zeros(T)
    u_hist = np.zeros(T)
    ec[0] = base[0]
    for k in range(1, T):
        meas = ec[k - 1] + rng.normal(0, 0.01)       # inline EC sensor noise
        u = pid.step(sp[k], meas)
        u_hist[k] = u
        u_delayed = buf[0]
        buf.append(u)
        ec_ss = base[k] + K * u_delayed
        ec[k] = ec[k - 1] + dt / tau * (ec_ss - ec[k - 1])
    band = 0.04
    metrics = {
        "overshoot_pct_first_step": float(max(0, (ec[:450].max() - 1.2) / (1.2 - 0.4) * 100)),
        "settle_s_first_step": settle_time(t, ec, sp, 0, 450, band),
        "recovery_s_after_FDI_drift": settle_time(t, ec, sp, 450, 700, band),
        "settle_s_heatwave_setpoint": settle_time(t, ec, sp, 700, T, band),
        "band_dS_m": band,
    }
    return t, ec, sp, u_hist, base, metrics


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="S.E.D.O.S.S. AI-brain training script")
    ap.add_argument("--lat", type=float, default=1.85)
    ap.add_argument("--lon", type=float, default=102.93)
    ap.add_argument("--start", default="20150101")
    ap.add_argument("--end", default="20241231")
    ap.add_argument("--out", default="sedoss_outputs")
    ap.add_argument("--offline", action="store_true", help="use synthetic weather")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- data + ET0 label -------------------------------------------------
    df, synthetic = get_weather(args, out)
    df = df.copy()
    df["doy"] = df.index.dayofyear
    df["et0"] = et0_pm(df["tmean"], df["tmax"], df["tmin"], df["rh"], df["u2"],
                       df["rs"], df["doy"], args.lat)
    print(f"[data] {len(df)} days  {df.index[0].date()} -> {df.index[-1].date()}  "
          f"mean ET0 = {df['et0'].mean():.2f} mm/day")

    # ---- stage 1: tomorrow's ET0 -------------------------------------------
    feats = build_et0_features(df)
    data = feats.join(df["et0"].shift(-1).rename("target")).dropna()
    xcols = [c for c in data.columns if c != "target"]
    n = len(data)
    i1, i2 = int(0.6 * n), int(0.8 * n)          # time-based 60/20/20 split
    tr, va, te = data.iloc[:i1], data.iloc[i1:i2], data.iloc[i2:]
    d1, d2 = data.index[i1], data.index[i2]
    print(f"[split] M1 train {tr.index[0].date()}-{tr.index[-1].date()} | "
          f"M2 train {va.index[0].date()}-{va.index[-1].date()} | "
          f"TEST {te.index[0].date()}-{te.index[-1].date()}")

    m1 = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                          random_state=args.seed, n_jobs=-1)
    m1.fit(tr[xcols], tr["target"])
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(tr[xcols], tr["target"])

    p_xgb = m1.predict(te[xcols])
    p_ridge = ridge.predict(te[xcols])
    p_persist = te["et0"].values                  # "tomorrow = today"
    res1 = {"XGBoost": reg_metrics(te["target"], p_xgb),
            "Ridge (linear baseline)": reg_metrics(te["target"], p_ridge),
            "Persistence (tomorrow = today)": reg_metrics(te["target"], p_persist)}
    print("\n=== STAGE 1: tomorrow's ET0 (test set, mm/day) ===")
    for k, v in res1.items():
        print(f"  {k:32s} R2={v['R2']:.3f}  RMSE={v['RMSE']:.3f}  MAE={v['MAE']:.3f}")
    m1.save_model(str(out / "models" / "et0_forecast_xgb.json"))

    # ---- stage 2: soil drying (SIMULATED soil) -----------------------------
    soil = simulate_soil(df, rng)
    s2 = pd.DataFrame(index=df.index)
    s2["theta"] = soil["theta_meas"]
    # ET0 forecast for day t is made on day t-1 by model #1 (out-of-sample for t >= d1)
    et0_hat = pd.Series(m1.predict(data[xcols]), index=data.index).reindex(df.index)
    s2["et0_hat"] = et0_hat.shift(1)
    # noisy rain / temperature "weather-API forecast" for day t
    s2["rain_fc"] = df["rain"] * rng.lognormal(0, 0.3, len(df)) \
        + (rng.random(len(df)) < 0.05) * rng.gamma(1, 3, len(df))
    s2["tmax_fc"] = df["tmax"] + rng.normal(0, 1.0, len(df))
    s2["rh_prev"] = df["rh"].shift(1)
    s2["rs_prev"] = df["rs"].shift(1)
    wetflag = (df["rain"] > 1).astype(int)
    s2["dry_days"] = (1 - wetflag).groupby(wetflag.cumsum()).cumsum().shift(1)
    s2["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    s2["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
    s2["target"] = soil["delta_nat"]
    s2 = s2.dropna()
    scols = [c for c in s2.columns if c != "target"]
    s2_tr = s2[(s2.index >= d1) & (s2.index < d2)]   # period model #1 never saw
    s2_te = s2[s2.index >= d2]

    m2 = xgb.XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                          random_state=args.seed, n_jobs=-1)
    m2.fit(s2_tr[scols], s2_tr["target"])
    ridge2 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(s2_tr[scols], s2_tr["target"])
    q_xgb = m2.predict(s2_te[scols])
    q_ridge = ridge2.predict(s2_te[scols])
    q_persist = soil["delta_nat"].shift(1).reindex(s2_te.index).fillna(0).values
    res2 = {"XGBoost": reg_metrics(s2_te["target"], q_xgb),
            "Ridge (linear baseline)": reg_metrics(s2_te["target"], q_ridge),
            "Persistence (yesterday's change)": reg_metrics(s2_te["target"], q_persist)}
    print("\n=== STAGE 2: next-day soil water change (SIMULATED soil, test set, mm/day) ===")
    for k, v in res2.items():
        print(f"  {k:32s} R2={v['R2']:.3f}  RMSE={v['RMSE']:.3f}  MAE={v['MAE']:.3f}")
    m2.save_model(str(out / "models" / "soil_drying_xgb.json"))

    # ---- decision engine demo on three contrasting test days --------------
    base_ec = 0.4
    picks = {"wettest forecast day": s2_te["rain_fc"].idxmax(),
             "hottest forecast day": s2_te["tmax_fc"].idxmax(),
             "driest soil day": s2_te["theta"].idxmin()}
    decisions = {}
    pred_delta = pd.Series(q_xgb, index=s2_te.index)
    for label, day in picks.items():
        r = s2_te.loc[day]
        decisions[label] = {
            "date": str(day.date()),
            "inputs": {"soil_water_mm": round(float(r["theta"]), 1),
                       "pred_delta_mm": round(float(pred_delta[day]), 2),
                       "rain_forecast_mm": round(float(r["rain_fc"]), 1),
                       "tmax_forecast_C": round(float(r["tmax_fc"]), 1)},
            "recipe": decide(r["theta"], pred_delta[day], r["rain_fc"], r["tmax_fc"], base_ec),
        }
    print("\n=== DECISION ENGINE demo ===")
    for k, v in decisions.items():
        print(f"  [{k}] {v['date']}: {v['recipe']['reason']}")

    # ---- PID demo -----------------------------------------------------------
    t, ec, sp, u_hist, base, pid_metrics = simulate_pid()
    print("\n=== PID loop (simulated plant) ===")
    for k, v in pid_metrics.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    # ---- plots --------------------------------------------------------------
    # 1. ET0 forecast
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    k = min(120, len(te))
    ax[0].plot(te.index[:k], te["target"].iloc[:k], "k-", lw=2, label="Actual (Penman-Monteith)")
    ax[0].plot(te.index[:k], p_xgb[:k], "-", color="#1f77b4", label="XGBoost forecast")
    ax[0].plot(te.index[:k], p_persist[:k], "--", color="#d62728", alpha=0.6, label="Persistence")
    ax[0].set(title="Tomorrow's ET0 - first 120 test days", ylabel="ET0 (mm/day)")
    ax[0].legend(); ax[0].tick_params(axis="x", rotation=30)
    ax[1].scatter(te["target"], p_xgb, s=8, alpha=0.5)
    lim = [te["target"].min(), te["target"].max()]
    ax[1].plot(lim, lim, "k--")
    ax[1].set(title=f"Test set: R2={res1['XGBoost']['R2']:.2f}, RMSE={res1['XGBoost']['RMSE']:.2f} mm/day",
              xlabel="Actual ET0", ylabel="Predicted ET0")
    fig.tight_layout(); fig.savefig(out / "fig1_et0_forecast.png", dpi=160); plt.close(fig)

    # 2. feature importance
    imp = pd.Series(m1.get_booster().get_score(importance_type="gain")).sort_values()[-12:]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color="#1f77b4")
    ax.set(title="Stage 1 feature importance (gain)")
    fig.tight_layout(); fig.savefig(out / "fig2_feature_importance.png", dpi=160); plt.close(fig)

    # 3. soil drying
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(s2_te["target"], q_xgb, s=8, alpha=0.5)
    lim = [s2_te["target"].min(), s2_te["target"].max()]
    ax[0].plot(lim, lim, "k--")
    ax[0].set(title=f"Soil water change (SIMULATED): R2={res2['XGBoost']['R2']:.2f}",
              xlabel="Actual change (mm/day)", ylabel="Predicted change (mm/day)")
    day = picks["hottest forecast day"]
    th0 = float(s2_te.loc[day, "theta"])
    ax[1].plot(range(25), hourly_drying_curve(th0, float(s2_te.loc[day, "target"])), "k-",
               lw=2, label="Actual")
    ax[1].plot(range(25), hourly_drying_curve(th0, float(pred_delta[day])), "-",
               color="#1f77b4", lw=2, label="XGBoost forecast")
    ax[1].axhline((1 - P_DEPLETION) * TAW_MM, color="#d62728", ls="--", label="Stress threshold")
    ax[1].set(title=f"24-h drying curve, {day.date()} (illustrative hourly split)",
              xlabel="Hour of day", ylabel="Root-zone water (mm)")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "fig3_soil_drying.png", dpi=160); plt.close(fig)

    # 4. PID
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(t, sp, "k--", label="EC setpoint")
    ax[0].plot(t, ec, color="#1f77b4", label="Outlet EC")
    ax[0].axvline(450, color="gray", ls=":"); ax[0].axvline(700, color="gray", ls=":")
    ax[0].text(455, 0.5, "FDI output drifts", fontsize=8)
    ax[0].text(705, 0.5, "Heatwave: dilute", fontsize=8)
    ax[0].set(ylabel="EC (dS/m)", title="PID control of Venturi-dosed EC (simulated plant)")
    ax[0].legend()
    ax[1].plot(t, u_hist, color="#2ca02c"); ax[1].set(ylabel="Venturi valve opening", xlabel="Time (s)")
    fig.tight_layout(); fig.savefig(out / "fig4_pid_response.png", dpi=160); plt.close(fig)

    # ---- save everything ----------------------------------------------------
    summary = {
        "synthetic_data": synthetic,
        "site": {"lat": args.lat, "lon": args.lon},
        "data_range": [str(df.index[0].date()), str(df.index[-1].date())],
        "split": {"model1_train_end": str(tr.index[-1].date()),
                  "model2_train": [str(va.index[0].date()), str(va.index[-1].date())],
                  "test": [str(te.index[0].date()), str(te.index[-1].date())]},
        "stage1_et0_forecast_test": res1,
        "stage2_soil_drying_SIMULATED_test": res2,
        "pid_metrics": pid_metrics,
        "assumptions": {"TAW_MM": TAW_MM, "P_DEPLETION": P_DEPLETION, "KC": KC,
                        "RAIN_EFF": RAIN_EFF, "RAIN_SKIP_MM": RAIN_SKIP_MM,
                        "HEAT_TMAX_C": HEAT_TMAX_C, "FERT_EC_INC": FERT_EC_INC},
        "decision_demo": decisions,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"actual": te["target"], "xgb": p_xgb, "persistence": p_persist}
                 ).to_csv(out / "et0_test_predictions.csv")
    print(f"\nDone. Outputs saved in ./{out}/  (results.json, 4 figures, 2 models)")
    if synthetic:
        print("*** REMINDER: data was SYNTHETIC. Re-run with real NASA POWER data before quoting numbers. ***")
    print("Stage-2 soil data is always SIMULATED - say so on your slide.")


if __name__ == "__main__":
    main()

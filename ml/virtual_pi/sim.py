"""Wires the simulated bench to the controllers and steps time forward."""

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from predict_today import load_models
from sedoss_ml import Params, et0_pm, fetch_nasa_power
from sites import get_site, load_index

from .bench import Bench, Sensors, Weather
from .controllers import EventLog, Esp32Dosing, Esp32Treatment, Pi5, prefill_history

ML_DIR = Path(__file__).resolve().parent.parent


def load_weather(site, start, end):
    """NASA POWER daily weather for the site (uses the training cache when present)."""
    cache_dir = ML_DIR / "sedoss_outputs" / "sites" / site["id"] / "cache"
    cached = sorted(cache_dir.glob("nasa_*.csv"))
    if cached:
        df = pd.read_csv(cached[0], index_col=0, parse_dates=True)
    else:
        cache = cache_dir / f"nasa_{site['latitude']}_{site['longitude']}_bench.csv"
        df = fetch_nasa_power(site["latitude"], site["longitude"],
                              start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), cache)
    df = df.copy()
    df["et0"] = et0_pm((df["tmax"] + df["tmin"]) / 2, df["tmax"], df["tmin"], df["rh"], df["u2"], df["rs"],
                       df.index.dayofyear, site["latitude"], site["elevation_m"])
    return df


class Simulation:
    def __init__(self, config_path, site_id=None, start=None, days=None):
        cfg = yaml.safe_load(Path(config_path).read_text())
        if site_id:
            cfg["site"] = site_id
        if start:
            cfg["replay"]["start"] = start
        if days:
            cfg["replay"]["days"] = days
        self.cfg = cfg
        self.site = get_site(cfg["site"], load_index())
        meta, m_fc, m2 = load_models(ML_DIR / self.site["models_dir"])
        self.P = Params(**meta["params"])
        rp = cfg["replay"]
        self.start = pd.Timestamp(rp["start"])
        self.end = self.start + pd.Timedelta(days=rp["days"])
        self.wx = load_weather(self.site, self.start - pd.Timedelta(days=rp["history_days"] + 1), self.end)
        missing = pd.date_range(self.start - pd.Timedelta(days=rp["history_days"]), self.end - pd.Timedelta(days=1))
        if not missing.isin(self.wx.index).all():
            raise ValueError(f"weather data does not cover {missing[0].date()}..{missing[-1].date()}")
        self.rng = np.random.default_rng(cfg.get("seed", 0))
        self.log = EventLog()
        self.bench = Bench(cfg, self.P)
        self.sensors = Sensors(self.bench, cfg, self.rng)
        hist = prefill_history(self.wx, self.start, rp["history_days"], self.rng, cfg["weather_station"])
        self.pi = Pi5(cfg, meta, m_fc, m2, hist, self.wx, self.rng, self.log)
        self.nodes = {"treatment": Esp32Treatment(cfg, self.log), "dosing": Esp32Dosing(cfg, self.log)}
        self.now = self.start
        self._delivered_last = 0.0
        self.daily, self.snapshots, self.deliveries = [], [], []
        self._day = None
        self._next_snapshot = self.start
        self.lock = threading.RLock()
        self.log.add(self.now, "info", "sim", f"Bench simulation started at {self.site['name']} "
                     f"(weather replay {self.start.date()} to {(self.end - pd.Timedelta(days=1)).date()})")

    @property
    def done(self):
        return self.now >= self.end

    # ---- time stepping ---------------------------------------------------------
    def step(self):
        dosing = self.nodes["dosing"]
        dt = 1 if dosing.active or (self.pi.plan and self.now >= self.pi.plan["start"] - pd.Timedelta(minutes=1)) else 60
        now, date = self.now, self.now.normalize()
        if self._day is None or self._day["date"] != date:
            self._close_day()
            self._day = {"date": date, "delivered_l": 0.0, "regens_start": self.nodes["treatment"].regens}
        row = self.wx.loc[date]
        hour = (now - date).total_seconds() / 3600
        s, b = self.sensors, self.bench
        self.pi.log_sensors(now, dt, s, row, hour, b.fdi_mode)
        self.pi.tick(now, dt, s, self.nodes)
        self.nodes["treatment"].step(now, s, b, self.pi.paused)
        res = dosing.step(now, dt, s, b, self.pi.paused, self._delivered_last)
        if res:
            self.pi.last_delivery = res
            self.deliveries.append(res)
        self._delivered_last = b.step(dt, row, hour)
        self._day["delivered_l"] += self._delivered_last
        self.now = now + pd.Timedelta(seconds=dt)
        if self.now >= self._next_snapshot:
            self.snapshots.append(self._snapshot())
            self._next_snapshot += pd.Timedelta(minutes=10)

    def run_until(self, t):
        while self.now < min(t, self.end):
            self.step()

    def run(self):
        self.run_until(self.end)
        self._close_day()

    def _close_day(self):
        d = self._day
        if d is None:
            return
        row = self.wx.loc[d["date"]]
        dec = self.pi.last_decision if self.pi.last_decision and self.pi.last_decision["date"] == str(d["date"].date()) else None
        rec = dec["recipe"] if dec else {}
        self.daily.append({
            "date": str(d["date"].date()), "rain_mm": round(float(row["rain"]), 1), "et0_actual_mm": round(float(row["et0"]), 2),
            "et0_forecast_mm": dec["model_outputs"]["forecast_et0_mm_per_day"] if dec else None,
            "soil_vwc_probe": dec["soil_vwc"] if dec else None,
            "soil_water_mm_at_decision": dec["sensor_and_forecast_inputs"]["soil_water_mm"] if dec else None,
            "soil_water_mm_end_true": round(self.bench.theta, 1),
            "decision": rec.get("scenario"), "volume_mm": rec.get("volume_mm"),
            "ai_ec_setpoint_dS_m": rec.get("ec_setpoint_dS_m") if rec.get("fertigate") else None,
            "applied_ec_setpoint_dS_m": dec.get("applied_ec_setpoint_dS_m") if dec else None,
            "delivered_l": round(d["delivered_l"], 2), "buffer_l": round(self.bench.buffer_l, 2),
            "brine_l": round(self.bench.brine_l, 2), "fdi_regenerations": self.nodes["treatment"].regens - d["regens_start"],
            "salt_recovered_g": round(self.bench.salt_recovered_g, 1)})
        self._day = None

    def _snapshot(self):
        b = self.bench
        return {"t": self.now.isoformat(), "soil_water_mm": round(b.theta, 2), "buffer_l": round(b.buffer_l, 3),
                "brine_l": round(b.brine_l, 3), "feed_l": round(b.feed_l, 2), "fdi_mode": b.fdi_mode,
                "fdi_load_g": round(b.fdi_load, 3), "outlet_ec": round(b.fdi_out_c / self.P.ec_to_ppt, 3),
                "buffer_ec": round(b.buffer_ec, 3), "salt_recovered_g": round(b.salt_recovered_g, 2)}

    # ---- website-facing ---------------------------------------------------------------
    def command(self, cmd, args=None):
        with self.lock:
            self.pi.command(self.now, cmd, args or {}, self.nodes, self.bench)

    def state(self):
        with self.lock:
            b, s, P = self.bench, self.sensors, self.P
            date = self.now.normalize()
            row = self.wx.loc[min(date, self.end - pd.Timedelta(days=1))]
            hour = (self.now - date).total_seconds() / 3600
            air = s.air_temp(row, hour)
            d = self.cfg["dosing"]
            raw = s.soil_probe_raw()
            sp = self.cfg["soil_probe"]
            vwc = sp["vwc_dry"] + (raw - sp["raw_dry"]) / (sp["raw_wet"] - sp["raw_dry"]) * (sp["vwc_wet"] - sp["vwc_dry"])
            return {
                "time": self.now.isoformat(), "finished": self.done, "paused": self.pi.paused,
                "site": {k: self.site[k] for k in ("id", "name", "latitude", "longitude")},
                "sensors": {"feed_ec_dS_m": round(s.feed_ec(), 3), "outlet_ec_dS_m": round(s.outlet_ec(), 3),
                            "final_ec_dS_m": round(s.final_ec(), 3), "ph": round(s.ph(), 2),
                            "water_temp_c": round(s.water_temp(air), 1), "air_temp_c": round(air, 1),
                            "rh_pct": round(s.rh(row), 1), "solar_w_m2": round(s.solar_w_m2(row, hour), 0),
                            "soil_probe_raw": raw, "soil_vwc": round(vwc, 3),
                            "buffer_high_float": s.high_float(), "buffer_low_float": s.low_float(),
                            "feed_flow_l_min": round(s.flow_in(), 3)},
                "actuators": {"feed_pump": b.feed_pump, "fdi_mode": b.fdi_mode, "three_way_valve": b.valve,
                              "delivery_pump": b.delivery_pump,
                              "dosing_pump_a_ml_min": round(b.dose_u * d["pump_max_ml_min"], 2),
                              "dosing_pump_b_ml_min": round(b.dose_u * d["pump_max_ml_min"], 2)},
                "tanks_l": {"feed": round(b.feed_l, 2), "buffer": round(b.buffer_l, 2), "brine": round(b.brine_l, 2),
                            "evaporation_trays": round(b.tray_l, 2), "nutrient_a": round(b.stock_a_l, 3),
                            "nutrient_b": round(b.stock_b_l, 3)},
                "salt_recovered_g": round(b.salt_recovered_g, 1),
                "fdi_last_batch_ec_dS_m": round(float(np.mean(self.pi.outlet_ec_recent)), 3) if self.pi.outlet_ec_recent else None,
                "fdi_regenerations": self.nodes["treatment"].regens,
                "today_plan": self.pi.plan_public(), "delivery_in_progress": self.nodes["dosing"].active,
                "delivery": self._delivery_public(),
                "farmer_overrides_today": {"skip": self.pi.overrides.get("skip") == date,
                                           "water_only": self.pi.overrides.get("water_only") == date},
                "replay": {"start": str(self.start.date()), "end": str((self.end - pd.Timedelta(days=1)).date())},
                "last_decision": self.pi.last_decision, "last_delivery": self.pi.last_delivery,
                "crop_safe_ec_limit_dS_m": round(self.pi.ec_limit, 2),
                "simulation_truth": {"_note": "hidden values a real bench cannot measure directly",
                                     "soil_water_mm": round(b.theta, 1), "stress_threshold_mm": round(P.raw_thr, 1),
                                     "fdi_electrode_load_g": round(b.fdi_load, 2), "buffer_ec_dS_m": round(b.buffer_ec, 3)},
            }

    def _delivery_public(self):
        j = self.nodes["dosing"].job
        if not j:
            return None
        return {"id": j["id"], "volume_l": round(j["volume_l"], 2), "delivered_l": round(j["delivered_l"], 2),
                "ec_setpoint_dS_m": j["ec_setpoint_dS_m"], "started": j["started"], "scenario": j.get("scenario")}

    def save(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.daily).to_csv(out / "daily_log.csv", index=False)
        pd.DataFrame(self.snapshots).to_csv(out / "snapshots_10min.csv", index=False)
        with open(out / "events.jsonl", "w") as f:
            for e in self.log.items:
                f.write(json.dumps(e, default=str) + "\n")
        dl = [x for x in self.deliveries if x.get("final_ec_mean_dS_m") is not None]
        days = pd.DataFrame(self.daily)
        summary = {
            "_note": "SIMULATED bench. Physical constants in config/bench.yaml are placeholders.",
            "site": self.site["name"], "replay": [str(self.start.date()), str((self.end - pd.Timedelta(days=1)).date())],
            "days": len(days), "irrigation_days": int((days["delivered_l"] > 0).sum()),
            "water_delivered_l": round(float(days["delivered_l"].sum()), 2),
            "rain_mm": round(float(days["rain_mm"].sum()), 1),
            "stress_days_true": int((days["soil_water_mm_end_true"] < self.P.raw_thr).sum()),
            "treated_water_l": round(self.bench.treated_l, 2), "diverted_to_brine_l": round(self.bench.diverted_l, 2),
            "fdi_regenerations": self.nodes["treatment"].regens,
            "salt_recovered_g": round(self.bench.salt_recovered_g, 1),
            "dosing": {"deliveries_with_fertilizer": len(dl),
                       "mean_abs_ec_error_dS_m": round(float(np.mean([abs(x["final_ec_mean_dS_m"] - x["ec_setpoint_dS_m"]) for x in dl])), 3) if dl else None,
                       "mean_settle_s": round(float(np.mean([x["settle_s"] for x in dl])), 1) if dl else None},
            "alarms": [e for e in self.log.items if e["level"] == "alarm"],
            "warnings": sorted({e["msg"] for e in self.log.items if e["level"] == "warn"}),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

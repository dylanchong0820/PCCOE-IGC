"""
Control code for the bench: one class per board on the parts list.

  Esp32Treatment - ESP32 #1: feed pump, FDI treat/regenerate, 3-way valve, float switches
  Esp32Dosing    - ESP32 #2: delivery pump, peristaltic dosing pumps A/B (PID on final EC)
  Pi5            - Raspberry Pi 5: daily AI decision (trained site models), scheduling,
                   safety interlocks, logging, commands from the website

They only use sensor readings and actuator settings, so the same logic can move to real
boards later (ESP32 code would be ported to C++/MicroPython; the Pi code can stay Python).
"""

import dataclasses
import numpy as np
import pandas as pd

from predict_today import REQUIRED, recommend
from sedoss_ml import PID


class EventLog:
    def __init__(self):
        self.items = []

    def add(self, t, level, source, msg, **data):
        self.items.append({"t": t.isoformat(), "level": level, "source": source, "msg": msg, **data})


class Esp32Treatment:
    name = "esp32_treatment"

    def __init__(self, cfg, log):
        self.cfg, self.log = cfg, log
        self.filling = False
        self.regen_started = None
        self.regens = 0
        self.force_regen = False
        self._alarm = set()

    def request_fill(self):
        self.filling = True

    def step(self, now, s, bench, paused):
        c = self.cfg["fdi"]
        if paused:
            bench.feed_pump, bench.fdi_mode = False, "idle"
            return
        # feed tank
        if bench.feed_l <= self.cfg["feed_tank"]["low_alarm_l"]:
            if self.cfg["feed_tank"].get("auto_refill"):
                bench.feed_l = self.cfg["feed_tank"]["capacity_l"]
                self.log.add(now, "info", "operator", "Feed tank refilled (auto, batch run)")
            elif "feed_low" not in self._alarm:
                self._alarm.add("feed_low")
                self.log.add(now, "alarm", self.name, "Feed tank low - treatment stopped, refill needed")
        else:
            self._alarm.discard("feed_low")
        feed_ok = "feed_low" not in self._alarm

        if bench.fdi_mode == "regen":
            if bench.brine_l >= self.cfg["brine_tank"]["capacity_l"]:
                bench.feed_pump, bench.fdi_mode = False, "idle"
                if "brine_full" not in self._alarm:
                    self._alarm.add("brine_full")
                    self.log.add(now, "alarm", self.name, "Brine tank full - regeneration paused")
                return
            self._alarm.discard("brine_full")
            done = (now - self.regen_started).total_seconds() > 120 and s.outlet_ec() <= s.feed_ec() * 1.05
            if done:
                bench.fdi_mode, bench.valve = "idle", "buffer"
                self.log.add(now, "info", self.name, "FDI regeneration finished")
            return

        if s.low_float():
            self.filling = True
        if s.high_float():
            self.filling = False
        if self.force_regen or (bench.fdi_mode == "treat" and s.outlet_ec() > c["outlet_ec_limit_dS_m"]):
            reason = "manual request" if self.force_regen else f"outlet EC {s.outlet_ec():.2f} > {c['outlet_ec_limit_dS_m']} dS/m"
            self.force_regen = False
            bench.fdi_mode, bench.valve, bench.feed_pump = "regen", "brine", True
            self.regen_started = now
            self.regens += 1
            self.log.add(now, "info", self.name, f"Electrodes saturated ({reason}): valve -> brine, FDI regenerating")
            return
        if self.filling and feed_ok:
            bench.fdi_mode, bench.valve, bench.feed_pump = "treat", "buffer", True
        else:
            bench.fdi_mode, bench.feed_pump = "idle", False


class Esp32Dosing:
    name = "esp32_dosing"

    def __init__(self, cfg, log):
        self.cfg, self.log = cfg, log
        self.job = None
        self.pid = None
        self._waiting = False

    @property
    def active(self):
        return self.job is not None

    def start(self, now, job):
        d = self.cfg["dosing"]
        self.job = {**job, "delivered_l": 0.0, "started": now.isoformat(), "ec_samples": [], "ph": []}
        self.pid = PID(d["pid"]["kp"], d["pid"]["ki"], d["pid"]["kd"], dt=1.0)
        sp = job["ec_setpoint_dS_m"]
        self.log.add(now, "info", self.name,
                     f"Delivering {job['volume_l']:.2f} L" + (f", dosing to EC {sp:.2f} dS/m" if sp else ", water only"))

    def step(self, now, dt, s, bench, paused, delivered_last):
        if self.job is None:
            bench.delivery_pump, bench.dose_u = False, 0.0
            return None
        j, d = self.job, self.cfg["dosing"]
        j["delivered_l"] += delivered_last
        if j["delivered_l"] >= j["volume_l"]:
            return self._finish(now, bench)
        if paused:
            bench.delivery_pump, bench.dose_u = False, 0.0
            return None
        if s.low_float():
            bench.delivery_pump, bench.dose_u = False, 0.0
            if not self._waiting:
                self._waiting = True
                self.log.add(now, "warn", self.name, "Buffer tank low - delivery waiting for treated water")
            return None
        self._waiting = False
        bench.delivery_pump = True
        sp = j["ec_setpoint_dS_m"]
        stock_ok = min(bench.stock_a_l, bench.stock_b_l) > d["stock_low_alarm_l"]
        if sp and stock_ok:
            ec = s.final_ec()
            bench.dose_u = self.pid.step(sp, ec)
            j["ec_samples"].append(ec)
            j["ph"].append(s.ph())
        else:
            bench.dose_u = 0.0
            if sp and not stock_ok and not j.get("stock_alarm"):
                j["stock_alarm"] = True
                self.log.add(now, "alarm", self.name, "Nutrient stock low - delivering water only")
        return None

    def _finish(self, now, bench):
        j = self.job
        bench.delivery_pump, bench.dose_u = False, 0.0
        sp = j["ec_setpoint_dS_m"]
        ecs = np.array(j["ec_samples"])
        res = {"id": j["id"], "volume_l": round(j["delivered_l"], 2), "ec_setpoint_dS_m": sp,
               "started": j["started"], "finished": now.isoformat()}
        if sp and len(ecs) > 10:
            tail = ecs[len(ecs) // 2:]                    # second half: after the loop has settled
            band = 0.04                                   # dS/m, same band as the training PID test
            settled = np.where(np.abs(ecs - sp) > band)[0]
            res.update(final_ec_mean_dS_m=round(float(tail.mean()), 3),
                       final_ec_max_dS_m=round(float(ecs.max()), 3),
                       settle_s=int(settled[-1] + 1) if len(settled) else 0,
                       ph_min=round(float(min(j["ph"])), 2), ph_max=round(float(max(j["ph"])), 2))
            lo, hi = self.cfg["dosing"]["ph_range"]
            if res["ph_min"] < lo or res["ph_max"] > hi:
                self.log.add(now, "alarm", self.name, f"pH left the safe range {lo}-{hi}")
        self.log.add(now, "info", self.name, f"Delivery finished: {res['volume_l']} L", result=res)
        self.job = None
        return res


class Pi5:
    name = "pi5"

    def __init__(self, cfg, meta, m_fc, m2, history, weather_df, rng, log):
        self.cfg, self.meta, self.m_fc, self.m2 = cfg, meta, m_fc, m2
        self.P = __import__("sedoss_ml").Params(**meta["params"])
        self.history = history                  # daily weather the Pi has logged (REQUIRED cols)
        self.wx = weather_df                    # "weather API": wind, rain, forecast (NASA replay)
        self.rng, self.log = rng, log
        self.paused = False
        self.today = None                       # running aggregation of today's sensor readings
        self.plan = None                        # today's job, waiting for its start time
        self.last_decision = None
        self.last_delivery = None
        self.overrides = {}
        self.irrigations = 0
        self.outlet_ec_recent = []
        self.ec_limit = self.P.ec_limit_ppt / self.P.ec_to_ppt
        sch = cfg["schedule"]
        self.t_decide = pd.Timedelta(sch["decision_time"] + ":00")
        self.t_morning = pd.Timedelta(sch["morning_time"] + ":00")
        self.t_evening = pd.Timedelta(sch["evening_time"] + ":00")

    # ---- sensor logging --------------------------------------------------------
    def log_sensors(self, now, dt, s, row, hour, treatment_mode):
        if self.today is None or self.today["date"] != now.normalize():
            self._close_day(now)
            self.today = {"date": now.normalize(), "tmax": -99.0, "tmin": 99.0, "rh_sum": 0.0, "n": 0, "rs_mj": 0.0}
        t = s.air_temp(row, hour)
        a = self.today
        a["tmax"], a["tmin"] = max(a["tmax"], t), min(a["tmin"], t)
        a["rh_sum"] += s.rh(row)
        a["n"] += 1
        a["rs_mj"] += s.solar_w_m2(row, hour) * dt / 1e6
        if treatment_mode == "treat":
            self.outlet_ec_recent = (self.outlet_ec_recent + [s.outlet_ec()])[-240:]

    def _close_day(self, now):
        a = self.today
        if a is None or a["n"] == 0:
            return
        d = a["date"]
        api = self.wx.loc[d]
        self.history.loc[d] = {"tmax": a["tmax"], "tmin": a["tmin"], "rh": a["rh_sum"] / a["n"],
                               "u2": api["u2"], "rs": a["rs_mj"], "rain": api["rain"]}

    # ---- daily AI decision ----------------------------------------------------
    def forecast_today(self, date):
        fe, r = self.cfg["forecast_error"], self.wx.loc[date]
        n = self.rng.normal
        return {"tmax": float(r["tmax"] + n(0, fe["tmax_sd_c"])), "tmin": float(r["tmin"] + n(0, fe["tmin_sd_c"])),
                "rh": float(np.clip(r["rh"] + n(0, fe["rh_sd_pct"]), 0, 100)),
                "u2": float(max(r["u2"] * (1 + n(0, fe["u2_rel_sd"])), 0.1)),
                "rs": float(max(r["rs"] * (1 + n(0, fe["rs_rel_sd"])), 1.0)),
                "rain": float(max(r["rain"] * (1 + n(0, fe["rain_rel_sd"])), 0.0))}

    def decide(self, now, s, treatment):
        date = now.normalize()
        sp = self.cfg["soil_probe"]
        raw = s.soil_probe_raw()
        vwc = sp["vwc_dry"] + (raw - sp["raw_dry"]) / (sp["raw_wet"] - sp["raw_dry"]) * (sp["vwc_wet"] - sp["vwc_dry"])
        outlet = float(np.mean(self.outlet_ec_recent)) if self.outlet_ec_recent else self.P.base_ec
        fert_due = (self.irrigations % max(self.P.fert_every_n, 1)) == 0
        if self.overrides.get("water_only") == date:
            fert_due = False
        readings = {"date": str(date.date()), "soil_vwc": float(vwc), "outlet_ec_dS_m": outlet,
                    "forecast_today": self.forecast_today(date), "fertigation_due": fert_due}
        hist = self.history.loc[:date - pd.Timedelta(days=1)].tail(30)
        out = recommend(hist, readings, self.meta, self.m_fc, self.m2)
        rec = out["recipe"]
        out["soil_probe_raw"] = raw
        out["soil_vwc"] = round(float(vwc), 3)
        self.last_decision = {"time": now.isoformat(), **out}
        self.log.add(now, "info", self.name, f"AI decision: {rec['reason']}", decision=out)

        if self.overrides.get("skip") == date:
            self.log.add(now, "info", self.name, "Skipped today by the farmer (website command)")
            self.plan = None
            return out
        if not rec["irrigate"]:
            self.plan = None
            return out
        setpoint = rec["ec_setpoint_dS_m"] if rec["fertigate"] else None
        if setpoint and setpoint > self.ec_limit:
            self.log.add(now, "warn", self.name,
                         f"Interlock: EC setpoint {setpoint:.2f} dS/m is above the crop-safe limit "
                         f"{self.ec_limit:.2f} dS/m ({self.P.ec_limit_ppt} ppt) - capped")
            setpoint = round(self.ec_limit, 2)
        self.last_decision["applied_ec_setpoint_dS_m"] = setpoint
        start = date + (self.t_evening if rec["timing"] == "evening" else self.t_morning)
        self.plan = {"id": f"{date.date()}-{rec['timing']}", "start": start, "volume_mm": rec["volume_mm"],
                     "volume_l": rec["volume_mm"] * self.cfg["test_bed"]["area_m2"], "ec_setpoint_dS_m": setpoint,
                     "dose_factor": rec["venturi_dose_factor"], "timing": rec["timing"], "scenario": rec["scenario"]}
        self.irrigations += 1
        treatment.request_fill()                # top up the buffer before the job starts
        return out

    def plan_public(self):
        if not self.plan:
            return None
        return {**{k: v for k, v in self.plan.items() if k != "start"}, "start": self.plan["start"].isoformat()}

    # ---- commands from the website ------------------------------------------------
    def command(self, now, cmd, args, nodes, bench):
        date = now.normalize()
        if cmd == "pause":
            self.paused = True
        elif cmd == "resume":
            self.paused = False
        elif cmd == "skip_today":
            self.overrides["skip"] = date
            self.plan = None
        elif cmd == "water_only_today":
            self.overrides["water_only"] = date
            if self.plan:
                self.plan["ec_setpoint_dS_m"] = None
        elif cmd == "irrigate_now":
            mm = float(args.get("mm", 5.0))
            if not 0 < mm <= self.P.max_irrig:
                raise ValueError(f"mm must be between 0 and {self.P.max_irrig}")
            dose = bool(args.get("fertigate", False))
            sp = round(min(self.P.base_ec + self.P.fert_inc, self.ec_limit), 2) if dose else None
            self.plan = {"id": f"manual-{now.isoformat()}", "start": now, "volume_mm": mm,
                         "volume_l": mm * self.cfg["test_bed"]["area_m2"], "ec_setpoint_dS_m": sp,
                         "dose_factor": 1.0 if dose else 0.0, "timing": "now", "scenario": "manual"}
        elif cmd == "regenerate_fdi":
            nodes["treatment"].force_regen = True
        elif cmd == "refill_feed":
            bench.feed_l = self.cfg["feed_tank"]["capacity_l"]
        elif cmd == "refill_nutrients":
            bench.stock_a_l, bench.stock_b_l = self.cfg["dosing"]["stock_a_l"], self.cfg["dosing"]["stock_b_l"]
        else:
            raise ValueError(f"unknown command '{cmd}'")
        self.log.add(now, "info", "website", f"Command: {cmd}", args=args)

    # ---- per tick ---------------------------------------------------------------
    def tick(self, now, dt, s, nodes):
        date = now.normalize()
        t_decide = date + self.t_decide
        if now >= t_decide and (self.last_decision is None or self.last_decision["date"] != str(date.date())):
            self.decide(now, s, nodes["treatment"])
        if self.plan and now >= self.plan["start"] and not nodes["dosing"].active and not self.paused:
            nodes["dosing"].start(now, self.plan)
            self.plan = None


def prefill_history(wx, start, days, rng, ws):
    """Weather the Pi had logged before the replay starts (sensor noise added)."""
    h = wx.loc[start - pd.Timedelta(days=days):start - pd.Timedelta(days=1), REQUIRED].copy()
    h["tmax"] += rng.normal(0, ws["t_noise_c"], len(h))
    h["tmin"] += rng.normal(0, ws["t_noise_c"], len(h))
    h["rh"] = np.clip(h["rh"] + rng.normal(0, ws["rh_noise_pct"], len(h)), 0, 100)
    h["rs"] *= 1 + rng.normal(0, ws["rs_noise_frac"], len(h))
    return h

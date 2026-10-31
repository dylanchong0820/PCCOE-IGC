"""
Simulated physics of the prototype bench and the sensors that read it (all SIMULATED).

The controllers in controllers.py only see sensor readings and only set actuators, the same
interface they would have on real hardware. Everything in this file is what the real bench
would replace.

Salt is tracked in grams; EC (dS/m) <-> concentration uses g/L = EC x ec_to_ppt.
"""

import math
from collections import deque


class Weather:
    """Replays NASA POWER daily weather as a smooth intraday signal."""

    def __init__(self, df):
        self.df = df

    def day(self, date):
        return self.df.loc[date]

    @staticmethod
    def air_temp(row, hour):
        # minimum around 06:00, maximum around 14:00
        lo, hi = row["tmin"], row["tmax"]
        if 6 <= hour < 14:
            f = (1 - math.cos(math.pi * (hour - 6) / 8)) / 2
        else:
            h = (hour - 14) % 24
            f = (1 + math.cos(math.pi * h / 16)) / 2
        return lo + (hi - lo) * f

    @staticmethod
    def sun_weight(hour):
        """Share of the day's sunshine per hour (06:00-18:00 sine), integrates to 1 over the day."""
        if 6 <= hour < 18:
            return math.sin(math.pi * (hour - 6) / 12) * math.pi / 24
        return 0.0


class Bench:
    """Tanks, FDI cell, mixer, test bed and evaporation trays."""

    def __init__(self, cfg, P):
        self.cfg, self.P = cfg, P
        g = P.ec_to_ppt
        f, b = cfg["feed_tank"], cfg["buffer_tank"]
        self.feed_l = f["initial_l"]
        self.feed_c = f["ec_dS_m"] * g            # g/L
        self.buffer_l = b["initial_l"]
        self.buffer_salt = self.buffer_l * P.base_ec * g
        self.brine_l, self.brine_salt = 0.0, 0.0
        self.tray_l, self.tray_salt, self.salt_recovered_g = 0.0, 0.0, 0.0
        self.fdi_load = 0.0                      # g of salt held on the electrodes
        self.fdi_out_c = P.base_ec * g           # g/L in the cell outlet (holds last value when idle)
        self.stock_a_l = cfg["dosing"]["stock_a_l"]
        self.stock_b_l = cfg["dosing"]["stock_b_l"]
        self.theta = cfg["test_bed"]["initial_fraction"] * P.taw   # soil water, mm
        self.mix_ec = P.base_ec                  # EC at the final sensor (after mixer lag)
        self.mix_ph = cfg["feed_tank"]["ph"]
        d = cfg["dosing"]
        self._mix_queue = deque([(P.base_ec, cfg["feed_tank"]["ph"])] * max(1, int(d["mixer_delay_s"])))
        # actuators (set by controllers)
        self.feed_pump = False
        self.fdi_mode = "idle"                   # idle | treat | regen
        self.valve = "buffer"                    # buffer | brine
        self.delivery_pump = False
        self.dose_u = 0.0                        # 0..1 for both dosing pumps
        # counters
        self.treated_l = self.diverted_l = self.delivered_l_total = 0.0
        self.flow_in_l_min = self.flow_out_l_min = 0.0

    # ---- derived ---------------------------------------------------------------
    def ec(self, salt_g, vol_l):
        return (salt_g / vol_l) / self.P.ec_to_ppt if vol_l > 1e-9 else 0.0

    @property
    def buffer_ec(self):
        return self.ec(self.buffer_salt, self.buffer_l)

    # ---- physics step ------------------------------------------------------------
    def step(self, dt, row, hour):
        cfg, P = self.cfg, self.P
        self._fdi(dt)
        delivered_l = self._delivery(dt)
        self._soil(dt, row, hour, delivered_l)
        self._evaporation(dt, row, hour)
        return delivered_l

    def _fdi(self, dt):
        c, f = self.cfg["fdi"], self.cfg
        self.flow_in_l_min = self.flow_out_l_min = 0.0
        if not self.feed_pump or self.fdi_mode == "idle" or self.feed_l <= 0:
            return
        flow = c["regen_flow_l_min"] if self.fdi_mode == "regen" else f["feed_pump"]["flow_l_min"]
        q = min(flow * dt / 60, self.feed_l)
        salt_in = self.feed_c * q
        self.feed_l -= q
        if self.fdi_mode == "treat":
            frac = min(self.fdi_load / c["capacity_g"], 1.0)
            r = c["max_removal"] * (1 - frac ** c["saturation_shape"])
            removed = salt_in * r
            self.fdi_load += removed
            salt_out = salt_in - removed
        else:                                    # regenerate: electrodes release salt
            release = min(self.fdi_load, c["regen_release_g_min"] * dt / 60)
            self.fdi_load -= release
            salt_out = salt_in + release
        self.fdi_out_c = salt_out / q if q > 0 else 0.0
        self.flow_in_l_min = self.flow_out_l_min = q * 60 / dt
        if self.valve == "buffer":
            room = self.cfg["buffer_tank"]["capacity_l"] - self.buffer_l
            take = min(q, max(room, 0.0))
            self.buffer_l += take
            self.buffer_salt += salt_out * (take / q)
            self.treated_l += take
            spill = q - take
            if spill > 0:                        # overflow goes to brine (real bench: overflow line)
                self.brine_l += spill
                self.brine_salt += salt_out * (spill / q)
        else:
            self.brine_l += q
            self.brine_salt += salt_out
            self.diverted_l += q

    def _delivery(self, dt):
        d = self.cfg["dosing"]
        q = 0.0
        if self.delivery_pump and self.buffer_l > 0.05:
            q = min(self.cfg["delivery_pump"]["flow_l_min"] * dt / 60, self.buffer_l)
        buf_ec = self.buffer_ec
        qd_each = self.dose_u * d["pump_max_ml_min"] / 1000 * dt / 60 if q > 0 else 0.0
        qd_each = min(qd_each, self.stock_a_l, self.stock_b_l)
        qd = 2 * qd_each
        if q > 0:
            self.buffer_salt -= self.buffer_salt * (q / self.buffer_l)
            self.buffer_l -= q
            self.stock_a_l -= qd_each
            self.stock_b_l -= qd_each
            inst_ec = (q * buf_ec + qd * d["stock_ec_dS_m"]) / (q + qd)
            inst_ph = self.cfg["feed_tank"]["ph"] - d["ph_drop_per_dS_m"] * (inst_ec - buf_ec)
        else:
            inst_ec, inst_ph = self.mix_ec, self.mix_ph   # no flow: sensor holds last value
        # transport delay + first-order lag through the static mixer
        self._mix_queue.append((inst_ec, inst_ph))
        ec_d, ph_d = self._mix_queue.popleft()
        a = min(dt / d["mixer_tau_s"], 1.0)
        self.mix_ec += a * (ec_d - self.mix_ec)
        self.mix_ph += a * (ph_d - self.mix_ph)
        self.delivered_l_total += q + qd
        return q + qd

    def _soil(self, dt, row, hour, delivered_l):
        P, tb = self.P, self.cfg["test_bed"]
        self.theta += delivered_l / tb["area_m2"]          # 1 L on 1 m2 = 1 mm
        if tb["receives_rain"] and int(hour) == 15 and (hour - 15) * 3600 < dt:
            self.theta += P.rain_eff * row["rain"]         # tropical afternoon shower
        ks = min(max(self.theta / P.raw_thr, 0.0), 1.0) if P.raw_thr > 0 else 1.0
        self.theta -= P.kc * ks * row["et0"] * Weather.sun_weight(hour) * dt / 3600
        self.theta = min(max(self.theta, 0.0), P.taw)      # excess drains below the roots

    def _evaporation(self, dt, row, hour):
        e = self.cfg["evaporation"]
        move = min(self.brine_l, e["brine_to_tray_l_h"] * dt / 3600, max(e["tray_capacity_l"] - self.tray_l, 0))
        if move > 0:
            self.tray_salt += self.brine_salt * (move / self.brine_l)
            self.brine_salt -= self.brine_salt * (move / self.brine_l)
            self.brine_l -= move
            self.tray_l += move
        evap = row["et0"] * e["photothermal_boost"] * Weather.sun_weight(hour) * dt / 3600 * e["tray_area_m2"]
        self.tray_l = max(self.tray_l - evap, 0.0)
        if self.tray_l < 1e-6:
            self.salt_recovered_g += self.tray_salt
            self.tray_salt = 0.0
        elif self.tray_salt / self.tray_l > e["crystallize_g_l"]:
            excess = self.tray_salt - e["crystallize_g_l"] * self.tray_l
            self.salt_recovered_g += excess
            self.tray_salt -= excess


class Sensors:
    """What the ESP32s and the Pi can read. Adds noise; never exposes hidden state directly."""

    def __init__(self, bench, cfg, rng):
        self.b, self.cfg, self.rng = bench, cfg, rng

    def _n(self, sd):
        return float(self.rng.normal(0, sd))

    # Stage 2 / 3 (ESP32 #1)
    def feed_ec(self):
        return max(self.b.feed_c / self.b.P.ec_to_ppt + self._n(0.01), 0.0)

    def outlet_ec(self):
        return max(self.b.fdi_out_c / self.b.P.ec_to_ppt + self._n(0.01), 0.0)

    def high_float(self):
        return self.b.buffer_l >= self.cfg["buffer_tank"]["high_float_frac"] * self.cfg["buffer_tank"]["capacity_l"]

    def low_float(self):
        return self.b.buffer_l <= self.cfg["buffer_tank"]["low_float_frac"] * self.cfg["buffer_tank"]["capacity_l"]

    def flow_in(self):
        return self.b.flow_in_l_min

    # Stage 4 (ESP32 #2)
    def final_ec(self):
        return max(self.b.mix_ec + self._n(0.01), 0.0)

    def ph(self):
        return self.b.mix_ph + self._n(0.02)

    def water_temp(self, air_c):
        return air_c - 2.0 + self._n(0.2)

    def soil_probe_raw(self):
        """ADS1115 counts from the capacitive probe (lower counts = wetter soil)."""
        P, sp = self.b.P, self.cfg["soil_probe"]
        vwc = P.vwc_wp + (self.b.theta / P.taw) * (P.vwc_fc - P.vwc_wp)
        frac = (vwc - sp["vwc_dry"]) / (sp["vwc_wet"] - sp["vwc_dry"])
        return int(round(sp["raw_dry"] + frac * (sp["raw_wet"] - sp["raw_dry"]) + self._n(sp["noise_counts"])))

    # Stage 5 (Pi)
    def air_temp(self, row, hour):
        return Weather.air_temp(row, hour) + self._n(self.cfg["weather_station"]["t_noise_c"])

    def rh(self, row):
        return min(max(row["rh"] + self._n(self.cfg["weather_station"]["rh_noise_pct"]), 0), 100)

    def solar_w_m2(self, row, hour):
        """Instantaneous solar radiation (W/m2) from the day's MJ/m2 total."""
        w = Weather.sun_weight(hour) * row["rs"] * 1e6 / 3600
        return max(w * (1 + self._n(self.cfg["weather_station"]["rs_noise_frac"])), 0.0)

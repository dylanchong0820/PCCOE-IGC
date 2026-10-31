#!/usr/bin/env python3
"""
Virtual Pi: simulate the S.E.D.O.S.S. prototype bench (parts list in config/bench.yaml).

The Raspberry Pi 5 logic runs the trained AI for a site each morning, the two ESP32s run the
treatment and dosing hardware, and the bench physics stands in for the real tanks, FDI cell,
pumps and sensors. Everything is SIMULATED; the physical constants are placeholders.

Usage
-----
  python3 run_bench.py                         # 60-day replay, results in sedoss_outputs/bench_sim/
  python3 run_bench.py --site raub --start 2016-02-15 --days 45
  python3 run_bench.py --serve                 # live API for the website on http://127.0.0.1:8765
  python3 run_bench.py --serve --speed 60      # 1 simulated minute per real second

Needs trained sites first: python3 train_sites.py
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from virtual_pi.sim import Simulation  # noqa: E402


def plot(sim, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    days = pd.DataFrame(sim.daily)
    days["date"] = pd.to_datetime(days["date"])
    snaps = pd.DataFrame(sim.snapshots)
    snaps["t"] = pd.to_datetime(snaps["t"])
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    a = ax[0, 0]
    a.plot(snaps["t"], snaps["soil_water_mm"], color="#2b6a4f", label="Soil water (simulated, mm)")
    a.axhline(sim.P.raw_thr, color="#b33", ls="--", lw=1, label="Stress threshold")
    a2 = a.twinx()
    a2.bar(days["date"], days["rain_mm"], color="#7aa6c2", alpha=0.5, width=0.8, label="Rain (mm)")
    irr = days[days["delivered_l"] > 0]
    a.scatter(irr["date"] + pd.Timedelta(hours=7), [sim.P.raw_thr] * len(irr), marker="v", color="#e08a00",
              zorder=5, label="Irrigation")
    a.set(title="Test bed: soil water, rain and AI irrigation", ylabel="mm")
    a2.set_ylabel("rain mm")
    a.legend(loc="lower left", fontsize=8)

    a = ax[0, 1]
    a.plot(snaps["t"], snaps["buffer_l"], color="#1f5fa8", label="Buffer tank (L)")
    a.plot(snaps["t"], snaps["brine_l"], color="#8a5a2b", label="Brine tank (L)")
    a.plot(snaps["t"], snaps["feed_l"], color="#888", lw=0.8, label="Feed tank (L)")
    a.set(title="Tanks", ylabel="litres")
    a.legend(fontsize=8)

    a = ax[1, 0]
    a.plot(snaps["t"], snaps["outlet_ec"].where(snaps["fdi_mode"] == "treat"), ".", ms=2, color="#1f5fa8",
           label="FDI outlet EC while treating")
    a.axhline(sim.cfg["fdi"]["outlet_ec_limit_dS_m"], color="#b33", ls="--", lw=1, label="Regenerate above")
    a.plot(snaps["t"], snaps["buffer_ec"], color="#2b6a4f", label="Buffer water EC")
    a.set(title="Desalination (FDI) output", ylabel="EC dS/m")
    a.legend(fontsize=8)

    a = ax[1, 1]
    a.plot(days["date"], days["salt_recovered_g"], color="#8a5a2b")
    a.set(title="Salt recovered from brine (cumulative)", ylabel="grams")

    fig.suptitle(f"S.E.D.O.S.S. virtual bench - {sim.site['name']} weather - SIMULATED, placeholder constants",
                 fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "fig_bench.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="S.E.D.O.S.S. virtual Pi / bench simulator")
    ap.add_argument("--config", default=str(HERE / "config" / "bench.yaml"))
    ap.add_argument("--site", help="trained site id (overrides the config)")
    ap.add_argument("--start", help="weather replay start date, YYYY-MM-DD")
    ap.add_argument("--days", type=int)
    ap.add_argument("--out", default=str(HERE / "sedoss_outputs" / "bench_sim"))
    ap.add_argument("--serve", action="store_true", help="run live and serve the website API")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--speed", type=float, default=600, help="--serve: simulated seconds per real second")
    args = ap.parse_args()

    sim = Simulation(args.config, args.site, args.start, args.days)
    if args.serve:
        from virtual_pi.server import serve
        serve(lambda site: Simulation(args.config, site, args.start, args.days), sim,
              args.host, args.port, args.speed)
        return

    sim.run()
    out = Path(args.out)
    s = sim.save(out)
    plot(sim, out)
    print(f"\nVirtual bench - {s['site']}, {s['replay'][0]} to {s['replay'][1]} (SIMULATED)")
    print(f"  irrigation days        : {s['irrigation_days']} of {s['days']}  ({s['water_delivered_l']} L delivered, rain {s['rain_mm']} mm)")
    print(f"  stress days (true soil): {s['stress_days_true']}")
    print(f"  treated / diverted     : {s['treated_water_l']} L / {s['diverted_to_brine_l']} L, {s['fdi_regenerations']} FDI regenerations")
    print(f"  salt recovered         : {s['salt_recovered_g']} g")
    d = s["dosing"]
    print(f"  dosing                 : {d['deliveries_with_fertilizer']} fertigated deliveries, "
          f"mean EC error {d['mean_abs_ec_error_dS_m']} dS/m, settles in {d['mean_settle_s']} s")
    for w in s["warnings"]:
        print(f"  WARN : {w}")
    for a in s["alarms"]:
        print(f"  ALARM: {a['t']} {a['msg']}")
    print(f"  outputs in {out}/ (daily_log.csv, events.jsonl, snapshots_10min.csv, summary.json, fig_bench.png)")


if __name__ == "__main__":
    main()

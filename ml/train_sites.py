#!/usr/bin/env python3
"""
Train the S.E.D.O.S.S. models for every orchard site and write the site index.

Each site has a YAML file in config/sites/. For each one this runs sedoss_ml.py with that config
(NASA POWER weather for that location) and saves everything to sedoss_outputs/sites/<id>/.
It then writes sedoss_outputs/sites/index.json, the list of trained sites that the website map
and predict_today.py read.

Usage
-----
  python3 train_sites.py                    # all sites, NASA POWER data
  python3 train_sites.py --only raub segamat
  python3 train_sites.py --full             # also run the extra-seed stability runs (slower)
  python3 train_sites.py --offline          # synthetic weather, pipeline test only

Adding a site: copy a file in config/sites/, change id/name/latitude/longitude/elevation_m,
and run this script again.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import yaml

from sites import FAR_KM, haversine_km

HERE = Path(__file__).parent
SITES_DIR = HERE / "config" / "sites"
OUT_ROOT = HERE / "sedoss_outputs" / "sites"


def site_summary(sid, cfg, out):
    r = json.loads((out / "results.json").read_text())
    res = r["results_primary_seed"]
    tb = res["task_B_forecast_tomorrow_et0"]
    s2 = res["stage2_soil_drying_SIMULATED"]["XGBoost"]
    s = cfg["site"]
    return {
        "id": sid,
        "name": s["name"],
        "latitude": s["latitude"],
        "longitude": s["longitude"],
        "elevation_m": s["elevation_m"],
        "synthetic_data": r["synthetic_data"],
        "data_range": r["data_range"],
        "heat_tmax_c": round(r["heat_tmax_c_used"], 1),
        "config_warnings": r["config_warnings"],
        "metrics": {
            "et0_forecast": {k: {m: round(v[m], 3) for m in ("R2", "RMSE", "MAE")}
                             for k, v in tb.items() if isinstance(v, dict) and "RMSE" in v},
            "soil_change_xgb_SIMULATED": {m: round(s2[m], 3) for m in ("R2", "RMSE", "MAE")},
        },
        "models_dir": str((out / "models").relative_to(HERE)),
        "results_file": str((out / "results.json").relative_to(HERE)),
    }


def main():
    ap = argparse.ArgumentParser(description="Train S.E.D.O.S.S. models for every site")
    ap.add_argument("--only", nargs="*", help="site ids to (re)train; others keep their last results")
    ap.add_argument("--full", action="store_true", help="include extra-seed stability runs")
    ap.add_argument("--offline", action="store_true", help="synthetic weather (pipeline test only)")
    args = ap.parse_args()

    configs = {}
    for f in sorted(SITES_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text())
        sid = cfg["site"].get("id") or f.stem
        if sid != f.stem:
            sys.exit(f"{f.name}: site.id '{sid}' must match the file name")
        configs[sid] = (f, cfg)
    if not configs:
        sys.exit(f"no site configs found in {SITES_DIR}")

    # Sites closer than one NASA POWER grid cell get (almost) the same weather.
    for (a, (_, ca)), (b, (_, cb)) in combinations(configs.items(), 2):
        d = haversine_km(ca["site"]["latitude"], ca["site"]["longitude"],
                         cb["site"]["latitude"], cb["site"]["longitude"])
        if d < FAR_KM:
            print(f"[warn] {a} and {b} are only {d:.0f} km apart - likely the same NASA POWER grid cell")

    todo = args.only or list(configs)
    unknown = [s for s in todo if s not in configs]
    if unknown:
        sys.exit(f"unknown site ids: {unknown}. Known: {list(configs)}")

    failed = []
    for sid in todo:
        f, _ = configs[sid]
        out = OUT_ROOT / sid
        cmd = [sys.executable, str(HERE / "sedoss_ml.py"), "--config", str(f), "--out", str(out)]
        if not args.full:
            cmd.append("--quick")
        if args.offline:
            cmd.append("--offline")
        print(f"\n===== training {sid} =====", flush=True)
        if subprocess.run(cmd, cwd=HERE).returncode != 0:
            failed.append(sid)

    sites = []
    for sid, (_, cfg) in configs.items():
        out = OUT_ROOT / sid
        if not (out / "results.json").exists() or not (out / "models" / "meta.json").exists():
            print(f"[skip] {sid}: no trained models yet")
            continue
        s = site_summary(sid, cfg, out)
        if s["synthetic_data"] and not args.offline:
            print(f"[skip] {sid}: trained on SYNTHETIC weather (NASA download failed) - not listed")
            continue
        sites.append(s)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    index = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "weather_source": "synthetic (offline test)" if args.offline else "NASA POWER daily",
             "nearest_site_warning_km": FAR_KM,
             "sites": sites}
    (OUT_ROOT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nWrote {OUT_ROOT / 'index.json'} with {len(sites)} trained site(s): {[s['id'] for s in sites]}")
    if failed:
        sys.exit(f"training failed for: {failed}")


if __name__ == "__main__":
    main()

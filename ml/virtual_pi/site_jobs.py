"""
Add a new orchard site from the website: write its config, then train it in the background.

Training runs train_sites.py --only <id> (NASA POWER download + models, usually 15-60 s).
Jobs run one at a time so two trainings never write the site index together.
"""

import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

from sites import haversine_km, load_index

ML_DIR = Path(__file__).resolve().parent.parent
SITES_DIR = ML_DIR / "config" / "sites"


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s[:40]


class SiteJobs:
    def __init__(self):
        self.jobs = {}                 # id -> job dict (newest state)
        self.lock = threading.Lock()
        self.q = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- request -----------------------------------------------------------------
    def add(self, body):
        name = str(body.get("name", "")).strip()
        try:
            lat, lon = float(body["latitude"]), float(body["longitude"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Latitude and longitude must be numbers.")
        elev_raw = body.get("elevation_m")
        elev = 50.0 if elev_raw in (None, "") else float(elev_raw)
        if not 2 <= len(name) <= 60:
            raise ValueError("Give the site a name (2 to 60 characters).")
        if not (-60 <= lat <= 60 and -180 <= lon <= 180):
            raise ValueError("Latitude must be between -60 and 60, longitude between -180 and 180.")
        if not -100 <= elev <= 3000:
            raise ValueError("Elevation must be between -100 and 3000 m.")
        sid = slugify(name)
        if not sid:
            raise ValueError("The name needs at least one letter or number.")
        with self.lock:
            busy = self.jobs.get(sid, {}).get("status") in ("queued", "training")
        if busy:
            raise ValueError(f"{name} is already being trained.")
        if (SITES_DIR / f"{sid}.yaml").exists():
            raise ValueError(f"A site called {name} already exists.")

        nearest = None
        try:
            for s in load_index()["sites"]:
                d = haversine_km(lat, lon, s["latitude"], s["longitude"])
                if nearest is None or d < nearest[1]:
                    nearest = (s["name"], d)
        except FileNotFoundError:
            pass

        cfg = {"site": {"id": sid, "name": name, "latitude": round(lat, 4), "longitude": round(lon, 4),
                        "elevation_m": round(elev, 1)}}
        header = ("# Added from the farmer hub. Elevation is approximate unless surveyed.\n"
                  "# Only the site block is set here; everything else uses DEFAULT_CONFIG in sedoss_ml.py.\n")
        (SITES_DIR / f"{sid}.yaml").write_text(header + yaml.safe_dump(cfg, sort_keys=False))

        job = {"id": sid, "name": name, "latitude": lat, "longitude": lon, "status": "queued",
               "queued_at": time.time(), "started_at": None, "finished_at": None, "error": None,
               "nearest": {"name": nearest[0], "km": round(nearest[1], 1)} if nearest else None}
        with self.lock:
            self.jobs[sid] = job
        self.q.put(sid)
        return dict(job)

    def list(self):
        with self.lock:
            return sorted((dict(j) for j in self.jobs.values()), key=lambda j: j["queued_at"], reverse=True)

    # ---- background training ------------------------------------------------------
    def _set(self, sid, **kw):
        with self.lock:
            self.jobs[sid].update(kw)

    def _worker(self):
        while True:
            sid = self.q.get()
            self._set(sid, status="training", started_at=time.time())
            try:
                r = subprocess.run([sys.executable, str(ML_DIR / "train_sites.py"), "--only", sid],
                                   cwd=ML_DIR, capture_output=True, text=True, timeout=900)
                trained = any(s["id"] == sid for s in load_index()["sites"])
                if trained:
                    self._set(sid, status="done", finished_at=time.time())
                else:
                    out = (r.stdout + r.stderr)
                    reason = ("The NASA weather download failed (check the internet connection)."
                              if "SYNTHETIC" in out or "download failed" in out
                              else "Training did not finish. See the server terminal for details.")
                    print(out[-3000:])
                    self._fail(sid, reason)
            except subprocess.TimeoutExpired:
                self._fail(sid, "Training took longer than 15 minutes and was stopped.")
            except Exception as e:  # noqa: BLE001 - report any failure to the farmer
                self._fail(sid, f"Training failed: {e}")

    def _fail(self, sid, reason):
        # remove the config so the farmer can try again with the same name
        (SITES_DIR / f"{sid}.yaml").unlink(missing_ok=True)
        self._set(sid, status="failed", finished_at=time.time(), error=reason)

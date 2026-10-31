"""
Small HTTP server for the farmer hub: serves the website (web/) and the virtual bench API.

  GET  /                   the hub (files from web/)
  GET  /api/state          live sensors, actuators, tanks, today's plan, last AI decision
  GET  /api/events?limit=N recent event log (newest last)
  GET  /api/daily          one row per simulated day
  GET  /api/hardware       parts list -> what each part does in the simulation
  GET  /api/sites          trained sites (from sedoss_outputs/sites/index.json)
  POST /api/sites          {"name", "latitude", "longitude", "elevation_m"?}  add a site and train it
  GET  /api/sites/jobs     status of sites being added (queued / training / done / failed)
  POST /api/site           {"site": "raub"}  restart the unit on another trained site
  POST /api/command        {"cmd": "...", "args": {...}}
       cmds: pause, resume, skip_today, water_only_today, irrigate_now {mm, fertigate},
             regenerate_fdi, refill_feed, refill_nutrients
  POST /api/speed          {"sim_seconds_per_second": 600}

The real Pi can serve the same endpoints later, so the website does not need to change.
"""

import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from sites import load_index

from .site_jobs import SiteJobs

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def serve(make_sim, sim, host="127.0.0.1", port=8765, speed=600.0):
    ctl = {"speed": float(speed), "stop": False, "sim": sim}
    switch_lock = threading.Lock()
    site_jobs = SiteJobs()

    def run():
        last = time.monotonic()
        while not ctl["stop"]:
            time.sleep(0.1)
            now = time.monotonic()
            s = ctl["sim"]
            with s.lock:
                if not s.done:
                    s.run_until(s.now + pd.Timedelta(seconds=ctl["speed"] * (now - last)))
            last = now

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def _static(self, path):
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            f = (WEB_DIR / rel).resolve()
            if WEB_DIR not in f.parents or not f.is_file():   # stay inside web/
                return self._send(404, {"error": "not found"})
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            self._send(200, f.read_bytes(), ctype)

        def do_GET(self):
            u = urlparse(self.path)
            s = ctl["sim"]
            if u.path == "/api/state":
                return self._send(200, {**s.state(), "speed": ctl["speed"]})
            if u.path == "/api/events":
                n = int(parse_qs(u.query).get("limit", ["50"])[0])
                with s.lock:
                    return self._send(200, s.log.items[-n:])
            if u.path == "/api/daily":
                with s.lock:
                    return self._send(200, list(s.daily))
            if u.path == "/api/hardware":
                return self._send(200, s.cfg["hardware"])
            if u.path == "/api/sites":
                return self._send(200, load_index())
            if u.path == "/api/sites/jobs":
                return self._send(200, site_jobs.list())
            if u.path.startswith("/api/"):
                return self._send(404, {"error": "not found"})
            self._static(u.path)

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/command":
                    ctl["sim"].command(body["cmd"], body.get("args"))
                    return self._send(200, {"ok": True})
                if self.path == "/api/speed":
                    ctl["speed"] = min(max(0.0, float(body["sim_seconds_per_second"])), 7200.0)
                    return self._send(200, {"ok": True, "speed": ctl["speed"]})
                if self.path == "/api/sites":
                    return self._send(202, site_jobs.add(body))
                if self.path == "/api/site":
                    with switch_lock:
                        new = make_sim(str(body["site"]))   # raises KeyError for an untrained site
                        ctl["sim"] = new
                    return self._send(200, {"ok": True, "site": new.site["id"]})
                self._send(404, {"error": "not found"})
            except KeyError as e:
                self._send(400, {"error": str(e).strip("'\"")})
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})

    threading.Thread(target=run, daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Farmer hub on http://{host}:{port}/   (virtual Pi API under /api, sim speed {speed:g}x, Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ctl["stop"] = True
        httpd.server_close()

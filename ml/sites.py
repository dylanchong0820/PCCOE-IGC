"""
Trained-site registry: find which trained site to use for an orchard location.

train_sites.py writes sedoss_outputs/sites/index.json. The website map and predict_today.py
both read it. Only sites listed there have models; any other location is mapped to the
nearest trained site, and the distance is reported so the farmer knows how far off it is.
"""

import json
import math
from pathlib import Path

INDEX_PATH = Path(__file__).parent / "sedoss_outputs" / "sites" / "index.json"

# NASA POWER weather is on a ~0.5 x 0.625 degree grid (~55-70 km here). Beyond this distance the
# trained site's weather is likely a different grid cell from the orchard's.
FAR_KM = 50.0


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_index(path=INDEX_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run train_sites.py first")
    return json.loads(path.read_text())


def get_site(site_id, index):
    for s in index["sites"]:
        if s["id"] == site_id:
            return s
    raise KeyError(f"site '{site_id}' is not trained. Trained sites: {[s['id'] for s in index['sites']]}")


def nearest_site(lat, lon, index):
    """Return (site, distance_km, is_far) for the trained site closest to (lat, lon)."""
    if not index["sites"]:
        raise ValueError("no trained sites in the index")
    best = min(index["sites"], key=lambda s: haversine_km(lat, lon, s["latitude"], s["longitude"]))
    d = haversine_km(lat, lon, best["latitude"], best["longitude"])
    return best, d, d > FAR_KM

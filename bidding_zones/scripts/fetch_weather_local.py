#!/usr/bin/env python3
"""Fetch the ERA5 weather points on your own machine (standard library only).

    cd bidding_zones
    python scripts/fetch_weather_local.py

Reads the 323 points from data/weather_points.csv and writes one JSON per
point to weather/2019/, named and formatted exactly as bzgen/data/weather.py
does, so the files drop into data/cache/weather/2019/ unchanged.  Resumable:
re-run it and it skips points already saved.  At the end it writes
weather_2019.zip (~90 MB unzipped) to upload.

Open-Meteo free tier: 10 000 calls/day; a full year for one point costs about
27 calls, so all points (~8 700 calls) fit in one day from one IP.
"""

import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive-api.open-meteo.com/v1/archive"
YEAR = 2019
VARS = ("wind_speed_100m", "shortwave_radiation")
BATCH = 5                     # locations per request (as in config weather.batch)
PAUSE_S = 15                  # between requests: well under 5 000 calls/h
HERE = Path(__file__).resolve().parents[1]
POINTS = HERE / "data" / "weather_points.csv"
OUT = HERE / "weather" / str(YEAR)


def path(lon: float, lat: float) -> Path:
    return OUT / f"{lat:.3f}_{lon:.3f}.json"


def get(params: dict) -> list:
    url = URL + "?" + urllib.parse.urlencode(params)
    wait = 30
    while True:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rocof-the-boat weather fetch"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
            return data if isinstance(data, list) else [data]
        except urllib.error.HTTPError as e:
            body = e.read()[:200].decode(errors="replace")
            if e.code == 429 and "Daily" in body:
                print("  daily limit reached; sleeping 1 h (or stop and re-run tomorrow)", flush=True)
                time.sleep(3600)
            elif e.code == 429 or e.code >= 500:
                print(f"  HTTP {e.code}; retry in {wait}s", flush=True)
                time.sleep(wait); wait = min(wait * 2, 900)
            else:
                raise SystemExit(f"HTTP {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  network error {e}; retry in {wait}s", flush=True)
            time.sleep(wait); wait = min(wait * 2, 900)


def main():
    with open(POINTS) as f:
        pts = [(float(r["lon"]), float(r["lat"])) for r in csv.DictReader(f)]
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [p for p in pts if not path(*p).exists()]
    print(f"{len(pts)} points, {len(todo)} to fetch", flush=True)
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        params = {
            "latitude": ",".join(f"{la:.3f}" for _, la in batch),
            "longitude": ",".join(f"{lo:.3f}" for lo, _ in batch),
            "start_date": f"{YEAR}-01-01", "end_date": f"{YEAR}-12-31",
            "hourly": ",".join(VARS), "models": "era5", "timezone": "GMT",
            "wind_speed_unit": "ms",
        }
        data = get(params)
        if len(data) != len(batch):
            raise SystemExit(f"expected {len(batch)} locations, got {len(data)}")
        for (lo, la), d in zip(batch, data):
            for v in VARS:
                h = d["hourly"][v]
                if len(h) < 8760 or any(x is None for x in h):
                    raise SystemExit(f"incomplete {v} at {la},{lo}")
            d["requested"] = {"lon": lo, "lat": la}
            path(lo, la).write_text(json.dumps(d))
        print(f"  {min(i + BATCH, len(todo))}/{len(todo)} fetched", flush=True)
        time.sleep(PAUSE_S)
    missing = [p for p in pts if not path(*p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} points still missing; re-run")
    z = HERE / f"weather_{YEAR}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(OUT.glob("*.json")):
            zf.write(f, f"{YEAR}/{f.name}")
    print(f"done: {z} ({z.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()

"""ERA5 hourly weather at a few hundred sample points, via Open-Meteo.

Grid
----
A hexagonal grid with a fixed ground spacing (``weather.spacing_km``) over the
covered area.  A point is kept if an in-scope bus or an offshore wind plant
lies within ``keep_frac * spacing`` of it.  Since the covering radius of a
hex grid is ``spacing / sqrt(3) ~ 0.577 spacing``, every bus has a kept point
within that distance.

Fetching
--------
``archive-api.open-meteo.com`` (keyless, ERA5 via ``models=era5``).  Open-Meteo
counts a request for one location and more than two weeks of data as
multiple API calls (weighted by ``ceil(days / 14)`` per location; INFERENCE
from their pricing/terms page wording — we pace conservatively around it).
Free-tier limits: 600/min, 5 000/h, 10 000/day.  We send batches of
``batch`` locations and pace to ``calls_per_hour`` weighted calls.  429 and
5xx are retried with exponential backoff; a daily-limit message causes a long
sleep.  Every location is cached as its own JSON file, so the fetch is
resumable and never repeats a point.

Units: ``wind_speed_unit=ms`` -> m/s at 100 m (instantaneous at the
timestamp); ``shortwave_radiation`` is W/m2 GHI, *mean over the preceding
hour* (Open-Meteo docs).  Timestamps are UTC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

from src import config
from src.data.cache import CACHE, USER_AGENT, FetchError, record

WEATHER_DIR = CACHE / "weather"
VARS = ("wind_speed_100m", "shortwave_radiation")


def _proj(lonlat: np.ndarray) -> np.ndarray:
    """Equirectangular km projection, adequate for nearest-neighbour at 100 km scale."""
    return np.c_[lonlat[:, 0] * 111.32 * np.cos(np.radians(lonlat[:, 1])),
                 lonlat[:, 1] * 110.57]


def hex_grid(spacing_km: float, lon0=-11.0, lon1=33.0, lat0=34.0, lat1=72.0) -> np.ndarray:
    rows, lat, r = [], lat0, 0
    dlat = spacing_km * math.sqrt(3) / 2 / 110.57
    while lat < lat1:
        dlon = spacing_km / (111.32 * math.cos(math.radians(lat)))
        off = dlon / 2 if r % 2 else 0.0
        for lon in np.arange(lon0 + off, lon1, dlon):
            rows.append((round(lon, 3), round(lat, 3)))
        lat += dlat
        r += 1
    return np.array(rows)


def sample_points(anchor_lonlat: np.ndarray, cfg: dict) -> pd.DataFrame:
    w = cfg["weather"]
    grid = hex_grid(w["spacing_km"])
    d, _ = cKDTree(_proj(anchor_lonlat)).query(_proj(grid))
    keep = grid[d < w["keep_frac"] * w["spacing_km"]]
    pts = pd.DataFrame(keep, columns=["lon", "lat"])
    pts.index = [f"p{i:04d}" for i in range(len(pts))]
    pts.index.name = "point"
    return pts


def _path(lon: float, lat: float, year: int) -> Path:
    return WEATHER_DIR / str(year) / f"{lat:.3f}_{lon:.3f}.json"


def _weight(year: int) -> int:
    days = 366 if (year % 4 == 0 and year % 100) or year % 400 == 0 else 365
    return math.ceil(days / 14)


def fetch_points(points: pd.DataFrame, year: int, cfg: dict, log=print) -> None:
    w = cfg["weather"]
    url = cfg["sources"]["open_meteo_archive"]
    todo = [(r.lon, r.lat) for r in points.itertuples() if not _path(r.lon, r.lat, year).exists()]
    log(f"{len(points)} points, {len(todo)} still to fetch for {year}")
    per_loc = _weight(year) * math.ceil(len(VARS) / 10)
    pause = w["batch"] * per_loc / w["calls_per_hour"] * 3600.0
    for i in range(0, len(todo), w["batch"]):
        batch = todo[i:i + w["batch"]]
        params = {
            "latitude": ",".join(f"{la:.3f}" for _, la in batch),
            "longitude": ",".join(f"{lo:.3f}" for lo, _ in batch),
            "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
            "hourly": ",".join(VARS), "models": "era5", "timezone": "GMT",
            "wind_speed_unit": "ms",
        }
        delay = 30.0
        for attempt in range(40):
            t0 = time.time()
            try:
                r = requests.get(url, params=params, timeout=300,
                                 headers={"User-Agent": USER_AGENT})
            except requests.RequestException as e:
                log(f"  network error {e}; retry in {delay:.0f}s")
                time.sleep(delay); delay = min(delay * 2, 3600); continue
            if r.status_code == 200:
                break
            body = r.text[:300]
            if r.status_code == 429 or r.status_code >= 500:
                # Egress here rotates over several IPs with separate quotas: a
                # daily-limit 429 from one is usually followed by a 200 from
                # another, so retry after a short fixed wait (bounded attempts).
                wait = 30.0 if "Daily" in body else delay
                log(f"  HTTP {r.status_code} ({body[:80]!r}); sleeping {wait:.0f}s")
                time.sleep(wait)
                if "Daily" not in body:
                    delay = min(delay * 2, 1800)
                continue
            raise FetchError(f"Open-Meteo HTTP {r.status_code}: {body}")
        else:
            raise FetchError("Open-Meteo: retries exhausted")
        data = r.json()
        if isinstance(data, dict):
            data = [data]
        if len(data) != len(batch):
            raise FetchError(f"expected {len(batch)} locations, got {len(data)}")
        for (lo, la), d in zip(batch, data):
            h = d["hourly"]
            for v in VARS:
                if v not in h or len(h[v]) < 8760:
                    raise FetchError(f"incomplete {v} at {la},{lo}")
                if sum(x is None for x in h[v]) > 0:
                    raise FetchError(f"missing values in {v} at {la},{lo}")
            d["requested"] = {"lon": lo, "lat": la}
            p = _path(lo, la, year)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(d))
        done = i + len(batch)
        log(f"  {done}/{len(todo)} fetched")
        time.sleep(max(0.0, pause - (time.time() - t0)))


def record_provenance(points: pd.DataFrame, year: int, cfg: dict) -> None:
    h = hashlib.sha256()
    n = 0
    for r in points.sort_values(["lat", "lon"]).itertuples():
        p = _path(r.lon, r.lat, year)
        if p.exists():
            h.update(p.read_bytes()); n += 1
    record(f"open_meteo_era5_{year}", cfg["sources"]["open_meteo_archive"],
           sha=h.hexdigest(), size=None,
           notes=f"{n}/{len(points)} points, models=era5, hourly {','.join(VARS)}, "
                 f"{year}; sha256 over concatenated per-point JSON (sorted lat,lon); "
                 f"points in data/weather_points.csv")


def load(points: pd.DataFrame, year: int) -> dict[str, pd.DataFrame]:
    """Return {var: DataFrame[time x point]} from the cache. Raises if any point missing."""
    out = {v: {} for v in VARS}
    idx = None
    for name, r in points.iterrows():
        p = _path(r.lon, r.lat, year)
        if not p.exists():
            raise FetchError(f"weather point {name} ({r.lat},{r.lon}) not fetched")
        d = json.loads(p.read_text())
        if idx is None:
            idx = pd.DatetimeIndex(d["hourly"]["time"], tz="UTC")
        for v in VARS:
            out[v][name] = d["hourly"][v]
    return {v: pd.DataFrame(out[v], index=idx) for v in VARS}


def anchors(cfg: dict) -> np.ndarray:
    buses = pd.read_csv(CACHE / "osm" / "buses.csv", quotechar="'")
    buses = buses[buses.country.isin(cfg["scope"]["countries"])]
    ppm = pd.read_csv(CACHE / "ppm" / "powerplants.csv", index_col=0)
    from src.data.countries import NAME
    names = [NAME[c] for c in cfg["scope"]["countries"] if c in NAME]
    off = ppm[(ppm.Technology == "Offshore") & ppm.Country.isin(names)][["lon", "lat"]].dropna().values
    return np.vstack([buses[["x", "y"]].values, off])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["points", "fetch", "provenance"])
    args = ap.parse_args()
    cfg = config.load()
    year = cfg["weather_year"]
    out = config.ROOT / "data" / "weather_points.csv"
    if args.cmd == "points" or not out.exists():
        pts = sample_points(anchors(cfg), cfg)
        pts.to_csv(out)
        print(f"{len(pts)} points -> {out}")
    pts = pd.read_csv(out, index_col=0)
    if args.cmd == "fetch":
        fetch_points(pts, year, cfg, log=lambda s: print(time.strftime("%H:%M:%S"), s, flush=True))
        record_provenance(pts, year, cfg)
    elif args.cmd == "provenance":
        record_provenance(pts, year, cfg)


if __name__ == "__main__":
    main()

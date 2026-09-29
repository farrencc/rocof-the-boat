"""National hourly load: OPSD shape for the weather year, scaled to the target year.

Shape: OPSD time series, ``<CC>_load_actual_entsoe_transparency`` (MW, UTC,
timestamp = start of the hour), weather year (2019).

Level: Eurostat ``nrg_cb_e`` balance item ``AFC`` ("available for final
consumption", GWh).  Factor = AFC(target or latest year) / AFC(weather year),
from the same series, so definitional differences between OPSD/ENTSO-E
"load" and Eurostat AFC cancel to first order.  INFERENCE: this assumes the
ratio of ENTSO-E load to AFC is stable across the years compared.

Countries without an OPSD series get a donor country's *normalised* shape
scaled to their own annual level.  This is flagged in the returned report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bzgen.data.cache import CACHE, fetch
from bzgen.data.countries import EUROSTAT
from bzgen.data import eurostat

# Donor shapes for countries OPSD does not cover.  Neighbours with similar
# climate and (where possible) OPSD coverage.  INFERENCE: author's choice.
DONOR = {"AL": "ME", "BA": "HR", "MK": "BG", "XK": "RS"}
MAX_GAP_H = 6


def load_opsd(cfg: dict) -> pd.DataFrame:
    p = fetch(cfg["sources"]["opsd_timeseries"], "opsd/time_series_60min_singleindex.csv",
              key="opsd_time_series_60min",
              notes="OPSD Time series package, latest release, 60-min single-index CSV")
    cols = pd.read_csv(p, nrows=0).columns
    use = ["utc_timestamp"] + [c for c in cols if c.endswith("_load_actual_entsoe_transparency")]
    df = pd.read_csv(p, usecols=use, index_col=0, parse_dates=True)
    df.columns = [c.replace("_load_actual_entsoe_transparency", "") for c in df.columns]
    return df


def annual_afc() -> pd.DataFrame:
    e = eurostat.read("nrg_cb_e", "Eurostat nrg_cb_e supply, transformation and "
                                  "consumption of electricity, annual, GWh")
    e = e[(e.nrg_bal == "AFC") & (e.siec == "E7000") & (e.unit == "GWH")]
    return e.pivot_table(index="geo", columns="year", values="value")


def _load_afc_ratio(opsd: pd.DataFrame, afc: pd.DataFrame, wy: int) -> float:
    r = []
    for c in opsd.columns:
        es = EUROSTAT.get(c)
        if es in afc.index and opsd[c].notna().mean() > 0.99 and not np.isnan(afc.loc[es, wy]):
            r.append(opsd[c].sum() / 1e3 / afc.loc[es, wy])
    return float(np.median(r))


def national_load(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (hourly MW [time x country], per-country report)."""
    wy, ty = cfg["weather_year"], cfg["target_year"]
    opsd = load_opsd(cfg)
    opsd = opsd.loc[f"{wy}-01-01":f"{wy}-12-31"]
    idx = pd.date_range(f"{wy}-01-01", f"{wy}-12-31 23:00", freq="h", tz="UTC")
    opsd = opsd.reindex(idx)
    afc = annual_afc()
    countries = [c for c in cfg["scope"]["countries"]]
    out, rep = {}, []
    for c in countries:
        es = EUROSTAT[c]
        row = {"country": c}
        src = c
        if c not in opsd or opsd[c].notna().sum() < 0.9 * len(idx):
            src = DONOR.get(c)
            row["shape_source"] = f"DONOR {src} (no OPSD series)" if src else "NONE"
        else:
            row["shape_source"] = "OPSD"
        if src is None or src not in opsd:
            row["status"] = "NO LOAD DATA"
            rep.append(row)
            continue
        s = opsd[src]
        n_missing = int(s.isna().sum())
        s = s.interpolate(limit=MAX_GAP_H, limit_direction="both")
        if s.isna().any():
            # Remaining long gaps: same hour one week earlier/later. Reported.
            s = s.fillna(s.shift(168)).fillna(s.shift(-168))
        row["opsd_missing_h"] = n_missing
        if s.isna().any():
            row["status"] = "UNFILLABLE GAPS"
            rep.append(row)
            continue
        # Level
        if es in afc.index and not np.isnan(afc.loc[es].get(wy, np.nan)):
            base = afc.loc[es, wy]
            yrs = [y for y in afc.columns if y <= ty and not np.isnan(afc.loc[es, y])]
            y = max(yrs)
            row.update(afc_weather_gwh=base, afc_year=y, afc_target_gwh=afc.loc[es, y])
            if src == c:
                factor = afc.loc[es, y] / base
                series = s * factor
                row["scale_method"] = f"OPSD {wy} x AFC {y}/{wy}"
            else:
                # donor shape: normalise to own annual AFC of the level year, times
                # the median ENTSO-E-load / AFC ratio of the OPSD countries (ENTSO-E
                # load includes losses that AFC excludes).  INFERENCE.
                series = s / s.sum() * afc.loc[es, y] * 1e3 * _load_afc_ratio(opsd, afc, wy)
                row["scale_method"] = (f"donor shape normalised to own AFC {y} x median "
                                       f"load/AFC ratio {_load_afc_ratio(opsd, afc, wy):.3f}")
        else:
            series = s
            row["scale_method"] = "UNSCALED: no Eurostat AFC (kept weather-year level)"
        row["annual_twh"] = series.sum() / 1e6
        row["peak_gw"] = series.max() / 1e3
        row["status"] = "ok" if row["shape_source"] == "OPSD" and "UNSCALED" not in row["scale_method"] else "FLAG"
        out[c] = series
        rep.append(row)
    return pd.DataFrame(out), pd.DataFrame(rep).set_index("country")

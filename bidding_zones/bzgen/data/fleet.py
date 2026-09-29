"""Generation fleet: powerplantmatching plants, carriers, marginal costs, targets.

* Plants: powerplantmatching's prepared (matched) dataset, fetched through our
  cache from the URL the package itself uses (``matched_data_url`` at the
  package's ``latest_release`` tag).  Kept if in scope, operating in the
  target year (``DateIn <= year`` or unknown, and ``DateOut >= year`` or
  unknown) and geolocated.  Storage (pumped hydro, batteries, H2, heat) is
  excluded: the brief fixes "no storage".
* Wind/solar national targets: Eurostat ``nrg_inf_epcrw`` net capacity for the
  latest year (2024 at time of writing; 2025 not yet published).  Where it
  exceeds powerplantmatching, the gap is added at network assembly.
* Hydro capacity factor per country (weather year): Eurostat ``nrg_bal_peh``
  GEP of RA100 minus RA130 (pure pumped storage) over ``nrg_inf_epcrw``
  RA110 + RA120 capacity of the same year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bzgen.data import eurostat
from bzgen.data.cache import fetch
from bzgen.data.countries import EUROSTAT, ISO

EXCLUDE_FUEL = {"Battery", "Hydrogen Storage", "Heat Storage", "Mechanical Storage"}


def _carrier(fuel: str, tech) -> str | None:
    tech = tech if isinstance(tech, str) else ""
    if fuel == "Nuclear":
        return "nuclear"
    if fuel == "Lignite":
        return "lignite"
    if fuel == "Hard Coal":
        return "coal"
    if fuel == "Natural Gas":
        if tech == "CCGT":
            return "CCGT"
        if tech in ("OCGT", "Combustion Engine"):
            return "OCGT"
        return "gas_steam"   # Steam Turbine or unknown technology. INFERENCE.
    if fuel in ("Oil", "Other"):
        return "oil"         # "Other" (~2 GW) costed like oil. INFERENCE.
    if fuel in ("Solid Biomass", "Biogas", "Waste"):
        return "biomass"
    if fuel == "Geothermal":
        return "geothermal"
    if fuel == "Hydro":
        if tech == "Run-Of-River":
            return "ror"
        if tech == "Pumped Storage":
            return None      # storage: excluded
        return "reservoir"   # Reservoir or unknown. INFERENCE.
    if fuel == "Wind":
        return "offwind" if tech == "Offshore" else "onwind"
    if fuel == "Solar":
        return "solar"       # PV and CSP (CSP ~2.5 GW, no storage modelled)
    return None


def plants(cfg: dict) -> tuple[pd.DataFrame, dict]:
    import powerplantmatching as pm
    from powerplantmatching import latest_release

    url = pm.get_config()["matched_data_url"].format(tag="v" + latest_release)
    p = fetch(url, "ppm/powerplants.csv", key="powerplantmatching",
              notes=f"powerplantmatching {pm.__version__} prepared dataset, tag v{latest_release}")
    df = pd.read_csv(p, index_col=0)
    y = cfg["target_year"]
    rep = {"ppm_tag": latest_release, "rows": len(df)}
    df["country"] = df.Country.map(ISO)
    df = df[df.country.isin(cfg["scope"]["countries"])]
    df = df[~df.Fueltype.isin(EXCLUDE_FUEL)]
    df["carrier"] = [_carrier(f, t) for f, t in zip(df.Fueltype, df.Technology)]
    rep["excluded_storage_mw"] = float(df[df.carrier.isna()].Capacity.sum())
    df = df[df.carrier.notna()]
    op = (df.DateIn.isna() | (df.DateIn <= y)) & (df.DateOut.isna() | (df.DateOut >= y))
    rep["not_operating_mw"] = float(df[~op].Capacity.sum())
    df = df[op]
    nogeo = df.lat.isna() | df.lon.isna()
    rep["no_location_mw"] = float(df[nogeo].Capacity.sum())
    df = df[~nogeo & (df.Capacity > 0)]
    # Marginal cost
    c = cfg["costs"]
    mc, eff_used = [], []
    for car, eff in zip(df.carrier, df.Efficiency):
        spec = c["carriers"][car]
        if "mc" in spec:
            mc.append(spec["mc"]); eff_used.append(np.nan); continue
        lo, hi = spec["eff_range"]
        e = eff if (isinstance(eff, float) and lo <= eff <= hi) else spec["eff"]
        f = spec["fuel"]
        mc.append((c["fuel_eur_mwh_th"][f] + c["co2_eur_t"] * c["emission_t_mwh_th"][f]) / e
                  + spec["vom"])
        eff_used.append(e)
    df["marginal_cost"] = mc
    df["efficiency"] = eff_used
    out = df[["Name", "country", "carrier", "Capacity", "efficiency", "marginal_cost",
              "lat", "lon", "DateIn", "DateOut"]].rename(columns={"Capacity": "p_nom"})
    return out, rep


def res_targets(cfg: dict) -> pd.DataFrame:
    """Eurostat installed capacity (MW) for onwind/offwind/solar, latest year."""
    r = eurostat.read("nrg_inf_epcrw", "Eurostat electricity production capacities for "
                                       "renewables and wastes, MW")
    r = r[(r.plant_tec == "CAP_NET_ELC") & (r.unit == "MW")]
    codes = {"RA310": "onwind", "RA320": "offwind", "RA420": "solar"}
    r = r[r.siec.isin(codes)].dropna(subset=["value"])
    out = []
    for c in cfg["scope"]["countries"]:
        g = r[r.geo == EUROSTAT[c]]
        if g.empty:
            continue
        yr = g.year.max()
        g = g[g.year == yr]
        for s, car in codes.items():
            v = g[g.siec == s].value
            out.append({"country": c, "carrier": car, "year": yr,
                        "eurostat_mw": float(v.iloc[0]) if len(v) else np.nan})
    return pd.DataFrame(out)


def hydro_cf(cfg: dict, ppm: pd.DataFrame) -> pd.DataFrame:
    wy = cfg["weather_year"]
    gen = eurostat.read("nrg_bal_peh", "Eurostat gross and net production of electricity "
                                       "and derived heat by type of plant and operator, GWh")
    gen = gen[(gen.nrg_bal == "GEP") & (gen.unit == "GWH") & (gen.year == wy)]
    g = gen.pivot_table(index="geo", columns="siec", values="value")
    cap = eurostat.read("nrg_inf_epcrw")
    cap = cap[(cap.plant_tec == "CAP_NET_ELC") & (cap.unit == "MW") & (cap.year == wy)]
    k = cap.pivot_table(index="geo", columns="siec", values="value")
    lo, hi = cfg["costs"]["hydro_cf_clip"]
    rows = []
    for c in cfg["scope"]["countries"]:
        es = EUROSTAT[c]
        row = {"country": c}
        ppm_cap = ppm[(ppm.country == c) & ppm.carrier.isin(["ror", "reservoir"])].p_nom.sum()
        gwh = (g.loc[es, "RA100"] - g.loc[es].get("RA130", 0.0)) if es in g.index else np.nan
        mw = (k.loc[es, "RA110"] + k.loc[es].get("RA120", 0.0)) if es in k.index else np.nan
        if np.isfinite(gwh) and np.isfinite(mw) and mw > 0:
            cf, src = gwh * 1e3 / (mw * 8760), f"Eurostat gen/cap {wy}"
        elif np.isfinite(gwh) and ppm_cap > 0:
            cf, src = gwh * 1e3 / (ppm_cap * 8760), f"Eurostat gen {wy} / ppm capacity"
        else:
            cf, src = cfg["costs"]["hydro_cf_default"], "DEFAULT (no data)"
        row.update(gen_gwh=gwh, cap_mw=mw, ppm_cap_mw=ppm_cap, cf_raw=cf,
                   cf=float(np.clip(cf, lo, hi)) if np.isfinite(cf) else cfg["costs"]["hydro_cf_default"],
                   source=src)
        rows.append(row)
    return pd.DataFrame(rows).set_index("country")

"""Attach load and generation to the simplified topology; persist to data/interim/.

Load
----
National hourly load (``src.data.demand``) is split over the country's buses
by a static key ``w_gdp * gdp_share + w_pop * pop_share`` (PyPSA-Eur's
default 0.6/0.4).  Each bus is assigned to the NUTS3 region containing it
(point-in-polygon); a region's population and GDP are split equally among its
buses; a region containing no bus hands its values to the nearest bus of the
same country.  Countries without NUTS3 values fall back as reported.

Generation
----------
Plants go to the nearest bus of the same country.  Wind/solar national totals
are matched to Eurostat (latest year) in both directions:
* ppm > Eurostat: ppm plants scaled down proportionally;
* ppm < Eurostat: onshore gap distributed in proportion to ppm onshore capacity
  at each bus (uniform over buses if the country has <20 % coverage), solar gap
  in proportion to the load key (distributed/rooftop PV proxy), offshore ppm
  plants scaled up (no new sites).  INFERENCE: author's allocation rules.
Wind/solar CF of a plant comes from its nearest weather point (so offshore farms
see offshore wind even though their bus is onshore); a (bus, carrier)
generator's profile is the capacity-weighted mean.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src import config
from src.data import demand, fleet, regional, renewables, weather
from src.network.build import build_topology, cluster_countries

INTERIM = config.ROOT / "data" / "interim"
RES = ("onwind", "offwind", "solar")


def _proj(lon, lat):
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return np.c_[lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57]


def nearest_bus(buses: pd.DataFrame, lon, lat, country) -> np.ndarray:
    out = np.empty(len(lon), dtype=object)
    country = np.asarray(country)
    for c in np.unique(country):
        bb = buses[buses.country == c]
        m = country == c
        if bb.empty:
            out[m] = None
            continue
        _, i = cKDTree(_proj(bb.x, bb.y)).query(_proj(np.asarray(lon)[m], np.asarray(lat)[m]))
        out[m] = bb.index.values[i]
    return out


def voronoi_key(buses: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.DataFrame]:
    """PyPSA-Eur's method (``redistribute_attribute``): each NUTS3 region's
    population and GDP are split over the buses' Voronoi cells (per country,
    clipped to the country) in proportion to overlapping area."""
    from shapely.geometry import MultiPoint
    from shapely.ops import voronoi_diagram
    nuts = regional.nuts3_table()
    nuts["cc"] = nuts.CNTR_CODE.replace({"EL": "GR"})
    nuts = nuts.to_crs(3035)
    wg, wp = cfg["demand"]["w_gdp"], cfg["demand"]["w_pop"]
    key = pd.Series(0.0, index=buses.index)
    rows = []
    for c, bb in buses.groupby("country"):
        nc = nuts[nuts.cc == c]
        row = {"country": c, "buses": len(bb), "nuts3_regions": len(nc)}
        have_pop = len(nc) and nc["pop"].notna().all()
        have_gdp = len(nc) and nc["gdp"].notna().all()
        if not len(nc) or not (have_pop or have_gdp):
            key[bb.index] = 1.0 / len(bb)
            row["method"] = "FLAG uniform per bus (no NUTS3 values)"
            rows.append(row)
            continue
        shape = nc.geometry.union_all()
        pts = gpd.GeoSeries(gpd.points_from_xy(bb.x, bb.y), crs=4326).to_crs(3035)
        if len(bb) == 1:
            cells = gpd.GeoDataFrame({"bus": bb.index}, geometry=[shape], crs=3035)
        else:
            from shapely.geometry import box
            x0, y0, x1, y1 = shape.bounds
            # (buffering the coastline itself is prohibitively expensive, e.g. FI)
            vor = voronoi_diagram(MultiPoint(list(pts)), envelope=box(x0 - 1e5, y0 - 1e5, x1 + 1e5, y1 + 1e5))
            cells = gpd.GeoDataFrame(geometry=list(vor.geoms), crs=3035)
            j = gpd.sjoin(gpd.GeoDataFrame({"bus": bb.index}, geometry=pts.values, crs=3035),
                          cells, predicate="within")
            cells = cells.loc[j.index_right.values].assign(bus=j.bus.values)
            cells["geometry"] = cells.geometry.intersection(shape)
        ov = gpd.overlay(nc[["geo", "pop", "gdp", "geometry"]], cells[["bus", "geometry"]],
                         how="intersection", keep_geom_type=True)
        ov["a"] = ov.area
        ov["share"] = ov.a / ov.groupby("geo").a.transform("sum")
        pop = (ov["pop"] * ov.share).groupby(ov.bus).sum().reindex(bb.index).fillna(0.0)
        gdp = (ov["gdp"] * ov.share).groupby(ov.bus).sum().reindex(bb.index).fillna(0.0)
        if have_pop and have_gdp:
            k = wg * gdp / gdp.sum() + wp * pop / pop.sum()
            row["method"] = f"voronoi {wg} gdp + {wp} pop"
        elif have_pop:
            k = pop / pop.sum()
            row["method"] = "voronoi FLAG population only (GDP incomplete)"
        else:
            k = gdp / gdp.sum()
            row["method"] = "voronoi FLAG GDP only (population incomplete)"
        key[bb.index] = k / k.sum()
        row["buses_zero_load"] = int((k == 0).sum())
        rows.append(row)
    return key, pd.DataFrame(rows).set_index("country")


def load_key(buses: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.DataFrame]:
    if cfg["demand"].get("distribution", "point_in_polygon") == "voronoi":
        return voronoi_key(buses, cfg)
    nuts = regional.nuts3_table()
    nuts["cc"] = nuts.CNTR_CODE.replace({"EL": "GR"})
    pts = gpd.GeoDataFrame(buses[["country"]].copy(),
                           geometry=gpd.points_from_xy(buses.x, buses.y), crs=4326)
    j = gpd.sjoin(pts, nuts[["geo", "cc", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    # only accept a region of the bus's own country
    j.loc[j.cc != j.country, "geo"] = np.nan
    buses = buses.assign(nuts3=j.geo)
    wg, wp = cfg["demand"]["w_gdp"], cfg["demand"]["w_pop"]
    key = pd.Series(0.0, index=buses.index)
    rows = []
    for c, bb in buses.groupby("country"):
        nc = nuts[nuts.cc == c].copy()
        row = {"country": c, "buses": len(bb), "buses_in_nuts3": int(bb.nuts3.notna().sum()),
               "nuts3_regions": len(nc)}
        have_pop = len(nc) and nc["pop"].notna().all()
        have_gdp = len(nc) and nc["gdp"].notna().all()
        if not len(nc) or not (have_pop or have_gdp):
            key[bb.index] = 1.0 / len(bb)
            row["method"] = "FLAG uniform per bus (no NUTS3 values)"
            rows.append(row)
            continue
        # assign each region to buses: its own buses, else nearest bus
        cent = nc.geometry.representative_point()
        nb = nearest_bus(bb, cent.x.values, cent.y.values, [c] * len(nc))
        pop = pd.Series(0.0, index=bb.index)
        gdp = pd.Series(0.0, index=bb.index)
        for (_, r), nearest in zip(nc.iterrows(), nb):
            own = bb.index[bb.nuts3 == r.geo]
            tgt = own if len(own) else [nearest]
            if have_pop:
                pop[tgt] += r["pop"] / len(tgt)
            if have_gdp:
                gdp[tgt] += r["gdp"] / len(tgt)
        if have_pop and have_gdp:
            k = wg * gdp / gdp.sum() + wp * pop / pop.sum()
            row["method"] = f"{wg} gdp + {wp} pop"
        elif have_pop:
            k = pop / pop.sum()
            row["method"] = "FLAG population only (GDP incomplete)"
        else:
            k = gdp / gdp.sum()
            row["method"] = "FLAG GDP only (population incomplete)"
        key[bb.index] = k / k.sum()
        row["regions_without_bus"] = int(sum((bb.nuts3 == g).sum() == 0 for g in nc.geo))
        rows.append(row)
    return key, pd.DataFrame(rows).set_index("country")


def assemble(cfg: dict | None = None) -> dict:
    cfg = cfg or config.load()
    INTERIM.mkdir(parents=True, exist_ok=True)
    buses, lines, links, topo_rep = build_topology(cfg)
    cc, merge_rep = cluster_countries(buses, lines, cfg)
    buses["cluster_country"] = cc

    # ---- load
    nat, dem_rep = demand.national_load(cfg)
    key, key_rep = load_key(buses, cfg)
    missing = sorted(set(buses.country) - set(nat.columns))
    if missing:
        raise RuntimeError(f"no national load for countries with buses: {missing}")

    # ---- weather
    pts = pd.read_csv(config.ROOT / "data" / "weather_points.csv", index_col=0)
    wx = weather.load(pts, cfg["weather_year"])
    idx = pd.date_range(f"{cfg['weather_year']}-01-01", periods=len(wx["wind_speed_100m"]),
                        freq="h", tz="UTC")
    wind = renewables.wind_cf(wx["wind_speed_100m"], cfg["weather"]["wind"]).set_axis(idx)
    pv = renewables.pv_cf(wx["shortwave_radiation"], cfg["weather"]["pv"]).set_axis(idx)
    ptree = cKDTree(_proj(pts.lon, pts.lat))

    # ---- plants
    pl, fleet_rep = fleet.plants(cfg)
    pl = pl[pl.country.isin(buses.country.unique())].copy()
    pl["bus"] = nearest_bus(buses, pl.lon.values, pl.lat.values, pl.country.values)
    pl["point"] = pts.index.values[ptree.query(_proj(pl.lon, pl.lat))[1]]
    pl["source"] = "ppm"

    # ---- RES targets
    tg = fleet.res_targets(cfg)
    res_rows, extra = [], []
    bus_point = pd.Series(pts.index.values[ptree.query(_proj(buses.x, buses.y))[1]], index=buses.index)
    for c in sorted(buses.country.unique()):
        for car in RES:
            m = (pl.country == c) & (pl.carrier == car)
            have = pl.loc[m, "p_nom"].sum()
            t = tg[(tg.country == c) & (tg.carrier == car)]
            target = float(t.eurostat_mw.iloc[0]) if len(t) and np.isfinite(t.eurostat_mw.iloc[0]) else np.nan
            row = {"country": c, "carrier": car, "ppm_mw": have, "eurostat_mw": target,
                   "eurostat_year": int(t.year.iloc[0]) if len(t) else None}
            if not np.isfinite(target):
                row["action"] = "FLAG no Eurostat target: ppm as is"
            elif target < have:
                pl.loc[m, "p_nom"] *= target / have
                row["action"] = f"scaled ppm down x{target / have:.3f}"
            elif target > have:
                gap = target - have
                bb = buses.index[buses.country == c]
                if car == "offwind":
                    if have > 0:
                        pl.loc[m, "p_nom"] *= target / have
                        row["action"] = f"scaled ppm offshore up x{target / have:.3f}"
                    else:
                        row["action"] = f"FLAG {gap:.0f} MW offshore target, no ppm sites: not added"
                    res_rows.append(row); continue
                if car == "onwind" and have >= 0.2 * target:
                    w = pl[m].groupby("bus").p_nom.sum().reindex(bb).fillna(0.0)
                    how = "prop. to ppm onwind at bus"
                elif car == "onwind":
                    w = pd.Series(1.0, index=bb)
                    how = "FLAG uniform over buses (ppm coverage <20%)"
                else:
                    w = key[bb]
                    how = "prop. to load key"
                w = w / w.sum()
                for b, s in w[w > 0].items():
                    extra.append({"Name": f"topup-{car}-{b}", "country": c, "carrier": car,
                                  "p_nom": gap * s, "efficiency": np.nan, "marginal_cost": 0.0,
                                  "lat": buses.at[b, "y"], "lon": buses.at[b, "x"],
                                  "bus": b, "point": bus_point[b], "source": "eurostat_topup"})
                row["action"] = f"added {gap:.0f} MW {how}"
            else:
                row["action"] = "equal"
            res_rows.append(row)
    pl = pd.concat([pl, pd.DataFrame(extra)], ignore_index=True)

    # ---- hydro CF
    hcf = fleet.hydro_cf(cfg, pl)

    # ---- aggregate to (bus, carrier)
    pl["mc_x_p"] = pl.marginal_cost * pl.p_nom
    gens = pl.groupby(["bus", "carrier"]).agg(p_nom=("p_nom", "sum"), mc_x_p=("mc_x_p", "sum"),
                                                n_plants=("p_nom", "size")).reset_index()
    gens["marginal_cost"] = gens.mc_x_p / gens.p_nom
    gens = gens.drop(columns="mc_x_p")
    gens = gens[gens.p_nom > 0.1]
    gens["country"] = gens.bus.map(buses.country)
    gens.index = gens.bus + " " + gens.carrier
    spec = cfg["costs"]["carriers"]
    gens["p_max_pu"] = [spec[c].get("availability", np.nan) for c in gens.carrier]
    # Run-of-river: flat national hydro CF.  Reservoirs: dispatchable up to their
    # availability at the water value; with no storage there is no energy limit,
    # so annual reservoir output is reported against Eurostat after the solve.
    hydro = gens.carrier == "ror"
    gens.loc[hydro, "p_max_pu"] = gens.loc[hydro, "country"].map(hcf.cf)

    # time-varying CF for RES: capacity-weighted mean of plant CFs
    prof = {}
    for car, cf in (("onwind", wind), ("offwind", wind), ("solar", pv)):
        sub = pl[pl.carrier == car]
        w = sub.groupby(["bus", "point"]).p_nom.sum()
        for b, g in w.groupby(level=0):
            pts_b = g.index.get_level_values(1)
            prof[f"{b} {car}"] = (cf[pts_b].to_numpy() @ g.to_numpy()) / g.sum()
    prof = pd.DataFrame(prof, index=idx)
    prof = prof[[c for c in prof.columns if c in gens.index]]

    # ---- persist
    buses.to_csv(INTERIM / "buses.csv")
    lines[["bus0", "bus1", "voltage", "circuits", "s_nom", "x", "r", "x_orig_ohm", "length",
           "type", "border"]].to_csv(INTERIM / "lines.csv")
    links[["bus0", "bus1", "p_nom", "length", "border"]].to_csv(INTERIM / "links.csv")
    key.rename("load_key").to_csv(INTERIM / "load_key.csv")
    nat.to_parquet(INTERIM / "national_load.parquet")
    gens.to_csv(INTERIM / "generators.csv")
    prof.to_parquet(INTERIM / "res_profiles.parquet")
    pl.drop(columns="mc_x_p").to_csv(INTERIM / "plants.csv", index=False)
    wind.to_parquet(INTERIM / "wind_cf_points.parquet")
    pv.to_parquet(INTERIM / "pv_cf_points.parquet")
    reports = {"topology": topo_rep, "merge": merge_rep, "demand": dem_rep, "load_key": key_rep,
               "fleet": fleet_rep, "res_targets": pd.DataFrame(res_rows), "hydro_cf": hcf}
    return {"buses": buses, "lines": lines, "links": links, "gens": gens, "prof": prof,
            "nat": nat, "key": key, "plants": pl, "wind": wind, "pv": pv, "pts": pts,
            "reports": reports}

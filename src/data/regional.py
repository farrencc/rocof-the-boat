"""NUTS3 population and GDP, with GISCO geometry, for distributing load to buses.

Eurostat's regional tables are coded in NUTS 2024 (checked: 1301 of 1309 NUTS3
codes in ``nama_10r_3popgdp`` match GISCO NUTS 2024; the rest are ``ZZZ``
extra-regio codes), so NUTS 2024 geometry is used.

* population: ``demo_r_pjanaggr3`` (1 January, sex=T, age=TOTAL), latest year
  with values per country (covers CH, NO, AL which ``nama_10r_3popgdp`` lacks
  for recent years).
* GDP: ``nama_10r_3gdp`` unit ``MIO_EUR``, latest year with values per country.

BA has no NUTS; XK has geometry but no values.  NO has no NUTS3 GDP.  Those
cases are reported by :func:`nuts3_table`, not filled silently.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from src.data import eurostat
from src.data.cache import fetch

NUTS_URL = ("https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
            "NUTS_RG_01M_2024_4326_LEVL_3.geojson")
NUTS0_URL = ("https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
             "NUTS_RG_10M_2024_4326_LEVL_0.geojson")
CNTR_URL = ("https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/"
            "CNTR_RG_10M_2024_4326.geojson")


def _latest(df: pd.DataFrame) -> pd.DataFrame:
    """Per country, the latest year in which every NUTS3 region with data has a value."""
    df = df[(df.geo.str.len() == 5) & ~df.geo.str.endswith("ZZZ")].dropna(subset=["value"]).copy()
    df["cc"] = df.geo.str[:2]
    out = []
    for cc, g in df.groupby("cc"):
        y = g.year.max()
        out.append(g[g.year == y][["geo", "value", "year"]])
    return pd.concat(out)


def nuts3_table() -> gpd.GeoDataFrame:
    geo = gpd.read_file(fetch(NUTS_URL, "gisco/NUTS_RG_01M_2024_4326_LEVL_3.geojson",
                              key="gisco_nuts2024_l3",
                              notes="GISCO NUTS 2024 level 3, 1:1M, EPSG:4326"))
    pop = eurostat.read("demo_r_pjanaggr3", "Eurostat population on 1 January by broad "
                                            "age group, sex and NUTS 3, SDMX TSV")
    pop = _latest(pop[(pop.sex == "T") & (pop.age == "TOTAL") & (pop.unit == "NR")])
    gdp = eurostat.read("nama_10r_3gdp", "Eurostat GDP at current market prices by NUTS 3, "
                                         "SDMX TSV")
    gdp = _latest(gdp[gdp.unit == "MIO_EUR"])
    geo = geo[["NUTS_ID", "CNTR_CODE", "NAME_LATN", "geometry"]].rename(columns={"NUTS_ID": "geo"})
    geo = geo.merge(pop.rename(columns={"value": "pop", "year": "pop_year"}), on="geo", how="left")
    geo = geo.merge(gdp.rename(columns={"value": "gdp", "year": "gdp_year"}), on="geo", how="left")
    return geo


def country_outlines() -> gpd.GeoDataFrame:
    """GISCO country polygons (for maps and for clipping zone cells)."""
    g = gpd.read_file(fetch(CNTR_URL, "gisco/CNTR_RG_10M_2024_4326.geojson",
                            key="gisco_countries_2024",
                            notes="GISCO countries 2024, 1:10M, EPSG:4326"))
    return g

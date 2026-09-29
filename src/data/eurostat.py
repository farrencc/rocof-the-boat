"""Eurostat SDMX-TSV reader (keyless dissemination API)."""

from __future__ import annotations

import pandas as pd

from src.data.cache import fetch

URL = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{ds}?format=TSV&compressed=true"


def read(ds: str, notes: str = "") -> pd.DataFrame:
    """Long table: dimension columns + ``year`` + ``value`` (+ ``flag``)."""
    p = fetch(URL.format(ds=ds), f"eurostat/{ds}.tsv.gz", key=f"eurostat_{ds}",
              notes=notes or f"Eurostat {ds}, SDMX TSV (full dataset)")
    df = pd.read_csv(p, sep="\t", compression="gzip", dtype=str)
    key = df.columns[0]
    dims = key.split("\\")[0].split(",")
    df[dims] = df[key].str.split(",", expand=True)
    df = df.drop(columns=key)
    df.columns = [c.strip() for c in df.columns]
    years = [c for c in df.columns if c[:2] in ("19", "20")]
    long = df.melt(id_vars=dims, value_vars=years, var_name="year", value_name="raw")
    raw = long["raw"].fillna(":").str.strip()
    long["value"] = pd.to_numeric(raw.str.split(" ").str[0].replace(":", None), errors="coerce")
    long["flag"] = raw.str.split(" ").str[1].fillna("")
    long["year"] = long["year"].astype(int)
    return long.drop(columns="raw")

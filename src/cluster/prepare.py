"""Edge statistics + the normalisation checkpoint (gate) -> reports/normalisation.md.

    python -m src.cluster.prepare

Reads the solved prices (never re-solves), applies the load-shedding policy,
computes per-edge congestion statistics and per-country normalisation, and
writes diagnostics, figures and ``results/edges.parquet``.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.cluster import edges as E

INTERIM = config.ROOT / "data" / "interim"
SOLVED = config.ROOT / "data" / "solved"
RESULTS = config.ROOT / "results"
REPORTS = config.ROOT / "reports"
FIGS = config.ROOT / "figures"


def load_solution(cfg):
    buses = pd.read_csv(INTERIM / "buses.csv", index_col=0)
    lines = pd.read_csv(INTERIM / "lines.csv", index_col=0)
    links = pd.read_csv(INTERIM / "links.csv", index_col=0)
    prices = pd.read_parquet(SOLVED / "prices.parquet")
    sn = pd.read_csv(SOLVED / "snapshots.csv", index_col=0, parse_dates=True).weight
    shed = pd.read_parquet(SOLVED / "shed.parquet")
    return buses, lines, links, prices, sn, shed


def shedding_policy(prices: pd.DataFrame, weights: pd.Series, shed: pd.DataFrame, buses, cfg):
    """Return (prices_used, weights_used, report dict)."""
    pol = cfg["solve"]["shedding_policy"]
    tol = cfg["solve"].get("shed_tol_mw", 1.0)
    s = shed[shed.mw > tol]
    s = s.assign(country=s.bus.map(buses.country))
    shed_snaps = pd.DatetimeIndex(s.snapshot.unique())
    by_c = s.groupby("country").agg(hours=("snapshot", "nunique"), mwh=("mw", "sum"),
                                     buses=("bus", "nunique"))
    top_bus = s.groupby("bus").agg(hours=("snapshot", "nunique"), mwh=("mw", "sum")).sort_values(
        "hours", ascending=False).head(15)
    top_bus["country"] = top_bus.index.map(buses.country)
    top_bus["lon"] = top_bus.index.map(buses.x)
    top_bus["lat"] = top_bus.index.map(buses.y)
    rep = {"policy": pol, "tol_mw": tol, "n_snapshots": int(len(prices)),
           "shed_snapshots": int(len(shed_snaps)),
           "shed_weight_share": float(weights.reindex(shed_snaps).sum() / weights.sum()),
           "shed_twh": float((s.mw * weights.reindex(pd.DatetimeIndex(s.snapshot)).to_numpy()).sum() / 1e6),
           "by_country": by_c, "top_buses": top_bus}
    cap = cfg["solve"]["price_clip"]
    rep["price_extremes"] = {"min": float(np.nanmin(prices.to_numpy())),
                             "max": float(np.nanmax(prices.to_numpy())),
                             "share_bus_hours_above_cap": float((prices.to_numpy() > cap).mean()),
                             "share_bus_hours_below_minus_cap": float((prices.to_numpy() < -cap).mean())}
    if pol == "exclude":
        keep = prices.index.difference(shed_snaps)
        return prices.loc[keep], weights.reindex(keep), rep
    if pol == "clip":
        return prices.clip(-cap, cap), weights.reindex(prices.index), rep
    if pol == "exclude_country":
        # per-country exclusion is applied in edge statistics (see prepare())
        return prices, weights.reindex(prices.index), rep
    raise ValueError(pol)


def degeneracy(prices: pd.DataFrame, cfg, buses) -> dict:
    p2f = SOLVED / "prices_seed2.parquet"
    if not p2f.exists():
        return {"note": "no second-seed solve found"}
    p2 = pd.read_parquet(p2f)
    p1 = prices.reindex(p2.index)
    d = (p1 - p2).abs()
    tol = cfg["solve"]["degeneracy_tol"]
    flag = d > tol
    by_c = flag.T.groupby(buses.country.reindex(flag.columns).to_numpy()).mean().mean(axis=1)
    return {"n_snapshots": int(len(p2)), "tol": tol, "share_bus_hours": float(flag.to_numpy().mean()),
            "share_snapshots_any": float(flag.any(axis=1).mean()),
            "median_abs_diff": float(np.nanmedian(d.to_numpy())),
            "p99_abs_diff": float(np.nanquantile(d.to_numpy(), 0.99)), "by_country": by_c}


def plot_distributions(norm: pd.DataFrame, fn, title):
    cs = sorted(norm.country.unique())
    ncol = 6
    nrow = int(np.ceil(len(cs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.1 * nrow), squeeze=False)
    for ax, c in zip(axes.ravel(), cs):
        g = norm[norm.country == c]
        dp = g.dp_n.to_numpy()
        J = g.J_n[~g.is_dc].to_numpy()
        hi = max(np.quantile(dp, 0.995) if len(dp) else 1, np.quantile(J, 0.995) if len(J) else 1, 2)
        bins = np.linspace(0, hi, 30)
        ax.hist(np.clip(J, 0, hi), bins=bins, alpha=0.6, color="C0", label="J̃")
        ax.hist(np.clip(dp, 0, hi), bins=bins, alpha=0.6, color="C3", label="Δp̃")
        ax.set_title(f"{c} (n={len(g)})", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_yscale("log")
    for ax in axes.ravel()[len(cs):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(fn, dpi=150)
    plt.close(fig)


def prepare(cfg: dict | None = None, write=True):
    cfg = cfg or config.load()
    buses, lines, links, prices, sn, shed = load_solution(cfg)
    P, w, shed_rep = shedding_policy(prices, sn, shed, buses, cfg)
    deg = degeneracy(prices, cfg, buses)
    et = E.edge_table(buses, lines, links)
    st = E.congestion_stats(et, P, w, cfg)
    raw_cfg = json.loads(json.dumps(cfg))
    raw_cfg["edges"]["dp_remedy"] = "none"
    raw_cfg["edges"]["J_remedy"] = "none"
    raw = E.normalise(st, raw_cfg)
    diag_raw = E.diagnostics(raw)
    norm = E.normalise(st, cfg)
    diag = E.diagnostics(norm)
    if write:
        RESULTS.mkdir(exist_ok=True)
        norm.to_parquet(RESULTS / "edges.parquet")
        plot_distributions(raw, FIGS / "normalisation_raw.png",
                           "Per-country distributions before remedy (log counts)")
        plot_distributions(norm, FIGS / "normalisation_remedied.png",
                           f"After remedy: Δp {cfg['edges']['dp_remedy']}, J {cfg['edges']['J_remedy']}")
    return {"edges": norm, "raw": raw, "diag_raw": diag_raw, "diag": diag, "shed": shed_rep,
            "deg": deg, "prices_used": P, "weights_used": w}


if __name__ == "__main__":
    out = prepare()
    pd.set_option("display.width", 250)
    print(out["diag_raw"].round(3).to_string())
    print(out["diag"].round(3).to_string())
    print({k: v for k, v in out["shed"].items() if not isinstance(v, pd.DataFrame)})
    print({k: v for k, v in out["deg"].items() if not isinstance(v, pd.Series)})

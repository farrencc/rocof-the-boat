"""Figures written by the sweep and by the reporting stages."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bzgen import config
from bzgen.plot import maps

ROOT = config.ROOT
FIGS = ROOT / "figures"
INTERIM = ROOT / "data" / "interim"


def _net():
    lines = pd.read_csv(INTERIM / "lines.csv", index_col=0)
    links = pd.read_csv(INTERIM / "links.csv", index_col=0)
    return lines, links


def members_of(buses: pd.DataFrame) -> dict:
    return {c: sorted(buses[buses.cluster_country == c].country.unique())
            for c in buses.cluster_country.unique()}


def config_figures(cid: str, labels: pd.DataFrame, rows: pd.DataFrame, buses: pd.DataFrame,
                   dpi: int = 110) -> None:
    lines, links = _net()
    mem = members_of(buses)
    lab = labels.set_index("bus").zone
    (FIGS / "zones").mkdir(parents=True, exist_ok=True)
    (FIGS / "europe").mkdir(parents=True, exist_ok=True)
    cs = [c for c in rows.sort_values("n_nodes", ascending=False).country
          if c in set(labels.country)]
    ncol = 6
    nrow = int(np.ceil(len(cs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
    for ax, c in zip(axes.ravel(), cs):
        r = rows.set_index("country").loc[c]
        l = lab[labels.set_index("bus").country == c]
        flag = "" if r.contiguous else " NONCONTIG"
        flag += " DEGEN" if r.get("degenerate", False) else ""
        maps.draw_country(ax, buses, lines, links, l, mem[c],
                          title=f"{c} k={int(r.k)}{'' if r.elbow_convincing else '*'}{flag}")
    for ax in axes.ravel()[len(cs):]:
        ax.axis("off")
    fig.suptitle(f"Zones, configuration {cid}   (* = k from fallback, not a convincing elbow; "
                 "dashed = HVDC)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "zones" / f"{cid}.png", dpi=dpi)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 10))
    maps.europe_map(ax, buses, lines, links, lab, mem, title=f"Stitched zones — {cid}")
    fig.tight_layout()
    fig.savefig(FIGS / "europe" / f"{cid}.png", dpi=150)
    plt.close(fig)


def country_maps(cid: str, labels: pd.DataFrame, rows: pd.DataFrame, buses: pd.DataFrame) -> None:
    """High-resolution individual country maps (headline configuration)."""
    lines, links = _net()
    mem = members_of(buses)
    lab = labels.set_index("bus").zone
    out = FIGS / "zones" / cid
    out.mkdir(parents=True, exist_ok=True)
    for c in sorted(set(labels.country)):
        r = rows.set_index("country").loc[c]
        fig, ax = plt.subplots(figsize=(6, 6))
        maps.draw_country(ax, buses, lines, links, lab[labels.set_index("bus").country == c],
                          mem[c], title=f"{c}: k={int(r.k)} "
                                        f"({'eigengap' if r.elbow_convincing else 'fallback k'}); "
                                        f"load shares {r.zone_load_share}")
        fig.tight_layout()
        fig.savefig(out / f"{c}.png", dpi=200)
        plt.close(fig)


def sensitivity(df: pd.DataFrame, cfg: dict) -> None:
    ok = df[df.status == "ok"]
    base = cfg["sweep"]["baseline"]
    a = ok[(ok.lambda_c_initial == base["lambda_c_initial"]) & (ok.lambda_b == base["lambda_b"])
           & (ok.dk == base["dk"])]
    if a.alpha.nunique() < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for c, g in a.groupby("country"):
        g = g.sort_values("alpha")
        big = g.n_nodes.iloc[0] >= 100
        kw = dict(lw=1.2 if big else 0.6, alpha=0.9 if big else 0.35)
        axes[0].plot(g.alpha, g.n_zones_above_floor, marker="o", ms=3, **kw,
                     label=c if big else None)
        axes[1].plot(g.alpha, g.n_zones_effective, marker="o", ms=3, **kw)
        axes[2].plot(g.alpha, g.energy / g.n_nodes, marker="o", ms=3, **kw)
    axes[0].set_ylabel("zones with max(load, gen) share ≥ floor")
    axes[1].set_ylabel("effective number of zones (1/Σ load share²)")
    axes[2].set_ylabel("final energy per node")
    for ax in axes:
        ax.set_xlabel("α"); ax.set_xscale("log")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Sensitivity to α (k fixed per country; other parameters at baseline). "
                 "α is not comparable across countries (per-country normalisation).", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "sensitivity_alpha.png", dpi=150)
    plt.close(fig)
    if "ari_vs_baseline" in a:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for c, g in a.groupby("country"):
            g = g.sort_values("alpha")
            y = g.ari_vs_baseline.fillna(1.0).where(g.alpha != base["alpha"], 1.0)
            ax.plot(g.alpha, y, marker="o", ms=3, lw=0.8, alpha=0.6)
        ax.set_xscale("log"); ax.set_xlabel("α"); ax.set_ylabel("ARI vs baseline map (α=1)")
        ax.set_title("How much the map changes with α", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIGS / "sensitivity_alpha_ari.png", dpi=150)
        plt.close(fig)


def price_maps(prices: pd.DataFrame, weights: pd.Series, buses: pd.DataFrame, edges: pd.DataFrame,
               cfg: dict) -> None:
    lines, _ = _net()
    w = weights.reindex(prices.index).to_numpy()
    mean = pd.Series((w[:, None] * prices.to_numpy()).sum(0) / w.sum(), index=prices.columns)
    lo, hi = np.nanquantile(mean, [0.02, 0.98])
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    sc = maps.price_map(axes[0], buses, lines, mean, vmin=lo, vmax=hi)
    fig.colorbar(sc, ax=axes[0], shrink=0.6, label="weighted mean nodal price (EUR/MWh)")
    axes[0].set_title("Time-weighted mean nodal price (colour clipped to 2–98 %)", fontsize=10)
    stat = {"mean": "dp_mean", "duration": "dp_duration", "quantile": "dp_quantile"}[cfg["edges"]["statistic"]]
    ac = edges[~edges.is_dc]
    ev = pd.Series(ac[stat].to_numpy(), index=[s.split(";")[0] for s in ac.line_ids])
    maps.price_map(axes[1], buses, lines, pd.Series(np.nan, index=buses.index), edge_values=ev)
    sm = plt.cm.ScalarMappable(cmap="magma_r", norm=plt.Normalize(ev.min(), ev.max()))
    fig.colorbar(sm, ax=axes[1], shrink=0.6, label=f"{stat} across intra-country edges")
    axes[1].set_title(f"The clustering input: {stat} of |Δp| per intra-country line "
                      "(cross-border lines not shown: they never enter)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "nodal_prices.png", dpi=150)
    plt.close(fig)

"""Exploratory scan: dDD (split - k=1) over the sweep's alpha maps x ATC derating.

    python -m bzgen.validate.scan [--tag rep200]

Reads cached runs ``data/validate/<tag>/runs/{k1,sw_<cid>}_d<derating>/hourly.parquet``
(produced by ``bzgen.validate.run --sweep-maps ...``), pairs hours feasible in both,
and writes results/validate/scan_<tag>.csv and figures/validate/scan_<tag>.png.

EXPLORATORY: scanning settings for the one that reduces dispatch down selects on
noise.  In-sample maps, weighted representative hours, no bootstrap: any setting
found here is a candidate to confirm out of sample (refit on odd weeks, score on
even weeks), not a result.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap

import matplotlib
import matplotlib.patches
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bzgen import config

ROOT = config.ROOT


def collect(tag: str) -> pd.DataFrame:
    base = ROOT / "data" / "validate" / tag
    w = pd.read_csv(base / "snapshots.csv", index_col=0, parse_dates=True).weight
    rows = []
    for d in sorted((base / "runs").glob("sw_*_d*")):
        if not (d / "done.json").exists():
            continue
        m = re.match(r"sw_(a([\d.]+)_.+)_d([\d.]+)$", d.name)
        cid, alpha, der = m.group(1), float(m.group(2)), float(m.group(3))
        kf = base / "runs" / f"k1_d{der:g}" / "hourly.parquet"
        if not kf.exists():
            continue
        hs, hk = pd.read_parquet(d / "hourly.parquet"), pd.read_parquet(kf)
        ix = hs.index.intersection(hk.index)
        ok = hs.stageB_ok.reindex(ix).astype(bool) & hk.stageB_ok.reindex(ix).astype(bool)
        ix = ix[ok.to_numpy()]
        ww = w.reindex(ix)
        sc = 8760.0 / ww.sum() / 1e3
        f = lambda h, c: float((h.loc[ix, c] * ww).sum() * sc)
        rows.append({"config": cid, "alpha": alpha, "derating": der, "hours": len(ix),
                     "DD_k1_GWh": f(hk, "DD_RES"), "DD_split_GWh": f(hs, "DD_RES"),
                     "dDD_RES_GWh": f(hs, "DD_RES") - f(hk, "DD_RES"),
                     "dDD_RES_TH_GWh": f(hs, "DD_RES_TH") - f(hk, "DD_RES_TH"),
                     "dSpill_market_GWh": f(hs, "spill_res") - f(hk, "spill_res"),
                     "dRedispatch_down_res_GWh": f(hs, "down_res") - f(hk, "down_res"),
                     "dNP_slack_focus_GWh": f(hs, "np_slack_focus") - f(hk, "np_slack_focus"),
                     "dCost_total_EUR": (f(hs, "C_market") + f(hs, "C_redispatch")
                                         - f(hk, "C_market") - f(hk, "C_redispatch")) * 1e3})
    return pd.DataFrame(rows).sort_values(["alpha", "derating"])


def heatmap(df: pd.DataFrame, fn, tag: str) -> None:
    piv = df.pivot(index="alpha", columns="derating", values="dDD_RES_GWh") / 1e3
    lim = float(np.nanmax(np.abs(piv.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{a:g}" for a in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 * lim else "#0b0b0b")
    ax.set_xlabel("inter-zone ATC derating", fontsize=8)
    ax.set_ylabel("coupling α (sweep map)", fontsize=8)
    ax.set_title(f"ΔDD_RES, DE split − DE = 1 zone (TWh/yr; blue = split reduces DD)\n"
                 f"EXPLORATORY: in-sample maps, {tag} representative hours", fontsize=8)
    fig.colorbar(im, ax=ax, label="TWh/yr")
    fig.tight_layout()
    fig.savefig(fn, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rep200")
    args = ap.parse_args()
    df = collect(args.tag)
    (ROOT / "results" / "validate").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results" / "validate" / f"scan_{args.tag}.csv", index=False)
    (ROOT / "figures" / "validate").mkdir(parents=True, exist_ok=True)
    if len(df):
        heatmap(df, ROOT / "figures" / "validate" / f"scan_{args.tag}.png", args.tag)
    print(df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()


def derating_panel(df: pd.DataFrame, der: float, tag: str) -> pd.DataFrame:
    """Figure + table at one derating: dDD by alpha (scan) and its two components, with the
    out-of-sample point for the headline alpha (results/validate.csv) where it exists."""
    d = df[np.isclose(df.derating, der)].sort_values("alpha").copy()
    d["dDD_pct"] = 100 * d.dDD_RES_GWh / d.DD_k1_GWh
    oos = None
    f = ROOT / "results" / "validate.csv"
    if f.exists():
        r = pd.read_csv(f)
        r = r[(r.scoring == "oos_even") & np.isclose(r.derating, der) & (r["map"] == "oos_headline")]
        if len(r):
            cfg = config.load()
            p = json.loads((ROOT / "results" / "configs" / cfg["sweep"]["headline"] / "done.json")
                           .read_text())["params"]
            oos = dict(alpha=p["alpha"], **r.iloc[0][["dDD_RES_GWh", "dDD_RES_lo", "dDD_RES_hi"]].to_dict())
    x = np.arange(len(d))
    lab = [f"{a:g}" for a in d.alpha]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax = axes[0]
    ax.bar(x, d.dDD_RES_GWh / 1e3, 0.6, color="#2a78d6", edgecolor="white",
           label="scan: in-sample map, 200 representative hours")
    for xi, v, pc in zip(x, d.dDD_RES_GWh / 1e3, d.dDD_pct):
        ax.text(xi, v + (0.05 if v >= 0 else -0.05), f"{v:+.2f}\n({pc:+.0f}%)", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7, color="#0b0b0b")
    if oos is not None and oos["alpha"] in list(d.alpha):
        i = list(d.alpha).index(oos["alpha"])
        ax.errorbar(i + 0.38, oos["dDD_RES_GWh"] / 1e3,
                    yerr=[[(oos["dDD_RES_GWh"] - oos["dDD_RES_lo"]) / 1e3],
                          [(oos["dDD_RES_hi"] - oos["dDD_RES_GWh"]) / 1e3]],
                    fmt="o", color="#eb6834", ms=6, capsize=3, lw=2,
                    label="out-of-sample check (refit map, held-out weeks, 95% CI)")
    ax.axhline(0, color="#52514e", lw=1)
    ax.set_xticks(x, lab)
    ax.set_xlabel("coupling α", fontsize=8)
    ax.set_ylabel("ΔDD_RES, DE split − DE = 1 zone (TWh/yr)", fontsize=8)
    ax.set_title(f"Dispatch-down change at derating {der:g} (below 0 = split reduces DD)", fontsize=9)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.1 * (hi - lo))
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    ax = axes[1]
    sp, rd = d.dSpill_market_GWh.to_numpy() / 1e3, d.dRedispatch_down_res_GWh.to_numpy() / 1e3
    ax.bar(x, sp, 0.6, color="#eb6834", edgecolor="white", label="market curtailment")
    ax.bar(x, rd, 0.6, color="#1baf7a", edgecolor="white", label="redispatch down")
    ax.plot(x, sp + rd, "o", color="#0b0b0b", ms=5, label="net ΔDD")
    ax.axhline(0, color="#52514e", lw=1)
    ax.set_xticks(x, lab)
    ax.set_xlabel("coupling α", fontsize=8)
    ax.set_ylabel("change vs DE = 1 zone (TWh/yr)", fontsize=8)
    ax.set_title("Where the change comes from", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(axis="y", color="#e6e5e0", lw=0.6)
        a.set_axisbelow(True)
        a.tick_params(labelsize=8)
    fig.suptitle(f"EXPLORATORY scan ({tag}); the out-of-sample check is the test", fontsize=8,
                 color="#52514e")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "validate" / f"scan_{tag}_d{der:g}.png", dpi=150)
    plt.close(fig)
    t = d[["alpha", "DD_k1_GWh", "DD_split_GWh", "dDD_RES_GWh", "dDD_pct", "dSpill_market_GWh",
           "dRedispatch_down_res_GWh", "dDD_RES_TH_GWh", "dNP_slack_focus_GWh", "dCost_total_EUR"]]
    t.to_csv(ROOT / "results" / "validate" / f"scan_{tag}_d{der:g}.csv", index=False)
    return t, oos


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, colors="#52514e")


def _tint(hex_color: str, a: float) -> str:
    """Colour as drawn at opacity ``a`` on white (how the zone maps render bzgen.plot.maps.PALETTE)."""
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return matplotlib.colors.to_hex(a * rgb + (1 - a) * 1.0)


# the EU-wide zone maps' palette (bzgen.plot.maps.PALETTE at map opacity): blue, sandy, green
_A = 0.65
BLUE, SAND, GREEN = (_tint(c, _A) for c in ("#4e79a7", "#f28e2b", "#59a14f"))
XLABEL = "Connectivity coupling α"


def readable_panels(df: pd.DataFrame, der: float, tag: str) -> list:
    """The derating panel as two short, stand-alone figures (no footnotes)."""
    d = df[np.isclose(df.derating, der)].sort_values("alpha").copy()
    d["pct"] = 100 * d.dDD_RES_GWh / d.DD_k1_GWh
    x = np.arange(len(d))
    lab = [f"{a:g}" for a in d.alpha]
    out = []

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    v = d.dDD_RES_GWh.to_numpy() / 1e3
    ax.bar(x, v, 0.6, color=BLUE, edgecolor="white")
    for xi, vi, pc in zip(x, v, d.pct):
        ax.text(xi, vi + (0.05 if vi >= 0 else -0.05), f"{vi:+.1f} TWh\n({pc:+.0f}%)", ha="center",
                va="bottom" if vi >= 0 else "top", fontsize=8, color="#0b0b0b")
    ax.axhline(0, color="#52514e", lw=1)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.12 * (hi - lo))
    ax.set_xticks(x, lab)
    ax.set_xlabel(XLABEL, fontsize=9, color="#52514e")
    ax.set_ylabel("Change in wind + solar dispatch down (TWh/yr)", fontsize=9, color="#52514e")
    ax.set_title("Split DE vs one zone (below 0 = less dispatch down)", fontsize=10,
                 color="#0b0b0b", loc="left")
    _clean(ax)
    fig.tight_layout()
    f1 = ROOT / "figures" / "validate" / f"readable_dd_change_d{der:g}.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    out.append(f1)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    sp, rd = d.dSpill_market_GWh.to_numpy() / 1e3, d.dRedispatch_down_res_GWh.to_numpy() / 1e3
    ax.bar(x, sp, 0.6, color=SAND, edgecolor="white", label="Market dispatch down")
    ax.bar(x, rd, 0.6, color=GREEN, edgecolor="white", label="Redispatch dispatch down")
    ax.plot(x, sp + rd, "o", color="#2e4a6b", ms=6, label="Net change")
    ax.axhline(0, color="#52514e", lw=1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%+.1f"))
    ax.set_xticks(x, lab)
    ax.set_xlabel(XLABEL, fontsize=9, color="#52514e")
    ax.set_ylabel("Change in wind + solar dispatch down (TWh/yr)", fontsize=9, color="#52514e")
    ax.set_title("Where the change comes from", fontsize=10, color="#0b0b0b", loc="left")
    _clean(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    f2 = ROOT / "figures" / "validate" / f"readable_dd_sources_d{der:g}.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    out.append(f2)
    return out


def de_zone_map(cid: str, focus: str = "DE"):
    """Map of the focus country's zones for one sweep configuration (EU-map colours)."""
    from bzgen.plot import maps as pm
    buses = pd.read_csv(ROOT / "data" / "interim" / "buses.csv", index_col=0)
    lines = pd.read_csv(ROOT / "data" / "interim" / "lines.csv", index_col=0)
    links = pd.read_csv(ROOT / "data" / "interim" / "links.csv", index_col=0)
    lab = pd.read_csv(ROOT / "results" / "configs" / cid / "labels.csv")
    lab = lab[lab.country == focus].set_index("bus").zone.astype(int)
    row = pd.read_csv(ROOT / "results" / "configs" / cid / "rows.csv").set_index("country").loc[focus]
    shares = [float(x) for x in str(row.zone_load_share).split("/")]
    alpha = float(re.match(r"a([\d.]+)_", cid).group(1))
    mem = sorted(buses[buses.cluster_country == focus].country.unique())
    fig, ax = plt.subplots(figsize=(5.4, 6.4))
    pm.draw_country(ax, buses, lines, links, lab, mem)
    order = lab.value_counts().sort_index().index
    handles = [matplotlib.patches.Patch(color=_tint(pm.PALETTE[int(z) % len(pm.PALETTE)], 0.55),
                                        label=f"Zone {i + 1}: {s:.0%} of load")
               for i, (z, s) in enumerate(zip(order, _shares_by_label(lab, cid, focus, shares)))]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower left")
    ax.set_title(f"Germany: 3 bidding zones, {XLABEL.lower()} = {alpha:g}", fontsize=10, loc="left")
    fig.tight_layout()
    f = ROOT / "figures" / "validate" / f"de_zones_a{alpha:g}.png"
    fig.savefig(f, dpi=160)
    plt.close(fig)
    return f


def _shares_by_label(lab, cid, focus, shares_sorted):
    """rows.csv lists load shares sorted by zone size; recompute per label from the load key."""
    from bzgen.cluster.sweep import node_attributes
    L, _ = node_attributes(config.load())
    lz = L.reindex(lab.index).fillna(0.0).groupby(lab).sum()
    return list((lz / lz.sum()).sort_index())

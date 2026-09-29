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
import re

import matplotlib
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

"""DBZ figures -> figures/dbz/."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bzgen import config

FIGS = config.ROOT / "figures" / "dbz"


def _save(fig, name: str, dpi: int = 130):
    FIGS.mkdir(parents=True, exist_ok=True)
    tmp = FIGS / f".{name}.tmp.png"
    fig.savefig(tmp, dpi=dpi)
    plt.close(fig)
    tmp.replace(FIGS / name)
    return FIGS / name


def dp_distribution(edges_n: pd.DataFrame, name: str = "dp_distribution.png"):
    """dp~ and J~ on intra-country vs cross-border edges (AC and HVDC separately for dp~)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    intra = edges_n[~edges_n.is_cross]
    xb = edges_n[edges_n.is_cross]
    hi = float(np.quantile(edges_n.dp_n, 0.999))
    bins = np.linspace(0, max(hi, 2.0), 60)
    ax = axes[0]
    ax.hist(np.clip(intra.dp_n, 0, bins[-1]), bins=bins, density=True, alpha=0.55,
            label=f"intra-country (n={len(intra)})", color="C0")
    ax.hist(np.clip(xb.dp_n, 0, bins[-1]), bins=bins, density=True, alpha=0.55,
            label=f"cross-border (n={len(xb)})", color="C3")
    ax.set_yscale("log"); ax.set_xlabel("Δp̃ (clipped at the 99.9th pct for display)")
    ax.set_ylabel("density"); ax.legend(fontsize=8); ax.set_title("Δp̃ by edge class")
    ax = axes[1]
    for lab, sub, c in (("intra", intra, "C0"), ("cross AC", xb[~xb.is_dc], "C3"),
                        ("cross HVDC", xb[xb.is_dc], "k")):
        if len(sub):
            x = np.sort(sub.dp_n.to_numpy())
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, label=f"{lab} (n={len(x)})")
    ax.set_xscale("symlog", linthresh=0.1); ax.set_xlabel("Δp̃"); ax.set_ylabel("ECDF")
    ax.legend(fontsize=8); ax.set_title("Δp̃ ECDF")
    ax = axes[2]
    ac_i = intra[~intra.is_dc].J_n
    ac_x = xb[~xb.is_dc].J_n
    b2 = np.linspace(0, max(float(np.quantile(edges_n[~edges_n.is_dc].J_n, 0.999)), 2.0), 60)
    ax.hist(ac_i, bins=b2, density=True, alpha=0.55, color="C0", label="intra AC")
    ax.hist(ac_x, bins=b2, density=True, alpha=0.55, color="C3", label="cross-border AC")
    ax.set_yscale("log"); ax.set_xlabel("J̃"); ax.legend(fontsize=8); ax.set_title("J̃ (AC) by edge class")
    fig.suptitle("European graph: node-averaged normalisation (country scales from "
                 "intra-country edges only)", fontsize=10)
    fig.tight_layout()
    return _save(fig, name)

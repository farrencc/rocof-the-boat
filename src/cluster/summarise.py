"""Summarise the sweep -> reports/results.md, headline maps, sensitivity figures.

    python -m src.cluster.summarise
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.cluster.sweep import CONFIGS, RESULTS, cid_of
from src.plot import figures

REPORTS = config.ROOT / "reports"


def main(narrative: str = ""):
    cfg = config.load()
    df = pd.read_csv(RESULTS / "sweep.csv")
    buses = pd.read_csv(config.ROOT / "data" / "interim" / "buses.csv", index_col=0)
    head = cfg["sweep"].get("headline") or cid_of(cfg["sweep"]["baseline"])
    figures.sensitivity(df, cfg)
    lab = pd.read_csv(CONFIGS / head / "labels.csv")
    rows = pd.read_csv(CONFIGS / head / "rows.csv")
    figures.country_maps(head, lab, rows, buses)
    ok = df[df.status == "ok"]
    per_cfg = ok.groupby("config_id").agg(
        role=("role", "first"), alpha=("alpha", "first"), lambda_c_initial=("lambda_c_initial", "first"),
        lambda_b=("lambda_b", "first"), dk=("dk", "first"), countries=("country", "size"),
        non_contiguous=("contiguous", lambda s: int((~s.astype(bool)).sum())),
        degenerate=("degenerate", lambda s: int(s.fillna(False).astype(bool).sum())),
        median_frac_w_pos=("frac_edges_w_positive", "median"),
        median_cut_ratio=("cut_ratio", "median"),
        median_ari_best_vs_2nd=("ari_best_vs_2nd", "median"),
    ).sort_values(["role", "alpha", "lambda_c_initial", "lambda_b", "dk"])
    if "ari_vs_baseline" in ok:
        per_cfg["median_ari_vs_baseline"] = ok.groupby("config_id").ari_vs_baseline.median()
    below = ok.assign(nb=ok.k - ok.n_zones_above_floor).groupby("config_id").nb.sum()
    per_cfg["zones_below_floor_total"] = below
    h = ok[ok.config_id == head].set_index("country")
    big = h[h.n_nodes >= 100]
    hcols = ["n_nodes", "k", "elbow_convincing", "energy", "E_spread", "E_gap_2nd", "ari_best_vs_2nd",
             "restarts_contiguous", "frac_edges_w_positive", "cut_ratio", "zone_buses", "zone_load_share",
             "n_zones_above_floor"]
    sig = pd.read_csv(RESULTS / "spectrum.csv", index_col=0)
    lines = [
        "# Results: candidate bidding zones",
        "",
        "> Nodal prices are a proxy: these zones are drawn from prices that would change if the",
        "> zones were imposed (see README). Nothing here closes that loop.",
        "",
        f"Headline configuration: **`{head}`** (α, λ_c,initial, λ_b, Δk). "
        f"{len(df.config_id.unique())} configurations × {df.country.nunique()} countries in "
        "`results/sweep.csv`.",
        "",
        narrative,
        "",
        "![europe](../figures/europe/" + head + ".png)",
        "",
        "## Headline configuration, per country",
        "",
        "`E_gap_2nd` and `ari_best_vs_2nd` compare the best restart with the second best: a small",
        "energy gap with a low ARI means structurally different maps of almost equal energy",
        "(degeneracy). `frac_edges_w_positive` is the share of edges on which the Potts term is",
        "repulsive (w = Δp̃ − αJ̃ > 0); `cut_ratio` is the share of intra-country edges cut.",
        f"k was the fallback default in {int((~sig.convincing).sum())} of {len(sig)} countries.",
        "",
        h[[c for c in hcols if c in h]].round(3).to_markdown(),
        "",
        "Per-country maps: `figures/zones/" + head + "/<CC>.png`; all countries on one sheet: "
        "`figures/zones/" + head + ".png`.",
        "",
        "## All configurations (medians over countries)",
        "",
        per_cfg.round(3).to_markdown(),
        "",
        "## Sensitivity to α",
        "",
        "![alpha](../figures/sensitivity_alpha.png)",
        "",
        "![alpha-ari](../figures/sensitivity_alpha_ari.png)",
    ]
    conv = RESULTS / "convergence.json"
    if conv.exists():
        c = json.loads(conv.read_text())
        lines += ["", "## Convergence check (headline configuration)", "",
                  pd.DataFrame(c).round(3).to_markdown(index=False)]
    (REPORTS / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    nf = REPORTS / "results_narrative.md"
    main(nf.read_text() if nf.exists() else "")

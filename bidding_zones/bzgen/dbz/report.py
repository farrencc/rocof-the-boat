"""DBZ reports.

    python -m bzgen.dbz.report graph     # stage 1 -> results/dbz/edges.parquet, reports/dbz_graph.md
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from bzgen import config
from bzgen.cluster.sweep import cid_of, configurations, node_attributes
from bzgen.dbz import anchor as AN
from bzgen.dbz import balance_floor, io, load_config, plots
from bzgen.dbz import graph as DG

ROOT = config.ROOT
REPORTS = ROOT / "reports"
RESULTS = io.RESULTS


def anchor_params(cfg: dict) -> dict:
    """Sweep parameters (alpha, lambda_b, ...) of the anchor configuration."""
    cid = cfg["dbz"]["anchor_config_id"]
    for p in configurations(cfg):
        if cid_of(p) == cid:
            return p
    raise ValueError(f"anchor_config_id {cid!r} is not a configuration of the static sweep")


def md(df: pd.DataFrame, digits=3) -> str:
    return df.round(digits).to_markdown()


# --------------------------------------------------------------------------- #
# stage 1
# --------------------------------------------------------------------------- #

def graph_stage(cfg: dict | None = None, write: bool = True) -> dict:
    cfg = cfg or load_config()
    p = anchor_params(cfg)
    buses, en, sc = DG.compute_edges(cfg)
    L, G = node_attributes(cfg)
    cap = DG.installed_capacity(buses)
    nodes = DG.node_order(buses)
    an = AN.load(cfg["dbz"]["anchor_config_id"], nodes, buses)
    K = an["K"]
    floor = balance_floor(cfg["anneal"], K)
    g, nodes, e = DG.build(buses, en, p["alpha"], L, G, cap, K, cfg["edges"]["dc_in_energy"],
                           nodes=nodes)
    contig = AN.check_contiguity(an["A"], K, g, nodes, buses)
    summ = DG.summary(g, p["lambda_b"], floor)
    # reproduction check of the rebuilt inputs against the committed static edge table
    com = pd.read_parquet(ROOT / "results" / "edges.parquet")
    m = en[~en.is_cross].merge(com, on=["bus_a", "bus_b"], suffixes=("", "_static"))
    repro = {"n_intra": int((~en.is_cross).sum()), "n_static": len(com), "n_matched": len(m)}
    for c in ("dp_duration", "dp_mean", "dp_quantile", "dp_n", "J_n"):
        repro[f"max_abs_diff_{c}"] = float(np.abs(m[c] - m[f"{c}_static"]).max())
    out = {"cfg": cfg, "params": p, "buses": buses, "edges": en, "scales": sc, "graph": g,
           "nodes": nodes, "anchor": an, "contig": contig, "summary": summ, "repro": repro,
           "floor": floor, "L": L, "G": G, "cap": cap}
    if write:
        io.write_parquet(RESULTS / "edges.parquet", en)
        io.write_json(RESULTS / "graph.json", {"summary": summ, "repro": repro,
                                               "anchor_config_id": an["cid"],
                                               "alpha": p["alpha"], "lambda_b": p["lambda_b"],
                                               "isolated": contig["isolated"]})
        plots.dp_distribution(en)
        from bzgen.cluster.sweep import INTERIM
        plots.anchor_map(buses, pd.read_csv(INTERIM / "lines.csv", index_col=0),
                         pd.read_csv(INTERIM / "links.csv", index_col=0),
                         AN.read_static(an["cid"]), contig["isolated"], an["cid"])
        io.write_text(REPORTS / "dbz_graph.md", graph_report(out))
    return out


def _quantiles(x: pd.Series) -> dict:
    q = np.quantile(x, [0.5, 0.9, 0.99]) if len(x) else [np.nan] * 3
    return {"n": len(x), "mean": x.mean(), "median": q[0], "p90": q[1], "p99": q[2],
            "max": x.max(), "zero_frac": float((x == 0).mean()) if len(x) else np.nan}


def graph_report(o: dict) -> str:
    cfg, en, sc, g, summ = o["cfg"], o["edges"], o["scales"], o["graph"], o["summary"]
    e = cfg["edges"]
    xb = en[en.is_cross]
    intra = en[~en.is_cross]
    pairs = (xb.assign(pair=[f"{min(a, b)}–{max(a, b)}" for a, b in zip(xb.country_a, xb.country_b)])
             .groupby("pair").agg(edges=("is_dc", "size"), hvdc=("is_dc", "sum"),
                                  dp_raw_mean=("dp_raw", "mean"), dp_n_mean=("dp_n", "mean"),
                                  dp_n_max=("dp_n", "max"), J_n_mean=("J_n", "mean")))
    pairs["ac"] = pairs.edges - pairs.hvdc
    pairs = pairs[["edges", "ac", "hvdc", "dp_raw_mean", "dp_n_mean", "dp_n_max", "J_n_mean"]]
    dist = pd.DataFrame({"intra (all)": _quantiles(intra.dp_n),
                         "cross-border (all)": _quantiles(xb.dp_n),
                         "cross-border AC": _quantiles(xb[~xb.is_dc].dp_n),
                         "cross-border HVDC": _quantiles(xb[xb.is_dc].dp_n),
                         "intra raw stat": _quantiles(intra.dp_raw),
                         "cross-border raw stat": _quantiles(xb.dp_raw)}).T
    jdist = pd.DataFrame({"intra AC": _quantiles(intra[~intra.is_dc].J_n),
                          "cross-border AC": _quantiles(xb[~xb.is_dc].J_n)}).T
    w_und = g.undirected()
    w, isx = w_und[2], w_und[3]
    ns = DG.no_signal_table(en, sc, cfg)
    flagged = ns[ns.no_congestion_signal]
    iso = o["contig"]["isolated"]
    b = o["buses"]
    iso_rows = []
    for x in iso:
        nb = en[(en.bus_a == x) | (en.bus_b == x)]
        other = [bb if aa == x else aa for aa, bb in zip(nb.bus_a, nb.bus_b)]
        iso_rows.append({"bus": x, "country": b.at[x, "cluster_country"],
                         "neighbours": ", ".join(f"{b.at[u, 'cluster_country']}" for u in other),
                         "anchor_zone": int(o["anchor"]["A"][o["nodes"].index(x)])})
    iso_df = pd.DataFrame(iso_rows).set_index("bus")
    rp = o["repro"]
    n = cfg["network"]
    lines = [
        "# DBZ stage 1 — the European graph",
        "",
        "Generated by `python -m bzgen.dbz.report graph` (`bzgen/dbz/graph.py`, `bzgen/dbz/anchor.py`).",
        "One graph over every bus; lines and links between cluster-countries are kept. Nothing is",
        "re-solved: the per-edge statistics come from `data/solved/prices.parquet` under the static",
        f"shedding policy (`{cfg['solve']['shedding_policy']}`, ±{cfg['solve']['price_clip']:g} EUR/MWh).",
        "",
        "## Inputs and reproduction of the static edge table",
        "",
        "`data/interim` and `data/solved` are gitignored. They were regenerated in this environment",
        "with the unchanged pipeline (topology → weather → assembly → the seeded 8760 h DC-OPF), and",
        "the intra-country rows of the European table were checked against the committed",
        "`results/edges.parquet`:",
        "",
        f"- intra-country edges: {rp['n_intra']} (static table: {rp['n_static']}, matched on bus pair: {rp['n_matched']})",
        "- max |difference| per column: " + ", ".join(
            f"`{k.replace('max_abs_diff_', '')}` {v:.3g}" for k, v in rp.items() if k.startswith("max_abs")),
        "",
        "**Network build settings.** `bzgen/network/build.py` has no `remove_stubs_across_borders`",
        "switch: stage 5 hard-codes the PyPSA-Eur `remove_stubs_across_borders: false` behaviour",
        "(only degree-1 buses whose single AC line stays in the *same* country are absorbed;",
        "`buses.country` is compared, and cross-border stubs survive). Other settings:",
        f"`v_ref_kv={n['v_ref_kv']}`, `s_max_pu={n['s_max_pu']}`, `min_component={n['min_component']}`,",
        f"`drop_under_construction={n['drop_under_construction']}`, merged countries",
        f"`{cfg['scope']['merge_always']}` + any with < {cfg['scope']['min_buses']} buses. Countries",
        "are `buses.cluster_country` (after merging, e.g. LU → DE, ME → BA), as in the static pipeline.",
        "",
        "## Edges",
        "",
        f"- European edges: **{len(en)}** = {len(intra)} intra-country + **{len(xb)} cross-border**",
        f"  ({int((~xb.is_dc).sum())} AC corridors, {int(xb.is_dc.sum())} HVDC links) over {len(pairs)} country pairs.",
        f"- Convention: `country` is the cluster-country for intra-country edges and `\"{DG.XB}\"` for",
        "  every cross-border edge; `country_a`, `country_b`, `is_cross` carry the endpoints.",
        f"- Energy weights at the anchor's α = {o['params']['alpha']:g}: share of edges with w > 0 (repulsive):",
        f"  intra {float((w[~isx] > 0).mean()):.3f}, cross-border {float((w[isx] > 0).mean()):.3f}.",
        "",
        "Cross-border edges by country pair:",
        "",
        md(pairs, 3),
        "",
        "## Normalisation: intra-country vs cross-border Δp̃",
        "",
        f"Remedies: Δp `{e['dp_remedy']}` (q = {e['dp_clip_q']}), J `{e['J_remedy']}`; cross-border rule",
        f"`edges.cross_border_remedy: {e.get('cross_border_remedy')}` (mean of the two countries' clip thresholds /",
        "log1p medians, then division by the mean of the two country means). Country scales use",
        "intra-country edges only; restricted to those edges the result equals `edges.normalise`",
        "bit for bit (`tests/test_dbz.py::test_normalisation_reduces_to_static`).",
        "",
        "![dp distribution](../figures/dbz/dp_distribution.png)",
        "",
        md(dist, 3),
        "",
        "J̃ (AC only; HVDC has J = 0):",
        "",
        md(jdist, 3),
        "",
        _dist_narrative(intra, xb),
        "",
        "## `signal_min`: countries with no intra-country congestion signal",
        "",
        f"The static gate flags a country when the mean of its raw statistic over intra-country edges",
        f"is below `edges.signal_min = {e.get('signal_min')}` (`bzgen/cluster/prepare.py::no_signal`). The",
        "flag is **informational only** in the static pipeline: flagged countries are still",
        "normalised by their own (tiny) mean and partitioned with fixed k, and nothing is excluded.",
        "",
        f"**Decision (`edges.no_signal_policy: {e.get('no_signal_policy')}`).** The flag is recomputed from",
        "intra-country edges only, so it is unchanged, and it still excludes nothing: every bus of a",
        "flagged country is a European node, with its own country scale as its half of the",
        "denominator of each of its cross-border edges. Cross-border edges do not give a flagged",
        "country a new *scale*, but they can give it a *signal*: a border spread divided by a",
        "near-zero intra-country mean. Two things bound it. The clip threshold of a cross-border",
        "edge is the mean of the two countries' q99 thresholds, so a flagged country's small",
        "threshold pulls the clip down as much as its small mean pulls the denominator down. And",
        "when one endpoint has a real signal, its mean dominates the average denominator. The table",
        "shows, for every flagged country, what its cross-border edges carry (`xb_signal`: their",
        "raw mean ≥ `signal_min`, i.e. a signal the static model never saw).",
        "",
        md(flagged, 4) if len(flagged) else "_none_",
        "",
        "## Node set",
        "",
        f"- European nodes: **N = {g.n}** = every bus (static: 3763 annealed nodes + {len(iso)} buses",
        "  attached post hoc). No European node lacks an edge.",
        f"- The {len(iso)} buses that were *isolated within their country* (every line crosses a border) are",
        "  ordinary nodes now. The anchor gives them their static label, i.e. the zone of the nearest",
        "  same-country bus, which they are **not adjacent to**. Each is therefore a singleton fragment",
        "  of its anchor zone on the European graph. Every other anchor zone is connected (asserted in",
        "  `anchor.check_contiguity`; adding edges can only merge components). So A has",
        f"  Σ(C_s − 1) = {o['contig']['excess_components']} on the European graph, all from these buses, in zones",
        f"  {o['contig']['zones_with_fragment']}. A contiguous B must move them to a neighbouring (foreign) zone,",
        "  which costs a small, fixed amount of rigidity at every λ_rigid.",
        "",
        "![anchor and isolated buses](../figures/dbz/anchor_isolated.png)",
        "",
        md(iso_df, 0),
        "",
        "## Scales",
        "",
        f"- anchor configuration: `{o['anchor']['cid']}` (α = {o['params']['alpha']:g}, λ_b = {o['params']['lambda_b']:g});",
        f"  **K = {summ['K']}** = Σ_c k_c over {len(o['anchor']['k_c'])} cluster-countries",
        f"- total European load L_tot = {summ['Ltot_mw']:,.0f} MW (time-weighted mean)",
        f"- balance floor: `balance_floor_frac_of_mean` {cfg['anneal']['balance_floor_frac_of_mean']} / K = **{summ['floor']:.5f}**",
        f"  (static: absolute {cfg['anneal']['balance_floor']} against *national* load at k ≈ 3)",
        f"- rigidity weights g_i: installed capacity, mean ḡ = {summ['mean_cap_mw']:.1f} MW;",
        f"  buses with g_i = 0: {int((g.cap == 0).sum())} of {g.n}",
        f"- rigidity normalisation **Z = 2N/K · ḡ = {summ['Z']:,.1f}** MW",
        f"- `contiguity_guarantee(g, λ_b)` = Σ|w|/2 + 2λ_b + 1 = **{summ['contiguity_guarantee']:,.1f}**",
        f"- max degree {summ['maxdeg']}; BFS workspace `Q` = {summ['Q_bytes'] / 2**20:.2f} MiB (asserted < 64 MiB)",
    ]
    return "\n".join(lines) + "\n"


def _dist_narrative(intra: pd.DataFrame, xb: pd.DataFrame) -> str:
    r = xb.dp_n.mean() / intra.dp_n.mean()
    med_i, med_x = intra.dp_n.median(), xb.dp_n.median()
    return (f"Cross-border Δp̃ averages **{r:.2f}×** the intra-country mean (medians {med_x:.2f} vs "
            f"{med_i:.2f}). The numerator is a border spread; the denominator is the mean of two "
            "intra-country means. " +
            ("So, as expected, the borderless map still prefers to cut near national borders, "
             "by this factor on average." if r > 1 else
             "Contrary to the expectation, cross-border edges are *not* more repulsive on average; "
             "the borderless map has no built-in preference for national borders."))


# --------------------------------------------------------------------------- #
# stage 2
# --------------------------------------------------------------------------- #

def results_stage(cfg: dict, out, df: pd.DataFrame) -> None:
    """Figures + reports/dbz_results.md from every completed lambda in ``out``."""
    import json
    from pathlib import Path
    out = Path(out)
    if df.empty:
        return
    is_main = out.resolve() == RESULTS.resolve()
    fig_dir = None if is_main else out / "figures"
    anc = json.loads((out / "anchor_energy.json").read_text())
    rs = {r.lambda_rigid: pd.read_csv(out / f"lr{r.lambda_rigid:g}" / "restarts.csv")
          for r in df.itertuples()}
    tr = {}
    for r in df.itertuples():
        z = np.load(out / f"lr{r.lambda_rigid:g}" / "traces.npz")
        best = rs[r.lambda_rigid].sort_values("energy").restart.iloc[0]
        tr[r.lambda_rigid] = z[f"r{best}"]
    plots.pareto(df, rs, anc["physical"], anc["N"], fig_dir=fig_dir)
    plots.acceptance(tr, fig_dir=fig_dir)
    text = results_report(cfg, df, anc, rs, figs="../figures/dbz" if is_main else "figures")
    io.write_text(REPORTS / "dbz_results.md" if is_main else out / "dbz_results.md", text)


def results_report(cfg, df, anc, rs, figs="../figures/dbz") -> str:
    a = cfg["anneal"]
    p = anchor_params(cfg)
    cols = {"lambda_rigid": "λ_rigid", "physical": "physical", "potts": "Potts",
            "contig_excess": "Σ(C−1)", "balance_pen": "balance", "rigid": "rigidity",
            "energy_excl_contig": "E", "transfer_distance": "transfer",
            "transfer_share": "transfer %", "transfer_distance_cap_mw": "transfer MW",
            "ari_vs_anchor": "ARI vs A", "n_zones": "zones", "n_zones_multinational": "multinat.",
            "zone_buses_min": "size min", "zone_buses_median": "size med",
            "zone_buses_max": "size max", "n_zones_above_floor": "≥ floor"}
    t1 = df[list(cols)].rename(columns=cols).set_index("λ_rigid")
    cols2 = {"lambda_rigid": "λ_rigid", "cut_edges": "cut edges",
             "cut_share_cross_border": "cut: XB share", "cut_wpos_share_cross_border": "cut w+: XB",
             "cross_border_edges_cut": "XB edges cut", "cut_share_dc": "cut: DC share",
             "cut_wpos_share_dc": "cut w+: DC", "anchor_cut_edges_kept": "A cuts kept"}
    t2 = df[list(cols2)].rename(columns=cols2).set_index("λ_rigid")
    cols3 = {"lambda_rigid": "λ_rigid", "restarts": "restarts", "E_spread": "E spread",
             "E_gap_2nd": "gap 2nd", "ari_best_vs_2nd": "ARI best/2nd",
             "ari_restarts_mean": "ARI mean", "ari_restarts_min": "ARI min",
             "init_best": "best init", "E_min_voronoi": "E best voronoi",
             "E_min_anchor_seeded": "E best anchor-seeded", "T0": "T0",
             "accept_early_best": "accept (first 10 %)", "accept_collapse": "collapse < 1 %",
             "wall_s_restart_median": "s / restart", "wall_s": "wall s"}
    t3 = df[[c for c in cols3 if c in df]].rename(columns=cols3).set_index("λ_rigid")
    cols4 = {"lambda_rigid": "λ_rigid", "physical": "found: physical",
             "transfer_distance": "found: transfer", "energy_excl_contig": "found: E",
             "ref_physical": "ref: physical", "ref_transfer": "ref: transfer",
             "ref_energy": "ref: E", "gap_to_ref": "E gap (found − ref)",
             "ref_dominates_best": "ref dominates"}
    t4 = df[list(cols4)].rename(columns=cols4).set_index("λ_rigid")
    n_dom = int(df.ref_dominates_best.sum())
    collapse = df[df.accept_collapse]
    lines = [
        "# DBZ stage 2 — Dynamical Bidding Zones: results",
        "",
        "Generated by `python -m bzgen.dbz.sweep` (`bzgen/dbz/report.py`), regenerated after every",
        "completed λ_rigid.",
        "",
        "## The anchor (the most consequential input)",
        "",
        f"**A = `{anc['anchor_config_id']}`**, the static headline map (`sweep.headline`): α = {p['alpha']:g},",
        f"λ_c,initial = {p['lambda_c_initial']:g}, λ_b = {p['lambda_b']:g}, Δk = {p['dk']:+d}. It is the largest α on the",
        "static grid with ≥ 95 % of zones above the balance floor (`reports/results.md`), which makes",
        "it the static map the project stands behind; the sweep baseline (α = 1) was not used. B is",
        f"optimised with the same α and λ_b, so A is a static optimum of the physical terms B faces.",
        f"K = {anc['K']} zones, N = {anc['N']} nodes, Z = {anc['Z']:,.1f} MW, balance floor = 0.15/K =",
        f"{anc['floor']:.5f}. On the European graph A has physical objective **{anc['physical']:.1f}**",
        f"(Potts {anc['potts']:.1f}, balance {anc['balance']:.3f}) and Σ(C_s − 1) = {anc['contig']:.0f}: the 8",
        "formerly-isolated buses (`reports/dbz_graph.md`).",
        "",
        "## Method as run",
        "",
        f"- H = Potts + λ_c Σ(C_s−1) + λ_b Σ hinge + λ_rigid · H_rigid, with λ_c pinned at",
        f"  `contiguity_guarantee` ({df.lambda_c.iloc[0]:,.1f}) for the whole schedule.",
        "- H_rigid = (1/Z) Σ_i g_i |BZ_B(i) Δ BZ_A(i)|, g_i = installed capacity. It is kept",
        "  incrementally from K × K contingency tables (O(1) per single-node move, O(#A-zones) per fragment",
        "  move) and re-synchronised from the tables after every temperature; the best restart's totals",
        "  are re-checked against a full recomputation.",
        f"- schedule: {int(df.n_temps.iloc[0])} temperatures, **{df.sweeps_per_temp.iloc[0]:g} sweeps per temperature**",
        f"  (static: {config.load()['anneal']['sweeps_per_temp']}), t_final_ratio {a['t_final_ratio']}, quench {df.quench_sweeps.iloc[0]:g} sweeps,",
        f"  {int(df.restarts.iloc[0])} restarts per λ. T0 is calibrated on Potts + balance + rigidity (contiguity",
        "  excluded), so it grows with λ_rigid, and must lie in (1e-3, 1e3).",
        "- initial labels: never A. Half the restarts use `graph_voronoi` per connected component (random",
        "  seeds). The other half use graph-Voronoi growth from one random seed inside each anchor zone",
        "  (it starts ~50 % of nodes away from A). The lowest-energy restart under each λ's own",
        "  objective is reported, together with its init.",
        "",
        "## Pareto front: what stability costs",
        "",
        f"![pareto]({figs}/pareto.png)",
        "",
        "Physical objective (Potts + λ_b · balance, lower is better) against the transfer distance",
        "from A (N minus the maximum one-to-one matching of B-zones to A-zones on the contingency",
        "table: the number of nodes that changed zone). One point per λ_rigid (best restart); grey:",
        "the other restarts; red star: A itself (not contiguous on the European graph, see above).",
        "",
        md(t1, 3),
        "",
        "## Where the cuts are",
        "",
        "Shares of cut edges (and of the repulsive weight w⁺ on cut edges) that are cross-border (XB) or",
        "HVDC; `XB edges cut`: the share of cross-border edges that are still zone boundaries;",
        "`A cuts kept`: the share of A's cut edges that are also cut in B.",
        "",
        md(t2, 3),
        "",
        "## Restart spread, reproducibility, schedule",
        "",
        md(t3, 3),
        "",
        f"![acceptance]({figs}/acceptance.png)",
        "",
        ("**Acceptance collapse** (< 1 % over the first 10 % of the schedule) at λ_rigid = "
         + ", ".join(f"{x:g}" for x in collapse.lambda_rigid) + ": the high-temperature phase is doing little there."
         if len(collapse) else
         "No λ_rigid shows an acceptance collapse (< 1 %) in the first 10 % of the schedule."),
        "",
        "## Optimisation quality: a reference quench from A′",
        "",
        "Diagnostic, not part of the sweep. A′ is A with each formerly-isolated bus moved to its",
        "neighbouring (foreign) zone, the nearest contiguous map to A. A T = 0 quench (50 sweeps) from",
        "A′ under each λ's objective gives a point near the anchored optimum. If it beats the sweep's",
        "best restart, the annealer has not found the optimum at that λ, and the Pareto point there",
        "is an upper bound on the true cost of stability, not the cost itself.",
        "",
        md(t4, 2),
        "",
        (f"**The reference dominates the sweep's result (better physical objective *and* smaller transfer "
         f"distance) at {n_dom} of {len(df)} λ values.** The annealer, started away from A with λ_c "
         "pinned, does not find the anchored basin under single-node and fragment moves. Where "
         "this holds, read the front as what this search finds, not as the Pareto frontier."
         if n_dom else
         "The reference never dominates the sweep's best restart."),
        "",
        "## Zone maps",
        "",
    ]
    for r in df.itertuples():
        lines += [f"### λ_rigid = {r.lambda_rigid:g}", "",
                  f"![zones]({figs}/zones_lr{r.lambda_rigid:g}.png)", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "graph"
    if what == "graph":
        o = graph_stage()
        print(o["summary"], o["repro"])
    else:
        raise SystemExit(f"unknown report {what!r}")

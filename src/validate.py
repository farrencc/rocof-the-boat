"""Step 6: validation on fresh seeds + full-network post-relief recheck.

    python src/validate.py --scope 26

* Re-evaluates every completed configuration's best-E and best-D groupings, the
  initial shift-factor grouping, and both WDT baselines on VALIDATION_SEEDS x
  THERMAL_SCALES (seeds never used for anchors, normalisation or annealing).
  Identical frozen cases for every grouping.
* Full-network recheck: the security screen is frozen from BASELINE flows at a
  90% threshold, so relief can push flow onto a branch (or branch/outage pair)
  that is not watched. After relief we rebuild flows on ALL branches,
      F = branch_flows + curtail @ (H[:, balance] - H[:, node_bus]).T
  and check intact loading of every branch and N-1 loading of every
  (monitor, outage) pair via upstream's LODF. Any > 100% that is not a screened
  state is reported as an invisible overload, with the WDT baseline checked the
  same way so the grouping's own contribution is separable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import upstream as up  # noqa: E402
import anchors as A  # noqa: E402
import plots  # noqa: E402
from evaluator import build_ensemble  # noqa: E402
from hamiltonian import structural_terms, TERMS  # noqa: E402
from sweep import Problem, RESULTS, FIGURES, sweep_configs  # noqa: E402

TOL = 1e-6


class FullNetwork:
    """Post-relief flows on every branch, intact and N-1."""

    def __init__(self, grid, balance_bus: str, node_buses=None, replacement_ptdf=None):
        self.grid = grid
        H = up.ptdf(grid, balance_bus)
        self.lodf, self.valid = up.lodf(grid, H)
        if node_buses is None:
            node_buses = grid.template.bus.astype(str)
        node_bus = [grid.model.bus_index[str(b)] for b in node_buses]
        # node x branch: 1 MW curtailed at node, replaced at the balance bus (upstream convention)
        repl = H[:, grid.model.bus_index[balance_bus]] if replacement_ptdf is None else replacement_ptdf
        self.bns = (repl[None, :] - H[:, node_bus].T)

    def check(self, cases, curtail_mw: np.ndarray, thermal_scale: float, screened: set):
        F = cases.branch_flows_mw + curtail_mw @ self.bns          # snapshot x branch
        lim = up.branch_limits(self.grid, thermal_scale)
        w = cases.case_weights
        rows = []
        load0 = np.abs(F) / lim[None, :]
        for m in np.flatnonzero((load0 > 1 + TOL).any(axis=0)):
            s = load0[:, m] > 1 + TOL
            rows.append(dict(monitor=int(m), outage=-1, max_loading=float(load0[:, m].max()),
                             weight=float(w[s].sum()), screened=(int(m), -1) in screened))
        for k in np.flatnonzero(self.valid):
            coeff = self.lodf[:, k]
            fin = np.isfinite(coeff)
            post = F + F[:, [k]] * np.where(fin, coeff, 0.0)[None, :]
            ld = np.abs(post) / lim[None, :]
            ld[:, ~fin] = 0
            ld[:, k] = 0
            for m in np.flatnonzero((ld > 1 + TOL).any(axis=0)):
                s = ld[:, m] > 1 + TOL
                rows.append(dict(monitor=int(m), outage=int(k), max_loading=float(ld[:, m].max()),
                                 weight=float(w[s].sum()), screened=(int(m), int(k)) in screened))
        return pd.DataFrame(rows, columns=["monitor", "outage", "max_loading", "weight", "screened"])


def baseline_group_for_anchor(P: Problem) -> np.ndarray:
    """For the sigma-distribution plot: per anchor, the WDT group that curtailed the most
    MW while one of that anchor's states was the worst state (training ensemble)."""
    K = P.sigma.shape[1]
    rel = np.zeros((K, P.M_wdt.shape[1]))
    for mem, r in zip(P.ens.members, P.ens(P.M_wdt, keep_results=True)["results"]):
        for k, e in enumerate(P.anchor_meta["branches"]):
            js = np.flatnonzero(mem.cases.state_monitor == e)
            rel[k] += r.relief_mw[js].sum(axis=0)
    return np.column_stack([P.M_wdt[:, int(np.argmax(rel[k]))] for k in range(K)])


def _v1_figures(P, val, groupings, best_name, best_cid, names, scope):
    g = P.grid
    args = argparse.Namespace(scope=scope)
    def pivot(name):
        return val[val.grouping == name].set_index(["seed", "thermal_scale"]).D

    vp = pd.DataFrame({"D_wdt": pivot("wdt_multi"), "D_excl": pivot("wdt_exclusive"),
                       "D_opt": pivot(best_name)}).reset_index()
    plots.validation_plot(vp, FIGURES / f"validation_dd_{args.scope}.png", label_opt=f"optimised ({best_cid})")
    if "c00_dd_only:bestE" in groupings:
        vp2 = pd.DataFrame({"D_wdt": pivot("wdt_multi"), "D_excl": pivot("wdt_exclusive"),
                            "D_opt": pivot("c00_dd_only:bestE")}).reset_index()
        plots.validation_plot(vp2, FIGURES / f"validation_dd_{args.scope}_dd_only.png",
                              label_opt="optimised (c00_dd_only)")

    # ---------------- maps and sigma distributions
    buses = g.network.buses.set_index("bus")
    br = g.model.branches.reset_index(drop=True)
    axy = [((buses.loc[br.loc[m, "bus0"], "lon"], buses.loc[br.loc[m, "bus0"], "lat"]),
            (buses.loc[br.loc[m, "bus1"], "lon"], buses.loc[br.loc[m, "bus1"], "lat"]))
           for m in P.anchor_meta["branches"]]
    rings = up.land_outlines()
    plots.group_map(g.template, P.M0, names, rings, FIGURES / f"map_init_{args.scope}.png",
                    "initial shift-factor groups", axy)
    plots.group_map(g.template, groupings[best_name], names, rings, FIGURES / f"map_optimised_{args.scope}.png",
                    f"optimised groups ({best_cid})", axy)
    bmap = baseline_group_for_anchor(P)
    plots.sigma_distributions(P.sigma, P.mec, {"baseline WDT group": bmap, "initial": P.M0,
                                               f"optimised {best_cid}": groupings[best_name]},
                              names, FIGURES / f"sigma_distributions_{args.scope}.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="26")
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    args = ap.parse_args(argv)
    P = Problem(args.scope, args.version)
    g = P.grid
    v2 = args.version == "v2"
    out = (P.root / "validation") if v2 else RESULTS / f"validation_{args.scope}"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = FIGURES / "v2" if v2 else FIGURES
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = [a["name"] for a in P.anchor_meta["anchors"]]

    # ---------------- sweep summary (training ensemble) + Pareto
    rows = []
    for cid, lam in sweep_configs(args.version):
        f = P.root / cid / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        r = dict(config_id=cid, **{f"lam_{t}": v for t, v in s["lambdas"].items()},
                 E=s["best_E"]["E"], D=s["best_E"]["D"], security_pct=s["best_E"]["security_pct"],
                 sizes="|".join(map(str, s["best_E"]["sizes"])), covered=s["best_E"]["covered"],
                 overlap=s["best_E"]["overlap"], bestD_D=s["best_D"]["D"], calls=s["calls"],
                 C=s["best_E"].get("C", 0.0), min_size=min(s["best_E"]["sizes"]),
                 runtime_s=s["runtime_s"])
        r.update({f"raw_{t}": s["best_E"]["raw"].get(t, 0.0) for t in TERMS})
        r.update({f"bestD_raw_{t}": s["best_D"]["raw"].get(t, 0.0) for t in TERMS})
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "sweep_summary.csv", index=False)
    init_desc = P.describe(P.M0)
    init_row = dict(D=init_desc["D"], **{f"raw_{t}": v for t, v in init_desc["raw_terms"].items()})
    if not v2:
        plots.pareto_plot(summ, FIGURES / f"pareto_{args.scope}.png", baseline=init_row)
        plots.anchors_plot(RESULTS / f"setup_{args.scope}", FIGURES / f"anchors_pe_{args.scope}.png")

    # ---------------- groupings to validate
    groupings = {"wdt_multi": P.M_wdt, "wdt_exclusive": P.M_excl, "init_shift_factor": P.M0}
    for cid in summ.config_id:
        z = np.load(P.root / cid / "final.npz")
        groupings[f"{cid}:bestE"] = z["best_E_M"].astype(bool)
        groupings[f"{cid}:bestD"] = z["best_D_M"].astype(bool)

    if v2:
        val_ens = build_ensemble(g, A.VALIDATION_SEEDS, P.thermal_scales, builder=P.builder)
    else:
        val_ens = build_ensemble(g, A.VALIDATION_SEEDS, A.THERMAL_SCALES)
    val_ens.set_baseline(P.M_wdt)
    c0 = val_ens.members[0].cases
    fn = FullNetwork(g, c0.balance_bus, getattr(c0, "node_bus", None), getattr(c0, "replacement_ptdf", None))
    base_full = {}
    vrows, frows, nfrows = [], [], []
    for name, M in groupings.items():
        ev = val_ens(M, keep_results=True)
        for mem, r, d, nf, sec in zip(val_ens.members, ev["results"], ev["D_members"],
                                      ev["new_failures_members"], ev["security_members"]):
            c = mem.cases
            screened = set(zip(c.state_monitor.tolist(), c.state_outage.tolist()))
            full = fn.check(c, r.curtail_mw, mem.thermal_scale, screened)
            key = (mem.seed, mem.thermal_scale)
            post = c.state_base_flows_mw + r.curtail_mw @ c.node_state_sensitivity
            for snap in np.flatnonzero(mem.baseline_secure & ~r.secure):
                j = int(np.argmax(np.abs(post[snap]) / c.state_limits_mw))
                nfrows.append(dict(grouping=name, seed=mem.seed, thermal_scale=mem.thermal_scale, snapshot=int(snap),
                                   case_weight=float(c.case_weights[snap]), worst_monitor=int(c.state_monitor[j]),
                                   worst_outage=int(c.state_outage[j]), worst_loading=float(r.worst_loading[snap])))
            if name == "wdt_multi":
                base_full[key] = set(zip(full.monitor, full.outage))
            unscreened = full[~full.screened]
            new_vs_base = unscreened[[(m, o) not in base_full.get(key, set())
                                      for m, o in zip(unscreened.monitor, unscreened.outage)]]
            vrows.append(dict(grouping=name, seed=mem.seed, thermal_scale=mem.thermal_scale, D=d, C=r.conv_pct,
                              security_pct=sec, new_failures_vs_wdt=int(nf),
                              unscreened_overloaded_pairs=int(len(unscreened)),
                              unscreened_overload_weight=float(unscreened.weight.max()) if len(unscreened) else 0.0,
                              unscreened_max_loading=float(unscreened.max_loading.max()) if len(unscreened) else 0.0,
                              unscreened_new_vs_wdt=int(len(new_vs_base))))
            for _, fr in unscreened.iterrows():
                frows.append(dict(grouping=name, seed=mem.seed, thermal_scale=mem.thermal_scale, **fr.to_dict()))
    val = pd.DataFrame(vrows)
    val.to_csv(out / "validation.csv", index=False)
    full_df = pd.DataFrame(frows)
    if len(full_df):
        br = g.model.branches.reset_index(drop=True)
        st = g.network.buses.set_index("bus")["station"].astype(str)
        full_df["monitor_name"] = [f"{st[br.loc[m, 'bus0']]} - {st[br.loc[m, 'bus1']]}" for m in full_df.monitor]
        full_df["outage_name"] = ["intact" if o < 0 else f"{st[br.loc[o, 'bus0']]} - {st[br.loc[o, 'bus1']]}"
                                  for o in full_df.outage]
    full_df.to_csv(out / "full_network_unscreened_overloads.csv", index=False)
    pd.DataFrame(nfrows).to_csv(out / "new_security_failures.csv", index=False)

    agg = val.groupby("grouping").agg(D_mean=("D", "mean"), D_min=("D", "min"), D_max=("D", "max"),
                                      C_mean=("C", "mean"),
                                      security_mean=("security_pct", "mean"),
                                      new_failures=("new_failures_vs_wdt", "sum"),
                                      unscreened_pairs=("unscreened_overloaded_pairs", "sum"),
                                      unscreened_new_vs_wdt=("unscreened_new_vs_wdt", "sum"),
                                      unscreened_max_loading=("unscreened_max_loading", "max"))
    wdt_D = agg.loc["wdt_multi", "D_mean"]
    agg["dD_vs_wdt_pp"] = agg.D_mean - wdt_D
    agg["dD_vs_wdt_pct"] = 100 * agg.dD_vs_wdt_pp / wdt_D
    agg = agg.sort_values("D_mean")
    agg.to_csv(out / "validation_summary.csv")

    # ---------------- pick the configurations to headline
    bestE = agg[agg.index.str.endswith(":bestE")].copy()
    # v2: renewable dispatch-down alone would reward pushing all relief onto conventional
    # plant, so the headline minimises total redispatch D + C (both % of renewable potential).
    bestE["score"] = bestE.D_mean + (bestE.C_mean if v2 else 0.0)
    what = "D + C (renewable dispatch-down + conventional redispatch)" if v2 else "D"
    cand = bestE[bestE.new_failures == 0]
    if len(cand):
        best_name = cand.score.idxmin()
        rule = f"lowest mean validation {what} among best-E groupings with 0 new failures vs WDT"
    else:
        best_name = bestE.score.idxmin()
        rule = (f"NO best-E grouping had 0 new failures vs WDT on validation; picked lowest mean validation {what}. "
                "See new_security_failures.csv")
    best_cid = best_name.split(":")[0]
    pick = dict(best_config=best_cid, best_grouping=best_name, rule=rule,
                new_failures=int(agg.loc[best_name, "new_failures"]))
    (out / "headline.json").write_text(json.dumps(pick, indent=2))

    if not v2:
        _v1_figures(P, val, groupings, best_name, best_cid, names, args.scope)
    print(agg.round(4).to_string())
    print(json.dumps(pick))
    return agg


if __name__ == "__main__":
    main()

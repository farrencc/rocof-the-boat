"""nDBZ reports.

    python -m bzgen.ndbz.report scenarios     # -> reports/ndbz_scenarios.md (+ results/ndbz/scenarios/)

``scenarios`` computes every scenario x scope x clip handling edge table from the
solved prices (never re-solves), persists them under ``results/ndbz/scenarios/`` so
later steps (and a fresh clone without ``data/solved/``) can use them, and writes the
scenario report with the price-clip diagnostic first.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from bzgen import config
from bzgen.ndbz import figures_dir, load_config, results_dir, write_csv, write_parquet, write_text
from bzgen.ndbz import plots, scenarios as S

REPORTS = config.ROOT / "reports"
SCOPES = ("country", "europe")


def _rel(p) -> str:
    return "../" + str(p.relative_to(config.ROOT))


# --------------------------------------------------------------------------- #
# compute
# --------------------------------------------------------------------------- #

def compute_scenarios(cfg: dict, log=print) -> dict:
    t0 = time.time()
    inp = S.load_inputs(cfg)
    ref = S.reference_stats(inp, cfg)
    log(f"inputs + reference: {time.time() - t0:.0f}s")
    names = cfg["scenario"]["names"]
    scns, tables, clip, cmp_excl, cmp_unc, boot = [], {}, [], [], [], []
    for scope in SCOPES:
        for n in names:
            if scope == "europe" and n == "baseline":
                continue                                    # identical to the country baseline
            scn = S.select(inp, n, cfg, scope)
            scns.append(scn)
            for ch in S.CLIP_HANDLING:
                if scope == "europe" and ch != "as_is":
                    continue
                st = S.scenario_stats(inp, scn, cfg, ch)
                tables[(scope, n, ch)] = S.normalise(st, ref.loc[st.index], cfg, "baseline")
            key = dict(scope=scope, scenario=n)
            clip.append(S.clip_diagnostic(inp, scn, cfg).assign(**key))
            if scope == "country":
                a = tables[(scope, n, "as_is")]
                cmp_excl.append(S.clip_comparison(a, tables[(scope, n, "exclude_clipped")], cfg).assign(**key))
                cmp_unc.append(S.clip_comparison(a, tables[(scope, n, "unclipped")], cfg).assign(**key))
            boot.append(S.bootstrap(inp, scn, cfg).assign(**key))
            log(f"{scope}/{n}: {time.time() - t0:.0f}s")
    static = config.ROOT / "results" / "edges.parquet"
    repro = None
    if static.exists():
        old = pd.read_parquet(static)
        b = tables[("country", "baseline", "as_is")]
        repro = {"max_abs_dp_n": float(np.abs(b.dp_n - old.loc[b.index].dp_n).max()),
                 "max_abs_J_n": float(np.abs(b.J_n - old.loc[b.index].J_n).max()),
                 "n_edges": int(len(b)), "n_static": int(len(old))}
    cat = lambda xs: pd.concat([x.reset_index() for x in xs], ignore_index=True)
    return {"inputs": inp, "ref": ref, "scenarios": scns, "tables": tables,
            "params": S.params_table(scns), "snapshots": S.snapshot_table(scns, inp),
            "clip": cat(clip), "cmp_excl": cat(cmp_excl), "cmp_unc": cat(cmp_unc),
            "boot": cat(boot), "repro": repro}


def long_edges(tables: dict, cfg: dict) -> pd.DataFrame:
    stat = S.STAT_COL[cfg["edges"]["statistic"]]
    parts = []
    for (scope, n, ch), t in tables.items():
        parts.append(pd.DataFrame({
            "scope": scope, "scenario": n, "clip_handling": ch, "edge_id": t.index,
            "country": t.country.to_numpy(), "dp_mean": t.dp_mean.to_numpy(),
            "dp_duration": t.dp_duration.to_numpy(), "dp_quantile": t.dp_quantile.to_numpy(),
            "stat": t[stat].to_numpy(), "dp_n_baseline": t.dp_n_baseline.to_numpy(),
            "dp_n_self": t.dp_n_self.to_numpy(), "dp_n_baseline_ownclip": t.dp_n_baseline_ownclip.to_numpy(),
            "above_ref_clip": t.above_ref_clip.to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def persist(out: dict, cfg: dict) -> None:
    d = results_dir(cfg) / "scenarios"
    write_parquet(d / "edges.parquet", long_edges(out["tables"], cfg))
    write_parquet(d / "snapshots.parquet", out["snapshots"])
    write_csv(d / "params.csv", out["params"], index=False)
    write_csv(d / "clip.csv", out["clip"], index=False)
    write_csv(d / "clip_compare_exclude.csv", out["cmp_excl"], index=False)
    write_csv(d / "clip_compare_unclipped.csv", out["cmp_unc"], index=False)
    write_csv(d / "bootstrap.csv", out["boot"], index=False)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def _pivot(df, value, scope="country", names=None, order=None):
    p = df[df.scope == scope].pivot(index="scenario", columns="country", values=value)
    if names:
        p = p.reindex([n for n in names if n in p.index])
    if order is not None:
        p = p.reindex(columns=[c for c in order if c in p.columns])
    return p


def _md(df: pd.DataFrame, digits=3) -> str:
    return df.round(digits).to_markdown()


def severity(tables: dict, scope: str, names, key="dp_n_baseline") -> pd.DataFrame:
    rows = {}
    for n in names:
        t = tables.get((scope, n, "as_is"))
        if t is None and scope == "europe" and n == "baseline":
            t = tables[("country", n, "as_is")]
        if t is not None:
            rows[n] = t.groupby("country")[key].mean()
    return pd.DataFrame(rows).T


def write_scenarios(out: dict, cfg: dict) -> None:
    sc, e = cfg["scenario"], cfg["edges"]
    names = sc["names"]
    fig = figures_dir(cfg)
    fig.mkdir(parents=True, exist_ok=True)
    inp = out["inputs"]
    order = list(inp.edges.groupby("country").size().sort_values(ascending=False).index)
    tables = out["tables"]
    clip, boot, params = out["clip"], out["boot"], out["params"]
    thr = sc["clip_material"]

    # --- figures
    f_clip = fig / "scenarios_clip_bus_hours.png"
    plots.heatmap(_pivot(clip, "bus_hours_at_clip", names=names, order=order) * 100, f_clip,
                  "Share of bus-hours at the ±price clip (%), country scope", "% of bus-hours",
                  fmt="{:.1f}")
    f_both = fig / "scenarios_clip_both_ends.png"
    plots.heatmap(_pivot(clip, "edge_hours_both_ends_clipped_same_side", names=names, order=order) * 100,
                  f_both, "Edge-hours with both ends clipped on the same side (%): |Δp| forced to 0",
                  "% of edge-hours", fmt="{:.1f}")
    mat = pd.concat([out["cmp_excl"].assign(v="excl"), out["cmp_unc"].assign(v="unc")])
    mat_any = mat.groupby(["scenario", "country"]).material.any().unstack().reindex(
        index=[n for n in names], columns=order)
    rel_unc = _pivot(out["cmp_unc"], "mean_rel_change", names=names, order=order) * 100
    f_cmp = fig / "scenarios_clip_unclipped_change.png"
    plots.heatmap(rel_unc, f_cmp, "Country-mean statistic, unclipped vs as_is: relative change (%) "
                  "— outlined = material (either comparison)", "% change", fmt="{:+.0f}",
                  cmap=plots.DIV, norm=plots.TwoSlopeNorm(0.0, min(-1, np.nanmin(rel_unc.to_numpy())),
                                                          max(1, np.nanmax(rel_unc.to_numpy()))),
                  flag=mat_any.reindex(index=rel_unc.index, columns=rel_unc.columns).fillna(False))
    sev_c = severity(tables, "country", names).reindex(columns=order)
    sev_e = severity(tables, "europe", names).reindex(columns=order)
    f_sev = fig / "scenarios_severity_country.png"
    plots.severity_heatmap(sev_c, f_sev, "Mean dp̃ per country under baseline normalisation "
                           "(country-scope scenarios)")
    f_sev_e = fig / "scenarios_severity_europe.png"
    plots.severity_heatmap(sev_e, f_sev_e, "Mean dp̃ per country under baseline normalisation "
                           "(europe-scope scenarios)")
    thin = _pivot(boot, "ci_width_over_edge_std", names=names, order=order)
    f_thin = fig / "scenarios_bootstrap.png"
    plots.heatmap(thin, f_thin, "Bootstrap: CI width of the country mean / between-edge std "
                  f"(outlined = thin, > {sc['bootstrap']['thin_ratio']})", "ratio", fmt="{:.2f}",
                  flag=_pivot(boot, "thin", names=names, order=order).astype(bool))
    show = [c for c in ["FR", "DE", "ES", "IT", "IE", "NO"] if c in order]
    f_ecdf = fig / "scenarios_dpn_ecdf.png"
    plots.dpn_ecdf({n: tables[("country", n, "as_is")] for n in names}, show, f_ecdf,
                   "dp̃ per edge: baseline vs self normalisation (country scope, as_is)")

    # --- tables
    cp = params[params.scope == "country"]
    n_snap = cp.pivot(index="country", columns="scenario", values="n_snapshots").reindex(
        index=order, columns=names)
    status = cp[cp.status != "ok"][["scenario", "country", "status"]]
    ep = params[(params.scope == "europe")].drop_duplicates("scenario").set_index("scenario")
    ep = ep.reindex([n for n in names if n in ep.index])
    q_eff = cp[cp.scenario.isin(["peak_demand", "wind_surplus", "max_dispersion"])].q_eff.dropna()
    df = cp[cp.scenario == "dunkelflaute"]
    # overlap of country-scope and europe-scope hour sets
    sn = out["snapshots"]
    jac = {}
    for n in names:
        if n == "baseline":
            continue
        eu = set(sn[(sn.scope == "europe") & (sn.scenario == n)].snapshot.drop_duplicates())
        row = {}
        for c in order:
            cs = set(sn[(sn.scope == "country") & (sn.scenario == n) & (sn.country == c)].snapshot)
            row[c] = len(cs & eu) / len(cs | eu) if cs else np.nan
        jac[n] = row
    jac = pd.DataFrame(jac).T
    # verdicts
    both = _pivot(clip, "edge_hours_both_ends_clipped_same_side", names=names)
    mat_list = mat[mat.material][["scenario", "country", "v", "mean_rel_change", "spearman"]]
    thin_list = boot[boot.thin & (boot.scope == "country")][["scenario", "country",
                                                              "n_snapshots", "ci_width_over_edge_std"]]
    stat = S.STAT_COL[e["statistic"]]
    no_sig = {}
    for n in names:
        t = tables[("country", n, "as_is")]
        m = t.groupby("country")[stat].mean()
        no_sig[n] = sorted(m.index[m < e.get("signal_min", 0.01)])
    above = {n: tables[("country", n, "as_is")].groupby("country").above_ref_clip.mean()
             for n in names}
    above = pd.DataFrame(above).T.reindex(columns=order)
    f_pin = fig / "scenarios_pinned_at_reference_cap.png"
    plots.heatmap(above * 100, f_pin, "Edges pinned at the reference-year Δp cap under baseline "
                  "normalisation (%)", "% of the country's edges", fmt="{:.0f}")
    sev_own = severity(tables, "country", names, "dp_n_baseline_ownclip").reindex(columns=order)
    f_sev_own = fig / "scenarios_severity_country_ownclip.png"
    plots.severity_heatmap(sev_own, f_sev_own, "Mean dp̃, diagnostic variant: clip at the scenario's "
                           "own 99th percentile, divide by the reference mean")

    L = []
    a = L.append
    a("# nDBZ scenarios: outlier hours, the price clip, normalisation")
    a("")
    a("Generated by `python -m bzgen.ndbz.report scenarios` (`bzgen/ndbz/scenarios.py`). "
      "Configuration: `config/ndbz.yaml` over `config/default.yaml`. No OPF re-solve: every "
      "scenario is a subset of the 8760 solved hours in `data/solved/prices.parquet`.")
    a("")
    a(f"- statistic: **{e['statistic']}** (threshold {e['duration_threshold']} EUR/MWh), remedies: "
      f"Δp `{e['dp_remedy']}` (q = {e['dp_clip_q']}), J `{e['J_remedy']}`; per country, cross-border "
      "edges dropped (as in the static pipeline)")
    a(f"- defaults: `scope: {sc['scope']}`, `q: {sc['q']}`, `min_snapshots: {sc['min_snapshots']}`, "
      f"`clip_handling: {sc['clip_handling']}`, `normalisation: {sc['normalisation']}`")
    a(f"- shedding policy `{cfg['solve']['shedding_policy']}`, price clip ±{cfg['solve']['price_clip']:.0f} EUR/MWh")
    if out["repro"]:
        r = out["repro"]
        a(f"- **reproduction check**: the `baseline` scenario (all hours, `as_is`, baseline "
          f"normalisation) against the static `results/edges.parquet`: max |Δ dp̃| = "
          f"{r['max_abs_dp_n']:.1e}, max |Δ J̃| = {r['max_abs_J_n']:.1e} over {r['n_edges']} edges. "
          "The re-solved prices therefore reproduce the static inputs, and the anchor map A "
          "was built from these exact numbers.")
    a("")
    a("## 1. The price clip (read first)")
    a("")
    a(f"Prices are clipped at ±{cfg['solve']['price_clip']:.0f} EUR/MWh before any statistic "
      "(`solve.shedding_policy: clip`). With the duration statistic, the clip destroys "
      "signal only where **both** ends of an edge are beyond the clip on the same side in "
      "the same hour: |Δp| is then forced to 0, although the unclipped prices may differ by "
      "hundreds of EUR/MWh. An edge with one end clipped still shows |Δp| > 1 EUR/MWh. "
      "The second table below counts exactly those forced-to-zero edge-hours.")
    a("")
    a("Three treatments are compared: `as_is` (the static pipeline's clipped prices), "
      "`exclude_clipped` (drop every edge-hour with an endpoint at the clip, renormalise that "
      "edge's weight), and `unclipped`. The solved duals in `prices.parquet` are stored "
      "*before* the clip: `prepare` applies it later. So `unclipped` is the statistic an "
      "unclipped solve would give, and **no re-solve is needed to check it**.")
    a("")
    a(f"![bus-hours at the clip]({_rel(f_clip)})")
    a("")
    a(f"![both ends clipped]({_rel(f_both)})")
    a("")
    a(f"![unclipped vs as_is]({_rel(f_cmp)})")
    a("")
    a(f"Material disagreement means Spearman ρ of the per-edge statistic < {thr['spearman_min']}, "
      f"or a relative change of the country mean > {thr['mean_rel_change']:.0%}, for either "
      "comparison with `as_is`.")
    a("")
    if len(mat_list):
        a(f"**{len(mat_list[['scenario', 'country']].drop_duplicates())} (scenario, country) pairs disagree "
          "materially.** Conclusions for them are provisional under `as_is`. Use the "
          "`unclipped` statistic, or report the two side by side:")
        a("")
        a(_md(mat_list.replace({"v": {"excl": "exclude_clipped", "unc": "unclipped"}})
              .rename(columns={"v": "vs as_is"}).set_index(["scenario", "country"])))
    else:
        a("**No (scenario, country) pair disagrees materially**: the clip has not removed the "
          "congestion signal from any scenario.")
    a("")
    a("<details><summary>Per-country clip tables (country scope)</summary>")
    a("")
    for col, lab in (("bus_hours_at_clip", "bus-hours at the clip"),
                     ("edge_hours_one_end_clipped", "edge-hours with ≥ 1 end clipped"),
                     ("edge_hours_both_ends_clipped_same_side", "edge-hours with both ends clipped, same side"),
                     ("edges_any_clipped_hour", "edges with any clipped hour"),
                     ("edges_clipped_ge10pct_hours", "edges with ≥ 10 % of hours clipped")):
        a(f"**{lab}**")
        a("")
        a(_md(_pivot(clip, col, names=names, order=order).T))
        a("")
    a("**Country-mean statistic, relative change vs as_is**")
    a("")
    a(_md(pd.concat({"exclude_clipped": _pivot(out["cmp_excl"], "mean_rel_change", names=names, order=order),
                     "unclipped": _pivot(out["cmp_unc"], "mean_rel_change", names=names, order=order)}).T))
    a("")
    a("**Spearman ρ of the per-edge statistic vs as_is**")
    a("")
    a(_md(pd.concat({"exclude_clipped": _pivot(out["cmp_excl"], "spearman", names=names, order=order),
                     "unclipped": _pivot(out["cmp_unc"], "spearman", names=names, order=order)}).T))
    a("")
    a("</details>")
    a("")
    a("## 2. Scenario definitions and snapshot counts")
    a("")
    a("| scenario | definition |")
    a("|---|---|")
    a("| `baseline` | all snapshots: the set the static map A was built on |")
    a(f"| `peak_demand` | top q of hours by total load (national load of the cluster-country) |")
    a(f"| `dunkelflaute` | bottom {sc['dunkelflaute']['vre_q']:.0%} of VRE (on/offshore wind + solar, "
      f"capacity-weighted) CF **and** top {sc['dunkelflaute']['load_q']:.0%} of load |")
    a(f"| `wind_surplus` | top q of hours by wind (on + offshore) capacity factor |")
    a(f"| `max_dispersion` | top q of hours by the cross-sectional std of the country's nodal prices (after the clip) |")
    a("")
    a(f"Load and CF come from the solve inputs (`data/interim/national_load.parquet`, "
      "`res_profiles.parquet`, `generators.csv`), not recomputed. \"Top q\" is by weight. "
      f"`q = {sc['q']}` is {sc['q'] * inp.weights.sum():.0f} h, below `min_snapshots = "
      f"{sc['min_snapshots']}`, so q is widened to **q_eff = {q_eff.max():.4f}** "
      f"({sc['min_snapshots']} h) in every country. The dunkelflaute quantiles widen together "
      f"in steps of {sc['dunkelflaute']['widen_step']} until the intersection reaches "
      f"{sc['min_snapshots']} h. Effective VRE quantile per country: "
      f"{df.vre_q_eff.min():.2f}–{df.vre_q_eff.max():.2f}; load quantile "
      f"{df.load_q_eff.min():.2f}–{df.load_q_eff.max():.2f}.")
    a("")
    a("`scope: country` (default) selects each country's own outlier hours. `scope: europe` "
      "selects one common set from system-wide load / CF / price dispersion. Europe-scope "
      "snapshot counts:")
    a("")
    a(_md(ep[["n_snapshots", "weight_share"] + [c for c in ("q_eff", "vre_q_eff", "load_q_eff") if c in ep]]))
    a("")
    if len(status):
        a("Scenarios undefined for a country (skipped there):")
        a("")
        a(_md(status.set_index(["scenario", "country"])))
        a("")
    a("<details><summary>Snapshot counts per country (country scope)</summary>")
    a("")
    a(_md(n_snap, 0))
    a("")
    a("</details>")
    a("")
    a("Overlap (Jaccard) of each country's own outlier hours with the europe-wide set. "
      "Low values mean the two scopes ask different questions:")
    a("")
    a(_md(jac.T, 2))
    a("")
    a("### Bootstrap stability")
    a("")
    a(f"Snapshot bootstrap ({sc['bootstrap']['n']} replicates, `as_is`) of the country mean of "
      f"`{stat}`. A scenario is **thin** for a country when the {sc['bootstrap']['ci']:.0%} "
      f"CI width exceeds {sc['bootstrap']['thin_ratio']} × the between-edge std of the "
      "statistic. The hours then cannot pin down the country's congestion level to better "
      "than the spread that separates its edges. Also recorded: the median per-edge "
      "bootstrap SE over the between-edge std (`results/ndbz/scenarios/bootstrap.csv`).")
    a("")
    a(f"![bootstrap]({_rel(f_thin)})")
    a("")
    if len(thin_list):
        a(f"**Thin (scenario, country) pairs: {len(thin_list)}.** Scenario conclusions for them "
          "are not supported by the data:")
        a("")
        a(_md(thin_list.set_index(["scenario", "country"]), 2))
    else:
        a("**No scenario is thin in any country** at this threshold.")
    a("")
    a("Median per-edge bootstrap SE / between-edge std:")
    a("")
    a(_md(_pivot(boot, "median_edge_se_over_edge_std", names=names, order=order).T, 2))
    a("")
    a("### Comparability of the statistic across scenarios")
    a("")
    a("`congestion_stats` divides by the subset's total weight, so `dp_duration` is a share "
      "of hours in [0, 1], whatever the scenario length. This is asserted on every scenario "
      "table (`scenarios._check_comparable`). `tests/test_ndbz.py` checks the "
      "weight-normalisation of all three statistics: invariance to scaling all weights, to "
      "duplicating every snapshot, and the full year as the weight-average of a scenario and "
      "its complement for `duration` and `mean`. `mean` shares the property. `quantile` is a "
      "quantile of the per-hour distribution and does not grow with length either, but it "
      "does not decompose over subsets.")
    a("")
    a("## 3. Normalisation: baseline vs self")
    a("")
    a("`self` renormalises each scenario to mean dp̃ = 1 per country, so only the spatial "
      "pattern survives. `baseline` (default) uses the full-year reference for the clip "
      f"threshold (q = {e['dp_clip_q']}) and the country mean. A scenario that is more "
      "congested than the year then has mean dp̃ > 1, which strengthens the Potts term "
      "against a fixed rigidity term. That is intended, and it is tested.")
    a("")
    a(f"![severity, country scope]({_rel(f_sev)})")
    a("")
    a(f"![severity, europe scope]({_rel(f_sev_e)})")
    a("")
    a(f"![ECDF]({_rel(f_ecdf)})")
    a("")
    a("Under `baseline`, a scenario's per-edge value is capped at the **reference** clip "
      "threshold, so congestion beyond the year's 99th-percentile edge is compressed. "
      "Share of edges pinned at that cap (all tied at the same dp̃):")
    a("")
    a(f"![pinned]({_rel(f_pin)})")
    a("")
    big = above.T.stack()
    big = big[big >= 0.25].sort_values(ascending=False)
    if len(big):
        a(f"**{len(big)} (country, scenario) pairs pin ≥ 25 % of their edges at the reference cap**, "
          "the worst being " + ", ".join(f"{c}/{n} {v:.0%}" for (c, n), v in big.head(8).items()) +
          ". Tied edges cannot be told apart by the Potts term, so the "
          "re-zoning signal there is flattened, not only rescaled. Diagnostic variant "
          "`dp_n_baseline_ownclip` (persisted, not used by default): clip at the scenario's "
          "own 99th percentile and divide by the reference mean. Severity is kept, and only "
          "1 % of edges are capped:")
        a("")
        a(f"![severity ownclip]({_rel(f_sev_own)})")
        a("")
    a("<details><summary>Pinned share per country and scenario</summary>")
    a("")
    a(_md(above.T, 3))
    a("")
    a("</details>")
    a("")
    a("<details><summary>Mean dp̃ per country (baseline normalisation)</summary>")
    a("")
    a(_md(pd.concat({"country scope": sev_c, "europe scope": sev_e}).T, 2))
    a("")
    a("</details>")
    a("")
    a("Countries with no congestion signal (country mean of the raw statistic below "
      f"`edges.signal_min` = {e.get('signal_min', 0.01)}) per scenario. Their dp̃ is noise "
      "amplified by a small divisor under `self`. Under `baseline` it is ≈ 0, as it should be:")
    a("")
    for n in names:
        a(f"- `{n}`: {', '.join(no_sig[n]) or '—'}")
    a("")
    a("J̃ is topology only. It is identical across all scenarios (asserted in "
      "`scenarios.normalise` and tested).")
    a("")
    narr = REPORTS / "ndbz_scenarios_narrative.md"
    if narr.exists():
        a("## 4. Reading")
        a("")
        a(narr.read_text().strip())
        a("")
    write_text(REPORTS / "ndbz_scenarios.md", "\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["scenarios"])
    args = ap.parse_args()
    cfg = load_config()
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    if args.what == "scenarios":
        out = compute_scenarios(cfg, log)
        persist(out, cfg)
        write_scenarios(out, cfg)
        log("wrote reports/ndbz_scenarios.md")


if __name__ == "__main__":
    main()

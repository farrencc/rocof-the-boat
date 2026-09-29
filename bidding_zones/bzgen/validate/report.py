"""Aggregation, validity checks, figures and reports/validate.md for the validation stage.

Called by ``bzgen.validate.run`` after the runs; ``--report-only`` re-aggregates cached runs.
"""

from __future__ import annotations

import json

import matplotlib
import matplotlib.colors
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bzgen import config
from bzgen.validate import inference as I
from bzgen.validate import metrics as M

ROOT = config.ROOT
FIGS = ROOT / "figures" / "validate"
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results"

from bzgen.plot.maps import (ALPHA_LABEL, BLUE, BLUE_LINE, DERATING_LABEL, GREEN, GREY, INK, INK2,  # noqa: E402
                             SAND, SAND_LINE, TEAL)

# zone-map palette; buckets in fixed order
BUCKET_COLORS = {"internal": BLUE, "border_intra": SAND, "border_cross": GREEN,
                 "pocket": TEAL, "unattributed": GREY}
BUCKET_LABELS = {"internal": "Inside a zone", "border_intra": "Between DE zones",
                 "border_cross": "DE border with neighbours", "pocket": "Load pockets",
                 "unattributed": "No binding line"}
DD_LABEL = "Wind + solar dispatch down (TWh/yr)"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def md(df, digits=2):
    return df.round(digits).to_markdown()


def load_hourly(out, mid, der):
    f = out / "runs" / f"{mid}_d{der:g}" / "hourly.parquet"
    return pd.read_parquet(f) if f.exists() else None


def summarise_market(h, w, cn):
    w = w.reindex(h.index)
    sc = 8760.0 / w.sum()
    return {"market_spill_res_GWh_allhours": float((h.spill_res * w).sum() * sc / 1e3),
            "C_market_EUR_allhours": float((h.C_market * w).sum() * sc),
            "C_nodal_EUR_allhours": float((cn.reindex(h.index) * w).sum() * sc)}


def infeasible_counts(out, mid, der, hours, cfg):
    f = out / "runs" / f"{mid}_d{der:g}" / "infeasible.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f, parse_dates=["snapshot"])
    df = df[df.snapshot.isin(hours)]
    s = df[df.kind == "shed"]
    tol = cfg["solve"]["shed_tol_mw"]
    beyond = s[(s.shed_mwh - s.pocket_mwh) > tol]
    return {"shed_hours_any": int(s.snapshot.nunique()),
            "shed_hours_excl_pockets": int(beyond.snapshot.nunique()),
            "shed_zone_hours_excl_pockets": int(len(beyond)),
            "lp_infeasible_hours_log": int((df.kind == "lp_infeasible").sum())}


def aggregate(cfg, sn, out, maps, stage):
    from bzgen.validate.run import scoring_views
    v = cfg["validate"]
    w = sn.weight
    cn = pd.read_parquet(out / "nodal" / "cost.parquet").C_nodal
    views = scoring_views(cfg, maps, sn)
    tol = lambda c: v["cost_tol_rel"] * np.abs(c) + v["cost_tol_abs"]
    rows, deltas, checks = [], [], []
    hourly = {}
    for scoring, mh in views.items():
        for der in v["deratings"]:
            hk = load_hourly(out, "k1", der)
            for mid, hours in mh.items():
                h = load_hourly(out, mid, der)
                if h is None:
                    continue
                h = h.loc[h.index.intersection(hours)]
                hourly[(scoring, mid, der)] = h
                row = {"scoring": scoring, "map": mid, "role": maps[mid]["role"],
                       "fit": maps[mid]["fit"], "derating": der}
                row.update(summarise_market(h, w, cn))
                if stage == "AB":
                    row.update(M.summarise(h, w, cn))
                    row.update(infeasible_counts(out, mid, der, h.index, cfg))
                # validity checks, hour by hour
                c_n = cn.reindex(h.index)
                v1 = int((h.C_market - c_n > tol(c_n)).sum())
                checks.append({"check": "C_market <= C_nodal", "scoring": scoring, "map": mid,
                               "derating": der, "hours": len(h), "violations": v1,
                               "asserted": bool(np.isclose(der, 1.0)),
                               "max_excess_EUR": float((h.C_market - c_n).max())})
                if stage == "AB":
                    ok = h.stageB_ok.astype(bool)
                    tot = h.C_market[ok] + h.C_redispatch[ok]
                    v3 = int((tot - c_n[ok] < -tol(c_n[ok])).sum())
                    checks.append({"check": "C_market + C_redispatch >= C_nodal", "scoring": scoring,
                                   "map": mid, "derating": der, "hours": int(ok.sum()),
                                   "violations": v3, "asserted": True,
                                   "max_excess_EUR": float((c_n[ok] - tot).max()) if ok.any() else np.nan})
                if mid != "k1" and hk is not None:
                    ix = h.index.intersection(hk.index)
                    d2 = hk.C_market.loc[ix] - h.C_market.loc[ix]
                    checks.append({"check": "C_market(k=1) <= C_market(split)", "scoring": scoring,
                                   "map": mid, "derating": der, "hours": len(ix),
                                   "violations": int((d2 > tol(h.C_market.loc[ix])).sum()),
                                   "asserted": True, "max_excess_EUR": float(d2.max())})
                    if stage == "AB":
                        p = I.paired(h, hk, "DD_RES", w)
                        b = I.block_bootstrap(p, cfg)
                        pt = I.paired(h, hk, "DD_RES_TH", w)
                        bt = I.block_bootstrap(pt, cfg)
                        row.update({"dDD_RES_GWh": b["point"], "dDD_RES_lo": b["lo"], "dDD_RES_hi": b["hi"],
                                    "dDD_RES_TH_GWh": bt["point"], "dDD_RES_TH_lo": bt["lo"],
                                    "dDD_RES_TH_hi": bt["hi"], "paired_hours": b["n_hours"],
                                    "boot_blocks": b["n_blocks"]})
                        deltas.append({"scoring": scoring, "map": mid, "role": maps[mid]["role"],
                                       "derating": der, **b})
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(deltas), pd.DataFrame(checks), hourly


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def _primary(df_scorings):
    return "oos_even" if "oos_even" in set(df_scorings) else sorted(set(df_scorings))[0]


def fig_delta(deltas, cfg):
    s = _primary(deltas.scoring)
    g = deltas[deltas.scoring == s]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    band = g[g.role.isin(["headline", "sibling"])].groupby("derating").point.agg(["min", "max"]) / 1e3
    if len(g[g.role == "sibling"]):
        ax.fill_between(band.index, band["min"], band["max"], color=BLUE, alpha=0.35, lw=0,
                        label="Alternative maps (range)")
    hd = g[g.role == "headline"].sort_values("derating")
    ax.errorbar(hd.derating, hd.point / 1e3, yerr=[(hd.point - hd.lo) / 1e3, (hd.hi - hd.point) / 1e3],
                color=BLUE_LINE, lw=2, fmt="o", ms=7, capsize=4, label="Split DE (95% range)")
    ax.axhline(0, color=SAND_LINE, lw=2, label="DE as one zone")
    ax.set_xlabel(DERATING_LABEL, fontsize=9, color=INK2)
    ax.set_ylabel("Change in " + DD_LABEL[0].lower() + DD_LABEL[1:], fontsize=9, color=INK2)
    ax.set_title("Split DE vs one zone, held-out weeks", fontsize=10, color=INK, loc="left")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "delta_dd_vs_derating.png", dpi=150)
    plt.close(fig)


def fig_attribution(rows, cfg):
    s = _primary(rows.scoring)
    g = rows[(rows.scoring == s) & rows.role.isin(["k1", "headline"])]
    ders = sorted(g.derating.unique())
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for j, role in enumerate(["k1", "headline"]):
        gg = g[g.role == role].set_index("derating").reindex(ders)
        x = np.arange(len(ders)) + (j - 0.5) * 0.4
        bottom = np.zeros(len(ders))
        for b in M.BUCKETS:
            val = gg[f"attr_{b}_GWh"].fillna(0).to_numpy() / 1e3
            ax.bar(x, val, 0.36, bottom=bottom, color=BUCKET_COLORS[b], edgecolor="white", lw=1,
                   label=BUCKET_LABELS[b] if j == 0 else None)
            bottom += val
        for xi, t in zip(x, bottom):
            ax.text(xi, t, "One zone" if role == "k1" else "Split", ha="center", va="bottom",
                    fontsize=7, color=INK2)
    ax.set_xticks(np.arange(len(ders)), [f"{d:.0%}" for d in ders])
    ax.set_xlabel(DERATING_LABEL, fontsize=9, color=INK2)
    ax.set_ylabel(DD_LABEL, fontsize=9, color=INK2)
    ax.set_title("Which lines cause the dispatch down", fontsize=10, color=INK, loc="left")
    _style(ax)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "attribution.png", dpi=150)
    plt.close(fig)


def fig_duration(hourly, sn, cfg):
    s = _primary([k[0] for k in hourly])
    head = "oos_headline" if s == "oos_even" else "is_headline"
    ders = sorted({k[2] for k in hourly if k[0] == s and k[1] == head})
    fig, axes = plt.subplots(1, len(ders), figsize=(4.4 * len(ders), 3.6), squeeze=False, sharey=True)
    for ax, der in zip(axes[0], ders):
        for mid, col, lab in (("k1", SAND_LINE, "DE as one zone"), (head, BLUE_LINE, "Split DE")):
            h = hourly.get((s, mid, der))
            if h is None:
                continue
            h = h[h.stageB_ok.astype(bool)]
            o = np.argsort(-h.DD_RES.to_numpy())
            wv = sn.weight.reindex(h.index).to_numpy()[o]
            ax.plot(np.cumsum(wv) / wv.sum(), h.DD_RES.to_numpy()[o] / 1e3, color=col, lw=2, label=lab)
        ax.set_xlabel("Share of hours", fontsize=9, color=INK2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(f"Trading capacity {der:.0%}", fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0, 0].set_ylabel("Wind + solar dispatch down (GW)", fontsize=9, color=INK2)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "dd_duration.png", dpi=150)
    plt.close(fig)


def dd_by_bus(out, mid, der, hours, w, gens, cfg):
    d = out / "runs" / f"{mid}_d{der:g}"
    h = pd.read_parquet(d / "hourly.parquet")
    ok = h.index[h.stageB_ok.astype(bool)].intersection(hours)
    sc = 8760.0 / w.reindex(ok).sum()
    res = gens.index[gens.carrier.isin(cfg["validate"]["res_carriers"])]
    sp = pd.read_parquet(d / "market_spill.parquet")
    rd = pd.read_parquet(d / "redispatch.parquet")
    sp = sp[sp.snapshot.isin(ok) & sp.generator.isin(res)]
    rd = rd[rd.snapshot.isin(ok) & rd.generator.isin(res)]
    x = pd.concat([pd.DataFrame({"g": sp.generator, "e": sp.mw * w.reindex(sp.snapshot).to_numpy()}),
                   pd.DataFrame({"g": rd.generator, "e": rd.down * w.reindex(rd.snapshot).to_numpy()})])
    return (x.groupby("g").e.sum() * sc / 1e3).groupby(gens.bus).sum()


def fig_map(out, maps, sn, cfg, buses, der=None):
    from bzgen.plot import maps as pm
    import geopandas as gpd
    from bzgen.validate.zonemap import iso_week
    der = cfg["validate"]["reference_derating"] if der is None else der
    gens = pd.read_csv(ROOT / "data" / "interim" / "generators.csv", index_col=0)
    f = cfg["validate"]["focus"]
    shape = pm.country_shape(sorted(buses[buses.cluster_country == f].country.unique()))
    head = "oos_headline" if "oos_headline" in maps else "is_headline"
    even = iso_week(sn.index).iso_week.to_numpy() % 2 == 0
    hours = sn.index[even] if maps[head]["fit"] == "oos" else sn.index
    lab = maps[head]["labels"]
    dd = {m: dd_by_bus(out, m, der, hours, sn.weight, gens, cfg) for m in ("k1", head)}
    vals = {m: dd[m].reindex(lab.index).fillna(0).groupby(lab).sum() / 1e3 for m in dd}
    vmax = max(float(v.max()) for v in vals.values()) or 1.0
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("zb", ["#ffffff", BLUE, "#2e4a6b"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.6))
    cells = pm.zone_cells(buses, lab, shape)
    for ax, m, title in ((axes[0], "k1", "DE as one zone"), (axes[1], head, "Split DE")):
        for _, r in cells.iterrows():
            val = vals[m].get(r.zone, 0.0)
            gpd.GeoSeries([r.geometry], crs=pm.CRS).plot(ax=ax, color=cmap(0.15 + 0.85 * val / vmax),
                                                         edgecolor="white", lw=1.2)
        gpd.GeoSeries([shape], crs=pm.CRS).plot(ax=ax, facecolor="none", edgecolor="0.3", lw=0.6)
        for z, val in vals[m].items():
            big = cells[cells.zone == z].geometry.explode(index_parts=False)
            c = big.iloc[int(np.argmax(big.area.to_numpy()))].representative_point()
            ax.text(c.x, c.y, f"{val:.1f} TWh", ha="center", va="center", fontsize=8, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_axis_off()
    fig.suptitle(f"Wind + solar dispatch down per zone (trading capacity {der:.0%})", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / f"dd_map_de_d{der:g}.png", dpi=150)
    plt.close(fig)
    return vals


def fig_maps_under_test(maps, buses):
    from bzgen.plot import maps as pm
    lines = pd.read_csv(ROOT / "data" / "interim" / "lines.csv", index_col=0)
    links = pd.read_csv(ROOT / "data" / "interim" / "links.csv", index_col=0)
    mem = sorted(buses[buses.cluster_country == "DE"].country.unique())
    names = {"is_headline": "Full year", "oos_headline": "Odd weeks (test map)"}
    ids = [m for m in ["is_headline", "oos_headline", "oos_sib1", "oos_sib2", "oos_sib3"] if m in maps]
    fig, axes = plt.subplots(1, len(ids), figsize=(3.3 * len(ids), 4.2), squeeze=False)
    for ax, i in zip(axes[0], ids):
        pm.draw_country(ax, buses, lines, links, maps[i]["labels"], mem,
                        title=names.get(i, i.replace("oos_sib", "Odd weeks, alternative ")))
        ax.title.set_fontsize(9)
    fig.suptitle(f"DE bidding zones under test ({ALPHA_LABEL.lower()} = 2)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "maps_under_test.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def write(cfg, sn, out, tag, maps, zmaps, pocket, buses, real_rep, stage):
    FIGS.mkdir(parents=True, exist_ok=True)
    v = cfg["validate"]
    rows, deltas, checks, hourly = aggregate(cfg, sn, out, maps, stage)
    csv = RESULTS / ("validate.csv" if tag == "all" else f"validate_{tag}.csv")
    rows.to_csv(csv, index=False)
    checks.to_csv(RESULTS / "validate" / f"checks_{tag}.csv", index=False)
    nodal_meta = json.loads((out / "nodal" / "done.json").read_text())
    failed = checks[checks.asserted & (checks.violations > 0)]
    bnd = pd.DataFrame()
    vals = None
    if stage == "AB" and len(deltas):
        bnd = I.bands(deltas, cfg)
        fig_delta(deltas, cfg)
        fig_attribution(rows, cfg)
        fig_duration(hourly, sn, cfg)
        try:
            for der in sorted(rows.derating.unique()):
                vv = fig_map(out, maps, sn, cfg, buses, der)
                if np.isclose(der, v["reference_derating"]):
                    vals = vv
            fig_maps_under_test(maps, buses)
        except Exception as e:           # a map figure must not hide the numbers
            vals = f"map figure failed: {e!r}"
    text = markdown(cfg, sn, tag, maps, rows, deltas, checks, bnd, nodal_meta, pocket, buses,
                    real_rep, stage, out, vals)
    name = "validate.md" if tag == "all" else f"validate_{tag}.md"
    (REPORTS / name).write_text(text)
    if len(failed):
        raise AssertionError("validity checks failed (see reports/" + name + "):\n"
                             + failed.to_string())


def markdown(cfg, sn, tag, maps, rows, deltas, checks, bnd, nodal_meta, pocket, buses, real_rep,
             stage, out, vals) -> str:
    v = cfg["validate"]
    meta = json.loads((ROOT / "results" / "validate" / "maps" / "maps.json").read_text())
    L = ["# Validation: does splitting DE into zones reduce dispatch down?", ""]
    L += ["Generated by `python -m bzgen.validate.run`" + ("" if tag == "all" else
          f" on **{len(sn)} representative hours (`{tag}`, development run — not a result)**") + ".", ""]
    # ---- flags at the top
    fl = []
    if tag != "all":
        fl.append(f"**Development subsample** ({len(sn)} k-means hours, weighted). The week-block "
                  "bootstrap is not meaningful on weighted representative hours.")
    if stage == "A":
        fl.append("**Stage A only**: no redispatch, so no dispatch-down result.")
    if v.get("np_slack_penalty") is not None:
        fl.append(f"**Net positions are soft, not fixed (quick-result fix; the brief asked for hard "
                  f"equalities and no relaxation).** With hard equalities most hours were LP-infeasible: the "
                  f"zonal market schedules exchanges the meshed grid cannot deliver even with load shedding. "
                  f"Deviation from a zone's market net position costs €{v['np_slack_penalty']:g}/MWh (above every "
                  "generator cost, below load shedding), so net positions hold wherever redispatch can deliver "
                  "them. The undeliverable volume is reported (`np_slack_*`, `np_undeliverable_hours*`) and "
                  "is itself a result: it measures how far the zonal market's exchanges exceed what the grid "
                  "can carry.")
    fl.append("**Zone polygons are not an ENTSO-E publication.** No authoritative bidding-zone polygon set "
              "was reachable (transparency.entsoe.eu and entsoe.eu blocked by the session's egress policy). "
              f"SE1–4, NO1–5, DK1–2 and six Italian zones come from Electricity Maps' curated open "
              f"geometry (`geo/world.geojson`, commit `{v['zones_source']['commit'][:12]}`), hash in "
              "`data/PROVENANCE.md`.")
    fl.append("**Italy has 6 zones, not the 7 in force since 2021**: CALA (Calabria) is not in the geometry and "
              "sits inside IT-SUD, as in the brief.")
    fl.append(f"**Sibling tolerance widened** to {v['sibling_rel_tol']:.3g} (from "
              f"`sweep.degeneracy_rel_tol` = {cfg['sweep']['degeneracy_rel_tol']:g}, at which DE has no "
              "near-degenerate restart). Siblings are re-run restarts (fixed crc32 seeds; the sweep's "
              "`hash()` seeds are not reproducible). Fewer than "
              f"{v['n_siblings']} siblings are reported as found, not padded.")
    fl.append(f"**BNetzA redispatch volume is not sourced**: the {v['bnetza_redispatch_twh'][0]:g}–"
              f"{v['bnetza_redispatch_twh'][1]:g} TWh/yr range is the order of magnitude given in the brief "
              "(bundesnetzagentur.de, smard.de, netztransparenz.de blocked).")
    fl.append("**Chronic load pockets stay in the model** (decision: exclude from statistics only). Their "
              "stage-B shedding forces equal down-regulation elsewhere in the same zone, which counts as "
              "dispatch down and lands differently under k = 1 (anywhere in DE) and the split map (only in "
              "the pocket's zone). This is a confounder of ΔDD, not a finding.")
    fl.append(f"**Inferred values** (config, not sourced): reference derating {v['reference_derating']:g}, "
              f"markup €{v['markup']:g}/MWh, pocket threshold {v['pocket_hour_share']:.0%} of hours, dual "
              f"tolerance {v['dual_tol']:g} €/MWh, RES = {', '.join(v['res_carriers'])}; the derating grid is "
              "the brief's.")
    fl.append("**α = 2 was chosen on the full year** (reports/results.md); the out-of-sample refit reuses it, "
              "a small leak of test weeks into the map definition.")
    fl.append("**Solver path**: stage A and B run on the benchmark's highspy LP (`bzgen/solve/lp.py`), not on "
              "`n.optimize()` (≈40 s/h outside the solver on this network); the PyPSA/linopy build of stage B "
              "with net positions added via `n.model.add_constraints` is the cross-check "
              "(`tests/test_validate.py`, and `--check` on real hours).")
    L += ["> ## Fallbacks and flags", ">"] + [f"> - {x}" for x in fl] + [""]
    L += ["Dispatch down in this model is **entirely constraint-driven**. A DC-OPF has no SNSP or "
          "inertia limit, so there is no system-wide pro-rata curtailment component. Every MWh counted "
          "here comes from a binding zonal ATC, a binding network branch, or zone-wide oversupply "
          "that no link can export.", ""]
    # ---- headline
    L += ["## Headline", ""]
    if stage == "AB" and len(bnd):
        for _, b in bnd.sort_values("scoring", key=lambda s: s != "oos_even").iterrows():
            nm = "Out-of-sample (primary)" if b.scoring == "oos_even" else "In-sample (appendix)"
            L += [f"**{nm}.** {I.verdict(b)}", "",
                  f"ΔDD_RES (split − k=1) at derating {b.reference_derating:g}: **{b.dDD_point:+,.0f} GWh/yr**; "
                  f"week-block 95% CI [{b.boot_lo:+,.0f}, {b.boot_hi:+,.0f}]; derating band "
                  f"[{b.derating_min:+,.0f}, {b.derating_max:+,.0f}]; sibling band "
                  f"[{b.sibling_min:+,.0f}, {b.sibling_max:+,.0f}] over {b.n_maps_in_sibling_band} map(s).", ""]
        L += ["Negative ΔDD means less dispatch down with DE split.", ""]
    else:
        L += ["No dispatch-down result in this run.", ""]
    # ---- method
    L += ["## Method", "",
          "1. **Zones.** DE-LU from the map under test (k = 1: one zone). SE, NO, DK, IT by real bidding "
          "zone (point-in-polygon; buses outside every polygon of their own country take the nearest one). "
          "Every other country is one zone.",
          f"2. **Stage A — market.** Buses collapsed to zones. Generators and loads keep their identity. "
          f"Inter-zone AC cuts become one Link per zone pair (`p_nom = derating · Σ s_nom · s_max_pu`, "
          f"s_max_pu = {cfg['network']['s_max_pu']}). HVDC is kept as it is, and intra-zone lines are "
          "deleted. No KVL.",
          "3. **Stage B — redispatch.** The full nodal network with KVL. Each unit gets an up part "
          "(`mc + markup`) and a down part (`−mc + markup` per MWh backed down). The market schedule is "
          "a fixed injection. One equality per zone and hour holds the net position (Σ up = Σ down). "
          "Net positions are soft in this run (flag above). Load shedding is available at "
          f"€{cfg['solve']['load_shedding_cost']:g}/MWh. No relaxation and no counter-trading.",
          "4. **Metrics** (DE-LU incl. offshore): DD_RES = market spill[RES] + redispatch down[RES]; "
          "DD_RES_TH adds thermal down-regulation. Attribution: market spill over binding zonal "
          "links (stage-A duals), redispatch down over binding nodal branches touching DE (stage-B duals), "
          "weighted by |μ| · capacity. Cost gap `(C_market + C_redispatch − C_nodal) / C_nodal` "
          "(physical costs, markup excluded).",
          "5. **Scoring.** Primary: maps refit on odd ISO weeks (edge statistics from `prices.parquet`, "
          "same anneal), with stages A and B scored on even weeks only. k = 1 needs no refit. Appendix: "
          "full-year maps scored on the full year. Paired by hour. The bootstrap resamples by ISO week "
          f"({v['bootstrap']['resamples']} resamples, {v['bootstrap']['ci']:.0%}).",
          "6. **Same inputs as the benchmark**: the same snapshots, the same seeded cost noise "
          f"(`anneal.seed`, U[{cfg['solve']['cost_noise'] / 10:g}, {cfg['solve']['cost_noise']:g}) €/MWh) "
          "and the same worker/block parallelism. The nodal reference was re-solved; its prices differ "
          f"from `data/solved/prices.parquet` by at most "
          f"{nodal_meta.get('price_check_max_abs', float('nan')):.2e} €/MWh.", ""]
    # ---- maps
    L += ["## Maps under test", ""]
    mt = pd.DataFrame([{"map": k, "role": m["role"], "fit": m["fit"], "energy": m.get("energy", np.nan),
                        "restart": m.get("restart", "")} for k, m in maps.items()]).set_index("map")
    L += [md(mt, 3), ""]
    if "ari_oos_vs_is_headline" in meta:
        L += [f"ARI between the out-of-sample headline and the in-sample headline: "
              f"{meta['ari_oos_vs_is_headline']:.2f}.", ""]
    for nm, fn in (("In-sample restarts", "is_restarts.csv"), ("Out-of-sample restarts", "oos_restarts.csv")):
        f = ROOT / "results" / "validate" / "maps" / fn
        if f.exists():
            L += [f"{nm} (`results/validate/maps/{fn}`):", "", md(pd.read_csv(f).set_index("restart"), 4), ""]
    L += ["Real foreign zones (bus counts):", "", md(real_rep, 1), ""]
    # ---- results
    L += ["## Results", ""]
    if stage == "AB" and "DD_RES_GWh" in rows:
        cols = ["scoring", "map", "derating", "DD_RES_GWh", "DD_RES_TH_GWh", "DD_share_of_avail_res",
                "internal_share", "border_share", "C_redispatch_EUR", "C_total_EUR", "efficiency_gap",
                "lp_infeasible_hours", "np_undeliverable_hours", "np_slack_focus_GWh",
                "shed_hours_excl_pockets", "shed_focus_GWh", "shed_pocket_GWh",
                "dDD_RES_GWh", "dDD_RES_lo", "dDD_RES_hi"]
        t = rows[[c for c in cols if c in rows]].copy()
        for c in ("C_redispatch_EUR", "C_total_EUR"):
            t[c] = t[c] / 1e6
        t = t.rename(columns={"C_redispatch_EUR": "C_redispatch_MEUR", "C_total_EUR": "C_total_MEUR"})
        L += ["All of `results/validate.csv`, abridged (GWh/yr, M€/yr; shed energy is reported separately "
              "and never counted as dispatch down):", "", md(t.set_index(["scoring", "map", "derating"]), 3), ""]
        L += ["![ΔDD versus derating](../figures/validate/delta_dd_vs_derating.png)", "",
              "![attribution](../figures/validate/attribution.png)", "",
              "![duration](../figures/validate/dd_duration.png)", "",
              f"![map](../figures/validate/dd_map_de_d{v['reference_derating']:g}.png)", ""]
        if isinstance(vals, str):
            L += [f"_{vals}_", ""]
        k1 = rows[(rows["map"] == "k1") & (rows.scoring == rows.scoring.iloc[0])]
        L += ["### k = 1 redispatch volume versus BNetzA", "",
              md(k1.set_index("derating")[["redispatch_up_focus_GWh", "redispatch_down_focus_GWh",
                                           "DD_RES_GWh"]].assign(
                  up_plus_down_TWh=lambda x: (x.redispatch_up_focus_GWh + x.redispatch_down_focus_GWh) / 1e3), 1),
              "",
              f"Reference: {v['bnetza_redispatch_twh'][0]:g}–{v['bnetza_redispatch_twh'][1]:g} TWh/yr (unsourced, "
              "from the brief). Nothing is tuned to match it. A large gap is expected from the 220 kV "
              "truncation (no 110 kV grid, load pockets) and from the idealised market (one perfect ATC "
              "auction, perfect redispatch).", ""]
    else:
        t = rows[["scoring", "map", "derating", "market_spill_res_GWh_allhours", "C_market_EUR_allhours",
                  "C_nodal_EUR_allhours"]].copy()
        L += [md(t.set_index(["scoring", "map", "derating"]), 1), ""]
    # ---- checks
    L += ["## Validity checks", "",
          "Asserted: `C_market ≤ C_nodal` every hour at derating 1 (the zonal model is a relaxation only "
          "there; at lower deratings the ATC is below what the physical cut carries, so violations are "
          "reported, not asserted); `C_market(k=1) ≤ C_market(split)` every hour; "
          "`C_market + C_redispatch ≥ C_nodal` every feasible hour. Tolerance "
          f"{v['cost_tol_rel']:g}·|C| + €{v['cost_tol_abs']:g}.", ""]
    ck = checks.groupby(["check", "derating", "asserted"]).agg(runs=("map", "size"), hours=("hours", "sum"),
                                                              violations=("violations", "sum"),
                                                              max_excess_EUR=("max_excess_EUR", "max"))
    L += [md(ck, 2), ""]
    nf = checks[checks.asserted & (checks.violations > 0)]
    L += ["**All asserted checks pass.**" if nf.empty else f"**{len(nf)} asserted checks FAIL** — see "
          "`results/validate/checks_*.csv`.", ""]
    # ---- pockets
    share = pd.read_csv(out / "nodal" / "shed_share.csv", index_col=0).shed_hour_share
    pk = share[share.index.isin(pocket)].sort_values(ascending=False).to_frame()
    pk["country"] = pk.index.map(buses.country)
    pk["lon"] = pk.index.map(buses.x).round(2)
    pk["lat"] = pk.index.map(buses.y).round(2)
    L += ["## Load pockets (excluded from the congestion statistics)", "",
          f"Buses shedding in ≥ {v['pocket_hour_share']:.0%} of nodal hours. Branches incident to them form "
          "the `pocket` attribution bucket, which is excluded from the internal/border shares. Shedding is "
          "reported separately (`shed_*` columns) and never counted as dispatch down.", "", md(pk, 3), ""]
    # ---- limitations
    L += ["## Limitations", "",
          "1. **Nodal network truncated at 220 kV** (reports/solve.md): the load pockets force redispatch that "
          "a real 110 kV grid would not need, and the k = 1 redispatch volume is not comparable to BNetzA's.",
          "2. **Idealised market**: one ATC auction with ATC = derating × thermal cut capacity, no flow-based "
          "coupling, no minimum-RAM rule, no intraday. Derating is a free parameter; the result is shown "
          "across it.",
          "3. **Idealised redispatch**: every unit can be redispatched at mc ± markup, with no gradients, "
          "must-run or cross-border redispatch limits. Net positions are fixed exactly.",
          "4. **No storage, hydro without energy limits, and 2019 weather** with the 2025 fleet (README).",
          "5. **Zone polygons are curated open data** (flag above), and Italy has 6 zones.",
          "6. **Degeneracy**: the DE map is one member of a near-degenerate family (reports/results.md). "
          "The sibling band measures how much that matters for dispatch down.",
          "7. **Cost-noise ties**: which of several zero-cost RES units spills in a copper-plate zone is decided "
          "by the seeded noise. Totals are meaningful; per-unit allocation within a zone is not.",
          "8. **Only DE is split.** Other countries keep one zone (or their real zones). Maps are "
          "partitions of DE-LU only.", ""]
    return "\n".join(L) + "\n"

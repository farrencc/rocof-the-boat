"""Presentation figures for the v2 model (figures/v2/fig*.png).

    python src/report.py --scope 26

Reads the v2 setup, sweep and validation outputs; explanations of every figure
are in docs/figures.md. Style rules: one message per figure (stated in the
title), one axis per quantity, fixed colour roles (blue = optimised / helpful,
grey = baseline / context, orange = conventional plant, red = problem),
small multiples instead of more than a few colours.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import upstream as up  # noqa: E402
from sweep import Problem, FIGURES, RESULTS  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"
INK, INK2, MUTED, GRID, FAINT = "#0b0b0b", "#52514e", "#9a9993", "#e4e3df", "#d6d5d0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 170,
})


def _subtitle(fig, text, y=0.995):
    fig.text(0.01, y, text, ha="left", va="top", fontsize=9, color=INK2)


def _short(name: str) -> str:
    return (name.replace("_A1", "").replace("POOLBEG SOUT", "POOLBEG S").replace("CARRICK ON S", "CARRICK-ON-S")
            .title().replace(" - ", " – "))


class Ctx:
    def __init__(self, scope):
        self.P = Problem(scope, "v2")
        self.g = self.P.grid
        self.root = self.P.root
        self.meta = self.P.anchor_meta
        self.anchors = self.meta["anchors"]
        self.nodes = self.P.nodes
        self.out = FIGURES / "v2"
        self.out.mkdir(parents=True, exist_ok=True)
        self.val = pd.read_csv(self.root / "validation" / "validation.csv")
        self.vsum = pd.read_csv(self.root / "validation" / "validation_summary.csv", index_col=0)
        self.sweep = pd.read_csv(self.root / "validation" / "sweep_summary.csv")
        self.head = json.loads((self.root / "validation" / "headline.json").read_text())
        self.best_cid = self.head["best_config"]
        self.M_best = np.load(self.root / self.best_cid / "final.npz")["best_E_M"].astype(bool)
        buses = self.g.network.buses.set_index("bus")
        br = self.g.model.branches.reset_index(drop=True)
        self.br, self.buses = br, buses
        self.rings = up.land_outlines()

    def line_xy(self, m):
        b0, b1 = self.br.loc[m, "bus0"], self.br.loc[m, "bus1"]
        return ((self.buses.loc[b0, "lon"], self.buses.loc[b0, "lat"]),
                (self.buses.loc[b1, "lon"], self.buses.loc[b1, "lat"]))

    def backdrop(self, ax, zoom=None):
        for r in self.rings:
            ax.fill(r[:, 0], r[:, 1], color="#f4f3f0", zorder=0)
            ax.plot(r[:, 0], r[:, 1], color=FAINT, lw=0.8, zorder=1)
        ax.set_aspect(1 / np.cos(np.deg2rad(53.4)))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        if zoom:
            ax.set_xlim(*zoom[0]); ax.set_ylim(*zoom[1])


# ------------------------------------------------------------------ fig 1
def fig1_line458(c: Ctx):
    """Why Poolbeg S – Carrickmines is overloaded, and who can relieve it."""
    P, g = c.P, c.g
    e, k = 458, 314
    rows = []
    sens = None
    for mem in P.ens.members:
        cs = mem.cases
        js = np.flatnonzero((cs.state_monitor == e) & (cs.state_outage == k))
        if not len(js):
            continue
        j = js[0]
        sens = cs.node_state_sensitivity[:, j]
        H = g.model.ptdf(cs.balance_bus)
        lodf, _ = up.lodf(g, H)
        row = H[e] + lodf[e, k] * H[k]
        bidx = g.model.bus_index
        dem = np.zeros((len(cs.case_weights), len(g.model.bus_ids)))
        lb = g.network.loads.set_index("load")["bus"].astype(str).to_dict()
        for l in g.network.load_profile.columns:
            if lb.get(l) in bidx:
                dem[:, bidx[lb[l]]] += g.network.load_profile[l].to_numpy()
        ren_idx = [bidx[b] for b in g.template.bus.astype(str)]
        n_ren = len(ren_idx)
        f = cs.state_base_flows_mw[:, j]
        over = np.abs(f) > cs.state_limits_mw[j]
        d_c = -(dem @ row)
        r_c = cs.pre_network_dispatch_mw[:, :n_ren] @ row[ren_idx]
        rest = f - d_c - r_c
        w = cs.case_weights * over
        if w.sum() > 0:
            rows.append(dict(demand=np.average(d_c, weights=w), renew=np.average(r_c, weights=w),
                             rest=np.average(rest, weights=w), net=np.average(f, weights=w),
                             limit=cs.state_limits_mw[j], share=float(np.sum(cs.case_weights * over))))
    d = pd.DataFrame(rows).mean()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    labels = ["Dublin & national demand", "Renewable output", "Conventional plant + HVDC", "Net flow (N-1)"]
    vals = [d.demand, d.renew, d.rest, d.net]
    cols = [RED, BLUE, ORANGE, INK]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=cols, height=0.55)
    for yy, v in zip(y, vals):
        ax.text(v + (8 if v >= 0 else -8), yy, f"{v:+.0f} MW", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK)
    lim = d.limit
    ax.axvline(-lim, color=RED, ls="--", lw=1.2)
    ax.text(-lim, -0.55, f"  limit −{lim:.0f} MW", color=RED, fontsize=8.5, va="center")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlim(min(vals) * 1.35, max(vals) * 1.6)
    ax.set_xlabel("contribution to flow on Poolbeg S → Carrickmines (MW, snapshots where overloaded)")
    ax.set_title("Demand pulls power into Dublin across the cable")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    kind = P.ens.members[0].cases.node_kind
    helpful = sens > 0          # 458 overloads negative: curtailment helps if it raises the flow
    order = np.argsort(sens)
    x = np.arange(len(sens))
    ren = kind[order] == 0
    ax.axhspan(0, sens.max() * 1.15, color="#e8f1fc", zorder=0)
    ax.text(3, sens.max() * 1.05, "helps relieve", color=BLUE, fontsize=8.5, va="top")
    ax.text(len(sens) * 0.55, sens.min() * 0.3, "below 0: makes it worse", color=RED, fontsize=8.5, va="center")
    ax.scatter(x[ren], sens[order][ren], s=10, color=MUTED, label="wind / solar farm")
    ax.scatter(x[~ren], sens[order][~ren], s=42, marker="s", color=ORANGE, edgecolor="white", lw=1,
               label="conventional unit")
    ir = int(np.argmax(np.where(kind == 1, sens, -np.inf)))
    ax.annotate("Irishtown (turn down)", (int(np.flatnonzero(order == ir)[0]), sens[ir]),
                xytext=(-110, 10), textcoords="offset points", fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("nodes, sorted by sensitivity")
    ax.set_ylabel("MW of relief per MW reduced")
    ax.set_xticks([])
    ax.set_title(f"Only {int(helpful.sum())} of {len(sens)} nodes can help")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Branch 458: an overload driven by demand, not by wind", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", y=1.06)
    _subtitle(fig, f"Poolbeg South – Carrickmines 220 kV under the Inchicore – Irishtown outage; overloaded in "
                   f"{100 * d.share:.0f}% of snapshots. The Carrickmines phase-shifter is fixed at 0° in the data.",
              y=1.0)
    fig.tight_layout()
    fig.savefig(c.out / "fig1_line458_diagnosis.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2
def fig2_anchor_ranking(c: Ctx):
    tab = pd.read_csv(c.P.setup / "branch_binding.csv", index_col=0)
    tab["score"] = tab.p_literal + tab.over_w / tab.over_w.sum()
    top = tab.sort_values("score", ascending=False).head(16)
    anchors = set(c.meta["branches"])
    guard = {a["monitor"] for a in c.anchors if a.get("guard_anchor")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    y = np.arange(len(top))[::-1]
    names = [f"{_short(n)}  ({m})" for m, n in zip(top.index, top.name)]
    colors = [RED if m in guard else BLUE if m in anchors else (MUTED if r else FAINT)
              for m, r in zip(top.index, top.relievable)]
    for ax, col, lab in ((axes[0], top.bind / tab.bind.sum(), "share of binding events"),
                         (axes[1], top.over_w / tab.over_w.sum(), "share of pre-relief overloads")):
        ax.barh(y, 100 * col, color=colors, height=0.62)
        for yy, v in zip(y, 100 * col):
            if v > 0.05:
                ax.text(v + 0.8, yy, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
        ax.set_xlabel(lab + " (%)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, names, fontsize=8.5)
    axes[0].set_title("Binding: argmax at each relief step")
    axes[1].set_title("Overloaded before any relief")
    handles = [Line2D([], [], color=BLUE, lw=8, label="anchor (≥1% on either measure)")]
    if guard:
        handles.append(Line2D([], [], color=RED, lw=8, label="guard anchor"))
    handles += [
               Line2D([], [], color=MUTED, lw=8, label="relievable, below 1%"),
               Line2D([], [], color=FAINT, lw=8, label="no grouping can relieve it")]
    axes[1].legend(handles=handles, loc="lower right", fontsize=8.5)
    fig.suptitle(f"{len(anchors)} problem lines become constraint-group anchors", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", y=1.04)
    _subtitle(fig, "Stressed ensemble: 5 weather seeds × thermal ratings 100/95/90%. Binding counts are masked by 458 "
                   "(the relief loop stops there), so pre-relief overloads are used as well.", y=0.99)
    fig.tight_layout()
    fig.savefig(c.out / "fig2_anchor_selection.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 3
def fig3_anchor_map(c: Ctx):
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, zoom, title in ((axes[0], None, "All anchors"),
                            (axes[1], ((-6.75, -6.0), (53.2, 53.78)), "Dublin & north-east (zoom)")):
        c.backdrop(ax, zoom)
        n = c.nodes
        ax.scatter(n.longitude, n.latitude, s=6, color=FAINT, zorder=2)
        for i, a in enumerate(c.anchors):
            (x0, y0), (x1, y1) = c.line_xy(a["monitor"])
            col = RED if a.get("guard_anchor") else BLUE
            ax.plot([x0, x1], [y0, y1], color=col, lw=4, solid_capstyle="round", zorder=4)
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            if zoom is None or (zoom[0][0] < xm < zoom[0][1] and zoom[1][0] < ym < zoom[1][1]):
                ax.text(xm, ym, str(i + 1), fontsize=8, fontweight="bold", color="white", ha="center",
                        va="center", zorder=6, bbox=dict(boxstyle="circle,pad=0.25", fc=col, ec="white", lw=1))
        ax.set_title(title)
    leg = "\n".join(f"{i + 1:>2}  {_short(a['name'])}" + ("  (guard)" if a.get("guard_anchor") else "")
                    for i, a in enumerate(c.anchors))
    axes[0].text(0.0, 0.0, leg, transform=axes[0].transAxes, fontsize=8, color=INK, va="bottom",
                 family="DejaVu Sans Mono", bbox=dict(fc="white", ec=GRID))
    fig.suptitle("Where the anchor lines are", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(c.out / "fig3_anchor_map.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 4
def fig4_groups(c: Ctx, M, tag, title):
    K = M.shape[1]
    ncol = 5
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    n = c.nodes
    cap = n.capacity_mw.to_numpy()
    size = 6 + 90 * np.sqrt(cap / cap.max())
    conv = (n.kind == "conventional").to_numpy()
    for k in range(len(axes)):
        ax = axes[k]
        if k >= K:
            ax.axis("off")
            continue
        c.backdrop(ax)
        a = c.anchors[k]
        sel = M[:, k]
        helpful = c.P.sigma[:, k] > 0
        ax.scatter(n.longitude[~sel], n.latitude[~sel], s=4, color=FAINT, zorder=2)
        r = sel & ~conv
        ax.scatter(n.longitude[r], n.latitude[r], s=size[r], color=np.where(helpful[r], BLUE, RED),
                   edgecolor="white", lw=0.7, zorder=3, alpha=0.9)
        cv = sel & conv
        ax.scatter(n.longitude[cv], n.latitude[cv], s=size[cv], marker="s", color=ORANGE, edgecolor="white",
                   lw=0.8, zorder=4)
        (x0, y0), (x1, y1) = c.line_xy(a["monitor"])
        ax.plot([x0, x1], [y0, y1], color=INK, lw=3.2, solid_capstyle="round", zorder=5)
        mw = cap[sel].sum()
        mu = np.sum(cap[sel] * c.P.sigma[sel, k]) / max(cap[sel].sum(), 1e-9)
        ax.set_title(f"{k + 1}. {_short(a['name'])}", fontsize=9)
        ax.text(0.02, 0.02, f"{int(sel.sum())} nodes · {mw:.0f} MW\nmean σ {mu:.3f}"
                + (f" · {int(cv.sum())} conv." if cv.any() else ""), transform=ax.transAxes, fontsize=7.5,
                color=INK2, va="bottom")
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, ms=7, label="farm that helps its anchor"),
               Line2D([], [], marker="o", ls="", color=RED, ms=7, label="farm with the wrong sign"),
               Line2D([], [], marker="s", ls="", color=ORANGE, ms=7, label="conventional unit (turns down)"),
               Line2D([], [], color=INK, lw=3, label="anchor line")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(c.out / f"fig4_groups_{tag}.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 5
def fig5_validation(c: Ctx):
    v = c.val
    best = f"{c.best_cid}:bestE"
    base = v[v.grouping == "wdt_multi"].set_index(["seed", "thermal_scale"])
    opt = v[v.grouping == best].set_index(["seed", "thermal_scale"])
    idx = base.index
    labels = [f"seed {s}\n{int(ts * 100)}% rating" for s, ts in idx]
    y = np.arange(len(idx))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.42 * len(idx) + 2), sharey=True,
                             gridspec_kw={"width_ratios": [2.2, 1, 1]})
    ax = axes[0]
    for yy, i in zip(y, idx):
        ax.plot([base.loc[i, "D"], opt.loc[i, "D"]], [yy, yy], color=FAINT, lw=3, zorder=1)
    ax.scatter(base.D, y, s=60, color=MUTED, zorder=2, label="WDT groups today")
    ax.scatter(opt.loc[idx, "D"], y, s=60, color=BLUE, zorder=3, label="anchored groups (optimised)")
    for yy, i in zip(y, idx):
        dd = opt.loc[i, "D"] - base.loc[i, "D"]
        ax.text(max(base.loc[i, "D"], opt.loc[i, "D"]) + 0.08, yy, f"{dd:+.2f} pp", va="center", fontsize=8,
                color=INK2)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("renewable dispatch-down (% of available energy)")
    ax.set_title("Dispatch-down on fresh seeds", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=2, fontsize=8.5)
    ax = axes[1]
    ax.barh(y, opt.loc[idx, "C"], color=ORANGE, height=0.55)
    ax.set_xlabel("conventional redispatch\n(% of renewable potential)")
    ax.set_title("Cost moved to plant", pad=28)
    ax = axes[2]
    ax.scatter(base.security_pct, y, s=40, color=MUTED)
    ax.scatter(opt.loc[idx, "security_pct"], y, s=40, color=BLUE, marker="D")
    nf = opt.loc[idx, "new_failures_vs_wdt"]
    for yy, n_ in zip(y, nf):
        if n_:
            ax.text(opt.security_pct.max() + 0.3, yy, f"{int(n_)} new", color=RED, fontsize=8, va="center")
    ax.set_xlabel("snapshots secure after relief (%)")
    ax.set_title("Security", pad=28)
    for a in axes:
        a.grid(axis="y", visible=False)
    s = c.vsum.loc[best]
    w0 = c.vsum.loc["wdt_multi"]
    fig.suptitle(f"Anchored groups cut wind/solar dispatch-down by {-s.dD_vs_wdt_pct:.0f}% and raise security "
                 f"from {w0.security_mean:.0f}% to {s.security_mean:.0f}% of snapshots",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.03)
    _subtitle(fig, f"Configuration {c.best_cid}; 5 unseen weather seeds × 3 thermal ratings, identical frozen cases "
                   f"for both groupings.", y=0.985)
    fig.tight_layout()
    fig.savefig(c.out / "fig5_validation.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 6
def fig6_tradeoff(c: Ctx):
    vs = c.vsum
    rows = vs[vs.index.str.endswith(":bestE")].copy()
    rows["cid"] = rows.index.str.split(":").str[0]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.scatter(rows.C_mean, rows.D_mean, s=70, color=BLUE, edgecolor="white", lw=1.5, zorder=3)
    keep = {"dd_only": "renewable DD only", "dd_conv": "DD + conv. cost", "base": "base", "P2": "λ_P=2 (headline)",
            "structural_only": "structural only", "N0": "λ_N=0"}
    for _, r in rows.iterrows():
        key = r.cid.split("_", 1)[1]
        if key not in keep:
            continue
        lab = keep[key]
        ax.annotate(lab, (r.C_mean, r.D_mean), xytext=(6, 3), textcoords="offset points", fontsize=8, color=INK2)
    w = vs.loc["wdt_multi"]
    ax.scatter([w.C_mean], [w.D_mean], s=160, marker="*", color=INK, zorder=4)
    ax.annotate("WDT groups today", (w.C_mean, w.D_mean), xytext=(8, -4), textcoords="offset points", fontsize=9)
    i0 = vs.loc["init_shift_factor"]
    ax.scatter([i0.C_mean], [i0.D_mean], s=70, marker="^", color=MUTED, zorder=4)
    ax.annotate("initial anchored groups", (i0.C_mean, i0.D_mean), xytext=(8, -4), textcoords="offset points",
                fontsize=9, color=INK2)
    ax.set_xlabel("conventional redispatch used for relief (% of renewable potential)")
    ax.set_ylabel("renewable dispatch-down (%)")
    ax.set_title("Each λ setting trades wind curtailment against plant redispatch", fontsize=12)
    fig.text(0.01, -0.02, "Validation means; unlabelled dots are the other one-at-a-time λ changes. Base (λ_DD=1, λ_S=λ_V=λ_P=0.5, "
                          "λ_N=0.1, λ_C=0.5).", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(c.out / "fig6_lambda_tradeoff.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 7
def fig7_fairness(c: Ctx):
    """Group sizes: v1 best (no floor, λ_N=0.5) vs v2 best (floor 5, λ_N=0.1)."""
    v1 = None
    p1 = RESULTS / "validation_26" / "headline.json"
    if p1.exists():
        cid1 = json.loads(p1.read_text())["best_config"]
        v1 = np.load(RESULTS / cid1 / "final.npz")["best_E_M"].astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    sets = [("v1 (no floor, λ_N=0.5)", v1, MUTED), (f"v2 {c.best_cid} (floor 5, λ_N=0.1)", c.M_best, BLUE)]
    for ax, metric, xl in ((axes[0], "n", "nodes per group"), (axes[1], "mw", "capacity per group (MW)")):
        for yy, (lab, M, col) in enumerate(sets[::-1]):
            if M is None:
                continue
            cap = c.nodes.capacity_mw.to_numpy()[:M.shape[0]]
            vals = M.sum(0) if metric == "n" else (M * cap[:, None]).sum(0)
            jit = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.25
            ax.scatter(vals, yy + jit, s=55, color=col, edgecolor="white", lw=1, zorder=3)
        ax.set_yticks([0, 1], [s[0] for s in sets[::-1]])
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axes[0].axvline(5, color=RED, ls="--", lw=1)
    axes[0].text(5.3, 1.45, "floor = 5", color=RED, fontsize=8.5)
    axes[0].set_xscale("log")
    axes[0].set_title("No more single-farm groups")
    axes[1].set_title("Each constraint is shared across real capacity")
    fig.tight_layout()
    fig.savefig(c.out / "fig7_group_fairness.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 8
def fig8_annealing(c: Ctx):
    h = pd.read_csv(c.root / c.best_cid / "history.csv")
    x = h.blackbox_calls / 1000
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.4, 1, 1]})
    axes[0].plot(x, h.current_energy, color=BLUE, alpha=0.35, lw=1, label="current")
    axes[0].plot(x, h.best_energy, color=BLUE, lw=2, label="best so far")
    axes[0].set_ylabel("energy E")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Energy falls as the schedule cools")
    axes[1].plot(x, h.current_D, color=BLUE, lw=1.6, label="dispatch-down D (%)")
    axes[1].plot(x, h.raw_C, color=ORANGE, lw=1.6, label="conventional redispatch C (%)")
    axes[1].set_ylabel("% of potential")
    axes[1].legend(fontsize=8)
    axes[1].set_title("…while D and C move against each other")
    axes[2].plot(x, h.temperature, color=MUTED)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("temperature")
    n_prop = h.blackbox_calls.diff().fillna(h.blackbox_calls.iloc[0]).clip(lower=1)
    axes[3].plot(x, h.acceptance_rate, color=AQUA, lw=1.4, label="accepted")
    axes[3].plot(x, h.infeasible_proposals / n_prop, color=RED, lw=1, alpha=0.7, label="rejected by security guard")
    axes[3].set_ylim(0, 1)
    axes[3].set_ylabel("fraction of moves")
    axes[3].set_xlabel("thousand ensemble evaluations")
    axes[3].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Annealing run {c.best_cid}", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(c.out / "fig8_annealing.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 9
def fig9_sigma(c: Ctx):
    K = c.M_best.shape[1]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    cap = c.nodes.capacity_mw.to_numpy()
    rng = np.random.default_rng(1)
    for k in range(K):
        for off, M, col in ((-0.18, c.P.M0, MUTED), (0.18, c.M_best, BLUE)):
            sel = np.flatnonzero(M[:, k])
            s = c.P.sigma[sel, k]
            ax.scatter(k + off + (rng.random(len(s)) - 0.5) * 0.22, s, s=6 + 40 * np.sqrt(cap[sel] / cap.max()),
                       color=col, alpha=0.75, edgecolor="none")
            mu = np.sum(cap[sel] * s) / cap[sel].sum()
            ax.plot([k + off - 0.14, k + off + 0.14], [mu, mu], color=INK, lw=2)
    ax.axhline(0, color=RED, lw=1, ls="--")
    ax.set_xticks(range(K), [f"{k + 1}. {_short(a['name'])}" for k, a in enumerate(c.anchors)], rotation=30,
                  ha="right", fontsize=8.5)
    ax.set_ylabel("σ: MW relief per MW reduced (>0 helps)")
    ax.set_yscale("symlog", linthresh=0.02)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, label="initial group"),
                       Line2D([], [], marker="o", ls="", color=BLUE, label="optimised group"),
                       Line2D([], [], color=INK, lw=2, label="capacity-weighted mean")], fontsize=8.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    ax.set_title("How strongly each member helps its own anchor line")
    fig.tight_layout()
    fig.savefig(c.out / "fig9_sigma_by_group.png")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="26")
    args = ap.parse_args(argv)
    c = Ctx(args.scope)
    for f in (fig1_line458, fig2_anchor_ranking, fig3_anchor_map, fig5_validation, fig6_tradeoff,
              fig7_fairness, fig8_annealing, fig9_sigma):
        f(c)
        print("wrote", f.__name__, flush=True)
    fig4_groups(c, c.P.M0, "initial", "Initial groups: shift-factor clustering per anchor")
    fig4_groups(c, c.M_best, "optimised", f"Optimised constraint groups ({c.best_cid})")
    print("wrote fig4", flush=True)


if __name__ == "__main__":
    main()

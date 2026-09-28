"""The figures.

Five of them, and the first two are the argument.  Each one is a Pareto front or
a map -- the spec's stage 3 says to plot the non-dominated set rather than blend
the scores into one number, on the grounds that if the front is narrow the
scalarisation never mattered and if it is wide that is real information about
the network.  On this grid it is wide, so the fronts are the result.

Colours are the data-viz reference palette's categorical slots, taken in fixed
order and never cycled; the ninth series would fold into "other" rather than get
a generated hue.  (The palette ships validated; the validator itself is a node
script and node is not installed on this machine, so the default slots are used
unchanged rather than a hand-mixed set.)  Text is in ink tokens, never in a
series colour, so identity is always carried by a mark rather than by the label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402
from scipy import stats                                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import base                                                  # noqa: E402
import greens as gr                                          # noqa: E402
import sweep                                                 # noqa: E402

FIGS = HERE / "figures"

#: Categorical slots, fixed order, from the data-viz reference palette.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#a8a79e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.grid": True,
    "legend.frameon": False, "figure.dpi": 130})


def _pareto(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated set, both objectives minimised."""
    order = np.argsort(x)
    keep, best = [], np.inf
    for i in order:
        if y[i] < best:
            keep.append(i)
            best = y[i]
    return np.array(keep)


def fig_pareto(df: pd.DataFrame, xk: str, yk: str, xlabel: str, ylabel: str,
               path: Path, title: str) -> None:
    new = df[~df["existing"]]
    x, y = new[xk].to_numpy(), new[yk].to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.scatter(x, y, s=7, c=C1, alpha=0.35, lw=0, label="candidate edge")
    front = _pareto(x, y)
    o = front[np.argsort(x[front])]
    ax.plot(x[o], y[o], color=C2, lw=2, marker="o", ms=4,
            label="Pareto front", zorder=3)
    ex = df[df["existing"]]
    ax.scatter(ex[xk], ex[yk], s=14, facecolors="none", edgecolors=C3,
               lw=1.0, label="reinforce an existing circuit", zorder=2)
    ax.scatter([1.0], [1.0], s=60, marker="*", color=INK, zorder=4,
               label="no edge")
    for k, lab in ((xk, "best " + xlabel.split(",")[0]),
                   (yk, "best " + ylabel.split(",")[0])):
        r = new.loc[new[k].idxmin()]
        ax.annotate(f"{int(r.a)}-{int(r.b)}", (r[xk], r[yk]),
                    textcoords="offset points", xytext=(6, 5),
                    fontsize=8, color=INK2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, loc="left", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_curves(bc, cur, df: pd.DataFrame, path: Path) -> None:
    """sigma_max(omega) for the base case and the winners, with the band marked."""
    w = cur["omega"] / base.TWO_PI
    phys, art = base.artefact_floor(bc)
    cut = base.physical_band(bc)[1]
    new = df[~df["existing"]]
    picks = [(int(new.loc[new["r_C_freq"].idxmin(), "a"]),
              int(new.loc[new["r_C_freq"].idxmin(), "b"]), C2, "best C_freq"),
             (int(new.loc[new["r_C_rocof"].idxmin(), "a"]),
              int(new.loc[new["r_C_rocof"].idxmin(), "b"]), C3, "best C_rocof"),
             (int(new.loc[new["r_C_H2"].idxmin(), "a"]),
              int(new.loc[new["r_C_H2"].idxmin(), "b"]), C4, "best H2")]
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(cur["pairs"])}

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    ax.axvspan(w.min(), cut, color=C1, alpha=0.06)
    ax.axvline(art, color=MUTED, lw=1.0, ls="--")
    ax.annotate("lowest regularisation mode", (art, ax.get_ylim()[1]),
                rotation=90, fontsize=7, color=INK2, va="top", ha="right",
                xytext=(-3, -4), textcoords="offset points")
    ax.plot(w, cur["ref"][0], color=INK, lw=2, label="no edge", zorder=4)
    for a, b, col, lab in picks:
        ax.plot(w, cur["sigma"][idx[(a, b)]], color=col, lw=1.6,
                label=f"{lab}  ({a}-{b})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\sigma_{\max}(\Pi\,G\,B)$")
    ax.set_title("the response, and the band it is scored over", color=INK,
                 loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    ax2.axvspan(w.min(), cut, color=C1, alpha=0.06)
    ax2.plot(w, base.BaseCase.weight_freq(cur["omega"]), color=C1, lw=2,
             label="frequency weight")
    ax2.plot(w, base.BaseCase.weight_rocof(cur["omega"]), color=C2, lw=2,
             label=f"windowed RoCoF weight ({base.T_ROCOF * 1e3:.0f} ms)")
    ax2.set_xlabel("frequency, Hz")
    ax2.set_ylabel("weight")
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_map(bc, df: pd.DataFrame, path: Path, n: int = 8) -> None:
    """Where the best edges want to go."""
    xy = np.array([(c[1], -c[0]) for c in bc.g.coords], float)
    new = df[~df["existing"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8))
    for ax, key, col, lab in ((axes[0], "r_C_freq", C2, "frequency deviation"),
                              (axes[1], "r_C_rocof", C3, "windowed RoCoF")):
        for i, j in bc.g.edges:
            ax.plot(*xy[[i, j]].T, color="#cfcdc4", lw=1.0, zorder=1)
        role = bc.g.role.astype(str)
        for r, c, m, s in (("sync", C1, "s", 42), ("wind", C3, "^", 26),
                           ("city", C4, "o", 30), ("hvdc", C2, "D", 30)):
            k = role == r
            ax.scatter(*xy[k].T, c=c, marker=m, s=s, zorder=3, lw=0,
                       label=r)
        k = (role == "load")
        ax.scatter(*xy[k].T, c=MUTED, marker=".", s=8, zorder=2, lw=0)
        on = bc.H > 1.0
        ax.scatter(*xy[on].T, facecolors="none", edgecolors=INK, s=95, lw=1.4,
                   zorder=4, label="synchronised")
        for rank, r in enumerate(new.nsmallest(n, key).itertuples()):
            ax.plot(*xy[[int(r.a), int(r.b)]].T, color=col, lw=2.4,
                    alpha=1.0 - 0.07 * rank, zorder=5)
        ax.set_title(f"best {n} by {lab}", color=INK, loc="left", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_length(df: pd.DataFrame, path: Path) -> None:
    """Is the best edge just the longest one?"""
    new = df[~df["existing"]]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.scatter(new["length"], new["r_C_freq"], s=7, c=C1, alpha=0.3, lw=0)
    grp = new.groupby(new["length"].round(0))["r_C_freq"]
    ax.plot(grp.mean().index, grp.min().to_numpy(), color=C2, lw=2,
            marker="o", ms=4, label="best at that length")
    ax.set_xlabel("lattice length of the new edge")
    ax.set_ylabel("C_freq, relative to no edge")
    ax.set_title("does the winner just buy distance?", color=INK, loc="left",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def equivalent_snsp(levels: np.ndarray, mean: np.ndarray, value: float) -> float:
    """The SNSP at which the ensemble mean risk equals `value`.

    Reading the benefit sideways rather than vertically.  This is an *equivalence*,
    not a movement -- SNSP does not change -- so it says: an edge that lowers the
    risk by some amount is worth however many points of non-synchronous share
    would have raised it by the same amount.  That is the framing the
    local-effects study found landed best, and the one an operator can price,
    because points of SNSP are curtailment.
    """
    o = np.argsort(mean)
    return float(np.interp(value, mean[o], levels[o]))


def fig_risk_vs_snsp(nl: pd.DataFrame, info: dict, path: Path) -> None:
    """What the edge buys, against the ensemble it has to be judged in.

    The layout of ../snsp/figures/fig11_topbus_vs_predictor.png, top row: the
    240 operating states of the SNSP ensemble, their risk against their
    non-synchronous share, shaded by system inertia.  Onto that go two points --
    this configuration with no edge, and the same configuration with the best
    edge in it -- so that "how much risk does one line buy" is answered in the
    units the rest of the ensemble is already drawn in.
    """
    ens = pd.read_csv(base.SNSP / "ensemble_src.csv")
    ref = info["baseline_risk"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    panels = (("dev_top5", "peak $|f-50\\,$Hz$|$ on the worst 5 buses (Hz)",
               "frequency deviation"),
              ("rocof_top5", "500 ms RoCoF on the worst 5 buses (Hz/s)",
               "windowed RoCoF"))
    for ax, (key, ylab, title) in zip(axes, panels):
        lev = np.sort(ens["snsp_target"].unique())
        grp = ens.groupby("snsp_target")[key]
        mean, lo, hi = grp.mean().to_numpy(), grp.min().to_numpy(), grp.max().to_numpy()
        ax.fill_between(100 * lev, lo, hi, color="#e6e5e0", lw=0, zorder=1,
                        label="range within a level")
        sc = ax.scatter(100 * ens["snsp_target"], ens[key],
                        c=ens["stored_energy"], cmap="Blues", vmin=5,
                        vmax=float(ens["stored_energy"].max()), s=13, lw=0,
                        zorder=2)
        ax.plot(100 * lev, mean, color=INK2, lw=1.5, zorder=3,
                label="mean at each SNSP level")

        best = nl.loc[nl[f"r_R_{key}"].idxmin()]
        x0, y0, y1 = 100 * info["snsp"], ref[key], best[f"R_{key}"]
        ax.annotate("", xy=(x0, y1), xytext=(x0, y0), zorder=5,
                    arrowprops=dict(arrowstyle="->", color=C2, lw=1.8))
        ax.scatter([x0], [y0], s=130, marker="*", color=INK, zorder=6,
                   label="this configuration, no edge")
        ax.scatter([x0], [y1], s=62, marker="D", color=C2, zorder=6, lw=0,
                   label=f"with the best edge ({int(best.a)}–{int(best.b)})")
        d = 100 * (equivalent_snsp(lev, mean, y0) - equivalent_snsp(lev, mean, y1))
        ax.annotate(f"−{100 * (1 - best[f'r_R_{key}']):.1f}%\n"
                    f"≈ {d:.1f} pts of SNSP",
                    xy=(x0, y1), xytext=(-16, -12), textcoords="offset points",
                    ha="right", va="top", fontsize=8.5, color=C2,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE,
                              ec="none", alpha=0.85))
        ax.set_yscale("log")
        ax.set_xlabel("SNSP of the configuration (%)")
        ax.set_ylabel(ylab)
        ax.set_title(title, color=INK, loc="left", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left")
    cb = fig.colorbar(sc, ax=axes, fraction=0.026, pad=0.015)
    cb.set_label("system inertia, $\\sum H\\,S$ (p.u. s)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_cost_vs_risk(nl: pd.DataFrame, path: Path) -> None:
    """The cost against the risk, which is the point of computing the cost.

    A diagnostic of the *cost*, not of the edge.  The question it has to answer
    is "does a low cost predict a low risk", and answering it needs three things
    the first version of this figure did not have:

    1.  A SAMPLE THAT IS ENTITLED TO A CORRELATION.  The original 45 candidates
        were drawn from the top and bottom of each ranking -- a design built to
        span the range of *risk*, which puts 27 of 45 in the bottom decile of
        cost and leaves the middle of the axis empty.  A Spearman over that draw
        estimates the design, not the population.  The quoted rho is therefore
        computed on the plain random draw alone, and the design points are shown
        in a muted colour so they read as annotation rather than as evidence.

    2.  AN INTERVAL.  With n = 45 a rho of +0.14 and a rho of 0 are the same
        measurement.  Only a confidence interval distinguishes "no relationship"
        from "not enough points", and that distinction is the whole question.

    3.  THE SHAPE.  A rank correlation collapses a scatter to one number and
        cannot tell a flat relationship from a strong one confined to a corner.
        The binned median makes the shape visible: if low cost genuinely buys
        low RoCoF, the line falls from left to right whatever rho says.

    Two rows, one per cost, so the outcome cannot be blamed on the weighting:
    the bottom row puts C_rocof -- the cost carrying the windowed-RoCoF weight,
    and so the one that ought to know about RoCoF -- on the horizontal axis.
    Columns share a risk (and its scale), rows share a cost.
    """
    import nonlinear as nlmod

    costs = (("exact_C_freq", "$C_{freq}$"), ("exact_C_rocof", "$C_{rocof}$"))
    risks = (("r_R_rocof_top5", "RoCoF risk, worst 5 buses", C2),
             ("r_R_dev_top5", "deviation risk, worst 5 buses", C1))
    unb = nl["source"].isin(nlmod.UNBIASED)
    strat = nl["source"] == "stratified"
    design = ~(unb | strat)
    # A stratified draw over-weights the sparse tails by construction, so it
    # cannot join the correlation -- but each bin is internally a fair draw, so
    # it may join the binned median.  `trend` is that union; `unb` alone is the
    # only thing a rho is quoted on.
    trend = unb | strat

    def interval(x, y, n_boot: int = 4000, seed: int = 7):
        r = stats.spearmanr(x, y).statistic
        if len(x) < 30:
            return r, None, None
        rng = np.random.default_rng(seed)
        xy = np.column_stack([np.asarray(x), np.asarray(y)])
        bs = [stats.spearmanr(*xy[rng.integers(0, len(xy), len(xy))].T).statistic
              for _ in range(n_boot)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return r, lo, hi

    # The two columns do NOT share a vertical scale, and cannot: an edge moves
    # RoCoF risk over a range several times wider than deviation risk, so a
    # common axis would flatten the right-hand column into a line.  The span is
    # put in the column heading so the difference is read as a number rather
    # than inferred from the tick labels.
    spans = {yk: 100 * (nl[yk].max() - nl[yk].min()) for yk, _, _ in risks}
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6), sharex="row",
                             sharey="col")
    for i, (xk, xlab) in enumerate(costs):
        for j, (yk, ylab, col) in enumerate(risks):
            ax = axes[i][j]
            ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
            ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
            if strat.any():
                ax.scatter(nl.loc[strat, xk], nl.loc[strat, yk], s=9,
                           marker="o", c=MUTED, alpha=0.45, lw=0,
                           label="stratified on cost")
            if design.any():
                ax.scatter(nl.loc[design, xk], nl.loc[design, yk], s=30,
                           marker="^", facecolors="none", edgecolors=INK2,
                           lw=0.7, alpha=0.75,
                           label="design (top/worst of a ranking)")
            ax.scatter(nl.loc[unb, xk], nl.loc[unb, yk], s=13, marker="o",
                       c=col, alpha=0.8, lw=0, label="random draw")

            # Binned median over the fair draws, with the count per bin, so a
            # bin resting on three points cannot be read as a trend.
            if trend.sum() >= 40:
                xt = nl.loc[trend, xk].to_numpy()
                yt = nl.loc[trend, yk].to_numpy()
                q = np.quantile(xt, np.linspace(0, 1, 9))
                q[-1] += 1e-12
                mid, med = [], []
                for lo_, hi_ in zip(q[:-1], q[1:]):
                    k = (xt >= lo_) & (xt < hi_)
                    if k.sum() >= 5:
                        mid.append(np.median(xt[k]))
                        med.append(np.median(yt[k]))
                if len(mid) >= 3:
                    ax.plot(mid, med, color=INK, lw=1.6, marker="o", ms=3.5,
                            zorder=5, label="median of the fair draws")

            r_u, lo, hi = interval(nl.loc[unb, xk], nl.loc[unb, yk])
            r_d = stats.spearmanr(nl.loc[design, xk],
                                  nl.loc[design, yk]).statistic
            ci = (f"  95% CI [{lo:+.2f}, {hi:+.2f}]" if lo is not None else "")
            rho_line = (f"$\\rho$ = {r_u:+.3f} on the random draw"
                        f" (n = {int(unb.sum())}){ci}")
            sub = f"design sample alone: {r_d:+.3f} (n = {int(design.sum())})"
            head = (f"{ylab}\nspans {spans[yk]:.1f}% across these candidates\n"
                    if i == 0 else "")
            ax.set_title(f"{head}{rho_line}\n{sub}", color=INK, loc="left",
                         fontsize=9)
            if i == len(costs) - 1:
                ax.set_xlabel(f"{xlab} (exact), relative to no edge")
            # Both columns get a label: they are different quantities on
            # different scales, so an unlabelled tick column would invite the
            # reader to compare them directly.
            ax.set_ylabel(f"{ylab},\nrelative to no edge")
        axes[i][0].annotate(f"{xlab} on the horizontal", xy=(-0.24, 0.5),
                            xycoords="axes fraction", rotation=90, va="center",
                            ha="center", fontsize=9.5, color=INK2)
    axes[0][0].set_xlabel(f"{costs[0][1]} (exact), relative to no edge")
    axes[0][1].set_xlabel(f"{costs[0][1]} (exact), relative to no edge")
    axes[0][0].legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_swarm(after: pd.DataFrame, meta: dict, path: Path) -> None:
    """Both swarms: the ensemble as it is, and with one more circuit in it.

    240 balanced operating states, each solved twice -- once on the frozen grid
    and once with the chosen edge built -- so the benefit is a shift of a whole
    population rather than one arrow on one point.

    The movement is purely vertical, and the figure has to show that: SNSP is a
    property of the dispatch, the dispatch is frozen, and an added edge cannot
    touch it.  So the two swarms share one jitter and each configuration's pair
    is drawn as a vertical segment.
    """
    before = pd.read_csv(base.SNSP / "ensemble_src.csv").set_index("config").sort_index()
    after = after.set_index("config").sort_index()
    a, b = meta["edge"]

    # SNSP does NOT change: the dispatch P is frozen and the share is computed
    # from P alone, so every configuration sits at exactly the same place on the
    # horizontal axis before and after (asserted below).  The scatter is jittered
    # only to separate overlapping points, and the SAME jitter is applied to both
    # swarms, so each configuration's pair is a strictly vertical segment and no
    # apparent sideways movement can be read into the picture.
    assert np.allclose(before["snsp_target"], after["snsp_target"]), \
        "SNSP moved -- the dispatch is supposed to be frozen"
    jit = np.random.default_rng(11).uniform(-1.15, 1.15, len(after))
    x = 100 * after["snsp_target"].to_numpy() + jit

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9))
    panels = (("dev_top5", "peak |f − 50 Hz| on the worst 5 buses (Hz)",
               "frequency deviation"),
              ("rocof_top5", "500 ms RoCoF on the worst 5 buses (Hz/s)",
               "windowed RoCoF"))
    for ax, (key, ylab, title) in zip(axes, panels):
        lev = np.sort(before["snsp_target"].unique())
        y0, y1 = before[key].to_numpy(), after[key].to_numpy()
        nan = np.full_like(x, np.nan)
        ax.plot(np.stack([x, x, nan]).T.ravel(),
                np.stack([y0, y1, nan]).T.ravel(),
                color=MUTED, lw=0.6, alpha=0.45, zorder=1,
                label="the same configuration, before and after")
        for df, y, col, lab in ((before, y0, INK2, "no edge"),
                                (after, y1, C2, f"with edge {a}–{b}")):
            ax.scatter(x, y, s=11, c=col, alpha=0.7, lw=0, zorder=2)
            m = df.groupby("snsp_target")[key].mean()
            ax.plot(100 * lev, m.to_numpy(), color=col, lw=2.0, zorder=3,
                    label=lab)
        # How far left the whole population moved: for each SNSP level, the
        # share at which the *unmodified* mean curve is as low as the modified
        # one is here.  Read sideways, the benefit is in points of SNSP, which
        # is a number an operator can price.
        mb = before.groupby("snsp_target")[key].mean().to_numpy()
        ma = after.groupby("snsp_target")[key].mean().to_numpy()
        shift = np.array([100 * (lev[i] - equivalent_snsp(lev, mb, ma[i]))
                          for i in range(len(lev))])
        mid = shift[(lev >= 0.35) & (lev <= 0.85)]
        r = after[f"r_{key}"]
        ax.set_yscale("log")
        ax.set_xlabel("SNSP of the configuration (%)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{title}   (mean ×{r.mean():.3f}"
                     f", lower in {int((r < 1).sum())}/{len(r)})",
                     color=INK, loc="left", fontsize=10)
        ax.annotate(f"same risk as running\n{np.median(mid):.1f} points lower in"
                    f" SNSP\n(median over 35–85%)", xy=(0.97, 0.06),
                    xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=8.5, color=C2)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_predictor(pr: pd.DataFrame, path: Path, xk: str = "C_H2",
                  xlab: str = "$\\|\\Omega_f\\,\\Pi\\,G\\,B\\|_F$ over the band"
                              "  (the Frobenius / $H_2$ norm)") -> None:
    """The Green's function as a predictor of risk, with no edges anywhere.

    240 operating states of the frozen grid: the horizontal axis is a norm of
    that configuration's Green's function, the vertical is the risk it actually
    incurs.  This is a different question from the rest of the folder, which asks
    whether a *change* in the cost predicts a *change* in the risk.

    Log-log, because both quantities vary over a factor of several across the
    ensemble and the relationship between them is a power law rather than a
    line.  Shaded by SNSP on a single-hue sequential ramp, so one can see whether
    the norm is doing anything more than reading the share off the dispatch.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7))
    panels = (("rocof_top5", "500 ms RoCoF on the worst 5 buses (Hz/s)"),
              ("dev_top5", "peak |f − 50 Hz| on the worst 5 buses (Hz)"))
    for ax, (yk, ylab) in zip(axes, panels):
        sc = ax.scatter(pr[xk], pr[yk], c=100 * pr["snsp"], cmap="Blues",
                        vmin=15, vmax=100, s=22, lw=0)
        lx, ly = np.log10(pr[xk]), np.log10(pr[yk])
        fit = stats.linregress(lx, ly)
        xs = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(10 ** xs, 10 ** (fit.intercept + fit.slope * xs), color=C2,
                lw=2, zorder=3,
                label=f"slope {fit.slope:.2f},  $R^2$ = {fit.rvalue ** 2:.3f}")
        rho = stats.spearmanr(pr[xk], pr[yk]).statistic
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"$\\rho$ = {rho:+.3f}", color=INK, loc="left", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    cb = fig.colorbar(sc, ax=axes, fraction=0.026, pad=0.015)
    cb.set_label("SNSP of the configuration (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="the figures")
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", type=int, default=None)
    args = ap.parse_args()
    FIGS.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    bc = base.load(config=args.config)
    cur = sweep.load_curves(tag=args.tag)
    df, meta = sweep.score(bc, cur)

    fig_pareto(df, "r_C_freq", "r_C_rocof",
               "C_freq, relative to no edge", "C_rocof, relative to no edge",
               FIGS / f"fig1_pareto_freq_rocof{suffix}.png",
               "the two objectives disagree")
    fig_pareto(df, "r_C_freq", "r_C_H2",
               "C_freq, relative to no edge", "H2, relative to no edge",
               FIGS / f"fig2_pareto_freq_h2{suffix}.png",
               "the screen is not the criterion")
    fig_curves(bc, cur, df, FIGS / f"fig3_curves{suffix}.png")
    fig_map(bc, df, FIGS / f"fig4_map{suffix}.png")
    fig_length(df, FIGS / f"fig5_length{suffix}.png")
    nlp = HERE / f"nonlinear{suffix}.csv"
    if nlp.exists():
        nl = pd.read_csv(nlp)
        info = json.loads((HERE / f"nonlinear{suffix}.json").read_text())
        fig_risk_vs_snsp(nl, info, FIGS / f"fig6_risk_vs_snsp{suffix}.png")
        fig_cost_vs_risk(nl, FIGS / f"fig7_cost_vs_risk{suffix}.png")
    ensp = HERE / f"ensemble_edge{suffix}.csv"
    if ensp.exists():
        meta = json.loads((HERE / f"ensemble_edge{suffix}.json").read_text())
        fig_swarm(pd.read_csv(ensp), meta, FIGS / f"fig8_swarm{suffix}.png")
    prp = HERE / f"predictor{suffix}.csv"
    if prp.exists():
        fig_predictor(pd.read_csv(prp),
                      FIGS / f"fig9_frobenius_predictor{suffix}.png")
    print(f"  figures -> {FIGS}")
    for p in sorted(FIGS.glob(f"*{suffix}.png")):
        print(f"    {p.name}")


if __name__ == "__main__":
    main()

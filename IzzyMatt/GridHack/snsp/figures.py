"""Five figures for the SNSP investigation.

Colour comes from the participant kit's validated palette, as in
../local_effects/figures.py: magnitude is always the single-hue sequential ramp,
series identity is the fixed categorical order plus a direct label, and no chart
here has two y-scales.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str((HERE / ".." / "repo" / "grid_TF_Wind"
                        / "participant-kit").resolve()))
import plotstyle as ps                                        # noqa: E402

import configs as C                                           # noqa: E402
import dynamics as dyn                                        # noqa: E402
import grid as G                                              # noqa: E402
import perturbations as PB                                    # noqa: E402

OUT = HERE / "figures"
SEQ = ps.sequential_cmap


def _load(tag: str = "") -> tuple[pd.DataFrame, dict]:
    suffix = f"_{tag}" if tag else ""
    df = pd.read_csv(HERE / f"ensemble{suffix}.csv")
    nodes = dict(np.load(HERE / f"ensemble{suffix}_nodes.npz"))
    return df, nodes


def _ends(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Masks for the bottom and top SNSP strata.

    Keyed on the *target*, not the realised share: a realised SNSP can land at
    0.34999999, and a "< 0.35" mask would then quietly pull half of the 35%
    stratum into the 30% one.
    """
    t = df["snsp_target"]
    return t == t.min(), t == t.max()


def _tidy(ax) -> None:
    ax.grid(True, alpha=0.35)
    ax.set_axisbelow(True)


def fig_cost_vs_snsp(df: pd.DataFrame) -> None:
    """The headline: what the cost does as the non-synchronous share rises."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    x, y = df["snsp"] * 100, df["J"]

    ax = axes[0]
    sc = ax.scatter(x, y, c=df["stored_energy"], cmap=SEQ, s=26,
                    edgecolors=ps.SURFACE, linewidths=0.4, zorder=3)
    lev = df.groupby("snsp_target")["J"].agg(["mean", "min", "max"])
    ax.plot(lev.index * 100, lev["mean"], color=ps.INK_SOFT, lw=1.6, zorder=4,
            label="mean at each SNSP level")
    ax.fill_between(lev.index * 100, lev["min"], lev["max"], color=ps.INK_MUTED,
                    alpha=0.13, zorder=1, label="range within a level")
    ax.set_yscale("log")
    ax.set_xlabel("SNSP of the configuration (%)")
    ax.set_ylabel(r"$J=\sum_{bus,event}\int_0^{T}[\tau^2\dot f^2+f^2]\,dt$"
                  "   (Hz$^2$ s)")
    rho = stats.spearmanr(df["snsp"], df["J"])[0]
    lo0, hi0 = _ends(df)
    ratio = df.loc[hi0, "J"].mean() / df.loc[lo0, "J"].mean()
    ax.set_title(f"The cost rises {ratio:.0f}-fold from 30% to 100% SNSP\n"
                 f"Spearman rho = {rho:.3f}, and the rise accelerates",
                 fontsize=10.5)
    ax.legend(loc="upper left")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("system inertia, sum H S (p.u. s)", fontsize=8)
    cb.outline.set_visible(False)
    _tidy(ax)

    ax = axes[1]
    for k, (key, label) in enumerate((
            ("J_rocof", r"gradient term  $\tau^2\dot f^2$"),
            ("J_freq", r"field term  $f^2$"),
            ("J_coi", "system-wide part (COI)"),
            ("J_local", "local part (J - COI)"))):
        m = df.groupby("snsp_target")[key].mean()
        ax.plot(m.index * 100, m, color=ps.CATEGORICAL[k], lw=2.0, marker="o",
                ms=4, label=label)
        ax.annotate(label, (m.index[-1] * 100, m.iloc[-1]), fontsize=8,
                    color=ps.CATEGORICAL[k], xytext=(6, (7, -7, 7, -7)[k]),
                    textcoords="offset points", va="center")
    ax.set_yscale("log")
    ax.set_xlim(28, 118)
    ax.set_xlabel("SNSP of the configuration (%)")
    ax.set_ylabel("contribution to J (Hz$^2$ s)")
    lo, hi = _ends(df)
    ax.set_title("The two terms do different things\n"
                 f"RoCoF term x{df.loc[hi, 'J_rocof'].mean() / df.loc[lo, 'J_rocof'].mean():.1f}, "
                 f"deviation term x{df.loc[hi, 'J_freq'].mean() / df.loc[lo, 'J_freq'].mean():.0f}"
                 " over the same range", fontsize=10.5)
    _tidy(ax)

    fig.tight_layout()
    fig.savefig(OUT / "fig1_cost_vs_snsp.png", dpi=170)
    plt.close(fig)


#: How many worst buses the grid-code scores keep.  Must match
#: `run_ensemble.TOP_N`'s companion in `dynamics.WORST_N` for the derived
#: per-event values to reduce to the CSV's `*_top5` columns.
TOP_N = 5

#: The two grid-code-shaped scores: CSV column, the `dynamics.Response` field
#: each is a bus-axis reduction of, headline phrase and axis label.  Both are
#: sums over the five worst buses, averaged over the disturbance set --
#: `dynamics.worst_n` -- so they carry the units of the underlying measurement
#: rather than J's Hz^2 s.
#:
#: The deviation rather than the nadir, deliberately.  A nadir is the deepest
#: *downward* excursion, so an event that disconnects demand -- which drives the
#: frequency up -- contributes exactly zero to it however severe it is, and a
#: whole scenario type drops out of the score.  |f| sees both directions.
TOP_BUS = (
    ("dev_top5", "peak_dev", "frequency deviation on the worst 5 buses",
     "peak |f - 50 Hz|, summed over the five worst buses (Hz)"),
    ("rocof_top5", "rocof_500ms", "500 ms RoCoF on the worst 5 buses",
     "500 ms RoCoF, summed over the five worst buses (Hz/s)"),
)


def fig_topbus(df: pd.DataFrame) -> None:
    """Figure 1 with a grid-code score on the vertical axis in place of J.

    J integrates over the whole horizon and every bus at once.  A grid code does
    not: it reads one number per bus per event -- how far the frequency moved,
    how fast it moved over the first 500 ms -- and it is written against the
    buses where that number is worst.  These are the same 240 configurations and
    the same disturbance set as figure 1, scored that way instead, one pane per
    score.

    Linear axes, where figure 1 needed a logarithmic one: J multiplies by ten
    across the sweep and these rise by about three, which a log axis would
    flatten into a straight line and hide the shape of.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    lo, hi = _ends(df)

    for ax, (key, _, phrase, ylabel) in zip(axes, TOP_BUS):
        y = df[key]
        sc = ax.scatter(df["snsp"] * 100, y, c=df["stored_energy"], cmap=SEQ,
                        s=26, edgecolors=ps.SURFACE, linewidths=0.4, zorder=3)
        lev = df.groupby("snsp_target")[key].agg(["mean", "min", "max"])
        ax.plot(lev.index * 100, lev["mean"], color=ps.INK_SOFT, lw=1.6,
                zorder=4, label="mean at each SNSP level")
        ax.fill_between(lev.index * 100, lev["min"], lev["max"],
                        color=ps.INK_MUTED, alpha=0.13, zorder=1,
                        label="range within a level")
        ax.set_xlabel("SNSP of the configuration (%)")
        ax.set_ylabel(ylabel)
        rho = stats.spearmanr(df["snsp"], y)[0]
        ratio = y[hi].mean() / y[lo].mean()
        ax.set_title(f"The {phrase} rises {ratio:.1f}-fold\n"
                     f"from 30% to 100% SNSP; Spearman rho = {rho:.3f}",
                     fontsize=10.5)
        ax.legend(loc="upper left")
        _tidy(ax)

    cb = fig.colorbar(sc, ax=axes, fraction=0.028, pad=0.015)
    cb.set_label("system inertia, sum H S (p.u. s)", fontsize=8)
    cb.outline.set_visible(False)
    fig.savefig(OUT / "fig1c_topbus.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_predictors(df: pd.DataFrame) -> None:
    """SNSP is a proxy.  What is the cost actually following?"""
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))

    def panel(ax, key, label, logx=False):
        v = df[key].to_numpy(float)
        ok = np.isfinite(v)
        sc = ax.scatter(v[ok], df["J"].to_numpy()[ok], c=df["snsp"][ok] * 100,
                        cmap=SEQ, s=24, edgecolors=ps.SURFACE, linewidths=0.4)
        p = np.polyfit(v[ok], np.log(df["J"].to_numpy()[ok]), 2)
        xs = np.linspace(v[ok].min(), v[ok].max(), 100)
        ax.plot(xs, np.exp(np.polyval(p, xs)), color=ps.INK_SOFT, lw=1.4,
                ls="--")
        r2 = 1 - (np.log(df["J"].to_numpy()[ok]) -
                  np.polyval(p, v[ok])).var() / np.log(
                      df["J"].to_numpy()[ok]).var()
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_title(f"R$^2$ = {r2:.3f} on log J", fontsize=10)
        _tidy(ax)
        return sc

    panel(axes[0], "stored_energy", "system inertia, sum H S (p.u. s)")
    panel(axes[1], "reserve", "primary reserve, sum droop gain (p.u./p.u. f)")
    sc = panel(axes[2], "machine_dist_mean",
               "mean hops from a bus to the nearest running machine")
    axes[0].set_ylabel("J (Hz$^2$ s)")
    cb = fig.colorbar(sc, ax=axes[2], fraction=0.045, pad=0.02)
    cb.set_label("SNSP (%)", fontsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("The cost follows inertia and reserve, not the share as such "
                 "-- and how far a bus is from a running machine is as good a "
                 "predictor as either", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig2_predictors.png", dpi=170)
    plt.close(fig)


def fig_traces(g: G.ToyGrid, events: list[PB.Event]) -> None:
    """What the functional is integrating, at two ends of the range."""
    ev = [e for e in events if e.name == "wind_lull@seaboard"][0]
    rng = np.random.default_rng(3)
    lo = C.draw(g, 0.30, rng, 0)
    hi = C.draw(g, 0.95, rng, 1)

    # Run each state twice: with primary response only, and with secondary
    # control added.  Droop is proportional control and *cannot* return the
    # frequency to 50 Hz -- the machines stop opening up the moment the frequency
    # stops falling -- so the offset in the first two panels is the correct
    # behaviour of the system being modelled, not an unfinished transient.  Only
    # the integral action of an AGC removes it, and it takes minutes to do so.
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))
    cum = {}
    for n, (cfg, label) in enumerate(((lo, "SNSP 30%"), (hi, "SNSP 95%"))):
        # With a sourced set this event is a fraction of the coastal wind that
        # is actually running, so its size differs between the two panels --
        # which is the whole point of the set and has to be said on the figure.
        U = ev.vector(cfg.P)[:, None]
        net = cfg.network(g)
        resp, tr = dyn.respond(net, U, trace=True)
        _, tr_agc = dyn.respond(net, U, agc=cfg.participation(g), trace=True)
        t, f = tr["t"], tr["f"][:, 0, :] * 1000
        ax = axes[n]
        ax.axhline(0, color=ps.INK_MUTED, lw=1.0, ls=":", zorder=1)
        ax.fill_between(t, f.min(axis=0), f.max(axis=0),
                        color=ps.CATEGORICAL[0], alpha=0.30, lw=0, zorder=2,
                        label="spread across the 100 buses")
        ax.plot(t, tr["f_coi"][0] * 1000, color=ps.CATEGORICAL[1], lw=2.2,
                zorder=3, label="system frequency, droop only")
        ax.plot(t, tr_agc["f_coi"][0] * 1000, color=ps.CATEGORICAL[2], lw=2.0,
                ls="--", zorder=3,
                label="system frequency, + secondary control")
        off = tr["f_coi"][0][-1] * 1000
        ax.annotate(f"droop offset {off:.0f} mHz\n(proportional control cannot\n"
                    f"remove it; AGC takes ~1 min)",
                    xy=(dyn.T_END, off), xytext=(-8, -34),
                    textcoords="offset points", ha="right", fontsize=7.6,
                    color=ps.INK_SOFT)
        ax.set_title(f"{label}: {cfg.meta['stored_energy']:.0f} p.u. s of "
                     f"inertia, {cfg.meta['n_committed']} machines on\n"
                     f"the same lull removes {abs(U.sum()):.2f} p.u. here",
                     fontsize=10)
        ax.set_xlabel("time since the disturbance (s)")
        ax.set_ylim(-1250, 250)
        ax.legend(loc="lower left")
        _tidy(ax)
        w = dyn.simpson_weights(t.size, t[1] - t[0])
        integ = ((dyn.TAU * tr["rocof"][:, 0, :]) ** 2 + (tr["f"][:, 0, :]) ** 2)
        cum[label] = np.cumsum((integ * w).sum(axis=0))
    axes[0].set_ylabel("frequency deviation from 50 Hz (mHz)")

    ax = axes[2]
    for n, (label, c) in enumerate(cum.items()):
        ax.plot(np.linspace(0, dyn.T_END, c.size), c, lw=2.2,
                color=ps.CATEGORICAL[n], label=label)
        ax.annotate(label, (dyn.T_END, c[-1]), xytext=(-4, 4),
                    textcoords="offset points", ha="right", fontsize=8,
                    color=ps.CATEGORICAL[n])
    ax.set_xlabel("upper limit of the integral (s)")
    ax.set_ylabel("cost accumulated by that time (Hz$^2$ s)")
    ax.set_title("What the cost accumulates\n"
                 "(a lull over the whole western seaboard)", fontsize=10)
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_traces.png", dpi=170)
    plt.close(fig)


def fig_maps(g: G.ToyGrid, df: pd.DataFrame, nodes: dict) -> None:
    """Where the cost lands on the map, and how that changes with SNSP."""
    by_bus, snsp = nodes["by_bus"], nodes["snsp"]
    lo = by_bus[snsp < 0.35].mean(axis=0)
    hi = by_bus[snsp > 0.97].mean(axis=0)
    x = np.array([c for _, c in g.coords], float)
    y = np.array([-r for r, _ in g.coords], float)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.8))
    # Each panel carries its own scale.  A common one would be honest about the
    # ninefold rise and useless about everything else -- the whole point of these
    # maps is the *pattern* within each state, and the third panel is where the
    # two are compared.
    for ax, v, title in (
            (axes[0], lo, f"SNSP 30%: cost per bus  "
                          f"({lo.max() / lo.min():.1f}x across the grid)"),
            (axes[1], hi, f"SNSP 100%: cost per bus  "
                          f"({hi.max() / hi.min():.2f}x across the grid)"),
            (axes[2], hi / lo, "how much each bus lost: ratio 100% / 30%")):
        for i, j in g.edges:
            ax.plot([x[i], x[j]], [y[i], y[j]], color="#dcdbd6", lw=0.7,
                    zorder=1)
        sc = ax.scatter(x, y, c=v, cmap=SEQ, s=90, zorder=3,
                        edgecolors=ps.SURFACE, linewidths=0.6)
        gen = np.isin(g.role.astype(str), ("wind", "hvdc"))
        ax.scatter(x[gen], y[gen], s=180, facecolors="none",
                   edgecolors=ps.CATEGORICAL[0], linewidths=0.9, zorder=2)
        syn = g.role == "sync"
        ax.scatter(x[syn], y[syn], s=200, marker="s", facecolors="none",
                   edgecolors=ps.CATEGORICAL[1], linewidths=1.2, zorder=2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.outline.set_visible(False)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].annotate("circles: converter buses\nsquares: synchronous stations",
                     xy=(0.02, -0.06), xycoords="axes fraction", fontsize=8,
                     color=ps.INK_SOFT, va="top")
    # Data-driven, so the caption cannot go stale when the disturbance set
    # changes underneath it.
    r = hi / lo
    gained = ", ".join(str(g.name[i]) for i in np.argsort(-r)[:3])
    axes[2].annotate(f"the buses that lose most sit next to a machine\n"
                     f"that stops running: {gained}",
                     xy=(0.02, -0.06), xycoords="axes fraction", fontsize=8,
                     color=ps.INK_SOFT, va="top")
    fig.suptitle(f"Exposure levels up: {lo.max() / lo.min():.1f}x between the "
                 f"best and worst bus at 30% SNSP, {hi.max() / hi.min():.2f}x "
                 f"at 100% -- by then every bus pays roughly what the worst "
                 f"used to", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig4_maps.png", dpi=170)
    plt.close(fig)


def fig_ffr(df: pd.DataFrame, ffr: pd.DataFrame) -> None:
    """How much of the correlation is reserve, and how much is inertia?"""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axes[0]
    for k, (d, label) in enumerate(((df, "converters give no response"),
                                    (ffr, "converters give 5% droop in 1 s"))):
        m = d.groupby("snsp_target")["J"].agg(["mean", "min", "max"])
        ax.plot(m.index * 100, m["mean"], color=ps.CATEGORICAL[k], lw=2.2,
                marker="o", ms=4, label=label)
        ax.fill_between(m.index * 100, m["min"], m["max"],
                        color=ps.CATEGORICAL[k], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("SNSP of the configuration (%)")
    ax.set_ylabel("J (Hz$^2$ s)")
    ax.legend(loc="upper left")
    lo, hi = _ends(df)
    lo2, hi2 = _ends(ffr)
    ax.set_title("Fast frequency response removes most of the rise\n"
                 f"30%->100%: {df.loc[hi, 'J'].mean() / df.loc[lo, 'J'].mean():.0f}x "
                 f"without it, "
                 f"{ffr.loc[hi2, 'J'].mean() / ffr.loc[lo2, 'J'].mean():.1f}x with it",
                 fontsize=10.5)
    _tidy(ax)

    ax = axes[1]
    keys = ["J", "J_rocof", "J_freq"]
    labels = ["J total", "gradient (RoCoF)", "field (deviation)"]
    w = 0.36
    pos = np.arange(len(keys))
    for k, (d, m_lo, m_hi, label) in enumerate((
            (df, lo, hi, "no converter response"),
            (ffr, lo2, hi2, "with fast frequency response"))):
        vals = [d.loc[m_hi, key].mean() / d.loc[m_lo, key].mean() for key in keys]
        ax.bar(pos + (k - 0.5) * w, vals, width=w * 0.92,
               color=ps.CATEGORICAL[k], label=label, zorder=3)
        for p, v in zip(pos + (k - 0.5) * w, vals):
            ax.annotate(f"{v:.1f}x", (p, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=ps.INK_SOFT)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("cost at 100% SNSP / cost at 30% SNSP")
    ax.set_title("What is left is the inertia part\n"
                 "the RoCoF term barely moves either way", fontsize=10.5)
    ax.legend(loc="upper left")
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_ffr.png", dpi=170)
    plt.close(fig)


#: The score family compared in fig6: the integral functional and its two terms,
#: the grid-code-shaped "n worst buses" scores, and the system-wide values a
#: single-machine-equivalent model would report.
METRICS = (
    ("J", "J  (both terms)"),
    ("J_rocof", "J, gradient term"),
    ("J_freq", "J, field term"),
    ("rocof_top5", "500 ms RoCoF, worst 5 buses"),
    ("nadir_top5", "nadir, worst 5 buses"),
    ("coi_rocof_500ms", "500 ms RoCoF, system-wide"),
    ("coi_nadir_mean", "nadir, system-wide"),
)


def fig_metrics(df: pd.DataFrame) -> None:
    """Different cost functions, same 240 configurations: do they agree?"""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    ax = axes[0]
    for k, (key, label) in enumerate(METRICS):
        m = df.groupby("snsp_target")[key].mean()
        m = m / m.iloc[0]
        ax.plot(m.index * 100, m, color=ps.CATEGORICAL[k], lw=2.0, marker="o",
                ms=3.5, label=f"{label}  (x{m.iloc[-1]:.1f})")
    ax.set_yscale("log")
    ax.set_xlabel("SNSP of the configuration (%)")
    ax.set_ylabel("score, relative to its own value at 30% SNSP")
    ax.legend(loc="upper left", fontsize=8)
    grow = [df.groupby("snsp_target")[k].mean().iloc[-1]
            / df.groupby("snsp_target")[k].mean().iloc[0] for k, _ in METRICS]
    ax.set_title(f"Every score rises with SNSP -- by factors from "
                 f"{min(grow):.1f} to {max(grow):.0f}\n"
                 "the choice of cost function sets the magnitude, not the sign",
                 fontsize=10.5)
    _tidy(ax)

    # Do they rank the 240 configurations the same way?  Spearman, because these
    # scores are on different scales and only the ordering is comparable.
    ax = axes[1]
    keys = [k for k, _ in METRICS]
    R = np.array([[stats.spearmanr(df[a], df[b])[0] for b in keys]
                  for a in keys])
    im = ax.imshow(R, cmap=SEQ, vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(keys)))
    short = [lbl.replace("  (both terms)", "").replace(" buses", "")
             for _, lbl in METRICS]
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.annotate(f"{R[i, j]:.2f}", (j, i), ha="center", va="center",
                        fontsize=7.5,
                        color=ps.SURFACE if R[i, j] > 0.8 else ps.INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Spearman rank correlation across the 240 configurations",
                 fontsize=8)
    cb.outline.set_visible(False)
    off = R[~np.eye(len(keys), dtype=bool)]
    ax.set_title(f"But they rank configurations the same way\n"
                 f"every pair agrees at rho >= {off.min():.2f}", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_metrics.png", dpi=170)
    plt.close(fig)


def fig_secondary(df: pd.DataFrame, agc: pd.DataFrame) -> None:
    """What secondary control changes, and what it does not."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axes[0]
    for k, (d, label) in enumerate(((df, "primary response only (droop)"),
                                    (agc, "+ secondary control (AGC)"))):
        m = d.groupby("snsp_target")["J"].agg(["mean", "min", "max"])
        ax.plot(m.index * 100, m["mean"], color=ps.CATEGORICAL[k], lw=2.2,
                marker="o", ms=4, label=label)
        ax.fill_between(m.index * 100, m["min"], m["max"],
                        color=ps.CATEGORICAL[k], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("SNSP of the configuration (%)")
    ax.set_ylabel("J (Hz$^2$ s)")
    ax.legend(loc="upper left")
    lo, hi = _ends(df)
    lo2, hi2 = _ends(agc)
    cut95 = 1 - (agc.groupby("snsp_target")["J"].mean().loc[0.95]
                 / df.groupby("snsp_target")["J"].mean().loc[0.95])
    ax.set_title("Secondary control cuts the cost wherever a machine runs\n"
                 f"-{cut95 * 100:.0f}% at 95% SNSP; nothing at 100%, where "
                 f"none is synchronised", fontsize=10.5)
    _tidy(ax)

    # Per level, not at the top stratum: at 100% SNSP nothing is synchronised,
    # so there is no secondary control either and every difference is exactly
    # zero.  That collapse is the interesting part, and a bar chart at the top
    # stratum alone would show only the zero.
    ax = axes[1]
    for k, (key, label) in enumerate((("J", "J"),
                                      ("J_freq", "J, field term"),
                                      ("J_rocof", "J, gradient term"),
                                      ("nadir_top5", "nadir, worst 5 buses"),
                                      ("rocof_top5", "500 ms RoCoF, worst 5"))):
        a = df.groupby("snsp_target")[key].mean()
        b = agc.groupby("snsp_target")[key].mean()
        v = (b / a - 1) * 100
        ax.plot(v.index * 100, v, color=ps.CATEGORICAL[k], lw=2.0, marker="o",
                ms=3.5, label=label)
    ax.axhline(0, color=ps.INK_MUTED, lw=1.0, zorder=1)
    ax.set_xlabel("SNSP of the configuration (%)")
    ax.set_ylabel("change from adding secondary control (%)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("It works on the deviation, not on RoCoF\n"
                 "integral action arrives far too late to change df/dt",
                 fontsize=10.5)
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_secondary.png", dpi=170)
    plt.close(fig)


#: The 2x2: how the hazard is modelled, against whether a minimum-units floor
#: holds the inertia up.  Tags are the `--tag` values run_ensemble writes.
CELLS = (("", "fixed hazard, free commitment"),
         ("scaled", "scaled hazard, free commitment"),
         ("muon", "fixed hazard, MUON floor"),
         ("scaled_muon", "scaled hazard, MUON floor"))


def fig_muon() -> dict[str, pd.DataFrame]:
    """Does SNSP still matter once an inertia floor is in place?

    Four ensembles on the identical grid.  Along one axis, whether the
    disturbance set is a fixed number of megawatts or a fixed fraction of what
    is running; along the other, whether a minimum-units constraint keeps four
    machines synchronised whatever the dispatch wants.  The second axis pins
    system inertia; the first decides whether SNSP has any channel left through
    which to act.
    """
    have = {}
    for tag, label in CELLS:
        try:
            have[tag] = _load(tag)[0]
        except FileNotFoundError:
            print(f"missing ensemble for '{tag or 'baseline'}' -- skipping fig8")
            return {}

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))

    ax = axes[0]
    for k, (tag, lbl) in enumerate((("", "free commitment"),
                                    ("muon", "MUON floor: 4 machines on"))):
        m = have[tag].groupby("snsp_target")["stored_energy"].mean()
        ax.plot(m.index * 100, m, color=ps.CATEGORICAL[k], lw=2.2, marker="o",
                ms=4, label=lbl)
    ax.set_xlabel("SNSP of the configuration (%)")
    ax.set_ylabel("system inertia, sum H S (p.u. s)")
    ax.set_ylim(0, None)
    ax.legend(loc="lower left")
    ax.set_title("The control: MUON pins the inertia\n"
                 "so anything left is not an inertia effect", fontsize=10.5)
    _tidy(ax)

    ax = axes[1]
    for k, (tag, lbl) in enumerate(CELLS):
        m = have[tag].groupby("snsp_target")["J"].mean()
        ax.plot(m.index * 100, m / m.iloc[0], color=ps.CATEGORICAL[k], lw=2.2,
                marker="o", ms=4, ls="-" if "scaled" in tag else "--",
                label=f"{lbl}  (x{m.iloc[-1] / m.iloc[0]:.0f})")
    ax.set_yscale("log")
    ax.set_xlabel("SNSP of the configuration (%)")
    ax.set_ylabel("J, relative to its own value at 30% SNSP")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("An inertia floor removes most of the rise -- but only\n"
                 "if the disturbance does not grow with the wind",
                 fontsize=10.5)
    _tidy(ax)

    # Absolute exceedances, not ratios: several of these are exactly zero at 30%
    # SNSP, and a ratio against zero is undefined rather than large.  These are
    # the readings a limit is actually written against.
    ax = axes[2]
    keys = ["share_rocof_over_code", "share_ufls"]
    labels = ["500 ms RoCoF past\n1 Hz/s (grid code)",
              "frequency past the first\nUFLS stage (-0.8 Hz)"]
    width = 0.2
    pos = np.arange(len(keys))
    for k, (tag, lbl) in enumerate(CELLS):
        d = have[tag]
        hi = d["snsp_target"] == d["snsp_target"].max()
        vals = [100 * d.loc[hi, key].mean() for key in keys]
        ax.bar(pos + (k - 1.5) * width, vals, width=width * 0.92,
               color=ps.CATEGORICAL[k], label=lbl, zorder=3)
        for p, v in zip(pos + (k - 1.5) * width, vals):
            ax.annotate(f"{v:.0f}", (p, v), xytext=(0, 3), fontsize=7,
                        textcoords="offset points", ha="center",
                        color=ps.INK_SOFT)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("share of bus-events at SNSP 100% (%)")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("Exceedances at 100% SNSP\n"
                 "under MUON the fixed hazard sees nothing at all",
                 fontsize=10.5)
    _tidy(ax)

    fig.tight_layout()
    fig.savefig(OUT / "fig8_muon.png", dpi=170)
    plt.close(fig)
    return have


#: Kinds of disturbance, in a fixed colour order.  Six slots, never cycled.
KINDS = ("infeed_loss", "wind_drop", "wind_lull", "load_step", "load_reject",
         "diffuse_load")
KIND_LABEL = {"infeed_loss": "unit / interconnector trip",
              "wind_drop": "gust front on one farm",
              "wind_lull": "lull over the whole seaboard",
              "load_step": "demand block switches in",
              "load_reject": "demand disconnects",
              "diffuse_load": "diffuse imbalance, no location"}


def _per_event(df: pd.DataFrame, nodes: dict, events: list[PB.Event]):
    """One row per (configuration, event): the long form of the ensemble.

    `by_event` is already summed over buses but not over events, so this is the
    ensemble with the event axis kept -- 240 x 20 points instead of 240.
    """
    J = nodes["by_event"]
    size = nodes["sizes"]
    n_cfg, n_ev = J.shape
    d = pd.DataFrame({
        "snsp": np.repeat(df["snsp"].to_numpy(), n_ev),
        "snsp_target": np.repeat(df["snsp_target"].to_numpy(), n_ev),
        "stored_energy": np.repeat(df["stored_energy"].to_numpy(), n_ev),
        "reserve": np.repeat(df["reserve"].to_numpy(), n_ev),
        "machine_dist_mean": np.repeat(df["machine_dist_mean"].to_numpy(), n_ev),
        "config": np.repeat(df["config"].to_numpy(), n_ev),
        "event": np.tile(np.arange(n_ev), n_cfg),
        "name": np.tile([e.name for e in events], n_cfg),
        "kind": np.tile([e.kind for e in events], n_cfg),
        "J": J.ravel(),
        "size": size.ravel(),
    })
    # The grid-code scores, reduced here rather than read: the ensemble saves
    # the raw (config, bus, event) readings, and every score in that family is a
    # sum over the worst few buses of one of them.  Deriving it at draw time is
    # what makes a different n, or a different field, a reprocess.
    for key, field, _, _ in TOP_BUS:
        raw = nodes.get(field)
        if raw is not None:
            d[key] = dyn.worst_n_by_event(np.abs(raw), TOP_N, axis=1).ravel()
    # Kept for the contrast fig1d draws: what the same score would read if it
    # watched only the downward excursion.
    if "nadir" in nodes:
        d["nadir_top5"] = dyn.worst_n_by_event(nodes["nadir"], TOP_N,
                                               axis=1).ravel()
    return d


def _kind_colour(kind: str) -> str:
    return ps.CATEGORICAL[KINDS.index(kind)] if kind in KINDS else ps.INK_MUTED


def fig_events(df: pd.DataFrame, nodes: dict, events: list[PB.Event]) -> None:
    """Figure 1 without the sum over events: every scenario on its own.

    Summing over the disturbance set is what makes a configuration a single
    point, and it hides two things: how much of the trend each kind of event
    contributes, and whether any individual scenario departs from it.  Here each
    (configuration, event) pair is one point -- 4800 of them.
    """
    d = _per_event(df, nodes, events)
    live = d["size"] > 1e-6
    dead = int((~live).sum())
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0))

    ax = axes[0]
    for kind in KINDS:
        m = live & (d["kind"] == kind)
        if not m.any():
            continue
        ax.scatter(d.loc[m, "snsp"] * 100, d.loc[m, "J"], s=7, alpha=0.32,
                   color=_kind_colour(kind), linewidths=0,
                   label=KIND_LABEL.get(kind, kind))
    med = d[live].groupby("snsp_target")["J"].median()
    ax.plot(med.index * 100, med, color=ps.INK, lw=1.8, zorder=5,
            label="median over all scenarios")
    ax.set_yscale("log")
    ax.set_xlabel("SNSP of the configuration (%)")
    ax.set_ylabel("J for one configuration and one event (Hz$^2$ s)")
    leg = ax.legend(loc="upper left", fontsize=7.5, markerscale=2.2)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax.set_title(f"Every configuration x every event: {int(live.sum())} points\n"
                 f"({dead} more are exactly zero: the unit was not running)",
                 fontsize=10.5)
    _tidy(ax)

    # Divide the hazard out exactly.  The response is linear, so a given
    # configuration and event shape gives J proportional to the square of the
    # imbalance: J / dP^2 is therefore pure vulnerability, with the size of the
    # disturbance removed rather than merely controlled for.
    ax = axes[1]
    v = d["J"] / d["size"] ** 2
    for kind in KINDS:
        m = live & (d["kind"] == kind)
        if not m.any():
            continue
        ax.scatter(d.loc[m, "snsp"] * 100, v[m], s=7, alpha=0.32,
                   color=_kind_colour(kind), linewidths=0)
    medv = v[live].groupby(d.loc[live, "snsp_target"]).median()
    ax.plot(medv.index * 100, medv, color=ps.INK, lw=1.8, zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("SNSP of the configuration (%)")
    ax.set_ylabel(r"$J/\Delta P^2$   (Hz$^2$ s per p.u.$^2$)")
    ax.set_title(f"Hazard divided out: vulnerability alone\n"
                 f"the median rises {medv.iloc[-1] / medv.iloc[0]:.1f}x across "
                 f"the sweep", fontsize=10.5)
    _tidy(ax)

    # Outliers.  Measured on the hazard-normalised cost, not on J: a wind_drop
    # on an inland farm varies by two orders of magnitude across the ensemble
    # simply because that farm's output does, and those are hazard-size
    # outliers, not response outliers.  Dividing by dP^2 first leaves only
    # configurations that responded unusually to a given scenario.
    ax = axes[2]
    logv = np.log10(v.where(live))
    base = logv.groupby([d["event"], d["snsp_target"]]).transform("median")
    resid = logv - base
    for kind in KINDS:
        m = live & (d["kind"] == kind)
        if not m.any():
            continue
        ax.scatter(d.loc[m, "snsp"] * 100, resid[m], s=7, alpha=0.32,
                   color=_kind_colour(kind), linewidths=0)
    ax.axhline(0, color=ps.INK_MUTED, lw=1.0)
    seen, n = set(), 0
    for i in resid.abs().sort_values(ascending=False).index:
        if d.loc[i, "name"] in seen:
            continue
        seen.add(d.loc[i, "name"])
        left = d.loc[i, "snsp"] < 0.7
        ax.annotate(f"{d.loc[i, 'name']}  (cfg {d.loc[i, 'config']})",
                    (d.loc[i, "snsp"] * 100, resid[i]),
                    xytext=(6 if left else -6, (-9, 9, -9)[n]),
                    textcoords="offset points",
                    fontsize=7, color=ps.INK_SOFT, va="center",
                    ha="left" if left else "right")
        n += 1
        if n == 3:
            break
    lim = float(np.nanmax(np.abs(resid))) * 1.2
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("SNSP of the configuration (%)")
    ax.set_ylabel(r"log$_{10}$ of $J/\Delta P^2$ minus its median"
                  "\nfor that event and SNSP level")
    q = np.nanpercentile(resid, [1, 99])
    ax.set_title(f"Response outliers, hazard removed\n"
                 f"98% lie within {10 ** q[1]:.2f}x / "
                 f"{1 / 10 ** q[0]:.2f}x of their own scenario",
                 fontsize=10.5)
    _tidy(ax)

    fig.tight_layout()
    fig.savefig(OUT / "fig1b_events.png", dpi=170)
    plt.close(fig)


def fig_topbus_events(df: pd.DataFrame, nodes: dict,
                      events: list[PB.Event]) -> None:
    """Figure 1b's grid-code counterpart: the same scores, one event at a time.

    `nadir_top5` and `rocof_top5` are averages over the disturbance set, and an
    average over twenty scenarios of very different sizes is a weak summary of
    what any one of them did.  Here the event axis is kept: each point is one
    configuration meeting one disturbance, scored over that event's own five
    worst buses.

    Linear, where figure 1b needed a logarithmic axis, and for a reason worth
    stating: J is quadratic in the imbalance, so a disturbance set spanning two
    orders of magnitude in size spreads J over four and only a log axis can hold
    it.  A deviation and a RoCoF are *linear* in the imbalance, so the same set
    spreads them over about one.

    This is also the figure that argues for |f| over the nadir.  The
    demand-disconnection scenarios drive the frequency up, so they have no
    downward excursion at all: a nadir-based score reads exactly zero for them
    however severe they are, and the count on the left-hand pane is how many
    points that silently discards.
    """
    d = _per_event(df, nodes, events)
    live = d["size"] > 1e-6
    dead = int((~live).sum())
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))

    for ax, (key, _, phrase, ylabel) in zip(axes, TOP_BUS):
        for kind in KINDS:
            m = live & (d["kind"] == kind)
            if not m.any():
                continue
            ax.scatter(d.loc[m, "snsp"] * 100, d.loc[m, key], s=7, alpha=0.32,
                       color=_kind_colour(kind), linewidths=0,
                       label=KIND_LABEL.get(kind, kind))
        med = d[live].groupby("snsp_target")[key].median()
        ax.plot(med.index * 100, med, color=ps.INK, lw=1.8, zorder=5,
                label="median over all scenarios")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("SNSP of the configuration (%)")
        ax.set_ylabel(ylabel.replace(" (", ",\nfor one configuration and "
                                     "one event ("))
        ax.set_title(f"The {phrase}, scenario by scenario\n"
                     f"the median rises {med.iloc[-1] / med.iloc[0]:.1f}x "
                     f"across the sweep", fontsize=10.5)
        _tidy(ax)

    # The zero row on the left-hand pane is a real reading, not missing data,
    # and it is the one thing in this figure a reader is likely to misread.
    if "nadir_top5" in d:
        blind = int((live & (d["nadir_top5"] < 1e-3)).sum())
        axes[0].annotate(f"the same score built on the nadir instead of |f|\n"
                         f"reads exactly zero for {blind} of these points:\n"
                         f"the events that push the frequency up",
                         (0.985, 0.03), xycoords="axes fraction", ha="right",
                         va="bottom", fontsize=7.5, color=ps.INK_SOFT,
                         bbox=dict(boxstyle="round,pad=0.4", fc=ps.SURFACE,
                                   ec=ps.INK_MUTED, lw=0.5, alpha=0.88))

    leg = axes[0].legend(loc="upper left", fontsize=7.5, markerscale=2.2)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    fig.tight_layout()
    fig.text(0.5, -0.015,
             f"{int(live.sum())} points per pane; {dead} more are dropped "
             f"entirely -- the unit that event trips was not running in that "
             f"configuration.",
             ha="center", va="top", fontsize=8, color=ps.INK_SOFT)
    fig.savefig(OUT / "fig1d_topbus_events.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_events_predictors(df: pd.DataFrame, nodes: dict,
                          events: list[PB.Event]) -> None:
    """Figure 2 without the sum over events: which scenarios follow which axis.

    The predictors are properties of a configuration, so each configuration's
    twenty events stack in a vertical column.  The height of that column is the
    thing figure 2 could not show: how differently the same grid state treats
    one disturbance and another.
    """
    d = _per_event(df, nodes, events)
    live = d["size"] > 1e-6
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0))

    for ax, key, label in (
            (axes[0], "stored_energy", "system inertia, sum H S (p.u. s)"),
            (axes[1], "reserve", "primary reserve, sum droop gain"),
            (axes[2], "machine_dist_mean",
             "mean hops to the nearest running machine")):
        for kind in KINDS:
            m = live & (d["kind"] == kind) & np.isfinite(d[key])
            if not m.any():
                continue
            ax.scatter(d.loc[m, key], d.loc[m, "J"], s=7, alpha=0.30,
                       color=_kind_colour(kind), linewidths=0,
                       label=KIND_LABEL.get(kind, kind))
            # One trend line per kind, so a kind that follows the axis and one
            # that ignores it are visibly different rather than averaged.
            x, y = d.loc[m, key].to_numpy(), np.log(d.loc[m, "J"].to_numpy())
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 20 and x[ok].std() > 1e-9:
                p = np.polyfit(x[ok], y[ok], 2)
                xs = np.linspace(x[ok].min(), x[ok].max(), 80)
                ax.plot(xs, np.exp(np.polyval(p, xs)),
                        color=_kind_colour(kind), lw=1.6, zorder=4)
        ax.set_yscale("log")
        ax.set_xlabel(label)
        _tidy(ax)
    axes[0].set_ylabel("J for one configuration and one event (Hz$^2$ s)")
    leg = axes[0].legend(loc="lower left", fontsize=7.5, markerscale=2.2)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    fig.suptitle("The same predictors, one point per configuration and event: "
                 "the vertical spread is how differently one grid state treats "
                 "different disturbances", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig2b_predictors_events.png", dpi=170)
    plt.close(fig)


#: Figures 1-7 are drawn from one *family* of runs: a disturbance set plus its
#: two control variants.  The headline family is `sourced`, where every event
#: scales with whatever causes it; the `fixed` family holds every event at a
#: constant size and is kept alongside as the vulnerability-only control.
FAMILIES = {
    "sourced": dict(base="src", ffr="src_ffr", agc="src_agc",
                    events="sourced", out="figures"),
    "fixed": dict(base="", ffr="ffr", agc="agc",
                  events="fixed", out="figures/fixed_hazard"),
}


def fig_clusters() -> None:
    """The spectral partition, and how firmly the eigengap chose it."""
    import clusters as CL
    g = G.load()
    part = CL.load()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    ax = axes[0]
    ks = sorted(part.gaps)
    vals = [part.gaps[k] for k in ks]
    top = max(vals)
    cols = [ps.CATEGORICAL[0] if v == top else ps.INK_MUTED for v in vals]
    ax.bar(ks, vals, color=cols, width=0.68)
    runner = sorted(vals, reverse=True)[1]
    ax.axhline(runner, color=ps.INK_SOFT, lw=1.0, ls="--")
    ax.annotate("the runner-up gap", (ks[-1], runner), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=8,
                color=ps.INK_SOFT)
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel(r"eigengap  $\lambda_{k+1}-\lambda_k$")
    ax.set_xticks(ks)
    ax.set_title(f"The eigengap picks k = {part.k} -- by {top / runner:.2f}x\n"
                 "a preference, not a partition hiding in the graph",
                 fontsize=10.5)
    _tidy(ax)

    ax = axes[1]
    x = np.array([c for _, c in g.coords], float)
    y = np.array([-r for r, _ in g.coords], float)
    for i, j in g.edges:
        same = part.label[i] == part.label[j]
        ax.plot([x[i], x[j]], [y[i], y[j]],
                color="#c9c8c3" if same else ps.INK_SOFT,
                lw=0.7 if same else 1.6, zorder=1,
                alpha=1.0 if same else 0.55)
    for c in range(part.k):
        m = part.label == c
        ax.scatter(x[m], y[m], s=95, color=ps.CATEGORICAL[c], zorder=3,
                   edgecolors=ps.SURFACE, linewidths=0.6,
                   label=f"region {c}  ({m.sum()} buses)")
    roles = g.role.astype(str)
    ns = np.isin(roles, ("wind", "hvdc"))
    ax.scatter(x[ns], y[ns], s=210, facecolors="none", edgecolors=ps.INK,
               linewidths=0.8, zorder=4)
    syn = roles == "sync"
    ax.scatter(x[syn], y[syn], s=230, marker="s", facecolors="none",
               edgecolors=ps.INK, linewidths=1.5, zorder=4)
    ax.annotate("circles: wind and HVDC     squares: synchronous stations\n"
                "heavy lines: coupling cut by the partition",
                (0.5, -0.03), xycoords="axes fraction", ha="center", va="top",
                fontsize=7.5, color=ps.INK_SOFT)
    cut = part.cut_weight(g) / np.triu(g.Kij, 1).sum() * 100
    ax.set_title(f"Four regions, cutting {cut:.0f}% of the coupling\n"
                 "wind-heavy west, demand-heavy east", fontsize=10.5)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
              frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig9_clusters.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _band(ax, obs, pred, colour, label):
    """Draw the 98% multiplicative residual band around a predictor."""
    res = np.log10(obs) - np.log10(pred)
    lo, hi = np.percentile(res, [1, 99])
    lim = np.array([min(obs.min(), pred.min()) * 0.93,
                    max(obs.max(), pred.max()) * 1.07])
    ax.plot(lim, lim, color=ps.INK, lw=1.2, zorder=4)
    ax.fill_between(lim, lim * 10 ** lo, lim * 10 ** hi, color=colour,
                    alpha=0.13, zorder=1)
    return lim, 10 ** (hi - lo)


def fig_predictor() -> None:
    """Does a cluster-built predictor beat SNSP -- and is it the clustering?"""
    pr = pd.read_csv(HERE / "predictor_predictions.csv")
    res = pd.read_csv(HERE / "predictor.csv").set_index("family")
    ctl = pd.read_csv(HERE / "predictor_controls.csv")
    bas = pd.read_csv(HERE / "predictor_bases.csv")
    best = str(pr["best_family"].iloc[0])
    te = pr["in_test"].to_numpy(bool)
    obs = pr["risk"].to_numpy(float)

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))

    # -- 1. the baseline, as figure 1c draws it -------------------------
    ax = axes[0]
    ax.scatter(pr.loc[~te, "snsp"] * 100, obs[~te], s=20, alpha=0.45,
               color=ps.INK_MUTED, linewidths=0, label="training")
    ax.scatter(pr.loc[te, "snsp"] * 100, obs[te], s=30,
               color=ps.CATEGORICAL[1], linewidths=0, label="held out")
    lev = pr.groupby("snsp_target")["risk"].mean()
    ax.plot(lev.index * 100, lev, color=ps.INK, lw=1.8, zorder=5,
            label="mean at each level")
    ax.set_xlabel("SNSP of the configuration (%)")
    ax.set_ylabel("risk: 500 ms RoCoF, worst 5 buses (Hz/s)")
    r = res.loc["snsp"]
    ax.set_title(f"What SNSP predicts\n"
                 f"held-out R$^2$ = {r['test_r2']:.3f}, "
                 f"98% of points within {r['test_band98']:.2f}x",
                 fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8)
    _tidy(ax)

    # -- 2. observed against predicted, both models ---------------------
    ax = axes[1]
    for name, colour, lbl in (("snsp", ps.INK_MUTED, "SNSP alone"),
                              (best, ps.CATEGORICAL[0], f"{best} model")):
        pred = pr[f"pred_{name}"].to_numpy(float)
        lim, band = _band(ax, obs[te], pred[te], colour, lbl)
        ax.scatter(pred[te], obs[te], s=30, color=colour, linewidths=0,
                   alpha=0.85, zorder=3,
                   label=f"{lbl}: 98% of points within {band:.2f}x")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("predicted risk (Hz/s)")
    ax.set_ylabel("observed risk (Hz/s)")
    b = res.loc[best]
    ax.set_title(f"The {best} model on the 60 held-out configurations\n"
                 f"R$^2$ {r['test_r2']:.3f} -> {b['test_r2']:.3f}, "
                 f"band {r['test_band98']:.2f}x -> {b['test_band98']:.2f}x, "
                 f"outliers {int(r['test_out'])} -> {int(b['test_out'])}",
                 fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8)
    _tidy(ax)

    # -- 3. the controls: is it the clustering? -------------------------
    ax = axes[2]
    rows, cols, labels = [], [], []
    rnd = ctl[ctl["partition"] == "random, same sizes"]["test_r2"].to_numpy()
    spec = float(ctl.iloc[0]["test_r2"])
    quad = ctl[ctl["partition"] == "geographic quarters"]["test_r2"]
    rb = bas[bas["basis"] == "random orthonormal"]["test_r2"].to_numpy()
    for lbl, v, c in (
            ("swing:\nlog dP, log H", float(res.loc["swing", "test_r2"]),
             ps.CATEGORICAL[2]),
            ("spectral,\nk = 4", spec, ps.CATEGORICAL[0]),
            ("geographic\nquarters", float(quad.iloc[0]) if len(quad) else np.nan,
             ps.CATEGORICAL[0]),
            ("random\npartitions", rnd.mean() if len(rnd) else np.nan,
             ps.INK_MUTED),
            ("slowest 10\nmodes", float(bas.iloc[0]["test_r2"]),
             ps.CATEGORICAL[0]),
            ("fastest 10\nmodes", float(bas.iloc[1]["test_r2"]),
             ps.CATEGORICAL[3]),
            ("random\nrotations", rb.mean() if len(rb) else np.nan,
             ps.INK_MUTED)):
        labels.append(lbl), rows.append(v), cols.append(c)
    xs = np.arange(len(rows))
    ax.bar(xs, rows, color=cols, width=0.66, zorder=2)
    if len(rnd):
        ax.plot([3, 3], [rnd.min(), rnd.max()], color=ps.INK, lw=1.4, zorder=4)
    if len(rb):
        ax.plot([6, 6], [rb.min(), rb.max()], color=ps.INK, lw=1.4, zorder=4)
    # The first three bars are the same 20-column partition construction on
    # three different partitions; the last three are the 31-column modal one on
    # three different bases.  Comparisons are within a group, not across, so the
    # divider is not decoration.
    ax.axvline(0.5, color=ps.INK_MUTED, lw=0.9, ls=":", zorder=1)
    ax.axvline(3.5, color=ps.INK_MUTED, lw=0.9, ls=":", zorder=1)
    for xc, txt in ((0.0, "physics\n(2 cols)"),
                    (2.0, "per-region features\n(20 columns)"),
                    (5.0, "projection onto modes\n(31 columns)")):
        ax.annotate(txt, (xc, 0.998), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7.5, color=ps.INK_SOFT)
    snsp_r2 = float(res.loc["snsp", "test_r2"])
    ax.axhline(snsp_r2, color=ps.CATEGORICAL[1], lw=1.4, ls="--", zorder=3)
    ax.annotate("SNSP alone", (-0.4, snsp_r2), xytext=(0, 4),
                textcoords="offset points", ha="left", fontsize=8,
                color=ps.CATEGORICAL[1])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("held-out R$^2$")
    ax.set_title("Two columns of physics match all of it\n"
                 "and every control scores the same",
                 fontsize=10.5)
    ax.annotate("black bars: range over the random draws",
                (0.5, -0.13), xycoords="axes fraction", ha="center", va="top",
                fontsize=7.5, color=ps.INK_SOFT)
    _tidy(ax)

    fig.tight_layout()
    fig.savefig(OUT / "fig10_predictor.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


#: Which array of `hazard_draw.npz` carries each of the two grid-code scores.
DRAW_KEY = {"dev_top5": "dev5", "rocof_top5": "top5"}


def fig_topbus_vs_predictor() -> None:
    """Figure 1c's two panes, over the axis that actually orders them.

    Both rows put the *same* quantity on the vertical axis: the score measured
    on hazard draw B -- sixty disturbances sampled from the hazard prior that
    nothing in this figure was fitted to.  That is the point of the figure.  A
    predictor scored against the disturbances used to build it can look
    arbitrarily good; this one is scored against a fresh draw.

    Top: the score against SNSP.  Bottom: the score against the two-column
    model `log Pi + log H`, whose two coefficients are fitted on draw A over
    the 180 training configurations.

    The bottom row then evaluates that fitted map at draw B's *own* hazard
    aggregate.  This is the distinction the whole exercise turned on, so it is
    worth being exact: `Pi` for a contingency list is a sum of injection steps,
    computable from the dispatch and the list alone.  Knowing which
    disturbances you are to be assessed against is not knowing what they will
    do -- an operator has the list before running any study, and that is the
    input, not the answer.  No solver output from draw B reaches any feature
    here; only the sizes the list defines.  For contrast the title also quotes
    what the same fitted model scores if it is left holding draw A's hazard,
    which is the situation of a predictor whose contingency list has gone
    stale.

    Two columns are wide, so the model is fitted for each score separately;
    putting a prediction of one quantity under a measurement of another would
    make the columns incomparable.
    """
    # `risk_predictor`, not `predictor`: dynamics.py puts ../local_effects
    # on sys.path[0] and that study has a `predictor.py` of its own, which
    # would shadow this one.
    import risk_predictor as PR
    draw = np.load(HERE / "hazard_draw.npz")
    d, part, _ = PR.build()
    tr, te = PR.split(d)
    H = np.log10(d["stored_energy"].to_numpy(float))
    # Two hazard columns: draw A's, which the coefficients are fitted against,
    # and draw B's, which the fitted map is then evaluated at.  Both are sums
    # of injection steps -- no solve involved.
    X_fit = np.column_stack([np.log10(draw["sizes_A"].mean(axis=1)), H])
    X_new = np.column_stack([np.log10(draw["sizes_B"].mean(axis=1)), H])
    X_snsp = d[["snsp"]].to_numpy(float)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.8))

    for col, (key, _, phrase, ylabel) in enumerate(TOP_BUS):
        a_obs = draw[f"{DRAW_KEY[key]}_A"].mean(axis=1)     # fitting target
        b_obs = draw[f"{DRAW_KEY[key]}_B"].mean(axis=1)     # everything is scored here
        yA, yB = np.log10(a_obs), np.log10(b_obs)
        cap = phrase[0].upper() + phrase[1:]
        ylab = ylabel.replace(" (", ",\nmeasured on the 60 test disturbances (")

        al, _ = PR.choose_alpha(X_snsp[tr], yA[tr])
        p_snsp = 10.0 ** PR.apply_fit(PR.ridge(X_snsp[tr], yA[tr], al), X_snsp)
        al, _ = PR.choose_alpha(X_fit[tr], yA[tr])
        fit = PR.ridge(X_fit[tr], yA[tr], al)
        p_swing = 10.0 ** PR.apply_fit(fit, X_new)     # the list being assessed
        p_stale = 10.0 ** PR.apply_fit(fit, X_fit)     # a list gone stale
        s_snsp = PR.Score.of(yB[te], np.log10(p_snsp[te]))
        s_swing = PR.Score.of(yB[te], np.log10(p_swing[te]))
        s_stale = PR.Score.of(yB[te], np.log10(p_stale[te]))

        # Both rows carry the same quantity vertically, so they carry the same
        # vertical axis: same log scale, same limits.  Only then is the height
        # of the scatter in the two rows a like-for-like comparison, which is
        # the whole reason the figure is stacked rather than side by side.
        lim = np.array([min(b_obs.min(), p_swing.min()) * 0.94,
                        max(b_obs.max(), p_swing.max()) * 1.06])

        # -- top: the score against SNSP -------------------------------
        ax = axes[0, col]
        sc = ax.scatter(d["snsp"] * 100, b_obs, c=d["stored_energy"], cmap=SEQ,
                        s=26, edgecolors=ps.SURFACE, linewidths=0.4, zorder=3)
        lev = pd.Series(b_obs).groupby(d["snsp_target"].to_numpy()).agg(
            ["mean", "min", "max"])
        ax.plot(lev.index * 100, lev["mean"], color=ps.INK_SOFT, lw=1.6,
                zorder=4, label="mean at each SNSP level")
        ax.fill_between(lev.index * 100, lev["min"], lev["max"],
                        color=ps.INK_MUTED, alpha=0.13, zorder=1,
                        label="range within a level")
        ax.set_yscale("log")
        ax.set_ylim(*lim)
        ax.set_xlabel("SNSP of the configuration (%)")
        ax.set_ylabel(ylab)
        rho = stats.spearmanr(d["snsp"], b_obs)[0]
        ax.set_title(f"{cap}, against SNSP\n"
                     f"Spearman rho = {rho:.3f}\n"
                     f"held-out R$^2$ = {s_snsp.r2:.3f}, "
                     f"98% within {s_snsp.band98:.2f}x", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        _tidy(ax)

        # -- bottom: the score against the two-column model ------------
        ax = axes[1, col]
        res = yB[te] - np.log10(p_swing[te])
        lo, hi = np.percentile(res, [1, 99])
        ax.fill_between(lim, lim * 10 ** lo, lim * 10 ** hi,
                        color=ps.CATEGORICAL[0], alpha=0.12, zorder=1,
                        label=f"98% of held-out points "
                              f"({s_swing.band98:.2f}x wide)")
        ax.plot(lim, lim, color=ps.INK, lw=1.3, zorder=5,
                label="perfect prediction")
        ax.scatter(p_swing[~te], b_obs[~te], s=20, alpha=0.4,
                   color=ps.INK_MUTED, linewidths=0, zorder=2,
                   label="training configurations (180)")
        ax.scatter(p_swing[te], b_obs[te], s=30, color=ps.CATEGORICAL[0],
                   linewidths=0, zorder=4, label="held out (60)")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlim(*lim), ax.set_ylim(*lim)
        ax.set_xlabel(r"two-column model  $\log\Pi + \log H$" "\n"
                      "(coefficients from draw A; $\\Pi$ from the list being "
                      "assessed)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{cap}, against $\\log\\Pi + \\log H$\n"
                     f"held-out R$^2$ = {s_swing.r2:.3f}, "
                     f"98% within {s_swing.band98:.2f}x "
                     f"({s_snsp.band98 / s_swing.band98:.2f}x tighter than "
                     f"SNSP)\n"
                     f"with draw A's hazard instead: R$^2$ = {s_stale.r2:.3f}",
                     fontsize=10)
        ax.legend(loc="upper left", fontsize=7.5)
        _tidy(ax)

    fig.tight_layout(rect=(0, 0.035, 0.915, 1))
    cax = fig.add_axes((0.935, 0.57, 0.013, 0.32))
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("system inertia, sum H S (p.u. s)", fontsize=8)
    cb.outline.set_visible(False)
    fig.text(0.46, 0.012,
             "Vertical axes throughout: the score under 60 disturbances drawn "
             "from the hazard prior, whose outcomes are never used to fit "
             "anything.  Coefficients come from a disjoint draw of 60; the "
             "bottom row evaluates them at the assessed list's own hazard "
             "aggregate, which needs the list but not its outcomes.",
             ha="center", va="bottom", fontsize=8, color=ps.INK_SOFT)
    fig.savefig(OUT / "fig11_topbus_vs_predictor.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)


def fig_topbus_vs_snsp() -> None:
    """The top row of figure 11 on its own: both scores against SNSP.

    Same data, same fits and same vertical limits as
    `fig_topbus_vs_predictor`; the limits still take the two-column model's
    predictions into account so that this pane is directly comparable with
    the stacked version it is cut from.
    """
    import risk_predictor as PR
    draw = np.load(HERE / "hazard_draw.npz")
    d, part, _ = PR.build()
    tr, te = PR.split(d)
    H = np.log10(d["stored_energy"].to_numpy(float))
    X_fit = np.column_stack([np.log10(draw["sizes_A"].mean(axis=1)), H])
    X_new = np.column_stack([np.log10(draw["sizes_B"].mean(axis=1)), H])
    X_snsp = d[["snsp"]].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))

    for col, (key, _, phrase, ylabel) in enumerate(TOP_BUS):
        a_obs = draw[f"{DRAW_KEY[key]}_A"].mean(axis=1)     # fitting target
        b_obs = draw[f"{DRAW_KEY[key]}_B"].mean(axis=1)     # scored here
        yA, yB = np.log10(a_obs), np.log10(b_obs)
        cap = phrase[0].upper() + phrase[1:]
        ylab = ylabel.replace(" (", ",\nmeasured on the 60 test disturbances (")

        al, _ = PR.choose_alpha(X_snsp[tr], yA[tr])
        p_snsp = 10.0 ** PR.apply_fit(PR.ridge(X_snsp[tr], yA[tr], al), X_snsp)
        al, _ = PR.choose_alpha(X_fit[tr], yA[tr])
        fit = PR.ridge(X_fit[tr], yA[tr], al)
        p_swing = 10.0 ** PR.apply_fit(fit, X_new)
        s_snsp = PR.Score.of(yB[te], np.log10(p_snsp[te]))

        # Identical to the stacked figure's limits, so the height of the
        # scatter reads the same in both places.
        lim = np.array([min(b_obs.min(), p_swing.min()) * 0.94,
                        max(b_obs.max(), p_swing.max()) * 1.06])

        ax = axes[col]
        sc = ax.scatter(d["snsp"] * 100, b_obs, c=d["stored_energy"], cmap=SEQ,
                        s=26, edgecolors=ps.SURFACE, linewidths=0.4, zorder=3)
        lev = pd.Series(b_obs).groupby(d["snsp_target"].to_numpy()).agg(
            ["mean", "min", "max"])
        ax.plot(lev.index * 100, lev["mean"], color=ps.INK_SOFT, lw=1.6,
                zorder=4, label="mean at each SNSP level")
        ax.fill_between(lev.index * 100, lev["min"], lev["max"],
                        color=ps.INK_MUTED, alpha=0.13, zorder=1,
                        label="range within a level")
        ax.set_yscale("log")
        ax.set_ylim(*lim)
        ax.set_xlabel("SNSP of the configuration (%)")
        ax.set_ylabel(ylab)
        rho = stats.spearmanr(d["snsp"], b_obs)[0]
        ax.set_title(f"{cap}, against SNSP\n"
                     f"Spearman rho = {rho:.3f}\n"
                     f"held-out R$^2$ = {s_snsp.r2:.3f}, "
                     f"98% within {s_snsp.band98:.2f}x", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        _tidy(ax)

    fig.tight_layout(rect=(0, 0.07, 0.915, 1))
    cax = fig.add_axes((0.935, 0.25, 0.013, 0.55))
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("system inertia, sum H S (p.u. s)", fontsize=8)
    cb.outline.set_visible(False)
    fig.text(0.46, 0.012,
             "Vertical axes: the score under 60 disturbances drawn from the "
             "hazard prior, whose outcomes are never used to fit anything.",
             ha="center", va="bottom", fontsize=8, color=ps.INK_SOFT)
    fig.savefig(OUT / "fig11a_topbus_vs_snsp.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", default="sourced", choices=list(FAMILIES),
                    help="which disturbance set figures 1-7 are drawn from")
    args = ap.parse_args()
    fam = FAMILIES[args.family]

    global OUT
    OUT = HERE / fam["out"]
    ps.use()
    OUT.mkdir(exist_ok=True, parents=True)
    g = G.load()
    events = PB.load(g, which=fam["events"])
    df, nodes = _load(fam["base"])
    fig_cost_vs_snsp(df)
    fig_predictors(df)
    fig_traces(g, events)
    fig_maps(g, df, nodes)
    fig_metrics(df)
    fig_events(df, nodes, events)
    fig_events_predictors(df, nodes, events)
    # Both grid-code figures need an ensemble written after the deviation score
    # was added: figure 1c reads the `dev_top5` column, and figure 1d reduces
    # the raw per-bus-per-event fields the same run started saving.  An older
    # family has neither, and re-running it is what fills them in.
    have_cols = all(k in df.columns for k, _, _, _ in TOP_BUS)
    have_fields = all(f in nodes for _, f, _, _ in TOP_BUS)
    if have_cols and have_fields:
        fig_topbus(df)
        fig_topbus_events(df, nodes, events)
    else:
        suffix = f"_{fam['base']}" if fam["base"] else ""
        missing = "the dev_top5 column" if not have_cols else ""
        missing += " and " if not have_cols and not have_fields else ""
        missing += "the raw per-bus-per-event fields" if not have_fields else ""
        print(f"ensemble{suffix}.csv / ensemble{suffix}_nodes.npz predate "
              f"{missing} -- skipping fig1c and fig1d.  Re-run  "
              f"run_ensemble.py --events {fam['events']} "
              f"--tag \"{fam['base']}\"  to fill them in.")
    for tag, fn, name in ((fam["ffr"], fig_ffr, "fig5"),
                          (fam["agc"], fig_secondary, "fig7")):
        try:
            other, _ = _load(tag)
        except FileNotFoundError:
            print(f"no ensemble_{tag}.csv -- skipping {name}")
        else:
            fn(df, other)
    if args.family == "sourced":
        fig_muon()          # the 2x2 belongs to no family: it *is* the contrast
        # The predictor study reads frozen outputs rather than refitting: the
        # controls take minutes, and a figure should not be where that happens.
        if (HERE / "predictor_predictions.csv").exists():
            fig_clusters()
            fig_predictor()
            if (HERE / "hazard_draw.npz").exists():
                fig_topbus_vs_predictor()
                fig_topbus_vs_snsp()
            else:
                print("no hazard_draw.npz -- run  python hazard_draw.py "
                      "--events 60 --freeze  first; skipping fig11")
        else:
            print("no predictor_predictions.csv -- run  python risk_predictor.py "
                  "--freeze  first; skipping fig9 and fig10")
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()

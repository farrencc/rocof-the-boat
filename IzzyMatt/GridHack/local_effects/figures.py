"""Seven figures for the local-effects study.

Colour comes from the participant kit's own palette module, so these read as part
of the same set as the kit's examples and inherit a palette that has already been
validated for colour-vision deficiency.  Magnitude is always carried by the
single-hue sequential ramp; converter membership is carried by an outline, never
by hue alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path("../repo/grid_TF_Wind/participant-kit").resolve()))
import plotstyle as ps

import stochastic as stoch
import swing as sw
from designed import (L, K_CONV, block, block_infill, dispersed, field,
                      largest_cluster, quadrant_shortcuts)

OUT = Path("figures")
SEQ = LinearSegmentedColormap.from_list("seq", ps.SEQUENTIAL)
PKT = "the pocket swings on its\nown, well past +/- 100 mHz"
TRK = "every bus tracks\nthe system"


def fig_propagation() -> None:
    """The mechanism: a clustered pocket runs its own frequency for a while.

    Frequency, not RoCoF, is plotted here.  RoCoF is the derivative of a lightly
    damped oscillation, so raw traces are a thicket of overlapping ringing and the
    disturbed bus's own spike sets a scale that hides everything else.  The
    integral is smooth, and the quantity of interest -- whether one region's
    frequency departs from the system's -- is a vertical gap you can read off.
    """
    lat = sw.Lattice(L, K=3.0, periodic=True)
    inside, far = lat.index(1, 1), lat.index(5, 5)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True)

    # Three panels, not two, because where the event happens matters as much as
    # the geography: a pocket is hit hardest by its own events.  Showing only the
    # first panel would overstate the effect roughly tenfold.  The governor is on
    # here -- it leaves the separation untouched but stops the common mode
    # falling forever, so the traces are the realistic shape.
    for ax, (ni, at, title) in zip(axes, (
            (block(lat), inside, "clustered - event inside the wind block"),
            (block(lat), far, "clustered - event on the far side"),
            (dispersed(lat), inside, "dispersed - same event as the left panel"))):
        net = sw.build_network(lat, ni, layout="flat", droop=sw.DROOP_GAIN)
        sim = sw.simulate_linear(net, sw.load_step(net, at, 0.5), t_end=5.0,
                                 dt=2e-3)
        t = sim["t"] * 1000
        mask = np.zeros(lat.N, bool)
        mask[list(ni)] = True
        mask[at] = False                       # the disturbed bus is its own story
        sync = ~mask.copy()
        sync[at] = False

        f = sim["freq_dev"] * 1000             # mHz
        ax.plot(t, f[mask].mean(axis=0), color=ps.CATEGORICAL[0], lw=2.2,
                label="converter buses", zorder=3)
        ax.plot(t, f[sync].mean(axis=0), color=ps.CATEGORICAL[1], lw=2.2,
                label="synchronous buses", zorder=3)
        ax.plot(t, sim["freq_coi"] * 1000, color=ps.INK_SOFT, lw=1.5, ls="--",
                label="system (centre of inertia)", zorder=2)
        gap = np.abs(f[mask].mean(axis=0) - f[sync].mean(axis=0)).max()
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("time since the disturbance (ms)")
        ax.set_xlim(0, 5000)
        ax.grid(True, alpha=0.35)
        ax.annotate(f"peak separation {gap:.0f} mHz", xy=(0.97, 0.05),
                    xycoords="axes fraction", ha="right", fontsize=9,
                    color=ps.INK_SOFT)

    axes[0].set_ylabel("frequency deviation (mHz)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("A clustered low-inertia region holds its own frequency - most "
                 "sharply for its own events", x=0.008, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig1_propagation.png", dpi=170)
    plt.close(fig)


def fig_noise() -> None:
    """The same question asked of the everyday regime rather than an event."""
    lat = sw.Lattice(L, K=3.0, periodic=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, (ni, title) in zip(axes, (
            (block(lat), "16 converters in one 4x4 corner"),
            (dispersed(lat), "the same 16, spread evenly"))):
        net = sw.build_network(lat, ni, layout="flat", droop=sw.DROOP_GAIN)
        sig = stoch.noise_profile(net, ni, "at_converters")
        t, f, coi = stoch.realise(net, sig, tau=1.0, t_end=60.0, seed=3)
        m = np.zeros(lat.N, bool)
        m[list(ni)] = True
        ax.plot(t, f[:, m].mean(axis=1) * 1000, color=ps.CATEGORICAL[0], lw=1.5,
                label="converter buses", zorder=3)
        ax.plot(t, f[:, ~m].mean(axis=1) * 1000, color=ps.CATEGORICAL[1], lw=1.5,
                label="synchronous buses", zorder=4)
        g = np.zeros(lat.N, bool)
        g[list(ni)] = True
        sd = stoch.separation_sd(net, sig, g, tau=1.0) * 1000
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.35)
        ax.annotate(f"sd of the gap: {sd:.0f} mHz", xy=(0.97, 0.05),
                    xycoords="axes fraction", ha="right", fontsize=9,
                    color=ps.INK_SOFT)
    axes[0].set_ylabel("frequency deviation (mHz)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Under continuous imbalance the pocket wanders away from the "
                 "system all the time", x=0.012, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig6_noise.png", dpi=170)
    plt.close(fig)


def _map(ax, g, ni, lat, vmin, vmax, title):
    im = ax.imshow(g, cmap=SEQ, vmin=vmin, vmax=vmax)
    for b in ni:
        r, c = lat.coords[b]
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor=ps.INK, lw=1.6))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=10)
    return im


def fig_maps() -> None:
    """The headline, in both quantities a grid code actually constrains.

    Top row is RoCoF, which loss-of-mains and RoCoF relays watch; bottom row is
    the frequency excursion, which under-frequency load shedding watches.  Each
    row has its own colour scale because the units differ, but the two columns
    within a row share one, so the left/right comparison is honest.
    """
    lat = sw.Lattice(L, K=3.0, periodic=True)
    out = {}
    for tag, ni in (("clustered", block(lat)), ("dispersed", dispersed(lat))):
        net = sw.build_network(lat, ni, layout="flat", droop=sw.DROOP_GAIN)
        out[tag] = (ni, field(net))

    rows = [("worst_window", 1.0, "worst 500 ms RoCoF (Hz/s)", "{:.3f} Hz/s", "RoCoF"),
            ("nadir", 1000.0, "frequency excursion (mHz)", "{:.0f} mHz", "frequency")]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.0))
    for r, (key, mul, cblab, fmt, rowlab) in enumerate(rows):
        vals = {t: f[key] * mul for t, (_, f) in out.items()}
        vmin = min(v.min() for v in vals.values())
        vmax = max(v.max() for v in vals.values())
        for c, (tag, label) in enumerate((("clustered", "16 converters in one 4x4 corner"),
                                          ("dispersed", "the same 16, spread evenly"))):
            ni, _ = out[tag]
            ax = axes[r, c]
            im = _map(ax, vals[tag].reshape(L, L), ni, lat, vmin, vmax,
                      label if r == 0 else "")
            ax.set_xlabel("worst bus " + fmt.format(vals[tag].max()),
                          fontsize=9, color=ps.INK_SOFT)
        cb = fig.colorbar(im, ax=axes[r, :], fraction=0.045, pad=0.02)
        cb.set_label(cblab, fontsize=9)
        cb.outline.set_visible(False)
        rr = vals["clustered"].max() / vals["dispersed"].max()
        lab = f"{rowlab}\nworst bus {rr:.2f}x worse when clustered"
        axes[r, 0].text(-0.10, 0.5, lab,
                        transform=axes[r, 0].transAxes, rotation=90, va="center",
                        ha="center", fontsize=9.5, color=ps.INK_SOFT)
    fig.suptitle("Clustering hurts both quantities the grid code limits - RoCoF "
                 "about twice as much", x=0.02, ha="left", fontsize=11.5)
    fig.text(0.02, 0.015, "dark outline = inverter-based (no inertia) bus.  8x8 "
                          "torus, governor on, averaged over a load step at every "
                          "bus.\nTotal inertia is identical between the two columns.",
             fontsize=8.5, color=ps.INK_MUTED)
    fig.savefig(OUT / "fig2_maps.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_ensemble() -> None:
    """Every placement tried, not just the two that make the point.

    The x-axis is the size of the largest contiguous all-converter patch, not the
    mean pairwise distance this figure first used.  Mean distance is close to
    blind here: all 120 random placements land between 3.69 and 4.23 on it, so
    the cloud is a vertical blob in the right-hand quarter of the axis and the
    only thing setting the slope is the four hand-designed points at the ends.
    It also scores `two blocks of 8` and `dispersed sublattice` identically
    (4.267 apiece) when their worst buses differ by 1.7x -- an argument against
    the caption, sitting in the middle of the figure.

    Patch size separates those two (8 against 1), spreads the random placements
    across most of the axis, and -- the point -- carries the trend on its own:
    Spearman rho is 0.94 on RoCoF and 0.96 on the excursion among the 120 random
    placements, with all four designed points left out.  The claim no longer
    rests on the hand-picked ends.

    The designed placements are one colour, not four.  They are one series, told
    apart by their labels; four hues would spend the categorical channel on
    nothing and read as a legend floating in the plot area.
    """
    lat = sw.Lattice(L, K=3.0, periodic=True)
    df = pd.read_csv("designed.csv")
    if "largest_cluster" not in df.columns:   # csv written before the column existed
        df["largest_cluster"] = [largest_cluster(lat, {int(b) for b in c.split("-")})
                                 for c in df.converters]
    rnd = df[df.config.str.startswith("random")]
    named = df[~df.config.str.startswith("random")]
    jitter = np.random.default_rng(4).uniform(-0.2, 0.2, len(rnd))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    panels = [("worst_window_worst", 1.0, "worst bus, worst 500 ms RoCoF (Hz/s)"),
              ("nadir_worst", 1000.0, "worst bus, frequency excursion (mHz)")]
    for ax, (col, mul, ylab) in zip(axes, panels):
        y = df[col] * mul
        ax.set_axisbelow(True)
        ax.grid(True, color=ps.INK_MUTED, alpha=0.22, lw=0.6)
        ax.scatter(rnd.largest_cluster + jitter, rnd[col] * mul, s=30,
                   color=ps.CATEGORICAL[0], edgecolor=ps.SURFACE, lw=0.8,
                   alpha=0.85, zorder=2,
                   label=f"random placements (n={len(rnd)})")

        # The median of the random placements at each patch size: the trend the
        # caption is claiming, drawn from the data that was not chosen to show
        # it.  Only sizes with at least three samples get a point -- past 9 the
        # random draws thin out to one or two, and a line through those would be
        # reporting noise as a trend.
        grp = rnd.groupby("largest_cluster")[col].agg(["median", "size"])
        grp = grp[grp["size"] >= 3]
        ax.plot(grp.index, grp["median"] * mul, color=ps.INK_SOFT, lw=1.8,
                marker="o", ms=5, mec=ps.SURFACE, mew=0.8, zorder=3,
                label="median of the random placements")

        ax.scatter(named.largest_cluster, named[col] * mul, s=110,
                   color=ps.CATEGORICAL[1], edgecolor=ps.SURFACE, lw=1.6,
                   zorder=5, label="designed placements")

        # Two designed placements share a patch size of 16, and which of them is
        # on top swaps between the panels, so the labels are pushed apart by the
        # values in this panel rather than by a fixed table of offsets.
        for size, same in named.groupby("largest_cluster"):
            same = same.sort_values(col)
            for n, (_, r) in enumerate(same.iterrows()):
                dy = (-13 if n == 0 else 7) if len(same) > 1 else -3
                # Label towards the empty upper-left of the trend, not into it.
                right = size >= 8
                ax.annotate(r.config.replace(" (all in one corner)", ""),
                            (size, r[col] * mul), textcoords="offset points",
                            xytext=(-12 if right else 11, dy), fontsize=8.5,
                            color=ps.INK_SOFT, ha="right" if right else "left",
                            zorder=6)

        rho = rnd.largest_cluster.corr(rnd[col], method="spearman")
        ax.annotate(f"random placements alone: Spearman $\\rho$ = {rho:.2f}",
                    xy=(0.97, 0.04), xycoords="axes fraction", ha="right",
                    fontsize=9, color=ps.INK_SOFT)
        ax.set_xlabel("buses in the largest all-converter patch\n"
                      "(left = spread out, right = clustered)", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
        ax.set_xlim(-0.6, 17.4)
        pad = 0.10 * (y.max() - y.min())
        ax.set_ylim(y.min() - pad, y.max() + pad)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("The worst bus tracks the size of the biggest inertia-free "
                 "patch - on both limits", x=0.008, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.text(0.008, 0.015, "124 placements of the same 16 converters on an 8x8 "
                           "torus; total inertia, and so the system-wide RoCoF, "
                           "is identical in every one.\nEach point is averaged "
                           "over a load step at every bus.  The rank correlation "
                           "excludes the four designed placements.",
             fontsize=8.5, color=ps.INK_MUTED)
    fig.savefig(OUT / "fig3_ensemble.png", dpi=170)
    plt.close(fig)


def fig_measurement() -> None:
    """How you measure decides whether you see the effect at all.

    This replaces an earlier version that plotted a single ratio-versus-window
    curve.  That curve carried about one number's worth of information, and it
    buried the actual claim in a caption: a window pinned to the moment of the
    event averages the local swing away and reports no clustering penalty, while
    a sliding window over the same solve reports a factor of two.  Here both
    curves are drawn, so the divergence is the picture rather than a footnote,
    and the left panel puts every measurement choice on one scale.
    """
    lat = sw.Lattice(L, K=3.0, periodic=True)
    sims = {}
    for tag, mk in (("c", block), ("d", dispersed)):
        net = sw.build_network(lat, mk(lat), layout="flat", droop=sw.DROOP_GAIN)
        sims[tag] = [sw.simulate_linear(net, sw.load_step(net, b, 0.5),
                                        t_end=5.0, dt=2e-3) for b in range(lat.N)]

    def step_ratio(key, window=0.5, absolute=False):
        out = {}
        for tag in ("c", "d"):
            ms = [sw.metrics(s, window=window) for s in sims[tag]]
            v = [np.abs(m[key]) if absolute else m[key] for m in ms]
            out[tag] = np.mean(v, axis=0).max()
        return out["c"] / out["d"]

    def noise_ratio(key, corr_len):
        out = {}
        for tag, mk in (("c", block), ("d", dispersed)):
            ni = mk(lat)
            net = sw.build_network(lat, ni, layout="flat", droop=sw.DROOP_GAIN)
            s = stoch.stationary(net, stoch.noise_profile(net, ni, "at_converters"),
                                 tau=1.0, corr=stoch.correlation(lat, corr_len))
            out[tag] = s[key].max()
        return out["c"] / out["d"]

    rows = [
        ("discrete event", "RoCoF | 500 ms pinned to event",
         step_ratio("rocof_500ms", absolute=True), 0),
        ("discrete event", "RoCoF | worst 500 ms window",
         step_ratio("worst_window"), 0),
        ("discrete event", "RoCoF | instantaneous peak",
         step_ratio("peak_instant"), 0),
        ("discrete event", "freq  | excursion (nadir)",
         step_ratio("nadir", absolute=True), 1),
        ("continuous noise", "RoCoF | independent wind",
         noise_ratio("rocof_window_sd", 0.0), 0),
        ("continuous noise", "RoCoF | correlated wind",
         noise_ratio("rocof_window_sd", 2.0), 0),
        ("continuous noise", "freq  | independent wind",
         noise_ratio("freq_sd", 0.0), 1),
        ("continuous noise", "freq  | correlated wind",
         noise_ratio("freq_sd", 2.0), 1),
    ]

    # Left column split into two stacked panels, one per regime: that labels the
    # grouping without putting text inside the plotting area, where it collided
    # with the reference line and the shortest bars.
    fig = plt.figure(figsize=(12.6, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.42, wspace=0.32)
    axL = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    axR = fig.add_subplot(gs[:, 1])

    groups = [("a discrete event", rows[:4]), ("continuous imbalance", rows[4:])]
    for ax, (gname, grp) in zip(axL, groups):
        ys = list(range(len(grp)))[::-1]
        for y, (_g, lab, val, kind) in zip(ys, grp):
            col = ps.CATEGORICAL[kind]
            ax.plot([1.0, val], [y, y], color=col, lw=2.0, alpha=0.5, zorder=2,
                    solid_capstyle="round")
            ax.scatter([val], [y], s=95, color=col, edgecolor=ps.SURFACE, lw=1.4,
                       zorder=4)
            ax.annotate(f"{val:.2f}x", (val, y), textcoords="offset points",
                        xytext=(11, -3.5), fontsize=9, color=ps.INK_SOFT)
        ax.axvline(1.0, color=ps.INK_SOFT, lw=1.4, ls="--", zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[1] for r in grp], fontsize=9)
        ax.set_xlim(0.9, 5.2)
        ax.set_ylim(-0.6, len(grp) - 0.4)
        ax.set_title(gname, loc="left", fontsize=9.5, color=ps.INK_SOFT)
        ax.grid(True, axis="x", alpha=0.3)
    axL[1].set_xlabel("worst bus: clustered / dispersed", fontsize=9.5)
    axL[0].annotate("1.0 = no effect visible", xy=(1.02, len(groups[0][1]) - 0.55),
                    fontsize=8.5, color=ps.INK_SOFT, va="top")
    h = [plt.Line2D([], [], marker="o", ls="", color=ps.CATEGORICAL[i], ms=8,
                    label=l) for i, l in ((0, "RoCoF"), (1, "frequency"))]
    axL[1].legend(handles=h, frameon=False, fontsize=8.5, loc="lower right")

    ax = axR
    wins = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    sliding = [step_ratio("worst_window", w) for w in wins]
    pinned = [step_ratio("rocof_500ms", w, absolute=True) for w in wins]
    xs = np.array(wins) * 1000
    ax.axvspan(455, 545, color=ps.INK_MUTED, alpha=0.16, zorder=0)
    ax.plot(xs, sliding, color=ps.CATEGORICAL[0], lw=2.2, marker="o", ms=6,
            mec=ps.SURFACE, mew=1.1, label="worst window anywhere in the event")
    ax.plot(xs, pinned, color=ps.CATEGORICAL[2], lw=2.2, marker="s", ms=6,
            mec=ps.SURFACE, mew=1.1, label="window pinned to the event")
    ax.axhline(1.0, color=ps.INK_SOFT, lw=1.4, ls="--")
    ax.annotate("500 ms: the grid-code window", xy=(500, 3.1), rotation=90,
                fontsize=8.5, color=ps.INK_SOFT, ha="center", va="center")
    ax.annotate("one solve,\nread two ways", xy=(520, 1.55), xytext=(690, 2.45),
                fontsize=8.5, color=ps.INK_SOFT,
                arrowprops=dict(arrowstyle="->", color=ps.INK_MUTED, lw=1.0))
    ax.set_xlabel("length of the measurement window (ms)", fontsize=9.5)
    ax.set_ylabel("worst bus: clustered / dispersed", fontsize=9.5)
    ax.set_title("Pinning the window to the event hides it", loc="left",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.35)
    fig.suptitle("Whether the local effect is visible at all is a choice of "
                 "measurement, not of physics", x=0.008, ha="left", fontsize=11.5)
    # explicit margins: this figure mixes a spanning axes with stacked ones,
    # which tight_layout cannot solve and silently clips the row labels for
    fig.subplots_adjust(left=0.175, right=0.988, top=0.865, bottom=0.115,
                        hspace=0.55, wspace=0.30)
    fig.savefig(OUT / "fig4_measurement.png", dpi=170)
    plt.close(fig)


def fig_shortcuts() -> None:
    """Does non-local connectivity undo the clustering penalty?

    Left column is the clustered baseline, identical to fig 2's left column.
    Middle column is the same grid with 48 long lines added, three from each
    converter to its counterpart in the other three quadrants.  Right column is
    the line-count-matched control: the same 48 lines, drawn at random from
    the pairs inside the block that the lattice does not already connect, so
    none of them reaches a synchronous bus.  These are the 48 that
    `selfenergy.py` and the README's control table use.

    Nothing else changes in any column: the injections are still zero
    everywhere, so the operating point is untouched, and no inertia is added, so
    the system-wide RoCoF and nadir are identical across the three.  Only the
    graph moves.

    The figure carries no title or note of its own: the numbers under each
    column are the whole of it, and what they mean is here.  Both treatments
    take about a fifth off the worst bus, but they do not get there the same
    way and they do not agree on which is better.  The long lines lower the
    whole grid (mean 17%, synchronous buses 13%); the inside ones only level
    the block (mean 14%, synchronous buses 3%).  So the inside lines win on
    RoCoF (26% against 21%) and lose on the nadir (18% against 19%) -- which of
    the two looks better is a choice of metric, not a fact about the grid.

    The control is the point of the third column.  If the outward lines worked
    by buying the pocket access to distant inertia, then lines that reach no
    inertia at all should do little -- and they do not behave that way, so the
    two columns are worth reading against each other rather than against the
    baseline alone.
    """
    lat = sw.Lattice(L, K=3.0, periodic=True)
    ni = block(lat)
    cols = [("baseline", (), "16 converters in one 4x4 corner"),
            ("quadrant", quadrant_shortcuts(lat),
             "plus 48 long lines out of the block"),
            ("inblock", block_infill(lat),
             "plus 48 lines inside the block")]
    out = {}
    for tag, extra, _ in cols:
        lt = sw.Lattice(L, K=3.0, periodic=True, extra_edges=extra)
        out[tag] = field(sw.build_network(lt, ni, layout="flat",
                                          droop=sw.DROOP_GAIN))

    rows = [("worst_window", 1.0, "worst 500 ms RoCoF (Hz/s)", "{:.3f} Hz/s", "RoCoF"),
            ("nadir", 1000.0, "frequency excursion (mHz)", "{:.0f} mHz", "frequency")]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.0))
    for r, (key, mul, cblab, fmt, rowlab) in enumerate(rows):
        vals = {t: f[key] * mul for t, f in out.items()}
        # One scale across all three columns of a row, so the comparison that
        # matters -- how much of the block's colour survives -- is read off the
        # image and not off three different ramps.
        vmin = min(v.min() for v in vals.values())
        vmax = max(v.max() for v in vals.values())
        for c, (tag, _extra, label) in enumerate(cols):
            ax = axes[r, c]
            im = _map(ax, vals[tag].reshape(L, L), ni, lat, vmin, vmax,
                      label if r == 0 else "")
            if tag == "quadrant":
                # the quadrant grid the long lines connect across
                for q in (3.5,):
                    ax.axhline(q, color=ps.INK_SOFT, lw=1.0, ls=":", alpha=0.8)
                    ax.axvline(q, color=ps.INK_SOFT, lw=1.0, ls=":", alpha=0.8)
            worst = vals[tag].max()
            note = "worst bus " + fmt.format(worst)
            if tag != "baseline":
                note += f"  ({1.0 - worst / vals['baseline'].max():.0%} lower)"
            ax.set_xlabel(note, fontsize=9, color=ps.INK_SOFT)
        cb = fig.colorbar(im, ax=axes[r, :], fraction=0.030, pad=0.02)
        cb.set_label(cblab, fontsize=9)
        cb.outline.set_visible(False)
        axes[r, 0].text(-0.10, 0.5, rowlab, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=9.5,
                        color=ps.INK_SOFT)
    fig.savefig(OUT / "fig7_shortcuts.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_snsp() -> None:
    """What clustering costs in carrying capacity, not just in Hz/s."""
    df = pd.read_csv("snsp.csv")
    sh = df["share"].to_numpy() * 100

    fig, axes2 = plt.subplots(2, 2, figsize=(9.8, 8.0))
    panels = [("step_norm", "step", "RoCoF - discrete event",
               "worst bus, rel. to all-synchronous", 1.0),
              ("noise", "noise", "RoCoF - continuous imbalance at the wind",
               "worst bus, 500 ms RoCoF sd (Hz/s)", 1.0),
              ("step_freq_norm", "step_freq", "frequency - discrete event",
               "worst bus, rel. to all-synchronous", 1.0),
              ("noise_freq", "noise_freq", "frequency - continuous imbalance",
               "worst bus, frequency sd (mHz)", 1000.0)]
    axes = axes2.ravel()

    for ax, (col, raw, title, ylab, mul) in zip(axes, panels):
        rnd = df[f"random_{col}"].to_numpy() * mul
        sd = df.get(f"random_{raw}_sd")
        if sd is not None:
            scale = np.where(df[f"random_{raw}"] != 0,
                             rnd / (df[f"random_{raw}"].to_numpy() * mul + 1e-30), 1.0)
            band = sd.to_numpy() * mul * scale
            ax.fill_between(sh, rnd - band, rnd + band,
                            color=ps.CATEGORICAL[0], alpha=0.16, lw=0)
        ax.plot(sh, rnd, color=ps.CATEGORICAL[0], lw=2.2, marker="o", ms=5,
                mec=ps.SURFACE, mew=1.0, label="random siting", zorder=3)
        ax.plot(sh, df[f"clustered_{col}"] * mul, color=ps.CATEGORICAL[1], lw=2.2,
                marker="s", ms=5, mec=ps.SURFACE, mew=1.0,
                label="all in one cluster", zorder=4)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("non-synchronous share of buses (%)", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(True, alpha=0.35)

    # the headroom construction, drawn on the RoCoF/step panel
    ax = axes[0]
    cl = df["clustered_step_norm"].to_numpy()
    rn = np.maximum.accumulate(df["random_step_norm"].to_numpy())
    i = int(np.argmin(np.abs(df["share"].to_numpy() - 0.25)))
    lvl = cl[i]
    j = int(np.searchsorted(rn, lvl))
    if j < len(rn):
        xr = np.interp(lvl, rn[j - 1:j + 1], sh[j - 1:j + 1])
        ax.annotate("", xy=(xr, lvl), xytext=(sh[i], lvl),
                    arrowprops=dict(arrowstyle="<->", color=ps.INK_SOFT, lw=1.4))
        ax.plot([sh[i], xr], [lvl, lvl], color=ps.INK_SOFT, lw=0.9, ls=":", zorder=1)
        ax.annotate(f"same worst bus at\n{xr - sh[i]:.0f} points more share",
                    xy=((sh[i] + xr) / 2, lvl), xytext=((sh[i] + xr) / 2, lvl * 0.72),
                    fontsize=8.5, color=ps.INK_SOFT, ha="center")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Clustering costs carrying capacity on both limits, RoCoF worst",
                 x=0.012, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig5_snsp.png", dpi=170)
    plt.close(fig)


def main() -> None:
    ps.use()
    OUT.mkdir(exist_ok=True)
    figs = [fig_propagation, fig_maps, fig_ensemble, fig_measurement, fig_noise,
            fig_shortcuts]
    if Path("snsp.csv").exists():
        figs.append(fig_snsp)
    for fn in figs:
        fn()
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()

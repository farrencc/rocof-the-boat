"""Draw the frozen toy grid: what is where, and what it is running at.

Two panels, because the grid has two things worth seeing and they do not fit in
one set of marks:

  left   the network itself -- every bus by role, every line by class.  This is
         the part that never changes, so it is the reference picture for the
         whole investigation.
  right  the net injection at the reference dispatch (SNSP 70%), which is one
         sample of the thing that *does* change.  Signed quantity, so a
         diverging ramp with a neutral middle; magnitude also carried by area,
         so the sign is never on colour alone.

Colour comes from the participant kit's validated palette, as in
../local_effects/figures.py.  Roles use four categorical slots plus a neutral
for the rural buses; every pair of those clears the all-pairs CVD check
(worst min(protan, deutan) Delta E 10.8, worst normal-vision Delta E 16.3) and
every role also carries its own marker shape, so identity is never on hue alone.
Line class is drawn with weight and dash, never with hue -- the hue channel is
spent entirely on the buses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str((HERE / ".." / "repo" / "grid_TF_Wind"
                        / "participant-kit").resolve()))
import plotstyle as ps

import grid as G

OUT = HERE / "figures"

#: Signed quantity, so a diverging ramp with a neutral middle -- reversed from
#: the kit's default so that injection is cool and demand is warm, which keeps
#: the wind buses blue in both panels.
DIVERGING_R = ps.diverging_cmap.reversed()

#: Bus roles: colour + marker.  The marker is not decoration -- it is the
#: secondary encoding that keeps the map readable in greyscale and under CVD.
ROLE_STYLE = {
    "wind": dict(colour=ps.CATEGORICAL[0], marker="o", label="wind farm"),
    "hvdc": dict(colour=ps.CATEGORICAL[4], marker="D", label="HVDC infeed"),
    "sync": dict(colour=ps.CATEGORICAL[1], marker="s",
                 label="synchronous station"),
    "city": dict(colour=ps.CATEGORICAL[6], marker="h", label="city demand"),
    "load": dict(colour="#6f6e69", marker=".", label="rural demand"),
}

#: Line classes: weight and dash, no hue.
LINE_STYLE = {
    "backbone": dict(lw=2.6, ls="-", colour=ps.INK_SOFT, z=2,
                     label="400 kV transfer corridor"),
    "main": dict(lw=1.0, ls="-", colour="#a9a8a2", z=1, label="220 kV mesh"),
    "spur": dict(lw=1.1, ls=(0, (3.5, 2.0)), colour="#a9a8a2", z=1,
                 label="110 kV western seaboard"),
}

#: Marker area (pt^2) for a unit of nameplate or demand, and the floor for a
#: bus that has neither.
AREA_PER_PU = 150.0
AREA_MIN = 26.0


def _xy(g: G.ToyGrid) -> tuple[np.ndarray, np.ndarray]:
    """Plotting coordinates: column east, row south, north at the top."""
    x = np.array([c for _, c in g.coords], float)
    y = np.array([-r for r, _ in g.coords], float)
    return x, y


def _draw_lines(ax, g: G.ToyGrid, plain: str | None = None) -> None:
    x, y = _xy(g)
    for i, j in g.edges:
        s = LINE_STYLE[g.klass(i, j)]
        ax.plot([x[i], x[j]], [y[i], y[j]],
                lw=s["lw"] if plain is None else 0.8,
                ls=s["ls"] if plain is None else "-",
                color=s["colour"] if plain is None else plain,
                zorder=s["z"], solid_capstyle="round")


def _size(g: G.ToyGrid) -> np.ndarray:
    """Area proportional to nameplate (generators) or demand (loads)."""
    weight = np.where(g.capacity > 0, g.capacity, g.demand)
    return AREA_MIN + AREA_PER_PU * weight


def panel_topology(ax, g: G.ToyGrid) -> None:
    x, y = _xy(g)
    _draw_lines(ax, g)
    size = _size(g)

    for role in ("load", "city", "wind", "hvdc", "sync"):
        m = g.role == role
        s = ROLE_STYLE[role]
        ax.scatter(x[m], y[m], s=size[m] if role != "load" else 11,
                   marker=s["marker"], c=s["colour"],
                   edgecolors=ps.SURFACE if role != "load" else "none",
                   linewidths=1.1, zorder=4)

    # A radial farm is a topological fact, not a role, so it gets its own mark:
    # an open ring around the bus.  These four are the buses with one line.
    radial = [g.index(r, 0) for r in G.STUB_ROWS]
    ax.scatter(x[radial], y[radial], s=size[radial] * 3.1, marker="o",
               facecolors="none", edgecolors=ps.CATEGORICAL[0], linewidths=1.1,
               linestyle=(0, (2, 1.6)), zorder=3)

    # Label placement, by where the bus sits: the boundary is crowded, so the
    # bottom row goes underneath on two staggered lines and the east coast goes
    # inboard, which is the only way these eleven names avoid each other.
    stagger = 0
    for i in sorted(np.r_[g.by_role("city"), g.by_role("sync"),
                          g.by_role("hvdc")], key=lambda b: g.coords[b][1]):
        r, c = g.coords[i]
        if r == g.L - 1:
            stagger += 1
            ax.annotate(str(g.name[i]), (x[i], y[i] - (0.45 + 0.45 * (stagger % 2))),
                        ha="center", va="top", fontsize=7.2, color=ps.INK_SOFT,
                        zorder=5)
        else:
            # Clear of the marker, which is sized by nameplate: Dublin's hexagon
            # is three times the width of a small farm's dot.
            off = 0.30 + 0.13 * max(g.capacity[i], g.demand[i])
            side = -1 if c >= g.L - 2 else 1
            ax.annotate(str(g.name[i]), (x[i] + side * off, y[i]),
                        ha="right" if side < 0 else "left", va="center",
                        fontsize=7.2, color=ps.INK_SOFT, zorder=5)

    ax.annotate("Atlantic seaboard:\n8 of 12 farms, 78% of the\n"
                "wind nameplate, 4 of them radial",
                xy=(-1.35, -4.5), ha="center", va="center", rotation=90,
                fontsize=7.4, color=ps.INK_SOFT)
    ax.annotate("demand on the east and south coasts",
                xy=(4.5, -10.7), ha="center", va="center", fontsize=7.4,
                color=ps.INK_SOFT)

    handles = [Line2D([], [], marker=s["marker"], ls="", color=s["colour"],
                      ms=7.5 if r != "load" else 5, label=s["label"])
               for r, s in ROLE_STYLE.items()]
    handles.append(Line2D([], [], marker="o", ls="", mfc="none",
                          mec=ps.CATEGORICAL[0], ms=10,
                          label="radial: one circuit out"))
    handles += [Line2D([], [], lw=s["lw"], ls=s["ls"], color=s["colour"],
                       label=s["label"]) for s in LINE_STYLE.values()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.02),
              ncol=2, fontsize=7.6, handletextpad=0.6, columnspacing=1.4,
              borderpad=0.2)
    ax.set_title("A toy grid with Ireland's shape\n"
                 "100 buses, 166 lines, 19 nearest-neighbour links absent",
                 fontsize=10.5)


def panel_injection(ax, g: G.ToyGrid, snsp: float) -> None:
    x, y = _xy(g)
    _draw_lines(ax, g, plain="#dcdbd6")
    P = G.dispatch(g, snsp)
    lim = float(np.abs(P).max())

    for role in ("load", "city", "wind", "hvdc", "sync"):
        m = g.role == role
        sc = ax.scatter(x[m], y[m], s=AREA_MIN + AREA_PER_PU * np.abs(P[m]),
                        marker=ROLE_STYLE[role]["marker"], c=P[m],
                        cmap=DIVERGING_R, vmin=-lim, vmax=lim,
                        edgecolors=ps.INK_MUTED, linewidths=0.5, zorder=4)
    cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("net injection (p.u.)", fontsize=8)
    cb.outline.set_visible(False)

    gen = P[P > 0].sum()
    ns = P[g.non_synchronous].sum()
    ax.set_title(f"Reference dispatch: SNSP {snsp * 100:.0f}%\n"
                 f"{ns:.1f} of {gen:.1f} p.u. from non-synchronous plant -- "
                 f"blue injects, red withdraws, shapes as on the left",
                 fontsize=10.5)


def _frame(ax) -> None:
    """The shared axis treatment: a map, so equal aspect and no furniture."""
    ax.set_aspect("equal")
    ax.set_xlim(-2.3, 10.4)
    ax.set_ylim(-11.5, 0.9)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def figure(g: G.ToyGrid, snsp: float = G.REFERENCE_SNSP) -> Path:
    ps.use()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6))
    panel_topology(axes[0], g)
    panel_injection(axes[1], g, snsp)
    for ax in axes:
        _frame(ax)
    fig.subplots_adjust(bottom=0.22)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "grid_layout.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_topology(g: G.ToyGrid) -> Path:
    """The left panel on its own: the network, with no dispatch on it.

    Same width per panel as the two-panel version, so marker areas and the
    two-column legend below the axes keep the proportions they were tuned at.
    """
    ps.use()
    fig, ax = plt.subplots(figsize=(6.9, 6.6))
    panel_topology(ax, g)
    _frame(ax)
    fig.subplots_adjust(bottom=0.22)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "grid_topology.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    g = G.load(HERE)          # the frozen grid, hash-checked -- never build()
    print(f"wrote {figure(g)}")
    print(f"wrote {figure_topology(g)}")


if __name__ == "__main__":
    main()

"""DBZ figures -> figures/dbz/."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

from bzgen import config

FIGS = config.ROOT / "figures" / "dbz"


def _save(fig, name: str, dpi: int = 130, **kw):
    FIGS.mkdir(parents=True, exist_ok=True)
    tmp = FIGS / f".{name}.tmp.png"
    fig.savefig(tmp, dpi=dpi, **kw)
    plt.close(fig)
    tmp.replace(FIGS / name)
    return FIGS / name


def dp_distribution(edges_n: pd.DataFrame, name: str = "dp_distribution.png"):
    """dp~ and J~ on intra-country vs cross-border edges (AC and HVDC separately for dp~)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    intra = edges_n[~edges_n.is_cross]
    xb = edges_n[edges_n.is_cross]
    hi = float(np.quantile(edges_n.dp_n, 0.999))
    bins = np.linspace(0, max(hi, 2.0), 60)
    ax = axes[0]
    ax.hist(np.clip(intra.dp_n, 0, bins[-1]), bins=bins, density=True, alpha=0.55,
            label=f"intra-country (n={len(intra)})", color="C0")
    ax.hist(np.clip(xb.dp_n, 0, bins[-1]), bins=bins, density=True, alpha=0.55,
            label=f"cross-border (n={len(xb)})", color="C3")
    ax.set_yscale("log"); ax.set_xlabel("Δp̃ (clipped at the 99.9th pct for display)")
    ax.set_ylabel("density"); ax.legend(fontsize=8); ax.set_title("Δp̃ by edge class")
    ax = axes[1]
    for lab, sub, c in (("intra", intra, "C0"), ("cross AC", xb[~xb.is_dc], "C3"),
                        ("cross HVDC", xb[xb.is_dc], "k")):
        if len(sub):
            x = np.sort(sub.dp_n.to_numpy())
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, label=f"{lab} (n={len(x)})")
    ax.set_xscale("symlog", linthresh=0.1); ax.set_xlabel("Δp̃"); ax.set_ylabel("ECDF")
    ax.legend(fontsize=8); ax.set_title("Δp̃ ECDF")
    ax = axes[2]
    ac_i = intra[~intra.is_dc].J_n
    ac_x = xb[~xb.is_dc].J_n
    b2 = np.linspace(0, max(float(np.quantile(edges_n[~edges_n.is_dc].J_n, 0.999)), 2.0), 60)
    ax.hist(ac_i, bins=b2, density=True, alpha=0.55, color="C0", label="intra AC")
    ax.hist(ac_x, bins=b2, density=True, alpha=0.55, color="C3", label="cross-border AC")
    ax.set_yscale("log"); ax.set_xlabel("J̃"); ax.legend(fontsize=8); ax.set_title("J̃ (AC) by edge class")
    fig.suptitle("European graph: node-averaged normalisation (country scales from "
                 "intra-country edges only)", fontsize=10)
    fig.tight_layout()
    return _save(fig, name)


def _zone_cells_all(buses, labels: pd.Series, members_of: dict):
    """Dissolved Voronoi cells per (country, static zone), computed once."""
    import geopandas as gpd
    from bzgen.plot import maps
    parts = []
    for c, members in members_of.items():
        lab = labels[labels.index.map(buses.cluster_country) == c]
        if lab.empty:
            continue
        cells = maps.zone_cells(buses, lab, maps.country_shape(members))
        cells["country"] = c
        parts.append(cells)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=maps.CRS)


def anchor_map(buses, lines, links, static_labels: pd.DataFrame, isolated: list, cid: str,
               name: str = "anchor_isolated.png", cells=None):
    """The anchor A (static map) with the buses that are isolated within their country
    highlighted.  Europe panel: zone colours as in figures/europe (per-country zone
    index), dashed = HVDC.  Zoom panels: only the bus's own anchor zone is coloured
    (everything else grey), red = the bus's lines, all of which cross a border, so
    on the European graph the bus is a one-bus fragment of its anchor zone."""
    import geopandas as gpd
    from matplotlib.collections import LineCollection
    from bzgen.plot import figures, maps
    lab = static_labels.set_index("bus").zone
    mem = figures.members_of(buses)
    cells = _zone_cells_all(buses, lab, mem) if cells is None else cells
    shapes = gpd.GeoSeries([maps.country_shape(m) for m in mem.values()], crs=maps.CRS)
    gb = maps._proj_buses(buses)
    xy = pd.DataFrame({"x": gb.geometry.x, "y": gb.geometry.y}, index=gb.index)
    cc = buses.cluster_country
    seg = lambda df: np.stack([xy.loc[df.bus0].to_numpy(), xy.loc[df.bus1].to_numpy()], axis=1)
    RED, GREY = "#d62728", "0.86"
    col = lambda z: maps.PALETTE[int(z) % len(maps.PALETTE)]

    def clean(ax):
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")

    iso_lines = {b: pd.concat([lines[(lines.bus0 == b) | (lines.bus1 == b)],
                               links[(links.bus0 == b) | (links.bus1 == b)]]) for b in isolated}
    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(3, 5, width_ratios=[1.3, 1.3, 1, 1, 1], wspace=0.08, hspace=0.42)
    ax = fig.add_subplot(gs[:, :2])
    maps.outlines().plot(ax=ax, facecolor="0.95", edgecolor="0.6", lw=0.3, zorder=0)
    cells.plot(ax=ax, color=[col(z) for z in cells.zone], alpha=0.6, lw=0.15,
               edgecolor="white", zorder=1)
    shapes.plot(ax=ax, facecolor="none", edgecolor="0.15", lw=0.6, zorder=3)
    ax.add_collection(LineCollection(seg(lines), colors="0.4", lw=0.12, zorder=2))
    ax.add_collection(LineCollection(seg(links), colors="k", lw=0.8, linestyles="--", zorder=4))
    for n, b in enumerate(isolated, 1):
        ax.add_collection(LineCollection(seg(iso_lines[b]), colors=RED, lw=1.6, zorder=6))
        ax.scatter(xy.at[b, "x"], xy.at[b, "y"], s=120, marker="*", c=RED,
                   edgecolors="k", linewidths=0.6, zorder=7)
        # cycle label sides so close pairs (SE/FI border: 2 and 6) do not overlap
        off = [(6, 4), (-22, 8), (6, -13), (-16, -13), (6, 4), (9, -15), (6, -13), (-16, -13)][(n - 1) % 8]
        ax.annotate(str(n), (xy.at[b, "x"], xy.at[b, "y"]), xytext=off,
                    textcoords="offset points", fontsize=9, fontweight="bold", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.3", lw=0.5))
    frame = gpd.GeoSeries([maps.box(-11, 35, 32, 71.5)], crs=4326).to_crs(maps.CRS).total_bounds
    ax.set_xlim(frame[0], frame[2]); ax.set_ylim(frame[1], frame[3])
    clean(ax)
    ax.set_title(f"Anchor A = static map {cid}\n★ = the {len(isolated)} buses isolated within "
                 "their country (every line crosses a border)", fontsize=10)
    for n, b in enumerate(isolated, 1):
        axz = fig.add_subplot(gs[(n - 1) // 3, 2 + (n - 1) % 3])
        others = sorted((set(iso_lines[b].bus0) | set(iso_lines[b].bus1)) - {b})
        bx, by = xy.at[b, "x"], xy.at[b, "y"]
        reach = max(float(np.hypot(xy.loc[others].x - bx, xy.loc[others].y - by).max()), 1.0)
        half = max(30_000.0, 1.35 * reach)
        z, c = lab[b], cc[b]
        own = (cells.country == c) & (cells.zone == z)
        cells[~own].plot(ax=axz, color=GREY, lw=0.3, edgecolor="white", zorder=1)
        cells[own].plot(ax=axz, color=col(z), alpha=0.8, lw=0, zorder=1)
        shapes.plot(ax=axz, facecolor="none", edgecolor="0.15", lw=0.9, zorder=3)
        axz.add_collection(LineCollection(seg(lines), colors="0.55", lw=0.5, zorder=2))
        axz.add_collection(LineCollection(seg(iso_lines[b]), colors=RED, lw=2.2, zorder=6))
        near = xy[(np.abs(xy.x - bx) < half) & (np.abs(xy.y - by) < half)]
        mine = (near.index.map(cc) == c) & (near.index.map(lab) == z)
        axz.scatter(near.x[~mine], near.y[~mine], s=9, c="0.45", lw=0, zorder=5)
        axz.scatter(near.x[mine], near.y[mine], s=14, c=col(z), edgecolors="k",
                    linewidths=0.3, zorder=5)
        axz.scatter(bx, by, s=280, marker="*", c=col(z), edgecolors=RED, linewidths=1.6, zorder=7)
        axz.set_xlim(bx - half, bx + half); axz.set_ylim(by - half, by + half)
        clean(axz)
        nb = ", ".join(sorted({cc[u] for u in others}))
        axz.set_title(f"{n}. {c} zone {int(z)}; its lines go only to {nb}\n"
                      f"{b}\n{2 * half / 1000:.0f} km across", fontsize=7.5)
    axt = fig.add_subplot(gs[2, 4])
    axt.axis("off")
    axt.text(0, 0.95, "Zoom panels\n\ncoloured: the bus's own anchor zone\n(static label = "
             "zone of its nearest\nsame-country bus)\ngrey: every other zone\nred: the bus's "
             "lines (all cross-border)\n\nOn the European graph each ★ is\na one-bus fragment "
             "of its anchor\nzone: Σ(C_s − 1) = 8 for A.", fontsize=8.5, va="top")
    return _save(fig, name, dpi=140, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# stage 2
# --------------------------------------------------------------------------- #

_CELLS: dict = {}


def bus_cells(buses: pd.DataFrame):
    """One Voronoi cell per bus over the whole of Europe, clipped to the (simplified)
    union of the in-scope outlines; computed once per process (display only)."""
    if "cells" in _CELLS:
        return _CELLS["cells"]
    import geopandas as gpd
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union, voronoi_diagram
    from bzgen.plot import figures, maps
    mem = figures.members_of(buses)
    shape = unary_union([maps.country_shape(m) for m in mem.values()]).simplify(1500)
    gb = maps._proj_buses(buses)
    vor = voronoi_diagram(MultiPoint(list(gb.geometry)), envelope=shape.buffer(50_000).envelope)
    cells = gpd.GeoDataFrame(geometry=list(vor.geoms), crs=maps.CRS)
    j = gpd.sjoin(gpd.GeoDataFrame({"bus": gb.index}, geometry=gb.geometry.values, crs=maps.CRS),
                  cells, predicate="within", how="left")
    cells = cells.loc[j.index_right.to_numpy()].reset_index(drop=True)
    cells["bus"] = j.bus.to_numpy()
    cells["geometry"] = cells.geometry.intersection(shape)
    _CELLS["cells"] = cells.set_index("bus")
    _CELLS["shapes"] = gpd.GeoSeries([maps.country_shape(m) for m in mem.values()], crs=maps.CRS)
    return _CELLS["cells"]


def anchor_colours(g, A: np.ndarray, K: int) -> np.ndarray:
    """Greedy colouring of the anchor-zone adjacency graph (colour index per A-zone)."""
    import networkx as nx
    src, dst = g.undirected()[:2]
    Z = nx.Graph()
    Z.add_nodes_from(range(K))
    Z.add_edges_from({(int(a), int(b)) for a, b in zip(A[src], A[dst]) if a != b})
    col = nx.greedy_color(Z, strategy="largest_first")
    return np.array([col[z] for z in range(K)])


def zone_map(W: dict, lab: pd.DataFrame, lam: float, row: dict, out_dir=None):
    """B for one lambda_rigid: zones (each B-zone takes the colour of its matched
    A-zone) and the nodes that changed zone relative to A."""
    from scipy.optimize import linear_sum_assignment
    from bzgen.dbz.anchor import contingency
    from bzgen.plot import maps
    g, A, K = W["g"], W["anchor"]["A"], W["K"]
    buses = W["buses"]
    cells = bus_cells(buses)
    B = lab.zone.to_numpy()
    colA = anchor_colours(g, A, K)
    t = contingency(B, A, K, K)
    r, c = linear_sum_assignment(t, maximize=True)
    match = np.empty(K, np.int64)
    match[r] = c
    colB = colA[match]
    changed = match[B] != A
    pal = maps.PALETTE
    fig, axes = plt.subplots(1, 2, figsize=(17, 9.5))
    for ax, title in zip(axes, ("zones B", "nodes whose zone differs from A (after matching)")):
        maps.outlines().plot(ax=ax, facecolor="0.95", edgecolor="0.6", lw=0.3, zorder=0)
        ax.set_title(title, fontsize=10)
    cb = cells.loc[lab.bus]
    cb = cb.assign(zone=B)
    d = cb.dissolve("zone").reset_index()
    d.plot(ax=axes[0], color=[pal[colB[z] % len(pal)] for z in d.zone], alpha=0.7, lw=0.4,
           edgecolor="white", zorder=1)
    d.boundary.plot(ax=axes[0], color="0.25", lw=0.35, zorder=2)
    ch = cb[changed]
    cb[~changed].plot(ax=axes[1], color="0.82", lw=0, zorder=1)
    if len(ch):
        ch.plot(ax=axes[1], color="#d62728", lw=0, zorder=2)
    dA = cells.loc[lab.bus].assign(zone=A).dissolve("zone").reset_index()
    dA.boundary.plot(ax=axes[1], color="0.35", lw=0.3, zorder=3)
    for ax in axes:
        _CELLS["shapes"].plot(ax=ax, facecolor="none", edgecolor="k", lw=0.7, zorder=4)
        frame = gpd_frame()
        ax.set_xlim(frame[0], frame[2]); ax.set_ylim(frame[1], frame[3])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")
    axes[1].text(0.01, 0.01, "grey lines: anchor-zone boundaries; black: national borders",
                 transform=axes[1].transAxes, fontsize=8)
    fig.suptitle(f"DBZ λ_rigid = {lam:g}: transfer distance from A = "
                 f"{row['transfer_distance']:.0f} of {g.n} nodes "
                 f"({row['transfer_share']:.1%}); {row['n_zones_multinational']} multinational zones; "
                 f"physical objective {row['physical']:.1f}, rigidity {row['rigid']:.2f}",
                 fontsize=11)
    fig.tight_layout()
    name = f"zones_lr{lam:g}.png"
    return _save_to(fig, name, None if out_dir is None else Path(out_dir) / "figures", dpi=120)


def gpd_frame():
    import geopandas as gpd
    from bzgen.plot import maps
    return gpd.GeoSeries([maps.box(-11, 35, 32, 71.5)], crs=4326).to_crs(maps.CRS).total_bounds


def pareto(df: pd.DataFrame, restarts: dict, anchor_phys: float, N: int, fig_dir=None,
           name="pareto.png"):
    """Physical objective (Potts + lambda_b balance) vs transfer distance from A."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for lam, r in restarts.items():
        ax.scatter(r.transfer_distance, r.physical, s=14, color="0.7", zorder=1, lw=0)
    ax.plot(df.transfer_distance, df.physical, "-", color="#4e79a7", lw=2, zorder=2)
    ax.scatter(df.transfer_distance, df.physical, s=60, color="#4e79a7", zorder=3,
               edgecolors="white", linewidths=1.5, label="best restart per λ_rigid")
    for _, r in df.iterrows():
        ax.annotate(f"λ={r.lambda_rigid:g}", (r.transfer_distance, r.physical),
                    xytext=(6, 4), textcoords="offset points", fontsize=8, color="0.25")
    ax.scatter([0], [anchor_phys], marker="*", s=180, color="#e15759", zorder=4,
               edgecolors="k", linewidths=0.5, label="anchor A (static map)")
    ax.scatter([], [], s=14, color="0.7", label="other restarts")
    ax.set_xlabel(f"transfer distance from A (nodes that changed zone, of N = {N})")
    ax.set_ylabel("physical objective  Σ w δ + λ_b Σ hinge  (lower is better)")
    ax.set_title("What stability costs: DBZ Pareto front over λ_rigid", fontsize=11)
    ax.grid(color="0.9", lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save_to(fig, name, fig_dir)


def acceptance(traces: dict, fig_dir=None, name="acceptance.png"):
    """Single-flip acceptance rate per temperature step (best restart), per lambda."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cols = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1",
            "#ff9da7"]
    for n, (lam, tr) in enumerate(sorted(traces.items())):
        a = np.clip(tr[:-1, 4], 1e-5, None)
        ax.plot(np.arange(len(a)), a, lw=2, color=cols[n % len(cols)], label=f"λ_rigid={lam:g}")
    ax.axhline(0.01, color="0.4", ls="--", lw=1)
    ax.text(0.5, 0.0107, "1 %", fontsize=8, color="0.3")
    ax.set_yscale("log")
    ax.set_xlabel("temperature step (T0 → T0 · t_final_ratio)")
    ax.set_ylabel("single-flip acceptance rate")
    ax.set_title("Acceptance with λ_c pinned at the contiguity guarantee", fontsize=11)
    ax.grid(color="0.9", lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    return _save_to(fig, name, fig_dir)


def _save_to(fig, name, fig_dir, **kw):
    global FIGS
    if fig_dir is None:
        return _save(fig, name, **kw)
    old = FIGS
    FIGS = Path(fig_dir)
    try:
        return _save(fig, name, **kw)
    finally:
        FIGS = old

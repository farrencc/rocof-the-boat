"""Frames -> GIF / MP4 of a country's zone map forming during annealing.

    python -m bzgen.ndbz.animate --country IE --scenario peak_demand --lr 1

Each frame: the zone map (bus Voronoi cells from ``bzgen.plot.maps``, so it looks
like the static figures); the temperature and lambda_rigid; the energy components
(Potts, lambda_b * balance, lambda_rigid * H_rigid; the contiguity excess as text:
it is 0 throughout with lambda_c at the guarantee) as traces with a cursor; and the
transfer distance from A.  A last, held frame shows the converged map beside A.

Colour stability: zone labels permute freely while annealing, so every frame's zones
are matched to A's zones (``scipy.optimize.linear_sum_assignment`` on the contingency
table, ties broken towards the previous frame's colours) and coloured by the matched
A-zone, in A's static palette colour.  Buses outside their A-zone are marked (heavy
outline, hatched cell), so the eye follows *what moved*.

``imageio`` (+ ``imageio-ffmpeg`` for MP4) is optional: the sweep runs without it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from scipy.optimize import linear_sum_assignment

from bzgen.ndbz.anneal import META
from bzgen.ndbz.plots import INK, INK2, SERIES, _style


# --------------------------------------------------------------------------- #
# colour matching
# --------------------------------------------------------------------------- #

def contingency(B: np.ndarray, A: np.ndarray, k: int) -> np.ndarray:
    n = np.zeros((k, k), np.int64)
    np.add.at(n, (B, A), 1)
    return n


def transfer_distance(B: np.ndarray, A: np.ndarray, k: int) -> int:
    """n - maximum one-to-one matching of B-zones to A-zones: nodes that changed zone."""
    n = contingency(B, A, k)
    r, c = linear_sum_assignment(-n)
    return int(len(B) - n[r, c].sum())


def match_colours(B: np.ndarray, A: np.ndarray, k: int, prev: np.ndarray | None = None) -> np.ndarray:
    """Per node: the A-zone its B-zone is matched to (maximum-overlap assignment).

    Ties between equally good assignments are broken towards ``prev`` (the previous
    frame's per-node colours): the secondary term is scaled below 1 so it never
    overrides the overlap itself.  Identical partitions therefore get identical
    colours, whatever their label indices."""
    n = contingency(B, A, k).astype(float)
    if prev is not None:
        eps = 0.5 / (len(B) + 1)
        n = n + eps * contingency(B, prev, k)
    r, c = linear_sum_assignment(-n)
    m = np.empty(k, np.int64)
    m[r] = c
    return m[B]


def colour_sequence(frames: np.ndarray, A: np.ndarray, k: int) -> np.ndarray:
    out = np.empty_like(frames, dtype=np.int64)
    prev = A
    for f in range(len(frames)):
        prev = out[f] = match_colours(frames[f].astype(np.int64), A, k, prev)
    return out


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def _path(geom) -> MPath:
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    verts, codes = [], []
    for p in polys:
        if p.is_empty or p.geom_type != "Polygon":
            continue
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            verts.append(xy)
            codes.append(np.r_[MPath.MOVETO, np.full(len(xy) - 2, MPath.LINETO), MPath.CLOSEPOLY])
    if not verts:
        return MPath(np.zeros((1, 2)))
    return MPath(np.vstack(verts), np.concatenate(codes))


def bus_cells(buses: pd.DataFrame, all_buses: list, members: list):
    """Voronoi cell per bus (``maps.zone_cells`` with every bus its own zone), the
    projected bus positions and the country outline."""
    from bzgen.plot import maps
    shape = maps.country_shape(members)
    lab = pd.Series(np.arange(len(all_buses)), index=all_buses)
    cells = maps.zone_cells(buses, lab, shape).set_index("zone").sort_index()
    gb = maps._proj_buses(buses.loc[all_buses])
    xy = np.c_[gb.geometry.x, gb.geometry.y]
    geoms = cells.geometry.reindex(np.arange(len(all_buses), dtype=float))
    paths = [MPath(np.zeros((1, 2))) if g is None or (isinstance(g, float) and np.isnan(g)) else _path(g)
             for g in geoms]
    return paths, xy, shape


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def render(frames: np.ndarray, meta: np.ndarray, nodes: list, host: dict, A: np.ndarray,
           zone_ids: list, k: int, buses: pd.DataFrame, lines: pd.DataFrame, members: list,
           title: str, lam_rigid: float, lam_b: float, out_base: Path, fps: float = 10.0,
           hold_s: float = 1.0, dpi: int = 80, mp4: bool = True) -> dict:
    """Write ``<out_base>.gif`` (and ``.mp4``).  Returns paths and per-frame diagnostics."""
    import imageio.v3 as iio
    from bzgen.plot import maps

    all_buses = list(nodes) + list(host)
    pos = {b: i for i, b in enumerate(nodes)}
    ext = np.array([pos[b] for b in nodes] + [pos[h] for h in host.values()])   # bus -> node
    col = colour_sequence(frames, A, k)
    moved = col != A[None, :]
    td = np.array([transfer_distance(f.astype(np.int64), A, k) for f in frames])
    paths, xy, shape = bus_cells(buses, all_buses, members)
    pal = np.array([matplotlib.colors.to_rgba(maps.PALETTE[int(z) % len(maps.PALETTE)])
                    for z in zone_ids])
    li = lines[lines.bus0.isin(all_buses) & lines.bus1.isin(all_buses)]
    bi = {b: i for i, b in enumerate(all_buses)}
    seg = np.stack([xy[[bi[b] for b in li.bus0]], xy[[bi[b] for b in li.bus1]]], axis=1) if len(li) else None

    F = len(frames)
    x = np.arange(F)
    potts, bal, rig = meta[:, 2], lam_b * meta[:, 4], lam_rigid * meta[:, 5]
    T = meta[:, 0]

    def map_axes(ax, colours, moved_nodes, subtitle):
        cc = pal[colours[ext]].copy()
        cc[:, 3] = 0.6
        pc = PatchCollection([PathPatch(p) for p in paths], facecolors=cc, edgecolors="none", zorder=1)
        ax.add_collection(pc)
        mv = moved_nodes[ext]
        if mv.any():
            ax.add_collection(PatchCollection([PathPatch(paths[i]) for i in np.flatnonzero(mv)],
                                              facecolor="none", edgecolor=INK, hatch="////",
                                              lw=0.0, alpha=0.55, zorder=2))
        if seg is not None:
            from matplotlib.collections import LineCollection
            ax.add_collection(LineCollection(seg, colors="0.35", lw=0.5, zorder=3))
        import geopandas as gpd
        gpd.GeoSeries([shape], crs=maps.CRS).plot(ax=ax, facecolor="none", edgecolor="0.3", lw=0.7, zorder=4)
        ax.scatter(xy[~mv, 0], xy[~mv, 1], s=10, c=pal[colours[ext]][~mv], edgecolors="k",
                   linewidths=0.3, zorder=5)
        if mv.any():
            ax.scatter(xy[mv, 0], xy[mv, 1], s=34, c=pal[colours[ext]][mv], edgecolors=INK,
                       linewidths=1.6, zorder=6)
        minx, miny, maxx, maxy = shape.bounds
        pad = 0.04 * max(maxx - minx, maxy - miny)
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(subtitle, fontsize=9, color=INK, loc="left")

    def canvas(fig):
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    images = []
    n_temps = int(np.sum(T[1:] > 0))
    for f in range(F):
        fig = plt.figure(figsize=(11.0, 6.0), dpi=dpi)
        ax = fig.add_axes([0.01, 0.04, 0.50, 0.84])
        phase = ("initial state (graph Voronoi)" if f == 0 else
                 f"temperature {f}/{n_temps}" if T[f] > 0 else "quench (T = 0)")
        map_axes(ax, col[f], moved[f], f"{phase}   T = {T[f]:.3g}   accept = {meta[f, 6]:.1%}")
        a1 = fig.add_axes([0.58, 0.55, 0.40, 0.33])
        a2 = fig.add_axes([0.58, 0.12, 0.40, 0.30])
        for a in (a1, a2):
            _style(a)
        for y, lab, c in ((potts, "Potts", SERIES[0]), (bal, "λ_b · balance", SERIES[1]),
                          (rig, "λ_rigid · H_rigid", SERIES[2])):
            a1.plot(x, y, color=c, lw=1.6, label=lab)
            a1.plot([f], [y[f]], "o", color=c, ms=5, mec="white", mew=1.0)
        a1.axvline(f, color=INK2, lw=0.7, ls=":")
        a1.legend(fontsize=6.5, frameon=False, loc="upper right")
        a1.set_title(f"energy terms (Σ(C−1) = {meta[f, 3]:.0f}: contiguous)", fontsize=8,
                     color=INK, loc="left")
        a1.text(0.0, -0.12, f"Potts {potts[f]:.1f} · balance {bal[f]:.2f} · rigidity {rig[f]:.2f}",
                transform=a1.transAxes, fontsize=7, color=INK2, va="top")
        a2.plot(x, td, color=SERIES[0], lw=1.6)
        a2.plot([f], [td[f]], "o", color=SERIES[0], ms=5, mec="white", mew=1.0)
        a2.axvline(f, color=INK2, lw=0.7, ls=":")
        a2.set_title(f"transfer distance from A: {td[f]} of {len(A)} buses", fontsize=8,
                     color=INK, loc="left")
        a2.set_xlabel("frame (temperature steps, then quench)", fontsize=7, color=INK2)
        a2.set_ylim(bottom=0)
        fig.suptitle(f"{title}    λ_rigid = {lam_rigid:g}", fontsize=11, color=INK, x=0.01,
                     ha="left", y=0.97)
        images.append(canvas(fig))
        plt.close(fig)
    # held comparison frame: converged map beside A
    fig = plt.figure(figsize=(11.0, 6.0), dpi=dpi)
    ax1 = fig.add_axes([0.01, 0.04, 0.48, 0.84])
    ax2 = fig.add_axes([0.51, 0.04, 0.48, 0.84])
    map_axes(ax1, A, np.zeros_like(A, bool), "A: status-quo static map")
    map_axes(ax2, col[-1], moved[-1], f"converged B: {td[-1]} buses moved (hatched)")
    fig.suptitle(f"{title}    λ_rigid = {lam_rigid:g}", fontsize=11, color=INK, x=0.01, ha="left",
                 y=0.97)
    last = canvas(fig)
    plt.close(fig)
    hold = max(1, int(round(hold_s * fps)))
    seq = images + [images[-1]] * hold + [last] * (2 * hold)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    gif = out_base.parent / (out_base.name + ".gif")
    tmp = gif.with_name("." + gif.name + ".tmp")
    iio.imwrite(tmp, np.stack(seq), extension=".gif", duration=int(1000 / fps), loop=0)
    tmp.replace(gif)
    out = {"gif": gif, "n_frames": F, "transfer": td, "colours": col}
    if mp4:
        try:
            import imageio.v2 as iio2
            mp = out_base.parent / (out_base.name + ".mp4")
            tmp = mp.with_name("." + mp.name + ".tmp.mp4")
            with iio2.get_writer(tmp, fps=fps, codec="libx264", macro_block_size=2,
                                 ffmpeg_log_level="error") as w:
                for im in seq:
                    w.append_data(im)
            tmp.replace(mp)
            out["mp4"] = mp
        except Exception as e:          # optional dependency
            out["mp4_error"] = repr(e)
    return out


# --------------------------------------------------------------------------- #
# command line: animate one (country, scenario, lambda_rigid) cell
# --------------------------------------------------------------------------- #

def animate_cell(ctx, c: str, scenario: str, lam_rigid: float, restart="best", clip_handling=None,
                 fps: float | None = None, log=print) -> dict:
    """Run the cell's restarts without capture, re-run the chosen one with capture
    (bit-identical, tests/test_ndbz.py) and render it."""
    from bzgen.ndbz import figures_dir
    from bzgen.ndbz import sweep as SW
    from bzgen.ndbz.anneal import anneal_ndbz, country_seed
    cfg = ctx.cfg
    a = cfg["anneal"]
    an = cfg.get("animate", {})
    g, nodes, host, anc, floor, _ = SW.country_problem(ctx, c, scenario, clip_handling=clip_handling)
    kw = dict(lam_b=ctx.params["lambda_b"], floor=floor, n_temps=a["n_temps"],
              sweeps_per_temp=a["sweeps_per_temp"], t_final_ratio=a["t_final_ratio"],
              quench_sweeps=a["quench_sweeps"], T0_band=tuple(a["T0_band"]))
    seeds = [country_seed(c, a["seed"], cfg["rigidity"]["seed_offset"], r) for r in range(a["restarts"])]
    if restart == "best":
        E = [anneal_ndbz(g, anc.k, anc.A, anc.cap, anc.Z, lam_rigid, seed=s, **kw)["energy"] for s in seeds]
        r = int(np.argmin(E))
    else:
        r = int(restart)
    res = anneal_ndbz(g, anc.k, anc.A, anc.cap, anc.Z, lam_rigid, seed=seeds[r], capture=True,
                      quench_frames=an.get("quench_frames", 20), **kw)
    lines = pd.read_csv(SW.INTERIM / "lines.csv", index_col=0)
    mem = sorted(ctx.buses[ctx.buses.cluster_country == c].country.unique())
    tag = f"{c}_{scenario}{'' if not clip_handling else '_' + clip_handling}_lr{lam_rigid:g}"
    out = render(res["frames"], res["frame_meta"], nodes, host, anc.A, anc.zone_ids, anc.k,
                 ctx.buses, lines, mem, f"{c} · {scenario}", lam_rigid, ctx.params["lambda_b"],
                 figures_dir(cfg) / "anim" / tag, fps=fps or an.get("fps", 10.0),
                 hold_s=an.get("hold_s", 1.0), dpi=an.get("dpi", 80), mp4=an.get("mp4", True))
    out.update(restart=r, result=res)
    log(f"{tag}: restart {r}, E = {res['energy']:.2f}, transfer {out['transfer'][-1]}, "
        f"{out['n_frames']} frames -> {out['gif'].name}")
    return out


def main():
    from bzgen.ndbz import load_config
    from bzgen.ndbz import sweep as SW
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="IE")
    ap.add_argument("--scenario", nargs="+", default=["peak_demand"])
    ap.add_argument("--lr", nargs="+", type=float, default=[1.0])
    ap.add_argument("--restart", default="best")
    args = ap.parse_args()
    cfg = load_config()
    ctx = SW.load_context(cfg)
    for sc in args.scenario:
        for lr in args.lr:
            animate_cell(ctx, args.country, sc, lr, args.restart)


if __name__ == "__main__":
    main()

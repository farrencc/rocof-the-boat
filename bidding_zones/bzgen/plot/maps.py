"""Geographic figures: zone maps, stitched European map, nodal price maps.

Zones are drawn as the union of the Voronoi cells of their buses, clipped to
the (cluster-)country outline (dissolved GISCO NUTS 2024 level-3 polygons; BA
from GISCO countries).  Cells are a display device only — nothing is computed
from them.  AC lines grey, HVDC links dashed black; buses as small dots.
"""

from __future__ import annotations

import functools

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from shapely.geometry import MultiPoint, box
from shapely.ops import unary_union, voronoi_diagram

from bzgen.data import regional

PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1",
           "#ff9da7", "#9c755f", "#bab0ac", "#86bcb6", "#d37295", "#a0cbe8", "#ffbe7d"]
CRS = 3035   # ETRS89-LAEA, equal area, for drawing


def tint(hex_color: str, a: float) -> str:
    """Colour as drawn at opacity ``a`` on white (how the zone maps render PALETTE)."""
    import matplotlib.colors as mc
    rgb = np.array(mc.to_rgb(hex_color))
    return mc.to_hex(a * rgb + (1 - a))


# chart colours matching the zone maps: fills at map opacity, lines a little stronger
MAP_ALPHA = 0.65
BLUE, SAND, GREEN, TEAL = (tint(c, MAP_ALPHA) for c in ("#4e79a7", "#f28e2b", "#59a14f", "#76b7b2"))
BLUE_LINE, SAND_LINE, GREEN_LINE = (tint(c, a) for c, a in zip( ("#4e79a7", "#f28e2b", "#59a14f"), (0.9, 0.75, 0.9)))
GREY = "#bab0ac"
INK, INK2 = "#0b0b0b", "#52514e"
ALPHA_LABEL = "Connectivity coupling α"
DERATING_LABEL = "Zone trading capacity (share of line capacity)"


@functools.lru_cache(maxsize=1)
def outlines() -> gpd.GeoDataFrame:
    """Country outlines for drawing: GISCO NUTS 2024 level 0 at 1:10M (display only)."""
    from bzgen.data.cache import fetch
    n0 = gpd.read_file(fetch(regional.NUTS0_URL, "gisco/NUTS_RG_10M_2024_4326_LEVL_0.geojson",
                             key="gisco_nuts2024_l0_10m",
                             notes="GISCO NUTS 2024 level 0, 1:10M, EPSG:4326 (maps only)"))
    n0["cc"] = n0.CNTR_CODE.replace({"EL": "GR"})
    g = n0.set_index("cc")[["geometry"]]
    cn = regional.country_outlines()
    if "BA" not in g.index:
        ba = cn[cn.CNTR_ID == "BA"][["geometry"]].set_index(pd.Index(["BA"]))
        g = pd.concat([g, ba])
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=4326).to_crs(CRS)
    # drop far-away overseas territories (FR DOM, ES Canarias, PT Azores/Madeira)
    frame = gpd.GeoSeries([box(-12, 34, 35, 72)], crs=4326).to_crs(CRS).iloc[0]
    g["geometry"] = g.geometry.intersection(frame)
    return g


def country_shape(members: list[str]):
    o = outlines()
    return unary_union([o.loc[c, "geometry"] for c in members if c in o.index])


def _proj_buses(buses: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(buses, geometry=gpd.points_from_xy(buses.x, buses.y),
                            crs=4326).to_crs(CRS)


def zone_cells(buses: pd.DataFrame, labels: pd.Series, shape) -> gpd.GeoDataFrame:
    """Dissolved Voronoi cells per zone, clipped to shape."""
    gb = _proj_buses(buses.loc[labels.index])
    pts = MultiPoint(list(gb.geometry))
    env = shape.buffer(50_000).envelope
    vor = voronoi_diagram(pts, envelope=env)
    cells = gpd.GeoDataFrame(geometry=list(vor.geoms), crs=CRS)
    j = gpd.sjoin(gpd.GeoDataFrame({"zone": labels.values}, geometry=gb.geometry.values, crs=CRS),
                  cells, predicate="within", how="left")
    cells["zone"] = np.nan
    cells.loc[j.index_right.dropna().astype(int).values, "zone"] = j.zone.values[j.index_right.notna()]
    cells = cells.dropna(subset=["zone"])
    cells["geometry"] = cells.geometry.intersection(shape)
    return cells.dissolve("zone").reset_index()


def draw_country(ax, buses, lines, links, labels: pd.Series, members, title=None,
                 show_lines=True):
    shape = country_shape(members)
    gpd.GeoSeries([shape], crs=CRS).plot(ax=ax, facecolor="none", edgecolor="0.3", lw=0.6, zorder=3)
    cells = zone_cells(buses, labels, shape)
    for _, r in cells.iterrows():
        gpd.GeoSeries([r.geometry], crs=CRS).plot(ax=ax, color=PALETTE[int(r.zone) % len(PALETTE)],
                                                   alpha=0.55, lw=0, zorder=1)
    gb = _proj_buses(buses)
    xy = pd.DataFrame({"x": gb.geometry.x, "y": gb.geometry.y}, index=gb.index)
    inside = set(labels.index)
    if show_lines:
        l = lines[lines.bus0.isin(inside) & lines.bus1.isin(inside)]
        seg = np.stack([xy.loc[l.bus0].to_numpy(), xy.loc[l.bus1].to_numpy()], axis=1)
        ax.add_collection(LineCollection(seg, colors="0.35", lw=0.35, zorder=2))
    k = links[links.bus0.isin(inside) & links.bus1.isin(inside)]
    if len(k):
        seg = np.stack([xy.loc[k.bus0].to_numpy(), xy.loc[k.bus1].to_numpy()], axis=1)
        ax.add_collection(LineCollection(seg, colors="k", lw=1.2, linestyles="--", zorder=4))
    p = xy.loc[labels.index]
    ax.scatter(p.x, p.y, s=3, c=[PALETTE[int(z) % len(PALETTE)] for z in labels.values],
               edgecolors="k", linewidths=0.15, zorder=5)
    minx, miny, maxx, maxy = shape.bounds
    pad = 0.04 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=8)


def europe_map(ax, buses, lines, links, labels: pd.Series, members_of: dict, title=None):
    o = outlines()
    o.plot(ax=ax, facecolor="0.95", edgecolor="0.6", lw=0.3, zorder=0)
    for c, members in members_of.items():
        lab = labels[labels.index.map(buses.cluster_country) == c]
        if lab.empty:
            continue
        shape = country_shape(members)
        cells = zone_cells(buses, lab, shape)
        for _, r in cells.iterrows():
            gpd.GeoSeries([r.geometry], crs=CRS).plot(
                ax=ax, color=PALETTE[int(r.zone) % len(PALETTE)], alpha=0.65, lw=0.15,
                edgecolor="white", zorder=1)
        gpd.GeoSeries([shape], crs=CRS).plot(ax=ax, facecolor="none", edgecolor="0.15",
                                             lw=0.5, zorder=3)
    gb = _proj_buses(buses)
    xy = pd.DataFrame({"x": gb.geometry.x, "y": gb.geometry.y}, index=gb.index)
    k = links
    seg = np.stack([xy.loc[k.bus0].to_numpy(), xy.loc[k.bus1].to_numpy()], axis=1)
    ax.add_collection(LineCollection(seg, colors="k", lw=0.8, linestyles="--", zorder=4))
    frame = gpd.GeoSeries([box(-11, 35, 32, 71.5)], crs=4326).to_crs(CRS).total_bounds
    ax.set_xlim(frame[0], frame[2]); ax.set_ylim(frame[1], frame[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=10)


def price_map(ax, buses, lines, values: pd.Series, cmap="viridis", label="", vmin=None, vmax=None,
              edge_values: pd.Series | None = None):
    o = outlines()
    o.plot(ax=ax, facecolor="0.96", edgecolor="0.6", lw=0.3, zorder=0)
    gb = _proj_buses(buses)
    xy = pd.DataFrame({"x": gb.geometry.x, "y": gb.geometry.y}, index=gb.index)
    if edge_values is not None:
        l = lines.loc[edge_values.index]
        seg = np.stack([xy.loc[l.bus0].to_numpy(), xy.loc[l.bus1].to_numpy()], axis=1)
        lc = LineCollection(seg, array=edge_values.to_numpy(), cmap="magma_r", lw=0.8, zorder=2)
        ax.add_collection(lc)
    else:
        seg = np.stack([xy.loc[lines.bus0].to_numpy(), xy.loc[lines.bus1].to_numpy()], axis=1)
        ax.add_collection(LineCollection(seg, colors="0.7", lw=0.25, zorder=1))
    v = values.reindex(buses.index)
    sc = ax.scatter(xy.x, xy.y, c=v, s=4, cmap=cmap, vmin=vmin, vmax=vmax, zorder=3, lw=0)
    frame = gpd.GeoSeries([box(-11, 35, 32, 71.5)], crs=4326).to_crs(CRS).total_bounds
    ax.set_xlim(frame[0], frame[2]); ax.set_ylim(frame[1], frame[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    return sc

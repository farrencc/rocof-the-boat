"""Topology: scope filter + an emulation of PyPSA-Eur's ``simplify_network``.

Stages (each counted in the returned report):

1. **Scope**: keep buses in ``scope.countries``; drop GB (incl. NI), UA, MD and
   every line/link touching them.
2. **HVDC**: contract AC-DC converters (DC bus -> its AC bus), so each HVDC link
   joins two AC buses.  Links whose ends cannot be mapped to an in-scope AC
   bus are dropped (e.g. those landing in GB).
3. **Single voltage level**: contract transformers (union-find over the buses
   they join; representative = highest-voltage bus of the group), and convert
   each line's series reactance to a 380 kV equivalent ``x (380/v)^2`` — the
   per-unit-preserving transformation PyPSA-Eur applies (lines keep their
   ``s_nom`` in MVA).  Lines that become self-loops are dropped.
4. **Islands**: connected components (AC lines + HVDC links).  Components with
   fewer than ``min_component`` buses are removed; their load and generation
   are later assigned to the nearest surviving bus of the same country.
5. **Stubs**: iteratively absorb degree-1 buses whose only connection is an
   AC line to a bus in the *same* country (PyPSA-Eur ``remove_stubs`` with
   ``remove_stubs_across_borders: false``).

Parallel lines are kept as separate PyPSA lines for the OPF (so each keeps its
own rating); they are merged only in the clustering graph (susceptances add).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bzgen.data.cache import CACHE

R = dict(quotechar="'")


class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _components(bus_ids, edges: list[tuple[str, str]]) -> dict[str, int]:
    uf = UnionFind(bus_ids)
    for a, b in edges:
        uf.union(a, b)
    roots = {b: uf.find(b) for b in bus_ids}
    lab = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {b: lab[r] for b, r in roots.items()}


def build_topology(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """Return (buses, lines, links, report_rows)."""
    d = CACHE / "osm"
    buses = pd.read_csv(d / "buses.csv", **R).set_index("bus_id")
    lines = pd.read_csv(d / "lines.csv", **R).set_index("line_id")
    links = pd.read_csv(d / "links.csv", **R).set_index("link_id")
    conv = pd.read_csv(d / "converters.csv", **R).set_index("converter_id")
    trafo = pd.read_csv(d / "transformers.csv", **R).set_index("transformer_id")
    rep = []

    def log(stage, **kw):
        rep.append({"stage": stage, **kw})

    log("raw", buses=len(buses), ac_buses=int((buses.dc == "f").sum()), lines=len(lines),
        links=len(links), converters=len(conv), transformers=len(trafo))

    if cfg["network"]["drop_under_construction"]:
        lines = lines[lines.under_construction == "f"]
        links = links[links.under_construction == "f"]

    # 1. scope
    scope = set(cfg["scope"]["countries"])
    out_c = buses.loc[~buses.country.isin(scope), "country"].value_counts().to_dict()
    keep = buses.country.isin(scope)
    buses = buses[keep]
    inb = lines.bus0.isin(buses.index) & lines.bus1.isin(buses.index)
    log("scope", buses=len(buses), lines=int(inb.sum()),
        note=f"dropped buses by country {out_c}; dropped {int((~inb).sum())} lines touching them")
    lines = lines[inb]

    # 2. HVDC: DC bus -> AC bus via converters
    dc2ac = {}
    for _, c in conv.iterrows():
        if c.bus0 in buses.index and c.bus1 in buses.index:
            dc2ac[c.bus0] = c.bus1
    links = links.copy()
    links["bus0"] = links.bus0.map(lambda b: dc2ac.get(b, b))
    links["bus1"] = links.bus1.map(lambda b: dc2ac.get(b, b))
    acb = buses.index[buses.dc == "f"]
    okl = links.bus0.isin(acb) & links.bus1.isin(acb) & (links.bus0 != links.bus1)
    log("hvdc", links=int(okl.sum()),
        note=f"dropped links (end outside scope or unmapped DC bus): {sorted(links.index[~okl])}")
    links = links[okl]
    dc_lines = lines.bus0.isin(buses.index[buses.dc == "t"]) | lines.bus1.isin(buses.index[buses.dc == "t"])
    if dc_lines.any():
        log("hvdc", note=f"WARNING {int(dc_lines.sum())} lines touch DC buses; dropped")
        lines = lines[~dc_lines]
    buses = buses[buses.dc == "f"]

    # 3. transformers -> contract
    uf = UnionFind(buses.index)
    tr = trafo[trafo.bus0.isin(buses.index) & trafo.bus1.isin(buses.index)]
    for _, t in tr.iterrows():
        uf.union(t.bus0, t.bus1)
    groups: dict[str, list[str]] = {}
    for b in buses.index:
        groups.setdefault(uf.find(b), []).append(b)
    rep_of = {}
    for members in groups.values():
        r = max(members, key=lambda b: (buses.at[b, "voltage"], b))
        for m in members:
            rep_of[m] = r
    mixed = sum(1 for m in groups.values() if len({buses.at[b, "country"] for b in m}) > 1)
    lines = lines.copy()
    lines["bus0"] = lines.bus0.map(rep_of)
    lines["bus1"] = lines.bus1.map(rep_of)
    links["bus0"] = links.bus0.map(rep_of)
    links["bus1"] = links.bus1.map(rep_of)
    v = cfg["network"]["v_ref_kv"]
    lines["x_orig_ohm"] = lines["x"]
    lines["x"] = lines["x"] * (v / lines["voltage"]) ** 2
    lines["r"] = lines["r"] * (v / lines["voltage"]) ** 2
    selfl = lines.bus0 == lines.bus1
    buses = buses.loc[sorted(set(rep_of.values()))]
    log("transformers", buses=len(buses), lines=int((~selfl).sum()),
        note=f"contracted {len(tr)} transformers into {len(buses)} buses; "
             f"{int(selfl.sum())} lines became self-loops and were dropped; "
             f"{mixed} transformer groups spanned two countries (WARNING if >0)")
    lines = lines[~selfl]
    links = links[links.bus0 != links.bus1]

    # 4. islands
    edges = list(zip(lines.bus0, lines.bus1)) + list(zip(links.bus0, links.bus1))
    comp = pd.Series(_components(buses.index, edges))
    size = comp.value_counts()
    small = size[size < cfg["network"]["min_component"]].index
    drop = comp[comp.isin(small)].index
    big = size[size >= cfg["network"]["min_component"]]
    log("islands", buses=len(buses) - len(drop),
        note=f"{len(size)} components; kept {len(big)} with sizes {big.tolist()}; "
             f"removed {len(drop)} buses in {len(small)} components smaller than "
             f"{cfg['network']['min_component']}")
    buses = buses.drop(drop)
    lines = lines[lines.bus0.isin(buses.index) & lines.bus1.isin(buses.index)]
    links = links[links.bus0.isin(buses.index) & links.bus1.isin(buses.index)]

    # 5. stubs (same-country only, AC only)
    n_stub = 0
    while True:
        ends = pd.concat([lines[["bus0", "bus1"]].rename(columns={"bus0": "a", "bus1": "b"}),
                          lines[["bus1", "bus0"]].rename(columns={"bus1": "a", "bus0": "b"})])
        nb = ends.groupby("a").b.nunique()
        dc_ends = set(links.bus0) | set(links.bus1)
        cand = [b for b in nb[nb == 1].index if b not in dc_ends]
        other = ends.drop_duplicates("a").set_index("a").b
        cand = [b for b in cand if buses.at[b, "country"] == buses.at[other[b], "country"]]
        if not cand:
            break
        cand = set(cand)
        # never remove both ends of an isolated two-bus pair in one pass
        cand = {b for b in cand if not (other[b] in cand and b > other[b])}
        n_stub += len(cand)
        buses = buses.drop(list(cand))
        lines = lines[lines.bus0.isin(buses.index) & lines.bus1.isin(buses.index)]
    log("stubs", buses=len(buses), lines=len(lines), links=len(links),
        note=f"absorbed {n_stub} degree-1 stub buses (same-country AC stubs only)")

    buses = buses[["voltage", "x", "y", "country", "symbol"]].copy()
    buses["v_nom"] = v
    lines["border"] = lines.bus0.map(buses.country) != lines.bus1.map(buses.country)
    links["border"] = links.bus0.map(buses.country) != links.bus1.map(buses.country)
    return buses, lines, links, rep


def cluster_countries(buses: pd.DataFrame, lines: pd.DataFrame, cfg: dict) -> tuple[pd.Series, list[dict]]:
    """Country label used for partitioning: microstates/small countries merged
    into the neighbour with the largest total cross-border susceptance."""
    lab = buses.country.copy()
    counts = lab.value_counts()
    merge = [c for c in counts.index
             if c in cfg["scope"]["merge_always"] or counts[c] < cfg["scope"]["min_buses"]]
    rows = []
    for c in merge:
        b = 1.0 / lines.x
        c0 = lines.bus0.map(lab)
        c1 = lines.bus1.map(lab)
        m = (c0 == c) ^ (c1 == c)
        nbr = np.where(c0[m] == c, c1[m], c0[m])
        s = pd.Series(b[m].values, index=nbr).groupby(level=0).sum().sort_values(ascending=False)
        if s.empty:
            rows.append({"country": c, "into": None, "note": "WARNING no AC neighbour; kept alone"})
            continue
        tgt = s.index[0]
        lab[lab == c] = tgt
        rows.append({"country": c, "buses": int(counts[c]), "into": tgt,
                     "note": "; ".join(f"{k}: b={v:.3g}" for k, v in s.items())})
    return lab, rows

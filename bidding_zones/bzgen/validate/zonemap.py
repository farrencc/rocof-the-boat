"""Bus -> zone assignment for the validation stage, and the DE maps under test.

Zones
-----
* Focus cluster-country (DE, with LU merged into it — DE-LU is one real bidding
  zone): labels from a map under test, or one zone ``DE`` for the k = 1 case.
* ``validate.split_real`` countries (SE, NO, DK, IT): real bidding zones by a
  point-in-polygon join of bus x/y against zone polygons.  Buses outside every
  polygon of their own country (coastline and offshore buses) take the nearest
  polygon of that country; the count is reported.
* Every other country (``buses.country``, so ME is its own zone although it is
  merged into BA for clustering): one zone.

Polygons: no ENTSO-E-published polygon set is reachable from this session, so
the source is Electricity Maps' curated open geometry pinned to a commit
(``validate.zones_source``), recorded in data/PROVENANCE.md and flagged at the
top of reports/validate.md.  It has six Italian zones; CALA (separate since
2021) lies inside IT-SUD there.

Maps under test
---------------
``k1``; the sweep headline (``results/configs/<sweep.headline>/labels.csv``);
up to ``validate.n_siblings`` near-degenerate siblings from a re-run of the
focus-country anneal at the headline parameters; and, for the out-of-sample
scoring set, the same three roles refit on odd ISO weeks only.  Siblings are
restarts within ``validate.sibling_rel_tol`` of the best energy whose map
differs from the headline (ARI < ``sweep.degenerate_ari``); fewer than
``n_siblings`` is reported, not padded.

The sweep's restart seeds use Python's salted ``hash(c)`` and are therefore
not reproducible across interpreter runs; the re-runs here use ``zlib.crc32``.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

from bzgen import config
from bzgen.cluster import edges as E
from bzgen.cluster import graphs
from bzgen.cluster.anneal import anneal

ROOT = config.ROOT
INTERIM = ROOT / "data" / "interim"
SOLVED = ROOT / "data" / "solved"
MAPS = ROOT / "results" / "validate" / "maps"


# --------------------------------------------------------------------------- #
# real foreign zones
# --------------------------------------------------------------------------- #

def zone_polygons(cfg: dict):
    import geopandas as gpd
    from bzgen.data.cache import fetch
    src = cfg["validate"]["zones_source"]
    path = fetch(src["url"], f"electricitymaps/world_{src['commit'][:12]}.geojson",
                 key="electricitymaps_zones",
                 notes=f"Electricity Maps electricitymaps-contrib geo/world.geojson at commit "
                       f"{src['commit']}; curated open zone geometry, NOT an ENTSO-E publication "
                       "(fallback: no authoritative bidding-zone polygons reachable)")
    g = gpd.read_file(path)
    ren = src["rename"]
    g = g[g.zoneName.isin(ren)].copy()
    g["zone"] = g.zoneName.map(ren)
    g["country"] = g.zoneName.str.split("-").str[0]
    return g[["zone", "country", "geometry"]].reset_index(drop=True)


def real_zones(buses: pd.DataFrame, cfg: dict, polys=None) -> tuple[pd.Series, pd.DataFrame]:
    """Zone of every bus in ``validate.split_real`` countries, and a per-country report."""
    import geopandas as gpd
    polys = zone_polygons(cfg) if polys is None else polys
    out, rep = [], []
    for c in cfg["validate"]["split_real"]:
        b = buses[buses.country == c]
        if b.empty:
            continue
        pc = polys[polys.country == c]
        if pc.empty:
            raise RuntimeError(f"no zone polygons for {c}: refusing to proceed silently")
        pts = gpd.GeoDataFrame(index=b.index, geometry=gpd.points_from_xy(b.x, b.y), crs=4326)
        j = gpd.sjoin(pts, pc, predicate="within", how="left")
        j = j[~j.index.duplicated(keep="first")]
        z = j.zone.reindex(b.index)
        miss = z.index[z.isna()]
        if len(miss):
            p3 = pts.loc[miss].to_crs(3035)
            nn = gpd.sjoin_nearest(p3, pc.to_crs(3035), how="left", distance_col="d")
            nn = nn[~nn.index.duplicated(keep="first")]
            z.loc[miss] = nn.zone.reindex(miss).to_numpy()
            dmax = float(nn.d.max()) / 1e3
        else:
            dmax = 0.0
        out.append(z)
        rep.append({"country": c, "buses": len(b), "in_polygon": int(len(b) - len(miss)),
                    "nearest_polygon": int(len(miss)), "max_nearest_km": round(dmax, 1),
                    "zones": " ".join(f"{k}:{v}" for k, v in z.value_counts().sort_index().items())})
    return pd.concat(out).rename("zone"), pd.DataFrame(rep).set_index("country")


def zone_map(buses: pd.DataFrame, focus_labels: pd.Series | None, cfg: dict,
             real: pd.Series | None = None) -> pd.Series:
    """bus -> zone for every bus.  ``focus_labels`` None means the k = 1 map."""
    f = cfg["validate"]["focus"]
    real = real_zones(buses, cfg)[0] if real is None else real
    z = buses.country.astype(str).copy()
    z.loc[real.index] = real
    inf = buses.index[buses.cluster_country == f]
    if focus_labels is None:
        z.loc[inf] = f
    else:
        lab = focus_labels.reindex(inf)
        if lab.isna().any():
            raise ValueError(f"{int(lab.isna().sum())} {f} buses have no label in the map")
        z.loc[inf] = [f"{f}-{int(v)}" for v in lab]
    check_cover(z, buses)
    return z.rename("zone")


def check_cover(z: pd.Series, buses: pd.DataFrame) -> None:
    """Every bus exactly once, no missing zone."""
    if not z.index.is_unique:
        raise AssertionError("zone map lists a bus more than once")
    if set(z.index) != set(buses.index):
        raise AssertionError(f"zone map covers {len(set(z.index) & set(buses.index))} of "
                             f"{len(buses)} buses")
    if z.isna().any():
        raise AssertionError(f"{int(z.isna().sum())} buses without a zone")


# --------------------------------------------------------------------------- #
# maps under test
# --------------------------------------------------------------------------- #

def sweep_labels(cid: str, focus: str) -> pd.Series:
    lab = pd.read_csv(ROOT / "results" / "configs" / cid / "labels.csv")
    lab = lab[lab.country == focus]
    return lab.set_index("bus").zone.astype(int)


def _seed(a: dict, c: str, r: int, salt: int = 0) -> int:
    return (a["seed"] * 1_000_003 + zlib.crc32(c.encode()) % 100_000 * 101 + r + salt) % (2**31 - 1)


def anneal_restarts(cfg: dict, edges: pd.DataFrame, L: pd.Series, G: pd.Series,
                    params: dict, salt: int = 0) -> list[dict]:
    """All restarts of the focus-country anneal: energy + bus labels for each."""
    c = cfg["validate"]["focus"]
    a = cfg["anneal"]
    buses = pd.read_csv(INTERIM / "buses.csv", index_col=0)
    spec = pd.read_csv(ROOT / "results" / "spectrum.csv", index_col=0)
    g, nodes, host, _ = graphs.build(c, buses, edges, params["alpha"], L, G,
                                     cfg["edges"]["dc_in_energy"])
    k = int(spec.loc[c, "k_used"]) + params["dk"]
    out = []
    for r in range(a["restarts"]):
        res = anneal(g, k, lam_b=params["lambda_b"], lam_c0=params["lambda_c_initial"],
                     lam_c1=a["lambda_c_final"], floor=a["balance_floor"], n_temps=a["n_temps"],
                     sweeps_per_temp=a["sweeps_per_temp"], t_final_ratio=a["t_final_ratio"],
                     seed=_seed(a, c, r, salt), quench_sweeps=a["quench_sweeps"])
        lab = pd.Series(res["labels"], index=nodes)
        for b, h in host.items():
            lab[b] = lab[h]
        out.append({"restart": r, "energy": float(res["energy"]), "contig": int(res["contig"]),
                    "labels": lab.sort_index()})
    return out


def map_energy(cfg: dict, edges: pd.DataFrame, L: pd.Series, G: pd.Series, params: dict,
               labels: pd.Series) -> float:
    """Energy of a given focus-country map on a given edge table (for comparing a
    committed map with re-run restarts)."""
    from bzgen.cluster.anneal import contiguity_guarantee, energy_terms
    c = cfg["validate"]["focus"]
    a = cfg["anneal"]
    buses = pd.read_csv(INTERIM / "buses.csv", index_col=0)
    spec = pd.read_csv(ROOT / "results" / "spectrum.csv", index_col=0)
    g, nodes, host, _ = graphs.build(c, buses, edges, params["alpha"], L, G,
                                     cfg["edges"]["dc_in_energy"])
    k = int(spec.loc[c, "k_used"]) + params["dk"]
    lab = labels.reindex(nodes).to_numpy().astype(np.int64)
    ep, cs, pb = energy_terms(lab, k, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, a["balance_floor"])
    lam_c1 = a["lambda_c_final"]
    if lam_c1 in (None, "auto"):
        lam_c1 = contiguity_guarantee(g, params["lambda_b"])
    return float(ep + lam_c1 * cs + params["lambda_b"] * pb)


def pick_siblings(restarts: list[dict], head: pd.Series, head_energy: float, cfg: dict) -> tuple[list[dict], pd.DataFrame]:
    """Restarts within sibling_rel_tol of the best energy whose map differs from the headline."""
    v = cfg["validate"]
    best = min([head_energy] + [r["energy"] for r in restarts])
    tol = v["sibling_rel_tol"] * max(1.0, abs(best))
    rows, sib = [], []
    for r in sorted(restarts, key=lambda r: r["energy"]):
        a_ = ari(head.reindex(r["labels"].index).to_numpy(), r["labels"].to_numpy())
        within = r["energy"] - best <= tol
        distinct = a_ < cfg["sweep"]["degenerate_ari"]
        chosen = within and distinct and r["contig"] == 0 and len(sib) < v["n_siblings"]
        rows.append({"restart": r["restart"], "energy": r["energy"],
                     "rel_gap_to_best": (r["energy"] - best) / max(1.0, abs(best)),
                     "ari_vs_headline": a_, "contiguous": r["contig"] == 0,
                     "within_tol": within, "distinct": distinct, "sibling": chosen})
        if chosen:
            sib.append(r)
    return sib, pd.DataFrame(rows)


def node_attributes_weighted(cfg: dict, w: pd.Series, gen_mean: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Time-weighted mean load and generation per bus over the hours in ``w`` (the
    sweep's ``node_attributes`` restricted to a subset of hours)."""
    buses = pd.read_csv(INTERIM / "buses.csv", index_col=0)
    key = pd.read_csv(INTERIM / "load_key.csv", index_col=0).load_key
    nat = pd.read_parquet(INTERIM / "national_load.parquet")
    nat.index = nat.index.tz_convert("UTC").tz_localize(None) if nat.index.tz is not None else nat.index
    m = nat.loc[w.index].mul(w.to_numpy(), axis=0).sum() / w.sum()
    L = key * buses.country.map(m)
    gens = pd.read_csv(INTERIM / "generators.csv", index_col=0)
    Gb = gen_mean.groupby(gens.bus.reindex(gen_mean.index)).sum()
    return L.rename("L"), Gb.reindex(buses.index).fillna(0.0).rename("G")


def iso_week(idx: pd.DatetimeIndex) -> pd.DataFrame:
    ic = idx.isocalendar()
    return pd.DataFrame({"iso_year": ic.year.to_numpy(), "iso_week": ic.week.to_numpy()}, index=idx)


def odd_week_edges(cfg: dict) -> pd.DataFrame:
    """Edge statistics refit on odd ISO weeks only (prices.parquet reused, not re-solved)."""
    from bzgen.cluster import prepare
    buses, lines, links, prices, sn, shed = prepare.load_solution(cfg)
    P, w, _ = prepare.shedding_policy(prices, sn, shed, buses, cfg)
    odd = iso_week(P.index).iso_week.to_numpy() % 2 == 1
    P, w = P.loc[odd], w[odd]
    et = E.edge_table(buses, lines, links)
    return E.normalise(E.congestion_stats(et, P, w, cfg), cfg)


def build_maps(cfg: dict, gen_mean_odd: pd.Series | None, log=print) -> dict:
    """Write results/validate/maps/<map_id>.csv for every map under test (cached, incremental).

    In-sample: the committed sweep headline, and siblings from a re-run of the focus
    anneal on the committed edge table (balance term from the sweep's node attributes).
    Out-of-sample (needs ``gen_mean_odd``, the nodal reference's mean dispatch over odd
    ISO weeks): edge statistics and anneal refit on odd weeks only.
    Returns {map_id: {"labels": Series | None, "fit": "none"|"insample"|"oos", "role": ...}}.
    """
    from bzgen.cluster.sweep import node_attributes
    MAPS.mkdir(parents=True, exist_ok=True)
    f = cfg["validate"]["focus"]
    cid = cfg["sweep"]["headline"]
    params = json.loads((ROOT / "results" / "configs" / cid / "done.json").read_text())["params"]
    meta_f = MAPS / "maps.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {
        "headline_config": cid, "params": params, "sibling_rel_tol": cfg["validate"]["sibling_rel_tol"],
        "maps": {"k1": {"fit": "none", "role": "k1"}}}
    head = sweep_labels(cid, f)
    if "is_headline" not in meta["maps"]:
        head.to_csv(MAPS / "is_headline.csv")
        e_in = pd.read_parquet(ROOT / "results" / "edges.parquet")
        L, G = node_attributes(cfg)
        eh = map_energy(cfg, e_in, L, G, params, head)
        log(f"in-sample headline energy (recomputed on committed edges) {eh:.3f}")
        rs = anneal_restarts(cfg, e_in, L, G, params)
        sib, tab = pick_siblings(rs, head, eh, cfg)
        tab.to_csv(MAPS / "is_restarts.csv", index=False)
        meta["is_headline_energy"] = eh
        meta["maps"]["is_headline"] = {"fit": "insample", "role": "headline", "energy": eh}
        for i, s_ in enumerate(sib):
            s_["labels"].to_csv(MAPS / f"is_sib{i + 1}.csv")
            meta["maps"][f"is_sib{i + 1}"] = {"fit": "insample", "role": "sibling",
                                              "energy": s_["energy"], "restart": s_["restart"]}
        meta_f.write_text(json.dumps(meta, indent=1))
    if "oos_headline" not in meta["maps"] and gen_mean_odd is not None:
        sn = pd.read_csv(SOLVED / "snapshots.csv", index_col=0, parse_dates=True).weight
        e_odd = odd_week_edges(cfg)
        odd = iso_week(pd.DatetimeIndex(sn.index)).iso_week.to_numpy() % 2 == 1
        Lo, Go = node_attributes_weighted(cfg, sn[odd], gen_mean_odd)
        ro = anneal_restarts(cfg, e_odd, Lo, Go, params, salt=7919)
        ok = [r for r in ro if r["contig"] == 0] or ro
        bo = min(ok, key=lambda r: r["energy"])
        bo["labels"].to_csv(MAPS / "oos_headline.csv")
        meta["maps"]["oos_headline"] = {"fit": "oos", "role": "headline", "energy": bo["energy"],
                                        "restart": bo["restart"]}
        sib_o, tab_o = pick_siblings([r for r in ro if r is not bo], bo["labels"], bo["energy"], cfg)
        tab_o.to_csv(MAPS / "oos_restarts.csv", index=False)
        for i, s_ in enumerate(sib_o):
            s_["labels"].to_csv(MAPS / f"oos_sib{i + 1}.csv")
            meta["maps"][f"oos_sib{i + 1}"] = {"fit": "oos", "role": "sibling",
                                               "energy": s_["energy"], "restart": s_["restart"]}
        meta["ari_oos_vs_is_headline"] = float(ari(head.reindex(bo["labels"].index), bo["labels"]))
        meta_f.write_text(json.dumps(meta, indent=1))
    out = {}
    for mid, m in meta["maps"].items():
        lab = None if mid == "k1" else pd.read_csv(MAPS / f"{mid}.csv", index_col=0).iloc[:, 0].astype(int)
        out[mid] = dict(m, labels=lab)
    return out

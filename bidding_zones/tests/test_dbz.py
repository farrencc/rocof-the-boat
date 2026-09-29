"""Dynamical Bidding Zones: normalisation, floor, anchor (stage 1)."""

import numpy as np
import pandas as pd
import pytest

from bzgen import config
from bzgen.cluster import edges as E
from bzgen.dbz import anchor as AN
from bzgen.dbz import balance_floor, load_config
from bzgen.dbz import graph as DG

ROOT = config.ROOT
INTERIM = ROOT / "data" / "interim"
STATIC_EDGES = ROOT / "results" / "edges.parquet"
DBZ_EDGES = ROOT / "results" / "dbz" / "edges.parquet"
RAW = ["bus_a", "bus_b", "country", "J", "s_nom", "n_lines", "line_ids", "is_dc",
       "dp_mean", "dp_duration", "dp_quantile"]

need_interim = pytest.mark.skipif(not (INTERIM / "buses.csv").exists(),
                                  reason="data/interim not built (gitignored)")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def network():
    buses = pd.read_csv(INTERIM / "buses.csv", index_col=0)
    lines = pd.read_csv(INTERIM / "lines.csv", index_col=0)
    links = pd.read_csv(INTERIM / "links.csv", index_col=0)
    return buses, lines, links


# --------------------------------------------------------------------------- #
# 1. normalisation reduction
# --------------------------------------------------------------------------- #

def _as_european(static: pd.DataFrame) -> pd.DataFrame:
    t = static[RAW].copy()
    t["country_a"] = t.country
    t["country_b"] = t.country
    t["is_cross"] = False
    return t


def test_normalisation_reduces_to_static(cfg):
    """On the real (committed) intra-country statistics, node-averaged normalisation
    reproduces edges.normalise exactly."""
    com = pd.read_parquet(STATIC_EDGES)
    raw = _as_european(com)
    ref = E.normalise(com[RAW], cfg).set_index(["bus_a", "bus_b"]).sort_index()
    out = DG.normalise(raw, cfg).set_index(["bus_a", "bus_b"]).sort_index()
    assert len(out) == len(ref) == len(com)
    np.testing.assert_allclose(out.dp_n, ref.dp_n, rtol=0, atol=0)
    np.testing.assert_allclose(out.J_n, ref.J_n, rtol=0, atol=0)
    np.testing.assert_allclose(out.dp_country_mean, ref.dp_country_mean, rtol=0, atol=0)
    # and the committed table itself
    c = com.set_index(["bus_a", "bus_b"]).sort_index()
    np.testing.assert_allclose(out.dp_n, c.dp_n, rtol=0, atol=0)
    np.testing.assert_allclose(out.J_n, c.J_n, rtol=0, atol=0)


@pytest.mark.skipif(not DBZ_EDGES.exists(), reason="results/dbz/edges.parquet not built")
def test_european_table_intra_rows_match_static(cfg):
    """The European table (cross-border edges present) leaves intra rows untouched:
    cross-border edges never enter the country scales."""
    eu = pd.read_parquet(DBZ_EDGES)
    com = pd.read_parquet(STATIC_EDGES).set_index(["bus_a", "bus_b"]).sort_index()
    intra = eu[~eu.is_cross].set_index(["bus_a", "bus_b"]).sort_index()
    assert intra.index.equals(com.index)
    np.testing.assert_allclose(intra.dp_n, com.dp_n, rtol=0, atol=1e-12)
    np.testing.assert_allclose(intra.J_n, com.J_n, rtol=0, atol=1e-12)
    xb = eu[eu.is_cross]
    assert (xb.country == DG.XB).all() and (xb.country_a != xb.country_b).all()
    # recompute from scratch: scales from intra only
    re = DG.normalise(eu[RAW + ["country_a", "country_b", "is_cross"]], cfg)
    np.testing.assert_allclose(re.dp_n, eu.dp_n, rtol=0, atol=0)


def test_cross_border_average_rule(cfg):
    """A cross-border edge is clipped at, and divided by, the mean of its countries' values."""
    rows = []
    for c, vals in (("AA", [0.1, 0.2, 0.3, 0.4]), ("BB", [1.0, 2.0, 3.0, 4.0])):
        for i, v in enumerate(vals):
            rows.append(dict(bus_a=f"{c}{i}", bus_b=f"{c}{i + 1}", country=c, J=1.0 + i,
                             s_nom=1.0, n_lines=1, line_ids="x", is_dc=False,
                             dp_mean=v, dp_duration=v, dp_quantile=v,
                             country_a=c, country_b=c, is_cross=False))
    rows.append(dict(bus_a="AA0", bus_b="BB0", country=DG.XB, J=2.0, s_nom=1.0, n_lines=1,
                     line_ids="y", is_dc=False, dp_mean=9.0, dp_duration=9.0, dp_quantile=9.0,
                     country_a="AA", country_b="BB", is_cross=True))
    t = pd.DataFrame(rows)
    c2 = dict(cfg, edges=dict(cfg["edges"], statistic="duration", dp_remedy="clip",
                              dp_clip_q=0.99, J_remedy="log1p"))
    sc = DG.country_scales(t, c2)
    out = DG.normalise(t, c2, sc)
    x = out[out.is_cross].iloc[0]
    thr = (sc.at["AA", "dp_theta"] + sc.at["BB", "dp_theta"]) / 2
    m = (sc.at["AA", "dp_mean"] + sc.at["BB", "dp_mean"]) / 2
    assert x.dp_n == pytest.approx(min(9.0, thr) / m)
    med = (sc.at["AA", "J_theta"] + sc.at["BB", "J_theta"]) / 2
    mJ = (sc.at["AA", "J_mean"] + sc.at["BB", "J_mean"]) / 2
    assert x.J_n == pytest.approx(np.log1p(2.0 / med) / mJ)
    # scales are unchanged by the cross-border edge
    sc0 = DG.country_scales(t[~t.is_cross], c2)
    pd.testing.assert_frame_equal(sc, sc0)


def test_rank_remedy_rejected(cfg):
    t = _as_european(pd.read_parquet(STATIC_EDGES))
    for key in ("dp_remedy", "J_remedy"):
        c2 = dict(cfg, edges=dict(cfg["edges"], **{key: "rank"}))
        with pytest.raises(ValueError, match="rank"):
            DG.normalise(t, c2)


@need_interim
def test_european_edge_table_intra_rows_identical(network):
    buses, lines, links = network
    eu = DG.edge_table(buses, lines, links)
    st = E.edge_table(buses, lines, links)
    a = (eu[~eu.is_cross].drop(columns=["country_a", "country_b", "is_cross"])
         .set_index(["bus_a", "bus_b"]).sort_index())
    s = st.set_index(["bus_a", "bus_b"]).sort_index()[a.columns]
    pd.testing.assert_frame_equal(a, s, check_dtype=False)
    xb = eu[eu.is_cross]
    assert len(xb) > 0 and xb.is_dc.any() and (~xb.is_dc).any()
    # the AC-corridor de-duplication of HVDC links carries over
    acpairs = set(zip(eu[~eu.is_dc].bus_a, eu[~eu.is_dc].bus_b))
    assert not any(p in acpairs for p in zip(eu[eu.is_dc].bus_a, eu[eu.is_dc].bus_b))


# --------------------------------------------------------------------------- #
# 3. balance floor rescale
# --------------------------------------------------------------------------- #

def test_floor_rescale():
    a = {"balance_floor": 0.05, "balance_floor_frac_of_mean": 0.15}
    assert balance_floor(a, 3) == pytest.approx(0.05, rel=1e-12)
    assert balance_floor(a, 90) == pytest.approx(0.15 / 90)
    # static default: absolute floor, untouched
    assert balance_floor({"balance_floor": 0.05}, 3) == 0.05
    assert balance_floor({"balance_floor": 0.05, "balance_floor_frac_of_mean": None}, 90) == 0.05
    assert balance_floor(config.load()["anneal"], 90) == config.load()["anneal"]["balance_floor"]


def test_overlay_does_not_touch_static_keys(cfg):
    base = config.load()
    assert cfg["anneal"]["balance_floor"] == base["anneal"]["balance_floor"]
    for k in ("statistic", "dp_remedy", "J_remedy", "dp_clip_q", "signal_min"):
        assert cfg["edges"][k] == base["edges"][k]
    assert cfg["dbz"]["anchor_config_id"] == base["sweep"]["headline"]


# --------------------------------------------------------------------------- #
# 7. anchor
# --------------------------------------------------------------------------- #

def test_relabel_bijection_toy():
    lab = pd.DataFrame({"bus": list("abcdefg"), "zone": [0, 1, 0, 1, 0, 2, 1],
                        "country": ["FR"] * 2 + ["DE"] * 2 + ["AT"] * 3})
    A, pairs = AN.relabel(lab)
    assert len(pairs) == 2 + 2 + 3                 # FR 2, DE 2, AT 3 zones
    assert sorted(A.unique()) == list(range(len(pairs)))
    # FR zone 0 and DE zone 0 are different global zones
    assert A["a"] != A["c"]
    back = pairs.set_index("global").loc[A.to_numpy()]
    assert (back.country.to_numpy() == lab.country.to_numpy()).all()
    assert (back.zone.to_numpy() == lab.zone.to_numpy()).all()


@need_interim
def test_anchor_real(cfg, network):
    buses, lines, links = network
    cid = cfg["dbz"]["anchor_config_id"]
    lab = AN.read_static(cid)
    nodes = DG.node_order(buses)
    an = AN.load(cid, nodes, buses)
    k_c = lab.groupby("country").zone.nunique()
    assert an["K"] == int(k_c.sum())
    rows = pd.read_csv(ROOT / "results" / "sweep.csv")
    assert an["K"] == int(rows[(rows.config_id == cid) & (rows.status == "ok")].k.sum())
    assert len(an["A"]) == len(nodes) == len(buses) == len(lab)
    assert (np.bincount(an["A"], minlength=an["K"]) > 0).all()
    # bijection (country, zone) <-> global
    p = an["pairs"]
    assert p["global"].tolist() == list(range(an["K"]))
    assert not p[["country", "zone"]].duplicated().any()
    # coverage: every European node labelled, labels agree with the file
    A = pd.Series(an["A"], index=nodes)
    back = p.set_index("global").loc[A.reindex(lab.bus).to_numpy()]
    assert (back.zone.to_numpy() == lab.zone.to_numpy()).all()
    assert (back.country.to_numpy() == lab.country.to_numpy()).all()
    # contiguity on the European graph (structure asserted inside check_contiguity)
    et = DG.edge_table(buses, lines, links).assign(dp_n=0.0, J_n=0.0)
    one = pd.Series(1.0, index=buses.index)
    g, nodes2, _ = DG.build(buses, et, 1.0, one, one * 0, one, an["K"], nodes=nodes)
    r = AN.check_contiguity(an["A"], an["K"], g, nodes, buses)
    static = rows[(rows.config_id == cid) & (rows.status == "ok")]
    assert len(r["isolated"]) == int(static.n_isolated_attached.sum())
    assert r["excess_components"] == len(r["isolated"])


def test_transfer_distance_toy():
    A = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    assert AN.transfer_distance(A, A) == 0
    B = np.array([2, 2, 2, 0, 0, 0, 1, 1])          # pure relabelling
    assert AN.transfer_distance(B, A) == 0
    B = np.array([0, 0, 1, 1, 1, 1, 2, 2])          # one node changed zone
    assert AN.transfer_distance(B, A) == 1


# --------------------------------------------------------------------------- #
# 4-6. rigidity
# --------------------------------------------------------------------------- #

from bzgen.dbz import rigidity as RG  # noqa: E402


def mirkin_direct(B, A, cap):
    """Definition: sum_i g_i |BZ_B(i) symdiff BZ_A(i)|."""
    s = 0.0
    for i in range(len(B)):
        inB = B == B[i]
        inA = A == A[i]
        s += cap[i] * np.count_nonzero(inB ^ inA)
    return s


def _caps(kind, n, rng):
    if kind == "ones":
        return np.ones(n)
    c = rng.exponential(100.0, n)
    c[rng.random(n) < 0.4] = 0.0
    return c


@pytest.mark.parametrize("kind", ["ones", "random"])
@pytest.mark.parametrize("seed", range(3))
def test_rigidity_full_matches_definition(kind, seed):
    rng = np.random.default_rng(seed)
    n, KA, KB = 60, 5, 6
    A = rng.integers(0, KA, n)
    B = rng.integers(0, KB, n)
    cap = _caps(kind, n, rng)
    assert RG.rigidity_full(B, A, cap, KB, KA) == pytest.approx(mirkin_direct(B, A, cap))


@pytest.mark.parametrize("kind", ["ones", "random"])
@pytest.mark.parametrize("seed", range(3))
def test_rigidity_incremental_single(kind, seed):
    rng = np.random.default_rng(seed)
    n, KA, KB = 60, 5, 5
    A = rng.integers(0, KA, n)
    B = rng.integers(0, KB, n)
    cap = _caps(kind, n, rng)
    t = RG.build_tables(B, A, cap, KB, KA)
    H = RG.rigidity_full(B, A, cap, KB, KA)
    for step in range(3000):
        i = int(rng.integers(n))
        q = int(rng.integers(KB))
        p = int(B[i])
        d = RG.rigidity_delta(p, q, A[i], cap[i], *t)
        RG.rigidity_apply(p, q, A[i], cap[i], *t) if p != q else None
        B[i] = q
        H += d
        if step % 250 == 0:
            ref = RG.rigidity_full(B, A, cap, KB, KA)
            assert H == pytest.approx(ref, rel=1e-9, abs=1e-6)
    assert H == pytest.approx(RG.rigidity_full(B, A, cap, KB, KA), rel=1e-9, abs=1e-6)
    assert RG.rigidity_from_tables(*t) == pytest.approx(H, rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("kind", ["ones", "random"])
@pytest.mark.parametrize("seed", range(3))
def test_rigidity_incremental_fragment(kind, seed):
    """Whole-component moves: move a random subset of one zone to another at once."""
    rng = np.random.default_rng(seed)
    n, KA, KB = 60, 5, 5
    A = rng.integers(0, KA, n)
    B = rng.integers(0, KB, n)
    cap = _caps(kind, n, rng)
    t = RG.build_tables(B, A, cap, KB, KA)
    H = RG.rigidity_full(B, A, cap, KB, KA)
    fa = np.zeros(KA, np.int64); fn = np.zeros(KA, np.int64); fg = np.zeros(KA)
    for step in range(2000):
        p = int(rng.integers(KB))
        pool = np.flatnonzero(B == p)
        if len(pool) == 0:
            continue
        members = rng.choice(pool, size=int(rng.integers(1, len(pool) + 1)), replace=False)
        q = int(rng.integers(KB))
        m, NF, GF = RG.fragment_profile(members.astype(np.int64), A, cap, fa, fn, fg)
        d = RG.rigidity_delta_fragment(p, q, fa, m, fn, fg, NF, GF, *t)
        if p != q:
            RG.rigidity_apply_fragment(p, q, fa, m, fn, fg, NF, GF, *t)
        RG.fragment_clear(fa, m, fn, fg)
        assert not fn.any() and not fg.any()
        B[members] = q
        H += d
        if step % 200 == 0:
            assert H == pytest.approx(RG.rigidity_full(B, A, cap, KB, KA), rel=1e-9, abs=1e-6)
    assert H == pytest.approx(RG.rigidity_full(B, A, cap, KB, KA), rel=1e-9, abs=1e-6)
    # a single-node fragment agrees with the single-node delta
    i = 0
    m, NF, GF = RG.fragment_profile(np.array([i]), A, cap, fa, fn, fg)
    q = (B[i] + 1) % KB
    assert (RG.rigidity_delta_fragment(B[i], q, fa, m, fn, fg, NF, GF, *t)
            == pytest.approx(RG.rigidity_delta(B[i], q, A[i], cap[i], *t)))
    RG.fragment_clear(fa, m, fn, fg)


def test_rigidity_known_values():
    rng = np.random.default_rng(0)
    A = rng.integers(0, 4, 40)
    cap = rng.exponential(1.0, 40)
    assert RG.rigidity_full(A, A, cap, 4, 4) == 0.0
    # relabelling B's zones does not change the penalty
    perm = np.array([2, 0, 3, 1])
    assert RG.rigidity_full(perm[A], A, cap, 4, 4) == 0.0
    # hand-computed toy: A = {0,1 | 2,3}, B = {0 | 1,2,3}, capacities g = (1, 2, 3, 4)
    #   i=0: B {0},     A {0,1}  -> symdiff {1}       -> 1 * 1
    #   i=1: B {1,2,3}, A {0,1}  -> symdiff {0,2,3}   -> 2 * 3
    #   i=2: B {1,2,3}, A {2,3}  -> symdiff {1}       -> 3 * 1
    #   i=3: same as i=2                                -> 4 * 1
    A = np.array([0, 0, 1, 1]); B = np.array([0, 1, 1, 1]); g = np.array([1.0, 2.0, 3.0, 4.0])
    assert RG.rigidity_full(B, A, g, 2, 2) == pytest.approx(1 + 6 + 3 + 4)
    # load-only nodes (g = 0) contribute nothing to the outer sum but count inside it
    g0 = np.array([1.0, 0.0, 0.0, 0.0])
    assert RG.rigidity_full(B, A, g0, 2, 2) == pytest.approx(1.0)
    # each node's term |B(i) symdiff A(i)| is symmetric, so the weighted sum is too
    assert RG.rigidity_full(A, B, g, 2, 2) == RG.rigidity_full(B, A, g, 2, 2)


# --------------------------------------------------------------------------- #
# 8. T0 with lambda_c pinned; the DBZ annealer's bookkeeping
# --------------------------------------------------------------------------- #

from bzgen.cluster.anneal import (contiguity_guarantee, energy_terms,  # noqa: E402
                                  graph_voronoi, initial_temperature)
from bzgen.dbz import anneal as DA  # noqa: E402


def _toy_europe(n=150, seed=3, K=6):
    import networkx as nx
    G = nx.random_geometric_graph(n, 0.16, seed=seed)
    comps = list(nx.connected_components(G))
    for a, b in zip(comps[:-1], comps[1:]):
        G.add_edge(next(iter(a)), next(iter(b)))
    rng = np.random.default_rng(seed)
    e = np.array(G.edges())
    w = rng.normal(0.3, 1.0, len(e))
    L = rng.exponential(1.0, n)
    Gn = rng.exponential(1.0, n) * (rng.random(n) < 0.3)
    cap = rng.exponential(50.0, n) * (rng.random(n) < 0.5)
    g = DG.EuropeGraph(n, e[:, 0], e[:, 1], w, L, Gn, cap, K)
    A = graph_voronoi(g, K, np.random.default_rng(seed + 1))
    return g, A


def test_T0_exclude_contiguity():
    g, A = _toy_europe()
    K = int(A.max()) + 1
    lam_c = contiguity_guarantee(g, 0.5)
    lab = graph_voronoi(g, K, np.random.default_rng(0))
    T_fix = initial_temperature(g, lab, K, 0.5, lam_c, 0.02, np.random.default_rng(1),
                                exclude_contiguity=True)
    assert np.isfinite(T_fix) and 1e-3 < T_fix < 1e3
    # default path unchanged: identical to passing lambda_c explicitly, and exclusion
    # equals the default with lambda_c = 0
    T_def = initial_temperature(g, lab, K, 0.5, lam_c, 0.02, np.random.default_rng(1))
    T_zero = initial_temperature(g, lab, K, 0.5, 0.0, 0.02, np.random.default_rng(1))
    assert T_fix == T_zero
    assert T_def >= T_fix       # contiguity-breaking samples only add uphill energy


@pytest.mark.parametrize("lam_r", [0.0, 1.0, 10.0])
def test_dbz_anneal_bookkeeping(lam_r):
    """Incremental totals of _run_dbz equal full recomputation; result contiguous; K kept."""
    g, A = _toy_europe()
    K = int(A.max()) + 1
    r = DA.anneal_dbz(g, A, K, lam_b=0.5, floor=0.15 / K, lam_rigid=lam_r, n_temps=12,
                      sweeps_per_temp=5, t_final_ratio=0.01, seed=7, quench_sweeps=3)
    lab = r["labels"]
    ep, cs, pb = energy_terms(lab, K, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, 0.15 / K)
    assert r["potts"] == pytest.approx(ep, abs=1e-8)
    assert r["contig"] == cs == 0
    assert r["balance"] == pytest.approx(pb, abs=1e-8)
    assert r["rigid_raw"] == pytest.approx(RG.rigidity_full(lab, A, g.cap, K, K), rel=1e-9)
    assert len(np.unique(lab)) == K


def test_rigidity_pulls_towards_anchor():
    g, A = _toy_europe()
    K = int(A.max()) + 1
    kw = dict(lam_b=0.5, floor=0.15 / K, n_temps=15, sweeps_per_temp=5, t_final_ratio=0.01,
              seed=11, quench_sweeps=3)
    free = DA.anneal_dbz(g, A, K, lam_rigid=0.0, **kw)
    stiff = DA.anneal_dbz(g, A, K, lam_rigid=50.0, **kw)
    assert AN.transfer_distance(stiff["labels"], A) < AN.transfer_distance(free["labels"], A)


@need_interim
def test_european_edge_table_has_no_parallel_edges(network):
    eu = DG.edge_table(*network)
    assert not eu.duplicated(["bus_a", "bus_b"]).any()
    assert (eu.n_lines > 1).any()


def test_europe_graph_rejects_parallel_edges():
    with pytest.raises(ValueError, match="duplicate"):
        DG.EuropeGraph(3, np.array([0, 0, 1]), np.array([1, 1, 2]), np.zeros(3), np.ones(3),
                       np.zeros(3), np.ones(3), 2)


def test_T0_dbz_reduces_to_exclude_contiguity():
    g, A = _toy_europe()
    K = int(A.max()) + 1
    lab = graph_voronoi(g, K, np.random.default_rng(0))
    lam_c = contiguity_guarantee(g, 0.5)
    ref = initial_temperature(g, lab, K, 0.5, lam_c, 0.02, np.random.default_rng(1),
                              exclude_contiguity=True)
    t0 = DA.initial_temperature_dbz(g, lab, K, 0.5, 0.02, np.random.default_rng(1), A, K, 0.0)
    assert t0 == ref
    t10 = DA.initial_temperature_dbz(g, lab, K, 0.5, 0.02, np.random.default_rng(1), A, K, 10.0)
    assert np.isfinite(t10) and t10 != ref


# --------------------------------------------------------------------------- #
# 2. static reproduction
# --------------------------------------------------------------------------- #

need_solved = pytest.mark.skipif(
    not (DBZ_EDGES.exists() and (ROOT / "data" / "solved" / "gen_mean.csv").exists()),
    reason="results/dbz/edges.parquet or data/solved not built")


@need_solved
@pytest.mark.parametrize("country", ["HU", "SK"])
def test_static_reproduction(cfg, country):
    """European graph restricted to one country's nodes and intra-country edges is the
    static country graph, and the static annealer on it reproduces results/sweep.csv."""
    from bzgen.cluster import graphs
    from bzgen.cluster.anneal import anneal
    from bzgen.cluster.sweep import node_attributes
    cid = cfg["dbz"]["anchor_config_id"]
    buses = pd.read_csv(INTERIM / "buses.csv", index_col=0)
    en = pd.read_parquet(DBZ_EDGES)
    L, G = node_attributes(cfg)
    ref = pd.read_csv(ROOT / "results" / "sweep.csv")
    ref = ref[(ref.config_id == cid) & (ref.country == country)].iloc[0]
    if ref.n_isolated_attached:
        pytest.skip("country has isolated buses: node sets differ by construction")
    nodes, _ = graphs.country_nodes(country, buses, en[~en.is_cross])
    intra = en[~en.is_cross]
    gE, nodesE, _ = DG.build(buses, intra, ref.alpha, L, G, pd.Series(1.0, index=buses.index),
                             int(ref.k), nodes=nodes)
    gS, nodesS, host, _ = graphs.build(country, buses, pd.read_parquet(STATIC_EDGES), ref.alpha,
                                       L, G, cfg["edges"]["dc_in_energy"])
    assert nodesE == nodesS and not host
    np.testing.assert_array_equal(gE.ptr, gS.ptr)
    np.testing.assert_array_equal(gE.idx, gS.idx)
    np.testing.assert_array_equal(gE.w, gS.w)
    np.testing.assert_array_equal(gE.Lnode, gS.Lnode)
    a = cfg["anneal"]
    E = [anneal(gE, int(ref.k), lam_b=ref.lambda_b, lam_c0=ref.lambda_c_initial,
                lam_c1=a["lambda_c_final"], floor=a["balance_floor"], n_temps=a["n_temps"],
                sweeps_per_temp=50, t_final_ratio=a["t_final_ratio"], seed=s,
                quench_sweeps=a["quench_sweeps"])["energy"] for s in range(6)]
    # within the restart spread: the static run's own spread bounds every restart, and the
    # median lies inside the static [E_min, E_max]; a restart below the static E_min
    # (the static best missed a lower minimum, e.g. SK) or above E_max (an outlier
    # restart) is allowed within one spread
    tol = 1e-6 * max(1.0, abs(ref.E_min))
    lo, hi = ref.E_min - ref.E_spread - tol, ref.E_max + ref.E_spread + tol
    assert all(lo <= e <= hi for e in E), (E, ref.E_min, ref.E_max)
    assert ref.E_min - tol <= float(np.median(E)) <= ref.E_max + tol, (E, ref.E_min, ref.E_max)


# --------------------------------------------------------------------------- #
# 9. resume after an interruption
# --------------------------------------------------------------------------- #

@need_solved
def test_resume_after_interrupt(tmp_path):
    import json
    import os
    import subprocess
    import sys
    out = tmp_path / "dbz"
    cmd = [sys.executable, "-m", "bzgen.dbz.sweep", "--smoke", "--no-figures", "--no-git",
           "--out", str(out), "--lambdas", "0", "1", "3"]
    env = dict(os.environ, DBZ_ABORT_AT="1")
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 3, r.stderr[-2000:]            # died mid-checkpoint at lambda = 1
    assert (out / "lr0" / "done.json").exists()
    assert (out / "lr1" / "labels.csv").exists() and not (out / "lr1" / "done.json").exists()
    assert not list(out.rglob(".*.tmp")), "atomic writes left a temp file behind"
    snap = {p.name: p.read_bytes() for p in (out / "lr0").iterdir()}
    table = pd.read_csv(out / "sweep.csv")
    assert table.lambda_rigid.tolist() == [0.0]           # only completed lambda listed
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "skip lr0 (done)" in r.stdout
    assert {p.name: p.read_bytes() for p in (out / "lr0").iterdir()} == snap   # untouched
    for lam in ("lr1", "lr3"):
        assert (out / lam / "done.json").exists()
        assert json.loads((out / lam / "done.json").read_text())["schedule"]["restarts"] == 2
    table = pd.read_csv(out / "sweep.csv")
    assert table.lambda_rigid.tolist() == [0.0, 1.0, 3.0]
    lab = pd.read_csv(out / "lr1" / "labels.csv")
    assert len(lab) == 3771 and lab.zone.nunique() == 90

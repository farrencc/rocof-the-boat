"""Energy function, incremental update vs full recomputation, contiguity counter."""

import networkx as nx
import numpy as np
import pytest

from bzgen.cluster.anneal import (CountryGraph, anneal, count_components_among, energy_terms,
                                graph_voronoi, hinge, move_delta, zone_components)


def random_graph(n, p, seed, planar=False):
    rng = np.random.default_rng(seed)
    if planar:
        G = nx.random_geometric_graph(n, 0.18, seed=seed)
        # ensure connected
        comps = list(nx.connected_components(G))
        for a, b in zip(comps[:-1], comps[1:]):
            G.add_edge(next(iter(a)), next(iter(b)))
    else:
        G = nx.connected_watts_strogatz_graph(n, 4, p, seed=seed)
    e = np.array(G.edges())
    w = rng.normal(0, 1, len(e))
    L = rng.exponential(1.0, n)
    Gn = rng.exponential(1.0, n) * (rng.random(n) < 0.3)
    return G, CountryGraph(n, e[:, 0], e[:, 1], w, L, Gn)


def full_H(labels, g, k, lam_c, lam_b, floor):
    ep, cs, pb = energy_terms(labels, k, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, floor)
    return ep + lam_c * cs + lam_b * pb, ep, cs, pb


def test_hinge():
    assert hinge(10, 0, 100, 0.05) == 0.0
    assert hinge(0, 0, 100, 0.05) == 1.0
    assert hinge(2.5, 1.0, 100, 0.05) == pytest.approx(0.25)
    assert hinge(1.0, 2.5, 100, 0.05) == pytest.approx(0.25)   # max(load, gen)


def test_energy_matches_networkx_definition():
    G, g = random_graph(60, 0.2, 1)
    rng = np.random.default_rng(0)
    k = 4
    lab = rng.integers(0, k, g.n)
    ep, cs, pb = energy_terms(lab, k, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, 0.05)
    # Potts: sum of weights on intra-zone edges, each undirected edge once
    wmap = {}
    for i in range(g.n):
        for e in range(g.ptr[i], g.ptr[i + 1]):
            wmap[(i, g.idx[e])] = g.w[e]
    ref = sum(wmap[(a, b)] for a, b in G.edges() if lab[a] == lab[b])
    assert ep == pytest.approx(ref)
    ncomp = sum(nx.number_connected_components(G.subgraph(np.flatnonzero(lab == s)))
                for s in range(k))
    assert cs == ncomp - k


@pytest.mark.parametrize("seed", range(6))
def test_zone_components_vs_networkx(seed):
    G, g = random_graph(80, 0.1, seed, planar=seed % 2 == 0)
    rng = np.random.default_rng(seed)
    k = 5
    lab = rng.integers(0, k, g.n)
    C = zone_components(lab, k, g.ptr, g.idx)
    for s in range(k):
        nodes = np.flatnonzero(lab == s)
        assert C[s] == nx.number_connected_components(G.subgraph(nodes))


@pytest.mark.parametrize("seed", range(6))
def test_local_counter_vs_networkx(seed):
    """count_components_among == number of zone-z components (without excl) hit by seeds."""
    G, g = random_graph(100, 0.05, seed, planar=True)
    rng = np.random.default_rng(seed)
    k = 3
    lab = graph_voronoi(g, k, rng)
    ws = g.ws
    for _ in range(200):
        i = int(rng.integers(g.n))
        z = int(rng.integers(k))
        nbr = sorted({int(j) for j in g.idx[g.ptr[i]:g.ptr[i + 1]] if lab[j] == z and j != i})
        m = len(nbr)
        ws.seeds[:m] = nbr
        ws.stamp[0] += 1
        got = count_components_among(ws.seeds, m, z, i, lab, g.ptr, g.idx, ws.mark, ws.owner,
                                     ws.stamp[0], ws.Q, ws.qh, ws.qt, ws.parent, ws.active)
        H = G.subgraph([v for v in np.flatnonzero(lab == z) if v != i])
        comp = {v: c for c, cc in enumerate(nx.connected_components(H)) for v in cc}
        assert got == len({comp[v] for v in nbr})


@pytest.mark.parametrize("seed", range(4))
def test_incremental_equals_full(seed):
    """Random move sequence: sum of move_delta equals full recomputation, every step."""
    G, g = random_graph(70, 0.15, seed, planar=seed % 2 == 1)
    rng = np.random.default_rng(100 + seed)
    k, lam_c, lam_b, floor = 4, 2.3, 0.7, 0.2
    lab = rng.integers(0, k, g.n)
    for s in range(k):
        lab[s] = s
    zL = np.zeros(k); zG = np.zeros(k)
    np.add.at(zL, lab, g.Lnode); np.add.at(zG, lab, g.Gnode)
    H, ep, cs, pb = full_H(lab, g, k, lam_c, lam_b, floor)
    ws = g.ws
    for _ in range(400):
        i = int(rng.integers(g.n))
        b = int(rng.integers(k))
        a = int(lab[i])
        if b == a or (lab == a).sum() == 1:
            continue
        dp, dca, dcb, dh = move_delta(i, b, lab, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, zL, zG,
                                      g.Ltot, floor, ws.mark, ws.owner, ws.stamp, ws.Q, ws.qh,
                                      ws.qt, ws.parent, ws.active, ws.seeds)
        lab[i] = b
        zL[a] -= g.Lnode[i]; zG[a] -= g.Gnode[i]; zL[b] += g.Lnode[i]; zG[b] += g.Gnode[i]
        H2, ep2, cs2, pb2 = full_H(lab, g, k, lam_c, lam_b, floor)
        assert ep2 - ep == pytest.approx(dp, abs=1e-9)
        assert cs2 - cs == dca + dcb
        assert pb2 - pb == pytest.approx(dh, abs=1e-12)
        H, ep, cs, pb = H2, ep2, cs2, pb2


def test_anneal_bookkeeping_consistent():
    """Energy tracked inside the annealer equals full recomputation of its output."""
    G, g = random_graph(120, 0.1, 7, planar=True)
    out = anneal(g, 4, lam_b=0.3, lam_c0=0.1, lam_c1=5.0, floor=0.05, n_temps=20,
                 sweeps_per_temp=5, t_final_ratio=1e-3, seed=3)
    ep, cs, pb = energy_terms(out["labels"], 4, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, 0.05)
    assert out["potts"] == pytest.approx(ep)
    assert out["contig"] == cs
    assert out["balance"] == pytest.approx(pb)
    assert len(np.unique(out["labels"])) == 4       # no zone emptied


def test_planted_partition_recovery():
    """The optimiser must reach at least the planted energy, contiguously; on this
    instance (planted partition contiguous and the ground state) it must recover it.
    Note: on sparser instances a *different* partition can have lower energy than
    the planted one (seen at n=250, seed=1: ARI 0.49 at lower energy) - that is
    degeneracy of the model, not an optimiser failure, hence the energy test."""
    from sklearn.metrics import adjusted_rand_score as ari
    from bzgen.cluster.synthetic import planted, to_graph
    p = planted(n=400, k=3, seed=1)
    g = to_graph(p)
    ep, cs, pb = energy_terms(p["truth"], 3, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, 0.02)
    assert cs == 0
    res = [anneal(g, 3, lam_b=0.1, lam_c0=0.1, lam_c1="auto", floor=0.02, n_temps=40,
                  sweeps_per_temp=15, t_final_ratio=1e-2, seed=s) for s in range(4)]
    best = min(res, key=lambda r: r["energy"])
    assert best["contig"] == 0
    assert best["potts"] + 0.1 * best["balance"] <= ep + 0.1 * pb + 1e-6
    assert ari(p["truth"], best["labels"]) > 0.95

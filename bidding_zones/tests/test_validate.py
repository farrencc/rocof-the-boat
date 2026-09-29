"""Validation stage: zone map, market bounds, net-position constraint, attribution."""

import copy

import numpy as np
import pandas as pd
import pytest

from bzgen import config
from bzgen.solve.lp import DCOPF
from bzgen.validate import metrics, zonemap
from bzgen.validate.market import ZonalMarket
from bzgen.validate.redispatch import Redispatch, redispatch_pypsa


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(config.load())
    c["network"]["s_max_pu"] = 1.0
    c["validate"]["markup"] = 1.0
    return c


def synth(buses, lines, gens, loads, snaps, links=()):
    """Small PyPSA network with the attributes bzgen.solve.opf.build_network provides.
    buses: {name: (country, x, y)}; lines: [(b0, b1, x, s_nom)]; gens: [(name, bus, carrier,
    p_nom, mc, p_max_pu array)]; loads: {bus: array}; links: [(name, b0, b1, p_nom)]."""
    import pypsa
    n = pypsa.Network()
    n.set_snapshots(snaps)
    n.add("Carrier", ["AC", "DC", "load_shedding", "onwind", "solar", "CCGT", "coal"])
    n.add("Bus", list(buses), v_nom=380.0, x=[v[1] for v in buses.values()],
          y=[v[2] for v in buses.values()], country=[v[0] for v in buses.values()], carrier="AC")
    for i, (a, b, x, s) in enumerate(lines):
        n.add("Line", f"L{i}", bus0=a, bus1=b, x=x, r=0.0, s_nom=s)
    for nm, a, b, p in links:
        n.add("Link", nm, bus0=a, bus1=b, p_nom=p, p_min_pu=-1.0, efficiency=1.0, carrier="DC")
    for nm, b, car, p, mc, pu in gens:
        n.add("Generator", nm, bus=b, carrier=car, p_nom=p, marginal_cost=mc,
              p_max_pu=pd.Series(pu, index=snaps))
    for b, L in loads.items():
        n.add("Load", f"{b} load", bus=b, p_set=pd.Series(L, index=snaps))
        n.add("Generator", f"{b} load_shedding", bus=b, carrier="load_shedding",
              p_nom=float(np.max(L)) + 1.0, marginal_cost=3000.0)
    n.generators["mc_base"] = n.generators.marginal_cost
    return n


def hour_inputs(n, s):
    pmpu = n.get_switchable_as_dense("Generator", "p_max_pu")
    pmax = pmpu.loc[s].to_numpy() * n.generators.p_nom.to_numpy()
    load = n.loads_t.p_set.loc[s].groupby(n.loads.bus).sum().reindex(n.buses.index).fillna(0.0)
    return pmax, load.to_numpy()


@pytest.fixture(scope="module")
def mesh():
    """Nine buses: DE (six, a 2x3 grid), FR (two), SE (one); random hours."""
    rng = np.random.default_rng(3)
    snaps = pd.date_range("2019-01-07", periods=6, freq="h")
    T = len(snaps)
    buses = {f"de{i}": ("DE", i % 3, i // 3) for i in range(6)}
    buses.update({"fr0": ("FR", -1, 0), "fr1": ("FR", -1, 1), "se0": ("SE", 3, 2)})
    lines = [("de0", "de1", 10, 300), ("de1", "de2", 10, 120), ("de3", "de4", 10, 300),
             ("de4", "de5", 12, 150), ("de0", "de3", 8, 200), ("de1", "de4", 9, 80),
             ("de2", "de5", 11, 200), ("fr0", "de0", 10, 150), ("fr1", "de3", 10, 100),
             ("fr0", "fr1", 5, 400)]
    links = [("hvdc", "de5", "se0", 150.0)]
    gens = []
    for b in buses:
        gens.append((f"{b} onwind", b, "onwind", rng.uniform(50, 300), 0.0, rng.uniform(0, 1, T)))
        gens.append((f"{b} CCGT", b, "CCGT", rng.uniform(50, 250), rng.uniform(40, 90), np.full(T, 0.9)))
    gens.append(("de2 solar", "de2", "solar", 400.0, 0.0, rng.uniform(0, 1, T)))
    loads = {b: rng.uniform(40, 220, T) for b in buses}
    n = synth(buses, lines, gens, loads, snaps, links)
    bdf = pd.DataFrame({"country": [v[0] for v in buses.values()],
                        "cluster_country": [v[0] for v in buses.values()],
                        "x": [v[1] for v in buses.values()], "y": [v[2] for v in buses.values()]},
                       index=list(buses))
    return n, bdf


def maps_for(bdf, cfg):
    real = pd.Series({"se0": "SE3"})
    split = pd.Series([0, 0, 1, 0, 0, 1], index=[f"de{i}" for i in range(6)])
    return (zonemap.zone_map(bdf, None, cfg, real=real),
            zonemap.zone_map(bdf, split, cfg, real=real))


def test_zone_map_covers_every_bus_once(mesh, cfg):
    _, bdf = mesh
    k1, sp = maps_for(bdf, cfg)
    for z in (k1, sp):
        zonemap.check_cover(z, bdf)
        assert z.index.is_unique and len(z) == len(bdf)
    assert set(k1[bdf.country == "DE"]) == {"DE"}
    assert set(sp[bdf.country == "DE"]) == {"DE-0", "DE-1"}
    assert k1["se0"] == "SE3" and k1["fr0"] == k1["fr1"] == "FR"
    with pytest.raises(AssertionError):
        zonemap.check_cover(pd.concat([k1, k1.iloc[:1]]), bdf)
    with pytest.raises(AssertionError):
        zonemap.check_cover(k1.iloc[1:], bdf)
    with pytest.raises(ValueError):
        zonemap.zone_map(bdf, pd.Series([0, 1], index=["de0", "de1"]), cfg,
                         real=pd.Series({"se0": "SE3"}))


def test_market_cost_bounds(mesh, cfg):
    n, bdf = mesh
    k1, sp = maps_for(bdf, cfg)
    costs = n.generators.mc_base.to_numpy() + np.linspace(0.001, 0.01, len(n.generators))
    nodal = DCOPF(n, cfg["network"]["s_max_pu"], costs=costs)
    for der in (0.2, 0.5, 1.0):
        mk = ZonalMarket(n, k1, der, cfg, costs)
        ms = ZonalMarket(n, sp, der, cfg, costs)
        rds = Redispatch(n, sp, cfg, costs)
        for s in n.snapshots:
            pmax, load = hour_inputs(n, s)
            cn = nodal.solve(pmax, load)["objective"]
            a1, a2 = mk.solve(pmax, load), ms.solve(pmax, load)
            tol = 1e-6 * abs(cn) + 1e-3
            assert a1["cost"] <= a2["cost"] + tol           # k = 1 relaxes the split
            if der == 1.0:
                assert a2["cost"] <= cn + tol               # zonal relaxes nodal at derating 1
            b = rds.solve(pmax, load, a2["p"])
            assert b["status"] == "ok"
            assert a2["cost"] + b["cost_phys"] >= cn - tol   # redispatched dispatch is nodal-feasible


def test_net_position_constraint_binds(cfg):
    snaps = pd.date_range("2019-01-07", periods=1, freq="h")
    buses = {"a": ("DE", 0, 0), "b": ("FR", 1, 0)}
    n = synth(buses, [("a", "b", 10, 100)],
              [("a CCGT", "a", "CCGT", 100, 10.0, [1.0]), ("b CCGT", "b", "CCGT", 100, 50.0, [1.0])],
              {"b": [80.0]}, snaps)
    bdf = pd.DataFrame({"country": ["DE", "FR"], "cluster_country": ["DE", "FR"],
                        "x": [0, 1], "y": [0, 0]}, index=["a", "b"])
    zm = zonemap.zone_map(bdf, None, cfg, real=pd.Series(dtype=object))
    costs = n.generators.mc_base.to_numpy()
    mk = ZonalMarket(n, zm, 0.5, cfg, costs)            # ATC 50 < physical 100
    pmax, load = hour_inputs(n, snaps[0])
    a = mk.solve(pmax, load)
    np_a = a["net_position"][list(mk.zones).index("DE")]
    assert np_a == pytest.approx(50.0)
    b = Redispatch(n, zm, cfg, costs).solve(pmax, load, a["p"])
    ga = n.generators.bus.to_numpy() == "a"
    inj_a = (a["p"] + b["up"] - b["down"])[ga].sum()
    assert inj_a == pytest.approx(np_a)                  # net position held
    assert b["up"].sum() == pytest.approx(b["down"].sum())
    # the same redispatch with one zone for everything (no zonal net positions) re-routes
    one = pd.Series("ALL", index=zm.index)
    b1 = Redispatch(n, one, cfg, costs).solve(pmax, load, a["p"])
    assert (a["p"] + b1["up"] - b1["down"])[ga].sum() == pytest.approx(80.0)
    assert b1["objective"] < b["objective"] - 1.0


def test_pypsa_crosscheck_matches(mesh, cfg):
    n, bdf = mesh
    _, sp = maps_for(bdf, cfg)
    costs = n.generators.mc_base.to_numpy() + 0.001
    ms = ZonalMarket(n, sp, 0.5, cfg, costs)
    rd = Redispatch(n, sp, cfg, costs)
    snaps = n.snapshots[:2]
    pz, obj = {}, {}
    for s in snaps:
        pmax, load = hour_inputs(n, s)
        a = ms.solve(pmax, load)
        pz[s] = a["p"]
        obj[s] = rd.solve(pmax, load, a["p"])["objective"]
    pz = pd.DataFrame(pz, index=n.generators.index).T
    ref = redispatch_pypsa(n, sp, cfg, costs, snaps, pz)
    for s in snaps:
        assert ref[s] == pytest.approx(obj[s], rel=1e-6, abs=1e-3)


def test_attribution_sums_to_dispatch_down(mesh, cfg):
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = rng.integers(0, 20)
        mu = rng.normal(0, 5, m) * (rng.random(m) < 0.5)
        cat = rng.integers(-1, 4, m)
        dd = float(rng.uniform(0, 100))
        a = metrics.attribute(dd, mu, rng.uniform(1, 100, m), cat, 1e-3)
        assert sum(a.values()) == pytest.approx(dd)
    n, bdf = mesh
    _, sp = maps_for(bdf, cfg)
    costs = n.generators.mc_base.to_numpy() + 0.001
    ms = ZonalMarket(n, sp, 0.2, cfg, costs)
    rd = Redispatch(n, sp, cfg, costs)
    ctx = metrics.Context(n, sp, cfg, set(), ms.nz.links, bdf.cluster_country == "DE")
    for s in n.snapshots:
        pmax, load = hour_inputs(n, s)
        a = ms.solve(pmax, load)
        h = metrics.hour_metrics(ctx, pmax, a, rd.solve(pmax, load, a["p"]))
        assert sum(h[f"attr_{b}"] for b in metrics.BUCKETS) == pytest.approx(h["DD_RES"], abs=1e-6)


def test_planted_border_congestion_is_border_attributed(cfg):
    """Two DE zones joined by one weak line; all wind north, all load south."""
    snaps = pd.date_range("2019-01-07", periods=3, freq="h")
    buses = {"n0": ("DE", 0, 1), "n1": ("DE", 1, 1), "s0": ("DE", 0, 0), "s1": ("DE", 1, 0)}
    lines = [("n0", "n1", 5, 2000), ("s0", "s1", 5, 2000), ("n1", "s1", 10, 100)]
    gens = [("n0 onwind", "n0", "onwind", 500, 0.0, [0.9, 0.8, 1.0]),   # never above load: no global oversupply
            ("s1 CCGT", "s1", "CCGT", 1000, 60.0, [1.0] * 3)]
    n = synth(buses, lines, gens, {"s0": [500.0] * 3}, snaps)
    bdf = pd.DataFrame({"country": "DE", "cluster_country": "DE",
                        "x": [0, 1, 0, 1], "y": [1, 1, 0, 0]}, index=list(buses))
    split = zonemap.zone_map(bdf, pd.Series([0, 0, 1, 1], index=list(buses)), cfg,
                             real=pd.Series(dtype=object))
    k1 = zonemap.zone_map(bdf, None, cfg, real=pd.Series(dtype=object))
    costs = n.generators.mc_base.to_numpy() + 0.001
    fb = bdf.cluster_country == "DE"
    res = {}
    for name, zm in (("split", split), ("k1", k1)):
        ms = ZonalMarket(n, zm, 1.0, cfg, costs)
        rd = Redispatch(n, zm, cfg, costs)
        ctx = metrics.Context(n, zm, cfg, set(), ms.nz.links, fb)
        rows = []
        for s in snaps:
            pmax, load = hour_inputs(n, s)
            a = ms.solve(pmax, load)
            rows.append(metrics.hour_metrics(ctx, pmax, a, rd.solve(pmax, load, a["p"])))
        res[name] = pd.DataFrame(rows).sum()
    sp = res["split"]
    border = sp.attr_border_intra + sp.attr_border_cross
    assert sp.DD_RES > 100
    assert border / sp.DD_RES > 0.95
    # the same congestion is internal when DE is one zone
    kk = res["k1"]
    assert kk.DD_RES > 100 and kk.attr_internal / kk.DD_RES > 0.95


def test_week_block_bootstrap_and_pairing(cfg):
    from bzgen.validate import inference as I
    idx = pd.date_range("2019-01-07", periods=24 * 7 * 6, freq="h")
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 100, len(idx))
    hk = pd.DataFrame({"DD_RES": base, "stageB_ok": True}, index=idx)
    hs = pd.DataFrame({"DD_RES": base - 10.0, "stageB_ok": True}, index=idx)
    hs.iloc[5, hs.columns.get_loc("stageB_ok")] = False      # infeasible under one map: dropped
    w = pd.Series(1.0, index=idx)
    p = I.paired(hs, hk, "DD_RES", w)
    assert len(p) == len(idx) - 1 and p.block.nunique() == 6
    b = I.block_bootstrap(p, cfg)
    assert b["point"] == pytest.approx(-10.0 * 8760 / 1e3)
    assert b["lo"] == pytest.approx(b["point"]) and b["hi"] == pytest.approx(b["point"])
    hs["DD_RES"] = base - 10.0 + rng.normal(0, 30, len(idx))
    b = I.block_bootstrap(I.paired(hs, hk, "DD_RES", w), cfg)
    assert b["lo"] < b["point"] < b["hi"]

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

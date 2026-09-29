"""nDBZ: scenarios (tests 2-4 of the task; later tests are added with their modules).

Synthetic fixtures throughout, so the tests run on a fresh clone (``data/solved/`` is
gitignored).  One check against the persisted scenario table runs when it exists.
"""

import json

import numpy as np
import pandas as pd
import pytest

from bzgen import config
from bzgen.cluster import edges as E
from bzgen.ndbz import load_config, scenarios as S


# --------------------------------------------------------------------------- #
# fixture: two countries, a cross-border line, an intra-country HVDC link
# --------------------------------------------------------------------------- #

def _cfg(**edges):
    cfg = load_config()
    cfg = json.loads(json.dumps(cfg))
    cfg["edges"].update(edges)
    cfg["scenario"]["min_snapshots"] = 40
    cfg["scenario"]["q"] = 0.05
    cfg["scenario"]["bootstrap"]["n"] = 60
    return cfg


def synthetic(seed=0, T=400, weights="ones"):
    rng = np.random.default_rng(seed)
    buses, lines, links = [], [], []
    for c, nb in (("AA", 12), ("BB", 9)):
        ids = [f"{c}{i:02d}" for i in range(nb)]
        buses += [{"bus": b, "country": c, "cluster_country": c} for b in ids]
        for i in range(nb):
            lines.append((ids[i], ids[(i + 1) % nb]))
        lines += [(ids[0], ids[nb // 2]), (ids[1], ids[nb // 2 + 1]), (ids[0], ids[1])]   # chords + parallel
    lines.append(("AA03", "BB04"))                          # cross-border: must be dropped
    links = pd.DataFrame({"bus0": ["AA02"], "bus1": ["AA08"], "p_nom": [500.0]}, index=["dc0"])
    buses = pd.DataFrame(buses).set_index("bus")
    lines = pd.DataFrame(lines, columns=["bus0", "bus1"])
    lines.index = [f"l{i}" for i in range(len(lines))]
    lines["x"] = rng.uniform(1, 30, len(lines))
    lines["s_nom"] = rng.uniform(500, 2000, len(lines))
    et = E.edge_table(buses, lines, links)
    idx = pd.date_range("2019-01-01", periods=T, freq="h", name="snapshot")
    # prices: a common level, a regional gradient that grows with load, noise, some spikes
    load = pd.DataFrame({"AA": 100 + 40 * rng.random(T), "BB": 80 + 30 * rng.random(T)}, index=idx)
    pos = np.array([int(b[2:]) for b in buses.index], float)
    lvl = 60 + 30 * rng.random(T)
    own = load[buses.country.to_numpy()].to_numpy()          # each bus sees its own country's load
    grad = (own - own.min(0)) / 12
    P = lvl[:, None] + grad * pos[None, :] + rng.normal(0, 0.7, (T, len(buses)))
    spike = rng.random((T, len(buses))) < 0.03
    P[spike] = rng.uniform(400, 900, spike.sum())
    raw = pd.DataFrame(P, index=idx, columns=buses.index)
    cfg = _cfg()
    cap = cfg["solve"]["price_clip"]
    w = pd.Series(1.0 if weights == "ones" else rng.integers(1, 40, T).astype(float), index=idx,
                  name="weight")
    gens = pd.DataFrame({"bus": ["AA00", "AA05", "BB00", "BB03", "AA07"],
                         "carrier": ["onwind", "solar", "offwind", "solar", "CCGT"],
                         "p_nom": [100.0, 50.0, 80.0, 40.0, 300.0]},
                        index=["g0", "g1", "g2", "g3", "g4"])
    prof = pd.DataFrame(rng.random((T, 4)), index=idx, columns=["g0", "g1", "g2", "g3"])
    feats = S.features(buses, raw.clip(-cap, cap), load, gens, prof)
    return S.Inputs(prices_raw=raw, prices=raw.clip(-cap, cap), weights=w, buses=buses, edges=et,
                    features=feats)


@pytest.fixture(scope="module")
def inp():
    return synthetic()


@pytest.fixture(scope="module")
def inp_w():
    return synthetic(seed=3, weights="kmeans")


# --------------------------------------------------------------------------- #
# test 2: scenario subsetting
# --------------------------------------------------------------------------- #

def test_edge_table_is_national(inp):
    assert "AA03" not in set(inp.edges[inp.edges.country == "BB"].bus_a)
    assert not ((inp.edges.bus_a == "AA03") & (inp.edges.bus_b == "BB04")).any()
    assert inp.edges.is_dc.sum() == 1


@pytest.mark.parametrize("fx", ["inp", "inp_w"])
def test_baseline_reproduces_congestion_stats(fx, request):
    inp = request.getfixturevalue(fx)
    cfg = _cfg()
    scn = S.select(inp, "baseline", cfg)
    got = S.scenario_stats(inp, scn, cfg, "as_is")
    ref = E.congestion_stats(inp.edges, inp.prices, inp.weights, cfg).sort_index()
    assert got.index.equals(ref.index)
    for c in ("dp_mean", "dp_duration", "dp_quantile"):
        np.testing.assert_array_equal(got[c].to_numpy(), ref[c].to_numpy())


def test_weights_subset_consistently(inp_w):
    cfg = _cfg()
    scn = S.select(inp_w, "peak_demand", cfg)
    for c, idx in scn.snapshots.items():
        ec = inp_w.edges[inp_w.edges.country == c]
        want = E.congestion_stats(ec, inp_w.prices.loc[idx], inp_w.weights.loc[idx], cfg)
        got = S.scenario_stats(inp_w, scn, cfg, "as_is").loc[ec.index]
        np.testing.assert_array_equal(got.dp_duration.to_numpy(), want.dp_duration.to_numpy())
        # a shuffled weight vector is aligned by label, not by position
        shuf = inp_w.weights.sample(frac=1.0, random_state=1)
        P, w = S.subset(inp_w.prices, shuf, idx)
        np.testing.assert_array_equal(w.to_numpy(), inp_w.weights.loc[idx].to_numpy())
        # the weight actually carried is the representative-hour weight, not 1
        assert scn.params[c]["weight"] == pytest.approx(inp_w.weights.loc[idx].sum())
    with pytest.raises(AssertionError):                     # a snapshot without a weight
        S.subset(inp_w.prices, inp_w.weights.iloc[1:], inp_w.weights.index[:5])
    with pytest.raises(AssertionError):                     # duplicated snapshot
        S.subset(inp_w.prices, inp_w.weights, inp_w.weights.index[[0, 0, 1]])


@pytest.mark.parametrize("stat", ["dp_mean", "dp_duration", "dp_quantile"])
def test_statistics_are_weight_normalised(inp_w, stat):
    """No statistic grows with the number of hours: invariant to scaling all weights
    and to duplicating every snapshot."""
    cfg = _cfg()
    et, P, w = inp_w.edges, inp_w.prices, inp_w.weights
    base = E.congestion_stats(et, P, w, cfg)[stat].to_numpy()
    scaled = E.congestion_stats(et, P, w * 7.0, cfg)[stat].to_numpy()
    np.testing.assert_allclose(scaled, base, rtol=1e-12, atol=1e-12)
    P2 = pd.concat([P, P.set_axis(P.index + pd.Timedelta(days=400))])
    w2 = pd.concat([w, w.set_axis(w.index + pd.Timedelta(days=400))])
    dup = E.congestion_stats(et, P2, w2, cfg)[stat].to_numpy()
    np.testing.assert_allclose(dup, base, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("stat", ["dp_mean", "dp_duration"])
def test_full_year_is_weight_average_of_subset_and_complement(inp_w, stat):
    cfg = _cfg()
    et, P, w = inp_w.edges, inp_w.prices, inp_w.weights
    idx = S.select(inp_w, "peak_demand", cfg, "europe").snapshots["AA"]
    comp = w.index.difference(idx)
    a = E.congestion_stats(et, P.loc[idx], w.loc[idx], cfg)[stat].to_numpy()
    b = E.congestion_stats(et, P.loc[comp], w.loc[comp], cfg)[stat].to_numpy()
    full = E.congestion_stats(et, P, w, cfg)[stat].to_numpy()
    Wa, Wb = w.loc[idx].sum(), w.loc[comp].sum()
    np.testing.assert_allclose((Wa * a + Wb * b) / (Wa + Wb), full, rtol=1e-10)
    if stat == "dp_duration":
        assert ((a >= 0) & (a <= 1)).all()


def test_masked_stats_without_mask_equal_congestion_stats(inp_w):
    cfg = _cfg()
    none = pd.DataFrame(False, index=inp_w.prices.index, columns=inp_w.prices.columns)
    m = S.masked_congestion_stats(inp_w.edges, inp_w.prices, inp_w.weights, none, cfg)
    ref = E.congestion_stats(inp_w.edges, inp_w.prices, inp_w.weights, cfg)
    for c in ("dp_mean", "dp_duration", "dp_quantile"):
        np.testing.assert_allclose(m[c].to_numpy(), ref[c].to_numpy(), rtol=1e-12, atol=1e-12)
    assert (m.kept_weight_share == 1).all()


def test_clip_handling_variants(inp):
    cfg = _cfg()
    scn = S.select(inp, "baseline", cfg)
    a = S.scenario_stats(inp, scn, cfg, "as_is")
    x = S.scenario_stats(inp, scn, cfg, "exclude_clipped")
    u = S.scenario_stats(inp, scn, cfg, "unclipped")
    assert not np.array_equal(a.dp_mean, u.dp_mean)          # spikes above the cap exist
    assert (x.kept_weight_share < 1).any()
    # by hand, for one edge
    r = inp.edges.iloc[0]
    raw = inp.prices_raw
    cap = cfg["solve"]["price_clip"]
    keep = (raw[r.bus_a].abs() < cap) & (raw[r.bus_b].abs() < cap)
    d = (raw[r.bus_a] - raw[r.bus_b]).abs()[keep]
    assert x.loc[inp.edges.index[0], "dp_duration"] == pytest.approx((d > 1.0).mean())
    with pytest.raises(ValueError):
        S.scenario_stats(inp, scn, cfg, "nonsense")


def test_selectors(inp_w):
    cfg = _cfg()
    W = inp_w.weights.sum()
    need = cfg["scenario"]["min_snapshots"]
    for name in ("peak_demand", "wind_surplus", "max_dispersion", "dunkelflaute"):
        for scope in ("country", "europe"):
            scn = S.select(inp_w, name, cfg, scope)
            for c, idx in scn.snapshots.items():
                w = inp_w.weights.loc[idx].sum()
                assert w >= min(need, W) - 1e-9, (name, scope, c)
                assert idx.is_unique and idx.is_monotonic_increasing
            if scope == "europe":
                assert scn.snapshots["AA"].equals(scn.snapshots["BB"])
    # top-q really is the top: every selected hour has at least the load of any other
    scn = S.select(inp_w, "peak_demand", cfg, "country")
    x = inp_w.features["load"]["AA"]
    sel = scn.snapshots["AA"]
    assert x.loc[sel].min() >= x.drop(sel).max()
    q = scn.params["AA"]["q_eff"]
    assert q == max(cfg["scenario"]["q"], need / W)
    # dunkelflaute: the intersection of low VRE and high load, at the recorded quantiles
    d = S.select(inp_w, "dunkelflaute", cfg, "country")
    p = d.params["BB"]
    vre, load = inp_w.features["vre_cf"]["BB"], inp_w.features["load"]["BB"]
    tv = S.weighted_quantile_1d(vre.to_numpy(), inp_w.weights.to_numpy(), p["vre_q_eff"])
    tl = S.weighted_quantile_1d(load.to_numpy(), inp_w.weights.to_numpy(), 1 - p["load_q_eff"])
    assert (vre.loc[d.snapshots["BB"]] <= tv).all() and (load.loc[d.snapshots["BB"]] >= tl).all()


def test_undefined_scenario_is_skipped(inp):
    cfg = _cfg()
    inp2 = S.Inputs(**{**inp.__dict__, "features": {**inp.features,
                                                    "wind_cf": inp.features["wind_cf"].drop(columns="BB")}})
    scn = S.select(inp2, "wind_surplus", cfg, "country")
    assert scn.snapshots["BB"] is None and scn.params["BB"]["status"].startswith("undefined")
    st = S.scenario_stats(inp2, scn, cfg)
    assert set(st.country) == {"AA"}


# --------------------------------------------------------------------------- #
# test 3: baseline vs self normalisation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("remedy", ["none", "clip", "log1p", "rank"])
def test_baseline_normalisation_of_baseline_equals_static(inp_w, remedy):
    cfg = _cfg(dp_remedy=remedy)
    ref = S.reference_stats(inp_w, cfg)
    n = S.normalise(ref, ref, cfg, "baseline")
    static = E.normalise(ref, cfg).sort_index()
    np.testing.assert_allclose(n.dp_n.to_numpy(), static.dp_n.to_numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(n.dp_n_self.to_numpy(), static.dp_n.to_numpy(), rtol=0, atol=0)


def test_remedy_fn_matches_edges_remedy():
    rng = np.random.default_rng(5)
    x = np.r_[rng.exponential(1, 200), np.zeros(20), [3.0, 3.0, 3.0]]
    for how in ("none", "clip", "log1p", "rank"):
        np.testing.assert_allclose(S.remedy_fn(x, how, 0.99)(x), E._remedy(x, how, 0.99), rtol=1e-12)


def test_baseline_normalisation_keeps_severity(inp):
    """Under ``baseline``, a non-baseline scenario's dp~ does NOT have mean 1: that is
    the point of it (a more congested scenario pushes harder against rigidity).
    Do not "fix" this."""
    cfg = _cfg()
    ref = S.reference_stats(inp, cfg)
    for name in ("peak_demand", "max_dispersion"):
        st = S.scenario_stats(inp, S.select(inp, name, cfg, "country"), cfg)
        n = S.normalise(st, ref.loc[st.index], cfg, "baseline")
        m = n.groupby("country").dp_n.mean()
        assert (np.abs(m - 1.0) > 0.005).all(), (name, m)
        assert (n.normalisation == "baseline").all()
        np.testing.assert_allclose(n.groupby("country").dp_n_self.mean(), 1.0, rtol=1e-12)
    # the synthetic gradient grows with load, so peak demand is more congested than the year
    st = S.scenario_stats(inp, S.select(inp, "peak_demand", cfg, "country"), cfg)
    assert (S.normalise(st, ref.loc[st.index], cfg, "baseline").groupby("country").dp_n.mean() > 1.05).all()


@pytest.mark.parametrize("remedy", ["none", "clip", "log1p", "rank"])
def test_rescaling_dp(inp, remedy):
    """Rescale every scenario dp by c.  baseline: dp~ scales by c (remedy none; with a
    remedy it is monotone, and at least not invariant).  self: mean 1, invariant."""
    cfg = _cfg(dp_remedy=remedy)
    stat = S.STAT_COL[cfg["edges"]["statistic"]]
    ref = S.reference_stats(inp, cfg)
    st = S.scenario_stats(inp, S.select(inp, "peak_demand", cfg, "country"), cfg)
    c = 0.6
    st2 = st.copy()
    st2[stat] = st2[stat] * c
    n1 = S.normalise(st, ref.loc[st.index], cfg, "baseline")
    n2 = S.normalise(st2, ref.loc[st.index], cfg, "baseline")
    if remedy == "none":
        np.testing.assert_allclose(n2.dp_n.to_numpy(), c * n1.dp_n.to_numpy(), rtol=1e-12)
    else:
        assert not np.allclose(n2.dp_n.to_numpy(), n1.dp_n.to_numpy())
        assert (n2.dp_n.to_numpy() <= n1.dp_n.to_numpy() + 1e-12).all()
    s1 = S.normalise(st, ref.loc[st.index], cfg, "self")
    s2 = S.normalise(st2, ref.loc[st.index], cfg, "self")
    np.testing.assert_allclose(s2.dp_n.to_numpy(), s1.dp_n.to_numpy(), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(s1.groupby("country").dp_n.mean().to_numpy(), 1.0, rtol=1e-12)


# --------------------------------------------------------------------------- #
# test 4: J~ invariance
# --------------------------------------------------------------------------- #

def test_J_identical_across_scenarios(inp):
    cfg = _cfg()
    ref = S.reference_stats(inp, cfg)
    base = S.edge_table(inp, cfg, "baseline", ref=ref).J_n
    static = E.normalise(ref, cfg).sort_index().J_n
    np.testing.assert_array_equal(base.to_numpy(), static.to_numpy())
    for name in S.SELECTORS:
        for scope in ("country", "europe"):
            for ch in S.CLIP_HANDLING:
                for mode in ("baseline", "self"):
                    t = S.edge_table(inp, cfg, name, scope, ch, mode, ref=ref)
                    np.testing.assert_array_equal(t.J_n.to_numpy(), static.loc[t.index].to_numpy())


def test_normalise_rejects_scenario_dependent_J(inp):
    cfg = _cfg()
    ref = S.reference_stats(inp, cfg)
    bad = ref.copy()
    bad["J"] = bad.J * np.linspace(1, 2, len(bad))
    with pytest.raises(AssertionError):
        S.normalise(bad, ref, cfg, "baseline")


# --------------------------------------------------------------------------- #
# the persisted scenario table (results/ndbz/scenarios/) against the static table
# --------------------------------------------------------------------------- #

PERSISTED = config.ROOT / "results" / "ndbz" / "scenarios" / "edges.parquet"


@pytest.mark.skipif(not PERSISTED.exists(), reason="scenario report not generated")
def test_persisted_baseline_reproduces_static_edges():
    t = pd.read_parquet(PERSISTED)
    b = t[(t.scope == "country") & (t.scenario == "baseline") & (t.clip_handling == "as_is")]
    static = pd.read_parquet(config.ROOT / "results" / "edges.parquet")
    b = b.set_index("edge_id").sort_index()
    np.testing.assert_array_equal(b.dp_n_baseline.to_numpy(), static.loc[b.index].dp_n.to_numpy())
    np.testing.assert_array_equal(b.dp_n_self.to_numpy(), static.loc[b.index].dp_n.to_numpy())
    np.testing.assert_array_equal(b.stat.to_numpy(), static.loc[b.index].dp_raw.to_numpy())
    # comparability: duration is a share of hours in every persisted scenario
    assert t.dp_duration.between(0, 1).all()
    others = t[(t.scenario != "baseline") & (t.clip_handling == "as_is") & (t.scope == "country")]
    means = others.groupby(["scenario", "country"]).dp_n_self.mean()
    np.testing.assert_allclose(means[means > 0].to_numpy(), 1.0, rtol=1e-9)

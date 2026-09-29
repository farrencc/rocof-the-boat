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
    """Rescale every scenario dp by c.  baseline: dp~ scales exactly by c for the
    scale-equivariant remedies (none, and clip at the scenario's own quantile); for
    log1p / rank (parameters frozen on the reference) it is monotone and not
    invariant.  self: mean 1 and invariant."""
    cfg = _cfg(dp_remedy=remedy)
    stat = S.STAT_COL[cfg["edges"]["statistic"]]
    ref = S.reference_stats(inp, cfg)
    st = S.scenario_stats(inp, S.select(inp, "peak_demand", cfg, "country"), cfg)
    c = 0.6
    st2 = st.copy()
    st2[stat] = st2[stat] * c
    n1 = S.normalise(st, ref.loc[st.index], cfg, "baseline")
    n2 = S.normalise(st2, ref.loc[st.index], cfg, "baseline")
    if remedy in ("none", "clip"):
        np.testing.assert_allclose(n2.dp_n.to_numpy(), c * n1.dp_n.to_numpy(), rtol=1e-12)
    else:
        assert not np.allclose(n2.dp_n.to_numpy(), n1.dp_n.to_numpy())
        assert (n2.dp_n.to_numpy() <= n1.dp_n.to_numpy() + 1e-12).all()
    # baseline_refclip (the original specification): monotone, not invariant
    r1 = S.normalise(st, ref.loc[st.index], cfg, "baseline_refclip")
    r2 = S.normalise(st2, ref.loc[st.index], cfg, "baseline_refclip")
    assert not np.allclose(r2.dp_n.to_numpy(), r1.dp_n.to_numpy())
    assert (r2.dp_n.to_numpy() <= r1.dp_n.to_numpy() + 1e-12).all()
    s1 = S.normalise(st, ref.loc[st.index], cfg, "self")
    s2 = S.normalise(st2, ref.loc[st.index], cfg, "self")
    np.testing.assert_allclose(s2.dp_n.to_numpy(), s1.dp_n.to_numpy(), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(s1.groupby("country").dp_n.mean().to_numpy(), 1.0, rtol=1e-12)


def test_refclip_pins_edges_and_baseline_does_not(inp):
    cfg = _cfg()
    stat = S.STAT_COL[cfg["edges"]["statistic"]]
    ref = S.reference_stats(inp, cfg)
    st = S.scenario_stats(inp, S.select(inp, "peak_demand", cfg, "country"), cfg)
    st[stat] = st[stat] * 3.0 + 0.05 * np.arange(len(st)) / len(st)   # much more congested, no ties
    n = S.normalise(st, ref.loc[st.index], cfg, "baseline")
    for c, g in n.groupby("country"):
        top = g.dp_n_baseline_refclip.max()
        assert (g.dp_n_baseline_refclip == top).mean() >= (g.above_ref_clip.mean() - 1e-12)
        # own-quantile clip: at most the top (1 - q) share is capped
        assert (g.dp_n_baseline >= g.dp_n_baseline.max() - 1e-12).mean() <= 0.5


def test_no_signal_flag(inp):
    cfg = _cfg()
    stat = S.STAT_COL[cfg["edges"]["statistic"]]
    ref = S.reference_stats(inp, cfg)
    quiet = ref.copy()
    quiet.loc[quiet.country == "BB", stat] = 1e-4
    n = S.normalise(ref, quiet, cfg, "baseline")
    assert n[n.country == "BB"].no_signal.all() and not n[n.country == "AA"].no_signal.any()


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


# --------------------------------------------------------------------------- #
# tests 5-7: rigidity (active implementation and the fallback copy)
# --------------------------------------------------------------------------- #

from bzgen.ndbz import _rigidity_core, balance_floor, rigidity as R

IMPLS = [R, _rigidity_core]


def _symdiff_reference(B, A, cap):
    """sum_i g_i |BZ_B(i) symdiff BZ_A(i)|, straight from the definition."""
    n = len(B)
    s = 0.0
    for i in range(n):
        zb = B == B[i]
        za = A == A[i]
        s += cap[i] * np.sum(zb ^ za)
    return s


def _caps(kind, n, rng):
    if kind == "ones":
        return np.ones(n)
    c = rng.exponential(300.0, n)
    c[rng.random(n) < 0.4] = 0.0
    return c


@pytest.mark.parametrize("impl", IMPLS, ids=lambda m: m.__name__.split(".")[-1])
@pytest.mark.parametrize("kind", ["ones", "random_with_zeros"])
def test_rigidity_incremental_single_moves(impl, kind):
    rng = np.random.default_rng(11)
    n, k = 60, 5
    A = rng.integers(0, k, n).astype(np.int64)
    B = rng.integers(0, k, n).astype(np.int64)
    cap = _caps(kind, n, rng)
    assert impl.rigidity_full(B, A, cap, k, k) == pytest.approx(_symdiff_reference(B, A, cap))
    t = impl.build_tables(B, A, cap, k, k)
    H = impl.rigidity_full(B, A, cap, k, k)
    for _ in range(4000):
        i = int(rng.integers(n))
        p, q = int(B[i]), int(rng.integers(k))
        d = impl.rigidity_delta(p, q, int(A[i]), float(cap[i]), *t)
        impl.rigidity_apply(p, q, int(A[i]), float(cap[i]), *t) if p != q else None
        B[i] = q
        H += d
    full = impl.rigidity_full(B, A, cap, k, k)
    assert H == pytest.approx(full, rel=1e-9, abs=1e-6)
    assert impl.rigidity_from_tables(*t) == pytest.approx(full, rel=1e-12, abs=1e-9)
    assert full == pytest.approx(_symdiff_reference(B, A, cap))


@pytest.mark.parametrize("impl", IMPLS, ids=lambda m: m.__name__.split(".")[-1])
@pytest.mark.parametrize("kind", ["ones", "random_with_zeros"])
def test_rigidity_incremental_fragment_moves(impl, kind):
    rng = np.random.default_rng(12)
    n, k = 60, 5
    A = rng.integers(0, k, n).astype(np.int64)
    B = rng.integers(0, k, n).astype(np.int64)
    cap = _caps(kind, n, rng)
    t = impl.build_tables(B, A, cap, k, k)
    H = impl.rigidity_full(B, A, cap, k, k)
    fa = np.zeros(k, np.int64)
    fn = np.zeros(k, np.int64)
    fg = np.zeros(k)
    for _ in range(1500):
        p = int(rng.integers(k))
        pool = np.flatnonzero(B == p)
        if len(pool) == 0:
            continue
        members = rng.choice(pool, size=int(rng.integers(1, len(pool) + 1)), replace=False).astype(np.int64)
        q = int(rng.integers(k))
        m, NF, GF = impl.fragment_profile(members, A, cap, fa, fn, fg)
        d = impl.rigidity_delta_fragment(p, q, fa, m, fn, fg, NF, GF, *t)
        if p != q:
            impl.rigidity_apply_fragment(p, q, fa, m, fn, fg, NF, GF, *t)
        impl.fragment_clear(fa, m, fn, fg)
        B[members] = q
        H += d
    assert (fn == 0).all() and (fg == 0).all()
    full = impl.rigidity_full(B, A, cap, k, k)
    assert H == pytest.approx(full, rel=1e-9, abs=1e-6)
    assert impl.rigidity_from_tables(*t) == pytest.approx(full, rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("impl", IMPLS, ids=lambda m: m.__name__.split(".")[-1])
def test_rigidity_known_values(impl):
    rng = np.random.default_rng(13)
    A = rng.integers(0, 4, 50).astype(np.int64)
    cap = _caps("random_with_zeros", 50, rng)
    assert impl.rigidity_full(A, A, cap, 4, 4) == 0.0
    # label invariance: permuting B's labels changes nothing (no matching needed)
    B = rng.integers(0, 4, 50).astype(np.int64)
    perm = np.array([2, 0, 3, 1])
    assert impl.rigidity_full(perm[B], A, cap, 4, 4) == pytest.approx(impl.rigidity_full(B, A, cap, 4, 4))
    # two-zone toy, by hand:
    #   A = {0,1}{2,3}, B = {0}{1,2,3}, g = (1, 2, 0, 3)
    #   node 0: A-zone {0,1}, B-zone {0}      -> {1}      -> 1 x 1 = 1
    #   node 1: A-zone {0,1}, B-zone {1,2,3}  -> {0,2,3}  -> 3 x 2 = 6
    #   node 2: g = 0                                        -> 0
    #   node 3: A-zone {2,3}, B-zone {1,2,3}  -> {1}      -> 1 x 3 = 3
    A2 = np.array([0, 0, 1, 1], np.int64)
    B2 = np.array([0, 1, 1, 1], np.int64)
    g2 = np.array([1.0, 2.0, 0.0, 3.0])
    assert impl.rigidity_full(B2, A2, g2, 2, 2) == pytest.approx(10.0)
    # load-only buses are counted inside the other terms: g_2 = 0, yet moving node 2
    # alone changes the penalty
    B3 = np.array([0, 0, 0, 1], np.int64)
    assert impl.rigidity_full(B3, A2, g2, 2, 2) > 0


def test_rigidity_source_is_reported():
    assert R.SOURCE in ("bzgen.dbz.rigidity", "bzgen.ndbz._rigidity_core")


# --------------------------------------------------------------------------- #
# test 8: Z_c normalisation
# --------------------------------------------------------------------------- #

def _country_like(n, k, seed):
    import networkx as nx
    from bzgen.cluster.anneal import CountryGraph, graph_voronoi
    G = nx.random_geometric_graph(n, 1.6 / np.sqrt(n), seed=seed)
    comps = list(nx.connected_components(G))
    for a, b in zip(comps[:-1], comps[1:]):
        G.add_edge(next(iter(a)), next(iter(b)))
    e = np.array(G.edges())
    rng = np.random.default_rng(seed)
    g = CountryGraph(n, e[:, 0], e[:, 1], np.zeros(len(e)), np.ones(n), np.ones(n))
    A = graph_voronoi(g, k, rng)
    cap = rng.exponential(400.0, n) * (rng.random(n) < 0.7)
    return g, A, cap


def test_normaliser_makes_moves_comparable_synthetic():
    med = []
    for n, k in ((25, 3), (800, 3), (300, 6)):
        g, A, cap = _country_like(n, k, seed=n)
        Z, gbar, flagged = R.normaliser(cap, k)
        assert not flagged and Z == pytest.approx(2 * n / k * cap.mean())
        med.append(np.median(R.single_move_deltas(A, cap, k, g.ptr, g.idx, Z)))
    assert max(med) / min(med) < 3.0, med
    # raw (unnormalised) costs differ by an order of magnitude, which is why Z_c exists
    Z, gbar, _ = R.normaliser(np.zeros(10), 3)
    assert gbar == 1.0 and Z == pytest.approx(2 * 10 / 3)


DATA = (config.ROOT / "data" / "interim" / "buses.csv").exists()
needs_data = pytest.mark.skipif(not DATA, reason="data/interim not built (gitignored)")


@pytest.fixture(scope="module")
def real_anchors():
    from bzgen.ndbz import anchor as AN
    return AN.anchors(load_config())


@needs_data
def test_normaliser_real_countries(real_anchors):
    from bzgen.cluster import graphs
    buses = pd.read_csv(config.ROOT / "data" / "interim" / "buses.csv", index_col=0)
    ed = pd.read_parquet(config.ROOT / "results" / "edges.parquet")
    zero = pd.Series(0.0, index=buses.index)
    med = {}
    for c, a in real_anchors.items():
        if a.skip:
            continue
        g = graphs.build(c, buses, ed, 1.0, zero, zero)[0]
        med[c] = np.median(R.single_move_deltas(a.A, a.cap, a.k, g.ptr, g.idx, a.Z))
    big = max(real_anchors, key=lambda c: real_anchors[c].info["n_nodes"])
    small = min(real_anchors, key=lambda c: real_anchors[c].info["n_nodes"])
    assert max(med[big], med[small]) / min(med[big], med[small]) < 3.0
    assert max(med.values()) / min(med.values()) < 3.0, med


# --------------------------------------------------------------------------- #
# test 9: anchor
# --------------------------------------------------------------------------- #

@needs_data
def test_anchor(real_anchors):
    from bzgen.cluster import graphs
    from bzgen.cluster.anneal import zone_components
    from bzgen.ndbz import anchor as AN
    cfg = load_config()
    sweep = pd.read_csv(config.ROOT / "results" / "sweep.csv")
    rows = sweep[sweep.config_id == cfg["ndbz"]["anchor_config_id"]].set_index("country")
    labels = AN.load_labels(cfg)
    buses = pd.read_csv(config.ROOT / "data" / "interim" / "buses.csv", index_col=0)
    ed = pd.read_parquet(config.ROOT / "results" / "edges.parquet")
    zero = pd.Series(0.0, index=buses.index)
    n_nodes = n_iso = 0
    for c, a in real_anchors.items():
        assert a.k == int(rows.loc[c, "k"]) == labels[labels.country == c].zone.nunique()
        g, nodes, host, _ = graphs.build(c, buses, ed, 1.0, zero, zero)
        assert a.nodes == nodes and a.host == host
        assert (zone_components(a.A, a.k, g.ptr, g.idx) == 1).all()
        # the isolated buses take their host's zone, exactly as the static pipeline did
        lab = labels[labels.country == c].set_index("bus").zone
        back = AN.attach_isolated(pd.Series(np.array(a.zone_ids)[a.A], index=nodes), host)
        assert back.sort_index().equals(lab.sort_index())
        n_nodes += len(nodes)
        n_iso += len(host)
        assert a.skip is None or a.k == 1
    assert n_nodes + n_iso == len(labels) == len(buses)
    assert n_iso == 8                          # "isolated within their country", graphs.py


def test_anchor_k1_is_skipped():
    from bzgen.cluster.anneal import CountryGraph
    from bzgen.ndbz import anchor as AN
    buses = pd.DataFrame({"country": "ZZ", "cluster_country": "ZZ", "x": [0.0, 1.0, 2.0],
                          "y": [50.0, 50.0, 50.0]}, index=["b0", "b1", "b2"])
    g = CountryGraph(3, np.array([0, 1]), np.array([1, 2]), np.zeros(2), np.ones(3), np.ones(3))
    labels = pd.DataFrame({"bus": ["b0", "b1", "b2"], "zone": [7, 7, 7], "country": "ZZ"})
    a = AN.country_anchor("ZZ", labels, buses, None, pd.Series(1.0, index=buses.index), 1,
                          g=g, nodes=["b0", "b1", "b2"], host={})
    assert a.k == 1 and a.skip and a.skip.startswith("k_c = 1")
    with pytest.raises(AssertionError):           # k disagreeing with sweep.csv fails loudly
        AN.country_anchor("ZZ", labels, buses, None, pd.Series(1.0, index=buses.index), 3,
                          g=g, nodes=["b0", "b1", "b2"], host={})
    bad = labels.assign(zone=[0, 1, 0])           # non-contiguous A fails loudly
    with pytest.raises(AssertionError):
        AN.country_anchor("ZZ", bad, buses, None, pd.Series(1.0, index=buses.index), 2,
                          g=g, nodes=["b0", "b1", "b2"], host={})


# --------------------------------------------------------------------------- #
# test 10: balance floor
# --------------------------------------------------------------------------- #

def test_balance_floor_parameterisations_agree_at_k3():
    cfg = load_config()
    static = config.load()
    assert "balance_floor_frac_of_mean" not in static["anneal"]      # static config unchanged
    assert balance_floor(cfg["anneal"], 3) == pytest.approx(static["anneal"]["balance_floor"])
    assert balance_floor(static["anneal"], 3) == static["anneal"]["balance_floor"]
    assert balance_floor(static["anneal"], 5) == static["anneal"]["balance_floor"]
    assert balance_floor({**cfg["anneal"], "balance_floor_frac_of_mean": None}, 2) == \
        static["anneal"]["balance_floor"]
    assert balance_floor(cfg["anneal"], 2) == pytest.approx(0.075)


# --------------------------------------------------------------------------- #
# tests 1, 11, 12: the forked annealer
# --------------------------------------------------------------------------- #

from bzgen.cluster import anneal as static_anneal
from bzgen.cluster.anneal import CountryGraph, contiguity_guarantee, energy_terms, graph_voronoi
from bzgen.ndbz import anneal as NA


def _toy(n=90, k=4, seed=3):
    import networkx as nx
    G = nx.random_geometric_graph(n, 0.2, seed=seed)
    comps = list(nx.connected_components(G))
    for a, b in zip(comps[:-1], comps[1:]):
        G.add_edge(next(iter(a)), next(iter(b)))
    e = np.array(G.edges())
    rng = np.random.default_rng(seed)
    g = CountryGraph(n, e[:, 0], e[:, 1], rng.normal(0, 1, len(e)), rng.exponential(1, n),
                     rng.exponential(1, n) * (rng.random(n) < 0.4))
    A = graph_voronoi(g, k, np.random.default_rng(seed + 1))
    cap = rng.exponential(200.0, n) * (rng.random(n) < 0.7)
    Z = R.normaliser(cap, k)[0]
    return g, k, A, cap, Z


SCHED = dict(lam_b=0.5, floor=0.05, n_temps=25, sweeps_per_temp=8, t_final_ratio=0.01)


@pytest.mark.parametrize("seed", [1, 7, 123])
def test_fork_reproduces_static_annealer_exactly(seed):
    """Test 1, strict form: lambda_rigid = 0 with the static schedule follows the
    static RNG stream, so labels, energy terms and trace are bit-identical to
    ``bzgen.cluster.anneal.anneal``.  The fork did not change the physics."""
    g, k, A, cap, Z = _toy()
    s = static_anneal.anneal(g, k, SCHED["lam_b"], 0.1, "auto", SCHED["floor"], SCHED["n_temps"],
                             SCHED["sweeps_per_temp"], SCHED["t_final_ratio"], seed=seed)
    x = NA.anneal_ndbz(g, k, A, cap, Z, 0.0, seed=seed, mode="static", lam_c0=0.1, **SCHED)
    np.testing.assert_array_equal(x["labels"], s["labels"])
    assert (x["potts"], x["contig"], x["balance"], x["energy"]) == \
        (s["potts"], s["contig"], s["balance"], s["energy"])
    np.testing.assert_array_equal(x["trace"][:, :5], s["trace"])
    assert x["T0"] == s["T0"]


def test_bookkeeping_matches_full_recomputation():
    g, k, A, cap, Z = _toy()
    for lr in (0.0, 0.3, 3.0):
        for blocks in (False, True):
            x = NA.anneal_ndbz(g, k, A, cap, Z, lr, seed=5, blocks=blocks, **SCHED)
            ep, cs, pb = energy_terms(x["labels"], k, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot,
                                      SCHED["floor"])
            assert x["potts"] == pytest.approx(ep, abs=1e-9)
            assert x["contig"] == cs == 0
            assert x["balance"] == pytest.approx(pb, abs=1e-12)
            assert x["rigid"] == pytest.approx(R.rigidity_full(x["labels"], A, cap, k, k) / Z, abs=1e-9)
            assert np.bincount(x["labels"], minlength=k).min() >= 1


def test_block_moves_reach_the_anchor_at_high_rigidity():
    """At large lambda_rigid the optimum is ~A; single flips alone get stuck in
    pair-counting local minima, the anchor-block moves do not (bzgen/ndbz/anneal.py)."""
    g, k, A, cap, Z = _toy()
    ref = NA.reference_quench(g, k, A, cap, Z, 30.0, SCHED["lam_b"], SCHED["floor"])
    best = min(NA.anneal_ndbz(g, k, A, cap, Z, 30.0, seed=s, **SCHED)["energy"] for s in range(3))
    assert best <= ref["energy"] + 1e-9


def test_reference_quench_from_anchor_is_contiguous():
    g, k, A, cap, Z = _toy()
    r = NA.reference_quench(g, k, A, cap, Z, 1.0, SCHED["lam_b"], SCHED["floor"])
    assert r["contig"] == 0


def test_T0_default_path_unchanged():
    g, k, A, cap, Z = _toy()
    lab = graph_voronoi(g, k, np.random.default_rng(0))
    f = lambda **kw: static_anneal.initial_temperature(g, lab, k, 0.5, 7.0, 0.05,
                                                       np.random.default_rng(4), **kw)
    assert f() == f(exclude_contiguity=False)
    assert f(exclude_contiguity=True) == static_anneal.initial_temperature(
        g, lab, k, 0.5, 0.0, 0.05, np.random.default_rng(4))
    # the nDBZ T0 at lambda_rigid = 0 is exactly initial_temperature(exclude_contiguity=True)
    t = NA.initial_temperature_ndbz(g, lab, k, 0.5, 0.05, np.random.default_rng(4), A, cap, 0.0)
    assert t == f(exclude_contiguity=True)


def test_T0_band_fails_loudly():
    g, k, A, cap, Z = _toy()
    with pytest.raises(RuntimeError, match="T0"):
        NA.anneal_ndbz(g, k, A, cap, Z, 0.0, seed=1, T0_band=(1e3, 1e4), **SCHED)


def test_frames():
    """Test 12, strict: the final frame is the returned labelling, the frame count
    follows the stride, and capture on/off is bit-identical (the buffer does not
    touch the RNG stream)."""
    g, k, A, cap, Z = _toy()
    for lr in (0.0, 1.0):
        on = NA.anneal_ndbz(g, k, A, cap, Z, lr, seed=9, capture=True, quench_frames=7, **SCHED)
        off = NA.anneal_ndbz(g, k, A, cap, Z, lr, seed=9, capture=False, **SCHED)
        np.testing.assert_array_equal(on["labels"], off["labels"])
        np.testing.assert_array_equal(on["trace"], off["trace"])
        for key in ("potts", "contig", "balance", "rigid", "energy", "T0"):
            assert on[key] == off[key]
        assert off["frames"].shape == (0, g.n)
        np.testing.assert_array_equal(on["frames"][-1], on["labels"])
        qsteps = int(20.0 * g.n)
        stride = on["quench_stride"]
        assert stride == max(1, qsteps // 7)
        assert on["frames"].shape == (NA.n_frames(SCHED["n_temps"], qsteps, stride), g.n)
        assert on["frames"].shape[0] == SCHED["n_temps"] + (qsteps - 1) // stride + 1
        # per-temperature frames carry the trace's temperature and rigidity
        np.testing.assert_array_equal(on["frame_meta"][:SCHED["n_temps"], 0], on["trace"][:-1, 0])
        np.testing.assert_allclose(on["frame_meta"][:SCHED["n_temps"], 5], on["trace"][:-1, 5])
        assert on["frame_meta"][-1, 5] == pytest.approx(on["rigid"])
        # every frame is a valid labelling with no empty zone
        assert all(np.bincount(f, minlength=k).min() >= 1 for f in on["frames"])


def test_country_seed_is_deterministic_and_distinct():
    s = [NA.country_seed("IE", 12345, 10000, r) for r in range(12)]
    assert len(set(s)) == 12 and s == [NA.country_seed("IE", 12345, 10000, r) for r in range(12)]
    assert NA.country_seed("IE", 12345, 0, 0) != s[0]


# --- real data: test 1 against results/sweep.csv, test 11 T0 band -------------

@pytest.fixture(scope="module")
def ctx():
    from bzgen.ndbz import sweep as SW
    return SW.load_context(load_config())


needs_solved = pytest.mark.skipif(
    not (DATA and (config.ROOT / "data" / "solved" / "gen_mean.csv").exists()
         and S.persisted_path(load_config()).exists()),
    reason="data/interim + data/solved + results/ndbz/scenarios needed")


@needs_solved
def test_static_reproduction_against_sweep_csv(ctx):
    """Test 1 on the real graphs: lambda_rigid = 0, baseline scenario, static schedule
    and floor.  The static seeds used Python's randomised ``hash(c)`` and cannot be
    replayed, so: the best of 3 restarts lies within the recorded restart range
    [E_min - spread, E_max] of results/sweep.csv for every country."""
    from bzgen.ndbz import sweep as SW
    cfg = ctx.cfg
    a = cfg["anneal"]
    rows = pd.read_csv(config.ROOT / "results" / "sweep.csv")
    rows = rows[rows.config_id == cfg["ndbz"]["anchor_config_id"]].set_index("country")
    for c in rows.index:
        g, nodes, host, anc, _, _ = SW.country_problem(ctx, c, "baseline")
        E = min(NA.anneal_ndbz(g, anc.k, anc.A, anc.cap, anc.Z, 0.0, ctx.params["lambda_b"],
                               a["balance_floor"], a["n_temps"], a["sweeps_per_temp"],
                               a["t_final_ratio"], seed=NA.country_seed(c, a["seed"], 0, r),
                               quench_sweeps=a["quench_sweeps"], mode="static",
                               lam_c0=ctx.params["lambda_c_initial"])["energy"] for r in range(3))
        r = rows.loc[c]
        assert r.E_min - max(r.E_spread, 1e-6 * abs(r.E_min)) - 1e-9 <= E <= r.E_max + 1e-9, (c, E, r.E_min, r.E_max)


@needs_solved
def test_T0_in_band_for_every_country_and_scenario(ctx):
    """Test 11: with lambda_c at the contiguity guarantee, the physical-term T0 is in the
    sane band everywhere; with the guarantee in the sampled dE it would not be."""
    from bzgen.ndbz import sweep as SW
    lo, hi = ctx.cfg["anneal"]["T0_band"]
    for sc in ctx.cfg["scenario"]["names"]:
        for c, anc in ctx.anchors.items():
            try:
                g, nodes, host, anc, floor, _ = SW.country_problem(ctx, c, sc)
            except SW.ScenarioUndefined:
                assert (sc, c) == ("wind_surplus", "AL")
                continue
            lab = graph_voronoi(g, anc.k, np.random.default_rng(0))
            lam_c = contiguity_guarantee(g, ctx.params["lambda_b"])
            T0 = static_anneal.initial_temperature(g, lab, anc.k, ctx.params["lambda_b"], lam_c,
                                                   floor, np.random.default_rng(1),
                                                   exclude_contiguity=True)
            assert np.isfinite(T0) and lo < T0 < hi, (sc, c, T0)
            for lr in ctx.cfg["rigidity"]["lambda"]:
                T = NA.initial_temperature_ndbz(g, lab, anc.k, ctx.params["lambda_b"], floor,
                                                np.random.default_rng(1), anc.A, anc.cap, lr / anc.Z)
                assert np.isfinite(T) and lo < T < hi, (sc, c, lr, T)

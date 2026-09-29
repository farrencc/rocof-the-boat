"""The anchor A: a static map, relabelled globally, on the European node order.

``results/configs/<cid>/labels.csv`` holds per-country zone labels ``0 .. k_c-1``
(so FR zone 0 and DE zone 0 collide once borders are open).  They are mapped to
global labels ``0 .. K-1`` by sorting ``(country, zone)``.  The file already
contains the post-hoc assignments of the buses that were isolated within their
country (3771 rows for 3763 annealed nodes), so every European node is covered.

Contiguity of A on the European graph
-------------------------------------
Adding cross-border edges only merges a zone's components, so every static zone
stays connected *over the nodes that were annealed*.  The formerly-isolated
buses are the exception by construction: their every edge crosses a border, so
under their (national) static label each is a singleton component of its zone.
``check_contiguity`` asserts exactly this structure — every zone is connected
once those buses are set aside, and each of them is a singleton fragment — and
anything else fails (it would mean the relabelling is wrong).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from bzgen import config
from bzgen.cluster.anneal import zone_components

CONFIGS = config.ROOT / "results" / "configs"


def read_static(cid: str) -> pd.DataFrame:
    f = CONFIGS / cid / "labels.csv"
    if not f.exists():
        raise FileNotFoundError(f"anchor labels not found: {f}")
    return pd.read_csv(f)


def relabel(lab: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """(global label per bus, mapping table country, zone -> global)."""
    if lab.bus.duplicated().any():
        raise ValueError("duplicate buses in anchor labels")
    pairs = lab[["country", "zone"]].drop_duplicates().sort_values(["country", "zone"])
    pairs = pairs.reset_index(drop=True)
    pairs["global"] = np.arange(len(pairs))
    k_c = lab.groupby("country").zone.nunique()
    zmax = lab.groupby("country").zone.max()
    if not (zmax + 1 == k_c).all():
        bad = k_c[zmax + 1 != k_c].index.tolist()
        raise ValueError(f"per-country labels are not 0..k_c-1 in {bad}")
    K = len(pairs)
    assert K == int(k_c.sum()), (K, int(k_c.sum()))
    key = pd.MultiIndex.from_frame(pairs[["country", "zone"]])
    g = pd.Series(pairs["global"].to_numpy(), index=key)
    A = pd.Series(g.reindex(pd.MultiIndex.from_frame(lab[["country", "zone"]])).to_numpy(),
                  index=lab.bus.to_numpy(), name="A")
    assert A.notna().all()
    # bijection: (country, zone) <-> global, every global label used
    assert np.array_equal(np.sort(A.unique()), np.arange(K))
    assert pairs["global"].is_unique and not pairs[["country", "zone"]].duplicated().any()
    return A.astype(np.int64), pairs


def load(cid: str, nodes: list, buses: pd.DataFrame | None = None) -> dict:
    """Anchor for European node order ``nodes``: {A (int64 array), K, pairs, k_c}."""
    lab = read_static(cid)
    if buses is not None:
        cc = lab.bus.map(buses.cluster_country)
        if cc.isna().any() or (cc != lab.country).any():
            raise ValueError("anchor country column disagrees with buses.cluster_country")
    A, pairs = relabel(lab)
    missing = [b for b in nodes if b not in A.index]
    if missing:
        raise ValueError(f"{len(missing)} European nodes have no anchor label: {missing[:5]}")
    extra = sorted(set(A.index) - set(nodes))
    if extra:
        raise ValueError(f"{len(extra)} anchor buses are not European nodes: {extra[:5]}")
    a = A.reindex(nodes).to_numpy(np.int64)
    K = len(pairs)
    cnt = np.bincount(a, minlength=K)
    assert (cnt > 0).all(), "empty global anchor zone"
    return {"A": a, "K": K, "pairs": pairs, "k_c": lab.groupby("country").zone.nunique(),
            "cid": cid}


def isolated_within_country(g, nodes: list, buses: pd.DataFrame) -> list:
    """Nodes all of whose European edges cross a border."""
    cc = buses.cluster_country.reindex(nodes).to_numpy()
    out = []
    for i in range(g.n):
        nb = g.idx[g.ptr[i]:g.ptr[i + 1]]
        if len(nb) and (cc[nb] != cc[i]).all():
            out.append(nodes[i])
    return out


def check_contiguity(A: np.ndarray, K: int, g, nodes: list, buses: pd.DataFrame) -> dict:
    """Assert A's contiguity structure on the European graph (module docstring)."""
    C = zone_components(A, K, g.ptr, g.idx)
    iso = isolated_within_country(g, nodes, buses)
    pos = {b: i for i, b in enumerate(nodes)}
    iso_idx = np.array([pos[b] for b in iso], np.int64)
    # every formerly-isolated bus is a singleton fragment of its zone
    for i in iso_idx:
        nb = g.idx[g.ptr[i]:g.ptr[i + 1]]
        assert (A[nb] != A[i]).all(), f"isolated bus {nodes[i]} shares a zone with a neighbour"
    # removing them, every zone is exactly one component
    expect = np.ones(K, np.int64)
    np.add.at(expect, A[iso_idx], 1)
    if not np.array_equal(C, expect):
        bad = np.flatnonzero(C != expect)
        raise AssertionError(f"anchor zones {bad.tolist()} have {C[bad].tolist()} components, "
                             f"expected {expect[bad].tolist()}: relabelling is wrong")
    return {"components": C, "isolated": iso, "excess_components": int(C.sum() - K),
            "zones_with_fragment": sorted({int(A[i]) for i in iso_idx})}


# --------------------------------------------------------------------------- #
# contingency tables and the transfer distance (diagnostic, never in the loop)
# --------------------------------------------------------------------------- #

def contingency(B: np.ndarray, A: np.ndarray, KB: int, KA: int, w=None) -> np.ndarray:
    """n[p, a] = #{i : B_i = p, A_i = a} (or the sum of w_i)."""
    t = np.zeros((KB, KA))
    np.add.at(t, (B, A), 1.0 if w is None else w)
    return t


def transfer_distance(B: np.ndarray, A: np.ndarray, KB: int | None = None,
                      KA: int | None = None, w=None) -> float:
    """N minus the maximum-weight one-to-one matching of B-zones to A-zones on the
    contingency table: the number (or w-weight) of nodes that changed zone."""
    KB = int(B.max()) + 1 if KB is None else KB
    KA = int(A.max()) + 1 if KA is None else KA
    t = contingency(B, A, KB, KA, w)
    r, c = linear_sum_assignment(t, maximize=True)
    return float(t.sum() - t[r, c].sum())

"""Normalised-Laplacian spectrum per country, eigengap heuristic for k.

W: J~ (after the normalisation remedy) on AC edges; HVDC edges get the country's
median AC J~ so islands attached only by HVDC do not produce spurious zero
eigenvalues.  L_sym = I - D^-1/2 W D^-1/2.  Eigenvalues sorted ascending
lambda_1 = 0 <= lambda_2 <= ...

Eigengap heuristic, corrected for the growth of gaps with index (see
``choose_k``).  Where no gap stands out the documented fallback k is used and
flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def laplacian_spectrum(nodes: list, edges: pd.DataFrame) -> np.ndarray:
    pos = {b: i for i, b in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n))
    ac = ~edges.is_dc.to_numpy()
    med = float(np.median(edges.J_n[ac])) if ac.any() else 1.0
    for a, b, j, dc in zip(edges.bus_a, edges.bus_b, edges.J_n, edges.is_dc):
        v = med if dc else j
        W[pos[a], pos[b]] += v
        W[pos[b], pos[a]] += v
    d = W.sum(1)
    with np.errstate(divide="ignore"):
        dm = np.where(d > 0, 1 / np.sqrt(d), 0.0)
    L = np.eye(n) - dm[:, None] * W * dm[None, :]
    return np.sort(np.linalg.eigvalsh(L))


def choose_k(ev: np.ndarray, k_max: int, gap_ratio: float, k_fallback: int, n: int,
             window: int = 3) -> dict:
    """Eigengap with a correction for the systematic growth of gaps with index.

    On near-planar sparse graphs the small eigenvalues crowd near 0 and the
    spacing grows with k, so the *largest* gap in 2..k_max is biased towards
    k_max.  Each gap g_k = lambda_{k+1} - lambda_k is therefore compared with
    the median of its neighbouring gaps (|j - k| <= window, j != k):
    significance s_k = g_k / median(neighbours).  k* = argmax s_k; the elbow is
    *convincing* if s_k >= gap_ratio.  The plain largest-gap k is reported too.
    """
    # at least ~4 buses per zone on average: a "gap" at k close to n is not a zoning signal
    k_max = int(min(k_max, max(2, n // 4), n - 2))
    gaps = np.diff(ev)                      # gaps[k-1] = lambda_{k+1} - lambda_k
    if k_max < 2:
        return {"k_elbow": None, "gap": np.nan, "significance": np.nan, "convincing": False,
                "k_used": min(2, n), "k_largest_gap": None, "gap_k1": float(gaps[0]) if len(gaps) else np.nan,
                "lambda2": float(ev[1]) if len(ev) > 1 else np.nan,
                "note": f"FALLBACK k={min(2, n)}: too few nodes for an eigengap"}
    sig = {}
    for k in range(2, k_max + 1):
        nb = [gaps[j - 1] for j in range(max(1, k - window), min(len(gaps), k + window) + 1) if j != k]
        med = float(np.median(nb)) if nb else np.nan
        sig[k] = gaps[k - 1] / med if med > 0 else np.inf
    k_star = max(sig, key=sig.get)
    k_big = int(np.argmax(gaps[1:k_max]) + 2)
    conv = bool(sig[k_star] >= gap_ratio)
    k_used = k_star if conv else int(max(2, min(k_fallback, k_max)))
    return {"k_elbow": int(k_star), "gap": float(gaps[k_star - 1]), "significance": float(sig[k_star]),
            "convincing": conv, "k_used": k_used, "k_largest_gap": k_big,
            "gap_k1": float(gaps[0]), "lambda2": float(ev[1]),
            "note": "eigengap" if conv else f"FALLBACK k={k_used}: no gap >= {gap_ratio}x local spacing"}

"""Normalised-Laplacian spectrum per country, eigengap heuristic for k.

W: J~ (after the normalisation remedy) on AC edges; HVDC edges get the country's
median AC J~ so islands attached only by HVDC do not produce spurious zero
eigenvalues.  L_sym = I - D^-1/2 W D^-1/2.  Eigenvalues sorted ascending
lambda_1 = 0 <= lambda_2 <= ...

Eigengap heuristic: k* = argmax_{2 <= k <= k_max} (lambda_{k+1} - lambda_k).
"Convincing" when that gap is at least ``gap_ratio`` times the next-largest gap
in the same range.  Otherwise the documented fallback k is used and flagged.
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


def choose_k(ev: np.ndarray, k_max: int, gap_ratio: float, k_fallback: int, n: int) -> dict:
    k_max = int(min(k_max, n - 1))
    gaps = np.diff(ev[: k_max + 1])          # gaps[k-1] = lambda_{k+1} - lambda_k
    cand = np.arange(2, k_max + 1)
    if len(cand) == 0:
        return {"k_elbow": None, "gap": np.nan, "gap_ratio": np.nan, "convincing": False,
                "k_used": 1, "gap_k1": float(gaps[0]) if len(gaps) else np.nan,
                "note": "too few nodes for k >= 2"}
    g = gaps[cand - 1]
    order = np.argsort(g)[::-1]
    k_star = int(cand[order[0]])
    ratio = float(g[order[0]] / g[order[1]]) if len(g) > 1 and g[order[1]] > 0 else np.inf
    conv = bool(ratio >= gap_ratio)
    k_used = k_star if conv else int(min(k_fallback, k_max))
    return {"k_elbow": k_star, "gap": float(g[order[0]]), "gap_ratio": ratio,
            "convincing": conv, "k_used": k_used, "gap_k1": float(gaps[0]),
            "lambda2": float(ev[1]) if len(ev) > 1 else np.nan,
            "note": "eigengap" if conv else f"FALLBACK k={k_used}: eigengap not convincing"}

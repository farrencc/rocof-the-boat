"""Step 3: structural Hamiltonian over an N x K anchored membership matrix.

Sign convention (from upstream ``_required_reduction``): curtailing a group
relieves a POSITIVE overload only if its pro-rata sensitivity S_g < 0. Anchor k
has a fixed orientation s_k (its dominant overload direction) and

    sigma_ik = -s_k * node_state_sensitivity[i, state(e_k)]      (see anchors.py)

so sigma_ik > 0 means "curtailing node i helps relieve anchor k". The
sensitivity is frozen (PTDF/LODF are topology, not dispatch), so sigma is a
constant matrix; only D(G) depends on the frozen cases.

Raw terms (m_i = mec_mw, members of group k = {i : M[i,k]}):

  T1  DD       D(G): mean dispatch-down % over the training ensemble
  T2  mean     -(1/K) sum_k  mu_k,   mu_k = sum_i m_i sigma_ik / sum_i m_i
  T3  variance  (1/K) sum_k  sum_i m_i (sigma_ik - mu_k)^2 / sum_i m_i
  T4  size      sum_k n_k^2
  T5  hinge     sum_k sum_{i in k} m_i * max(0, -sigma_ik)     (soft wrong-sign penalty)

Normalisation (chosen: z-score against a random-membership probe). Each term is
mapped to  (T_j - T_j(G0)) / sd_j,  where G0 is the initial (shift-factor)
grouping and sd_j is the standard deviation of T_j over R random memberships
with the same per-group density as G0. So E(G0) = 0 and lam_j = 1 means "one
random-spread unit of term j". Dividing by the G0 value was rejected because
T5(G0) = 0 by construction (G0 holds only helpful-sign nodes) and T2 is negative.
The probe's D(G) ignores the security guard (random groupings are mostly
infeasible, but their D is still a valid scale).

    E = lam_DD*T1' + lam_S*T2' + lam_V*T3' + lam_N*T4' + lam_P*T5'

The upstream security guard is kept: a candidate that makes any snapshot of any
training member insecure that was secure under the baseline (multi-membership
WDT) returns E = +inf. Raw D(G) is always reported alongside E.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TERMS = ("DD", "S", "V", "N", "P")


def structural_terms(M: np.ndarray, sigma: np.ndarray, mec: np.ndarray) -> np.ndarray:
    """Raw T2..T5 for membership M (N x K bool). Empty groups contribute nothing."""
    Mf = M.astype(float)
    wm = Mf * mec[:, None]                      # N x K MEC weights of members
    tot = wm.sum(axis=0)
    safe = np.where(tot > 0, tot, 1.0)
    mu = (wm * sigma).sum(axis=0) / safe
    var = (wm * (sigma - mu[None, :]) ** 2).sum(axis=0) / safe
    live = tot > 0
    K = max(int(live.sum()), 1)
    T2 = -float(mu[live].sum()) / K
    T3 = float(var[live].sum()) / K
    T4 = float((Mf.sum(axis=0) ** 2).sum())
    T5 = float((wm * np.maximum(0.0, -sigma)).sum())
    return np.array([T2, T3, T4, T5])


def group_stats(M: np.ndarray, sigma: np.ndarray, mec: np.ndarray) -> list[dict]:
    out = []
    for k in range(M.shape[1]):
        idx = np.flatnonzero(M[:, k])
        w = mec[idx]
        s = sigma[idx, k]
        mu = float(np.sum(w * s) / w.sum()) if len(idx) else np.nan
        out.append(dict(k=k, n=int(len(idx)), mec_mw=float(w.sum()), mean_sigma=mu,
                        var_sigma=float(np.sum(w * (s - mu) ** 2) / w.sum()) if len(idx) else np.nan,
                        n_wrong_sign=int(np.sum(s < 0)), wrong_sign_mec=float(w[s < 0].sum())))
    return out


def random_membership(rng: np.random.Generator, N: int, density: np.ndarray) -> np.ndarray:
    M = rng.random((N, len(density))) < density[None, :]
    for k in range(M.shape[1]):
        if not M[:, k].any():
            M[rng.integers(N), k] = True
    return M


@dataclass
class Normaliser:
    offset: np.ndarray   # raw terms at G0 (5,)
    scale: np.ndarray    # probe sd (5,)
    probe: np.ndarray    # R x 5 raw probe terms, kept for the record


class Hamiltonian:
    def __init__(self, ens, sigma: np.ndarray, mec: np.ndarray, lambdas, normaliser: Normaliser):
        self.ens = ens
        self.sigma = np.asarray(sigma, float)
        self.mec = np.asarray(mec, float)
        self.lam = np.asarray(lambdas, float)
        assert self.lam.shape == (5,)
        self.norm = normaliser

    @staticmethod
    def build_normaliser(ens, sigma, mec, M0: np.ndarray, n_probe: int = 64, seed: int = 12345) -> Normaliser:
        rng = np.random.default_rng(seed)
        density = M0.mean(axis=0)

        def raw(M):
            return np.concatenate([[ens(M)["D"]], structural_terms(M, sigma, mec)])

        probe = np.array([raw(random_membership(rng, M0.shape[0], density)) for _ in range(n_probe)])
        scale = probe.std(axis=0, ddof=1)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return Normaliser(offset=raw(M0), scale=scale, probe=probe)

    def evaluate(self, M: np.ndarray) -> dict:
        ev = self.ens(M)
        raw = np.concatenate([[ev["D"]], structural_terms(M, self.sigma, self.mec)])
        z = (raw - self.norm.offset) / self.norm.scale
        contrib = self.lam * z
        E = float(contrib.sum()) if ev["feasible"] else float("inf")
        return dict(E=E, D=ev["D"], raw=raw, z=z, contrib=contrib, feasible=ev["feasible"],
                    new_failures=ev["new_failures"], security_pct=ev["security_pct"])

    __call__ = evaluate

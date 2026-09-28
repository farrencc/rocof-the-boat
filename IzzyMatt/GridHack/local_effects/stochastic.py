"""Everyday frequency: continuous random imbalance, not one big event.

A generator trip is the wrong model for what a grid actually does most of the
time.  Frequency wanders continuously because load and weather-driven generation
never balance exactly, and it is that wandering -- not the rare trip -- that most
RoCoF measurements are taken from.

So the forcing here is a stationary random process at every bus: an
Ornstein-Uhlenbeck power imbalance

    d(xi_i) = -xi_i/tau dt + sigma_i sqrt(2/tau) dW_i,

which has stationary standard deviation sigma_i and correlation time tau.  That
is a crude stand-in for a real load/wind spectrum -- real ones are broadband and
closer to 1/f -- so `tau` is swept rather than trusted.

Because the whole system is linear, none of this needs Monte Carlo.  The state
[x; omega; p; xi] obeys dz = A z dt + B dW, whose stationary covariance solves the
Lyapunov equation

    A S + S A^T + B B^T = 0,

and every quantity wanted here is a linear functional of the state, so it comes
out of S exactly.  `montecarlo_check` verifies it against brute-force simulation.

One wrinkle: the uniform phase shift is a zero mode -- the whole grid is free to
rotate, so the absolute angle random-walks and has no stationary variance.  Only
*relative* angles are stationary, so the angle block is expressed in an orthonormal
basis of the space orthogonal to the all-ones vector, which removes the zero mode
without touching any physical quantity (the coupling only ever sees L x, and
L annihilates the all-ones vector).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

import swing as sw

TAU_C = 1.0          # correlation time of the imbalance, seconds
TOTAL_SIGMA = 0.5    # total injected imbalance, p.u., held fixed across configs


def _basis(N: int) -> np.ndarray:
    """Orthonormal basis (N x N-1) of the subspace orthogonal to the all-ones vector."""
    Q, _ = np.linalg.qr(np.column_stack([np.ones(N), np.eye(N)[:, :N - 1]]))
    return Q[:, 1:]


def noise_profile(net: sw.Network, non_inertial: set[int], where: str = "uniform",
                  total: float = TOTAL_SIGMA, scale: str = "total") -> np.ndarray:
    """Per-bus imbalance magnitude sigma_i.

    Placement:

    ``uniform``        every bus fluctuates equally.  The fluctuation pattern is
                       then identical across configurations, so any difference in
                       the answer is the inertia map and nothing else.
    ``at_converters``  the fluctuation enters at the converter buses, which is the
                       physical situation: wind and solar are what vary.  More
                       realistic, but it moves the *source* of the noise at the same
                       time as the inertia, so the two effects are confounded --
                       which is exactly why both are reported.

    Scaling -- this matters as soon as the *number* of converters varies:

    ``total``    total injected variance is held fixed however many converters
                 there are.  Right for comparing placements at a fixed share,
                 which is what the controlled sweeps do.
    ``per_bus``  each converter carries the same variability regardless of how
                 many there are, so aggregate fluctuation grows with the share --
                 sub-linearly in sigma, since independent farms partly cancel.
                 Right for an SNSP sweep: building more wind adds variability, it
                 does not redistribute a fixed amount of it.  Using ``total`` for
                 a share sweep makes each farm quieter as you add farms, which
                 gets the sign of the whole effect backwards.
    """
    N = net.lat.N
    s = np.zeros(N)
    per = total / np.sqrt(N)
    if where == "uniform":
        s[:] = per
    elif where == "at_converters":
        idx = sorted(non_inertial)
        if not idx:
            raise ValueError("no converter buses to put the noise on")
        s[idx] = per if scale == "per_bus" else total / np.sqrt(len(idx))
    else:
        raise ValueError(f"unknown noise placement: {where!r}")
    if scale not in ("total", "per_bus"):
        raise ValueError(f"unknown noise scaling: {scale!r}")
    return s


def correlation(lat: sw.Lattice, length: float) -> np.ndarray:
    """Spatial correlation of the imbalance between buses, exp(-d/length).

    ``length = 0`` makes every bus fluctuate independently.  That is the wrong
    picture for wind: farms a few kilometres apart see much the same weather, so
    a geographic cluster of them fluctuates largely in step.  Independent noise
    lets a cluster's fluctuations cancel in the aggregate, which flatters
    clustering; correlated noise removes that cancellation.
    """
    if length <= 0:
        return np.eye(lat.N)
    return np.exp(-lat.dist / length)


def _system(net: sw.Network, sigma: np.ndarray, tau: float,
            corr: np.ndarray | None = None):
    """Assemble (A, BB^T, index slices) for the stationary problem."""
    N = net.lat.N
    Minv = 1.0 / net.M
    L = sw.laplacian(net)
    V = _basis(N)
    na = N - 1
    n = na + 3 * N
    a, w, p, x = (slice(0, na), slice(na, na + N),
                  slice(na + N, na + 2 * N), slice(na + 2 * N, n))

    A = np.zeros((n, n))
    A[a, w] = V.T
    A[w, a] = -Minv[:, None] * (L @ V)
    A[w, w] = -np.diag(Minv * net.D)
    A[w, p] = np.diag(Minv)
    A[w, x] = np.diag(Minv)
    A[p, w] = -np.diag(net.droop / (sw.OMEGA_S * net.T_gov))
    A[p, p] = -np.diag(1.0 / net.T_gov)
    A[x, x] = -np.eye(N) / tau

    C = np.eye(N) if corr is None else corr
    Q = np.zeros((n, n))
    Q[x, x] = (sigma[:, None] * C * sigma[None, :]) * 2.0 / tau
    return A, Q, (a, w, p, x), V, L, Minv


def covariance(net: sw.Network, sigma: np.ndarray, tau: float = TAU_C,
               corr: np.ndarray | None = None):
    """Stationary covariance of the full state, plus the pieces to use it.

    Returns (S, A, slices).  Any quantity that is a linear functional c of the
    state has variance c S c^T, which is how the group-separation statistics in
    `separation_sd` are obtained without simulating anything.
    """
    A, Q, sl, V, L, Minv = _system(net, sigma, tau, corr)
    S = solve_continuous_lyapunov(A, -Q)
    return 0.5 * (S + S.T), A, sl


def separation_sd(net: sw.Network, sigma: np.ndarray, group: np.ndarray,
                  tau: float = TAU_C, corr: np.ndarray | None = None) -> float:
    """Standard deviation of (mean frequency over `group`) - (system frequency).

    This is the stochastic version of the question figure 1 asks of a single
    event: does a region hold a frequency of its own, distinct from the system's?
    Exact, in Hz.
    """
    S, _, (a, w, p, x) = covariance(net, sigma, tau, corr)
    n = S.shape[0]
    c = np.zeros(n)
    g = np.asarray(group, bool)
    c[w] = g / g.sum() - net.M / net.M_total
    return float(np.sqrt(max(c @ S @ c, 0.0)) / (2.0 * np.pi))


def realise(net: sw.Network, sigma: np.ndarray, tau: float = TAU_C,
            t_end: float = 120.0, dt: float = 5e-3, seed: int = 0):
    """One time-domain realisation of the noise-driven grid.

    Exact exponential stepping, so the sample path is a true draw from the
    stationary process rather than an Euler approximation of one.  Returns
    (t, frequency deviation per bus in Hz, system frequency in Hz).
    """
    A, Q, (a, w, p, x), V, L, Minv = _system(net, sigma, tau)
    n = A.shape[0]
    Ad = expm(A * dt)
    big = np.zeros((2 * n, 2 * n))
    big[:n, :n] = -A
    big[:n, n:] = Q
    big[n:, n:] = A.T
    E = expm(big * dt)
    Qd = 0.5 * (Ad @ E[:n, n:] + (Ad @ E[:n, n:]).T)
    ev, evec = np.linalg.eigh(Qd)
    Lc = evec @ np.diag(np.sqrt(np.maximum(ev, 0.0)))

    rng = np.random.default_rng(seed)
    steps = int(t_end / dt)
    S0, _, _ = covariance(net, sigma, tau)
    ev0, evec0 = np.linalg.eigh(0.5 * (S0 + S0.T))
    z = evec0 @ (np.sqrt(np.maximum(ev0, 0.0)) * rng.standard_normal(n))
    out = np.empty((steps, net.lat.N))
    for k in range(steps):
        z = Ad @ z + Lc @ rng.standard_normal(n)
        out[k] = z[w]
    t = np.arange(steps) * dt
    m = net.M / net.M_total
    return t, out / (2.0 * np.pi), (out @ m) / (2.0 * np.pi)


def stationary(net: sw.Network, sigma: np.ndarray, tau: float = TAU_C,
               window: float = 0.5, corr: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Exact per-bus frequency and RoCoF statistics under continuous forcing.

    Returns standard deviations, in Hz and Hz/s:

      freq_sd        of the bus frequency itself
      rocof_sd       of the instantaneous RoCoF
      rocof_window_sd of RoCoF measured over `window` -- the grid-code form.  This
                     uses the two-time covariance E[z(t+T) z(t)^T] = exp(A T) S,
                     since a windowed derivative depends on the state at two times.
      coi_rocof_window_sd  the same for the system as a whole, i.e. what a purely
                     global inertia argument would predict for every bus.
    """
    N = net.lat.N
    A, Q, (a, w, p, x), V, L, Minv = _system(net, sigma, tau, corr)
    S = solve_continuous_lyapunov(A, -Q)
    S = 0.5 * (S + S.T)

    # instantaneous RoCoF is a linear functional of the state
    n = A.shape[0]
    C = np.zeros((N, n))
    C[:, a] = -Minv[:, None] * (L @ V)
    C[:, w] = -np.diag(Minv * net.D)
    C[:, p] = np.diag(Minv)
    C[:, x] = np.diag(Minv)
    rocof_var = np.einsum("ij,jk,ik->i", C, S, C)

    # windowed RoCoF: Var[(w(t+T)-w(t))/T] = 2(S_ii - (e^{AT} S)_ii) / T^2
    ST = expm(A * window) @ S
    wv = np.diag(S)[w]
    wc = np.diag(ST)[w]
    win_var = 2.0 * (wv - wc) / window ** 2

    # the system-wide (centre of inertia) equivalent
    m = net.M / net.M_total
    e = np.zeros(n)
    e[w] = m
    coi_var = 2.0 * (e @ S @ e - e @ ST @ e) / window ** 2

    two_pi = 2.0 * np.pi
    return {
        "freq_sd": np.sqrt(np.maximum(wv, 0.0)) / two_pi,
        "rocof_sd": np.sqrt(np.maximum(rocof_var, 0.0)) / two_pi,
        "rocof_window_sd": np.sqrt(np.maximum(win_var, 0.0)) / two_pi,
        "coi_rocof_window_sd": np.full(N, np.sqrt(max(coi_var, 0.0)) / two_pi),
    }


def montecarlo_check(net: sw.Network, sigma: np.ndarray, tau: float = TAU_C,
                     window: float = 0.5, t_end: float = 4000.0,
                     dt: float = 2e-3, seed: int = 0) -> dict[str, np.ndarray]:
    """Brute-force the same statistics, to confirm the Lyapunov result.

    Exact exponential stepping of the linear SDE: the deterministic part is
    propagated with expm and the noise increment is drawn from its exact
    discrete-time covariance, so the only error is sampling, not discretisation.
    """
    N = net.lat.N
    A, Q, (a, w, p, x), V, L, Minv = _system(net, sigma, tau)
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    Ad = expm(A * dt)
    # discrete noise covariance over one step, by the standard augmented trick
    big = np.zeros((2 * n, 2 * n))
    big[:n, :n] = -A
    big[:n, n:] = Q
    big[n:, n:] = A.T
    E = expm(big * dt)
    Qd = Ad @ E[:n, n:]
    Qd = 0.5 * (Qd + Qd.T)
    ev, evec = np.linalg.eigh(Qd)
    Lc = evec @ np.diag(np.sqrt(np.maximum(ev, 0.0)))

    steps = int(t_end / dt)
    z = np.zeros(n)
    keep = int(window / dt)
    hist = np.empty((steps, N))
    for k in range(steps):
        z = Ad @ z + Lc @ rng.standard_normal(n)
        hist[k] = z[w]
    burn = steps // 10
    h = hist[burn:]
    win = (h[keep:] - h[:-keep]) / window
    return {"freq_sd": h.std(axis=0) / (2 * np.pi),
            "rocof_window_sd": win.std(axis=0) / (2 * np.pi)}

"""Second-order Kuramoto (swing equation) on a toy square lattice.

Implements Eq. (1) of Park, Kim & Kahng, *Optimal location of reinforced
inertia to stabilize power grids* (arXiv:2502.09024),

    M_i d2(theta_i)/dt2 + D_i d(theta_i)/dt = P_i - sum_j K_ij sin(theta_i - theta_j)

with one departure: the paper works in its own reduced units, and we work in
per-unit-on-a-system-base with M_i = 2 H_i / omega_s, so that the second
derivative divides by 2*pi into a RoCoF in Hz/s -- the quantity the grid code is
actually written in (the all-island RoCoF limit is 1 Hz/s over 500 ms).

The experiment this file supports (problem sheet section 3.5) needs one property
above all: the steady state of the equation above depends on P and K only, never
on M.  So a family of networks that share P and differ only in how inertia is
spread over the buses all share *the same* pre-fault operating point.  Every
difference in the transient is then attributable to the geography of inertia and
to nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

OMEGA_S = 2.0 * np.pi * 50.0  # rad/s, nominal system frequency

# Inertia constants H (seconds), on each unit's own 1 p.u. rating.
H_SYNC = 4.0     # a synchronous machine: gas, coal, hydro
H_INV = 0.1      # an inverter-based unit: wind, solar.  Not zero -- see README.
H_LOAD = 0.5     # residual motor inertia behind a load bus

# Damping, p.u. power per rad/s, identical at every bus.  Load damping in a real
# system is roughly 1-2% of load per 1% of frequency, i.e. of order 1 p.u. per
# p.u. frequency, which is 1/omega_s here.  That puts the system damping time
# constant M_total/sum(D) at a couple of seconds -- frequency keeps falling for
# seconds after an infeed loss, as it does in reality, rather than settling
# inside the 500 ms measurement window.
D_UNIFORM = 1.5 / OMEGA_S


def inertia_from_H(H: float) -> float:
    """M = 2H/omega_s, the coefficient on d2(theta)/dt2 in per-unit."""
    return 2.0 * H / OMEGA_S


@dataclass
class Lattice:
    """An L x L nearest-neighbour square lattice of buses, indexed row-major.

    With ``periodic=True`` the lattice closes into a torus.  That matters more
    than it looks: on an open square lattice a corner bus has two neighbours and
    an interior bus has four, so a bus far from the converters is also a bus
    weakly tied to the rest of the grid, and the two explanations for a high
    RoCoF cannot be told apart.  On a torus every bus has exactly four
    neighbours and the graph is vertex-transitive, so the inertia map is the only
    thing that breaks the symmetry and any spatial structure in the answer has
    only one possible cause.
    """

    L: int
    K: float = 3.0
    periodic: bool = False
    #: Extra (i, j) links beyond the nearest-neighbour grid -- long-range
    #: connections that short-circuit the lattice.  They carry the same coupling
    #: K as every other line unless `extra_K` says otherwise.
    extra_edges: tuple[tuple[int, int], ...] = ()
    extra_K: float | None = None

    def __post_init__(self) -> None:
        if self.periodic and self.L < 3:
            raise ValueError("periodic boundaries need L >= 3 to avoid double edges")
        self.N = self.L * self.L
        self.coords = [(r, c) for r in range(self.L) for c in range(self.L)]
        A = np.zeros((self.N, self.N))
        for i, (r, c) in enumerate(self.coords):
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if self.periodic:
                    rr, cc = rr % self.L, cc % self.L
                elif rr >= self.L or cc >= self.L:
                    continue
                j = rr * self.L + cc
                A[i, j] = A[j, i] = 1.0
        self.A = A
        self.Kij = self.K * A
        for i, j in self.extra_edges:
            if i == j:
                raise ValueError(f"self-loop in extra_edges: {i}")
            A[i, j] = A[j, i] = 1.0
            k = self.K if self.extra_K is None else self.extra_K
            self.Kij[i, j] = self.Kij[j, i] = k

        # Hop distance through the graph.  On a plain lattice this is exactly the
        # (wrapped) Manhattan distance, which `validate.py` asserts; with extra
        # links it is not, and every consumer of `dist` wants the real thing.
        self.dist = shortest_path(csr_matrix(A), unweighted=True, directed=False)

    def index(self, r: int, c: int) -> int:
        return r * self.L + c


def checkerboard_power(lat: Lattice) -> np.ndarray:
    """Generators and loads on alternating sites, scaled so that sum(P) = 0.

    The checkerboard is chosen because it is the most spatially symmetric way to
    lay generation over a lattice.  Any asymmetry in the results is then imposed
    by the inertia configuration under test and is not inherited from the
    injection pattern.
    """
    sign = np.array([1.0 if (r + c) % 2 == 0 else -1.0 for r, c in lat.coords])
    P = sign.copy()
    gen, load = sign > 0, sign < 0
    P[gen] = 1.0
    P[load] = -gen.sum() / load.sum()
    return P


def steady_state(lat: Lattice, P: np.ndarray) -> np.ndarray:
    """Solve P_i = sum_j K_ij sin(theta_i - theta_j) for the phase angles.

    The solution is defined only up to a global rotation, so bus 0 is pinned at
    zero and its (redundant, since sum(P) = 0) balance equation is dropped.
    """

    def residual(x: np.ndarray) -> np.ndarray:
        th = np.concatenate(([0.0], x))
        d = th[:, None] - th[None, :]
        return (P - (lat.Kij * np.sin(d)).sum(axis=1))[1:]

    def jac(x: np.ndarray) -> np.ndarray:
        th = np.concatenate(([0.0], x))
        d = th[:, None] - th[None, :]
        C = lat.Kij * np.cos(d)
        J = C - np.diag(C.sum(axis=1))
        return J[1:, 1:]

    sol = root(residual, np.zeros(lat.N - 1), jac=jac, method="hybr", tol=1e-12)
    if not sol.success:
        raise RuntimeError(f"no synchronous steady state: {sol.message}")
    theta = np.concatenate(([0.0], sol.x))
    resid = P - (lat.Kij * np.sin(theta[:, None] - theta[None, :])).sum(axis=1)
    if np.abs(resid).max() > 1e-8:
        raise RuntimeError("steady state did not converge")
    return theta


#: Governor droop, p.u. power per p.u. frequency.  R = 5% is the usual setting,
#: so a machine raises output by 1/0.05 = 20 p.u. per p.u. of frequency error.
DROOP_GAIN = 20.0
#: Governor time constant, seconds.  A thermal or gas unit takes several seconds
#: to actually move steam or fuel; this lag is the whole reason RoCoF exists as a
#: separate problem from frequency nadir.
T_GOV = 5.0
#: Fast frequency response: same droop law, but delivered in about a second.
#: Ireland's FFR products are specified to arrive inside 2 s.
T_FFR = 1.0


@dataclass
class Network:
    """A lattice plus an inertia/damping assignment: everything Eq. (1) needs.

    ``droop`` and ``T_gov`` add primary frequency response: each bus carries an
    extra state p_i obeying

        T_i dp_i/dt = -p_i - droop_i * (omega_i / omega_s),

    and p_i is added to the injection in the swing equation.  Physically, a
    governor sees the frequency fall and opens up, restoring the balance over
    several seconds.  With droop = 0 (the default) the model reduces exactly to
    the pure swing equation used for the RoCoF results.
    """

    lat: Lattice
    P: np.ndarray
    M: np.ndarray
    D: np.ndarray
    droop: np.ndarray | None = None
    T_gov: np.ndarray | None = None
    theta0: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.theta0 = steady_state(self.lat, self.P)
        if self.droop is None:
            self.droop = np.zeros(self.lat.N)
        if self.T_gov is None:
            self.T_gov = np.full(self.lat.N, T_GOV)

    @property
    def has_governor(self) -> bool:
        return bool(np.any(self.droop != 0.0))

    @property
    def M_total(self) -> float:
        return float(self.M.sum())

    def max_angle_gap(self) -> float:
        """Largest |theta_i - theta_j| over lines: how hard the grid is pushed."""
        d = np.abs(self.theta0[:, None] - self.theta0[None, :])
        return float(d[self.lat.A > 0].max())


#: Damping ratio of the local swing mode, for the "equal_ratio" damping model.
ZETA = 0.15


def damping_vector(lat: Lattice, M: np.ndarray, theta0: np.ndarray,
                   model: str = "equal_ratio", zeta: float = ZETA,
                   D0: float = D_UNIFORM) -> np.ndarray:
    """Damping at each bus, under one of three models.

    ``uniform``      D_i = D0 everywhere.  Isolates the inertia perfectly -- M is
                     then the only thing that differs between configurations --
                     but it leaves a near-zero-inertia bus almost undamped, and
                     such a bus rings at tens of Hz/s in a way no real converter
                     does.  Kept as a robustness check, not as the default.
    ``proportional`` D_i proportional to M_i, the linear fit the paper reports
                     between gamma_i and m_i (its Fig. 1(b)), normalised to the
                     same total damping as ``uniform``.
    ``equal_ratio``  D_i = 2 zeta sqrt(k_i M_i), where k_i = sum_j K_ij cos(...)
                     is the synchronising stiffness seen at bus i.  Every bus
                     then has the same damping ratio for its local swing mode.
                     This is the physically well-posed choice: a converter bus
                     has little inertia but its current control still damps, so
                     it should not be modelled as an undamped resonator.

    ``equal_ratio`` and ``proportional`` both make D a function of M, so inertia
    and damping move together between configurations.  That is unavoidable --
    holding absolute damping fixed while sending inertia to zero is what is
    unphysical -- and it is why every headline result is repeated under
    ``uniform`` in the robustness sweep.
    """
    if model == "uniform":
        return np.full(lat.N, D0)
    if model == "proportional":
        D = M / M.mean() * D0
        return D
    if model == "equal_ratio":
        d = theta0[:, None] - theta0[None, :]
        k = (lat.Kij * np.cos(d)).sum(axis=1)
        return 2.0 * zeta * np.sqrt(k * M)
    raise ValueError(f"unknown damping model: {model!r}")


def build_network(lat: Lattice, non_inertial: set[int], layout: str = "checkerboard",
                  H_inv: float = H_INV, damping: str = "uniform",
                  zeta: float = ZETA, D0: float = D_UNIFORM,
                  droop: float = 0.0, T_gov: float = T_GOV,
                  converter_ffr: float = 0.0, T_ffr: float = T_FFR) -> Network:
    """Assign inertia given the set of buses whose machine carries none.

    ``checkerboard`` alternates generators and loads, so the grid carries real
    pre-fault flows -- but generators then sit at least two hops apart, which
    caps how tightly the converters can be clustered.

    ``flat`` gives every bus a machine and zero net injection.  There are no
    pre-fault flows, which is less realistic, but the synchronising stiffness is
    then exactly uniform across the lattice and *the inertia map is the only
    spatial structure in the whole problem*.  It is the controlled version of the
    experiment, and it lets a converter be placed at any bus, so a tight corner
    cluster is actually reachable.
    """
    if layout == "checkerboard":
        P = checkerboard_power(lat)
        base = np.where(P > 0, H_SYNC, H_LOAD)
    elif layout == "flat":
        P = np.zeros(lat.N)
        base = np.full(lat.N, H_SYNC)
    else:
        raise ValueError(f"unknown layout: {layout!r}")

    H = base.copy()
    for i in non_inertial:
        H[i] = H_inv
    M = inertia_from_H(H)
    theta0 = steady_state(lat, P)
    D = damping_vector(lat, M, theta0, model=damping, zeta=zeta, D0=D0)

    # Governor response is a property of the synchronous plant.  A converter
    # contributes none of it unless it is explicitly given fast frequency
    # response, which is a control choice rather than a physical consequence of
    # the machine -- so it is a separate, faster path with its own gain.
    g = np.full(lat.N, droop)
    T = np.full(lat.N, T_gov)
    for i in non_inertial:
        g[i] = converter_ffr
        T[i] = T_ffr
    return Network(lat=lat, P=P, M=M, D=D, droop=g, T_gov=T)


def gen_trip(net: Network, bus: int) -> np.ndarray:
    """Power step for losing the generator at `bus` outright."""
    dP = np.zeros(net.lat.N)
    dP[bus] = -net.P[bus]
    return dP


def load_step(net: Network, bus: int, size: float = 1.0) -> np.ndarray:
    """Power step for `size` p.u. of extra load appearing at `bus`.

    Every configuration gives load buses the same inertia, so a load step is a
    disturbance that is identical in both magnitude *and* local environment
    across configurations.  It is the cleanest excitation available here: a
    generator trip at a converter bus removes a near-massless node's injection,
    which is a harsher and less well-posed perturbation.
    """
    dP = np.zeros(net.lat.N)
    dP[bus] = -abs(size)
    return dP


def simulate(net: Network, dP: np.ndarray, t_end: float = 1.5,
             dt: float = 1e-3) -> dict[str, np.ndarray]:
    """Apply the power step `dP` at t = 0 and integrate Eq. (1).

    Returns frequency and RoCoF traces in Hz and Hz/s.  RoCoF is evaluated from
    the right-hand side of the equation of motion rather than by differencing the
    frequency trace, so it carries no numerical-differentiation error.
    """
    lat, N = net.lat, net.lat.N
    P = net.P + dP
    delta_P = float(dP.sum())
    Kij, M, D = lat.Kij, net.M, net.D
    g, T = net.droop, net.T_gov

    def accel(theta: np.ndarray, omega: np.ndarray, p: np.ndarray) -> np.ndarray:
        coupling = (Kij * np.sin(theta[:, None] - theta[None, :])).sum(axis=1)
        return (P + p - D * omega - coupling) / M

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        theta, omega, p = y[:N], y[N:2 * N], y[2 * N:]
        dp = (-p - g * omega / OMEGA_S) / T
        return np.concatenate((omega, accel(theta, omega, p), dp))

    t_eval = np.arange(0.0, t_end + 0.5 * dt, dt)
    y0 = np.concatenate((net.theta0, np.zeros(N), np.zeros(N)))
    # Radau: with H_inv small the system is stiff (M/D down to ~1 ms), and an
    # explicit method would either crawl or go unstable.
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="Radau", t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")

    theta, omega, pg = sol.y[:N], sol.y[N:2 * N], sol.y[2 * N:]
    alpha = np.empty_like(omega)
    for k in range(sol.t.size):
        alpha[:, k] = accel(theta[:, k], omega[:, k], pg[:, k])

    M_tot = net.M_total
    # relative angles must stay bounded, or the run has left the synchronous
    # regime and no frequency metric computed from it means anything
    spread = theta - theta.mean(axis=0)
    return {
        "t": sol.t,
        "freq_dev": omega / (2.0 * np.pi),                 # Hz, deviation from 50
        "rocof": alpha / (2.0 * np.pi),                    # Hz/s, per bus
        "rocof_coi": (M @ alpha) / M_tot / (2.0 * np.pi),  # Hz/s, system average
        "freq_coi": (M @ omega) / M_tot / (2.0 * np.pi),
        "delta_P": delta_P,
        "max_angle_spread": float(np.abs(spread).max()),
    }


def laplacian(net: Network) -> np.ndarray:
    """Synchronising stiffness about the operating point: L_ij = K_ij cos(theta_i-theta_j)."""
    d = net.theta0[:, None] - net.theta0[None, :]
    C = net.lat.Kij * np.cos(d)
    return np.diag(C.sum(axis=1)) - C


def resistance_distance(net: Network) -> np.ndarray:
    """Effective electrical distance R_ij between every pair of buses.

    Built from the pseudo-inverse of the synchronising-stiffness Laplacian, so it
    measures nearness through the network as power actually sees it, rather than
    counting hops on the map.
    """
    L = laplacian(net)
    Lp = np.linalg.pinv(L)
    d = np.diag(Lp)
    return d[:, None] + d[None, :] - 2.0 * Lp


def local_inertia(net: Network, lam: float = 0.5) -> np.ndarray:
    """Inertia in a bus's electrical neighbourhood, as an H in seconds.

    A weighted average of every bus's inertia constant, with weight falling off as
    exp(-R_ij / lam) in effective resistance.  As lam -> infinity this tends to the
    system-wide inertia that a global argument would use; at finite lam it is the
    quantity a bus can actually lean on in the first few hundred milliseconds,
    before the disturbance has had time to reach the rest of the grid.

    This is the cheap predictor: it needs one pseudo-inverse and no simulation, so
    it can rank candidate sites for a synchronous condenser directly (section 3.6).
    """
    R = resistance_distance(net)
    W = np.exp(-R / lam)
    H = net.M * OMEGA_S / 2.0
    return (W @ H) / W.sum(axis=1)


def state_matrix(net: Network) -> np.ndarray:
    """The linearised dynamics as one 3N x 3N matrix, in the state [x; omega; p].

    Linearising Eq. (1) about the operating point, with the governor state
    attached, gives

        dx/dt      = omega
        d(omega)/dt = M^-1 ( dP + p - D omega - L x )
        dp/dt      = ( -p - droop * omega / omega_s ) / T

    Shared by the step-response solver and the stochastic model, so both are
    guaranteed to be describing the same system.
    """
    N = net.lat.N
    Minv = 1.0 / net.M
    L = laplacian(net)
    S = np.zeros((3 * N, 3 * N))
    S[:N, N:2 * N] = np.eye(N)
    S[N:2 * N, :N] = -Minv[:, None] * L
    S[N:2 * N, N:2 * N] = -np.diag(Minv * net.D)
    S[N:2 * N, 2 * N:] = np.diag(Minv)
    S[2 * N:, N:2 * N] = -np.diag(net.droop / (OMEGA_S * net.T_gov))
    S[2 * N:, 2 * N:] = -np.diag(1.0 / net.T_gov)
    return S


def simulate_linear(net: Network, dP: np.ndarray, t_end: float = 1.5,
                    dt: float = 1e-3) -> dict[str, np.ndarray]:
    """Same disturbance, solved on the linearisation about the operating point.

    Linearising Eq. (1) in the angle deviations gives

        M d2(x)/dt2 + D d(x)/dt + L x = dP,     L = the stiffness above,

    a linear time-invariant system driven by a step.  It is propagated exactly
    with one matrix exponential of the augmented state [x; dx/dt; 1], which folds
    the constant forcing into the homogeneous problem and sidesteps the fact that
    the state matrix is singular (the grid is free to rotate as a whole).

    This is ~100x faster than integrating the nonlinear equation and, at the
    disturbance sizes used here, agrees with it to a fraction of a percent --
    ``validate.py`` checks that.  It is what makes the exhaustive 1820-configuration
    sweep affordable.
    """
    from scipy.linalg import expm

    N = net.lat.N
    Minv = 1.0 / net.M
    L = laplacian(net)
    S = state_matrix(net)
    n = 3 * N
    b = np.concatenate((np.zeros(N), Minv * dP, np.zeros(N)))

    aug = np.zeros((n + 1, n + 1))
    aug[:n, :n] = S
    aug[:n, n] = b
    step = expm(aug * dt)

    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    z = np.zeros((n + 1, t.size))
    z[n, :] = 1.0
    for k in range(1, t.size):
        z[:, k] = step @ z[:, k - 1]

    x, omega, pg = z[:N], z[N:2 * N], z[2 * N:3 * N]
    alpha = Minv[:, None] * (dP[:, None] + pg - net.D[:, None] * omega - L @ x)

    M_tot = net.M_total
    return {
        "t": t,
        "freq_dev": omega / (2.0 * np.pi),
        "rocof": alpha / (2.0 * np.pi),
        "rocof_coi": (net.M @ alpha) / M_tot / (2.0 * np.pi),
        "freq_coi": (net.M @ omega) / M_tot / (2.0 * np.pi),
        "delta_P": float(dP.sum()),
        "max_angle_spread": float(np.abs(x - x.mean(axis=0)).max()),
    }


def metrics(sim: dict[str, np.ndarray], window: float = 0.5) -> dict[str, np.ndarray]:
    """Per-bus summaries of one disturbance.

    rocof_500ms   mean RoCoF over the first `window` seconds after the event.
                  This is the grid-code form of the quantity: the all-island
                  limit of 1 Hz/s is written against a 500 ms measurement, not
                  against an instantaneous derivative.  It is also the metric
                  that is robust to lightly damped local ringing.
    worst_window  the largest |RoCoF| over any sliding `window` in the run --
                  what a RoCoF relay would actually pick up, whenever it happens.
    peak_instant  largest instantaneous |RoCoF| in the first second.  Reported
                  for completeness; it is dominated by the first quarter-swing of
                  the local mode and is sensitive to the damping model, so it is
                  never used as a headline number.
    t_peak        when the instantaneous peak arrives -- the propagation delay,
                  which is what makes the effect local in the first place.
    local_excess  |rocof_500ms| minus |RoCoF of the centre of inertia| over the
                  same window: the part of the swing that is *local*, with the
                  system-wide component that every bus shares removed.  This is
                  the quantity section 3.5 is asking for.
    """
    t, rocof, f = sim["t"], sim["rocof"], sim["freq_dev"]
    dt = t[1] - t[0]
    early = t <= 1.0
    idx = np.argmax(np.abs(rocof[:, early]), axis=1)
    peak = np.abs(rocof[:, early])[np.arange(rocof.shape[0]), idx]

    kw = int(round(window / dt))
    rocof_500 = (f[:, kw] - f[:, 0]) / window
    coi_500 = (sim["freq_coi"][kw] - sim["freq_coi"][0]) / window
    # every sliding window of the same length, not just the one starting at t=0
    sliding = (f[:, kw:] - f[:, :-kw]) / window
    worst = np.abs(sliding).max(axis=1)

    # Frequency excursion, the other half of what a grid code constrains.  RoCoF
    # relays and loss-of-mains protection watch df/dt; under-frequency load
    # shedding watches f itself.  A bus can be comfortable on one and not the
    # other, so both are reported.  `nadir` is signed (negative for a deficit)
    # and only means anything if primary response is enabled -- without a
    # governor the frequency never turns round and the "nadir" is just wherever
    # the run happened to stop.
    ni = np.argmin(f, axis=1)
    nadir = f[np.arange(f.shape[0]), ni]
    coi_nadir = sim["freq_coi"].min()

    n = rocof.shape[0]
    return {
        "rocof_500ms": rocof_500,
        "worst_window": worst,
        "peak_instant": peak,
        "t_peak": t[early][idx],
        "coi_rocof_500ms": np.full(n, coi_500),
        "local_excess": np.abs(rocof_500) - abs(coi_500),
        "nadir": nadir,
        "t_nadir": t[ni],
        "excursion": np.abs(f).max(axis=1),
        "coi_nadir": np.full(n, coi_nadir),
    }

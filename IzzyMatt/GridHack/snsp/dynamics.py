"""Integrate the swing equations for a configuration and score the response.

The model is the one from ../local_effects/swing.py -- second-order Kuramoto,
Eq. (1) of arXiv:2502.09024 -- in per-unit with M_i = 2 H_i S_i / omega_s (the
inertia constant on the bus's own rating; see grid.inertia), so that
theta-double-dot / 2 pi is a RoCoF in Hz/s:

    M_i th_i'' + D_i th_i' = P_i + p_i - sum_j K_ij sin(th_i - th_j)
    T_i p_i'  = -p_i - droop_i * th_i' / omega_s          (primary response)

with the governor state p_i carried at every committed machine (5% droop, 5 s
lag) and, optionally, fast frequency response at the converters (same droop law,
1 s).  Everything the frequency actually does after a disturbance -- the initial
RoCoF set by inertia, the nadir set by how fast reserve arrives, the settling
offset set by droop and load damping -- is in there.

THE COST FUNCTIONAL

For one configuration, one disturbance and one bus,

    J_i = int_0^T [ tau^2 (df_i/dt)^2 + f_i^2 ] dt,

f being the frequency deviation from 50 Hz in Hz.  Summed over buses and over
the fixed disturbance set, this is the score of a configuration: a
Landau-Ginzburg-shaped functional in which the gradient term penalises RoCoF and
the field term penalises frequency deviation, so a configuration is only cheap
if it is calm on both counts, everywhere, for every event.

`tau` (default 1 s) is what makes the two terms commensurable -- (df/dt)^2 is
Hz^2/s^2 and f^2 is Hz^2, so they cannot be added without a time scale.  It is a
weighting choice, not a physical constant, so the two terms are always reported
separately as well and `--tau` sweeps it.

HOW IT IS SOLVED

Linearised about each configuration's own operating point and propagated with one
matrix exponential.  Writing the state x = [angle; omega; governor] and folding
the constant disturbance in as extra states,

    d/dt [x; u] = [[S, B], [0, 0]] [x; u],   u = dP held constant,

so a *single* 4N x 4N exponential per configuration propagates **every** event in
the set at once -- the whole disturbance set becomes one matrix multiply per time
step.  The propagation is exact at the sample times (no integrator error, no
stiffness limit, and converter buses with H = 0.1 s are stiff); the only
approximation left is the quadrature of the cost integrand, which is Simpson's
rule on the same grid.

`nonlinear_cost()` re-runs a configuration through the full nonlinear solver as a
cross-check; at these disturbance sizes the two agree to 0.03%, which `validate.py`
asserts along with the time-step convergence and the analytic COI RoCoF.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "local_effects"))
import swing as sw                                            # noqa: E402

import grid as G                                              # noqa: E402

TWO_PI = 2.0 * np.pi

#: Cost horizon (s).  Long enough for the initial swing, the nadir at a few
#: seconds and most of the governor recovery; short enough that the settling
#: offset does not simply dominate by being integrated forever.
T_END = 10.0
#: Sample step (s).  The fastest local mode on a converter bus is ~25 Hz, so
#: this is ~20 samples per cycle -- what the Simpson quadrature of the squared
#: derivative needs, the propagation itself being exact at any step.
DT = 2.5e-3
#: The time scale that makes Hz^2/s^2 and Hz^2 addable.  A weighting choice.
TAU = 1.0
#: Measurement time constant (s).  The cost is evaluated on the frequency as it
#: can be *measured*, not on the raw state derivative, and this is why:
#:
#: an injection step landing directly on a converter bus (H = 0.1 s on its own
#: rating) produces theta-double-dot / 2 pi of order 250 Hz/s in the first
#: instant, because M there is ~1e-4 p.u.  That number is an artefact of
#: regularising what is really an algebraic bus with a small mass -- no relay,
#: PLL or PMU reports it, and nothing in the grid responds to it.  Left in, it
#: dominates the integral of (df/dt)^2 completely and the cost function ends up
#: measuring how massless the disturbed bus is rather than how the system
#: behaved.
#:
#: So f is passed through a first-order lag of 100 ms -- the measurement window a
#: RoCoF relay is specified over -- and the functional is evaluated on that.  The
#: filtered derivative is available in closed form, (f - f_meas) / T_meas, so no
#: numerical differentiation enters anywhere.  `--t-meas 0` recovers the raw
#: functional, and the sensitivity to this choice is reported rather than assumed.
T_MEAS = 0.1


def simpson_weights(n: int, dt: float) -> np.ndarray:
    """Composite Simpson weights over n samples (trapezoid if n is even)."""
    w = np.full(n, dt)
    if n % 2 == 1:
        w[:] = dt / 3.0
        w[1:-1:2] = 4.0 * dt / 3.0
        w[2:-1:2] = 2.0 * dt / 3.0
    else:
        w[0] = w[-1] = dt / 2.0
    return w


@dataclass
class Response:
    """What one configuration did, over the whole disturbance set.

    Costs are (n_bus, n_event) arrays in Hz^2 s; `total` sums both axes.
    """

    cost: np.ndarray             # tau^2 (df/dt)^2 + f^2, per bus per event
    cost_rocof: np.ndarray       # the gradient term alone
    cost_freq: np.ndarray        # the field term alone
    coi_cost: np.ndarray         # per event, the same functional on f_coi
    coi_rocof: np.ndarray
    coi_freq: np.ndarray
    peak_rocof: np.ndarray       # per bus per event, |df/dt| max, Hz/s
    peak_dev: np.ndarray         # per bus per event, |f| max, Hz
    rocof_500ms: np.ndarray      # per bus per event, the grid-code window, Hz/s
    nadir: np.ndarray            # per bus per event, deepest dip, Hz (>= 0)
    coi_nadir: np.ndarray        # per event, the same for the system frequency
    coi_rocof_500ms: np.ndarray  # per event, the system 500 ms RoCoF, Hz/s
    #: True where the bus is one the event lands on directly, for events local
    #: enough for that to mean anything (the diffuse event hits every bus and is
    #: never masked).  See `summarise`: the cost is reported both with and
    #: without these, because a step applied *at* a low-inertia bus produces a
    #: large excursion there whose size depends mostly on that bus's own
    #: inertia, and it is worth knowing whether a conclusion rests on it.
    self_mask: np.ndarray

    @property
    def total(self) -> float:
        return float(self.cost.sum())

    @property
    def by_bus(self) -> np.ndarray:
        return self.cost.sum(axis=1)

    @property
    def by_event(self) -> np.ndarray:
        return self.cost.sum(axis=0)


#: Load damping: how much demand falls of its own accord per unit of frequency
#: deviation, p.u. power per p.u. frequency on the *load's* base.  1.5 is the
#: textbook value (a 1% frequency drop sheds about 1.5% of load).
D_LOAD = 1.5
#: Damping from the plant itself, on each unit's own rating: damper windings and
#: a machine's own losses for a synchronised set, the converter's current control
#: for an inverter.  A decommitted station contributes neither.
D_SYNC, D_CONVERTER = 2.0, 1.0


def damping_vector(g, H: np.ndarray, M: np.ndarray, theta0: np.ndarray,
                   model: str = "rating") -> np.ndarray:
    """Damping at each bus, p.u. power per rad/s.

    ``rating`` (the default and the only one that is calibrated) builds it from
    what is physically there: load damping on the demand at the bus, plus plant
    damping on the rating of whatever is synchronised to it.

    The alternatives from ../local_effects/swing.py are kept for the robustness
    sweep but are **not** interchangeable here, and the reason is worth
    recording.  That study gave every bus a machine on a 1 p.u. base, so its
    per-bus damping constants were per-unit-correct.  Here a bus's rating varies
    by two orders of magnitude, and its ``equal_ratio`` rule (D = 2 zeta sqrt(kM))
    applied to a hundred small buses sums to roughly 50 times the physical load
    damping of the whole system -- enough to arrest the frequency in 20 ms, which
    hides the entire effect being measured.  Scaling with the rating is what
    makes the two studies say the same thing about the same physics.
    """
    if model == "rating":
        sync_on = (g.role == "sync") & (H > 1.0)
        conv = np.isin(g.role.astype(str), ("wind", "hvdc"))
        d = D_LOAD * g.demand.copy()
        d[sync_on] += D_SYNC * g.S[sync_on]
        d[conv] += D_CONVERTER * g.S[conv]
        return d / sw.OMEGA_S
    return sw.damping_vector(g, M, theta0, model=model)


def build_network(g, P: np.ndarray, H: np.ndarray, droop: np.ndarray,
                  T_gov: np.ndarray, damping: str = "rating") -> sw.Network:
    """A `swing.Network` for one configuration: its dispatch and its inertia.

    Both change together across the ensemble and that is the point -- a
    configuration with less synchronous generation running has both less inertia
    and less governor response, because it is the same machines that provide
    each.
    """
    M = G.inertia(H, g.S)
    theta0 = sw.steady_state(g, P)
    D = damping_vector(g, H, M, theta0, model=damping)
    return sw.Network(lat=g, P=P, M=M, D=D, droop=droop, T_gov=T_gov)


#: Secondary control (AGC) gain, p.u. power per Hz per second, when it is
#: enabled.  Droop is *proportional* control, so it leaves a standing frequency
#: error by construction: the machines stop increasing output the moment the
#: frequency stops falling, and where they stop is set by dP / (droop + damping).
#: Only integral action returns the frequency to 50 Hz exactly, and in a real
#: system that is secondary control acting over minutes, not seconds.  At this
#: gain a 0.6 p.u. deficit is made up in about a minute, which is realistic --
#: and six times longer than the cost horizon, which is the point: within the
#: 10 s window the frequency *should* still be sitting at its droop offset.
AGC_GAIN = 0.1


def _propagator(net: sw.Network, dt: float, agc: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """exp(A dt) for the augmented system, and the stiffness matrix L.

    A = [[S, B], [0, 0]] on the state [angle; omega; governor; dP], with
    B = [0; M^-1; 0] mapping a held injection step into acceleration.  Every
    event in the set is then just a different initial value of the dP block, so
    one exponential serves all of them.

    With `agc` (participation factors summing to one over the machines that are
    synchronised) a single further state z is appended, carrying integral action
    on the centre-of-inertia frequency:

        z' = -K f_coi ,   and each machine injects agc_i * z

    -- one shared secondary controller, shared out by participation, exactly as
    an AGC does.
    """
    N = net.lat.N
    S = sw.state_matrix(net)
    L = sw.laplacian(net)
    n = 4 * N + (1 if agc is not None else 0)
    A = np.zeros((n, n))
    A[:3 * N, :3 * N] = S
    A[N:2 * N, 3 * N:4 * N] = np.diag(1.0 / net.M)
    if agc is not None:
        Mw = net.M / net.M.sum()
        A[N:2 * N, -1] = agc / net.M
        A[-1, N:2 * N] = -AGC_GAIN * Mw / TWO_PI
    return expm(A * dt), L


def respond(net: sw.Network, dP: np.ndarray, t_end: float = T_END,
            dt: float = DT, tau: float = TAU, t_meas: float = T_MEAS,
            agc: np.ndarray | None = None,
            trace: bool = False) -> Response | tuple[Response, dict]:
    """Propagate every event in `dP` (N x n_events) and score the responses.

    `t_meas` is the measurement lag the cost is evaluated through (see T_MEAS);
    set it to 0 to score the raw state variables instead.  `agc` turns on
    secondary control with those participation factors (see `_propagator`);
    without it the frequency settles at its droop offset and stays there, which
    is what proportional control does.
    """
    N = net.lat.N
    U = np.atleast_2d(dP.T).T if dP.ndim > 1 else dP[:, None]
    E = U.shape[1]
    step, L = _propagator(net, dt, agc)

    n = int(round(t_end / dt)) + 1
    w = simpson_weights(n, dt)
    Minv = 1.0 / net.M

    Y = np.zeros((step.shape[0], E))
    Y[3 * N:4 * N] = U                             # the held disturbance

    cost_r = np.zeros((N, E))
    cost_f = np.zeros((N, E))
    coi_r = np.zeros(E)
    coi_f = np.zeros(E)
    peak_r = np.zeros((N, E))
    peak_d = np.zeros((N, E))
    nadir = np.zeros((N, E))
    coi_nadir = np.zeros(E)
    Mw = net.M / net.M.sum()
    kw = int(round(0.5 / dt))                      # the 500 ms grid-code window
    f0 = np.zeros((N, E))
    f_at_500 = np.zeros((N, E))
    fc0 = np.zeros(E)
    fc_at_500 = np.zeros(E)
    traces: dict[str, list] = {"t": [], "f": [], "rocof": [], "f_coi": []}

    # Measurement lag, held exactly: with a zero-order hold on the sample grid,
    # f_meas[k+1] = a f_meas[k] + (1 - a) f[k], and the derivative of a
    # first-order lag is (f - f_meas) / t_meas with no differencing at all.
    filtered = t_meas > 0.0
    a = np.exp(-dt / t_meas) if filtered else 0.0
    fm = np.zeros((N, E))
    fmc = np.zeros(E)

    for k in range(n):
        x, omega, p = Y[:N], Y[N:2 * N], Y[2 * N:3 * N]
        secondary = agc[:, None] * Y[-1] if agc is not None else 0.0
        alpha = Minv[:, None] * (U + p + secondary
                                 - net.D[:, None] * omega - L @ x)
        f = omega / TWO_PI                                     # Hz
        r = alpha / TWO_PI                                     # Hz/s
        fc = Mw @ f
        rc = Mw @ r
        if filtered:
            f_use, r_use = fm, (f - fm) / t_meas
            fc_use, rc_use = fmc, (fc - fmc) / t_meas
        else:
            f_use, r_use, fc_use, rc_use = f, r, fc, rc
        cost_r += w[k] * (tau * r_use) ** 2
        cost_f += w[k] * f_use ** 2
        coi_r += w[k] * (tau * rc_use) ** 2
        coi_f += w[k] * fc_use ** 2
        np.maximum(peak_r, np.abs(r_use), out=peak_r)
        np.maximum(peak_d, np.abs(f_use), out=peak_d)
        # The nadir is the deepest *downward* excursion: it is what
        # under-frequency load shedding watches, so an over-frequency event
        # contributes nothing to it rather than contributing its mirror image.
        np.maximum(nadir, -f_use, out=nadir)
        np.maximum(coi_nadir, -np.atleast_1d(fc_use), out=coi_nadir)
        if k == 0:
            f0, fc0 = f.copy(), np.atleast_1d(fc).copy()
        if k == kw:
            f_at_500, fc_at_500 = f.copy(), np.atleast_1d(fc).copy()
        if trace:
            traces["t"].append(k * dt)
            traces["f"].append(f_use.copy())
            traces["rocof"].append(r_use.copy())
            traces["f_coi"].append(np.atleast_1d(fc_use).copy())
        if k < n - 1:
            if filtered:
                fm = a * fm + (1.0 - a) * f
                fmc = a * fmc + (1.0 - a) * fc
            Y = step @ Y

    hit = np.abs(U) > 0
    local_event = hit.sum(axis=0) <= 8
    resp = Response(cost=cost_r + cost_f, cost_rocof=cost_r, cost_freq=cost_f,
                    coi_cost=coi_r + coi_f, coi_rocof=coi_r, coi_freq=coi_f,
                    peak_rocof=peak_r, peak_dev=peak_d,
                    rocof_500ms=(f_at_500 - f0) / 0.5,
                    nadir=nadir, coi_nadir=coi_nadir,
                    coi_rocof_500ms=(fc_at_500 - fc0) / 0.5,
                    self_mask=hit & local_event[None, :])
    if trace:
        out = {"t": np.array(traces["t"]),
               "f": np.stack(traces["f"], axis=-1),
               "rocof": np.stack(traces["rocof"], axis=-1),
               "f_coi": np.stack(traces["f_coi"], axis=-1)}
        return resp, out
    return resp


def post_fault(net: sw.Network, dP: np.ndarray) -> tuple[bool, float]:
    """Does a post-disturbance equilibrium exist, and how hard is it pushed?

    The linearised solver above cannot answer this: linear equations always have
    a solution, so a disturbance can be made arbitrarily large and the response
    simply scales.  The real network has a limit -- the power a line can carry is
    K sin(angle difference), which cannot exceed K -- and past it there is no
    synchronous steady state to settle into at all.

    Solving for it is cheap and exact.  At the post-event equilibrium every bus
    turns at the same offset frequency,

        omega_inf = sum(dP) / (sum(D) + sum(droop) / omega_s),

    each bus injects P_i + dP_i less its damping and governor contribution, and
    the angles solve the same nonlinear power-flow equation as the pre-fault
    state.  If that solve fails, the configuration cannot reach a new operating
    point: it loses synchronism.  Returns (feasible, worst line loading), the
    loading being |sin(angle difference)|, so 1.0 is the static limit.
    """
    denom = net.D.sum() + net.droop.sum() / sw.OMEGA_S
    w_inf = float(dP.sum()) / denom
    P_eff = net.P + dP - net.D * w_inf - net.droop * w_inf / sw.OMEGA_S
    try:
        theta = sw.steady_state(net.lat, P_eff)
    except RuntimeError:
        return False, float("inf")
    d = theta[:, None] - theta[None, :]
    return True, float(np.abs(np.sin(d))[net.lat.A > 0].max())


def nonlinear_cost(net: sw.Network, dP: np.ndarray, t_end: float = T_END,
                   dt: float = DT, tau: float = TAU) -> float:
    """The same functional from the full nonlinear solver, for cross-checking.

    One event at a time and far slower, so this is a validation path, not the
    one the ensemble runs on.
    """
    sim = sw.simulate(net, dP, t_end=t_end, dt=dt)
    w = simpson_weights(sim["t"].size, dt)
    integrand = (tau * sim["rocof"]) ** 2 + sim["freq_dev"] ** 2
    return float((integrand * w).sum())


#: How many of the worst buses the "n worst nodes" scores add up.  n = 1 is the
#: single worst bus (what a limit written on the worst point would see), n = 10
#: is a tenth of the grid (what a limit written on a region would see).
WORST_N = (1, 5, 10)


def worst_n_by_event(field: np.ndarray, n: int, axis: int = 0) -> np.ndarray:
    """The sum of the n largest bus values, with the event axis kept.

    `worst_n` averages this over the disturbance set; keeping the event axis is
    what lets a figure show each scenario separately, the way `Response.by_event`
    does for J.

    `axis` is the bus axis, so this works both on a single response's
    (bus, event) field and on a whole ensemble's (config, bus, event) stack --
    which is the point of saving those stacks: every score in this family is a
    reduction over the bus axis, so a new one is a reprocess and never a
    re-solve.
    """
    if n >= field.shape[axis]:
        return field.sum(axis=axis)
    part = np.partition(field, -n, axis=axis)
    keep = range(part.shape[axis] - n, part.shape[axis])
    return np.take(part, keep, axis=axis).sum(axis=axis)


def worst_n(field: np.ndarray, n: int) -> float:
    """Mean over events of the sum of the n largest bus values.

    The family of scores this supports is deliberately *not* an integral: it
    takes one number per bus per event -- a 500 ms RoCoF, a nadir -- keeps the n
    worst buses and adds them.  Where J asks "how much did the whole grid move,
    for how long", this asks "how bad was it at the places where it was worst",
    which is the shape a grid code is actually written in.  Averaged over events
    rather than summed so the units stay those of the underlying quantity.
    """
    return float(worst_n_by_event(field, n).mean())


def summarise(resp: Response, n_bus: int) -> dict[str, float]:
    """The scalar scores a configuration is placed on the plot by.

    Five families, because they are not the same question and they do not have
    to agree:

      J, J_rocof, J_freq   the Ginzburg-Landau functional and each of its two
                           terms on its own -- how much the grid moved, and for
                           how long, everywhere.
      rocof_top{n}         the 500 ms RoCoF at the n worst buses, added up and
                           averaged over events.  The grid-code measurement.
      nadir_top{n}         the same for the deepest downward excursion, which is
                           what under-frequency load shedding watches.
      dev_top{n}           the same for the largest excursion in *either*
                           direction.  Not a duplicate of the nadir: an event
                           that disconnects demand drives the frequency up, so
                           it contributes nothing at all to a nadir however
                           severe it is, and a score meant to rank
                           configurations rather than to size a load-shedding
                           scheme should see it.
      coi_*                the system-wide values of both, for reference: the
                           number a single-machine-equivalent model would give.
      peak_*, share_*      extremes and exceedances.
    """
    coi_total = float(resp.coi_cost.sum()) * n_bus
    extra = {}
    for n in WORST_N:
        extra[f"rocof_top{n}"] = worst_n(np.abs(resp.rocof_500ms), n)
        extra[f"nadir_top{n}"] = worst_n(resp.nadir, n)
        extra[f"dev_top{n}"] = worst_n(resp.peak_dev, n)
    extra["coi_rocof_500ms"] = float(np.abs(resp.coi_rocof_500ms).mean())
    extra["coi_nadir_mean"] = float(resp.coi_nadir.mean())
    # How much of the worst bus is local: the same quantity at the worst bus,
    # divided by what the system as a whole did.  1.0 would mean every bus moves
    # with the system and there is nothing local to find.
    extra["rocof_local_excess"] = (extra["rocof_top1"]
                                   / max(extra["coi_rocof_500ms"], 1e-12))
    extra["nadir_local_excess"] = (extra["nadir_top1"]
                                   / max(extra["coi_nadir_mean"], 1e-12))
    return extra | {
        "J": resp.total,
        "J_rocof": float(resp.cost_rocof.sum()),
        "J_freq": float(resp.cost_freq.sum()),
        "J_per_bus_event": resp.total / resp.cost.size,
        "J_worst_bus": float(resp.by_bus.max()),
        "J_worst_event": float(resp.by_event.max()),
        # The system-wide part: the same functional evaluated on the centre-of-
        # inertia frequency, i.e. what a configuration would score if every bus
        # moved together.  What is left over is local.
        "J_coi": coi_total,
        "J_local": resp.total - coi_total,
        # The same total with the directly-disturbed buses left out: the part
        # of the score that is about the system rather than about the inertia of
        # whichever bus the step happened to land on.
        "J_ex_self": float(resp.cost[~resp.self_mask].sum()),
        "J_self": float(resp.cost[resp.self_mask].sum()),
        "peak_rocof": float(resp.peak_rocof.max()),
        "peak_rocof_ex_self": float(
            np.where(resp.self_mask, 0.0, resp.peak_rocof).max()),
        "peak_dev_ex_self": float(
            np.where(resp.self_mask, 0.0, resp.peak_dev).max()),
        "peak_dev": float(resp.peak_dev.max()),
        "worst_rocof_500ms": float(np.abs(resp.rocof_500ms).max()),
    }

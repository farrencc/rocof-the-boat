"""The ensemble: power-balanced operating states from 30% to 100% SNSP.

A configuration is a complete, balanced operating point of the frozen grid --
which machines are synchronised, what every unit is producing, and therefore
what inertia and what primary response the system has.  The grid underneath does
not move; only this does.

WHAT VARIES, AND WHY IT IS DRAWN RATHER THAN SET

Three things differ between configurations at the same SNSP:

  which machines are committed   Not every station has to run.  A configuration
      needing 3 p.u. of synchronous output can get it from two big stations or
      three small ones, and those are different grids dynamically even though
      they have the same SNSP.  Commitment is drawn at random from the subsets
      that can cover the residual demand.
  how the wind is distributed    The same wind total can sit mostly on the
      radial west-coast farms or mostly inland.  Three patterns are drawn -
      coastal-heavy, inland-heavy and neutral - so that geography varies
      independently of the share.
  the interconnector position    Drawn in [50%, 100%] of nameplate.

That spread is deliberate.  If every configuration at a given SNSP were
identical, the plot would be a curve and would say only "SNSP predicts risk".
With the geography varying too, the *scatter at fixed SNSP* measures how much of
the risk SNSP does not explain -- which is the more interesting half of the
question, and the half the local-effects study says should be large.

INERTIA FOLLOWS COMMITMENT, NOT SNSP DIRECTLY

A synchronised machine carries H = 4 s on its own nameplate.  A station that is
not committed carries almost nothing (H = 0.05 s, a bus with a transformer on it)
and provides no governor response either.  So inertia and reserve both fall as SNSP rises, in steps, as
individual stations come off -- which is how it actually happens, and why the
relationship between SNSP and risk need not be smooth.

A committed machine runs between `P_MIN_FRACTION` of its nameplate and its
nameplate.  Where the residual is smaller than any single machine's minimum, one
machine is committed below its minimum and flagged (`below_min`): that is the
synchronous-condenser corner of the operating space, and it is real, but it
should be visible in the results rather than hidden.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import dynamics as dyn
import grid as G

HERE = Path(__file__).resolve().parent

#: Inertia constant of a synchronised machine, and of a station that is off.
#: Both are quoted on the station's own nameplate (see grid.inertia), so
#: committing a 3.4 p.u. machine adds 13.6 p.u. s of stored energy and
#: decommitting it removes all but a token amount.
H_ON, H_OFF = 4.0, 0.05
#: A committed machine runs no lower than this fraction of its nameplate.
P_MIN_FRACTION = 0.20
#: Governor: 5% droop (gain 1/R = 20) and a 5 s lag, at committed machines only.
#: The gain is quoted on the machine's own rating, so the injection it delivers
#: scales with nameplate: a 3.4 p.u. station at 5% droop is worth 68 p.u. of
#: power per p.u. of frequency, a 1.6 p.u. station only 32.
DROOP, T_GOV = 20.0, 5.0
#: Fast frequency response at the converters: off in the baseline ensemble,
#: swept separately.  A control choice, not a property of the plant.
FFR_DROOP, T_FFR = 0.0, 1.0

#: The SNSP levels the ensemble is stratified over, and how many draws each.
SNSP_LEVELS = np.round(np.arange(0.30, 1.0001, 0.05), 4)
DRAWS_PER_LEVEL = 16


@dataclass
class Config:
    """One balanced operating state of the frozen grid."""

    index: int
    snsp_target: float
    P: np.ndarray
    H: np.ndarray
    droop: np.ndarray
    T_gov: np.ndarray
    committed: tuple[int, ...]
    condensers: tuple[int, ...]
    wind_pattern: str
    below_min: bool
    meta: dict = field(default_factory=dict)

    def network(self, g: G.ToyGrid, **kw):
        return dyn.build_network(g, self.P, self.H, self.droop, self.T_gov, **kw)

    def participation(self, g: G.ToyGrid) -> np.ndarray | None:
        """Secondary-control participation: the committed machines, by rating.

        None when nothing is synchronised -- a system with no machine on has no
        secondary control either, and that is a statement about the operating
        state rather than a modelling convenience.
        """
        if not self.committed:
            return None
        a = np.zeros(g.N)
        on = np.array(self.committed)
        a[on] = g.capacity[on] / g.capacity[on].sum()
        return a


def _allocate(total: float, cap: np.ndarray, rng: np.random.Generator,
              shape: np.ndarray | None = None, lo: float = 0.25,
              hi: float = 1.0) -> np.ndarray:
    """Split `total` over units with headroom `cap`, unevenly but feasibly.

    `shape` sets the relative loading wanted (drawn from `cap * U(lo, hi)` if not
    given), then it is water-filled: any unit that would exceed its headroom is
    pinned there and the remainder is redistributed over the rest.  Returns
    exactly `total` whenever total <= cap.sum().
    """
    if total <= 0:
        return np.zeros_like(cap)
    if total > cap.sum() + 1e-9:
        raise ValueError("asked for more than the units can produce")
    w = cap * rng.uniform(lo, hi, size=cap.size) if shape is None else shape
    out = np.zeros_like(cap)
    free = np.ones(cap.size, bool)
    left = total
    for _ in range(cap.size + 1):
        if left <= 1e-12 or not free.any():
            break
        share = w[free] / w[free].sum() * left
        take = np.minimum(share, cap[free] - out[free])
        out[free] += take
        left -= take.sum()
        free &= out < cap - 1e-12
    return out


def _wind_weights(g: G.ToyGrid, pattern: str, rng: np.random.Generator
                  ) -> tuple[float, float]:
    """Draw ranges for coastal and inland farms under one geography pattern."""
    if pattern == "coastal":
        return (0.60, 1.00), (0.10, 0.45)
    if pattern == "inland":
        return (0.15, 0.55), (0.70, 1.00)
    return (0.25, 1.00), (0.25, 1.00)


def draw(g: G.ToyGrid, snsp: float, rng: np.random.Generator, index: int,
         ffr: float = FFR_DROOP, muon: int = 0) -> Config:
    """One configuration at (approximately) the requested SNSP.

    `muon` is a minimum-units floor: at least this many stations are
    synchronised whatever the energy balance needs (see below).
    """
    demand = float(g.demand.sum())
    wind = g.by_role("wind")
    hv = g.by_role("hvdc")
    syn = g.by_role("sync")

    # -- non-synchronous side -------------------------------------------------
    want_ns = snsp * demand
    hvdc = float(min(g.capacity[hv].sum() * rng.uniform(0.5, 1.0), want_ns))
    want_wind = want_ns - hvdc
    if want_wind > g.capacity[wind].sum():
        hvdc += want_wind - g.capacity[wind].sum()
        want_wind = g.capacity[wind].sum()

    pattern = str(rng.choice(["coastal", "inland", "neutral"]))
    coast_rng, inland_rng = _wind_weights(g, pattern, rng)
    is_coast = np.array([g.coords[i][1] == 0 for i in wind])
    w = np.empty(len(wind))
    w[is_coast] = rng.uniform(*coast_rng, size=is_coast.sum())
    w[~is_coast] = rng.uniform(*inland_rng, size=(~is_coast).sum())
    cap_w = g.capacity[wind]
    P_wind = _allocate(want_wind, cap_w, rng, shape=cap_w * w)

    # -- synchronous side: commitment ----------------------------------------
    residual = demand - hvdc - P_wind.sum()
    cap_s = g.capacity[syn]
    pmin = P_MIN_FRACTION * cap_s
    below_min = False
    order = rng.permutation(len(syn))
    committed: list[int] = []
    if residual > 1e-9:
        for k in order:                     # commit until the residual fits
            committed.append(int(k))
            if cap_s[committed].sum() >= residual - 1e-9 and \
                    pmin[committed].sum() <= residual + 1e-9:
                break
        while committed and pmin[committed].sum() > residual + 1e-9:
            committed.pop()                 # too much minimum generation
        if not committed:
            # Residual smaller than any machine's minimum: one machine runs
            # below it, which is the synchronous-condenser corner.
            committed = [int(np.argmin(pmin))]
            below_min = True
    committed = sorted(committed)

    P_sync = np.zeros(len(syn))
    if committed:
        c = np.array(committed)
        base = np.minimum(pmin[c], residual)
        extra = _allocate(max(residual - base.sum(), 0.0), cap_s[c] - base, rng)
        P_sync[c] = base + extra

    # -- the MUON floor -------------------------------------------------------
    # A minimum-units constraint of the kind EirGrid operates alongside its SNSP
    # limit: at least `muon` stations synchronised at all times, whatever the
    # dispatch wants.  Stations the energy balance does not need are synchronised
    # anyway, as **condensers** -- spinning, carrying their full inertia, but
    # producing nothing and therefore offering no governor response.  That is
    # what makes this the clean experiment: it pins system inertia while leaving
    # SNSP free to reach 100%, and it separates inertia from reserve, which are
    # otherwise supplied by the same machines and impossible to tell apart.
    condensers: list[int] = []
    if muon:
        spare = [int(k) for k in order if int(k) not in committed]
        while len(committed) + len(condensers) < min(muon, len(syn)) and spare:
            condensers.append(spare.pop(0))
    condensers = sorted(condensers)

    # -- assemble -------------------------------------------------------------
    P = -g.demand.astype(float).copy()
    P[wind] += P_wind
    P[hv] += hvdc / max(len(hv), 1)
    P[syn] += P_sync
    if abs(P.sum()) > 1e-7:
        raise RuntimeError(f"configuration does not balance: {P.sum():.2e}")

    H = g.H.astype(float).copy()
    H[syn] = H_OFF
    spinning = committed + condensers
    if spinning:
        H[syn[np.array(spinning)]] = H_ON

    droop = np.zeros(g.N)
    T = np.full(g.N, T_GOV)
    if committed:
        on_bus = syn[np.array(committed)]
        droop[on_bus] = DROOP * g.capacity[on_bus]
    if ffr:
        ns = g.non_synchronous
        droop[ns] = ffr * g.capacity[ns]
        T[ns] = T_FFR

    gen = np.clip(P, 0.0, None)
    realised = float(gen[g.non_synchronous].sum() / gen.sum())
    # "Distance to a machine" means distance to a *spinning* machine: a
    # condenser anchors the local frequency just as well as a generating unit,
    # which is the whole reason for running one.
    on = syn[np.array(spinning)] if spinning else np.array([], int)
    meta = {
        "snsp": realised,
        "wind_pu": float(P_wind.sum()),
        "hvdc_pu": float(hvdc),
        "sync_pu": float(P_sync.sum()),
        "n_committed": len(committed),
        "n_condensers": len(condensers),
        "n_spinning": len(spinning),
        "stored_energy": g.stored_energy(H),
        "reserve": float(droop.sum()),
        "below_min": below_min,
        "wind_pattern": pattern,
        # Geography of what is left: how far a bus is from the nearest machine
        # that is actually spinning.  The local-effects study says this, not the
        # share, is what a bus experiences.
        "machine_dist_mean": float(g.dist[:, on].min(axis=1).mean())
        if len(on) else float("nan"),
        "machine_dist_max": float(g.dist[:, on].min(axis=1).max())
        if len(on) else float("nan"),
        "coastal_wind_share": float(
            P_wind[is_coast].sum() / max(P_wind.sum(), 1e-12)),
        "radial_wind_pu": float(sum(
            P[g.index(r, 0)] for r in G.STUB_ROWS)),
    }
    return Config(index=index, snsp_target=snsp, P=P, H=H, droop=droop, T_gov=T,
                  committed=tuple(int(syn[k]) for k in committed),
                  condensers=tuple(int(syn[k]) for k in condensers),
                  wind_pattern=pattern, below_min=below_min, meta=meta)


def ensemble(g: G.ToyGrid, levels: np.ndarray = SNSP_LEVELS,
             draws: int = DRAWS_PER_LEVEL, seed: int = 20260909,
             ffr: float = FFR_DROOP, muon: int = 0) -> list[Config]:
    """The stratified ensemble: `draws` configurations at each SNSP level.

    Stratified rather than uniformly random in SNSP so that every part of the
    range is equally represented in the fit, and seeded so the ensemble is
    reproducible bit for bit.
    """
    rng = np.random.default_rng(seed)
    out = []
    for s in levels:
        for _ in range(draws):
            out.append(draw(g, float(s), rng, index=len(out), ffr=ffr,
                            muon=muon))
    return out


def frame(configs: list[Config]) -> pd.DataFrame:
    rows = []
    for c in configs:
        r = {"config": c.index, "snsp_target": c.snsp_target}
        r.update(c.meta)
        r["committed"] = "|".join(str(b) for b in c.committed)
        r["condensers"] = "|".join(str(b) for b in c.condensers)
        rows.append(r)
    return pd.DataFrame(rows)


def save(configs: list[Config], out: Path = HERE) -> None:
    """Freeze the ensemble: the summary table and every dispatch vector."""
    frame(configs).to_csv(out / "configs.csv", index=False, lineterminator="\n")
    np.savez_compressed(
        out / "configs.npz",
        P=np.stack([c.P for c in configs]),
        H=np.stack([c.H for c in configs]),
        droop=np.stack([c.droop for c in configs]),
        T_gov=np.stack([c.T_gov for c in configs]))


def load(out: Path = HERE) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    return pd.read_csv(out / "configs.csv"), dict(np.load(out / "configs.npz"))


def report(g: G.ToyGrid, configs: list[Config]) -> str:
    df = frame(configs)
    out = ["ENSEMBLE OF OPERATING CONFIGURATIONS", "=" * 70, "",
           f"  {len(configs)} configurations, {len(SNSP_LEVELS)} SNSP levels x "
           f"{DRAWS_PER_LEVEL} draws", "",
           f"  {'SNSP':>6s} {'n':>4s} {'machines on':>12s} {'stored (pu s)':>14s} "
           f"{'reserve':>9s} {'dist to machine':>16s}", "-" * 70]
    for s, part in df.groupby("snsp_target"):
        out.append(f"  {s * 100:5.0f}% {len(part):4d} "
                   f"{part['n_committed'].mean():12.2f} "
                   f"{part['stored_energy'].mean():14.1f} "
                   f"{part['reserve'].mean():9.1f} "
                   f"{part['machine_dist_mean'].mean():16.2f}")
    out += ["",
            f"  realised SNSP spans {df['snsp'].min() * 100:.1f}% to "
            f"{df['snsp'].max() * 100:.1f}%",
            f"  stored energy falls from {df['stored_energy'].max():.0f} to "
            f"{df['stored_energy'].min():.0f} p.u. s",
            f"  configurations with a machine below minimum generation: "
            f"{int(df['below_min'].sum())}",
            f"  wind patterns: " + ", ".join(
                f"{k} {v}" for k, v in df['wind_pattern'].value_counts().items())]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", action="store_true",
                    help="write configs.csv and configs.npz")
    args = ap.parse_args()
    g = G.load()
    cfgs = ensemble(g)
    text = report(g, cfgs)
    print(text)
    if args.freeze:
        save(cfgs)
        (HERE / "configs_report.txt").write_text(text)
        print("wrote configs.csv, configs.npz, configs_report.txt")


if __name__ == "__main__":
    main()

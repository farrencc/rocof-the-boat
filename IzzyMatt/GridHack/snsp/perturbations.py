"""The disturbance set -- defined once, then applied to every configuration.

The experiment compares configurations, so the excitation must not vary with the
configuration.  If it did, a configuration could look bad because it was hit
harder rather than because it responded worse, and the two would be impossible
to separate afterwards.

So every disturbance here is a **fixed injection step in p.u.**, identical in
location and magnitude across the whole ensemble, applied at t = 0 and held.
That has one consequence worth being explicit about: a "generator trip" in this
set is a fixed-size infeed loss *at that bus*, not the removal of whatever that
unit happened to be producing in this configuration.  The alternative -- trip
each unit's actual output -- makes the disturbance a function of the dispatch,
which is exactly the confound the fixed set exists to avoid.  It is offered
separately, as `n1_events()`, for the N-1 question, and is never mixed into the
ensemble cost.

The 20 events span the imbalance types the system actually sees:

  load_step      a block of demand switches in at a city.
  wind_drop      a farm's output falls -- a gust front passing, or curtailment.
  infeed_loss    a large unit or the interconnector is lost.
  wind_lull      the whole western seaboard drops together, which is the
                 correlated event a single-bus disturbance cannot represent.
  load_reject    demand disconnects: the same physics with the sign reversed.
  diffuse_load   a small increase spread over every bus, i.e. an imbalance with
                 no location at all -- the control case for everything else.

Sizes are deliberately of one order (0.4-0.65 p.u. of a 12 p.u. system, i.e.
3-5% of demand) so that no single event dominates the summed cost by being
larger than the rest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import grid as G

HERE = Path(__file__).resolve().parent
EVENTS_CSV, EVENTS_JSON = "perturbations.csv", "perturbations.json"
SCALED_CSV, SCALED_JSON = "perturbations_scaled.csv", "perturbations_scaled.json"
SOURCED_CSV = "perturbations_sourced.csv"
SOURCED_JSON = "perturbations_sourced.json"

#: The three frozen sets, and what each is for.
SETS = {
    "sourced": (SOURCED_CSV, SOURCED_JSON),   # the headline set: each event
                                              # scales with its own source
    "fixed": (EVENTS_CSV, EVENTS_JSON),       # hazard held constant: the
                                              # vulnerability-only control
    "scaled": (SCALED_CSV, SCALED_JSON),      # wind-only, all scaling: the
                                              # hazard-only contrast
}


def _files(which: str) -> tuple[str, str]:
    """The frozen files for one of the three sets."""
    try:
        return SETS[which]
    except KeyError:
        raise ValueError(f"unknown event set: {which!r}") from None

#: Cities take a block of demand: 0.4 p.u., about a quarter of Cork.
LOAD_STEP = 0.40
#: A farm loses output.  Fixed size, so it is the same event in every
#: configuration; 0.35 p.u. is roughly a quarter of a big coastal farm.
WIND_DROP = 0.35
#: A large unit or the HVDC link is lost outright.
INFEED_LOSS = 0.60
#: The seaboard lull: this much at each of the eight coastal farms at once.
LULL_EACH = 0.08
#: Demand disconnects at Dublin.
LOAD_REJECT = 0.50
#: A diffuse imbalance with no location: this much extra load at every bus.
DIFFUSE_TOTAL = 0.50


@dataclass
class Event:
    """One disturbance, in one of two flavours.

    ``step``    a fixed injection step in p.u., identical in every
                configuration.  Holds the *hazard* constant so that only the
                response differs -- the control the ensemble comparison needs.
    ``scaled``  a fixed **fraction of what each bus is currently generating**.
                The rule is identical in every configuration; the megawatts are
                not, because they cannot be: a wind front that takes 20% of the
                west coast's output removes four times as much power when four
                times as much wind is running.

    The two answer different questions and the study needs both.  A fixed-size
    set measures *vulnerability* at constant hazard.  A scaled set measures
    *risk*, hazard and vulnerability together -- and it is the only one of the
    two in which the non-synchronous share can act on the answer through
    anything other than the inertia it displaces.

    A third form, ``rule``, is a scaled event whose *location* is chosen per
    configuration -- "the largest running wind farm trips", which is one event
    even though the bus it lands on changes with the dispatch.  The rule is what
    is frozen, not the bus.
    """

    name: str
    kind: str
    dP: np.ndarray | None = None      # fixed step, p.u. (a `step` event)
    frac: np.ndarray | None = None    # fraction of own output (a `scaled` one)
    rule: str | None = None           # "largest": trip the biggest of `pool`
    pool: tuple[int, ...] = ()        # candidate buses for a rule event
    cap: float = 0.0                  # ceiling on a scaled event, p.u. (0 = none)

    @property
    def scaled(self) -> bool:
        return self.frac is not None or self.rule is not None

    def vector(self, P: np.ndarray | None = None) -> np.ndarray:
        """The injection step this event applies to a configuration."""
        if not self.scaled:
            return self.dP
        if P is None:
            raise ValueError(f"{self.name} is a scaled event and needs the "
                             f"configuration's dispatch")
        out = np.zeros_like(P)
        if self.rule == "largest":
            pool = np.array(self.pool)
            out[pool[np.argmax(P[pool])]] = -max(float(P[pool].max()), 0.0)
            return out
        out = -self.frac * np.clip(P, 0.0, None)
        if self.cap:
            # A station is several units, and the credible contingency is the
            # loss of one of them, not of the whole site.  So the event still
            # scales with what the station is producing -- a machine at minimum
            # load loses less, a decommitted one loses nothing -- but it cannot
            # exceed the size of a single unit.
            out = np.maximum(out, -self.cap)
        return out

    def size(self, P: np.ndarray | None = None) -> float:
        """Net imbalance, p.u.  Negative for a deficit."""
        return float(self.vector(P).sum())

    @property
    def buses(self) -> list[int]:
        if self.rule is not None:
            return list(self.pool)
        v = self.dP if self.frac is None else self.frac
        return np.flatnonzero(v != 0.0).tolist()


def build(g: G.ToyGrid) -> list[Event]:
    """The fixed set of 20 events."""
    ev: list[Event] = []

    def step(name: str, kind: str, pairs: list[tuple[int, float]]) -> None:
        dP = np.zeros(g.N)
        for b, v in pairs:
            dP[b] += v
        ev.append(Event(name, kind, dP))

    # 5: a demand block switches in at each city.
    for i in g.by_role("city"):
        step(f"load_step@{g.name[i]}", "load_step", [(int(i), -LOAD_STEP)])

    # 8: a farm's output falls.  All four radial farms, plus two other coastal
    # and two inland, so the set covers strong and weak points of connection.
    farms = [g.index(r, 0) for r in G.STUB_ROWS]
    farms += [g.index(1, 0), g.index(8, 0), g.index(2, 2), g.index(8, 3)]
    for i in farms:
        step(f"wind_drop@{g.name[i]}", "wind_drop", [(int(i), -WIND_DROP)])

    # 4: a large infeed is lost -- three stations spread over the country, and
    # the interconnector.
    for r, c in ((5, 8), (7, 3), (1, 6)):
        i = g.index(r, c)
        step(f"infeed_loss@{g.name[i]}", "infeed_loss", [(int(i), -INFEED_LOSS)])
    i = int(g.by_role("hvdc")[0])
    step(f"infeed_loss@{g.name[i]}", "infeed_loss", [(i, -INFEED_LOSS)])

    # 1: the correlated event -- the whole seaboard drops at once.
    coast = [int(g.index(r, 0)) for r, _, _ in G.WIND_COAST]
    step("wind_lull@seaboard", "wind_lull", [(b, -LULL_EACH) for b in coast])

    # 1: demand disconnects at Dublin.  Same physics, opposite sign; in the
    # linearised model the response is exactly antisymmetric, which is a useful
    # thing to have in the set as a check rather than an assumption.
    i = int(g.by_role("city")[np.argmax(g.demand[g.by_role("city")])])
    step(f"load_reject@{g.name[i]}", "load_reject", [(i, +LOAD_REJECT)])

    # 1: an imbalance with no location at all.
    step("diffuse_load@all", "diffuse_load",
         [(b, -DIFFUSE_TOTAL / g.N) for b in range(g.N)])

    return ev


# ---------------------------------------------------------------------------
# The sourced set: the same 20 events, each scaling with its own source
# ---------------------------------------------------------------------------

#: A gust front takes this much of the farm it passes over, and a broad seaboard
#: lull this much of every coastal farm.  Chosen so that at the reference
#: dispatch the events are the same size as the fixed set's 0.35 and 0.64 p.u. --
#: the set is recalibrated, not rescaled.
DROP_FRACTION, LULL_FRACTION = 0.40, 0.10
#: Largest single unit, p.u.  A trip event scales with the station's output but
#: is capped here: the stations on this toy grid carry 13-28% of system demand
#: apiece, so "the whole site trips" would be a far larger contingency than any
#: real system dimensions against (Ireland's largest single infeed is about a
#: tenth of demand).  A station is several units; losing one is the credible
#: event.  `--sourced-uncapped` runs without it, and the difference is reported.
LARGEST_UNIT = 1.0


def build_sourced(g: G.ToyGrid, cap_units: float = LARGEST_UNIT
                  ) -> list[Event]:
    """The 20-event set again, with every event scaled to its own source.

    This is the physically consistent version of `build()`, and it is the one
    the headline results use.  The rule is simple: an event scales with whatever
    causes it.

      wind_drop, wind_lull   weather, so a fraction of what the farms are
                             *generating*.  The same front removes four times
                             the power when four times the wind is running.
      infeed_loss            a unit trip, so **all** of what that unit is
                             producing -- and nothing at all if it is not
                             running.  This is the half that moves the other way:
                             at low SNSP the machines are loaded and their trip
                             is the largest single infeed on the system, while at
                             high SNSP they are at minimum load or off, and the
                             event shrinks towards zero.
      load_step, load_reject, diffuse_load
                             demand, which is held fixed across the whole
                             ensemble by construction, so these do not scale.
                             They are the events that stay put, and they matter:
                             without them the set would only contain hazards
                             that grow with the wind.

    The net effect of the two directions is not obvious in advance, which is why
    it is worth measuring rather than assuming.
    """
    ev: list[Event] = []

    def step(name: str, kind: str, pairs: list[tuple[int, float]]) -> None:
        dP = np.zeros(g.N)
        for b, v in pairs:
            dP[b] += v
        ev.append(Event(name, kind, dP=dP))

    def sourced(name: str, kind: str, frac: dict[int, float],
                cap: float = 0.0) -> None:
        f = np.zeros(g.N)
        for b, v in frac.items():
            f[b] = v
        ev.append(Event(name, kind, frac=f, cap=cap))

    # 5 + 1 + 1: demand-side, unchanged and unscaled.
    for i in g.by_role("city"):
        step(f"load_step@{g.name[i]}", "load_step", [(int(i), -LOAD_STEP)])
    i = int(g.by_role("city")[np.argmax(g.demand[g.by_role("city")])])
    step(f"load_reject@{g.name[i]}", "load_reject", [(i, +LOAD_REJECT)])
    step("diffuse_load@all", "diffuse_load",
         [(b, -DIFFUSE_TOTAL / g.N) for b in range(g.N)])

    # 8: a front over one farm, as a fraction of that farm's output.
    farms = [g.index(r, 0) for r in G.STUB_ROWS]
    farms += [g.index(1, 0), g.index(8, 0), g.index(2, 2), g.index(8, 3)]
    for i in farms:
        sourced(f"wind_drop@{g.name[i]}", "wind_drop",
                {int(i): DROP_FRACTION})

    # 1: the correlated event, over the whole seaboard.
    coast = [int(g.index(r, 0)) for r, _, _ in G.WIND_COAST]
    sourced("wind_lull@seaboard", "wind_lull",
            {b: LULL_FRACTION for b in coast})

    # 4: unit trips -- all of what that unit is producing, up to one unit's
    # worth.  Nothing at all if the station is not running.
    for r, c in ((5, 8), (7, 3), (1, 6)):
        i = int(g.index(r, c))
        sourced(f"infeed_loss@{g.name[i]}", "infeed_loss", {i: 1.0},
                cap=cap_units)
    i = int(g.by_role("hvdc")[0])
    sourced(f"infeed_loss@{g.name[i]}", "infeed_loss", {i: 1.0}, cap=cap_units)
    return ev


# ---------------------------------------------------------------------------
# The hazard-scaled set: weather, whose size follows the wind
# ---------------------------------------------------------------------------

#: Depth of a correlated wind front at its centre, as a fraction of the output
#: of the farms it passes over.
FRONT_DEPTH = 0.35
#: How far that front reaches, in hops through the network.  Wind is correlated
#: over hundreds of kilometres; on this grid a couple of hops is the analogue.
FRONT_REACH = 2.5
#: A broad front over the whole seaboard, and a system-wide forecast error.
SEABOARD_DEPTH, ALLWIND_DEPTH = 0.20, 0.15
#: An over-frequency event: wind coming up rather than going down.
SPIKE_DEPTH = 0.20


def build_scaled(g: G.ToyGrid) -> list[Event]:
    """Wind-driven events, defined as fractions of what is actually generating.

    Eight events.  Three moving fronts centred on the north, middle and south of
    the seaboard, falling off with distance through the network; a broad front
    over the whole coast; a system-wide forecast error over every farm; the same
    broad front with the sign reversed; the largest running farm tripping; and
    the interconnector tripping.

    None of these is a fixed number of megawatts, and that is the point.  The
    same weather removes a different amount of power depending on how much wind
    is running -- which is precisely the channel through which the
    non-synchronous share can matter for reasons that have nothing to do with
    how much inertia it displaced.
    """
    wind = g.by_role("wind")
    coast = [g.index(r, 0) for r, _, _ in G.WIND_COAST]
    ev: list[Event] = []

    def scaled(name: str, kind: str, frac: dict[int, float]) -> None:
        f = np.zeros(g.N)
        for b, v in frac.items():
            f[b] = v
        ev.append(Event(name, kind, frac=f))

    # Three fronts, each centred on a coastal farm and decaying through the
    # network, so that they are correlated over a region rather than at a point.
    for label, r0 in (("north", 1), ("mid", 4), ("south", 7)):
        centre = g.index(r0, 0)
        scaled(f"front@{label}", "wind_front",
               {int(b): FRONT_DEPTH * float(np.exp(-g.dist[centre, b]
                                                   / FRONT_REACH))
                for b in wind})

    scaled("front@seaboard", "wind_front",
           {int(b): SEABOARD_DEPTH for b in coast})
    scaled("forecast@allwind", "forecast_error",
           {int(b): ALLWIND_DEPTH for b in wind})
    scaled("spike@seaboard", "wind_spike",
           {int(b): -SPIKE_DEPTH for b in coast})
    scaled("trip@hvdc", "unit_trip",
           {int(b): 1.0 for b in g.by_role("hvdc")})
    ev.append(Event("trip@largest_farm", "unit_trip", rule="largest",
                    pool=tuple(int(b) for b in wind)))
    return ev


def n1_events(g: G.ToyGrid, P: np.ndarray, min_size: float = 0.05) -> list[Event]:
    """The other kind of disturbance: trip each unit's *actual* output.

    Configuration-dependent by construction -- the size of the event changes with
    the dispatch -- so this is a separate question ("is the largest single infeed
    survivable in this state?") and is never summed into the ensemble cost.
    """
    ev = []
    for i in g.generators:
        if P[i] > min_size:
            dP = np.zeros(g.N)
            dP[i] = -P[i]
            ev.append(Event(f"trip@{g.name[i]}", "unit_trip", dP))
    return ev


def matrix(events: list[Event], P: np.ndarray | None = None,
           g: G.ToyGrid | None = None) -> np.ndarray:
    """The whole set as one (N, n_events) array, ready to drive the solver.

    `P` is needed only if the set contains scaled events, whose megawatts depend
    on what the configuration is generating.
    """
    if any(e.scaled for e in events) and P is None:
        raise ValueError("a scaled event set needs the configuration's dispatch")
    return np.column_stack([e.vector(P) for e in events])


def frame(g: G.ToyGrid, events: list[Event]) -> pd.DataFrame:
    rows = []
    for k, e in enumerate(events):
        for b in e.buses:
            rows.append({"event": k, "name": e.name, "kind": e.kind,
                         "bus": b, "bus_name": str(g.name[b]),
                         "scaled": e.scaled, "rule": e.rule or "",
                         "cap_pu": e.cap,
                         "dP_pu": 0.0 if e.scaled else float(e.dP[b]),
                         "frac": float(e.frac[b]) if e.frac is not None else 0.0,
                         "net_pu": 0.0 if e.scaled else e.size()})
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(g: G.ToyGrid, events: list[Event], out: Path = HERE,
         which: str = "fixed") -> dict:
    csv, js = _files(which)
    frame(g, events).to_csv(out / csv, index=False, lineterminator="\n")
    spec = {
        "description": f"{which} disturbance set for the SNSP ensemble",
        "n_events": len(events),
        "sizes_pu": {"load_step": LOAD_STEP, "wind_drop": WIND_DROP,
                     "infeed_loss": INFEED_LOSS, "lull_each": LULL_EACH,
                     "load_reject": LOAD_REJECT, "diffuse_total": DIFFUSE_TOTAL,
                     "front_depth": FRONT_DEPTH, "front_reach": FRONT_REACH,
                     "seaboard_depth": SEABOARD_DEPTH,
                     "allwind_depth": ALLWIND_DEPTH, "spike_depth": SPIKE_DEPTH,
                     "drop_fraction": DROP_FRACTION,
                     "lull_fraction": LULL_FRACTION,
                     "largest_unit": LARGEST_UNIT},
        "events": [{"name": e.name, "kind": e.kind, "scaled": e.scaled,
                    "net_pu": 0.0 if e.scaled else round(e.size(), 6),
                    "buses": e.buses} for e in events],
        "sha256": {csv: _sha256(out / csv)},
    }
    (out / js).write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def load(g: G.ToyGrid, out: Path = HERE, verify: bool = True,
         which: str = "fixed") -> list[Event]:
    """Read a frozen event set back, hash-checked, as the solver's input."""
    csv, js = _files(which)
    df = pd.read_csv(out / csv)
    if verify:
        spec = json.loads((out / js).read_text())
        for fname, want in spec["sha256"].items():
            if _sha256(out / fname) != want:
                raise RuntimeError(f"{fname} has been edited since it was "
                                   f"frozen; re-run perturbations.py --freeze "
                                   f"only if that was intended")
    events = []
    for _, part in df.groupby("event", sort=True):
        name = str(part["name"].iloc[0])
        kind = str(part["kind"].iloc[0])
        rule = str(part["rule"].iloc[0]) if "rule" in part else ""
        if rule and rule != "nan":
            events.append(Event(name, kind, rule=rule,
                                pool=tuple(int(b) for b in part["bus"])))
        elif bool(part["scaled"].iloc[0]):
            f = np.zeros(g.N)
            for _, r in part.iterrows():
                f[int(r["bus"])] = float(r["frac"])
            cap = float(part["cap_pu"].iloc[0]) if "cap_pu" in part else 0.0
            events.append(Event(name, kind, frac=f, cap=cap))
        else:
            dP = np.zeros(g.N)
            for _, r in part.iterrows():
                dP[int(r["bus"])] = float(r["dP_pu"])
            events.append(Event(name, kind, dP=dP))
    return events


def report(g: G.ToyGrid, events: list[Event],
           P: np.ndarray | None = None) -> str:
    scaled = any(e.scaled for e in events)
    title = ("HAZARD-SCALED DISTURBANCE SET" if scaled
             else "FIXED DISTURBANCE SET")
    out = [title, "=" * 70, "",
           f"  {len(events)} events, the same rule applied to every "
           f"configuration", ""]
    if scaled:
        out += ["  Sizes below are for the reference dispatch (SNSP 70%); the "
                "same events",
                "  are larger at higher SNSP and smaller at lower, which is "
                "the point.", ""]
        P = G.dispatch(g, G.REFERENCE_SNSP) if P is None else P
    out += [f"  {'event':28s} {'kind':14s} {'net (p.u.)':>11s}  buses", "-" * 70]
    sizes = []
    for e in events:
        v = e.vector(P)
        sizes.append(abs(v.sum()))
        out.append(f"  {e.name:28s} {e.kind:14s} {v.sum():11.3f}  "
                   f"{int((v != 0).sum())}")
    sizes = np.array(sizes)
    out += ["", f"  net imbalance: min {sizes.min():.2f}, max {sizes.max():.2f} "
                f"p.u. against {g.demand.sum():.0f} p.u. of demand",
            f"  ({sizes.min() / g.demand.sum() * 100:.1f}% to "
            f"{sizes.max() / g.demand.sum() * 100:.1f}% of system demand)"]
    if scaled:
        out += ["", "  the same set at the two ends of the SNSP sweep:", ""]
        for s in (0.30, 0.65, 1.00):
            Ps = G.dispatch(g, s)
            tot = sum(abs(e.vector(Ps).sum()) for e in events)
            out.append(f"    SNSP {s * 100:5.0f}%   total imbalance across the "
                       f"set {tot:6.2f} p.u.")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", action="store_true",
                    help="write the frozen CSV and JSON for both sets")
    args = ap.parse_args()

    g = G.load()
    for which, builder in (("sourced", build_sourced), ("fixed", build),
                           ("scaled", build_scaled)):
        events = builder(g)
        text = report(g, events)
        print(text)
        if args.freeze:
            spec = save(g, events, which=which)
            csv, js = _files(which)
            (HERE / f"perturbations_{which}_report.txt").write_text(text)
            print(f"wrote {csv}, {js}, perturbations_{which}_report.txt")
            print(f"sha256 {spec['sha256'][csv][:16]}...")
            back = load(g, which=which)
            assert len(back) == len(events)
            P = G.dispatch(g, G.REFERENCE_SNSP)
            assert all(np.allclose(a.vector(P), b.vector(P))
                       for a, b in zip(back, events))
            print("read back and matched\n")


if __name__ == "__main__":
    main()

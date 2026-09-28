"""Sample disturbances from a hazard prior, so a predictor can be tested on a
genuinely fresh draw.

The disjoint-half test in `risk_predictor.cross_event_scores` splits the frozen
twenty-event set in two.  That is a weaker test than it looks, and in one
respect a misleading one: the twenty events are a *designed* set with a fixed
composition (five demand steps, eight wind drops, four unit trips, ...), so a
random half of them is not a representative sample of anything.  The two halves
carry different mixtures, and an aggregate computed over one half is measuring a
different quantity than the same aggregate over the other.

This module does the thing that test was standing in for.  It defines a hazard
*prior* -- a distribution over disturbances of the same families, with the
location and the severity drawn rather than chosen -- and samples independent
representative sets from it.  Two such sets contain entirely different
disturbances at different buses with different severities, but they estimate the
same population quantities, which is exactly the situation an operator is in:
the contingency list is a sample of what could happen, not the list of what
will.

The prior deliberately mirrors the frozen set's composition and scaling rules
(`perturbations.build_sourced`), because the question is whether a predictor
generalises across *draws*, not whether it survives a change of hazard model --
`risk_predictor.cross_hazard_model` asks that separately.

Run `python hazard_draw.py --freeze` to sample, solve and write
`hazard_draw.npz`.  It is the expensive step: 240 configurations against
`--events` disturbances, about a minute per twenty events.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import configs as C
import dynamics as dyn
import grid as G
import perturbations as PB

HERE = Path(__file__).resolve().parent

#: The mixture of disturbance families, as proportions.  These are the
#: frequencies in the frozen twenty-event set, so a sample from this prior is
#: the same kind of contingency list, drawn rather than designed.
MIX = {"load_step": 5 / 20, "load_reject": 1 / 20, "diffuse_load": 1 / 20,
       "wind_drop": 8 / 20, "wind_lull": 1 / 20, "infeed_loss": 4 / 20}

#: Severity ranges.  Centred on the frozen set's fixed values, spread wide
#: enough that two draws really do differ: a sampled demand block is anywhere
#: from half to one and a half times the designed 0.4 p.u., and a sampled front
#: takes between a third and nearly all of a farm's output.
LOAD_STEP_PU = (0.20, 0.60)
LOAD_REJECT_PU = (0.30, 0.70)
DIFFUSE_PU = (0.30, 0.70)
DROP_FRAC = (0.30, 0.90)
LULL_FRAC = (0.05, 0.20)

#: Unit trips are capped at one machine's worth, as in the frozen set.
CAP = PB.LARGEST_UNIT


def sample(g: G.ToyGrid, n: int, seed: int) -> list[PB.Event]:
    """Draw `n` disturbances from the prior, stratified by family.

    Stratified rather than multinomial so that a set of sixty always carries
    the intended mixture: the point of the exercise is two *representative*
    samples, and letting the mixture itself wander would put the variation in
    the wrong place.
    """
    rng = np.random.default_rng(seed)
    counts = _allocate(n, MIX, rng)
    cities = g.by_role("city")
    farms = g.by_role("wind")
    coast = [int(g.index(r, 0)) for r, _, _ in G.WIND_COAST]
    infeed = np.concatenate([g.by_role("sync"), g.by_role("hvdc")])

    ev: list[PB.Event] = []

    def step(name, kind, pairs):
        dP = np.zeros(g.N)
        for b, v in pairs:
            dP[b] += v
        ev.append(PB.Event(name, kind, dP=dP))

    def sourced(name, kind, frac, cap=0.0):
        f = np.zeros(g.N)
        for b, v in frac.items():
            f[b] = v
        ev.append(PB.Event(name, kind, frac=f, cap=cap))

    for _ in range(counts["load_step"]):
        b = int(rng.choice(cities))
        step(f"load_step@{g.name[b]}#{len(ev)}", "load_step",
             [(b, -rng.uniform(*LOAD_STEP_PU))])
    for _ in range(counts["load_reject"]):
        b = int(rng.choice(cities))
        step(f"load_reject@{g.name[b]}#{len(ev)}", "load_reject",
             [(b, +rng.uniform(*LOAD_REJECT_PU))])
    for _ in range(counts["diffuse_load"]):
        tot = rng.uniform(*DIFFUSE_PU)
        step(f"diffuse_load@all#{len(ev)}", "diffuse_load",
             [(b, -tot / g.N) for b in range(g.N)])
    for _ in range(counts["wind_drop"]):
        b = int(rng.choice(farms))
        sourced(f"wind_drop@{g.name[b]}#{len(ev)}", "wind_drop",
                {b: float(rng.uniform(*DROP_FRAC))})
    for _ in range(counts["wind_lull"]):
        f = float(rng.uniform(*LULL_FRAC))
        sourced(f"wind_lull@seaboard#{len(ev)}", "wind_lull",
                {b: f for b in coast})
    for _ in range(counts["infeed_loss"]):
        b = int(rng.choice(infeed))
        sourced(f"infeed_loss@{g.name[b]}#{len(ev)}", "infeed_loss",
                {b: 1.0}, cap=CAP)
    return ev


def _allocate(n: int, mix: dict[str, float],
              rng: np.random.Generator) -> dict[str, int]:
    """Split `n` over the families, largest-remainder, ties broken at random."""
    exact = {k: n * v for k, v in mix.items()}
    out = {k: int(np.floor(v)) for k, v in exact.items()}
    short = n - sum(out.values())
    if short:
        rem = sorted(exact, key=lambda k: (-(exact[k] - out[k]),
                                           rng.random()))
        for k in rem[:short]:
            out[k] += 1
    return out


def solve(g: G.ToyGrid, cfgs: list[C.Config], events: list[PB.Event],
          n_bus: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-event scores and disturbance size, for every configuration.

    Both grid-code scores are kept -- the 500 ms RoCoF and the peak absolute
    deviation, each over the five worst buses -- because the figure this feeds
    puts one in each column.

    Only the arrays the test needs, so this is much cheaper than a full
    `run_ensemble` pass: no post-fault feasibility solve, no summary columns.
    """
    top, dev, size = [], [], []
    t0 = time.time()
    for i, c in enumerate(cfgs):
        U = PB.matrix(events, c.P)
        resp = dyn.respond(c.network(g), U, t_end=dyn.T_END, dt=dyn.DT,
                           tau=dyn.TAU, t_meas=dyn.T_MEAS)
        top.append(dyn.worst_n_by_event(np.abs(resp.rocof_500ms), n_bus))
        dev.append(dyn.worst_n_by_event(resp.peak_dev, n_bus))
        size.append(np.abs(U.sum(axis=0)))
        if (i + 1) % 40 == 0:
            el = time.time() - t0
            done = (i + 1) / len(cfgs)
            print(f"  {i + 1:4d}/{len(cfgs)}  {el:5.0f}s elapsed, "
                  f"{el / done - el:5.0f}s left", flush=True)
    return np.array(top), np.array(dev), np.array(size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", type=int, default=60,
                    help="disturbances in EACH of the two independent sets")
    ap.add_argument("--seed-a", type=int, default=10111)
    ap.add_argument("--seed-b", type=int, default=20222)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    g = G.load()
    cfgs = C.ensemble(g, draws=C.DRAWS_PER_LEVEL, seed=20260909)
    A = sample(g, args.events, args.seed_a)
    B = sample(g, args.events, args.seed_b)
    print(f"{len(cfgs)} configurations x {len(A)} + {len(B)} sampled "
          f"disturbances")
    print(f"  set A families: "
          f"{ {k: sum(e.kind == k for e in A) for k in MIX} }")
    print(f"  set B families: "
          f"{ {k: sum(e.kind == k for e in B) for k in MIX} }")
    overlap = {e.name.split('#')[0] for e in A} & {e.name.split('#')[0]
                                                   for e in B}
    print(f"  bus-level overlap between the two sets: {len(overlap)} locations "
          f"in common (severities still differ)")

    top, dev, size = solve(g, cfgs, A + B)
    nA = len(A)
    if args.freeze:
        np.savez_compressed(
            HERE / "hazard_draw.npz",
            top5_A=top[:, :nA], top5_B=top[:, nA:],
            dev5_A=dev[:, :nA], dev5_B=dev[:, nA:],
            sizes_A=size[:, :nA], sizes_B=size[:, nA:],
            kinds_A=np.array([e.kind for e in A]),
            kinds_B=np.array([e.kind for e in B]))
        print("wrote hazard_draw.npz")


if __name__ == "__main__":
    main()

"""The headline comparison: the same converters, clustered or spread out.

The exhaustive 4x4 sweeps establish that the geography of inertia matters, but a
4x4 torus is too small to ask the question the way it was posed -- a "cluster" of
4 buses out of 16 wraps around the whole system, and a row of four *is* a closed
ring.  This runs the comparison on an 8x8 torus, where a 4x4 block of converters
is genuinely a corner of the grid and the rest of the network is genuinely far
away.

Every configuration converts exactly the same number of buses, so total system
inertia -- and therefore the system-wide RoCoF that a global argument predicts --
is identical across all of them.  The lattice is a torus, so every bus has four
neighbours and no bus is topologically special.  The only thing that changes is
where the inertia sits.

Three metrics are reported for each bus, because they do not agree, and the
disagreement is the interesting part:

  rocof_500ms   the grid-code measurement, over the 500 ms window starting at the
                event.  Slow, heavily averaged, close to the system value.
  worst_window  the worst 500 ms window anywhere in the event -- what a RoCoF
                relay would actually latch, whenever the swing arrives.
  peak_instant  the instantaneous peak.  What a fast PMU or a converter's own
                phase-locked loop sees.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import swing as sw

L = 8
K_CONV = 16          # converters, in every configuration


def block(lat: sw.Lattice, side: int = 4) -> set[int]:
    """A contiguous square of converters: all the wind in one corner."""
    return {lat.index(r, c) for r in range(side) for c in range(side)}


def band(lat: sw.Lattice, rows: int = 2) -> set[int]:
    """A stripe of converters right across the grid."""
    return {lat.index(r, c) for r in range(rows) for c in range(lat.L)}


def dispersed(lat: sw.Lattice) -> set[int]:
    """A regular sublattice: the same converters spread as evenly as possible."""
    return {lat.index(r, c) for r in range(0, lat.L, 2) for c in range(0, lat.L, 2)}


def two_blocks(lat: sw.Lattice) -> set[int]:
    """Half the converters in each of two opposite corners."""
    out = {lat.index(r, c) for r in range(2) for c in range(4)}
    out |= {lat.index(r, c) for r in range(4, 6) for c in range(4, 8)}
    return out


def quadrant_shortcuts(lat: sw.Lattice, side: int = 4) -> tuple[tuple[int, int], ...]:
    """Link every converter to its counterpart in the other three quadrants.

    An 8x8 torus divides into four 4x4 quadrants; the converter block occupies
    one of them.  Each converter at (r, c) gains a long line to (r, c+4),
    (r+4, c) and (r+4, c+4) -- the same position in each of the other three.
    That is 48 new lines: three per converter, and exactly one arriving at each
    of the 48 synchronous buses, so the reinforcement is spread evenly over the
    rest of the grid rather than landing anywhere in particular.
    """
    return tuple((lat.index(r, c), lat.index((r + dr) % lat.L, (c + dc) % lat.L))
                 for r in range(side) for c in range(side)
                 for dr, dc in ((0, side), (side, 0), (side, side)))


def block_infill(lat: sw.Lattice, side: int = 4,
                 seed: int = 0) -> tuple[tuple[int, int], ...]:
    """`quadrant_shortcuts`' line-count-matched control: 48 lines kept inside.

    The quadrant construction spends 48 lines wiring the converter block out to
    the rest of the grid; this spends 48 wiring it to itself, drawn uniformly
    from the 96 pairs inside the block that the lattice does not already
    connect.  The two differ in where the reinforcement goes and in nothing
    else -- not the number of lines, not their coupling, and not the inertia,
    which neither touches.

    Drawn rather than constructed, and deliberately so: any regular rule for
    picking 48 of the 96 also imposes a symmetry, and the question here is what
    48 lines inside the block buy in general, not what one flattering
    arrangement of them buys.  The draw is lumpy in consequence -- at seed 0 the
    converters gain between 3 and 10 lines each, against the flat 3 per
    converter the quadrant construction gives.

    `selfenergy.py` and the control table in the README are the same 48 lines.
    """
    blk = sorted(block(lat, side))
    cands = [(i, j) for n, i in enumerate(blk) for j in blk[n + 1:]
             if lat.A[i, j] == 0]
    rng = np.random.default_rng(seed)
    return tuple(cands[i] for i in rng.choice(len(cands), 48, replace=False))


def field(net: sw.Network, step: float = 0.5) -> dict[str, np.ndarray]:
    """Every metric at every bus, averaged over a load step at every bus in turn.

    Run out to 5 s so the frequency nadir is actually reached -- with a 5 s
    governor the turning point is around 2 s, and a shorter run would report
    wherever the trace happened to stop instead.
    """
    N = net.lat.N
    keys = ("rocof_500ms", "worst_window", "peak_instant", "nadir")
    acc = {k: np.zeros(N) for k in keys}
    coi = coi_n = 0.0
    for b in range(N):
        m = sw.metrics(sw.simulate_linear(net, sw.load_step(net, b, step),
                                          t_end=5.0, dt=2e-3))
        acc["rocof_500ms"] += np.abs(m["rocof_500ms"])
        acc["worst_window"] += m["worst_window"]
        acc["peak_instant"] += m["peak_instant"]
        acc["nadir"] += np.abs(m["nadir"])
        coi += abs(m["coi_rocof_500ms"][0])
        coi_n += abs(m["coi_nadir"][0])
    for k in acc:
        acc[k] /= N
    acc["coi"] = np.full(N, coi / N)
    acc["coi_nadir"] = np.full(N, coi_n / N)
    return acc


def cluster_index(lat: sw.Lattice, buses: set[int]) -> float:
    """Mean pairwise distance within the converter set: low = clustered."""
    b = sorted(buses)
    d = [lat.dist[i, j] for n, i in enumerate(b) for j in b[n + 1:]]
    return float(np.mean(d))


def largest_cluster(lat: sw.Lattice, buses: set[int]) -> int:
    """Buses in the largest contiguous all-converter patch: the biggest hole in
    the inertia map.

    `cluster_index` describes how the converters are arranged relative to each
    other, which turns out not to be the thing the swing responds to.  It cannot
    tell two compact blobs placed far apart from an evenly spread sublattice --
    `two blocks of 8` and `dispersed sublattice` score an identical 4.267 on it,
    and their worst buses differ by 1.7x.  It is also nearly blind on random
    placements, which all land between 3.69 and 4.23 out of a designed range of
    2.67 to 4.27.

    What a bus can actually lean on in the first few hundred milliseconds is the
    inertia within reach of it, so what matters is how deep inside an
    inertia-free region a bus can get -- the size of the patch, not how the
    patches sit relative to one another.  Across the 120 random placements alone,
    this orders the worst 500 ms window with Spearman rho 0.94 and the frequency
    excursion with 0.96, against 0.39 and 0.37 for `cluster_index`.
    """
    remaining = set(buses)
    best = 0
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            i = stack.pop()
            size += 1
            for j in np.nonzero(lat.A[i])[0]:
                j = int(j)
                if j in remaining:
                    remaining.discard(j)
                    stack.append(j)
        best = max(best, size)
    return best


def summarise(tag: str, lat: sw.Lattice, ni: set[int], f: dict[str, np.ndarray]) -> dict:
    mask = np.zeros(lat.N, bool)
    mask[list(ni)] = True
    row = {"config": tag, "n_converters": len(ni),
           "cluster_index": cluster_index(lat, ni),
           "largest_cluster": largest_cluster(lat, ni),
           "converters": "-".join(map(str, sorted(ni)))}
    for k in ("rocof_500ms", "worst_window", "peak_instant", "nadir"):
        row[f"{k}_conv"] = f[k][mask].mean()
        row[f"{k}_sync"] = f[k][~mask].mean()
        row[f"{k}_ratio"] = f[k][mask].mean() / f[k][~mask].mean()
        row[f"{k}_worst"] = f[k].max()
        row[f"{k}_sd"] = f[k].std()
    row["coi"] = f["coi"][0]
    row["coi_nadir"] = f["coi_nadir"][0]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="designed.csv")
    args = ap.parse_args()

    lat = sw.Lattice(L, K=3.0, periodic=True)
    named = {
        "block 4x4 (all in one corner)": block(lat),
        "two blocks of 8": two_blocks(lat),
        "band of 2 rows": band(lat),
        "dispersed sublattice": dispersed(lat),
    }
    for tag, ni in named.items():
        assert len(ni) == K_CONV, (tag, len(ni))

    rows, fields = [], {}
    print(f"{L}x{L} torus, {lat.N} buses, {K_CONV} of them converted to inverter")
    ref = sw.build_network(lat, set(), layout="flat", droop=sw.DROOP_GAIN)
    print(f"total inertia identical in every configuration; "
          f"all-synchronous COI RoCoF would be "
          f"{0.5 / ref.M_total / (2 * np.pi):.4f} Hz/s\n")

    for tag, ni in named.items():
        net = sw.build_network(lat, ni, layout="flat", droop=sw.DROOP_GAIN)
        f = field(net)
        fields[tag] = (ni, f)
        rows.append(summarise(tag, lat, ni, f))
        print(f"  ran {tag}")

    rng = np.random.default_rng(args.seed)
    for i in range(args.random):
        ni = set(rng.choice(lat.N, K_CONV, replace=False).tolist())
        net = sw.build_network(lat, ni, layout="flat", droop=sw.DROOP_GAIN)
        rows.append(summarise(f"random {i}", lat, ni, field(net)))
    print(f"  ran {args.random} random placements\n")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    named_df = df[~df.config.str.startswith("random")]
    rnd = df[df.config.str.startswith("random")]

    print("=" * 78)
    print(f"COI (system-wide) 500 ms RoCoF, same in every row: "
          f"{df['coi'].iloc[0]:.4f} Hz/s")
    print("\nRoCoF AT THE CONVERTER BUSES, RELATIVE TO THE SYNCHRONOUS BUSES")
    print("(ratio > 1 means the converter buses are the exposed ones)\n")
    print(f"  {'configuration':<32s} {'500ms':>9s} {'worst 500ms':>13s} "
          f"{'instantaneous':>15s}")
    for _, r in named_df.iterrows():
        print(f"  {r.config:<32s} {r.rocof_500ms_ratio:>9.3f} "
              f"{r.worst_window_ratio:>13.3f} {r.peak_instant_ratio:>15.2f}")
    print(f"  {'random placements (mean)':<32s} {rnd.rocof_500ms_ratio.mean():>9.3f} "
          f"{rnd.worst_window_ratio.mean():>13.3f} "
          f"{rnd.peak_instant_ratio.mean():>15.2f}")

    print("\nABSOLUTE NUMBERS (Hz/s), averaged over a load step at every bus\n")
    print(f"  {'configuration':<32s} {'500ms conv':>11s} {'500ms sync':>11s} "
          f"{'worst conv':>11s} {'worst sync':>11s}")
    for _, r in named_df.iterrows():
        print(f"  {r.config:<32s} {r.rocof_500ms_conv:>11.4f} "
              f"{r.rocof_500ms_sync:>11.4f} {r.worst_window_conv:>11.4f} "
              f"{r.worst_window_sync:>11.4f}")

    print("\nSPATIAL UNEVENNESS OF THE FIELD (sd across buses, Hz/s)\n")
    print(f"  {'configuration':<32s} {'500ms':>9s} {'worst 500ms':>13s}")
    for _, r in named_df.iterrows():
        print(f"  {r.config:<32s} {r.rocof_500ms_sd:>9.5f} {r.worst_window_sd:>13.5f}")
    print(f"  {'random placements (mean)':<32s} {rnd.rocof_500ms_sd.mean():>9.5f} "
          f"{rnd.worst_window_sd.mean():>13.5f}")

    print("\nWORST SINGLE BUS IN THE GRID (Hz/s)\n")
    print(f"  {'configuration':<32s} {'worst 500ms window':>19s}")
    for _, r in named_df.iterrows():
        print(f"  {r.config:<32s} {r.worst_window_worst:>19.4f}")
    print(f"  {'random placements (mean)':<32s} "
          f"{rnd.worst_window_worst.mean():>19.4f}")
    b = named_df.set_index("config")
    clus = b.loc["block 4x4 (all in one corner)", "worst_window_worst"]
    disp = b.loc["dispersed sublattice", "worst_window_worst"]
    print(f"\n  clustering the same 16 converters into one corner raises the worst "
          f"bus\n  from {disp:.4f} to {clus:.4f} Hz/s "
          f"({100 * (clus / disp - 1):+.1f}%)")

    print("\nMAPS: worst 500 ms window at each bus (Hz/s), converters marked C\n")
    for tag in ("block 4x4 (all in one corner)", "dispersed sublattice"):
        ni, f = fields[tag]
        g = f["worst_window"].reshape(L, L)
        print(f"  {tag}   [system value {f['coi'][0]:.3f}]")
        for r in range(L):
            cells = " ".join(
                ("*" if lat.index(r, c) in ni else " ") + f"{g[r, c]:.3f}"
                for c in range(L))
            print("   ", cells)
        print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

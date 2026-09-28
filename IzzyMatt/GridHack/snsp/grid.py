"""The toy grid this SNSP investigation runs on -- defined once, then frozen.

The local-effects study (../local_effects) deliberately used a torus: every bus
had four neighbours and no bus was topologically special, so any spatial
structure in the answer could only have come from the inertia map.  That is the
right lattice for isolating a mechanism and the wrong one for asking what the
mechanism costs *Ireland*, because the whole point about the Irish system is that
it is not symmetric:

  - the wind is on the Atlantic seaboard, the demand is on the east and south
    coasts, and the two are a long way apart;
  - a good part of that wind connects radially -- one circuit, no alternative
    path -- so it is electrically remote as well as geographically remote;
  - non-synchronous plant already supplies the majority of the energy at the
    times that matter, which is exactly what an SNSP limit is a limit on;
  - the machines that are left are few, and they are scattered.

So this module builds a 10x10 toy grid with those properties written into it
(see LAYOUT NOTES below), and nothing else.  It is a *caricature* of the Irish
system, not a model of it: no real bus, line rating or geography is claimed.

Everything here is geometry, topology and the fixed nameplate of each unit.
None of it varies in the investigation that follows -- **only the dispatch
does**, via `dispatch()`.  That separation is why this file exists on its own and
writes its output to disk: every later experiment loads the same frozen grid from
`grid_buses.csv` / `grid_lines.csv` and checks it against the hash recorded in
`grid.json`, so a comparison between two SNSP levels can never be contaminated by
a quietly edited network.

The `ToyGrid` object is deliberately attribute-compatible with
`local_effects/swing.py`'s `Lattice` (`N`, `coords`, `A`, `Kij`, `dist`), so the
swing-equation machinery already written there runs on it unchanged.

LAYOUT NOTES (rows run north 0 -> south 9, columns west 0 -> east 9)

  wind     Twelve non-synchronous farms.  Eight sit on the west coast (column 0)
           and four inland; 77% of the wind nameplate is on the coast.  Four of
           the coastal farms (rows 2, 3, 4, 7) are *radial*: their north-south
           coastal circuits are absent, so the only way out is a single 110 kV
           circuit east into column 1.  This is the Donegal/Mayo/Kerry situation
           and it is the single most important feature of the grid for an SNSP
           study -- a radial low-inertia infeed has no local machine to lean on
           and no second path to share the swing.
  hvdc     One HVDC infeed on the east coast beside Dublin (the East-West
           Interconnector).  Non-synchronous, so it counts against SNSP, but it
           is not wind and is held flat by default.
  sync     Five synchronous stations, spread so that no two are within five
           squares of each other on the map (three hops through the graph,
           since the backbone shortens some of those): north-east, midlands,
           south-west, east and south.  Few, but
           nowhere near a cluster -- they are what is left after the fleet
           retires, and where they sit is what the rest of this study is about.
  city     Five large demand buses on the east and south boundaries: Dublin far
           the largest, then Belfast, Cork, Limerick, Waterford.  Together they
           are 65% of demand.
  load     Every other bus carries a small, equal share of rural demand.

  lines    Three classes, because a uniform coupling would hide the thing that
           makes radial wind fragile:
             backbone  the 400 kV transfers, including the west-to-east and
                       south-to-east corridors: five long lines that skip over
                       the lattice.
             main      the 220 kV mesh: the ordinary nearest-neighbour links.
             spur      the western seaboard's 110 kV circuits.
           Nineteen nearest-neighbour links are absent (mountains, bog, the
           Shannon), including a partial cut across the middle of the country
           that forces north-south transfer onto the east and west flanks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# The specification.  Every number below is frozen for the whole investigation.
# ---------------------------------------------------------------------------

L = 10                      # 10 x 10 = 100 buses

#: Inertia constant H (s) by role, on each unit's own rating.  Same values as
#: ../local_effects/swing.py, so the two studies are directly comparable.
#: Stored energy at a bus is H_i * S_i, not H_i: an inertia constant is quoted on
#: the unit's *own* rating, so a 3.4 p.u. machine at H = 4 s carries a hundred
#: times the stored energy of a 0.05 p.u. rural load at H = 0.2 s.  Ignoring the
#: rating would give every load bus as much inertia as a power station and would
#: flatten the very thing this study measures -- system inertia falling as
#: machines come off.  `S_pu` below is the rating each bus is quoted on.
H_BY_ROLE = {"sync": 4.0, "wind": 0.1, "hvdc": 0.1, "city": 0.2, "load": 0.2}

#: Minimum rating a bus is quoted on.  A "load bus" here stands for a chunk of
#: the distribution system rather than one customer, so its rating does not fall
#: below this even where its demand does; without a floor the near-massless
#: buses ring at tens of Hz, which is a numerical artefact rather than a grid.
S_BASE_FLOOR = 0.5

#: Coupling K (p.u.) by line class.  A line's class is fixed; its *flow* is not.
K_BY_CLASS = {"backbone": 8.0, "main": 4.0, "spur": 2.5}

#: Total system demand, p.u.  Fixed: the investigation varies the generation
#: mix, not the load, so that SNSP is the only thing moving.
DEMAND_TOTAL = 12.0

#: The operating point every structural claim about this grid is stated at.  The
#: investigation sweeps around it; nothing else in this file moves with it.
REFERENCE_SNSP = 0.70

#: West-coast wind: (row, name, nameplate p.u.) at column 0.
#:
#: The wind nameplate totals 13.2 p.u. against 12.0 p.u. of demand, so that
#: **SNSP = 100% is reachable** -- the investigation sweeps to a system with no
#: synchronous machine running at all, and that end point has to be inside the
#: installed capacity or the sweep stops short of the question.  At 100% the
#: farms run at 86% of nameplate, which still leaves room to vary *which* farms
#: are producing.  Position, topology and role are untouched by this: the
#: nameplate is a number on a bus, not a piece of geography.
WIND_COAST = (
    (1, "WF-Donegal", 1.35),
    (2, "WF-Mayo", 1.20),
    (3, "WF-Connemara", 1.20),
    (4, "WF-Clare", 1.20),
    (5, "WF-Kerry", 1.35),
    (6, "WF-CorkWest", 1.35),
    (7, "WF-Bantry", 1.20),
    (8, "WF-Mizen", 1.35),
)
#: Coastal farms whose north-south circuits are missing: radial, one way out.
STUB_ROWS = (2, 3, 4, 7)

#: Inland wind: (row, col, name, nameplate).
WIND_INLAND = (
    (0, 4, "WF-Sperrin", 0.75),
    (2, 2, "WF-Sligo", 0.75),
    (6, 2, "WF-Shannon", 0.75),
    (8, 3, "WF-Blackwater", 0.75),
)

#: The HVDC infeed: (row, col, name, nameplate).  East coast, beside Dublin.
HVDC = (4, 9, "HVDC-East", 0.6)

#: Synchronous plant: (row, col, name, nameplate).  Five stations, minimum
#: pairwise hop distance 5 -- spread, not clustered, and not everywhere.
#: Nameplate totals 12.6 p.u. against 12.0 p.u. of demand: the machines can
#: still carry the whole system on their own (SNSP = 0), which is what makes the
#: SNSP sweep a sweep rather than a set of separate systems.  What they cannot do
#: is be in more than five places.
SYNC = (
    (1, 6, "SY-North", 1.9),
    (3, 2, "SY-Midlands", 1.6),
    (5, 8, "SY-East", 3.4),
    (7, 3, "SY-Shannon", 3.0),
    (9, 6, "SY-Harbour", 2.7),
)

#: Cities: (row, col, name, demand p.u.).  East and south boundaries only.
CITIES = (
    (0, 9, "Belfast", 1.6),
    (3, 9, "Dublin", 3.0),
    (9, 2, "Limerick", 1.0),
    (9, 5, "Cork", 1.5),
    (9, 8, "Waterford", 0.7),
)

#: Nearest-neighbour links that do not exist.  The first six isolate the radial
#: coastal farms; the next three are the partial cut across the middle; the rest
#: thin the mesh, so that the grid is a uniform lattice nowhere.
MISSING = (
    ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (4, 0)), ((4, 0), (5, 0)),
    ((6, 0), (7, 0)), ((7, 0), (8, 0)),
    ((4, 4), (5, 4)), ((4, 5), (5, 5)), ((4, 6), (5, 6)),
    ((0, 7), (1, 7)), ((1, 2), (1, 3)), ((2, 5), (2, 6)), ((2, 3), (3, 3)),
    ((3, 4), (3, 5)), ((5, 2), (6, 2)), ((6, 4), (7, 4)), ((6, 7), (6, 8)),
    ((7, 5), (7, 6)), ((8, 1), (8, 2)),
)

#: Long 400 kV lines that skip over the lattice: the transfer corridors.
BACKBONE = (
    ((6, 1), (7, 3)),      # west-coast wind into the Shannon station
    ((7, 3), (5, 6)),      # Shannon -> midlands
    ((5, 6), (3, 9)),      # midlands -> Dublin
    ((9, 6), (6, 7)),      # south -> east
    ((1, 6), (0, 9)),      # north -> Belfast
)

ROLES = ("wind", "hvdc", "sync", "city", "load")


def inertia(H: np.ndarray, S: np.ndarray) -> np.ndarray:
    """M = 2 H S / omega_s: the coefficient on theta-double-dot, in per-unit.

    Same convention as ../local_effects/swing.py, with the unit rating S carried
    explicitly rather than assumed to be 1 p.u. everywhere.
    """
    return 2.0 * H * S / (2.0 * np.pi * 50.0)


# ---------------------------------------------------------------------------
# The grid object
# ---------------------------------------------------------------------------

@dataclass
class ToyGrid:
    """A fixed set of buses, lines and nameplates.  Nothing here ever varies.

    Attribute-compatible with ``local_effects/swing.py``'s ``Lattice``: it
    exposes ``N``, ``coords``, ``A``, ``Kij`` and ``dist``, which is everything
    the swing-equation solvers there touch.
    """

    L: int
    coords: list[tuple[int, int]]
    role: np.ndarray            # one of ROLES, per bus
    name: np.ndarray            # str, per bus
    capacity: np.ndarray        # nameplate p.u. (generators; 0 elsewhere)
    demand: np.ndarray          # p.u. (loads; 0 at generator buses)
    H: np.ndarray               # inertia constant, s
    A: np.ndarray               # adjacency
    Kij: np.ndarray             # coupling matrix
    line_class: dict[tuple[int, int], str]

    def __post_init__(self) -> None:
        self.N = self.L * self.L
        self.dist = shortest_path(csr_matrix(self.A), unweighted=True,
                                  directed=False)
        #: Rating each bus's inertia constant is quoted on: its own nameplate if
        #: it generates, its own demand if it does not, floored so that a small
        #: rural bus still stands for a chunk of network.
        self.S = np.maximum(np.maximum(self.capacity, self.demand),
                            S_BASE_FLOOR)
        self.M = inertia(self.H, self.S)

    def stored_energy(self, H: np.ndarray | None = None) -> float:
        """System inertia, sum_i H_i S_i (p.u. seconds) -- what falls with SNSP."""
        return float(((self.H if H is None else H) * self.S).sum())

    # -- lookups ------------------------------------------------------------
    def index(self, r: int, c: int) -> int:
        return r * self.L + c

    def by_role(self, role: str) -> np.ndarray:
        return np.flatnonzero(self.role == role)

    @property
    def generators(self) -> np.ndarray:
        return np.flatnonzero(np.isin(self.role.astype(str),
                                      ("wind", "hvdc", "sync")))

    @property
    def non_synchronous(self) -> np.ndarray:
        return np.flatnonzero(np.isin(self.role.astype(str), ("wind", "hvdc")))

    @property
    def degree(self) -> np.ndarray:
        return self.A.sum(axis=1).astype(int)

    @property
    def edges(self) -> list[tuple[int, int]]:
        i, j = np.triu(self.A, 1).nonzero()
        return list(zip(i.tolist(), j.tolist()))

    def klass(self, i: int, j: int) -> str:
        return self.line_class[(min(i, j), max(i, j))]

    # -- checks -------------------------------------------------------------
    def check(self) -> None:
        """Everything that must be true of the grid, asserted at build time."""
        A = self.A
        assert np.array_equal(A, A.T), "adjacency is not symmetric"
        assert np.all(np.diag(A) == 0), "self-loop"
        assert np.all(self.degree >= 1), "isolated bus"
        n_comp, _ = connected_components(csr_matrix(A), directed=False)
        assert n_comp == 1, f"grid is not connected ({n_comp} components)"
        assert np.isfinite(self.dist).all(), "unreachable bus pair"
        assert np.allclose(self.Kij, self.Kij.T), "asymmetric coupling"
        assert np.all((self.Kij > 0) == (A > 0)), "coupling/adjacency mismatch"

        # The structural requirements this grid exists to satisfy.
        assert abs(self.demand.sum() - DEMAND_TOTAL) < 1e-9
        assert np.all(self.capacity[self.role == "load"] == 0.0), \
            "generation at a non-generator bus"
        wind = self.by_role("wind")
        west = [i for i in wind if self.coords[i][1] == 0]
        assert self.capacity[west].sum() > 0.5 * self.capacity[wind].sum(), \
            "most of the wind is not on the west coast"
        # "The majority of the power in the grid comes from non-inertial
        # sources" is a statement about the operating point, not the nameplate:
        # the machines keep enough nameplate to carry the system alone, but at
        # the reference dispatch they are the minority of what is running.
        assert snsp_of(self, dispatch(self, REFERENCE_SNSP)) > 0.5, \
            "the reference dispatch is not majority non-synchronous"
        for r in STUB_ROWS:
            i = self.index(r, 0)
            assert self.degree[i] == 1, f"radial farm at row {r} is not radial"
            assert self.role[i] == "wind"
        # "Evenly distributed, not all in a cluster": checked two ways, because
        # the backbone lines make the machines electrically closer than they
        # look on the map.  Geographically no two are within five squares; over
        # the graph no two are within three hops, i.e. never neighbours and
        # never neighbours-but-one.
        syncs = self.by_role("sync")
        pairs = [(a, b) for n, a in enumerate(syncs) for b in syncs[n + 1:]]
        geo = min(abs(self.coords[a][0] - self.coords[b][0])
                  + abs(self.coords[a][1] - self.coords[b][1]) for a, b in pairs)
        hop = min(self.dist[a, b] for a, b in pairs)
        assert geo >= 5, f"synchronous plant is clustered on the map ({geo})"
        assert hop >= 3, f"synchronous plant is clustered electrically ({hop})"

    # -- tables -------------------------------------------------------------
    def bus_frame(self) -> pd.DataFrame:
        rows = [c[0] for c in self.coords]
        cols = [c[1] for c in self.coords]
        return pd.DataFrame({
            "bus": np.arange(self.N), "name": self.name.astype(str),
            "row": rows, "col": cols,
            "x": cols, "y": [-v for v in rows],     # plotting: north at the top
            "role": self.role.astype(str), "capacity_pu": self.capacity,
            "demand_pu": self.demand, "H_s": self.H, "S_pu": self.S,
            "degree": self.degree,
        })

    def line_frame(self) -> pd.DataFrame:
        rows = []
        for i, j in self.edges:
            ri, ci = self.coords[i]
            rj, cj = self.coords[j]
            rows.append({
                "line": len(rows), "from_bus": i, "to_bus": j,
                "from_name": str(self.name[i]), "to_name": str(self.name[j]),
                "class": self.klass(i, j), "K_pu": self.Kij[i, j],
                "length_hops": abs(ri - rj) + abs(ci - cj),
            })
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        wind = self.by_role("wind")
        west = [i for i in wind if self.coords[i][1] == 0]
        ns = self.non_synchronous
        syncs = self.by_role("sync")
        cl = pd.Series([self.klass(i, j) for i, j in self.edges]).value_counts()
        return {
            "buses": self.N,
            "lines": len(self.edges),
            "lines_by_class": {k: int(v) for k, v in cl.items()},
            "missing_nn_links": len(MISSING),
            "mean_degree": round(float(self.degree.mean()), 3),
            "degree_1_buses": int((self.degree == 1).sum()),
            "generators": int(len(self.generators)),
            "wind_farms": int(len(wind)),
            "radial_wind_farms": len(STUB_ROWS),
            "synchronous_stations": int(len(syncs)),
            "cities": int(len(self.by_role("city"))),
            "demand_total_pu": float(self.demand.sum()),
            "stored_energy_all_on_pu_s": round(self.stored_energy(), 2),
            "city_share_of_demand": round(float(
                self.demand[self.by_role("city")].sum() / self.demand.sum()), 4),
            "nameplate_non_synchronous_pu": float(self.capacity[ns].sum()),
            "nameplate_synchronous_pu": float(self.capacity[syncs].sum()),
            "west_coast_share_of_wind_nameplate": round(float(
                self.capacity[west].sum() / self.capacity[wind].sum()), 4),
            "max_snsp": round(float(min(
                1.0, self.capacity[ns].sum() / self.demand.sum())), 4),
            "min_synchronous_separation_hops": int(min(
                self.dist[a, b] for n, a in enumerate(syncs)
                for b in syncs[n + 1:])),
            "min_synchronous_separation_squares": int(min(
                abs(self.coords[a][0] - self.coords[b][0])
                + abs(self.coords[a][1] - self.coords[b][1])
                for n, a in enumerate(syncs) for b in syncs[n + 1:])),
            "reference_snsp": REFERENCE_SNSP,
        }


def build() -> ToyGrid:
    """Assemble the grid from the specification constants above."""
    N = L * L
    coords = [(r, c) for r in range(L) for c in range(L)]
    idx = {rc: i for i, rc in enumerate(coords)}

    role = np.array(["load"] * N, dtype=object)
    name = np.array([f"b{r}{c}" for r, c in coords], dtype=object)
    capacity = np.zeros(N)
    demand = np.zeros(N)

    for r, nm, cap in WIND_COAST:
        i = idx[(r, 0)]
        role[i], name[i], capacity[i] = "wind", nm, cap
    for r, c, nm, cap in WIND_INLAND:
        i = idx[(r, c)]
        role[i], name[i], capacity[i] = "wind", nm, cap
    r, c, nm, cap = HVDC
    i = idx[(r, c)]
    role[i], name[i], capacity[i] = "hvdc", nm, cap
    for r, c, nm, cap in SYNC:
        i = idx[(r, c)]
        role[i], name[i], capacity[i] = "sync", nm, cap
    for r, c, nm, dem in CITIES:
        i = idx[(r, c)]
        role[i], name[i], demand[i] = "city", nm, dem

    # Rural demand: whatever the cities do not take, spread equally over every
    # bus that is neither a generator nor a city.  Equal, because any pattern
    # here would be a second piece of geography competing with the wind map.
    rural = np.flatnonzero(role == "load")
    demand[rural] = (DEMAND_TOTAL - demand.sum()) / len(rural)

    H = np.array([H_BY_ROLE[x] for x in role])

    # -- topology -----------------------------------------------------------
    missing = {(idx[a], idx[b]) for a, b in MISSING}
    missing |= {(b, a) for a, b in missing}
    A = np.zeros((N, N))
    line_class: dict[tuple[int, int], str] = {}

    def link(i: int, j: int, kind: str) -> None:
        A[i, j] = A[j, i] = 1.0
        line_class[(min(i, j), max(i, j))] = kind

    for (r, c), i in idx.items():
        for dr, dc in ((0, 1), (1, 0)):
            rr, cc = r + dr, c + dc
            if rr >= L or cc >= L:
                continue
            j = idx[(rr, cc)]
            if (i, j) in missing:
                continue
            # The western seaboard is 110 kV: any link touching column 0.
            kind = "spur" if (c == 0 or cc == 0) else "main"
            link(i, j, kind)
    for a, b in BACKBONE:
        link(idx[a], idx[b], "backbone")

    Kij = np.zeros((N, N))
    for (i, j), kind in line_class.items():
        Kij[i, j] = Kij[j, i] = K_BY_CLASS[kind]

    g = ToyGrid(L=L, coords=coords, role=role, name=name, capacity=capacity,
                demand=demand, H=H, A=A, Kij=Kij, line_class=line_class)
    g.check()
    return g


# ---------------------------------------------------------------------------
# Dispatch: the one thing that varies
# ---------------------------------------------------------------------------

def dispatch(g: ToyGrid, snsp: float = 0.70, hvdc: float | None = None,
             wind_weights: np.ndarray | None = None) -> np.ndarray:
    """Net injection at every bus, p.u., for a given non-synchronous share.

    `snsp` is the fraction of demand met by non-synchronous plant -- wind plus
    the HVDC infeed -- which is the system operator's SNSP definition with no
    exports to complicate it.  The HVDC is held at its nameplate by default (an
    interconnector position is commercial, not a wind forecast) and the wind is
    loaded pro rata to nameplate unless `wind_weights` says otherwise.  Whatever
    is left is picked up by the synchronous stations, also pro rata to nameplate.

    Loads never move.  Sum(P) = 0 by construction, and the grid's *geometry*,
    *topology* and *inertia map* are untouched -- which is exactly the control
    the SNSP comparison needs.
    """
    if not 0.0 <= snsp <= 1.0:
        raise ValueError(f"snsp must be in [0, 1], got {snsp}")
    P = -g.demand.copy()
    total = float(g.demand.sum())

    hv = g.by_role("hvdc")
    # The interconnector runs at nameplate whenever the SNSP target leaves room
    # for it; below that it is the first non-synchronous infeed to be backed off,
    # so that SNSP = 0 stays reachable and the sweep spans the whole range.
    hvdc_pu = float(min(g.capacity[hv].sum(), snsp * total)
                    if hvdc is None else hvdc)
    if hvdc_pu > g.capacity[hv].sum() + 1e-12:
        raise ValueError("HVDC set above nameplate")

    wind = g.by_role("wind")
    want_wind = snsp * total - hvdc_pu
    if want_wind < -1e-9:
        raise ValueError(f"snsp={snsp:.3f} is below the fixed HVDC infeed alone")
    w = g.capacity[wind] if wind_weights is None else np.asarray(wind_weights,
                                                                float)
    if w.shape != wind.shape:
        raise ValueError("wind_weights must have one entry per wind farm")
    if w.sum() <= 0:
        raise ValueError("wind_weights sum to zero")
    wind_pu = w / w.sum() * max(want_wind, 0.0)
    if np.any(wind_pu > g.capacity[wind] + 1e-9):
        raise ValueError(f"snsp={snsp:.3f} needs more wind than is installed")

    syn = g.by_role("sync")
    rest = total - hvdc_pu - wind_pu.sum()
    cap = g.capacity[syn]
    if rest < -1e-9:
        raise ValueError("non-synchronous generation exceeds demand")
    if rest > cap.sum() + 1e-9:
        raise ValueError(f"snsp={snsp:.3f} leaves more than the machines can cover")

    P[hv] += hvdc_pu / max(len(hv), 1)
    P[wind] += wind_pu
    P[syn] += cap / cap.sum() * max(rest, 0.0)
    assert abs(P.sum()) < 1e-9, "dispatch does not balance"
    return P


def snsp_of(g: ToyGrid, P: np.ndarray) -> float:
    """The non-synchronous share of generation for an injection vector."""
    gen = np.clip(P, 0.0, None)
    return float(gen[g.non_synchronous].sum() / gen.sum())


# ---------------------------------------------------------------------------
# Freezing it to disk
# ---------------------------------------------------------------------------

BUSES_CSV, LINES_CSV, SPEC_JSON = "grid_buses.csv", "grid_lines.csv", "grid.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(g: ToyGrid, out: Path = HERE) -> dict:
    """Write the frozen grid, plus a hash of it, so later runs can verify."""
    out.mkdir(parents=True, exist_ok=True)
    g.bus_frame().to_csv(out / BUSES_CSV, index=False, lineterminator="\n")
    g.line_frame().to_csv(out / LINES_CSV, index=False, lineterminator="\n")
    spec = {
        "description": "Irish-caricature toy grid for the SNSP investigation",
        "L": g.L,
        "H_by_role": H_BY_ROLE,
        "S_base_floor_pu": S_BASE_FLOOR,
        "K_by_class": K_BY_CLASS,
        "demand_total_pu": DEMAND_TOTAL,
        "radial_wind_rows": list(STUB_ROWS),
        "backbone_lines": [[list(a), list(b)] for a, b in BACKBONE],
        "missing_nn_links": [[list(a), list(b)] for a, b in MISSING],
        "summary": g.summary(),
        "sha256": {BUSES_CSV: _sha256(out / BUSES_CSV),
                   LINES_CSV: _sha256(out / LINES_CSV)},
    }
    (out / SPEC_JSON).write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def load(out: Path = HERE, verify: bool = True) -> ToyGrid:
    """Rebuild the grid from the frozen CSVs -- the source of truth downstream.

    Reads the files rather than the constants at the top of this module, so an
    edit to the specification that has not been re-frozen and re-checked cannot
    silently change an experiment underneath you.
    """
    buses = pd.read_csv(out / BUSES_CSV)
    lines = pd.read_csv(out / LINES_CSV)
    if verify:
        spec = json.loads((out / SPEC_JSON).read_text())
        for fname, want in spec["sha256"].items():
            if _sha256(out / fname) != want:
                raise RuntimeError(
                    f"{fname} does not match the hash in {SPEC_JSON}: the "
                    f"frozen grid has been edited. Re-run "
                    f"`python grid.py --freeze` only if that was intended.")

    side = int(buses["row"].max()) + 1
    N = len(buses)
    A = np.zeros((N, N))
    Kij = np.zeros((N, N))
    line_class: dict[tuple[int, int], str] = {}
    for _, r in lines.iterrows():
        i, j = int(r["from_bus"]), int(r["to_bus"])
        A[i, j] = A[j, i] = 1.0
        Kij[i, j] = Kij[j, i] = float(r["K_pu"])
        line_class[(min(i, j), max(i, j))] = str(r["class"])
    g = ToyGrid(L=side,
                coords=list(zip(buses["row"].tolist(), buses["col"].tolist())),
                role=buses["role"].to_numpy(dtype=object),
                name=buses["name"].to_numpy(dtype=object),
                capacity=buses["capacity_pu"].to_numpy(float),
                demand=buses["demand_pu"].to_numpy(float),
                H=buses["H_s"].to_numpy(float), A=A, Kij=Kij,
                line_class=line_class)
    g.check()
    return g


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(g: ToyGrid) -> str:
    out = ["TOY GRID FOR THE SNSP INVESTIGATION", "=" * 70, "", "Structure"]
    for k, v in g.summary().items():
        out.append(f"  {k:38s} {v}")

    out += ["", "Generators (nameplate, p.u.)", "-" * 70]
    for i in g.generators:
        r, c = g.coords[i]
        out.append(f"  {g.name[i]:14s} ({r},{c})  {g.role[i]:5s}  "
                   f"cap {g.capacity[i]:5.2f}  H {g.H[i]:4.1f} s  "
                   f"degree {g.degree[i]}")

    out += ["", "Demand (p.u.)", "-" * 70]
    for i in g.by_role("city"):
        r, c = g.coords[i]
        out.append(f"  {g.name[i]:14s} ({r},{c})  {g.demand[i]:5.2f}")
    rural = g.by_role("load")
    out.append(f"  rural x{len(rural):<3d}             "
               f"{g.demand[rural].sum():5.2f} total "
               f"({g.demand[rural][0]:.4f} each)")

    syn = g.by_role("sync")
    ns = g.non_synchronous
    out += ["", f"Reference dispatch (SNSP {REFERENCE_SNSP * 100:.0f}%)",
            "-" * 70]
    P = dispatch(g, REFERENCE_SNSP)
    out.append(f"  non-synchronous {P[ns].sum():5.2f} p.u. "
               f"({snsp_of(g, P) * 100:.1f}% of generation, "
               f"{P[ns].sum() / g.capacity[ns].sum() * 100:.0f}% of nameplate)")
    out.append(f"  synchronous     {P[syn].sum():5.2f} p.u. "
               f"({P[syn].sum() / g.capacity[syn].sum() * 100:.0f}% of nameplate)")
    out.append(f"  stored energy   {g.stored_energy():.1f} p.u. s with every "
               f"machine synchronised;\n                  "
               f"{g.stored_energy(np.where(g.role == 'sync', 0.05, g.H)):.1f} "
               f"with none of them -- the range the ensemble moves over")

    out += ["", "Feasible SNSP range", "-" * 70]
    top = float(min(1.0, g.capacity[ns].sum() / g.demand.sum()))
    for x in (0.0, 0.2, 0.4, 0.6, REFERENCE_SNSP, 0.8, top):
        try:
            P = dispatch(g, float(x))
            sy = P[syn].sum()
            out.append(f"  SNSP {x * 100:5.1f}%   wind {P[g.by_role('wind')].sum():5.2f}"
                       f"   sync {sy:5.2f}   machines at "
                       f"{sy / g.capacity[syn].sum() * 100:5.1f}% of nameplate")
        except ValueError as e:
            out.append(f"  SNSP {x * 100:5.1f}%   infeasible: {e}")

    out += ["", "Radial (single-circuit) wind farms", "-" * 70]
    for r in STUB_ROWS:
        i = g.index(r, 0)
        j = int(np.flatnonzero(g.A[i])[0])
        out.append(f"  {g.name[i]:14s} -> {g.name[j]:6s} only, "
                   f"K = {g.Kij[i, j]:.1f} ({g.klass(i, j)}), "
                   f"nameplate {g.capacity[i]:.2f}")
    return "\n".join(out) + "\n"


def steady_check(g: ToyGrid, n: int = 9) -> str:
    """Does the frozen network actually hold a synchronous steady state?

    A toy grid can be drawn with couplings too weak for the transfers its own
    dispatch demands, and the failure would not appear until the first dynamic
    run.  This solves P_i = sum_j K_ij sin(theta_i - theta_j) at SNSP levels
    across the whole feasible range and reports the most-loaded line, where
    "loading" is sin(angle difference) -- 1.0 is the static stability limit.

    Uses the solver from ../local_effects/swing.py rather than a second copy of
    it; `ToyGrid` is attribute-compatible with the `Lattice` it expects.
    """
    sys.path.insert(0, str(HERE.parent / "local_effects"))
    import swing as sw

    ns = g.non_synchronous
    top = float(min(1.0, g.capacity[ns].sum() / g.demand.sum()))
    out = ["Static feasibility across the SNSP range", "-" * 70,
           "  SNSP    worst line loading   max angle gap   worst line"]
    for x in np.linspace(0.0, top, n):
        P = dispatch(g, float(x))
        theta = sw.steady_state(g, P)
        d = theta[:, None] - theta[None, :]
        s = np.where(g.A > 0, np.abs(np.sin(d)), 0.0)
        i, j = np.unravel_index(int(np.argmax(s)), s.shape)
        out.append(f"  {x * 100:5.1f}%   {s[i, j]:14.3f}   "
                   f"{np.abs(d)[g.A > 0].max() * 180 / np.pi:11.1f} deg   "
                   f"{g.name[i]}-{g.name[j]} ({g.klass(i, j)})")
    out.append("")
    out.append("  A steady state exists at every level, and the most-loaded "
               "circuit peaks\n  at about half its static limit at SNSP = 100%, "
               "so nothing below is an\n  artefact of the network being pushed "
               "past the point where it can hold\n  synchronism at all.  Note "
               "which circuit it is: above ~40% SNSP the\n  binding line stops "
               "being an urban import and becomes a west-coast spur\n  "
               "exporting wind.")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", action="store_true",
                    help="write grid_buses.csv, grid_lines.csv, grid.json")
    ap.add_argument("--verify", action="store_true",
                    help="rebuild from the spec and check it against the "
                         "frozen files")
    ap.add_argument("--steady", action="store_true",
                    help="solve the steady state across the SNSP range "
                         "(needs ../local_effects/swing.py)")
    args = ap.parse_args()

    g = build()
    text = report(g)
    if args.steady:
        text += "\n" + steady_check(g)
    print(text)

    if args.freeze:
        spec = save(g, HERE)
        (HERE / "grid_report.txt").write_text(text)
        print(f"wrote {BUSES_CSV}, {LINES_CSV}, {SPEC_JSON}, grid_report.txt")
        print(f"sha256 {spec['sha256'][BUSES_CSV][:16]}... "
              f"{spec['sha256'][LINES_CSV][:16]}...")

    if args.verify:
        f = load(HERE)
        assert np.array_equal(f.A, g.A), "adjacency differs from the frozen file"
        assert np.allclose(f.Kij, g.Kij), "coupling differs"
        assert np.array_equal(f.role.astype(str), g.role.astype(str)), \
            "roles differ"
        assert np.allclose(f.capacity, g.capacity), "nameplates differ"
        assert np.allclose(f.demand, g.demand), "demand differs"
        assert np.allclose(f.H, g.H), "inertia differs"
        assert np.allclose(dispatch(f, 0.7), dispatch(g, 0.7)), "dispatch differs"
        print("verify: the frozen grid matches the specification exactly")


if __name__ == "__main__":
    main()

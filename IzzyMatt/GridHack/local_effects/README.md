# §3.5 Local effects — inertia geography and RoCoF on a toy grid

Second-order Kuramoto (the swing equation), as set out in
[arXiv:2502.09024](https://arxiv.org/html/2502.09024), on a square lattice, used to
ask whether inertia is only a system-wide quantity or whether **where** you put it
changes what a given bus experiences.

**Headline:** it changes it a lot. Holding total system inertia and the pre-fault
operating point exactly constant, clustering non-inertial generation into one
corner **roughly doubles the RoCoF at the worst bus** (0.238 vs 0.116 Hz/s) and raises
the worst frequency excursion by half (143 vs 99 mHz) versus
spreading the same generators evenly, and concentrates the exposure onto the
cluster itself. The hypothesis that motivated this study is confirmed — but only
if RoCoF is measured on a window short enough to see it.

## The experiment

The model is Eq. (1) of the paper,

```
M_i θ̈_i + D_i θ̇_i = P_i − Σ_j K_ij sin(θ_i − θ_j)
```

with `M_i = 2H_i/ω_s` in per-unit, so that `θ̈/2π` is a RoCoF in Hz/s — the units
the grid code is written in. Synchronous machines get `H = 4 s`, inverter-based
units `H = 0.1 s`, load buses `H = 0.5 s`.

The design rests on one property of the equation: **its steady state depends on
`P` and `K`, never on `M`.** So the sweep fixes the power injection at every bus
and changes only *which* generators are synchronous and which are inverter-based.
Every configuration therefore has

- an identical pre-fault operating point — same angles, same line flows,
- identical total system inertia, hence an identical system-wide (centre-of-inertia)
  RoCoF,

and the only thing that differs is the geography of inertia. Disturbances are
averaged over a fixed, spatially balanced set (a load step at every bus, or a trip
of every generator in turn) identical across configurations, so the excitation
cannot imprint a pattern of its own. The main lattice is a **torus**, so every bus
has exactly four neighbours and no bus is topologically special — on an open
lattice, "far from the converters" is confounded with "weakly connected", and the
two explanations cannot be separated.

### Two disturbance regimes

**A discrete event** — a step change to the injection at one bus, held from t = 0:
a unit tripping (`gen_trip`) or a load block switching in (`load_step`). Every
configuration sees the same family of events, swept over every bus.

**Continuous stochastic imbalance** (`stochastic.py`) — the everyday regime, where
frequency wanders because load and weather-driven output never balance exactly.
Each converter bus carries **its own** Ornstein–Uhlenbeck power imbalance with
correlation time `τ`, spatially correlated between buses as `exp(-d/ℓ)` — sixteen
separate processes, not one block-wide forcing. The fluctuation is placed **at the
converter buses**, because wind and solar output is what actually varies.
Since the system is linear, this needs no Monte
Carlo: the stationary covariance solves a Lyapunov equation and every statistic
falls out exactly (verified against brute-force simulation to <0.5%).

### Primary frequency response

Each synchronous bus carries a governor state with 5% droop and a 5 s lag, so a
power deficit is actually made good rather than left open forever. Converters
contribute nothing unless explicitly given fast frequency response, which is a
control choice rather than a property of the machine.

This matters for realism but not for the result: **RoCoF over the first 500 ms is
governor-independent** (it moves by ~1%), because a 5 s governor has delivered
almost no extra MW by then. What the governor fixes is everything after — there is
now a frequency nadir at ~2 s and a droop offset, instead of a frequency that
falls for 30 seconds. All the RoCoF findings below were reproduced with the
governor enabled and the clustered/spread ratio was unchanged (1.96 → 2.04).

Four metrics are reported, because they disagree and the disagreement matters:

| metric | what it is |
|---|---|
| `rocof_500ms` | mean RoCoF over the 500 ms window starting at the event — the grid-code measurement |
| `nadir` / `excursion` | how far the bus frequency actually goes — what under-frequency load shedding watches. Only meaningful with the governor enabled. |
| `worst_window` | the worst 500 ms window *anywhere* in the event — what a RoCoF relay would actually latch |
| `peak_instant` | the instantaneous peak — what a fast PMU or a converter's own PLL sees |

## What was found

**1. Local effects are real, and large.** A bus's RoCoF is not the system value.
In the nonlinear checkerboard run — real pre-fault flows, generator trips, no
averaging — per-bus 500 ms RoCoF spans **0.03 to 2.66 Hz/s about a system-wide
value of 1.06**, a local excess of up to 151% of the global number. The mechanism
is visible in the timing: the peak swing arrives 69 ms after the event one hop
away and 492 ms six hops away. The disturbance takes time to cross the grid, and
the local modes ring while it does.

Averaging over every disturbance location deliberately cancels the fault-position
part of that, leaving only what the inertia map itself imposes; even then the 4×4
torus sweep (all 1820 placements) still spans 0.219–0.271 Hz/s about 0.243, ±12%.
Every number below is of that averaged, conservative kind.

**2. Clustering roughly doubles the worst bus on RoCoF, and raises it by half on
frequency.** Both are reported throughout, because a grid code constrains both and
they are guarding against different things: RoCoF relays and loss-of-mains
protection watch d*f*/d*t*, while under-frequency load shedding watches *f*
itself. 8×8 torus, 16 of 64 buses converted, governor on, identical total inertia
in every row:

| converter placement | worst 500 ms RoCoF (Hz/s) | worst frequency excursion (mHz) |
|---|---|---|
| one 4×4 block | **0.2378** | **142.7** |
| band of 2 rows | 0.2423 | 140.9 |
| two blocks of 8 | 0.1974 | 131.1 |
| spread on a regular sublattice | **0.1164** | **98.8** |
| 120 random placements (mean) | 0.1590 | 120.4 |
| *system-wide (COI) value* | *0.0595* | *74.8* |

Clustered ÷ dispersed is **2.04× on RoCoF and 1.44× on frequency**. RoCoF is hurt
about twice as hard, and the reason is visible in the last row: the frequency
excursion is a large common-mode dip (74.8 mHz, identical everywhere) with the
local swing added on top, so the shared component dilutes the ratio. RoCoF has no
such large common part to hide behind.

Spreading the converters evenly makes both fields nearly flat — the spatial
standard deviation falls from 0.048 to 0.005 Hz/s on RoCoF (10×) and from 18.1 to
1.0 mHz on frequency (18×). The extra exposure sits squarely *on* the cluster in
both (`fig2_maps.png`), and on the instantaneous RoCoF metric the clustered
converter buses see 17.5× the synchronous buses.

**3. The mechanism is that a clustered low-inertia region stops following the
system frequency.** A lone converter is surrounded by machines that anchor it; a
block of them is anchored only around its perimeter, so the interior is free to
swing. `fig1_propagation.png` shows it — with an important qualification, because
**where the event happens matters as much as the geography**:

| configuration | event location | peak pocket-vs-system separation |
|---|---|---|
| clustered | inside the wind block | 186 mHz, at 30 ms |
| clustered | far side of the grid | 16 mHz, at 1.4 s |
| dispersed | either | 3 mHz |

A pocket is hit hardest by its own events. The effect is real regardless — even
for a distant event, clustering gives 5.5× the separation of dispersed siting —
but an illustration that only shows the first row overstates it roughly tenfold,
so all three are drawn. (The headline results in findings 2, 6 and 7 average over
a disturbance at *every* bus, so they are not sensitive to this choice.)

Adding the governor leaves the separation completely unchanged (186.2 mHz either
way): it is established within ~30 ms, long before any governor moves. What the
governor changes is the depth of the common-mode dip the pocket oscillates about.

**Under continuous forcing the mechanism is present but weaker**
(`fig6_noise.png`). Comparing like with like — the same metrics, with the forcing
placed at the converter buses in both regimes:

| metric (clustered ÷ dispersed) | discrete event | continuous noise, independent | continuous noise, correlated |
|---|---|---|---|
| pocket-vs-system frequency | 25.4× | 2.2× | 3.0× |
| worst-bus 500 ms RoCoF | 4.08× | 1.26× | **1.7–2.0×** |

The pocket does hold a frequency of its own continuously, not only after events —
but a single coherent event excites it far more efficiently than diffuse noise
does. See finding 8 for why the last column is the honest one.

**4. Whether the effect is visible at all is a choice of measurement, not of
physics.** One set of solves, read eight ways, gives clustered ÷ dispersed ratios
from 1.04 to 4.37 (`fig4_measurement.png`):

| measurement | ratio |
|---|---|
| RoCoF, 500 ms window pinned to the event | **1.04** — nothing to see |
| RoCoF, worst 500 ms window anywhere | 2.04 |
| RoCoF, instantaneous peak | **4.37** |
| frequency excursion | 1.44 |
| RoCoF sd under noise, independent wind | 1.26 |
| RoCoF sd under noise, correlated wind | 1.99 |
| frequency sd under noise, independent wind | 1.13 |
| frequency sd under noise, correlated wind | 1.54 |

The pinned-window result is the one to worry about. Fixing the window to the
moment of the event averages the local swing away and leaves only the system-wide
value, so **a compliance test of that shape reports no local problem here at
all** — while a sliding window over the identical solve reports a factor of two.
The right-hand panel sweeps the window length and shows the two readings
diverging: they agree below ~100 ms and separate completely by 500 ms.

**5. Exposure is predictable without simulating.** A bus's inertia averaged over
its electrical neighbourhood, weighted `exp(−R_ij/λ)` in effective resistance
(λ ≈ 0.2), explains **72%** of the bus-to-bus variance in the worst-window metric
(r = −0.85), and its top pick is among the three worst buses 93% of the time.
Most of that is carried by whether the bus is itself a converter (r = −0.82
alone); the neighbourhood weighting adds a modest amount. This is the direct
hook into §3.6: it ranks candidate synchronous-condenser sites from one
pseudo-inverse, with no simulation.

**6. Under everyday stochastic forcing the effect is real, and it needs the
fluctuation to sit on the wind.** With the share held at 25% and only the geometry
changed, clustering raises the worst bus's 500 ms RoCoF standard deviation by
**about 26%** if wind farms fluctuate independently, rising to **70–100%** once
their fluctuations are spatially correlated as they really are (finding 8). The
independent-noise ratio is stable at 1.24–1.29 across noise correlation *times*
from 0.2 s to 20 s, so it does not hinge on that parameter — what it hinges on is
the spatial correlation, not the temporal one.

The control run is what localises the mechanism: spreading the *same total*
fluctuation uniformly over every bus instead of concentrating it on the
converters collapses the ratio to 1.03–1.10. So it is not that a low-inertia
region is fragile in the abstract. It is that **co-locating the fluctuating
sources with the missing inertia** removes both the nearby machines that would
have absorbed the imbalance and the spatial averaging between independent
fluctuations. That is your point 4a as a measurable quantity.

**7. Clustering costs about 20 points of non-synchronous share.** The operational
question is not "how many Hz/s" but "how much wind can this grid carry before
RoCoF binds" — which is what an SNSP cap expresses and what curtailment enforces.
Sweeping the share and comparing a single compact cluster against randomly-sited
plant (`snsp.py`, `fig5_snsp.png`), the same worst-bus RoCoF is reached at:

| clustered at | RoCoF: event / noise | frequency: event | headroom lost |
|---|---|---|---|
| 12.5% | 30.0% / 21.6% | 26.0% | 9–18 points |
| **25.0%** | **46.7% / 47.6%** | **41.9%** | **17–23 points** |
| 37.5% | 55.1% / 57.8% | 53.7% | 16–20 points |
| 50.0% | 61.0% / 58.3% | 63.1% | 8–13 points |

Both limits and both disturbance regimes agree on roughly **15–22 points of lost
headroom** at moderate shares, which they did not have to. The advantage
correctly collapses to nothing above ~70% share, where converting most of the
grid leaves "clustered" and "spread" barely distinguishable — a useful sanity
check that the metric is measuring geography and not something else.

One panel of `fig5_snsp.png` is close to a null and is worth reading as such:
frequency *under continuous noise* shows almost no clustering penalty. That is
the independence assumption of finding 8 again — frequency is the more
common-mode-dominated of the two quantities, so it is the one most flattered by
letting each wind farm fluctuate on its own.

**And a global limit cannot see any of it.** At every share the centre-of-inertia
RoCoF is identical for the two geographies to four significant figures (e.g.
0.03224 vs 0.03223 Hz/s at 25%). A cap written on system-wide inertia or SNSP
alone is by construction blind to where the plant is — so two grids with the same
SNSP, one of which has twice the worst-bus RoCoF, are treated identically.

**8. Most of the gap between the two regimes was an artefact of assuming wind
farms fluctuate independently.** Independent per-bus noise lets the fluctuations
inside a cluster cancel against each other in aggregate (√n averaging), which
flatters clustering badly. Real farms a few tens of kilometres apart see much the
same weather. Adding spatial correlation `exp(−d/ℓ)` to the imbalance:

| correlation length ℓ (hops) | pocket separation, clustered | worst-bus RoCoF ratio |
|---|---|---|
| 0 (independent) | 46 mHz | 1.26× |
| 1 | 79 mHz | 1.69× |
| 2 | 108 mHz | 1.99× |
| ∞ (fully coherent) | 170 mHz | 2.27× |

If a lattice hop stands for something like 50–100 km of transmission, plausible
wind correlation lengths are 1–2 hops, giving a RoCoF ratio of **1.7–2.0×** —
essentially the discrete-event answer. So the two regimes agree once the forcing
is modelled realistically, and the "noise is much weaker" reading was a
consequence of the independence assumption, not of the physics.

**9. The governor is powerless against local effects, and inertia is not.** This
is the sharpest practical result here. Under continuous noise, increasing droop
gain from 0 to 100 (5× the standard setting) cuts the system-wide frequency
standard deviation almost in half — and leaves the pocket separation *completely
untouched*:

| droop gain | system frequency sd | pocket separation sd | worst-bus RoCoF |
|---|---|---|---|
| 0 | 116.1 mHz | 45.99 mHz | 0.2953 |
| 20 (standard) | 71.9 mHz | 45.98 mHz | 0.2976 |
| 100 | 49.0 mHz | 45.94 mHz | 0.3007 |

Making the governor five times faster (T = 1 s) does not help either: system sd
falls to 35.0 mHz, pocket separation stays at 45.9. Nor does giving the
converters their own fast frequency response (45.3 mHz).

The reason is structural, not a matter of tuning. Governor response is a
first-order lag acting on each machine's own frequency deviation, which is
dominated by the *common* mode — and the common mode is precisely what it is
designed to arrest. The pocket separation is a *differential* mode between two
regions, and it lives at the local swing frequency (period ~0.1–0.3 s), where a
5 s lag attenuates the response by roughly a factor of 100. A controller that
slow cannot see the mode at all.

Inertia can. Raising the inertia constant at the converter buses — synthetic
inertia, or a synchronous condenser sited in the pocket — attacks it directly:

| H at converter buses | pocket separation sd | worst-bus RoCoF |
|---|---|---|
| 0.1 s | 45.98 mHz | 0.2976 |
| 2.0 s | 30.39 mHz | 0.2392 |
| 4.0 s | 25.57 mHz | 0.1884 |

So **local RoCoF is an inertia problem and not a reserve problem**: buying more or
faster primary response does nothing for it, while putting inertia in the right
place does. That is the direct argument for §3.6.

**10. Non-local lines out of the block undo about a fifth of the penalty — but
which lines you add matters more than how many.** Adding 48 long lines, three
from each converter to its counterpart in the other three quadrants of the torus
(`fig7_shortcuts.png`), halves the graph diameter from 8 hops to 4 and takes the
worst bus from 0.2378 to 0.1869 Hz/s (−21%) and 142.7 to 116.0 mHz (−19%). No
inertia is added and the operating point is untouched, so this is purely
redistributing *access* to the inertia that was already there — the system-wide
RoCoF moves by 0.07%.

The controls are what make that number mean anything, because 48 extra lines is
a 37.5% increase in total coupling and some improvement is bought by the copper
alone:

| intervention (all 48 extra lines, same K) | step: RoCoF / nadir | noise: RoCoF sd / freq sd | separation, step / noise |
|---|---|---|---|
| clustered baseline | 0.2378 / 142.7 | 0.5131 / 246.0 | 39.3 / 107.7 |
| existing lines uprated 37.5%, none added | 0.2167 / 132.8 | — | — |
| 48 lines placed at random | 0.2155 / 131.0 | — | — |
| 48 lines **inside** the wind block | **0.1764** / 117.5 | 0.4874 / **259.3** | 41.4 / **126.0** |
| 48 **quadrant shortcuts** out of the block | 0.1869 / **116.0** | **0.3065 / 198.3** | **16.6 / 49.0** |
| *dispersed siting, for reference* | *0.1164 / 98.8* | *0.2580 / 160.0* | *4.3 / 38.7* |

Random reinforcement buys exactly what uprating every line buys (−9%), so it is
the *placement* that earns the rest.

**The two targeted interventions are not the same fix, and which one looks better
depends entirely on the disturbance regime.** Under a discrete event, stiffening
the block internally is genuinely the stronger option on RoCoF (−26% vs −21%),
and it is not a measurement artefact: the block *mean* excursion improves from
139.5 to 110.9 mHz, so the block really is sharing the shock rather than merely
being homogenised. A single localised event lands on one bus, and internal mesh
spreads it across sixteen.

Under continuous imbalance that advantage evaporates and partly reverses: −5% on
RoCoF sd and **+5% worse on frequency sd**, against −40% and −19% for the
shortcuts. The reason is that everyday forcing does not arrive at one bus. It
enters at all sixteen converters at once and, with realistic spatial correlation,
largely in step — so tying them tightly together makes them move as one large
noisy low-inertia mass with no new path to system inertia, and the pocket
separation grows from 107.7 to 126.0 mHz. Stable across three random choices of
the 48 internal lines and a structured all-2-hop-pairs variant.

Two practical caveats on the internal option. 48 lines inside a 16-bus block is
72 of the 120 possible pairs — a 60%-dense mesh that is closer to collapsing the
cluster into a single node than to any reinforcement anyone would build, whereas
three long lines per converter is at least conceivable. And it leaves the pocket
running its own frequency, which is what anti-islanding and vector-shift
protection react to and what no bus-level metric in this study captures.

**11. The right way to describe a placement is the size of its largest contiguous
patch, not how spread out it is on average.** The obvious descriptor — mean
pairwise distance between converters, `cluster_index` — turns out not to be the
thing the swing equation responds to. `largest_cluster`, the number of buses in
the biggest all-converter patch, orders the outcomes far better:

| descriptor | ρ with worst 500 ms RoCoF | ρ with frequency excursion |
|---|---|---|
| `cluster_index` (mean pairwise distance) | −0.39 | −0.37 |
| `largest_cluster` (biggest patch) | **+0.94** | **+0.96** |

over the 120 random placements. The decisive case is in the named configurations:
`two blocks of 8` and `dispersed sublattice` score an **identical** 4.267 on
cluster index, yet their largest patches are 8 and 1 and their worst buses differ
by 1.7×. Mean separation cannot tell two compact blobs held far apart from an
evenly spread sublattice, and it is nearly blind across random placements, which
all land in 3.69–4.23 out of a designed range of 2.67–4.27.

This follows from finding 3: what a bus can lean on in the first few hundred
milliseconds is the inertia within reach, so what hurts is **how deep inside an
inertia-free region a bus can get** — the depth of the hole, not how the holes sit
relative to one another. `fig3_ensemble.png` is plotted against patch size for
this reason.

**12. All of this is a self-energy, and the inertia a bus carries is dressed and
frequency-dependent.** Writing the linearised system as a matrix inverse
propagator `A(s) = Ms² + Ds + L` and eliminating every other bus is a Schur
complement, `A_eff(s) = A_ii − A_i,r (A_rr)⁻¹ A_r,i` — the operation that produces
a self-energy when fields are integrated out. Its `s²` coefficient is a correction
to the bus's inertia, so a bus carries the whole grid's inertia at zero frequency
(193.6 s) and only its own rotor at infinite frequency (0.10 s), interpolating in
between:

| ω (rad/s) | dressed H at bus 17 |
|---|---|
| → 0 | 191.75 s ≈ the whole grid |
| → ∞ | 0.10 s = the bare rotor |

This is exact rather than perturbative — the linearised problem is Gaussian, so
there is no coupling constant to expand in and no loop series to truncate. The
genuine interaction vertex is `sin(θᵢ−θⱼ)` beyond quadratic order, and everything
here is verified to sit in the linear regime to 0.1%. It is also unrelated to
"higher-order interactions" in the hypergraph sense: adding pairwise lines creates
none.

What the picture buys is the explanation of finding 10. The two reinforcements
dress a block bus out of **different inertia reservoirs** — internal lines from
the block's own (16 × 0.1 s), long lines from the synchronous sector's
(48 × 4 s). Under the real correlated-wind forcing, 70% of the RoCoF variance sits
in the 25–80 rad/s band, where the long lines remove 99% of the contribution and
internal meshing removes 9%. `selfenergy.py` computes both, and its spectral route
reproduces the Lyapunov solver to three decimals.

## Robustness

The ~2× clustered/spread ratio survives every axis swept one at a time
(`robustness.py`): damping law (2.4–3.2×), coupling strength `K` from 1 to 12
(1.7–2.0×), open lattice instead of a torus (1.97×), and measurement windows from
50 ms to 1 s (1.67–4.79×). It fades correctly as converters stop being different —
at `H_inv = 2 s` against `H_sync = 4 s` the ratio is 1.00.

`validate.py` runs 23 checks: the operating point is independent of the inertia
map, initial COI RoCoF equals `ΔP/M_total`, the response is linear in disturbance
size to 0.1%, the linearised solver agrees with the nonlinear equation to
< 0.01 Hz/s while running up to 300× faster, and — the sharpest check — **uniform inertia
produces a perfectly flat field** (spread < 1e-12 Hz/s) once averaged over
disturbances, so any structure reported here is genuinely the inertia map.

## Caveats

- A toy lattice, not a real network: uniform line ratings, uniform coupling, no
  voltage dynamics, DC-style coupling only. The absolute Hz/s numbers are not
  predictions for Ireland; the comparisons between placements are the result.
- Governor response has no headroom limit and no deadband — a machine can always
  deliver whatever droop asks of it. Real headroom is finite, and running out of
  it is itself part of why curtailment exists.
- The noise model is a single-timescale Ornstein–Uhlenbeck process. Real load and
  wind spectra are broadband and closer to 1/f, and real grid frequency is
  non-Gaussian and heavy-tailed, which an OU process cannot reproduce. `τ` is
  therefore swept (0.2–20 s) rather than trusted; the clustering ratio is flat
  across that range.
- Converter buses are modelled as low-inertia machines with the same damping law
  as everything else, not as grid-following inverters with a PLL and current
  control. That is the right first-order representation of "no inertia" — which
  is itself a consequence of the AC-DC-AC decoupling — but it is not a converter
  model, and it does not capture Type 3 DFIGs, which are only partly decoupled
  and do contribute some inherent inertia.
- Converters are modelled as `H = 0.1 s` rather than exactly zero. A true
  zero-inertia bus makes the swing equation a DAE; the small-but-finite value keeps
  it an ODE, and `robustness.py` sweeps `H_inv` from 0.02 to 2 s.
- The main sweeps use a `flat` layout (every bus a machine, zero net injection) so
  that the synchronising stiffness is exactly uniform and the inertia map is the
  only spatial structure. A `checkerboard` layout with real pre-fault flows is also
  provided and run with the full nonlinear solver (`results.csv`).
- **That nonlinear checkerboard run confirms finding 1 but cannot test finding 2.**
  It reproduces the locality and the propagation delay convincingly, but its
  clustered-vs-spread comparison is weak and not significant (r = −0.15, p = 0.21;
  quartile contrast p = 0.058). The reason is structural: on a checkerboard,
  generators are never closer than two hops, so the cluster index only spans
  2.00–3.67 and there is almost no clustering contrast to detect. That limitation
  is exactly why the headline comparison is done on the 8×8 torus, where
  converters can actually be placed adjacent to one another.
- Damping is uniform across buses. That keeps `M` the only thing varying between
  configurations, at the cost of leaving low-inertia buses lightly damped;
  `proportional` (the paper's `γ ∝ m` fit) and `equal_ratio` are both available and
  both *strengthen* the result.
- `peak_instant` is sensitive to the damping model and is never used as a headline.
- **The measurement operator is not a detail, and getting it wrong reverses
  conclusions.** RoCoF averaged over a window `T` has gain `2|sin(ωT/2)|/T` on a
  frequency-deviation component — flat below `1/T`, falling as `1/ω` above it.
  The instantaneous derivative has gain `ω`, rising without limit. Any
  frequency-domain analysis of these results must use the windowed form, or the
  far tail dominates and the ordering of the interventions in finding 10 flips.
  `selfenergy.py` defaults to the window and agrees with `stochastic.stationary`
  to three decimals; with the instantaneous form the two disagree outright.
- The fault-averaged fields in `designed.py` include, for each bus, the one event
  out of 64 in which that bus is itself the one hit. A 0.5 p.u. step landing
  directly on a near-massless converter bus dips it to about 880 mHz, so that
  single event is worth ~9% of the clustered worst-bus figure. Dropping each
  bus's own event moves the headline ratios from 2.04 to 1.89 (RoCoF) and 1.44
  to 1.35 (frequency) — the conclusions are unaffected, but the published numbers
  are mildly flattered by it. It is kept in because a load step at a converter bus
  is a real event that removes nothing from the network (unlike a generator trip,
  which *is* excluded in `experiments.py`), and because the exclusion is not
  neutral either: it changes which bus is worst.
  That last point is itself informative. Under the dispersed layout, excluding
  self-events moves the worst bus off the converters entirely — a scattered
  converter's exposure comes almost wholly from its *own* event. Under the
  clustered layout the block converter stays worst, because it is also exposed to
  its fifteen neighbours' events. That is the clustering mechanism restated.
- Lattice size is a parameter (`--L`). A 3×3 was run first and is too small to
  resolve anything: only 10 distinct placements, and its generators are all at
  least two hops apart, so there is no clustered case to compare against.

## Figures

| figure | what it shows | finding |
|---|---|---|
| `fig1_propagation.png` | frequency traces: a clustered pocket holding its own frequency, and how much that depends on where the event is | 3 |
| `fig2_maps.png` | RoCoF and frequency-excursion maps, clustered vs dispersed, at identical total inertia | 2 |
| `fig3_ensemble.png` | worst bus against largest converter patch, over 120 random placements plus the named ones | 11 |
| `fig4_measurement.png` | the same solves read eight ways: ratios from 1.04 to 4.37, and the window sweep | 4 |
| `fig5_snsp.png` | worst bus against non-synchronous share, and the headroom clustering costs | 7 |
| `fig6_noise.png` | the everyday regime: the pocket wandering under continuous imbalance | 6 |
| `fig7_shortcuts.png` | the same clustered grid with 48 long lines added out of the block | 10 |

## Files

| file | what it does |
|---|---|
| `swing.py` | the model: lattice, steady state, governor, nonlinear and linearised solvers, metrics |
| `stochastic.py` | continuous random imbalance; exact stationary statistics via Lyapunov |
| `validate.py` | 44 correctness checks — **run this first** |
| `designed.py` | the headline 8×8 comparison, clustered vs spread vs random |
| `snsp.py` | the non-synchronous share sweep and the headroom clustering costs |
| `robustness.py` | sweeps every modelling choice about the default point |
| `experiments.py` | exhaustive 4×4 sweeps over all placements |
| `analysis.py` | statistics for an `experiments.py` sweep |
| `predictor.py` | the neighbourhood-inertia siting metric |
| `selfenergy.py` | integrating out the grid: dressed inertia and frequency-resolved response |
| `figures.py` | the seven figures |

## Running it

```bash
cd local_effects
PY=../repo/grid_TF_Wind/participant-kit/.venv/Scripts/python.exe

$PY validate.py                 # 44 checks, ~3 min
$PY designed.py --random 120    # the headline comparison, ~2 min
$PY snsp.py                     # share sweep and headroom, ~5 min
$PY robustness.py               # the parameter sweeps, ~3 min
$PY figures.py                  # figures/*.png

# exhaustive 4x4 sweeps
$PY experiments.py --layout flat --solver linear --kinds load_step \
      --step 0.5 --aggregate --periodic --out results_torus.csv   # 1820 configs, ~2 min
$PY analysis.py --csv results_torus.csv
$PY predictor.py --csv results_torus.csv
$PY experiments.py --out results.csv                              # nonlinear, ~15 min
```

## Where this goes next

- **§3.6, synchronous condenser siting.** Finding 9 makes the case: local RoCoF is
  an inertia problem, not a reserve problem — no amount of droop touches it, while
  inertia at the converter buses cuts it by a third. The predictor (finding 5)
  gives a ranking rule for *where*. The natural next experiment is to add a fixed
  budget of inertia at the top-ranked bus and measure the worst-bus RoCoF it buys
  against placing it at the centre — the paper's own finding is that the
  *periphery* is the effective place, which this setup can now test directly.
- **The real network.** The North West constraint groups are exactly the clustered
  case: a lot of wind in one electrically remote corner. Running the same
  measurement on the kit's `north-west` scope would say whether the ~2× effect
  shows up at realistic topology and inertia levels.
- **The measurement question.** Finding 4 is a concrete argument that a single
  500 ms window from the event is the wrong compliance test for a grid with
  clustered converters, and that is a claim worth putting to EirGrid's own
  methodology.

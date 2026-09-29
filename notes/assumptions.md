# Modelling assumptions and known limitations

Everything below is inherited from upstream (`ciangregg/EIEG_Hackathon26`,
`NI_Grid_Simulator/all_island_annealer_api.py` @ `35da52e`) unless marked
**[ours]**. Decisions agreed with the project owner during the build are marked
**[decision]**.

## Physics and data (inherited)

- **DC load flow.** Lossless, |P| ≈ |S| against `s_nom`. No reactive power, voltage or
  angle-stability limits.
- **N-1 only.** One intact state plus single-branch outages through LODF. No N-2 and no
  busbar or common-mode faults.
- **Frozen 90% security screen.** States are screened once, from the *baseline* frozen
  flows: every (monitor, outage) pair that reaches ≥ 90% in any snapshot, capped at
  800 states. Candidates are only checked against the screened states. We keep 90% for
  compute reasons. The mitigation is the full-network post-relief recheck in
  `src/validate.py`, which checks all branches intact and all branch/outage pairs N-1.
- **Synthetic weather.** SV2024 has no wind/solar availability series, so upstream
  generates one shared wind series, a daylight-limited solar series and lognormal
  per-plant noise. The weather seed and the case-weight sampling seed are the **same
  `seed`** upstream, so "fresh weather" and "fresh simulation" seeds are one knob here.
- **Case weights.** 10,000 draws with replacement from the 168 SV2024 demand snapshots.
- **Dispatch.** Conventional plant runs merit order from p_min. Surplus is curtailed
  pro-rata across all renewables before the network is considered (pre-network
  dispatch-down: 3.42% at seed 42). Curtailed renewables are replaced at the
  balance bus (Great Island 220 kV, the largest conventional unit).
- **Relief model.** A greedy, myopic loop: take the worst screened state, pick the group
  with the smallest required MW, cut it pro-rata, and repeat up to 24 times. When the
  worst state cannot be relieved by any group, **the loop stops**, so every other
  overload in that snapshot is also left unrelieved (see "Upstream behaviour").
- **Not modelled:** inertia, RoCoF, SNSP, reserve, ramping, unit commitment, and any
  priority/firm-access rules in real WDT dispatch. HVDC runs at fixed schedules.
- **Node table.** 174 renewable nodes (26-county), MEC from the Jan-2024 controllability
  report, mapped to 110/220 kV buses. Coordinates are station coordinates, not plant
  centroids.

## Upstream behaviour that differs from the task brief

1. **`make_emulator` does expose the frozen cases** (`dispatch_down.cases`). We still
   rebuild them ourselves as the brief asks. The regression test checks that our
   construction is bit-identical.
2. **Seed coupling:** see "Synthetic weather" above.
3. **The baseline is insecure on about half the snapshots.** At seed 42, only 52.2% of
   weighted snapshots are secure after relief under the exclusive grouping. The cause is
   branch 458, Poolbeg South–Carrickmines PST 220 kV, monitored under the
   Inchicore–Irishtown outage. It is overloaded in the negative direction in about 50% of
   snapshots, with a median excess of 60–75 MW. Curtailing *every* helpful-sign renewable
   in full gives only about 8 MW of relief, so no grouping can fix it.
4. **The greedy break hides other overloads.** Because 458 is so often the worst state,
   Lanesboro–Mullingar (380) is overloaded before relief about twice as often as 458, yet
   it rarely binds.
5. The "three-panel style" plot is in upstream `plot_hist.py`, not in
   `annealer+blackbox_26+32.py`. We reproduce it.

## Our modelling choices

- **[ours] Overlapping membership.** An N×K bool matrix. The pro-rata weights are
  recomputed from current dispatch on every action, so a node in two called groups is
  cut twice from a shrinking base and the clip at 0 holds. This needs no special code.
- **[ours] Uncovered nodes allowed.** There is no coverage constraint. The security guard
  catches any overload left unrelievable.
- **[decision] Baseline = true multi-membership WDT** (`wdt_groups`: 36 groups including
  IE-ALL with 172/174 nodes). It is the validation comparator and the reference for the
  guard. The exclusive scalar grouping is reported alongside. Multi-membership WDT has
  *higher* DD than exclusive (6.43% vs 6.37% at seed 42), because the greedy picks the
  group needing the least MW for the current worst state, and that is myopic.
- **[decision] Anchor ensemble:** seeds {11, 23, 42, 76, 101} × thermal_scale {1.0, 0.95}.
  The same 10 cases are the **training ensemble** for annealing. D(G) is their
  unweighted mean, and the guard applies to every member. Validation uses seeds
  {2001..2005} × {1.0, 0.95}.
- **[ours] Binding definition.** A state binds in a relief iteration if it is the argmax
  loading at the top of the iteration and that loading is > 1. This includes the final
  iteration in which nothing can relieve it. Counts are case-weighted, summed over
  outage variants and over the ensemble.
- **[decision] Relievability screen.** A monitored branch is dropped from p_e if, in most
  of its case-weighted overload events, curtailing all helpful-sign renewables in full
  could not relieve it. This drops 458, 227, 249 and 250.
- **[decision] K cutoff 0.95, not 0.5.** The p_e curve is a cliff: K=2 at 0.5–0.9 and K=3
  at 0.95 (relievable branches, WDT-multi baseline). With the literal counts including
  458, K=1 for every threshold from 0.3 to 0.8. The 0.5 cutoff is arbitrary either way.
  Anchors: 380 Lanesboro–Mullingar, 462 Platin–Oldbridge, 438 Moy–Glenree.
- **[decision] Guard anchor 458.** The K=3 initial grouping broke the guard in 37
  snapshot×member cases, all ending on 458. In marginal snapshots (90–103% loading) the
  baseline does relieve 458, using its 16 Wexford/Wicklow helpful-sign nodes, while all
  of the K=3 groups' MEC is wrong-signed for it. A fourth group anchored to 458
  (orientation −1), seeded with all 16 helpful nodes, restores feasibility. It does
  **not** count toward the p_e/K rule.
- **[decision] Sigma = binding-weighted mean** of node_state_sensitivity over the
  anchor's outage variants. In practice one variant carries all binding for each anchor.
  Signs agree across variants for at least 95% of nodes (380), and fully for 462, 438
  and 458. No anchor overloads in both directions: every anchor is 100/0.
- **[ours] Initialisation:** sigma > 0 and |sigma| above the 75th percentile, per
  anchor. The guard anchor is seeded with all helpful nodes, because the 75th-percentile
  filter leaves one node and fails the guard.
- **[ours] Normalisation:** each term is z-scored as (T − T(G0)) / sd over 64 random
  memberships with G0's per-group density. The probe's D ignores the guard. So E(G0) = 0
  and λ = 1 is one random-spread unit. Division by G0 values was rejected because
  T5(G0) = 0 and T2 < 0. **Caveat:** the probe spread of T4 (Σn²) is small relative to the
  range the annealer can reach by shrinking groups, so T4 can dominate E (see findings).
- **Soft wrong-sign penalty.** T5 is a weighted term, not a hard reject, as the brief
  asked.
- **Security guard kept hard (+inf).** It only penalises *new* insecurity. **D(G)
  rewards giving up:** an overload that is not relieved costs no dispatch-down. Security
  pass % is therefore reported next to every D.
- **[ours] Annealer.** Move = flip M[i,k] with p = 0.5, otherwise move a node k1→k2.
  Moves that would empty a group are redrawn. T is calibrated as upstream: median |dE|
  at acceptance 0.5, 5th percentile at 1e-3, from 40 probe moves. Cooling α = 0.98 with
  25 moves per temperature.

## Known limitations

- The 90% frozen screen (see above). Unscreened overloads are reported, not optimised
  against.
- Synthetic weather. 10 training cases, 10 validation cases.
- DC, N-1 only, no inertia/reserve/ramping.
- A soft wrong-sign penalty, not a hard one.
- The 0.5 anchor cutoff is arbitrary. We use 0.95, and the curve is a cliff.
- One annealing chain per configuration (seed 0). No restarts, so run-to-run variance
  is not quantified.
- 26-county only so far.

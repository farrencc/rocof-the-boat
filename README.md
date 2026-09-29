# rocof-the-boat: structural Hamiltonian for constraint-group generation (PoC)

This is a proof of concept. It replaces upstream's exclusive, size-conserving
constraint-group annealer with one that works on an **N×K overlapping membership
matrix**. Each group is **anchored** to a frequently binding monitored branch, and the
energy is E = λ·(dispatch-down, mean oriented sensitivity, sensitivity variance, Σn²,
wrong-sign hinge). The upstream hard security guard is kept.

Upstream ([ciangregg/EIEG_Hackathon26](https://github.com/ciangregg/EIEG_Hackathon26),
no LICENSE) is **never copied**. It is cloned into the gitignored `external/` folder and
imported only by `src/upstream.py`.

## v2 model (current): conventional plant, SNSP, fairness, 11 anchor lines

v2 answers four review points:
1. **Nodes are real farms.** Each of the 174 renewable nodes is one wind or solar farm
   (sometimes one phase of a farm), not an average. v2 adds the 14 conventional units as
   group members. They act by turning **down** to p_min, and that redispatch is priced
   by a separate term T6 (λ_C), not counted as dispatch-down. SNSP is capped at 75%
   before the network is checked; see the finding below. Fairness comes from λ_N = 0.1
   plus a **hard minimum of 5 nodes per group**.
2. **Branch 458 (Poolbeg S – Carrickmines) is explained** (`figures/v2/fig1`). The
   overload is caused by demand pulling power into Dublin. With only wind and solar
   allowed to act, 0% of its overloads can be relieved. With conventional plant it is
   88%, and 100% with a distributed slack. The Carrickmines phase-shifter is fixed at 0°
   in the data, and the 26-county file lacks most Dublin generation.
3. **11 anchor lines** from a stressed ensemble (ratings 100/95/90%), each with its own
   group (`fig3`, `fig4`).
4. **New figures** are in `figures/v2/`, explained in [`docs/figures.md`](docs/figures.md).

**Headline (validation, fresh seeds; config `d13_P2`, chosen for lowest D + C with no new
failures):**

| grouping | renewable DD % | conv. redispatch % | secure snapshots % | new failures | unscreened >100% pairs (max) |
|---|---|---|---|---|---|
| WDT today (multi-membership) | 6.80 | 0 | 49.3 | — | 10 (108%) |
| initial anchored groups | 6.87 | 5.89 | 82.8 | 0 | 30 (112%) |
| **optimised `d13_P2`** | **4.88 (−28%)** | 6.24 | **82.1** | **0** | 15 (113%) |
| renewable-DD-only λ | 4.50 | 10.42 | 68.3 | 1 | 23 (112%) |

- **Renewable dispatch-down falls 28%, and security rises from 49% to 82% of snapshots.**
- **The total MW moved (DD + C) goes up, from 6.8% to 11.1%.** Most of the extra
  redispatch pays for relieving overloads that WDT leaves unrelieved: WDT has no
  conventional plant in its groups and gives up on half the snapshots. There is no
  like-for-like secure baseline, so read D and C together, never D alone.
- **Two things were found and fixed on the way.**
  - **Balance-bus replacement:** replacing everything at Great Island exceeded its
    464.5 MW rating in about 45% of snapshots and overloaded its exit lines (up to
    167%). That run is archived in `results/v2_26_balance_slack/`. v2 now shares the
    replacement across all conventional units by capacity (the distributed slack).
  - **Anchor orientation:** anchors that never bind now take their overload direction
    from pre-relief overloads.
- **Open issues.**
  - **The SNSP cap never binds:** synthetic SNSP peaks at 62.8%.
  - **Constraint generation is deferred:** it was requested for later. The frozen 90%
    screen still misses a few overloads (15 pairs, up to 113%, for the headline).

Reproduce v2 (after the setup below):

```bash
python src/sweep.py --version v2 --scope 26 --alpha 0.98 --sweeps 25   # 15 configs, ~13 min on 4 cores
python src/validate.py --version v2 --scope 26                          # ~3 min
python src/report.py --scope 26                                         # figures/v2/
```

The v1 write-up below is kept for reference.

## Reproduce from a clean clone

```bash
pip install numpy scipy pandas h5py matplotlib numba pytest
./scripts/setup_upstream.sh            # shallow clone -> external/EIEG_Hackathon26
python -m pytest -q tests              # gate: evaluator == make_emulator (bit-identical)
python src/sweep.py --scope 26         # 18 lambda configs, 4 workers, ~5 min; checkpoints + commits per config
python src/validate.py --scope 26      # fresh seeds, full-network recheck, figures/  (~2 min)
```

`sweep.py --no-commit` skips the git commits. Delete `results/<config_id>/` to re-run
one config. An interrupted config resumes from `results/<config_id>/checkpoint.npz`;
we verified that a kill mid-run followed by a resume reproduces the uninterrupted run
exactly.

| file | role |
|---|---|
| `src/upstream.py` | the only importer of `external/`; rebuilds the grid and frozen cases as `make_emulator` does |
| `src/evaluator.py` | numba reimplementation of `evaluate_assignment` for N×K membership; ensemble evaluator |
| `src/anchors.py` | ensemble binding statistics, relievability screen, p_e, K, orientation, sigma, initial groups |
| `src/hamiltonian.py` | T1–T5, z-score normalisation, security guard |
| `src/anneal.py` | Metropolis loop with flip/move moves, upstream temperature calibration, checkpoint/resume |
| `src/sweep.py`, `src/validate.py`, `src/plots.py` | sweep, validation, figures |
| `results/setup_26/` | anchors (`anchors.json`, `branch_binding.csv`) and normaliser |
| `results/<config>/` | `history.csv`, `summary.json`, `final.npz`, energy trace, three-panel diagnostics |
| `results/validation_26/` | `validation_summary.csv`, `new_security_failures.csv`, `full_network_unscreened_overloads.csv` |
| `figures/` | p_e/K, Pareto, validation DD, maps, sigma distributions |

`notes/assumptions.md` lists every modelling assumption and decision, and the places
where upstream behaves differently from the original brief.

## Findings (26-county)

**Regression gate.** With upstream's exclusive `group` column, our evaluator matches
`make_emulator` bit-for-bit (difference 0.0), across three seed/thermal-scale cases and
a set of permuted candidates, including the guard verdict.

**Anchors.** One line dominates, and nothing can relieve it. Branch 458, Poolbeg
South–Carrickmines PST 220 kV (under the Inchicore–Irishtown outage), accounts for 80%
of binding events. It is overloaded in about 50% of snapshots, always in the negative
direction, and no grouping can fix it: all helpful-sign curtailment together gives
about 8 MW against 60–75 MW of excess. It is the reason the baseline is only about 50%
secure.

With the literal counts, K = 1 for every threshold from 0.3 to 0.8: a cliff, not a
plateau. After dropping unrelievable branches, K = 2 from 0.5 to 0.9 and K = 3 at 0.95.
We used 0.95, which gives 380 Lanesboro–Mullingar, 462 Platin–Oldbridge and 438
Moy–Glenree. All three overload in one direction only (100/0), and node signs agree
across outage variants for at least 95% of nodes.

The K = 3 initial grouping broke the guard, every time on marginal 458 overloads.
Adding 458 back as a fourth "guard anchor" restored feasibility.

**Did the structural terms reduce DD or trade against it? They reduced it, and they
beat DD-only annealing.** On fresh validation seeds (5 seeds × thermal scale {1.0, 0.95},
identical frozen cases for every grouping):

| grouping | mean DD % | vs WDT-multi | security % | new insecure snapshots vs WDT |
|---|---|---|---|---|
| WDT multi-membership (baseline) | 6.42 | — | 49.35 | — |
| WDT exclusive (upstream) | 6.25 | −0.16 pp | 49.75 | 0 |
| initial shift-factor groups | 6.48 | +0.06 pp | 49.56 | 0 |
| DD-only annealing (λ = 1,0,0,0,0) | 6.04 | −0.37 pp (−5.8%) | 49.34 | 3 |
| **structural-only (λ = 0,1,1,1,1)** | **5.61** | **−0.81 pp (−12.7%)** | 49.47 | 1 |

Every one of the 18 configurations improved on WDT out of sample, by 7.8–12.7%. The
improvement held in all 10 validation cases, by 0.70–1.00 pp each
(`figures/validation_dd_26.png`). Security stays at the baseline level, so the saving
does **not** come from giving up on overloads. The Pareto plot shows no trade-off: lower
T2, T4 and T5 go with lower D. The DD-only run sits in the worst corner.

The reason is not that structure is "better physics". **T4 (Σn²) dominates E**, because
its random-probe spread is small compared with how far the annealer can shrink groups.
Shrinking groups to a few high-sensitivity nodes is also what makes the greedy relief
cheap: the required MW scales with 1/|S_g|, and pro-rata cuts stop landing on
low-sensitivity members. DD-only annealing, driven by a noisy and nearly flat D, never
found that region within 5.7k calls.

**Caveats. Read these before trusting the headline.**
- **The optimised groups cover only 15–31 of 174 nodes,** with 1-node groups for
  462 and 438. That is operationally implausible and probably fragile. Nothing
  rewards coverage, as the brief intended.
- **No annealed grouping is guard-clean out of sample.** Each creates 1–3 new insecure
  snapshots on the validation seeds, all on 458 at 100.04–101.3% loading, each about
  0.5–0.7% of one case's weight. The guard held on the training ensemble only.
- **Full-network recheck:** the optimised groupings produce 0–9 unscreened N-1 overloads
  (up to 107.7%). The WDT baseline produces 11 (up to 109.7%), and the optimised ones
  are almost entirely a subset of the baseline's (`unscreened_new_vs_wdt` = 0 for all
  bestE groupings). So anchoring did not create new invisible overloads, but the frozen
  90% screen already misses about 10% overloads even at baseline. This is a known
  limitation of the upstream screen.
- One annealing chain per configuration, with a short schedule (α = 0.98, 25 moves per
  temperature, about 5.7k calls at about 22 ms each). Configs c02 and c13 are the same
  λ up to a factor of 4 and give identical runs, as expected, since T is calibrated
  from dE. That confirms the scale invariance but also shows that only λ *ratios*
  matter.

**Runtime:** the whole 18-config sweep ran in 4.7 min on 4 cores, and validation in
about 2 min. There is plenty of room to expand the λ grid, add restarts, or lengthen
the schedule.

32-county has not been attempted yet. The 26-county scope now runs end to end.

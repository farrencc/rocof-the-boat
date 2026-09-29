## Findings, in order of importance

1. **For most large countries the minimum-energy map is not identifiable.** In the
   headline configuration, the best and second-best of 12 restarts differ structurally
   (ARI 0.12–0.33 for FR, ES, DE, IT, PL, PT) at energy gaps of 0.2–8.6 on energies of
   several hundred. The convergence check makes this sharper. Annealing 4× longer with 24
   restarts lowers the minimum by only 0.1–1 %, yet it finds a **different** map for FR, ES,
   DE, IT, PL and SE (ARI 0.10–0.29 against the sweep's map). Only NO, BG and CH return
   essentially the same map (ARI ≥ 0.85). This is degeneracy of the energy landscape: many
   structurally different partitions lie within ~1 % of the best energy found. The maps for
   these countries should be read as *one* low-energy member of a large near-degenerate set,
   not as *the* partition. Small countries (≤ ~40 buses) are reproducible (ARI 1.0 across
   restarts).
2. **α brackets two failure modes, and no value of α avoids both.** With both terms
   normalised to mean 1, the Potts weight w = Δp̃ − αJ̃ is repulsive on 49 % of edges at α = 1
   (median over large countries), and on 65 % at α = 0.25. The minimisers are then contiguous
   but interdigitated (cut ratio 0.14–0.16). As α grows the zones compact: repulsive share
   0.17 and cut ratio 0.07 at α = 5. But at α = 10 almost everything is attractive, and with
   k fixed the cheapest solution carves off **microzones** (FR 99.7 % / 0.2 % / 0.1 % of load;
   only 71 of 90 zones above the 5 % floor). The small balance term (λ_b = 0.5, deliberately
   kept small) does not prevent this. The headline α = 2 is the largest value on the grid at
   which ≥ 95 % of zones clear the floor (86 of 90). That criterion was chosen after seeing
   the sweep and is stated as such. The α = 1 baseline map is in `figures/europe/`.
3. **Maps are sensitive to every parameter.** Median ARI against the α = 1 map is 0.07–0.34
   across the α grid for large countries (`sensitivity_alpha_ari.png`). Given (1), part of
   this is restart noise rather than a response to the parameter.
4. **k carries no information from the spectrum.** No country's Laplacian eigengap passes a
   Poisson-spacing null test (min p = 0.10, SI), so every k is the fallback (3, or 2 for
   the smallest countries). The Δk sweep (k = 2..5) is in `results/sweep.csv`.
5. **Why the landscape is glassy.** In a meshed DC network, a congested corridor creates
   price *gradients* over a wide area rather than a step at the corridor. Every edge in that
   area shows persistent |Δp| > €1/MWh (duration statistic), so the congestion signal does
   not pick out a boundary. It rewards cutting anywhere in the region, and the fixed-k Potts
   energy then has many equivalent cuts. This is a property of using edge-wise nodal price
   differences as the splitting signal, which the proxy limitation compounds. It is the
   main methodological lesson of this PoC.
6. **Load pockets.** The 220 kV truncation leaves chronic shedding pockets (Munich, Cádiz,
   Rennes, Uppsala, Oslo; `reports/solve.md`). Zone boundaries next to them should be treated
   as artefacts until the network includes sub-transmission.

All 17 configurations × 31 countries end with contiguous zones (HVDC counted as adjacency;
λ_c ramped to a value that guarantees it).

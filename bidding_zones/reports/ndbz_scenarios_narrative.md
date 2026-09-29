Written after inspecting the numbers above. This is the step-1 checkpoint: nothing has
been annealed yet.

**1. The clip has not eaten the signal, except in three countries.** In every scenario,
under 1 % of bus-hours sit at the clip in 24 of the 31 cluster-countries. The clip is
concentrated where the solve sheds chronically: FR (13 % of bus-hours in `peak_demand`),
NO (9 %), SE (7 %), IE (3.5 %). There it bites the way the brief warned, in the
direction that makes the extreme scenarios look *calmer*.

- With the unclipped prices, the FR / NO / SE country-mean duration rises by 15–19 %
  (SE: 7 %) in `peak_demand`, `dunkelflaute` and `max_dispersion`.
- The per-edge ranking changes: Spearman ρ falls to 0.69–0.87.
- `exclude_clipped` barely moves them. Dropping the tied hours does not recover the
  gaps that the clip hid; only the unclipped prices do.
- IE `dunkelflaute` is the one case where `exclude_clipped` is material (−21 %). There
  the unclipped statistic agrees with `as_is` to within 3 %.
- NO is material even on the full year (`baseline`, +14 %). The clip already shaped the
  static map A there.

**Conclusions for FR, NO and SE are provisional under `as_is`.** No re-solve is needed
to settle them: `prices.parquet` holds the unclipped duals. Recommendation: run the
re-zoning for those three countries under both `as_is` and `unclipped` and report both.
Alternatively, make `unclipped` the scenario default and keep `as_is` as the check. With
the duration statistic, the clip only hides gaps (it never creates them). The static
pipeline adopted the clip to protect the *mean* statistic from 3000+ EUR/MWh shedding
prices. So `unclipped` is safe for `duration` but not for `mean`.

**2. Severity survives under baseline normalisation, and it is not uniform.**

- `peak_demand` and `max_dispersion` are more congested than the year almost everywhere:
  mean dp̃ is 1.1–1.7 in the large countries and 1.7 / 2.2 in IE.
- `dunkelflaute` is *less* congested in DE (0.63), PL (0.72) and IE (0.45). With no
  wind, the wind-export corridors are not loaded, so a still-and-cold hour is a
  low-congestion hour for those grids.
- `wind_surplus` is where IE (2.1), PL (2.2) and DE (1.4) are stressed, and ES is calmer
  (0.85).
- Under `self`, all of this is divided out by construction (mean 1).

**3. Decision needed: the reference-year Δp cap pins many edges in extreme scenarios.**
As specified, `baseline` normalisation takes the `clip` remedy's threshold (the
reference 99th percentile) from the full year. In a scenario more congested than the
year, every edge above that threshold is set to the same value:

- 45–50 % of NO's edges and 61–64 % of SE's are pinned in `peak_demand` /
  `dunkelflaute` / `max_dispersion`.
- IE: 35 % in `peak_demand`, 67–73 % in `wind_surplus` / `max_dispersion`.
- PL: 44–51 % in `wind_surplus` / `max_dispersion`.

Tied edges carry the same w_ij, so for them the physical term only says "congested";
it no longer says *which* boundary is more congested. The re-zoning signal is
flattened there, not only rescaled.

The diagnostic variant (own 99th-percentile clip, reference mean) avoids the ties, but
has the opposite failure. In countries with almost no full-year signal (BG, GR, LT, LV,
MK, AL, XK), the reference mean is tiny and dp̃ reaches ~40. Such a scenario would
overwhelm the rigidity term there on the strength of a few congested hours.

Options, in the order I would recommend them:

- **(a)** Own-quantile clip with the reference mean, restricted to countries that have a
  full-year congestion signal. Countries without one are skipped, as the task already
  plans for "no congestion signal".
- **(b)** Keep the reference cap, as specified, and report the pinned share next to
  every result.
- **(c)** `self` normalisation, which gives up severity.

**4. Sample size is adequate wherever there is a signal.**

- `q = 1 %` is 88 h, so it is widened to 200 h (2.3 %) everywhere.
- The dunkelflaute quantiles widen to at most a VRE 20 % / load 43 % intersection.
- Every (scenario, country) pair flagged thin is a country with no or negligible
  congestion signal (EE, LT, LV, MK, AL, BG, GR, XK), plus SI `peak_demand` and BA
  `wind_surplus`. For those, the between-edge spread is itself ≈ 0.
- For the large countries and IE, the bootstrap CI of the country mean is narrow. The
  median per-edge SE is ≤ 0.11 of the between-edge spread: the edges are well ranked on
  200 h.

**5. Country and europe scope ask different questions.**

- For `peak_demand`, the hour sets overlap moderately (Jaccard 0.3–0.5 in most large
  countries; IT 0.08, PT 0.14).
- For `wind_surplus`, `dunkelflaute` and `max_dispersion`, the overlap is mostly
  0.0–0.2. Europe's windiest hours are not Italy's or Norway's.
- The default stays `country`. Both scopes are persisted in `results/ndbz/scenarios/`.

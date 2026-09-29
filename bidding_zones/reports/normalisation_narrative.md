## Finding and decision

**Statistic.** The default Δp statistic is **duration**: the weighted share of hours with
|Δp| > €1/MWh. Art. 14 asks for *structural*, persistent congestion, and a mean conflates
that with rare spikes. The data support the choice. Across countries the duration statistic
has median skew 1.1 and a median top-1 % share of 0.10, whereas the weighted mean and the q90
have skew up to 17.5 and 16.1: the pocket-driven price spikes dominate them. Duration does
not saturate: at most 15 % of a country's edges exceed 0.9 (AT), and most countries have
none.

**Δp̃ outliers.** Three countries have Δp̃ concentrated on a handful of lines: **BG** (88 %
of edges at zero, one line at 24× the mean, top-1 % share 0.32), **GR** (77 % zero,
top-1 % 0.44) and **DK** (51 % zero, top-1 % 0.39). Here the price signal really is carried
by a few congested lines. That is a property of these countries' nodal prices, not a
numerical artefact, so no transformation can "fix" it without inventing signal. Their
partitions will be driven by those few lines, and that is what the result will say. As a
guard against single-edge dominance, Δp̃ is clipped at each country's 99th percentile before
normalising (`dp_remedy: clip`). This changes at most the top 1 % of edges and does not
change the concentration. **EE has no intra-country price divergence at all** (every edge
zero): it has no congestion signal and its partition is driven by J̃ and the balance term only.

**J̃ outliers — remedy applied.** Raw susceptance is dominated by outliers, as expected,
because reactance spans three orders of magnitude (short lines have tiny x). Raw skew is up
to 10.7 (FR, max 46× the mean), and in small countries the top 1 % of lines (often a single
line) carries 50–75 % of the total (BA, LT, SI, XK). Left like that, the energy would be
driven by a few short lines. **Remedy: `J_remedy: log1p`**, J → log(1 + J / median J), then
normalised by the country mean. Across countries this brings the maximum skew to 2.2 and the
top-1 % share of large countries (≥ 100 edges) to ≤ 0.075, while keeping both the ordering
and relative magnitudes. Clipping at q99 (max skew 5.9) was not enough. A rank transform
(uniform, skew 0) discards all magnitude information and was rejected.

**What remains true after the remedy, and matters for interpretation.** With both terms
normalised to mean 1, the median edge still has w = Δp̃ − αJ̃ > 0 in **45 % of edges at α = 1**
(country median), and 26 % at α = 3. So at α ≈ 1 the Potts energy is *repulsive* on almost
half the edges. Its minimisers at fixed k are contiguous but tend to maximise cut weight.
The reason is that nodal price differences in a meshed DC network are spatial gradients
around congested corridors, not steps at them. The α sweep is where this shows up (share of
repulsive edges and cut ratio are recorded per configuration).

**Degeneracy of the duals.** Re-solving 500 random hours with a second cost-noise seed moves
0.07 % of bus-hours by more than €0.5/MWh (99th percentile of |Δprice| €0.09/MWh). That is
far below the €1/MWh duration threshold, so dual degeneracy does not affect the statistic
used. 25 % of those hours have at least one bus that moves.

**Load shedding.** Shedding occurs in 97 % of hours at five chronic load pockets
(`reports/solve.md`), so excluding shedding hours is impossible. Prices are clipped to
±€500/MWh. Under the duration statistic, a pocket bus's incident lines show |Δp| > €1 in
most hours, so partitions may carve out pockets. Wherever that happens it is flagged as an
artefact of the 220 kV truncation.

**Gate status: passed with the remedies above.** Proceeding to the spectrum and the annealer.

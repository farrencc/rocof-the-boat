# Optimal single-edge placement in a second-order Kuramoto network

**Purpose of this document.** A design discussion, worked through to the point where it can be implemented. It records the physics, the derivations, the objective-function choice, and — importantly — several dead ends and degeneracies that were identified along the way. An implementing agent should read the "Traps" sections before writing code; a naive implementation will silently optimise the wrong thing.

---

## 1. Problem statement

We have a network of coupled oscillators obeying the second-order (inertial) Kuramoto / swing equation

$$m_i\,\ddot\theta_i + \gamma_i\,\dot\theta_i = P_i + \sum_j K_{ij}\,(\theta_j - \theta_i),$$

with $K_{ij} = K_{ji} \ge 0$ given.

**Decision variable.** We may add exactly one new edge $(a,b)$ of *fixed* strength $\kappa$, i.e. set $K_{ab} = K_{ba} = \kappa$ for one pair currently unconnected (or increment an existing pair, if that is admissible). There are $O(N^2)$ candidates.

**Objectives.** Against perturbations $\Delta P$ applied at arbitrary locations, minimise

1. the maximum frequency deviation $\max_{i,t}|\Delta\dot\theta_i|$, and
2. the maximum angular acceleration $\max_{i,t}|\ddot\theta_i|$,

both taken across the whole network.

---

## 2. Linearised model and the Green's function

Linearise about a stable synchronous state $\theta^*$ in the rotating frame. Writing $\theta \to \theta^* + \Delta\theta$:

$$M\,\Delta\ddot\theta + \Gamma\,\Delta\dot\theta + L\,\Delta\theta = \Delta P,$$

with $M = \mathrm{diag}(m_i)$, $\Gamma = \mathrm{diag}(\gamma_i)$, and $L = D - \tilde K$ the weighted graph Laplacian.

> **Important.** The correct entries are the *synchronising coefficients* at the operating point,
> $$\tilde K_{ij} = K_{ij}\cos(\theta^*_j - \theta^*_i),$$
> not the bare $K_{ij}$. See §9 — adding an edge moves $\theta^*$, so the update is not purely additive in the nonlinear model.

In Laplace space, with $A(s) = Ms^2 + \Gamma s + L$,

$$\Delta\theta(s) = G(s)\,\Delta P(s), \qquad G(s) = A(s)^{-1}.$$

$G$ is the dynamical Green's function. Frequency response is $s\,G(s)$; acceleration response is $s^2 G(s)$. On the imaginary axis $s = i\omega$, the entry $G_{ij}(i\omega)$ is the complex transfer function from a disturbance at $j$ to the phase at $i$: injecting $\Delta P_j(t) = \varepsilon\cos\omega t$ gives, in steady state,

$$\Delta\theta_i(t) = \varepsilon\,|G_{ij}(i\omega)|\,\cos\!\big(\omega t + \arg G_{ij}(i\omega)\big).$$

**Notation.** $G_{ij}$ *with indices* is a scalar entry, and $|G_{ij}|$ is a complex modulus. A determinant is always written $\det(\cdot)$ explicitly in this document to avoid the collision with $|A|$ for matrices.

---

## 3. Schur complements and self-energies

### 3.1 Single node

Partition node $i$ against the rest $r$:

$$S_i(s) = m_is^2 + \gamma_i s + L_{ii} - \underbrace{L_{ir}A_{rr}(s)^{-1}L_{ri}}_{\textstyle \Sigma_i(s)}, \qquad G_{ii}(s) = \frac{1}{S_i(s)}.$$

$\Sigma_i$ is a genuine self-energy in the QFT sense: it is what remains after integrating out the rest of the network, it is frequency dependent, and it is complex on the imaginary axis.

### 3.2 Block generalisation — this is the right object for $(i,j)$

For any target set $T$ with complement $r$,

$$S_T(s) = A_{TT}(s) - A_{Tr}(s)A_{rr}(s)^{-1}A_{rT}(s), \qquad G_{TT}(s) = S_T(s)^{-1},$$

exactly (the block of the inverse equals the inverse of the block Schur complement). Since $A_{Tr} = L_{Tr} = -\tilde K_{Tr}$,

$$\Sigma_T(s) = \tilde K_{Tr}\,A_{rr}(s)^{-1}\,\tilde K_{rT},$$

a $|T|\times|T|$ complex matrix. For $T = \{i,j\}$ its off-diagonal entry gives the **renormalised coupling**

$$\tilde K^{\rm eff}_{ij}(s) = \tilde K_{ij} + \Sigma_{ij}(s),$$

frequency dependent and complex even for real static bare coupling. So: yes, the Schur complement is the correct tool for effective $(i,j)$ coupling; the generalisation is a matrix-valued self-energy, not a scalar one.

### 3.3 Consistency of nested elimination (Dyson-like identity)

Write $S_T = \begin{pmatrix} a & b\\ b & d\end{pmatrix}$ with $a = m_is^2+\gamma_is+L_{ii}-\Sigma_{ii}$, $d$ likewise at $j$, and $b = -(\tilde K_{ij}+\Sigma_{ij})$. Then

$$G_{ii} = \frac{d}{\det S_T} = \frac{1}{a - b^2/d}, \qquad G_{ij} = \frac{-b}{\det S_T}.$$

By the quotient property (Crabtree–Haynsworth), the one-shot single-node reduction must agree, giving

$$\boxed{\;\Sigma_i(s) = \Sigma_{ii}(s) + \frac{\big(\tilde K_{ij}+\Sigma_{ij}(s)\big)^2}{S_{jj}(s)}\;}$$

The total self-energy at $i$ splits into "everything not through $j$" plus a $j$-channel weighted by the *renormalised* coupling squared over $j$'s own inverse propagator. **Use this as a unit test** (§10).

### 3.4 Static condensation — what "effective mass between $i$ and $j$" means

Expand $A_{rr}^{-1}$ about $s=0$. With $L_{rr}$ the grounded (Dirichlet) Laplacian, which is nonsingular once $T$ is removed, define the **harmonic weight vectors**

$$w^{(i)} = L_{rr}^{-1}\tilde K_{ri},$$

i.e. the potential on the bath when node $i$ is held at 1 and the other target node at 0. For $T=\{i,j\}$, $w^{(i)}+w^{(j)}=\mathbb 1$. Then $S_T(s) \approx M^{\rm eff}s^2 + \Gamma^{\rm eff}s + L^{\rm eff}$ with

$$M^{\rm eff}_{ij} = \sum_{r} m_r\,w^{(i)}_r w^{(j)}_r + O(\Gamma^2), \qquad \Gamma^{\rm eff}_{ij} = \sum_r \gamma_r\,w^{(i)}_r w^{(j)}_r,$$
$$M^{\rm eff}_{ii} = m_i + \sum_r m_r\big(w^{(i)}_r\big)^2 .$$

This is Guyan / Craig–Bampton condensation; the $w$'s are the constraint modes. Sanity check: for $T=\{i\}$, $w^{(i)}=\mathbb 1$ and the added mass is the total bath mass.

> **Trap.** This Taylor expansion has a finite radius of convergence, set by the lowest eigenvalue of the grounded pencil $(L_{rr}, M_r)$ — the first Dirichlet mode of the bath. In a realistic network that pole lies *inside* the band of interest. Do not treat "the $s^2$ prefactor" as a single band-wide number. Evaluate $\Sigma_T(i\omega)$ pointwise on the frequency grid instead.

---

## 4. Traps: why "maximise the minimum effective mass" is the wrong objective

The original proposal was to maximise $\min_i m^{\rm eff}_i$. The machinery is right; this particular scalar is not. Four distinct reasons, all of which an implementation must avoid re-introducing.

**(a) Mass and damping renormalise together and are not separately observable.** Only the total $S_i(i\omega)$ — one complex number per frequency — is physical. Splitting it into $m^{\rm eff}s^2+\gamma^{\rm eff}s+k^{\rm eff}$ is a choice. Two masses coupled by $k$:

$$S_1(s) = m_1s^2+\gamma_1s+k-\frac{k^2}{m_2s^2+\gamma_2s+k} \xrightarrow{s\to0} (m_1+m_2)s^2+(\gamma_1+\gamma_2)s+O(s^3).$$

Mass *and* damping both rise, because at low frequency the pair moves as one lump. Scoring on mass alone discards the damping information, which is the more important of the two for peak frequency deviation.

**(b) Mass does not monotonically help the stated objectives.** For a single DOF, $H(\omega)=1/(-m\omega^2+i\gamma\omega+k)$, $\omega_r=\sqrt{k/m}$:

| quantity | scaling | effect of increasing $m$ |
|---|---|---|
| peak phase deviation | $\sqrt{m/k}/\gamma$ | **worse** |
| peak frequency deviation | $1/\gamma$ | **neutral** |
| peak acceleration | $\sqrt{k/m}/\gamma$ | better |

Adding mass also *lowers* $\omega_r$, which can drag a resonance into a band where the disturbance actually has power.

**(c) The diagonal alone cannot express a min–max over disturbance location.** $1/S_i = G_{ii}$ is the response at $i$ to a disturbance at $i$. The stated objective ranges over all source sites, which is controlled by the whole of $G$. A node with huge $m^{\rm eff}_i$ can still be driven hard through $G_{ij}$.

**(d) The sign of the objective flips between diagonal and off-diagonal.** On the diagonal, effective mass sits in a *denominator*: $G_{ii}=1/S_{ii}$, so inflating it shrinks the response. Off-diagonal there is no denominator to inflate —

$$G_{ij} = \frac{\tilde K_{ij}+\Sigma_{ij}}{\det S_T},$$

the effective coupling is in the **numerator**. Large $M^{\rm eff}_{ij}$ means a kick at $j$ drags $i$ along inertially, which is exactly the transmission we want to suppress. A rule of "maximise the minimum $s^2$ prefactor over all entries" would actively select the worst edges off-diagonal.

The genuine physical tension: linking $i$ to a heavy, well-damped region raises $\Sigma_{ii}$ (good) *and* raises $\Sigma_{ij}$ to nodes in that region (bad). The determinant structure balances the two. This is why the score must be $|G_{ij}|$ itself, not any single Taylor coefficient of $\Sigma$.

Additional hazard: $\Sigma_i(s)$ has poles at the eigenvalues of the grounded subsystem. Near such a pole $|S_i|\to\infty$ and any extracted $m^{\rm eff}$ blows up or changes sign (the negative-effective-mass phenomenon of mass-in-mass metamaterials). An optimiser maximising $m^{\rm eff}$ will chase these narrowband antiresonances.

---

## 5. Traps: two structural degeneracies to check before optimising

**(a) Steady-state frequency offset is topology-independent.** $L$ is singular with $\mathbb 1$ in its kernel, so $S_i(0)=0$ and

$$\lim_{t\to\infty}\Delta\dot\theta = \frac{\sum_j \Delta P_j}{\sum_j \gamma_j}$$

for *any* connected topology. No added edge changes it. Only the transient — nadir, overshoot, damping of inter-area modes — is available to optimise.

**(b) For step disturbances, peak acceleration is also topology-independent.** From equilibrium, $\Delta\ddot\theta(0^+) = M^{-1}\Delta P$: initial RoCoF sees bare inertia only, since $\Sigma_i(s) = O(s^{-2})$ at large $s$ and $m^{\rm eff}\to m_i$. For a step at node $j$ the global max over $i,t$ is generically $\Delta P_j/m_j$ at $t=0^+$.

**Consequence for implementation:** objective 2 is degenerate unless you either (i) restrict to band-limited / smooth disturbances, or (ii) use a windowed RoCoF (power-systems convention: average over $\approx 500$ ms). **Decide which, and record the choice**, before spending compute. The band-limited route is naturally handled by restricting the frequency grid $[\omega_1,\omega_2]$; the windowed route means multiplying $s^2G$ by the transfer function of the averaging window.

---

## 6. Rank-one update: how to screen all candidate edges cheaply

Adding edge $(a,b)$ of strength $\kappa$ is a rank-one update of $L$ and hence of $A$:

$$A \to A + \kappa\,u u^\top, \qquad u = e_a - e_b.$$

Sherman–Morrison gives, exactly,

$$G_{\rm new}(s) = G(s) - \kappa\,\frac{\big(G(s)u\big)\big(u^\top G(s)\big)}{1 + \kappa\,z_{ab}(s)}, \qquad z_{ab}(s) = u^\top G(s)\,u,$$

where $z_{ab}$ is a complex frequency-dependent effective impedance between $a$ and $b$ — the dynamical generalisation of effective resistance. Because $u = e_a-e_b$:

- $G u = G_{:,a} - G_{:,b}$ (difference of two columns, already available),
- $u^\top G = G_{a,:} - G_{b,:}$,
- $z_{ab} = G_{aa} - G_{ab} - G_{ba} + G_{bb}$ (four entries).

**One factorisation of $A(i\omega)$ per frequency serves every candidate pair.** This is the key computational win: $O(N^3)$ per $\omega$ rather than $O(N^2)$ separate solves.

> $A(i\omega)$ is complex symmetric (not Hermitian). Use a complex LU / `scipy.linalg.lu_factor` or a complex-symmetric solver; do **not** assume Hermitian and reach for Cholesky.

> Watch $1+\kappa z_{ab}(s)$: near-zero denominators signal that the added edge nearly cancels a mode. Flag these rather than dividing blindly.

---

## 7. Choosing the cost function

### 7.1 The candidates

| Quantity | Meaning | Implied disturbance model |
|---|---|---|
| $\max_{i,j}\lvert G_{ij}\rvert$ | entrywise max-norm | one node perturbed at a time |
| $\max_i\sum_j\lvert G_{ij}\rvert$ | induced $\ell^\infty$, max row sum | each $\Delta P_j$ individually bounded, worst-case phases |
| $\sigma_{\max}(G)$ | induced $\ell^2$ | bounded-energy pattern |
| $\lVert G\rVert_F$ | Frobenius; swept over $\omega$ this is $\mathcal H_2$ | RMS under isotropic random forcing |
| $\det G$ | $=1/\det A$; locates poles | **not a gain at all** |

$\det G$ is a useful stability/resonance diagnostic and a useless objective: it can be tiny while individual entries are huge, and vice versa.

### 7.2 There is no trade-off between single trips and correlated disturbances

Since $G_{ij} = e_i^\top G e_j$ with $\|e_j\|_2=1$,

$$\max_{i,j}|G_{ij}| \;\le\; \sigma_{\max}(G).$$

Single-site disturbances are simply $N$ of the points on the unit sphere that $\sigma_{\max}$ already maximises over. Bounding $\sigma_{\max}$ bounds every single-trip response for free.

### 7.3 Frobenius is a screen, not a criterion

$\|G\|_F^2 = \sum_k\sigma_k^2$ is an *average* over disturbance directions, so it hedges against nothing in the worst case. With $\sigma_{\max}\le\|G\|_F\le\sqrt r\,\sigma_{\max}$, for $N\sim10^2$ the two can differ by an order of magnitude. Worked counterexample:

- Spectrum A: one mode at gain 10, ninety-nine at gain 1 → $\|G\|_F\approx14.1$. Halving the dangerous mode to 5 improves it by only 21%; killing one benign mode improves it by 0.04%.
- Spectrum B: thirty modes at gain 3, nothing higher → $\|G\|_F\approx16.4$, *worse* than A despite a peak three times smaller.

An optimiser minimising $\|G\|_F$ will accept an edge that sharpens one resonance in exchange for flattening harmless background. Backwards for a robustness criterion — but fine as a cheap first-pass filter.

### 7.4 The objective to actually use

$$J_{\rm freq}(a,b) \;=\; \max_{\omega\in[\omega_1,\omega_2]}\ \sigma_{\max}\!\big(\,\omega\,\Pi\,G_{\rm new}(i\omega)\,B\,\big), \qquad J_{\rm acc}(a,b)\;=\;\max_{\omega}\ \sigma_{\max}\!\big(\,\omega^2\,\Pi\,G_{\rm new}(i\omega)\,B\,\big).$$

Two modifiers, both cheap and both necessary:

**Input restriction $B$.** The worst singular direction of bare $G$ may be a physically impossible pattern (alternating $\pm$ on neighbouring buses). Let $B$'s columns be the disturbance patterns actually credited — individual trips scaled by plant size, a region's load swinging coherently, etc. If a covariance $\Sigma_P$ of plausible patterns is available, $B=\Sigma_P^{1/2}$ interpolates smoothly between "one node at a time" and "worst case over everything".

**Centre-of-inertia projection $\Pi$.** By §5(a) the uniform mode is untouchable, and it otherwise dominates $\sigma_{\max}(\omega G)$ at small $\omega$, adding a near-constant to every candidate's score and compressing the differences we are trying to resolve. Use

$$\Pi = I - \frac{\mathbb 1\,\mathbb 1^\top M}{\mathbb 1^\top M\,\mathbb 1},$$

which measures deviations relative to the centre of inertia (standard in the network-coherence literature).

Also compute $\max_i\sum_j|G_{ij}|$ alongside: if the disturbance model is really "each node's $\Delta P$ is individually bounded" — arguably the more physical box for a grid — that induced $\ell^\infty$ norm is the exact worst case. Non-smooth and more conservative, but free once $G$ is in hand.

---

## 8. Recommended algorithm

The decision variable is a single discrete choice among $\binom N2$ candidates, so there is no need to commit to one scalarisation.

**Stage 0 — setup.**
1. Solve for $\theta^*$, form $\tilde K_{ij}=K_{ij}\cos(\theta_j^*-\theta_i^*)$, build $M,\Gamma,L$.
2. Verify the base case is stable (all roots of $\det A(s)=0$ in the open left half-plane, modulo the $\mathbb 1$ zero mode).
3. Choose the band $[\omega_1,\omega_2]$. Sensible default: from below the slowest inter-area mode to above the fastest mode you care about — get these from the eigenvalues of the quadratic pencil $(M,\Gamma,L)$. Grid logarithmically, then refine adaptively near peaks (a fixed grid *underestimates* an $\mathcal H_\infty$ peak).
4. Define $B$ and $\Pi$. Record the disturbance model and the objective-2 convention from §5(b).

**Stage 1 — screen with $\mathcal H_2$/Frobenius, $O(N^3)$ per $\omega$ for *all* candidates.**
Under the rank-one update, with $c = \kappa/(1+\kappa z_{ab})$,

$$\|G_{\rm new}\|_F^2 = \|G\|_F^2 - 2\,\mathrm{Re}\!\left[c\;u^\top W u\right] + |c|^2\,\big(u^\top P u\big)\big(u^\top Q u\big),$$

with the precomputed matrices

$$W = G\,G^H G, \qquad P = G^H G, \qquad Q = G\,G^H .$$

Because $u=e_a-e_b$, each quadratic form is four matrix entries: $u^\top W u = W_{aa}-W_{ab}-W_{ba}+W_{bb}$, likewise for $P,Q$. So after one $O(N^3)$ precomputation per frequency, **every candidate costs $O(1)$**. Keep the best few hundred.

**Stage 2 — rank the shortlist by $\sigma_{\max}$.**
For each surviving candidate, sweep $\omega$ and evaluate $J_{\rm freq}$ and $J_{\rm acc}$. Never form $G_{\rm new}$ explicitly if avoidable: matrix–vector products with the Sherman–Morrison form are $O(N^2)$, so warm-started power iteration (or `scipy.sparse.linalg.svds` with a `LinearOperator`) gives $\sigma_{\max}$ cheaply. Warm-start from the previous frequency's singular vector.

**Stage 3 — Pareto front, not a blend.**
Plot the surviving candidates in (peak, $\mathcal H_2$) space, and separately for frequency vs acceleration. Pick from the non-dominated set. If the front is narrow, the scalarisation never mattered; if it is wide, that is real information about the network which a single blended cost would have hidden. Only fall back to $\max_\omega\sigma_{\max} + \lambda\|G\|_{\mathcal H_2}$ if the final pick must be automated.

**Stage 4 — explain the winner.**
Reduce onto the two or three nodes carrying the worst entries and report how $M^{\rm eff}$, $\Gamma^{\rm eff}$, $L^{\rm eff}$ (§3.4) moved. This is where the Schur/self-energy picture earns its keep — as diagnosis, not as objective.

**Stage 5 — nonlinear validation.** See §9.

---

## 9. Nonlinear caveats

- Adding a line **moves $\theta^*$**, so $\tilde K \to \tilde K + \kappa\,uu^\top$ is only first-order correct. For the shortlist, re-solve the power flow with the edge present and rebuild $\tilde K$ from the new $\theta^*$ before final ranking.
- **Braess's paradox is real in oscillator networks and in real grids**: adding a line can degrade synchronisation. No monotone "more coupling is better" heuristic is safe — including the effective-mass one. Do not prune candidates on such grounds.
- Simulate the top handful on the full nonlinear model with the actual contingency set before recommending anything.

---

## 10. Unit tests / sanity checks

An implementation should assert all of these:

1. **Two-mass limit.** $N=2$, coupling $k$: check $S_1(s)\to(m_1+m_2)s^2+(\gamma_1+\gamma_2)s+O(s^3)$ as $s\to0$.
2. **Zero mode.** $S_i(0)=0$ for every $i$; $L\mathbb 1 = 0$ to machine precision.
3. **Steady state.** Time-domain simulation reproduces $\Delta\dot\theta_\infty = \sum_j\Delta P_j/\sum_j\gamma_j$, and it is *unchanged* by any added edge (§5a).
4. **Initial RoCoF.** Step at $j$ gives $\ddot\theta_j(0^+)=\Delta P_j/m_j$, unchanged by any added edge (§5b).
5. **Woodbury correctness.** $G_{\rm new}$ from Sherman–Morrison equals $\big(A+\kappa uu^\top\big)^{-1}$ computed directly, for random $(a,b,\kappa,\omega)$.
6. **Quotient property.** $\Sigma_i = \Sigma_{ii} + (\tilde K_{ij}+\Sigma_{ij})^2/S_{jj}$ (§3.3) for random $i,j,\omega$.
7. **Guyan limit.** For $T=\{i\}$, $\sum_r m_r (w^{(i)}_r)^2$ with $w^{(i)}=\mathbb 1$ equals the total bath mass.
8. **Norm ordering.** $\max_{ij}|G_{ij}| \le \sigma_{\max}(G) \le \|G\|_F$ at every evaluated frequency.
9. **Frequency-domain vs time-domain.** For one candidate, compare the predicted peak $|\omega G_{ij}|$ against a direct time-domain simulation with a sinusoidal disturbance at that $\omega$.

---

## 11. Implementation notes

- **Complexity.** Stage 1: $O(N^3)$ per frequency total (all candidates). Stage 2: $O(N^2\times\text{iters})$ per candidate per frequency. For $N\lesssim500$ and a few hundred shortlisted candidates this is comfortable in NumPy/SciPy; beyond that, move the Stage-1 precomputation to GPU and consider a structured $\mathcal H_\infty$ algorithm instead of a frequency sweep.
- **Frequency grid.** A fixed grid systematically underestimates peaks. Use adaptive bisection refinement around local maxima, or the Boyd–Balakrishnan / Bruinsma–Steinbuch level-set method if exactness matters.
- **Conditioning.** Near a Dirichlet pole of the bath (§4) and near $1+\kappa z_{ab}\approx0$ (§6), quantities blow up. Log and flag rather than silently propagating.
- **Complex symmetry.** $A(i\omega)$ is complex symmetric; $\sigma_{\max}$ still requires the SVD (or the Hermitian eigenproblem of $G^HG$), not the eigenvalues of $G$.
- **Suggested stack.** NumPy/SciPy for the core; `python-control` or `slycot` if you want off-the-shelf $\mathcal H_\infty$/$\mathcal H_2$ norms for cross-validation on small cases; `networkx` for graph bookkeeping.

---

## 12. Summary of the design decisions

| Question | Answer |
|---|---|
| Is the Schur-complement / self-energy framework right? | Yes — as machinery, and as post-hoc diagnosis. |
| Use $\max\min m^{\rm eff}$ as objective? | **No.** §4 — mass and damping are inseparable, mass is not monotonically good, the diagonal is insufficient, and the sign flips off-diagonal. |
| Effective $(i,j)$ coupling via Schur? | Yes — block Schur gives a matrix self-energy; $\tilde K^{\rm eff}_{ij}=\tilde K_{ij}+\Sigma_{ij}(s)$. |
| Screen candidates how? | Sherman–Morrison rank-one update; one factorisation per frequency serves all pairs. |
| Cost function? | $\max_\omega\sigma_{\max}(\Omega\,\Pi\,G_{\rm new}B)$ with $\Omega\in\{\omega I,\omega^2I\}$. Frobenius/$\mathcal H_2$ as a fast screen only. $\det G$ never. |
| Single trips vs correlated disturbances? | Not a trade-off; $\sigma_{\max}$ subsumes single trips by construction. |
| Final selection? | Pareto front over (peak, $\mathcal H_2$) and over (frequency, acceleration), then nonlinear validation. |

---

## 13. Pointers to the literature

- Poolla, Bolognani & Dörfler — optimal placement of virtual inertia in power grids.
- Bamieh & Jovanović *et al.* — network coherence, $\mathcal H_2$ norms and their Laplacian-spectral expressions.
- Craig & Bampton; Guyan — component-mode synthesis / static condensation (§3.4).
- Witthaut & Timme; Coletta & Jacquod — Braess's paradox in oscillator networks and power grids.
- Boyd & Balakrishnan; Bruinsma & Steinbuch — level-set algorithms for the $\mathcal H_\infty$ norm.
- Mass-in-mass metamaterial literature — for the negative-effective-mass phenomenon flagged in §4.

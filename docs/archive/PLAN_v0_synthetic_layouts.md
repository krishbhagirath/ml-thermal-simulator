# Project Plan — Physics-Informed Neural Thermal Simulation and Design Optimization

> **Status: DRAFT, awaiting approval.** Nothing here has been implemented yet. The only things measured so far are labelled **[measured locally]**: they come from a throwaway solver prototype run on the development laptop (Windows, 14 logical cores, CPU only). Every other number is an **estimate** and is labelled as one.
>
> Working title only. This is not a final name. The repository is `ml-thermal-simulator` (chosen by the owner) and the Python package is the neutral `thermal_research`.

---

## 0. Executive summary

**What we build**
- **Physics model:** a validated 2D finite-volume model of a thin "package layer". Silicon chiplets (high conductivity) sit in mold compound (low conductivity) above a cold plate that cools non-uniformly, with configurable edge convection.
- **Data and surrogates:** procedurally generated layouts produce leakage-free train, validation, test and out-of-distribution (OOD) datasets. On these we train a CNN, a U-Net, a Fourier Neural Operator (FNO) and a physics-informed FNO.
- **Optimization:** a constrained chiplet-placement problem. Peak temperature is minimised subject to no overlap, minimum spacing, staying inside the domain, fixed power and a wirelength budget. Every final design is re-checked by the numerical solver.

**Primary research contribution (one)**
- *When does a neural thermal surrogate's predicted improvement survive solver verification?* Optimizers push surrogates off their training distribution and exploit their errors.
- We measure that effect (H1). We then compare gate signals that might detect it (H2):
  - the raw FV residual, as already used by DeepOHeat-v1;
  - a conditioning-aware Jacobi/energy residual (one sparse mat-vec);
  - deep-ensemble disagreement (5× inference);
  - conformal calibrations of each.

  We also test a specific prediction: the raw residual degrades as conductivity contrast rises (H2b).
- Finally we test whether the best gate can spend a small budget of solver calls well enough to match solver-only SA or CMA-ES at a fraction of their solver calls (H3).
- Whether physics-informed training helps out-of-distribution, which is the user's question A, is a **secondary ablation** (H4). The literature suggests it gives little gain when labels are plentiful (§2).

**Why it is credible**
- The surrogate learns the solution map of the same discrete operator the solver uses, so the physics loss is exactly consistent with the labels.
- The solver is verified against analytic and manufactured solutions before any model is trained. The prototype already shows observed order 2.00 and energy closure at 1e-14 [measured locally].
- Every claimed improvement is re-measured with the solver.
- Speed claims count the solver calls spent on training data.

**Honest framing risk, stated up front**
- At 128², one direct solve costs about 80 ms on a laptop CPU [measured locally].
- The surrogate's advantage is therefore not raw speed on this toy-scale problem. It comes from three things:
  1. batched, differentiable evaluation;
  2. far fewer solver calls per optimization once the training data is amortised over many design instances;
  3. how solver cost scales with resolution, which we measured locally: 15 ms → 83 ms → 455 ms → 3.2 s for 64² → 512².
- The plan reports the break-even point rather than a headline speedup.

**DGX usage**
- Week 1: environment and storage smoke test, about 15 minutes, no GPU-hours.
- Weeks 3–9: data generation (CPU, under 1 hour) and about 70–150 GPU-hours of training and evaluation in total (estimate). All of it runs on one A100 and can be resumed.

---

## 1. Assumptions (made without asking; any of them can be overridden)

1. **Geometry is snapped to the grid.** Every material and source rectangle has edges on the 128×128 cell faces. Conductivity is therefore exactly piecewise constant per cell, and refining the grid (256, 512) keeps the geometry exactly the same.
2. **Steady state and linear.** Conductivity does not depend on temperature. Temperature-dependent silicon conductivity is a stretch goal.
3. **Thin-layer (compact) 2D model.** Temperature is taken as uniform through the layer thickness. Section 4 explains why this holds in silicon but fails in low-conductivity mold.
4. **One A100 for every essential experiment.** Multi-GPU use is optional and never needed.
5. **The repository is public.** It is cloned on the DGX with `git clone` / `git pull`. Unlike the simulator project there are no secrets, so `scp` is not needed. Data and checkpoints are never committed.
6. **Server environment discrepancy.** The owner's notes list two envs on the lab server: `base` (Python 3.13) and `rl-train` (Python 3.11). The brief says Python 3.12 with torch 2.10.0+cu128. The first smoke test lists the envs so we can tell which one is meant; we do not assume.

---

## 2. Research motivation and literature positioning

Source: a citation-checked review by the research subagent (58 references; each was checked on its abstract or proceedings page, or through index metadata). Items not confirmed that way are listed separately below and are **not** relied on. Every quantitative claim below is the original authors' claim; none has been reproduced by us.

### 2.1 What already exists (so we don't claim it)
- **Neural operators and Darcy-type problems.** Our PDE is structurally a Darcy/diffusion problem with heterogeneous coefficients.
  - FNO [1] introduced spectral-convolution operators. Its Darcy benchmark is only 4:1 contrast with zero-Dirichlet BCs. Ours is up to about 400:1 with Robin edges and a sink term, which is much harder.
  - PDEBench Darcy (128², 10k samples) [4]: the authors write that **U-Net is consistently better than FNO** there. PDEArena [5] also found U-Nets competitive with or better than FNOs (on time-dependent fluids, so only indirect evidence).
  - ConDiff [22]: every model degrades as coefficient contrast rises, and no architecture wins clearly.
  - Lu et al. [11]: on piecewise-constant Darcy, methods end up close to each other, and FNO degrades on complex geometry.
  - **Implication:** the FNO is our required neural-operator model, but we do **not** assume it wins. The U-Net is an equally serious contender, and the primary contribution uses whichever backbone is best on validation.
- **Physics-informed operator learning.**
  - PINO's [2] large Darcy gains come mainly from imposing the PDE at a *finer* resolution than the data. At matched resolution its gains are small.
  - Liu et al. [16] report "marginal and often unstable" gains from physics-informed variants over data-driven training.
  - FC-PINO [8] shows spectral differentiation fails on non-periodic, non-smooth problems. **That supports our choice to use the solver's own discrete FV operator instead of FFT derivatives.**
  - Label-free physics-constrained CNNs exist, including for heat-source layouts [17, 19].
  - **Implication:** physics loss (the user's question A) is a weak *primary* contribution. It stays as an ablation (H4).
- **Heat-source-layout surrogates and optimization.** There is a whole line of work on 2D steady conduction, rectangular sources, CNN surrogates and layout search: the HSL-TFP benchmark [28] (200², 12 sources, *constant k*) and surrogate-assisted search [27, 29, 30]. Ref [30] in particular updates the surrogate online, because offline surrogates misdirect the search.
  - **Implication:** "surrogate-guided thermal layout optimization" (question C) is crowded and **not novel alone**.
- **Thermal-aware chiplet placement.**
  - Our optimization task is essentially **TAP-2.5D's formulation** [31]: simulated annealing with HotSpot over spacing and wirelength. It has open-source code at github.com/bu-icsg/TAP-2.5D.
  - Others: RLPlanner [32], an RBF-surrogate + SA placer [33], ATMPlace [34], TDPNavigator-Placer [35]. Classical background: B*-trees [36], HotSpot [38].
  - **Implication:** the placement problem is not our contribution. SA becomes an *essential* solver-only baseline.
- **Closest prior art to our primary question: DeepOHeat-v1 [7].**
  - It already gates surrogate predictions inside an SA placer. The gate is the relative FV residual ‖AT̂ − b‖/‖b‖ < α, and rejected predictions are refined with GMRES.
  - Other related UQ work:
    - ensembles correlate with error in neural operators, and several UQ methods fail out of distribution [26];
    - conformal operator UQ [44] and label-free physics-residual conformal prediction [45];
    - active learning for PDE solvers [41–43];
    - residual-based "trust" reaching AUROC ≈ 0.9 on worst-decile cases [46] (a 2026 preprint).
- **Optimizers exploit surrogates.**
  - Generic model-based optimization documents that designs can easily fool the model: COMs [54], CbAS [55].
  - Optimizing through learned dynamics models falls into adversarial modes [53].
  - Trust regions are the standard remedy [56, 57].
  - The review found **no paper that quantifies FNO/U-Net error inflation specifically at optimizer-selected thermal layouts.** That is a search result, not proof that none exists.
- **Baseline quality matters.** McGreivy & Hakim [39] found that most ML-for-PDE papers they surveyed compared against weak baselines. We therefore include strong solver-only optimizers (SA and CMA-ES) and an exact accounting of solver calls.

### 2.2 The gap we target
- **No head-to-head comparison of gate signals on optimizer-selected designs.** Nobody has compared (a) the raw residual (DeepOHeat-v1 style), (b) the conditioning-aware *preconditioned / energy-norm* residual, (c) deep-ensemble disagreement and (d) their conformalized versions **on the designs an optimizer actually selects**. Nobody has scored them by *verified* design quality against solver calls, or measured **how the residual-to-error relationship degrades with conductivity contrast**.
- **A gap on our side of the derivation** (standard linear algebra, ours):
  - For the SPD system Aθ = b, the error e satisfies Ae = r, so ‖e‖_A = ‖r‖_{A⁻¹}.
  - The raw Euclidean residual bounds the error only up to cond(A), which grows with contrast and resolution.
  - So the raw-residual gate *should* get looser at 400:1 contrast, and a Jacobi- or energy-scaled residual should hold up better. This is a concrete, testable prediction.
- **Novelty level, honestly stated:** an empirical methodology study with a sharp, testable question, not a new architecture. If a closer prior work turns up during implementation, the contribution becomes a rigorous thermal-domain benchmark of known gates, and we will say so.

### 2.3 Metrics conventions to mirror
- **Field accuracy:** per-sample relative L2 / nRMSE [1, 4, 11], computed on θ = T − T_amb so the ambient offset does not inflate it.
- **PDEBench extras [4]:** boundary RMSE and frequency-band RMSE (low k ≤ 4, mid 5–12, high ≥ 13).
- **HSL-TFP-style decision metrics [28]:** peak-temperature absolute error, hotspot location error, and the **Spearman rank correlation of predicted vs true peak temperature** within a batch. Ranking is what an optimizer actually consumes.
- **UQ:** coverage and width of conformal bands, Spearman(uncertainty, error), and AUROC for catching worst-decile errors. Each is reported on random test designs *and* on optimizer-visited designs.

### 2.4 Key references (verified; full list of 58 goes to RESEARCH_NOTES.md)
[1] Li et al. 2020, FNO — https://arxiv.org/abs/2010.08895 · [2] Li et al. 2021, PINO — https://arxiv.org/abs/2111.03794 · [3] Lu et al. 2019, DeepONet — https://arxiv.org/abs/1910.03193 · [4] Takamoto et al. 2022, PDEBench (NeurIPS D&B) — https://arxiv.org/abs/2210.07182 · [5] Gupta & Brandstetter 2022, PDEArena (TMLR 2023) — https://arxiv.org/abs/2209.15616 · [6] Liu et al. 2023, DeepOHeat (DAC) — https://arxiv.org/abs/2302.12949 · [7] Yu et al. 2025, DeepOHeat-v1 (IEEE TCPMT) — https://arxiv.org/abs/2504.03955 · [8] Ganeshram et al. 2022, FC-PINO — https://arxiv.org/abs/2211.15960 · [9] Lakshminarayanan et al. 2017, Deep Ensembles — https://arxiv.org/abs/1612.01474 · [10] Ronneberger et al. 2015, U-Net — https://arxiv.org/abs/1505.04597 · [11] Lu et al. 2021, fair comparison of neural operators (CMAME 2022) — https://arxiv.org/abs/2111.05512 · [12] Wen et al. 2021, U-FNO — https://arxiv.org/abs/2109.03697 · [16] Liu et al. 2022, multi-resolution PDE-preserved learning — https://arxiv.org/abs/2205.03990 · [17] Zhu et al. 2019, physics-constrained surrogates without labels — https://arxiv.org/abs/1901.06314 · [19] Zhao et al. 2021, PI-CNN heat-source layout — https://arxiv.org/abs/2109.12482 · [22] Trifonov et al. 2024, ConDiff — https://arxiv.org/abs/2406.04709 · [26] Mouli et al. 2024, UQ for out-of-domain PDE learning (ICML) — https://arxiv.org/abs/2403.10642 · [28] Chen et al. 2021, HSL-TFP benchmark — https://arxiv.org/abs/2103.11177 · [30] Zhao et al. 2024, online surrogate-assisted thermal layout optimization — https://doi.org/10.1007/s40747-023-01276-0 · [31] Ma et al. 2021, TAP-2.5D (DATE) — https://past.date-conference.com/proceedings-archive/2021/html/1516.html · [32] Duan et al. 2023, RLPlanner — https://arxiv.org/abs/2312.16895 · [38] Huang et al. 2006, HotSpot (IEEE TVLSI) — https://doi.org/10.1109/TVLSI.2006.876103 · [39] McGreivy & Hakim 2024, weak baselines in ML-for-PDE (Nat. Mach. Intell.) — https://arxiv.org/abs/2407.07218 · [44] Ma et al. 2024, conformal UQ for operator learning (TMLR) — https://arxiv.org/abs/2402.01960 · [45] Gopakumar et al. 2025, calibrated physics-informed UQ (ICML) — https://arxiv.org/abs/2502.04406 · [46] Jabbary & Ghanavati 2026, NeuroForge (preprint) — https://arxiv.org/abs/2607.10333 · [47] Babuška & Rheinboldt 1978, a-posteriori error estimates — https://doi.org/10.1137/0715049 · [48] E & Yu 2017, Deep Ritz — https://arxiv.org/abs/1710.00211 · [53] Wu et al. 2024, Compositional generative inverse design (ICLR) — https://arxiv.org/abs/2401.13171 · [54] Trabucco et al. 2021, COMs (ICML) — https://arxiv.org/abs/2107.06882 · [55] Brookes et al. 2019, CbAS — https://arxiv.org/abs/1901.10060 · [57] Eriksson et al. 2019, TuRBO (NeurIPS) — https://arxiv.org/abs/1910.01739

**Unverified, so not relied on:**
- the "ThermalNet" paper (not found);
- DGenNO;
- "Roy et al. 2025";
- whether Allen et al. 2022 and Augenstein et al. 2023 re-verify their designs with the true solver.

---

## 3. Primary hypothesis and contribution

### 3.1 Primary research question (merges the brief's questions C and D; A is used as a supporting ablation)

**Context**
- A learned surrogate is accurate on its test distribution.
- An optimizer actively looks for inputs where the surrogate predicts low temperature. Some of those predictions are low only because the surrogate is wrong there, so the optimizer is drawn to them.
- Our problem has a property most surrogate-optimization settings lack. The governing operator is exactly known and cheap to apply, which gives a **label-free, physically grounded error indicator**: the residual of the prediction.

**The question**
- Does optimizer-induced error happen in neural thermal surrogates, and how large is it?
- Can physics residuals or ensemble disagreement detect it?
- Can those indicators decide where to spend a few solver calls, so that verified design quality is reached cheaply?

### 3.2 Hypotheses (each is pre-registered and evaluated with confidence intervals)

| ID | Hypothesis | Test | Negative result means |
|---|---|---|---|
| **H1** | On designs the optimizer selects, surrogate error in peak temperature is significantly larger than on in-distribution test designs. | Paired bootstrap CI on the difference in absolute peak-θ error: top-k optimizer outputs vs matched ID test layouts, 30 instances × 3 seeds. | The surrogate is robust to optimizer pressure. Ungated surrogate optimization is then enough, which is still a useful result. |
| **H2** | Label-free gate signals rank the true error on optimizer-selected designs. Target: Spearman ρ > 0.5 and AUROC > 0.8 for "peak error > 2 K". These thresholds are fixed now, before any data. **Sub-hypothesis H2b:** the raw residual (DeepOHeat-v1 style [7]) gets worse as conductivity contrast rises, while the Jacobi/energy-scaled residual holds up better (§2.2). | Correlation and AUROC with bootstrap CIs, on random test designs and on optimizer-selected designs, stratified by contrast. Signals compared: raw residual, Jacobi-preconditioned residual, 5-member ensemble std, and split-conformal versions of each. | Label-free indicators cannot be trusted under optimizer pressure. Every candidate needs solver verification, and we quantify what that costs. |
| **H3** | The best gate reaches ≥ 90% of the verified peak-θ reduction of the best solver-only optimizer (SA or CMA-ES) using ≤ 15% of its solver calls *per instance*. It also beats both ungated surrogate optimization and a **random/fixed-interval verification gate** given the same solver budget. | Per-instance paired comparison on 30 held-out instances. Wilcoxon signed-rank test plus bootstrap CI. Reported both excluding and **including** the training-data solves, amortised over the number of instances. | For a cheap 2D solver, surrogate-guided search does not pay off. We report the break-even instance count and resolution. |
| **H4** (secondary) | Physics-informed training (exact discrete residual or energy loss) reduces OOD and optimizer-induced error more than it reduces ID error. | FNO vs PI-FNO across the OOD suites and on H1 designs, 3 seeds. | In the data-rich regime, physics losses add nothing beyond the labels. That is an informative negative result. |

### 3.3 Why it matters, the gap it addresses, and its cost
- **Why it matters.** In hardware design, the usual way a surrogate fails is a design that looks cool in the surrogate but runs hot in signoff. Whether the surrogate can be trusted on optimizer outputs matters more in practice than its average test error.
- **What it adds.** DeepOHeat-v1 [7] already *uses* a raw-residual gate, so having a gate is not new. What is missing (§2.2):
  - a controlled *comparison* of gate signals (raw residual, preconditioned residual, ensemble, conformal) on optimizer-selected designs;
  - a direct measurement of optimizer-induced error inflation (H1);
  - the contrast-dependence prediction (H2b);
  - verified quality per solver call against strong solver-only baselines, which [39] flags as commonly missing.

  This is an empirical methodology contribution, and we describe it as exactly that.
- **Added complexity:**
  - deep ensembles (5× training of one configuration);
  - a torch implementation of the FV operator, which the physics loss needs anyway;
  - the gated optimization loop, about 400 lines.

  No new architectures are needed.
- **Feasible on one A100.** The main extra cost is the 5-member ensemble, estimated at about 5 GPU-hours.

---

## 4. Mathematical problem definition

### 4.1 Physical setting
- **The layer.** A square package layer Ω = [0, L]² of uniform thickness *t*.
- **What it contains:**
  - silicon chiplets carrying functional blocks that dissipate power;
  - mold or underfill filling the rest;
  - optional copper heat-spreader inserts.
- **How heat leaves:**
  - through the bottom face into a cold plate with a spatially varying effective heat-transfer coefficient h_b(x, y). This lumps the thermal interface material and the cold-plate resistance;
  - through the side edges by convection with h_e (zero means adiabatic).
- The top face is adiabatic.

### 4.2 Governing equation (depth-integrated; units W/m²)
Let θ(x, y) = T − T_amb [K]. Integrating the 3D steady heat equation through the thickness, with T uniform in z, gives:

```
  −∇·( k(x,y) · t · ∇θ ) + h_b(x,y) · θ = q''(x,y)        in Ω
  −k ∂θ/∂n = h_e,j · θ                                     on edge j ∈ {W, E, S, N}
```

| Symbol | Meaning | Units |
|---|---|---|
| k | in-plane thermal conductivity, piecewise constant | W m⁻¹ K⁻¹ |
| t | layer thickness (fixed at 0.5 mm) | m |
| q'' | dissipated power per unit **area** | W m⁻² |
| h_b | effective bottom heat-transfer coefficient to the cold plate | W m⁻² K⁻¹ |
| h_e | edge-convection coefficient (applies to the side face of area t per unit length; t cancels) | W m⁻² K⁻¹ |

- **Relation to the brief's form.** Dividing by *t* gives ∇·(k∇T) + q = 0 with volumetric q = q''/t [W m⁻³] and an extra sink term −(h_b/t)·θ.
- **Why the extra sink term.** It is the cold-plate term that a pure ∇·(k∇T) + q = 0 model leaves out. Without it, all heat would have to leave through the die edges, which is physically wrong for chips.
- **Dimensional check:**
  - [k·t·∇²θ] = W m⁻¹ K⁻¹ · m · K m⁻² = W m⁻²
  - [h_b θ] = W m⁻²
  - [q''] = W m⁻²
- **Characteristic spreading length ℓ = √(k t / h_b).** This is the range of the Green's function, i.e. how far a heat source's influence reaches:

  | Case | ℓ | In cells (156 µm each) |
  |---|---|---|
  | Si (k = 150), h_b = 2·10⁴ | ≈ 1.9 mm | ≈ 12 |
  | Si (k = 150), h_b = 10³ | ≈ 8.7 mm | about half the die |
  | Mold (k = 1), h_b = 2·10⁴ | ≈ 0.16 mm | ≈ 1 (an under-resolution risk, see §5.4) |

  Changing the cooling therefore changes **how far the solution operator reaches**. That gives a physical interpretation to the cooling-shift OOD tests: the model must handle a different kernel range.

### 4.3 Properties we rely on and test
- **Well-posedness.** h_b ≥ h_b,min > 0 everywhere. This makes the operator strictly coercive: there is a unique solution, and the discrete matrix is symmetric positive definite.
- **Maximum principle.** The discrete matrix is an M-matrix: positive diagonal, non-positive off-diagonals, diagonally dominant. So q'' ≥ 0 implies θ ≥ 0.
- **Linearity in q''.** For fixed k, h_b and h_e, θ is linear in q''. Consequences:
  - We normalise targets by total power (§7.2), which makes the model exactly equivariant to power scaling.
  - "Higher power magnitude" is therefore **not** a meaningful OOD test. It is replaced by power-*distribution* shifts: more concentrated hotspots.
- **Energy conservation:** ∫ q'' dA = ∫ h_b θ dA + Σ_j ∮ h_e,j t θ ds.
- **Symmetry / reciprocity:** the Green's function satisfies G(x, y) = G(y, x).

### 4.4 Explicit limitations of the 2D model (they go in the report)
- **No vertical gradients.** The model assumes the Biot number Bi = h_b t / k ≪ 1. This holds in silicon (Bi ≈ 0.07 at h_b = 2·10⁴). It fails in mold (Bi ≈ 10). Temperatures there are qualitative only.
- **Not 3D.** There are no layer stacks, through-silicon-via resolution, convection modelling, radiation or transients.
- **Lumped cooling.** h_b stands in for a cold plate and thermal interface material; it is not a fluid simulation.
- **Not signoff-accurate.** The numbers are physically plausible but not signoff-grade. No claim of production fidelity is made.

---

## 5. Reference solver design

### 5.1 Discretisation (prototype validated locally)
- **Grid.** Cell-centred finite volume on an N×N uniform grid, Δ = L/N. The production setting is N = 128, Δ = 156.25 µm.
- **Interior face conductance.** Between cells P and Q (per unit depth): G_PQ = t · k_f with k_f = 2 k_P k_Q / (k_P + k_Q), the **harmonic mean**.
  - Justification: with the material interface on the face, flux continuity makes the two half-cells act as series resistors. The harmonic mean is then the exact two-point flux.
  - Arithmetic averaging is wrong here. It overestimates flux through low-k barriers, by up to about 200× at our contrasts. In the 1:400 composite test it drops to first-order convergence and is about 110× less accurate at N = 128 [measured locally].
  - V2 and V5 below test this directly.
- **Boundary face.** The half-cell conduction resistance is in series with the film resistance: G_b = Δ·t / (Δ/(2k_P) + 1/h_e).
- **Sink term.** h_b,P · Δ², integrated over the cell.
- **Linear system.** A θ = b, where b = q''·Δ² (+ the T_amb terms if we work in T). A is a symmetric positive definite M-matrix with 5 diagonals and n = 16,384 unknowns.

### 5.2 Linear solvers
| Solver | Role | Timing at 128² |
|---|---|---|
| `scipy.sparse.linalg.spsolve` (SuperLU) | Primary reference | 83 ms [measured locally] |
| `splu` once, then reuse the factorisation for new right-hand sides | Optimization variants where only q'' changes | 80 ms to factorise + 3 ms per solve [measured locally] |
| Conjugate gradient with Jacobi preconditioning, tol 1e-12 | Independent cross-check; must agree with the direct solve to 1e-9 relative | n/a |
| `pyamg` (optional) | Only for 512²–1024² grid-convergence references, if direct solves become slow | n/a |

Scaling measured locally: 64² takes 15 ms, 128² 83 ms, 256² 455 ms and 512² 3.2 s, roughly ∝ n^1.3.

**GPU solver (stretch):** a batched preconditioned conjugate-gradient solver in torch. Without it, every speed comparison is explicitly labelled "vs CPU direct solver".

### 5.3 Validation suite (every item must pass before any model is trained)
| ID | Test | Pass criterion |
|---|---|---|
| V1 | 1D uniform slab, Robin on both ends, h_b = 0: (a) q'' = 0, linear profile; (b) uniform q'', analytic quadratic | (a) exact, max relative error < 1e-10; (b) observed order ∈ [1.9, 2.1]. The cell-centred scheme is exact only for piecewise-linear θ. |
| V2 | 1D two-material composite, contrast 1:400, interface on a face: (a) q'' = 0; (b) uniform q'' | (a) < 1e-10; (b) order ≈ 2 with the harmonic mean. In the prototype, harmonic gives order **2.00** (1.8e-4 max relative error at N = 128) while arithmetic gives order **0.99** (2.0e-2) [measured locally]. |
| V3 | 1D fin equation (h_b > 0) against the analytic cosh solution | observed order ∈ [1.9, 2.1] |
| V4 | 2D manufactured solution with smooth k, sink term and Neumann-compatible θ | order ∈ [1.9, 2.1]. Prototype: **2.00** at N = 16→256 [measured locally] |
| V5 | 2D manufactured solution with discontinuous k aligned to faces (piecewise θ satisfying flux continuity); arithmetic-mean ablation | harmonic mean: order ≈ 2. Arithmetic mean documented as degraded (goes in DECISIONS.md) |
| V6 | Energy balance on 1,000 random layouts | \|P_in − P_out\| / P_in < 1e-10. Prototype: 4.7e-14 [measured locally] |
| V7 | Positivity and maximum principle | min θ ≥ −1e-12 · max θ |
| V8 | Linearity, superposition, mirror symmetry, reciprocity (Aᵀ = A) | relative error < 1e-10 |
| V9 | Grid convergence on 50 generator layouts at 128 / 256 / 512, geometry preserved; Richardson extrapolation | median relative peak-θ difference (128 vs 512) < 1%, p95 < 3% (see §5.4) |
| V10 | Units: P_total recomputed from q''·area matches the layout spec | < 1e-12 relative |
| V11 | Torch FV operator (used for the physics loss) vs scipy A·θ | < 1e-12 in fp64, < 1e-5 in fp32 |

### 5.4 Label-accuracy requirement
- **What the labels are.** Labels are the *discrete* 128² solution. We document the gap between that discrete solution and the continuum (the discretisation error) separately.
- **Requirement (V9).** The discretisation error in peak θ must be well below the surrogate errors we hope to resolve. Concretely, below 1%, against a target surrogate error of about 1–3%.
- **Fallbacks if V9 fails** (thin mold regions are the likely cause):
  - (a) raise k_bg,min or reduce h_b,max, so that the spreading length ℓ_mold covers more cells;
  - (b) solve at 256² and restrict to 128², at 5.5× the label cost (still cheap).

---

## 6. Synthetic dataset specification

### 6.1 Layout generator (a "2.5D package layer")
- **Fixed constants:** L = 20 mm, N = 128, t = 0.5 mm, T_amb = 25 °C.
- **Generation steps:**
  1. **Background:** mold, with k_bg sampled.
  2. **Chiplets:** M non-overlapping rectangles, minimum spacing 3 cells (0.47 mm), placed by rejection sampling. Each gets k_c.
  3. **Inside each chiplet:** a **slicing-tree** floorplan (recursive guillotine cuts, as in classical floorplanning) into 1–6 functional blocks. Each block gets a type (logic, memory or IO) and a power density from that type's distribution. Optional hotspots are small sub-rectangles with a density multiplier.
  4. **Optional copper inserts** (heat-spreader slugs) in the mold region.
  5. **Cooling map h_b:** either uniform or a "partial cold plate", where a rectangular footprint is cooled at the base h_b and the rest at a reduced factor.
  6. **Edges:** each edge is independently adiabatic or Robin.

**In-distribution (ID) parameter table**
| Quantity | Units | ID distribution |
|---|---|---|
| k_bg (mold) | W/m/K | log-uniform [1, 5] |
| k_c (Si) | W/m/K | uniform [100, 150] |
| Cu inserts | – | 0–3 (p = 0.5), k uniform [200, 400], sides 1–4 mm |
| Chiplets M | – | 2–8 |
| Chiplet side | mm | 2–8 (snapped), aspect ratio ≤ 3 |
| Blocks per chiplet | – | 1–6 |
| q'' logic / memory / IO | W/cm² | log-uniform [30, 150] / [3, 20] / [5, 30] |
| Hotspots | – | 0–2 per chiplet, density ×[2, 5], side 4–8 cells |
| Min feature size | cells | ≥ 4 (0.625 mm) |
| h_b base | W/m²/K | log-uniform [5·10³, 5·10⁴]. Pattern: uniform (50%) or partial plate (50%; outside factor uniform [0.1, 0.5], footprint 40–90% of the area) |
| h_e per edge | W/m²/K | adiabatic with p = 0.6, else log-uniform [10, 10³] |

### 6.2 OOD suites
The suites fall into three classes:
- **Interpolation gap:** a parameter band held out *inside* the training range.
- **Extrapolation:** one parameter pushed beyond its training range.
- **Structural OOD:** qualitatively new features.

Each axis has three severity levels (s1 < s2 < s3), with 500 samples per level.

| Suite | Class | s1 / s2 / s3 |
|---|---|---|
| GAP | interpolation | k_bg band [2.0, 2.6] and h_b band [1.2·10⁴, 1.6·10⁴] removed from train and val, tested here |
| COUNT | extrapolation | M = 9–10 / 11–12 / 13–14 (chiplet sizes shrink to fit, a confound that is documented) |
| FEATURE | extrapolation (spectral) | min feature 3 / 2 / 1 cells. Probes FNO mode truncation |
| CONTRAST | extrapolation | k_bg [0.5, 1] / [0.2, 0.5] / plus voids (k = 0.03) |
| COOLING | extrapolation, then structural | h_b base [2–5]·10³ / [0.5–2]·10³ (longer spreading length) / unseen "channel stripes" pattern |
| BC | extrapolation, then structural | 3–4 Robin edges / h_e up to 10⁴ / one Dirichlet (isothermal) edge |
| POWER | distribution shift | hotspot multiplier ×[5, 10] / ×[10, 20] / with single-block concentration |
| COMBINED | structural | all s2 shifts together |
| RES | discretisation | ID distribution at 256² (FNO zero-shot; U-Net/CNN compared via downsample → predict → upsample) |
| OOD-DEV | (selection only) | a separate 200-sample COMBINED-s2 set, generated with different seeds. It is the **only** OOD data allowed for model selection, e.g. choosing λ_pde. OOD test suites are evaluated once per final model. |

### 6.3 Splits and leakage control
- **Independent random streams.** Every sample is an independent draw. Seeds come from `numpy.random.SeedSequence(entropy=PROJECT_SEED, spawn_key=(split_id, index))`, so splits use disjoint streams.
- **One geometry per sample in the main datasets.** No power-only variants are made, so near-duplicate layouts cannot appear across splits.
  - Wherever we *do* make power variants of a geometry (for example in POWER), the split is grouped by `geometry_id`.
- **Near-duplicate audit.** Each layout gets a fingerprint: a 16×16 down-sampled material map plus the chiplet count. For every test, OOD and optimization-instance sample we find its nearest train neighbour by IoU. Acceptance: no test sample with IoU > 0.95.
- **Optimization instances (30 test + 10 dev)** come from their own seed stream. They are checked against train in the same way.

### 6.4 Sizes and storage (explicit arithmetic)
- **Per sample:** 4 float32 fields (k, q'', h_b, θ) at 128² = 4 × 16,384 × 4 B = **256 KiB**, plus about 2 KiB of JSON spec.
- **Format:** HDF5 via h5py, chunked per sample with lossless gzip. Piecewise-constant fields compress very well, so the sizes below are upper bounds.

| Set | Samples | Size (uncompressed) | Label cost (estimate: 0.1 s/sample/core, from the 83 ms measured solve) |
|---|---|---|---|
| debug | 256 | 64 MiB | under 1 min on the laptop |
| pilot | 2,000 | 0.5 GiB | about 3 core-minutes |
| train_v1 | 10,000 (scalable to 20,000 based on the learning curve) | 2.5 GiB (5 GiB) | about 17 (33) core-minutes |
| val_v1 | 1,000 | 0.25 GiB | 2 core-minutes |
| test_v1 | 2,000 | 0.5 GiB | 3 core-minutes |
| OOD: 8 suites × 3 levels × 500 + OOD-DEV 200 | 12,200 | 3.0 GiB | about 20 core-minutes |
| RES (256²) | 500 | 0.5 GiB | about 5 core-minutes (0.5 s each) |
| V9 references (512²) | 50 | 0.2 GiB | about 3 core-minutes |
| **Total** | about 28k | **≈ 7.5–10 GiB** | **about 1–1.5 core-hours.** Labelling is **not** a bottleneck. |

### 6.5 Integrity checks (run automatically after generation)
- every value is finite;
- θ ≥ 0;
- energy balance < 1e-9;
- P_total matches the spec;
- the field statistics histogram per split goes into a report PNG;
- the duplicate audit (§6.3);
- an SHA-256 manifest per shard.
- **Cross-machine check:** a 64-sample reference set is generated on both the laptop and the DGX and compared with a tolerance of 1e-10 relative. SuperLU may differ at the 1e-13 level, so the check is tolerance-based, not bitwise.

---

## 7. Model architectures and tensor shapes

### 7.1 Inputs
`x ∈ ℝ^{B×6×128×128}`, float32.

| Channel | Content | Normalisation |
|---|---|---|
| 0 | log₁₀(k / 1 W m⁻¹K⁻¹) | standardised (train mean/std) |
| 1 | q̂ = q''·L² / P_total, a dimensionless power fraction with ∑ q̂ (Δ/L)² = 1 | standardised |
| 2 | log₁₀(h_b / 10⁴ W m⁻²K⁻¹) | standardised |
| 3 | edge ring: h_e,j / 10³ on boundary cells of edge j, 0 elsewhere and on adiabatic edges | none |
| 4, 5 | x/L, y/L at cell centres | none |

### 7.2 Target
`y ∈ ℝ^{B×1×128×128}`.
- **Dimensionless form:** θ̂ = θ · k_ref t / P_total, with k_ref = 150 W m⁻¹K⁻¹.
- **Standardisation:** a scalar train mean/std is applied on top. Inverting it to kelvin is exact.
- **Why this form:** it makes the model exactly equivariant to scaling total power (§4.3). A unit test checks this.

### 7.3 Models
| Model | Specification | Parameters (approx.) | Purpose |
|---|---|---|---|
| **B0 Homogenised physics** | Solve the same PDE with uniform k̄ (area-weighted harmonic mean) and the true h_b, q''. No learning. | 0 | Shows how much the heterogeneity matters. Any learned model must beat it. |
| **B1 CNN** | 8 conv layers 3×3, width 64, dilations [1, 1, 2, 4, 8, 16, 1, 1] (receptive field about 70 cells), GELU, residual connections | ≈ 0.3M | Minimum reference |
| **B2 U-Net** | 4 down-levels (128→8), channels 32–64–128–256–512, 2×(conv3×3 + GroupNorm + GELU) per level, skip connections, bilinear upsampling | ≈ 7.8M (base 32). A width-matched variant is built to match the FNO's parameter count. | Strong image-to-image baseline |
| **M1 FNO-2D** | Lift 6→W (1×1 conv) → 4 Fourier layers → project W→128→1. Each layer computes v ← GELU(W_l v + 𝓕⁻¹(R_l · 𝓕v)), keeping m×m modes (both signs in ky, rfft along kx). Defaults W = 64, m = 16. Zero padding 128→144 for non-periodic boundaries. Implemented with `torch.fft.rfft2` / `irfft2` in fp32. | ≈ 16.8M (4 · 2 · W² · m² · 2) | Main neural operator |
| **M2 PI-FNO** | M1 plus the physics loss (§8) | same as M1 | RQ-A / H4 |
| **Ensemble** | 5 independently seeded members of the best ID model (FNO or U-Net, chosen on val) | 5 × | H2 / H3 uncertainty |
| U-FNO (stretch) | FNO layers with a U-Net branch | – | Only if the FNO fails on fine features |

**FNO intuition.**
- For constant coefficients, the solution operator is a convolution with the Green's function, which is diagonal in Fourier space. A learned spectral multiplier R_l represents that global, smooth part cheaply.
- Heterogeneous k breaks the diagonal structure. The pointwise path W_l and the nonlinearities rebuild local, interface-scale effects.
- **Expected weakness:** sharp temperature gradients at low-k interfaces have high-frequency content beyond the m truncated modes. FEATURE and CONTRAST are designed to expose this. PDEBench reports U-Net > FNO on 128² Darcy [4], and ConDiff [22] reports that every model degrades with contrast, so **a U-Net win is a likely outcome, and it would be reported.**

**Pipeline sanity check (cheap, no download).** Before trusting our own comparison, we regenerate an FNO-paper-style Darcy setup with our own FV solver: two-phase coefficient 12/3, zero-Dirichlet BCs, f = 1, 1,000 train / 200 test. We then confirm that our FNO reaches the same *order of magnitude* of relative error as the published one. If it doesn't, the bug is ours. The published numbers are the authors' claims, and we compare only qualitatively.

---

## 8. Losses and training strategy

### 8.1 Losses
- **Data loss.** L_data is the per-sample relative L2, ‖ŷ − y‖₂ / ‖y‖₂, averaged over the batch (the FNO-literature convention). MSE on standardised targets is logged too.
- **Physics losses, in exact discrete form.** A torch module `FVOperator(k, h_b, h_e)` builds the same 5-point stencil as the solver: harmonic face conductances via shifted tensors, no sparse matrices. Since A θ* = b holds to about 1e-12 for the labels, every physics loss below is **exactly consistent**: it is zero at the true solution.
  - **(i) Raw residual:** ‖Aθ − b‖² / ‖b‖². Ill-conditioned: it weights high-frequency error by about κ(A).
  - **(ii) Jacobi-preconditioned residual:** ‖D⁻¹(Aθ − b)‖² / ‖D⁻¹b‖². This balances high-k and low-k regions, whose diagonals differ by up to about 400×.
  - **(iii) Energy (Ritz) loss:** E(θ) = ½ θᵀAθ − bᵀθ.
    - It is minimised exactly at θ*.
    - E(θ) − E(θ*) = ½‖θ − θ*‖²_A, which is a squared error norm that needs no label.
    - Its conditioning sits between that of (i) and the plain data loss.
  - All three are compared in a pilot on val only. The winner goes forward.
  - **Prior expectation:** the squared residual (i) is effectively ‖e‖²_{A²}, so it squares the conditioning. The energy loss (iii) is ½‖e‖²_A + const. So (iii), or else (ii), is expected to train best. The pilot checks this rather than assuming it.
- **Boundary loss, L_bc.** In the finite-volume form the boundary condition lives inside the boundary rows of A, so the residual already includes BC violation. We log the boundary-row residual as its own metric. A separate weight λ_bc on those rows is tested once; it is kept only if it helps on val.
- **Total:** L = L_data + λ_pde · L_pde (+ λ_bc · L_bc).
- **Weighting:**
  - (a) A fixed grid λ ∈ {0.01, 0.1, 1, 10} applied *after* scale calibration, i.e. λ_eff = λ · L_data(θ₀) / L_pde(θ₀) at initialisation.
  - (b) One adaptive scheme: gradient-norm balancing, which sets λ so that ‖∇L_pde‖ ≈ ‖∇L_data‖ (exponential moving average).
- **Label-free uses of L_pde:**
  - semi-supervised training on extra unlabeled ID layouts;
  - **test-time physics fine-tuning** on the optimizer's current candidate population. These designs are known but unlabeled, so this is legitimate in the optimization setting, and it feeds H3 and H4.

### 8.2 Training system
- **Optimizer and schedule:** AdamW (lr 1e-3 for FNO, 3e-4 for U-Net; tuned on val), weight decay 1e-4, cosine schedule with 2% warm-up.
- **Budget:** fixed at 300 epochs-equivalent, identical across models.
  - Early stopping only *selects* the best-on-val checkpoint; training always runs the full budget, so compute stays matched.
  - Gradient clipping at 1.0.
- **Data on the GPU.** The whole dataset sits in GPU memory (2.5–5 GiB), so there are no DataLoader workers or CPU bottleneck. Shuffling is a seeded permutation saved in the checkpoint.
- **Precision:**
  - FP32 baseline, with TF32 matmuls allowed.
  - BF16 autocast is tried only after the FP32 results are fixed, and **never inside FFTs**: torch 2.10 CUDA FFT rejects bf16 [verified in the v2.10.0 source by the dependency audit]. The spectral path is cast to fp32.
  - The physics residual is always computed in fp32 or fp64.
- **Checkpoints:**
  - written atomically (tmp + `os.replace`), every 10 minutes of wall time and at each epoch's end;
  - each holds model, optimizer, scheduler, step, data permutation and every RNG state;
  - `--resume auto` resumes from `last.pt`; SIGTERM/SIGINT saves and exits cleanly.
- **Logging:**
  - `metrics.csv` and `events.jsonl` (loss components, learning rate, grad norm, peak `torch.cuda.max_memory_allocated`, step time);
  - a NaN/Inf guard that halts and saves a diagnostic checkpoint;
  - `env.json` (package versions, GPU, driver, git SHA, dirty flag).
- **Run ID:** `{exp}-{model}-{data}-s{seed}-{UTCtimestamp}-{gitsha7}`.
- **`torch.compile`:** off by default. It is enabled only if the DGX smoke test passes *and* profiling shows ≥ 15% step-time gain.

---

## 9. Generalization experiments
- **Models compared:** every final model (B0, B1, B2, M1, M2; 3 seeds each) is evaluated once on test_v1 and on each OOD suite at levels s1–s3, plus RES.
- **Headline curve:** error vs shift severity, per axis (Fig. 3).
- **Spectral diagnostics:** error spectra (radially averaged |FFT(error)|) show whether failures are high-frequency. They connect FEATURE and CONTRAST to the FNO mode count.
- **Error localisation:** error on interface cells (cells adjacent to a k jump) is reported separately from error on bulk cells.

---

## 10. Design-optimization methodology

### 10.1 Problem: thermal-aware chiplet placement
- **Instance contents:**
  - M = 5 chiplets, with fixed sizes, fixed internal power maps and fixed total powers;
  - fixed k_bg and Cu inserts;
  - a fixed **non-uniform** cold-plate map h_b (partial plate), so placement matters beyond "spread out";
  - fixed edge BCs;
  - a netlist of weighted chiplet pairs, e.g. a compute–memory star.
- **Variables:** chiplet centres c_i = (x_i, y_i) ∈ ℝ². Orientation is fixed (rotation is a stretch goal).
- **Hard constraints:**
  - (1) stay inside Ω with a 1-cell margin;
  - (2) no overlap and edge-to-edge spacing ≥ 0.47 mm (3 cells);
  - (3) weighted Manhattan wirelength WL = Σ w_ij ‖c_i − c_j‖₁ ≤ 1.25 · WL_min, where WL_min comes from a wirelength-only placement;
  - (4) power and sizes fixed by construction.
- **Objective:** minimise **verified** peak temperature max_Ω θ.
  - Gradient methods optimise a smooth max, (1/τ) log Σ exp(τ θ), with τ annealed.
  - All results report the true max.
- **Feasibility pipeline:**
  - Every candidate first goes through a deterministic *legalisation* step: clamp to the domain, an iterative pairwise push-apart, then a wirelength check.
  - It is then snapped to the grid and rasterised exactly (hard masks).
  - Candidates that are still infeasible are rejected and counted. No infeasible design is ever scored.
- **Differentiable path (surrogate-gradient methods only):**
  - Soft rasterisation: a box indicator σ((x − x₀)/ε)·σ((x₁ − x)/ε) with ε ≈ 0.5 cells, composed into the k and q'' fields.
  - Constraints become penalties during the gradient phase, followed by legalisation.
  - The soft-vs-hard rasterisation gap is measured and reported.

### 10.2 Methods compared (each gets a fixed budget; final scoring always uses the solver)
| ID | Method | Evaluations | Notes |
|---|---|---|---|
| O1 | Random feasible search | solver, 2,000 per instance | lower bound |
| O2 | **Simulated annealing on positions** (TAP-2.5D-style [31]; perturb / swap moves) | solver, 2,000 | **essential**: the established method in the chiplet-placement literature |
| O3 | **CMA-ES with the solver in the loop** (`cma` package; repair + penalty) | solver, 2,000 | classical reference ("ground-truth optimizer") |
| O4 | CMA-ES on the surrogate, then verify the top-8 | surrogate 50k, solver 8 | ungated surrogate |
| O5 | Multi-start gradient descent on the surrogate (Adam, 64 starts batched), then verify the top-8 | surrogate gradients, solver 8 | tests the value of differentiability |
| O6 | **Gated loop (the method under study).** Each round:<br>1. optimise a pessimistic acquisition μ + β·u over a CMA-ES population inside a trust region;<br>2. send the candidates that the **gate** flags, plus the top-k (k = 4), to the solver;<br>3. add the verified designs to the fine-tuning set (optional label-free physics fine-tuning on the population);<br>4. grow or shrink the trust region from predicted-vs-verified agreement.<br><br>**Gate variants, given identical solver budgets:**<br>- G0 none (= O4);<br>- G1 random / fixed-interval verification;<br>- G2 raw relative residual ‖r‖/‖b‖ (DeepOHeat-v1 style [7]);<br>- G3 Jacobi-scaled residual √(rᵀD⁻¹r), a cheap approximation to the energy-norm error;<br>- G4 ensemble std of peak θ;<br>- G5 split-conformal calibration of G2–G4 on val. | solver 100–300 | H2 / H3 |

- **Instances:** 30 test instances plus 10 dev instances.
  - β, k, trust-region parameters and step sizes are tuned **only** on the dev instances.
- **Reporting:**
  - verified Δ peak-θ vs solver calls (Fig. 6);
  - wall-clock time;
  - surrogate error on the final designs (this is H1);
  - constraint-violation and failure counts.

### 10.3 Fair cost accounting
- **Solver calls are counted honestly:** per-instance calls, and also total calls including the 10k training labels.
- **Amortisation:** N_break-even = training solves / (O3 calls − O6 calls).
- **Wall clock** is reported for the CPU solver (1 core and all cores) and for GPU surrogate evaluation, on the same server.
- **Resolution extrapolation:** solver cost at 256² and 512² is measured, which shows where the surrogate starts to pay off.

---

## 11. Baselines and evaluation metrics

**Accuracy metrics** (physical units where possible)
| Metric | Definition | Units |
|---|---|---|
| Relative L2 | per-sample ‖θ_pred − θ‖₂ / ‖θ‖₂, then the mean | – |
| MAE, RMSE | pixelwise | K |
| Peak error | \|max θ_pred − max θ\|, absolute and relative | K, % |
| Hotspot localisation | distance between argmax locations; IoU of the top-1% hottest cells | mm, – |
| Physics residual | Jacobi-preconditioned relative residual of the prediction | – |
| Boundary residual | boundary-row residual | – |
| Global energy error | (∫q'' − ∫h_b θ_pred − edge flux) / ∫q'' | – |
| Interface vs bulk error | MAE restricted to interface-adjacent cells vs the rest | K |
| Boundary RMSE, frequency-band RMSE | PDEBench conventions [4]: low k ≤ 4, mid 5–12, high ≥ 13 | K |
| Peak-rank correlation | Spearman ρ between predicted and true peak θ across a batch (HSL-TFP-style [28]); this is what the optimizer consumes | – |
| UQ quality | conformal coverage and width; Spearman(signal, error); AUROC for worst-decile errors, on random and optimizer-selected designs | – |

**Statistics**
- **Seeds:** 3 training seeds per configuration; 5 for the primary models (FNO, U-Net) and for the ensemble members.
- **Confidence intervals:** 95% CIs from a hierarchical bootstrap that resamples seeds, then test layouts (10,000 resamples).
- **Paired comparisons:** model comparisons use *paired* bootstrap differences on the same layouts. Optimization uses per-instance paired differences (Wilcoxon signed-rank test plus bootstrap CI).
- **Model selection:** on val only (and OOD-DEV for physics-loss settings). Test and OOD test sets are evaluated once per frozen model. Every evaluation run is logged in EXPERIMENTS.md, so any re-run is visible.

**Performance benchmarks**
- **Latency protocol:** `torch.cuda.synchronize()` timing, 20 warm-up and 100 timed iterations, reporting the median and IQR.
- **Batch sizes:** 1, 8, 32, 128, 512, 1024, measured separately for single-sample latency and batched throughput.
- **Also recorded:** peak memory (`max_memory_allocated`), parameter count and FLOPs (via `torch.utils.flop_counter`).
- **Solver comparison:** solver timings use the same server's CPU, single-thread and parallel-process. Comparisons always state the hardware on each side.

**Ablations** (in priority order, cheapest and most informative first)
1. FNO vs U-Net (E1)
2. FNO vs PI-FNO, physics-loss form (E2a)
3. λ_pde sweep (E2b)
4. dataset size 1k / 2k / 5k / 10k / 20k (E3)
5. FNO modes 8 / 16 / 24 / 32 (E4)
6. shift severity (E5, comes for free from §9)
7. O6 with ensemble vs residual vs no gating (E7)

---

## 12. GPU, RAM and disk budgets (all estimates, to be replaced by measured values in M4)

**Per-run estimates**
- FNO (W = 64, m = 16, batch 32, 144² padded):
  - activations ≈ 32·64·144²·4 B ≈ 170 MB per tensor, about 25 tensors saved → **peak VRAM about 6–10 GB**;
  - plus 2.5–5 GB of GPU-resident data;
  - **under 16 GB in total**, which leaves room on a shared 80 GB card.
- U-Net (base 32): similar or less.
- Step time: guessed at 10–30 ms. At 313 steps per epoch × 300 epochs that is **0.3–0.8 GPU-hours per run**. We plan with 1 GPU-hour as the upper bound.

| Experiment | Runs | GPU-h (estimate) | Essential? |
|---|---|---|---|
| E0 pilot (debug + pilot datasets, all models, 1 seed) | 5 | 2 | yes |
| E1 baselines B1, B2, M1 (5 seeds for B2 and M1, 3 for B1) | 13 | 13 | yes |
| E2 physics: loss-form pilot (3) + λ grid (4 × 3 seeds) + adaptive (3) | 18 | 22 | yes |
| E3 data scaling (5 sizes × U-Net/FNO/PI-FNO × 3 seeds; small sizes cheaper) | 45 | 25 | yes |
| E4 modes/width (4 × 3 seeds) | 12 | 12 | yes (reduced to 1 seed if pressed) |
| E5 OOD evaluation of all frozen models | – | 2 | yes |
| E6 optimization O1–O5 (30 + 10 instances) | – | 6 GPU-h + about 2 CPU core-hours | yes |
| E7 ensemble (2 extra members of the best backbone + 5-member PI ensemble for H4) + gated loop O6 (G0–G5) + fine-tunes | – | 20 | yes |
| E8 benchmarking (latency, throughput, memory) | – | 2 | yes |
| E9 BF16 / torch.compile / U-FNO / multi-GPU | – | 10–20 | **optional** |
| **Total** | about 95 runs | **≈ 100 essential (+20 optional)**, planned with a 1.5× margin of about 150 | |

**Storage**
- Datasets: ≈ 10 GiB.
- Checkpoints:
  - FNO about 67 MB of weights; with Adam state about 200 MB.
  - Policy: keep full `last.pt` only while a run is active. Keep `best.pt` as weights only.
  - About 100 runs × 70 MB ≈ 7 GB.
- Environment: about 2 GB (a venv that inherits the existing torch install).
- **Total about 20 GB.** That fits the 40 GB home quota with little spare, so `/raid` is preferred if the write test passes.

**CPU and RAM**
- Data generation: about 1.5 core-hours at about 200 MB RAM per worker.
- Training: about 8 GB of host RAM (data staging).

---

## 13. Dependency compatibility assessment
Source: an audit of the PyPI JSON metadata and the PyTorch v2.10.0 source on 2026-11-20. Nothing was installed on the DGX. **[V]** marks items verified from a source; **[I]** marks inferences.

| Package | Verdict | Notes |
|---|---|---|
| numpy, scipy, matplotlib, PyYAML, tqdm, pytest | **USE** | cp312 wheels with manylinux ≤ 2_28 (glibc 2.31 is fine) [V] |
| h5py | **USE** | manylinux_2_28 wheel bundles HDF5, no system libraries [V]. Fallback: `.npy` shards + JSONL if the smoke test fails. |
| cma | **USE** (O3) | pure Python, numpy only [V] |
| pyamg | optional | wheel available, CPU only [V]. Only for 512²+ references. |
| pandas | optional | 3.x is current. Only for post-processing tables. Avoided in core code. |
| botorch / gpytorch | optional, **not planned** | torch ≥ 2.4 satisfied [V]. Bayesian optimization is not in the essential plan; CMA-ES is the classical baseline. |
| optuna, tensorboard, zarr | not needed | avoided, to keep the environment small |
| neuraloperator | **AVOID** | pulls in wandb, tensorly and more [V]. The FNO is about 150 lines with `torch.fft`. |
| nvidia-physicsnemo | **AVOID** | requires torch ≥ 2.13 [V], which would replace the working torch |
| NVIDIA Warp | **not used** | No measured need. The SciPy solver costs about 80 ms per sample and labelling is not the bottleneck. Reconsidered only if a GPU batched solver becomes essential. |

**Risks**
- **torch.compile / Triton on driver 535 with a cu128 build** [I: probably fine on sm_80, but unproven]. Off by default, and the smoke test checks it.
- **bf16 FFT** is unsupported in torch 2.10 [V]. fp16 FFT works only with power-of-two sizes [V].
- **cuFFT/cuSPARSE under CUDA minor-version compatibility.** No known issue was found [I]. The smoke test checks it.

**Environment strategy**
- `python -m venv --system-site-packages ~/envs/thermal` built from the Python 3.12 env that already has torch 2.10.
- Plus `PYTHONNOUSERSITE=1`.
- Plus a **pip constraints file** that freezes torch==2.10.0+cu128, triton, numpy and warp-lang at their installed versions.
- Every install is run with `--dry-run` first. If the dry run lists `torch` or `nvidia-*`, stop.

---

## 14. Repository architecture

```
ml-thermal-simulator/
├── README.md  PROJECT_PLAN.md  TASKS.md  DECISIONS.md  EXPERIMENTS.md  SERVER_RUNBOOK.md  RESEARCH_NOTES.md
├── CLAUDE.md                     # session-start protocol for future Claude Code sessions
├── pyproject.toml                # package metadata, pytest markers (gpu, slow), ruff config
├── requirements.txt  requirements-opt.txt  constraints.example.txt
├── configs/
│   ├── physics/default.yaml      # constants, material table, units
│   ├── data/{debug,pilot,train_v1,val_v1,test_v1,ood_*,opt_instances}.yaml
│   ├── model/{cnn,unet,fno}.yaml
│   ├── train/{base,physics}.yaml
│   └── experiments/E{0..9}_*.yaml  # sweeps = list of overrides, expanded by scripts/launch.py
├── src/thermal_research/
│   ├── physics/      units.py materials.py fv_assembly.py solver.py torch_operator.py analytic.py
│   ├── layouts/      primitives.py slicing_tree.py generator.py shifts.py rasterize.py fingerprint.py
│   ├── data/         schema.py generate.py storage.py dataset.py normalization.py integrity.py
│   ├── models/       cnn.py unet.py fno.py registry.py
│   ├── training/     losses.py trainer.py checkpoint.py metrics_log.py seed.py
│   ├── evaluation/   metrics.py evaluate.py stats.py spectra.py
│   ├── optimization/ problem.py legalize.py soft_raster.py baselines.py surrogate_opt.py gated_loop.py verify.py
│   ├── benchmarking/ latency.py solver_bench.py memory.py
│   ├── viz/          figures.py style.py            # Agg backend only
│   └── utils/        config.py paths.py env_report.py
├── scripts/          env_check.py generate_data.py train.py evaluate.py optimize.py benchmark.py make_figures.py launch.py
├── tests/            physics/ layouts/ data/ models/ training/ optimization/ evaluation/ gpu/
├── report/           outline.md  figures/ (generated)  tables/ (generated)
└── .github/workflows/ci.yml      # CPU tests on ubuntu-latest, Python 3.12
```
- **Config handling:** YAML plus typed dataclasses, with `key=value` CLI overrides. No Hydra, to avoid a dependency.
- **Paths:** every data and run path comes from `THERMAL_DATA_ROOT` / `THERMAL_RUN_ROOT` environment variables, never hard-coded.
- **`.gitignore`** excludes `data/`, `runs/`, `*.h5`, `*.pt`.

**Test plan** (CPU unless marked `@pytest.mark.gpu`)

| Area | Tests |
|---|---|
| physics | V1–V11 |
| layouts | determinism; non-overlap and spacing; grid snapping; power totals; OOD shift parameters applied |
| data | HDF5 round trip; manifest/checksum; normalisation inverse; leakage audit on synthetic duplicates |
| models | shapes; FNO spectral conv vs a naive DFT reference on 16²; FNO at 256² input; parameter-count formula; power-scaling equivariance through the full pipeline |
| losses | L_pde(θ*) ≈ 0; gradcheck in fp64; energy-loss minimiser equals the solve on 16² |
| training | overfit 8 samples; checkpoint→kill→resume gives bitwise-identical params on CPU; NaN guard |
| optimization | legaliser output always feasible (10k random candidates); hard vs soft raster agree as ε→0; verification uses the solver |
| evaluation | bootstrap CI coverage on synthetic data; metric correctness on hand-built fields |
| CI target | under 10 min on GitHub's free runner |

---

## 15. Agent delegation matrix
Claude Code subagents are available in this environment and were used for this plan's literature and dependency research. During implementation, agents that write code in parallel each get their **own git worktree** (`isolation: worktree`) and a disjoint file-ownership set. The lead reviews each diff, merges it and runs the cross-component tests.

| Agent | Owns (writes) | Reads | Main tasks |
|---|---|---|---|
| Lead (me) | `utils/`, `configs/`, tracking docs, `scripts/launch.py`, integration | everything | interfaces, reviews, merges, DGX handoffs, decisions |
| A Research Scientist | `RESEARCH_NOTES.md`, report §2 | papers | literature, hypotheses, novelty check, related work |
| B Computational Physics | `physics/`, `tests/physics/` | layouts schema | solver, torch operator, V1–V11, grid convergence |
| C Data Engineering | `layouts/`, `data/`, `tests/{layouts,data}/`, `configs/data/` | physics API | generator, OOD shifts, storage, integrity, leakage |
| D DL Architecture | `models/`, `training/`, `tests/{models,training}/`, `configs/{model,train}/` | data API, `physics/torch_operator.py` | CNN, U-Net, FNO, losses, trainer, checkpointing |
| E GPU/ML Systems | `benchmarking/`, `tests/gpu/` | models, solver | profiling, latency/throughput/memory, bf16/compile studies |
| F Optimization & Evaluation | `optimization/`, `evaluation/`, `tests/{optimization,evaluation}/` | models, physics | metrics, bootstrap stats, O1–O6, verification |
| G Quality & Docs | `.github/`, `pyproject.toml`, `README.md`, `SERVER_RUNBOOK.md`, `report/` | everything | CI, test coverage, reproduction docs, report skeleton |

- **Interface contracts are defined first, by the lead,** so that B, C and D can work in parallel. The lead writes `physics/units.py`, the `PhysicalFields` dataclass and `data/schema.py` before any parallel work starts.
- **Disagreements** are resolved with a small experiment on val or with a test. The outcome is recorded in DECISIONS.md.

---

## 16. Week-by-week roadmap (10 weeks; GPU-hours are estimates)

| Wk | Milestone | Tasks (agent) | GPU | Deliverables |
|---|---|---|---|---|
| 1 | **M0 Foundations** | scaffold, config, CI (G); interfaces, tracking docs (Lead); literature deep-dive (A); DGX env and storage smoke test (owner runs it, about 15 min) | 0 | green CI, env report, RESEARCH_NOTES |
| 2 | **M1 Validated solver** | FV assembly, solver, analytic cases, V1–V11 (B); torch operator (B); layout primitives and slicing tree in parallel (C) | 0 | validation report with convergence plots |
| 3 | **M2 Dataset v1** | generator, OOD shifts, storage, integrity, leakage audit (C); V9 grid convergence on generator layouts (B); debug and pilot data locally; full data on DGX CPU (owner, about 30 min) | 0 | dataset card + manifests |
| 4 | **M3 Baselines + training system** | trainer, checkpoint/resume, CNN, U-Net, B0 (D); metrics and bootstrap (F); E0 pilot on DGX | about 3 | pilot curves, resume test passed |
| 5 | **M4 FNO + E1** | FNO (D); profiling harness (E); E1 with 5 seeds; learning curve decides 10k vs 20k | about 15 | Table 1 draft, measured budgets replace §12 |
| 6 | **M5 Physics-informed + OOD** | FVOperator losses, normalisation pilot, λ sweep (D); OOD evaluation (F) | about 25 | Fig. 3, 5; Table 3, 4 drafts |
| 7 | **M6 Optimization baselines** | problem, legaliser, soft raster, O1–O5 (F); H1 measurement | about 6 + CPU | H1 result |
| 8 | **M7 Gated loop** | ensembles, residual estimator, O6, dev-instance tuning, then test instances (F, D) | about 15 | H2/H3 results, Fig. 6, 7 |
| 9 | **M8 Ablations + benchmarks** | E3, E4, E8 (E, D); stats, all figures from logs (F, G) | about 35 | all figures and tables |
| 10 | **M9 Report + reproducibility** | technical report (A, G, Lead); reproduce from a clean clone; README; release tag | about 2 | report PDF/MD, tagged release |

**Buffer:** weeks 9–10 absorb up to about 1 week of slip. If GPU access is scarce, drop E4 to 1 seed and run E3 with 2 seeds.

---

## 17. Milestone acceptance criteria
- **M0**
  - `pytest -m "not gpu"` passes on CI.
  - The DGX smoke test confirms: torch 2.10 sees the A100; an fp32 FFT round trip has error < 1e-5; h5py works; the storage location is writable with ≥ 30 GB free.
  - torch.compile status is recorded either way.
- **M1**
  - V1–V11 all pass.
  - Observed order ∈ [1.9, 2.1] for V3–V5.
  - Energy closure < 1e-10.
  - The validation report is committed (plots generated by script).
- **M2**
  - The generator is deterministic (identical bytes for the same seed).
  - Integrity checks pass on 100% of samples.
  - The duplicate audit finds 0 test samples with IoU > 0.95 against train.
  - V9: median peak discretisation error < 1%, p95 < 3%. If not, apply the §5.4 fallback before M3.
  - Measured storage is within ±25% of §6.4.
  - The laptop-vs-DGX 64-sample cross-check passes.
- **M3**
  - Overfitting 8 samples reaches relative L2 < 1e-3.
  - The kill/resume test is bitwise identical on CPU and within 1e-6 on GPU.
  - Pilot runs finish with no NaN, all metrics logged, and resume tested on the DGX.
  - *No accuracy threshold is required*: results are reported as they are.
- **M4**
  - FNO unit tests pass, including the naive-DFT equivalence and running at 256².
  - E1 is complete with every seed, and Table 1 is generated from logs with CIs.
  - Measured step time, VRAM and GPU-hours are recorded in EXPERIMENTS.md.
- **M5**
  - L_pde(θ*) < 1e-8 (fp64) on labels.
  - The λ sweep and loss-form choice use val and OOD-DEV only.
  - All OOD suites are evaluated exactly once per frozen model, and that is logged.
- **M6**
  - The legaliser is 100% feasible on 10k random proposals.
  - O1, O3, O4 and O5 are complete on 30 test instances, with all finals verified by the solver.
  - The H1 paired CI is reported.
- **M7**
  - O6 hyperparameters are fixed on dev instances, then run once on test instances.
  - H2 (ρ, AUROC with CIs) and H3 (paired tests, amortised and per-instance cost) are reported, whatever the outcome.
- **M8**
  - Every figure and table is produced by `scripts/make_figures.py` from logged results. No number is typed by hand.
- **M9**
  - A fresh clone runs `scripts/` end to end on the debug config on CPU in under 15 minutes.
  - One E1 run reproduced on the DGX matches within seed variance.
  - The report lists limitations and negative results.

---

## 18. Risks and fallbacks
| # | Risk | Likelihood | Mitigation / fallback |
|---|---|---|---|
| R1 | Labels under-resolved at thin low-k regions (ℓ_mold ≈ 1 cell) | medium | Measured in V9. Fallback: raise k_bg,min, or 256²→128² restriction (§5.4). |
| R2 | FNO underperforms the U-Net (sharp interfaces) | medium-high | Report it; it is a result. Analyse error spectra. Optional U-FNO. The primary contribution uses whichever model is best. |
| R3 | Physics loss gives no gain in the data-rich regime | high | That answers H4. The primary contribution does not depend on it. The residual is still used as an *estimator* (H2). |
| R4 | The surrogate's speed advantage is negligible because the solver is cheap | high at 128² | The contribution is framed around solver-call efficiency and verification reliability. Report break-even instance counts and the measured resolution scaling. |
| R5 | The optimization problem is trivial (e.g. "spread to corners") | medium | Wirelength budget plus non-uniform cold plate. Gate: CMA-ES must beat random significantly on dev instances, else tighten the constraints before M7. |
| R6 | Shared GPU unavailable or jobs pre-empted | medium | Everything resumable; runs under 1 h; ≤ 16 GB VRAM so we can co-locate; CNN and debug runs work on a laptop CPU. |
| R7 | torch.compile / h5py fail on the DGX | low-medium | Eager mode; `.npy` + JSONL storage fallback. |
| R8 | 40 GB home quota exceeded | medium | `/raid` write test; checkpoint pruning policy; compressed HDF5. |
| R9 | Soft-vs-hard rasterisation gap misleads gradient search | medium | Hard-raster verification always; measure the gap; anneal ε. |
| R10 | Contention with the owner's other workloads on the same server | medium | Check `nvidia-smi` / `top` before launching. Never run data-generation multiprocessing alongside `SubprocVecEnv` RL jobs. One GPU only. |
| R11 | Literature shows H2/H3 is already well covered. DeepOHeat-v1 [7] already uses a residual gate, and 2025–26 preprints on residual and conformal trust are appearing fast. | medium, rising | Our novelty is the *comparison under optimizer exploitation* and the contrast-dependence test (H2b), not the gate itself. Re-run the literature search at M6. If it is pre-empted, reframe as a rigorous thermal-domain benchmark and say so in the report. |
| R12 | SA/CMA-ES in the solver loop is so cheap at 128² (about 80 ms per call; 2,000 calls ≈ 3 min per instance on one core) that the gated loop's savings look trivial | high | Expected, and it is why H3 is about *solver calls* with amortisation and resolution scaling, not wall-clock. Also run O3 and O6 at 256² on a 10-instance subset (solve about 0.45 s), where the trade-off starts to matter. |

---

## 19. Final technical report structure
1. Abstract (measured results only)
2. Introduction: motivation, contribution, hypotheses H1–H4
3. Related work: neural operators, physics-informed operator learning, chip-thermal ML, surrogate exploitation and UQ
4. Physical model: equations, units, assumptions, limitations (Bi number, 2D)
5. Reference solver and verification: V1–V11, grid convergence
6. Dataset: generator, distributions, splits, leakage audit, OOD taxonomy
7. Surrogates: architectures, losses, training protocol, compute
8. Results I: accuracy and generalization (Tables 1, 3; Figs. 1–5, 7)
9. Results II: optimization and verification, H1–H3 (Table 6; Fig. 6)
10. Computational cost and fair-comparison accounting (Table 5; Figs. 2, 8)
11. Ablations (Table 4)
12. Limitations, negative results, threats to validity
13. Conclusion
14. Appendices: hyperparameters, full per-suite results, reproducibility checklist, environment

**Figures:**
1. Layout, reference field, prediction and error
2. Accuracy vs latency
3. OOD error vs severity
4. Data scaling
5. λ_pde effect
6. Verified improvement vs solver calls
7. Failure cases and hotspot errors
8. Throughput vs batch size

**Tables:**
1. Models
2. Dataset configuration
3. OOD
4. Ablations
5. Resources
6. Optimization

---

## 20. First implementation tasks after approval (milestone M0 → M1)
1. **T0.1 (Lead):** commit the tracking docs (TASKS, DECISIONS, EXPERIMENTS, SERVER_RUNBOOK, RESEARCH_NOTES, project CLAUDE.md).
2. **T0.2 (G):** `pyproject.toml`, package skeleton, pytest markers, ruff, GitHub Actions CPU CI.
3. **T0.3 (Lead):** `physics/units.py`, the `PhysicalFields` / `LayoutSpec` dataclasses, and `utils/config.py`. These are the interface contracts.
4. **T0.4 (owner, DGX):** the environment and storage smoke test (Appendix B).
5. **T1.1–T1.6 (B):** solver plus V1–V11. **T1.7 (C, in parallel):** layout primitives and the slicing tree.

**Recommended first milestone: M0 + M1, "Validated reference solver."** It runs entirely locally on CPU and needs no GPU. Its result decides whether the 128² labels are trustworthy, and everything else depends on that.

---

## Appendix A — Dependency-aware task graph
Where: **L** = local CPU, **D-CPU** = DGX CPU, **D-GPU** = DGX GPU. Tasks with the same ∥ group can run in parallel (separate worktrees, disjoint files).

| ID | Task | Agent | Deps | Files | Definition of done / tests | Where | ∥ |
|---|---|---|---|---|---|---|---|
| T0.1 | Tracking docs and project CLAUDE.md | Lead | approval | *.md | files exist, populated from this plan | L | a |
| T0.2 | Packaging, CI, lint, pytest markers | G | approval | pyproject.toml, .github/, tests/conftest.py | CI green on an empty test suite | L | a |
| T0.3 | Interface contracts: units, PhysicalFields, LayoutSpec, config loader | Lead | approval | physics/units.py, data/schema.py, utils/config.py | type-checked; round-trip config test | L | a |
| T0.4 | DGX env + storage smoke test | owner | T0.2 | — | Appendix B outputs returned | D | a |
| T1.1 | FV assembly (scipy) with harmonic faces, Robin/adiabatic/Dirichlet edges, sink term | B | T0.3 | physics/fv_assembly.py | V8 symmetry and M-matrix tests | L | b |
| T1.2 | Solver wrapper (spsolve, splu reuse, CG cross-check) | B | T1.1 | physics/solver.py | CG vs direct < 1e-9 | L | b |
| T1.3 | Analytic and manufactured solutions | B | T0.3 | physics/analytic.py | used by V1–V5 | L | b |
| T1.4 | Validation suite V1–V10 plus a convergence-plot script | B | T1.1–1.3 | tests/physics/, scripts/validate_solver.py | all pass; plots generated | L | b |
| T1.5 | Torch FVOperator (residual, Jacobi, energy) | B | T1.1 | physics/torch_operator.py | V11; gradcheck | L | b |
| T1.6 | Solver benchmark (64–512, 1 core vs N processes) | E | T1.2 | benchmarking/solver_bench.py | CSV output; reused on DGX later | L | b |
| T1.7 | Rect primitives, snapping, slicing-tree floorplans | C | T0.3 | layouts/primitives.py, slicing_tree.py | non-overlap, snap, area tests | L | b |
| T2.1 | ID generator (chiplets, blocks, hotspots, inserts, cooling, edges) | C | T1.7 | layouts/generator.py, rasterize.py | determinism; power totals; parameter ranges | L | c |
| T2.2 | OOD shift definitions (8 suites × 3 levels, GAP, OOD-DEV) | C | T2.1 | layouts/shifts.py, configs/data/ood_*.yaml | each shift verified by summary statistics | L | c |
| T2.3 | Fingerprints and leakage audit | C | T2.1 | layouts/fingerprint.py, data/integrity.py | synthetic duplicates detected | L | c |
| T2.4 | Generation CLI (multiprocess, seeded), HDF5 storage, manifests | C | T2.1, T1.2 | data/generate.py, storage.py | round trip; resumable generation | L | c |
| T2.5 | V9 grid convergence on generator layouts | B | T2.1, T1.4 | scripts/grid_convergence.py | §5.4 criterion; result goes in DECISIONS | L | c |
| T2.6 | Debug + pilot datasets locally; dataset card | C | T2.4 | data/ (gitignored), docs | integrity 100% | L | — |
| T2.7 | Full datasets + 64-sample cross-check on the DGX | owner/C | T2.6, T0.4 | — | manifests match; cross-check passes | D-CPU | — |
| T3.1 | GPU-resident dataset, normalisation, power-scaling transform | D | T2.4 | data/dataset.py, normalization.py | inverse-transform and equivariance tests | L | d |
| T3.2 | Trainer, checkpoint/resume, logging, NaN guard, seeds | D | T0.3 | training/*.py | overfit-8; kill/resume bitwise | L | d |
| T3.3 | CNN, U-Net, B0 homogenised baseline | D | T3.1 | models/cnn.py, unet.py; physics/ (B0 uses the solver) | shape tests | L | d |
| T3.4 | Metrics, bootstrap, evaluation CLI | F | T3.1 | evaluation/*.py | metric hand-cases; CI coverage test | L | d |
| T3.5 | E0 pilot on the DGX | owner | T3.2–3.4, T2.7 | configs/experiments/E0 | runs finish, resume tested | D-GPU | — |
| T4.1 | FNO (SpectralConv2d, padding, modes) | D | T3.2 | models/fno.py | naive-DFT equivalence; 256² run; parameter formula | L | e |
| T4.2 | Latency/throughput/memory harness | E | T3.3 | benchmarking/latency.py, memory.py | runs on CPU with a tiny config | L | e |
| T4.3 | E1 baselines (5/5/3 seeds) | owner | T4.1, T3.5 | configs/experiments/E1 | Table 1 generated | D-GPU | — |
| T5.1 | Physics losses + weighting schemes in the trainer | D | T1.5, T4.1 | training/losses.py | L_pde(θ*) ≈ 0; gradcheck | L | f |
| T5.2 | OOD evaluation driver and spectra | F | T3.4, T2.2 | evaluation/ood.py, spectra.py | runs on debug OOD | L | f |
| T5.3 | E2 physics sweeps; E5 OOD evaluation | owner | T5.1, T5.2 | configs/experiments/E2, E5 | Fig. 3, 5 | D-GPU | — |
| T6.1 | Optimization problem, legaliser, netlists, instances | F | T2.1 | optimization/problem.py, legalize.py | 10k-feasibility test | L | g |
| T6.2 | Soft rasteriser (differentiable) | F | T2.1 | optimization/soft_raster.py | ε→0 matches the hard raster | L | g |
| T6.3 | O1–O5 implementations + solver verification | F | T6.1, T6.2, T4.1 | optimization/baselines.py, surrogate_opt.py, verify.py | budgets enforced; all finals verified | L | g |
| T6.4 | E6 runs + H1 analysis | owner | T6.3, T4.3 | configs/experiments/E6 | H1 CI | D-GPU/CPU | — |
| T7.1 | Ensemble training config + residual error estimator | D/F | T5.1 | evaluation/uncertainty.py | estimator unit tests | L | h |
| T7.2 | Gated loop O6 | F | T7.1, T6.3 | optimization/gated_loop.py | runs end to end on CPU with a tiny model | L | h |
| T7.3 | E7 dev tuning, then a test run; H2/H3 analysis | owner | T7.2 | configs/experiments/E7 | H2/H3 CIs | D-GPU | — |
| T8.1 | E3, E4 ablations; E8 benchmarks | owner/E | T4.3 | configs/experiments/E3, E4, E8 | Tables 4, 5 | D-GPU | — |
| T8.2 | All figures and tables from logs | F/G | all results | viz/figures.py, scripts/make_figures.py | no hand-entered numbers | L | — |
| T9.1 | Report, README, reproducibility check, release | A/G/Lead | T8.2 | report/, README.md | §17 M9 criteria | L + D | — |

---

## Appendix B — First DGX checkpoint (prepared; **not yet run**)
**When:** right after approval, alongside M0. It costs about 15 minutes of the owner's time and no GPU-hours.
**Purpose:**
- confirm which conda env has Python 3.12 + torch 2.10.0+cu128;
- check GPU FFT, h5py and SciPy on the server;
- check whether `torch.compile` works on driver 535;
- find a writable storage location with free space.

**Prerequisites:** you are logged in to the DGX; any shell is fine. These are short commands, so no tmux is needed. Nothing is installed and nothing is changed apart from creating and then removing a probe file.

**Commands** (one per line; copy each on its own):
1. Envs, GPUs, glibc, disk, CPU and RAM:
   `conda env list && nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv && ldd --version | head -1 && df -h ~ && nproc && free -g`
2. Activate the env that has Python 3.12 + torch 2.10 (you'll know its name from step 1):
   `conda activate <ENV_WITH_PY312_TORCH210>`
3. Torch, CUDA, FFT and bf16. Pick a GPU index that step 1 shows as idle:
   `CUDA_VISIBLE_DEVICES=<FREE_GPU_INDEX> python -c "import sys,platform,torch;print(sys.version.split()[0],platform.libc_ver());print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0));x=torch.randn(4,64,144,144,device='cuda');y=torch.fft.irfft2(torch.fft.rfft2(x),s=(144,144));print('fft fp32 roundtrip err',(x-y).abs().max().item());a=torch.randn(512,512,device='cuda');c=torch.autocast('cuda',dtype=torch.bfloat16);c.__enter__();print('autocast matmul dtype',(a@a).dtype);c.__exit__(None,None,None)"`
4. Scientific Python stack:
   `python -c "import numpy,scipy,matplotlib;print('numpy',numpy.__version__,'scipy',scipy.__version__,'mpl',matplotlib.__version__)" ; python -c "import h5py;print('h5py',h5py.__version__)" || echo "h5py missing"`
5. Whether torch.compile works on driver 535. It may take a minute, and a failure here is fine, it just tells us to stay in eager mode:
   `CUDA_VISIBLE_DEVICES=<FREE_GPU_INDEX> timeout 600 python -c "import torch;f=torch.compile(lambda x:torch.sin(x)+x*x);x=torch.randn(1024,device='cuda');print('compile ok, err',(f(x)-(torch.sin(x)+x*x)).abs().max().item())" 2>&1 | tail -5`
6. Whether `/raid` is writable. This only creates and then removes our own probe directory:
   `(mkdir -p /raid/$USER/thermal_write_test && touch /raid/$USER/thermal_write_test/probe && echo RAID_WRITABLE && df -h /raid && rm /raid/$USER/thermal_write_test/probe && rmdir /raid/$USER/thermal_write_test) || echo RAID_NOT_WRITABLE`
7. Installed versions we must not disturb:
   `pip list 2>/dev/null | grep -i -E "^(torch|triton|numpy|scipy|h5py|warp-lang|nvidia-cufft)"`

**Expected resource use:** under 1 GB of GPU memory for a few seconds on one GPU, no disk beyond the probe, about 2 minutes in total.
**Signs of trouble:**
- `cuda.is_available()` prints False;
- the FFT error is above 1e-4;
- step 5 shows "unsupported toolchain" or a PTX error (expected to be possible, and not a blocker);
- RAID_NOT_WRITABLE (then everything goes under `~`, with checkpoint pruning).

**What to send back:** the full terminal output of steps 1–7.
**What happens next:** I record the results in SERVER_RUNBOOK.md and DECISIONS.md, which settles the env name, the storage root, whether torch.compile is on or off, and whether to use h5py or the `.npy` fallback. I then write the venv and constraints setup commands.

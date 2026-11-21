# Project Plan — Sim-to-Real Thermal-Map Prediction for a Commercial Processor

> **Status: APPROVED by the owner (2026-11-21). Building.**
> Earlier drafts: the synthetic-layout plan v0 is in `docs/archive/`. The real-data "wait for full dataset" plan v1 is superseded and summarised in DECISIONS.md (D-002).
>
> **How numbers are labelled:**
> - **[measured locally]**: computed in this repo from the public dataset excerpts.
> - **[paper-reported]**: the original authors' claim.
> - **[estimate]**: everything else.

---

## 0. Executive summary

**Goal**
- Predict the full-chip temperature map of a real processor (Intel i7-8650U) from its live activity telemetry.
- Use **physics simulation for training** and **real infrared measurements for calibration and testing**.

**System**

```
telemetry window ──► Translator (small, learned on 1–10 real workloads) ──► power map q(x,y) ≥ 0 + base temperature
power map        ──► Heat model (physics; learned from SIMULATION only, then frozen) ──► temperature map
```

**Two heat-model variants**
- **H-spec:** an exact, differentiable spectral solver (uniform effective properties). Pure physics, no learning.
- **H-net:** a neural operator trained only on simulated data from a 2-layer finite-volume model with randomised physical parameters ("domain randomisation"). It learns heat spreading the closed-form solver can't represent.

**Data**
- The public UC Riverside excerpt for the i7: **19 workloads × 100 IR frames, 170 telemetry features, 185×154 maps** [measured locally].
- Real data is used **only** to calibrate on k workloads (k ∈ {1, 3, 5, 10}). The remaining workloads are a held-out **test** set the model never sees.
- No dependence on the authors or on the full dataset.

**Questions answered**
1. How much error does a random-frame split hide? Leakage analysis.
2. How well can a simulation-trained physics model represent real chip heat? The sim-to-real gap.
3. With only k real workloads, does the hybrid (physics + translator) beat pure-ML models trained on the same k workloads, on unseen workloads?
4. How many real workloads does pure ML need to catch up?
5. Is the hybrid's uncertainty calibrated on unseen workloads?

**Compute**
- Everything runs on the laptop CPU except H-net training, which is the first DGX job if local training proves too slow. The decision will be measured, not assumed. Estimated 2–6 A100 GPU-hours.

---

## 1. Key facts established so far

- **Public data per i7 workload:** 100 frames. Telemetry: 170 columns (unnamed; 10 constant). Maps: 185×154 °C [measured locally].
- **The telemetry contains on-die temperature sensors:**
  - About 9 integer columns, range 0–72, correlate at **−0.98** with the mean map temperature [measured locally].
  - This is consistent with Intel PCM's "thermal headroom", i.e. degrees below the maximum junction temperature. That reading is an inference from the value pattern.
  - ThermGAN also found core temperatures to be the top inputs [paper-reported].
  - **So every headline result is reported with and without sensor columns.**
- **Frames are time-ordered.**
  - Lag-1 autocorrelation of the mean map temperature within a workload: 0.994.
  - Drift within a workload is up to 25 °C over 100 frames, which means heating transients are present [measured locally].
  - The interval between frames isn't stated. The original runs were 4 min at 60 Hz [paper-reported], so the excerpt is probably subsampled (inference).
- **Leakage:** under a random-frame split, copying the nearest-in-time training frame gives **0.37 °C RMSE** with no model [measured locally]. Published "0.36 °C"-class numbers are therefore not evidence of workload generalization.
- **Correction:** an earlier audit quoted "ridge LOWO 8.0 °C". That ridge used fixed, excessive regularisation and is **not** a fair baseline. M0 replaces it with a tuned ridge.
- **Pixel pitch is 50 µm** for the i5/i7 rigs [paper-reported], giving a map of 9.25 × 7.70 mm.
- **Rig:** heatsink removed, back-side Peltier cooling calibrated to the stock operating point; emissivity and ambient conditions not stated [paper-reported]. Claims are scoped to this rig.
- **Data licence:** none. Data is never committed (`data/` is gitignored). Pickles are loaded only through a whitelist unpickler.

---

## 2. Physics

### 2.1 Model family
Use θ = T − T_base, where T_base is a learned base temperature: a translator output, roughly the Peltier operating point.

**H-spec: one layer, uniform effective properties, steady state**
```
  −∇·(G ∇θ) + h θ = q           G = k t  [W/K],  h [W m⁻² K⁻¹],  q [W m⁻²]
  ∂θ/∂n = 0 on the map edges
```
- **Exact solve.** On the 185×154 cell-centred grid (Δ = 50 µm), the 5-point FV operator is diagonalised exactly by the 2-D DCT-II, with eigenvalues λ_mn = (G/Δ²)·[4 sin²(πm/2H) + 4 sin²(πn/2W)] + h.
- **Steady state:** θ̂ = q̂ / λ.
- **Transient extension** (implicit Euler): θ̂ⁿ⁺¹ = (C/Δt · θ̂ⁿ + q̂ⁿ⁺¹) / (C/Δt + λ).
- Cost is O(HW log HW), and the solve is differentiable.
- **Identifiability.**
  - Only the ratio q/h and the spreading length ℓ = √(G/h) affect θ.
  - Without an absolute power anchor (the i7 column names are unknown), we fix G and learn h and q's scale. q is then a *relative* power map. That is fine for temperature prediction, and the limits on interpreting it are documented.
  - On chips with a power column (R7: `Socket0Power`), the anchor ∑qΔ² = P is applied.

**Simulator for H-net training data: a 2-layer stack, finite volume, sparse direct solve**
- **Layer 1:** die, conductance G₁ = k_Si·t_die.
- **Layer 2:** package/spreader, conductance G₂, coupled to layer 1 by a vertical conductance g₁₂ per unit area.
- **Bottom:** layer 2 loses heat to the Peltier through h₂.
- **Sources:** power is injected in layer 1, and the temperature of layer 1 is observed.
- Captures lateral spreading in the package that a single uniform layer cannot.
- **Domain randomisation.** Each simulated sample draws (G₁, G₂, g₁₂, h₂) log-uniformly over physically plausible ranges:

  | Parameter | Range |
  |---|---|
  | k_Si | 100–150 W/m·K |
  | t_die | 0.1–0.8 mm |
  | G₂ | 0.5–20 W/K |
  | g₁₂ | 1e3–1e5 W/m²K |
  | h₂ | 100–5000 W/m²K, bracketing the 600 W/m²K COMSOL fit [paper-reported] |

  Power maps are drawn from a generic family: random rectangles, Gaussian blobs and uniform background, at random intensities. **Nothing is fitted to real maps.**
- **Validation:** the FV assembly is reused from the verified prototype (second-order manufactured-solution convergence, energy closure ≈ 1e-14 [measured locally]). Test: the 2-layer system with g₁₂ → ∞ and identical layers reduces to the 1-layer case, and the 1-layer FV result equals H-spec within 1e-10.

### 2.2 Laplacian inversion (diagnostic and initialisation)
- VSCLab's steady inverse: q ∝ (−∇²T)₊ after denoising [paper-reported method].
- Used to **check the layout**: inverted maps should show consistent core and GPU regions across workloads.
- Used to **initialise** the translator's basis maps, via non-negative factorisation of inverted maps from **calibration workloads only**.

---

## 3. Models

### 3.1 Translator (the only part trained on real data)
- **Input:** a telemetry window of the last w frames (w ∈ {1, 5}), z-scored with calibration-set statistics.
- **Output:**
  - K ≥ 1 non-negative coefficients a_k(t) (softplus) for K learned non-negative, smooth **basis power maps** B_k(x, y), which roughly correspond to cores, GPU and uncore;
  - plus T_base(t).
- Power map: q = Σ_k a_k B_k.
- **Size:** tiny on purpose (risk #4). A linear or one-hidden-layer MLP for a(t), with K ≈ 4–10 basis maps parameterised at low resolution (37×31) and upsampled bilinearly.
- **Training:** through the **frozen** heat model on calibration workloads. MSE loss against the real maps, plus three regularisers:
  - basis smoothness (total variation),
  - non-negativity (by construction),
  - ℓ1 on the coefficients.
- The heat model's global parameters (h, and for H-net the physical-parameter channels) are fitted jointly on calibration data only.

### 3.2 Heat models
| ID | What | Trained on |
|---|---|---|
| H-spec | exact DCT solver, uniform properties (§2.1) | nothing; 1–2 scalars fitted at calibration |
| H-net | FNO (fallback U-Net): input power map + broadcast physical-parameter channels → layer-1 temperature rise | simulation only (§2.1); frozen afterwards |

### 3.3 Baselines (trained on the same calibration workloads unless stated)
| ID | Baseline |
|---|---|
| B-MEAN | calibration mean map |
| B-RIDGE | ridge regression telemetry → PCA-k maps, with λ and k chosen by inner leave-one-workload-out CV on the calibration set |
| B-MLP | MLP telemetry → PCA coefficients |
| B-CNN | MLP → conv decoder → map |
| B-TF | small transformer over feature tokens (iTransformer/TherMapNet-style) → conv decoder. Included because the field uses transformers; data-starved by design here. |
| B-COPY | nearest-in-time training frame. **Random-split leakage diagnostic only.** |
| REF-ALL | each pure-ML baseline trained on all 18 non-test workloads (leave-one-workload-out). The "how much real data would pure ML need" reference. |

---

## 4. Evaluation protocol

**Splits**
| ID | Split | Purpose |
|---|---|---|
| S-RAND | random 80/20 over frames | leakage demo only (B-COPY, B-RIDGE, B-CNN) |
| S-CAL-k | pick k calibration workloads; test on the other 19 − k | main protocol |
| S-LOWO | leave one workload out over all 19 | REF-ALL only |

- **Repeats:** R = 20 random calibration draws per k ∈ {1, 3, 5, 10}, with a fixed seed list. The **same draws are used for every model**, so comparisons are paired.
- **Validation inside calibration:** the last 20% of frames of each calibration workload, for early stopping and hyperparameters. **Test workloads are never used for any selection.**

**Metrics** (°C)
- mean per-map RMSE, plus max and std over maps;
- pooled RMSE; MAE;
- **peak-temperature error**;
- **hotspot location error** (mm, 50 µm/px);
- top-1% hotspot IoU;
- 2-D correlation.

**Statistics**
- The unit is the **test workload**, nested in calibration draws.
- Report mean and 95% CI via a hierarchical bootstrap (draws → workloads).
- Paired model differences per (draw, workload), with a bootstrap CI and a Wilcoxon test.

**Ablations** (priority order)
1. with vs without sensor columns
2. k ∈ {1, 3, 5, 10}
3. H-spec vs H-net
4. window w = 1 vs 5
5. number of basis maps K
6. steady vs transient H-spec (if the frame interval can be estimated or fitted)

**UQ**
- deep ensembles of translators (5 seeds);
- split-conformal intervals calibrated at the **workload level** on calibration workloads;
- coverage measured per test workload;
- AUROC for flagging the worst test frames.

**Optional second chip:** i5-3337U (5 workloads; calibrate on 2, test on 3) as a replication.

---

## 5. Gates (go / no-go, decided from data)
- **G1: physics expressiveness.**
  - For calibration frames, fit q ≥ 0 per frame directly (inverse problem, with smoothness), using H-spec and fitted global scalars.
  - This gives the **representation floor**: the best error that physics family can reach.
  - If the median floor is ≤ 1.0 °C, H-spec is adequate.
  - If > 1.5 °C, H-spec is too crude: prioritise H-net, and consider a small learned residual. Record it in DECISIONS.md.
- **G2: layout consistency.** Laplacian-inverted power maps from different calibration workloads should share hot regions (cosine similarity of normalised maps). If inconsistent, the translator gets more basis maps and a weaker smoothness prior.
- **G3: H-net sim accuracy.**
  - Held-out *simulated* error < 2% of the temperature rise before it is used on real data.
  - Local training time is measured: if a reduced config trains in under about 2 h on the laptop CPU, train locally; otherwise hand off to the DGX.

---

## 6. Risks and mitigations
| # | Risk | Detection | Mitigation |
|---|---|---|---|
| 1 | Simulation ≠ reality | G1 floor; calibration-workload fit | 2-layer domain-randomised H-net; parameters fitted on calibration only; report the gap |
| 2 | Wrong layout | G2; basis maps vs inverted maps | Layout is *learned* (basis maps), not hand-drawn from die shots; initialised from calibration-only inversion |
| 3 | Translator cheats (power in wrong places) | basis maps leave plausible regions; implausible magnitudes | non-negativity, smoothness, low-resolution basis, ℓ1; anchor total power where a power column exists |
| 4 | Overfitting with k small | calibration-vs-test gap | tiny translator; k-curve; tuned regularisation on the calibration validation split |
| 5 | Lucky calibration choice | spread across 20 draws | report the distribution, never the best draw |
| 6 | Sensors make it trivial | no-sensor ablation | every headline reported with and without sensors |
| 7 | Transients / history | error vs within-workload drift | window w = 5; transient H-spec variant |
| 8 | Pure ML wins | the paired comparison | report honestly; compare at k and at REF-ALL |
| 9 | Rig ≠ real laptop | – | stated limitation |
| 10 | Single chip | – | optional i5 replication; scoped claims |
| 11 | Small n, wide CIs | – | workload-level statistics; no over-claiming |
| 12 | Unknown column semantics | – | sensor columns found by a data rule on calibration only (integer, range ≤ 110, \|corr\| > 0.8 with mean temperature); the rule is documented |

---

## 7. Repository layout

```
src/thermal_research/
  data/        safe_pickle.py  ucr.py (load/convert/cache)  splits.py  features.py (z-score, sensor detection)
  physics/     dct.py  spectral.py (H-spec)  fv.py (1- and 2-layer FV assembly + solve)  inversion.py  powermaps.py (random sim maps)
  models/      translator.py  baselines.py  heatnet.py (FNO/U-Net)  pca.py
  training/    fit.py (generic trainer, checkpoints)  calibrate.py (translator through frozen heat model)
  evaluation/  metrics.py  stats.py  protocol.py (S-RAND / S-CAL-k / S-LOWO runner)
  uq/          ensemble.py  conformal.py
  viz/         figures.py
scripts/       audit_data.py  gate_g1_g2.py  gen_sim.py  train_heatnet.py  run_protocol.py  make_figures.py  env_check.py
tests/         one test module per package module
results/       small JSON/CSV summaries (committed); figures (committed PNGs)
data/, runs/   gitignored
```

---

## 8. Milestones (build order; each one ends with a commit and push to `dev`)
| M | Content | Done when |
|---|---|---|
| M0 | scaffold, CI, safe loader + cache, splits, metrics, B-MEAN / B-RIDGE / B-COPY, leakage result | tests green; `results/leakage.json` + figure |
| M1 | DCT, H-spec, 1/2-layer FV, inversion, random power maps | FV == H-spec < 1e-10; energy closure; tests green |
| M2 | Gates G1 and G2 on the calibration protocol | `results/gates.json`, decisions logged |
| M3 | translator + calibration through H-spec; B-MLP, B-CNN, B-TF; full S-CAL-k protocol, sensor ablation | main table (H-spec) with CIs |
| M4 | simulation dataset, H-net training (local or DGX), hybrid with H-net | G3 passed; H-net rows in the main table |
| M5 | UQ (ensembles, workload conformal), ablations, latency | UQ table and figures |
| M6 | figures, README results, technical report | all numbers from logged results |

**DGX:** only M4 might need it. When it does, a **DGX SERVER ACTION REQUIRED** block goes in chat and in `SERVER_RUNBOOK.md`.

---

## 9. Conventions
- Commits go on `dev` as `krishbhagirath`, with informal messages and **no co-author trailers of any kind**. `main` is updated only with the owner's OK.
- Never commit data, checkpoints or raw pickles. `results/` holds only small summaries and figures.
- Every number in the README and report is generated by a script from `results/`.

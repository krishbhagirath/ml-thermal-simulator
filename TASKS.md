# Tasks

Status: `[ ]` pending · `[~]` active · `[x]` done · `[!]` blocked

## RESUME HERE (last session ended 2026-11-21)
- Leakage audit DONE -> results/leakage.json (copy-nearest S-RAND 0.22 C; ridge LOWO 2.46 C with sensors, 10.4 C without).
- Gates G1/G2 were running when the session ended: rerun `python -u scripts/gate_g1_g2.py` (resumable).
- Hybrid redesigned as `models/hybrid.py` HybridRidge (physics factorisation + tuned ridge). First check (k=3, draw 0):
  with sensors hybrid 9.3 C vs ridge 12.8 C; no sensors 22.9 vs 22.5. BUT factorisation fits calibration maps
  only to ~4.6 C, so the physics family is too restrictive. Static background map now added (untested on real data).
  NEXT: re-check factor_rmse with background + K in {8,16}; then launch `scripts/run_protocol.py` (resumable).
- Checkpointing: protocol tasks -> runs/protocol/results.jsonl; mid-fit ckpts for hybrid factorisation and NN
  baselines; audit/gates save per config.

## M0: scaffold, data, splits, metrics, baselines
- [x] T0.1 pyproject, package skeleton, pytest config, CI (CPU)
- [x] T0.2 `data/safe_pickle.py` (whitelist unpickler) + malicious-pickle test
- [x] T0.3 `data/ucr.py`: load the i7/i5 excerpts → cached `.npz` in `data/cache/`
- [x] T0.4 `data/splits.py`: S-RAND, S-CAL-k (fixed seed draws), S-LOWO, calibration validation tail; overlap tests
- [x] T0.5 `data/features.py`: calibration-only z-score, sensor-column rule (D-004)
- [x] T0.6 `evaluation/metrics.py` (per-map RMSE, peak, hotspot location, IoU, corr) + `stats.py` (hierarchical bootstrap)
- [x] T0.7 `models/baselines.py`: B-MEAN, B-COPY, B-RIDGE (inner-CV λ and k)
- [x] T0.8 `scripts/audit_data.py` → `results/leakage.json` + leakage figure

## M1: physics
- [x] T1.1 `physics/dct.py` (torch DCT-II/III via FFT) + scipy cross-check
- [x] T1.2 `physics/spectral.py` H-spec steady and transient
- [x] T1.3 `physics/fv.py` 1-layer and 2-layer FV assembly + solve; FV == H-spec test
- [x] T1.4 `physics/inversion.py` Laplacian inversion with denoising
- [x] T1.5 `physics/powermaps.py` random simulation power maps

## M2: gates
- [~] T2.1 G1 representation floor (per-frame non-negative fit) on calibration frames
- [ ] T2.2 G2 layout consistency of inverted maps
- [ ] T2.3 record outcomes in DECISIONS.md

## M3: hybrid (H-spec) + baselines + protocol
- [x] T3.1 `models/translator.py` (basis maps + coefficients + T_base)
- [x] T3.2 `training/calibrate.py` through the frozen heat model
- [x] T3.3 B-MLP, B-CNN, B-TF
- [~] T3.4 `evaluation/protocol.py` S-CAL-k runner, k ∈ {1, 3, 5, 10} × 20 draws, ± sensors
- [ ] T3.5 REF-ALL (S-LOWO) for the pure-ML baselines

## M4: H-net
- [ ] T4.1 `scripts/gen_sim.py` domain-randomised 2-layer simulation dataset
- [ ] T4.2 `models/heatnet.py` FNO / U-Net + `scripts/train_heatnet.py` (resumable)
- [ ] T4.3 measure local training speed → local or DGX (G3)
- [ ] T4.4 hybrid with H-net through the protocol

## M5: UQ, ablations, latency
- [ ] T5.1 translator ensembles; workload-level conformal; coverage and AUROC
- [ ] T5.2 ablations: window w, K basis maps, transient H-spec
- [ ] T5.3 latency (CPU single-sample)

## M6: report
- [ ] T6.1 figures from `results/`
- [ ] T6.2 README results + technical report

## Blocked / waiting on owner
- (none)

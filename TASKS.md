# Tasks

Status: `[ ]` pending · `[~]` active · `[x]` done · `[!]` blocked

## M0: scaffold, data, splits, metrics, baselines
- [~] T0.1 pyproject, package skeleton, pytest config, CI (CPU)
- [ ] T0.2 `data/safe_pickle.py` (whitelist unpickler) + malicious-pickle test
- [ ] T0.3 `data/ucr.py`: load the i7/i5 excerpts → cached `.npz` in `data/cache/`
- [ ] T0.4 `data/splits.py`: S-RAND, S-CAL-k (fixed seed draws), S-LOWO, calibration validation tail; overlap tests
- [ ] T0.5 `data/features.py`: calibration-only z-score, sensor-column rule (D-004)
- [ ] T0.6 `evaluation/metrics.py` (per-map RMSE, peak, hotspot location, IoU, corr) + `stats.py` (hierarchical bootstrap)
- [ ] T0.7 `models/baselines.py`: B-MEAN, B-COPY, B-RIDGE (inner-CV λ and k)
- [ ] T0.8 `scripts/audit_data.py` → `results/leakage.json` + leakage figure

## M1: physics
- [ ] T1.1 `physics/dct.py` (torch DCT-II/III via FFT) + scipy cross-check
- [ ] T1.2 `physics/spectral.py` H-spec steady and transient
- [ ] T1.3 `physics/fv.py` 1-layer and 2-layer FV assembly + solve; FV == H-spec test
- [ ] T1.4 `physics/inversion.py` Laplacian inversion with denoising
- [ ] T1.5 `physics/powermaps.py` random simulation power maps

## M2: gates
- [ ] T2.1 G1 representation floor (per-frame non-negative fit) on calibration frames
- [ ] T2.2 G2 layout consistency of inverted maps
- [ ] T2.3 record outcomes in DECISIONS.md

## M3: hybrid (H-spec) + baselines + protocol
- [ ] T3.1 `models/translator.py` (basis maps + coefficients + T_base)
- [ ] T3.2 `training/calibrate.py` through the frozen heat model
- [ ] T3.3 B-MLP, B-CNN, B-TF
- [ ] T3.4 `evaluation/protocol.py` S-CAL-k runner, k ∈ {1, 3, 5, 10} × 20 draws, ± sensors
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

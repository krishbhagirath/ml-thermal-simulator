# Research notes

## Dataset and prior work (verified by reading full texts or abstracts; see the plan history for details)
- **Dataset:** Lu & Tan, "Thermal Map Dataset for Commercial Multi/Many Core CPU/GPU/TPU", MLCAD 2024, DOI 10.1145/3670474.3685963. Repo: https://github.com/sheldonucr/commercial_thermal_map_dataset. Public files are excerpts of 100 samples each; the full data is on request; no licence.
- **ThermTransformer** (Lu, Zhang & Tan, ICCAD'23, DOI 10.1109/ICCAD57390.2023.10323817): R7 4800U, 158 uProf metrics, window of 100 at 10 Hz, 223×280 maps. 0.360 °C mean per-map RMSE [paper-reported]. **Split method not stated.**
- **ThermGAN** (Jin et al., ICCAD'20; NSF PAR 10393031): i7-8650U, 170 PCM inputs, random 25% test 0.47 °C, held-out Gimp 0.83 °C [paper-reported]. Core temperature inputs are the most important.
- **RealMaps** (Sadiqbatcha et al., IEEE TC 2021; NSF PAR 10324767) and the ASP-DAC'20 LSTM (NSF PAR 10195478): LSTM + DCT output; tested on 17 newly recorded unseen workloads (not public).
- **Power-density maps** (Zhang et al., DATE'20; NSF PAR 10195484): steady Laplacian inversion p = κΔz(−∇²T)₊, κ ≈ 174 W/m·K; COMSOL validation 1.8 °C RMSE, h ≈ 600 W/m²K [paper-reported].
- **Lu PhD dissertation** (UCR 2026, https://escholarship.org/uc/item/2bh104zn): warns about random-shuffle leakage; proposes a decoupled counters → power map → steady spectral solver pipeline (simulated only); transient modelling left as future work.
- **TherMapNet** (arXiv 2608.21887, Aug 2026): Transformer + dual CBAM CNN, 0.251 °C R7 [paper-reported]; split not stated; no code, no UQ, no held-out workloads.
- **Closest physics-inverse prior art:** TCPMT'26 transient inverse with a frozen U-Net surrogate (abstract only, DOI 10.1109/TCPMT.2026.3667684). Not telemetry-driven.

## Sim-to-real and physics background
- Domain randomisation is the standard sim-to-real remedy: train on randomised simulator parameters so the real system falls inside the training range.
- FNO (Li et al., arXiv 2010.08895). PDEBench (arXiv 2210.07182) reports U-Net > FNO on 128² Darcy [paper-reported].
- McGreivy & Hakim 2024 (arXiv 2407.07218): weak baselines are common in ML-for-PDE work, so we use strong, tuned baselines.

## Hypotheses (from PROJECT_PLAN.md)
1. Random-frame splits hide most of the error (leakage).
2. A simulation-trained physics model can represent real maps to within a small floor (G1).
3. With k ≤ 5 real workloads, the hybrid beats pure ML on unseen workloads.
4. Pure ML needs many more real workloads to catch up.
5. Workload-level conformal intervals are calibrated on unseen workloads; frame-level ones are not.

## Open questions
- Frame interval of the excerpt (needed for transient modelling).
- Column semantics of the 170 i7 features (only inferred via the data rule).

## Negative results
- (none yet)

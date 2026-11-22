# Experiments registry

Columns: ID · date · what · config / seed · data · where · result (with label) · notes.

| ID | Date | What | Data | Where | Result | Notes |
|---|---|---|---|---|---|---|
| A-001 | 2026-11-20 | FV solver prototype: manufactured solution, energy, timing | synthetic | laptop | order 2.00; energy closure 4.7e-14; 83 ms per 128² solve [measured locally] | scratch prototype; reused in `physics/fv.py` |
| A-002 | 2026-11-20 | Harmonic vs arithmetic face conductivity, 1:400 interface | synthetic | laptop | harmonic order 2.00 (1.8e-4 at N = 128); arithmetic order 0.99 (2.0e-2) [measured locally] | |
| A-003 | 2026-11-20 | Public excerpt shapes (whitelist loader) | UCR public | laptop | i7 19 × 100 frames, 170 features, 185×154; R7 100 windows | |
| A-004 | 2026-11-20 | Leakage audit, i7 | UCR i7 | laptop | copy-nearest (random split) 0.37 °C; consecutive-frame change 0.29 °C [measured locally] | ridge numbers retracted (D-005) |
| A-005 | 2026-11-21 | Feature profiling, i7 | UCR i7 | laptop | ~9 sensor-like columns, corr −0.98 with mean temperature; within-workload drift up to 25 °C [measured locally] | |
| A-006 | 2026-11-21 | Gate G1 (37x31 per-frame non-negative power fit through H-spec, 380 frames) | UCR i7 | laptop | physics floor median 0.495 C (p90 0.774); smooth-surface control 0.073 C; h = 13,666 W/m2K, l = 2.34 mm [measured locally] | G1 passes (<= 1.0 C). The 4.6 C shared-layout factorisation error therefore comes from the K = 8 shared-basis constraint, not the physics |
| A-007 | 2026-11-22 | Gate G1 at 19x16 basis + Gate G2 layout consistency | UCR i7 | laptop | G1 19x16 floor 0.498 C (control 0.101 C), same h; G2 cosine similarity of per-workload inverted power maps mean 0.808, min 0.512 [measured locally] | results/gates.json, results/figures/g1_worst_fit.png, g2_power_maps.png |

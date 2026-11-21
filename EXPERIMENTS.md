# Experiments registry

Columns: ID · date · what · config / seed · data · where · result (with label) · notes.

| ID | Date | What | Data | Where | Result | Notes |
|---|---|---|---|---|---|---|
| A-001 | 2026-11-20 | FV solver prototype: manufactured solution, energy, timing | synthetic | laptop | order 2.00; energy closure 4.7e-14; 83 ms per 128² solve [measured locally] | scratch prototype; reused in `physics/fv.py` |
| A-002 | 2026-11-20 | Harmonic vs arithmetic face conductivity, 1:400 interface | synthetic | laptop | harmonic order 2.00 (1.8e-4 at N = 128); arithmetic order 0.99 (2.0e-2) [measured locally] | |
| A-003 | 2026-11-20 | Public excerpt shapes (whitelist loader) | UCR public | laptop | i7 19 × 100 frames, 170 features, 185×154; R7 100 windows | |
| A-004 | 2026-11-20 | Leakage audit, i7 | UCR i7 | laptop | copy-nearest (random split) 0.37 °C; consecutive-frame change 0.29 °C [measured locally] | ridge numbers retracted (D-005) |
| A-005 | 2026-11-21 | Feature profiling, i7 | UCR i7 | laptop | ~9 sensor-like columns, corr −0.98 with mean temperature; within-workload drift up to 25 °C [measured locally] | |

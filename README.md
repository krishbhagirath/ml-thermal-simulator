# ml-thermal-simulator

Research code that predicts the **full-chip temperature map of a real processor** (Intel i7-8650U) from its live activity telemetry. The approach is sim-to-real:

```
telemetry ──► translator (calibrated on a few real workloads) ──► power map ──► heat model (trained on physics simulation) ──► temperature map
```

- **Real data:** public infrared thermal-map excerpts from UC Riverside's [Commercial Thermal Map Dataset](https://github.com/sheldonucr/commercial_thermal_map_dataset): 19 workloads × 100 frames. It's used only for calibration (k workloads) and **testing on unseen workloads**.
- **Physics:** a verified finite-volume heat solver and an exact spectral (DCT) solver. A neural operator is trained on domain-randomised simulations.
- **Baselines:** tuned ridge, MLP, CNN and a small transformer, all trained on the same calibration workloads. Plus a leakage analysis of random-frame splits.

**Status:** under active development. See [PROJECT_PLAN.md](PROJECT_PLAN.md) and [TASKS.md](TASKS.md). Results will appear here once generated from `results/`.

## Quick start (CPU)
```bash
pip install numpy scipy matplotlib PyYAML pytest torch
git clone https://github.com/sheldonucr/commercial_thermal_map_dataset.git data/raw/ucr_commercial_thermal
python -m pytest -q
```

## Data notice
The dataset belongs to its authors (Lu & Tan, MLCAD 2024) and has no licence file, so this repo **does not redistribute any data**. Clone it from the source above. The pickles are loaded only through a whitelist unpickler (`thermal_research.data.safe_pickle`).

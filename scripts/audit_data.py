"""M0 leakage audit on the public i7 excerpt.

Compares random-frame splits (S-RAND, 5 seeds) with leave-one-workload-out (S-LOWO, 19 folds) for
non-DL baselines, with and without on-die temperature-sensor columns.
Writes results/leakage.json and results/figures/leakage.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from thermal_research.data import splits, ucr  # noqa: E402
from thermal_research.evaluation import protocol, stats  # noqa: E402
from thermal_research.models.baselines import CopyNearest, MeanMap, RidgePCA  # noqa: E402
from thermal_research.training.checkpoint import JsonProgress  # noqa: E402
from thermal_research.viz import figures  # noqa: E402


def main() -> None:
    cd = ucr.load_chip("i7")
    print(f"i7: {cd.n} frames, {len(cd.workloads)} workloads, X{cd.X.shape}, maps {cd.shape}")
    models = {"mean": MeanMap, "ridge": RidgePCA, "copy-nearest": CopyNearest}
    out: dict = {"chip": "i7", "n_frames": cd.n, "results": {}}
    progress = JsonProgress(ROOT / "runs" / "audit" / "progress.json")  # resume after disconnects

    rand_splits = [splits.random_frames(cd.n, 0.2, seed=s) for s in range(5)]
    lowo_splits = [splits.lowo(cd.group, cd.frame, held=g) for g in range(len(cd.workloads))]

    for split_name, split_list in (("S-RAND", rand_splits), ("S-LOWO", lowo_splits)):
        for mname, factory in models.items():
            if mname == "copy-nearest" and split_name == "S-LOWO":
                continue
            for use_sensors in (True, False):
                if mname != "ridge" and not use_sensors:
                    continue  # mean / copy don't use telemetry
                key = f"{split_name}|{mname}|{'sensors' if use_sensors else 'no-sensors'}"
                if key in progress:
                    out["results"][key] = progress[key]
                    print(f"{key:40s} (cached)")
                    continue
                res = [protocol.evaluate_split(factory, cd, s, use_sensors=use_sensors) for s in split_list]
                summary = {m: stats.hier_bootstrap(protocol.collect(res, m), n_boot=2000)
                           for m in ("rmse", "peak_err", "hotspot_mm", "corr")}
                summary["n_sensor_cols"] = [r["n_sensor_cols"] for r in res]
                summary["chosen"] = [r["chosen"] for r in res]
                if split_name == "S-LOWO":
                    summary["per_workload_rmse"] = {cd.workloads[w]: v["rmse"] for r in res
                                                    for w, v in r["per_workload"].items()}
                out["results"][key] = summary
                progress.put(key, summary)
                r = summary["rmse"]
                print(f"{key:40s} RMSE {r['mean']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] C   "
                      f"peak {summary['peak_err']['mean']:.2f} C")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "leakage.json").write_text(json.dumps(out, indent=2))

    def get(key):
        r = out["results"][key]["rmse"]
        return (r["mean"], r["lo"], r["hi"])

    nan = (np.nan, np.nan, np.nan)
    figures.grouped_bars(
        ["random frames\n(S-RAND)", "unseen workload\n(S-LOWO)"],
        {
            "copy nearest frame (no model)": [get("S-RAND|copy-nearest|sensors"), nan],
            "ridge, with sensors": [get("S-RAND|ridge|sensors"), get("S-LOWO|ridge|sensors")],
            "ridge, no sensors": [get("S-RAND|ridge|no-sensors"), get("S-LOWO|ridge|no-sensors")],
            "mean map": [get("S-RAND|mean|sensors"), get("S-LOWO|mean|sensors")],
        },
        "per-map RMSE (°C), 95% CI", "i7 excerpt: random-frame vs unseen-workload evaluation",
        ROOT / "results" / "figures" / "leakage.png",
    )
    print("wrote results/leakage.json and results/figures/leakage.png")


if __name__ == "__main__":
    main()

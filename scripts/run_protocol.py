"""Main experiment: S-CAL-k (calibrate on k workloads, test on the rest) for every model, +/- sensors,
plus S-LOWO reference runs. Resumable: each finished (model, split, k, draw, sensors) is appended to
runs/protocol/results.jsonl and skipped on re-run.

Examples:
  python scripts/run_protocol.py --models ridge,hyb-spec --ks 3,5 --draws 20 --workers 4
  python scripts/run_protocol.py --summarize
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUN_DIR = ROOT / "runs" / "protocol"
CACHE = RUN_DIR / "results.jsonl"
PITCH = 50e-6
G_FIXED = 150 * 0.5e-3


def make_factory(name: str, out_hw):
    from thermal_research.models.baselines import MeanMap, RidgePCA
    from thermal_research.models.blackbox import TorchRegressor

    if name == "mean":
        return MeanMap
    if name == "ridge":
        return RidgePCA
    if name in ("mlp", "cnn", "tf"):
        return lambda: TorchRegressor(name, out_hw)
    if name.startswith("hyb-spec-e2e"):  # end-to-end translator (kept as an ablation)
        from thermal_research.models.translator import SpecOperator
        from thermal_research.physics.spectral import SpectralHeat
        from thermal_research.training.calibrate import CalibConfig, TranslatorModel

        cfg = CalibConfig(K=8)
        return lambda: TranslatorModel(lambda: SpecOperator(SpectralHeat(*out_hw, PITCH), G_FIXED, 1000.0),
                                       cfg, out_hw, PITCH, G_FIXED, 1000.0)
    if name.startswith("hyb-spec"):  # two-stage: physics factorisation + tuned ridge (main hybrid)
        from thermal_research.models.hybrid import FactorConfig, HybridRidge
        from thermal_research.models.translator import SpecOperator
        from thermal_research.physics.spectral import SpectralHeat

        K = int(name.split("-K")[1]) if "-K" in name else 8
        return lambda: HybridRidge(lambda: SpecOperator(SpectralHeat(*out_hw, PITCH), G_FIXED, 1000.0),
                                   PITCH, G_FIXED, 1000.0, FactorConfig(K=K))
    if name.startswith("hyb-net"):
        from thermal_research.models.heatnet import load_heat_operator
        from thermal_research.training.calibrate import CalibConfig, TranslatorModel

        cfg = CalibConfig(K=8)
        return lambda: TranslatorModel(lambda: load_heat_operator(ROOT / "runs" / "heatnet" / "best.pt", out_hw),
                                       cfg, out_hw, PITCH, G_FIXED, 1000.0)
    raise ValueError(name)


def task_key(t: dict) -> str:
    return f"{t['model']}|{t['split']}|k{t['k']}|d{t['draw']}|{'s' if t['sensors'] else 'ns'}|w{t['w']}"


def build_tasks(models, ks, n_draws, sensors_opts, w, lowo_models, n_workloads):
    from thermal_research.data import splits

    tasks = []
    for k in ks:
        draws = splits.calibration_draws(n_workloads, k, n_draws)
        for d, cal in enumerate(draws):
            for s in sensors_opts:
                for m in models:
                    tasks.append({"model": m, "split": "S-CAL", "k": k, "draw": d, "cal": list(cal), "sensors": s, "w": w})
    for m in lowo_models:
        for held in range(n_workloads):
            for s in sensors_opts:
                tasks.append({"model": m, "split": "S-LOWO", "k": n_workloads - 1, "draw": held, "cal": [held],
                              "sensors": s, "w": w})
    return tasks


_CD = None


def _init_worker(threads: int):
    global _CD
    import torch

    torch.set_num_threads(threads)
    from thermal_research.data import ucr

    _CD = ucr.load_chip("i7")


def run_task(t: dict) -> dict:
    from thermal_research.data import splits
    from thermal_research.evaluation import protocol

    cd = _CD
    if t["split"] == "S-CAL":
        sp = splits.calibration_split(cd.group, cd.frame, tuple(t["cal"]))
    else:
        sp = splits.lowo(cd.group, cd.frame, held=t["cal"][0])
    base_factory = make_factory(t["model"], cd.shape)
    ckpt = RUN_DIR / "ckpt" / (task_key(t).replace("|", "__") + ".pt")

    def factory():  # attach a per-task checkpoint so long fits resume mid-way after a disconnect
        m = base_factory()
        if hasattr(m, "ckpt_path"):
            m.ckpt_path = ckpt
        return m

    t0 = time.perf_counter()
    r = protocol.evaluate_split(factory, cd, sp, use_sensors=t["sensors"], w=t["w"])
    r["per_workload"] = {str(k): v for k, v in r["per_workload"].items()}
    return {**t, "key": task_key(t), "seconds": time.perf_counter() - t0, **r}


def done_keys() -> set[str]:
    if not CACHE.exists():
        return set()
    return {json.loads(line)["key"] for line in CACHE.read_text().splitlines() if line.strip()}


def run(args):
    from thermal_research.data import ucr

    n_w = len(ucr.load_chip("i7").workloads)
    tasks = build_tasks(args.models, args.ks, args.draws, args.sensors, args.w, args.lowo_models, n_w)
    finished = done_keys()
    todo = [t for t in tasks if task_key(t) not in finished]
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} cached, {len(todo)} to run", flush=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if args.workers <= 1:
        _init_worker(args.threads)
        it = map(run_task, todo)
    else:
        import multiprocessing as mp

        pool = mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(args.threads,))
        it = pool.imap_unordered(run_task, todo)
    with open(CACHE, "a") as f:
        for i, r in enumerate(it, 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {r['key']}: rmse {r['frame_mean']['rmse']:.3f} C ({r['seconds']:.0f}s)", flush=True)


def summarize():
    import numpy as np

    from thermal_research.evaluation import stats

    rows = [json.loads(line) for line in CACHE.read_text().splitlines() if line.strip()]
    groups: dict[tuple, dict] = {}
    for r in rows:
        g = (r["model"], r["split"], r["k"], r["sensors"], r["w"])
        groups.setdefault(g, {})[r["draw"]] = r["per_workload"]
    summary = {}
    for g, draws in sorted(groups.items()):
        entry = {"n_draws": len(draws)}
        for metric in ("rmse", "peak_err", "hotspot_mm", "corr"):
            vals = {d: {int(w): m[metric] for w, m in pw.items()} for d, pw in draws.items()}
            if g[1] == "S-LOWO":  # one draw per held-out workload -> pool into a single "draw"
                vals = {0: {w: v for dd in vals.values() for w, v in dd.items()}}
            entry[metric] = stats.hier_bootstrap(vals, n_boot=2000)
        summary["|".join(map(str, g))] = entry
    # paired comparisons: each hybrid vs each baseline on identical (k, draw, workload, sensors)
    paired = {}
    models = sorted({g[0] for g in groups})
    for hyb in [m for m in models if m.startswith("hyb")]:
        for base in [m for m in models if not m.startswith("hyb")]:
            for (m, sp, k, s, w), draws in groups.items():
                if m != hyb or sp != "S-CAL" or (base, sp, k, s, w) not in groups:
                    continue
                a = {d: {int(x): v["rmse"] for x, v in pw.items()} for d, pw in draws.items()}
                b = {d: {int(x): v["rmse"] for x, v in pw.items()} for d, pw in groups[(base, sp, k, s, w)].items()}
                paired[f"{hyb} - {base}|k{k}|{'sensors' if s else 'no-sensors'}"] = stats.paired_diff(a, b, n_boot=2000)
    out = {"summary": summary, "paired_rmse": paired}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "protocol.json").write_text(json.dumps(out, indent=2))
    for k_, v in summary.items():
        r = v["rmse"]
        print(f"{k_:40s} n={v['n_draws']:2d}  RMSE {r['mean']:6.2f} [{r['lo']:.2f}, {r['hi']:.2f}]  "
              f"peak {v['peak_err']['mean']:.2f}  hotspot {v['hotspot_mm']['mean']:.2f} mm")
    for k_, v in paired.items():
        print(f"{k_:45s} diff {v['mean']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]  a-better {v['frac_a_better']:.2f}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="mean,ridge,mlp,tf,hyb-spec")
    p.add_argument("--lowo-models", default="")
    p.add_argument("--ks", default="1,3,5,10")
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--sensors", default="0,1")
    p.add_argument("--w", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    if a.summarize:
        summarize()
        return
    a.models = [m for m in a.models.split(",") if m]
    a.lowo_models = [m for m in a.lowo_models.split(",") if m]
    a.ks = [int(k) for k in a.ks.split(",")]
    a.sensors = [bool(int(s)) for s in a.sensors.split(",")]
    run(a)


if __name__ == "__main__":
    main()

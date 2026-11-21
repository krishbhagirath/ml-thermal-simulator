# Project instructions for Claude Code

## Session-start protocol (do this first, every session)
1. Read `PROJECT_PLAN.md`, then `TASKS.md` (find the active milestone and blocked items), `DECISIONS.md` and `EXPERIMENTS.md`.
2. Run `git status` and `git log --oneline -15` on branch `dev`.
3. Run `python -m pytest -q -m "not slow"` to confirm the tree is green.
4. Continue with the highest-priority unblocked task in `TASKS.md`. Do not re-ask the owner things already decided in `DECISIONS.md`.

## Git conventions (strict)
- Work on `dev`. Commit and push regularly with **informal** messages.
- Author is `krishbhagirath` (repo-local git config). **Never add `Co-Authored-By` or any Claude/AI attribution** to commits, PRs, or files.
- Update `main` only when the owner says so.
- Never commit `data/`, `runs/`, `*.pkl`, `*.pt`, `*.npz` caches. Only small summaries and figures go in `results/`.

## Data safety
- The UC Riverside pickles are untrusted. Load them **only** via `thermal_research.data.safe_pickle` (a whitelist unpickler).
- The dataset has no licence. Never redistribute raw or derived data.

## Research integrity
- Label numbers as measured, paper-reported or estimate. Never fabricate results.
- Test workloads are never used for any selection or tuning. Calibration draws are fixed seed lists shared by all models.
- Every number in the README/report comes from a script reading `results/`.

## Compute
- Default to local CPU. When the DGX A100 is needed, write a **DGX SERVER ACTION REQUIRED** block (purpose, single-line commands, resources, monitoring, outputs to send back) in chat and in `SERVER_RUNBOOK.md`.
- Server commands must be single-line (chain with `&&`), use absolute paths, and go in tmux for long jobs. One GPU only.
- Never upgrade torch on the server (2.10.0+cu128). Use a venv with `--system-site-packages` plus a constraints file.

## Owner preferences
- Answers: TL;DR bullets first, then detail. Plain language for ML concepts.
- One clear recommendation, not a menu. Make low-stakes calls yourself; surface real strategic forks.

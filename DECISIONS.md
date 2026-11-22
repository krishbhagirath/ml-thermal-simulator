# Decisions log

Format: ID · date · decision · alternatives · why · trade-offs.

**D-001 · 2026-11-20 · Drop the synthetic-only layout plan (v0).**
- Alternatives: keep synthetic chiplet layouts plus surrogate optimization.
- Why: the owner wants real measured data as the evidence. Synthetic-only results support method claims, not hardware claims.
- Trade-off: lose the optimization study. The verified FV solver from v0 is reused.

**D-002 · 2026-11-21 · Don't depend on the full dataset or the authors (v1 superseded).**
- Alternatives: request the full R7 data and reproduce ThermTransformer.
- Why: the public excerpt has 100 samples per file, and there is no response channel the owner wants to rely on.
- Trade-off: no faithful reproduction of the 0.36 °C R7 result.

**D-003 · 2026-11-21 · Sim-to-real hybrid on the public i7 excerpt.**
- Design: physics (simulation-trained) heat model plus a tiny translator calibrated on k real workloads; test on the remaining workloads.
- Alternatives: pure ML on 1,900 frames; switch to a battery dataset.
- Why: real data is mostly kept for testing; physics supplies spatial structure; it answers a clear sim-to-real question.
- Trade-off: depends on how well simulation matches the rig (Gate G1).

**D-004 · 2026-11-21 · On-die temperature-sensor columns are detected by a data rule** (integer-valued, range ≤ 110, |corr| > 0.8 with mean map temperature), computed on calibration workloads only. Every headline result is reported with and without them.
- Why: about 9 columns correlate at −0.98 with the mean map temperature [measured locally]; ThermGAN found sensors to be the top inputs [paper-reported].

**D-005 · 2026-11-21 · The audit's "ridge LOWO 8.0 °C" number is retracted as a baseline.**
- Why: it used fixed λ = 10·n, which is excessive. M0's ridge tunes λ and the PCA rank by inner CV on calibration workloads.

**D-006 · 2026-11-21 · H-spec uses relative power without an absolute anchor on the i7.**
- Why: the column names are unknown, so no reliable power column exists. Only q/h and √(G/h) affect θ. G is fixed and h and q's scale are fitted.
- Trade-off: power maps are relative, not in watts.

**D-007 · 2026-11-21 · Pixel pitch is fixed at 50 µm** for the i7 maps [paper-reported], giving a map of 9.25 × 7.70 mm.

**D-008 · 2026-11-21 · Work on branch `dev`; author `krishbhagirath`; no co-author trailers anywhere.** The initial commit was rewritten to remove a Claude co-author trailer, which needed a one-time force-push of `main`.

**D-009 · 2026-11-22 · Gates G1 and G2 passed; the shared-layout factorisation needs more capacity.**
- G1: per-frame physics floor about 0.50 °C at both 37×31 and 19×16 basis resolution, against a 1.0 °C threshold, so H-spec is adequate.
- G2: per-workload power maps share hot regions (mean cosine 0.81), but some differ (min 0.51), for example single-core vs all-core workloads.
- Action: raise the factorisation's K from 8 to 16 or more, and initialise h near the G1 fit scale (~1e4, a generic prior, not fitted to test data). The 4.6 °C shared-basis error is a capacity/initialisation issue, not a physics failure.

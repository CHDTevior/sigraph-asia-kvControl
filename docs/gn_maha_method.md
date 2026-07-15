# GN-Maha: Anchor-Space Control via Codebook-Metric Gauss–Newton

An **analytic-step iterative solver** — a damped Gauss–Newton / Levenberg–Marquardt method whose
trial updates solve a linearized Mahalanobis least-squares subproblem in closed form. It is *not*
a one-shot closed-form solver: steps are damped, trial-based, and rollback-guarded.

**Status**: validated 2026-07-13/14 (5-repeat formal eval, test split, MaskControl protocol).
**Scope**: replaces the inherited 2,525-iteration two-stage test-time optimization (M3) for
**pelvis-trajectory control at any density** (the regimes tested). Multi-joint control is NOT
claimed — see §5 for the measured boundary and the full attempted-fix matrix.
**Training required**: none. Everything below runs on the released frozen checkpoints.

---

## 1. Problem

Trajectory-controlled text-to-motion on a frozen masked transformer (PartVQ + T-Concat) with a
K/V control adapter. At inference the user supplies **anchors** — (frame, joint, xyz) targets —
and the decoded motion must pass through them.

The inherited pipeline (from MaskControl) achieves this with test-time optimization:

- **Stage-1**: at each of 10 unmask steps, `(s+1)×35` Adam iterations on the token logits
  (1,925 total);
- **Stage-2**: 600 Adam iterations on the decoded continuous embeddings;
- total ≈ **2,525 iterations, tens of seconds per sample**, KPS 0.40 ± 0.02 cm.

Per-iteration probes (`scripts/a1_trace_probe.py`, 32 train-split clips) show this budget is
structurally misallocated *given our adapter's strong feed-forward start* (M0 ≈ 30 cm vs
MaskControl's 40 cm):

| finding | measured |
|---|---|
| samples entering Stage-2 already below 1e-5 m² anchor MSE | **97%** |
| samples whose Stage-2 EXIT is *worse* than its best iterate | **75%** |
| iterations to 95% of each unmask step's gain (schedule allocates 35→350) | median **12–23** |
| accuracy lost by deleting Stage-1 outright (S2-only probe) | **none** |

## 2. Key observation

Because the adapter already places the feed-forward output near the data manifold, "control"
degenerates from an optimization problem into a **tiny nonlinear least-squares problem**:

- residual `r(e) ∈ R^{3A}` — anchor position errors (A anchors; 5-anchor control ⇒ **15 dims**);
- variables `e ∈ R^{~37,632}` — the six quantizers' decoded embeddings (6 × 128 × 49).

Massively underdetermined ⇒ solve it **analytically in the dual (anchor) space**.

## 3. Method

One damped Gauss–Newton / Levenberg–Marquardt step:

```
Δ = − Σ Jᵀ ( J Σ Jᵀ + λ·s·I )⁻¹ r
```

| symbol | meaning | cost |
|---|---|---|
| `r` (3A,) | anchor residual at current embeddings | 1 decoder forward |
| `J` (3A × P) | Jacobian, one `autograd.grad` per residual component | 3A backward passes |
| `J Σ Jᵀ` (3A × 3A) | dual-space normal matrix | trivial |
| linear solve | a **15×15 system** at 5-anchor control | microseconds |
| `λ` | LM damping, trust-region adapted (accept ⇒ ×0.5, reject ⇒ ×10, exact-restore rollback) | — |
| stop | full-mask anchor MSE < τ (default 1e-5 m²) or ≤ 8 steps | — |

Dense control caps the Jacobian at `anchor_cap=48` rows (evenly-spaced collocation), but
**acceptance / stopping / reporting always use the full mask** so the solver cannot overfit the
collocation subset.

### 3.1 Why the metric Σ is the heart of the method

With the plain Euclidean metric (`Σ = I`, minimum-L2-norm step) anchors are nailed
(KPS 0.13 cm) but **FID collapses 0.065 → 0.137**. Diagnosis (each alternative excluded by a
dedicated control experiment — batching artifact, anchor-local discontinuity, stopping time,
step size): the minimum-L2 direction is **agnostic to the decoder's data manifold**. Adam's
thousands of small steps implicitly follow the loss geometry (costing only +0.016 FID); a single
analytic jump to the constraint surface has no such protection.

Fix: measure "smallest change" with a **Mahalanobis norm** whose metric the model already owns.
For each quantizer `q`, let `Σ_q` be the covariance of its 128 codebook vectors
(+`ridge·mean(diag)·I`, scale-normalized). Directions along which the codebook has spread are
directions the decoder was trained on — they are cheap; directions the codebook never spans are
expensive:

```
min ‖Δ‖²_{Σ⁻¹}   s.t.  JΔ = −r      ⇒      Δ = −Σ Jᵀ (J Σ Jᵀ + λsI)⁻¹ r
```

Block-diagonal, position-independent, **zero training** — Σ comes from the frozen VQ.

Matched-pair ablation (same τ, same resulting KPS 0.25 cm, the metric is the *only* variable):

| metric | KPS | FID |
|---|---|---|
| Euclidean (Σ=I) | 0.25 | 0.134 |
| **codebook covariance** | 0.25 | **0.062–0.070** |

τ ∈ {1e-5, 3e-5} leaves FID unchanged; heavier ridge (more isotropic ⇒ closer to L2)
monotonically degrades FID — the direction of both trends matches the mechanism.

## 4. Results (formal protocol: test split, bs=32 metric groups, density mixture, H100)

| | M3 (2,525 iters) | **GN-Maha (≤8 solves)** |
|---|---|---|
| KPS (5-repeat) | 0.40 ± 0.02 cm | **0.26 ± 0.03 cm** |
| FID (5-repeat) | 0.065 | **0.062 ± 0.007** (comparable point estimates) |
| Top-3 | 0.799 | **0.799 ± 0.008** |
| foot-skate | 0.0444 | 0.0497 (+12% — the one residual regression; for context, M3 itself sits at 0.0495 on multi-joint) |

### 4.1 Formal latency (A100, same 16 picks, CUDA-synchronized, matched pair)

| keyframes | GN-Maha median (fast J / row loop) | M3 median | speedup |
|---|---|---|---|
| 1 | **0.50 s** / 0.50 s | 15.0 s (± 0.35) | **30×** |
| 5 | **0.50 s** / 0.60 s | 18.8 s (± 0.48) | **38×** |
| dense (per-frame) | **0.60 s** (p90 0.7) / 1.70 s | 18.2 s (± 0.50) | **30×** |

(The earlier "~61 s" M3 figure came from a different measurement path; all paper numbers use
this same-GPU matched-pair table.)

**Fast Jacobian implementation** (`--gn_jac_chunk 64`): the per-row backward loop is replaced
by a replicated-batch backward (m rows per decode+backward). Adjudicated footnote: "Latency is
measured using a numerically optimized Jacobian implementation; its Jacobian differs from the
reference by at most 1.48e-9 elementwise and yields metric-equivalent results under the
identical five-repeat evaluation protocol" (fast-path 5r: KPS 0.260 ± 0.013, FID 0.0637 ±
0.0048, Top3 0.7996 ± 0.0064, skate 0.0496 — 0.00–0.20σ from the reference means). Wall-clock:
2.8× at dense (1.70 → 0.60 s); full-protocol eval drops from ~9 h to ~70 min per repeat.
Default OFF — published metric numbers are reproduced bit-exact on the row loop.

### 4.2 Fixed-density stress tests (beyond the standard mixture)

| density | M3 (5r) | GN-Maha (5r) |
|---|---|---|
| 49 | KPS 0.80, FID 0.1267 ± 0.0042 | **KPS 0.304 ± 0.015**, FID 0.1293 ± 0.0049 |
| 196 | KPS 0.70, FID 0.1745 ± 0.0075 | **KPS 0.340**, FID 0.1749 ± 0.0070 |

The FID rise with density is a property of the *regime* (the 2,525-iteration baseline shows the
same trend on its own: 0.065 → 0.127 → 0.175); GN keeps a 2×+ KPS advantage throughout with
matching FID.

Skeleton rigidity on the picks *improves* (worst clip 79 → 24 mm bone-length drift).
Frame-level visual QA (6 clips incl. the worst tail samples): no pose degradation; see
`output/aaai_pareto_sweep/visqa/sbs_*.mp4`.

Honest caveats: foot-skate +12% is the one residual quality regression (temporal artifact,
being reviewed on video); "restores most of the realism lost by Euclidean GN" is the
supported claim — baseline-FID equivalence rests on 5-repeat overlap, not superiority.
The solver returns the best observed iterate of its executed prefix; there is no guarantee
against a longer optimizer run (empirically it wins).

## 5. Multi-joint boundary (measured, honest scoping)

On cross control (up to 6 joints simultaneously, standard protocol) the feed-forward start is
much worse (40.6 cm vs 29.9 cm single-joint) and the pure analytic solver does not hold.
M3 baseline (5r): KPS 0.806 ± 0.060, FID 0.0600 ± 0.0090. Attempted-fix matrix (all measured,
all standard protocol, 1r):

| config | KPS | FID | reading |
|---|---|---|---|
| GN-Maha cap48/8 | 2.18 | 0.271 | Jacobian row starvation (p90 residual 7.2 cm) |
| GN-Maha cap192/16 | 1.02 | 0.288 | rows fixed (p90 0.7 cm) — FID does not recover |
| + proximal filter β∈{.01,.03,.1} | 1.07–1.32 | 0.266–0.281 | drift *magnitude* is not the cause |
| GN-**L2** cap192/16 | 0.94 | 0.176 | metric flip! see below |
| Stage-1(35,dyn,per-sample) + GN | **0.76** | 0.177 | beats M3 on KPS; FID floor persists |
| **GNbs1: BATCHED Stage-1 + per-sample GN (5r)** | **0.664 ± 0.070** | 0.0751 ± 0.0127 | see verdict below |

### 5.1 GNbs1 — the engineered fix, and its adjudicated verdict

Replacing the per-sample Stage-1 with the baseline's **batched** Stage-1 (killing the batch=1
logit-overfitting FID tax) and keeping GN as a per-sample finisher recovers almost everything
(5 repeats): KPS 0.664 ± 0.070 (**17.6% better than M3**), Top3 0.7962 ± 0.0038, ~7 s/sample
(2.1–2.7× faster than M3). Mean FID was 0.0751 vs 0.0600 for M3 (Δ = +0.0151; +25.2%),
narrowly exceeding the pre-registered ceiling of 0.0750 by 0.0001. GNbs1 therefore did **not**
satisfy the composite promotion gate; all five repeats, including the 0.0999 repeat, were
retained. Matched-repeat pairing shows the FID deficit is consistent (worse in all 5 paired
repeats), so it is reported as a systematic ~+25% relative FID cost, not noise. Disposition:
the multi-joint main-table entry remains M3; GNbs1 is reported as the fast secondary
operating point.

Two mechanism findings worth keeping:

1. **The metric conclusion inverts across control types.** On pelvis control Mahalanobis ≫ L2
   (FID 0.062 vs 0.137); on multi-joint L2 > Mahalanobis (0.176 vs 0.288). Consistent reading:
   the codebook covariance's high-variance directions are dominated by global/pelvis motion —
   exactly what pelvis control needs to move, exactly what multi-joint control must NOT abuse
   (the solver satisfies limb constraints through global dims → whole-body distortion). A
   control-subspace-local metric is the natural next step.
2. **The per-sample Stage-1 penalty.** Batch=1 logit TTT alone raises FID ~0.08 even on
   pelvis control (s1C probe: 0.141 vs 0.065) — single-sample overfitting, not a GN issue
   (pure GN per-sample on pelvis: FID 0.062, clean). A batched-Stage-1 + per-sample-GN
   implementation is the identified engineering path for multi-joint.

Path-validity control: batched vs per-sample feed-forward on cross differ by only +0.010 FID
(0.066 vs 0.076) — the floor above is real, not a harness artifact.

## 6. Code map

| file | what |
|---|---|
| `models/mask_transformer/control_transformer_t_concat_v8_kv_v4_noqres.py` | the `"gn"` Stage-2 branch (search `_opt_choice == "gn"`): residual/Jacobian, dual LM solve, trust region with exact-restore, `codebook_cov` metric (`_gn_apply_sigma`); RGAR (rollback + plateau gating, kept as the iterative-family ablation); `TTT_TRACE_DIR` per-iteration probes |
| `scripts/D1_stage2_kv_inference.py` | protocol registry: `GN-Maha` (canonical), `GN-L2` (pathology ablation), `RGAR-M3/M2`, `B*` budget grid, `S2x*` stage ablations; `anchor_frames` explicit-anchor picks; `--out_dir` sweep guard |
| `scripts/eval_v4_ctrlnet_ttt.py` | formal harness flags: `--gn --gn_tau --gn_metric codebook_cov --gn_ridge --last_iter 1` (guarded: bare `--gn` without Stage-2 armed is refused), `--rgar`, `--density`, `--seq_gen` |
| `utils/eval_t2m.py` | per-sample GN generation inside unchanged 32-sample metric groups; fixed-density branch fix |
| `scripts/a1_trace_probe.py` | the A1 convergence probe (build/run/analyze, train-split, tag-paired traces) |
| `scripts/pareto_metrics.py` | accuracy-vs-latency aggregation + plot |

### Canonical invocations

```bash
# formal eval, canonical method
python scripts/eval_v4_ctrlnet_ttt.py --ckpt <gold_ft_v2>/latest.tar \
  --split test --n_layers 20 --n_heads 8 --latent_dim 384 --ff_size 1536 \
  --kv_rank 64 --cross_attn_interval 2 --factorized_attn none \
  --time_steps 10 --each_iter 0 --last_iter 1 --cond_scale 3.25 \
  --gn --gn_tau 1e-5 --gn_metric codebook_cov --repeat_times 5 --control trajectory

# per-pick inference (demos / figures)
USE_V8_V4_CONTROL_NOQRES=1 python scripts/D1_stage2_kv_inference.py \
  --protocols GN-Maha --out_dir <dir> --overwrite
```

All numbers above are reproducible from `output/aaai_pareto_sweep/formal_*.log`; per-run manifests
and the full experiment narrative live in
`.codex-research/plan/20260713_070000_rgar_innovation_day1.md`.

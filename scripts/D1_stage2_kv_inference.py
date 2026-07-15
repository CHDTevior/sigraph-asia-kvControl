"""
D1 Stage-2 KV-Control inference on hand-picked HumanML3D test-split samples.

Runs OUR paper-final v4 KV-Control checkpoint (Gold FT v2; PartVQ tokenizer,
NOT MoMask-KV which is still training as A5) on each of the 15 picks in
`/scratch/ts1v23/workspace/MaskControl/analysis/d1_picks/_index.json` at both
paper headline protocols:

  - M2  : timesteps=10, each_iter=100, ttt_dynamic=False, last_iter=600
  - M3  : timesteps=10, each_iter=35,  ttt_dynamic=True,  last_iter=600

Per pick we save:
  <pick>/pred_joints_M2.npy   shape (T, 22, 3), T == 4 * (n_frames // 4)
  <pick>/pred_joints_M3.npy   shape (T, 22, 3)

Only single-sample inference is performed (no eval-loop / no FID / no KPS):
the consumer (D1 visualization / qualitative comparison) handles metrics on
its own.

Reference model-loading pattern: scripts/eval_v4_ctrlnet_ttt.py.

Usage (defaults pick the paper-final ckpt; override with --ckpt if needed):
  python scripts/D1_stage2_kv_inference.py --gpu_id 0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Ensure repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fixseed import fixseed                                # noqa: E402
import models.vqvae as vqvae                                     # noqa: E402
from models.mask_transformer.control_transformer_t_concat_v8_kv_v4_noqres import (  # noqa: E402
    ControlTransformerTConcatV4 as ControlTransformerKVV4,
)


# ---------------------------------------------------------------------------
# Defaults (paper-final v4 KV-Control + PartVQ tokenizer).
# ---------------------------------------------------------------------------
GOLD_FT_V2_DIR = (
    "/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/"
    "z2026-04-23-21-23-19_v4_0_kv_gold_ft_h200x2_bs1024_k133/model"
)
DEFAULT_CKPT = os.path.join(GOLD_FT_V2_DIR, "latest.tar")

V4_BASE = (
    "/iridisfs/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/"
    "z2026-03-27-17-56-13_t_concat_v4_d384_ff1536_dense_xattn/model/net_best_fid.tar"
)

DEFAULT_VQ_CKPT = (
    "/iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/output/vq/"
    "2026-03-02-14-06-17_vq_overlap_20260302/net_best_fid.pth"
)
DEFAULT_VQ_PARTITION = (
    "/iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis/"
    "skeleton_partition.json"
)

MEAN_NPY = (
    "/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/"
    "VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
)
STD_NPY = (
    "/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/"
    "VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"
)

DEFAULT_PICKS_INDEX = (
    "/scratch/ts1v23/workspace/MaskControl/analysis/d1_picks/_index.json"
)

VQ_CFG = {
    "dataname": "t2m", "batch_size": 256, "window_size": 64, "total_iter": 300000,
    "warm_up_iter": 1000, "lr": 2e-4, "lr_scheduler": [200000], "gamma": 0.05,
    "weight_decay": 0.0, "commit": 0.02, "loss_vel": 0.5, "recons_loss": "l1_smooth",
    "code_dim": 128, "nb_code": 128, "mu": 0.99, "down_t": 2, "stride_t": 2,
    "width": 512, "depth": 3, "dilation_growth_rate": 3, "output_emb_width": 128,
    "vq_act": "relu", "vq_norm": None, "quantizer": "ema_reset", "beta": 1.0,
    "resume_pth": None, "resume_gpt": None, "out_dir": "output",
    "results_dir": "visual_results/", "visual_name": "baseline", "exp_name": "exp_debug",
    "print_iter": 200, "eval_iter": 5000, "seed": 3407, "vis_gt": False,
    "nb_vis": 20, "sep_uplow": False,
}

# Paper headline protocols. All other knobs are kept at the paper defaults
# used in eval_v4_ctrlnet_ttt.py (cond_scale=3.25, temp=1.0, topkr=0.9,
# each_lr=6e-2, last_lr=6e-2, Adam Stage-2).
PROTOCOLS = {
    "M2": dict(time_steps=10, each_iter=100, ttt_dynamic=False, last_iter=600),
    "M3": dict(time_steps=10, each_iter=35,  ttt_dynamic=True,  last_iter=600),
    # Budget-sweep protocols (efficiency exploration 2026-07-13): same machinery, smaller
    # iteration budgets, for the accuracy-vs-latency Pareto. Naming: B<stage1-per-step>x<stage2>.
    "B10x50":   dict(time_steps=10, each_iter=10,  ttt_dynamic=False, last_iter=50),
    "B25x100":  dict(time_steps=10, each_iter=25,  ttt_dynamic=False, last_iter=100),
    "B50x200":  dict(time_steps=10, each_iter=50,  ttt_dynamic=False, last_iter=200),
    "B100x300": dict(time_steps=10, each_iter=100, ttt_dynamic=False, last_iter=300),
    # RGAR-lite (A1-calibrated 2026-07-13, run 20260713_060928): same M3/M2 iteration CAPS, but
    # best-iterate rollback + within-step plateau stop + Stage-2 entry gate. Anytime behaviour:
    # never exceeds the parent budget, exits when converged.
    "RGAR-M3":  dict(time_steps=10, each_iter=35,  ttt_dynamic=True,  last_iter=600,
                     rgar=dict(check_every=10, plateau_rtol=0.01, s2_entry_tau=1e-5)),
    "RGAR-M2":  dict(time_steps=10, each_iter=100, ttt_dynamic=False, last_iter=600,
                     rgar=dict(check_every=10, plateau_rtol=0.01, s2_entry_tau=1e-5)),
    # S2-only: NO Stage-1 TTT at all (each_iter=0) — probes whether the adapter's feed-forward
    # start + Stage-2 alone reaches usable accuracy, i.e. whether Stage-1's 1925 iterations can
    # be REPLACED outright (refinement-replacement track, 2026-07-13).
    "S2x600":   dict(time_steps=10, each_iter=0, ttt_dynamic=False, last_iter=600),
    "S2x100":   dict(time_steps=10, each_iter=0, ttt_dynamic=False, last_iter=100),
    # GN replacement: NO iterative optimization anywhere — feed-forward + <=8 Gauss-Newton
    # solves in the 3A-dim anchor-dual space (batch=1). last_iter=1 only arms the Stage-2
    # block; the GN branch replaces the loop entirely.
    # CANONICAL method protocol (paper): Mahalanobis metric, tau=1e-5. "GN-L2" is the
    # ablation that exposes the off-manifold pathology (FID 0.137 vs 0.070).
    "GN-Maha":  dict(time_steps=10, each_iter=0, ttt_dynamic=False, last_iter=1,
                     s2_optimizer="gn",
                     gn=dict(max_steps=8, tau=1e-5, lam0=1e-2, metric="codebook_cov", ridge=0.1)),
    "GN-L2":    dict(time_steps=10, each_iter=0, ttt_dynamic=False, last_iter=1,
                     s2_optimizer="gn", gn=dict(max_steps=8, tau=1e-6, lam0=1e-2)),
    # RGAR-gated Stage-1 (NOT full M3 Stage-1) + GN Stage-2 — named accordingly (codex P2)
    "RGAR-S1+GN": dict(time_steps=10, each_iter=35, ttt_dynamic=True, last_iter=1,
                     s2_optimizer="gn", gn=dict(max_steps=8, tau=1e-6, lam0=1e-2),
                     rgar=dict(check_every=10, plateau_rtol=0.01, s2_entry_tau=1e-5)),
}


def _load_vq(vq_ckpt: str, vq_partition: str | None):
    """Build PartVQ HumanVQVAE matching VQ_CFG, load weights."""
    vq_cfg = dict(VQ_CFG)
    vq_cfg["load_dir_vqvae"] = vq_ckpt
    vq_cfg["partition_file"] = vq_partition if vq_partition else None
    vq_args = argparse.Namespace(**vq_cfg)
    vq_model = vqvae.HumanVQVAE(
        vq_args, vq_args.nb_code, vq_args.code_dim, vq_args.output_emb_width,
        vq_args.down_t, vq_args.stride_t, vq_args.width, vq_args.depth,
        vq_args.dilation_growth_rate, vq_args.vq_act, vq_args.vq_norm,
    )
    sd = torch.load(vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(sd["net"] if "net" in sd else sd)
    vq_model.eval()
    return vq_model


def _build_model(args, device, vq_model, mean_t, std_t):
    """Construct the v4-KV ControlNet matching the Gold FT v2 architecture."""
    opt = argparse.Namespace(
        joints_num=22, max_motion_len=55, num_tokens=VQ_CFG["nb_code"],
        num_quantizers=6,
        latent_dim=args.latent_dim, ff_size=args.ff_size,
        n_layers=args.n_layers, n_heads=args.n_heads,
        dropout=args.dropout, cond_drop_prob=args.cond_drop_prob,
        max_motion_length=196,
        cross_attn_interval=args.cross_attn_interval, cross_attn_heads=0,
        text_adapter_layers=args.text_adapter_layers, gate_init=args.gate_init,
        unit_length=4, max_token_len=49,
        factorized_attn=args.factorized_attn,
        mask_2d_hybrid=False, mask_2d_ratio=0.3,
        ttt_gumbel=False, ttt_dynamic=False,   # overwritten per-protocol below
        ctrl_net=True, each_lr=6e-2, each_iter=100,
        last_lr=6e-2, last_iter=0,             # overwritten per-protocol below
        device=device, control="trajectory", dataset_name="t2m",
        dataset_opt_path="./checkpoints/t2m/Comp_v6_KLD005/opt.txt",
        vq_checkpoint=args.vq_ckpt,
        vq_partition_file=args.vq_partition_file,
        mean_npy=MEAN_NPY, std_npy=STD_NPY,
    )

    os.environ.setdefault(
        "MASKCONTROL_CLIP_MODEL_PATH",
        "/scratch/ts1v23/workspace/MaskControl/artifacts/models/clip/ViT-B-32.pt",
    )

    ct2m = ControlTransformerKVV4(
        code_dim=128, cond_mode="text",
        latent_dim=args.latent_dim, ff_size=args.ff_size,
        num_layers=args.n_layers, num_heads=args.n_heads,
        dropout=args.dropout, clip_dim=512, cond_drop_prob=args.cond_drop_prob,
        clip_version="ViT-B/32", opt=opt,
        mean=mean_t, std=std_t,
        trans_path=args.base_ckpt,
        vq_model=vq_model,
        control="trajectory",
        kv_rank=args.kv_rank,
    )

    ckpt = torch.load(args.ckpt, map_location=device)
    key = "ct2m_transformer" if "ct2m_transformer" in ckpt else "trans"
    missing, unexpected = ct2m.load_state_dict(ckpt[key], strict=False)
    non_clip_missing = [k for k in missing if not k.startswith("clip_model.")]
    if non_clip_missing:
        bias_missing = [k for k in non_clip_missing if k.startswith("ctrl_attn_bias")]
        if bias_missing and len(bias_missing) == len(non_clip_missing):
            print(f"  Resetting {len(bias_missing)} ctrl_attn_bias to 0 (pre-BUG2 compat).")
            with torch.no_grad():
                for p in ct2m.ctrl_attn_bias.parameters():
                    p.zero_()
        else:
            raise RuntimeError(
                f"Non-CLIP, non-bias missing keys when loading ckpt — refusing to "
                f"silently degrade. missing={non_clip_missing}"
            )
    if unexpected:
        raise RuntimeError(f"Unexpected keys in ckpt: {unexpected}")

    ct2m.to(device)
    ct2m.eval()
    ct2m.ctrl_net = True
    return ct2m, opt


def _build_pick_inputs(pick, device, max_len=196):
    """Build per-pick model inputs.

    Returns:
      text      : single-line text string (first line of text.txt).
      m_length  : LongTensor [1] of original frame count (clipped to <= max_len).
      global_joint: FloatTensor [1, max_len, 22, 3] (gt_joints padded with zeros).
      global_joint_mask: BoolTensor [1, max_len, 22], only joint 0 (pelvis)
                       True for the valid frames; everything else False.
    """
    paths = pick["saved_paths"]
    with open(paths["text"], "r") as fh:
        text = fh.readline().strip()

    gt_joints = np.load(paths["gt_joints"])           # (T, 22, 3)
    n_frames = int(gt_joints.shape[0])
    if n_frames > max_len:
        gt_joints = gt_joints[:max_len]
        n_frames = max_len

    # codex fix: gj/gjm shape mismatch. generate_with_control derives seq_len
    # from m_lens.max() and returns pred_joints with T = 4 * (m_length // 4),
    # so _build_ctrl_cond does `global_joint - pred_joints` which crashes for
    # any pick with n_frames < 196. Trim BOTH gt_joints and global_joint_mask
    # to n_eff = 4 * (n_frames // 4) BEFORE building tensors / passing to model.
    n_eff = 4 * (n_frames // 4)
    gt_joints = gt_joints[:n_eff]

    gj = torch.zeros(1, n_eff, 22, 3, device=device, dtype=torch.float32)
    gj[0, :n_eff] = torch.from_numpy(gt_joints).to(device).float()

    gjm = torch.zeros(1, n_eff, 22, device=device, dtype=torch.bool)
    # KF_DENSITY: number of pelvis keyframes. Unset/"dense" = every frame (the historical
    # behaviour, unchanged). An integer k anchors k evenly spaced keyframes — the regime the
    # benchmark KPS is measured in (eval density=5) and the one training mass concentrates on.
    _dens = os.environ.get("KF_DENSITY", "dense")
    if pick.get("anchor_frames") is not None:
        # A pick can carry explicit anchor positions (probe sets use TRAINING-distribution
        # anchors: random positions, count scaled by length for the 49/196 categories — codex
        # 019f59cc P1: evenly spaced linspace anchors are a distribution shift that biases
        # entry-loss calibration optimistic). Takes precedence over KF_DENSITY.
        raw = np.asarray(pick["anchor_frames"])
        if raw.ndim != 1 or raw.size == 0 or raw.dtype == bool or \
                not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"{pick.get('idx')}: anchor_frames must be a non-empty 1-D "
                             f"integer array, got shape {raw.shape} dtype {raw.dtype}")
        sel = raw.astype(int)
        if sel.min() < 0 or sel.max() >= n_eff or len(np.unique(sel)) != len(sel):
            raise ValueError(f"{pick.get('idx')}: bad anchor_frames "
                             f"(n={len(sel)}, range [{sel.min()},{sel.max()}], n_eff={n_eff})")
        gjm[0, torch.from_numpy(sel).to(device).long(), 0] = True
    elif _dens == "dense":
        gjm[0, :n_eff, 0] = True
    else:
        k = int(_dens)
        if not (1 <= k <= n_eff):
            raise ValueError(f"KF_DENSITY={k} must be in [1, {n_eff}]")
        sel = np.linspace(0, n_eff - 1, k).round().astype(int)
        gjm[0, torch.from_numpy(sel).to(device).long(), 0] = True

    m_length = torch.tensor([n_eff], device=device, dtype=torch.long)
    return text, m_length, gj, gjm, n_frames


def _run_protocol(ct2m, vq_model, opt, name, cfg, text, m_length, gj, gjm,
                  mean_t, std_t, cond_scale, temperature, topkr):
    """Run one protocol and return pred_joints as numpy (T_eff, 22, 3)."""
    # Per-protocol opt overrides — generate() reads `opt.ttt_dynamic` directly.
    opt.ttt_dynamic = cfg["ttt_dynamic"]
    opt.each_iter = cfg["each_iter"]
    opt.last_iter = cfg["last_iter"]
    ct2m.opt.ttt_dynamic = cfg["ttt_dynamic"]

    # codex 019f59a4: synchronize so dt charges all queued GPU work to the sample, and use
    # perf_counter; dt excludes model load and (unlike B4 before this fix) any disk I/O.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        # `generate_with_control` opens grad internally for TTT / Stage-2.
        pred_motions_denorm, _ = ct2m.generate_with_control(
            [text], m_length, cfg["time_steps"], cond_scale,
            temperature=temperature, topkr=topkr, force_mask=False,
            vq_model=vq_model,
            global_joint=gj, global_joint_mask=gjm,
            _mean=mean_t, _std=std_t,
            control_opt={
                "each_lr": 6e-2, "each_iter": cfg["each_iter"],
                "lr": 6e-2,      "iter": cfg["last_iter"],
                "rgar": cfg.get("rgar"),
                "s2_optimizer": cfg.get("s2_optimizer"),
                # GN_JAC_CHUNK: opt-in fast replicated-batch Jacobian (metric-equivalent,
                # not bit-equivalent — leave unset to reproduce published picks exactly)
                "gn": (dict(cfg["gn"], jac_chunk=int(os.environ["GN_JAC_CHUNK"]))
                       if cfg.get("gn") is not None and os.environ.get("GN_JAC_CHUNK")
                       else cfg.get("gn")),
            },
            avoid_points=None,
        )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    # pred_motions_denorm: (1, T_eff, 22, 3) where T_eff = 4 * (n_frames // 4)
    pj = pred_motions_denorm[0].detach().cpu().numpy()
    n_eff = 4 * (int(m_length.item()) // 4)
    pj = pj[:n_eff]
    print(f"    [{name}] {dt:6.1f}s  out_shape={tuple(pj.shape)}")
    return pj


def main():
    # codex fix: leftover STAGE2_OPTIMIZER from a prior ablation could silently
    # route Stage-2 to LBFGS, violating feedback_lbfgs_internal_only (LBFGS
    # NEVER appears in the paper). Pin to adam explicitly at entry.
    os.environ["STAGE2_OPTIMIZER"] = "adam"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                        help="Paper-final v4 KV-Control ckpt (Gold FT v2 latest).")
    parser.add_argument("--base_ckpt", type=str, default=V4_BASE,
                        help="Frozen v4 base ckpt the ControlNet was trained on.")
    parser.add_argument("--vq_ckpt", type=str, default=DEFAULT_VQ_CKPT,
                        help="PartVQ tokenizer ckpt (NOT MoMask RVQ — D1 uses paper-final substrate).")
    parser.add_argument("--vq_partition_file", type=str, default=DEFAULT_VQ_PARTITION)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--picks_index", type=str, default=DEFAULT_PICKS_INDEX)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cond_scale", type=float, default=3.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topkr", type=float, default=0.9)
    parser.add_argument("--protocols", type=str, default="M2,M3",
                        help="Comma-separated subset of {M2,M3}.")
    # v4 architecture (Gold FT v2 matches the paper-default v4 base).
    parser.add_argument("--n_layers", type=int, default=20)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=384)
    parser.add_argument("--ff_size", type=int, default=1536)
    parser.add_argument("--kv_rank", type=int, default=64)
    parser.add_argument("--cross_attn_interval", type=int, default=2,
                        help="Paper main config: interval=2 (per feedback_v4_cli_flags_and_interval).")
    parser.add_argument("--text_adapter_layers", type=int, default=4)
    parser.add_argument("--gate_init", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--factorized_attn", type=str, default="none")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if pred_joints_<M>.npy already exists.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Write pred_joints_<M>.npy here instead of back into the pick "
                             "directory. REQUIRED for any sweep (e.g. KF_DENSITY): without it "
                             "this script overwrites the paper's picks in place, which has "
                             "already destroyed them once.")
    args = parser.parse_args()

    # A sweep that writes back into the pick directories silently replaces the paper's figures'
    # inputs with the sweep's outputs. Refuse.
    if os.environ.get("KF_DENSITY", "dense") != "dense" and not args.out_dir:
        raise SystemExit("KF_DENSITY is set but --out_dir is not: refusing to overwrite the "
                         "paper's pick directories with sweep outputs. Pass --out_dir.")

    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    for p in protocols:
        if p not in PROTOCOLS:
            raise ValueError(f"Unknown protocol {p!r}; expected one of {sorted(PROTOCOLS)}.")

    fixseed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}")

    print(f"[info] ckpt        = {args.ckpt}")
    print(f"[info] base_ckpt   = {args.base_ckpt}")
    print(f"[info] vq_ckpt     = {args.vq_ckpt}")
    print(f"[info] picks_index = {args.picks_index}")
    print(f"[info] protocols   = {protocols}")
    print(f"[info] device      = {device}")

    with open(args.picks_index, "r") as fh:
        index = json.load(fh)
    picks = index["picks"]
    print(f"[info] n_picks     = {len(picks)}")

    mean_np = np.load(MEAN_NPY)
    std_np = np.load(STD_NPY)
    mean_t = torch.tensor(mean_np, dtype=torch.float32, requires_grad=False, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, requires_grad=False, device=device)

    vq_model = _load_vq(args.vq_ckpt, args.vq_partition_file).to(device)
    print("[info] VQ loaded.")

    ct2m, opt = _build_model(args, device, vq_model, mean_t, std_t)
    print(f"[info] KV-ControlNet loaded from {args.ckpt}")

    t_all = time.time()
    for i, pick in enumerate(picks):
        idx = pick["idx"]
        pick_dir = os.path.dirname(pick["saved_paths"]["gt_joints"])
        print(f"\n[{i+1}/{len(picks)}] idx={idx}  ({pick.get('category', '?')})")

        text, m_length, gj, gjm, n_frames = _build_pick_inputs(pick, device)
        print(f"    text='{text}'  n_frames={n_frames}")

        write_dir = args.out_dir or pick_dir
        if args.out_dir:
            os.makedirs(write_dir, exist_ok=True)
        for name in protocols:
            fn = f"pred_joints_{name}.npy" if not args.out_dir else f"{idx}_pred_joints_{name}.npy"
            out_path = os.path.join(write_dir, fn)
            if (not args.overwrite) and os.path.exists(out_path):
                print(f"    [{name}] SKIP (exists): {out_path}")
                continue
            # Re-seed per (pick, protocol) so M2 and M3 share identical text
            # sampling — only the control schedule differs.
            fixseed(args.seed + i)
            if os.environ.get("TTT_TRACE_DIR"):
                # identity for the trace file — analyzers pair by this tag, never by timestamp
                os.environ["TTT_TRACE_TAG"] = f"{idx}:{name}:{i}"
            pj = _run_protocol(
                ct2m, vq_model, opt, name, PROTOCOLS[name], text, m_length,
                gj, gjm, mean_t, std_t,
                args.cond_scale, args.temperature, args.topkr,
            )
            # codex fix: non-atomic np.save. A killed run leaves a corrupt .npy
            # which then trips the --overwrite skip guard and silently poisons
            # downstream Blender renders. Write to .tmp.npy then os.replace for
            # atomic publish.
            tmp_path = out_path + ".tmp.npy"
            np.save(tmp_path, pj.astype(np.float32))
            os.replace(tmp_path, out_path)
            print(f"    [{name}] saved -> {out_path}")

    print(f"\n[done] total wallclock = {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()

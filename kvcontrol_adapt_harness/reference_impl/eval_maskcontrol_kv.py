"""Offline paper-formal eval for KV-on-MaskControl port.

Forks `scripts/eval_v4_ctrlnet_ttt.py` — the v4-substrate offline eval — and
retargets it at the MaskControl substrate with our KVControlTransformer.

Protocol: M1 (Stage-1 TTT only, matches KPSoursNoRefine cell in paper macros)
    time_steps   = 10
    cond_scale   = 3.25
    each_iter    = 35
    ttt_dynamic  = True     # step s gets (s+1)*35 inner iters
    last_iter    = 0        # Stage 2 disabled
    last_lr      = 6e-2
    repeat_times = 5        # 5-rep on full HumanML3D test split for CI

Output: FID, R-precision (top1/2/3), MatchScore, Diversity, KPS (cm), skate_ratio,
all reported as mean ± 95% CI = std × 1.96 / sqrt(N).

Usage:
    python scripts/eval_maskcontrol_kv.py \\
      --ckpt references/MaskControl/checkpoints/t2m/<run>/model/best_kps.tar \\
      --gpu_id 0 --repeat_times 5 \\
      --output_json output/<run>/eval_5r_M1.json
"""
import os, sys, argparse, json, time
from pathlib import Path

import numpy as np
import torch

# MaskControl substrate — put its repo first on sys.path so its own utils/modules resolve
MC_REPO = Path("/scratch/ts1v23/workspace/MaskControl/references/MaskControl")
PROJ = Path("/scratch/ts1v23/workspace/MaskControl")
sys.path.insert(0, str(MC_REPO))

from utils.fixseed import fixseed
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from models.vq.model import RVQVAE
from models.mask_transformer.control_transformer_kv import KVControlTransformer
import utils.eval_t2m as eval_t2m


def load_rvqvae(rvq_ckpt_path, rvq_opt_path, device):
    """Load MoMask's public RVQ-VAE (same one MaskControl uses)."""
    rvq_opt = argparse.Namespace()
    with open(rvq_opt_path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            else:
                try:
                    v = float(v) if ("." in v or "e" in v.lower()) else int(v)
                except ValueError:
                    pass
            setattr(rvq_opt, k, v)
    rvq_opt.dim_pose = 263
    rvq_opt.joints_num = 22
    rvq = RVQVAE(rvq_opt, rvq_opt.dim_pose, rvq_opt.nb_code, rvq_opt.code_dim,
                 rvq_opt.output_emb_width, rvq_opt.down_t, rvq_opt.stride_t,
                 rvq_opt.width, rvq_opt.depth, rvq_opt.dilation_growth_rate,
                 rvq_opt.vq_act, rvq_opt.vq_norm)
    ckpt = torch.load(rvq_ckpt_path, map_location="cpu")
    state_key = "vq_model" if "vq_model" in ckpt else "net"
    rvq.load_state_dict(ckpt[state_key])
    rvq.to(device).eval()
    for p in rvq.parameters():
        p.requires_grad = False
    return rvq, rvq_opt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="KV-Control ckpt (.tar) — best_kps.tar or latest.tar from v16 training")
    p.add_argument("--gpu_id", type=int, default=0)

    # Protocol: M1 defaults (paper-formal)
    p.add_argument("--repeat_times", type=int, default=5)
    p.add_argument("--time_steps", type=int, default=10)
    p.add_argument("--cond_scale", type=float, default=3.25)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--topkr", type=float, default=0.9)
    p.add_argument("--force_mask", action="store_true")
    p.add_argument("--each_iter", type=int, default=35, help="TTT iterations per timestep (M1=35)")
    p.add_argument("--each_lr", type=float, default=6e-2)
    p.add_argument("--ttt_dynamic", action=argparse.BooleanOptionalAction, default=False,
                   help="Dynamic TTT: step s gets (s+1)*each_iter iters, wired via the "
                        "substrate's negative-each_iter convention (control_transformer.py "
                        "L527-531: each_iter<0 -> dynamic). NOTE: published ep6000 M1/M2/M3 "
                        "numbers were all produced with UNIFORM iters (this flag was "
                        "historically cosmetic and never negated each_iter); enabling it "
                        "yields a NEW protocol not comparable to published numbers.")
    p.add_argument("--last_iter", type=int, default=0, help="Stage-2 embedding refinement iters (M1=0; M3=600)")
    p.add_argument("--last_lr", type=float, default=6e-2)
    p.add_argument("--seed", type=int, default=3407)

    # Substrate paths (MaskControl defaults)
    p.add_argument("--dataset_name", type=str, default="t2m")
    p.add_argument("--vq_name", type=str, default="rvq_nq6_dc512_nc512_noshare_qdp0.2")
    p.add_argument("--trans_name", type=str, default="t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns")
    p.add_argument("--eval_wrapper_opt", type=str,
                   default=str(MC_REPO / "checkpoints/t2m/Comp_v6_KLD005/opt.txt"))

    # Model architecture (must match training)
    p.add_argument("--latent_dim", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--ff_size", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--control", type=str, default="trajectory")
    p.add_argument("--kv_rank", type=int, default=64)
    p.add_argument("--ctrl_attn_bias_init", type=float, default=-5.0)
    p.add_argument("--no_q_residual", action="store_true")

    # Output
    p.add_argument("--output_json", type=str, required=True)

    args = p.parse_args()

    fixseed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}  ckpt={args.ckpt}")

    # ---- eval loader (MaskControl's Comp_v6 wrapper) ----
    print(f"[eval] loading eval_val_loader from {args.eval_wrapper_opt}")
    eval_val_loader, _ = get_dataset_motion_loader(args.eval_wrapper_opt, 32, "test", device=device)
    wrapper_opt = get_opt(args.eval_wrapper_opt, torch.device("cuda"))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    mean = torch.tensor(eval_val_loader.dataset.mean, requires_grad=False).to(device)
    std = torch.tensor(eval_val_loader.dataset.std, requires_grad=False).to(device)

    # ---- vq model (RVQ-VAE, 6 quantizers, MoMask-public) ----
    vq_ckpt_dir = MC_REPO / "checkpoints" / args.dataset_name / args.vq_name
    vq_opt_path = vq_ckpt_dir / "opt.txt"
    vq_ckpt_path = vq_ckpt_dir / "model" / "net_best_fid.tar"
    print(f"[eval] loading vq_model from {vq_ckpt_path}")
    vq_model, vq_opt = load_rvqvae(str(vq_ckpt_path), str(vq_opt_path), device)

    # ---- build KVControlTransformer + load ckpt ----
    class Opt: pass
    opt = Opt()
    opt.device = device
    opt.dataset_name = args.dataset_name
    opt.joints_num = 22
    opt.num_tokens = int(vq_opt.nb_code)
    opt.latent_dim = args.latent_dim
    opt.ff_size = args.ff_size
    opt.n_layers = args.n_layers
    opt.n_heads = args.n_heads
    opt.dropout = args.dropout
    opt.cond_drop_prob = 0.1
    opt.unit_length = 4
    opt.max_motion_length = 196
    opt.checkpoints_dir = str(MC_REPO / "checkpoints")

    # eval-time attrs the model expects
    opt.ctrl_net = True
    opt.each_lr = args.each_lr
    # negative-each_iter convention (control_transformer.py L527-531):
    # each_iter > 0 -> uniform iters per step; each_iter < 0 -> dynamic
    # (step s gets (s+1)*|each_iter|). Historical bug: this negation was
    # never applied, so --ttt_dynamic was cosmetic and published M1/M3
    # numbers are uniform-35. Now wired explicitly:
    opt.each_iter = -args.each_iter if args.ttt_dynamic else args.each_iter
    opt.last_lr = args.last_lr
    opt.last_iter = args.last_iter

    trans_path = MC_REPO / "checkpoints" / args.dataset_name / args.trans_name / "model" / "latest.tar"

    print(f"[eval] instantiating KVControlTransformer, trans_path={trans_path}")
    model = KVControlTransformer(
        code_dim=vq_opt.code_dim, cond_mode="text",
        latent_dim=args.latent_dim, ff_size=args.ff_size,
        num_layers=args.n_layers, num_heads=args.n_heads, dropout=args.dropout,
        clip_dim=512, cond_drop_prob=0.1,
        clip_version="ViT-B/32", opt=opt,
        mean=mean, std=std,
        trans_path=str(trans_path),
        vq_model=vq_model, control=args.control,
        kv_rank=args.kv_rank,
        ctrl_attn_bias_init=args.ctrl_attn_bias_init,
        use_q_residual=(not args.no_q_residual),
    ).to(device)
    model.eval()

    print(f"[eval] loading ckpt {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    key = "ct2m_transformer" if "ct2m_transformer" in ckpt else "trans"
    raw_sd = ckpt[key]
    # DDP-saved ckpts sometimes have 'module.' prefix (best_kps.tar might have it).
    # Strip if present.
    if any(k.startswith("module.") for k in raw_sd.keys()):
        raw_sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in raw_sd.items()}
        print(f"[eval] stripped 'module.' prefix (DDP ckpt) from state_dict")
    missing, unexpected = model.load_state_dict(raw_sd, strict=False)
    non_clip_missing = [k for k in missing if not k.startswith("clip_model.")]
    if non_clip_missing:
        print(f"[eval] WARN missing non-clip keys: {non_clip_missing[:10]}...")
    if unexpected:
        # tolerate the vq_model.* / ControlNet* leaked keys from DDP; we already have KV state.
        print(f"[eval] WARN unexpected keys: {unexpected[:10]}...")

    # ---- 5-rep eval loop (M1 protocol) ----
    ckpt_epoch = ckpt.get("ep", -1)
    # M1 = uniform-35 (the protocol the published ep6000 numbers were produced with).
    protocol = "M1" if (args.each_iter == 35 and args.time_steps == 10 and not args.ttt_dynamic and args.last_iter == 0) else "CUSTOM"
    print(f"\n[eval] protocol={protocol}  ckpt_epoch={ckpt_epoch}  repeat_times={args.repeat_times}")
    print(f"[eval] settings: time_steps={args.time_steps} cond_scale={args.cond_scale} "
          f"each_iter={args.each_iter} ttt_dynamic={args.ttt_dynamic} last_iter={args.last_iter}")

    reps = []
    t0 = time.time()
    for rep_id in range(args.repeat_times):
        rep_start = time.time()
        best_fid, best_div, Rprecision, best_matching, best_skate_ratio, best_mm, traj_err, _avoid, kps_mean = \
            eval_t2m.evaluation_mask_transformer_test_plus_res(
                eval_val_loader, vq_model, None, model, None, rep_id,
                eval_wrapper=eval_wrapper,
                time_steps=args.time_steps, cond_scale=args.cond_scale,
                temperature=args.temperature, topkr=args.topkr,
                # pred_num_batch=16 matches MaskControl's default; it means "accumulate 16
                # (bs=32) batches = 512 samples per generate call" — not "process 16 batches
                # total". Setting this too high silently disables all generation calls.
                force_mask=args.force_mask, cal_mm=False, f=None, pred_num_batch=16,
                logger=None, epoch=ckpt_epoch,
                control=args.control, density=-1, opt=opt)
        rep_dt = time.time() - rep_start
        top1, top2, top3 = float(Rprecision[0]), float(Rprecision[1]), float(Rprecision[2])
        print(f"[eval] rep {rep_id+1}/{args.repeat_times} done in {rep_dt:.1f}s | "
              f"FID={best_fid:.4f} Top3={top3:.4f} Match={best_matching:.4f} "
              f"Div={best_div:.4f} KPS={kps_mean*100:.3f}cm skate={best_skate_ratio:.4f}")
        reps.append(dict(
            fid=float(best_fid), diversity=float(best_div),
            top1=top1, top2=top2, top3=top3,
            matching_score=float(best_matching),
            skate_ratio=float(best_skate_ratio),
            kps_cm=float(kps_mean) * 100.0,
        ))

    # ---- summarize mean ± 95% CI ----
    N = len(reps)
    summary = {}
    for k in ("fid", "diversity", "top1", "top2", "top3", "matching_score", "skate_ratio", "kps_cm"):
        vals = np.array([r[k] for r in reps])
        mean_v = float(vals.mean())
        std_v = float(vals.std(ddof=1)) if N > 1 else 0.0
        ci95 = float(1.96 * std_v / np.sqrt(N)) if N > 1 else 0.0
        summary[k] = dict(mean=mean_v, std=std_v, ci95=ci95, n=N)

    wall = time.time() - t0
    print(f"\n{'='*72}")
    print(f"[eval] SUMMARY (N={N} reps, protocol={protocol}, wall={wall/60:.1f} min)")
    print('='*72)
    for k, v in summary.items():
        unit = " cm" if k == "kps_cm" else ""
        print(f"  {k:>18}: {v['mean']:8.4f} ± {v['ci95']:.4f} (std={v['std']:.4f}, n={N}){unit}")

    out = dict(
        ckpt_path=args.ckpt,
        ckpt_epoch=ckpt_epoch,
        protocol=protocol,
        wall_seconds=wall,
        settings=vars(args),
        reps=reps,
        summary=summary,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[eval] wrote {args.output_json}")


if __name__ == "__main__":
    main()

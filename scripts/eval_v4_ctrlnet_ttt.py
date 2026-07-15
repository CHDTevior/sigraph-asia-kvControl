"""
Evaluate v8 KV-Context ControlNet on v4 base with TTT.

Usage:
    python scripts/eval_v4_ctrlnet_ttt.py \
        --ckpt checkpoints/t2m/.../model/net_ep400.tar \
        --time_steps 25 --each_iter 200 --cond_scale 3.25 --gpu_id 0
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fixseed import fixseed
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
import models.vqvae as vqvae
import utils.eval_t2m as eval_t2m

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

V4_BASE = "/iridisfs/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/z2026-03-27-17-56-13_t_concat_v4_d384_ff1536_dense_xattn/model/net_best_fid.tar"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--each_iter", type=int, default=200, help="TTT iterations per timestep")
    parser.add_argument("--each_lr", type=float, default=6e-2)
    parser.add_argument("--cond_scale", type=float, default=3.25)
    parser.add_argument("--time_steps", type=int, default=25)
    parser.add_argument("--pred_num_batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--control", type=str, default="trajectory",
                        choices=["trajectory", "pelvis", "l_foot", "r_foot", "head",
                                 "left_wrist", "right_wrist", "lower", "cross", "random", "all"],
                        help="Control mode. trajectory==pelvis (single-joint). "
                             "{pelvis,l_foot,r_foot,head,left_wrist,right_wrist,lower}: fixed-joint single-joint eval. "
                             "cross: random per-sample joint from 6-set. random: per-sample random idx. all: 6 joints simultaneously.")
    parser.add_argument("--trans_path", type=str, default=None, help="Base model checkpoint (overrides V4_BASE)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--n_layers", type=int, default=20)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=384)
    parser.add_argument("--ff_size", type=int, default=1536)
    parser.add_argument("--kv_rank", type=int, default=64)
    parser.add_argument("--factorized_attn", type=str, default="none",
                        help="Must match training config. Use 'none' for old v4.0, 'same_q' for v4.68+")
    parser.add_argument("--mask_2d_hybrid", action="store_true")
    parser.add_argument("--mask_2d_ratio", type=float, default=0.3)
    parser.add_argument("--cross_attn_interval", type=int, default=1)
    parser.add_argument("--text_adapter_layers", type=int, default=4)
    parser.add_argument("--gate_init", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--ttt_gumbel", action="store_true", help="Add Gumbel noise during TTT (like MaskControl)")
    parser.add_argument("--ttt_dynamic", action="store_true", help="Dynamic TTT schedule: step s gets (s+1)*each_iter iters. When NOT passed, Stage 1 uses uniform per-step iter (M2 protocol).")
    parser.add_argument("--last_iter", type=int, default=0, help="Stage 2 (post-generation) embedding optimization iterations (MaskControl uses 600). 0 disables Stage 2.")
    parser.add_argument("--rgar", action="store_true",
                        help="RGAR-lite: best-iterate rollback + within-step plateau stop + "
                             "Stage-2 entry gate (A1-calibrated defaults).")
    parser.add_argument("--rgar_tau", type=float, default=1e-5)
    parser.add_argument("--density", type=int, default=-1,
                        help="Control density: -1 = protocol mixture {1,2,5,49,196} (paper "
                             "default); a positive k fixes k anchors for per-density rows.")
    parser.add_argument("--seq_gen", action="store_true",
                        help="Force per-sample sequential generation WITHOUT GN — control "
                             "experiment isolating the batch-1 path from the GN solver.")
    parser.add_argument("--gn_tau", type=float, default=1e-6)
    parser.add_argument("--gn_lam0", type=float, default=1e-2)
    parser.add_argument("--gn_metric", type=str, default="", choices=["", "codebook_cov"])
    parser.add_argument("--gn_ridge", type=float, default=0.1)
    parser.add_argument("--gn_steps", type=int, default=8)
    parser.add_argument("--gn_anchor_cap", type=int, default=48)
    parser.add_argument("--gn_prox_beta", type=float, default=0.0)
    parser.add_argument("--gn_jac_chunk", type=int, default=0,
                        help="Jacobian rows per replicated-batch backward (fast, metric-"
                             "equivalent but not bit-equivalent); 0 = per-row loop")
    parser.add_argument("--gn_batch", action="store_true",
                        help="batched Stage-1 + per-sample GN Stage-2 (instead of fully "
                             "per-sample generation)")
    parser.add_argument("--gn", action="store_true",
                        help="Replace Stage-2 with the Gauss-Newton anchor-dual solve "
                             "(batch=1 generation inside the unchanged metric groups).")
    parser.add_argument("--last_lr", type=float, default=6e-2, help="LR for Stage 2 embedding optimization (default 6e-2, MaskControl-aligned).")
    parser.add_argument("--vq_checkpoint", type=str, default=None,
                        help="VQ checkpoint path override. Default: old-vq 128 (for back-compat). Pass MMM path for new runs.")
    parser.add_argument("--vq_partition_file", type=str, default=None,
                        help='Skeleton partition JSON. Default: skeleton_partition.json. Pass "" to disable (MMM hardcoded partitions).')
    parser.add_argument("--nb_code", type=int, default=None,
                        help="VQ codebook size override. Default: VQ_CFG[nb_code]=128. Required for MMM 8192/512.")
    args = parser.parse_args()
    if args.gn and args.last_iter <= 0:
        parser.error("--gn requires --last_iter >= 1 (Stage-2 must be armed for the GN solve); "
                     "bare --gn would silently run NO refinement at all")


    if args.nb_code is not None:
        VQ_CFG["nb_code"] = int(args.nb_code)

    base_path = args.trans_path if args.trans_path else V4_BASE

    fixseed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}")

    vq_checkpoint = args.vq_checkpoint if args.vq_checkpoint else \
        "/iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/output/vq/2026-03-02-14-06-17_vq_overlap_20260302/net_best_fid.pth"
    # partition_file: None when user passes empty string (MMM hardcoded partitions)
    if args.vq_partition_file is None:
        vq_partition_file = "/iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis/skeleton_partition.json"
    elif args.vq_partition_file.strip() == "":
        vq_partition_file = None
    else:
        vq_partition_file = args.vq_partition_file
    dataset_opt_path = "./checkpoints/t2m/Comp_v6_KLD005/opt.txt"
    mean_npy = "/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
    std_npy = "/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"

    # Load VQ
    vq_cfg = dict(VQ_CFG)
    vq_cfg["load_dir_vqvae"] = vq_checkpoint
    # Pass None (not "") so vqvae.HumanVQVAE takes the hardcoded 6-part branch for MMM
    vq_cfg["partition_file"] = vq_partition_file if vq_partition_file else None
    vq_args = argparse.Namespace(**vq_cfg)
    vq_model = vqvae.HumanVQVAE(
        vq_args, vq_args.nb_code, vq_args.code_dim, vq_args.output_emb_width,
        vq_args.down_t, vq_args.stride_t, vq_args.width, vq_args.depth,
        vq_args.dilation_growth_rate, vq_args.vq_act, vq_args.vq_norm,
    )
    ckpt = torch.load(vq_checkpoint, map_location="cpu")
    vq_model.load_state_dict(ckpt["net"] if "net" in ckpt else ckpt)
    vq_model.eval()
    print("Loaded VQ model")

    # Load eval data
    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, args.split, device=device)
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # Build v4-based ControlNet
    # Detect which control class to use based on checkpoint keys
    _probe_ckpt = torch.load(args.ckpt, map_location="cpu")
    _probe_key = "ct2m_transformer" if "ct2m_transformer" in _probe_ckpt else "trans"
    _probe_keys = set(_probe_ckpt[_probe_key].keys())
    _has_kv = any(k.startswith("kv_down.") for k in _probe_keys)
    _has_qres = any(("q_down" in k) or ("q_up" in k) or ("q_gate" in k) for k in _probe_keys)
    del _probe_ckpt
    _env_noqres = os.environ.get("USE_V8_V4_CONTROL_NOQRES") == "1"
    _env_kv_cconcat = os.environ.get("USE_KV_CCONCAT_V4") == "1"
    _env_v5_v4 = os.environ.get("USE_V5_V4_CONTROL") == "1"
    _use_noqres = _has_kv and (_env_noqres or not _has_qres)
    if _env_v5_v4:
        # ControlNet-style parallel-branch on T-Concat v4 (paper §4.6 Pareto baseline,
        # ~48M trainable parallel branch, contrast vs KV-Control's 1.5M mech / 10.5M total).
        from models.mask_transformer.control_transformer_t_concat_v5_v4 import ControlTransformerTConcatV5V4 as ControlTransformerTConcatV4
        print("[INFO] Using ControlTransformerTConcatV5V4 (ControlNet-style parallel branch on T-Concat v4 backbone)")
    elif _env_kv_cconcat and _has_kv:
        from models.mask_transformer.control_transformer_kv_c_concat_v4 import ControlTransformerKVCConcatV4 as ControlTransformerTConcatV4
        print("[INFO] Using ControlTransformerKVCConcatV4 (KV adapter on c_concat_v4 backbone)")
    elif _use_noqres:
        from models.mask_transformer.control_transformer_t_concat_v8_kv_v4_noqres import ControlTransformerTConcatV4
        print("[INFO] Using ControlTransformerTConcatV4 noqres variant (no Q-Residual, paper main method per Codex 019db6bd)")
    elif _has_kv:
        from models.mask_transformer.control_transformer_t_concat_v8_kv_v4 import ControlTransformerTConcatV4
        print("[INFO] Using ControlTransformerTConcatV4 with Q-Residual (legacy/ablation variant)")
    else:
        from models.mask_transformer.control_transformer_t_concat import ControlTransformerTConcat as ControlTransformerTConcatV4
        print("[INFO] Using additive ControlTransformerTConcat (no kv_down keys in ckpt)")

    opt = argparse.Namespace(
        joints_num=22, max_motion_len=55, num_tokens=VQ_CFG["nb_code"], num_quantizers=6,
        # v4 architecture params (from CLI)
        latent_dim=args.latent_dim, ff_size=args.ff_size,
        n_layers=args.n_layers, n_heads=args.n_heads,
        dropout=args.dropout, cond_drop_prob=args.cond_drop_prob, max_motion_length=196,
        cross_attn_interval=args.cross_attn_interval, cross_attn_heads=0,
        text_adapter_layers=args.text_adapter_layers, gate_init=args.gate_init,
        unit_length=4, max_token_len=49,
        factorized_attn=args.factorized_attn,
        mask_2d_hybrid=args.mask_2d_hybrid,
        mask_2d_ratio=args.mask_2d_ratio,
        # control params
        ttt_gumbel=args.ttt_gumbel,
        ttt_dynamic=args.ttt_dynamic,
        rgar_cfg=(dict(check_every=10, plateau_rtol=0.01, s2_entry_tau=args.rgar_tau)
                  if args.rgar else None),
        gn_cfg=(dict(max_steps=args.gn_steps, tau=args.gn_tau, lam0=args.gn_lam0,
                     anchor_cap=args.gn_anchor_cap, prox_beta=args.gn_prox_beta,
                     jac_chunk=args.gn_jac_chunk,
                     metric=args.gn_metric, ridge=args.gn_ridge)
                if args.gn else None),
        seq_gen=args.seq_gen,
        gn_batch=args.gn_batch,
        ctrl_net=True, each_lr=args.each_lr, each_iter=args.each_iter,
        last_lr=args.last_lr if args.last_lr is not None else args.each_lr,
        last_iter=args.last_iter,
        device=device, control=args.control, dataset_name="t2m",
        dataset_opt_path=dataset_opt_path,
        vq_checkpoint=vq_checkpoint,
        vq_partition_file=vq_partition_file,
        mean_npy=mean_npy, std_npy=std_npy,
    )

    os.environ.setdefault("MASKCONTROL_CLIP_MODEL_PATH",
                          "/scratch/ts1v23/workspace/MaskControl/artifacts/models/clip/ViT-B-32.pt")

    ct2m = ControlTransformerTConcatV4(
        code_dim=128, cond_mode="text",
        latent_dim=args.latent_dim, ff_size=args.ff_size,
        num_layers=args.n_layers, num_heads=args.n_heads,
        dropout=args.dropout, clip_dim=512, cond_drop_prob=args.cond_drop_prob,
        clip_version="ViT-B/32", opt=opt,
        mean=torch.tensor(eval_val_loader.dataset.mean, requires_grad=False).to(device),
        std=torch.tensor(eval_val_loader.dataset.std, requires_grad=False).to(device),
        trans_path=base_path,
        vq_model=vq_model,
        control=args.control,
        kv_rank=args.kv_rank,
    )

    # Load ControlNet checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    key = "ct2m_transformer" if "ct2m_transformer" in ckpt else "trans"
    # GATE0 fix F2: filter embedded vq_model.* keys so a ckpt can never silently overwrite the pristine PartVQ used for decoding
    ckpt_sd = {k: v for k, v in ckpt[key].items() if not k.startswith("vq_model.")}
    missing, unexpected = ct2m.load_state_dict(ckpt_sd, strict=False)
    non_clip_missing = [k for k in missing
                        if not (k.startswith("clip_model.") or k.startswith("vq_model."))]
    print(f"Loaded ControlNet from {args.ckpt}")
    if non_clip_missing:
        # GATE0 fix F2: non-CLIP missing keys are FATAL — a partial load silently evaluates the wrong model
        raise RuntimeError(f"non-CLIP missing keys in {args.ckpt}: {non_clip_missing}")
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    # GATE0 fix F2: assert the model's vq_model stayed bit-equal to the independently loaded pristine PartVQ after ckpt load
    if hasattr(ct2m, "vq_model"):
        _pristine = torch.load(vq_checkpoint, map_location="cpu")
        _pristine = _pristine["net"] if "net" in _pristine else _pristine
        _model_vq_sd = ct2m.vq_model.state_dict()
        assert set(_model_vq_sd.keys()) == set(_pristine.keys()), \
            f"vq_model key set diverged from pristine PartVQ ({len(_model_vq_sd)} vs {len(_pristine)} keys)"
        for _k in _model_vq_sd:
            assert torch.equal(_model_vq_sd[_k].cpu(), _pristine[_k].cpu()), \
                f"vq_model weight diverged from pristine PartVQ after ckpt load: {_k}"
        print(f"  [F2] vq_model bit-equal to pristine PartVQ verified ({len(_model_vq_sd)} keys)")

    ct2m.to(device)
    ct2m.eval()
    ct2m.ctrl_net = True

    for repeat_id in range(args.repeat_times):
        print(f"\n{'='*60}")
        print(f"Repeat {repeat_id+1}/{args.repeat_times}: TTT each_iter={args.each_iter}, "
              f"ts={args.time_steps}, cfg={args.cond_scale}")
        print(f"{'='*60}\n")

        fid, diversity, R_precision, matching_score, skate_ratio, mm, traj_err, avoid_dist, kps_mean = \
            eval_t2m.evaluation_mask_transformer_test_plus_res(
                eval_val_loader, vq_model, None,
                ct2m, None,
                repeat_id, eval_wrapper=eval_wrapper,
                time_steps=args.time_steps, cond_scale=args.cond_scale,
                temperature=1, topkr=0.9,
                force_mask=False, cal_mm=False, f=None,
                pred_num_batch=args.pred_num_batch, logger=None, epoch=0,
                control=args.control, density=args.density, opt=opt,
            )

        print(f"\nRESULTS (repeat={repeat_id}):")
        print(f"  FID:         {fid:.4f}")
        print(f"  Top3:        {R_precision[2]:.4f}")
        print(f"  Top1:        {R_precision[0]:.4f}")
        print(f"  Match:       {matching_score:.4f}")
        print(f"  Diversity:   {diversity:.4f}")
        if kps_mean is not None:
            print(f"  KPS (cm):    {kps_mean*100:.2f}")
        if traj_err is not None:
            print(f"  Traj err:    {np.mean(traj_err):.4f}")


if __name__ == "__main__":
    main()

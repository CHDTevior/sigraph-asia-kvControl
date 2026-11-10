import argparse
import json
from argparse import Namespace
from math import sqrt
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import torch

from utils.fixseed import fixseed
from utils.get_opt import get_opt


ROOT = Path(__file__).resolve().parent
DEFAULT_MAIN_CHECKPOINT = Path(
    "/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/"
    "z2026-03-02-21-53-07_c_concat_main_ddp_part_aware_vae_overlap_20260302/"
    "model/net_best_fid.tar"
)
DEFAULT_VQ_CHECKPOINT = Path(
    "/scratch/ts1v23/workspace/part-aware-vqvae/output/vq/"
    "2026-03-02-14-06-17_vq_overlap_20260302/net_best_fid.pth"
)
DEFAULT_PARTITION_FILE = Path(
    "/scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis/skeleton_partition.json"
)
DEFAULT_DATASET_OPT = Path("checkpoints/t2m/Comp_v6_KLD005/opt.txt")
ALLOWED_DATASET = "t2m"
ALLOWED_SPLITS = ("val", "test")
ALLOWED_WHICH_EPOCH_ALIASES = {"best_fid": "net_best_fid.tar", "latest": "latest.tar"}

VQ_CFG = {
    "dataname": "t2m",
    "batch_size": 256,
    "window_size": 64,
    "total_iter": 300000,
    "warm_up_iter": 1000,
    "lr": 2e-4,
    "lr_scheduler": [200000],
    "gamma": 0.05,
    "weight_decay": 0.0,
    "commit": 0.02,
    "loss_vel": 0.5,
    "recons_loss": "l1_smooth",
    "code_dim": 128,
    "nb_code": 128,
    "mu": 0.99,
    "down_t": 2,
    "stride_t": 2,
    "width": 512,
    "depth": 3,
    "dilation_growth_rate": 3,
    "output_emb_width": 128,
    "vq_act": "relu",
    "vq_norm": None,
    "quantizer": "ema_reset",
    "beta": 1.0,
    "resume_pth": None,
    "resume_gpt": None,
    "out_dir": "output",
    "results_dir": "visual_results/",
    "visual_name": "baseline",
    "exp_name": "cfg_eval",
    "print_iter": 200,
    "eval_iter": 5000,
    "seed": 3407,
    "vis_gt": False,
    "nb_vis": 20,
    "sep_uplow": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Standalone CFG evaluation driver aligned to the training-time main "
            "evaluation path for the 2026-03-02 HumanML3D best-FID checkpoint."
        ),
    )
    parser.add_argument("--dataset_name", type=str, default=ALLOWED_DATASET)
    parser.add_argument("--split", type=str, default="val", choices=ALLOWED_SPLITS)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10107)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--which_epoch", type=str, default="best_fid")
    parser.add_argument("--repeat_times", type=int, default=5)
    parser.add_argument("--cond_scale", type=float, default=3.25)
    parser.add_argument("--time_steps", type=int, default=18)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--main_checkpoint", type=Path, default=DEFAULT_MAIN_CHECKPOINT)
    parser.add_argument("--vq_checkpoint", type=Path, default=DEFAULT_VQ_CHECKPOINT)
    parser.add_argument("--vq_partition_file", type=str, default=str(DEFAULT_PARTITION_FILE),
                        help='Path to skeleton partition JSON. Pass "" to disable (for MMM VQ with hardcoded partitions).')
    parser.add_argument("--nb_code", type=int, default=None,
                        help="Override VQ codebook size (default: VQ_CFG[nb_code]=128). Required for MMM 8192/512 etc.")
    parser.add_argument("--dataset_opt_path", type=Path, default=DEFAULT_DATASET_OPT)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(".codex-research/artifacts/cfg_scan/manual"),
    )
    parser.add_argument("--ext", type=str, default="cfg_eval")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate path resolution and output contract without running evaluation.",
    )
    return parser


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve_repo_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def resolve_main_checkpoint(authoritative_best_fid: Path, which_epoch: str) -> Path:
    require(authoritative_best_fid.is_file(), f"main checkpoint not found: {authoritative_best_fid}")
    model_dir = authoritative_best_fid.parent
    require(model_dir.name == "model", f"expected checkpoint under a model directory, got: {authoritative_best_fid}")
    if which_epoch == "all":
        raise RuntimeError(
            "--which_epoch=all is not supported in the CFG scan driver because it breaks deterministic provenance."
        )
    if which_epoch in ALLOWED_WHICH_EPOCH_ALIASES:
        resolved = model_dir / ALLOWED_WHICH_EPOCH_ALIASES[which_epoch]
    elif which_epoch.endswith(".tar"):
        resolved = model_dir / which_epoch
    else:
        raise RuntimeError(
            f"unsupported --which_epoch={which_epoch!r}; use one of "
            f"{sorted(ALLOWED_WHICH_EPOCH_ALIASES)} or an explicit '*.tar' filename."
        )
    require(resolved.is_file(), f"requested checkpoint not found: {resolved}")
    return resolved


def validate_scan_scope(args: argparse.Namespace) -> None:
    require(
        args.dataset_name == ALLOWED_DATASET,
        f"dataset drift detected: expected {ALLOWED_DATASET}, got {args.dataset_name}",
    )
    require(
        args.split in ALLOWED_SPLITS,
        f"unsupported split: {args.split}. Must be one of {ALLOWED_SPLITS}",
    )
    require(args.repeat_times > 0, f"repeat_times must be > 0, got {args.repeat_times}")
    require(args.batch_size > 0, f"batch_size must be > 0, got {args.batch_size}")
    require(args.time_steps > 0, f"time_steps must be > 0, got {args.time_steps}")
    require(args.temperature > 0, f"temperature must be > 0, got {args.temperature}")
    require(args.cond_scale > 0, f"cond_scale must be > 0, got {args.cond_scale}")
    if args.vq_partition_file and str(args.vq_partition_file).strip():
        require(resolve_repo_path(Path(args.vq_partition_file)).is_file(),
                f"vq_partition_file not found: {args.vq_partition_file}")


def load_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    require(isinstance(ckpt, dict), f"checkpoint must be a dict: {path}")
    return ckpt


def build_vq_args(vq_checkpoint: Path, partition_file) -> Namespace:
    cfg = dict(VQ_CFG)
    cfg["load_dir_vqvae"] = str(vq_checkpoint)
    cfg["partition_file"] = str(partition_file) if partition_file else None
    return Namespace(**cfg)


def load_vq_model(vq_checkpoint: Path, partition_file, device: torch.device):
    import models.vqvae as vqvae

    require(vq_checkpoint.is_file(), f"VQ checkpoint not found: {vq_checkpoint}")
    if partition_file:
        require(Path(partition_file).is_file(), f"partition file not found: {partition_file}")
    vq_args = build_vq_args(vq_checkpoint, partition_file)
    vq_model = vqvae.HumanVQVAE(
        vq_args,
        vq_args.nb_code,
        vq_args.code_dim,
        vq_args.output_emb_width,
        vq_args.down_t,
        vq_args.stride_t,
        vq_args.width,
        vq_args.depth,
        vq_args.dilation_growth_rate,
        vq_args.vq_act,
        vq_args.vq_norm,
    )
    ckpt = load_checkpoint(vq_checkpoint)
    if "net" in ckpt:
        state_dict = ckpt["net"]
    elif "vq_model" in ckpt:
        state_dict = ckpt["vq_model"]
    else:
        raise RuntimeError(
            f"VQ checkpoint {vq_checkpoint} does not contain 'net' or 'vq_model'; "
            "this does not look like the paired VQ checkpoint."
        )
    missing_keys, unexpected_keys = vq_model.load_state_dict(state_dict, strict=False)
    require(not unexpected_keys, f"unexpected VQ checkpoint keys: {unexpected_keys}")
    require(not missing_keys, f"missing VQ checkpoint keys: {missing_keys}")
    vq_model.to(device)
    vq_model.eval()
    return vq_model, vq_args


def load_transformer_model(checkpoint_path: Path, device: torch.device):
    from models.mask_transformer.factory import build_mask_transformer, normalize_transformer_variant

    root_dir = checkpoint_path.parent.parent
    model_opt_path = root_dir / "opt.txt"
    require(model_opt_path.is_file(), f"model opt.txt not found: {model_opt_path}")
    model_opt = get_opt(str(model_opt_path), device=device)
    require(
        model_opt.dataset_name == ALLOWED_DATASET,
        f"model opt dataset mismatch: expected {ALLOWED_DATASET}, got {model_opt.dataset_name}",
    )
    model_opt.num_tokens = VQ_CFG["nb_code"]
    model_opt.num_quantizers = 6
    model_opt.code_dim = VQ_CFG["code_dim"]
    model_opt.max_token_len = max(
        1,
        int(
            np.ceil(
                getattr(model_opt, "max_motion_length", 196)
                / max(1, getattr(model_opt, "unit_length", 4))
            )
        ),
    )
    model_opt.transformer_variant = normalize_transformer_variant(
        getattr(model_opt, "transformer_variant", "c_concat")
    )

    trans = build_mask_transformer(
        model_opt.transformer_variant,
        code_dim=model_opt.code_dim,
        cond_mode="text",
        latent_dim=model_opt.latent_dim,
        ff_size=model_opt.ff_size,
        num_layers=model_opt.n_layers,
        num_heads=model_opt.n_heads,
        dropout=model_opt.dropout,
        clip_dim=512,
        cond_drop_prob=model_opt.cond_drop_prob,
        clip_version="ViT-B/32",
        opt=model_opt,
    )
    ckpt = load_checkpoint(checkpoint_path)
    require(
        "t2m_transformer" in ckpt,
        f"checkpoint {checkpoint_path} does not contain 't2m_transformer'; "
        "refusing to load residual/control checkpoints in the CFG scan driver.",
    )
    missing_keys, unexpected_keys = trans.load_state_dict(ckpt["t2m_transformer"], strict=False)
    require(
        not unexpected_keys,
        f"unexpected transformer checkpoint keys: {unexpected_keys}",
    )
    require(
        all(key.startswith("clip_model.") for key in missing_keys),
        f"unexpected missing transformer keys: {missing_keys}",
    )
    trans.to(device)
    trans.eval()
    return trans, model_opt, ckpt


def build_eval_stack(args: argparse.Namespace, device: torch.device, model_opt=None):
    from models.t2m_eval_wrapper import EvaluatorModelWrapper
    from motion_loaders.dataset_motion_loader import get_dataset_motion_loader

    dataset_opt_path = resolve_repo_path(args.dataset_opt_path)
    require(dataset_opt_path.is_file(), f"dataset opt path not found: {dataset_opt_path}")
    mean_path = getattr(model_opt, "mean_npy", None) if model_opt is not None else None
    std_path = getattr(model_opt, "std_npy", None) if model_opt is not None else None

    eval_val_loader, _ = get_dataset_motion_loader(
        str(dataset_opt_path),
        args.batch_size,
        args.split,
        device=device,
        mean_path=mean_path,
        std_path=std_path,
    )
    wrapper_opt = get_opt(str(dataset_opt_path), device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    return eval_val_loader, eval_wrapper, dataset_opt_path


def evaluate_once(
    val_loader,
    transformer,
    vq_model,
    eval_wrapper,
    time_steps: int,
    cond_scale: float,
    temperature: float,
):
    from utils.metrics import (
        calculate_activation_statistics,
        calculate_diversity,
        calculate_frechet_distance,
        calculate_R_precision,
        euclidean_distance_matrix,
    )

    motion_annotation_list = []
    motion_pred_list = []
    r_precision_real = 0
    r_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    nb_sample = 0

    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, _token = batch
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]
        pred_tok_len = torch.ceil(m_length.float() / 4.0).long()

        mids, _logits = transformer.generate(
            clip_text,
            m_length // 4,
            time_steps,
            cond_scale,
            temperature=temperature,
        )
        mids = torch.stack([part.unsqueeze(-1) for part in mids], dim=-1).squeeze(2)

        pred_motions = torch.zeros((bs, seq, pose.shape[-1]), device=pose.device)
        for idx in range(bs):
            pred_pose = vq_model.vqvae.forward_decoder(mids[idx : idx + 1, : pred_tok_len[idx].item()])
            pred_motions[idx : idx + 1, : m_length[idx].item()] = pred_pose

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
        )
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)

        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_r = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        r_precision_real += temp_r
        matching_score_real += temp_match

        temp_r = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        r_precision += temp_r
        matching_score_pred += temp_match
        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    r_precision_real = r_precision_real / nb_sample
    r_precision = r_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    return {
        "fid": float(fid),
        "diversity_real": float(diversity_real),
        "diversity": float(diversity),
        "top1": float(r_precision[0]),
        "top2": float(r_precision[1]),
        "top3": float(r_precision[2]),
        "matching_score_real": float(matching_score_real),
        "matching_score_pred": float(matching_score_pred),
    }


def ci95(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.std(arr) * 1.96 / sqrt(len(arr)))


def summarise_results(results: list[dict]) -> dict:
    summary = {"repeat_times": len(results)}
    fid_values = [result["fid"] for result in results]
    best_repeat = min(results, key=lambda item: (item["fid"], item["repeat_id"]))
    summary["min_fid"] = float(min(fid_values))
    summary["min_fid_repeat_id"] = int(best_repeat["repeat_id"])
    for key in [
        "fid",
        "diversity",
        "top1",
        "top2",
        "top3",
        "matching_score_pred",
    ]:
        values = [result[key] for result in results]
        summary[f"mean_{key}"] = float(np.mean(values))
        summary[f"{key}_ci95"] = ci95(values)
    return summary


def emit(handle, payload):
    message = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
    print(message)
    print(message, file=handle, flush=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.nb_code is not None:
        VQ_CFG["nb_code"] = int(args.nb_code)
    validate_scan_scope(args)
    fixseed(args.seed)

    authoritative_best_fid = resolve_repo_path(args.main_checkpoint)
    resolved_checkpoint = resolve_main_checkpoint(authoritative_best_fid, args.which_epoch)
    vq_checkpoint = resolve_repo_path(args.vq_checkpoint)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.ext}.log"

    header = {
        "event": "cfg_eval_preflight",
        "dataset_name": args.dataset_name,
        "split": args.split,
        "which_epoch": args.which_epoch,
        "resolved_checkpoint": str(resolved_checkpoint),
        "authoritative_best_fid_checkpoint": str(authoritative_best_fid),
        "vq_checkpoint": str(vq_checkpoint),
        "cond_scale": args.cond_scale,
        "time_steps": args.time_steps,
        "temperature": args.temperature,
        "repeat_times": args.repeat_times,
        "batch_size": args.batch_size,
        "out_path": str(out_path),
    }

    with out_path.open("w", encoding="utf-8") as handle:
        emit(handle, header)

        if args.dry_run:
            main_ckpt = load_checkpoint(resolved_checkpoint)
            require(
                "t2m_transformer" in main_ckpt,
                f"checkpoint {resolved_checkpoint} does not contain 't2m_transformer'; "
                "refusing to load residual/control checkpoints in the CFG scan driver.",
            )
            vq_ckpt = load_checkpoint(vq_checkpoint)
            require(
                "net" in vq_ckpt or "vq_model" in vq_ckpt,
                f"VQ checkpoint {vq_checkpoint} does not contain 'net' or 'vq_model'.",
            )
            emit(
                handle,
                {
                    "event": "cfg_eval_dry_run_complete",
                    "resolved_checkpoint": str(resolved_checkpoint),
                    "vq_checkpoint": str(vq_checkpoint),
                    "out_path": str(out_path),
                },
            )
            return

        require(torch.cuda.is_available(), "CUDA is required for CFG evaluation, but no GPU is available.")
        require(args.gpu_id >= 0, f"gpu_id must be >= 0 for evaluation, got {args.gpu_id}")
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)

        transformer, model_opt, ckpt = load_transformer_model(resolved_checkpoint, device)
        if args.vq_partition_file and str(args.vq_partition_file).strip():
            vq_partition_file = resolve_repo_path(Path(args.vq_partition_file))
        else:
            vq_partition_file = None
        vq_model, _vq_args = load_vq_model(vq_checkpoint, vq_partition_file, device)
        eval_val_loader, eval_wrapper, dataset_opt_path = build_eval_stack(args, device, model_opt=model_opt)

        emit(
            handle,
            {
                "event": "cfg_eval_runtime",
                "model_opt_path": str((resolved_checkpoint.parent.parent / "opt.txt")),
                "dataset_opt_path": str(dataset_opt_path),
                "checkpoint_epoch": int(ckpt.get("ep", -1)),
                "model_name": model_opt.name,
            },
        )

        results = []
        with torch.no_grad():
            for repeat_id in range(args.repeat_times):
                metrics = evaluate_once(
                    eval_val_loader,
                    transformer,
                    vq_model,
                    eval_wrapper,
                    time_steps=args.time_steps,
                    cond_scale=args.cond_scale,
                    temperature=args.temperature,
                )
                metrics["repeat_id"] = repeat_id
                results.append(metrics)
                emit(handle, {"event": "cfg_eval_repeat", **metrics})

        summary = summarise_results(results)
        emit(
            handle,
            {
                "event": "cfg_eval_final",
                "source_checkpoint": str(resolved_checkpoint),
                "authoritative_best_fid_checkpoint": str(authoritative_best_fid),
                "vq_checkpoint": str(vq_checkpoint),
                "which_epoch": args.which_epoch,
                "checkpoint_epoch": int(ckpt.get("ep", -1)),
                **summary,
            },
        )


if __name__ == "__main__":
    main()

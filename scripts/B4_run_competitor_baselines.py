#!/usr/bin/env python3
"""B4 — Run competitor baselines (MaskControl / OmniControl) on the 15 D1 picks.

Produces head-to-head pred_joints for paper Tab comparison vs our KV-Control.

KEY FACT (see COMPETITOR_INTEGRATION.md §1):
  MaskControl ships TWO ckpts that must be reported as SEPARATE rows:
    - trajectory (pelvis-only): l=1.1*XEnt + 0.9*TTT
    - cross (multi-joint 6):    l=1.5*XEnt + 0.5*TTT
  Different loss weights => different TTT behavior => cannot be averaged.

Per the user's instruction we condition BOTH ckpts on pelvis-only
ctrlNet_cond derived from gt_trajectory.npy for an apples-to-apples
comparison against our M3 pelvis-only headline. The "cross" ckpt is
exercised at pelvis-only conditioning (its encoder is wider but it
accepts a 6-joint tensor with only joint 0 non-zero).

OmniControl path: TODO — needs format adapter (raw-xyz layout +
their Mean_raw/Std_raw stats + sampler injection); see B5.

Layout (per pick):
  analysis/d1_picks/<idx>/gt_trajectory.npy       # (T, 3) pelvis xyz
  analysis/d1_picks/<idx>/text.txt                # 1-3 text variants
  analysis/d1_picks/<idx>/pred_joints_baseline_<method>.npy  # OUT (T, 22, 3)

Methods: maskcontrol_traj | maskcontrol_cross | omnicontrol

Usage:
  conda activate tlcontrol
  python scripts/B4_run_competitor_baselines.py \
      --maskcontrol_traj_ckpt /scratch/ts1v23/workspace/competitors/ckpts/maskcontrol_trajectory/z2024-08-23-01-27-51_CtrlNet_randCond1-196_l1.1XEnt.9TTT__fixRandCond \
      --maskcontrol_cross_ckpt /scratch/ts1v23/workspace/competitors/ckpts/maskcontrol_cross/z2024-08-27-21-07-55_CtrlNet_randCond1-196_l1.5XEnt.5TTT__cross \
      --methods maskcontrol_traj maskcontrol_cross \
      --gpu_id 0

ETA: ~3 min/pick (TTT iters dominate) * 15 picks * N methods, sequential on one GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/scratch/ts1v23/workspace/MaskControl").resolve()
# Use the upstream-clean MaskControl reference (no pdb.set_trace, no fork drift).
MC_ROOT = REPO_ROOT / "references" / "MaskControl"
PICKS_ROOT = REPO_ROOT / "analysis" / "d1_picks"
INDEX_JSON = PICKS_ROOT / "_index.json"

# MaskControl always pairs ctrl ckpt with the SAME base mtrans + RVQVAE + residual
# (no swapping — these are fixed by the public release).
# codex round-2 fix: trans_name literal wrong — public MaskControl ckpts use
# "1_mtrans_lossAllMaskNoMask" (verified by direct grep on opt.txt of both
# maskcontrol_trajectory and maskcontrol_cross 2026-06-29). The previous
# literal "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns" was an arch-name
# guess that would never match and silently pass the None-branch.
MC_BASE_TRANS_NAME = "1_mtrans_lossAllMaskNoMask"
MC_VQ_NAME = "rvq_nq6_dc512_nc512_noshare_qdp0.2"
MC_RES_NAME = "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--maskcontrol_traj_ckpt", type=str, default=None,
                   help="Directory of MaskControl trajectory ctrl ckpt "
                        "(contains model/, opt.txt).")
    p.add_argument("--maskcontrol_cross_ckpt", type=str, default=None,
                   help="Directory of MaskControl cross ctrl ckpt.")
    p.add_argument("--omnicontrol_ckpt", type=str, default=None,
                   help="Path to OmniControl HumanML3D ckpt .pt file.")
    p.add_argument("--methods", nargs="+", required=True,
                   choices=["maskcontrol_traj", "maskcontrol_cross", "omnicontrol"],
                   help="Which baselines to run sequentially.")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--which_model", type=str, default="latest.tar",
                   choices=["latest.tar", "net_best_acc.tar", "net_best_fid.tar"],
                   help="Which ckpt file inside model/ to load (MaskControl uses latest).")
    p.add_argument("--time_steps", type=int, default=10)
    p.add_argument("--cond_scale", type=float, default=4.0)
    p.add_argument("--iter_each", type=int, default=100,
                   help="MaskControl logits-opt iters per unmask step (matches their default).")
    p.add_argument("--iter_last", type=int, default=600,
                   help="MaskControl logits-opt iters at last unmask step.")
    p.add_argument("--each_lr", type=float, default=6e-2)
    p.add_argument("--last_lr", type=float, default=6e-2)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--picks", nargs="+", default=None,
                   help="Optional subset of pick idx folders (default: all from _index.json).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run picks whose output .npy already exists.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Picks
# ---------------------------------------------------------------------------
def load_picks(args: argparse.Namespace) -> list[dict]:
    """Return list of pick dicts from _index.json (or numeric subdirs as fallback)."""
    if INDEX_JSON.exists():
        with open(INDEX_JSON) as f:
            picks = json.load(f)["picks"]
    else:
        picks = []
        for d in sorted(PICKS_ROOT.iterdir()):
            if d.is_dir() and d.name.isdigit():
                picks.append({"idx": d.name})
    if args.picks:
        wanted = set(args.picks)
        picks = [p for p in picks if p["idx"] in wanted]
    return picks


def read_pick_text(pick_dir: Path, pick_meta: dict) -> str:
    """Use first non-empty line of text.txt; fallback to meta['text']."""
    txt_file = pick_dir / "text.txt"
    if txt_file.exists():
        for line in txt_file.read_text().splitlines():
            line = line.strip()
            if line:
                return line
    return pick_meta.get("text", "a person moves.")


# ---------------------------------------------------------------------------
# MaskControl setup
# ---------------------------------------------------------------------------
def setup_maskcontrol_env() -> None:
    """Add references/MaskControl/ to sys.path so its modules win over our fork."""
    # Insert BEFORE the repo root so MaskControl's models/* are picked up.
    sys.path.insert(0, str(MC_ROOT))
    # chdir so relative paths inside MaskControl (e.g. ./generation/moment.npy,
    # ./checkpoints/...) resolve.
    os.chdir(str(MC_ROOT))


def ensure_ctrl_ckpt_symlink(user_ckpt_dir: str) -> str:
    """MaskControl resolves ckpts via opt.checkpoints_dir + opt.dataset_name + ctrl_name.
    The user passes an absolute path; symlink it into ./checkpoints/t2m/ if not there.
    Returns the ctrl_name (basename) to set on opt.
    """
    user_path = Path(user_ckpt_dir).resolve()
    if not user_path.exists():
        raise FileNotFoundError(f"ctrl ckpt dir not found: {user_path}")
    ctrl_name = user_path.name
    target = MC_ROOT / "checkpoints" / "t2m" / ctrl_name
    if target.is_symlink():
        # codex round-3 fix: stale broken symlinks under references/MaskControl/
        # checkpoints/t2m/ point to non-existent paths from old workspace
        # locations. Detect broken symlinks and silently replace them, since
        # they cannot resolve to anything meaningful anyway. Only refuse when
        # the existing symlink points to a real-but-different valid target.
        try:
            existing = target.resolve(strict=True)  # raises if broken
        except (FileNotFoundError, OSError):
            print(f"  [info] removing broken symlink {target}", file=sys.stderr)
            target.unlink()
            target.symlink_to(user_path)
            return ctrl_name
        if existing != user_path:
            raise FileExistsError(
                f"checkpoints/t2m/{ctrl_name} already a symlink resolving "
                f"to {existing}, but requested ckpt is {user_path}. "
                f"Refusing to run wrong ckpt. Remove or rename and re-run."
            )
    elif target.exists():
        # Real directory (not symlink) → cannot replace silently
        existing = target.resolve()
        if existing != user_path:
            raise FileExistsError(
                f"checkpoints/t2m/{ctrl_name} already exists as a directory "
                f"at {existing}, but requested ckpt is {user_path}. "
                f"Refusing to run wrong ckpt. Remove or rename and re-run."
            )
    else:
        target.symlink_to(user_path)
        print(f"[symlink] checkpoints/t2m/{ctrl_name} -> {user_path}")
    return ctrl_name


def build_maskcontrol_models(args: argparse.Namespace, ctrl_name: str,
                             control_mode: str, device: torch.device):
    """Reproduce generation.load_model.get_models() with our overrides.

    We DON'T call get_models() directly because it hardcodes ctrl_name and
    forces opening a log file in the ckpt dir (write-permission issues on
    read-only mirrors).
    """
    from utils.get_opt import get_opt  # noqa: E402  (MC sys.path)
    from generation.load_model import (  # noqa: E402
        load_vq_model, load_res_model, load_ctrltrans_model,
    )

    clip_version = "ViT-B/32"
    checkpoints_dir = "./checkpoints"
    dataset_name = "t2m"

    # Ctrl model_opt (this is what defines num_layers/n_heads/etc.)
    model_opt_path = pjoin(checkpoints_dir, dataset_name, ctrl_name, "opt.txt")
    model_opt = get_opt(model_opt_path, device=device)

    # codex fix: unused MC_*_NAME constants — assert the loaded ctrl opt
    # actually points at the public MaskControl base mtrans + RVQVAE pair.
    # MaskControl ships ckpts that are fixed-paired with exactly one
    # base trans + one VQ; a mis-pointed opt.txt would silently pull a
    # different base and produce nonsense paper numbers. Fail fast.
    got_vq = getattr(model_opt, "vq_name", None)
    if got_vq != MC_VQ_NAME:
        raise AssertionError(
            f"ctrl opt.txt at {model_opt_path} has vq_name={got_vq!r}, "
            f"expected {MC_VQ_NAME!r} (MaskControl public release)."
        )
    # codex round-2 fix: hard-assert trans_name — previously the assert
    # passed silently when getattr returned None (e.g. older opt.txt fields
    # renamed). Treat missing field as a fatal mismatch so we never quietly
    # exercise a non-public base on top of a public ctrl ckpt.
    got_trans = getattr(model_opt, "trans_name", None)
    if got_trans != MC_BASE_TRANS_NAME:
        raise AssertionError(
            f"ctrl opt.txt at {model_opt_path} has trans_name={got_trans!r}, "
            f"expected {MC_BASE_TRANS_NAME!r} (MaskControl public release)."
        )

    # VQ
    vq_opt_path = pjoin(checkpoints_dir, dataset_name, model_opt.vq_name, "opt.txt")
    vq_opt = get_opt(vq_opt_path, device=device)
    vq_model, vq_opt = load_vq_model(vq_opt)
    vq_model = vq_model.to(device).eval()
    for p in vq_model.parameters():
        p.requires_grad = False

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    # Residual
    res_opt_path = pjoin(checkpoints_dir, dataset_name, MC_RES_NAME, "opt.txt")
    res_opt = get_opt(res_opt_path, device=device)
    res_model = load_res_model(res_opt, vq_opt, clip_version)
    res_model = res_model.to(device).eval()
    for p in res_model.parameters():
        p.requires_grad = False

    # Moment
    moment = np.load(MC_ROOT / "generation" / "moment.npy", allow_pickle=True)

    # Ctrl model (load_ctrltrans_model reads its 'opt' arg for ctrl_name + control)
    loader_opt = argparse.Namespace(
        checkpoints_dir=checkpoints_dir, dataset_name=dataset_name,
        ctrl_name=ctrl_name, control=control_mode, gpu_id=args.gpu_id,
    )
    ct2m_transformer = load_ctrltrans_model(
        model_opt, loader_opt, args.which_model, clip_version, vq_model, moment,
    )
    ct2m_transformer = ct2m_transformer.to(device).eval()
    ct2m_transformer.res_model = res_model
    ct2m_transformer.res_model.process_embed_proj_weight()
    ct2m_transformer.ctrl_eval()
    ct2m_transformer.TTT = True
    ct2m_transformer.vq_model = vq_model
    ct2m_transformer.ctrl_net = True
    for p in ct2m_transformer.parameters():
        p.requires_grad = False

    return ct2m_transformer, vq_model, res_model, moment


def build_pelvis_cond(traj: np.ndarray, n_frames: int, device: torch.device,
                      pad_to: int = 196):
    """Pelvis-only ctrlNet conditioning tensor.

    Args:
        traj: (T, 3) pelvis xyz from gt_trajectory.npy
        n_frames: actual sequence length T (<= pad_to)
        pad_to: MaskControl expects fixed 196-frame buffer.

    Returns:
        global_joint:      (1, 196, 22, 3) float, joint-0 set on first T frames
        global_joint_mask: (1, 196, 22)    bool,  True only on (joint=0, frame<T)
    """
    assert traj.ndim == 2 and traj.shape[1] == 3, traj.shape
    T = min(int(n_frames), int(traj.shape[0]), pad_to)
    gj = torch.zeros((1, pad_to, 22, 3), device=device, dtype=torch.float32)
    gj[0, :T, 0, :] = torch.from_numpy(traj[:T]).to(device).float()
    # The dense pelvis trajectory mask. (sum(-1)!=0) would also work but
    # explicit mask is robust to a literal (0,0,0) pelvis frame.
    gjm = torch.zeros((1, pad_to, 22), device=device, dtype=torch.bool)
    # KF_DENSITY mirrors the same env in D1_stage2_kv_inference.py so both methods can be run
    # at the SAME control density. Unset/"dense" = every frame (historical behaviour, unchanged).
    _dens = os.environ.get("KF_DENSITY", "dense")
    if _dens == "dense":
        gjm[0, :T, 0] = True
    else:
        k = int(_dens)
        if not (1 <= k <= T):
            raise ValueError(f"KF_DENSITY={k} must be in [1, {T}]")
        sel = np.linspace(0, T - 1, k).round().astype(int)
        gjm[0, torch.from_numpy(sel).to(device).long(), 0] = True
    return gj, gjm, T


def run_maskcontrol_method(args: argparse.Namespace, ctrl_dir: str,
                           control_mode: str, method_tag: str,
                           picks: list[dict], device: torch.device) -> None:
    """Run all picks through one MaskControl ctrl ckpt."""
    print(f"\n{'='*72}\n[{method_tag}] loading models from {ctrl_dir}\n{'='*72}")
    ctrl_name = ensure_ctrl_ckpt_symlink(ctrl_dir)
    ct2m, vq_model, res_model, moment = build_maskcontrol_models(
        args, ctrl_name, control_mode, device,
    )
    mean_t = torch.tensor(moment[0]).to(device).float()
    std_t = torch.tensor(moment[1]).to(device).float()

    t_start_method = time.time()
    for i, pk in enumerate(picks):
        idx = pk["idx"]
        pick_dir = PICKS_ROOT / idx
        # B4_OUT_DIR redirects sweep outputs away from the pick directories (same hazard the
        # D1 --out_dir guard exists for: a sweep must never overwrite the paper's baselines).
        # B4_OUT_SUFFIX distinguishes budget variants. Defaults: byte-identical behaviour.
        _sweep_dir = os.environ.get("B4_OUT_DIR")
        _sfx = os.environ.get("B4_OUT_SUFFIX", "")
        if _sweep_dir:
            Path(_sweep_dir).mkdir(parents=True, exist_ok=True)
            out_npy = Path(_sweep_dir) / f"{idx}_baseline_{method_tag}{_sfx}.npy"
        elif _sfx:
            raise SystemExit("B4_OUT_SUFFIX without B4_OUT_DIR would write sweep outputs into "
                             "the pick directories; set B4_OUT_DIR.")
        else:
            out_npy = pick_dir / f"pred_joints_baseline_{method_tag}.npy"
        if out_npy.exists() and not args.overwrite:
            print(f"[{method_tag}][{idx}] skip (exists; pass --overwrite to redo)")
            continue

        if os.environ.get("B4_OUT_DIR"):
            # sweep mode: mirror D1's per-pick fixseed(seed+i) so every budget sees the same
            # RNG state per pick (public MaskControl consumes Gumbel draws during refinement,
            # so without this, budget changes shift the RNG stream of later picks).
            from utils.fixseed import fixseed as _fs_b4
            _fs_b4(args.seed + i)
        traj = np.load(pick_dir / "gt_trajectory.npy")  # (T, 3)
        text = read_pick_text(pick_dir, pk)
        n_frames = int(pk.get("n_frames", traj.shape[0]))
        # MaskControl requires m_length > 30 (assert in forward).
        n_frames = max(n_frames, 32)
        # codex round-2 fix: n_frames clamp asymmetry — some D1 picks are 199
        # frames but MaskControl's max sequence is 196. We clamp baseline
        # output to 196 here; for the D1 metric script to be fair, OUR M3
        # outputs MUST also be cropped to 196 on the same picks. Asymmetric
        # crop (baseline 196 vs ours 199) = silent comparison error in Tab.
        # Do NOT change the clamp value (196 is a MaskControl hard limit);
        # instead warn loudly so the user knows which picks were truncated
        # and can mirror the crop in the metric pairing step.
        orig_n_frames = n_frames
        n_frames = min(n_frames, 196)
        # codex round-3 fix: floor to multiple of 4 (RVQ stride×down). D1 already
        # does this via n_eff = 4*(n_frames//4) per D1_stage2_kv_inference.py:207.
        # B4 must match or metric pairing breaks (11/15 picks have non-divisible
        # clamped lengths in current set).
        n_eff = 4 * (n_frames // 4)
        if n_eff != n_frames:
            print(
                f"  [info][{method_tag}][{idx}] floor n_frames {n_frames}->{n_eff} "
                f"to RVQ unit_length=4 to match D1 protocol",
                file=sys.stderr,
            )
        n_frames = n_eff
        if n_frames < orig_n_frames:
            print(
                f"[WARN][{method_tag}][{idx}] n_frames {orig_n_frames}->{n_frames} "
                f"(MaskControl 196 cap + RVQ unit_length=4 floor); CROP M3 PRED "
                f"TO {n_frames} FOR THIS PICK IN THE METRIC SCRIPT.",
                file=sys.stderr,
            )

        global_joint, global_joint_mask, T = build_pelvis_cond(
            traj, n_frames, device, pad_to=196,
        )
        m_length = torch.tensor([n_frames], device=device, dtype=torch.long)

        t0 = time.time()
        torch.cuda.synchronize()
        t0_gen = time.perf_counter()
        with torch.set_grad_enabled(True):
            pred_motions_denorm, _pred_motions = ct2m.generate_with_control(
                [text], m_length,
                time_steps=args.time_steps, cond_scale=args.cond_scale,
                temperature=1.0, topkr=0.9, force_mask=False,
                vq_model=vq_model,
                global_joint=global_joint, global_joint_mask=global_joint_mask,
                _mean=mean_t, _std=std_t,
                res_cond_scale=5, res_model=res_model,
                control_opt={
                    "each_lr": args.each_lr, "each_iter": args.iter_each,
                    "lr": args.last_lr, "iter": args.iter_last,
                },
            )
        torch.cuda.synchronize()
        dt_gen = time.perf_counter() - t0_gen
        pred_joints = pred_motions_denorm[0, :n_frames].detach().cpu().numpy()
        # codex fix: non-atomic np.save — a killed run leaves a partial
        # .npy that the skip-if-exists guard at L272-274 would later treat
        # as a valid output, silently feeding wrong numbers to the paper
        # table. Write to a tmp path then os.replace for atomicity (D1 fix).
        # codex round-2 fix: tmp suffix bug — using
        # out_npy.with_suffix(out_npy.suffix + ".tmp") gives "foo.npy.tmp",
        # but np.save auto-appends .npy when the path lacks that suffix,
        # producing "foo.npy.tmp.npy" on disk → os.replace(tmp_npy, ...)
        # fails with FileNotFoundError. Use ".tmp.npy" so the on-disk path
        # equals tmp_npy and the rename succeeds.
        tmp_npy = out_npy.with_suffix(".tmp.npy")
        np.save(tmp_npy, pred_joints)
        os.replace(tmp_npy, out_npy)
        dt = time.time() - t0
        print(f"[{method_tag}][{i+1:02d}/{len(picks)}][{idx}] T={n_frames} "
              f"shape={pred_joints.shape} dt={dt:.1f}s dt_gen={dt_gen:.2f}s -> {out_npy.name}")

    print(f"[{method_tag}] done in {(time.time()-t_start_method)/60:.1f} min")
    # Free GPU memory before next method.
    del ct2m, vq_model, res_model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# OmniControl — TODO
# ---------------------------------------------------------------------------
def run_omnicontrol(args: argparse.Namespace, picks: list[dict],
                    device: torch.device) -> None:
    """Stub: OmniControl requires a trajectory format adapter (see B5)."""
    msg = (
        "OmniControl integration NOT YET IMPLEMENTED.\n"
        "Required pre-work (see COMPETITOR_INTEGRATION.md §2):\n"
        "  1. Build /scratch/ts1v23/workspace/competitors/adapters/"
        "omnicontrol_traj_adapter.py:\n"
        "     converts our (T,3) pelvis traj -> OmniControl's (n_frames,22,3) "
        "raw-xyz + Mean_raw/Std_raw normalization.\n"
        "  2. Inject our fixed picks into their dataset sampler "
        "(override data_loaders/humanml/data/dataset.py).\n"
        "  3. Use the omnicontrol conda env (separate from tlcontrol).\n"
        "  4. Call sample.generate per pick with --num_repetitions 1.\n"
        f"  5. Save to {PICKS_ROOT}/<idx>/pred_joints_baseline_omnicontrol.npy\n"
        "TODO: file this as B5 task; do not block MaskControl head-to-head on this."
    )
    raise NotImplementedError(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    # codex round-3 fix: torch.cuda.set_device must be called BEFORE any model
    # construction to ensure bare .cuda() calls (used by MaskControl upstream
    # ControlTransformer ctor for self.mean/self.std tensor attrs) bind to our
    # intended GPU. Without this, model components land on whatever CUDA default
    # is (cuda:0), breaking multi-GPU-visible setups. Assert that gpu_id matches
    # default when CUDA_VISIBLE_DEVICES has multiple GPUs.
    if torch.cuda.is_available():
        n_visible = torch.cuda.device_count()
        if n_visible > 1 and args.gpu_id != 0:
            raise SystemExit(
                f"--gpu_id={args.gpu_id} but {n_visible} CUDA devices visible. "
                f"MaskControl upstream uses bare .cuda() (default cuda:0); "
                f"either set CUDA_VISIBLE_DEVICES={args.gpu_id} to expose only "
                f"that device, or pass --gpu_id 0 with one visible device."
            )
        torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    picks = load_picks(args)
    print(f"[setup] {len(picks)} picks; methods={args.methods}; device={device}")

    # Determine if any MaskControl method is requested.
    need_mc = any(m.startswith("maskcontrol") for m in args.methods)
    if need_mc:
        if "maskcontrol_traj" in args.methods and not args.maskcontrol_traj_ckpt:
            raise SystemExit("--maskcontrol_traj_ckpt required for maskcontrol_traj")
        if "maskcontrol_cross" in args.methods and not args.maskcontrol_cross_ckpt:
            raise SystemExit("--maskcontrol_cross_ckpt required for maskcontrol_cross")
        setup_maskcontrol_env()  # sys.path + chdir AFTER arg parsing

    failed = []
    for method in args.methods:
        try:
            if method == "maskcontrol_traj":
                run_maskcontrol_method(
                    args, args.maskcontrol_traj_ckpt,
                    control_mode="trajectory", method_tag="maskcontrol_traj",
                    picks=picks, device=device,
                )
            elif method == "maskcontrol_cross":
                run_maskcontrol_method(
                    args, args.maskcontrol_cross_ckpt,
                    control_mode="cross", method_tag="maskcontrol_cross",
                    picks=picks, device=device,
                )
            elif method == "omnicontrol":
                run_omnicontrol(args, picks, device)
            else:
                raise ValueError(method)
        except Exception as exc:
            print(f"\n[FAIL] method={method}: {exc}\n", file=sys.stderr)
            traceback.print_exc()
            failed.append((method, repr(exc)))

    print("\n" + "="*72)
    if failed:
        print(f"[B4] {len(failed)} method(s) failed:")
        for m, err in failed:
            print(f"  - {m}: {err}")
        return 1
    print(f"[B4] all {len(args.methods)} method(s) completed across {len(picks)} picks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

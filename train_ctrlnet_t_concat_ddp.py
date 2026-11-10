"""
DDP Training script for T-concat ControlNet.

Usage (single-GPU, backward compatible):
    python train_ctrlnet_t_concat_ddp.py --name t_concat_ctrlnet_traj ...

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=2 train_ctrlnet_t_concat_ddp.py \
        --name t_concat_ctrlnet_traj_ddp2 \
        --dataset_name t2m \
        --batch_size 256 \
        --max_epoch 500 \
        --xent 0.5 --ctrl_loss 0.5 \
        --control trajectory \
        --trans_path /path/to/t_concat/model/net_best_fid.tar
"""
import os
import time
import datetime
import argparse
from collections import defaultdict, OrderedDict
from os.path import join as pjoin

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from options.train_option import TrainT2MOptions
from utils.fixseed import fixseed
from utils.get_opt import get_opt
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain
from utils.utils import print_current_loss
import utils.eval_t2m as eval_t2m

from data.t2m_dataset import Text2MotionDataset
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
import models.vqvae as vqvae

# ===================== VQ config =====================
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
    "exp_name": "exp_debug",
    "print_iter": 200,
    "eval_iter": 5000,
    "seed": 3407,
    "vis_gt": False,
    "nb_vis": 20,
    "sep_uplow": False,
}


def load_vq_model(opt):
    vq_cfg = dict(VQ_CFG)
    vq_cfg["load_dir_vqvae"] = opt.vq_checkpoint
    # Normalize empty string → None so VQ uses hardcoded partSeg (e.g. MMM tokenizer)
    _pf = opt.vq_partition_file
    if isinstance(_pf, str) and _pf.strip() == "":
        _pf = None
    elif hasattr(_pf, "__fspath__") and str(_pf).strip() == "":
        _pf = None
    vq_cfg["partition_file"] = _pf
    # Also override nb_code from opt (allows MMM 8192 without editing VQ_CFG defaults)
    if hasattr(opt, "nb_code") and opt.nb_code:
        vq_cfg["nb_code"] = int(opt.nb_code)
    vq_args = argparse.Namespace(**vq_cfg)

    vq_net = vqvae.HumanVQVAE(
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
    ckpt = torch.load(opt.vq_checkpoint, map_location="cpu")
    if "net" in ckpt:
        state_dict = ckpt["net"]
    elif "vq_model" in ckpt:
        state_dict = ckpt["vq_model"]
    else:
        state_dict = ckpt
    vq_net.load_state_dict(state_dict)
    print(f"Loading VQ Model {opt.vq_checkpoint}")
    return vq_net, vq_args


# ===================== DDP helpers =====================
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def is_master():
    return (not is_dist_avail_and_initialized()) or dist.get_rank() == 0


@torch.no_grad()
def ddp_reduce_mean(x: float, device: torch.device):
    if not is_dist_avail_and_initialized():
        return x
    t = torch.tensor([x], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t.item()


# ===================== Trainer =====================
def def_value():
    return 0.0


class TconcatCtrlNetTrainerDDP:
    def __init__(self, args, ct2m_transformer, vq_model):
        self.opt = args
        self.ct2m_transformer = ct2m_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()

        self.ddp = getattr(args, "is_ddp", False) and is_dist_avail_and_initialized()
        self.master = getattr(args, "is_master", is_master())

        if args.is_train and self.master:
            self.logger = SummaryWriter(args.log_dir)
        else:
            self.logger = None

    def unwrap(self, model):
        return model.module if hasattr(model, "module") else model

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr
        return current_lr

    def force_adam_safe_mode(self):
        """Avoid PyTorch 2.10 foreach Adam step-counter corruption on resume."""
        for group in self.opt_t2m_transformer.param_groups:
            group["foreach"] = False
            group["fused"] = False
            group["capturable"] = False

    def repair_adam_state_steps(self, completed_it):
        """Ensure every trainable param has a nonzero Adam step before optimizer.step()."""
        target_step = max(int(completed_it), 1)
        repaired = 0
        for group in self.opt_t2m_transformer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.opt_t2m_transformer.state[p]
                step = state.get("step")
                needs_repair = step is None
                if step is not None:
                    step_value = step.item() if torch.is_tensor(step) else step
                    needs_repair = step_value <= 0
                if needs_repair:
                    state["step"] = torch.tensor(float(target_step), dtype=torch.float32)
                    state.setdefault("exp_avg", torch.zeros_like(p, memory_format=torch.preserve_format))
                    state.setdefault("exp_avg_sq", torch.zeros_like(p, memory_format=torch.preserve_format))
                    repaired += 1
        if repaired and self.master:
            print(f"[WARN] Repaired {repaired} Adam state step counter(s) to {target_step}")

    def forward(self, batch_data, use_unwrapped=False):
        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device, non_blocking=True)
        m_lens = m_lens.detach().long().to(self.device, non_blocking=True)
        conds = conds.to(self.device, non_blocking=True).float() if torch.is_tensor(conds) else conds

        model = self.unwrap(self.ct2m_transformer) if use_unwrapped else self.ct2m_transformer
        loss_emb, ce_loss, _pred_ids, _acc, loss_tta = model(
            conds, m_lens, motion
        )
        return loss_emb, ce_loss, loss_tta, _acc

    def update(self, batch_data, opt, it=None):
        loss_emb, ce_loss, loss_TTT, acc = self.forward(batch_data)
        self.opt_t2m_transformer.zero_grad(set_to_none=True)
        (0 * loss_emb + opt.xent * ce_loss + opt.ctrl_loss * loss_TTT).backward()
        if getattr(opt, 'grad_clip_norm', 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.ct2m_transformer.parameters() if p.requires_grad],
                opt.grad_clip_norm,
            )
        if it is not None:
            self.repair_adam_state_steps(completed_it=it - 1)
        self.opt_t2m_transformer.step()
        # For smooth schedulers (linear/cosine), skip scheduler.step() during warmup
        # (warmup overwrites lr directly; stepping scheduler would desync).
        # MultiStepLR is unaffected by warmup (milestones typically far past warmup).
        sched_kind = getattr(opt, 'lr_scheduler', 'multistep')
        in_warmup = (it is not None) and (it < getattr(opt, 'warm_up_iter', 0))
        if sched_kind in ('linear', 'cosine') and in_warmup:
            pass  # hold scheduler until warmup ends
        else:
            self.scheduler.step()

        if self.ddp:
            return (
                ddp_reduce_mean(loss_emb.item(), self.device),
                ddp_reduce_mean(ce_loss.item(), self.device),
                ddp_reduce_mean(loss_TTT.item(), self.device),
                ddp_reduce_mean(float(acc), self.device),
            )
        return loss_emb.item(), ce_loss.item(), loss_TTT.item(), float(acc)

    def save(self, file_name, ep, total_it):
        if not self.master:
            return
        model = self.unwrap(self.ct2m_transformer)
        t2m_trans_state_dict = model.state_dict()
        # GATE0 fix F3: also strip frozen vq_model.* — embedded VQ keys silently overwrite the pristine PartVQ at eval load
        clip_weights = [
            e for e in t2m_trans_state_dict.keys()
            if e.startswith("clip_model.") or e.startswith("vq_model.")
        ]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            "ct2m_transformer": t2m_trans_state_dict,
            "opt_t2m_transformer": self.opt_t2m_transformer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ep": ep,
            "total_it": total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        model = self.unwrap(self.ct2m_transformer)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["ct2m_transformer"], strict=False
        )
        if len(unexpected_keys) > 0 and self.master:
            print(f"  WARNING: {len(unexpected_keys)} unexpected keys (ignored): {unexpected_keys[:5]}...")
        if missing_keys and self.master:
            # GATE0 fix F3: vq_model.* is intentionally stripped from ckpts now (frozen tokenizer) — not a real missing key
            non_clip = [k for k in missing_keys
                        if not (k.startswith("clip_model.") or k.startswith("vq_model."))]
            if non_clip:
                print(f"  WARNING: {len(non_clip)} non-CLIP missing keys: {non_clip[:5]}...")
        try:
            self.opt_t2m_transformer.load_state_dict(
                checkpoint["opt_t2m_transformer"]
            )
            self.force_adam_safe_mode()
            self.repair_adam_state_steps(completed_it=checkpoint["total_it"])
            if getattr(self.opt, 'reset_scheduler', False):
                if self.master:
                    print(f"  Scheduler RESET: milestones={self.opt.milestones}, lr={self.opt.lr}")
                # Override optimizer lr to match the new schedule start
                for pg in self.opt_t2m_transformer.param_groups:
                    pg['lr'] = self.opt.lr
            else:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception:
            if self.master:
                print("Resume without optimizer")
        return checkpoint["ep"], checkpoint["total_it"]

    def train(
        self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval
    ):
        self.ct2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        model = self.unwrap(self.ct2m_transformer)

        # Only optimize control-specific parameters
        if hasattr(model, 'kv_down'):
            # v8 / aligned-kvq: KV-Context control
            if hasattr(model, 'ctrl_adapter'):
                # Aligned KV+Q: adapter + temporal + Q-residual (no encoder_control)
                ctrl_params = (
                    list(model.ctrl_adapter.parameters())
                    + (list(model.ctrl_temporal.parameters()) if hasattr(model, 'ctrl_temporal') else [])
                    + (list(model.ctrl_out_proj.parameters()) if hasattr(model, 'ctrl_out_proj') else [])
                    + list(model.kv_down.parameters())
                    + list(model.kv_up_k.parameters())
                    + list(model.kv_up_v.parameters())
                    + list(model.q_down.parameters())
                    + list(model.q_up.parameters())
                    + list(model.q_gates.parameters())
                    + list(model.ctrl_attn_bias.parameters())
                )
            else:
                # Original v8 + Q-residual: Conv encoder + KV + Q
                ctrl_params = (
                    list(model.encoder_control.parameters())
                    + list(model.kv_down.parameters())
                    + list(model.kv_up_k.parameters())
                    + list(model.kv_up_v.parameters())
                    + (list(model.ctrl_attn_bias.parameters()) if hasattr(model, 'ctrl_attn_bias') else [])
                    + (list(model.q_down.parameters()) if hasattr(model, 'q_down') else [])
                    + (list(model.q_up.parameters()) if hasattr(model, 'q_up') else [])
                    + (list(model.q_gates.parameters()) if hasattr(model, 'q_gates') else [])
                )
        else:
            # v5/v6/v7: parallel transformer + mid_zero variants
            ctrl_params = (
                list(model.seqTransEncoder_control.parameters())
                + list(model.encoder_control.parameters())
                + list(model.first_zero_linear.parameters())
            )
            if hasattr(model, 'mid_zero_down'):
                ctrl_params += list(model.mid_zero_down.parameters())
                ctrl_params += list(model.mid_zero_up.parameters())
                ctrl_params += list(model.q_coord.parameters())
                ctrl_params += list(model.q_coord_gate)
            elif hasattr(model, 'mid_zero_linear'):
                ctrl_params += list(model.mid_zero_linear.parameters())
        self.opt_t2m_transformer = optim.AdamW(
            ctrl_params, betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5,
            foreach=False, fused=False,
        )
        # GATE0 fix F3: ctrl_attn_bias must train (May ckpts show it moving off its -5 init); fail loud if it fell out of the optimizer
        if hasattr(model, 'ctrl_attn_bias'):
            _opt_param_ids = {id(p) for g in self.opt_t2m_transformer.param_groups for p in g['params']}
            _bias_missing = [n for n, p in model.ctrl_attn_bias.named_parameters() if id(p) not in _opt_param_ids]
            assert not _bias_missing, f"ctrl_attn_bias params missing from optimizer (would silently freeze): {_bias_missing}"
            if self.master:
                print(f"  [F3] ctrl_attn_bias verified in optimizer param list "
                      f"({sum(1 for _ in model.ctrl_attn_bias.parameters())} params)")
        self.force_adam_safe_mode()
        sched_kind = getattr(self.opt, 'lr_scheduler', 'multistep')
        if sched_kind == 'linear':
            end_factor = float(self.opt.lr_floor) / float(self.opt.lr)
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.opt_t2m_transformer,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=self.opt.lr_decay_iters,
            )
            if self.master:
                print(f"  LR scheduler: LinearLR (lr={self.opt.lr:.2e} -> lr_floor={self.opt.lr_floor:.2e} "
                      f"over {self.opt.lr_decay_iters} iters, holds at floor after)")
        elif sched_kind == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.opt_t2m_transformer,
                T_max=self.opt.lr_decay_iters,
                eta_min=self.opt.lr_floor,
            )
            if self.master:
                print(f"  LR scheduler: CosineAnnealingLR (lr={self.opt.lr:.2e} -> eta_min={self.opt.lr_floor:.2e} "
                      f"over T_max={self.opt.lr_decay_iters} iters)")
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.opt_t2m_transformer,
                milestones=self.opt.milestones,
                gamma=self.opt.gamma,
            )
            if self.master:
                print(f"  LR scheduler: MultiStepLR (milestones={self.opt.milestones}, gamma={self.opt.gamma})")

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, "latest.tar")
            epoch, it = self.resume(model_dir)
            if self.master:
                print("Load model epoch:%d iterations:%d" % (epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)

        if self.master:
            print(f"Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}")
            print(
                "Iters Per Epoch, Training: %04d, Validation: %03d"
                % (len(train_loader), len(val_loader))
            )

        logs = defaultdict(def_value, OrderedDict())

        best_acc = 0.0
        best_fid = float("inf")
        best_top3 = 0.0
        best_kps = float("inf")

        # CtrlNet eval settings — MaskControl standard protocol
        # (ts=10, TTT=35 dynamic, cfg=3.25 default, bs=32 via eval_val_loader)
        # cfg override via env var TRAIN_EVAL_COND_SCALE (e.g. v5.1-mmmN512 base sweep best_fid @ cfg=4.0)
        opt = self.opt
        opt.time_steps = getattr(opt, 'eval_time_steps', 10)
        opt.cond_scale = float(os.environ.get('TRAIN_EVAL_COND_SCALE', '3.25'))
        opt.temperature = 1
        opt.topkr = 0.9
        opt.force_mask = False
        opt.which_epoch = "latest"
        opt.ttt_dynamic = getattr(opt, 'eval_ttt_dynamic', True)
        model.ctrl_net = True

        if self.ddp:
            dist.barrier()

        while epoch < self.opt.max_epoch:
            # DDP: set epoch for distributed sampler
            if self.ddp and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            model.ctrl_train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss_emb, loss, loss_TTT, acc = self.update(batch_data=batch, opt=opt, it=it)
                logs["loss_emb"] += loss_emb
                logs["loss"] += loss
                logs["loss_TTT"] += loss_TTT
                logs["acc"] += acc
                logs["lr"] += self.opt_t2m_transformer.param_groups[0]["lr"]

                if self.master and it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        if self.logger is not None:
                            try:
                                self.logger.add_scalar(
                                    "Train/%s" % tag, value / self.opt.log_every, it
                                )
                            except (FileNotFoundError, OSError, RuntimeError) as _e:
                                self.logger = None
                                print(f"[tb-disabled] {_e}")
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(
                        start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i
                    )

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)

            self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)
            epoch += 1

            # Validation (rank 0 only, use unwrapped model to avoid DDP sync)
            if self.master:
                print("Validation time:")
                self.vq_model.eval()
                model.ctrl_eval()

                val_loss_emb = []
                val_loss = []
                val_loss_TTT = []
                val_acc = []
                with torch.no_grad():
                    for i, batch_data in enumerate(val_loader):
                        loss_emb, loss, loss_TTT, acc = self.forward(batch_data, use_unwrapped=True)
                        val_loss_emb.append(loss_emb.item())
                        val_loss.append(loss.item())
                        val_loss_TTT.append(loss_TTT.item())
                        val_acc.append(acc.item() if torch.is_tensor(acc) else acc)

                print(
                    f"Validation loss:{np.mean(val_loss):.3f}, "
                    f"accuracy:{np.mean(val_acc):.3f}"
                )

                if self.logger is not None:
                    try:
                        self.logger.add_scalar("Val/loss_emb", np.mean(val_loss_emb), epoch)
                        self.logger.add_scalar("Val/loss", np.mean(val_loss), epoch)
                        self.logger.add_scalar("Val/loss_TTT", np.mean(val_loss_TTT), epoch)
                        self.logger.add_scalar("Val/acc", np.mean(val_acc), epoch)
                    except (FileNotFoundError, OSError, RuntimeError) as _e:
                        self.logger = None
                        print(f"[tb-disabled] {_e}")

                if np.mean(val_acc) > best_acc:
                    print(
                        f"Improved accuracy from {best_acc:.02f} to {np.mean(val_acc)}!!!"
                    )
                    self.save(pjoin(self.opt.model_dir, "net_best_acc.tar"), epoch, it)
                    best_acc = np.mean(val_acc)

                # Periodic checkpoint saving (save_every_e from opt)
                save_every = getattr(self.opt, 'save_every_e', 0)
                if save_every > 0 and epoch % save_every == 0:
                    self.save(
                        pjoin(self.opt.model_dir, f"net_ep{epoch:04d}.tar"), epoch, it
                    )

                # Periodic evaluation with control metrics
                eval_freq = getattr(self.opt, 'eval_every_e', 10)
                if epoch % eval_freq == 0 and epoch > 49:
                    if save_every <= 0:
                        self.save(
                            pjoin(self.opt.model_dir, f"net_ep{epoch}.tar"), epoch, it
                        )
                    try:
                        fid, diversity, R_precision, matching_score, skate_ratio, mm, traj_err, avoid_dist, kps_mean = \
                            eval_t2m.evaluation_mask_transformer_test_plus_res(
                                eval_val_loader, self.vq_model, None,
                                model, None,
                                i, eval_wrapper=eval_wrapper,
                                time_steps=opt.time_steps, cond_scale=opt.cond_scale,
                                temperature=opt.temperature, topkr=opt.topkr,
                                force_mask=opt.force_mask, cal_mm=False, f=None,
                                pred_num_batch=16, logger=self.logger, epoch=epoch,
                                control=opt.control, density=-1, opt=opt)
                        if fid < best_fid:
                            print(f"Improved FID from {best_fid:.4f} to {fid:.4f}!!!")
                            self.save(pjoin(self.opt.model_dir, "net_best_fid.tar"), epoch, it)
                            best_fid = fid
                        if R_precision[2] > best_top3:
                            print(f"Improved Top3 from {best_top3:.4f} to {R_precision[2]:.4f}!!!")
                            self.save(pjoin(self.opt.model_dir, "net_best_top3.tar"), epoch, it)
                            best_top3 = R_precision[2]
                        if kps_mean > 0 and kps_mean < best_kps:
                            print(f"Improved KPS from {best_kps:.4f} to {kps_mean:.4f}!!!")
                            self.save(pjoin(self.opt.model_dir, "net_best_kps.tar"), epoch, it)
                            best_kps = kps_mean
                    except Exception as e:
                        print(f"[WARN] Evaluation failed at epoch {epoch}: {e}")
                        import traceback
                        traceback.print_exc()

            if self.ddp:
                dist.barrier()


if __name__ == "__main__":
    parser = TrainT2MOptions()
    opt = parser.parse()

    # ControlNet-specific eval settings (MaskControl standard protocol)
    opt.ctrl_net = True
    opt.each_lr = 6e-2
    opt.each_iter = getattr(opt, 'eval_each_iter', 35)  # standard TTT iters per timestep
    opt.last_lr = 6e-2
    opt.last_iter = 0  # no final refine

    fixseed(opt.seed)

    # ---------- DDP init ----------
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    rank_env = int(os.environ.get("RANK", "0"))
    local_rank_env = int(os.environ.get("LOCAL_RANK", "0"))
    launched_by_torchrun = "LOCAL_RANK" in os.environ
    use_ddp = launched_by_torchrun and world_size_env > 1

    if use_ddp:
        # Set CUDA device BEFORE init_process_group, and pass device_id, to
        # avoid NCCL "init_process_group called before set_device" warning
        # and ensure the per-rank device binding is unambiguous (per Codex
        # review thread 019e033f, 2026-05-07).
        local_rank = local_rank_env
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl",
            rank=rank_env,
            world_size=world_size_env,
            timeout=datetime.timedelta(seconds=7200),
            device_id=device,
        )
        rank = rank_env
        world_size = world_size_env
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device(
            "cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id)
        )

    opt.device = device
    opt.rank = rank
    opt.world_size = world_size
    opt.local_rank = local_rank
    opt.is_ddp = use_ddp
    opt.is_master = (rank == 0)

    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, "model")
    opt.eval_dir = pjoin(opt.save_root, "animation")
    opt.log_dir = pjoin("./checkpoints/", opt.dataset_name, opt.name)

    if opt.is_master:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.eval_dir, exist_ok=True)
        os.makedirs(opt.log_dir, exist_ok=True)

    if use_ddp:
        dist.barrier()

    if opt.dataset_name == "t2m":
        opt.data_root = "./dataset/HumanML3D"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = opt.dataset_opt_path
    elif opt.dataset_name == "kit":
        opt.data_root = "./dataset/KIT-ML"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_len = 55
        kinematic_chain = kit_kinematic_chain
        dataset_opt_path = "./checkpoints/kit/Comp_v6_KLD005/opt.txt"
    else:
        raise KeyError("Dataset Does Not Exist")

    opt.text_dir = pjoin(opt.data_root, "texts")

    # Load VQ model
    vq_model, vq_opt = load_vq_model(opt)

    clip_version = "ViT-B/32"
    opt.num_tokens = vq_opt.nb_code  # 128
    opt.num_quantizers = 6

    # Load normalization stats
    mean = np.load(opt.mean_npy)
    std = np.load(opt.std_npy)

    # Datasets
    train_split_file = pjoin(opt.data_root, "train.txt")
    val_split_file = pjoin(opt.data_root, "val.txt")

    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler,
        shuffle=False,
        drop_last=True,
    )

    eval_split = getattr(opt, 'eval_split', 'test')
    eval_val_loader, _ = get_dataset_motion_loader(
        dataset_opt_path, 32, eval_split, device=device
    )

    wrapper_opt = get_opt(dataset_opt_path, torch.device("cuda"))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # Build T-concat ControlTransformer
    # USE_V5/V6_CONTROL: load backup code for resuming old checkpoints
    # USE_V8_CONTROL: load v8 KV-Context ControlNet
    import sys, importlib.util
    _bk_map = {
        # NOTE: more specific variants MUST come first — iter order decides priority
        "USE_KV_CCONCAT_V4": "control_transformer_kv_c_concat_v4.py",
        "USE_V8_V4_CONTROL_NOQRES": "control_transformer_t_concat_v8_kv_v4_noqres.py",
        "USE_V5_V4_CONTROL": "control_transformer_t_concat_v5_v4.py",
        "USE_V5_CONTROL": "control_transformer_t_concat_v5_bk.py",
        "USE_V6_CONTROL": "control_transformer_t_concat_v6_bk.py",
        "USE_V8_CONTROL": "control_transformer_t_concat_v8_kv.py",
        "USE_V8_V4_CONTROL": "control_transformer_t_concat_v8_kv_v4.py",
        "USE_ALIGNED_KVQ": "control_transformer_aligned_kvq.py",
    }
    _env_key = None
    for k in _bk_map:
        if os.environ.get(k):
            _env_key = k
            break
    if _env_key:
        _bk_path = os.path.join(
            os.path.dirname(__file__),
            "models", "mask_transformer", _bk_map[_env_key],
        )
        _bk_spec = importlib.util.spec_from_file_location(
            "models.mask_transformer.control_transformer_t_concat", _bk_path
        )
        _bk_mod = importlib.util.module_from_spec(_bk_spec)
        sys.modules["models.mask_transformer.control_transformer_t_concat"] = _bk_mod
        _bk_spec.loader.exec_module(_bk_mod)
        # Different class names per variant — most specific first
        if hasattr(_bk_mod, 'ControlTransformerKVCConcatV4'):
            ControlTransformerTConcat = _bk_mod.ControlTransformerKVCConcatV4
        elif hasattr(_bk_mod, 'AlignedKVQControlTransformer'):
            ControlTransformerTConcat = _bk_mod.AlignedKVQControlTransformer
        elif hasattr(_bk_mod, 'ControlTransformerTConcatV4'):
            ControlTransformerTConcat = _bk_mod.ControlTransformerTConcatV4
        else:
            ControlTransformerTConcat = _bk_mod.ControlTransformerTConcat
        print(f"[{_env_key}] Loaded from {_bk_path}")
    else:
        from models.mask_transformer.control_transformer_t_concat import (
            ControlTransformerTConcat,
        )

    trans_path = getattr(opt, "trans_path", "")
    if not trans_path:
        trans_dir = pjoin(
            opt.checkpoints_dir, opt.dataset_name, opt.trans_name, "model"
        )
        if os.path.exists(pjoin(trans_dir, "net_best_fid.tar")):
            trans_path = pjoin(trans_dir, "net_best_fid.tar")
        elif os.path.exists(pjoin(trans_dir, "latest.tar")):
            trans_path = pjoin(trans_dir, "latest.tar")
        else:
            raise FileNotFoundError(
                f"Cannot find T-concat base checkpoint in {trans_dir}. "
                f"Please provide --trans_path explicitly."
            )

    ct2m_transformer = ControlTransformerTConcat(
        code_dim=vq_opt.code_dim,
        cond_mode="text",
        latent_dim=opt.latent_dim,
        ff_size=opt.ff_size,
        num_layers=opt.n_layers,
        num_heads=opt.n_heads,
        dropout=opt.dropout,
        clip_dim=512,
        cond_drop_prob=opt.cond_drop_prob,
        clip_version=clip_version,
        opt=opt,
        mean=torch.tensor(eval_val_loader.dataset.mean, requires_grad=False).to(device),
        std=torch.tensor(eval_val_loader.dataset.std, requires_grad=False).to(device),
        trans_path=trans_path,
        vq_model=vq_model,
        control=opt.control,
        kv_rank=getattr(opt, 'kv_rank', 64),
    )

    pc_transformer = sum(
        param.numel() for param in ct2m_transformer.parameters_wo_clip()
    )
    if opt.is_master:
        print("Total parameters of T-concat ControlNet: {:.2f}M".format(pc_transformer / 1_000_000))

    ct2m_transformer = ct2m_transformer.to(device)

    if use_ddp:
        # ControlNet has frozen base params + CLIP/VQ submodules that may not participate
        # in every backward; use find_unused_parameters=True to avoid DDP deadlock
        ct2m_transformer = DDP(
            ct2m_transformer,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    trainer = TconcatCtrlNetTrainerDDP(opt, ct2m_transformer, vq_model)
    trainer.train(
        train_loader, val_loader, eval_val_loader, eval_wrapper=eval_wrapper, plot_eval=None
    )

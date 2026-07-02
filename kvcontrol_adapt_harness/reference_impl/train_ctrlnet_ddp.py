"""
Cross-alloc DDP fork of references/MaskControl/train_ctrlnet.py.

Designed for the cross-alloc DDP pattern verified 2026-06-01 (CLAUDE.md
section "同节点多 Slurm alloc 合并成 cross-alloc DDP"):
    - rendezvous is static (--node_rank/--master_addr/--master_port)
      set by the orchestrator (scripts/kv_mc_orchestrator.sh).
    - this script reads RANK / LOCAL_RANK / WORLD_SIZE from torchrun env
      BEFORE touching cuda, then init_process_group(backend='nccl',
      init_method='env://', timeout=2h).
    - rank-0-only for: prints, checkpoint writes, TB writes, and the
      EvaluatorModelWrapper / eval_val_loader (which are full-set eval-on-
      rank-0).
    - DistributedSampler on the train loader, set_epoch() each epoch.
    - The KVControlTransformer adapter param list is built from the
      unwrapped module (DDP wraps the same module).
    - dist.destroy_process_group() at exit.

ALL CLI args are inherited from TrainT2MOptions and forwarded unchanged.
Run via torchrun. Per the orchestrator, world_size=4 across 3 allocs
(2+1+1) on swarmh1002 with linear scaling (lr=8e-4 for 4x bs).
"""

import os
import datetime
import torch
import torch.distributed as dist
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from collections import defaultdict, OrderedDict

from models.mask_transformer.transformer import MaskTransformer
from models.mask_transformer.transformer_trainer import MaskTransformerTrainer
from models.vq.model import RVQVAE

from options.train_option import TrainT2MOptions

from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain

from data.t2m_dataset import Text2MotionDataset
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
import utils.eval_t2m as eval_t2m

from utils.utils import *
from utils.eval_t2m import evaluation_mask_transformer, evaluation_res_transformer
from models.mask_transformer.tools import *

from einops import rearrange, repeat


# ----------------------------------------------------------------------------
# DDP helpers
# ----------------------------------------------------------------------------
def ddp_init():
    """Read torchrun env BEFORE any cuda call; init NCCL process group."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1

    if use_ddp:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=7200),
        )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

    is_master = (rank == 0)
    # Print once at startup so cross-alloc topology is debuggable.
    print(f"[ddp] rank={rank} local_rank={local_rank} world={world_size} "
          f"is_master={is_master} device={device}")
    return rank, local_rank, world_size, is_master, device, use_ddp


def is_dist():
    return dist.is_available() and dist.is_initialized()


def barrier():
    if is_dist():
        dist.barrier()


@torch.no_grad()
def reduce_mean(x, device):
    if not is_dist():
        return float(x)
    t = torch.tensor([float(x)], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t.item()


def def_value():
    return 0.0


def plot_t2m(data, save_dir, captions, m_lengths):
    return


def load_vq_model(opt):
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    if opt.is_master:
        print(f'Loading VQ Model {opt.vq_name}')
    return vq_model, vq_opt


# ----------------------------------------------------------------------------
# CtrlNet trainer (DDP-aware)
# ----------------------------------------------------------------------------
class CtrlNetTrainer:
    def __init__(self, args, ct2m_transformer, vq_model):
        self.opt = args
        self.ct2m_transformer = ct2m_transformer
        # Set attributes on the unwrapped module so DDP-wrap doesn't hide them.
        self.unwrap(self.ct2m_transformer).vq_model = vq_model
        self.vq_model = vq_model
        self.device = args.device

        self.ddp = getattr(args, "is_ddp", False) and is_dist()
        self.master = getattr(args, "is_master", True)

        self.vq_model.eval()

        if args.is_train and self.master:
            self.logger = SummaryWriter(args.log_dir)
        else:
            self.logger = None

    @staticmethod
    def unwrap(m):
        return m.module if hasattr(m, "module") else m

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr
        return current_lr

    def forward(self, batch_data):
        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device, non_blocking=True)
        m_lens = m_lens.detach().long().to(self.device, non_blocking=True)

        conds = conds.to(self.device, non_blocking=True).float() if torch.is_tensor(conds) else conds

        # ct2m_transformer.__call__ -> DDP -> module.forward(conds, m_lens, motion)
        loss_emb, ce_loss, _pred_ids, _acc, loss_tta = self.ct2m_transformer(conds, m_lens, motion)
        return loss_emb, ce_loss, loss_tta, _acc

    def update(self, batch_data, opt):
        loss_emb, ce_loss, loss_TTT, acc = self.forward(batch_data)

        self.opt_t2m_transformer.zero_grad()
        (0 * loss_emb + opt.xent * ce_loss + opt.ctrl_loss * loss_TTT).backward()
        self.opt_t2m_transformer.step()
        self.scheduler.step()

        return loss_emb.item(), ce_loss.item(), loss_TTT.item(), acc

    def save(self, file_name, ep, total_it):
        # Rank-0-only writes — prevents cross-alloc shared-fs race.
        if not self.master:
            return
        model = self.unwrap(self.ct2m_transformer)
        t2m_trans_state_dict = model.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            'ct2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        # All ranks load the same checkpoint from the shared FS — deterministic
        # and avoids a manual broadcast. Same pattern as train_ctrlnet_ddp.py
        # (the project's other cross-alloc DDP trainer).
        checkpoint = torch.load(model_dir, map_location=self.device)
        model = self.unwrap(self.ct2m_transformer)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['ct2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        except Exception:
            if self.master:
                print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        TTT = True
        # DDP wrap happens in main; .to(device) already done there.
        # Just make sure vq_model lives on the right device.
        self.vq_model.to(self.device)

        # ---- Optimizer built from the unwrapped module (KV adapter access) ----
        _m = self.unwrap(self.ct2m_transformer)
        if hasattr(_m, 'kv_down'):
            # KV-Control adapter param list
            kv_params = (
                list(_m.encoder_control.parameters())
                + [p for mod in _m.kv_down for p in mod.parameters()]
                + [p for mod in _m.kv_up_k for p in mod.parameters()]
                + [p for mod in _m.kv_up_v for p in mod.parameters()]
                + list(_m.ctrl_attn_bias.parameters())
            )
            if getattr(_m, 'use_q_residual', False):
                kv_params += (
                    list(_m.q_down.parameters())
                    + list(_m.q_up.parameters())
                    + list(_m.q_gates.parameters())
                )
            self.opt_t2m_transformer = optim.AdamW(
                kv_params, betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
            n_trainable = sum(p.numel() for p in kv_params)
            if self.master:
                print(f'[KV-Control] optimizer: AdamW lr={self.opt.lr} weight_decay=1e-5 '
                      f'trainable_params={n_trainable / 1e6:.3f}M')
        else:
            # Legacy MaskControl ControlNet adapter
            self.opt_t2m_transformer = optim.AdamW(
                list(_m.seqTransEncoder_control.parameters()) +
                list(_m.encoder_control.parameters()) +
                list(_m.first_zero_linear.parameters()) +
                list(_m.mid_zero_linear.parameters()),
                betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.opt_t2m_transformer,
            milestones=self.opt.milestones,
            gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            if self.master:
                print("Load model epoch:%d iterations:%d" % (epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        if self.master:
            print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
            print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        # ---- pre-loop eval (rank-0 only; other ranks barrier) ----
        opt = self.opt
        i = 0
        opt.time_steps = 10
        opt.cond_scale = 4
        opt.temperature = 1
        opt.topkr = .9
        opt.force_mask = False
        _m.TTT = TTT
        opt.which_epoch = 'latest'

        best_kps_mean = float('inf')
        if self.master:
            best_fid, best_div, Rprecision, best_matching, best_skate_ratio, best_mm, traj_err, _avoid_dist, kps_mean = \
                eval_t2m.evaluation_mask_transformer_test_plus_res(
                    eval_val_loader, self.vq_model, None, _m, None,
                    i, eval_wrapper=eval_wrapper,
                    time_steps=opt.time_steps, cond_scale=opt.cond_scale,
                    temperature=opt.temperature, topkr=opt.topkr,
                    force_mask=opt.force_mask, cal_mm=True, f=None, pred_num_batch=16,
                    logger=self.logger, epoch=epoch,
                    control=opt.control, density=-1, opt=opt)
        barrier()
        best_acc = 0.

        while epoch < self.opt.max_epoch:
            # DistributedSampler: must set_epoch each epoch for proper shuffling.
            if self.ddp and hasattr(train_loader, "sampler") and \
                    hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            _m.ctrl_train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss_emb, loss, loss_TTT, acc = self.update(batch_data=batch, opt=opt)
                logs['loss_emb'] += loss_emb
                logs['loss'] += loss
                logs['loss_TTT'] += loss_TTT
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0 and self.master:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        if self.logger is not None:
                            self.logger.add_scalar('Train/%s' % tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)
                elif it % self.opt.log_every == 0:
                    # Non-master ranks also reset the rolling buffer to avoid drift.
                    logs = defaultdict(def_value, OrderedDict())

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            # ---- validation (each rank computes on its shard, master reduces) ----
            if self.master:
                print('Validation time:')
            self.vq_model.eval()
            _m.ctrl_eval()

            val_loss_emb = []
            val_loss = []
            val_loss_TTT = []
            val_acc = []
            with torch.no_grad():
                for j, batch_data in enumerate(val_loader):
                    le, l, lttt, a = self.forward(batch_data)
                    val_loss_emb.append(le.item())
                    val_loss.append(l.item())
                    val_loss_TTT.append(lttt.item())
                    val_acc.append(float(a))

            v_le = float(np.mean(val_loss_emb)) if val_loss_emb else 0.0
            v_l = float(np.mean(val_loss)) if val_loss else 0.0
            v_t = float(np.mean(val_loss_TTT)) if val_loss_TTT else 0.0
            v_a = float(np.mean(val_acc)) if val_acc else 0.0

            if self.ddp:
                v_le = reduce_mean(v_le, self.device)
                v_l = reduce_mean(v_l, self.device)
                v_t = reduce_mean(v_t, self.device)
                v_a = reduce_mean(v_a, self.device)

            if self.master:
                print(f"Validation loss:{v_l:.3f}, accuracy:{v_a:.3f}")
                if self.logger is not None:
                    self.logger.add_scalar('Val/loss_emb', v_le, epoch)
                    self.logger.add_scalar('Val/loss', v_l, epoch)
                    self.logger.add_scalar('Val/loss_TTT', v_t, epoch)
                    self.logger.add_scalar('Val/acc', v_a, epoch)

                if v_a > best_acc:
                    print(f"Improved accuracy from {best_acc:.02f} to {v_a:.02f}!!!")
                    self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                    best_acc = v_a

                if epoch % 5 == 0:
                    best_fid, best_div, Rprecision, best_matching, best_skate_ratio, best_mm, traj_err_key, _avoid_dist, kps_mean = \
                        eval_t2m.evaluation_mask_transformer_test_plus_res(
                            eval_val_loader, self.vq_model, None, _m, None,
                            i, eval_wrapper=eval_wrapper,
                            time_steps=opt.time_steps, cond_scale=opt.cond_scale,
                            temperature=opt.temperature, topkr=opt.topkr,
                            force_mask=opt.force_mask, cal_mm=True, f=None, pred_num_batch=16,
                            logger=self.logger, epoch=epoch,
                            control=opt.control, density=-1, opt=opt)
                    if best_kps_mean > kps_mean:
                        self.save(pjoin(self.opt.model_dir, 'best_kps.tar'), epoch, it)
                        best_kps_mean = kps_mean
                        print(f'[best_kps] new best_kps_mean={kps_mean:.4f} at epoch={epoch} iter={it}')
            barrier()


# ----------------------------------------------------------------------------
# main (DDP)
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()

    ### ADDED for CtrlNet EVAL ###
    opt.ctrl_net = True
    opt.each_lr = 6e-2
    opt.each_iter = 0
    opt.last_lr = 6e-2
    opt.last_iter = 0
    ############################

    fixseed(opt.seed)

    # DDP init: read RANK/LOCAL_RANK/WORLD_SIZE from env *before* cuda use.
    rank, local_rank, world_size, is_master, device, use_ddp = ddp_init()
    opt.rank = rank
    opt.local_rank = local_rank
    opt.world_size = world_size
    opt.is_master = is_master
    opt.is_ddp = use_ddp
    opt.device = device

    # set_detect_anomaly is OFF by default in DDP runs (perf hit); only
    # enable when explicitly requested for debug.
    if getattr(opt, "detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    if opt.name != 'TEMP':
        from exit.utils import init_save_folder
        if is_master:
            init_save_folder(opt.save_root)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./checkpoints/', opt.dataset_name, opt.name)

    if is_master:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.eval_dir, exist_ok=True)
        os.makedirs(opt.log_dir, exist_ok=True)
    barrier()

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_len = 55
        kinematic_chain = kit_kinematic_chain
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'
    else:
        raise KeyError('Dataset Does Not Exist')

    opt.text_dir = pjoin(opt.data_root, 'texts')

    vq_model, vq_opt = load_vq_model(opt)

    clip_version = 'ViT-B/32'
    opt.num_tokens = vq_opt.nb_code

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)

    # DDP samplers (only on rank>1 world); single-rank fallback keeps the
    # original shuffle=True behavior.
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                           rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size,
                                         rank=rank, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size,
                              num_workers=4,
                              pin_memory=True,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None),
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=opt.batch_size,
                            num_workers=4,
                            pin_memory=True,
                            sampler=val_sampler,
                            shuffle=False,
                            drop_last=True)

    # Eval val loader: built on ALL ranks because we use its dataset.mean / .std
    # for model construction (same source as the original single-GPU script — keeping
    # this consistent across ranks guarantees identical model init, which DDP
    # requires). EvaluatorModelWrapper is heavyweight (loads CLIP) — rank 0 only.
    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device)
    if is_master:
        wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    else:
        eval_wrapper = None
    barrier()

    # ----- Build ControlTransformer on ALL ranks (DDP requires identical init) -----
    _adapter_kwargs = {}
    if getattr(opt, 'control_adapter', 'ctrlnet') == 'kv':
        from models.mask_transformer.control_transformer_kv import KVControlTransformer as _CtrlModel
        _adapter_kwargs = dict(
            kv_rank=opt.kv_rank,
            ctrl_attn_bias_init=opt.ctrl_attn_bias_init,
            use_q_residual=(not opt.no_q_residual),
        )
        if is_master:
            print(f'[KV-Control] adapter=kv  rank={opt.kv_rank}  bias_init={opt.ctrl_attn_bias_init}  '
                  f'q_residual={(not opt.no_q_residual)}')
    else:
        from models.mask_transformer.control_transformer import ControlTransformer as _CtrlModel
        if is_master:
            print('[KV-Control] adapter=ctrlnet (MaskControl default)')

    # mean/std tensors must land on the per-rank device — sourced from
    # eval_val_loader.dataset on every rank (built above) so model init is
    # bitwise identical across ranks.
    mean_t = torch.tensor(eval_val_loader.dataset.mean, requires_grad=False).to(opt.device)
    std_t = torch.tensor(eval_val_loader.dataset.std, requires_grad=False).to(opt.device)

    ct2m_transformer = _CtrlModel(code_dim=vq_opt.code_dim,
                                  cond_mode='text',
                                  latent_dim=opt.latent_dim,
                                  ff_size=opt.ff_size,
                                  num_layers=opt.n_layers,
                                  num_heads=opt.n_heads,
                                  dropout=opt.dropout,
                                  clip_dim=512,
                                  cond_drop_prob=opt.cond_drop_prob,
                                  clip_version=clip_version,
                                  opt=opt,
                                  mean=mean_t,
                                  std=std_t,
                                  trans_path=f'./checkpoints/{opt.dataset_name}/{opt.trans_name}/model/latest.tar',
                                  vq_model=vq_model,
                                  control=opt.control,
                                  **_adapter_kwargs).to(opt.device)

    # Finalize freeze/unfreeze pattern BEFORE DDP wrap so the reducer sees the
    # right requires_grad set.
    ct2m_transformer.ctrl_train()

    if use_ddp:
        # find_unused_parameters=False: KV-Control adapters (encoder_control,
        # kv_down/up_k/up_v, ctrl_attn_bias, and — when use_q_residual —
        # q_down/q_up/q_gates) are touched on every forward via
        # _encode_ctrl_to_kv() + the per-layer loop, so no parameter goes
        # unused on a single training step.
        ct2m_transformer = DDP(ct2m_transformer,
                               device_ids=[local_rank],
                               output_device=local_rank,
                               broadcast_buffers=False,
                               find_unused_parameters=False)

    if is_master:
        try:
            _raw = ct2m_transformer.module if hasattr(ct2m_transformer, "module") else ct2m_transformer
            pc_transformer = sum(p.numel() for p in _raw.parameters_wo_clip())
            print('Total parameters of all models: {:.2f}M'.format(pc_transformer / 1_000_000))
        except Exception as e:
            print(f"[warn] param count failed: {e}")

    trainer = CtrlNetTrainer(opt, ct2m_transformer, vq_model)
    trainer.train(train_loader, val_loader, eval_val_loader,
                  eval_wrapper=eval_wrapper, plot_eval=plot_t2m)

    if use_ddp:
        dist.destroy_process_group()

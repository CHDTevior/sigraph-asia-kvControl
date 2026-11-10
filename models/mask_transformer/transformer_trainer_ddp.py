import os
import time
import torch
import torch.distributed as dist
from collections import defaultdict, OrderedDict
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.utils import print_current_loss
from os.path import join as pjoin
from utils.eval_t2m import evaluation_mask_transformer
from models.mask_transformer.tools import *

def def_value():
    return 0.0

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_master():
    return get_rank() == 0

@torch.no_grad()
def ddp_reduce_mean(x: float, device: torch.device):
    """把标量在所有进程求平均，返回 python float（rank0/非rank0 都能用）"""
    if not is_dist_avail_and_initialized():
        return x
    t = torch.tensor([x], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t.item()

class MaskTransformerTrainer:
    def __init__(self, args, t2m_transformer, vq_model):
        self.opt = args
        self.t2m_transformer = t2m_transformer
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

    # ------------------------------------------------------------------
    # EMA (Exponential Moving Average) support — v4.1
    # ------------------------------------------------------------------

    def _init_ema(self):
        """Initialize EMA shadow parameters from the current model state."""
        ema_decay = float(getattr(self.opt, 'ema_decay', 0.0))
        if ema_decay <= 0.0:
            self.ema_params = None
            return
        model = self.unwrap(self.t2m_transformer)
        # Deep-copy all parameters as shadow weights (detached, on same device)
        self.ema_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        if self.master:
            n_ema = len(self.ema_params)
            print(f'[v4.1] EMA initialized: decay={ema_decay}, tracking {n_ema} parameters')

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA shadow parameters after an optimizer step."""
        if self.ema_params is None:
            return
        decay = float(self.opt.ema_decay)
        model = self.unwrap(self.t2m_transformer)
        for name, param in model.named_parameters():
            if name in self.ema_params:
                self.ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def _swap_ema_weights(self):
        """Swap model weights with EMA weights (for eval/save)."""
        if self.ema_params is None:
            return {}
        model = self.unwrap(self.t2m_transformer)
        orig = {}
        for name, param in model.named_parameters():
            if name in self.ema_params:
                orig[name] = param.data.clone()
                param.data.copy_(self.ema_params[name])
        return orig

    def _restore_weights(self, orig):
        """Restore original weights after EMA swap."""
        if not orig:
            return
        model = self.unwrap(self.t2m_transformer)
        for name, param in model.named_parameters():
            if name in orig:
                param.data.copy_(orig[name])

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr
        return current_lr

    def get_loss_weights(self, cur_iter, total_iters):
        # Fixed CE weight override (--fixed_ce_weight >= 0 disables schedule)
        fixed = getattr(self.opt, 'fixed_ce_weight', -1.0)
        if fixed >= 0:
            return 1.0 - fixed, fixed

        # Three-stage schedule for metric-oriented optimization.
        progress = cur_iter / max(1, total_iters)
        if progress < 0.2:
            return 0.6, 0.4
        if progress < 0.8:
            return 0.4, 0.6
        return 0.3, 0.7

    def should_run_eval(self, epoch: int) -> bool:
        policy = getattr(self.opt, "eval_policy", "legacy")
        if policy == "legacy":
            return epoch % 10 == 0 and epoch >= 700
        if policy != "staged":
            raise RuntimeError(f"Unsupported eval_policy={policy!r}")
        if epoch < self.opt.eval_stage1_start:
            return False
        start_epoch = self.opt.eval_stage1_start
        interval = self.opt.eval_stage1_interval
        if epoch >= self.opt.eval_stage3_start:
            start_epoch = self.opt.eval_stage3_start
            interval = self.opt.eval_stage3_interval
        elif epoch >= self.opt.eval_stage2_start:
            start_epoch = self.opt.eval_stage2_start
            interval = self.opt.eval_stage2_interval
        return epoch == start_epoch or (epoch - start_epoch) % interval == 0

    def forward(self, batch_data):
        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device, non_blocking=True)
        m_lens = m_lens.detach().long().to(self.device, non_blocking=True)

        m_lens = m_lens // 4
        conds = conds.to(self.device, non_blocking=True).float() if torch.is_tensor(conds) else conds

        loss_emb, ce_loss, _pred_ids, _acc = self.t2m_transformer(conds, m_lens, motion, self.vq_model)
        return loss_emb, ce_loss, _acc

    def update(self, batch_data, cur_iter, total_iters):
        loss_emb, ce_loss, acc = self.forward(batch_data)
        w_emb, w_ce = self.get_loss_weights(cur_iter, total_iters)
        total_loss = w_emb * loss_emb + w_ce * ce_loss

        self.opt_t2m_transformer.zero_grad(set_to_none=True)
        total_loss.backward()

        # ---- v4.1: gradient clipping ----
        grad_clip_norm = float(getattr(self.opt, 'grad_clip_norm', 0.0))
        if grad_clip_norm > 0:
            model = self.unwrap(self.t2m_transformer)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)

        self.opt_t2m_transformer.step()
        self.scheduler.step()

        # ---- v4.1: EMA update (only after warmup) ----
        if self.ema_params is not None and cur_iter >= self.opt.warm_up_iter:
            self._update_ema()

        # 可选：把每个 rank 的指标做全局平均，让 rank0 的日志更“真实”
        if self.ddp:
            loss_emb_v = ddp_reduce_mean(loss_emb.item(), self.device)
            ce_loss_v  = ddp_reduce_mean(ce_loss.item(), self.device)
            acc_v      = ddp_reduce_mean(float(acc), self.device)
            total_loss_v = ddp_reduce_mean(total_loss.item(), self.device)
            return loss_emb_v, ce_loss_v, acc_v, total_loss_v, w_emb, w_ce

        return loss_emb.item(), ce_loss.item(), float(acc), total_loss.item(), w_emb, w_ce

    def save(self, file_name, ep, total_it):
        if not self.master:
            return

        model = self.unwrap(self.t2m_transformer)
        t2m_trans_state_dict = model.state_dict()

        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]

        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

        # ---- v4.1: save EMA checkpoint alongside raw checkpoint ----
        if self.ema_params is not None:
            orig = self._swap_ema_weights()
            ema_state_dict = model.state_dict()
            ema_clip_weights = [e for e in ema_state_dict.keys() if e.startswith('clip_model.')]
            for e in ema_clip_weights:
                del ema_state_dict[e]
            ema_state = {
                't2m_transformer': ema_state_dict,
                'ep': ep,
                'total_it': total_it,
            }
            ema_file = file_name.replace('.tar', '_ema.tar')
            torch.save(ema_state, ema_file)
            self._restore_weights(orig)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)

        model = self.unwrap(self.t2m_transformer)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['t2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            if self.master:
                print('Resume wo optimizer')

        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        # 注意：DDP 模型外面已经 .to(device) 了也没事，这里再 to 一次也无害
        self.t2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        # optimizer/scheduler：用 unwrap 后的参数更稳妥（避免某些情况下 DDP wrapper 的参数迭代问题）
        model = self.unwrap(self.t2m_transformer)

        self.opt_t2m_transformer = optim.AdamW(
            model.parameters(),
            betas=(0.9, 0.99),
            lr=self.opt.lr,
            weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.opt_t2m_transformer,
            milestones=self.opt.milestones,
            gamma=self.opt.gamma
        )

        # ---- v4.1: initialize EMA ----
        self._init_ema()

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
            print(
                "Eval policy: "
                f"{self.opt.eval_policy} "
                f"(stage1={getattr(self.opt, 'eval_stage1_start', 'n/a')}/{getattr(self.opt, 'eval_stage1_interval', 'n/a')}, "
                f"stage2={getattr(self.opt, 'eval_stage2_start', 'n/a')}/{getattr(self.opt, 'eval_stage2_interval', 'n/a')}, "
                f"stage3={getattr(self.opt, 'eval_stage3_start', 'n/a')}/{getattr(self.opt, 'eval_stage3_interval', 'n/a')})"
            )

        logs = defaultdict(def_value, OrderedDict())

        # 跳过初始 eval（非常耗时，且 epoch 0 无意义），直接开始训练
        # 若需初始 eval，可设 skip_initial_eval=False
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching = 100, 100, 0, 0, 0, 100
        if self.master:
            print('Skipping initial eval (use epoch 0 baseline). Training starts...')

        if self.ddp:
            dist.barrier()

        while epoch < self.opt.max_epoch:
            # DDP: 每个 epoch 设置 sampler 的 epoch，保证每个 rank 的 shuffle 不一样
            if self.ddp and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            self.t2m_transformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss_emb, ce_loss, token_acc, total_loss, w_emb, w_ce = self.update(
                    batch_data=batch, cur_iter=it, total_iters=total_iters
                )

                # GATE0 fix F6: --debug_max_steps N stops after N optimizer steps without saving (makes the 20-iter smoke expressible)
                _debug_max_steps = int(getattr(self.opt, 'debug_max_steps', 0))
                if _debug_max_steps > 0 and it >= _debug_max_steps:
                    if self.master:
                        print(f'[debug_max_steps] Reached {it} optimizer steps '
                              f'(loss_emb={loss_emb:.4f}, ce={ce_loss:.4f}, acc={token_acc:.4f}) — stopping, saving nothing.')
                    return
                logs['loss_emb'] += loss_emb
                logs['ce_loss'] += ce_loss
                logs['total_loss'] += total_loss
                logs['token_acc'] += token_acc
                logs['w_emb'] += w_emb
                logs['w_ce'] += w_ce
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

                if self.master and it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        if self.logger is not None:
                            # GATE0 fix F5: TB writes must never kill training (shared-fs FileNotFoundError killed May segment B)
                            try:
                                self.logger.add_scalar('Train/%s' % tag, value / self.opt.log_every, it)
                            except (FileNotFoundError, OSError) as _tb_err:
                                print(f"  WARNING: TensorBoard write failed (non-fatal): {_tb_err}")
                        mean_loss[tag] = value / self.opt.log_every

                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1

            # Periodic epoch checkpoint
            save_every = getattr(self.opt, 'save_every_e', 0)
            if save_every > 0 and epoch % save_every == 0 and self.master:
                self.save(pjoin(self.opt.model_dir, f'net_ep{epoch:04d}.tar'), epoch, it)

            # epoch 结束：可选验证/评估
            if self.master:
                print('Validation time:')
                self.vq_model.eval()
                self.t2m_transformer.eval()

                if self.should_run_eval(epoch):
                    # ---- v4.1: swap in EMA weights for evaluation ----
                    ema_orig = self._swap_ema_weights()
                    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
                        self.opt.save_root, eval_val_loader, model, self.vq_model, self.logger, epoch,
                        best_fid=best_fid, best_div=best_div,
                        best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                        best_matching=best_matching, eval_wrapper=eval_wrapper,
                        plot_func=plot_eval, save_ckpt=True,
                        save_anim=(epoch % self.opt.eval_every_e == 0)
                    )
                    self._restore_weights(ema_orig)

            if self.ddp:
                dist.barrier()

from options.base_option import BaseOptions
import argparse

class TrainT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
        self.parser.add_argument('--max_epoch', type=int, default=3000, help='Maximum number of epoch for training')
        # self.parser.add_argument('--max_iters', type=int, default=150_000, help='Training iterations')

        '''LR scheduler'''
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate schedule factor')
        self.parser.add_argument('--milestones', default=[50_000], nargs="+", type=int, # 100000
                            help="learning rate schedule (iterations, MultiStepLR)")
        self.parser.add_argument('--warm_up_iter', default=2000, type=int, help='number of total iterations for warmup')
        self.parser.add_argument('--lr_scheduler', type=str, default='multistep',
                            choices=['multistep', 'linear', 'cosine'],
                            help='LR scheduler type: multistep (default, legacy), linear (LinearLR decay then hold), cosine (CosineAnnealingLR)')
        self.parser.add_argument('--lr_decay_iters', type=int, default=237500,
                            help='For linear/cosine: iters over which to decay from lr to lr_floor; linear holds at lr_floor afterwards')
        self.parser.add_argument('--lr_floor', type=float, default=2e-5,
                            help='For linear/cosine: final (minimum) LR value')

        '''Condition'''
        self.parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Drop ratio of condition, for classifier-free guidance')
        self.parser.add_argument("--seed", default=3407, type=int, help="Seed")

        self.parser.add_argument('--is_continue', action="store_true", help='Is this trial continuing previous state?')
        self.parser.add_argument('--reset_scheduler', action="store_true", help='Reset LR scheduler on resume (keep model+optimizer, discard scheduler state)')
        self.parser.add_argument('--gumbel_sample', action="store_true", help='Strategy for token sampling, True: Gumbel sampling, False: Categorical sampling')
        self.parser.add_argument('--share_weight', action="store_true", help='Whether to share weight for projection/embedding, for residual transformer.')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress, (iteration)')
        self.parser.add_argument('--save_every_e', type=int, default=0, help='Save checkpoint every N epochs (0=disabled)')
        self.parser.add_argument('--eval_every_e', type=int, default=10, help='Frequency of animating eval results, (epoch)')
        self.parser.add_argument('--eval_policy', type=str, default='legacy',
                                 help='Evaluation cadence policy. Supported: legacy, staged')
        self.parser.add_argument('--eval_stage1_start', type=int, default=200,
                                 help='For staged eval_policy: first epoch eligible for evaluation.')
        self.parser.add_argument('--eval_stage1_interval', type=int, default=50,
                                 help='For staged eval_policy: evaluation interval once eval_stage1_start is reached.')
        self.parser.add_argument('--eval_stage2_start', type=int, default=700,
                                 help='For staged eval_policy: switch to eval_stage2_interval at this epoch.')
        self.parser.add_argument('--eval_stage2_interval', type=int, default=30,
                                 help='For staged eval_policy: evaluation interval from eval_stage2_start onward.')
        self.parser.add_argument('--eval_stage3_start', type=int, default=1600,
                                 help='For staged eval_policy: switch to eval_stage3_interval at this epoch.')
        self.parser.add_argument('--eval_stage3_interval', type=int, default=10,
                                 help='For staged eval_policy: evaluation interval from eval_stage3_start onward.')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving checkpoint, (iteration)')
        self.parser.add_argument('--trans_name', type=str, default="t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns", help='Name of this trial')
        self.parser.add_argument('--transformer_variant', type=str, default='c_concat',
                                 help='Mask transformer implementation to use. Supported: '
                                      'c_concat, c_concat_v4, t_concat, t_concat_v2, t_concat_v3, t_concat_v4')
        self.parser.add_argument('--pretrained_base', type=str, default='',
                                 help='Load pretrained base weights with strict=False (for v2 init from v1)')
        self.parser.add_argument('--cross_attn_interval', type=int, default=-1,
                                 help='[v2] Insert text cross-attention every N self-attention layers')
        self.parser.add_argument('--cross_attn_heads', type=int, default=0,
                                 help='[v2] Heads for text cross-attention (0 = same as self-attention)')
        self.parser.add_argument('--text_adapter_layers', type=int, default=-1,
                                 help='[v4] Number of transformer layers in CLIP text adapter (0 = no adapter)')
        self.parser.add_argument('--gate_init', type=float, default=-1.0,
                                 help='[v4] Initial value for cross-attention gate parameter (0.0 = v2 default)')
        self.parser.add_argument('--control', default='trajectory', type=str, help="trajectory, random, cross")
        self.parser.add_argument('--trans_path', type=str, default='',
                                 help='Explicit path to pretrained base transformer checkpoint for ControlNet training.')
        self.parser.add_argument('--vq_checkpoint', type=str,
                                 default='/iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/output/vq/2026-03-02-14-06-17_vq_overlap_20260302/net_best_fid.pth',
                                 help='Absolute path to the paired VQ checkpoint used by the mask transformer.')
        self.parser.add_argument('--vq_partition_file', type=str,
                                 default='/iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis/skeleton_partition.json',
                                 help='Absolute path to the part-aware VQ partition JSON. Pass "" (empty) to use hardcoded 6-part partSeg (MMM-style tokenizer).')
        self.parser.add_argument('--nb_code', type=int, default=128,
                                 help='VQ codebook size per part. 128 = vq_overlap baseline; 8192 = mmm-style new tokenizer. Overrides VQ_CFG default.')
        self.parser.add_argument('--dataset_opt_path', type=str, default='./checkpoints/t2m/Comp_v6_KLD005/opt.txt',
                                 help='Dataset opt.txt used to build the evaluation stack.')
        self.parser.add_argument('--mean_npy', type=str,
                                 default='/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy',
                                 help='Normalization mean.npy used by train/val/eval loaders.')
        self.parser.add_argument('--std_npy', type=str,
                                 default='/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy',
                                 help='Normalization std.npy used by train/val/eval loaders.')

        self.parser.add_argument('--kv_rank', type=int, default=64,
                                 help='Low-rank dimension for v8 KV-Context projections (ablation: 32/64/128)')

        self.parser.add_argument('--xent', default=.1, type=float, help="")
        self.parser.add_argument('--ctrl_loss', default=.9, type=float, help="")
        self.parser.add_argument('--fixed_ce_weight', default=-1.0, type=float,
                                 help='Fixed CE loss weight (0-1). -1 = use progressive schedule.')

        # ---- v4.1 training improvements (backward-compatible: defaults = v4.0 behavior) ----
        self.parser.add_argument('--masked_only_ce', action='store_true',
                                 help='[v4.1] Only compute CE on masked tokens (ignore clean context)')
        self.parser.add_argument('--label_smoothing', type=float, default=0.0,
                                 help='[v4.1] Label smoothing for CE loss (0.0=v4, 0.03=v4.1)')
        self.parser.add_argument('--noise_std', type=float, default=0.1,
                                 help='[v4.1] Embedding noise std (0.1=v4, 0.03=v4.1)')
        self.parser.add_argument('--noise_masked_only', action='store_true',
                                 help='[v4.1] Only add noise to corrupted tokens (mask/random-replaced)')
        self.parser.add_argument('--ema_decay', type=float, default=0.0,
                                 help='[v4.1] EMA decay for weight averaging (0.0=off, 0.9995=v4.1)')
        self.parser.add_argument('--grad_clip_norm', type=float, default=0.0,
                                 help='[v4.1] Max gradient norm for clipping (0.0=off)')

        # ---- v4.5/v4.6 architecture ablations (backward-compatible: defaults = v4.0 behavior) ----
        self.parser.add_argument('--per_q_head', action='store_true',
                                 help='[v4.5] Use per-quantizer output heads instead of shared head')
        self.parser.add_argument('--mask_2d_hybrid', action='store_true',
                                 help='[v4.6] Enable hybrid 2D masking (mix frame-level + per-Q independent)')
        self.parser.add_argument('--mask_2d_ratio', type=float, default=0.3,
                                 help='[v4.6] Fraction of samples using per-Q independent mask (0.3=default)')
        self.parser.add_argument('--factorized_attn', type=str, default='none',
                                 choices=['none', 'same_q', 'semantic_group', 'semantic_group_v2'],
                                 help='[v4.8/v4.9] Factorized attention mode for odd layers. '
                                      'semantic_group_v2: 3-group banded (lower↔core↔upper) with bridge connectivity')
        self.parser.add_argument('--eval_split', type=str, default='val', choices=['val', 'test'],
                                 help='Split used for FID/Top3 evaluation during training (default: val)')
        # GATE0 fix F6: expressible smoke runs — stop after N optimizer steps, save nothing (0 = disabled)
        self.parser.add_argument('--debug_max_steps', type=int, default=0,
                                 help='DEBUG: stop training after N optimizer steps without saving any checkpoint (0=off)')

        self.is_train = True


class TrainLenEstOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument("--gpu_id", type=int, default=-1, help='GPU id')

        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

        self.parser.add_argument("--unit_length", type=int, default=4, help="Length of motion")
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Length of motion")

        self.parser.add_argument('--max_epoch', type=int, default=300, help='Training iterations')

        self.parser.add_argument('--lr', type=float, default=1e-4, help='Layers of GRU')

        self.parser.add_argument('--is_continue', action="store_true", help='Training iterations')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
        self.parser.add_argument('--save_every_e', type=int, default=5, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=3, help='Frequency of printing training progress')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.is_train = True
        # args = vars(self.opt)
        return self.opt

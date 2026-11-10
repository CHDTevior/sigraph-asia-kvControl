import os
import torch
import numpy as np
from argparse import Namespace
from pathlib import Path

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.mask_transformer.factory import build_mask_transformer, normalize_transformer_variant
from models.mask_transformer.transformer_trainer_ddp import MaskTransformerTrainer

from options.train_option import TrainT2MOptions

from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain

from data.t2m_dataset import Text2MotionDataset
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper

import models.vqvae as vqvae
from util.tqdm_wrapper import tqdm
import util.parser_util
import argparse

from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_tokenize


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def resolve_repo_path(path_value):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()


def build_vq_args(opt):
    cfg = dict(VQ_CFG)
    cfg["load_dir_vqvae"] = str(opt.vq_checkpoint)
    # Allow empty partition_file → None for hardcoded-partSeg tokenizers (e.g., mmm-style v5.1-mmmV1)
    _pf = str(opt.vq_partition_file) if opt.vq_partition_file is not None else ""
    cfg["partition_file"] = None if _pf in ("", "None", "none") else _pf
    # Override VQ_CFG default nb_code with CLI value (128 for baseline, 8192 for mmm)
    cfg["nb_code"] = int(getattr(opt, "nb_code", cfg["nb_code"]))
    return Namespace(**cfg)

def plot_t2m(data, save_dir, captions, m_lengths):
    return
    data = train_dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        # print(joint.shape)
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)

def preflight_training_inputs(opt):
    opt.transformer_variant = normalize_transformer_variant(opt.transformer_variant)
    opt.eval_policy = str(opt.eval_policy).strip().lower()
    opt.vq_checkpoint = resolve_repo_path(opt.vq_checkpoint)
    # partition_file may be empty/None for mmm-style tokenizers that use hardcoded 6-part partSeg
    _pf_str = str(opt.vq_partition_file) if opt.vq_partition_file is not None else ""
    if _pf_str in ("", "None", "none"):
        opt.vq_partition_file = None
    else:
        opt.vq_partition_file = resolve_repo_path(opt.vq_partition_file)
    opt.dataset_opt_path = resolve_repo_path(opt.dataset_opt_path)
    opt.mean_npy = resolve_repo_path(opt.mean_npy)
    opt.std_npy = resolve_repo_path(opt.std_npy)

    require(opt.vq_checkpoint.is_file(), f"VQ checkpoint not found: {opt.vq_checkpoint}")
    if opt.vq_partition_file is not None:
        require(opt.vq_partition_file.is_file(), f"VQ partition file not found: {opt.vq_partition_file}")
    require(opt.dataset_opt_path.is_file(), f"dataset_opt_path not found: {opt.dataset_opt_path}")
    require(opt.mean_npy.is_file(), f"mean_npy not found: {opt.mean_npy}")
    require(opt.std_npy.is_file(), f"std_npy not found: {opt.std_npy}")

    require(opt.batch_size > 0, f"batch_size must be > 0, got {opt.batch_size}")
    require(opt.max_epoch > 0, f"max_epoch must be > 0, got {opt.max_epoch}")
    require(opt.lr > 0, f"lr must be > 0, got {opt.lr}")
    require(opt.latent_dim > 0, f"latent_dim must be > 0, got {opt.latent_dim}")
    require(opt.n_layers > 0, f"n_layers must be > 0, got {opt.n_layers}")
    require(opt.n_heads > 0, f"n_heads must be > 0, got {opt.n_heads}")
    require(
        opt.eval_policy in {"legacy", "staged"},
        f"Unsupported eval_policy={opt.eval_policy!r}. Supported: legacy, staged",
    )
    require(opt.eval_stage1_start >= 0, f"eval_stage1_start must be >= 0, got {opt.eval_stage1_start}")
    require(opt.eval_stage2_start >= opt.eval_stage1_start,
            f"eval_stage2_start must be >= eval_stage1_start, got {opt.eval_stage2_start} < {opt.eval_stage1_start}")
    require(opt.eval_stage3_start >= opt.eval_stage2_start,
            f"eval_stage3_start must be >= eval_stage2_start, got {opt.eval_stage3_start} < {opt.eval_stage2_start}")
    require(opt.eval_stage1_interval > 0, f"eval_stage1_interval must be > 0, got {opt.eval_stage1_interval}")
    require(opt.eval_stage2_interval > 0, f"eval_stage2_interval must be > 0, got {opt.eval_stage2_interval}")
    require(opt.eval_stage3_interval > 0, f"eval_stage3_interval must be > 0, got {opt.eval_stage3_interval}")


def load_vq_model(opt):
    vq_args = build_vq_args(opt)
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
        raise RuntimeError(
            f"VQ checkpoint {opt.vq_checkpoint} does not contain 'net' or 'vq_model'."
        )

    missing_keys, unexpected_keys = vq_net.load_state_dict(state_dict, strict=False)
    require(not unexpected_keys, f"unexpected VQ checkpoint keys: {unexpected_keys}")
    require(not missing_keys, f"missing VQ checkpoint keys: {missing_keys}")
    print(f"Loading VQ Model {opt.vq_checkpoint}")
    return vq_net, vq_args

if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)
    preflight_training_inputs(opt)

    # ---------- DDP init ----------
    # Always trust torchrun-provided env vars for rank topology.
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    rank_env = int(os.environ.get("RANK", "0"))
    local_rank_env = int(os.environ.get("LOCAL_RANK", "0"))
    launched_by_torchrun = "LOCAL_RANK" in os.environ
    use_ddp = launched_by_torchrun and world_size_env > 1

    if use_ddp:
        dist.init_process_group(
            backend="nccl",
            rank=rank_env,
            world_size=world_size_env,
        )
        local_rank = local_rank_env
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = rank_env
        world_size = world_size_env
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() and opt.gpu_id != -1 else "cpu")

    opt.device = device
    opt.rank = rank
    opt.world_size = world_size
    opt.local_rank = local_rank
    opt.is_ddp = use_ddp

    # 只让 rank0 打印/写 log
    opt.is_master = (rank == 0)

    torch.autograd.set_detect_anomaly(False)  # disable for production runs

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    # opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/t2m/', opt.dataset_name, opt.name)

    if opt.is_master:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.eval_dir, exist_ok=True)
        os.makedirs(opt.log_dir, exist_ok=True)

    if opt.is_ddp:
        dist.barrier()  # 等 rank0 建完目录


    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = opt.dataset_opt_path

    elif opt.dataset_name == 'kit': #TODO
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_len = 55
        kinematic_chain = kit_kinematic_chain
        dataset_opt_path = opt.dataset_opt_path

    else:
        raise KeyError('Dataset Does Not Exist')

    opt.text_dir = pjoin(opt.data_root, 'texts')
    opt.max_token_len = max(1, int(np.ceil(opt.max_motion_length / float(opt.unit_length))))

    vq_model, vq_opt = load_vq_model(opt)
    vq_model = vq_model.to(opt.device)

    clip_version = 'ViT-B/32'

    opt.num_tokens = vq_opt.nb_code
    opt.num_quantizers = len(vq_model.vqvae.quantizers)

    t2m_transformer = build_mask_transformer(
        opt.transformer_variant,
        code_dim=vq_opt.code_dim,
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
    )

    # if opt.fix_token_emb:
    #     t2m_transformer.load_and_freeze_token_emb(vq_model.quantizer.codebooks[0])

    # ---- load pretrained base (for v2 init from v1 checkpoint) ----
    if getattr(opt, 'pretrained_base', ''):
        ckpt = torch.load(opt.pretrained_base, map_location='cpu')
        state_dict = ckpt.get('t2m_transformer', ckpt)
        missing, unexpected = t2m_transformer.load_state_dict(state_dict, strict=False)
        if opt.is_master:
            print(f"[pretrained_base] Loaded: {opt.pretrained_base}")
            print(f"  Missing: {len(missing)} keys (expected for new modules)")
            if unexpected:
                print(f"  Unexpected: {len(unexpected)} keys: {unexpected[:5]}")

    all_params = 0
    pc_transformer = sum(param.numel() for param in t2m_transformer.parameters_wo_clip())

    # print(t2m_transformer)
    # print("Total parameters of t2m_transformer net: {:.2f}M".format(pc_transformer / 1000_000))
    all_params += pc_transformer

    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    mean = np.load(opt.mean_npy)
    std = np.load(opt.std_npy)

    if opt.is_master:
        print(f"Transformer variant: {opt.transformer_variant}")
        print(f"VQ checkpoint: {opt.vq_checkpoint}")
        print(f"VQ partition file: {opt.vq_partition_file}")
        print(f"dataset_opt_path: {dataset_opt_path}")
        print(f"mean_npy: {opt.mean_npy}")
        print(f"std_npy: {opt.std_npy}")
        print(f"max_token_len: {opt.max_token_len}")
        print(
            "eval schedule: "
            f"policy={opt.eval_policy}, "
            f"stage1={opt.eval_stage1_start}/{opt.eval_stage1_interval}, "
            f"stage2={opt.eval_stage2_start}/{opt.eval_stage2_interval}, "
            f"stage3={opt.eval_stage3_start}/{opt.eval_stage3_interval}"
        )

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)

    # train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)

    if opt.is_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        val_sampler   = DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler   = None

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

    t2m_transformer = t2m_transformer.to(opt.device)

    if opt.is_ddp:
        # DDP 建议 find_unused_parameters=False（除非你确实有分支导致 unused）
        t2m_transformer = DDP(
            t2m_transformer,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    eval_split = getattr(opt, 'eval_split', 'val')
    eval_val_loader, _ = get_dataset_motion_loader(
        str(dataset_opt_path),
        32,
        eval_split,
        device=opt.device,
        mean_path=str(opt.mean_npy),
        std_path=str(opt.std_npy),
    )

    wrapper_opt = get_opt(str(dataset_opt_path), torch.device('cuda'))
    # print(f"wrapper opt: {wrapper_opt}")
    # import sys
    # sys.exit()
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    trainer = MaskTransformerTrainer(opt, t2m_transformer, vq_model)

    # GATE0 fix F6: announce smoke mode loudly so a debug run can never be mistaken for a real launch
    if int(getattr(opt, 'debug_max_steps', 0)) > 0 and opt.is_master:
        print(f"[debug_max_steps] SMOKE MODE: stopping after {opt.debug_max_steps} optimizer steps; NO checkpoints will be saved")

    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper=eval_wrapper, plot_eval=plot_t2m)


'''
python train_t2m_transformer.py --name 1124_My_t2m_transformer_NEW_VAE --gpu_id 0 --dataset_name t2m --batch_size 64

nohup python train_t2m_transformer.py --name My_t2m_transformer_1500_2 --gpu_id 1 --dataset_name t2m --batch_size 64 > log_t2m_myself22_1500_2.txt 2>&1 &

nohup python train_t2m_transformer.py --name My_t2m_transformer_1500_new_vae --gpu_id 0 --dataset_name t2m --batch_size 64 > log_t2m_myself22_1500_new_vae.txt 2>&1 &

# nohup 挂起
nohup python train_t2m_transformer.py --name My_t2m_transformer --gpu_id 0 --dataset_name t2m --batch_size 64 > log_t2m_myself22.txt 2>&1 &


python train_t2m_transformer.py --name test --gpu_id 0 --dataset_name t2m --batch_size 64

python train_t2m_transformer.py --name 1225_My_t2m_transformer_NEW_VAE_Tconcat_64_real_20 --gpu_id 1 --dataset_name t2m --batch_size 64

nohup 挂起

nohup python train_t2m_transformer.py --name 11301_My_t2m_transformer_NEW_VAE_64_real --gpu_id 1 --dataset_name t2m --batch_size 64 > log_11301_My_t2m_transformer_NEW_VAE_64_real.txt 2>&1 &

nohup python train_t2m_transformer.py --name 11302_My_t2m_transformer_NEW_VAE_64_real --gpu_id 1 --dataset_name t2m --batch_size 64 > log_11302_My_t2m_transformer_NEW_VAE_64_real.txt 2>&1 &

nohup python train_t2m_transformer.py --name 1212_My_t2m_transformer_NEW_VAE_64_real --gpu_id 1 --dataset_name t2m --batch_size 64 > 1212_My_t2m_transformer_NEW_VAE_64_real.txt 2>&1 &

nohup python train_t2m_transformer.py --name 1217_My_t2m_transformer_NEW_VAE_64_real --gpu_id 1 --dataset_name t2m --batch_size 64 > 1217_My_t2m_transformer_NEW_VAE_64_real.txt 2>&1 &

nohup python train_t2m_transformer.py --name 1219_My_t2m_transformer_NEW_VAE_64_real_20 --gpu_id 1 --dataset_name t2m --batch_size 64 > 1219_My_t2m_transformer_NEW_VAE_64_real_20.txt 2>&1 &

nohup python train_t2m_transformer.py --name 0101_My_t2m_transformer_NEW_VAE_64_real_20 --gpu_id 1 --dataset_name t2m --batch_size 64 > 0101_My_t2m_transformer_NEW_VAE_64_real_20.txt 2>&1 &



nohup python train_t2m_transformer.py --name 1226_My_t2m_transformer_NEW_VAE_Tconcat_64_real_20 --gpu_id 1 --dataset_name t2m --batch_size 64 > 1226_My_t2m_transformer_NEW_VAE_Tconcat_64_real_20.txt 2>&1 &



# torchrun --nproc_per_node=2 train_t2m_transformer.py \
  --name test --dataset_name t2m --batch_size 128


nohup torchrun --nproc_per_node=2 train_t2m_transformer.py \
  --name 1229_My_t2m_transformer_NEW_VAE_Tconcat_64_real_20_ddp --dataset_name t2m --batch_size 64 > 1229_My_t2m_transformer_NEW_VAE_Tconcat_64_real_20_ddp.txt 2>&1 &
  

nohup torchrun --nproc_per_node=2 train_t2m_transformer.py \
  --name 1231_My_t2m_transformer_NEW_VAE_Tconcat_32_real_20_ddp --dataset_name t2m --batch_size 32 > 1231_My_t2m_transformer_NEW_VAE_Tconcat_32_real_20_ddp.txt 2>&1 &


nohup torchrun --nproc_per_node=2 train_t2m_transformer.py \
  --name 0101_My_t2m_transformer_NEW_VAE_Cconcat_64_real_20_ddp --dataset_name t2m --batch_size 64 > 0101_My_t2m_transformer_NEW_VAE_Cconcat_64_real_20_ddp.txt 2>&1 &
 
nohup torchrun --nproc_per_node=2 train_t2m_transformer.py \
  --name 0102_My_t2m_transformer_NEW_VAE_Cconcat_32_real_20_ddp --dataset_name t2m --batch_size 32 > 0102_My_t2m_transformer_NEW_VAE_Cconcat_32_real_20_ddp.txt 2>&1 &
 

nohup torchrun --nproc_per_node=2 --master_port=29511 train_t2m_transformer.py \
  --name 0103_My_t2m_transformer_NEW_VAE_Cconcat_16_real_20_ddp \
  --dataset_name t2m --batch_size 16 \
  > 0103_My_t2m_transformer_NEW_VAE_Cconcat_16_real_20_ddp.txt 2>&1 &


nohup torchrun --nproc_per_node=2 --master_port=29511 train_t2m_transformer.py \
  --name 0203_My_t2m_transformer_NEW_VAE_Cconcat_64_real_20_ddp \
  --dataset_name t2m --batch_size 64 \
  > 0203_My_t2m_transformer_NEW_VAE_Cconcat_64_real_20_ddp.txt 2>&1 &


torchrun --nproc_per_node=1 --master_port=29511 train_t2m_transformer.py \
--name My_t2m_data_driven_VAE_ddp \
--dataset_name t2m \
--batch_size 64

nohup torchrun --nproc_per_node=2 --master_port=29511 train_t2m_transformer.py \
--name My_t2m_data_driven_VAE_ddp_C \
--dataset_name t2m \
--batch_size 64 \
> My_t2m_data_driven_VAE_ddp_C.txt 2>&1 &



CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_t2m_transformer.py \
  --dataset_name t2m \
  --checkpoints_dir ./checkpoints \
  --name t_concat_main_ddp_part_aware_vae \
  --batch_size 32 \
  --max_epoch 3000 \
  --lr 2e-4 \
  --warm_up_iter 2000 \
  --milestones 50000 \
  --gamma 0.1 \
  --log_every 50 \
  --save_latest 500 \
  --eval_every_e 10

nohup torchrun --nproc_per_node=2 train_t2m_transformer.py \
  --dataset_name t2m \
  --checkpoints_dir ./checkpoints \
  --name t_concat_main_ddp_part_aware_vae_h100 \
  --batch_size 32 \
  --max_epoch 3000 \
  --lr 2e-4 \
  --warm_up_iter 2000 \
  --milestones 50000 \
  --gamma 0.1 \
  --log_every 50 \
  --save_latest 500 \
  --eval_every_e 10 \
  > t_concat_main_ddp_part_aware_vae_h100.txt 2>&1 &

# 改一下port，避免和上一个冲突
nohup torchrun --nproc_per_node=2 --master_port=29411 train_t2m_transformer.py \
  --dataset_name t2m \
  --checkpoints_dir ./checkpoints \
  --name t_concat_main_ddp_part_aware_vae_h100_0222 \
  --batch_size 128 \
  --max_epoch 3000 \
  --lr 2e-4 \
  --warm_up_iter 2000 \
  --milestones 50000 \
  --gamma 0.1 \
  --log_every 50 \
  --save_latest 500 \
  --eval_every_e 10 \
  > t_concat_main_ddp_part_aware_vae_h100_0222.txt 2>&1 &

cd /scratch/ts1v23/workspace/MaskControl
nohup torchrun --nproc_per_node=2 --master_port=29688 train_t2m_transformer.py \
  --dataset_name t2m \
  --checkpoints_dir ./checkpoints \
  --name c_concat_main_ddp_part_aware_vae_overlap_20260302 \
  --batch_size 64 \
  --max_epoch 3000 \
  --lr 2e-4 \
  --warm_up_iter 2000 \
  --milestones 50000 \
  --gamma 0.1 \
  --log_every 50 \
  --save_latest 500 \
  --eval_every_e 10 \
  > c_concat_main_ddp_part_aware_vae_overlap_20260302.txt 2>&1 &


'''

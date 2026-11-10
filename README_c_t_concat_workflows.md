# C/T Concat Training And Evaluation

This file is the single launch reference for the `c_concat` and `t_concat` mask-transformer variants.

## Fixed Shared Inputs

All commands below assume the same paired VQ and normalization stats:

- `VQ_CHECKPOINT=/iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/output/vq/2026-03-02-14-06-17_vq_overlap_20260302/net_best_fid.pth`
- `VQ_PARTITION_FILE=/iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis/skeleton_partition.json`
- `DATASET_OPT=/iridisfs/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/Comp_v6_KLD005/opt.txt`
- `MEAN_NPY=/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy`
- `STD_NPY=/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy`

The transformer variant is now controlled only by `--transformer_variant`. Do not copy any `T_latest.py` content over `transformer.py`.

## Environment Setup

Run this after SSH-ing into an allocated compute node:

```bash
cd /iridisfs/scratch/ts1v23/workspace/MaskControl
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tlcontrol
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

## 1-Epoch Smoke Test

### C-concat

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_t2m_transformer.py \
  --name smoke_c_concat_1ep \
  --gpu_id 0 \
  --dataset_name t2m \
  --batch_size 16 \
  --max_epoch 1 \
  --log_every 10 \
  --save_latest 1000000 \
  --eval_every_e 999999 \
  --transformer_variant c_concat \
  --vq_checkpoint /iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/output/vq/2026-03-02-14-06-17_vq_overlap_20260302/net_best_fid.pth \
  --vq_partition_file /iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis/skeleton_partition.json \
  --dataset_opt_path /iridisfs/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/Comp_v6_KLD005/opt.txt \
  --mean_npy /scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy \
  --std_npy /scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy
```

### T-concat

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_t2m_transformer.py \
  --name smoke_t_concat_1ep \
  --gpu_id 0 \
  --dataset_name t2m \
  --batch_size 16 \
  --max_epoch 1 \
  --log_every 10 \
  --save_latest 1000000 \
  --eval_every_e 999999 \
  --transformer_variant t_concat \
  --vq_checkpoint /iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/output/vq/2026-03-02-14-06-17_vq_overlap_20260302/net_best_fid.pth \
  --vq_partition_file /iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis/skeleton_partition.json \
  --dataset_opt_path /iridisfs/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/Comp_v6_KLD005/opt.txt \
  --mean_npy /scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy \
  --std_npy /scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy
```

## Recommended Full Training Launch

Pick a free two-GPU allocation and launch from that compute node.

### C-concat

```bash
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29681 train_t2m_transformer.py \
  --name c_concat_main_ddp_part_aware_vq \
  --dataset_name t2m \
  --batch_size 64 \
  --max_epoch 3000 \
  --lr 2e-4 \
  --warm_up_iter 2000 \
  --milestones 50000 \
  --gamma 0.1 \
  --log_every 50 \
  --save_latest 500 \
  --eval_every_e 10 \
  --transformer_variant c_concat \
  --vq_checkpoint /iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/output/vq/2026-03-02-14-06-17_vq_overlap_20260302/net_best_fid.pth \
  --vq_partition_file /iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis/skeleton_partition.json \
  --dataset_opt_path /iridisfs/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/Comp_v6_KLD005/opt.txt \
  --mean_npy /scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy \
  --std_npy /scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy \
  > logs/c_concat_main_ddp_part_aware_vq.log 2>&1 < /dev/null &
```

### T-concat

```bash
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29682 train_t2m_transformer.py \
  --name t_concat_main_ddp_part_aware_vq \
  --dataset_name t2m \
  --batch_size 64 \
  --max_epoch 3000 \
  --lr 2e-4 \
  --warm_up_iter 2000 \
  --milestones 50000 \
  --gamma 0.1 \
  --log_every 50 \
  --save_latest 500 \
  --eval_every_e 10 \
  --transformer_variant t_concat \
  --vq_checkpoint /iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/output/vq/2026-03-02-14-06-17_vq_overlap_20260302/net_best_fid.pth \
  --vq_partition_file /iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis/skeleton_partition.json \
  --dataset_opt_path /iridisfs/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/Comp_v6_KLD005/opt.txt \
  --mean_npy /scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy \
  --std_npy /scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy \
  > logs/t_concat_main_ddp_part_aware_vq.log 2>&1 < /dev/null &
```

## CFG Evaluation

`eval_t2m_trans_cfg.py` reads `transformer_variant` from the training run's `opt.txt`, so the same evaluation entrypoint works for both variants.

### Dry Run

```bash
python -u eval_t2m_trans_cfg.py \
  --gpu_id 0 \
  --batch_size 32 \
  --repeat_times 1 \
  --cond_scale 3.25 \
  --main_checkpoint /iridisfs/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/<RUN_DIR>/model/net_best_fid.tar \
  --which_epoch best_fid \
  --vq_checkpoint /iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/output/vq/2026-03-02-14-06-17_vq_overlap_20260302/net_best_fid.pth \
  --vq_partition_file /iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis/skeleton_partition.json \
  --dataset_opt_path /iridisfs/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/Comp_v6_KLD005/opt.txt \
  --out_dir .codex-research/artifacts/cfg_scan/<RUN_TAG> \
  --dry_run
```

### Best-FID Checkpoint Scan

```bash
python -u eval_t2m_trans_cfg.py \
  --gpu_id 0 \
  --batch_size 32 \
  --repeat_times 200 \
  --cond_scale 3.25 \
  --time_steps 18 \
  --temperature 1.0 \
  --main_checkpoint /iridisfs/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/<RUN_DIR>/model/net_best_fid.tar \
  --which_epoch best_fid \
  --vq_checkpoint /iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/output/vq/2026-03-02-14-06-17_vq_overlap_20260302/net_best_fid.pth \
  --vq_partition_file /iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis/skeleton_partition.json \
  --dataset_opt_path /iridisfs/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/Comp_v6_KLD005/opt.txt \
  --out_dir .codex-research/artifacts/cfg_scan/<RUN_TAG>
```

### Latest Checkpoint Debug Eval

Use this when the run is still young and does not have `net_best_fid.tar` yet:

```bash
python -u eval_t2m_trans_cfg.py \
  --gpu_id 0 \
  --batch_size 32 \
  --repeat_times 5 \
  --cond_scale 3.25 \
  --main_checkpoint /iridisfs/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/<RUN_DIR>/model/latest.tar \
  --which_epoch latest \
  --vq_checkpoint /iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/output/vq/2026-03-02-14-06-17_vq_overlap_20260302/net_best_fid.pth \
  --vq_partition_file /iridisfs/scratch/ts1v23/workspace/part-aware-vqvae/partition_analysis/skeleton_partition.json \
  --dataset_opt_path /iridisfs/scratch/ts1v23/workspace/MaskControl/checkpoints/t2m/Comp_v6_KLD005/opt.txt \
  --out_dir .codex-research/artifacts/cfg_scan/<RUN_TAG>
```

## Practical Notes

- `C/T` isolation is now config-driven. Switching variants only requires changing `--transformer_variant`.
- Training and evaluation both use the explicit `VQ checkpoint`, `partition file`, `mean.npy`, and `std.npy` passed on the command line.
- If you reuse one compute node for both variants, use different `master_port` values for the two DDP jobs.
- If you run a background job, record the `PID`, log path, and checkpoint directory immediately.

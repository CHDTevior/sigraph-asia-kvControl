# reference_impl — 活的参考实现 (self-contained)

本目录让 harness 在 GitHub 上自包含: 不依赖 cluster 本地的 `references/MaskControl/` clone。

| 文件 | 是什么 | 对应 harness 文档 |
|---|---|---|
| `control_transformer_kv.py` | **KV adapter 完整实现** (subclass MaskControl 的 ControlTransformer; per-layer K/V injection + Q-residual + near-identity init + ctrl_train override + base fallback) | docs/01, docs/02 Step 2 |
| `train_ctrlnet_ddp.py` | 4-GPU DDP trainer fork (DistributedSampler / rank-0-only IO / adapter-aware optimizer) | docs/02 Step 6 |
| `eval_maskcontrol_kv.py` | 5-rep M1/M2/M3 offline eval (negative-each_iter convention 已接通 / vq_model.* filter / frozen-snapshot 模式) | docs/03 |
| `smoke_kv_on_maskcontrol.py` | Tier1/Tier2 invariant smoke test | docs/02 Step 4 |
| `maskcontrol_substrate.patch` | 对 upstream [exitudio/MaskControl](https://github.com/exitudio/MaskControl) 5 个文件的最小 diff (isinstance dispatch 修复 / best_kps tracker 修复 / DDP mkdir 修复 / CLI flags / adapter-aware optimizer) | docs/04 P2/P4/P5/P6 |

## 复现步骤 (在新机器上)

```bash
git clone https://github.com/exitudio/MaskControl.git
cd MaskControl
git apply <this-dir>/maskcontrol_substrate.patch
cp <this-dir>/control_transformer_kv.py models/mask_transformer/
cp <this-dir>/train_ctrlnet_ddp.py .
# 下载 MaskControl README 里的 t2m ckpts (rvq / trans / evaluator) 后:
python train_ctrlnet.py --name kv_run --control_adapter kv --kv_rank 64 \
  --ctrl_attn_bias_init -5.0 --batch_size 64 --lr 2e-4 --max_epoch 6000 \
  --xent 0.1 --ctrl_loss 0.9 --control trajectory \
  --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2 \
  --trans_name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns \
  --latent_dim 384 --n_layers 8 --n_heads 6 --ff_size 1024 --dropout 0.2
# (4×GPU: torchrun --nproc_per_node=4 train_ctrlnet_ddp.py <同参数> --lr 8e-4)
```

训练前先跑 smoke (`smoke_kv_on_maskcontrol.py`), 评估用 `eval_maskcontrol_kv.py` — 协议细节见 `../docs/03_evaluation_protocol.md`。

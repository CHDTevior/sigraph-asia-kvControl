# KV-Control + CodeGeo: trajectory-controlled text-to-motion

System code for trajectory-controlled text-to-motion generation with four parts:

1. **PartVQ.** Per-body-part motion codebooks. The code lives in a separate repository: <https://github.com/CHDTevior/part-aware-vqvae>.
2. **T-Concat backbone.** A masked transformer that lays the per-part codes out along the sequence axis.
3. **KV-Control.** A thin adapter that feeds the commanded trajectory into every attention layer of the frozen backbone as key/value memory.
4. **CodeGeo.** A codebook-geometry correction that moves the generated motion onto the commanded anchors in at most 8 analytic damped Gauss–Newton steps. Each step uses a metric taken from the codebook covariance.

The code builds on [MaskControl](https://github.com/exitudio/MaskControl) and [MoMask](https://github.com/EricGuo5513/momask-codes). Data preparation, evaluator checkpoints and the GloVe files follow those repositories.

## Where things are

| Component | Files |
|---|---|
| T-Concat backbone | `models/mask_transformer/transformer_t_concat_v4.py`, `models/mask_transformer/factory.py` |
| KV-Control adapter (main variant) | `models/mask_transformer/control_transformer_t_concat_v8_kv_v4_noqres.py` |
| CodeGeo solver | `models/mask_transformer/coda_solver_vec_fullcap.py` (batched, full anchor capacity), `models/mask_transformer/coda_solver.py` |
| Backbone training | `train_t2m_transformer.py` (`--transformer_variant t_concat_v4`) |
| KV-Control training | `train_ctrlnet_t_concat_ddp.py` (set `USE_V8_V4_CONTROL_NOQRES=1`) |
| Backbone evaluation | `eval_t2m_trans_cfg.py` |
| Control evaluation, CodeGeo included | `scripts/eval_v4_ctrlnet_ttt.py` |
| Per-clip inference (demos) | `scripts/D1_stage2_kv_inference.py` |
| Porting KV-Control to other backbones | `kvcontrol_adapt_harness/` |

Launch commands for training and evaluating the backbone, with every architecture flag spelled out, are in `README_c_t_concat_workflows.md`. The architecture flags must be passed explicitly. If one is omitted, the checkpoint loads into a mismatched model.

## Setup

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
bash prepare/download_glove.sh
bash prepare/download_evaluator.sh
```

Put HumanML3D under `dataset/HumanML3D` as in MoMask. Checkpoints are not included in this repository.

**Paths.** Several default paths point to our cluster (`/iridisfs/scratch/ts1v23/...`): the PartVQ checkpoint, the partition file and the normalization statistics. Where a script offers flags, override them there: `--vq_checkpoint`, `--vq_partition_file` and `--trans_path` in `scripts/eval_v4_ctrlnet_ttt.py`, and the VQ options in `options/train_option.py`. Otherwise edit the defaults in these files:
- `options/train_option.py`
- `eval_t2m_trans_cfg.py`
- `train_t2m_transformer.py`
- `scripts/eval_v4_ctrlnet_ttt.py`
- `scripts/D1_stage2_kv_inference.py`
- `util/parser_util.py`

Run the scripts from the repository root, because some of them use relative paths such as `./checkpoints` and `./glove`.

## Evaluating CodeGeo (example: pelvis control)

```bash
python scripts/eval_v4_ctrlnet_ttt.py --ckpt <kv_control_ckpt>.tar --split test \
  --n_layers 20 --n_heads 8 --latent_dim 384 --ff_size 1536 --kv_rank 64 \
  --cross_attn_interval 2 --factorized_attn none \
  --time_steps 10 --each_iter 0 --last_iter 1 --cond_scale 3.25 \
  --gn --gn_vec --gn_metric codebook_cov --gn_ridge 10.0 --gn_tau 1e-5 \
  --gn_steps 8 --gn_lam0 1e-2 --gn_anchor_cap 196 \
  --control pelvis --repeat_times 5 --seed 3407
```

The eval script picks the KV-Control variant from the checkpoint's keys. These are the main CodeGeo flags:
- `--gn_metric codebook_cov` selects the codebook-covariance metric. Leave it out for the Euclidean step.
- `--gn_ridge` is the ridge ρ.
- `--gn_steps` is the step budget.
- `--gn_anchor_cap` is the anchor capacity.

For iterative refinement instead of CodeGeo, drop the `--gn*` flags and set `--each_iter` and `--last_iter`.

# KVControl Adapt Harness — 移植框架入口

## 这是什么

KVControl Adapt Harness 是一套 **docs + templates + checklists**，用于把 KV-Control 机制（frozen base transformer + per-layer K/V residual injection + soft-decode differentiable control）移植到**新的 substrate / 新项目**上。它诞生于三次实战：(1) **v4 native** — 机制的原生家园（PartVQ + T-Concat backbone，paper 主线，M3 KPS 0.40cm）；(2) **MaskControl port（成功）** — 3 天（含 debug）完成 mechanism hot-swap，KPS 63cm → 1.29cm，验证机制可移植；(3) **MoMask port（部分失败，教训来源）** — 4 层叠加 bug（重写 decode loop / static err channel / RT-cascade OOD / EMA codebook 污染），最终只 salvage 到 FID 0.581 / KPS 30.6cm。本 harness 的每一条 checklist 项都对应一个真实踩过的坑。读者：三个月后的我们自己 + collaborators。

## 结果表 — 机制的 substrate-portability 证据

KV-on-MaskControl（v16→v18，ep 6000，4×A100 DDP，single-joint pelvis trajectory，HumanML3D test split，5-rep mean±95CI）：

| Protocol | FID | Top3 | KPS | Diversity | skate |
|---|---|---|---|---|---|
| M1 (Stage-1 TTT only, **uniform-35**/step) | 0.152±0.011 | 0.797 | 11.75±0.22 cm | 9.700 | 0.046 |
| M2 (100 uniform + last_iter 600) | 0.098±0.006 | 0.791 | **1.10±0.01 cm** | 9.665 | 0.046 |
| M3 (**uniform-35** + last_iter 600) | **0.0875±0.008** | 0.789 | 1.29±0.02 cm | 9.578 | 0.047 |
| no-control baseline（同 substrate，见下方注 ①） | 0.1455 | ~0.80* | 63.05 cm | 9.94 | ~0.05* |
| paper v4 KV-Control（M3，own substrate，参照） | 0.065 | 0.799 | 0.40 cm | ~9.5 | ~0.05 |

> **协议标注勘误（2026-07 codex audit）**：M1/M3 此前被标为 "35 dynamic"（step s 得 (s+1)×35 次迭代，共 ~1925 次）——**该标注是错的**。substrate 的 `control_transformer.py` L527-531 只在 `each_iter<0` 时走 dynamic TTT；eval 脚本传的是 +35 且从未取负（`--ttt_dynamic` flag 只影响打印标签），所以上表 M1/M3 数字实际是 **uniform 35 iters/step（ts=10 共 350 次）**。已在 `scripts/eval_maskcontrol_kv.py` 与 `templates/eval_template.py` 里接通 negative-each_iter 约定供未来真跑 dynamic 用。
>
> **注 ①**：no-control 行不是正式 M0 5-rep JSON——FID 0.1455 / Div 9.94 / KPS 63.05 来自 dispatch-bug 期间的 no-control-equivalent evals（`eval_5r_M1_bestkps/ep875/ep1080`，KV 子类被 `type() is` 路由到 base 无控制路径，等价于 no-control 测量）；带 * 的 Top3/skate 是 informal estimate。正式引用前应补跑一次 formal M0 5-rep。

解读：
- **KPS 63cm → 1.10-1.29cm**（M2/M3），同时 FID 相对 no-control baseline 不劣化甚至改善（0.1455 → 0.0875）——机制在别人的 substrate 上原样成立。
- M1 (11.75cm) vs M2/M3 (1.1-1.3cm) 的 **10× gap 来自 Stage-2**（last_iter=600 直接优化 continuous embedding，绕过 codebook，是唯一没有 quantization floor 的杠杆）。
- 反面教材：KV-on-MoMask（failed port）最好只到 ep600 base-only-decode FID 0.581 / KPS 30.6cm——差距全部来自移植违规操作，不是机制本身。

## 目录结构

```
kvcontrol_adapt_harness/
├── README.md                        ← 本文件（入口）
├── docs/
│   ├── 01_kv_control_method.md      — KV-Control 机制解剖：K/V injection、init 不变量
│   │                                  (kv_down zero + kv_up standard = identity-with-live-grads;
│   │                                  ctrl_attn_bias=-5.0; Q-residual gate=0)、soft-decode 梯度链
│   ├── 02_porting_playbook.md       — 8+1 步移植 playbook（下方 Quick start 的展开版），
│   │                                  含 substrate 侦察问卷、A/B-clean 训练 config、DDP 排雷
│   ├── 03_evaluation_protocol.md    — M0/M1/M2/M3 eval protocol 钉死规范 + 运行手册 + 协议陷阱
│   ├── 04_pitfall_catalog.md        — 15 条 pitfall 全目录（症状 → 根因 → 修复），全部来自实战
│   └── 05_data_adaptation.md        — 换数据集/骨架适配：dim_pose 公式、可微 FK、指标可移植性
├── templates/
│   ├── kv_adapter_template.py       — subclass + delattr + KV modules 的起手模板
│   ├── smoke_test_template.py       — identity-at-init / gradient-flow / frozen-base 冒烟测试
│   └── eval_template.py             — M1/M2/M3 协议 5-rep eval 模板（PROTOCOLS 表内置）
└── checklists/
    ├── porting_checklist.md         — 端到端 checkbox 清单（每项标注防护的 pitfall #）
    └── launch_checklist.md          — 训练启动前 10 项 checklist（GPU 核验 → 监控计划）
```

## Quick start — 9 步移植到新 substrate

详细展开见 `docs/02_porting_playbook.md`，每步一行：

1. **读 substrate**：找到 generate()/decode loop、per-layer forward、tokenizer 下采样率（unit_length）、codebook 布局——先读后写（R8）。
2. **验差分 FK 路径**：确认 decoder 输出 → joints 可微（新数据集若非 `recover_from_ric`，先移植可微等价物并 unit-test 梯度）。
3. **设计 adapter**：control encoder 输入 C = n_ctrl_joints × 2 × 3、encoder stride 链匹配 tokenizer 下采样率、kv_rank=64。
4. **subclass + delattr**：继承 substrate 的 transformer 类，删掉它自己的 ControlNet 分支，加 KV modules（严守 init 不变量：kv_down=0、bias=-5.0、q_gates=0 但 q_down 标准初始化）。
5. **只 override trans_forward**：generate()/decode loop 与 Stage-1/Stage-2 TTT 一律原样复用，绝不重写（MoMask 失败的 #1 根因）。
6. **闭环 err channel**：ctrlNet_cond = cat([(gt−pred)*mask, gt*mask])，训练每 batch、推理每 iteration 都从当前预测重算。
7. **smoke test**：init 时输出 bit-近似 base、adapter 梯度全活、eval 确实路由到 KV subclass（isinstance，不是 type() is）。
8. **codex review（铁律）→ 训练**：hyperparams 与 substrate 自己的 ControlNet 完全一致（A/B-clean），多卡按 Goyal linear scaling。
9. **eval + 对比**：M1/M2/M3 protocol 参数逐项钉死并 assert，5-rep mean±95CI，与 no-control baseline + substrate 原生 ControlNet 同表对比。

## Repo pointers — 活的实现（真实路径）

| 内容 | 路径 |
|---|---|
| **MaskControl port KV adapter**（成功移植范本，339 行） | `<repo-root>/references/MaskControl/models/mask_transformer/control_transformer_kv.py` — `_layer_with_kv_injection` L41-122（Q-residual 防污染写法 L69-82）、`KVControlTransformer.__init__` L128-216（delattr L174-176、zero-init L190-191、bias L195-198）、`trans_forward` L281-319、`_base_trans_forward` L321-339 |
| **被移植的原 ControlNet**（对照物） | `<repo-root>/references/MaskControl/models/mask_transformer/control_transformer.py`（`freeze_block`/`unfreeze_block` L35-43） |
| **v4 native 实现**（paper 主线） | `<repo-root>/models/mask_transformer/control_transformer_t_concat_v8_kv_v4.py` — KV modules L289-304、ctrl_attn_bias L308-310、Q-residual L313-322、`forward_with_kv_context_and_crossattn` L64-192、`_build_ctrl_cond`（闭环 err channel）L376-387 |
| **MaskControl port 训练脚本**（cross-alloc DDP） | `<repo-root>/references/MaskControl/train_ctrlnet_ddp.py` |
| **Eval 脚本**（M1 默认 + M2/M3 via `--last_iter`） | `<repo-root>/scripts/eval_maskcontrol_kv.py`（CLI L78-121、protocol 自动检测 L212、`pred_num_batch=16` 语义注意 L227-230；2026-07 修正后行号） |
| v4 eval 脚本（被 fork 的母本） | `<repo-root>/scripts/eval_v4_ctrlnet_ttt.py` |
| 已验证的 5r M3 结果 JSON | `<repo-root>/output/v16_kv_maskcontrol_4xa100_20260701_013823/eval_5r_M3_ep6000/eval_5r.json` |
| ep6000 frozen ckpt snapshot（~244MB） | `<repo-root>/output/v18_ep6000_frozen_snapshot.tar` |
| Git remotes | upstream `origin`: https://github.com/exitudio/MaskControl.git；fork `chd`: git@github.com:CHDTevior/sigraph-asia-kvControl.git |

> 提醒：`utils/eval_t2m.py:1082` 曾用 `type(x) is ControlTransformer` 把 KV subclass 静默路由到 no-control 路径（pitfall #2；现已修复，isinstance 判断在 L1086，:1082 是历史 bug 位置、现为修复注释行）。任何新 port 的 eval 链路先确认此类判断已是 `isinstance()`。

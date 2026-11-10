# KV-Control Porting Checklist

逐项打勾，顺序执行。每项末尾的 `(P#)` 指向它防护的 pitfall 编号（全目录见 `../docs/04_pitfall_catalog.md`；15 条 pitfall 全部来自 v4 / MaskControl / MoMask 三次实战）。标 `(—)` 的是流程纪律项，不对应单一 pitfall。

## Phase 1 — Read substrate（先读后写）

- [ ] 通读 substrate 的 generate()/decode loop（pad/score init、sampling 顺序、re-mask 策略），确认它将被**原样复用**而非重写 (P12)
- [ ] 定位 per-layer forward 的唯一注入点（如 `trans_forward`），确认能在不碰 decode loop 的前提下 override (P12)
- [ ] 查清 tokenizer 下采样率（unit_length，通常 4）→ T_ctrl = T_frames/rate，决定 encoder_control stride 链 (—)
- [ ] 查清 codebook 布局：几个 quantizer、soft-decode 应该 mix 哪个/哪些 codebook（MoMask RVQ=6，MaskControl 用 base，v4 是 Q=6 part codebooks） (P11)
- [ ] 列出所有 frozen 下游模块（如 ResidualTransformer），标记为潜在 OOD 放大器，规划 base-only-decode 对照 (P11)
- [ ] 确认数据 feature dim（263 仅 HumanML3D；KIT=251）、joints_num、mean/std 文件、foot-contact 通道 (—)
- [ ] 确认 feature→joints 的 FK 路径**可微**（`recover_from_ric` 是 HumanML3D-tree-specific）；新骨架先移植可微等价物并 unit-test 其梯度 (—)
- [ ] 确认目标数据集有自己的 FID/R-prec evaluator ckpt，否则这两项指标无意义（KPS/traj-fail 是纯几何、永远可移植） (P15)

## Phase 2 — Design

- [ ] control encoder 输入通道 C = n_ctrl_joints × 2 × 3（err + abs 每关节），pelvis-only 轨迹 C=6 (—)
- [ ] ctrlNet_cond 设计为 **闭环**：cat([(gt−pred)*mask, gt*mask])，(residual, absolute) 顺序，err 通道每次从当前预测重算 (P13)
- [ ] 每层 low-rank K/V：kv_down (D→r=64, **zero-init**) + kv_up_k/kv_up_v (r→D, standard init)——zero-down + standard-up = identity at init 且梯度活；双零 = dead branch (—)
- [ ] ctrl_attn_bias per-layer scalar，init −5.0（每 ctrl token 相对单个 base token 权重 e^-5≈0.0067；S=50/T_ctrl=49 下总 ctrl mass ≈0.65%（<1%），near-identity at init） (—)
- [ ] Q-residual（若启用）：shared q_down/q_up standard init + per-upper-half-layer scalar q_gates **zero init**，只作用 layers ≥ num_layers//2 (P3)

## Phase 3 — Implement

- [ ] subclass substrate 的 transformer 类，delattr 其原生 ControlNet 分支（参照 `control_transformer_kv.py` L174-176） (—)
- [ ] q_delta 只加进 Q 流：qkv 从**未修改的 x** 投影后 chunk(3)，再对 q 加 `F.linear(q_delta, in_proj_weight[:D])`——绝不在 in_proj 前把 q_delta 加到 x (P3)
- [ ] ctrl_train() 内以及每次 `model.train()` 之后强制 `vq_model.eval()`；requires_grad=False **保护不了** EMA quantizer buffers (P1)
- [ ] ckpt save 剥离 `vq_model.*` keys；resume load 同样剥离；eval 侧加 byte-equal preflight assert 对照 pristine VQ (P1)
- [ ] eval load 剥离 DDP `module.` 前缀 (P14)
- [ ] 保留 base fallback 分支（ctrlNet_cond=None → bit-exact base forward，参照 `_base_trans_forward` L321-339），不 delegate 给可能被改的 parent 路径 (P2)
- [ ] DDP：save-folder 路径上所有 `os.makedirs(exist_ok=True)` / `copytree(dirs_exist_ok=True)`，ckpt 写入 rank-0-only (P5)
- [ ] `--is_continue` 时跳过 name 的日期前缀，否则 resume 解析到全新空目录 (P6)
- [ ] 所有 TensorBoard `writer.add_scalar`（per-iter 和 end-of-epoch）包 try/except，NFS 抖动不许杀训练 (P10)
- [ ] best_kps tracker：每次 save 后更新 best 值，且 in-loop eval 的 kps_mean 不许 unpack 进 `_`（两个 bug 都会冻结 best-ckpt 选择） (P4)
- [ ] 训练侧 eval 读 ckpt 前先 cp + atomic mv 到 frozen snapshot，绝不直接读训练正在写的 latest.tar (P7)

## Phase 4 — Smoke test

- [ ] identity-at-init：加载 frozen base，KV adapter 全新初始化，输出与纯 base forward bit-近似 (—)
- [ ] gradient-flow：init 后一步反传，**只要求 kv_down/ctrl_attn_bias/q_gates 梯度非零**（kv_down 全零 = 双零死支，q_down 若被零初始化会双零死支）。⚠ kv_up_k/kv_up_v/encoder_control/q_down/q_up 在 init 时梯度**恰好为 0** 是正确实现的数学必然（kv_down=0 ⇒ h=0；q_gates=0），不许判 FAIL——改为 1-2 个 optimizer step 后 assert 这些组梯度变非零 (P3)
- [ ] eval-routing：断言 eval 路径用 `isinstance()` 而非 `type() is`，确认真的调用了 KV subclass 的 forward (P2)
- [ ] soft-decode 梯度链 smoke：L_traj → FK → decoder → soft emb → logits → adapter 全链 requires_grad 且梯度非零 (P13)
- [ ] 短跑 2-3 ep：loss 下降、无 NaN、ckpt 可 save/load/resume roundtrip、vq_model.* 不在 ckpt 里 (P1)

## Phase 5 — Codex review（铁律，不可跳过）

- [ ] 全部新增/修改代码过 codex（gpt-5.5 xhigh）review，NEEDS-FIX 项修完复审到 PASS 再启动长训练 (—)

## Phase 6 — Train

- [ ] hyperparams 与 substrate **自己的** ControlNet 训练完全一致（A/B-clean），机制 delta 才可归因（MaskControl port: bs 64/rank×4=256, lr 8e-4 Goyal 4×, warm_up 2000, milestones [12500], xent 0.1/ctrl 0.9） (—)
- [ ] 多卡扩 batch 时 lr 按 Goyal linear scaling 同倍放大 (—)
- [ ] sparse-keyframe 若 NaN：先查 err channel 是否真闭环 + loss balance，别急着断言 sparse 不可训（MoMask 上 NaN、MaskControl 上 random density 稳训） (P9, P13)
- [ ] 逐 ckpt 监控 eval：**连续多次 eval 数字 bit-identical = 你评的不是你以为的模型**，立即停下查 routing (P2)

## Phase 7 — Eval

- [ ] M0/M1/M2/M3 每个 protocol 的 time_steps/cond_scale/each_iter/ttt_dynamic/last_iter 逐项钉死并在脚本内 assert，禁止默认值静默漂移 (P15)
- [ ] `pred_num_batch` 语义 = "每次 generate 累积 N 个 loader batch"，用 16；设 99999 会静默跳过全部生成然后 torch.cat 空表崩 (P8)
- [ ] 5-rep formal，报 mean ± 95CI（std×1.96/√N），test split (—)
- [ ] 有 residual/refinement 级联的 substrate：跑 base-only-decode vs full-cascade 对照，确认下游模块没把 KV-modified tokens 当 OOD 加噪 (P11)

## Phase 8 — Compare & report

- [ ] 同表汇报：KV port vs no-control baseline vs substrate 原生 ControlNet vs v4 参照（0.065/0.40cm），FID 与 KPS 一起看，不许单指标报捷 (—)
- [ ] Stage-2（last_iter）单独消融：它是唯一无 quantization floor 的杠杆（M1 11.75cm → M2/M3 1.1-1.3cm 的 10× 来自它） (—)
- [ ] 渲染可视化 demo 人眼检查（多帧 gif / GT-vs-pred 并排），metric 与视觉冲突时以视觉为准 (—)
- [ ] 结果 + 教训回写本 harness（新 pitfall 追加编号入 `docs/04_pitfall_catalog.md`） (—)

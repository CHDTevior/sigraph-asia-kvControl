# 03 — 控制评估协议 (Evaluation Protocol)

> KVControl Adapt Harness 文档 #3。本文定义把 KV-Control 移植到任何新 substrate 后，**必须以完全相同的方式跑的评估协议**：M0/M1/M2/M3 四档、固定采样参数、指标定义与计算位置、运行手册、协议陷阱、baseline 对照要求。
>
> 主参考实现（MaskControl port 的 eval 脚本，逐条行号皆指向它）:
> `<repo-root>/scripts/eval_maskcontrol_kv.py`
> （fork 自 v4 substrate 的 `<repo-root>/scripts/eval_v4_ctrlnet_ttt.py`）
>
> 底层 eval 函数: `evaluation_mask_transformer_test_plus_res`，位于
> `<repo-root>/references/MaskControl/utils/eval_t2m.py`（MaskControl substrate 侧）
> 及 `<repo-root>/utils/eval_t2m.py`（v4 substrate 侧）。

---

## 1. M0 / M1 / M2 / M3 协议定义表

四档协议只在 **test-time optimization (TTT) 的两个 lever** 上不同：
- **Stage-1**：每个 MaskGIT timestep 内，把 logits 当自由变量、用 Adam (lr = `each_lr` = 6e-2) 在 soft-decode 轨迹 MSE 上优化（`each_iter` / `ttt_dynamic` 控制迭代量）；
- **Stage-2**：生成完成后，绕过 codebook、直接对 **连续 embedding** 做 `last_iter` 步 Adam 优化（唯一没有量化下限 quantization floor 的 lever）。

| 协议 | `each_iter` | TTT 日程 | `last_iter` | `last_lr` | 用途 | 单 rep 相对耗时 |
|---|---|---|---|---|---|---|
| **M0** | 0 | — | 0 | — | 纯 KV adapter 前向，无任何 TTT。诊断 adapter 本体学到了什么；也是 no-control / base-only 对照的运行档 | 1×（最快，纯生成） |
| **M1** | 35 | **uniform**（每 step 固定 35 次） | 0 | 6e-2 (未使用) | Stage-1-only。论文 `KPSoursNoRefine` cell；衡量"离散 codebook 内"控制上限（有量化 floor） | ~中等；TTT 总迭代 = 35×10 = **350 次** @ ts=10 |
| **M2** | 100 | **uniform**（每 step 固定 100 次） | 600 | 6e-2 | Stage-1 重 + Stage-2。KPS 最优档（1.10 cm on MaskControl port） | 最慢（Stage-1 均匀 100×10 = 1000 次 + Stage-2 600 次） |
| **M3** | 35 | **uniform**（每 step 固定 35 次） | 600 | 6e-2 | **论文 headline 档**：Stage-1 轻 + Stage-2 600。FID/KPS 最佳折中（0.0875 / 1.29 cm） | ~M1 + Stage-2 600 次 |

> ⚠ **协议标注勘误（2026-07 codex audit）**：M1/M3 此前被标为 "35 dynamic（第 s 个 timestep 得 (s+1)×35 次，总 ~1925 次）"——**该标注对已发布的 JSON 是错的**。Ground truth：substrate 的 `references/MaskControl/models/mask_transformer/control_transformer.py` L527-531 只在 `each_iter<0` 时走 dynamic TTT；`scripts/eval_maskcontrol_kv.py` 传的是 `opt.each_iter=+35` 且从未在 `--ttt_dynamic` 时取负（旧 flag 只影响 protocol 打印标签，历史 L201、修正后在 L212）。所以 ep6000 的 M1/M3 数字全部产自 **uniform 35 iters/step（350 total @ ts=10）**。脚本现已接通 `opt.each_iter = -args.each_iter if args.ttt_dynamic else args.each_iter`（negative-each_iter 约定），未来真跑 dynamic 会得到不同（未发布）的数字，须另行标注。

> 注：单 rep 绝对耗时未做逐档正式计时，上表给相对量级；首次在新 substrate 上跑时把每档实测 wall-clock 记回本表。参考量级：MaskControl port 上 5-rep × 全 test split 的 M3 eval（4568 samples, bs=32）在单张 A100 上约数小时级。

**为什么 M2/M3 能把 KPS 从 11.75 cm 干到 1.1–1.3 cm（10×）**：Stage-1 无论迭代多少次都受 codebook 量化下限约束（soft-decode 最终仍要落到离散 token 的凸组合附近）；Stage-2 直接优化连续 embedding、完全绕过 codebook，是唯一无 floor 的 lever。所以 M1 (11.75 cm) → M2/M3 (1.10/1.29 cm) 的差距几乎全部来自 `last_iter=600`。

**脚本内的协议自动检测**（`scripts/eval_maskcontrol_kv.py`，修正后）：
```python
protocol = "M1" if (each_iter==35 and time_steps==10 and not ttt_dynamic and last_iter==0) else "CUSTOM"
```
（M1 = uniform-35；`--ttt_dynamic` 现在真正接通 negative-each_iter 约定并被排除在 M1 之外。）即只有 M1 的精确取值写在 argparse 默认里；M2/M3 通过覆盖 `--each_iter/--last_iter` 产生，输出 JSON 中会标 `protocol="CUSTOM"` —— **人工在结果文件名/目录名里标注 M2/M3**（如已验证的 `output/v16_kv_maskcontrol_4xa100_20260701_013823/eval_5r_M3_ep6000/eval_5r.json`）。

---

## 2. 固定参数（所有协议档共享，禁止漂移）

以下参数在 M0–M3 全档 **pin 死**，来自 `scripts/eval_maskcontrol_kv.py` L83-L100 的 argparse 默认值（2026-07 修正后行号）：

| 参数 | 值 | 行号 | 说明 |
|---|---|---|---|
| `--time_steps` | 10 | L84 | MaskGIT 迭代步数 |
| `--cond_scale` | 3.25 | L85 | CFG scale |
| `--temperature` | 1.0 | L86 | 采样温度 |
| `--topkr` | 0.9 | L87 | top-k ratio |
| `--each_lr` | 6e-2 | L90 | Stage-1 Adam lr |
| `--last_lr` | 6e-2 | L99 | Stage-2 Adam lr |
| `--seed` | 3407 | L100 | `fixseed(args.seed)` 在 L125 调用 |
| `--repeat_times` | 5 | L83 | 正式 (formal) 数字必须 5 reps |
| split | test | L131 | `get_dataset_motion_loader(..., 32, "test", ...)`；全 HumanML3D test split，不用 val（用户铁律） |
| `pred_num_batch` | 16 | L227-230 | 见 §5 陷阱 #8 |
| `density` | -1 | L232 附近 | 每 sample 随机 density，见 §5 |

**95% CI 公式**（L246-254 的 summary 计算）：

```
mean ± 1.96 · std / sqrt(N)      N = repeat_times = 5
```

对 fid / diversity / top1 / top2 / top3 / matching_score / skate_ratio / kps_cm 各自计算。正式报告一律写 `mean ± 95CI`；单 rep (N=1) 数字只能用于趋势监控，**不得**进论文/对照表（见 §5 角色分工）。

**注意 `time_steps=10` 是 MaskControl substrate 的评估约定**；v4 substrate 的历史约定是 `time_steps=25`（项目记忆 `feedback_timesteps25`）。移植到新 substrate 时用 **该 substrate 原生论文的 time_steps**，并在结果 JSON 里显式记录 —— 两个 substrate 的数字本来就不跨栈比（见 §6）。

---

## 3. 指标定义与计算位置

所有指标由 `evaluation_mask_transformer_test_plus_res` 一次性算出，`scripts/eval_maskcontrol_kv.py` L219-244 的每-rep 循环解包为：

```python
best_fid, best_div, Rprecision, best_matching, best_skate_ratio, \
    best_mm, traj_err, _avoid, kps_mean = eval_t2m.evaluation_mask_transformer_test_plus_res(...)
```
（L221；调用签名在 L222-232，`cal_mm=False`, `res_model=None`, `rt2m_transformer=None`。）

| 指标 | 定义 | 依赖 evaluator? | 备注 |
|---|---|---|---|
| **FID** | 生成动作 vs GT 动作在 **evaluator 特征空间**（`Comp_v6_KLD005` motion encoder）的 Fréchet 距离 | 是 | evaluator 从 `--eval_wrapper_opt`（L106-107，默认 `<MC_REPO>/checkpoints/t2m/Comp_v6_KLD005/opt.txt`）加载，L131-133 构建 `EvaluatorModelWrapper` |
| **R-precision top1/2/3** | 32 候选文本中，配对文本的 motion-text 特征相似度排进前 1/2/3 的比例 | 是 | 解包自 `Rprecision` 三元组 |
| **MatchScore** | 配对 text-motion 特征欧氏距离均值（越小越好） | 是 | `best_matching` |
| **Diversity** | 生成集合内随机 pair 的特征距离均值 | 是 | `best_div`；目标是接近 GT diversity（~9.5），不是越大越好 |
| **KPS** (keyframe position score) | keyframe 上控制关节的预测 vs 目标 xyz 平均误差。eval 函数返回 **米**，脚本在 L243 处 `kps_cm = kps_mean * 100` 转 **厘米** | 否（纯几何） | 控制精度 headline 指标；跨数据集可移植 |
| **traj_fail (20cm/50cm)** | keyframe 误差超过 20cm / 50cm 阈值的比例 | 否（纯几何） | 解包在 `traj_err` 元组内；论文报 fail-rate 曲线用 |
| **skate_ratio** | 足部接触帧上滑步（foot skating）比例 | 否（纯几何 + foot-contact 通道） | `best_skate_ratio`；物理合理性 sanity 指标 |

**移植警告（数据适配文档 #05 的核心结论在此复述）**：FID / R-precision / MatchScore / Diversity 全部绑定 **数据集专属 evaluator ckpt**（HumanML3D 的 `Comp_v6_KLD005` / `text_mot_match`）。换数据集必须换/重训 evaluator，否则这四个数字**无意义**。KPS / traj_fail / skate_ratio 是 evaluator-free 的纯几何量，永远可移植 —— 新 substrate 起步阶段先用它们做 gate。

**dispatch 位置（关联陷阱 #2）**：eval 函数内部按模型类型选 generate 路径，历史 bug 位置在 MaskControl substrate 的 `utils/eval_t2m.py:1082`（现已修复：isinstance 判断在 L1086，:1082 现为修复注释行）—— 详见 §5。

---

## 4. 运行手册 (Runbook)

### 4.1 三条完整 CLI（M1 / M2 / M3）

CLI 全量参数定义在 `scripts/eval_maskcontrol_kv.py` L78-121。必填项只有 `--ckpt`（L79）和 `--output_json`（L121）。架构参数（L110-118：`--latent_dim 384 --n_layers 8 --n_heads 6 --ff_size 1024 --dropout 0.2 --control trajectory --kv_rank 64 --ctrl_attn_bias_init -5.0`）**必须与训练时一致** —— 与 v4 的教训相同（`feedback_v4_arch_params_mandatory`：架构参数不匹配 → `load_state_dict(strict=False)` 静默丢权重）。

```bash
cd <repo-root>
CKPT=<repo-root>/output/v18_ep6000_frozen_snapshot.tar  # 冻结快照, 见 4.2

# —— M1: Stage-1 TTT only (全部走默认值; protocol 自动标 "M1") ——
python scripts/eval_maskcontrol_kv.py \
    --ckpt "$CKPT" \
    --output_json output/v16_kv_maskcontrol_4xa100_20260701_013823/eval_5r_M1_ep6000/eval_5r.json

# —— M2: 100 uniform + Stage-2 600 ——
python scripts/eval_maskcontrol_kv.py \
    --ckpt "$CKPT" \
    --each_iter 100 \
    --last_iter 600 \
    --output_json output/v16_kv_maskcontrol_4xa100_20260701_013823/eval_5r_M2_ep6000/eval_5r.json

# —— M3: 35 uniform + Stage-2 600 (论文 headline) ——
python scripts/eval_maskcontrol_kv.py \
    --ckpt "$CKPT" \
    --each_iter 35 \
    --last_iter 600 \
    --output_json output/v16_kv_maskcontrol_4xa100_20260701_013823/eval_5r_M3_ep6000/eval_5r.json
```

要点：
- `--ttt_dynamic` 是 `BooleanOptionalAction`，**修正后默认 False（uniform，与已发布 M1/M2/M3 数字一致）**，且现在真正接通 negative-each_iter 约定（传了它 = step s 得 (s+1)×each_iter 次迭代，产生的是与已发布数字**不可比**的新协议，须另行标注）。M1/M2/M3 正式复现一律**不传**该 flag。
- M0（无 TTT 诊断档）= `--each_iter 0 --last_iter 0`。
- Q-residual 消融加 `--no_q_residual`（L188：`use_q_residual=(not args.no_q_residual)`）——必须与训练该 ckpt 时的开关一致。
- ckpt 加载（L193-208）：state key 取 `ct2m_transformer` else `trans`（L194）；DDP `module.` 前缀自动 strip（L198-200，陷阱 #14）。⚠ **`vq_model.*` 泄漏 keys 的正确处理是 load 前显式过滤，不能指望 `strict=False`**：模型有同名 `vq_model` 子模块，名字匹配得上，`strict=False` 会**照常加载**这些 keys、静默覆盖 pristine RVQ（strict=False 只忽略匹配不上的 keys；v18_ep6000 snapshot 实测带 72 个 `vq_model.*` keys——本 port 训练侧没有 strip）。eval 必须做 `raw_sd = {k:v for k,v in raw_sd.items() if not k.startswith("vq_model.")}` + 与官方 RVQ ckpt 的 byte-equal preflight。旧 ControlNet keys（`seqTransEncoder_control.*` 等已 delattr 的）匹配不上、确实被 strict=False 丢弃。其他 unexpected keys 按 `feedback_unexpected_keys_critical` 处理（立即停下排查）。

### 4.2 冻结快照模式（陷阱 #7 的标准操作）

**绝不**直接 eval 一个正在被训练进程写入的 `latest.tar` —— 读写竞态会抛 `PytorchStreamReader failed reading file`，更糟的是可能读到半写状态不报错。标准做法：

```bash
RUN=<repo-root>/output/v16_kv_maskcontrol_4xa100_20260701_013823
SNAP=<repo-root>/output/v18_ep6000_frozen_snapshot.tar

# cp 到临时名 + 原子 mv, 保证快照文件永远是完整字节
cp "$RUN/model/latest.tar" "${SNAP}.tmp" && mv "${SNAP}.tmp" "$SNAP"

# 之后所有 eval 只指向 $SNAP, 永不指向 live ckpt
```

已验证的实例：`<repo-root>/output/v18_ep6000_frozen_snapshot.tar`（~244 MB）就是 v16→v18 ep6000 的冻结快照，§1 表里的 5-rep 数字全部 eval 自它。

### 4.3 srun --overlap 到 eval 卡

Eval 是重 GPU 任务（5 reps × 全 test split × TTT），不许在跳板机/登录节点跑。用 `--overlap` 进入 **自己项目** 的既有 Slurm alloc 的空闲卡（先 `srun --jobid=<j> --overlap nvidia-smi` 确认目标卡空闲、不与训练 rank 或**别的项目**冲突 —— 跨项目铁律：不抢别项目正在用的卡）：

```bash
srun --jobid=<ALLOC_JOBID> --overlap --gres=gpu:1 --cpus-per-task=8 --pty bash -lc '
  cd <repo-root> && \
  conda activate tlcontrol && \
  CUDA_VISIBLE_DEVICES=<空闲卡序号> python scripts/eval_maskcontrol_kv.py \
      --ckpt output/v18_ep6000_frozen_snapshot.tar \
      --each_iter 35 --last_iter 600 \
      --output_json output/.../eval_5r_M3_ep6000/eval_5r.json \
      2>&1 | tee output/.../eval_5r_M3_ep6000/eval.log'
```

`--gres` / `--cpus-per-task` 必须显式给（cross-alloc 经验第 3 条：不显式给可能拿不到卡或 CPU 被限死）。长 eval 用 `setsid nohup` 落到 compute node 上跑（durable 模式），不要依赖登录 shell 存活。

### 4.4 环境与路径前提

- `sys.path.insert(0, MC_REPO)`（L33）让 MaskControl 自己的 `utils/models/motion_loaders` 优先解析 —— eval 必须从主 repo 根目录启动，且不要在 `PYTHONPATH` 里塞会遮蔽它的路径。
- RVQ-VAE 经 `load_rvqvae`（L44-74）加载：解析 `opt.txt`、强制 `dim_pose=263` / `joints_num=22`、读 `model/net_best_fid.tar`、state key `vq_model` else `net`（L69）、全参数 `requires_grad=False`（L72-73）。**新数据集这里的 263/22 硬编码必须改**（见 docs/05 数据适配）。
- base transformer 快照从 `<trans_name>/model/latest.tar` 读（L174）；substrate 默认名 L102-105（`rvq_nq6_dc512_nc512_noshare_qdp0.2` / `t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns`）。

---

## 5. 协议陷阱（本项目全部真实踩过）

### 5.1 `pred_num_batch` 语义（pitfall #8）

`evaluation_mask_transformer_test_plus_res` 的 `pred_num_batch` 意思是 **"每次 generate 调用累积 N 个 loader batch"**（N=16 × bs=32 = 512 samples/次），**不是**"总共处理 N 个 batch"。曾把它设成 99999 想"跑全量"，结果一个 generate 调用都没触发、静默跳过全部生成，最后在 `motion_annotation_list` 处 `torch.cat` 空列表崩溃。**永远用 16**（脚本已固化在 L227-230，带注释）。症状识别：eval 秒退 + cat 空列表 traceback = pred_num_batch 设错。

### 5.2 `type() is` vs `isinstance` dispatch（pitfall #2）

MaskControl 的 `utils/eval_t2m.py:1082` 原文曾是 `type(x) is ControlTransformer` —— 我们的 `KVControlTransformer` 是它的**子类**，`type() is` 判 False，eval 被静默路由到 **base 无控制路径**。三次连续 eval 输出到小数点后 4 位完全一致（FID 0.1455 / KPS 63.05）才暴露。修复：改 `isinstance()`（现已修复于 L1086；:1082 是历史 bug 位置、现为修复注释行）。

**通用检测启发式（写进任何新 port 的验收 gate）**：不同 ckpt / 不同 TTT 配置下 eval 输出 bit-identical ⇒ 你以为在评的模型根本没被调用。新 substrate 接好后第一件事：跑一个 M0 vs M1 的 sanity 对比，两者 KPS 必须显著不同（M1 应大幅低于 M0），否则 dispatch 一定有问题。

### 5.3 `density` 参数含义与 paper 可比性

脚本传 `density=-1`（L232）= **每个 sample 随机抽 keyframe density**。这与 MaskControl 原论文按固定 density 分档报告（如 dense / 25% / 5 keyframes / 1 keyframe）**不是同一协议** —— random-density 的 KPS 是各 density 的混合均值，直接与人家论文里某个固定-density cell 并排会被审稿人打。规则：
- 内部 A/B（我们的 KV vs 我们复跑的 ControlNet，同 density 协议）→ random density 可用且公平；
- 要引用/对照 substrate 原论文的表格数字 → 必须复现它的固定-density 协议逐档跑；
- 结果 JSON / 表格里永远标注 density 协议。这就是项目里 "§4 全表只内部比、cross-stack 数字抛弃"（`feedback_paper_section4_controlnet_self_only`）决策的评估侧根源之一。

### 5.4 in-loop eval vs offline 5-rep 的角色分工

- **in-loop eval**（训练器每若干 epoch 跑一次，N=1、常缩小样本量、无 CI）：只用于 (a) 趋势监控 / 早停判断，(b) `best_kps.tar` / `best_fid.tar` 的 ckpt 选择。它的绝对数值不进任何对照表。附带警告：MaskControl trainer 的 best_kps tracker 有两个 bug（pitfall #4 —— save 后 `best_kps_mean` 不更新导致每次 eval 都覆盖 `best_kps.tar`；in-loop 解包把 `kps_mean` 丢进 `_` 导致比较用的是循环前的陈值），**不修这两处，best-ckpt 选择毫无意义**。
- **offline 5-rep formal eval**（本文档协议）：唯一产生可报告数字的通道。固定 seed 3407、全 test split、5 reps、mean±95CI。M1/M2/M3 各跑一遍，与 baseline 同批跑。

### 5.5 参数 pin 与漂移断言（pitfall #15）

`time_steps / cond_scale / each_iter / ttt_dynamic / last_iter / last_lr` 六个量任何一个静默漂移，数字就不可比。要求：(a) 每次 eval 的 output JSON 里回写全部六个实参 + protocol 标签 + ckpt epoch（脚本已做：`ckpt_epoch = ckpt.get("ep", -1)`，L210）；(b) 汇总对照表前，用脚本 assert 各 cell 的六元组与其协议档定义一致，不靠人眼。

---

## 6. Baseline 对照要求

每一次正式 eval 交付，必须同批产出以下三行，缺一不可：

| 行 | 是什么 | 怎么跑 |
|---|---|---|
| **no-control baseline** | 同一 substrate、同一 base ckpt、**不加任何控制**的生成质量 + 无控制时的 KPS（≈ 关节自由漂移的自然误差） | KV ckpt 走 M0 且 `ctrl_net` 关闭 → 触发 `_base_trans_forward` fallback（adapter 文件 `references/MaskControl/models/mask_transformer/control_transformer_kv.py` L284-285: `ctrl_net is None / not ctrl_net / ctrlNet_cond is None` 三条件任一即走 L321-339 的 bit-exact base 路径）。MaskControl port 目前只有 dispatch-bug 期间的 no-control-equivalent 实测（`eval_5r_M1_bestkps/ep875/ep1080`）：FID 0.1455 / Div 9.94 / KPS 63.05 cm；**尚无正式 M0 5-rep JSON**，正式引用前应补跑 |
| **substrate 原生控制机制 baseline** | 该 substrate 自带的控制方案（MaskControl = 它自己的 ControlNet；MoMask port 时 = 无原生控制，此行标 N/A 并说明） | 用 **我们自己复跑** 的原生机制 ckpt，同协议同 density 同 evaluator 评估；不引用对方论文数字（§5.3） |
| **KV-Control（被评方法）** | 我们的 adapter，M1/M2/M3 三档 | §4.1 三条 CLI |

**公平性 checklist**（全部打勾才允许把数字写进文档/论文）：

- [ ] 训练超参 A/B-clean：KV 训练与原生机制训练的 bs / lr / warm_up / milestones / xent:ctrl_loss / cond_drop / epoch 数 **逐项相同**（本 port：bs 64×4=256, lr 8e-4, warm_up 2000, milestones [12500], 0.1:0.9, cdp 0.1, 6000 ep），使差异可归因到机制本身
- [ ] 同一 base transformer 快照、同一 RVQ ckpt（byte-equal preflight，防 pitfall #1 的 EMA codebook 污染）
- [ ] 同一 evaluator ckpt（`Comp_v6_KLD005`）、同一 test split、同 bs=32 loader
- [ ] 同 seed 3407、同 5 reps、同 §2 固定参数六元组
- [ ] 同 density 协议（random-density 对 random-density；固定档对固定档）
- [ ] 所有行来自 **冻结快照** ckpt（§4.2），不是 live 文件
- [ ] M0-vs-M1 dispatch sanity 已通过（§5.2）
- [ ] 数字之外做了可视化检查（跨项目铁律：CV 任务可视化 demo 准确度 > metric —— 至少渲染 GT-vs-pred keyframe 叠加的动画，确认 KPS 数值与视觉一致）

**已验证参考数字**（MaskControl port, v16→v18 ep6000, 5-rep mean±95CI, single-joint pelvis trajectory；原始 JSON: `<repo-root>/output/v16_kv_maskcontrol_4xa100_20260701_013823/eval_5r_M3_ep6000/eval_5r.json`）：

| 行 / 协议 | FID | Top3 | KPS (cm) | Diversity | skate |
|---|---|---|---|---|---|
| no-control baseline ① | 0.1455 | ~0.80* | 63.05 | 9.94 | ~0.05* |
| KV M1 (uniform-35) | 0.152±0.011 | 0.797 | 11.75±0.22 | 9.700 | 0.046 |
| KV M2 (uniform-100 + 600) | 0.098±0.006 | 0.791 | 1.10±0.01 | 9.665 | 0.046 |
| KV M3 (uniform-35 + 600) | 0.0875±0.008 | 0.789 | 1.29±0.02 | 9.578 | 0.047 |
| (参考) v4 substrate KV-Control M3 | 0.065 | 0.799 | 0.40 | ~9.5 | ~0.05 |

最后一行仅作"机制在原生家园的上限"参考，**不与 MaskControl port 行跨栈比较**（不同 base、不同 tokenizer、不同 time_steps 约定）。

> **注 ①**：no-control 行不可追溯到 `output/v16_kv_maskcontrol_4xa100_20260701_013823/` 下的任何正式 JSON——数字来自 dispatch-bug 期间的 no-control-equivalent evals（`eval_5r_M1_bestkps` / `ep875` / `ep1080`，KV 子类被 `type() is` 路由进 base 无控制路径，等价于 no-control 测量：FID 0.1455 / Div 9.94 / KPS 63.05）。带 * 的 Top3/skate 为 informal estimate。正式表格引用前应跑一次 formal M0 no-control 5-rep 并回填 JSON 路径。

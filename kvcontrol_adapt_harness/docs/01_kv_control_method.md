# 01 — KV-Control 方法精讲（以人的视角过一遍）

> KVControl Adapt Harness · 文档 1/N
> 目标读者：三个月后的我们自己 + 新加入的 collaborator。读完这篇你应该能在**不看论文**的情况下，对着一个陌生 substrate 说清楚"KV-Control 到底往哪里插了什么、为什么这么插、初始化为什么必须是这样"。
>
> 代码依据（本文所有行号都指向这两份实现 + eval 脚本）：
> - **MaskControl port 版**（结构最干净，推荐先读）：`<repo-root>/references/MaskControl/models/mask_transformer/control_transformer_kv.py`（339 行）
> - **v4 native 版**（paper headline）：`<repo-root>/models/mask_transformer/control_transformer_t_concat_v8_kv_v4.py`
> - **原版 ControlNet（被替换的对照物）**：`<repo-root>/references/MaskControl/models/mask_transformer/control_transformer.py`（1161 行）
> - **Eval 协议**：`<repo-root>/scripts/eval_maskcontrol_kv.py`

---

## 1. 设计哲学：frozen base + 轻量 adapter，near-identity init

KV-Control 的核心 bet 只有一句话：

> **不动 frozen base transformer 的任何权重，只给每一层 self-attention 追加一段"可以看的控制上下文"（extra K/V tokens），并让整个 adapter 在第 0 步严格等价于 identity。**

拆开是三个决策：

### 1.1 为什么 frozen base？

Base transformer（text-to-motion masked transformer）已经花了几千个 epoch 学会了"文本 → 动作分布"。控制任务（让 pelvis 走给定轨迹）是在这个分布上做**局部修正**，不是重学分布。Finetune 全模型的代价是灾难性遗忘 + 每个控制任务一份全量权重；frozen base + adapter 则是：

- 生成质量的下界由 base 保底（adapter 初始为 identity，见 1.3）；
- adapter 只有 ~9.6M 可训练参数（MaskControl port 上：`encoder_control` 9.0M + KV projections 0.59M + Q-residual 0.05M + 8 个 scalar bias），单卡即可训；
- 同一 base 可以挂多个不同控制任务的 adapter。

代码上"frozen"是显式执行的：`control_transformer_kv.py` 的 `ctrl_train()` override（L220-242）先 `freeze_block(self)`（L228）冻结全模型，再逐个 `unfreeze_block` adapter 模块（L229-242）；`ctrl_eval()`（L244-246）则只做 `freeze_block(self)`。`freeze_block`/`unfreeze_block` 定义在原版 `control_transformer.py` L35-43。

### 1.2 为什么改 K/V 而不是改 Q？

Attention 里 Q、K、V 的角色不对称：

- **Q 决定"每个 motion token 想看什么"** —— 改 Q 等于扭曲 base 已学好的查询语义。frozen backbone 的每一层 Q 都是针对 base 的 K/V 分布校准的，动了 Q，base token 之间的 attention pattern 会整体漂移，破坏面积大、且难以做到 identity init（Q 的任何非零扰动都会重排 softmax 权重）。
- **K/V 决定"有什么内容可以被看"** —— 在 kv-seq 轴上 concat 额外的控制 token（`K_aug=[K_base; K_ctrl]`，`V_aug=[V_base; V_ctrl]`），base token 之间原有的 attention 结构一个数都不用变，只是多了几列**候选上下文**。控制信号以"新的可看内容"的身份进入，模型可以学着看、也可以学着不看（由 `ctrl_attn_bias` 门控，见 §2.3）。

代码位置：MaskControl port 在 `_layer_with_kv_injection` L91-93 做 `k_aug = cat([k_base, k_ctrl], dim=1)`、`v_aug` 同理；v4 版在 `forward_with_kv_context_and_crossattn` L172-173 做 `k_aug = cat([output, ctrl_k], 0)`、`v_aug = cat([output, ctrl_v], 0)`。

（Q 不是完全不能动——paper headline config 里有一个受严格约束的 Q-residual，但它是锦上添花的二阶修正，且必须小心避开 pitfall #3，见 §2.4。）

### 1.3 Near-identity init：三重保险

整个 adapter 在 step 0 必须**数值上几乎等价于 base 前向**，否则 frozen base 的输出分布一开始就被打乱，训练从"修正"变成"重学"。三个机制各管一段：

| 机制 | 做法 | 效果 | 代码 |
|---|---|---|---|
| `kv_down` zero-init | down 投影零初始化，up 标准初始化 | `K_ctrl=V_ctrl=0`，但梯度活着（见 §2.2 的"双零死支"分析） | kv 版 L190-191；v4 版 L303-304 |
| `ctrl_attn_bias = -5.0` | 控制列 attention logits 上加 -5 的可学习标量 | 每个 ctrl token 相对单个 base token 的权重 ≈ e^-5≈0.0067；在本 substrate 的 S=50/T_ctrl=49 下 per-ctrl-token 归一化质量 ≈ 1.3e-4，**总** ctrl 质量 ≈ 0.65%（<1%），近似看不见 | kv 版 L195-198；v4 版 L308-310 |
| `q_gates = 0` | Q-residual 出口标量门零初始化 | Q 通路 step 0 完全关闭 | kv 版 L208；v4 版 L320-322 |

三个门是**串联冗余**的：即使 `kv_up` 标准初始化产出了非零的 K_ctrl/V_ctrl（它不会，因为 down 是零），bias=-5 也把这些列压到总计 <1% 质量（见 §2.3 的算术）；即使 Q-residual 的 `q_down/q_up` 都是标准初始化（有意为之），gate=0 也让它输出严格为 0。

---

## 2. 模块拆解

Adapter 由四类模块组成。每个模块按"是什么 / 为什么 / 初始化 / 代码位置"过一遍。

### 2.1 `encoder_control` — 控制信号编码器（9.0M，adapter 的大头）

**是什么**：一个 Conv1D 下采样编码器，把 per-frame 的控制条件 `ctrlNet_cond`（帧级，196 帧 × C 通道）编码成 token 级的控制特征（49 token × D=latent_dim）。结构是 `input C→512→D`，`down_t=2, stride_t=2`，即 4× 时间下采样。

**为什么**：base transformer 工作在 VQ token 序列上（unit_length=4，196 帧 → 49 token）。控制信号是帧级几何量，必须先降到与 motion token **相同的时间分辨率**，K_ctrl/V_ctrl 的每一列才能和 motion token 一一对应。⚠ 这意味着 encoder 的 stride 链**必须匹配 substrate tokenizer 的下采样率** —— 换了 tokenizer 率就要改 stride（数据适配文档会展开）。

**初始化**：标准初始化。它不需要 zero-init —— identity 保证由下游的 `kv_down` 和 `ctrl_attn_bias` 负责，encoder 从第一步就可以放心学表征。

**代码**：v4 版 L275-285（196→49 token）；MaskControl port 在 `_encode_ctrl_to_kv`（kv 版 L268-279）里调用：L271 `encoder_control(ctrlNet_cond.permute(0,2,1))` → L272 permute 成 `(T_ctrl, B, D)`。输入通道数 C 由控制任务决定：pelvis trajectory C=6，多关节 C = n_joints × 2 × 3（见 §3）。

### 2.2 `kv_down` / `kv_up_k` / `kv_up_v` — per-layer 低秩 K/V 投影（0.59M）

**是什么**：每层一组 low-rank 投影：共享的 `kv_down: D→r`（r=64, bias=False），分叉的 `kv_up_k: r→D` 和 `kv_up_v: r→D`。对 encoder 输出的控制特征做 `h = kv_down[i](ctrl_feat)`，然后 `k_ctrl = kv_up_k[i](h)`，`v_ctrl = kv_up_v[i](h)`（kv 版 L274-278；v4 版按层从 `ctrl_kv_list[layer_idx]` 取用，L108）。

**为什么 per-layer**：base 每一层的 K/V 语义不同（低层局部运动学、高层全局语义），控制信号需要 per-layer 的适配才能在每层都以"该层的语言"呈现。**为什么低秩**：r=64 vs D=384 把每层参数从 2×D² 压到 ~3×D×r，8 层总共 0.59M；控制信号本身是低维几何量（C=6），不需要满秩表达。

**初始化（关键不变量）**：`kv_down` **零初始化** + `kv_up_*` **标准初始化**。这是"identity at init WITH live gradients"的标准构型：

- 前向：`down=0` ⇒ `h=0` ⇒ `k_ctrl=v_ctrl=0`，严格 identity；
- 反向：`∂L/∂W_down ∝ W_upᵀ · (∂L/∂k_ctrl) · ctrl_featᵀ`，因为 `W_up ≠ 0`，梯度非零，通路活着。
- **反例（真踩过的坑，v4 历史文件名即证据）**：down 和 up **都**零初始化 ⇒ 两个方向的梯度都含对方权重因子 ⇒ 双零相乘 ⇒ 梯度恒为 0，分支永久死亡。仓库里留着尸体：`models/mask_transformer/control_transformer_t_concat_v8_kv_bk_20260322_deadgrad.py`。同一规则也约束 Q-residual（`q_down` 必须标准初始化，只有 `q_gates` 是零，v4 版 L317-318 有 NOTE 注释）。

**代码**：kv 版 L186-192（定义）+ L190-191（zero-init loop）；v4 版 L289-304（L303-304 是 zero-init：`C^K = C^V = 0 at step 0`）。

### 2.3 `ctrl_attn_bias` — per-layer 可学习标量门（8 个标量）

**是什么**：每层一个可学习标量（`torch.full((1,), -5.0)`），以 additive bias 的形式加在 attention logits 的**控制列**上（pre-softmax）。

**为什么**：它是控制信息流入量的**总闸门**。即使 K_ctrl 训到和 K_base 同量级，softmax 前的 -5 也让每个 ctrl token 只有 `e^-5≈0.0067` 的相对权重（vs 单个 base token）。归一化后（等基线 logits，S=50 个 base token + T_ctrl=49 个 ctrl token）：per-ctrl-token 质量 = e^-5/(50+49·e^-5) ≈ **1.3e-4**，全部 ctrl 列合计 ≈ **0.65%**，base attn 输出只被缩放 ≈0.9935——init 时近似看不见。训练中每层各自学会把闸门开多大——这给了模型一个平滑的、per-layer 的"要不要看控制"旋钮，而不是硬开关。它也是可解释性探针：训完看各层 bias 值就知道控制在哪些层起作用。

**初始化**：-5.0（`ctrl_attn_bias_init` 可调，eval 脚本 CLI `--ctrl_attn_bias_init` 默认 -5.0，`eval_maskcontrol_kv.py` L112 附近）。

**代码 — bias 只落在控制列，这点必须构造正确**：
- kv 版 L95-103：`attn_mask = q.new_zeros(B*H, S, S+T_ctrl)`（L96），然后 `attn_mask[:, :, S:] = ctrl_bias_scalar`（L97）——只有最后 T_ctrl 列吃 bias；base-token 列另行处理 key_padding_mask（L99-103，`masked_fill -inf` 只作用于 `[:, :, :S]`）。
- v4 版 L127-138：零矩阵 `[S_q, S_kv]`，`bias_mask[:, -T_ctrl:] = ctrl_bias`，再与 factorized attention mask 合并（L135-136）。
- 定义：kv 版 L195-198；v4 版 L308-310。

### 2.4 Q-residual — 上半层共享低秩 Q 修正（0.05M，paper headline config 含它）

**是什么**：一个**共享**（非 per-layer）的 `q_down: D→r` + `q_up: r→D`，配 per-layer 标量门 `q_gates`，只作用于 `layer >= num_layers//2` 的上半层：`q_delta = q_up(q_down(x)) * q_gate`，加到该层的 Q 上。

**为什么**：K/V 注入给了"新的可看内容"，但 base 的 Q 是在无控制分布上校准的——上半层（语义层）的 query 有时需要微调"朝控制 token 看的角度"。限制在上半层 + 共享投影 + 零门，是把 §1.2 里"动 Q 危险"的破坏面压到最小的折中。历史上它带来 KPS ~10% 改善（Ep1000 5r：4.90 vs 5.44，见项目记忆），是二阶增益，**移植新 substrate 时可以先不带**（eval 脚本有 `--no_q_residual` 开关，L113/L178）。

**初始化**：`q_down/q_up` 标准初始化，`q_gates` 全零 —— 同 §2.2 的"单零门 + 活梯度"构型。

**⚠ Pitfall #3 — Q-residual 的 K/V 污染（正确实现必须长这样）**：

大多数 PyTorch transformer layer 的 QKV 是**一次** `F.linear(x, in_proj_weight)` 再 `chunk(3)` 出来的。如果把 `q_delta` 加在 **x 上再投影**，residual 会通过共享的 in_proj **同时泄漏进 K_base 和 V_base**。init 时 gate=0 完全无症状；gates 一打开，frozen backbone 的 K/V 语义就开始静默漂移——这正是"你以为只动了 Q，其实把地基也动了"。

正确写法（kv 版 L69-82，代码即规范）：

```python
# 注释 L69-71 + 代码 L72-73: 先用【未修改的 x】算出 q, k_base, v_base
qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(3)
q, k_base, v_base = qkv
# L77-82: q_delta 单独过 Q 的那 1/3 权重，只加到 q 上
q_delta = q_up(q_down(x)) * q_gate
W_Q = attn.in_proj_weight[:D]          # L81: 只切 Q 分块
q = q + F.linear(q_delta, W_Q)         # L82: k_base/v_base 原封不动
```

v4 版的等价实现走的是另一条路（post-norm 路径 L166-182）：因为 v4 直接调 `mod.self_attn(q_input, k_aug, v_aug, ...)` 且 q/k/v 输入本来就分离，只需保证 `q_input = output.clone(); q_input[1:1+S_motion] += gate_val * q_delta_exp`（L169-171）只喂给 query 参数、而 `k_aug/v_aug` 从未修改的 `output` 构造（L172-173）。**两条路殊途同归的检查标准：K_base/V_base 的计算图里不允许出现 q_delta。**

### 2.5 结构收口：injection 怎么接进前向

MaskControl port 的 `trans_forward`（kv 版 L281-319）展示了最小接法：

1. **base fallback**（L284-285）：`ctrl_net` 关闭或 `ctrlNet_cond is None` 时走 `_base_trans_forward`（L321-339）——一条**逐 bit 复刻** base MaskTransformer 的路径（L337 直接 `self.seqTransEncoder(xseq, ...)`），保证无控制时行为与 base 完全一致。这条 fallback 也是 eval 的对照组来源（no-control baseline）。
2. L299：`kv_list = self._encode_ctrl_to_kv(ctrlNet_cond)` 一次算好全部层的 (K_ctrl, V_ctrl)。
3. L302-312：逐层调 `_layer_with_kv_injection`，替代 `nn.TransformerEncoderLayer` 的默认前向（L116-122 手工复刻其 post-norm residual + FF，dropout1/norm1/linear1→activation→dropout→linear2/dropout2/norm2）。
4. 继承层面：`KVControlTransformer(ControlTransformer)`（L125）先 `super().__init__`，然后 **delattr 掉父类的 ControlNet 分支**（L174-176：`del self.seqTransEncoder_control / first_zero_linear / mid_zero_linear`）——这是"subclass 换机制"的移植范式（详见两 substrate 对照 §6）。附带一个教训化的细节：`ctrl_train()` 会被父类 `__init__` 在 adapter 存在前调用，所以 L224-227 有 `if not hasattr(self, "kv_down"): return` 的 hasattr guard，`__init__` 末尾（L213）再显式补调一次。

---

## 3. 闭环控制条件 `ctrlNet_cond`：(residual, absolute) 双通道

### 3.1 结构

```python
# v4 版 _build_ctrl_cond, L376-387
ctrlNet_cond  = (global_joint - pred_joints) * global_joint_mask.unsqueeze(-1)  # L377: 残差通道 err
ctrlNet_cond2 = global_joint * global_joint_mask.unsqueeze(-1)                  # L378: 绝对目标通道 abs
# trajectory 模式只取 pelvis (joint 0), L380-382
# 最终 cat([(gt - pred)*mask, gt*mask], dim=-1) → C = 3+3 = 6（pelvis）
```

顺序约定是 **(residual, absolute)** —— err 在前、abs 在后，encoder 的输入通道语义依赖这个顺序，移植时不要交换。多关节推广：C = n_ctrl_joints × 2 × 3。mask 把非 keyframe 的帧清零，天然支持稀疏 keyframe 控制。

### 3.2 为什么必须闭环（err 通道每一步重算）

两个通道分工明确：

- **abs 通道**告诉网络"目标在哪"——静态信息；
- **err 通道**告诉网络"**当前预测**离目标还差多少、差在哪个方向"——它必须从**当前这一步的 decode 结果**实时算出，训练时每个 batch 重算一次，推理时每个 MaskGIT iteration 重算一次。

这就是 iterative trajectory error correction 的本体：网络学的映射是"给定残差 → 输出修正后的 token 分布"，每一轮迭代里残差缩小、修正变细。`pred_joints` 是 iterative current estimate（不是 frozen base 的一次性输出——这一点论文 §3 曾被 gpt_advice 纠正过，见记忆 `feedback_p_pred_iterative_estimate`）。

**反面教材（pitfall #13，MoMask port 的四大败因之一）**：MoMask port 早期实现里 err 通道被喂了常数 0（static err）。后果是网络只知道目标、不知道自己差多远——闭环原则被整个移除，控制退化成"看一眼目标然后凭感觉"，KPS 停在 30.6cm 量级（对照 MaskControl port 正确闭环后的 1.29cm）。同一 pitfall 还解释了 pitfall #9 的表象差异：MoMask 上稀疏 keyframe 训练 4 次尝试全部 NaN（ep 11-39），而 MaskControl substrate 上配合**正确的闭环 err 通道**+ 其损失配比，random density 训练全程稳定——说明稀疏不稳定是"static err × 稀疏"的交互问题，不是稀疏控制本身不可训。

**移植检查项**：在新 substrate 上，grep 你的训练 loop 和推理 loop，确认 `ctrlNet_cond` 的第一通道来自 `gt - 当前 pred` 且 `pred` 在每次迭代后更新。任何"预计算一次 err 然后复用"的写法都是 pitfall #13 复发。

---

## 4. 越过离散码本的可微路径：soft-decode 三层结构

控制损失是几何空间的（关节位置 L1/MSE），但 base 输出的是**离散 codebook index 的 logits**——argmax/采样不可微。KV-Control 用同一条 soft-decode 路径在三个不同阶段建立梯度通道：

### 4.1 第一层：训练期 soft-decode loss

```
emb = softmax(logits) @ codebook[0]     # 期望 embedding，codebook 向量的凸组合，完全可微
    → frozen VQ decoder → denorm → recover_from_ric (FK) → masked keyframe 上的 L1
```

梯度链：`L_traj → FK → decoder → soft emb → logits → transformer → KV adapter`。每一环都必须可微——尤其 **FK（`recover_from_ric`）是 load-bearing 的**：它是 HumanML3D 树结构专用的，换数据集必须先移植一个可微等价物并单测其梯度（数据适配文档展开）。注意 soft-decode 混合的是**哪个 codebook** 也是 substrate 相关的：MaskControl 有效地只用 base codebook（`codebook[0]`）；MoMask RVQ 有 6 个 quantizer（pitfall #11 的 OOD 雷区）；v4 substrate 是 Q=6 个 part codebook 沿序列 unpack。

### 4.2 第二层：Stage-1 TTT（推理期 logits 优化）

推理时在**每个 MaskGIT step 内**，把该步的 logits 当自由变量，用 Adam（lr=6e-2）沿同一条 soft path 优化 trajectory MSE。两种日程：

- **uniform**（已发布的 M1/M2/M3 数字全部用它）：每步固定 `each_iter` 次（M1/M3 = 35/step，ts=10 共 350 次；M2 = 100/step 共 1000 次）；
- **dynamic**：step s 分到 `(s+1) × |each_iter|` 次迭代——后期 step 决定细节，多给预算。substrate 的触发约定是 **`each_iter` 传负值**（`control_transformer.py` L527-531：`each_iter>0` 走 uniform，`<0` 走 dynamic）。

> ⚠ **标注勘误**：本 harness 曾把 M1/M3 标成 "35 dynamic"——错误。eval 脚本传的是 +35 且从未取负（旧 `--ttt_dynamic` flag 只影响打印标签，不改行为），所以已发布的 ep6000 M1/M3 数字实际是 **uniform 35 iters/step**。脚本现已接通 `opt.each_iter = -each_iter if ttt_dynamic else each_iter`，未来若真跑 dynamic 会产生不同（未发布）的数字。

Stage-1 优化的仍是 logits ⇒ 最终输出仍要过 codebook 量化 ⇒ **精度存在量化下限**（见 4.4）。

### 4.3 第三层：Stage-2（last_iter，连续 embedding 直接优化）

生成完成后，**绕过 codebook**，直接把连续 embedding 当变量，Adam 优化 keyframe MSE，然后过 decoder 出动作。`--last_iter 600 --last_lr 6e-2`（eval 脚本 L93-94）。这是三层里**唯一没有量化下限的杠杆**。

### 4.4 为什么 Stage-2 带来 10× 提升：量化下限分析

5-rep 实测（KV-on-MaskControl v16→v18 ep6000，HumanML3D test，pelvis trajectory）：

| protocol | Stage-1 | Stage-2 | FID | KPS |
|---|---|---|---|---|
| M1 | 35 uniform | **0** | 0.152±0.011 | **11.75±0.22 cm** |
| M2 | 100 uniform | 600 | 0.098±0.006 | **1.10±0.01 cm** |
| M3 | 35 uniform | 600 | 0.0875±0.008 | **1.29±0.02 cm** |
| no-control baseline（dispatch-bug evals 实测，非正式 M0 JSON） | — | — | 0.1455 | 63.05 cm |

解读：Stage-1 无论优化多狠，输出必须落在 codebook 的离散格点上——每个 token 只能表达码本里最近的那个 4 帧动作片段，pelvis 位置的可达精度被**码本分辨率**卡死在 ~10cm 量级（M1 的 11.75cm 基本就是这个 floor 的实测值）。Stage-2 把变量换成连续 embedding，等于在码点之间连续插值，floor 消失，KPS 直落 1.1-1.3cm（**≈10×**）。旁证一：M2（Stage-1 预算 ~3× 于 M3）最终 KPS 只比 M3 好 0.19cm——说明 Stage-1 打满也只是更贴近同一个 floor；旁证二：FID 在 M2/M3 反而**更好**（0.098/0.0875 vs M1 0.152），说明 Stage-2 的连续微调不以分布质量为代价。

**移植含义**：如果新 substrate 上 M1 类协议的误差停在某个不再下降的平台，先怀疑量化 floor 而不是训练失败——上 Stage-2 再下结论。同时协议参数（time_steps/cond_scale/each_iter/ttt_dynamic/last_iter）必须逐协议 pin 死并 assert（pitfall #15）：eval 脚本用谓词自动识别 M1（`each_iter==35 and time_steps==10 and not ttt_dynamic and last_iter==0`，uniform-35 是 M1 的定义），M2/M3 靠 override `--last_iter` 产生，默认值漂移会让数字之间不可比。

---

## 5. 训练目标：CE(0.1) + traj L1(0.9)

```
L = 0.1 × L_xent(masked-token CE)  +  0.9 × L_traj(soft-decode keyframe L1)
```

（`xent=0.1, ctrl_loss=0.9`；完整训练 config：bs=64/rank × 4 = global 256，lr=8e-4 = Goyal 4× 线性缩放自 2e-4，warm_up_iter=2000，milestones=[12500]，gamma=0.1，cond_drop_prob=0.1，6000 ep，~11 s/ep on 4×A100-80GB。）

**为什么这个 balance**：

- **CE 权重小（0.1）但不能为零**。base 已经 frozen，token 分布的"对不对"主要由 base 保证；CE 在这里的角色是**正则**——把 adapter 的输出 logits 锚在 base 的语言里，防止 traj loss 把 logits 推到分布外（推到 decoder 没见过的 token 组合上，FID 会崩）。
- **traj 权重大（0.9）因为它是 adapter 存在的唯一理由**。adapter 从 identity 出发，所有"偏离 identity 的动机"都来自 traj loss；signal 太弱则 `ctrl_attn_bias` 的闸门开不起来。
- **这组数字是 substrate 原配，不要擅自调**。MaskControl port 的一条铁律是所有超参与 MaskControl 自己的 ControlNet 训练**完全一致**（A/B-clean），使得最终 delta 可完全归因于机制本身（ControlNet-add vs KV-inject）。这个配比在他们 substrate 上被验证过稳定（对照 pitfall #9：MoMask 上另一套 loss balance + static err 就 NaN 了）。移植新 substrate 时：第一优先复用目标 substrate 自己 ControlNet/控制基线的 loss 配比；没有基线才从 0.1/0.9 起步。

---

## 6. 两个 substrate 上的具体形态对照

同一个机制，在两个 substrate 上长得不一样。移植第三个 substrate 时对着这张表找自己的位置：

| 维度 | v4 native（paper headline） | MaskControl port（3 天完成的 hot-swap 验证） |
|---|---|---|
| 实现文件 | `models/mask_transformer/control_transformer_t_concat_v8_kv_v4.py` | `references/MaskControl/models/mask_transformer/control_transformer_kv.py` |
| 父类 | `MaskTransformer`（`transformer_t_concat_v4.py`，L38 import；直接继承 base） | `ControlTransformer(MaskTransformer)`（继承**他们的 ControlNet 类**，再 delattr 其分支，L174-176） |
| Tokenizer / codebook | PartVQ，Q=6 个 part codebook 沿序列 unpack（~295 token 有效序列） | RVQ `rvq_nq6_dc512_nc512_noshare_qdp0.2`，soft-decode 只混 base codebook；49 motion token |
| Text conditioning | CLIP text **序列** + dense text cross-attn block 每 `cross_attn_interval` 层一次（父类提供 `encode_text_with_seq`；injection 循环 L185-190 在 self-attn+FFN **之后**插 cross-attn） | 单个 CLIP cond token prepend 在序列头（`trans_forward` L287-296 的 cond_tok cat；无 per-layer cross-attn） |
| Base 规模 | latent 384, 20L（factorized attention, Q_per_tok 展开） | latent 384, 8L, 6H, ff 1024, dropout 0.2 |
| 注入点实现 | 自写 `forward_with_kv_context_and_crossattn`（L64-192）：直接调 `mod.self_attn(q_input, k_aug, v_aug, ...)`，需处理 factorized_mask 合并（L135-136）和 `q_delta.repeat_interleave(Q_per_tok)`（L141-146） | 自写 `_layer_with_kv_injection`（L41-122）：手动 `F.linear` in_proj + `F.scaled_dot_product_attention`（L105-110），需处理 key_padding_mask → attn_mask 融合（L99-103） |
| Q-residual 形态 | 共享 q_down/q_up + 上半层（`num_layers//2` 起，L313-322）标量门；加在 `q_input`（clone 后只改 query 入参，L169-171） | 同构（L201-210）；但因 QKV 共用 in_proj，必须走 pitfall #3 的 `W_Q = in_proj_weight[:D]` 切块加法（L77-82） |
| decode / TTT loop | 自家 generate（本来就是我们写的） | **原样复用**他们的 `generate_with_control` + Stage-1/Stage-2 TTT，一行不改——只 override `ctrl_train/ctrl_eval/trans_forward` + base fallback（pitfall #12 的正解） |
| 无控制 fallback | base fallback 分支（`_build_ctrl_cond` 上游判 None） | `_base_trans_forward`（L321-339），bit-exact base 路径，**不 delegate 给父类**（父类 forward 会摸已 delattr 的分支） |
| 训练入口 | `train_ctrlnet_ddp.py` 系 | `references/MaskControl/train_ctrlnet_ddp.py`（cross-alloc DDP fork，rank-0-only ckpt/TB/eval） |
| 结果 | M3 KPS **0.40 cm** / FID 0.065（co-designed substrate 的上限） | M3 KPS **1.29 cm** / FID 0.0875（vs 同 substrate no-control ~63cm；机制可移植性的存在性证明） |
| Eval | `scripts/eval_v4_ctrlnet_ttt.py` | `scripts/eval_maskcontrol_kv.py`（fork 自前者；5-rep，`pred_num_batch=16`——语义是"每次 generate 累积 16 个 loader batch"而非"总共 16 个 batch"，设 99999 会静默跳过全部生成，pitfall #8） |

两条移植总结（详细版在 pitfall 文档，这里只留方法层面的两句）：

1. **MaskControl port 成功的原因**是把改动面压缩到 KV-Control 的定义域内：只换 `trans_forward` 的每层前向 + 挂 adapter 模块，substrate 的 generate loop、TTT、optimizer、data、loss 全部原样——这正是 §1 设计哲学在移植维度上的投影。
2. **MoMask port 失败的原因**是越界：手写 decode loop（pitfall #12）、static err（#13）、EMA codebook 被训练模式污染（#1）、RT-cascade 把 KV-modified token 当 OOD（#11）。四个坑没有一个出在 KV 机制本身——全部出在"替换了不该替换的 substrate 部件"。

> 下一篇（02）：移植操作手册 —— 对新 substrate 逐步骤应用本文机制，含 pitfall checklist 与验收 gate。

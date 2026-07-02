# 02 — Porting Playbook: 把 KV-Control 移植到新 substrate 的标准流程

> KVControl Adapt Harness · doc 02
> 本文把 MaskControl port(成功, 3 天含 debug, KPS 63→1.29cm)重述为可复用的 8+1 步流程,
> 每一步都标注对应的 pitfall #(见 `04_pitfall_catalog.md`,编号与本文一致)和真实代码位置。
> 反面教材是 MoMask port(部分失败, 4 层 bug 叠加, 最终 FID 0.581 / KPS 30.6cm)——
> 本 playbook 里每条 "绝不 / 必须" 都是从那次失败里换来的。

**两次移植的最终数字(判定 Step 8 的锚点):**

| run | protocol | FID | Top3 | KPS |
|---|---|---|---|---|
| KV-on-MaskControl (v16→v18, ep6000) | M1 (Stage-1 TTT only, **uniform-35**/step) | 0.152±0.011 | 0.797 | 11.75±0.22 cm |
| 同上 | M2 (100 uniform + last_iter 600) | 0.098±0.006 | 0.791 | 1.10±0.01 cm |
| 同上 | M3 (**uniform-35** + last_iter 600) | 0.0875±0.008 | 0.789 | 1.29±0.02 cm |
| MaskControl substrate no-control baseline (dispatch-bug evals 实测; Top3 为 informal estimate, 无正式 M0 JSON) | — | 0.1455 | ~0.80 | 63.05 cm |
| paper v4 KV-Control (自家 substrate, 参照) | M3 | 0.065 | 0.799 | 0.40 cm |
| KV-on-MoMask (失败教材) | best ep600 base-only decode | 0.581 | — | 30.6 cm |

---

## Step 0 — 侦察 (Recon): 动手前必须读完的 substrate 代码

**规则来源: Karpathy R8 (Read before you write) + pitfall #12。** MoMask port 的根因就是没读透
substrate 的 decode loop 就自己重写了一个。

必读四件套,以及每件要回答的问题:

1. **Base transformer forward。** 找到 per-layer self-attention 的实现位置。
   - 关键问题: 用的是 `nn.TransformerEncoderLayer`(in_proj 融合 QKV)还是自定义 attention?
     norm 是 pre-norm 还是 post-norm? 有没有 cond token / padding column 的前缀处理?
   - MaskControl 的答案在 `references/MaskControl/models/mask_transformer/transformer.py`
     (标准 `nn.TransformerEncoderLayer`, post-norm),这直接决定了我们的注入 helper
     `_layer_with_kv_injection` 必须手工镜像 dropout1/norm1/linear1→activation→dropout→linear2/dropout2/norm2
     (见 `references/MaskControl/models/mask_transformer/control_transformer_kv.py` L116-122)。
   - v4 substrate 是 pre-norm/post-norm 双支 + factorized attention mask,对照
     `models/mask_transformer/control_transformer_t_concat_v8_kv_v4.py` L148-182。

2. **generate / decode loop。** 找到 MaskGIT(或等价)采样循环:pad 初始化、score 初始化、
   remask 策略、每步调用哪个 forward。
   - MaskControl: `references/MaskControl/models/mask_transformer/control_transformer.py`
     的 `generate_with_control` + Stage-1/Stage-2 TTT 逻辑(全文件 1161 行,decode 相关都在这里)。
   - **回答一个问题即可: "decode loop 每步调用的那个函数叫什么、签名是什么"** ——
     那就是你唯一要 override 的入口(Step 3)。MaskControl 里是 `trans_forward`。

3. **已有的 control 机制(如有)。** 如果 substrate 自带 ControlNet 类的东西,读它的:
   - 训练入口(loss 组合、cond 构造):MaskControl 的 `ControlTransformer.__init__`
     在 `control_transformer.py` L48-104,ControlNet 分支 = `seqTransEncoder_control`
     (deepcopy encoder)+ `first_zero_linear`/`mid_zero_linear`,注入方式是
     `forward_with_condition` L26-33(逐层 `output = output + control_feat`, L30)。
   - freeze/unfreeze 工具: `freeze_block` L35-38 / `unfreeze_block` L40-43 —— 直接复用。
   - **它的 ctrlNet_cond 格式**:必须确认是不是 closed-loop
     (`cat([(gt-pred)*mask, gt*mask], -1)`,err 通道每个 iteration 从当前 prediction 重算)。
     MoMask port 喂了 static `err_xyz=0`,直接摧毁了 iterative-correction 原理(pitfall #13)。
     v4 的正确实现参照 `_build_ctrl_cond`,
     `models/mask_transformer/control_transformer_t_concat_v8_kv_v4.py` L376-387。

4. **Tokenizer / VQ 结构。** 回答:
   - downsample rate(unit_length)是多少? → 决定 encoder_control 的 stride 链
     (MaskControl: 196 frames → 49 tokens = 4×,对应 down_t=2, stride_t=2)。
   - 有几个 codebook? soft-decode 应该 mix 哪个? MoMask RVQ = base + 5 residual
     (小心 pitfall #11 RT-cascade OOD);MaskControl 实际只用 base codebook;
     v4 是 Q=6 part codebooks 沿 sequence 展开。
   - **quantizer 是不是 EMA 更新?** 是的话 pitfall #1(EMA buffer 污染)在等你。
   - feature→joints 的映射是不是 `recover_from_ric`? 不是的话先移植可微 FK 并单测梯度
     (见 `05_data_adaptation.md`;这条是 load-bearing,soft-decode 全链路
     L_traj → FK → decoder → soft emb → logits 断在 FK 就全断)。

产出物:一页 recon note,回答上面所有加粗问题。答不出任何一个 → 不许进 Step 1(Karpathy 第 1 条:停下来问)。

---

## Step 1 — 设计决策树: subclass 谁、复用什么、替换什么

```
substrate 自带 ControlTransformer 类的机制?
├── 是 → subclass 它 (MaskControl 路线)
│        复用: loss 组合 / ctrlNet_cond 构造 / generate_with_control / Stage-1+Stage-2 TTT
│               / optimizer 与数据管线 / eval 协议
│        替换: ControlNet 注入分支 (delattr) → 换成 KV adapter
│        override: ctrl_train / ctrl_eval / trans_forward (+ _base_trans_forward fallback)
└── 否 → subclass base transformer (v4 路线)
         自己带: encoder_control + ctrlNet_cond 构造 (_build_ctrl_cond 那套)
                 + ctrl_train/ctrl_eval + 带注入的 forward
         仍然复用: substrate 的 generate loop (Step 3 THE RULE 不变)
```

MaskControl port 的实例化:

```python
# references/MaskControl/models/mask_transformer/control_transformer_kv.py
L38 :  from ... import ControlTransformer, freeze_block, unfreeze_block
L125:  class KVControlTransformer(ControlTransformer):
```

复用/替换清单(经验证的最小切面):

| 组件 | 决策 | 位置 |
|---|---|---|
| `generate_with_control` + TTT | **复用, 一行不改** | control_transformer.py (父类) |
| ctrl loss / cond_drop / mask 采样 | 复用 (父类 ctrl_train 训练循环) | 同上 |
| `encoder_control` (Conv1D C→512→D) | 复用父类的实例 (它就是 KV 的 ctrl encoder) | 父类 `__init__` |
| `seqTransEncoder_control` + zero_linears | **delattr 删除** | kv.py L174-176 |
| per-layer forward | **override** `trans_forward` | kv.py L281-319 |
| no-control 路径 | **手写** `_base_trans_forward` (不许 delegate) | kv.py L321-339 |

**A/B-clean 原则从这一步就锁定**:除了注入机制,optimizer/data/loss/超参与 substrate
自家 ControlNet 训练**逐项相同**(Step 6),否则 mechanism delta 不可归因。

---

## Step 2 — 实现 KV adapter 类 (MaskControl 实际 diff 摘要)

全部代码在 `references/MaskControl/models/mask_transformer/control_transformer_kv.py`(339 行)。
逐块拆解,每块都是可移植模板:

### 2.1 `__init__`: delattr 旧机制 + 加 KV 模块 (L128-216)

```python
# L152-170  super().__init__(...)   ← 父类会构建 ControlNet 分支并 load+freeze base ckpt
# L174-176  删掉父类的 ControlNet 分支:
del self.seqTransEncoder_control
del self.first_zero_linear
del self.mid_zero_linear
# L186-192  KV 低秩投影 (per-layer):
#   kv_down: D→r=64, bias=False, **ZERO-init** (L190-191)
#   kv_up_k / kv_up_v: r→D, **standard init**
# L195-198  ctrl_attn_bias: per-layer scalar Parameter, init -5.0
# L201-210  Q-residual: SHARED q_down/q_up (standard init) + per-layer q_gates (zero init),
#            q_start_layer = num_layers // 2 (只作用于上半层)
# L213      self.ctrl_train()   ← adapter 建好后重新调一次 (见 2.2 hasattr guard)
```

**Init 不变量(三条,缺一即废):**
1. `kv_down=0` + `kv_up` standard = 注入分支 identity at init **且梯度活着**。
   双零(down 和 up 都 zero)= dead branch,梯度全死——v4 的 q_down 注释原话:
   `control_transformer_t_concat_v8_kv_v4.py` L317-318
   "q_down standard init (NOT zero) so gradients flow through q_gates. q_gates=0 ensures identity init. Both-zero → dead branch."
2. `ctrl_attn_bias = -5.0`:每个 ctrl token 相对单个 base token 的权重 e^-5≈0.0067;
   在 S=50/T_ctrl=49 下 per-ctrl-token 归一化质量 ≈1.3e-4,**总** ctrl 质量 ≈0.65%(<1%)
   = near-identity at init,但可学习地打开。v4 对应 L308-310。
3. Q-residual 用 **shared 投影 + per-layer scalar gate(zero init)**,不是 per-layer 满投影
   (paper headline config;参数量 0.05M vs 0.59M KV)。

参数量 sanity(MaskControl 上): encoder_control 9.0M + KV projections 0.59M
+ Q-residual 0.05M + 8 scalar biases ≈ **9.6M trainable**。`_compute_param_counts`
(kv.py L248-264)在 `__init__` 末尾(L216)打出这张表——移植后第一件事就是核对它。

### 2.2 `ctrl_train` override + **hasattr guard** (L220-242) — 必踩坑预防

父类 `__init__` 内部会调 `self.ctrl_train()`,此时子类的 adapter 还没建
(Python MRO:子类 `__init__` 里 `super().__init__()` 先跑)。不加 guard 直接 AttributeError,
或更糟——摸到已 delattr 的模块。

```python
# L224-227
if not hasattr(self, "kv_down"):
    return   # no-op: 被父类 __init__ 提前调用时静默跳过
# L228-242: freeze_block(self) → unfreeze encoder_control / kv_down / kv_up_k / kv_up_v
#           / ctrl_attn_bias / (q_down, q_up, q_gates)
```

`ctrl_eval`(L244-246)只做 `freeze_block(self)` —— 安全,不访问已删模块。

**同场加映 pitfall #1(EMA-codebook 污染,MoMask 血案):** 如果 substrate 的 VQ 是
EMA quantizer,`requires_grad=False` **保护不了 EMA buffer**。必须四层防御:
(a) `ctrl_train()` 内部以及每次 `model.train()` 之后都 `vq_model.eval()`;
(b) ckpt 保存时 strip `vq_model.*` keys;(c) resume load 时同样 strip;
(d) eval 侧 filter + 与 pristine RVQ 做 byte-equal preflight assert。
MoMask 当时的症状:训练悄悄改写 codebook → ckpt 携带漂移的 `vq_model.*` → eval load
覆盖干净 RVQ → **FID 22-25(纯垃圾)**,且全程无报错。

### 2.3 `trans_forward` override + base fallback (L281-319, L321-339)

```python
# L284-285  base fallback 分支:
if (self.ctrl_net is None) or (not self.ctrl_net) or (ctrlNet_cond is None):
    return self._base_trans_forward(...)
# L299      kv_list = self._encode_ctrl_to_kv(ctrlNet_cond)
# L302-312  per-layer loop over self.seqTransEncoder.layers:
#           取 (K_ctrl_i, V_ctrl_i), bias_i;
#           i >= q_start_layer 时组 q_resid=(q_down, q_up, q_gates[i-q_start_layer]) 否则 None;
#           调 _layer_with_kv_injection(...)
```

**`_base_trans_forward`(L321-339)必须手写、不许 delegate 到 `super().trans_forward`** ——
父类的实现会摸 `seqTransEncoder_control` 等已被 delattr 的模块(docstring L322-326 明说
"bit-exact base MaskTransformer path; must not delegate to parent")。它复制 preamble
(L327-336)然后走 plain encoder:
`output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[1:]`(L337)。
这条路径就是 Step 4 Tier-1 bit-exact 测试的被测对象。

### 2.4 注入 helper `_layer_with_kv_injection` (L41-122) — 机制核心

```python
# 注释 L69-71 + 代码 L72-73  Q/K/V 从 **未修改的 x** 投影:
#         F.linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(3) → q, k_base, v_base
# L77-82  Q-residual 只加到 q:
#         q_delta = q_up(q_down(x)) * q_gate
#         W_Q = attn.in_proj_weight[:D]          (L81, 切出 Q 的投影权重)
#         q = q + F.linear(q_delta, W_Q)          (L82; k_base/v_base 不动)
# L91-93  k_aug = cat([k_base, k_ctrl], dim=1); v_aug 同理   (kv-seq 轴 concat)
# L95-103 attn_mask: 零矩阵 (B*H, S, S+T_ctrl); ctrl 列 [:, :, S:] = ctrl_bias_scalar;
#         key_padding_mask 只 masked_fill base 段 [:, :, :S]
# L105-110 F.scaled_dot_product_attention(q, k_aug, v_aug, attn_mask=attn_mask, ...)
# L116-122 手工镜像 post-norm FF (对齐 nn.TransformerEncoderLayer 语义)
```

L69-82 就是 **pitfall #3(Q-residual K/V 污染)的修复本体**,细节见 Step 5。
ctrl encoder → KV 的编码在 `_encode_ctrl_to_kv`(L268-279):
`encoder_control(cond.permute(0,2,1))` → per-layer `kv_up_k[i](kv_down[i](feat))`。

v4 上同一机制的对照实现(post-norm 支):
`control_transformer_t_concat_v8_kv_v4.py` L166-182 —— `q_input = output.clone();
q_input[1:1+S_motion] += gate_val * q_delta_exp`(L169-171),
`k_aug/v_aug = cat([output, ctrl_k/v], 0)`(L172-173)。同一思想,两种 codebase 形态。

---

## Step 3 — THE RULE: 绝不 fork substrate 的 decode/generate loop

**这是整个 harness 的第 1 移植铁律,也是 MoMask port 的根因(pitfall #12)。**

MoMask port 当时手工重写了 MaskGIT decode loop,pad/score 初始化和 sampling order
与原版结构性偏离 → 单这一项就造成大幅 FID 损伤,而且和 pitfall #1/#11/#13 叠加后
根本无法归因。最终打捞结果 FID 0.581 / KPS 30.6cm。

正确做法(MaskControl port 实证):

- **REUSE substrate 的 `generate()` / `generate_with_control` / Stage-1+Stage-2 TTT,逐字复用。**
- **唯一 override 点是 per-layer forward 入口**(MaskControl 里叫 `trans_forward`),
  在它内部拼接 KV 注入(kv.py L281-319)。decode loop 通过多态自动调到子类实现,
  循环本身一个字符不改。
- 推论:如果 substrate 的 decode loop 没有一个干净的 per-step forward 入口可 override,
  **先给 substrate 提一个最小重构**(抽出 `trans_forward`),再做 port——
  也绝不 copy-paste loop 出来改。

同族坑一起锁死:
- **pitfall #13**:复用 loop 的同时,确认 ctrlNet_cond 的 err 通道在**每个 inference iteration**
  从当前 decode 重算(closed-loop)。static err = 机制名存实亡。
- **pitfall #11**:substrate 下游若有 frozen 模块(如 MoMask 的 ResidualTransformer)
  是在 un-controlled token 分布上训练的,它会把 KV-modified tokens 当 OOD 放大噪声
  (MoMask 实测 base-only decode FID 0.715→0.581 反而更好)。port 初期先 base-only decode
  评,确认机制本身工作,再决定下游模块去留。

---

## Step 4 — Smoke test: 四道门,全过才许训练

按 R9(验证手段必须真能抓到失败)设计,每道门都有明确判定数值:

| Tier | 测什么 | 怎么测 | 判定标准 |
|---|---|---|---|
| **Tier 1: base fallback 健全** | `ctrlNet_cond=None` fallback 路径行为正确 | 同 seed 同输入,`ctrlNet_cond=None` 走 fallback (kv.py L284-285) | **实际 smoke 脚本检查的是形状正确 + 无 NaN**。与原生 MaskTransformer 的 `torch.equal()` 逐位对照是**推荐的可选加强项**(需单独实例化原类并共享权重,当时未实装;模板亦标 optional)——若实装,同权重同算子理应逐位相同 |
| **Tier 2: near-identity at init** | 注入路径在 init 时≈无控制 | 刚构建的 model,带 ctrlNet_cond 跑 `trans_forward` vs `_base_trans_forward` | max abs diff 只来自 ctrl_attn_bias=-5 的 attention 质量分流(**总** ctrl mass <1%: per-token ≈1.3e-4 × 49 列 ≈0.65%,base 输出缩放 ≈0.9935);diff 应是小量;kv_down=0 保证 K_ctrl/V_ctrl 精确为 0(此时唯一差异源是 bias 列本身) |
| **Tier 3: gradient flow** | adapter 梯度通路活着 | 一个 batch,`loss.backward()`,遍历 trainable params | **init 后单次 backward 只要求 kv_down / ctrl_attn_bias / q_gates 的 grad 非零**(kv_down 经 standard-init 的 kv_up 回传,必须非零;全零 = 双零 dead branch,2.1 不变量 1)。⚠ kv_up_k/kv_up_v/encoder_control 在 init 时 grad **恰好为 0**(因 kv_down=0 ⇒ h=0),q_down/q_up 同理(q_gates=0)——这是正确实现的数学必然,**不许把它们的零梯度判 FAIL**;应改为 1-2 个 optimizer step 后再 assert 这些组的 grad 变非零(或对这些组只查非 None/无 NaN) |
| **Tier 4: frozen base** | base 权重训练中不动 | 训 N step 前后对 base 参数 hash;若 VQ 是 EMA,额外对 codebook buffer 做 byte-equal | hash 完全一致。**codebook buffer 一致性是对 pitfall #1 的直接测试**——requires_grad 检查抓不到它 |

另外一条**差分 FK 单测**(数据侧,pitfall 见 04 文档):新数据集若 feature→joints
不是 `recover_from_ric`,先对移植的 differentiable FK 做 `gradcheck`/有限差分,
再进任何训练。

Smoke 没全绿之前启动长训练 = 违反 fail-fast(R12)。MoMask 的 4 层 bug 中至少 #1 和 #13
会被 Tier 4 与一个 closed-loop assert 当场抓住。

---

## Step 5 — Codex review 要点: 我们真被抓过的 2 个 critical

铁律:adapter 代码写完必过 codex(gpt-5.5 xhigh)review 再部署。给 reviewer 的 brief
必须点名以下两类历史 critical(它们隐蔽到 smoke test 都不一定报警):

### 5.1 Q-residual K/V 污染 (pitfall #3)

错误写法:`x = x + q_delta` 然后才 `F.linear(x, in_proj_weight).chunk(3)` ——
`nn.TransformerEncoderLayer` 的 in_proj 是 QKV 融合的,residual 会经共享 in_proj
**泄漏进 K_base 和 V_base**。init 时 gate=0 完全无症状;gates 打开后 frozen-backbone
语义静默漂移,任何单点数值检查都看不出来。

正确写法即 kv.py L69-82:qkv 从**未修改的 x** 投影;单独切 `W_Q = in_proj_weight[:D]`
(L81),只把 `F.linear(q_delta, W_Q)` 加到 q(L82)。
Review checklist 措辞:**"证明 q_delta 到 K_base/V_base 之间不存在任何计算路径。"**

### 5.2 best_kps tracker 双 bug (pitfall #4)

MaskControl trainer 里两个独立 bug,叠加后 best-ckpt 选择完全失效:
(a) 保存 `best_kps.tar` 后 **best_kps_mean 从未更新** → 每次 eval 都覆盖保存;
(b) in-loop eval 把 kps_mean unpack 进 `_` 丢弃,比较用的是循环前的 stale 值 → tracker
冻结在第一次 eval。两个都修,`best_kps.tar` 才有意义。
Review checklist 措辞:**"追踪 best_* 变量的完整生命周期:更新点、比较点、保存点三者闭环。"**

### 5.3 顺带让 reviewer 扫的次级项

- `type(x) is Class` vs `isinstance`(pitfall #2,eval 路由;见 Step 7)。
- hasattr guard 的存在性(2.2)。
- ckpt save/load 的 `vq_model.*` strip(pitfall #1 的 b/c 层)。⚠ 注意:本 port 训练侧
  **没有** strip,ckpt 里实际带着 72 个 `vq_model.*` keys;且模型有同名 `vq_model` 子模块,
  `load_state_dict(strict=False)` 会**加载**(不是忽略)这些 keys、覆盖 pristine RVQ——
  eval 侧必须显式 `if not k.startswith("vq_model.")` 过滤 + byte-equal preflight(见 Step 7.3)。
- TB writer 的 try/except 包裹(pitfall #10,NFS 上 `writer.add_scalar` 的
  FileNotFoundError 曾在 ep17 和 ep152 杀死过两次训练——**per-iter 和 end-of-epoch
  两类调用点都要包**)。

---

## Step 6 — 训练: A/B-clean config + DDP 排雷

### 6.1 A/B-clean 原则

与 substrate 原生 control 方法**完全同 config**,机制 delta 才可归因。
MaskControl port 的锁定值(与其自家 ControlNet 训练逐项相同):

```
bs=64/rank × 4 ranks = global 256      lr=8e-4 (Goyal 线性缩放: 4 × 2e-4)
warm_up_iter=2000                       milestones=[12500], gamma=0.1
xent=0.1, ctrl_loss=0.9                 cond_drop_prob=0.1
6000 ep, ~11 s/ep on 4×A100-80GB (~18.5 hr)
latent 384, 8L, 6H, ff 1024, dropout 0.2, CLIP ViT-B/32
```

Goyal 规则提醒:如果你改了卡数/global batch,lr 同倍缩放、milestones 反向缩放——
但**先保证与 substrate 参考 config 的等效性可论证**,否则 A/B 不 clean。

### 6.2 DDP 排雷清单(全部真实炸过)

| # | 坑 | 症状 | 修复 |
|---|---|---|---|
| pitfall #5 | mkdir race | rank>0 在共享 FS 上 crash:`options/base_option.py` 的 `os.makedirs(exist_ok=False)`、`utils/utils.py` `init_save_folder`、`shutil.copytree` | 该路径上全部 `exist_ok=True` / `dirs_exist_ok=True` |
| pitfall #6 | `--is_continue` vs date-prefix | base_option 无条件给 `opt.name` 加 `z<date>_` 前缀 → resume 解析到**全新空目录**,静默从零训 | `is_continue` 时跳过 date prefix |
| pitfall #14 | `module.` prefix | DDP ckpt 的 state key 带 `module.` → eval load 大量 missing keys | eval 侧 strip(参照 `scripts/eval_maskcontrol_kv.py` L198-200, 2026-07 修正后行号) |
| pitfall #10 | TB writer on NFS | 训练中途 FileNotFoundError 直接杀进程 | 所有 `writer.add_scalar` try/except |
| pitfall #1 | EMA codebook | 见 2.2;**训练侧**的防御是 `vq_model.eval()` + save-strip | 四层防御全上 |
| — | unexpected keys | resume 出现 unexpected keys | 立即停训排查(既有铁律),多半是 #1 或 #14 的前兆 |

Cross-alloc DDP 的参考实现: `references/MaskControl/train_ctrlnet_ddp.py`
(L1-22 docstring:static rendezvous via `--node_rank/--master_addr/--master_port`
由 `scripts/kv_mc_orchestrator.sh` 注入;rank-0-only print/ckpt/TB/eval;
DistributedSampler + set_epoch;adapter param list 从 unwrapped module 取)。

### 6.3 训练期健康监控

- 稀疏 keyframe 密度若出现 NaN(pitfall #9):MoMask 上 densities {1,2,5} 四次尝试都在
  ep 11-39 NaN,但同样的 random-density 在 MaskControl(closed-loop err + 它们的
  loss balance)上稳定训完——**先怀疑 err 通道和 loss 配比,不要先怪 sparse 本身**。
- eval-while-training 的 ckpt 读写 race(pitfall #7):eval 读 `latest.tar` 时训练正在写
  → `PytorchStreamReader failed reading file`。修复:`cp` 到 frozen snapshot 路径
  (cp + atomic mv),eval 只碰 snapshot。这也是 Step 7 冻结快照的由来。

---

## Step 7 — Eval: 冻结快照 → M1 先行 → M2/M3 确认

参考脚本: `<repo-root>/scripts/eval_maskcontrol_kv.py`
(fork 自 `scripts/eval_v4_ctrlnet_ttt.py`,重定向到 MaskControl substrate)。

### 7.1 冻结快照 (pitfall #7)

绝不直接 eval `latest.tar`。实例:
`<repo-root>/output/v18_ep6000_frozen_snapshot.tar`(~244MB)
→ 产出 `<repo-root>/output/v16_kv_maskcontrol_4xa100_20260701_013823/eval_5r_M3_ep6000/eval_5r.json`。

### 7.2 协议矩阵与执行顺序

| protocol | 定义 | 用途 | 速度 |
|---|---|---|---|
| M1 | Stage-1 TTT only: time_steps=10, cond_scale=3.25, **each_iter=35 uniform**(每 step 固定 35 次,共 350 @ts=10), **last_iter=0** | 快筛:机制是否工作 | ~7 min/rep 量级,**先跑** |
| M2 | each_iter=100 uniform + **last_iter=600** | Stage-2 确认 | 慢 |
| M3 | **each_iter=35 uniform** + **last_iter=600** | paper headline | 慢 |

> ⚠ 勘误:M1/M3 曾被标 "35 dynamic"((s+1)×35/step)——错误。substrate 只在 `each_iter<0` 时走 dynamic(`control_transformer.py` L527-531),eval 脚本传 +35 且旧 `--ttt_dynamic` flag 只是打印标签。已发布 ep6000 数字全部是 uniform-35。脚本现已接通 negative-each_iter 约定。

脚本把 M1 编码为 argparse defaults,并自动检测:
`protocol = "M1" if (each_iter==35 and time_steps==10 and not ttt_dynamic and last_iter==0) else "CUSTOM"`。
M2/M3 **不是硬编码**,通过 `--last_iter 600`(+ each_iter)覆盖产生,报告为 CUSTOM。

**为什么 M1 先行且 M1 差不等于失败:** Stage-1 有 quantization floor
(优化的是 logits,输出仍被 codebook 离散化);Stage-2(last_iter=600)直接优化
**连续 embedding**,绕开 codebook,是唯一没有 quantization floor 的杠杆——
这就是 M1 11.75cm → M2/M3 1.10-1.29cm 这一 10× 落差的机制解释。
所以 M1 的判定只看"KPS 是否已比 no-control(~63cm)低一个量级"(11.75cm ✓),
最终数字看 M2/M3。

### 7.3 Eval 侧陷阱(全部踩过)

- **pitfall #2, `type() is` 路由 bug**:`utils/eval_t2m.py:1082` 的
  `type(x) is ControlTransformer` 把 KV **子类**静默路由到 base no-control 路径
  (现已修复:isinstance 在 L1086,:1082 是历史 bug 位置、现为修复注释行)。
  症状指纹:**不同 ckpt 的三次 eval 输出 4 位小数逐位相同**(FID 0.1455 / KPS 63.05)
  = 你以为在评的模型根本没被调用。修复 `isinstance()`。
  通用启发式:**identical eval outputs across different ckpts ⇒ 被评对象不对**。
- **pitfall #8, `pred_num_batch` 语义**:它是"每次 generate 调用累积 N 个 loader batch"
  而**不是**"总共处理 N 个 batch"。设 99999 → 静默跳过全部 generation →
  `torch.cat` 空列表在 motion_annotation_list 处炸。用 16(= 16×bs32 = 512 samples/call,
  脚本 L227-230 有注释, 2026-07 修正后行号)。
- **pitfall #15, 协议参数漂移**:time_steps / cond_scale / each_iter / ttt_dynamic /
  last_iter 每个协议 pin 死并 assert;默认值静默漂移 = 数字不可比。
- **pitfall #14 + #1(eval 侧)**:ckpt load 时 strip `module.`(脚本 L198-200, 2026-07 修正后行号),
  `load_state_dict(strict=False)` 后对非 `clip_model.` 的 missing keys 告警。
  ⚠ **`strict=False` 不会"忽略"ckpt 里泄漏的 `vq_model.*` keys**——模型本身有同名
  `vq_model` 子模块,名字能匹配上,`strict=False` 会**照常加载它们、静默覆盖 pristine RVQ**
  (strict=False 只忽略**匹配不上**的 keys)。实测 v18_ep6000 snapshot 就带 72 个 `vq_model.*` keys
  (本 port 训练侧并未 strip)。所以 eval 必须在 load 前显式过滤:
  `raw_sd = {k: v for k, v in raw_sd.items() if not k.startswith("vq_model.")}`,
  外加与官方 RVQ ckpt 的 byte-equal preflight。真正被 strict=False 丢弃的只有
  匹配不上的旧 ControlNet keys(`seqTransEncoder_control.*` 等,已被 delattr)。
- 统计口径:5-rep, mean ± 95% CI = std × 1.96 / √N;seed=3407(L95, L120)。

---

## Step 8 — 判定标准: 什么算 port 成功

四条 gate,对 M2/M3 结果判:

1. **保真 gate**: Top3 / Diversity / skate 应 **match substrate 自己的 baseline**
   (Top3 ~0.80 / Div ~9.5 / skate ~0.05)。
   实测: Top3 0.789-0.797 ✓, Div 9.578-9.700 ✓, skate 0.046-0.047 ✓。
2. **控制 gate**: KPS 比 no-control **降一个数量级以上**。
   实测: 63cm → 1.10 (M2) / 1.29 (M3),≈50× ✓。
3. **FID 代价 gate**: FID ≤ 2× no-control baseline。
   实测: M3 0.0875 vs 0.1455(dispatch-bug evals 实测的 no-control-equivalent 值)——
   不但 <2×,还**更好** ✓(KV 注入 + TTT 有正则效应)。
4. **协议内一致性**: M1→M2/M3 的落差应与 quantization-floor 解释一致
   (Stage-2 是唯一无 floor 的杠杆);若 M2/M3 没有相对 M1 的大幅 KPS 改善,
   查 Stage-2 是否真的在跑(last_iter 传没传进去,pitfall #15)。

**校准预期**:port 的绝对 KPS 不需要追平 co-designed substrate
(v4 M3 = 0.40cm vs MaskControl port M3 = 1.29cm)。port 的 claim 是
**mechanism hot-swap 成立**——同一 9.6M adapter 配方在 alien substrate 上
把 63cm 打到 1.29cm,且不伤 Top3/Div/skate。这就是 SUCCESS。

反之,出现 MoMask 型指纹(FID >0.5、KPS 只降到 30cm 量级、base-only decode 反而
比 full pipeline 好)→ 回到 Step 3/Step 4:大概率是 decode loop fork(#12)、
static err(#13)、下游 OOD(#11)、或 codebook 污染(#1)中至少一个。

---

## 附: 一页速查 (checklist)

```
[ ] Step 0  recon note 四件套答完 (attention 形态 / decode 入口函数名 / closed-loop cond / VQ 结构+EMA?)
[ ] Step 1  决策树走完: subclass 对象 + 复用/替换/override 清单
[ ] Step 2  delattr 旧机制; kv_down ZERO-init + up standard; bias=-5; hasattr guard;
            _base_trans_forward 手写不 delegate; vq_model.eval() + ckpt strip (EMA 时)
[ ] Step 3  decode loop 逐字复用, 只 override trans_forward; err 通道每 iter 重算
[ ] Step 4  Tier1 shape+无NaN (torch.equal 对照为可选加强) / Tier2 near-identity / Tier3 kv_down+bias+q_gates grad≠0 (kv_up/encoder/q_down/q_up init 时零梯度是正常的) / Tier4 base+codebook hash 不变
[ ] Step 5  codex xhigh review, brief 点名: Q-residual 污染路径证明 + best_* tracker 生命周期
[ ] Step 6  A/B-clean config; DDP: exist_ok / is_continue prefix / module. strip / TB try-except
[ ] Step 7  frozen snapshot → M1 5-rep 快筛 → M2/M3 确认; 警惕 type() is / pred_num_batch / 协议 pin
[ ] Step 8  四 gate: Top3-Div-skate match / KPS ≥10× 降 / FID <2× / M1→M3 落差符合 floor 解释
```

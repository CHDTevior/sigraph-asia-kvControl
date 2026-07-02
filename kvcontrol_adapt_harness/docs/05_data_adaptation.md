# 05 — 数据适配指南：当目标数据不是 HumanML3D / 不是 263 维

> KVControl Adapt Harness 文档 5/N。
> 面向场景：把 KV-Control 移植到一个**新数据集 / 新骨架**（用户明确预警：之后的数据迁移很可能不是 263 维）。
> 核心结论先行：**263 是 HumanML3D 的 dataset-specific 特征布局，不是 KV-Control 方法的一部分**。方法本身只依赖"feature → joints 可微"这一条链路性质；其余全是可替换的适配点。

---

## 1. 263 维到底是什么 — 拆解与 KIT 251 对照

263 = HumanML3D 的 redundant RIC (rotation-invariant coordinates) 特征。通用公式（J = joints_num）：

```
dim_pose = 4 + (J-1)*3 + (J-1)*6 + J*3 + 4
           │   │          │          │      └─ foot contacts (4, 双脚×前后脚掌)
           │   │          │          └─ local velocity, 全部 J 个 joint × 3
           │   │          └─ local rotation (6D cont. rep), 非 root 的 J-1 个 joint × 6
           │   └─ local position, 非 root 的 J-1 个 joint × 3
           └─ root: rot_vel(1) + lin_vel_xz(2) + height(1)
```

| 数据集 | joints_num | 代入 | dim_pose |
|---|---|---|---|
| HumanML3D (t2m) | 22 | 4 + 21×3 + 21×6 + 22×3 + 4 = 4+63+126+66+4 | **263** |
| KIT-ML | 21 | 4 + 20×3 + 20×6 + 21×3 + 4 = 4+60+120+63+4 | **251** |

代码证据：`<repo-root>/utils/get_opt.py:58-76` 里两个 dataset 分支硬编码了这组数（t2m: `joints_num=22`, `dim_pose=263` @ L62-63；kit: `joints_num=21`, `dim_pose=251` @ L71-72）。**同一份代码、同一个公式、换 J 就换维度** —— 这就是"dataset-specific 不是 method-specific"的直接证明：KV adapter 本身（kv_down/kv_up/ctrl_attn_bias/Q-residual，见 `models/mask_transformer/control_transformer_t_concat_v8_kv_v4.py:289-322`）没有任何一行依赖 263。

特征切片的隐式契约在 `utils/motion_process.py:400-416`（`recover_from_ric`）：
- `data[..., :4]` → root 四通道（由 `recover_root_rot_pos`, L361 消费）
- `data[..., 4:(joints_num-1)*3+4]` → local positions（L402）
- rotation 6D 和 velocity 段在 RIC 反解 joints 时**不被使用**（只用 root + local pos），foot contacts 是最后 4 维 `data[..., -4:]`
- 注意：velocity 段是 **J×3（含 root）**，不是 (J-1)×3 —— 手推新数据集维度时最容易错在这。

## 2. 换数据集的 6 个必改点清单

| # | 必改点 | 当前值 (HumanML3D) | 代码锚点 | 改法 |
|---|---|---|---|---|
| 1 | `dim_pose` | 263 | `utils/get_opt.py:63`；eval 脚本 `scripts/eval_maskcontrol_kv.py` L44-74 的 `load_rvqvae` **强制** `dim_pose=263` | 新增 dataset 分支；eval 脚本里的 hard-force 必须同步改，否则 VQ 输入维度 silent mismatch |
| 2 | `joints_num` | 22 | `utils/get_opt.py:62`；`control_transformer_t_concat_v8_kv_v4.py:374` (`recover_from_ric(pred, self.opt.joints_num)`) | 同上；FK 与 KPS 计算全走这个数 |
| 3 | `recover_from_ric` (FK!) | HumanML3D 树专用 | `utils/motion_process.py:400` | **不能沿用**——它假设 RIC 布局 + Y-up + root-XZ 平移 + 特定 joint 顺序。新骨架必须移植一个可微等价物，见 §3 |
| 4 | mean/std 归一化文件 | `<repo-root>/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy` + `std.npy` | v8_kv_v4 的 `_motion_to_joints` (L372-374): `pred*std+mean` 后才 FK | 新数据集重算；denorm 错 = FK 输出的 joints 全错 = err channel 全错（闭环污染） |
| 5 | foot contact 通道 | 最后 4 维 | skate 指标与部分后处理消费 | 新骨架脚 joint id 不同 → contact 检测阈值/通道数都要重定义；没有就砍掉这 4 维并同步 dim_pose 公式 |
| 6 | evaluator ckpt | `Comp_v6_KLD005` (text_mot_match) | `scripts/eval_maskcontrol_kv.py` L98-102 (`--eval_wrapper_opt`), L126-128 (`EvaluatorModelWrapper`) | **dataset-specific**。新数据集没有对应 evaluator → FID/R-prec/MatchScore 无意义，见 §6 |

另有一个隐藏第 7 点：**VQ tokenizer 本身**。换数据 = 换 VQ-VAE（重训或已有），soft-decode 用的 `codebook` 与 frozen decoder 全换，见 §7。

## 3. 关键：可微 FK 是 load-bearing 的

soft-decode 控制链的完整梯度路径（训练与 Stage-1 TTT 共用）：

```
L_traj → FK(recover_from_ric) → frozen VQ decoder → denorm → soft emb (softmax@codebook) → logits → transformer → KV adapter
```

其中 FK 这一环在 `control_transformer_t_concat_v8_kv_v4.py`：
- 训练闭环：L604 (`gt_joints = recover_from_ric(...)`) → L638 (`_build_ctrl_cond`, L376-387，(residual, absolute) 双通道)
- 推理闭环：L785 每个 MaskGIT step 重建 ctrl_cond；L837 TTT 内 `pred_joints_ttt = recover_from_ric(...)`；L944 Stage-2 同理

**如果新数据的 feature→joints 映射不可微（或有 in-place / numpy / detach 断点），整个 soft-decode 控制机制直接失效** —— 训练时 ctrl loss 无梯度回传，TTT 时 Adam 拿不到 logits 的梯度。这是 MoMask 移植教训 #13（static err channel）的孪生兄弟：断掉的不是"值"而是"导数"。

移植顺序铁律：**先移植可微 FK，先单测梯度，再动模型代码**。单测模板：

```python
import torch
from new_dataset.motion_process import recover_from_ric_newskel  # 你的新 FK

def test_fk_differentiable():
    J, D = 25, 287                      # 新骨架的 joints_num / dim_pose
    feat = torch.randn(2, 196, D, dtype=torch.float64, requires_grad=True)
    joints = recover_from_ric_newskel(feat, J)   # (2, 196, J, 3)
    assert joints.requires_grad, "FK output detached — check .numpy()/detach()/in-place ops"

    loss = joints.square().mean()
    loss.backward()
    g = feat.grad
    assert g is not None
    assert torch.isfinite(g).all(), "NaN/Inf in FK gradient"
    assert g.abs().sum() > 0, "all-zero gradient — dead FK branch"

    # 每个输出 joint 的 xyz 都应对某些输入通道敏感（防 silent slice 错位）
    per_channel = g.abs().sum(dim=(0, 1))        # (D,)
    used = (per_channel > 0).sum().item()
    print(f"FK consumes {used}/{D} feature channels")
    # HumanML3D 参考: RIC 反解只用 root(4) + local pos((J-1)*3), rot/vel 段梯度为 0 是正常的

def test_fk_gradcheck():
    feat = torch.randn(1, 8, 287, dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(lambda x: recover_from_ric_newskel(x, 25), (feat,), eps=1e-6)
```

注意点（都是 `utils/motion_process.py:400-416` 里真实存在的模式）：`positions[..., 0] += r_pos[..., 0:1]` 这类 in-place add 在 autograd 下是安全的，但如果新实现里对 **leaf 或被多次使用的 tensor** 做 in-place 会炸 backward；quaternion 归一化处除零要加 eps；`float()` cast 保留梯度、`.numpy()` 不保留。

## 4. Control encoder 输入维度 C 的推导

公式：**C = n_ctrl_joints × 2 × 3**（每个受控 joint：err 通道 3 + abs 目标通道 3）。

代码锚点 `control_transformer_t_concat_v8_kv_v4.py:271-274`：

```python
input_emb_width = (
    6 if self.control == "trajectory"
    else len(control_joint_ids) * 2 * 3
)
```

双通道语义在 `_build_ctrl_cond` (L376-387)：`ctrlNet_cond = (gt - pred) * mask`（residual，闭环误差）与 `ctrlNet_cond2 = gt * mask`（absolute 目标），`trajectory` 分支只切 joint 0（pelvis）。**顺序是 (residual, absolute)，不能颠倒** —— ckpt 权重按这个通道顺序训练。

| 控制集 | n_ctrl_joints | C |
|---|---|---|
| pelvis trajectory（paper 主线） | 1 | **6** |
| pelvis + 双手 (wrists) | 3 | **18** |
| 全身 22 joints (HumanML3D) | 22 | **132** |
| 新骨架任意 joint 子集 | n | 6n |

唯一前提：err 通道必须能从 decoded joints 算出来 —— 也就是 §3 的可微 FK 输出必须覆盖你选的 control joints。任何 joint 子集都合法。

## 5. Tokenizer 率适配：unit_length 与 encoder_control 的 stride 链

契约：**T_ctrl = T_frames / unit_length**。control tokens 必须与 motion tokens 一一对齐（KV 注入按 token 拼接，`_encode_ctrl_to_kv` 输出的 T_ctrl 直接 cat 到每层 K/V 的 seq 轴，参照 `references/MaskControl/models/mask_transformer/control_transformer_kv.py:268-279` 与 v4 的 L107-113 注入循环）。

当前值：`unit_length=4`（`options/base_option.py:26`，`utils/get_opt.py:78-79` fallback 同值）→ 196 frames → 49 tokens。

encoder_control 的降采样由 `models/vq/encdec.py:5-34` 的 `Encoder` 决定：`down_t` 个 stage，每个 stage 一个 stride 为 `stride_t` 的 Conv1d（L23-28，kernel `filter_t = stride_t*2`, pad `stride_t//2`）。总降采样：

```
downsample_factor = stride_t ** down_t        必须满足  stride_t ** down_t == unit_length
```

当前配置 `down_t=2, stride_t=2`（`control_transformer_t_concat_v8_kv_v4.py:278-279`）→ 2² = 4 ✓。

新 substrate 换率的改法：

| 新 unit_length | (stride_t, down_t) 解 | 备注 |
|---|---|---|
| 2 | (2, 1) | |
| 4 | (2, 2) | 当前 |
| 8 | (2, 3) | 加一层 stage，参数量↑ ~1 个 Conv+Resnet1D block |
| 5 / 非 2 幂 | 无整数解 | 用 (5,1) 单层 stride-5，或 encoder 后接 `F.interpolate`/avg-pool 对齐——**务必 assert `ctrl_feat.shape[t] == motion_tokens.shape[t]`**，静默广播错位不会报错但会毁掉对齐 |

fail-loud 建议：在 `_encode_ctrl_to_kv` 入口加 `assert T_ctrl == T_motion_tokens, f"{T_ctrl} vs {T_motion_tokens}: encoder stride chain does not match tokenizer rate"`（pitfall 精神同 R12）。

## 6. 评估指标可移植性分级

| 级别 | 指标 | 依赖 | 新数据集可用性 |
|---|---|---|---|
| **A：纯几何，永远可移植** | KPS (keyframe position score)、traj_fail、skate ratio | 只需 decoded joints + GT joints（米制） | ✅ 无条件可报。这是新数据集上**唯一开箱即用的成功度量** |
| **B：需要 dataset-specific evaluator** | FID、R-precision (Top1/2/3)、MatchingScore、Diversity、MModality | text_mot_match 特征提取器（HumanML3D = `Comp_v6_KLD005`，`scripts/eval_maskcontrol_kv.py` L98-102/L126-128 加载） | ❌ 没有对应 evaluator ckpt 就**不能报** —— 用错 evaluator 算出来的 FID 是无意义数字，不是"近似值" |

实践规则：
1. 移植初期只用 A 级指标 gate 训练（本项目 KPS 63cm→1.29cm 的全部中间判断都可以只靠 KPS + 可视化完成；MaskControl port 的 M1/M2/M3 表里 KPS 才是控制机制的直接证据，FID 只证明"没有破坏生成质量"）。
2. 要报 B 级 → 两条路：目标数据集社区已有 evaluator（KIT 有官方 Comp_v6 变体）；或者自己训一个 text-motion matching evaluator（成本约等于再训一个 base 模型，立项前先确认值不值）。
3. 记住 pitfall #15：即便指标可用，M0/M1/M2/M3 的 `time_steps/cond_scale/each_iter/ttt_dynamic/last_iter` 必须 pin 死并 assert，否则数字不可比。
4. **可视化 demo 优先于一切数字**（跨项目铁律）：新数据集连 KPS 都可能因 FK/denorm bug 而"虚好"，多帧 GT-vs-pred 并排动画是第一道 gate。

## 7. Codebook 结构侦察：soft-decode 该 mix 哪个 codebook

soft-decode 核心操作：`emb = softmax(logits) @ codebook[0]`（期望嵌入，凸组合，全程可微）。**换 substrate 前必须先侦察它的量化器结构** —— 三种已知形态，对应三种处理：

| 形态 | 实例 | soft-decode 处理 | 风险 |
|---|---|---|---|
| 单码本 / 只用 base 层 | MaskControl（RVQ `rvq_nq6...` 但 mask transformer 只操作 base quantizer tokens） | mix `codebook[0]`，decoder 输入 = base 层期望嵌入 | 最低。MaskControl port 3 天成功的结构性原因之一 |
| RVQ 残差级联 | MoMask（6 quantizers = base + 5 residual） | mask transformer 只管 base 层 → soft-decode 仍 mix `codebook[0]`；但 residual 层由 **ResidualTransformer** 补 | **pitfall #11**：RT 在 vanilla token 分布上训练，KV 修改后的 base tokens 对它是 OOD → RT 加噪。实测 base-only decode 反而更好 (FID 0.715→0.581)。默认策略：**先 base-only decode 出结果，RT-on 作为 ablation 单独验证**，不要默认打开 |
| multi-part 多码本 | v4 substrate（PartVQ，Q=6 个 part codebooks 沿 seq 轴 unpack，~295 tokens） | 每个 seq 位置属于哪个 part 是确定的 → 按位置 mix 对应 part 的 codebook；FK 前要按 part 重组特征 | 中等。对齐逻辑复杂，但每个码本内仍是标准 softmax-mix |

侦察 checklist（动手前跑一遍）：
1. `state_dict` 里数一下 `quantizer`/`codebook` 相关 key 的数量与形状 → 判断 nq 与 code_dim。
2. 确认 mask transformer 的 logits 词表大小 == 哪个 codebook 的 code 数（+mask/pad token 偏移）。
3. 确认 decoder 的输入是"base 层嵌入"还是"所有层嵌入之和" —— 决定 soft emb 送进 decoder 前要不要补 residual 层（以及 pitfall #11 的暴露面）。
4. **EMA 检查**（pitfall #1）：codebook 若是 EMA buffer（非 Parameter），`requires_grad=False` 不保护它 —— train() 模式下 forward 就会污染。必须 `vq_model.eval()` 常驻 + ckpt 保存时 strip `vq_model.*` + eval 侧 byte-equal preflight。

另注意 Stage-2（`last_iter`）**天然绕过 codebook**（直接优化连续 embedding），所以它是唯一没有量化地板的杠杆（M1 11.75cm → M2/M3 1.1-1.3cm 的 10× 差距来源）——这一条在任何 codebook 形态下都成立，是换 substrate 时最稳的保底手段。

---

## 附：新数据集移植的最小执行顺序（把本文压成 checklist）

1. 写公式算出新 `dim_pose`（§1），在 `utils/get_opt.py` 加 dataset 分支（§2 #1-2）。
2. 移植可微 FK + 跑 §3 两个单测（gradcheck 用 float64）。**过不了不许往下走。**
3. 重算 mean/std（§2 #4）；确定 foot contact 有无（§2 #5）。
4. 侦察 substrate codebook 结构（§7 checklist 4 条），确定 soft-decode mix 目标。
5. 选 control joints → 算 C（§4），确认 unit_length → 配 stride 链 + 对齐 assert（§5）。
6. 只用 KPS + GT-vs-pred 可视化 gate 前期训练（§6）；evaluator 问题立项时单独决策。
7. 全程对照 pitfall catalog（`docs/` 内 pitfall 篇）：#1 EMA、#11 RT-OOD、#12 复用 substrate decode loop、#13 闭环 err channel 是与数据适配交互最强的四条。

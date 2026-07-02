"""KV-Control adapter TEMPLATE — 把 KV-Control 机制移植到新 substrate 的骨架代码。

改编自 (已验证可工作, KPS 63cm→1.29cm on MaskControl port):
    <repo-root>/references/MaskControl/models/mask_transformer/control_transformer_kv.py
v4 原始参考 (paper headline, M3 KPS 0.40cm):
    <repo-root>/models/mask_transformer/control_transformer_t_concat_v8_kv_v4.py

═══════════════════════════════════════════════════════════════════════════
本模板守护的 pitfalls (编号对应 harness pitfall catalog):
─────────────────────────────────────────────────────────────────────────
  P3  Q-residual K/V contamination — q_delta 绝不能在 F.linear(x, in_proj_weight)
      之前加到 x 上, 否则通过共享 in_proj 泄漏进 K_base/V_base。本模板的
      _layer_with_kv_injection 已内置修复: qkv 从未修改的 x 计算,
      q_delta 单独过 W_Q = in_proj_weight[:D] 只加到 q。
      症状: init 时 (gate=0) 完全静默; gate 打开后 frozen backbone 语义悄悄漂移。
  P12 手写 substrate decode loop (MoMask port 根因失败, FID 0.581) —
      本模板 ONLY 提供 trans_forward 覆盖 + per-layer 注入;
      generate()/decode loop 必须 VERBATIM 复用 substrate 自己的实现。
      移植第一铁律: 只 override per-layer forward, 别碰采样循环。
  P13 static err channel — ctrlNet_cond 的 residual 通道必须每个 training batch
      / 每个 inference iteration 从当前 prediction 重算 (closed-loop)。
      见文件底部 build_ctrl_cond() 参考实现。
  (dead-branch trap) kv_down 零初始化 + kv_up 标准初始化 = identity at init
      且梯度存活。两者都零 → 梯度全死 (v4 历史踩过)。q_down/q_up 同理:
      标准初始化, q_gates=0 负责 identity。绝不把 q_down 也零初始化。
═══════════════════════════════════════════════════════════════════════════

Init invariants (paper-faithful near-identity at init):
- kv_down[i].weight = 0        → K_ctrl_i = V_ctrl_i = 0 at init
- kv_up_k/kv_up_v 标准 init    → 梯度经 kv_down 流动 (zero-down + std-up = live gradients)
- ctrl_attn_bias[i] = -5.0     → 每 ctrl token 相对单个 base token 权重 e^-5≈0.0067;
  S=50/T_ctrl=49 下 per-ctrl-token 归一化质量 ≈1.3e-4, 总 ctrl mass ≈0.65% (<1%) → near-identity at init
- q_gates = 0, q_down/q_up 标准 init → Q-residual 贡献为 0 at init, 梯度存活
- Q-residual 只作用于 upper-half layers (layer >= num_layers // 2)

Frozen vs trainable (MaskControl port 实测 9.6M trainable):
- Frozen:    整个 base transformer (从 trans_path 加载)
- Trainable: encoder_control (~9.0M) + kv_down/kv_up_k/kv_up_v (~0.59M)
             + ctrl_attn_bias (num_layers 个标量) [+ q_down/q_up/q_gates ~0.05M]
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# TODO(substrate): 换成目标 substrate 的 base transformer 类。
#   - MaskControl port 用的是 `ControlTransformer` (subclass 后 delattr 其
#     ControlNet 分支); 若目标 substrate 没有现成 control 子类, 直接
#     subclass 其 MaskTransformer / base transformer。
#   - freeze_block/unfreeze_block 只是 requires_grad_(False/True) 的循环,
#     substrate 没有的话在下面自己定义。
# ============================================================================
# from <substrate>.models import BaseTransformer  # TODO(substrate)


def freeze_block(block: nn.Module):
    for p in block.parameters():
        p.requires_grad_(False)


def unfreeze_block(block: nn.Module):
    for p in block.parameters():
        p.requires_grad_(True)


def _layer_with_kv_injection(
    layer: nn.TransformerEncoderLayer,
    x: torch.Tensor,
    K_ctrl: torch.Tensor,
    V_ctrl: torch.Tensor,
    ctrl_bias_scalar: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
    q_residual: Optional[Tuple[nn.Linear, nn.Linear, torch.Tensor]] = None,
) -> torch.Tensor:
    """单层 TransformerEncoderLayer + K/V 增广注入。

    注意: 本实现假设 substrate 的 layer 是 PyTorch 标准 post-norm
    nn.TransformerEncoderLayer (self_attn + norm1 + linear1/2 + norm2)。
    TODO(substrate): 如果目标 substrate 用 pre-norm / 自定义 block /
    factorized attention (如 v4 的 Q_per_tok), 必须按其真实 forward
    逐行镜像重写残差路径 — 参见 v4 版本
    control_transformer_t_concat_v8_kv_v4.py:64-192 的 pre/post-norm 双路径。

    Inputs:
        layer            — nn.TransformerEncoderLayer (post-norm)
        x                — (S, B, D) motion query stream
        K_ctrl, V_ctrl   — (T_ctrl, B, D) per-layer control K/V residual (init 时为 0)
        ctrl_bias_scalar — 标量 Parameter, pre-softmax 加在 ctrl-token attention 列上
        key_padding_mask — (B, S) bool, True = padded (ctrl token 永远不 pad)
        q_residual       — optional (q_down, q_up, q_gate), 本层的 Q-residual
    """
    S, B, D = x.shape
    T_ctrl = K_ctrl.shape[0]
    attn = layer.self_attn
    num_heads = attn.num_heads
    head_dim = D // num_heads

    # ------------------------------------------------------------------
    # PITFALL P3 修复 (关键, 勿改): base Q/K/V 从「未修改的 x」计算。
    # 如果先把 q_delta 加进 x 再过 in_proj, residual 会经共享权重泄漏进
    # K_base/V_base — init 时 (gate=0) 完全静默, 训练后期悄悄破坏 frozen
    # backbone 语义。
    # ------------------------------------------------------------------
    qkv_base = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)  # (S, B, 3D)
    q, k_base, v_base = qkv_base.chunk(3, dim=-1)                    # each (S, B, D)

    # Q-residual: 只作用于 Q stream。q_delta 单独经 W_Q (in_proj 的 Q 切片)
    # 投影后加到 q; k_base/v_base 保持不动。init 时 q_gate=0 → 贡献为 0。
    if q_residual is not None:
        q_down, q_up, q_gate = q_residual
        q_delta = q_up(q_down(x)) * q_gate                           # (S, B, D)
        # in_proj_weight 布局为 [W_Q; W_K; W_V] 按行 (PyTorch MHA 约定)
        W_Q = attn.in_proj_weight[:D]                                 # (D, D)
        q = q + F.linear(q_delta, W_Q)

    # Multi-head reshape: (seq, batch, D) → (batch*heads, seq, head_dim)
    q = q.contiguous().view(S, B * num_heads, head_dim).transpose(0, 1)
    k_base = k_base.contiguous().view(S, B * num_heads, head_dim).transpose(0, 1)
    v_base = v_base.contiguous().view(S, B * num_heads, head_dim).transpose(0, 1)
    k_ctrl = K_ctrl.contiguous().view(T_ctrl, B * num_heads, head_dim).transpose(0, 1)
    v_ctrl = V_ctrl.contiguous().view(T_ctrl, B * num_heads, head_dim).transpose(0, 1)

    # KV-Control 核心: 沿 kv-seq 轴拼接
    k_aug = torch.cat([k_base, k_ctrl], dim=1)  # (B*H, S + T_ctrl, head_dim)
    v_aug = torch.cat([v_base, v_ctrl], dim=1)

    # Additive attention mask: ctrl 列加 learnable 标量 bias (init -5.0)
    attn_mask = q.new_zeros(B * num_heads, S, S + T_ctrl)
    attn_mask[:, :, S:] = ctrl_bias_scalar

    if key_padding_mask is not None:
        # (B, S), True = padded → -inf; 只作用于 base-token 切片 [:, :, :S]
        kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)             # (B, 1, 1, S)
        kpm = kpm.expand(-1, num_heads, -1, -1).reshape(B * num_heads, 1, S)
        attn_mask[:, :, :S] = attn_mask[:, :, :S].masked_fill(kpm, float("-inf"))

    attn_out = F.scaled_dot_product_attention(
        q, k_aug, v_aug,
        attn_mask=attn_mask,
        dropout_p=attn.dropout if attn.training else 0.0,
    )  # (B*H, S, head_dim)

    attn_out = attn_out.transpose(0, 1).contiguous().view(S, B, D)
    attn_out = attn.out_proj(attn_out)

    # 标准 post-norm 残差 + FF (镜像 nn.TransformerEncoderLayer.forward)
    # TODO(substrate): pre-norm substrate 须重排此段 (norm 前置)
    src = x + layer.dropout1(attn_out)
    src = layer.norm1(src)
    src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
    src = src + layer.dropout2(src2)
    src = layer.norm2(src)
    return src


# TODO(substrate): 把 `BaseTransformer` 换成目标 substrate 的类
class KVControlAdapter(nn.Module):  # TODO(substrate): class KVControlAdapter(BaseTransformer)
    """KV-Control adapter on <SUBSTRATE>。

    移植操作序 (MaskControl port 验证过的 3 天路径):
      1. subclass substrate 的 transformer
      2. 若 parent 自带 control 分支 (ControlNet 等) → delattr 掉
      3. 加 KV 模块 (本 __init__)
      4. override ctrl_train / ctrl_eval / trans_forward + base fallback
      5. 【P12】substrate 的 generate()/decode/TTT 循环 VERBATIM 复用, 一行不改
    """

    def __init__(
        self,
        # ... TODO(substrate): 透传 parent 全部构造参数 (code_dim, cond_mode,
        #     latent_dim, ff_size, num_layers, num_heads, dropout, trans_path,
        #     vq_model, control, ...)
        latent_dim=384,        # TODO(substrate): 必须 = base transformer 的 D
        num_layers=8,          # TODO(substrate): 必须 = base transformer 层数
        # KV-Control specific (paper 配置, 一般不用改)
        kv_rank=64,
        ctrl_attn_bias_init=-5.0,
        use_q_residual=True,
        **kargs,
    ):
        super().__init__()  # TODO(substrate): super().__init__(<parent 全部参数>)

        # ------------------------------------------------------------------
        # TODO(substrate): 若 parent 是 ControlNet 类, 删掉其 control 分支,
        # 换成 KV 模块 (MaskControl port: control_transformer_kv.py:174-176):
        #   del self.seqTransEncoder_control
        #   del self.first_zero_linear
        #   del self.mid_zero_linear
        # 注意: delattr 之后 parent 的 ctrl_train() 会 crash → 必须 override (见下)。
        # ------------------------------------------------------------------

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.kv_rank = int(kv_rank)
        self.ctrl_attn_bias_init = float(ctrl_attn_bias_init)
        self.use_q_residual = bool(use_q_residual)

        D = self.latent_dim
        r = self.kv_rank

        # ==================================================================
        # Control encoder: 帧级控制信号 → token 级特征
        # TODO(substrate): ctrl_in_dim C = n_ctrl_joints × 2 × 3
        #   (每个受控 joint 两个通道: err residual + absolute target, 各 3 维 xyz)
        #   trajectory (pelvis-only) → C = 6; k 个 joint → C = 6k。
        # TODO(substrate): 下采样率必须匹配 substrate tokenizer 的 unit_length!
        #   HumanML3D 系: unit_length=4 → down_t=2, stride_t=2 (196 帧 → 49 token)。
        #   tokenizer 率不同 → 改 stride 链, 保证 T_ctrl == T_tokens。
        # MaskControl port 用的是 substrate 自带的 Conv1D encoder
        # (C → 512 → D, ~9.0M params); 复用 substrate 的实现最稳。
        # ==================================================================
        ctrl_in_dim = 6  # TODO(substrate): n_ctrl_joints * 2 * 3
        self.encoder_control = None  # TODO(substrate): 复用/构造 Conv1D encoder,
        #   输入 (B, C, T_frames) → 输出 (B, D, T_ctrl)

        # Per-layer low-rank K/V projections: D → r → D
        self.kv_down = nn.ModuleList([nn.Linear(D, r, bias=False) for _ in range(self.num_layers)])
        self.kv_up_k = nn.ModuleList([nn.Linear(r, D, bias=False) for _ in range(self.num_layers)])
        self.kv_up_v = nn.ModuleList([nn.Linear(r, D, bias=False) for _ in range(self.num_layers)])
        # 【dead-branch trap】kv_down 零初始化 → K_ctrl=V_ctrl=0 at init;
        # kv_up 保持默认 Kaiming init → 梯度存活。两者都零 = 梯度全死, 勿改!
        for m in self.kv_down:
            nn.init.zeros_(m.weight)

        # Per-layer learnable 标量 bias, init -5.0 → ctrl token 相对权重 e^-5≈0.0067
        # (总 ctrl attn mass <1% at S=50/T_ctrl=49, base 输出缩放 ≈0.9935)
        self.ctrl_attn_bias = nn.ParameterList([
            nn.Parameter(torch.full((1,), self.ctrl_attn_bias_init))
            for _ in range(self.num_layers)
        ])

        # Q-residual: SHARED q_down/q_up + per-upper-layer 标量 gate (paper headline)
        if self.use_q_residual:
            self.q_start_layer = self.num_layers // 2
            n_q = self.num_layers - self.q_start_layer
            # 标准 init, 绝不零初始化 (q_gates=0 已保证 identity; 双零 = dead branch)
            self.q_down = nn.Linear(D, r, bias=False)
            self.q_up = nn.Linear(r, D, bias=False)
            self.q_gates = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(n_q)])
        else:
            self.q_start_layer = self.num_layers  # disabled

        # adapter 建好后重算 freeze/unfreeze
        self.ctrl_train()
        self._param_counts = self._compute_param_counts()

    # --- Lifecycle / freezing -------------------------------------------------

    def ctrl_train(self):
        """Override parent: parent.ctrl_train 会 unfreeze 已被 delattr 的
        ControlNet 分支 → crash。重现其意图: freeze 全部, unfreeze adapter。"""
        if not hasattr(self, "kv_down"):
            # parent.__init__ 内部调用时 adapter 尚不存在 — 安全 no-op,
            # 我们在自己 __init__ 末尾再调一次。
            return
        freeze_block(self)
        if self.encoder_control is not None:
            unfreeze_block(self.encoder_control)
        for m in self.kv_down:
            unfreeze_block(m)
        for m in self.kv_up_k:
            unfreeze_block(m)
        for m in self.kv_up_v:
            unfreeze_block(m)
        for p in self.ctrl_attn_bias:
            p.requires_grad_(True)
        if self.use_q_residual:
            unfreeze_block(self.q_down)
            unfreeze_block(self.q_up)
            for p in self.q_gates:
                p.requires_grad_(True)
        # ------------------------------------------------------------------
        # PITFALL P1 提醒: 若 substrate 的 vq_model 挂在 self 上且用 EMA
        # codebook, requires_grad=False 保护不了 EMA BUFFER。必须在这里
        # (以及每次 model.train() 之后) 强制 self.vq_model.eval(),
        # 否则 codebook 被训练静默污染 → eval FID 22-25 量级垃圾。
        # TODO(substrate): if hasattr(self, "vq_model"): self.vq_model.eval()
        # ------------------------------------------------------------------

    def ctrl_eval(self):
        freeze_block(self)  # 安全: 不触碰已删除模块

    def _compute_param_counts(self):
        def num(p_iter):
            return sum(p.numel() for p in p_iter)
        cnt = {
            "kv_down_M": num(p for m in self.kv_down for p in m.parameters()) / 1e6,
            "kv_up_k_M": num(p for m in self.kv_up_k for p in m.parameters()) / 1e6,
            "kv_up_v_M": num(p for m in self.kv_up_v for p in m.parameters()) / 1e6,
            "ctrl_attn_bias_K": num(self.ctrl_attn_bias) / 1e3,
        }
        if self.encoder_control is not None:
            cnt["encoder_control_M"] = num(self.encoder_control.parameters()) / 1e6
        if self.use_q_residual:
            cnt["q_down_M"] = num(self.q_down.parameters()) / 1e6
            cnt["q_up_M"] = num(self.q_up.parameters()) / 1e6
            cnt["q_gates_K"] = num(self.q_gates) / 1e3
        cnt["total_trainable_M"] = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        return cnt

    # --- Core forward ---------------------------------------------------------

    def _encode_ctrl_to_kv(self, ctrlNet_cond: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """ctrlNet_cond: (B, T_frames, C) → per-layer (K_ctrl_i, V_ctrl_i), 各 (T_ctrl, B, D)。"""
        # encoder_control 期望 (B, C, T_frames), 输出 (B, D, T_ctrl)
        ctrl_feat = self.encoder_control(ctrlNet_cond.permute(0, 2, 1))
        ctrl_feat = ctrl_feat.permute(2, 0, 1)  # (T_ctrl, B, D)
        kv_list = []
        for i in range(self.num_layers):
            h = self.kv_down[i](ctrl_feat)   # (T_ctrl, B, r) — init 时为 0
            k_ctrl = self.kv_up_k[i](h)      # (T_ctrl, B, D) — init 时为 0
            v_ctrl = self.kv_up_v[i](h)
            kv_list.append((k_ctrl, v_ctrl))
        return kv_list

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False, ctrlNet_cond=None):
        """Per-layer forward, 唯一允许 override 的入口 (P12)。

        TODO(substrate): 前处理段 (token_emb / input_process / cond_emb /
        position_enc / padding_mask 前缀列) 必须逐行照抄 substrate 自己的
        trans_forward — 下面是 MaskControl 版本, 仅作结构参考。
        """
        # ctrl 未启用 / 无控制信号 → 纯 base 路径 (不能 delegate 给 parent,
        # parent 的实现会访问已删除的 ControlNet 分支)
        if ctrlNet_cond is None:
            return self._base_trans_forward(motion_ids, cond, padding_mask, force_mask)

        # -- TODO(substrate): 以下前处理照抄 substrate ----------------------
        cond = self.mask_cond(cond, force_mask=force_mask)
        x = self.token_emb(motion_ids) if len(motion_ids.shape) == 2 else motion_ids
        x = self.input_process(x)
        cond_tok = self.cond_emb(cond).unsqueeze(0)            # (1, B, D)
        x = self.position_enc(x)
        xseq = torch.cat([cond_tok, x], dim=0)                 # (S+1, B, D)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)
        # -------------------------------------------------------------------

        kv_list = self._encode_ctrl_to_kv(ctrlNet_cond)

        output = xseq
        for i, layer in enumerate(self.seqTransEncoder.layers):
            K_ctrl_i, V_ctrl_i = kv_list[i]
            bias_i = self.ctrl_attn_bias[i]
            if self.use_q_residual and i >= self.q_start_layer:
                q_resid = (self.q_down, self.q_up, self.q_gates[i - self.q_start_layer])
            else:
                q_resid = None
            output = _layer_with_kv_injection(
                layer, output, K_ctrl_i, V_ctrl_i, bias_i, padding_mask, q_resid
            )
        if self.seqTransEncoder.norm is not None:
            output = self.seqTransEncoder.norm(output)

        output = output[1:]                                    # 去掉 cond token
        logits = self.output_process(output)
        return logits

    def _base_trans_forward(self, motion_ids, cond, padding_mask, force_mask=False):
        """ctrlNet_cond=None fallback: 与 base transformer bit-exact 的前向。

        TODO(substrate): 逐行照抄 substrate 的 base trans_forward。
        不许 delegate 给 parent (parent 路径可能访问已删除模块)。
        smoke test Tier 1 会验证此路径与 frozen base bit-exact。
        """
        cond = self.mask_cond(cond, force_mask=force_mask)
        x = self.token_emb(motion_ids) if len(motion_ids.shape) == 2 else motion_ids
        x = self.input_process(x)
        cond_tok = self.cond_emb(cond).unsqueeze(0)
        x = self.position_enc(x)
        xseq = torch.cat([cond_tok, x], dim=0)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[1:]
        logits = self.output_process(output)
        return logits


# ============================================================================
# 参考: closed-loop 控制信号构造 (PITFALL P13)
# 出处: control_transformer_t_concat_v8_kv_v4.py:376-387 (_build_ctrl_cond)
# ============================================================================
def build_ctrl_cond(global_joint, pred_joints, global_joint_mask):
    """ctrlNet_cond = cat([(gt - pred) * mask, gt * mask], dim=-1)  — (residual, absolute) 顺序。

    P13 铁律: err 通道 (gt - pred) 必须在【每个 training batch】和【每个
    inference iteration】从当前 prediction 重算。喂 err=0 (static) 会彻底
    移除 iterative-correction 原理 — MoMask port 就是这样失败的。

    TODO(substrate): pred_joints 来自「可微 FK 路径」:
      soft-decode logits → expected embedding → frozen decoder → denorm
      → recover_from_ric (FK)。新数据集若 feature→joints 映射不是
      recover_from_ric, 先移植可微等价物并单测其梯度 (data notes 文档)。
    """
    err = (global_joint - pred_joints) * global_joint_mask.unsqueeze(-1)   # residual 通道
    tgt = global_joint * global_joint_mask.unsqueeze(-1)                    # absolute 通道
    # TODO(substrate): joint 子集切片 + reshape 到 (B, T_frames, C=n_joints*2*3)
    return torch.cat([err, tgt], dim=-1)

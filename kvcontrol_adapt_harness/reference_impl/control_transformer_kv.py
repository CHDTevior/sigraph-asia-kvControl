"""KV-Control on MaskControl substrate (KV-on-MaskControl port).

Subclasses MaskControl's `ControlTransformer` (control_transformer.py:46) and
swaps the ControlNet mechanism (deepcopy seqTransEncoder + zero-conv branches)
with per-layer K/V residual injection identical in spirit to v4 paper:

    forward_with_kv_context:
        K_aug = [K_base; K_ctrl_i],  V_aug = [V_base; V_ctrl_i]
        + additive ctrl_attn_bias[i] on ctrl-token columns (additive log-softmax bias)
        + (optional) Q-residual on upper-half layers via shared q_down/q_up + per-layer q_gates

References:
- v4 paper KV-Control: /scratch/ts1v23/workspace/MaskControl/models/mask_transformer/control_transformer_t_concat_v8_kv_v4.py
- v4 reference doc: /scratch/ts1v23/workspace/MaskControl/.codex-research/plan/v4_kv_control_reference_for_port.md
- codex final review (NEEDS_FIX 13 items) at tasks/wl4zqaeu9.output — all items addressed here.

Init invariants (paper-faithful near-identity at init):
- kv_down[i].weight = 0  → K_ctrl_i = V_ctrl_i = 0 at init
- kv_up_k[i] / kv_up_v[i] standard PyTorch Linear init → gradients flow during training
- ctrl_attn_bias[i] = -5.0 init (paper-parity) → softmax mass on ctrl ≈ 4e-3 per ctrl token
  at MaskControl's S_base=49+1, T_ctrl=49 (vs v4's S_base=295)
- q_gates init=0 → Q-residual contribution=0 at init (identity)
- q_down / q_up standard init (NEVER zero — would kill gradient through q_gates,
  same Q-residual dead-branch trap as in v4 history)

Frozen vs trainable:
- Frozen: entire base MaskTransformer (loaded from trans_path, inherited from parent)
- Trainable: encoder_control + kv_down/kv_up_k/kv_up_v + ctrl_attn_bias [+ q_down/q_up/q_gates if --use_q_residual]
- Parent's ctrl_train() unfreezes ControlNet branches we delattr'd — fully overridden below.
"""
import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mask_transformer.control_transformer import ControlTransformer, freeze_block, unfreeze_block


def _layer_with_kv_injection(
    layer: nn.TransformerEncoderLayer,
    x: torch.Tensor,
    K_ctrl: torch.Tensor,
    V_ctrl: torch.Tensor,
    ctrl_bias_scalar: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
    q_residual: Optional[Tuple[nn.Linear, nn.Linear, torch.Tensor]] = None,
) -> torch.Tensor:
    """Run a single TransformerEncoderLayer with K/V augmented by control residuals.

    Inputs:
        layer            — nn.TransformerEncoderLayer (post-norm, GELU)
        x                — (S, B, D) motion query stream
        K_ctrl, V_ctrl   — (T_ctrl, B, D) per-layer control K/V residuals (zero at init)
        ctrl_bias_scalar — scalar Parameter, additive bias on ctrl-token attention columns
        key_padding_mask — (B, S) bool, True = padded (ctrl tokens NEVER padded)
        q_residual       — optional (q_down, q_up, q_gate) for Q-residual injection on this layer

    Output:
        (S, B, D)
    """
    S, B, D = x.shape
    T_ctrl = K_ctrl.shape[0]
    attn = layer.self_attn
    num_heads = attn.num_heads
    head_dim = D // num_heads

    # Compute base Q, K, V from UNMODIFIED x via the layer's MultiheadAttention in_proj
    # (codex critical-bug fix #1: q-residual must NOT contaminate K_base / V_base via the
    # shared in_proj_weight chunked split; v4 reference modifies Q stream only).
    qkv_base = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)  # (S, B, 3D)
    q, k_base, v_base = qkv_base.chunk(3, dim=-1)                    # each (S, B, D)

    # Q-residual ONLY on the Q stream: project q_delta through the Q-slice of in_proj_weight,
    # add to q, leave k_base / v_base untouched. At init q_gate=0 so q_residual_proj=0.
    if q_residual is not None:
        q_down, q_up, q_gate = q_residual
        q_delta = q_up(q_down(x)) * q_gate                            # (S, B, D)
        # in_proj_weight is laid out as [W_Q; W_K; W_V] along rows (PyTorch MHA convention).
        W_Q = attn.in_proj_weight[:D]                                  # (D, D)
        q = q + F.linear(q_delta, W_Q)                                # (S, B, D)

    # Multi-head reshape: (seq, batch, D) → (batch*heads, seq, head_dim)
    q = q.contiguous().view(S, B * num_heads, head_dim).transpose(0, 1)
    k_base = k_base.contiguous().view(S, B * num_heads, head_dim).transpose(0, 1)
    v_base = v_base.contiguous().view(S, B * num_heads, head_dim).transpose(0, 1)
    k_ctrl = K_ctrl.contiguous().view(T_ctrl, B * num_heads, head_dim).transpose(0, 1)
    v_ctrl = V_ctrl.contiguous().view(T_ctrl, B * num_heads, head_dim).transpose(0, 1)

    # Concatenate along kv-sequence axis
    k_aug = torch.cat([k_base, k_ctrl], dim=1)  # (B*H, S + T_ctrl, head_dim)
    v_aug = torch.cat([v_base, v_ctrl], dim=1)

    # Additive attention mask: shape (B*H, S, S + T_ctrl)
    attn_mask = q.new_zeros(B * num_heads, S, S + T_ctrl)
    attn_mask[:, :, S:] = ctrl_bias_scalar  # broadcasted scalar onto ctrl columns

    if key_padding_mask is not None:
        # key_padding_mask: (B, S), True = padded → -inf in attention logits
        kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)             # (B, 1, 1, S)
        kpm = kpm.expand(-1, num_heads, -1, -1).reshape(B * num_heads, 1, S)
        attn_mask[:, :, :S] = attn_mask[:, :, :S].masked_fill(kpm, float("-inf"))

    # Scaled dot-product attention
    attn_out = F.scaled_dot_product_attention(
        q, k_aug, v_aug,
        attn_mask=attn_mask,
        dropout_p=attn.dropout if attn.training else 0.0,
    )  # (B*H, S, head_dim)

    # Reshape back: (B*H, S, head_dim) → (S, B, D)
    attn_out = attn_out.transpose(0, 1).contiguous().view(S, B, D)
    attn_out = attn.out_proj(attn_out)

    # Standard post-norm TransformerEncoderLayer residual + FF (mirrors nn.TransformerEncoderLayer.forward)
    src = x + layer.dropout1(attn_out)
    src = layer.norm1(src)
    src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
    src = src + layer.dropout2(src2)
    src = layer.norm2(src)
    return src


class KVControlTransformer(ControlTransformer):
    """KV-Control variant of MaskControl's ControlTransformer."""

    def __init__(
        self,
        code_dim,
        cond_mode,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        clip_dim=512,
        cond_drop_prob=0.1,
        clip_version=None,
        opt=None,
        mean=None,
        std=None,
        trans_path="",
        vq_model=None,
        control=None,
        # KV-Control specific
        kv_rank=64,
        ctrl_attn_bias_init=-5.0,
        use_q_residual=True,
        **kargs,
    ):
        super().__init__(
            code_dim=code_dim,
            cond_mode=cond_mode,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            clip_dim=clip_dim,
            cond_drop_prob=cond_drop_prob,
            clip_version=clip_version,
            opt=opt,
            mean=mean,
            std=std,
            trans_path=trans_path,
            vq_model=vq_model,
            control=control,
            **kargs,
        )

        # Parent built ControlNet branches (seqTransEncoder_control, first_zero_linear,
        # mid_zero_linear). Replace them with KV-Control modules.
        del self.seqTransEncoder_control
        del self.first_zero_linear
        del self.mid_zero_linear

        self.kv_rank = int(kv_rank)
        self.ctrl_attn_bias_init = float(ctrl_attn_bias_init)
        self.use_q_residual = bool(use_q_residual)

        D = self.latent_dim
        r = self.kv_rank

        # Per-layer low-rank K/V projections: D → r → D
        self.kv_down = nn.ModuleList([nn.Linear(D, r, bias=False) for _ in range(self.num_layers)])
        self.kv_up_k = nn.ModuleList([nn.Linear(r, D, bias=False) for _ in range(self.num_layers)])
        self.kv_up_v = nn.ModuleList([nn.Linear(r, D, bias=False) for _ in range(self.num_layers)])
        # Zero-init kv_down → K_ctrl = V_ctrl = 0 at init (identity invariant)
        for m in self.kv_down:
            nn.init.zeros_(m.weight)
        # kv_up_{k,v} keep default Kaiming uniform init so gradients flow

        # Per-layer learnable scalar additive bias on control-token attention columns
        self.ctrl_attn_bias = nn.ParameterList([
            nn.Parameter(torch.full((1,), self.ctrl_attn_bias_init))
            for _ in range(self.num_layers)
        ])

        # Q-residual on upper-half layers (paper v4 headline config)
        if self.use_q_residual:
            self.q_start_layer = self.num_layers // 2
            n_q = self.num_layers - self.q_start_layer
            # Shared q_down / q_up (paper convention) — standard init, NOT zero
            self.q_down = nn.Linear(D, r, bias=False)
            self.q_up = nn.Linear(r, D, bias=False)
            # Per-upper-layer scalar gates init=0 → identity at init, gradients flow
            self.q_gates = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(n_q)])
        else:
            self.q_start_layer = self.num_layers  # disabled

        # Recompute trainable freeze/unfreeze with KV modules now present
        self.ctrl_train()

        # Bookkeeping for diagnostic logs
        self._param_counts = self._compute_param_counts()

    # --- Lifecycle / freezing -------------------------------------------------

    def ctrl_train(self):
        """Override parent: parent.ctrl_train unfreezes seqTransEncoder_control / first_zero_linear /
        mid_zero_linear — we deleted those modules, so calling parent crashes. Reproduce the
        intent (freeze everything, unfreeze adapter) with the KV-Control trainable set."""
        if not hasattr(self, "kv_down"):
            # Called from parent.__init__ before our adapters exist — safe no-op; we re-call
            # ctrl_train() at the end of our __init__ once everything is in place.
            return
        freeze_block(self)
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

    def ctrl_eval(self):
        # Parent freezes everything. Safe — does not touch deleted modules.
        freeze_block(self)

    def _compute_param_counts(self):
        def num(p_iter):
            return sum(p.numel() for p in p_iter)
        cnt = {
            "encoder_control_M": num(self.encoder_control.parameters()) / 1e6,
            "kv_down_M": num(p for m in self.kv_down for p in m.parameters()) / 1e6,
            "kv_up_k_M": num(p for m in self.kv_up_k for p in m.parameters()) / 1e6,
            "kv_up_v_M": num(p for m in self.kv_up_v for p in m.parameters()) / 1e6,
            "ctrl_attn_bias_K": num(self.ctrl_attn_bias) / 1e3,
        }
        if self.use_q_residual:
            cnt["q_down_M"] = num(self.q_down.parameters()) / 1e6
            cnt["q_up_M"] = num(self.q_up.parameters()) / 1e6
            cnt["q_gates_K"] = num(self.q_gates) / 1e3
        trainable_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        cnt["total_trainable_M"] = trainable_M
        return cnt

    # --- Core forward ---------------------------------------------------------

    def _encode_ctrl_to_kv(self, ctrlNet_cond: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """ctrlNet_cond: (B, T_frames, C_in) → per-layer (K_ctrl_i, V_ctrl_i) each (T_ctrl, B, D)."""
        # encoder_control expects (B, C_in, T_frames); returns (B, D, T_ctrl=49)
        ctrl_feat = self.encoder_control(ctrlNet_cond.permute(0, 2, 1))
        ctrl_feat = ctrl_feat.permute(2, 0, 1)  # (T_ctrl, B, D)
        kv_list = []
        for i in range(self.num_layers):
            h = self.kv_down[i](ctrl_feat)   # (T_ctrl, B, r) — zero at init
            k_ctrl = self.kv_up_k[i](h)      # (T_ctrl, B, D) — zero at init
            v_ctrl = self.kv_up_v[i](h)      # (T_ctrl, B, D) — zero at init
            kv_list.append((k_ctrl, v_ctrl))
        return kv_list

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False, ctrlNet_cond=None):
        # ctrlNet_cond=None / ctrl_net disabled → run pure base path (cannot delegate to
        # parent because parent assumes seqTransEncoder_control/first_zero_linear/mid_zero_linear).
        if (self.ctrl_net is None) or (not self.ctrl_net) or (ctrlNet_cond is None):
            return self._base_trans_forward(motion_ids, cond, padding_mask, force_mask)

        cond = self.mask_cond(cond, force_mask=force_mask)
        if len(motion_ids.shape) == 2:
            x = self.token_emb(motion_ids)
        else:
            x = motion_ids
        x = self.input_process(x)
        cond_tok = self.cond_emb(cond).unsqueeze(0)            # (1, B, D)
        x = self.position_enc(x)
        xseq = torch.cat([cond_tok, x], dim=0)                 # (S+1, B, D)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)

        # Encode control to per-layer (K_ctrl, V_ctrl)
        kv_list = self._encode_ctrl_to_kv(ctrlNet_cond)

        # Layer-by-layer forward with KV injection (+ Q-residual on upper half)
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

        # Drop cond token, project to logits
        output = output[1:]                                    # (S, B, D)
        logits = self.output_process(output)                   # (B, ntoken, S)
        return logits

    def _base_trans_forward(self, motion_ids, cond, padding_mask, force_mask=False):
        """ctrlNet_cond=None fallback: bit-exact base MaskTransformer forward (no KV, no Q-residual).

        Must not delegate to parent.trans_forward / parent.super().trans_forward — both code
        paths attempt to access the deleted ControlNet branches at one place or another.
        """
        cond = self.mask_cond(cond, force_mask=force_mask)
        if len(motion_ids.shape) == 2:
            x = self.token_emb(motion_ids)
        else:
            x = motion_ids
        x = self.input_process(x)
        cond_tok = self.cond_emb(cond).unsqueeze(0)
        x = self.position_enc(x)
        xseq = torch.cat([cond_tok, x], dim=0)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[1:]
        logits = self.output_process(output)
        return logits

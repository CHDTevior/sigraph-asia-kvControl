"""
v8 KV-Context ControlNet for T-concat v4 MaskTransformer.

This is the v4-compatible version of control_transformer_t_concat_v8_kv.py.
It inherits from the v4 base transformer (which has CLIP text adapter + dense
cross-attention) instead of v1, and combines the KV-injection control path
with v4's text conditioning path.

Key idea (same as original v8):
  - NO parallel control transformer (saves 50.4M params)
  - Control signal encoded to 49 tokens via 1D conv
  - Per-layer zero-init low-rank KV projections: ctrl -> dK, dV
  - Concatenate ctrl KV to base self-attention: K=[K_base; K_ctrl], V=[V_base; V_ctrl]
  - Attention mechanism automatically learns weight allocation

Changes from original v8 (control_transformer_t_concat_v8_kv.py):
  A. Inherits from v4 (MaskTransformer) instead of v1
  B. trans_forward accepts text_seq and text_pad_mask (v4 signature)
  C. trans_forward preserves v4's text adapter + cross-attention path alongside
     KV injection — uses forward_with_kv_context_and_crossattn helper
  D. forward_with_cond_scale inherited from v2 (via v4) already passes
     text_seq/text_pad_mask through — no override needed
  E. forward() (training) uses encode_text_with_seq instead of encode_text
  F. generate() uses encode_text_with_seq instead of encode_text

Everything else (encoder_control, kv_down/up, TTT, generate_with_control,
loss computation) is identical to the original v8.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from einops import repeat

# CHANGE: inherit from v4 instead of v1
from models.mask_transformer.transformer_t_concat_v4 import MaskTransformer
from models.vq.encdec import Encoder
from models.mask_transformer.tools import (
    lengths_to_mask, uniform, cosine_schedule, eval_decorator,
    top_k, gumbel_sample, get_mask_subset_prob, gumbel_noise,
)
from utils.motion_process import recover_from_ric
from utils.metrics import control_joint_ids


# ============================================================
# helpers
# ============================================================

def freeze_block(block):
    block.eval()
    for p in block.parameters():
        p.requires_grad = False


def unfreeze_block(block):
    block.train()
    for p in block.parameters():
        p.requires_grad = True


def forward_with_kv_context_and_crossattn(
    trans_layers,
    src,
    ctrl_kv_list,
    cross_attn_blocks,
    cross_attn_layer_ids,
    text_memory,
    text_pad_mask,
    cond_keep,
    src_key_padding_mask=None,
    factorized_mask=None,
    ctrl_attn_bias_list=None,
):
    """
    Forward through base transformer layers, injecting control KV at each layer
    AND applying text cross-attention at designated layers.

    This combines the KV-injection logic from the original v8's
    forward_with_kv_context with the cross-attention injection from v2/v4's
    trans_forward.

    Args:
        trans_layers: iterable of TransformerEncoderLayer modules (from seqTransEncoder.layers)
        src:                [S+1, B, D] base input sequence (with cond token)
        ctrl_kv_list:       list of (ctrl_K, ctrl_V) per layer, each [T_ctrl, B, D]
        cross_attn_blocks:  nn.ModuleList of TextCrossAttentionBlock
        cross_attn_layer_ids: set of layer indices that have cross-attention
        text_memory:        [L, B, D] projected+adapted CLIP text tokens (or None)
        text_pad_mask:      [B, L] True=padding (or None)
        cond_keep:          [B, 1] 1.0=keep, 0.0=drop (for CFG)
        src_key_padding_mask: [B, S+1] padding mask for the input sequence
        factorized_mask:    [S+1, S+1] boolean mask from _build_factorized_attn_mask
                            (True=blocked). Applied on odd layers only. When provided,
                            ctrl KV tokens are NOT constrained by factorized mask
                            (control is global signal), so the mask is padded with
                            False columns for ctrl tokens.
    """
    output = src
    cross_idx = 0
    for layer_idx, mod in enumerate(trans_layers):
        ctrl_k, ctrl_v = ctrl_kv_list[layer_idx]  # [T_ctrl, B, D]
        T_ctrl = ctrl_k.shape[0]

        # Build augmented padding mask for KV
        if src_key_padding_mask is not None:
            kv_padding_mask = F.pad(src_key_padding_mask, (0, T_ctrl), value=False)
        else:
            kv_padding_mask = None

        # Build attn_mask for factorized layers (odd layers only)
        use_factorized = factorized_mask is not None and (layer_idx % 2 == 1)
        if use_factorized:
            # Expand [S+1, S+1] -> [S+1, S+1 + T_ctrl]:
            # ctrl tokens are global — every query can attend to them (False = allowed)
            attn_mask = F.pad(factorized_mask, (0, T_ctrl), value=False)
        else:
            attn_mask = None

        # Apply learnable control attention bias (init=-5 for true identity)
        if ctrl_attn_bias_list is not None:
            S_q = output.shape[0]
            S_kv = S_q + T_ctrl
            ctrl_bias = ctrl_attn_bias_list[layer_idx].expand(S_q, T_ctrl)
            bias_mask = torch.zeros(S_q, S_kv, device=output.device)
            bias_mask[:, -T_ctrl:] = ctrl_bias
            if attn_mask is not None:
                # Convert bool mask (True=blocked) to float (blocked=-inf, allowed=0)
                attn_mask = torch.where(attn_mask, torch.tensor(-float('inf'), device=attn_mask.device), torch.tensor(0.0, device=attn_mask.device))
                attn_mask = attn_mask + bias_mask
            else:
                attn_mask = bias_mask

        if getattr(mod, 'norm_first', False):
            # Pre-norm path
            x = mod.norm1(output)
            k_aug = torch.cat([x, ctrl_k], dim=0)
            v_aug = torch.cat([x, ctrl_v], dim=0)
            x = mod.self_attn(
                x, k_aug, v_aug,
                attn_mask=attn_mask,
                key_padding_mask=kv_padding_mask,
                need_weights=False,
            )[0]
            output = output + mod.dropout1(x)
            # FFN
            output = output + mod._ff_block(mod.norm2(output))
        else:
            # Post-norm path (default)
            k_aug = torch.cat([output, ctrl_k], dim=0)
            v_aug = torch.cat([output, ctrl_v], dim=0)
            x = mod.self_attn(
                output, k_aug, v_aug,
                attn_mask=attn_mask,
                key_padding_mask=kv_padding_mask,
                need_weights=False,
            )[0]
            output = mod.norm1(output + mod.dropout1(x))
            # FFN
            output = mod.norm2(output + mod._ff_block(output))

        # Cross-attention (from v2/v4) — AFTER self-attention + FFN
        if layer_idx in cross_attn_layer_ids:
            if text_memory is not None:
                output = cross_attn_blocks[cross_idx](
                    output, text_memory, text_pad_mask, cond_keep=cond_keep
                )
            cross_idx += 1

    return output


class ControlTransformerTConcatV4(MaskTransformer):
    """v8 KV-Context ControlNet for v4 base — no parallel transformer, control via KV injection.

    Inherits from v4 MaskTransformer which provides:
      - CLIP text adapter (CLIPTextAdapter)
      - Dense text cross-attention (TextCrossAttentionBlock at every cross_attn_interval layers)
      - encode_text_with_seq (pooled + sequence tokens)
      - forward_with_cond_scale (passes text_seq/text_pad_mask through)

    Adds (same as original v8):
      - Control encoder (1D conv, 196 frames -> 49 tokens)
      - Per-layer low-rank KV projections for control injection
      - TTT (test-time training) during generation
    """

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
        kv_rank=64,
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
            **kargs,
        )

        self.num_layers = num_layers
        self.mean = mean
        self.std = std
        self.vq_model = vq_model
        self.control = control
        self.ctrl_net = True
        self.kv_rank = kv_rank

        # ---- load pretrained v4 base transformer ----
        if trans_path != "":
            ckpt = torch.load(trans_path, map_location=opt.device)
            model_key = "t2m_transformer" if "t2m_transformer" in ckpt else "trans"
            missing_keys, unexpected_keys = self.load_state_dict(
                ckpt[model_key], strict=False
            )
            freeze_block(self)
            print(
                f"Loaded v4 T-concat Transformer from {trans_path}\n"
                f"  missing keys: {missing_keys}\n"
                f"  unexpected keys: {unexpected_keys}"
            )

        # ============= v8 KV-Context modules =============

        # Control encoder: 196 frames -> 49 tokens (same as v5/v8)
        input_emb_width = (
            6 if self.control == "trajectory"
            else len(control_joint_ids) * 2 * 3
        )
        self.encoder_control = Encoder(
            input_emb_width=input_emb_width,
            output_emb_width=self.latent_dim,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation="relu",
            norm=None,
        )

        # Per-layer low-rank KV projections: D -> r -> D (zero-init down)
        # ctrl_feat [49, B, D] -> K_ctrl [49, B, D], V_ctrl [49, B, D]
        self.kv_down = nn.ModuleList([
            nn.Linear(self.latent_dim, self.kv_rank, bias=False)
            for _ in range(self.num_layers)
        ])
        self.kv_up_k = nn.ModuleList([
            nn.Linear(self.kv_rank, self.latent_dim, bias=False)
            for _ in range(self.num_layers)
        ])
        self.kv_up_v = nn.ModuleList([
            nn.Linear(self.kv_rank, self.latent_dim, bias=False)
            for _ in range(self.num_layers)
        ])

        # Zero-init the down projection so ctrl KV starts at zero
        for m in self.kv_down:
            nn.init.zeros_(m.weight)

        # Learnable attention bias for control KV columns, init to -5
        # so zero-init K/V don't steal softmax mass at start (true identity init)
        self.ctrl_attn_bias = nn.ParameterList([
            nn.Parameter(torch.full((1,), -5.0)) for _ in range(self.num_layers)
        ])

        # Per-codebook mean embeddings for masked positions
        self.mask_emb_vq_list = [
            vq_model.vqvae.quantizers[i].codebook.mean(0) for i in range(self.Q)
        ]

        # Print param counts
        enc_params = sum(p.numel() for p in self.encoder_control.parameters())
        kv_params = sum(
            p.numel() for p in list(self.kv_down.parameters())
            + list(self.kv_up_k.parameters())
            + list(self.kv_up_v.parameters())
        )
        print(f"[v8-v4-noQres] Control encoder: {enc_params/1e6:.1f}M, "
              f"KV (rank={kv_rank}): {kv_params/1e6:.1f}M, "
              f"Total trainable: {(enc_params+kv_params)/1e6:.1f}M "
              f"(Q-Residual REMOVED per Codex verdict 019db6bd — paper main method)")

    # ============= train/eval mode =============

    def ctrl_train(self):
        freeze_block(self)
        # BUG FIX: freeze_block calls self.eval(), which recursively sets
        # self.training=False on ALL modules. This disables mask_2d_hybrid
        # and cond_dropout (both check self.training on the top-level module).
        # Only restore the top-level training flag (non-recursive) so frozen
        # backbone submodules keep eval semantics (dropout off).
        self.training = True
        unfreeze_block(self.encoder_control)
        unfreeze_block(self.kv_down)
        unfreeze_block(self.kv_up_k)
        unfreeze_block(self.kv_up_v)
        # Also unfreeze control attention bias
        for p in self.ctrl_attn_bias.parameters():
            p.requires_grad = True

    def ctrl_eval(self):
        self.eval()

    # ============= control signal building =============

    def _motion_to_joints(self, pred_motions):
        pred_denorm = pred_motions * self.std + self.mean
        return recover_from_ric(pred_denorm.float(), self.opt.joints_num)

    def _build_ctrl_cond(self, global_joint, pred_joints, global_joint_mask, B):
        ctrlNet_cond = (global_joint - pred_joints) * global_joint_mask.unsqueeze(-1)
        ctrlNet_cond2 = global_joint * global_joint_mask.unsqueeze(-1)

        if self.control == "trajectory":
            ctrlNet_cond = ctrlNet_cond[:, :, 0]
            ctrlNet_cond2 = ctrlNet_cond2[:, :, 0]
        else:
            ctrlNet_cond = ctrlNet_cond[..., control_joint_ids, :].reshape(B, -1, len(control_joint_ids) * 3)
            ctrlNet_cond2 = ctrlNet_cond2[..., control_joint_ids, :].reshape(B, -1, len(control_joint_ids) * 3)

        return torch.cat([ctrlNet_cond, ctrlNet_cond2], dim=-1)

    def _decode_motion_from_local_ids(self, local_ids_bt6):
        x_q_list = self.vq_model.vqvae.get_x_quantized_from_x_ids(local_ids_bt6)
        pred_motions = self.vq_model.vqvae.forward_decoder_from_quantized_codes(x_q_list)
        pred_motions = pred_motions.permute(0, 2, 3, 1).squeeze(1)
        return pred_motions

    def _encode_ctrl_to_kv(self, ctrlNet_cond):
        """
        Encode control signal to per-layer KV pairs.

        ctrlNet_cond: [B, 196, 6/36]
        Returns: list of (K_ctrl, V_ctrl) per layer, each [T=49, B, D]
        (no-Q-Res variant: Q-side residual mechanism removed, KV injection only)
        """
        # Encode: [B, 196, C] -> [B, D, 49]
        ctrl_feat = self.encoder_control(ctrlNet_cond.permute(0, 2, 1))
        ctrl_feat = ctrl_feat.permute(2, 0, 1)  # [49, B, D]

        kv_list = []
        for i in range(self.num_layers):
            h = self.kv_down[i](ctrl_feat)  # [49, B, r]
            k_ctrl = self.kv_up_k[i](h)     # [49, B, D]
            v_ctrl = self.kv_up_v[i](h)     # [49, B, D]
            kv_list.append((k_ctrl, v_ctrl))

        return kv_list

    # ============= trans_forward with KV injection + text cross-attention =============

    def trans_forward(
        self,
        motion_emb,
        cond,
        padding_mask,
        force_mask=False,
        text_seq=None,
        text_pad_mask=None,
        ctrlNet_cond=None,
    ):
        """
        Override trans_forward to inject control via KV-Context while preserving
        v4's text conditioning path (text adapter + cross-attention).

        When ctrlNet_cond is None, falls back to v4's standard trans_forward
        (inherited via super()). When ctrlNet_cond is provided, uses
        forward_with_kv_context_and_crossattn for layer-by-layer forward with
        both KV injection and cross-attention.

        Args:
            motion_emb:    [B, S, code_dim] where S = T * Q
            cond:          [B, clip_dim]       pooled CLIP embedding
            padding_mask:  [B, S]              True = pad
            force_mask:    bool                force unconditional (for CFG)
            text_seq:      [B, L, clip_dim]    CLIP sequence tokens (optional)
            text_pad_mask: [B, L]              True = padding (optional)
            ctrlNet_cond:  [B, 196, 6/36]      control signal (optional)
        """
        if not self.ctrl_net or ctrlNet_cond is None:
            # Fall back to v4's trans_forward (with text adapter + cross-attn)
            return super().trans_forward(
                motion_emb, cond, padding_mask, force_mask,
                text_seq=text_seq, text_pad_mask=text_pad_mask,
            )

        bs = cond.shape[0]
        cond_keep = self._get_cond_mask(bs, cond.device, force_mask)

        # ---- pooled condition (same as v2/v4) ----
        cond = cond * cond_keep

        # ---- motion embedding ----
        motion_emb = self._inject_quantizer_token_identity(motion_emb)
        x = self.input_process(motion_emb)  # [S, B, D]
        cond_token = self.cond_emb(cond).unsqueeze(0)  # [1, B, D]

        x = self._add_factorized_position_encoding(x)
        xseq = torch.cat([cond_token, x], dim=0)  # [S+1, B, D]

        padding_mask_with_cond = torch.cat(
            [torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1
        )

        # ---- text memory for cross-attention (v4 path: proj + adapter + mask) ----
        if text_seq is not None:
            text_memory = self.text_seq_proj(text_seq)          # [B, L, D]
            text_memory = self.text_adapter(text_memory,
                                            src_key_padding_mask=text_pad_mask)  # [B, L, D]
            text_memory = text_memory * cond_keep.unsqueeze(-1)  # zero dropped samples
            text_memory = text_memory.permute(1, 0, 2)           # [L, B, D]
        else:
            text_memory = None

        # ---- encode control to per-layer KV pairs ----
        ctrl_kv_list = self._encode_ctrl_to_kv(ctrlNet_cond)

        # ---- v4.8/v4.9/v4.68b: build factorized mask for odd layers ----
        if self.factorized_attn != "none":
            factorized_mask = self._build_factorized_attn_mask(xseq.shape[0], xseq.device)
        else:
            factorized_mask = None

        # ---- forward through base transformer with KV injection + cross-attention ----
        output = forward_with_kv_context_and_crossattn(
            trans_layers=self.seqTransEncoder.layers,
            src=xseq,
            ctrl_kv_list=ctrl_kv_list,
            cross_attn_blocks=self.text_cross_attn,
            cross_attn_layer_ids=self.cross_attn_layer_ids,
            text_memory=text_memory,
            text_pad_mask=text_pad_mask,
            cond_keep=cond_keep,
            src_key_padding_mask=padding_mask_with_cond,
            factorized_mask=factorized_mask,
            ctrl_attn_bias_list=self.ctrl_attn_bias,
        )

        # Apply final norm if present
        if self.seqTransEncoder.norm is not None:
            output = self.seqTransEncoder.norm(output)

        # Remove cond token, project to logits
        out = output[1:]  # [S, B, D]

        if self.per_q_head:
            S_out, B_out, D_out = out.shape
            T_out = S_out // self.Q
            out_tqbd = out.view(T_out, self.Q, B_out, D_out)
            logits_list = []
            for q in range(self.Q):
                lq = self.output_heads[q](out_tqbd[:, q, :, :])  # [B, vocab, T]
                logits_list.append(lq)
            logits = torch.stack(logits_list, dim=3).reshape(B_out, self.vocab, T_out * self.Q)
        else:
            logits = self.output_head(out)  # [B, vocab, S]
        return logits

    # ============= forward (training) =============

    def forward(self, y, m_lens, pose):
        """Training forward pass. Signature matches trainer: model(conds, m_lens, motion)."""
        device = next(self.parameters()).device
        m_length = m_lens
        bs = len(y)

        # VQ encode
        with torch.no_grad():
            pose_4d = pose.float().permute(0, 2, 1).unsqueeze(2)  # [B, 263, 1, T]
            code_idx = self.vq_model.get_code_idx(pose_4d)  # [B, Q, T]
        code_idx = code_idx.to(device)

        # Flatten to local sequence: interleave [t0q0, t0q1, ..., t0q5, t1q0, ...]
        code_idx_bt6 = code_idx.permute(0, 2, 1).contiguous()  # [B, T, Q]
        T_tok = code_idx_bt6.shape[1]
        local_ids = code_idx_bt6.reshape(bs, T_tok * self.Q)

        m_tok_lens = (m_length.float() / 4).ceil().long()  # frame -> token length
        m_tokens_len_seq = m_tok_lens * self.Q
        padding_mask = ~lengths_to_mask(m_tokens_len_seq, T_tok * self.Q)

        # BERT-style masking
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = cosine_schedule(rand_time)
        num_token_masked = (m_tok_lens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((bs, T_tok), device=device)
        batch_randperm[~lengths_to_mask(m_tok_lens, T_tok)] = -1
        sorted_indices = batch_randperm.argsort(dim=-1, descending=True)
        ranks = sorted_indices.argsort(dim=-1)
        is_mask_t = (ranks < num_token_masked.unsqueeze(-1))
        is_mask = is_mask_t.unsqueeze(-1).expand(bs, T_tok, self.Q).reshape(bs, T_tok * self.Q)

        # v4.68b: hybrid 2D masking (match base training distribution)
        if getattr(self, 'mask_2d_hybrid', False) and self.training:
            use_2d = torch.rand(bs, device=device) < getattr(self, 'mask_2d_ratio', 0.3)
            non_pad_mask_t = lengths_to_mask(m_tok_lens, T_tok)
            perm_2d = torch.rand((bs, T_tok, self.Q), device=device).argsort(dim=1)
            mask_2d = perm_2d < num_token_masked.view(bs, 1, 1)
            mask_2d &= non_pad_mask_t.unsqueeze(-1)
            mask_2d = mask_2d.reshape(bs, T_tok * self.Q)
            is_mask = torch.where(use_2d.unsqueeze(1), mask_2d, is_mask)

        # Build target and input
        target = local_ids.clone()
        target[~is_mask] = -100
        target[padding_mask] = -100

        # 10% keep, 88% mask, 2% random
        r = torch.rand_like(local_ids.float())
        x_ids = local_ids.clone()
        x_ids[is_mask & (r < 0.88)] = self.mask_id
        x_ids[is_mask & (r >= 0.88) & (r < 0.90)] = local_ids[is_mask & (r >= 0.88) & (r < 0.90)]
        noise_ids = torch.randint_like(x_ids, 0, self.K)
        x_ids[is_mask & (r >= 0.90)] = noise_ids[is_mask & (r >= 0.90)]

        x_emb = self.token_emb(x_ids)

        # Build control signal
        with torch.no_grad():
            # Decode current (partially masked) prediction
            ids_safe = x_ids.clone()
            overflow = (ids_safe >= self.K) | (ids_safe < 0)
            ids_safe[overflow] = 0
            local_bt6 = ids_safe.view(bs, T_tok, self.Q)
            pred_m = self._decode_motion_from_local_ids(local_bt6)
            pred_joints = self._motion_to_joints(pred_m)

            # GT joints
            gt_denorm = pose * self.std + self.mean
            gt_joints = recover_from_ric(gt_denorm.float(), self.opt.joints_num)

            # Random control mask
            global_joint_mask = torch.zeros((bs, pose.shape[1], self.opt.joints_num),
                                           device=device, dtype=torch.bool)
            import random
            for i in range(bs):
                dens = random.choice([1, 2, 5, 49, 196])
                if dens in [1, 2, 5]:
                    rand_dens = dens
                else:
                    rand_dens = int(m_length[i] * (dens / 196))
                import numpy as np
                selected = np.random.choice(m_length[i].cpu().numpy(), rand_dens, replace=False)
                global_joint_mask[i, selected, :] = True

            if self.control == "trajectory":
                global_joint_mask[..., 1:] = False
            elif self.control == "cross":
                from utils.metrics import cross_combination_joints
                cross_joints = cross_combination_joints()
                choose = np.random.choice(len(cross_joints), 1).item()
                choose_joint = cross_joints[choose]
                tmp = global_joint_mask.clone()
                global_joint_mask = torch.zeros_like(tmp)
                global_joint_mask[..., choose_joint] = tmp[..., choose_joint]
            elif self.control == "random":
                _mask = global_joint_mask.clone()
                global_joint_mask = torch.zeros_like(_mask)
                ctrl_joints = torch.tensor([0, 10, 11, 15, 20, 21], device=device)
                rand_idx = torch.randint(len(ctrl_joints), (bs,))
                for i in range(bs):
                    global_joint_mask[i, :, ctrl_joints[rand_idx[i]]] = _mask[i, :, ctrl_joints[rand_idx[i]]]

            ctrlNet_cond = self._build_ctrl_cond(gt_joints, pred_joints, global_joint_mask, bs)

        # CHANGE: encode text with sequence tokens (v4/v2 path) instead of pooled-only
        text_seq, text_pad_mask = None, None
        if self.cond_mode == "text":
            with torch.no_grad():
                cond_vector, text_seq, text_pad_mask = self.encode_text_with_seq(y)
        else:
            cond_vector = self.encode_text(y)

        # Forward with control + text
        logits = self.trans_forward(
            x_emb, cond_vector, padding_mask,
            text_seq=text_seq,
            text_pad_mask=text_pad_mask,
            ctrlNet_cond=ctrlNet_cond,
        )

        # CE loss
        ce_loss = F.cross_entropy(logits, target, ignore_index=-100)

        # Trajectory control loss — differentiable soft decode
        padding_mask_t = ~lengths_to_mask(m_tok_lens, T_tok)
        logits_bsv = logits.permute(0, 2, 1)  # [B, S, vocab]
        probs_local = F.softmax(logits_bsv, dim=-1).view(bs, T_tok, self.Q, self.K)
        emb_list = []
        for q in range(self.Q):
            cb = self.vq_model.vqvae.quantizers[q].codebook  # [K, C]
            emb_q = probs_local[:, :, q, :] @ cb  # [B, T_tok, C]
            emb_q = emb_q.masked_fill(padding_mask_t.unsqueeze(-1), 0.0)
            emb_list.append(emb_q.permute(0, 2, 1))  # [B, C, T_tok]

        pred_motions2 = self.vq_model.vqvae.forward_decoder_from_quantized_codes(emb_list)
        pred_motions2 = pred_motions2.squeeze(2).permute(0, 2, 1)  # [B, 196, 263]
        pred_joints2 = self._motion_to_joints(pred_motions2)

        loss_traj = F.l1_loss(
            pred_joints2[global_joint_mask],
            gt_joints[global_joint_mask],
            reduction="mean",
        )
        acc = (logits.argmax(dim=1) == target).float()
        acc = acc[target != -100].mean()

        # Return 5 values to match trainer: (loss_emb, ce_loss, pred_ids, acc, loss_tta)
        return torch.tensor(0.0, device=device), ce_loss, None, acc, loss_traj

    def _ttt_loss_per_sample(self, pred_motions_denorm, global_joint, global_joint_mask):
        """(B,) anchor MSE per sample — the RGAR gate/rollback signal (a batch-mean scalar
        must never gate or roll back a whole batch; codex 019f59cc round-3)."""
        expanded_mask = global_joint_mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        squared_error = (pred_motions_denorm - global_joint) ** 2
        masked_error = squared_error * expanded_mask
        valid_counts = expanded_mask.sum(dim=(1, 2, 3)).clamp(min=1)
        return masked_error.sum(dim=(1, 2, 3)) / valid_counts

    def _ttt_loss(self, pred_motions_denorm, global_joint, global_joint_mask):
        return self._ttt_loss_per_sample(
            pred_motions_denorm, global_joint, global_joint_mask).mean()

    # ============= generate with control =============

    @eval_decorator
    def generate(
        self,
        conds,
        m_lens,
        timesteps: int,
        cond_scale: int,
        temperature=1.0,
        topk_filter_thres=0.0,
        gsample=False,
        force_mask=False,
        vq_model=None,
        global_joint=None,
        global_joint_mask=None,
        _mean=None,
        _std=None,
        lr=6e-2,
        each_iter=50,
        avoid_points=None,
        abitary_func=None,
        is_relative=False,
        rgar=None,
    ):
        device = next(self.parameters()).device
        m_lens_t = torch.as_tensor(m_lens, device=device)
        if m_lens_t.ndim == 0:
            m_lens_t = m_lens_t.unsqueeze(0)
        batch_size = m_lens_t.shape[0]

        seq_len = int(m_lens_t.max().item())
        seq_len = max(1, seq_len)

        if vq_model is None:
            vq_model = self.vq_model

        # CHANGE: encode text with sequence tokens (v4/v2 path) instead of pooled-only
        text_seq, text_pad_mask = None, None
        if conds is None:
            cond_vector = torch.zeros((batch_size, self.clip_dim), device=device)
        elif self.cond_mode == "text":
            cond_vector, text_seq, text_pad_mask = self.encode_text_with_seq(conds)
        else:
            cond_vector = self.encode_text(conds)

        S = seq_len * self.Q
        m_lens_seq = m_lens_t * self.Q

        padding_mask_seq = ~lengths_to_mask(m_lens_seq, S).to(device)
        padding_mask_t = ~lengths_to_mask(m_lens_t, seq_len).to(device)

        # Init ids
        ids = torch.where(
            padding_mask_seq,
            torch.full((batch_size, S), self.pad_id, device=device),
            torch.full((batch_size, S), self.mask_id, device=device),
        )

        scores = torch.where(
            padding_mask_seq,
            torch.full_like(ids, 1e5, dtype=torch.float),
            torch.zeros_like(ids, dtype=torch.float),
        )

        starting_temperature = temperature

        with torch.no_grad():
          for _step_idx, timestep in enumerate(torch.linspace(0, 1, timesteps, device=device)):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked_t = (m_lens_t * rand_mask_prob).round().clamp(min=1)

            frame_scores = scores.view(batch_size, seq_len, self.Q).mean(dim=-1)
            sorted_indices = frame_scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask_t = (ranks < num_token_masked_t.unsqueeze(-1)) & ~padding_mask_t
            is_mask = is_mask_t.unsqueeze(-1).expand(batch_size, seq_len, self.Q).reshape(batch_size, S)

            emb = self.token_emb(ids)
            mask_emb = self.token_emb.weight[self.mask_id]
            emb = torch.where(is_mask.unsqueeze(-1), mask_emb, emb)

            # Build control signal from current predictions
            ids_safe = ids.clone()
            overflow = (ids_safe >= self.K) | (ids_safe < 0)
            ids_safe[overflow] = 0
            local_bt6 = ids_safe.view(batch_size, seq_len, self.Q)

            x_q_list = vq_model.vqvae.get_x_quantized_from_x_ids(local_bt6)
            pred_m = vq_model.vqvae.forward_decoder_from_quantized_codes(x_q_list)
            pred_m = pred_m.squeeze(2).permute(0, 2, 1)
            pred_joints = self._motion_to_joints(pred_m)

            ctrlNet_cond = self._build_ctrl_cond(
                global_joint, pred_joints, global_joint_mask, batch_size
            )

            # CHANGE: pass text_seq and text_pad_mask to forward_with_cond_scale
            logits = self.forward_with_cond_scale(
                emb,
                cond_vector=cond_vector,
                padding_mask=padding_mask_seq,
                cond_scale=cond_scale,
                force_mask=force_mask,
                text_seq=text_seq,
                text_pad_mask=text_pad_mask,
                ctrlNet_cond=ctrlNet_cond,
            )

            logits_bsv = logits.permute(0, 2, 1).contiguous()

            if topk_filter_thres > 0:
                logits_bsv = top_k(logits_bsv, topk_filter_thres, dim=-1)

            # ======= Test-Time Training (TTT) =======
            if each_iter > 0 and global_joint is not None and global_joint_mask.any():
                # Dynamic schedule: more TTT at later steps (like MaskControl)
                if getattr(self.opt, 'ttt_dynamic', False):
                    _n_iter = (_step_idx + 1) * each_iter
                else:
                    _n_iter = each_iter
                logits_bsv = logits_bsv.detach().requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    [logits_bsv], lr=lr, betas=(0.5, 0.9), weight_decay=1e-6
                )
                # TTT_TRACE probe (armed by generate_with_control only when TTT_TRACE_DIR is
                # set; None in every normal run -> zero behaviour change). Records this unmask
                # step's per-iteration loss. The per-iter .item() forces a sync, so traced runs
                # are for CONVERGENCE analysis only, never latency.
                _tr = getattr(self, "_ttt_trace", None)
                _tr_losses = [] if _tr is not None else None
                # ---- RGAR-lite (A1-calibrated, 2026-07-13): (i) best-iterate ROLLBACK — on the
                # A1 probe 75% of samples end refinement worse than their best iterate; tracking
                # is branchless on-GPU (torch.where), no per-iter host sync. (ii) within-step
                # PLATEAU STOP — each unmask step reaches 95% of its gain in ~12-23 iters
                # (median; p90 <=58) while the schedule allocates 35..350; check every
                # `check_every` iters (one sync per check). Both OFF unless rgar is passed —
                # the M2/M3 paths are byte-identical.
                _rg_on = bool(rgar)
                if _rg_on:
                    if abitary_func is not None:
                        raise RuntimeError("rgar + abitary_func unsupported: the per-sample "
                                           "gate signal cannot decompose an arbitrary scalar")
                    _rg_check = int(rgar.get("check_every", 10))
                    if _rg_check <= 0:
                        raise ValueError(f"rgar check_every must be > 0, got {_rg_check}")
                    _rg_rtol = float(rgar.get("plateau_rtol", 0.01))
                    _best_l = None          # (B,) per-sample best — batched RGAR: each sample
                    _best_logits = None     # rolls back to ITS OWN best iterate
                    _win_ref = None
                _rg_used = 0
                with torch.enable_grad():
                    for _ttt_step in range(_n_iter):
                        logits_btqk = logits_bsv.view(batch_size, seq_len, self.Q, self.K)
                        emb_list = []
                        for q in range(self.Q):
                            cb = vq_model.vqvae.quantizers[q].codebook
                            # Gumbel noise for stochastic exploration (like MaskControl)
                            if getattr(self.opt, 'ttt_gumbel', False):
                                from models.mask_transformer.tools import gumbel_noise
                                noisy_logits = logits_btqk[:, :, q, :] / max(temperature, 1e-10) + gumbel_noise(logits_btqk[:, :, q, :])
                                probs_q = F.softmax(noisy_logits, dim=-1)
                            else:
                                probs_q = F.softmax(logits_btqk[:, :, q, :], dim=-1)
                            emb_q = probs_q @ cb
                            emb_q = emb_q.masked_fill(padding_mask_t.unsqueeze(-1), 0.0)
                            emb_list.append(emb_q.permute(0, 2, 1))

                        pred_motions = vq_model.vqvae.forward_decoder_from_quantized_codes(emb_list)
                        pred_motions = pred_motions.squeeze(2).permute(0, 2, 1)
                        pred_motions_denorm = pred_motions * self.std + self.mean
                        pred_joints_ttt = recover_from_ric(pred_motions_denorm.float(), self.opt.joints_num)

                        loss_ttt = self._ttt_loss(pred_joints_ttt, global_joint, global_joint_mask)

                        if abitary_func is not None:
                            loss_ttt = loss_ttt + abitary_func(pred_joints_ttt)

                        if _rg_on:
                            # track BEFORE the update: the loss was computed at the CURRENT
                            # logits, so the snapshot must pair with them, not with the
                            # post-step iterate (off-by-one otherwise). Per-sample, branchless,
                            # no host sync.
                            with torch.no_grad():
                                _lps = self._ttt_loss_per_sample(
                                    pred_joints_ttt, global_joint, global_joint_mask).detach()
                                if _best_l is None:
                                    _best_l = _lps.clone()
                                    _best_logits = logits_bsv.detach().clone()
                                else:
                                    _imp = _lps < _best_l                       # (B,)
                                    _best_l = torch.where(_imp, _lps, _best_l)
                                    _bview = _imp.view(-1, *([1] * (logits_bsv.dim() - 1)))
                                    _best_logits = torch.where(
                                        _bview, logits_bsv.detach(), _best_logits)
                        optimizer.zero_grad()
                        loss_ttt.backward()
                        optimizer.step()
                        if _tr_losses is not None:
                            _tr_losses.append(float(loss_ttt.item()))
                        _rg_used = _ttt_step + 1
                        if _rg_on and (_ttt_step + 1) % _rg_check == 0:
                            # plateau check: ONE host sync per `check_every` iterations. The
                            # stop signal is the MEAN of per-sample bests — conservative: the
                            # step keeps running while any sample keeps improving the mean.
                            _cur = float(_best_l.mean().item())
                            if _win_ref is not None and _cur > _win_ref * (1.0 - _rg_rtol):
                                break                # no >rtol improvement this window
                            _win_ref = _cur
                if _rg_on and _best_logits is not None:
                    # codex round-3 P1: the loop observes x_i then steps to x_{i+1}; on a cap or
                    # plateau break the final post-step iterate was never evaluated. Score it
                    # once so a last-update improvement is not thrown away by the rollback.
                    with torch.no_grad():
                        logits_btqk = logits_bsv.view(batch_size, seq_len, self.Q, self.K)
                        emb_fin = []
                        for q in range(self.Q):
                            cb = vq_model.vqvae.quantizers[q].codebook
                            p_q = F.softmax(logits_btqk[:, :, q, :], dim=-1) @ cb
                            p_q = p_q.masked_fill(padding_mask_t.unsqueeze(-1), 0.0)
                            emb_fin.append(p_q.permute(0, 2, 1))
                        pm = vq_model.vqvae.forward_decoder_from_quantized_codes(emb_fin)
                        pm = pm.squeeze(2).permute(0, 2, 1) * self.std + self.mean
                        pj_fin = recover_from_ric(pm.float(), self.opt.joints_num)
                        l_fin = self._ttt_loss_per_sample(
                            pj_fin, global_joint, global_joint_mask).detach()      # (B,)
                        _impf = (l_fin < _best_l).view(-1, *([1] * (logits_bsv.dim() - 1)))
                        _best_logits = torch.where(_impf, logits_bsv.detach(), _best_logits)
                        logits_bsv.data.copy_(_best_logits)   # C1: BEST OBSERVED per sample
                if _tr is not None:
                    _tr["stage1"].append({"step": int(_step_idx), "n_iter": int(_n_iter),
                                          "used": int(_rg_used), "losses": _tr_losses})

            if gsample:
                pred_local = gumbel_sample(logits_bsv, temperature=starting_temperature, dim=-1)
                probs_local = F.softmax(logits_bsv / max(temperature, 1e-8), dim=-1)
            else:
                probs_local = F.softmax(logits_bsv / max(temperature, 1e-8), dim=-1)
                pred_local = Categorical(probs_local).sample()

            ids = torch.where(is_mask, pred_local, ids)

            conf = probs_local.gather(-1, pred_local.unsqueeze(-1)).squeeze(-1)
            frame_conf = conf.view(batch_size, seq_len, self.Q).mean(dim=-1)
            frame_scores = frame_conf.masked_fill(~is_mask_t, 1e5).masked_fill(padding_mask_t, 1e5)
            scores = frame_scores.unsqueeze(-1).expand(batch_size, seq_len, self.Q).reshape(batch_size, S)

        ids_final = ids.view(batch_size, seq_len, self.Q)
        ids_final = torch.where(
            padding_mask_t.unsqueeze(-1).expand_as(ids_final),
            torch.full_like(ids_final, -1),
            ids_final,
        )

        logits_final = logits_bsv.view(batch_size, seq_len, self.Q, self.K)
        logits_final = logits_final.permute(0, 2, 1, 3).contiguous()

        return ids_final, logits_final

    def generate_with_control(
        self, clip_text, m_length, time_steps, cond_scale,
        temperature, topkr, force_mask,
        vq_model, global_joint, global_joint_mask,
        _mean, _std,
        res_cond_scale=None,
        res_model=None,
        control_opt=None,
        avoid_points=None,
        abitary_func=None,
        is_relative=False,
    ):
        # TTT_TRACE_DIR arms per-iteration convergence tracing in generate()'s Stage-1 loop and
        # the Stage-2 loop below; the trace is dumped to one json per call. Unset (every normal
        # run): self._ttt_trace stays absent and no traced branch executes. codex 019f59cc P0:
        # the attribute is removed in a finally block — an exception mid-generate must NOT leave
        # it armed for a later untraced call — and traced runs demand batch=1 + Adam Stage-2
        # (Stage-1 loss is a batch mean; the LBFGS branch has no tracing).
        import os as _os_tr
        _trace_dir = _os_tr.environ.get("TTT_TRACE_DIR")
        if _trace_dir:
            if len(clip_text) != 1 or int(m_length.shape[0]) != 1:
                raise RuntimeError(f"TTT_TRACE_DIR requires batch=1 (got {len(clip_text)}); "
                                   f"a traced batch would record batch-MEAN losses.")
            if _os_tr.environ.get("STAGE2_OPTIMIZER", "adam").lower() != "adam":
                raise RuntimeError("TTT_TRACE_DIR requires STAGE2_OPTIMIZER=adam — the LBFGS "
                                   "branch is untraced and would yield empty stage2 stats.")
            self._ttt_trace = {"stage1": [], "stage2": [], "stage2_exit": None,
                               "batch_size": 1, "complete": False,
                               "tag": _os_tr.environ.get("TTT_TRACE_TAG", ""),
                               "text": str(clip_text[0])[:120]}

        try:
            return self._generate_with_control_impl(
                clip_text, m_length, time_steps, cond_scale, temperature, topkr, force_mask,
                vq_model, global_joint, global_joint_mask, _mean, _std,
                res_cond_scale=res_cond_scale, res_model=res_model, control_opt=control_opt,
                avoid_points=avoid_points, abitary_func=abitary_func, is_relative=is_relative)
        finally:
            if _trace_dir and hasattr(self, "_ttt_trace"):
                import json as _json_tr, time as _time_tr, sys as _sys_tr
                tr = self._ttt_trace
                del self._ttt_trace              # cleanup FIRST — dump I/O must not re-arm
                # codex 019f59cc round-3: capture "are we unwinding?" BEFORE the dump try —
                # inside the except clause sys.exc_info() reports the dump error itself, so
                # testing there is always-true and would silently swallow dump failures of
                # perfectly successful calls.
                _unwinding = _sys_tr.exc_info()[0] is not None
                try:
                    _os_tr.makedirs(_trace_dir, exist_ok=True)
                    fn = _os_tr.path.join(
                        _trace_dir, f"trace_{int(_time_tr.time()*1000)}_{_os_tr.getpid()}.json")
                    with open(fn + ".tmp", "w") as fh:
                        _json_tr.dump(tr, fh)
                    _os_tr.replace(fn + ".tmp", fn)
                except Exception as _dump_err:
                    if _unwinding:
                        print(f"[ttt-trace] dump failed while unwinding: {_dump_err}")
                    else:
                        raise

    def _generate_with_control_impl(
        self, clip_text, m_length, time_steps, cond_scale,
        temperature, topkr, force_mask,
        vq_model, global_joint, global_joint_mask,
        _mean, _std,
        res_cond_scale=None,
        res_model=None,
        control_opt=None,
        avoid_points=None,
        abitary_func=None,
        is_relative=False,
    ):
        m_lens = m_length // 4
        ids_bt6, logits = self.generate(
            clip_text, m_lens, time_steps, cond_scale,
            temperature=temperature,
            topk_filter_thres=topkr,
            force_mask=force_mask,
            vq_model=vq_model,
            global_joint=global_joint,
            global_joint_mask=global_joint_mask,
            _mean=_mean, _std=_std,
            lr=control_opt.get("each_lr", 6e-2) if control_opt else 6e-2,
            each_iter=control_opt.get("each_iter", 50) if control_opt else 50,
            avoid_points=avoid_points,
            abitary_func=abitary_func,
            is_relative=is_relative,
            rgar=(control_opt or {}).get("rgar"),
        )

        seq_len = ids_bt6.shape[1]
        padding_mask = ~lengths_to_mask(m_lens, seq_len)

        emb_list = []
        for i in range(self.Q):
            cb = vq_model.vqvae.quantizers[i].codebook
            probs_i = F.softmax(logits[:, i, :, :] / max(temperature, 1e-8), dim=-1)
            emb_i = probs_i @ cb
            emb_i = emb_i.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            emb_list.append(emb_i.permute(0, 2, 1))

        # === Stage 2: post-generation embedding optimization (MaskControl-style) ===
        # When last_iter > 0, refine emb_list directly against the keypoint loss.
        # When last_iter == 0, this block is a no-op and the original control
        # flow (M1 baseline) is preserved exactly.
        last_iter = control_opt.get("iter", 0) if control_opt else 0
        last_lr = control_opt.get("lr", 6e-2) if control_opt else 6e-2
        if last_lr is None:
            last_lr = 6e-2

        _rgar_s2 = (control_opt or {}).get("rgar")
        if last_iter > 0 and global_joint is not None and global_joint_mask is not None and global_joint_mask.any():
            # contiguous() so LBFGS' .view(-1) on .grad works (else RuntimeError "view size not compatible").
            emb_optim_list = [e.clone().detach().contiguous().requires_grad_(True) for e in emb_list]
            # Optimizer choice via env var STAGE2_OPTIMIZER=adam|lbfgs (default adam).
            # MaskControl baseline uses 600-iter LBFGS; set STAGE2_OPTIMIZER=lbfgs to match.
            import os as _os_s2
            _opt_choice = _os_s2.environ.get("STAGE2_OPTIMIZER", "adam").lower()
            if (control_opt or {}).get("s2_optimizer"):
                _opt_choice = str(control_opt["s2_optimizer"]).lower()
            if _opt_choice == "gn":
                # ---- Gauss-Newton / Levenberg-Marquardt REPLACEMENT for Stage-2 ----
                # The Stage-2 objective is a small least-squares problem: the residual lives in
                # R^{3A} (A = controlled anchor-frames; 15 dims for 5-anchor control) while the
                # variables are the ~37.6k embedding entries. Solve in the DUAL space:
                #     delta = -J^T (J J^T + lambda I)^{-1} r
                # (minimum-norm LM step; the linear system is 3A x 3A — trivially small).
                # Each LM step costs 3A backward passes for the Jacobian rows + 1 decode.
                # K<=8 steps replace the 600-iteration Adam loop. The Jacobian is per-sample,
                # so batch mode loops samples AFTER the (batched) Stage-1 — this is the
                # "batched-S1 + per-sample-GN" configuration: Stage-1 keeps its batch
                # amortization and stays on-distribution (no batch=1 logit overfitting),
                # GN only performs the small finishing moves it is FID-safe for.
                _gn_cfg = (control_opt or {}).get("gn", {}) if control_opt else {}
                _gn_steps = int(_gn_cfg.get("max_steps", 8))
                _gn_tau = float(_gn_cfg.get("tau", 1e-6))       # target anchor MSE (m^2)
                _gn_lam0 = float(_gn_cfg.get("lam0", 1e-2))
                _gn_amax = int(_gn_cfg.get("anchor_cap", 48))   # Jacobian row budget
                # prox_beta > 0: proximal acceptance — a trial must reduce anchor MSE
                # AND the combined C_beta = mse/mse0 + beta * D_Sigma(e, e0), where
                # D_Sigma is the mean per-dim Mahalanobis distance from the feed-forward
                # start e0. Controls cumulative off-manifold drift (multi-joint FID fix).
                _gn_beta = float(_gn_cfg.get("prox_beta", 0.0))
                # jac_chunk: rows per replicated-batch backward (0 = per-row loop).
                # DEFAULT 0: the fast path is metric-equivalent but NOT bit-equivalent
                # (conv backward float reassociation diverges the LM branch path within
                # solver noise) — published numbers stay reproducible on the row loop;
                # opt in explicitly for latency-critical runs.
                _gn_jchunk = int(_gn_cfg.get("jac_chunk", 0))

                # metric="codebook_cov": minimum-MAHALANOBIS-norm steps instead of minimum-L2.
                # Diagnosis 2026-07-13: plain min-L2 dual steps jump to the constraint surface
                # along directions the decoder never saw in training (FID 0.067->0.134), while
                # Adam's many small steps implicitly follow the loss geometry (0.067->0.083).
                # The codebook covariance Sigma_q (per quantizer, over its 128 code vectors) is
                # a training-free prior over on-manifold directions: solve
                #     (J Sigma J^T + lam*scale*I) z = r,   delta = -Sigma J^T z.
                _gn_metric = str(_gn_cfg.get("metric", "") or "")
                _sigma_q = None
                if _gn_metric == "codebook_cov":
                    _gn_ridge = float(_gn_cfg.get("ridge", 0.1))
                    _sigma_q = []
                    for q in range(self.Q):
                        cb = vq_model.vqvae.quantizers[q].codebook.detach().float()  # (K, C)
                        c = cb - cb.mean(0, keepdim=True)
                        S = (c.T @ c) / max(1, cb.shape[0] - 1)               # (C, C)
                        mu = float(S.diagonal().mean().item())
                        if not (mu > 0) or not torch.isfinite(S).all():       # codex P1 guard
                            raise RuntimeError(f"quantizer {q}: degenerate codebook covariance "
                                               f"(mean diag {mu}); cannot build the metric")
                        # ridge keeps the metric positive-definite (min eig >= ridge/(1+ridge))
                        S = S + _gn_ridge * mu * torch.eye(
                            S.shape[0], device=S.device, dtype=S.dtype)
                        _sigma_q.append(S / S.diagonal().mean())              # scale-normalized
                elif _gn_metric:
                    raise ValueError(f"unknown gn metric {_gn_metric!r}")

                _chol_q = None
                if _gn_beta > 0.0 and _sigma_q is not None:
                    _chol_q = [torch.linalg.cholesky(S) for S in _sigma_q]

                def _gn_solve_one(_emb, _gj, _gjm, _tag):
                    # Solve ONE sample: _emb = list of six (1, C, Tk) leaf tensors,
                    # _gj/_gjm = that sample's (1, T, 22[, 3]) control tensors.
                    _gn_lam = _gn_lam0
                    b_idx, f_idx_full, j_idx_full = torch.where(_gjm)
                    if not bool((b_idx == 0).all()):  # runtime check — python -O strips asserts
                        raise RuntimeError("GN per-sample slice violated: mask beyond sample 0")
                    # Dense control: the JACOBIAN is capped for cost, but acceptance / stopping /
                    # reporting use the FULL mask — otherwise the solver could fit the 48
                    # collocation rows while degrading the frames between them (codex P1).
                    f_idx, j_idx = f_idx_full, j_idx_full
                    if len(f_idx) > _gn_amax:
                        sel = torch.linspace(0, len(f_idx) - 1, _gn_amax).round().long()
                        f_idx, j_idx = f_idx[sel], j_idx[sel]
                    _A = len(f_idx)

                    def _gn_apply_sigma(vec):
                        """Apply the blockwise codebook metric to a flat (P,) vector."""
                        if _sigma_q is None:
                            return vec
                        out, off = [], 0
                        for qi, e in enumerate(_emb):
                            n = e.numel()
                            blk = vec[off:off + n].view(e.shape)              # (1, C, Tk)
                            out.append(torch.einsum("cd,bdt->bct", _sigma_q[qi], blk).reshape(-1))
                            off += n
                        return torch.cat(out)

                    _e0 = None
                    if _gn_beta > 0.0:
                        _e0 = [e.detach().clone() for e in _emb]

                    def _gn_prox_dist():
                        # mean per-dim Mahalanobis (or L2) distance of e from the start e0
                        tot, P = 0.0, 0
                        for qi, (e, e0) in enumerate(zip(_emb, _e0)):
                            d = (e.detach() - e0).view(e.shape[1], -1)        # (C, Tk)
                            if _chol_q is not None:
                                sid = torch.cholesky_solve(d, _chol_q[qi])    # Sigma^-1 d
                            else:
                                sid = d
                            tot += float((d * sid).sum().item())
                            P += d.numel()
                        return tot / max(P, 1)

                    def _gn_decode():
                        pm = vq_model.vqvae.forward_decoder_from_quantized_codes(_emb)
                        pm = pm.squeeze(2).permute(0, 2, 1) * _std + _mean
                        return recover_from_ric(pm.float(), self.opt.joints_num)

                    def _gn_residual(pj):
                        # (3A,) SOLVE residual (capped rows), metres
                        return (pj[0, f_idx, j_idx] - _gj[0, f_idx, j_idx]).reshape(-1)

                    def _gn_mse_full(pj):
                        # scalar anchor MSE over the FULL mask — the accept/stop signal
                        d = pj[0, f_idx_full, j_idx_full] - _gj[0, f_idx_full, j_idx_full]
                        return float((d ** 2).mean().item())

                    _mse0 = None   # anchor MSE at the feed-forward start (C_beta denominator)
                    _C_prev = None  # last ACCEPTED combined objective
                    with torch.enable_grad():
                        for _gn_it in range(_gn_steps):
                            pj = _gn_decode()
                            r = _gn_residual(pj)
                            mse = _gn_mse_full(pj)
                            if _mse0 is None:
                                _mse0 = max(mse, 1e-12)
                                _C_prev = mse / _mse0        # = 1.0, D_Sigma(e0,e0)=0
                            if mse < _gn_tau:
                                break
                            # Jacobian rows. Fast path: REPLICATED-BATCH backward — decode a
                            # batch of m identical copies of the sample, each replica selects
                            # its own residual component, one backward returns m rows at once
                            # through the decoder's native batch parallelism. (A vmap /
                            # is_grads_batched variant fails here: the decoder backward hits
                            # as_strided, which has no vmap rule.) The sequential version
                            # launched 3A tiny backwards and left the GPU at ~40% util.
                            _nr = 3 * _A
                            J = None
                            if _gn_jchunk > 0:
                                try:
                                    _a_all = torch.arange(_nr, device=r.device)
                                    _chunks = []
                                    for _i0 in range(0, _nr, _gn_jchunk):
                                        _rows = _a_all[_i0:_i0 + _gn_jchunk]
                                        _m = _rows.numel()
                                        _f_sel = f_idx[_rows // 3]           # (m,) frames
                                        _j_sel = j_idx[_rows // 3]           # (m,) joints
                                        _c_sel = _rows % 3                   # (m,) coords
                                        _erep = [e.detach().repeat(_m, 1, 1)
                                                  .requires_grad_(True) for e in _emb]
                                        _pmr = vq_model.vqvae.forward_decoder_from_quantized_codes(_erep)
                                        _pmr = _pmr.squeeze(2).permute(0, 2, 1) * _std + _mean
                                        _pjr = recover_from_ric(_pmr.float(), self.opt.joints_num)
                                        _sc = _pjr[torch.arange(_m, device=r.device),
                                                   _f_sel, _j_sel, _c_sel].sum()
                                        _g = torch.autograd.grad(_sc, _erep, allow_unused=True)
                                        _chunks.append(torch.cat(
                                            [(gi if gi is not None else torch.zeros(
                                                (_m,) + tuple(e.shape[1:]),
                                                device=r.device, dtype=r.dtype))
                                             .reshape(_m, -1)
                                             for gi, e in zip(_g, _emb)], dim=1))
                                    J = torch.cat(_chunks)                   # (3A, P)
                                except RuntimeError as _je:
                                    print(f"  [Stage2 GN]{_tag} batched-J unavailable "
                                          f"({str(_je)[:80]}) — per-row fallback")
                                    J = None
                            if J is None:
                                J_rows = []
                                for k in range(_nr):
                                    grads = torch.autograd.grad(
                                        r[k], _emb, retain_graph=(k < _nr - 1),
                                        allow_unused=True)
                                    J_rows.append(torch.cat(
                                        [(g if g is not None else torch.zeros_like(e)).reshape(-1)
                                         for g, e in zip(grads, _emb)]))
                                J = torch.stack(J_rows)                      # (3A, P)
                            with torch.no_grad():
                                if _sigma_q is not None:
                                    # one einsum per quantizer over ALL rows (codex P2: the
                                    # per-row loop launched up to 864 kernels per LM step)
                                    JS_parts, off = [], 0
                                    for qi, e in enumerate(_emb):
                                        n = e.numel()
                                        blk = J[:, off:off + n].view(J.shape[0], *e.shape[1:])
                                        JS_parts.append(torch.einsum(
                                            "cd,mdt->mct", _sigma_q[qi], blk).reshape(J.shape[0], -1))
                                        off += n
                                    JS = torch.cat(JS_parts, dim=1)          # (3A, P) = J Sigma
                                    JJt = J @ JS.T                            # J Sigma J^T
                                else:
                                    JJt = J @ J.T
                                _scale = float(JJt.diagonal().mean().item())
                                if not (_scale > 1e-20):
                                    break     # degenerate Jacobian — LM stalled (codex P1 guard)
                                eye = torch.eye(JJt.shape[0], device=JJt.device, dtype=JJt.dtype)
                                # exact-restore base — subtractive undo accumulates float drift
                                # and later lambda trials would start from a perturbed point
                                _base = [e.detach().clone() for e in _emb]
                                accepted = False
                                for _try in range(4):                        # trust region on lambda
                                    for e, bb in zip(_emb, _base):
                                        e.copy_(bb)
                                    try:
                                        z = torch.linalg.solve(JJt + _gn_lam * _scale * eye,
                                                               r.detach())
                                    except Exception:
                                        _gn_lam *= 10.0
                                        continue
                                    if not torch.isfinite(z).all():
                                        _gn_lam *= 10.0
                                        continue
                                    delta = -_gn_apply_sigma(J.T @ z)        # (P,) [Sigma J^T z]
                                    off = 0
                                    for e in _emb:
                                        n = e.numel()
                                        e.add_(delta[off:off + n].view_as(e))
                                        off += n
                                    pj2 = _gn_decode()
                                    mse2 = _gn_mse_full(pj2)
                                    if _gn_beta > 0.0:
                                        _C2 = mse2 / _mse0 + _gn_beta * _gn_prox_dist()
                                        _ok = (mse2 < mse) and (_C2 < _C_prev)
                                    else:
                                        _ok = mse2 < mse
                                    if _ok:
                                        if _gn_beta > 0.0:
                                            _C_prev = _C2
                                        _gn_lam = max(_gn_lam * 0.5, 1e-6)
                                        accepted = True
                                        break
                                    _gn_lam *= 10.0
                                if not accepted:
                                    for e, bb in zip(_emb, _base):
                                        e.copy_(bb)                          # exact restore
                                    break                                    # LM stalled — stop
                            print(f"  [Stage2 GN]{_tag} step {_gn_it+1}/{_gn_steps} "
                                  f"full-mse {mse:.2e}->{mse2 if accepted else mse:.2e} "
                                  f"lam={_gn_lam:.1e}")
                    return [e.detach() for e in _emb]

                if _gn_amax <= 0:
                    raise ValueError(f"gn anchor_cap must be positive, got {_gn_amax}")
                _emb_out = [e.detach().clone() for e in emb_list]
                for _bi in range(len(clip_text)):
                    _gjm_b = global_joint_mask[_bi:_bi + 1]
                    if not bool(_gjm_b.any()):
                        continue                     # no anchors — keep feed-forward output
                    _emb_b = [e[_bi:_bi + 1].clone().detach().contiguous().requires_grad_(True)
                              for e in emb_list]
                    _tag = f" s{_bi}" if len(clip_text) > 1 else ""
                    _sol = _gn_solve_one(_emb_b, global_joint[_bi:_bi + 1], _gjm_b, _tag)
                    for _qi in range(self.Q):
                        _emb_out[_qi][_bi:_bi + 1] = _sol[_qi]
                # codex P0: the common footer below re-derives emb_list from emb_optim_list,
                # so the solutions MUST be copied back into those leaves (in place) —
                # assigning emb_list here would be silently discarded.
                with torch.no_grad():
                    for _qi in range(self.Q):
                        emb_optim_list[_qi].data.copy_(_emb_out[_qi])
            elif _opt_choice == "lbfgs":
                # LBFGS hyper-params chosen for matched-budget comparison: each
                # outer step runs up to 20 inner LBFGS iters, so 30 outer × 20 inner = 600 total.
                _lbfgs_outer = max(1, last_iter // 20)
                optimizer = torch.optim.LBFGS(
                    emb_optim_list, lr=1.0, max_iter=20, tolerance_grad=1e-7,
                    tolerance_change=1e-9, history_size=10, line_search_fn="strong_wolfe"
                )
            else:
                optimizer = torch.optim.AdamW(
                    emb_optim_list, lr=last_lr, betas=(0.5, 0.9), weight_decay=1e-6
                )

            def _stage2_loss_ps():
                """(B,) per-sample anchor loss at the current emb_optim_list."""
                pred_motions_s2 = vq_model.vqvae.forward_decoder_from_quantized_codes(emb_optim_list)
                pred_motions_s2 = pred_motions_s2.squeeze(2).permute(0, 2, 1)
                pred_motions_s2_denorm = pred_motions_s2 * _std + _mean
                pred_joints_s2 = recover_from_ric(
                    pred_motions_s2_denorm.float(), self.opt.joints_num
                )
                return self._ttt_loss_per_sample(
                    pred_joints_s2, global_joint, global_joint_mask), pred_joints_s2

            def _stage2_loss_eval():
                _lps, pred_joints_s2 = _stage2_loss_ps()
                _l = _lps.mean()
                if abitary_func is not None:
                    _l = _l + abitary_func(pred_joints_s2)
                return _l

            with torch.enable_grad():
                if _opt_choice == "gn":
                    pass          # Stage-2 already solved by the GN branch above; nothing to run
                elif _opt_choice == "lbfgs":
                    for _step in range(_lbfgs_outer):
                        def _closure():
                            optimizer.zero_grad()
                            _l = _stage2_loss_eval()
                            _l.backward()
                            return _l
                        loss_s2 = optimizer.step(_closure)
                        if loss_s2 is not None and loss_s2.item() < 1e-10:
                            break
                        if _step == 0 or (_step + 1) % max(1, _lbfgs_outer // 10) == 0 or _step == _lbfgs_outer - 1:
                            _v = loss_s2.item() if loss_s2 is not None else float('nan')
                            print(f"  [Stage2 LBFGS] outer {_step+1}/{_lbfgs_outer}  loss={_v:.6f}")
                else:
                    _tr_s2 = getattr(self, "_ttt_trace", None)
                    # ---- RGAR-lite Stage-2 (A1-calibrated): entry GATE — 97% of probe samples
                    # arrive below 1e-5 m^2 anchor MSE, where 600 Adam iterations are wasted and
                    # 75% of samples actually EXIT WORSE; rollback + plateau stop as in Stage-1.
                    _s2_on = bool(_rgar_s2)
                    _s2_run = True
                    if _s2_on:
                        if abitary_func is not None:
                            raise RuntimeError("rgar + abitary_func unsupported in Stage-2")
                        _tau = float(_rgar_s2.get("s2_entry_tau", 1e-5))
                        with torch.no_grad():
                            _entry_ps, _ = _stage2_loss_ps()                    # (B,)
                            _entry_mean = float(_entry_ps.mean().item())
                            _all_below = bool((_entry_ps < _tau).all().item())
                        if _tr_s2 is not None:
                            _tr_s2["stage2_entry"] = _entry_mean
                        if _all_below:
                            # batched semantics: the whole batch is skipped only when EVERY
                            # sample is already below tau (per-sample rollback protects any
                            # sample that would have been hurt by running on regardless).
                            _s2_run = False
                            if _tr_s2 is not None:
                                _tr_s2["stage2_skipped"] = True
                            print(f"  [Stage2 RGAR] SKIP (all entries < tau {_tau:.0e}; "
                                  f"mean {_entry_mean:.2e})")
                        _s2_check = int(_rgar_s2.get("check_every", 10))
                        if _s2_check <= 0:
                            raise ValueError(f"rgar check_every must be > 0, got {_s2_check}")
                        _s2_rtol = float(_rgar_s2.get("plateau_rtol", 0.01))
                        _s2_best_l, _s2_best, _s2_ref = None, None, None
                        _s2_loss_ps = None
                    for _step in range(last_iter if _s2_run else 0):
                        if _s2_on:
                            _lps2, _pj_unused = _stage2_loss_ps()
                            loss_s2 = _lps2.mean()
                        else:
                            loss_s2 = _stage2_loss_eval()
                        if _tr_s2 is not None:
                            # loss BEFORE this step's update; index 0 is the Stage-2 ENTRY loss
                            _tr_s2["stage2"].append(float(loss_s2.item()))
                        if _s2_on:
                            with torch.no_grad():
                                _ld2 = _lps2.detach()                           # (B,)
                                if _s2_best_l is None:
                                    _s2_best_l = _ld2.clone()
                                    _s2_best = [e.detach().clone() for e in emb_optim_list]
                                else:
                                    _imp2 = _ld2 < _s2_best_l
                                    _s2_best_l = torch.where(_imp2, _ld2, _s2_best_l)
                                    _s2_best = [torch.where(
                                        _imp2.view(-1, *([1] * (e.dim() - 1))), e.detach(), b)
                                        for e, b in zip(emb_optim_list, _s2_best)]
                        else:
                            # baseline-only early exit; its per-iter .item() sync is historical
                            # behaviour. In RGAR mode the same test rides the every-10 check
                            # below — a per-iteration sync would eat the latency win (codex P0).
                            if loss_s2.item() < 1e-10:
                                break
                        optimizer.zero_grad()
                        loss_s2.backward()
                        optimizer.step()
                        if _s2_on:
                            if (_step + 1) % _s2_check == 0:
                                _cur2 = float(_s2_best_l.mean().item())
                                print(f"  [Stage2 Adam/RGAR] step {_step+1}/{last_iter}  "
                                      f"best={_cur2:.6f}")
                                if _cur2 < 1e-10:
                                    break
                                if _s2_ref is not None and _cur2 > _s2_ref * (1.0 - _s2_rtol):
                                    print(f"  [Stage2 RGAR] plateau stop at iter {_step+1}")
                                    break
                                _s2_ref = _cur2
                        elif _step == 0 or (_step + 1) % max(1, last_iter // 10) == 0 \
                                or _step == last_iter - 1:
                            print(f"  [Stage2 Adam] step {_step+1}/{last_iter}  loss={loss_s2.item():.6f}")
                    if _s2_on and _s2_run and _s2_best is not None:
                        with torch.no_grad():
                            # codex round-3 P1: score the final post-step iterate too, then
                            # exit at the best OBSERVED point — per sample.
                            _l_fin2, _ = _stage2_loss_ps()
                            _impf2 = _l_fin2.detach() < _s2_best_l              # (B,)
                            _s2_best = [torch.where(
                                _impf2.view(-1, *([1] * (e.dim() - 1))), e.detach(), b)
                                for e, b in zip(emb_optim_list, _s2_best)]
                            for e, b in zip(emb_optim_list, _s2_best):
                                e.data.copy_(b)
                    if _tr_s2 is not None and last_iter > 0:
                        with torch.no_grad():
                            _tr_s2["stage2_exit"] = float(_stage2_loss_eval().item())
            emb_list = emb_optim_list
        # === end Stage 2 ===

        pred_motions = vq_model.vqvae.forward_decoder_from_quantized_codes(emb_list)
        pred_motions = pred_motions.squeeze(2).permute(0, 2, 1)

        pred_motions_denorm = pred_motions * _std + _mean
        pred_motions_denorm = recover_from_ric(pred_motions_denorm.float(), self.opt.joints_num)

        _tr_done = getattr(self, "_ttt_trace", None)
        if _tr_done is not None:
            _tr_done["complete"] = True     # dumped by the caller's finally; partial stays False

        return pred_motions_denorm, pred_motions


# Backward-compatible alias so eval code that imports ControlTransformerTConcat works
ControlTransformerTConcat = ControlTransformerTConcatV4

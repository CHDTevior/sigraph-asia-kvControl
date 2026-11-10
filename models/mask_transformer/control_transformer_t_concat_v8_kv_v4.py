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
        print(f"[v8-v4] Control encoder: {enc_params/1e6:.1f}M, "
              f"KV projections (rank={kv_rank}): {kv_params/1e6:.1f}M, "
              f"Total trainable: {(enc_params+kv_params)/1e6:.1f}M")

    # ============= train/eval mode =============

    def ctrl_train(self):
        freeze_block(self)
        unfreeze_block(self.encoder_control)
        unfreeze_block(self.kv_down)
        unfreeze_block(self.kv_up_k)
        unfreeze_block(self.kv_up_v)

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
            text_memory = self.text_adapter(text_memory)         # [B, L, D]  (v4 text adapter)
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

    def _ttt_loss(self, pred_motions_denorm, global_joint, global_joint_mask):
        expanded_mask = global_joint_mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        squared_error = (pred_motions_denorm - global_joint) ** 2
        masked_error = squared_error * expanded_mask
        valid_counts = expanded_mask.sum(dim=(1, 2, 3)).clamp(min=1)
        per_sample_loss = masked_error.sum(dim=(1, 2, 3)) / valid_counts
        return per_sample_loss.mean()

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
            if each_iter > 0 and global_joint is not None and global_joint[global_joint_mask].sum() != 0:
                # Dynamic schedule: more TTT at later steps (like MaskControl)
                if getattr(self.opt, 'ttt_dynamic', False):
                    _n_iter = (_step_idx + 1) * each_iter
                else:
                    _n_iter = each_iter
                logits_bsv = logits_bsv.detach().requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    [logits_bsv], lr=lr, betas=(0.5, 0.9), weight_decay=1e-6
                )
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

                        optimizer.zero_grad()
                        loss_ttt.backward()
                        optimizer.step()

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

        pred_motions = vq_model.vqvae.forward_decoder_from_quantized_codes(emb_list)
        pred_motions = pred_motions.squeeze(2).permute(0, 2, 1)

        pred_motions_denorm = pred_motions * _std + _mean
        pred_motions_denorm = recover_from_ric(pred_motions_denorm.float(), self.opt.joints_num)

        return pred_motions_denorm, pred_motions


# Backward-compatible alias so eval code that imports ControlTransformerTConcat works
ControlTransformerTConcat = ControlTransformerTConcatV4

"""
v5_v4 = v5_bk's classical parallel-branch ControlNet adapted to the v4 backbone
(dense cross-attention + CLIP text adapter), for matched-backbone ablation
against KV-Control. NOT KV injection — preserves v5_bk's parallel deepcopy
control transformer + per-Q first_zero_linear + per-layer mid_zero_linear
additive residual; v4 text path (text_seq_proj + text_adapter + text_cross_attn)
is preserved on the main branch and frozen with the rest of the base.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn.attention import sdpa_kernel, SDPBackend
from einops import repeat

# CHANGE: inherit from v4 instead of v1 (matches v8_kv_v4_noqres pattern)
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


def return_all_layers(trans, src, mask=None, src_key_padding_mask=None):
    """Run a TransformerEncoder layer-by-layer, return list of per-layer outputs."""
    output = src
    all_layers = []
    for mod in trans.layers:
        output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        all_layers.append(output)
    if trans.norm is not None:
        output = trans.norm(output)
    return all_layers


def forward_with_parallel_control_and_crossattn(
    trans_layers,
    trans_norm,
    src,
    list_of_controlnet_output,
    cross_attn_blocks,
    cross_attn_layer_ids,
    text_memory,
    text_pad_mask,
    cond_keep,
    src_key_padding_mask=None,
    factorized_mask=None,
):
    """
    Forward through the FROZEN base transformer layer-by-layer, adding the
    pre-projected control branch residual after each base layer's output, and
    applying text cross-attention at designated layers. Mirrors v5_bk's
    forward_with_condition + v4's per-layer cross-attention insertion.

    Args:
        trans_layers:           iterable of TransformerEncoderLayer modules (base)
        trans_norm:             optional final LayerNorm of the base encoder
        src:                    [S+1, B, D] base input (cond token prepended)
        list_of_controlnet_output:
                                list of L tensors [S+1, B, D] (already passed
                                through self.mid_zero_linear[i])
        cross_attn_blocks:      nn.ModuleList of TextCrossAttentionBlock
        cross_attn_layer_ids:   set of layer indices that have cross-attention
        text_memory:            [L_text, B, D] adapter-refined CLIP tokens (or None)
        text_pad_mask:          [B, L_text] True=padding (or None)
        cond_keep:              [B, 1] 1.0=keep / 0.0=drop (CFG)
        src_key_padding_mask:   [B, S+1] True=pad
        factorized_mask:        [S+1, S+1] bool mask (True=blocked) or None
    """
    output = src
    cross_idx = 0
    for layer_idx, mod in enumerate(trans_layers):
        # Optional factorized self-attention mask on odd layers (v4.8/v4.9)
        use_factorized = factorized_mask is not None and (layer_idx % 2 == 1)
        src_mask = factorized_mask if use_factorized else None

        # Base self-attention + FFN block
        output = mod(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Additive control residual from the parallel branch (v5_bk semantics)
        ctrl_residual = list_of_controlnet_output[layer_idx]
        output = output + ctrl_residual

        # Text cross-attention at designated layers (after self-attn + ctrl injection)
        if layer_idx in cross_attn_layer_ids:
            if text_memory is not None:
                output = cross_attn_blocks[cross_idx](
                    output, text_memory, text_pad_mask, cond_keep=cond_keep
                )
            cross_idx += 1

    if trans_norm is not None:
        output = trans_norm(output)

    return output


# ============================================================
# Model
# ============================================================

class ControlTransformerTConcatV5V4(MaskTransformer):
    """v5_v4: classical parallel-branch ControlNet on top of the v4 base.

    Inherits from v4 MaskTransformer which provides:
      - CLIP text adapter (CLIPTextAdapter)
      - Dense text cross-attention (TextCrossAttentionBlock at every cross_attn_interval layers)
      - encode_text_with_seq (pooled + sequence tokens)
      - forward_with_cond_scale (passes text_seq/text_pad_mask through)

    Adds (matching v5_bk):
      - encoder_control: 1D conv encoder, 196 frames -> 49 tokens
      - first_zero_linear: per-Q (per body part) zero-init Linear projection
      - seqTransEncoder_control: deepcopy of the frozen base self-attention encoder
      - mid_zero_linear: per-base-layer zero-init Linear (gates the residual)
      - additive injection of control output into the main branch, layer by layer
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

        # ---- load pretrained v4 base transformer ----
        # IMPORTANT: load BEFORE deepcopy so seqTransEncoder_control inherits the
        # trained base weights as initialization (v5_bk semantics).
        if trans_path != "":
            ckpt = torch.load(trans_path, map_location=opt.device)
            model_key = "t2m_transformer" if "t2m_transformer" in ckpt else "trans"
            missing_keys, unexpected_keys = self.load_state_dict(
                ckpt[model_key], strict=False
            )
            # Strict assertion on unexpected_keys: v4 ckpt must align 1:1 with v4 backbone.
            # Any unexpected key indicates a config mismatch (e.g. different
            # num_layers / latent_dim / cross_attn_interval / text_adapter_layers)
            # and would silently drop loaded weights — fail loudly.
            assert unexpected_keys == [], (
                f"[v5_v4] Unexpected keys when loading v4 ckpt — config mismatch:\n"
                f"  trans_path: {trans_path}\n"
                f"  unexpected_keys (first 20): {unexpected_keys[:20]}"
            )
            # Missing keys are EXPECTED for control-branch params (encoder_control,
            # first_zero_linear, mid_zero_linear, seqTransEncoder_control).
            print(
                f"[v5_v4] Loaded v4 T-concat Transformer from {trans_path}\n"
                f"  unexpected_keys: 0 (assertion passed)\n"
                f"  missing_keys: {len(missing_keys)} (expected: ControlNet branch)"
            )
            # Freeze entire backbone (every parameter that loaded). Control modules
            # built below will be unfrozen via ctrl_train().
            freeze_block(self)

        # ============= v5 parallel-branch ControlNet modules =============

        # Per-quantizer (body part) control projection: each part learns its own
        # control strength. Zero-init means trajectory-irrelevant parts (head, neck)
        # start at zero and only root/leg parts learn strong projections.
        self.first_zero_linear = nn.ModuleList(
            [nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.Q)]
        )

        # Per-base-layer zero-init linear that gates the residual flowing back into
        # the main branch (one per base layer).
        self.mid_zero_linear = nn.ModuleList(
            [nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)]
        )

        # Control encoder: 196 frames -> 49 tokens via 1D conv
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

        # Control branch encoder: deepcopy of the (now frozen + loaded) base
        # self-attention stack. NOTE: the deepcopy intentionally does NOT include
        # text_seq_proj / text_adapter / text_cross_attn — those live on the main
        # branch only. The control branch sees only motion+cond tokens (no text
        # cross-attn columns), matching v5_bk semantics.
        self.seqTransEncoder_control = copy.deepcopy(self.seqTransEncoder)

        # Zero-init the gating linears so the control residual starts at zero
        # (identity initialization — model behaves like the frozen base at step 0).
        def init_zero(m):
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.first_zero_linear.apply(init_zero)
        self.mid_zero_linear.apply(init_zero)

        # Per-codebook mean embeddings for masked positions during ctrl signal computation
        self.mask_emb_vq_list = [
            vq_model.vqvae.quantizers[i].codebook.mean(0) for i in range(self.Q)
        ]

        # ---- report ----
        enc_params = sum(p.numel() for p in self.encoder_control.parameters())
        first_params = sum(p.numel() for p in self.first_zero_linear.parameters())
        mid_params = sum(p.numel() for p in self.mid_zero_linear.parameters())
        ctrl_branch_params = sum(p.numel() for p in self.seqTransEncoder_control.parameters())
        ctrl_total = enc_params + first_params + mid_params + ctrl_branch_params
        print(
            f"[v5_v4] Parallel-branch ControlNet on v4 backbone\n"
            f"  encoder_control: {enc_params/1e6:.2f}M\n"
            f"  first_zero_linear (Q={self.Q}): {first_params/1e6:.2f}M\n"
            f"  mid_zero_linear (L={self.num_layers}): {mid_params/1e6:.2f}M\n"
            f"  seqTransEncoder_control (deepcopy of base, L={self.num_layers}): "
            f"{ctrl_branch_params/1e6:.2f}M\n"
            f"  Total trainable: {ctrl_total/1e6:.2f}M"
        )

        # Must be called after all submodules are attached
        self.ctrl_train()

    # ============= train/eval mode =============

    def ctrl_eval(self):
        self.eval()

    def ctrl_train(self):
        # Freeze entire backbone (sets self.training=False recursively)
        freeze_block(self)
        # Restore TOP-LEVEL training flag (non-recursive) so future mode-dependent
        # logic on the wrapper still functions, while frozen submodules keep
        # eval semantics (dropout off). Matches v8_kv_v4_noqres.ctrl_train().
        self.training = True
        # Unfreeze all four control modules
        unfreeze_block(self.seqTransEncoder_control)
        unfreeze_block(self.encoder_control)
        unfreeze_block(self.first_zero_linear)
        unfreeze_block(self.mid_zero_linear)

    # ============= utils =============

    def inv_transform(self, data):
        assert self.std is not None and self.mean is not None
        return data * self.std + self.mean

    def _build_control_mask(self, m_length, pose, device):
        """Build random control joint mask over raw 196-frame space."""
        B = pose.shape[0]
        num_frames = pose.shape[1]

        if self.control in ("trajectory", "random"):
            rand_length = (
                torch.rand(m_length.shape, device=device) * 196
            ).round().clamp(min=1, max=196)

            all_len_mask = lengths_to_mask(m_length.to(device), num_frames)
            batch_randperm = torch.rand((B, num_frames), device=device)
            batch_randperm[~all_len_mask] = 1
            batch_randperm = batch_randperm.argsort(dim=-1)
            frame_mask = (batch_randperm < rand_length.unsqueeze(-1)) & all_len_mask

            if self.control == "trajectory":
                joint_mask = repeat(frame_mask, "b f -> b f j", j=self.opt.joints_num).clone()
                joint_mask[..., 1:] = False
            else:  # random
                joint_mask = torch.zeros(
                    (*frame_mask.shape, self.opt.joints_num), device=device, dtype=torch.bool
                )
                ctrl_joints = torch.tensor(control_joint_ids, device=device)
                rand_idx = torch.randint(len(ctrl_joints), (B,), device=device)
                joint_mask[
                    torch.arange(B, device=device), :, ctrl_joints[rand_idx]
                ] = frame_mask

        elif self.control == "cross":
            rand_length = (
                torch.rand(m_length.shape, device=device) * m_length.to(device) * len(control_joint_ids)
            ).round().clamp(min=1)

            all_len_mask = lengths_to_mask(m_length.to(device), num_frames)
            batch_randperm = torch.rand(
                (*all_len_mask.shape, len(control_joint_ids)), device=device
            )
            batch_randperm[~all_len_mask] = 1
            batch_randperm = batch_randperm.reshape((B, -1)).argsort(dim=-1).reshape((*all_len_mask.shape, -1))
            _mask = (batch_randperm < rand_length.unsqueeze(-1).unsqueeze(-1)) & all_len_mask.unsqueeze(-1)

            joint_mask = torch.zeros(
                (*pose.shape[:2], self.opt.joints_num), device=device, dtype=torch.bool
            )
            joint_mask[..., control_joint_ids] = _mask
        else:
            raise NotImplementedError(f"{self.control} is not implemented yet!")

        return joint_mask  # [B, 196, 22]

    def _build_ctrl_cond(self, global_joint, pred_joints, global_joint_mask, B):
        """Build ctrlNet_cond from GT and predicted joints."""
        ctrlNet_cond = (global_joint - pred_joints) * global_joint_mask.unsqueeze(-1)
        ctrlNet_cond2 = global_joint * global_joint_mask.unsqueeze(-1)

        if self.control == "trajectory":
            ctrlNet_cond = ctrlNet_cond[:, :, 0]    # [B, 196, 3]
            ctrlNet_cond2 = ctrlNet_cond2[:, :, 0]  # [B, 196, 3]
        else:
            ctrlNet_cond = ctrlNet_cond[..., control_joint_ids, :].reshape(B, -1, len(control_joint_ids) * 3)
            ctrlNet_cond2 = ctrlNet_cond2[..., control_joint_ids, :].reshape(B, -1, len(control_joint_ids) * 3)

        return torch.cat([ctrlNet_cond, ctrlNet_cond2], dim=-1)  # [B, 196, 6 or 36]

    def _decode_motion_from_local_ids(self, local_ids_bt6):
        """Decode motion from local code indices [B, T, Q]."""
        x_q_list = self.vq_model.vqvae.get_x_quantized_from_x_ids(local_ids_bt6)
        pred_motions = self.vq_model.vqvae.forward_decoder_from_quantized_codes(x_q_list)
        pred_motions = pred_motions.permute(0, 2, 3, 1).squeeze(1)
        return pred_motions

    def _motion_to_joints(self, pred_motions):
        """Denormalize and convert to joint positions."""
        pred_denorm = pred_motions * self.std + self.mean
        return recover_from_ric(pred_denorm.float(), self.opt.joints_num)

    # ============= trans_forward with parallel-branch control + text cross-attn =============

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
        Override trans_forward to inject parallel-branch ControlNet residual on
        the main path while preserving v4's text conditioning (text adapter +
        cross-attention).

        When ctrlNet_cond is None, falls back to v4's standard trans_forward
        (inherited via super()). When ctrlNet_cond is provided, runs the
        parallel control transformer and injects per-layer additive residual.

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
            return super().trans_forward(
                motion_emb, cond, padding_mask, force_mask,
                text_seq=text_seq, text_pad_mask=text_pad_mask,
            )

        bs = cond.shape[0]
        cond_keep = self._get_cond_mask(bs, cond.device, force_mask)

        # ---- pooled condition (same as v2/v4) ----
        cond = cond * cond_keep

        # ---- motion embedding + cond token + factorized PE ----
        motion_emb = self._inject_quantizer_token_identity(motion_emb)
        x = self.input_process(motion_emb)              # [S, B, D]
        cond_token = self.cond_emb(cond).unsqueeze(0)   # [1, B, D]
        x = self._add_factorized_position_encoding(x)
        xseq = torch.cat([cond_token, x], dim=0)        # [S+1, B, D]

        padding_mask_with_cond = torch.cat(
            [torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1
        )                                                # [B, S+1]

        # ---- text memory for cross-attention (v4 path: proj + adapter + mask) ----
        if text_seq is not None and self._has_cross_attn:
            text_memory = self.text_seq_proj(text_seq)                     # [B, L, D]
            text_memory = self.text_adapter(text_memory,
                                            src_key_padding_mask=text_pad_mask)  # [B, L, D]
            text_memory = text_memory * cond_keep.unsqueeze(-1)            # CFG zeroing
            text_memory = text_memory.permute(1, 0, 2)                     # [L, B, D]
        else:
            text_memory = None

        # ---- v4.8/v4.9 factorized self-attention mask (odd layers only) ----
        if self.factorized_attn != "none":
            factorized_mask = self._build_factorized_attn_mask(xseq.shape[0], xseq.device)
        else:
            factorized_mask = None

        # ========= Control branch =========
        # Encode control signal: [B, 196, 6/36] -> [B, D, 49] -> [49, B, D]
        ctrl_feat = self.encoder_control(ctrlNet_cond.permute(0, 2, 1))    # [B, D, T_ctrl]
        ctrl_feat = ctrl_feat.permute(2, 0, 1)                              # [T_ctrl, B, D]
        T_ctrl = ctrl_feat.shape[0]

        # Per-quantizer (body part) zero-init projection. Output interleaved as
        # [t0q0, t0q1, ..., t0q(Q-1), t1q0, ...] to match motion-token layout.
        ctrl_per_part = torch.stack(
            [self.first_zero_linear[q](ctrl_feat) for q in range(self.Q)], dim=1
        )                                                                   # [T_ctrl, Q, B, D]
        ctrl_feat = ctrl_per_part.reshape(T_ctrl * self.Q, ctrl_feat.shape[1], -1)
                                                                            # [S, B, D]

        # Inject into the SAME pre-block sequence (cond token + motion tokens).
        # Prepend a zero so the cond token position receives no control signal.
        # NOTE (v5_bk semantics): we add to the motion-token positions only;
        # the cond-token row (index 0) gets zero injection.
        control_input = xseq + torch.cat(
            [torch.zeros_like(cond_token), ctrl_feat], dim=0
        )                                                                   # [S+1, B, D]

        # Run the parallel control transformer over the same masked sequence.
        # NOTE: the control branch does NOT use factorized attention masks; it
        # mirrors v5_bk which only applies the standard self-attention pattern.
        # (Even/odd factorization is a v4.8/v4.9 main-branch choice; replicating
        # it on the control branch would require carrying the same mask there
        # too — which v5_bk did not do, so we don't either, to keep the ablation
        # strictly faithful to the reference design.)
        # Force MATH SDPA backend: PyTorch 2.8 mem-efficient SDPA backward
        # produces NaN on the trainable parallel branch (Codex 019dd602).
        with sdpa_kernel(SDPBackend.MATH):
            control_output_list = return_all_layers(
                self.seqTransEncoder_control,
                control_input,
                src_key_padding_mask=padding_mask_with_cond,
            )                                                                   # L * [S+1, B, D]

        # Per-layer zero-init projection (acts as gate; identity at init).
        for i in range(self.num_layers):
            control_output_list[i] = self.mid_zero_linear[i](control_output_list[i])

        # ========= Main branch with control residual + text cross-attention =========
        output = forward_with_parallel_control_and_crossattn(
            trans_layers=self.seqTransEncoder.layers,
            trans_norm=self.seqTransEncoder.norm,
            src=xseq,
            list_of_controlnet_output=control_output_list,
            cross_attn_blocks=self.text_cross_attn if self._has_cross_attn else None,
            cross_attn_layer_ids=self.cross_attn_layer_ids if self._has_cross_attn else set(),
            text_memory=text_memory,
            text_pad_mask=text_pad_mask,
            cond_keep=cond_keep,
            src_key_padding_mask=padding_mask_with_cond,
            factorized_mask=factorized_mask,
        )                                                                   # [S+1, B, D]

        # Drop cond token, project to logits
        out = output[1:]                                                    # [S, B, D]

        if self.per_q_head:
            S_out, B_out, D_out = out.shape
            T_out = S_out // self.Q
            out_tqbd = out.view(T_out, self.Q, B_out, D_out)
            logits_list = []
            for q in range(self.Q):
                lq = self.output_heads[q](out_tqbd[:, q, :, :])             # [B, vocab, T]
                logits_list.append(lq)
            logits = torch.stack(logits_list, dim=3).reshape(B_out, self.vocab, T_out * self.Q)
        else:
            logits = self.output_head(out)                                  # [B, vocab, S]
        return logits

    # ============= forward (training) =============

    def forward(self, y, m_lens, pose):
        """
        Training forward pass. Trainer signature: model(conds, m_lens, motion).

        Returns: (loss_emb, ce_loss, pred_id, acc, loss_tta)
        """
        device = next(self.parameters()).device
        m_length = m_lens
        bs = len(y)

        # VQ encode (no_grad)
        with torch.no_grad():
            pose_4d = pose.float().permute(0, 2, 1).unsqueeze(2)            # [B, 263, 1, T]
            code_idx = self.vq_model.get_code_idx(pose_4d)                  # [B, Q, T]
        code_idx = code_idx.to(device)

        # Flatten to local sequence: interleave [t0q0, t0q1, ..., t1q0, ...]
        code_idx_bt6 = code_idx.permute(0, 2, 1).contiguous()               # [B, T, Q]
        T_tok = code_idx_bt6.shape[1]
        local_ids = code_idx_bt6.reshape(bs, T_tok * self.Q)

        m_tok_lens = (m_length.float() / 4).ceil().long()                   # frame -> token
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

        # NOTE: v5_bk does NOT use mask_2d_hybrid, and per the user's KV training
        # rule the ControlNet stage does not enable hybrid 2D masking either.

        # Build target and input
        target = local_ids.clone()
        target[~is_mask] = -100
        target[padding_mask] = -100

        # 88% mask, 2% real, 10% random (matches v8_v4)
        r = torch.rand_like(local_ids.float())
        x_ids = local_ids.clone()
        x_ids[is_mask & (r < 0.88)] = self.mask_id
        x_ids[is_mask & (r >= 0.88) & (r < 0.90)] = local_ids[is_mask & (r >= 0.88) & (r < 0.90)]
        noise_ids = torch.randint_like(x_ids, 0, self.K)
        x_ids[is_mask & (r >= 0.90)] = noise_ids[is_mask & (r >= 0.90)]

        x_emb = self.token_emb(x_ids)

        # Build control signal
        with torch.no_grad():
            ids_safe = x_ids.clone()
            overflow = (ids_safe >= self.K) | (ids_safe < 0)
            ids_safe[overflow] = 0
            local_bt6 = ids_safe.view(bs, T_tok, self.Q)
            pred_m = self._decode_motion_from_local_ids(local_bt6)
            pred_joints = self._motion_to_joints(pred_m)

            # GT joints
            gt_denorm = pose * self.std + self.mean
            gt_joints = recover_from_ric(gt_denorm.float(), self.opt.joints_num)

            # Random control mask (matches v8_v4 distribution)
            global_joint_mask = torch.zeros(
                (bs, pose.shape[1], self.opt.joints_num), device=device, dtype=torch.bool
            )
            import random
            import numpy as np
            for i in range(bs):
                dens = random.choice([1, 2, 5, 49, 196])
                if dens in [1, 2, 5]:
                    rand_dens = dens
                else:
                    rand_dens = int(m_length[i] * (dens / 196))
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

        # CHANGE (v4 path): encode_text_with_seq returns (pooled, text_seq, text_pad_mask)
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
        )                                                                    # [B, vocab, S]

        # CE loss
        ce_loss = F.cross_entropy(logits, target, ignore_index=-100)

        # Trajectory control loss — differentiable soft decode
        padding_mask_t = ~lengths_to_mask(m_tok_lens, T_tok)
        logits_bsv = logits.permute(0, 2, 1)                                # [B, S, vocab]
        probs_local = F.softmax(logits_bsv, dim=-1).view(bs, T_tok, self.Q, self.K)
        emb_list = []
        for q in range(self.Q):
            cb = self.vq_model.vqvae.quantizers[q].codebook                  # [K, C]
            emb_q = probs_local[:, :, q, :] @ cb                             # [B, T_tok, C]
            emb_q = emb_q.masked_fill(padding_mask_t.unsqueeze(-1), 0.0)
            emb_list.append(emb_q.permute(0, 2, 1))                          # [B, C, T_tok]

        pred_motions2 = self.vq_model.vqvae.forward_decoder_from_quantized_codes(emb_list)
        pred_motions2 = pred_motions2.squeeze(2).permute(0, 2, 1)            # [B, 196, 263]
        pred_joints2 = self._motion_to_joints(pred_motions2)

        loss_traj = F.l1_loss(
            pred_joints2[global_joint_mask],
            gt_joints[global_joint_mask],
            reduction="mean",
        )
        acc = (logits.argmax(dim=1) == target).float()
        acc = acc[target != -100].mean()

        # Trainer expects 5 values: (loss_emb, ce_loss, pred_ids, acc, loss_tta)
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

        # CHANGE (v4 path): encode_text_with_seq instead of pooled-only encode_text
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

            # Pass text_seq + text_pad_mask through forward_with_cond_scale (v4 path)
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
                with torch.enable_grad():
                    for _ttt_step in range(_n_iter):
                        logits_btqk = logits_bsv.view(batch_size, seq_len, self.Q, self.K)
                        emb_list = []
                        for q in range(self.Q):
                            cb = vq_model.vqvae.quantizers[q].codebook
                            if getattr(self.opt, 'ttt_gumbel', False):
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

        # === Stage 2: post-generation embedding optimization (MaskControl-style) ===
        last_iter = control_opt.get("iter", 0) if control_opt else 0
        last_lr = control_opt.get("lr", 6e-2) if control_opt else 6e-2
        if last_lr is None:
            last_lr = 6e-2

        if last_iter > 0 and global_joint is not None and global_joint_mask is not None and global_joint_mask.any():
            emb_optim_list = [e.clone().detach().requires_grad_(True) for e in emb_list]
            optimizer = torch.optim.AdamW(
                emb_optim_list, lr=last_lr, betas=(0.5, 0.9), weight_decay=1e-6
            )
            with torch.enable_grad():
                for _step in range(last_iter):
                    pred_motions_s2 = vq_model.vqvae.forward_decoder_from_quantized_codes(emb_optim_list)
                    pred_motions_s2 = pred_motions_s2.squeeze(2).permute(0, 2, 1)
                    pred_motions_s2_denorm = pred_motions_s2 * _std + _mean
                    pred_joints_s2 = recover_from_ric(
                        pred_motions_s2_denorm.float(), self.opt.joints_num
                    )

                    loss_s2 = self._ttt_loss(pred_joints_s2, global_joint, global_joint_mask)
                    if abitary_func is not None:
                        loss_s2 = loss_s2 + abitary_func(pred_joints_s2)

                    if loss_s2.item() < 1e-10:
                        break

                    optimizer.zero_grad()
                    loss_s2.backward()
                    optimizer.step()

                    if _step == 0 or (_step + 1) % max(1, last_iter // 10) == 0 or _step == last_iter - 1:
                        print(f"  [Stage2] step {_step+1}/{last_iter}  loss={loss_s2.item():.6f}")
            emb_list = emb_optim_list
        # === end Stage 2 ===

        pred_motions = vq_model.vqvae.forward_decoder_from_quantized_codes(emb_list)
        pred_motions = pred_motions.squeeze(2).permute(0, 2, 1)

        pred_motions_denorm = pred_motions * _std + _mean
        pred_motions_denorm = recover_from_ric(pred_motions_denorm.float(), self.opt.joints_num)

        return pred_motions_denorm, pred_motions


# Backward-compatible alias so eval/loader code that imports the canonical class
# name still works (mirrors v8_kv_v4_noqres pattern).
ControlTransformerTConcat = ControlTransformerTConcatV5V4

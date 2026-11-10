"""KV-Context ControlNet for c_concat v4 MaskTransformer.

Direct port of control_transformer_t_concat_v8_kv_v4_noqres.py to c_concat
backbone (channel-concat input, per-Q output heads, S=T+1=50, 1-D positional
encoding, NO factorized attention, NO 2D pos emb, NO sequence-axis token
unpacking).

Inheritance:
  models.mask_transformer.transformer.MaskTransformer (c_concat v1, parent)
    |
    +— transformer_c_concat_v4.MaskTransformer (adds v4 dual-stream text path)
        |
        +— THIS class (adds KV-Context ControlNet adapter modules)

Adapter parameters (same as T-concat noqres variant):
  - encoder_control: 1D conv 196→49 tokens (Encoder from models/vq/encdec.py)
  - kv_down/up_k/up_v: per-layer low-rank KV projections (D → r → D)
  - ctrl_attn_bias: per-layer learnable scalar bias (init=-5 so zero-init
                    KV doesn't steal softmax mass at start)

Differences from T-concat KV:
  - Sequence length S = T + 1 = 50 (no Q axis flattening)
  - factorized_mask is always None (c_concat has no factorized attention)
  - Per-Q output heads via self.output_processes[i] (not self.output_head/output_heads)
  - Per-T BERT masking shared across all 6 codebooks (vs per-(T,Q) in T-concat)
  - forward() uses c_concat per-codebook embedding via self.token_embs[i] +
    channel-concat to motion_emb [B, T, 6*code_dim]
  - CE loss via cal_performance2 (per-Q logits vs labels_list)

Everything else (KV adapter logic, TTT generation, stage-2 emb optimization,
ctrl_train/ctrl_eval, ctrlNet_cond construction) follows the T-concat noqres
template exactly.

Verified design points (codex-pending):
  1. Inherits from c_concat_v4 (NOT T-concat) — paired ablation
  2. KV injection point: inside seqTransEncoder.layers self-attention via
     forward_with_kv_context_and_crossattn helper (re-used from noqres)
  3. text adapter / cross-attn untouched (matches v4 dual-stream path)
  4. trainer interface: forward returns 5-tuple (loss_emb=0, ce_loss, None, acc, loss_traj)
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from einops import repeat

# Inherit from c_concat_v4 base (NOT T-concat)
from models.mask_transformer.transformer_c_concat_v4 import MaskTransformer
# Re-use the layer-by-layer KV injection helper from T-concat noqres (it accepts
# factorized_mask=None and handles that case correctly — see helper docstring)
from models.mask_transformer.control_transformer_t_concat_v8_kv_v4_noqres import (
    forward_with_kv_context_and_crossattn,
    freeze_block,
    unfreeze_block,
)
from models.vq.encdec import Encoder
from models.mask_transformer.tools import (
    lengths_to_mask, uniform, cosine_schedule, eval_decorator,
    top_k, gumbel_sample, get_mask_subset_prob, gumbel_noise,
    cal_performance2,
)
from utils.motion_process import recover_from_ric
from utils.metrics import control_joint_ids


# ============================================================
# ControlTransformerKVCConcatV4
# ============================================================

class ControlTransformerKVCConcatV4(MaskTransformer):
    """KV-Context ControlNet on c_concat v4 backbone.

    Inherits c_concat_v4 dual-stream text path (CLIPTextAdapter + sparse
    text cross-attention every cross_attn_interval layers, with cond_keep CFG
    mask). Adds the KV adapter modules (encoder_control + per-layer KV projs)
    used in the T-concat noqres paper-main variant.
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
        # c_concat default seq length (matches transformer_c_concat_v4.generate)
        self.cconcat_seq_len = 49

        # ---- load pretrained c_concat_v4 base transformer ----
        if trans_path != "":
            # Codex audit 019deb51 fix: load to CPU, let .to(device) move final model
            # (avoids CUDA pressure during init before model fully constructed)
            ckpt = torch.load(trans_path, map_location="cpu")
            model_key = "t2m_transformer" if "t2m_transformer" in ckpt else "trans"
            missing_keys, unexpected_keys = self.load_state_dict(
                ckpt[model_key], strict=False
            )
            freeze_block(self)
            print(
                f"[KV-c_concat-v4] Loaded c_concat_v4 base Transformer from {trans_path}\n"
                f"  missing keys: {len(missing_keys)} (expected: encoder_control + kv_down/up + ctrl_attn_bias)\n"
                f"  unexpected keys: {len(unexpected_keys)}"
            )

        # ============= KV-Context modules (identical to T-concat noqres) =============

        # Control encoder: 196 frames -> 49 tokens (down_t=2, stride_t=2 → /4)
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

        # Per-codebook mean embeddings for masked positions (matches noqres)
        # NOTE: Q is c_concat default 6 (matches token_embs ModuleList length)
        self.Q = 6
        # Codex audit 019deb51: detach + clone for hygiene (avoid VQ graph reference)
        self.mask_emb_vq_list = [
            vq_model.vqvae.quantizers[i].codebook.mean(0).detach().clone()
            for i in range(self.Q)
        ]

        # Print param counts
        enc_params = sum(p.numel() for p in self.encoder_control.parameters())
        kv_params = sum(
            p.numel() for p in list(self.kv_down.parameters())
            + list(self.kv_up_k.parameters())
            + list(self.kv_up_v.parameters())
        )
        bias_params = sum(p.numel() for p in self.ctrl_attn_bias.parameters())
        print(
            f"[KV-c_concat-v4] Control encoder: {enc_params/1e6:.2f}M, "
            f"KV (rank={kv_rank}): {kv_params/1e6:.2f}M, "
            f"bias: {bias_params}, "
            f"Total trainable adapter: {(enc_params+kv_params+bias_params)/1e6:.2f}M"
        )

    # ============= train/eval mode (same as noqres) =============

    def ctrl_train(self):
        freeze_block(self)
        # BUG FIX (preserved from noqres): freeze_block calls self.eval(), which
        # recursively sets self.training=False on ALL modules. We need
        # self.training=True for cond_dropout to work, but the frozen backbone
        # submodules should keep eval semantics (dropout off). Set non-recursively.
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

    # ============= control signal building (same as noqres) =============

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
            ctrlNet_cond = ctrlNet_cond[..., control_joint_ids, :].reshape(
                B, -1, len(control_joint_ids) * 3
            )
            ctrlNet_cond2 = ctrlNet_cond2[..., control_joint_ids, :].reshape(
                B, -1, len(control_joint_ids) * 3
            )

        return torch.cat([ctrlNet_cond, ctrlNet_cond2], dim=-1)

    def _decode_motion_from_local_ids(self, local_ids_bt6):
        """Decode VQ ids [B, T, 6] -> motion [B, T_frame=196, 263]."""
        x_q_list = self.vq_model.vqvae.get_x_quantized_from_x_ids(local_ids_bt6)
        pred_motions = self.vq_model.vqvae.forward_decoder_from_quantized_codes(x_q_list)
        # forward_decoder_from_quantized_codes returns [B, 263, 1, T_frame]
        # squeeze H axis and permute to [B, T_frame, 263]
        pred_motions = pred_motions.squeeze(2).permute(0, 2, 1)
        return pred_motions

    def _encode_ctrl_to_kv(self, ctrlNet_cond):
        """Encode control signal to per-layer KV pairs.

        ctrlNet_cond: [B, 196, 6/36] (6 for trajectory, 36 = len(control_joint_ids)*2*3 for cross/random)
        Returns: list of (K_ctrl, V_ctrl) per layer, each [T_ctrl=49, B, D]
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

    # ============= trans_forward (c_concat layout + KV injection + cross-attn) =============

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
        """Override c_concat_v4 trans_forward to inject control KV.

        When ctrlNet_cond is None, falls back to c_concat_v4's standard
        trans_forward (inherited via super()). When ctrlNet_cond is provided,
        uses forward_with_kv_context_and_crossattn for layer-by-layer forward
        with both KV injection and v4 cross-attention.

        Args:
            motion_emb:    [B, T, 6*code_dim] channel-concat per-codebook embeddings
            cond:          [B, clip_dim]      pooled CLIP embedding
            padding_mask:  [B, T]             True = pad (per-frame)
            force_mask:    bool               force unconditional (for CFG)
            text_seq:      [B, L, clip_dim]   CLIP sequence tokens (optional)
            text_pad_mask: [B, L]             True = padding (optional)
            ctrlNet_cond:  [B, 196, 6/36]     control signal (optional)

        Returns:
            logits: [B, 6, K, T] per-codebook logits (matches c_concat_v4)
        """
        if not self.ctrl_net or ctrlNet_cond is None:
            # Fall back to c_concat_v4's trans_forward (with text adapter + cross-attn)
            return super().trans_forward(
                motion_emb, cond, padding_mask, force_mask=force_mask,
                text_seq=text_seq, text_pad_mask=text_pad_mask,
            )

        bs = cond.shape[0]
        cond_keep = self._get_cond_mask(bs, cond.device, force_mask)

        # ---- pooled condition (gated by cond_keep, matches c_concat_v4) ----
        cond = cond * cond_keep

        # (B, T, 6*code_dim) -> (T, B, latent_dim) via input_process
        x = self.input_process(motion_emb)
        cond_token = self.cond_emb(cond).unsqueeze(0)  # [1, B, D]

        x = self.position_enc(x)
        xseq = torch.cat([cond_token, x], dim=0)  # [T+1, B, D]

        # padding_mask: [B, T+1] — first slot = cond, never masked
        padding_mask_with_cond = torch.cat(
            [torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1
        )

        # ---- text memory for cross-attention (v4 path: proj + adapter + mask) ----
        if text_seq is not None and self._has_cross_attn:
            text_memory = self.text_seq_proj(text_seq)  # [B, L, D]
            text_memory = self.text_adapter(
                text_memory, src_key_padding_mask=text_pad_mask
            )  # [B, L, D]
            text_memory = text_memory * cond_keep.unsqueeze(-1)  # zero dropped samples
            text_memory = text_memory.permute(1, 0, 2)  # [L, B, D]
        else:
            text_memory = None

        # ---- encode control to per-layer KV pairs ----
        ctrl_kv_list = self._encode_ctrl_to_kv(ctrlNet_cond)

        # ---- forward through base layers with KV injection + cross-attention ----
        # NOTE: factorized_mask=None (c_concat has no factorized attention)
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
            factorized_mask=None,
            ctrl_attn_bias_list=self.ctrl_attn_bias,
        )

        # Apply final encoder norm if present (matches nn.TransformerEncoder behavior)
        if (
            hasattr(self.seqTransEncoder, "norm")
            and self.seqTransEncoder.norm is not None
        ):
            output = self.seqTransEncoder.norm(output)

        # Drop the cond token row → [T, B, D]
        output = output[1:]

        # ---- per-Q output heads (c_concat layout: 6 heads, each [B, K, T]) ----
        logits_list = [head(output) for head in self.output_processes]
        logits = torch.stack(logits_list, dim=1)  # [B, 6, K, T]
        return logits

    # ============= forward (training) =============

    def forward(self, y, m_lens, pose):
        """Training forward pass. Signature matches trainer: model(conds, m_lens, motion).

        Returns 5-tuple: (loss_emb=0, ce_loss, None, acc, loss_traj)
        matching train_ctrlnet_t_concat_ddp.py expectation.
        """
        device = next(self.parameters()).device
        m_length = m_lens
        bs = len(y)

        vq_model = self.vq_model

        # ----- VQ encode (c_concat-style: returns [B, 6, T_tok]) -----
        with torch.no_grad():
            pose_4d = pose.float().permute(0, 2, 1).unsqueeze(2)  # [B, 263, 1, T]
            code_idx = vq_model.get_code_idx(pose_4d)  # [B, 6, T_tok]
        code_idx = code_idx.to(device)

        # Convert to per-codebook ids list [B, T_tok]
        Q = self.Q
        T_tok = code_idx.shape[2]
        code_idx_btq = code_idx.permute(0, 2, 1).contiguous()  # [B, T_tok, 6]
        ids_list = [code_idx_btq[..., i].clone() for i in range(Q)]

        m_tok_lens = (m_length.float() / 4).ceil().long()  # frame -> token length
        non_pad_mask = lengths_to_mask(m_tok_lens, T_tok).to(device)  # [B, T_tok]
        padding_mask = ~non_pad_mask  # True = pad

        # Set pad positions to pad_id
        for i in range(Q):
            ids_list[i] = torch.where(
                non_pad_mask, ids_list[i], torch.full_like(ids_list[i], self.pad_id)
            )

        # ----- Per-T BERT-style mask (c_concat: shared across 6 quantizers) -----
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)  # [B]
        num_token_masked = (m_tok_lens.float() * rand_mask_probs).round().clamp(min=1)  # [B]

        batch_randperm = torch.rand((bs, T_tok), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)  # [B, T_tok]
        mask &= non_pad_mask  # only mask non-pad positions

        # Build labels (per-codebook target ids) and noisy x_ids
        labels_list = [ids_list[i].clone() for i in range(Q)]
        x_ids_list = [ids_list[i].clone() for i in range(Q)]

        # 10% of masked → random token (per-codebook, c_concat semantics)
        mask_rid = get_mask_subset_prob(mask, 0.1)
        for i in range(Q):
            rand_ids = torch.randint(
                low=0, high=self.opt.num_tokens,
                size=x_ids_list[i].shape, device=device,
            )
            x_ids_list[i] = torch.where(mask_rid, rand_ids, x_ids_list[i])

        # 88% of remaining masked → mask_id
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        for i in range(Q):
            x_ids_list[i] = torch.where(
                mask_mid,
                torch.full_like(x_ids_list[i], self.mask_id),
                x_ids_list[i],
            )
        # remaining ~2% keep original token

        # ----- Channel-concat noisy embedding (c_concat layout) -----
        num_cb_list = [
            vq_model.vqvae.quantizers[i].codebook.shape[0] for i in range(Q)
        ]
        std = 0.1
        trans_emb_parts = []

        for i in range(Q):
            num_cb = num_cb_list[i]
            raw_ids = x_ids_list[i]  # [B, T_tok]
            safe_codes = raw_ids.clamp(min=0, max=num_cb - 1)
            one_hot = F.one_hot(safe_codes, num_classes=num_cb).float()
            noise = torch.randn_like(one_hot) * std
            noisy_one_hot = one_hot + noise
            part_emb = noisy_one_hot @ self.token_embs[i].weight[:num_cb]

            mask_pos = raw_ids == self.mask_id
            pad_pos = raw_ids == self.pad_id
            if mask_pos.any():
                part_emb[mask_pos] = self.token_embs[i].weight[self.mask_id]
            if pad_pos.any():
                part_emb[pad_pos] = self.token_embs[i].weight[self.pad_id]

            trans_emb_parts.append(part_emb)

        trans_emb_all = torch.cat(trans_emb_parts, dim=-1)  # [B, T_tok, 6*code_dim]

        # ----- Build control signal -----
        with torch.no_grad():
            # Decode current (partially masked) prediction to motion → joints
            local_ids_bt6 = torch.stack(x_ids_list, dim=-1)  # [B, T_tok, 6]
            local_safe = local_ids_bt6.clone()
            # Clamp out-of-range (mask_id, pad_id, random noise) for safe VQ decode
            for q in range(Q):
                num_cb = num_cb_list[q]
                slot = local_safe[..., q]
                overflow = (slot >= num_cb) | (slot < 0)
                slot[overflow] = 0
                local_safe[..., q] = slot
            pred_m = self._decode_motion_from_local_ids(local_safe)
            pred_joints = self._motion_to_joints(pred_m)

            # GT joints
            gt_denorm = pose * self.std + self.mean
            gt_joints = recover_from_ric(gt_denorm.float(), self.opt.joints_num)

            # Random control mask (same density schedule as noqres)
            global_joint_mask = torch.zeros(
                (bs, pose.shape[1], self.opt.joints_num),
                device=device, dtype=torch.bool,
            )
            import random
            import numpy as np
            for i in range(bs):
                dens = random.choice([1, 2, 5, 49, 196])
                if dens in [1, 2, 5]:
                    rand_dens = dens
                else:
                    rand_dens = int(m_length[i].item() * (dens / 196))
                rand_dens = max(1, rand_dens)
                m_len_i = int(m_length[i].item())
                rand_dens = min(rand_dens, m_len_i)
                selected = np.random.choice(m_len_i, rand_dens, replace=False)
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
                    global_joint_mask[i, :, ctrl_joints[rand_idx[i]]] = (
                        _mask[i, :, ctrl_joints[rand_idx[i]]]
                    )

            ctrlNet_cond = self._build_ctrl_cond(gt_joints, pred_joints, global_joint_mask, bs)

        # ----- Text encoding (v4 path: pooled + sequence) -----
        text_seq, text_pad_mask = None, None
        force_mask = False
        if self.cond_mode == "text":
            with torch.no_grad():
                cond_vector, text_seq, text_pad_mask = self.encode_text_with_seq(y)
        elif self.cond_mode == "action":
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == "uncond":
            cond_vector = torch.zeros(bs, self.latent_dim, device=device).float()
            force_mask = True
        else:
            raise NotImplementedError("Unsupported cond_mode")

        # ----- Forward with control + text -----
        logits = self.trans_forward(
            trans_emb_all, cond_vector, padding_mask, force_mask=force_mask,
            text_seq=text_seq, text_pad_mask=text_pad_mask,
            ctrlNet_cond=ctrlNet_cond,
        )  # [B, 6, K, T_tok]

        # ----- CE loss (per-codebook, c_concat-style via cal_performance2) -----
        ce_loss, _pred_id, acc = cal_performance2(
            logits, labels_list, ignore_index=self.pad_id
        )

        # ----- Trajectory control loss (differentiable soft decode) -----
        # logits: [B, 6, K, T_tok] -> softmax over K -> [B, 6, T_tok, K]
        logits_for_emb = logits.permute(0, 1, 3, 2)  # [B, 6, T_tok, K]
        emb_list = []
        for q in range(Q):
            cb = vq_model.vqvae.quantizers[q].codebook  # [K, C]
            probs_q = F.softmax(logits_for_emb[:, q, :, :], dim=-1)  # [B, T_tok, K]
            emb_q = probs_q @ cb  # [B, T_tok, C]
            emb_q = emb_q.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            emb_list.append(emb_q.permute(0, 2, 1))  # [B, C, T_tok]

        pred_motions2 = vq_model.vqvae.forward_decoder_from_quantized_codes(emb_list)
        # forward_decoder_from_quantized_codes returns [B, 263, 1, T_frame]
        pred_motions2 = pred_motions2.squeeze(2).permute(0, 2, 1)  # [B, 196, 263]
        pred_joints2 = self._motion_to_joints(pred_motions2)

        if global_joint_mask.any():
            loss_traj = F.l1_loss(
                pred_joints2[global_joint_mask],
                gt_joints[global_joint_mask],
                reduction="mean",
            )
        else:
            loss_traj = torch.tensor(0.0, device=device)

        # Return 5-tuple to match trainer: (loss_emb=0, ce_loss, pred_ids=None, acc, loss_tta)
        return torch.tensor(0.0, device=device), ce_loss, None, acc, loss_traj

    def _ttt_loss(self, pred_motions_denorm, global_joint, global_joint_mask):
        """Test-time training loss (same as noqres)."""
        expanded_mask = global_joint_mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        squared_error = (pred_motions_denorm - global_joint) ** 2
        masked_error = squared_error * expanded_mask
        valid_counts = expanded_mask.sum(dim=(1, 2, 3)).clamp(min=1)
        per_sample_loss = masked_error.sum(dim=(1, 2, 3)) / valid_counts
        return per_sample_loss.mean()

    # ============= generate (with TTT) =============

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
        """Iterative sampling with TTT (test-time training) for control.

        c_concat layout: per-T mask schedule (mask whole timestep for all 6 quantizers).
        Returns: (ids_list_final, last_logits) matching c_concat_v4.generate contract,
                 plus the TTT-optimized logits embedded in last_logits.
        """
        device = next(self.parameters()).device
        m_lens_t = torch.as_tensor(m_lens, device=device)
        if m_lens_t.ndim == 0:
            m_lens_t = m_lens_t.unsqueeze(0)
        batch_size = m_lens_t.shape[0]

        seq_len = self.cconcat_seq_len  # = 49

        if vq_model is None:
            vq_model = self.vq_model

        # ---- Text encoding (v4: pooled + sequence) ----
        text_seq, text_pad_mask = None, None
        if conds is None:
            cond_vector = torch.zeros((batch_size, self.clip_dim), device=device)
        elif self.cond_mode == "text":
            cond_vector, text_seq, text_pad_mask = self.encode_text_with_seq(conds)
        else:
            cond_vector = self.encode_text(conds)

        padding_mask = ~lengths_to_mask(m_lens_t, seq_len).to(device)  # [B, T] True=pad

        # Init per-codebook ids (mask everywhere except pad)
        init_ids = torch.where(
            padding_mask,
            torch.full((batch_size, seq_len), self.pad_id, device=device),
            torch.full((batch_size, seq_len), self.mask_id, device=device),
        )
        ids_list = [init_ids.clone() for _ in range(self.Q)]

        # Per-T scores (single value per timestep, shared across codebooks)
        scores = torch.where(
            padding_mask,
            torch.full_like(init_ids, 1e5, dtype=torch.float),
            torch.zeros_like(init_ids, dtype=torch.float),
        )

        starting_temperature = temperature
        last_logits = None

        with torch.no_grad():
          for _step_idx, timestep in enumerate(torch.linspace(0, 1, timesteps, device=device)):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = (m_lens_t.float() * rand_mask_prob).round().clamp(min=1)

            # Per-T mask selection (c_concat: same mask for all 6 quantizers)
            sorted_indices = scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_token_masked.unsqueeze(-1)) & ~padding_mask  # [B, T]

            # Build per-codebook embeddings, channel-concat
            emb_parts = []
            for q in range(self.Q):
                emb_q = self.token_embs[q](ids_list[q])  # [B, T, code_dim]
                mask_emb_q = self.token_embs[q].weight[self.mask_id]
                emb_q = torch.where(is_mask.unsqueeze(-1), mask_emb_q, emb_q)
                emb_parts.append(emb_q)
            emb_all = torch.cat(emb_parts, dim=-1)  # [B, T, 6*code_dim]

            # ---- Build control signal from current predictions ----
            ids_safe_list = []
            num_cb_list = [
                vq_model.vqvae.quantizers[q].codebook.shape[0] for q in range(self.Q)
            ]
            for q in range(self.Q):
                slot = ids_list[q].clone()
                overflow = (slot >= num_cb_list[q]) | (slot < 0)
                slot[overflow] = 0
                ids_safe_list.append(slot)
            local_bt6 = torch.stack(ids_safe_list, dim=-1)  # [B, T, 6]

            x_q_list = vq_model.vqvae.get_x_quantized_from_x_ids(local_bt6)
            pred_m = vq_model.vqvae.forward_decoder_from_quantized_codes(x_q_list)
            pred_m = pred_m.squeeze(2).permute(0, 2, 1)  # [B, 196, 263]
            pred_joints = self._motion_to_joints(pred_m)

            ctrlNet_cond = self._build_ctrl_cond(
                global_joint, pred_joints, global_joint_mask, batch_size
            )

            # ---- CFG forward through trans_forward ----
            logits = self.forward_with_cond_scale(
                emb_all,
                cond_vector=cond_vector,
                padding_mask=padding_mask,
                cond_scale=cond_scale,
                force_mask=force_mask,
                text_seq=text_seq,
                text_pad_mask=text_pad_mask,
                ctrlNet_cond=ctrlNet_cond,
            )  # [B, 6, K, T]

            # logits_btqk for sampling: [B, 6, T, K]
            logits_btqk = logits.permute(0, 1, 3, 2)

            # ======= Test-Time Training (TTT) =======
            if each_iter > 0 and global_joint is not None and global_joint_mask is not None and global_joint_mask.any():
                # Dynamic schedule: more TTT at later steps (matches MaskControl)
                if getattr(self.opt, 'ttt_dynamic', False):
                    _n_iter = (_step_idx + 1) * each_iter
                else:
                    _n_iter = each_iter
                logits_btqk = logits_btqk.detach().requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    [logits_btqk], lr=lr, betas=(0.5, 0.9), weight_decay=1e-6
                )
                with torch.enable_grad():
                    for _ttt_step in range(_n_iter):
                        emb_list = []
                        for q in range(self.Q):
                            cb = vq_model.vqvae.quantizers[q].codebook
                            if getattr(self.opt, 'ttt_gumbel', False):
                                from models.mask_transformer.tools import gumbel_noise as _gn
                                noisy_logits = (
                                    logits_btqk[:, q, :, :] / max(temperature, 1e-10)
                                    + _gn(logits_btqk[:, q, :, :])
                                )
                                probs_q = F.softmax(noisy_logits, dim=-1)
                            else:
                                probs_q = F.softmax(logits_btqk[:, q, :, :], dim=-1)
                            emb_q = probs_q @ cb  # [B, T, C]
                            emb_q = emb_q.masked_fill(padding_mask.unsqueeze(-1), 0.0)
                            emb_list.append(emb_q.permute(0, 2, 1))  # [B, C, T]

                        pred_motions = vq_model.vqvae.forward_decoder_from_quantized_codes(emb_list)
                        pred_motions = pred_motions.squeeze(2).permute(0, 2, 1)
                        pred_motions_denorm = pred_motions * self.std + self.mean
                        pred_joints_ttt = recover_from_ric(
                            pred_motions_denorm.float(), self.opt.joints_num
                        )

                        loss_ttt = self._ttt_loss(pred_joints_ttt, global_joint, global_joint_mask)
                        if abitary_func is not None:
                            loss_ttt = loss_ttt + abitary_func(pred_joints_ttt)

                        optimizer.zero_grad()
                        loss_ttt.backward()
                        optimizer.step()

            # Update last_logits AFTER TTT (codex audit 019deb32 RED bug fix:
            # pre-fix `last_logits = logits` was set BEFORE TTT, so returned
            # tensor was stale pre-TTT; downstream generate_with_control
            # soft-decoded from stale logits, wasting all TTT optimization.)
            last_logits = logits_btqk.detach().permute(0, 1, 3, 2).contiguous()  # [B, 6, K, T]

            # ---- Sample from logits_btqk (per-codebook) ----
            pred_ids_list = []
            probs_list = []
            for q in range(self.Q):
                logit_q = logits_btqk[:, q, :, :].detach()  # [B, T, K]

                if topk_filter_thres > 0:
                    logit_q = top_k(logit_q, topk_filter_thres, dim=-1)

                if gsample:
                    pred_ids_q = gumbel_sample(
                        logit_q, temperature=starting_temperature, dim=-1
                    )
                    probs_q = F.softmax(logit_q / max(temperature, 1e-8), dim=-1)
                else:
                    probs_q = F.softmax(logit_q / max(temperature, 1e-8), dim=-1)
                    pred_ids_q = Categorical(probs_q).sample()

                pred_ids_list.append(pred_ids_q)
                probs_list.append(probs_q)

            # Update ids per-codebook (mask positions only)
            for q in range(self.Q):
                ids_list[q] = torch.where(is_mask, pred_ids_list[q], ids_list[q])

            # Update scores (per-T): mean confidence across 6 codebooks
            scores_q = []
            for q in range(self.Q):
                conf_q = probs_list[q].gather(-1, pred_ids_list[q].unsqueeze(-1)).squeeze(-1)
                scores_q.append(conf_q)
            scores = torch.stack(scores_q, dim=1).mean(dim=1)  # [B, T]
            scores = scores.masked_fill(~is_mask, 1e5).masked_fill(padding_mask, 1e5)

        # Mark pad slots with -1
        for q in range(self.Q):
            ids_list[q] = torch.where(
                padding_mask, torch.full_like(ids_list[q], -1), ids_list[q]
            )

        return ids_list, last_logits

    # ============= generate_with_control (paper-main API for eval pipeline) =============

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
        """Top-level generation API used by scripts/eval_v4_ctrlnet_ttt.py.

        Mirrors T-concat noqres signature exactly. Returns (pred_motions_denorm, pred_motions).
        """
        m_lens = m_length // 4  # frame -> token length
        ids_list, logits = self.generate(
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

        seq_len = ids_list[0].shape[1]
        padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Convert per-codebook logits → per-codebook embeddings
        # logits shape: [B, 6, K, T] (from trans_forward); we use it directly
        emb_list = []
        for q in range(self.Q):
            cb = vq_model.vqvae.quantizers[q].codebook  # [K, C]
            # logits[:, q] is [B, K, T] → permute to [B, T, K] for softmax
            logit_q = logits[:, q, :, :].permute(0, 2, 1)  # [B, T, K]
            probs_q = F.softmax(logit_q / max(temperature, 1e-8), dim=-1)  # [B, T, K]
            emb_q = probs_q @ cb  # [B, T, C]
            emb_q = emb_q.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            emb_list.append(emb_q.permute(0, 2, 1))  # [B, C, T]

        # === Stage 2: post-generation embedding optimization (MaskControl-style) ===
        last_iter = control_opt.get("iter", 0) if control_opt else 0
        last_lr = control_opt.get("lr", 6e-2) if control_opt else 6e-2
        if last_lr is None:
            last_lr = 6e-2

        if last_iter > 0 and global_joint is not None and global_joint_mask is not None and global_joint_mask.any():
            emb_optim_list = [e.clone().detach().contiguous().requires_grad_(True) for e in emb_list]
            # CODA dispatch, same contract as the T-Concat class: control_opt
            # ["s2_optimizer"]="gn" (or STAGE2_OPTIMIZER=gn) swaps the AdamW
            # Stage-2 loop for the analytic anchor solver. Without this branch
            # the class silently ignored s2_optimizer and ran plain AdamW, so a
            # "CODA" row measured on this backbone would in fact have been
            # one AdamW step mislabelled as CODA (codex 019f789d).
            import os as _os_s2
            _opt_choice = _os_s2.environ.get("STAGE2_OPTIMIZER", "adam").lower()
            if (control_opt or {}).get("s2_optimizer"):
                _opt_choice = str(control_opt["s2_optimizer"]).lower()
            if _opt_choice == "gn":
                from models.mask_transformer.coda_solver import coda_solve
                with torch.enable_grad():
                    coda_solve(emb_optim_list, emb_list, vq_model, global_joint,
                               global_joint_mask, _mean, _std, control_opt,
                               self.Q, self.opt.joints_num)
                emb_list = [e.detach() for e in emb_optim_list]
            else:
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
        pred_motions = pred_motions.squeeze(2).permute(0, 2, 1)  # [B, 196, 263]

        pred_motions_denorm = pred_motions * _std + _mean
        pred_motions_denorm = recover_from_ric(pred_motions_denorm.float(), self.opt.joints_num)

        return pred_motions_denorm, pred_motions


# Backward-compatible aliases:
# - ControlTransformerCConcatV4: cleaner name for new code
# - ControlTransformerTConcat: legacy name expected by eval_t2m.py:1126 import
#   (trainer dynamic-imports my module under "control_transformer_t_concat" path,
#   so eval needs to find this exact class name)
# - ControlTransformerTConcatV4: legacy name some scripts use
ControlTransformerCConcatV4 = ControlTransformerKVCConcatV4
ControlTransformerTConcat = ControlTransformerKVCConcatV4
ControlTransformerTConcatV4 = ControlTransformerKVCConcatV4

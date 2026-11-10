"""
ControlNet for T-concat MaskTransformer (v7).

Inherits from transformer_t_concat.MaskTransformer (local vocab, factorized PE,
shared token embedding, single output head).

Key design:
  - Per-quantizer control encoders (from v6): each body part has its own 1D-conv
    encoder that learns a part-specific representation of the control signal.
  - Low-rank per-quantizer per-layer mid-stage projection:
    Linear(D, r) → Linear(r, D) with r=128 instead of Linear(D, D).
    Reduces mid_zero params from 50.4M → 12.6M to avoid overpowering the base model.
  - Per-layer cross-Q attention: after per-Q projection, 6 quantizer residuals
    attend to each other via MultiheadAttention(D, 4) to coordinate control
    injection across body parts. Zero-gated so it starts inactive.
  - Zero-initialized injection into the frozen base transformer layers.

Changes from v6:
  - mid_zero_linear[L][Q]: Linear(D, D) → mid_zero_down[L][Q] Linear(D, 128)
    + mid_zero_up[L][Q] Linear(128, D)  (low-rank, ↓75% params)
  - NEW q_coord[L]: MultiheadAttention(D, 4) for cross-Q coordination
  - NEW q_coord_gate[L]: zero-init scalar gate per layer
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from einops import repeat

from models.mask_transformer.transformer_t_concat import MaskTransformer
from models.vq.encdec import Encoder
from models.mask_transformer.tools import (
    lengths_to_mask, uniform, cosine_schedule, eval_decorator,
    top_k, gumbel_sample, get_mask_subset_prob, gumbel_noise,
)
from utils.motion_process import recover_from_ric
from utils.metrics import control_joint_ids


# ============================================================
# helpers for ControlNet injection
# ============================================================

def return_all_layers(trans, src, mask=None, src_key_padding_mask=None):
    output = src
    all_layers = []
    for mod in trans.layers:
        output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        all_layers.append(output)
    if trans.norm is not None:
        output = trans.norm(output)
    return all_layers


def forward_with_condition(trans, src, list_of_controlnet_output, mask=None, src_key_padding_mask=None):
    output = src
    for mod, control_feat in zip(trans.layers, list_of_controlnet_output):
        output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        output = output + control_feat
    if trans.norm is not None:
        output = trans.norm(output)
    return output


def freeze_block(block):
    block.eval()
    for p in block.parameters():
        p.requires_grad = False


def unfreeze_block(block):
    block.train()
    for p in block.parameters():
        p.requires_grad = True


class ControlTransformerTConcat(MaskTransformer):
    """ControlNet wrapper around T-concat MaskTransformer."""

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

        # ---- load pretrained T-concat base transformer ----
        if trans_path != "":
            ckpt = torch.load(trans_path, map_location=opt.device)
            model_key = "t2m_transformer" if "t2m_transformer" in ckpt else "trans"
            missing_keys, unexpected_keys = self.load_state_dict(
                ckpt[model_key], strict=False
            )
            freeze_block(self)
            print(
                f"Loaded T-concat Transformer from {trans_path}\n"
                f"  missing keys: {missing_keys}\n"
                f"  unexpected keys: {unexpected_keys}"
            )

        # ============= ControlNet modules =============
        # Per-quantizer (body part) control projection: each part learns its own
        # control strength. Zero-init means trajectory-irrelevant parts (head, neck)
        # start at zero and only root/leg parts learn strong projections.
        self.first_zero_linear = nn.ModuleList(
            [nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.Q)]
        )
        # Low-rank per-layer per-quantizer mid-stage projection.
        # D → r → D with r=128, reduces 50.4M → 12.6M params.
        self.mid_zero_rank = 128
        self.mid_zero_down = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.latent_dim, self.mid_zero_rank) for _ in range(self.Q)
            ])
            for _ in range(self.num_layers)
        ])
        self.mid_zero_up = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.mid_zero_rank, self.latent_dim) for _ in range(self.Q)
            ])
            for _ in range(self.num_layers)
        ])

        # Cross-Q coordination: per-layer attention lets 6 quantizer residuals
        # communicate before injection into the base model.
        self.q_coord = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.latent_dim, num_heads=4, batch_first=False,
            )
            for _ in range(self.num_layers)
        ])
        # Zero-init gate: tanh(0)=0, so cross-Q coord starts inactive
        self.q_coord_gate = nn.ParameterList([
            nn.Parameter(torch.zeros(1))
            for _ in range(self.num_layers)
        ])

        # Per-quantizer control encoder: each body part has its own 1D-conv
        # encoder (width=256) that learns a part-specific representation of the
        # control signal.  For trajectory mode all encoders receive the same 6D
        # root input but learn independent temporal features; for random/cross
        # mode this naturally extends to per-part input splits (future TODO).
        input_emb_width = (
            6 if self.control == "trajectory"
            else len(control_joint_ids) * 2 * 3
        )
        self.encoder_control = nn.ModuleList([
            Encoder(
                input_emb_width=input_emb_width,
                output_emb_width=self.latent_dim,
                down_t=2,
                stride_t=2,
                width=256,
                depth=3,
                dilation_growth_rate=3,
                activation="relu",
                norm=None,
            )
            for _ in range(self.Q)
        ])

        # Control branch encoder (deepcopy of frozen base)
        self.seqTransEncoder_control = copy.deepcopy(self.seqTransEncoder)

        # Zero-init linear layers
        def init_zero(m):
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.first_zero_linear.apply(init_zero)
        # Zero-init only mid_zero_up so product is zero initially
        self.mid_zero_up.apply(init_zero)

        # Per-codebook mean embeddings for masked positions during ctrl signal computation
        self.mask_emb_vq_list = [
            vq_model.vqvae.quantizers[i].codebook.mean(0) for i in range(self.Q)
        ]

        # Must be called after all submodules are attached
        self.ctrl_train()

    # ============= train/eval =============
    def ctrl_eval(self):
        freeze_block(self)

    def ctrl_train(self):
        freeze_block(self)
        unfreeze_block(self.seqTransEncoder_control)
        unfreeze_block(self.encoder_control)
        unfreeze_block(self.first_zero_linear)
        unfreeze_block(self.mid_zero_down)
        unfreeze_block(self.mid_zero_up)
        unfreeze_block(self.q_coord)
        # q_coord_gate is a ParameterList, handle manually
        for g in self.q_coord_gate:
            g.requires_grad = True

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
        x_q_list = self.vq_model.vqvae.get_x_quantized_from_x_ids(local_ids_bt6)  # Q * [B, C, T]
        pred_motions = self.vq_model.vqvae.forward_decoder_from_quantized_codes(x_q_list)  # [B, 263, 1, T*4]
        pred_motions = pred_motions.permute(0, 2, 3, 1).squeeze(1)  # [B, T*4, 263]
        return pred_motions

    def _motion_to_joints(self, pred_motions):
        """Denormalize and convert to joint positions."""
        pred_denorm = pred_motions * self.std + self.mean
        return recover_from_ric(pred_denorm.float(), self.opt.joints_num)

    # ============= trans_forward with ControlNet injection =============
    def trans_forward(self, motion_emb, cond, padding_mask, force_mask=False, ctrlNet_cond=None):
        """
        Override trans_forward to inject ControlNet features.

        motion_emb: [B, S, code_dim] where S = T * Q
        padding_mask: [B, S], TRUE=pad
        cond: [B, clip_dim]
        ctrlNet_cond: [B, 196, 6/36] control signal in raw frame space
        """
        if not self.ctrl_net or ctrlNet_cond is None:
            return super().trans_forward(motion_emb, cond, padding_mask, force_mask)

        cond = self.mask_cond(cond, force_mask=force_mask)

        # Inject quantizer token identity
        motion_emb = self._inject_quantizer_token_identity(motion_emb)
        x = self.input_process(motion_emb)  # [S, B, D]
        cond_lat = self.cond_emb(cond).unsqueeze(0)  # [1, B, D]

        # Factorized positional encoding
        x = self._add_factorized_position_encoding(x)
        xseq = torch.cat([cond_lat, x], dim=0)  # [S+1, B, D]

        padding_mask = torch.cat(
            [torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1
        )  # [B, S+1]

        # ========= Control branch =========
        # Per-quantizer control encoding: each body part encoder processes the
        # control signal independently, producing part-specific features.
        ctrl_input = ctrlNet_cond.permute(0, 2, 1)  # [B, C_ctrl, 196]
        ctrl_per_part = []
        for q in range(self.Q):
            feat_q = self.encoder_control[q](ctrl_input)  # [B, D, T]
            feat_q = feat_q.permute(2, 0, 1)              # [T, B, D]
            feat_q = self.first_zero_linear[q](feat_q)     # [T, B, D]
            ctrl_per_part.append(feat_q)
        # Interleave as [t0q0, t0q1, ..., t0q5, t1q0, ...]
        ctrl_feat = torch.stack(ctrl_per_part, dim=1)  # [T, Q, B, D]
        T_ctrl = ctrl_feat.shape[0]
        B_ctrl = ctrl_feat.shape[2]
        D_ctrl = ctrl_feat.shape[3]
        ctrl_feat = ctrl_feat.reshape(T_ctrl * self.Q, B_ctrl, D_ctrl)  # [S, B, D]

        # Add to xseq (prepend zero for cond token position)
        control_input = xseq + torch.cat(
            [torch.zeros_like(cond_lat), ctrl_feat], dim=0
        )  # [S+1, B, D]

        # Run control branch encoder
        control_output_list = return_all_layers(
            self.seqTransEncoder_control,
            control_input,
            src_key_padding_mask=padding_mask,
        )  # L * [S+1, B, D]

        # Per-layer: low-rank per-Q projection + cross-Q attention coordination
        for i in range(self.num_layers):
            out = control_output_list[i]  # [S+1, B, D]
            cond_part = out[:1]           # [1, B, D]
            seq_part = out[1:]            # [S=T*Q, B, D]
            seq_tqbd = seq_part.view(T_ctrl, self.Q, B_ctrl, D_ctrl)

            # Low-rank per-Q projection: D → 128 → D
            proj = torch.stack(
                [self.mid_zero_up[i][q](self.mid_zero_down[i][q](seq_tqbd[:, q]))
                 for q in range(self.Q)],
                dim=1,
            )  # [T, Q, B, D]

            # Cross-Q attention: 6 quantizer residuals attend to each other
            # Reshape so Q is the sequence dim, T*B is the batch dim
            proj_for_attn = proj.permute(1, 0, 2, 3).reshape(
                self.Q, T_ctrl * B_ctrl, D_ctrl
            )  # [Q, T*B, D]
            coord_out, _ = self.q_coord[i](
                proj_for_attn, proj_for_attn, proj_for_attn
            )  # [Q, T*B, D]
            coord_out = coord_out.reshape(
                self.Q, T_ctrl, B_ctrl, D_ctrl
            ).permute(1, 0, 2, 3)  # [T, Q, B, D]

            # Zero-gated residual: tanh(0)=0 initially, learns over time
            proj = proj + self.q_coord_gate[i].tanh() * coord_out

            seq_part = proj.reshape(T_ctrl * self.Q, B_ctrl, D_ctrl)
            control_output_list[i] = torch.cat([cond_part, seq_part], dim=0)

        # Run main encoder with injected control features
        out = forward_with_condition(
            self.seqTransEncoder,
            control_input,
            list_of_controlnet_output=control_output_list,
            src_key_padding_mask=padding_mask,
        )[1:]  # [S, B, D]

        logits = self.output_head(out)  # [B, K, S]
        return logits

    # ============= training forward =============
    def forward(self, y, m_length, pose):
        """
        y: list[str] text batch
        m_length: [B] raw lengths in frames (≤196)
        pose: [B, 196, 263]
        Returns: (loss_emb, ce_loss, pred_id, acc, loss_tta)
        """
        device = pose.device

        # GT joints for control signal
        global_joint = recover_from_ric(
            self.inv_transform(pose).float(), self.opt.joints_num
        )  # [B, 196, 22, 3]

        # VQ encode
        pose_4d = pose.permute(0, 2, 1).unsqueeze(2)  # [B, 263, 1, 196]
        code_idx = self.vq_model.get_code_idx(pose_4d)  # [B, Q, T]
        code_idx_bt6 = code_idx.permute(0, 2, 1).contiguous()  # [B, T, Q]
        B, T, Q = code_idx_bt6.shape

        assert m_length.min() > 30
        m_lens = (m_length // 4).to(device)  # [B] token-frame scale

        # Build control mask
        global_joint_mask = self._build_control_mask(m_length, pose, device)  # [B, 196, 22]

        # Flatten to local IDs [B, S] where S = T * Q
        local_ids = code_idx_bt6.reshape(B, T * Q).contiguous()
        S = local_ids.shape[1]

        # Padding masks
        non_pad_mask_t = lengths_to_mask(m_lens, T).to(device)  # [B, T]
        non_pad_mask_seq = non_pad_mask_t.unsqueeze(-1).expand(B, T, Q).reshape(B, S)
        padding_mask_seq = ~non_pad_mask_seq

        # Labels
        labels = torch.where(
            non_pad_mask_seq, local_ids, torch.full_like(local_ids, self.pad_id)
        )

        # Cond vector
        with torch.no_grad():
            cond_vector = self.encode_text(y)  # [B, 512]

        # BERT-style masking (frame-level, broadcast to Q)
        rand_time = uniform((B,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (m_lens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((B, T), device=device).argsort(dim=-1)
        mask_t = (batch_randperm < num_token_masked.unsqueeze(-1)) & non_pad_mask_t
        mask = mask_t.unsqueeze(-1).expand(B, T, Q).reshape(B, S)

        x_ids = labels.clone()

        # 10% → random real token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_ids = torch.randint(low=0, high=self.K, size=x_ids.shape, device=device)
        x_ids = torch.where(mask_rid, rand_ids, x_ids)

        # 88% → mask_id
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids = torch.where(mask_mid, torch.full_like(x_ids, self.mask_id), x_ids)

        # ======= build ctrlNet_cond from current masked tokens =======
        x_ids_safe = x_ids.clone()
        overflow = x_ids_safe >= self.K
        x_ids_safe[overflow] = 0

        local_bt6 = x_ids_safe.view(B, T, Q)
        # Fill overflow positions with codebook mean
        overflow_bt6 = overflow.view(B, T, Q)
        x_q_list = self.vq_model.vqvae.get_x_quantized_from_x_ids(local_bt6)  # Q * [B, C, T]
        for i in range(Q):
            mask_i = overflow_bt6[:, :, i]  # [B, T]
            if mask_i.any():
                x_q_list[i].permute(0, 2, 1)[mask_i] = self.mask_emb_vq_list[i].to(device)

        pred_motions = self.vq_model.vqvae.forward_decoder_from_quantized_codes(x_q_list)
        pred_motions = pred_motions.permute(0, 2, 3, 1).squeeze(1)  # [B, 196, 263]
        pred_joints = self._motion_to_joints(pred_motions)

        ctrlNet_cond = self._build_ctrl_cond(global_joint, pred_joints, global_joint_mask, B)

        # ======= noisy embedding for transformer input =======
        std_noise = 0.1
        safe_codes = x_ids.clamp(min=0, max=self.K - 1)
        one_hot = F.one_hot(safe_codes, num_classes=self.K).float()
        noise = torch.randn_like(one_hot) * std_noise
        motion_emb = (one_hot + noise) @ self.token_emb.weight[:self.K]  # [B, S, code_dim]

        mask_pos = x_ids == self.mask_id
        pad_pos = x_ids == self.pad_id
        if mask_pos.any():
            motion_emb[mask_pos] = self.token_emb.weight[self.mask_id]
        if pad_pos.any():
            motion_emb[pad_pos] = self.token_emb.weight[self.pad_id]

        # ======= transformer with ControlNet =======
        logits = self.trans_forward(
            motion_emb, cond_vector, padding_mask_seq, force_mask=False, ctrlNet_cond=ctrlNet_cond
        )  # [B, K, S]
        logits_bsv = logits.permute(0, 2, 1).contiguous()  # [B, S, K]

        # CE loss
        ce_loss = F.cross_entropy(
            logits_bsv.view(-1, self.K),
            labels.view(-1),
            ignore_index=self.pad_id,
        )

        # Accuracy
        pred_seq = logits_bsv.argmax(dim=-1)
        correct = (pred_seq == labels) & non_pad_mask_seq
        acc = correct.float().sum() / non_pad_mask_seq.float().sum().clamp_min(1.0)

        # pred_id for compatibility (Q * [B, T])
        pred_bt6 = pred_seq.view(B, T, Q)
        pred_id = [pred_bt6[:, :, i] for i in range(Q)]

        # ======= control loss (loss_tta) =======
        probs_local = F.softmax(logits_bsv, dim=-1).view(B, T, Q, self.K)
        emb_list = []
        for i in range(Q):
            cb = self.vq_model.vqvae.quantizers[i].codebook  # [K, C]
            emb_i = probs_local[:, :, i, :] @ cb  # [B, T, C]
            emb_i = emb_i.masked_fill(~non_pad_mask_t.unsqueeze(-1), 0.0)
            emb_list.append(emb_i.permute(0, 2, 1))  # [B, C, T]

        pred_motions2 = self.vq_model.vqvae.forward_decoder_from_quantized_codes(emb_list)
        pred_motions2 = pred_motions2.squeeze(2).permute(0, 2, 1)  # [B, 196, 263]
        pred_joints2 = self._motion_to_joints(pred_motions2)

        loss_tta = F.l1_loss(
            pred_joints2[global_joint_mask],
            global_joint[global_joint_mask],
            reduction="mean",
        )

        loss_emb = torch.zeros(1, device=device)
        return loss_emb, ce_loss, pred_id, acc, loss_tta

    def _ttt_loss(self, pred_motions_denorm, global_joint, global_joint_mask):
        """Per-sample MSE loss for test-time training (same as C-concat)."""
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

        # Cond vector
        if conds is None:
            cond_vector = torch.zeros((batch_size, self.clip_dim), device=device)
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
          for timestep in torch.linspace(0, 1, timesteps, device=device):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked_t = (m_lens_t * rand_mask_prob).round().clamp(min=1)

            # Frame-level masking based on scores
            frame_scores = scores.view(batch_size, seq_len, self.Q).mean(dim=-1)
            sorted_indices = frame_scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask_t = (ranks < num_token_masked_t.unsqueeze(-1)) & ~padding_mask_t
            is_mask = is_mask_t.unsqueeze(-1).expand(batch_size, seq_len, self.Q).reshape(batch_size, S)

            # Build embedding
            emb = self.token_emb(ids)
            mask_emb = self.token_emb.weight[self.mask_id]
            emb = torch.where(is_mask.unsqueeze(-1), mask_emb, emb)

            # Build ctrlNet_cond from current decoded motion
            ids_safe = ids.clone()
            overflow = (ids_safe >= self.K) | (ids_safe < 0)
            ids_safe[overflow] = 0
            local_bt6 = ids_safe.view(batch_size, seq_len, self.Q)

            x_q_list = vq_model.vqvae.get_x_quantized_from_x_ids(local_bt6)
            pred_m = vq_model.vqvae.forward_decoder_from_quantized_codes(x_q_list)
            pred_m = pred_m.squeeze(2).permute(0, 2, 1)  # [B, 196, 263]
            pred_joints = self._motion_to_joints(pred_m)

            ctrlNet_cond = self._build_ctrl_cond(
                global_joint, pred_joints, global_joint_mask, batch_size
            )

            # Forward with control
            logits = self.forward_with_cond_scale(
                emb,
                cond_vector=cond_vector,
                padding_mask=padding_mask_seq,
                cond_scale=cond_scale,
                force_mask=force_mask,
                ctrlNet_cond=ctrlNet_cond,
            )  # [B, K, S]

            logits_bsv = logits.permute(0, 2, 1).contiguous()  # [B, S, K]

            if topk_filter_thres > 0:
                logits_bsv = top_k(logits_bsv, topk_filter_thres, dim=-1)

            # ======= Test-Time Training (TTT) =======
            if each_iter > 0 and global_joint is not None and global_joint[global_joint_mask].sum() != 0:
                logits_bsv = logits_bsv.detach().requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    [logits_bsv], lr=lr, betas=(0.5, 0.9), weight_decay=1e-6
                )
                non_pad_mask_t = ~padding_mask_t  # [B, T]
                with torch.enable_grad():
                    for _ttt_step in range(each_iter):
                        # Reshape logits to [B, T, Q, K] for per-part decoding
                        logits_btqk = logits_bsv.view(batch_size, seq_len, self.Q, self.K)
                        emb_list = []
                        for q in range(self.Q):
                            cb = vq_model.vqvae.quantizers[q].codebook  # [K, C]
                            probs_q = F.softmax(logits_btqk[:, :, q, :], dim=-1)  # [B, T, K]
                            emb_q = probs_q @ cb  # [B, T, C]
                            emb_q = emb_q.masked_fill(padding_mask_t.unsqueeze(-1), 0.0)
                            emb_list.append(emb_q.permute(0, 2, 1))  # [B, C, T]

                        pred_motions = vq_model.vqvae.forward_decoder_from_quantized_codes(emb_list)
                        pred_motions = pred_motions.squeeze(2).permute(0, 2, 1)  # [B, 196, 263]
                        pred_motions_denorm = pred_motions * self.std + self.mean
                        pred_joints_ttt = recover_from_ric(pred_motions_denorm.float(), self.opt.joints_num)

                        loss_ttt = self._ttt_loss(pred_joints_ttt, global_joint, global_joint_mask)

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

            # Update scores
            conf = probs_local.gather(-1, pred_local.unsqueeze(-1)).squeeze(-1)
            frame_conf = conf.view(batch_size, seq_len, self.Q).mean(dim=-1)
            frame_scores = frame_conf.masked_fill(~is_mask_t, 1e5).masked_fill(padding_mask_t, 1e5)
            scores = frame_scores.unsqueeze(-1).expand(batch_size, seq_len, self.Q).reshape(batch_size, S)

        # Unpack to list format: Q * [B, T]
        ids_final = ids.view(batch_size, seq_len, self.Q)
        ids_final = torch.where(
            padding_mask_t.unsqueeze(-1).expand_as(ids_final),
            torch.full_like(ids_final, -1),
            ids_final,
        )
        ids_list = [ids_final[:, :, i] for i in range(self.Q)]

        # Also build logits in [B, Q, T, K] format for compatibility
        logits_final = logits_bsv.view(batch_size, seq_len, self.Q, self.K)
        logits_final = logits_final.permute(0, 2, 1, 3).contiguous()  # [B, Q, T, K]

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
        )  # ids_bt6: [B, T, Q], logits: [B, Q, T, K]

        seq_len = ids_bt6.shape[1]
        padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Decode to motion
        emb_list = []
        for i in range(self.Q):
            cb = vq_model.vqvae.quantizers[i].codebook
            probs_i = F.softmax(logits[:, i, :, :] / max(temperature, 1e-8), dim=-1)
            emb_i = probs_i @ cb
            emb_i = emb_i.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            emb_list.append(emb_i.permute(0, 2, 1))

        pred_motions = vq_model.vqvae.forward_decoder_from_quantized_codes(emb_list)
        pred_motions = pred_motions.squeeze(2).permute(0, 2, 1)  # [B, 196, 263]

        pred_motions_denorm = pred_motions * _std + _mean
        pred_motions_denorm = recover_from_ric(pred_motions_denorm.float(), self.opt.joints_num)

        return pred_motions_denorm, pred_motions

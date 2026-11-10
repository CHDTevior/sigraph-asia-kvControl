"""T-concat MaskTransformer v2 — sparse text cross-attention.

Changes from v1 (transformer_t_concat.py):
  1. Extracts CLIP sequence tokens (77 tokens) for word-level cross-attention
  2. Adds cross-attention blocks every `cross_attn_interval` self-attention layers
  3. Keeps prepended pooled text token for global conditioning (dual-path)
  4. State-dict compatible with v1 (load v1 checkpoint with strict=False)

Motivation:
  T-concat has 1 text token : 294 motion tokens (1:294 ratio).
  MoMask has 1:49. This 6x dilution causes Top3 R-precision gap (0.72 vs 0.80).
  Cross-attention gives motion tokens direct access to word-level text features
  without competing in self-attention.

Config (via opt / command line):
  --cross_attn_interval N   Insert cross-attention every N layers (default 4)
  --cross_attn_heads N      Heads for cross-attention (default: same as self-attention)

Version: v2
Base: transformer_t_concat.py (v1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torch.distributions.categorical import Categorical

from models.mask_transformer.transformer_t_concat import (
    MaskTransformer as MaskTransformerV1,
    _unpack_local_seq_to_list,
    require,
)
from models.mask_transformer.tools import (
    lengths_to_mask,
    uniform,
    eval_decorator,
    top_k,
    gumbel_sample,
    get_mask_subset_prob,
)


# ---------------------------------------------------------------------------
# Cross-attention block
# ---------------------------------------------------------------------------

class TextCrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention: Q = motion/cond tokens, K/V = CLIP text tokens.

    Design choices (from Codex GPT-5.4 review):
      - gate (scalar, init=0) on residual branch → output starts at zero, model = v1
      - QKV and out_proj keep STANDARD init (NOT zero) so gradients flow through
      - Only gate=0 blocks the output; once gate learns to open, the full branch activates
      - cond_keep mask on output ensures strict zero under CFG force_mask (no bias leak)
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False,
        )
        self.drop = nn.Dropout(dropout)
        # Learnable gate initialized to 0 — residual starts as identity (LayerScale)
        # QKV + out_proj keep standard init so d(gate)/d(loss) = out_proj(attn(x)) ≠ 0
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, memory, memory_key_padding_mask=None, cond_keep=None):
        """
        x:      [S+1, B, D]  (motion tokens + prepended cond token)
        memory: [L, B, D]    (projected CLIP text sequence, L <= 77)
        memory_key_padding_mask: [B, L]  True = ignore
        cond_keep: [B, 1]    1.0=keep, 0.0=drop (for CFG — ensures strict zero)
        """
        residual = x
        x_norm = self.norm(x)
        out, _ = self.cross_attn(
            query=x_norm,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )
        out = self.gate * self.drop(out)
        # Gate by cond_keep to ensure strict zero under force_mask (fixes bias leak)
        if cond_keep is not None:
            # cond_keep: [B, 1] → [1, B, 1] for broadcasting with [S+1, B, D]
            out = out * cond_keep.unsqueeze(0)
        return residual + out


# ---------------------------------------------------------------------------
# MaskTransformer v2
# ---------------------------------------------------------------------------

class MaskTransformer(MaskTransformerV1):
    """T-concat v2: original T-concat self-attention + sparse text cross-attention.

    Inherits all v1 functionality; adds cross-attention blocks after selected
    self-attention layers and extracts CLIP text *sequence* tokens (77-token)
    in addition to the pooled embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        opt = self.opt
        num_layers = len(self.seqTransEncoder.layers)

        # ---- config (from opt / command line) ----
        cross_attn_interval = int(getattr(opt, "cross_attn_interval", 4))
        nhead_cfg = getattr(opt, "cross_attn_heads", None)
        nhead = int(nhead_cfg) if nhead_cfg else int(getattr(opt, "n_heads", 8))

        # ---- which layers get cross-attention (0-indexed, applied AFTER layer i) ----
        self.cross_attn_layer_ids = set(
            range(cross_attn_interval - 1, num_layers, cross_attn_interval)
        )
        n_cross = len(self.cross_attn_layer_ids)
        self._cross_attn_sorted = sorted(self.cross_attn_layer_ids)

        # ---- new modules ----
        self.text_cross_attn = nn.ModuleList(
            [TextCrossAttentionBlock(self.latent_dim, nhead, self.dropout) for _ in range(n_cross)]
        )
        self.text_seq_proj = nn.Sequential(
            nn.Linear(self.clip_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )

        # Initialize text_seq_proj with standard init; cross-attn blocks handle
        # their own init (zero-init out_proj + gate=0 in TextCrossAttentionBlock)
        self.text_seq_proj.apply(self.init_weights)

        # ---- build layer-index → cross-attn-index mapping ----
        self._layer_to_cross_idx = {}
        for ci, li in enumerate(self._cross_attn_sorted):
            self._layer_to_cross_idx[li] = ci

        # ---- report ----
        n_new = sum(p.numel() for p in self.text_cross_attn.parameters()) + sum(
            p.numel() for p in self.text_seq_proj.parameters()
        )
        n_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[v2] Cross-attention every {cross_attn_interval} layers → "
            f"{n_cross} blocks at layers {self._cross_attn_sorted}"
        )
        print(f"[v2] New params: {n_new / 1e6:.1f}M | Total trainable: {n_total / 1e6:.1f}M")

    # ------------------------------------------------------------------
    # Text encoding — extract both pooled + sequence tokens
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_text_with_seq(self, raw_text):
        """Return (pooled, text_seq, text_pad_mask).

        pooled:        [B, clip_dim]   same as encode_text()
        text_seq:      [B, 77, clip_dim]  per-token features after ln_final
        text_pad_mask: [B, 77]  True for padding positions (to ignore in attn)
        """
        device = next(self.parameters()).device
        text_tokens = clip.tokenize(raw_text, truncate=True).to(device)

        cm = self.clip_model
        x = cm.token_embedding(text_tokens).type(cm.dtype)
        x = x + cm.positional_embedding.type(cm.dtype)
        x = x.permute(1, 0, 2)  # [77, B, width]
        x = cm.transformer(x)
        x = x.permute(1, 0, 2)  # [B, 77, width]
        x = cm.ln_final(x).type(cm.dtype)

        # pooled (same path as v1 encode_text)
        pooled = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ cm.text_projection

        # padding mask (CLIP pads with 0)
        text_pad_mask = text_tokens == 0  # True = padding

        return pooled.float(), x.float(), text_pad_mask

    # ------------------------------------------------------------------
    # Condition masking — shared mask for pooled + sequence tokens
    # ------------------------------------------------------------------

    def _get_cond_mask(self, bs, device, force_mask=False):
        """Per-sample binary mask [B, 1].  1.0 = keep, 0.0 = drop."""
        if force_mask:
            return torch.zeros(bs, 1, device=device)
        if self.training and self.cond_drop_prob > 0.0:
            drop = torch.bernoulli(torch.ones(bs, device=device) * self.cond_drop_prob)
            return (1.0 - drop).view(bs, 1)
        return torch.ones(bs, 1, device=device)

    # ------------------------------------------------------------------
    # Transformer forward — with cross-attention injection
    # ------------------------------------------------------------------

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
        """Forward pass with sparse text cross-attention.

        Args:
            motion_emb:   [B, S, code_dim]   S = T * Q
            cond:         [B, clip_dim]       pooled CLIP embedding
            padding_mask: [B, S]              True = pad
            text_seq:     [B, L, clip_dim]    CLIP sequence tokens (optional)
            text_pad_mask:[B, L]              True = padding (optional)
        """
        bs = cond.shape[0]
        cond_keep = self._get_cond_mask(bs, cond.device, force_mask)

        # ---- pooled condition (same as v1) ----
        cond = cond * cond_keep

        motion_emb = self._inject_quantizer_token_identity(motion_emb)
        x = self.input_process(motion_emb)  # [S, B, D]
        cond_token = self.cond_emb(cond).unsqueeze(0)  # [1, B, D]

        x = self._add_factorized_position_encoding(x)
        xseq = torch.cat([cond_token, x], dim=0)  # [S+1, B, D]

        padding_mask = torch.cat(
            [torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1
        )

        # ---- text memory for cross-attention ----
        if text_seq is not None:
            # project then mask — ensures params participate in compute graph for DDP
            text_memory = self.text_seq_proj(text_seq)          # [B, L, D]
            text_memory = text_memory * cond_keep.unsqueeze(-1) # zero dropped samples
            text_memory = text_memory.permute(1, 0, 2)          # [L, B, D]
        else:
            text_memory = None

        # ---- layer-by-layer with cross-attention injection ----
        cross_idx = 0
        for i, layer in enumerate(self.seqTransEncoder.layers):
            xseq = layer(xseq, src_key_padding_mask=padding_mask)
            if i in self.cross_attn_layer_ids:
                if text_memory is not None:
                    xseq = self.text_cross_attn[cross_idx](
                        xseq, text_memory, text_pad_mask, cond_keep=cond_keep
                    )
                cross_idx += 1

        out = xseq[1:]  # [S, B, D]  — drop cond token

        if self.per_q_head:
            S_out, B_out, D_out = out.shape
            T_out = S_out // self.Q
            out_tqbd = out.view(T_out, self.Q, B_out, D_out)
            logits_list = []
            for q in range(self.Q):
                lq = self.output_heads[q](out_tqbd[:, q, :, :])
                logits_list.append(lq)
            logits = torch.stack(logits_list, dim=3).reshape(B_out, self.vocab, T_out * self.Q)
        else:
            logits = self.output_head(out)  # [B, vocab, S]
        return logits

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(self, y, m_lens, pose, vq_model):
        device = pose.device
        self._require_vq_interface(vq_model)

        # ---- VQ encode (identical to v1) ----
        pose_4d = pose.permute(0, 2, 1).unsqueeze(2)
        code_idx = vq_model.get_code_idx(pose_4d)
        code_idx_bt6 = code_idx.permute(0, 2, 1)
        B, T, Q = code_idx_bt6.shape
        require(Q == self.Q, f"T-concat expects Q={self.Q}, got Q={Q}.")

        local_ids = code_idx_bt6.reshape(B, T * Q).contiguous()
        S = local_ids.shape[1]

        non_pad_mask_t = lengths_to_mask(m_lens, T).to(device)
        non_pad_mask_seq = non_pad_mask_t.unsqueeze(-1).expand(B, T, Q).reshape(B, S)
        padding_mask_seq = ~non_pad_mask_seq

        labels = torch.where(
            non_pad_mask_seq, local_ids, torch.full_like(local_ids, self.pad_id)
        )

        # ---- text encoding (v2: pooled + sequence) ----
        force_mask = False
        text_seq, text_pad_mask = None, None
        if self.cond_mode == "text":
            with torch.no_grad():
                cond_vector, text_seq, text_pad_mask = self.encode_text_with_seq(y)
        elif self.cond_mode == "action":
            cond_vector = self.encode_action(y).to(device).float()
        elif self.cond_mode == "uncond":
            cond_vector = torch.zeros(B, self.latent_dim, device=device).float()
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        # ---- BERT-style masking (identical to v1) ----
        rand_time = uniform((B,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (m_lens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((B, T), device=device).argsort(dim=-1)
        mask_t = batch_randperm < num_token_masked.unsqueeze(-1)
        mask_t &= non_pad_mask_t
        mask_frame = mask_t.unsqueeze(-1).expand(B, T, Q).reshape(B, S)

        # v4.6: hybrid 2D masking (per-Q independent mask for some samples)
        if getattr(self, 'mask_2d_hybrid', False) and self.training:
            use_2d = torch.rand(B, device=device) < getattr(self, 'mask_2d_ratio', 0.3)
            perm_2d = torch.rand((B, T, Q), device=device).argsort(dim=1)
            mask_2d = perm_2d < num_token_masked.view(B, 1, 1)
            mask_2d &= non_pad_mask_t.unsqueeze(-1)
            mask_2d = mask_2d.reshape(B, S)
            mask = torch.where(use_2d.unsqueeze(1), mask_2d, mask_frame)
        else:
            mask = mask_frame

        x_ids = labels.clone()

        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_ids = torch.randint(low=0, high=self.K, size=x_ids.shape, device=device)
        x_ids = torch.where(mask_rid, rand_ids, x_ids)

        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids = torch.where(mask_mid, torch.full_like(x_ids, self.mask_id), x_ids)

        # ---- noisy embedding (v4.1: configurable std + masked-only noise) ----
        std = float(getattr(self.opt, 'noise_std', 0.1))
        noise_masked_only = getattr(self.opt, 'noise_masked_only', False)
        raw_ids = x_ids
        safe_codes = raw_ids.clamp(min=0, max=self.vocab - 1)
        one_hot = F.one_hot(safe_codes, num_classes=self.vocab).float()
        noise = torch.randn_like(one_hot) * std
        if noise_masked_only:
            # Only add noise to positions that were corrupted (mask-replaced or random-replaced)
            corrupted = mask_mid | mask_rid  # [B, S]
            noise = noise * corrupted.unsqueeze(-1).float()
        noisy_one_hot = one_hot + noise
        motion_emb = noisy_one_hot @ self.token_emb.weight[: self.vocab]

        mask_pos = raw_ids == self.mask_id
        pad_pos = raw_ids == self.pad_id
        if mask_pos.any():
            motion_emb[mask_pos] = self.token_emb.weight[self.mask_id]
        if pad_pos.any():
            motion_emb[pad_pos] = self.token_emb.weight[self.pad_id]

        # ---- transformer forward (v2: with text sequence) ----
        logits = self.trans_forward(
            motion_emb,
            cond_vector,
            padding_mask_seq,
            force_mask,
            text_seq=text_seq,
            text_pad_mask=text_pad_mask,
        )
        logits_bsv = logits.permute(0, 2, 1).contiguous()

        # ---- CE + accuracy (v4.1: masked-only CE + label smoothing) ----
        ce_labels = labels.clone()
        if getattr(self.opt, 'masked_only_ce', False):
            # Only compute CE on masked positions; ignore clean context tokens
            ce_labels[~mask] = self.pad_id
        label_smoothing = float(getattr(self.opt, 'label_smoothing', 0.0))
        loss_ce = F.cross_entropy(
            logits_bsv.view(-1, self.vocab),
            ce_labels.view(-1),
            ignore_index=self.pad_id,
            label_smoothing=label_smoothing,
        )

        pred_seq = logits_bsv.argmax(dim=-1)
        correct = (pred_seq == labels) & non_pad_mask_seq
        acc = correct.float().sum() / non_pad_mask_seq.float().sum().clamp_min(1.0)

        pred_id = _unpack_local_seq_to_list(pred_seq, padding_mask_seq=padding_mask_seq, Q=self.Q)

        # ---- embedding alignment loss (identical to v1) ----
        probs_local = F.softmax(logits_bsv, dim=-1).view(B, T, self.Q, self.K)

        pred_emb_bt6c = []
        for i in range(self.Q):
            cb = vq_model.vqvae.quantizers[i].codebook
            pred_emb_i = probs_local[:, :, i, :] @ cb
            pred_emb_bt6c.append(pred_emb_i)
        pred_emb_bt6c = torch.stack(pred_emb_bt6c, dim=2)

        non_pad_bt6 = non_pad_mask_seq.view(B, T, self.Q)
        pred_emb_bt6c = pred_emb_bt6c.masked_fill(~non_pad_bt6.unsqueeze(-1), 0.0)

        x_encoder_list = vq_model.vqvae.get_quantized_codes(pose_4d, in_idx_format=False)
        loss_emb = 0.0
        for i in range(self.Q):
            x_enc = x_encoder_list[i].permute(0, 2, 1)
            valid = non_pad_bt6[:, :, i]
            loss_emb = loss_emb + (
                F.l1_loss(x_enc[valid], pred_emb_bt6c[:, :, i, :][valid], reduction="mean")
                / float(self.Q)
            )

        return loss_emb, loss_ce, pred_id, acc

    # ------------------------------------------------------------------
    # CFG forward
    # ------------------------------------------------------------------

    def forward_with_cond_scale(
        self,
        motion_emb,
        cond_vector,
        padding_mask,
        cond_scale=3,
        force_mask=False,
        text_seq=None,
        text_pad_mask=None,
        ctrlNet_cond=None,
    ):
        if force_mask:
            return self.trans_forward(
                motion_emb,
                cond_vector,
                padding_mask,
                force_mask=True,
                text_seq=text_seq,
                text_pad_mask=text_pad_mask,
                ctrlNet_cond=ctrlNet_cond,
            )

        logits = self.trans_forward(
            motion_emb,
            cond_vector,
            padding_mask,
            force_mask=False,
            text_seq=text_seq,
            text_pad_mask=text_pad_mask,
            ctrlNet_cond=ctrlNet_cond,
        )
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(
            motion_emb,
            cond_vector,
            padding_mask,
            force_mask=True,
            text_seq=text_seq,
            text_pad_mask=text_pad_mask,
            ctrlNet_cond=ctrlNet_cond,
        )
        return aux_logits + (logits - aux_logits) * cond_scale

    # ------------------------------------------------------------------
    # Generation (iterative masked decoding)
    # ------------------------------------------------------------------

    @torch.no_grad()
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
    ):
        device = next(self.parameters()).device
        require(self.Q > 0, f"num_quantizers must be > 0, got {self.Q}")

        m_lens_t = torch.as_tensor(m_lens, device=device)
        if m_lens_t.ndim == 0:
            m_lens_t = m_lens_t.unsqueeze(0)
        batch_size = m_lens_t.shape[0]

        default_seq_len = int(getattr(self.opt, "max_token_len", 49))
        seq_len = int(m_lens_t.max().item()) if m_lens_t.numel() > 0 else default_seq_len
        seq_len = max(1, seq_len)
        require(
            seq_len <= self.max_time_tokens,
            f"generate seq_len={seq_len} exceeds max_time_tokens={self.max_time_tokens}.",
        )

        # ---- text encoding (v2: pooled + sequence) ----
        text_seq, text_pad_mask = None, None
        if conds is None:
            if self.cond_mode == "uncond":
                cond_vector = torch.zeros((batch_size, self.latent_dim), device=device)
            elif self.cond_mode == "action":
                cond_vector = torch.zeros((batch_size, self.num_actions), device=device)
            else:
                cond_vector = torch.zeros((batch_size, self.clip_dim), device=device)
        else:
            cond_vector, text_seq, text_pad_mask = self.encode_text_with_seq(conds)

        # ---- setup (identical to v1) ----
        S = seq_len * self.Q
        m_lens_t = m_lens_t.clamp(min=0, max=seq_len)
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
        last_logits = None

        for timestep in torch.linspace(0, 1, timesteps, device=device):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked_t = (m_lens_t * rand_mask_prob).round().clamp(min=1)

            frame_scores = scores.view(batch_size, seq_len, self.Q).mean(dim=-1)
            sorted_indices = frame_scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask_t = ranks < num_token_masked_t.unsqueeze(-1)
            is_mask_t &= ~padding_mask_t
            is_mask = (
                is_mask_t.unsqueeze(-1)
                .expand(batch_size, seq_len, self.Q)
                .reshape(batch_size, S)
            )

            emb = self.token_emb(ids)
            mask_emb = self.token_emb.weight[self.mask_id]
            emb = torch.where(is_mask.unsqueeze(-1), mask_emb, emb)

            logits = self.forward_with_cond_scale(
                emb,
                cond_vector=cond_vector,
                padding_mask=padding_mask_seq,
                cond_scale=cond_scale,
                force_mask=force_mask,
                text_seq=text_seq,
                text_pad_mask=text_pad_mask,
            )
            last_logits = logits

            logits_bsv = logits.permute(0, 2, 1).contiguous()

            if topk_filter_thres > 0:
                logits_bsv = top_k(logits_bsv, topk_filter_thres, dim=-1)

            if gsample:
                pred_local = gumbel_sample(
                    logits_bsv, temperature=starting_temperature, dim=-1
                )
                probs_local = F.softmax(logits_bsv / max(temperature, 1e-8), dim=-1)
            else:
                probs_local = F.softmax(logits_bsv / max(temperature, 1e-8), dim=-1)
                pred_local = Categorical(probs_local).sample()

            pred_ids = pred_local
            ids = torch.where(is_mask, pred_ids, ids)

            conf = probs_local.gather(-1, pred_local.unsqueeze(-1)).squeeze(-1)
            frame_conf = conf.view(batch_size, seq_len, self.Q).mean(dim=-1)
            frame_scores = frame_conf.masked_fill(~is_mask_t, 1e5)
            frame_scores = frame_scores.masked_fill(padding_mask_t, 1e5)
            scores = (
                frame_scores.unsqueeze(-1)
                .expand(batch_size, seq_len, self.Q)
                .reshape(batch_size, S)
            )

        ids_list = _unpack_local_seq_to_list(ids, padding_mask_seq=padding_mask_seq, Q=self.Q)
        return ids_list, last_logits

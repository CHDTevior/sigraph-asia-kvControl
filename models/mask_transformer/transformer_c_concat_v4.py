"""c_concat MaskTransformer v4 — c_concat backbone + v4 dual-stream text path.

Paired ablation against transformer_t_concat_v4.py: same v4 dual-stream text
path (CLIP sequence tokens projected → CLIPTextAdapter → sparse cross-attention
gated residuals every cross_attn_interval self-attn layers, with cond_keep CFG
mask), but the spatio-temporal backbone is the original c_concat v1 layout
(channel-axis token concat, per-Q codebook embeddings, per-Q output heads,
sequence length S = T + 1 = 50, 1-D positional encoding, NO 2D pos emb,
NO factorized attention, NO mask_2d_hybrid, NO sequence-axis token unpacking).

Goal: isolate the *backbone layout* contribution (channel-concat vs T-concat
sequence-axis) by holding the text architecture identical.

Inheritance:
  models.mask_transformer.transformer.MaskTransformer  (c_concat v1, parent)
    |
    +— this class (adds v4 text path: text_seq_proj + text_adapter + cross_attn)

The added text path is identical (parameter-for-parameter) to v4's:
  - text_seq_proj: bare nn.Linear(clip_dim, latent_dim) — no LayerNorm
  - text_adapter:  CLIPTextAdapter (4-layer TransformerEncoder + LN)
  - text_cross_attn: ModuleList of TextCrossAttentionBlock, one per
                     cross_attn layer index
  - cross_attn_layer_ids: {2*i + (interval-1) for i in range(L // interval)}
                          (default interval=2, L=20 → {1,3,5,...,19})
  - gate_init = 0.01

Config flags (via opt / command line):
  --cross_attn_interval N   Insert cross-attention every N layers (default 2)
  --cross_attn_heads N      Heads for cross-attention (default: n_heads)
  --text_adapter_layers N   Number of CLIPTextAdapter layers (default 4)
  --gate_init F             Initial value for cross-attn gate params (default 0.01)

Version: c_concat_v4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torch.distributions.categorical import Categorical

from models.mask_transformer.transformer_c_concat import MaskTransformer as MaskTransformerC1
from models.mask_transformer.transformer_t_concat_v2 import TextCrossAttentionBlock
from models.mask_transformer.transformer_t_concat_v4 import CLIPTextAdapter
from models.mask_transformer.tools import (
    lengths_to_mask,
    uniform,
    eval_decorator,
    top_k,
    gumbel_sample,
    get_mask_subset_prob,
    cal_performance2,
)


# ---------------------------------------------------------------------------
# c_concat MaskTransformer v4
# ---------------------------------------------------------------------------

class MaskTransformer(MaskTransformerC1):
    """c_concat backbone (per-Q heads, channel-concat input, S=T+1) +
    v4 dual-stream text path (CLIP-seq → adapter → sparse gated cross-attn)."""

    # v4-specific defaults — applied when parser provides sentinel values (-1)
    _V4_DEFAULTS = {
        "cross_attn_interval": 2,
        "text_adapter_layers": 4,
        "gate_init": 0.01,
    }

    def __init__(self, **kwargs):
        # ---- enforce v4 defaults BEFORE building parent backbone (parent does
        # not read these flags but we record them on opt for downstream use) ----
        opt = kwargs.get("opt")
        if opt is not None:
            for key, default in self._V4_DEFAULTS.items():
                cur = getattr(opt, key, None)
                # Only override sentinel -1 / None — explicit user values pass through
                if cur is None or cur < 0:
                    setattr(opt, key, default)

        # Build c_concat v1 backbone untouched (input_process, position_enc,
        # seqTransEncoder, cond_emb, token_embs[0..5], output_processes[0..5],
        # CLIP frozen, etc.).
        super().__init__(**kwargs)

        opt = self.opt
        num_layers = len(self.seqTransEncoder.layers)

        # ---- v4 text-path config (now guaranteed to be real values) ----
        cross_attn_interval = int(getattr(opt, "cross_attn_interval", 2))
        text_adapter_layers = int(getattr(opt, "text_adapter_layers", 4))
        gate_init = float(getattr(opt, "gate_init", 0.01))
        nhead_cfg = getattr(opt, "cross_attn_heads", None)
        nhead = (
            int(nhead_cfg)
            if nhead_cfg
            else int(getattr(opt, "n_heads", None) or 8)
        )

        # ---- which self-attn layers get text cross-attention (0-indexed,
        #      applied AFTER layer i). Matches v4 default with interval=2,
        #      L=20 → {1,3,5,...,19}. ----
        if cross_attn_interval > 0 and cross_attn_interval <= num_layers:
            self.cross_attn_layer_ids = set(
                range(cross_attn_interval - 1, num_layers, cross_attn_interval)
            )
        else:
            self.cross_attn_layer_ids = set()
        self._cross_attn_sorted = sorted(self.cross_attn_layer_ids)
        n_cross = len(self.cross_attn_layer_ids)

        self._has_cross_attn = n_cross > 0

        if self._has_cross_attn:
            assert text_adapter_layers >= 1, (
                f"[c_concat_v4] text_adapter_layers must be >= 1, "
                f"got {text_adapter_layers}."
            )

            # ---- bare Linear text_seq_proj (no LayerNorm — adapter's LN
            #      replaces it, matching v4) ----
            self.text_seq_proj = nn.Linear(self.clip_dim, self.latent_dim)
            self.text_seq_proj.apply(self.init_weights)

            # ---- CLIP text adapter (identical to v4) ----
            self.text_adapter = CLIPTextAdapter(
                d_model=self.latent_dim,
                nhead=nhead,
                dim_feedforward=self.latent_dim * 4,
                num_layers=text_adapter_layers,
                dropout=self.dropout,
            )
            self.text_adapter.apply(self.init_weights)

            # ---- text cross-attention blocks (one per skipped self-attn
            #      layer). Default-init then re-init gate to gate_init. ----
            self.text_cross_attn = nn.ModuleList(
                [
                    TextCrossAttentionBlock(self.latent_dim, nhead, self.dropout)
                    for _ in range(n_cross)
                ]
            )
            with torch.no_grad():
                for block in self.text_cross_attn:
                    nn.init.constant_(block.gate, gate_init)

            # ---- layer-index → cross-attn-index map (sorted ascending) ----
            self._layer_to_cross_idx = {
                li: ci for ci, li in enumerate(self._cross_attn_sorted)
            }

            # ---- report ----
            n_proj = sum(p.numel() for p in self.text_seq_proj.parameters())
            n_adapter = sum(p.numel() for p in self.text_adapter.parameters())
            n_cross_p = sum(p.numel() for p in self.text_cross_attn.parameters())
            n_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(
                f"[c_concat_v4] Cross-attention every {cross_attn_interval} "
                f"layers → {n_cross} blocks at layers {self._cross_attn_sorted}"
            )
            print(
                f"[c_concat_v4] text_seq_proj (bare Linear): "
                f"{n_proj / 1e6:.2f}M params"
            )
            print(
                f"[c_concat_v4] Text adapter: {text_adapter_layers} layers, "
                f"d_model={self.latent_dim}, nhead={nhead}, "
                f"ff={self.latent_dim * 4} | {n_adapter / 1e6:.1f}M params"
            )
            print(
                f"[c_concat_v4] Cross-attn blocks: "
                f"{n_cross_p / 1e6:.1f}M params (gate_init={gate_init})"
            )
            print(f"[c_concat_v4] Total trainable: {n_total / 1e6:.1f}M")
        else:
            # No cross-attention — degenerate to plain c_concat v1
            self.text_seq_proj = None
            self.text_adapter = None
            self.text_cross_attn = nn.ModuleList()
            self._layer_to_cross_idx = {}
            print(
                f"[c_concat_v4] Cross-attention DISABLED "
                f"(interval={cross_attn_interval}, num_layers={num_layers})"
            )
            n_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(
                f"[c_concat_v4] Total trainable: {n_total / 1e6:.1f}M "
                f"(no text adapter/proj/cross-attn)"
            )

    # ------------------------------------------------------------------
    # Text encoding — extract pooled + sequence tokens (v2-style)
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

        # pooled (same path as encode_text)
        pooled = (
            x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
            @ cm.text_projection
        )

        # padding mask (CLIP pads with 0)
        text_pad_mask = text_tokens == 0  # True = padding

        return pooled.float(), x.float(), text_pad_mask

    # ------------------------------------------------------------------
    # Condition masking — shared between pooled token and cross-attn
    # ------------------------------------------------------------------

    def _get_cond_mask(self, bs, device, force_mask=False):
        """Per-sample binary mask [B, 1].  1.0 = keep, 0.0 = drop."""
        if force_mask:
            return torch.zeros(bs, 1, device=device)
        if self.training and self.cond_drop_prob > 0.0:
            drop = torch.bernoulli(
                torch.ones(bs, device=device) * self.cond_drop_prob
            )
            return (1.0 - drop).view(bs, 1)
        return torch.ones(bs, 1, device=device)

    # ------------------------------------------------------------------
    # Transformer forward — c_concat backbone + sparse text cross-attn
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
        """c_concat trans_forward with v4 text cross-attention injection.

        Args:
            motion_emb:   [B, T, 6*code_dim]  channel-concat embeddings
            cond:         [B, clip_dim]       pooled CLIP embedding
            padding_mask: [B, T]              True = pad
            text_seq:     [B, L, clip_dim]    CLIP sequence tokens (optional)
            text_pad_mask:[B, L]              True = padding (optional)
        """
        bs = cond.shape[0]
        cond_keep = self._get_cond_mask(bs, cond.device, force_mask)

        # ---- pooled condition (gated by cond_keep, replaces v1 mask_cond) ----
        cond = cond * cond_keep

        # (B, T, 6*code_dim) -> (T, B, latent_dim)
        x = self.input_process(motion_emb)
        cond_token = self.cond_emb(cond).unsqueeze(0)  # [1, B, latent_dim]

        x = self.position_enc(x)
        xseq = torch.cat([cond_token, x], dim=0)  # [T+1, B, latent_dim]

        # padding_mask: [B, T+1] — first slot = cond, never masked
        padding_mask = torch.cat(
            [torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1
        )

        # ---- text memory for cross-attention ----
        if text_seq is not None and self._has_cross_attn:
            text_memory = self.text_seq_proj(text_seq)  # [B, L, D]
            text_memory = self.text_adapter(
                text_memory, src_key_padding_mask=text_pad_mask
            )  # [B, L, D]
            text_memory = text_memory * cond_keep.unsqueeze(-1)  # zero dropped
            text_memory = text_memory.permute(1, 0, 2)  # [L, B, D]
        else:
            text_memory = None

        # ---- layer-by-layer self-attn with sparse cross-attn injection ----
        output = xseq
        cross_idx = 0
        for i, layer in enumerate(self.seqTransEncoder.layers):
            output = layer(output, src_key_padding_mask=padding_mask)
            if i in self.cross_attn_layer_ids:
                if text_memory is not None:
                    output = self.text_cross_attn[cross_idx](
                        output,
                        text_memory,
                        text_pad_mask,
                        cond_keep=cond_keep,
                    )
                cross_idx += 1

        # Apply final encoder norm if present (matches nn.TransformerEncoder
        # behavior when norm=None vs norm=LayerNorm).
        if (
            hasattr(self.seqTransEncoder, "norm")
            and self.seqTransEncoder.norm is not None
        ):
            output = self.seqTransEncoder.norm(output)

        # Drop the cond token row → [T, B, D]
        output = output[1:]

        # ---- per-Q output heads (c_concat v1 layout) ----
        logits_list = [head(output) for head in self.output_processes]  # 6 * [B, K, T]
        logits = torch.stack(logits_list, dim=1)  # [B, 6, K, T]
        return logits

    # ------------------------------------------------------------------
    # Training forward — same as c_concat v1 but pulls v4 text path
    # ------------------------------------------------------------------

    def forward(self, y, m_lens, pose, vq_model):
        """y: text or action. m_lens: (B,). pose: [B, 263, T]."""
        device = pose.device

        # ----- VQ encode (identical to c_concat v1) -----
        pose_4d = pose.permute(0, 2, 1).unsqueeze(2)  # [B, T, 1, 263]
        code_idx = vq_model.get_code_idx(pose_4d)  # [B, 6, T]
        code_idx = code_idx.permute(0, 2, 1)  # [B, T, 6]

        B, T, Q = code_idx.shape
        bs, ntokens = B, T
        assert Q == 6, "c_concat_v4 default 6 codebooks"

        ids_list = [code_idx[..., i].clone() for i in range(Q)]
        non_pad_mask = lengths_to_mask(m_lens, ntokens).to(device)  # [B, T]

        for i in range(Q):
            ids_list[i] = torch.where(
                non_pad_mask, ids_list[i], torch.full_like(ids_list[i], self.pad_id)
            )

        # ----- condition vector (v4: pooled + sequence) -----
        force_mask = False
        text_seq, text_pad_mask = None, None
        if self.cond_mode == "text":
            with torch.no_grad():
                cond_vector, text_seq, text_pad_mask = self.encode_text_with_seq(y)
        elif self.cond_mode == "action":
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == "uncond":
            cond_vector = torch.zeros(bs, self.latent_dim, device=device).float()
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        # ----- BERT-style mask strategy (identical to c_concat v1) -----
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)  # [B]
        num_token_masked = (m_lens * rand_mask_probs).round().clamp(min=1)  # [B]

        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)  # [B, T]
        mask &= non_pad_mask  # only mask non-pad

        labels_list = [ids_list[i].clone() for i in range(Q)]
        x_ids_list = [ids_list[i].clone() for i in range(Q)]

        # 10% masked → random token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        for i in range(Q):
            rand_ids = torch.randint(
                low=0,
                high=self.opt.num_tokens,
                size=x_ids_list[i].shape,
                device=device,
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
        # remaining ~2% keep original token (BERT habit)

        # ----- channel-concat noisy embedding (c_concat v1 layout) -----
        num_cb_list = [
            vq_model.vqvae.quantizers[i].codebook.shape[0] for i in range(Q)
        ]
        std = 0.1
        trans_emb_parts = []

        for i in range(Q):
            num_cb = num_cb_list[i]
            raw_ids = x_ids_list[i]  # [B, T]
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

        trans_emb_all = torch.cat(trans_emb_parts, dim=-1)  # [B, T, 6*code_dim]

        # ----- Transformer + per-Q CE loss (v4 text path threaded in) -----
        padding_mask = ~non_pad_mask  # TRUE = pad
        logits = self.trans_forward(
            trans_emb_all,
            cond_vector,
            padding_mask,
            force_mask,
            text_seq=text_seq,
            text_pad_mask=text_pad_mask,
        )  # [B, 6, K, T]

        ce_loss, pred_id, acc = cal_performance2(
            logits, labels_list, ignore_index=self.pad_id
        )

        # ----- embedding alignment loss (identical to c_concat v1) -----
        logits_for_emb = logits.permute(0, 1, 3, 2)  # [B, 6, T, K]

        emb_list = [
            F.softmax(logits_for_emb[:, i, :, :], dim=-1)
            @ vq_model.vqvae.quantizers[i].codebook
            for i in range(Q)
        ]

        emb_list = [
            emb_list[i].masked_fill(padding_mask.unsqueeze(-1), 0.0)
            for i in range(Q)
        ]

        x_encoder_list = vq_model.vqvae.get_quantized_codes(
            pose_4d, in_idx_format=False
        )

        loss_emb = 0.0
        for i in range(Q):
            x_enc = x_encoder_list[i].permute(0, 2, 1)
            loss_emb += (
                F.l1_loss(
                    x_enc[non_pad_mask],
                    emb_list[i][non_pad_mask],
                    reduction="mean",
                )
                / Q
            )

        return loss_emb, ce_loss, pred_id, acc

    # ------------------------------------------------------------------
    # CFG forward — thread text_seq / text_pad_mask through both branches
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
    # Generate — c_concat v1 iterative sampling + v4 text path threading
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
        seq_len = 49  # c_concat v1 default
        batch_size = len(m_lens)
        m_lens = (
            m_lens.to(device)
            if isinstance(m_lens, torch.Tensor)
            else torch.tensor(m_lens, device=device, dtype=torch.long)
        )

        # ---- text encoding (v4: pooled + sequence) ----
        text_seq, text_pad_mask = None, None
        if conds is None:
            cond_vector = torch.zeros((batch_size, self.clip_dim), device=device)
        else:
            cond_vector, text_seq, text_pad_mask = self.encode_text_with_seq(conds)

        padding_mask = ~lengths_to_mask(m_lens, seq_len).to(device)  # TRUE = pad

        init_ids = torch.where(
            padding_mask,
            torch.full((batch_size, seq_len), self.pad_id, device=device),
            torch.full((batch_size, seq_len), self.mask_id, device=device),
        )
        ids_list = [init_ids.clone() for _ in range(6)]

        scores = torch.where(
            padding_mask,
            torch.full_like(init_ids, 1e5, dtype=torch.float),
            torch.zeros_like(init_ids, dtype=torch.float),
        )

        starting_temperature = temperature
        last_logits = None

        for timestep in torch.linspace(0, 1, timesteps, device=device):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = (m_lens * rand_mask_prob).round().clamp(min=1)

            sorted_indices = scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = ranks < num_token_masked.unsqueeze(-1)
            is_mask &= ~padding_mask

            emb_parts = []
            for i in range(6):
                emb_i = self.token_embs[i](ids_list[i])  # [B, T, Cdim]
                mask_emb_i = self.token_embs[i].weight[self.mask_id]
                emb_i = torch.where(is_mask.unsqueeze(-1), mask_emb_i, emb_i)
                emb_parts.append(emb_i)

            emb_all = torch.cat(emb_parts, dim=-1)  # [B, T, 6*code_dim]

            logits = self.forward_with_cond_scale(
                emb_all,
                cond_vector=cond_vector,
                padding_mask=padding_mask,
                cond_scale=cond_scale,
                force_mask=force_mask,
                text_seq=text_seq,
                text_pad_mask=text_pad_mask,
            )  # [B, 6, K, T]
            last_logits = logits

            logits_btqk = logits.permute(0, 1, 3, 2)  # [B, 6, T, K]

            pred_ids_list = []
            probs_list = []
            for i in range(6):
                logit_i = logits_btqk[:, i, :, :]  # [B, T, K]

                if topk_filter_thres > 0:
                    logit_i = top_k(logit_i, topk_filter_thres, dim=-1)

                if gsample:
                    pred_ids_i = gumbel_sample(
                        logit_i, temperature=starting_temperature, dim=-1
                    )
                    probs_i = F.softmax(logit_i / max(temperature, 1e-8), dim=-1)
                else:
                    probs_i = F.softmax(logit_i / max(temperature, 1e-8), dim=-1)
                    pred_ids_i = Categorical(probs_i).sample()

                pred_ids_list.append(pred_ids_i)
                probs_list.append(probs_i)

            for i in range(6):
                ids_list[i] = torch.where(is_mask, pred_ids_list[i], ids_list[i])

            scores_q = []
            for i in range(6):
                conf_i = (
                    probs_list[i]
                    .gather(-1, pred_ids_list[i].unsqueeze(-1))
                    .squeeze(-1)
                )
                scores_q.append(conf_i)

            scores = torch.stack(scores_q, dim=1).mean(dim=1)  # [B, T]
            scores = scores.masked_fill(~is_mask, 1e5)
            scores = scores.masked_fill(padding_mask, 1e5)

        # mark pad slots with -1 (matches c_concat v1 contract)
        for i in range(6):
            ids_list[i] = torch.where(
                padding_mask, torch.full_like(ids_list[i], -1), ids_list[i]
            )

        return ids_list, last_logits

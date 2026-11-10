"""T-concat MaskTransformer v4 — v2 + CLIP text adapter + small gate init.

Changes from v2 (transformer_t_concat_v2.py):
  1. CLIP Text Adapter: a learned TransformerEncoder (4 layers by default) processes
     the projected CLIP sequence tokens before they are used as cross-attention K/V.
     Inspired by StableMoFusion (Huang et al., 2024): Linear -> TransformerEncoder -> LN.
     Reference: references/stablemofusion/models/unet.py lines 488-531.
     The adapter is applied AFTER text_seq_proj (Linear only, no LayerNorm) and BEFORE
     the cond_keep mask, so the adapter's final LayerNorm replaces the proj's old LN.

  2. text_seq_proj simplified: now a bare nn.Linear(clip_dim, latent_dim) instead of
     nn.Sequential(Linear, LayerNorm), because the adapter's trailing LN fills that role.

  3. Gate init = 0.01 (configurable via opt.gate_init): all cross-attention gates start
     at a small positive value so text signal flows immediately. This matches the v3
     cold-start fix (transformer_t_concat_v3.py line 166). Unlike v2's gate=0 which
     relies entirely on the prepended pooled token for early text signal, v4 allows
     cross-attention to contribute from step 0.

All other behavior (forward, generate, encode_text_with_seq, CFG, etc.) is inherited
from v2 unchanged. The only override is trans_forward, which inserts one line to call
the text adapter.

Config (via opt / command line):
  --text_adapter_layers N   Number of TransformerEncoder layers (default 4)
  --gate_init F             Initial value for cross-attn gate params (default 0.01)
  --cross_attn_interval N   Inherited from v2
  --cross_attn_heads N      Inherited from v2

Version: v4
Base: transformer_t_concat_v2.py (v2)
"""

import torch
import torch.nn as nn

from models.mask_transformer.transformer_t_concat_v2 import (
    MaskTransformer as MaskTransformerV2,
)


# ---------------------------------------------------------------------------
# CLIP Text Adapter (inspired by StableMoFusion)
# ---------------------------------------------------------------------------

class CLIPTextAdapter(nn.Module):
    """Learned adapter that refines projected CLIP sequence tokens.

    Architecture (faithful to StableMoFusion's text encoder pattern):
      TransformerEncoder (N layers, batch_first=False) -> LayerNorm

    The input is already projected from clip_dim to d_model by text_seq_proj.
    StableMoFusion uses batch_first=False and permutes [B,L,D] -> [L,B,D] before
    the encoder, then back after; we follow the same convention for compatibility.

    Args:
        d_model:          Hidden dimension (= latent_dim, typically 1024).
        nhead:            Number of attention heads.
        dim_feedforward:  FFN intermediate dimension (= d_model * 4).
        num_layers:       Number of TransformerEncoder layers.
        dropout:          Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # match StableMoFusion convention
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: [B, L, D] — projected CLIP text tokens.
            src_key_padding_mask: optional [B, L] bool, True = padded. Restores the
                May-worktree signature that transformer_c_concat_v4.trans_forward
                calls with (lost in the 2026-07-02 reset; GATE0 follow-up fix F7).
                t_concat_v4's own positional call `text_adapter(text_memory)` keeps
                mask=None → behavior identical to the committed HEAD version.
        Returns:
            [B, L, D] — adapter-refined text tokens.
        """
        # [B, L, D] -> [L, B, D] for batch_first=False encoder
        x = x.permute(1, 0, 2)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.ln(x)
        # [L, B, D] -> [B, L, D]
        x = x.permute(1, 0, 2)
        return x


# ---------------------------------------------------------------------------
# MaskTransformer v4
# ---------------------------------------------------------------------------

class MaskTransformer(MaskTransformerV2):
    """T-concat v4: v2 + CLIP text adapter + small gate init.

    Three changes from v2:
      1. text_seq_proj = bare Linear (adapter's LN replaces the old proj LN)
      2. self.text_adapter = CLIPTextAdapter inserted after text_seq_proj
      3. All cross-attn gates re-initialized to gate_init (default 0.01)

    Everything else (forward, forward_with_cond_scale, generate,
    encode_text_with_seq) is inherited from v2 unchanged.
    """

    # v4-specific defaults — applied when parser provides generic safe values
    _V4_DEFAULTS = {
        "text_adapter_layers": 4,
        "gate_init": 0.01,
        "cross_attn_interval": 2,
    }

    def __init__(self, **kwargs):
        # ---- enforce v4 defaults before super().__init__ builds cross-attn blocks ----
        opt = kwargs.get("opt")
        if opt is not None:
            for key, default in self._V4_DEFAULTS.items():
                cur = getattr(opt, key, None)
                # Apply v4 default only when the parser-level "safe zero" is in effect
                if key == "text_adapter_layers" and (cur is None or cur == 0):
                    setattr(opt, key, default)
                elif key == "gate_init" and (cur is None or cur == 0.0):
                    setattr(opt, key, default)
                elif key == "cross_attn_interval" and (cur is None or cur == 4):
                    setattr(opt, key, default)

        super().__init__(**kwargs)

        # ---- v4.8/v4.9: factorized attention ----
        self.factorized_attn = getattr(self.opt, "factorized_attn", "none")

        opt = self.opt

        # ---- v4 config (now guaranteed to have real values) ----
        text_adapter_layers = int(getattr(opt, "text_adapter_layers", 4))
        gate_init = float(getattr(opt, "gate_init", 0.01))
        nhead_cfg = getattr(opt, "cross_attn_heads", None)
        nhead = int(nhead_cfg) if nhead_cfg else int(getattr(opt, "n_heads", 8))

        assert text_adapter_layers >= 1, (
            f"[v4] text_adapter_layers must be >= 1, got {text_adapter_layers}. "
            f"Use t_concat_v2 if you do not want a text adapter."
        )

        # ---- Change 1: replace text_seq_proj with bare Linear (no LayerNorm) ----
        # The adapter's trailing LN replaces the old proj LayerNorm.
        self.text_seq_proj = nn.Linear(self.clip_dim, self.latent_dim)
        self.text_seq_proj.apply(self.init_weights)

        # ---- Change 2: CLIP text adapter ----
        self.text_adapter = CLIPTextAdapter(
            d_model=self.latent_dim,
            nhead=nhead,
            dim_feedforward=self.latent_dim * 4,
            num_layers=text_adapter_layers,
            dropout=self.dropout,
        )
        self.text_adapter.apply(self.init_weights)

        # ---- Change 3: re-init all cross-attn gates to gate_init ----
        with torch.no_grad():
            for block in self.text_cross_attn:
                nn.init.constant_(block.gate, gate_init)

        # ---- report ----
        n_adapter = sum(p.numel() for p in self.text_adapter.parameters())
        n_proj = sum(p.numel() for p in self.text_seq_proj.parameters())
        n_cross = sum(p.numel() for p in self.text_cross_attn.parameters())
        n_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[v4] Text adapter: {text_adapter_layers} layers, "
            f"d_model={self.latent_dim}, nhead={nhead}, "
            f"ff={self.latent_dim * 4} | {n_adapter / 1e6:.1f}M params"
        )
        print(f"[v4] text_seq_proj (bare Linear): {n_proj / 1e6:.2f}M params")
        print(f"[v4] Cross-attn blocks: {n_cross / 1e6:.1f}M params (gate_init={gate_init})")
        print(f"[v4] Total trainable: {n_total / 1e6:.1f}M")

    # ------------------------------------------------------------------
    # v4.8/v4.9: factorized attention mask
    # ------------------------------------------------------------------

    def _build_factorized_attn_mask(self, S_plus_1: int, device: torch.device) -> torch.Tensor:
        """Build boolean attention mask for factorized layers. True = blocked.

        For same_q: token i attends only to tokens with same quantizer index.
        For semantic_group: locomotion (Q=0,2,4) and manipulation (Q=1,3,5).
        Position 0 (text cond token) always attends to / is attended by all.
        """
        S = S_plus_1 - 1
        T = S // self.Q
        mask = torch.ones(S_plus_1, S_plus_1, dtype=torch.bool, device=device)

        # Cond token (pos 0) can attend everywhere and be attended by all
        mask[0, :] = False
        mask[:, 0] = False

        if self.factorized_attn == "same_q":
            # Token at flat index i (1-based in S+1) -> q = (i-1) % Q
            for q in range(self.Q):
                # Indices in the S+1 sequence: 1+q, 1+q+Q, 1+q+2Q, ...
                indices = torch.arange(q, S, self.Q, device=device) + 1  # +1 for cond token
                # Allow mutual attention within this quantizer stream
                mask[indices.unsqueeze(0), indices.unsqueeze(1)] = False
        elif self.factorized_attn == "semantic_group":
            assert self.Q == 6, f"semantic_group assumes 6 quantizers, got {self.Q}"
            groups = [[0, 2, 4], [1, 3, 5]]  # legacy binary split: locomotion, manipulation
            for group_qs in groups:
                indices_list = []
                for q in group_qs:
                    indices_list.append(torch.arange(q, S, self.Q, device=device) + 1)
                indices = torch.cat(indices_list)
                mask[indices.unsqueeze(0), indices.unsqueeze(1)] = False
        elif self.factorized_attn == "semantic_group_v2":
            assert self.Q == 6, f"semantic_group_v2 assumes 6 quantizers, got {self.Q}"
            # Anatomy-aware 3-group banded connectivity.
            # VQ partition joint coverage (by majority):
            #   Q=0: root, l_ankle                         → lower body
            #   Q=1: r_knee, head, both arms (6 joints)    → upper body (majority)
            #   Q=2: l_knee, r_hip, r_foot, spine3         → lower body (majority)
            #   Q=3: spine1, neck                           → torso core
            #   Q=4: l_hip, l_foot, r_ankle, spine2        → lower body (majority)
            #   Q=5: l_shoulder                             → upper body
            # Q=2 is anatomically mixed (lower-majority) but reassigned to core
            # to serve as bridge — it contains knee/hip joints that coordinate
            # upper and lower body during locomotion.
            # Core group bridges upper↔lower; non-adjacent groups must
            # communicate through core or full-attention even layers.
            lower = [0, 4]   # root / ankles / feet / hips
            core  = [2, 3]   # knees / hip / spine / neck (bridge)
            upper = [1, 5]   # arms / head / shoulders

            def _get_indices(qs):
                return torch.cat([torch.arange(q, S, self.Q, device=device) + 1 for q in qs])

            # Within-group self-attention
            for group in (lower, core, upper):
                idx = _get_indices(group)
                mask[idx.unsqueeze(0), idx.unsqueeze(1)] = False
            # Adjacent cross-group attention (lower↔core, core↔upper)
            for g1, g2 in ((lower, core), (core, upper)):
                idx1, idx2 = _get_indices(g1), _get_indices(g2)
                mask[idx1.unsqueeze(0), idx2.unsqueeze(1)] = False
                mask[idx2.unsqueeze(0), idx1.unsqueeze(1)] = False

        return mask  # [S+1, S+1], True = blocked

    # ------------------------------------------------------------------
    # Transformer forward — insert text adapter after text_seq_proj
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
        """Forward pass with text adapter + sparse text cross-attention.

        Identical to v2 trans_forward except for one inserted line:
            text_memory = self.text_adapter(text_memory)
        applied after text_seq_proj and before cond_keep masking.

        Args:
            motion_emb:   [B, S, code_dim]   S = T * Q
            cond:         [B, clip_dim]       pooled CLIP embedding
            padding_mask: [B, S]              True = pad
            text_seq:     [B, L, clip_dim]    CLIP sequence tokens (optional)
            text_pad_mask:[B, L]              True = padding (optional)
        """
        bs = cond.shape[0]
        cond_keep = self._get_cond_mask(bs, cond.device, force_mask)

        # ---- pooled condition (same as v1/v2) ----
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
            # project then adapt then mask
            text_memory = self.text_seq_proj(text_seq)          # [B, L, D]
            text_memory = self.text_adapter(text_memory)         # [B, L, D]  <- NEW in v4
            text_memory = text_memory * cond_keep.unsqueeze(-1)  # zero dropped samples
            text_memory = text_memory.permute(1, 0, 2)           # [L, B, D]
        else:
            text_memory = None

        # ---- v4.8/v4.9: build factorized mask for odd layers ----
        if self.factorized_attn != "none":
            factorized_mask = self._build_factorized_attn_mask(xseq.shape[0], xseq.device)
        else:
            factorized_mask = None

        # ---- layer-by-layer with cross-attention injection ----
        cross_idx = 0
        for i, layer in enumerate(self.seqTransEncoder.layers):
            use_factorized = factorized_mask is not None and (i % 2 == 1)
            src_mask = factorized_mask if use_factorized else None
            xseq = layer(xseq, src_mask=src_mask, src_key_padding_mask=padding_mask)
            if i in self.cross_attn_layer_ids:
                if text_memory is not None:
                    xseq = self.text_cross_attn[cross_idx](
                        xseq, text_memory, text_pad_mask, cond_keep=cond_keep
                    )
                cross_idx += 1

        out = xseq[1:]  # [S, B, D]  -- drop cond token

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

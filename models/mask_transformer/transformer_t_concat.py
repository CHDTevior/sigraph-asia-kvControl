import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import clip
import os
import math
from einops import rearrange, repeat
from functools import partial
from pathlib import Path
from torch.distributions.categorical import Categorical

from models.mask_transformer.tools import *  # lengths_to_mask / uniform / cosine_schedule / eval_decorator / top_k / gumbel_sample / get_mask_subset_prob / etc.


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIP_CACHE_DIR = PROJECT_ROOT / "artifacts" / "models" / "clip"
CLIP_MODEL_PATH_ENV = "MASKCONTROL_CLIP_MODEL_PATH"
CLIP_CACHE_DIR_ENV = "MASKCONTROL_CLIP_CACHE_DIR"
CLIP_MODEL_FILENAMES = {
    "RN50": "RN50.pt",
    "RN101": "RN101.pt",
    "RN50x4": "RN50x4.pt",
    "RN50x16": "RN50x16.pt",
    "RN50x64": "RN50x64.pt",
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
    "ViT-L/14": "ViT-L-14.pt",
    "ViT-L/14@336px": "ViT-L-14-336px.pt",
}


def resolve_local_clip_checkpoint(clip_version: str) -> str:
    explicit_path = os.environ.get(CLIP_MODEL_PATH_ENV)
    if explicit_path:
        explicit = Path(explicit_path).expanduser().resolve()
        if not explicit.is_file():
            raise RuntimeError(
                f"{CLIP_MODEL_PATH_ENV} points to a missing file: {explicit}"
            )
        return str(explicit)

    version_path = Path(clip_version).expanduser()
    if version_path.is_file():
        return str(version_path.resolve())

    if clip_version not in CLIP_MODEL_FILENAMES:
        raise RuntimeError(
            f"Unsupported CLIP model '{clip_version}'. "
            f"Provide a local checkpoint path or one of {clip.available_models()}."
        )

    cache_dir = Path(
        os.environ.get(CLIP_CACHE_DIR_ENV, str(DEFAULT_CLIP_CACHE_DIR))
    ).expanduser()
    candidate = cache_dir / CLIP_MODEL_FILENAMES[clip_version]
    if candidate.is_file():
        return str(candidate.resolve())

    raise RuntimeError(
        f"Offline CLIP checkpoint not found for '{clip_version}'. "
        f"Expected local file at {candidate}. "
        "Download it once on a networked node and rerun; runtime downloads are disabled."
    )


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # x: [B, S, input_feats]
        x = x.permute((1, 0, 2))  # [S, B, input_feats]
        x = self.poseEmbedding(x)  # [S, B, latent_dim]
        return x


class PositionalEncoding(nn.Module):
    # Borrow from MDM
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [S, B, D]
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats)  # Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [S, B, D]
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [S, B, out_feats]
        output = output.permute(1, 2, 0)  # [B, out_feats, S]
        return output


def _unpack_local_seq_to_list(local_ids: torch.Tensor, padding_mask_seq: torch.Tensor, Q: int = 6):
    """
    local_ids: [B, S] where S=T*Q, local in [0..K-1]
    padding_mask_seq: [B, S] TRUE=pad
    return ids_list: Q * [B, T] (local in [0..K-1], pad位置=-1)
    """
    B, S = local_ids.shape
    assert S % Q == 0
    T = S // Q
    seq_ids = torch.where(padding_mask_seq, torch.full_like(local_ids, -1), local_ids)
    local_b6t = seq_ids.view(B, T, Q).permute(0, 2, 1).contiguous()
    return [local_b6t[:, i, :] for i in range(Q)]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class MaskTransformer(nn.Module):
    """
    ✅ 入口/函数名保持不变：
      - forward(self, y, m_lens, pose, vq_model)
      - generate(...)
      - trans_forward(...)
      - forward_with_cond_scale(...)
    ✅ 改动点：
      - 不再在 channel 维 concat 6 个 codebook embedding
      - 改为 seq 维展开：S = T*6
      - 词表采用 local-id：real [0..K-1], mask=K, pad=K+1
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
        **kargs,
    ):
        super().__init__()
        print(
            f"latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, "
            f"nheads: {num_heads}, dropout: {dropout}"
        )

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == "action":
            assert "num_actions" in kargs
        self.num_actions = kargs.get("num_actions", 1)

        # ========= local vocab =========
        self.K = opt.num_tokens
        self.Q = int(getattr(opt, "num_quantizers", 6))
        self.vocab = self.K
        self.mask_id = self.K
        self.pad_id = self.K + 1
        self.num_tokens_total = self.K + 2

        # ========= network =========
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        default_max_token_len = max(
            1,
            int(
                math.ceil(
                    getattr(opt, "max_motion_length", 196)
                    / max(1, getattr(opt, "unit_length", 4))
                )
            ),
        )
        self.max_time_tokens = int(getattr(opt, "max_token_len", default_max_token_len))
        self.quantizer_token_emb = nn.Embedding(self.Q, self.code_dim)
        self.time_pos_emb = nn.Embedding(self.max_time_tokens, self.latent_dim)
        self.quantizer_pos_emb = nn.Embedding(self.Q, self.latent_dim)
        self.embedding_dropout = nn.Dropout(p=self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=num_layers
        )

        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        if self.cond_mode == "text":
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == "action":
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == "uncond":
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")

        self.token_emb = nn.Embedding(self.num_tokens_total, self.code_dim)

        # v4.5: per-quantizer output heads
        self.per_q_head = getattr(opt, "per_q_head", False)
        if self.per_q_head:
            self.output_heads = nn.ModuleList([
                OutputProcess_Bert(out_feats=self.vocab, latent_dim=latent_dim)
                for _ in range(self.Q)
            ])
            self.output_head = None  # not used
        else:
            self.output_head = OutputProcess_Bert(out_feats=self.vocab, latent_dim=latent_dim)

        # v4.6: hybrid 2D masking
        self.mask_2d_hybrid = getattr(opt, "mask_2d_hybrid", False)
        self.mask_2d_ratio = getattr(opt, "mask_2d_ratio", 0.3)

        self.apply(self.init_weights)

        if self.cond_mode == "text":
            print("Loading CLIP...")
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        self.noise_schedule = cosine_schedule

    # --------------- utils ---------------

    def load_and_freeze_clip(self, clip_version):
        local_clip_path = resolve_local_clip_checkpoint(clip_version)
        print(f"Loading CLIP from local checkpoint: {local_clip_path}")
        clip_model, _ = clip.load(local_clip_path, device="cpu", jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def _require_vq_interface(self, vq_model) -> None:
        require(
            hasattr(vq_model, "get_code_idx"),
            "T-concat MaskTransformer requires vq_model.get_code_idx(...).",
        )
        require(
            hasattr(vq_model, "vqvae"),
            "T-concat MaskTransformer requires vq_model.vqvae.",
        )
        require(
            hasattr(vq_model.vqvae, "quantizers"),
            "T-concat MaskTransformer requires vq_model.vqvae.quantizers.",
        )
        require(
            hasattr(vq_model.vqvae, "get_quantized_codes"),
            "T-concat MaskTransformer requires vq_model.vqvae.get_quantized_codes(...).",
        )
        require(
            len(vq_model.vqvae.quantizers) == self.Q,
            f"T-concat expects {self.Q} quantizers, got {len(vq_model.vqvae.quantizers)}.",
        )

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith("clip_model.")]

    def _build_factorized_indices(self, seq_len: int, device: torch.device):
        require(seq_len > 0, f"sequence length must be > 0, got {seq_len}")
        require(
            seq_len % self.Q == 0,
            f"T-concat expects flattened sequence length divisible by {self.Q}, got {seq_len}.",
        )
        time_len = seq_len // self.Q
        require(
            time_len <= self.max_time_tokens,
            f"T-concat received time_len={time_len}, exceeding max_time_tokens={self.max_time_tokens}.",
        )
        time_ids = torch.arange(time_len, device=device).repeat_interleave(self.Q)
        quantizer_ids = torch.arange(self.Q, device=device).repeat(time_len)
        return time_ids, quantizer_ids

    def _inject_quantizer_token_identity(self, motion_emb: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = motion_emb.shape
        _time_ids, quantizer_ids = self._build_factorized_indices(seq_len, motion_emb.device)
        quantizer_emb = self.quantizer_token_emb(quantizer_ids).unsqueeze(0)
        return motion_emb + quantizer_emb

    def _add_factorized_position_encoding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len, _batch_size, _dim = hidden_states.shape
        time_ids, quantizer_ids = self._build_factorized_indices(seq_len, hidden_states.device)
        pos = self.time_pos_emb(time_ids) + self.quantizer_pos_emb(quantizer_ids)
        hidden_states = hidden_states + pos.unsqueeze(1)
        return self.embedding_dropout(hidden_states)

    def mask_cond(self, cond, force_mask=False):
        bs, _ = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_drop_prob
            ).view(bs, 1)
            return cond * (1.0 - mask)
        else:
            return cond

    # --------------- transformer forward ---------------

    def trans_forward(self, motion_emb, cond, padding_mask, force_mask=False, ctrlNet_cond=None):
        """
        motion_emb: (B, S, code_dim), S=T*6
        padding_mask: (B, S), TRUE=pad
        cond: (B, d)
        """
        cond = self.mask_cond(cond, force_mask=force_mask)

        motion_emb = self._inject_quantizer_token_identity(motion_emb)
        x = self.input_process(motion_emb)  # [S,B,D]
        cond = self.cond_emb(cond).unsqueeze(0)  # [1,B,D]

        x = self._add_factorized_position_encoding(x)
        xseq = torch.cat([cond, x], dim=0)  # [S+1,B,D]

        padding_mask = torch.cat(
            [torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1
        )

        out = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[1:]  # [S,B,D]

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
            logits = self.output_head(out)  # [B,vocab,S]
        return logits

    # --------------- training forward (入口不变) ---------------

    def forward(self, y, m_lens, pose, vq_model):
        device = pose.device
        self._require_vq_interface(vq_model)

        # ----- VQ encode -----
        pose_4d = pose.permute(0, 2, 1).unsqueeze(2)  # [B, T, 1, 263]
        code_idx = vq_model.get_code_idx(pose_4d)  # [B, 6, T]
        code_idx_bt6 = code_idx.permute(0, 2, 1)  # [B, T, 6]

        B, T, Q = code_idx_bt6.shape
        require(Q == self.Q, f"T-concat expects Q={self.Q}, got Q={Q}.")

        local_ids = code_idx_bt6.reshape(B, T * Q).contiguous()  # [B,S], local id
        S = local_ids.shape[1]

        # mask by length (frame -> token), 与 C-concat 对齐：先在 T 维构造 mask 再广播到 Q
        non_pad_mask_t = lengths_to_mask(m_lens, T).to(device)  # [B,T]
        non_pad_mask_seq = non_pad_mask_t.unsqueeze(-1).expand(B, T, Q).reshape(B, S)  # [B,S]
        padding_mask_seq = ~non_pad_mask_seq  # TRUE=pad

        labels = torch.where(
            non_pad_mask_seq, local_ids, torch.full_like(local_ids, self.pad_id)
        )

        # ----- cond vector -----
        force_mask = False
        if self.cond_mode == "text":
            with torch.no_grad():
                cond_vector = self.encode_text(y)  # [B,512]
        elif self.cond_mode == "action":
            cond_vector = self.encode_action(y).to(device).float()
        elif self.cond_mode == "uncond":
            cond_vector = torch.zeros(B, self.latent_dim, device=device).float()
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        # ----- BERT-style mask -----
        rand_time = uniform((B,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)  # [B]
        num_token_masked = (m_lens * rand_mask_probs).round().clamp(min=1)  # [B]

        batch_randperm = torch.rand((B, T), device=device).argsort(dim=-1)
        mask_t = batch_randperm < num_token_masked.unsqueeze(-1)  # [B,T]
        mask_t &= non_pad_mask_t
        mask_frame = mask_t.unsqueeze(-1).expand(B, T, Q).reshape(B, S)  # [B,S]

        if self.mask_2d_hybrid and self.training:
            # v4.6: hybrid masking — some samples use per-Q independent mask
            use_2d = torch.rand(B, device=device) < self.mask_2d_ratio  # [B]
            # Per-Q independent: each quantizer independently picks which frames to mask
            perm_2d = torch.rand((B, T, Q), device=device).argsort(dim=1)  # [B,T,Q]
            mask_2d = perm_2d < num_token_masked.view(B, 1, 1)  # [B,T,Q]
            mask_2d &= non_pad_mask_t.unsqueeze(-1)
            mask_2d = mask_2d.reshape(B, S)
            mask = torch.where(use_2d.unsqueeze(1), mask_2d, mask_frame)
        else:
            mask = mask_frame

        x_ids = labels.clone()

        # 10% -> random real token，和 C-concat 一致只在 [0..K-1] 采样
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_ids = torch.randint(low=0, high=self.K, size=x_ids.shape, device=device)
        x_ids = torch.where(mask_rid, rand_ids, x_ids)

        # 88% -> mask_id
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids = torch.where(mask_mid, torch.full_like(x_ids, self.mask_id), x_ids)

        # ----- noisy embedding (one_hot + noise) -----
        std = 0.1
        raw_ids = x_ids
        safe_codes = raw_ids.clamp(min=0, max=self.vocab - 1)
        one_hot = F.one_hot(safe_codes, num_classes=self.vocab).float()  # [B,S,K]
        noise = torch.randn_like(one_hot) * std
        noisy_one_hot = one_hot + noise
        motion_emb = noisy_one_hot @ self.token_emb.weight[: self.vocab]  # [B,S,C]

        # override special token emb
        mask_pos = raw_ids == self.mask_id
        pad_pos = raw_ids == self.pad_id
        if mask_pos.any():
            motion_emb[mask_pos] = self.token_emb.weight[self.mask_id]
        if pad_pos.any():
            motion_emb[pad_pos] = self.token_emb.weight[self.pad_id]

        # ----- transformer -----
        logits = self.trans_forward(motion_emb, cond_vector, padding_mask_seq, force_mask)  # [B,K,S]
        logits_bsv = logits.permute(0, 2, 1).contiguous()  # [B,S,K]

        # ----- CE + acc -----
        loss_ce = F.cross_entropy(
            logits_bsv.view(-1, self.vocab),
            labels.view(-1),
            ignore_index=self.pad_id,
        )

        pred_seq = logits_bsv.argmax(dim=-1)  # [B,S] (local id)
        correct = (pred_seq == labels) & non_pad_mask_seq
        acc = correct.float().sum() / non_pad_mask_seq.float().sum().clamp_min(1.0)

        pred_id = _unpack_local_seq_to_list(pred_seq, padding_mask_seq=padding_mask_seq, Q=self.Q)

        # ----- embedding alignment loss -----
        _, S2, V = logits_bsv.shape
        assert S2 == S
        assert V == self.K
        assert S == T * self.Q

        probs_local = F.softmax(logits_bsv, dim=-1).view(B, T, self.Q, self.K)  # [B,T,6,K]

        pred_emb_bt6c = []
        for i in range(self.Q):
            cb = vq_model.vqvae.quantizers[i].codebook  # [K,C]
            pred_emb_i = probs_local[:, :, i, :] @ cb   # [B,T,C]
            pred_emb_bt6c.append(pred_emb_i)
        pred_emb_bt6c = torch.stack(pred_emb_bt6c, dim=2)  # [B,T,6,C]

        non_pad_bt6 = non_pad_mask_seq.view(B, T, self.Q)  # [B,T,6]
        pred_emb_bt6c = pred_emb_bt6c.masked_fill(~non_pad_bt6.unsqueeze(-1), 0.0)

        x_encoder_list = vq_model.vqvae.get_quantized_codes(pose_4d, in_idx_format=False)  # 6 * [B,C,T]
        loss_emb = 0.0
        for i in range(self.Q):
            x_enc = x_encoder_list[i].permute(0, 2, 1)  # [B,T,C]
            valid = non_pad_bt6[:, :, i]                # [B,T]
            loss_emb = loss_emb + (F.l1_loss(
                x_enc[valid], pred_emb_bt6c[:, :, i, :][valid], reduction="mean"
            ) / float(self.Q))

        return loss_emb, loss_ce, pred_id, acc

    # --------------- cond-scale forward (入口不变) ---------------

    def forward_with_cond_scale(
        self,
        motion_emb,
        cond_vector,
        padding_mask,
        cond_scale=3,
        force_mask=False,
        ctrlNet_cond=None,
    ):
        if force_mask:
            return self.trans_forward(
                motion_emb, cond_vector, padding_mask, force_mask=True, ctrlNet_cond=ctrlNet_cond
            )

        logits = self.trans_forward(
            motion_emb, cond_vector, padding_mask, force_mask=False, ctrlNet_cond=ctrlNet_cond
        )
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(
            motion_emb, cond_vector, padding_mask, force_mask=True, ctrlNet_cond=ctrlNet_cond
        )
        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    # --------------- generate (入口/出口不变) ---------------

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
        require(
            self.Q > 0,
            f"num_quantizers must be > 0, got {self.Q}",
        )
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

        # cond
        if conds is None:
            if self.cond_mode == "uncond":
                cond_vector = torch.zeros((batch_size, self.latent_dim), device=device)
            elif self.cond_mode == "action":
                cond_vector = torch.zeros((batch_size, self.num_actions), device=device)
            else:
                cond_vector = torch.zeros((batch_size, self.clip_dim), device=device)
        else:
            cond_vector = self.encode_text(conds)  # [B,512]

        # token seq length
        S = seq_len * self.Q
        m_lens_t = m_lens_t.clamp(min=0, max=seq_len)
        m_lens_seq = m_lens_t * self.Q

        padding_mask_seq = ~lengths_to_mask(m_lens_seq, S).to(device)  # TRUE=pad
        padding_mask_t = ~lengths_to_mask(m_lens_t, seq_len).to(device)  # TRUE=pad

        # init ids: non-pad=mask_id, pad=pad_id
        ids = torch.where(
            padding_mask_seq,
            torch.full((batch_size, S), self.pad_id, device=device),
            torch.full((batch_size, S), self.mask_id, device=device),
        )

        # scores: pad=1e5, else=0
        scores = torch.where(
            padding_mask_seq,
            torch.full_like(ids, 1e5, dtype=torch.float),
            torch.zeros_like(ids, dtype=torch.float),
        )

        starting_temperature = temperature
        last_logits = None

        for timestep in torch.linspace(0, 1, timesteps, device=device):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked_t = (m_lens_t * rand_mask_prob).round().clamp(min=1)  # [B]

            frame_scores = scores.view(batch_size, seq_len, self.Q).mean(dim=-1)
            sorted_indices = frame_scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask_t = ranks < num_token_masked_t.unsqueeze(-1)
            is_mask_t &= ~padding_mask_t
            is_mask = is_mask_t.unsqueeze(-1).expand(batch_size, seq_len, self.Q).reshape(batch_size, S)

            # emb: masked positions use mask_emb
            emb = self.token_emb(ids)  # [B,S,C]
            mask_emb = self.token_emb.weight[self.mask_id]
            emb = torch.where(is_mask.unsqueeze(-1), mask_emb, emb)

            logits = self.forward_with_cond_scale(
                emb,
                cond_vector=cond_vector,
                padding_mask=padding_mask_seq,
                cond_scale=cond_scale,
                force_mask=force_mask,
            )  # [B,K,S]
            last_logits = logits

            logits_bsv = logits.permute(0, 2, 1).contiguous()  # [B,S,K]
            B, S2, V = logits_bsv.shape
            assert S2 == S
            assert V == self.K

            if topk_filter_thres > 0:
                logits_bsv = top_k(logits_bsv, topk_filter_thres, dim=-1)

            if gsample:
                pred_local = gumbel_sample(
                    logits_bsv, temperature=starting_temperature, dim=-1
                )  # [B,S]
                probs_local = F.softmax(logits_bsv / max(temperature, 1e-8), dim=-1)  # [B,S,K]
            else:
                probs_local = F.softmax(logits_bsv / max(temperature, 1e-8), dim=-1)  # [B,S,K]
                pred_local = Categorical(probs_local).sample()  # [B,S]

            pred_ids = pred_local

            # update ids only where masked
            ids = torch.where(is_mask, pred_ids, ids)

            # update scores with local confidence
            conf = probs_local.gather(-1, pred_local.unsqueeze(-1)).squeeze(-1)  # [B,S]
            frame_conf = conf.view(batch_size, seq_len, self.Q).mean(dim=-1)
            frame_scores = frame_conf.masked_fill(~is_mask_t, 1e5)
            frame_scores = frame_scores.masked_fill(padding_mask_t, 1e5)
            scores = frame_scores.unsqueeze(-1).expand(batch_size, seq_len, self.Q).reshape(batch_size, S)

        ids_list = _unpack_local_seq_to_list(ids, padding_mask_seq=padding_mask_seq, Q=self.Q)
        return ids_list, last_logits

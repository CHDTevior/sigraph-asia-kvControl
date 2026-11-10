"""c-concat MaskTransformer (6-codebook channel-concat variant).

Factory key: transformer_variant='c_concat'. This implementation previously
lived in models/mask_transformer/transformer.py; it was moved here so that
transformer.py keeps the original MoMask-style MaskTransformer/ResidualTransformer
API (generate/edit/edit_beta) that legacy entrypoints (gen_t2m.py, edit_t2m.py,
eval_t2m_trans_res.py, train_res_transformer.py, generation/load_model.py) import.
"""

import torch
import torch.nn as nn
import numpy as np
# from networks.layers import *
import torch.nn.functional as F
import clip
import os
from einops import rearrange, repeat
import math
from random import random
from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict
from copy import deepcopy
from functools import partial
from pathlib import Path
from models.mask_transformer.tools import *
from torch.distributions.categorical import Categorical
import torch.nn.functional as F





from einops import rearrange, repeat


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
        f"Download it once on a networked node and rerun; runtime downloads are disabled."
    )


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2))  # [seqlen, bs, input_feats]
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
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
        # hidden_states: [seqlen, bs, d]
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, out_feats, seqlen]
        return output


class MaskTransformer(nn.Module):
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

        self.code_dim = code_dim  # 单个 codebook 的 dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == "action":
            assert "num_actions" in kargs
        self.num_actions = kargs.get("num_actions", 1)

        # ======= 网络结构 =======

        # 6 个 codebook，一次性 concat，输入维度 = 6 * code_dim
        self.input_process = InputProcess(self.code_dim * 6, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

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

        # vocab: [0 .. num_tokens-1] 真实 code，num_tokens 为 mask，num_tokens+1 为 pad
        _num_tokens = opt.num_tokens + 2
        self.mask_id = opt.num_tokens
        self.pad_id = opt.num_tokens + 1

        # 每个 head 一个输出头 + embedding
        self.output_processes = nn.ModuleList(
            [
                OutputProcess_Bert(out_feats=opt.num_tokens, latent_dim=latent_dim)
                for _ in range(6)
            ]
        )
        self.token_embs = nn.ModuleList(
            [nn.Embedding(_num_tokens, self.code_dim) for _ in range(6)]
        )

        self.apply(self.init_weights)

        # ======= 冻结 clip =======
        if self.cond_mode == "text":
            print("Loading CLIP...")
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        self.noise_schedule = cosine_schedule

    # --------------- 通用工具 ---------------

    def load_and_freeze_clip(self, clip_version):
        local_clip_path = resolve_local_clip_checkpoint(clip_version)
        print(f"Loading CLIP from local checkpoint: {local_clip_path}")
        clip_model, clip_preprocess = clip.load(
            local_clip_path, device="cpu", jit=False
        )  # Must set jit=False for training
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

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
        return [
            p for name, p in self.named_parameters() if not name.startswith("clip_model.")
        ]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_drop_prob
            ).view(bs, 1)
            return cond * (1.0 - mask)
        else:
            return cond

    # --------------- Transformer 前向 ---------------

    def trans_forward(self, motion_emb, cond, padding_mask, force_mask=False, ctrlNet_cond=None):
        """
        motion_emb: (B, T, 6*code_dim)  已经是 embedding 了
        padding_mask: (B, T), TRUE = pad
        cond: (B, d)
        """
        cond = self.mask_cond(cond, force_mask=force_mask)

        # (B, T, D_emb) -> (T, B, D_latent)
        x = self.input_process(motion_emb)  # [T, B, latent_dim]

        cond = self.cond_emb(cond).unsqueeze(0)  # [1, B, latent_dim]

        x = self.position_enc(x)
        xseq = torch.cat([cond, x], dim=0)  # [T+1, B, latent_dim]

        # padding_mask: (B, T+1)，第 0 位是 cond，不 mask
        padding_mask = torch.cat(
            [torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1
        )

        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[1:]  # [T, B, d]

        # 6 个 head 的 logits
        logits_list = [head(output) for head in self.output_processes]  # 6 * [B, K, T]
        logits = torch.stack(logits_list, dim=1)  # [B, 6, K, T]
        return logits

    # --------------- 训练 forward ---------------

    def forward(self, y, m_lens, pose, vq_model):
        """
        y: 文本或 action
        m_lens: (B,)
        pose: [B, 263, T]
        """
        device = pose.device

        # ----- VQ 编码 -----
        pose_4d = pose.permute(0, 2, 1).unsqueeze(2)  # [B, T, 1, 263]
        code_idx = vq_model.get_code_idx(pose_4d)     # [B, 6, T]
        code_idx = code_idx.permute(0, 2, 1)          # [B, T, 6]

        B, T, Q = code_idx.shape
        bs, ntokens = B, T
        assert Q == 6, "当前实现默认 6 个 codebook"

        # 每个 head 一个 ids
        ids_list = [code_idx[..., i].clone() for i in range(Q)]  # 6 * [B, T]

        # 非 pad mask：TRUE 表示有效
        non_pad_mask = lengths_to_mask(m_lens, ntokens).to(device)  # [B, T]

        # padding 位置写成 pad_id
        for i in range(Q):
            ids_list[i] = torch.where(
                non_pad_mask, ids_list[i], torch.full_like(ids_list[i], self.pad_id)
            )

        # ------- 条件向量 -------
        force_mask = False
        if self.cond_mode == "text":
            with torch.no_grad():
                cond_vector = self.encode_text(y)  # [B, 512]
        elif self.cond_mode == "action":
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == "uncond":
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        # ------- BERT-style mask 策略 -------
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)  # [B]
        # 按真实长度算 mask 数量，避免 pad 影响
        num_token_masked = (m_lens * rand_mask_probs).round().clamp(min=1)  # [B]

        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)  # [B, T]
        mask &= non_pad_mask  # 只 mask 非 pad

        # labels：有效位置 = 真实 code，pad 位置 = pad_id（loss 里 ignore）
        labels_list = [ids_list[i].clone() for i in range(Q)]

        # x_ids_list = 输入用的 noisy token ids
        x_ids_list = [ids_list[i].clone() for i in range(Q)]

        # Step 1: 10% masked positions 换成随机 token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        for i in range(Q):
            rand_ids = torch.randint(
                low=0,
                high=self.opt.num_tokens,  # 只在 [0..num_tokens-1] 内采样真实 code
                size=x_ids_list[i].shape,
                device=device,
            )
            x_ids_list[i] = torch.where(mask_rid, rand_ids, x_ids_list[i])

        # Step 2: 剩下 masked 中的 88% 换成 mask_id
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        for i in range(Q):
            x_ids_list[i] = torch.where(
                mask_mid, torch.full_like(x_ids_list[i], self.mask_id), x_ids_list[i]
            )
        # 剩余 ~2% 的 masked 位置保持原 token（BERT 习惯）

        # ------- 构造带噪声的 embedding 作为 transformer 输入 -------
        num_cb_list = [
            vq_model.vqvae.quantizers[i].codebook.shape[0] for i in range(Q)
        ]  # 每个 codebook 的 nb_code
        std = 0.1
        trans_emb_parts = []

        for i in range(Q):
            num_cb = num_cb_list[i]
            raw_ids = x_ids_list[i]  # [B, T]

            # 安全的 code index，防止 one_hot 越界
            safe_codes = raw_ids.clamp(min=0, max=num_cb - 1)
            one_hot = F.one_hot(safe_codes, num_classes=num_cb).float()  # [B, T, num_cb]
            noise = torch.randn_like(one_hot) * std
            noisy_one_hot = one_hot + noise

            # 用 token_embs 的前 num_cb 行作为 code embedding
            part_emb = noisy_one_hot @ self.token_embs[i].weight[:num_cb]  # [B, T, Cdim]

            # 覆盖特殊 token：mask / pad
            mask_pos = raw_ids == self.mask_id
            pad_pos = raw_ids == self.pad_id

            if mask_pos.any():
                part_emb[mask_pos] = self.token_embs[i].weight[self.mask_id]
            if pad_pos.any():
                part_emb[pad_pos] = self.token_embs[i].weight[self.pad_id]

            trans_emb_parts.append(part_emb)

        trans_emb_all = torch.cat(trans_emb_parts, dim=-1)  # [B, T, 6*code_dim]

        # ------- Transformer & CE loss -------
        padding_mask = ~non_pad_mask  # TRUE = pad
        logits = self.trans_forward(
            trans_emb_all, cond_vector, padding_mask, force_mask
        )  # [B, 6, K, T]

        # 注意：ignore_index 用 pad_id，而不是 mask_id
        ce_loss, pred_id, acc = cal_performance2(
            logits, labels_list, ignore_index=self.pad_id
        )

        # ------- Embedding 对齐 loss -------
        logits_for_emb = logits.permute(0, 1, 3, 2)  # [B, 6, T, K]

        emb_list = [
            F.softmax(logits_for_emb[:, i, :, :], dim=-1)
            @ vq_model.vqvae.quantizers[i].codebook  # [B, T, Cdim]
            for i in range(Q)
        ]

        emb_list = [
            emb_list[i].masked_fill(padding_mask.unsqueeze(-1), 0.0) for i in range(Q)
        ]

        x_encoder_list = vq_model.vqvae.get_quantized_codes(
            pose_4d, in_idx_format=False
        )  # 6 * [B, Cdim, T]

        loss_emb = 0.0
        for i in range(Q):
            x_enc = x_encoder_list[i].permute(0, 2, 1)  # [B, T, Cdim]
            loss_emb += (
                F.l1_loss(
                    x_enc[non_pad_mask], emb_list[i][non_pad_mask], reduction="mean"
                )
                / Q
            )

        return loss_emb, ce_loss, pred_id, acc

    # --------------- 带 cond_scale 的 forward（generate 用） ---------------

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

    # --------------- 生成 generate（6 个 codebook 版本） ---------------

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
        seq_len = 49  # 你原来的设定
        batch_size = len(m_lens)
        m_lens = m_lens.to(device) if isinstance(m_lens, torch.Tensor) else torch.tensor(m_lens, device=device, dtype=torch.long)

        # 条件
        if conds is None:
            cond_vector = torch.zeros((batch_size, self.clip_dim), device=device)
        else:
            with torch.no_grad():
                cond_vector = self.encode_text(conds)  # [B, 512]

        padding_mask = ~lengths_to_mask(m_lens, seq_len).to(device)  # TRUE = pad

        # 起始：非 pad 为 mask_id，pad 为 pad_id
        init_ids = torch.where(
            padding_mask,
            torch.full((batch_size, seq_len), self.pad_id, device=device),
            torch.full((batch_size, seq_len), self.mask_id, device=device),
        )
        ids_list = [init_ids.clone() for _ in range(6)]

        # scores 初始化：pad 位置 = 1e5（永远不会被 mask 出来），其他 = 0
        scores = torch.where(
            padding_mask,
            torch.full_like(init_ids, 1e5, dtype=torch.float),
            torch.zeros_like(init_ids, dtype=torch.float),
        )

        starting_temperature = temperature

        last_logits = None  # 仅用于返回

        for timestep in torch.linspace(0, 1, timesteps, device=device):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = (
                m_lens * rand_mask_prob
            ).round().clamp(min=1)  # [B]

            # 根据 scores 找出置信度最低的若干 token 做 mask
            sorted_indices = scores.argsort(dim=1)  # [B, T]
            ranks = sorted_indices.argsort(dim=1)   # [B, T]
            is_mask = ranks < num_token_masked.unsqueeze(-1)
            is_mask &= ~padding_mask  # 不 mask pad

            # 用 mask embedding 构造输入 embedding
            emb_parts = []
            for i in range(6):
                emb_i = self.token_embs[i](ids_list[i])  # [B, T, Cdim]
                mask_emb_i = self.token_embs[i].weight[self.mask_id]
                emb_i = torch.where(
                    is_mask.unsqueeze(-1), mask_emb_i, emb_i
                )  # 被选中的位置用 mask_emb
                emb_parts.append(emb_i)

            emb_all = torch.cat(emb_parts, dim=-1)  # [B, T, 6*code_dim]

            # Transformer 前向
            logits = self.forward_with_cond_scale(
                emb_all,
                cond_vector=cond_vector,
                padding_mask=padding_mask,
                cond_scale=cond_scale,
                force_mask=force_mask,
            )  # [B, 6, K, T]
            last_logits = logits

            # [B, 6, T, K]
            logits_btqk = logits.permute(0, 1, 3, 2)

            # 采样
            pred_ids_list = []
            probs_list = []
            for i in range(6):
                logit_i = logits_btqk[:, i, :, :]  # [B, T, K]

                # top-k 过滤
                if topk_filter_thres > 0:
                    logit_i = top_k(logit_i, topk_filter_thres, dim=-1)

                if gsample:
                    pred_ids_i = gumbel_sample(
                        logit_i, temperature=starting_temperature, dim=-1
                    )  # [B, T]
                    probs_i = F.softmax(logit_i / max(temperature, 1e-8), dim=-1)
                else:
                    probs_i = F.softmax(
                        logit_i / max(temperature, 1e-8), dim=-1
                    )  # [B, T, K]
                    pred_ids_i = Categorical(probs_i).sample()  # [B, T]

                pred_ids_list.append(pred_ids_i)
                probs_list.append(probs_i)

            # 更新 ids_list：只在 is_mask 的位置更新
            for i in range(6):
                ids_list[i] = torch.where(
                    is_mask, pred_ids_list[i], ids_list[i]
                )

            # 更新 scores：用当前采样 token 的概率作为置信度
            scores_q = []
            for i in range(6):
                conf_i = probs_list[i].gather(
                    -1, pred_ids_list[i].unsqueeze(-1)
                ).squeeze(-1)  # [B, T]
                scores_q.append(conf_i)

            # 这里我们用 6 个 head 的平均置信度
            scores = torch.stack(scores_q, dim=1).mean(dim=1)  # [B, T]
            # 不再重新 mask 的位置 & pad 位置设置成 1e5
            scores = scores.masked_fill(~is_mask, 1e5)
            scores = scores.masked_fill(padding_mask, 1e5)

        # 最后把 pad 位置改成 -1，方便 forward_decoder 识别
        for i in range(6):
            ids_list[i] = torch.where(
                padding_mask, torch.full_like(ids_list[i], -1), ids_list[i]
            )

        return ids_list, last_logits

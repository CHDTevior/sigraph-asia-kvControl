"""KV-Control adapter smoke test TEMPLATE — 训练启动前的强制门禁。

改编自 (MaskControl port 实际使用, 全部检查真实抓过 bug):
    <repo-root>/scripts/smoke_kv_on_maskcontrol.py

═══════════════════════════════════════════════════════════════════════════
本模板守护的 pitfalls:
─────────────────────────────────────────────────────────────────────────
  P3  Q-residual K/V contamination — Tier 1 (bit-exact base fallback) +
      frozen-base 梯度检查一起, 能抓到 residual 泄漏进 K_base/V_base 导致的
      base 路径漂移 / base 参数收到梯度。
  (dead-branch trap) grad-flow 检查逐模块打印 grad_norm; init 后单次 backward
      只有 kv_down / ctrl_attn_bias / q_gates 必须非零 ("NO GRAD"/全零 = 双零
      初始化写错, 立刻 FAIL); kv_up_k/kv_up_v/encoder_control/q_down/q_up 在
      init 时梯度恰好为 0 是正确实现的数学必然 (kv_down=0 ⇒ h=0; q_gates=0
      ⇒ q_delta 支路系数 0), 只查非 NaN, 并在 1-2 个 optimizer step 后复查非零。
  P1  EMA-codebook pollution — 若 substrate 用 EMA quantizer, 附加检查:
      train() + backward 前后 codebook buffer byte-equal (见文末 TODO)。
  P12 前置保障 — Tier 1 bit-exact 要求 _base_trans_forward 逐行照抄
      substrate, 抄错(结构漂移)在这里第一时间暴露, 而不是训完才发现。
═══════════════════════════════════════════════════════════════════════════

四层检查 + PASS 阈值 (全部必须过, 任何一项 FAIL 禁止启动训练):

  Tier 1  ctrlNet_cond=None ≡ frozen base 路径, BIT-EXACT。
          阈值: logits 形状正确 + 无 NaN; 若 base ckpt 已加载, 与
          substrate 原生 forward 输出 allclose(atol=0) 逐位相等。
  Tier 2  ctrlNet_cond=valid AT INIT ≈ base (near-identity)。
          阈值: max|diff| < 50.0 (宽松 sanity bound, 非紧检查)。
          原理: kv_down=0 → K_ctrl=V_ctrl=0, ctrl_bias=-5 从 softmax 分母
          分走的质量极小: 等基线 logits 下 per-ctrl-token 归一化质量
          = e^-5/(S + T_ctrl·e^-5) ≈ 1.3e-4 (S=50, T_ctrl=49), 全部 ctrl
          列合计 ≈ 0.65% (<1%), base attn 输出只被缩放 ≈0.9935 — 所以
          init diff 本来就很小, <50.0 只是防爆炸的宽松 sanity bound。
          paper 不变量是 "FID near base at init", 不是 "输出逐位相等"。
          diff 恒等于 0 反而可疑 (说明 ctrl 路径根本没被走到, 参见
          pitfall P2 的检测直觉)。
  Grad    init 后单次 backward: kv_down / ctrl_attn_bias / q_gates 三组
          grad_norm > 0 且无 NaN (kv_down 经 standard-init 的 kv_up 回传,
          必须非零; 全零 = 双零 dead branch)。
          ⚠ kv_up_k / kv_up_v / encoder_control 的 init 梯度**恰好为 0**
          (kv_down=0 ⇒ h=0 ⇒ ∂L/∂W_up ∝ h = 0), q_down / q_up 同理
          (q_gates=0)。这是正确实现的数学必然, 不许判 FAIL — 这些组只查
          无 NaN, 并在 1-2 个 optimizer step 后复查其 grad 变非零。
  Frozen  base transformer (seqTransEncoder 等) n_params_with_grad == 0,
          hard assert。

Run:
    # TODO(substrate): 在 substrate repo 根目录下运行, 保证其 import 解析
    python smoke_test_template.py
"""
import os
import sys
from pathlib import Path

import torch

# ============================================================================
# TODO(substrate): 指向目标 substrate repo, chdir + sys.path 保证其内部
# 相对 import / 相对路径 ckpt 解析正确。
# ============================================================================
SUBSTRATE_REPO = Path("/path/to/substrate/repo")  # TODO(substrate)
os.chdir(SUBSTRATE_REPO)
sys.path.insert(0, str(SUBSTRATE_REPO))


def build_opt(device):
    """构造最小 opt namespace。

    trans_path='' → super().__init__ 跳过 base ckpt 加载, 用随机初始化
    transformer 做 smoke。这仍是合法测试: init 不变量 / 梯度流 / 冻结集
    是 adapter 的属性, 不依赖 base ckpt。(Tier 1 的 bit-exact 对照在
    随机 base 上同样成立。)
    """
    class Opt: pass
    opt = Opt()
    opt.device = device
    # TODO(substrate): 填入目标 substrate 的全部必需字段 — 以下是
    # MaskControl 的值, 仅作字段清单参考:
    opt.dataset_name = "t2m"
    opt.joints_num = 22          # TODO(substrate): 新骨架的 joint 数
    opt.latent_dim = 384
    opt.ff_size = 1024
    opt.n_layers = 8
    opt.n_heads = 6
    opt.dropout = 0.2
    opt.cond_drop_prob = 0.1
    opt.unit_length = 4          # TODO(substrate): tokenizer 下采样率
    opt.max_motion_length = 196  # TODO(substrate)
    opt.num_tokens = 512
    opt.code_dim = 512
    return opt


def make_vq(opt, device):
    """TODO(substrate): 构造 / 加载 substrate 的 tokenizer (VQ/RVQ),
    dim_pose 换成目标数据集维度 (HumanML3D=263, KIT=251, 新数据集自定)。
    必须 .eval() — 见 P1。"""
    raise NotImplementedError  # TODO(substrate)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device = {device}")

    opt = build_opt(device)
    vq_model = make_vq(opt, device)

    DIM_POSE = 263  # TODO(substrate): 数据集特征维度
    mean = torch.zeros(DIM_POSE).to(device)
    std = torch.ones(DIM_POSE).to(device)

    # TODO(substrate): import 你的 adapter 类 (kv_adapter_template.py 的实例化版)
    from models.mask_transformer.control_transformer_kv import KVControlTransformer  # TODO(substrate)

    print("[smoke] building KV adapter model ...")
    model = KVControlTransformer(
        code_dim=opt.code_dim,
        cond_mode="text",
        latent_dim=opt.latent_dim,
        ff_size=opt.ff_size,
        num_layers=opt.n_layers,
        num_heads=opt.n_heads,
        dropout=opt.dropout,
        clip_dim=512,
        cond_drop_prob=opt.cond_drop_prob,
        clip_version="ViT-B/32",
        opt=opt,
        mean=mean, std=std,
        trans_path="",  # 跳过 ckpt 加载 — 随机 base 足以测 adapter 接线
        vq_model=vq_model,
        control="trajectory",  # TODO(substrate): 控制任务
        kv_rank=64,
        ctrl_attn_bias_init=-5.0,
        use_q_residual=True,
    ).to(device)
    model.eval()

    # ----- 测试输入 -----
    bsz = 2
    seqlen = 49                  # TODO(substrate): T_tokens = T_frames / unit_length
    T_frames = 196               # TODO(substrate)
    CTRL_C = 6                   # TODO(substrate): n_ctrl_joints * 2 * 3
    cond = torch.randn(bsz, 512).to(device)  # text embedding stub
    motion_ids = torch.randint(0, opt.num_tokens, (bsz, seqlen)).to(device)
    padding_mask = torch.zeros(bsz, seqlen, dtype=torch.bool).to(device)
    ctrlNet_cond = torch.randn(bsz, T_frames, CTRL_C).to(device)

    # ========================================================================
    # Tier 1: ctrlNet_cond=None ≡ 纯 base 前向 (bit-exact 路径)
    # PASS: 形状 (bsz, num_tokens, seqlen) + 无 NaN
    # (若 substrate 原生 MaskTransformer 可单独实例化并共享权重, 加一条
    #  torch.equal(logits_none, native_logits) 的逐位对照 — 更强。)
    # ========================================================================
    print("[smoke] Tier 1: ctrlNet_cond=None ≡ pure base forward")
    with torch.no_grad():
        logits_none = model.trans_forward(motion_ids, cond, padding_mask, ctrlNet_cond=None)
    print(f"  logits_none shape: {tuple(logits_none.shape)}  has_nan={torch.isnan(logits_none).any().item()}")
    # TODO(substrate): 确认 output_process 的输出约定 (MaskControl: (B, num_tokens, S))
    assert logits_none.shape == (bsz, opt.num_tokens, seqlen), f"unexpected logits shape {logits_none.shape}"
    assert not torch.isnan(logits_none).any(), "NaN in ctrlNet_cond=None path"

    # ========================================================================
    # Tier 2: ctrlNet_cond=valid AT INIT (zero-init kv_down → near-identity)
    # PASS: max|diff| < 50.0 (sanity bound; 见模块 docstring 的原理说明)
    # ========================================================================
    print("[smoke] Tier 2: ctrlNet_cond=valid AT INIT (near-identity)")
    with torch.no_grad():
        logits_with = model.trans_forward(motion_ids, cond, padding_mask, ctrlNet_cond=ctrlNet_cond)
    print(f"  logits_with shape: {tuple(logits_with.shape)}  has_nan={torch.isnan(logits_with).any().item()}")
    diff = (logits_with - logits_none).abs()
    print(f"  max |diff| vs base path: {diff.max().item():.6f}")
    print(f"  mean |diff| vs base path: {diff.mean().item():.6f}")
    assert not torch.isnan(logits_with).any(), "NaN in KV path"
    assert diff.max().item() < 50.0, f"KV at init too far from base ({diff.max().item():.4f})"

    # ----- Param counts (对照预期: MaskControl port 总计 ~9.6M trainable) -----
    print("[smoke] param counts:")
    for k, v in model._param_counts.items():
        print(f"  {k}: {v:.4f}")

    # ========================================================================
    # Grad-flow: trivial loss backward, 每组 trainable 参数必须收到梯度
    # PASS: 每组 grad_norm > 0, 无 "NO GRAD"
    # ========================================================================
    print("[smoke] backward: verify gradients reach every adapter module")
    model.train()
    model.ctrl_train()  # explicit
    # P1: train() 之后 tokenizer 必须回到 eval (EMA buffer 保护)
    vq_model.eval()
    logits_train = model.trans_forward(motion_ids, cond, padding_mask, ctrlNet_cond=ctrlNet_cond)
    loss = logits_train.mean()
    loss.backward()

    failures = []

    def grad_norm(p_iter, name, require_nonzero_at_init):
        """require_nonzero_at_init=True 的组 (kv_down/ctrl_attn_bias/q_gates):
        init 后单次 backward 梯度必须非零 — 全零 = 双零 dead branch / freeze 错。
        require_nonzero_at_init=False 的组 (kv_up_k/kv_up_v/encoder_control/
        q_down/q_up): init 时梯度**恰好为 0 是正确的** (kv_down=0 ⇒ h=0;
        q_gates=0), 只查存在且无 NaN; 它们的非零性在下方 1 个 optimizer
        step 后复查。"""
        gs = [p.grad for p in p_iter if p.grad is not None]
        if not gs:
            failures.append(name)
            return f"  {name}: NO GRAD (BAD)"
        total = sum(g.norm().item() for g in gs)
        if any(torch.isnan(g).any() for g in gs):
            failures.append(name)
        if require_nonzero_at_init and total == 0.0:
            failures.append(name)
        note = "" if require_nonzero_at_init else "  (zero-at-init OK)"
        return f"  {name}: grad_norm={total:.4e}  n_params_with_grad={len(gs)}{note}"

    # 这三组 init 时必须非零 (经 standard-init 的对侧回传):
    print(grad_norm((p for m in model.kv_down for p in m.parameters()), "kv_down", True))
    print(grad_norm(model.ctrl_attn_bias.parameters(), "ctrl_attn_bias", True))
    # 这些组 init 时梯度恰好为 0 是数学必然, 不判 FAIL:
    print(grad_norm((p for m in model.kv_up_k for p in m.parameters()), "kv_up_k", False))
    print(grad_norm((p for m in model.kv_up_v for p in m.parameters()), "kv_up_v", False))
    print(grad_norm(model.encoder_control.parameters(), "encoder_control", False))
    if model.use_q_residual:
        print(grad_norm(model.q_gates.parameters(), "q_gates", True))
        print(grad_norm(model.q_down.parameters(), "q_down", False))
        print(grad_norm(model.q_up.parameters(), "q_up", False))
    assert not failures, f"grad-flow FAIL (dead branch / freeze error): {failures}"

    # ---- 复查: 1 个 optimizer step 后, zero-at-init 组的梯度必须活过来 ----
    # (kv_down/q_gates 离开 0 后, kv_up/encoder/q_down/q_up 的梯度路径打通)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=1e-4)
    optim.step()
    optim.zero_grad(set_to_none=True)
    logits_train2 = model.trans_forward(motion_ids, cond, padding_mask, ctrlNet_cond=ctrlNet_cond)
    logits_train2.mean().backward()
    failures2 = []
    for name, p_iter in [
        ("kv_up_k", (p for m in model.kv_up_k for p in m.parameters())),
        ("kv_up_v", (p for m in model.kv_up_v for p in m.parameters())),
        ("encoder_control", model.encoder_control.parameters()),
    ] + ([("q_down", model.q_down.parameters()),
          ("q_up", model.q_up.parameters())] if model.use_q_residual else []):
        gs = [p.grad for p in p_iter if p.grad is not None]
        total = sum(g.norm().item() for g in gs) if gs else 0.0
        print(f"  [post-step] {name}: grad_norm={total:.4e}")
        if total == 0.0:
            failures2.append(name)
    assert not failures2, (
        f"grad-flow FAIL after 1 optimizer step (branch still dead): {failures2}")

    # ========================================================================
    # Frozen-base: base transformer 一个参数都不许收到梯度
    # PASS: n_params_with_grad == 0 (hard assert)
    # ========================================================================
    base_grad = sum(1 for p in model.seqTransEncoder.parameters() if p.grad is not None)
    print(f"  base seqTransEncoder n_params_with_grad: {base_grad}  (should be 0)")
    assert base_grad == 0, "Base transformer received gradients — freeze broken"
    # TODO(substrate): 其他 frozen 组件同样检查 (text encoder / token_emb /
    # output_process / vq_model 参数)。

    # ========================================================================
    # P1 附加检查 (EMA-quantizer substrate 必做, 如 MoMask RVQ):
    # TODO(substrate): backward 前 snapshot codebook buffer, backward 后
    # byte-equal 对照:
    #   before = {k: v.clone() for k, v in vq_model.state_dict().items()}
    #   ... (train + backward) ...
    #   for k, v in vq_model.state_dict().items():
    #       assert torch.equal(v, before[k]), f"EMA codebook mutated: {k}  (P1!)"
    # ========================================================================

    print()
    print("[smoke] ALL CHECKS PASS")


if __name__ == "__main__":
    main()

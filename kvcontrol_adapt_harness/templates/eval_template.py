"""KV-Control offline paper-formal eval TEMPLATE — M1/M2/M3 协议 + 5-rep CI。

改编自 (MaskControl port 实际使用, 产出 v16→v18 ep6000 全部 paper 数字):
    <repo-root>/scripts/eval_maskcontrol_kv.py
其本身 fork 自 v4 substrate 的 <repo-root>/scripts/eval_v4_ctrlnet_ttt.py。

═══════════════════════════════════════════════════════════════════════════
本模板守护的 pitfalls:
─────────────────────────────────────────────────────────────────────────
  P2  type() is vs isinstance — substrate eval 代码若用
      `type(x) is ControlTransformer` 路由, KV 子类会被静默送进 base
      no-control 路径 (症状: 不同 ckpt 的 eval 输出 4 位小数逐位相同,
      FID 0.1455/KPS 63.05 恒定)。移植前 grep substrate 的 eval 路径,
      全部改 isinstance()。本模板附 sanity 检测 (见 rep 循环后注释)。
  P7  并发 ckpt 读写 race — eval 读 latest.tar 时训练在写 →
      'PytorchStreamReader failed reading file'。必须先 cp 到 frozen
      snapshot (cp + atomic mv) 再 eval snapshot (见 --ckpt help + main 开头)。
  P8  pred_num_batch 语义 — 是「每次 generate call 累积 N 个 loader batch」,
      不是「总共处理 N 个 batch」。设 99999 会静默跳过全部生成 →
      motion_annotation_list torch.cat 空列表 crash。固定用 16。
  P14 DDP ckpt 'module.' 前缀 — load 时检测并 strip。
  P15 eval 协议漂移 — time_steps/cond_scale/each_iter/ttt_dynamic/last_iter
      必须按协议 (M1/M2/M3) pin 死并写进 output_json; 默认值静默漂移会让
      数字不可比。本模板把全部 settings 落盘。
  P1  (eval 侧) — ⚠ strict=False **不会**忽略 ckpt 里泄漏的 vq_model.* keys:
      模型有同名 vq_model 子模块, 名字匹配得上, load_state_dict(strict=False)
      会照常加载它们、静默覆盖 pristine tokenizer (strict=False 只忽略
      匹配不上的 keys; 实测 v18_ep6000 snapshot 带 72 个 vq_model.* keys,
      MaskControl port 训练侧并未 strip)。所以本模板在 load 前**显式过滤**
      vq_model.* keys, 并建议加与官方 ckpt 的 byte-equal preflight;
      tokenizer 永远从官方 ckpt 独立加载 + freeze。
═══════════════════════════════════════════════════════════════════════════

协议定义 (MaskControl port 实测数字, HumanML3D test split, 5-rep):
  M1 = Stage-1 TTT only:       each_iter=35 uniform/step (350 total @ts=10), last_iter=0
       → FID 0.152, KPS 11.75cm  (受 quantization floor 限制)
  M2 = 100 uniform + Stage-2:  each_iter=100 uniform, last_iter=600
       → FID 0.098, KPS 1.10cm
  M3 = 35 uniform + Stage-2:   each_iter=35 uniform, last_iter=600
       → FID 0.0875, KPS 1.29cm  (paper headline 协议; v4 substrate 上 0.40cm)

  ⚠ 协议标注勘误 (2026-07 codex audit): M1/M3 曾被标 "35 dynamic ((s+1)*35/step,
  ~1925 total)" — 错误。substrate 的 control_transformer.py L527-531 只在
  each_iter<0 时走 dynamic TTT; 原 eval 脚本传 +35 且从未取负 (--ttt_dynamic
  只是打印标签)。已发布 ep6000 M1/M3 数字全部是 uniform-35。本模板已接通
  negative-each_iter 约定 (见 main() 中 opt.each_iter 的赋值); 真跑 dynamic
  产生的是与已发布数字不可比的新协议。

  Stage-2 (last_iter) 直接优化连续 embedding, 绕过 codebook — 唯一没有
  quantization floor 的杠杆, 解释 M1→M2/M3 的 ~10× KPS 差距。

输出: FID / R-precision (top1/2/3) / MatchScore / Diversity / KPS(cm) /
skate_ratio, 全部 mean ± 95% CI (= std × 1.96 / sqrt(N))。

Usage:
    # P7: 先做 frozen snapshot, 再 eval!
    cp <run>/model/latest.tar <run>/model/eval_snapshot.tar.tmp \\
      && mv <run>/model/eval_snapshot.tar.tmp <run>/model/eval_snapshot.tar
    python eval_template.py --ckpt <run>/model/eval_snapshot.tar \\
      --protocol M3 --output_json output/<run>/eval_5r_M3.json
"""
import os, sys, argparse, json, time
from pathlib import Path

import numpy as np
import torch

# ============================================================================
# TODO(substrate): substrate repo 置于 sys.path 首位, 使其自身
# utils/ models/ motion_loaders/ 优先解析。
# ============================================================================
SUBSTRATE_REPO = Path("/path/to/substrate/repo")  # TODO(substrate)
sys.path.insert(0, str(SUBSTRATE_REPO))

from utils.fixseed import fixseed                                        # TODO(substrate)
from utils.get_opt import get_opt                                        # TODO(substrate)
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader  # TODO(substrate)
from models.t2m_eval_wrapper import EvaluatorModelWrapper                # TODO(substrate)
import utils.eval_t2m as eval_t2m                                        # TODO(substrate)
# TODO(substrate): import 你的 KV adapter 类
# from models.mask_transformer.control_transformer_kv import KVControlTransformer

# M1/M2/M3 协议表 — P15: 协议参数只在这里定义, 不散落在 CLI 默认值里
# 全部 ttt_dynamic=False (uniform): 已发布的 M1/M2/M3 数字全部产自 uniform
# schedule (见模块 docstring 的勘误)。dynamic 是另一个 (未发布的) 协议,
# 只能经 --protocol CUSTOM --ttt_dynamic 显式开启。
PROTOCOLS = {
    # (time_steps, cond_scale, each_iter, ttt_dynamic, last_iter, last_lr)
    "M1": dict(time_steps=10, cond_scale=3.25, each_iter=35,  ttt_dynamic=False, last_iter=0,   last_lr=6e-2),
    "M2": dict(time_steps=10, cond_scale=3.25, each_iter=100, ttt_dynamic=False, last_iter=600, last_lr=6e-2),
    "M3": dict(time_steps=10, cond_scale=3.25, each_iter=35,  ttt_dynamic=False, last_iter=600, last_lr=6e-2),
}


def load_tokenizer(ckpt_path, opt_path, device):
    """TODO(substrate): 加载 substrate 的 tokenizer (VQ/RVQ)。

    要点 (照抄 eval_maskcontrol_kv.py:44-74 的模式):
    - 从 opt.txt 解析构造参数
    - dim_pose / joints_num 强制为目标数据集的值:
        HumanML3D: dim_pose=263, joints_num=22
        KIT:       dim_pose=251, joints_num=21
        新数据集:  TODO(substrate) — 同时需要新的 recover_from_ric 等价 FK!
    - state key 兼容 ("vq_model" else "net")
    - .to(device).eval() + 全参数 requires_grad=False (P1: eval 侧也要
      保证用的是 pristine tokenizer; 若之前发生过 P1 污染, 这里加
      byte-equal preflight assert 对照官方 ckpt)
    """
    raise NotImplementedError  # TODO(substrate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="KV-Control ckpt (.tar)。P7: 训练还在跑时, 必须先 cp+mv 出 "
                        "frozen snapshot 再传进来, 绝不直接 eval 正在被写的 latest.tar")
    p.add_argument("--gpu_id", type=int, default=0)

    # ---- 协议选择 (P15) ----
    p.add_argument("--protocol", type=str, default="M1", choices=list(PROTOCOLS.keys()) + ["CUSTOM"],
                   help="M1=Stage-1 only (uniform-35); M2=100 uniform+Stage-2; "
                        "M3=35 uniform+Stage-2 (paper headline)。CUSTOM 时用下面的显式覆盖值。")
    p.add_argument("--repeat_times", type=int, default=5)
    # CUSTOM 覆盖 (仅 --protocol CUSTOM 时生效; 命名协议下这些值被 PROTOCOLS 表覆盖)
    p.add_argument("--time_steps", type=int, default=10)
    p.add_argument("--cond_scale", type=float, default=3.25)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--topkr", type=float, default=0.9)
    p.add_argument("--force_mask", action="store_true")
    p.add_argument("--each_iter", type=int, default=35, help="Stage-1 TTT iters per timestep")
    p.add_argument("--each_lr", type=float, default=6e-2)
    p.add_argument("--ttt_dynamic", action=argparse.BooleanOptionalAction, default=False,
                   help="Dynamic TTT: step s gets (s+1)*each_iter iters (经 substrate 的 "
                        "negative-each_iter 约定接通, control_transformer.py L527-531)。"
                        "已发布 M1/M2/M3 数字全部是 uniform (False); 开启 = 不可比的新协议")
    p.add_argument("--last_iter", type=int, default=0, help="Stage-2 embedding refinement iters")
    p.add_argument("--last_lr", type=float, default=6e-2)
    p.add_argument("--seed", type=int, default=3407)

    # ---- Substrate 路径 (TODO(substrate): 全部换成目标 substrate 的默认值) ----
    p.add_argument("--dataset_name", type=str, default="t2m")   # TODO(substrate)
    p.add_argument("--vq_name", type=str, default="TODO_vq_run_name")      # TODO(substrate)
    p.add_argument("--trans_name", type=str, default="TODO_base_trans_run")  # TODO(substrate)
    p.add_argument("--eval_wrapper_opt", type=str,
                   default="TODO/path/to/evaluator/opt.txt",     # TODO(substrate)
                   help="FID/R-prec evaluator 是 dataset-specific 的 "
                        "(HumanML3D 用 Comp_v6_KLD005)。新数据集必须有自己的 "
                        "evaluator ckpt, 否则 FID/R-prec 无意义; "
                        "KPS/traj-fail 是纯几何、永远可移植。")

    # ---- 模型架构 (必须与训练一致) ----
    p.add_argument("--latent_dim", type=int, default=384)   # TODO(substrate)
    p.add_argument("--n_layers", type=int, default=8)       # TODO(substrate)
    p.add_argument("--n_heads", type=int, default=6)        # TODO(substrate)
    p.add_argument("--ff_size", type=int, default=1024)     # TODO(substrate)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--control", type=str, default="trajectory")  # TODO(substrate)
    p.add_argument("--kv_rank", type=int, default=64)
    p.add_argument("--ctrl_attn_bias_init", type=float, default=-5.0)
    p.add_argument("--no_q_residual", action="store_true")

    p.add_argument("--output_json", type=str, required=True)
    args = p.parse_args()

    # ---- P15: 命名协议强制覆盖 CLI 采样参数, 杜绝静默漂移 ----
    if args.protocol in PROTOCOLS:
        for k, v in PROTOCOLS[args.protocol].items():
            setattr(args, k, v)
    protocol = args.protocol

    # ---- P7: 拒绝直接 eval 可能正在被写的 latest.tar ----
    if Path(args.ckpt).name == "latest.tar":
        print("[eval] WARNING (P7): eval'ing latest.tar directly — if training is "
              "still running this WILL race the writer (PytorchStreamReader error). "
              "cp + atomic mv to a frozen snapshot first.")

    fixseed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}  ckpt={args.ckpt}  protocol={protocol}")

    # ---- eval loader + evaluator wrapper (TODO(substrate)) ----
    # 统一用 test split (项目铁律), bs=32
    print(f"[eval] loading eval_val_loader from {args.eval_wrapper_opt}")
    eval_val_loader, _ = get_dataset_motion_loader(args.eval_wrapper_opt, 32, "test", device=device)
    wrapper_opt = get_opt(args.eval_wrapper_opt, torch.device("cuda"))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    mean = torch.tensor(eval_val_loader.dataset.mean, requires_grad=False).to(device)
    std = torch.tensor(eval_val_loader.dataset.std, requires_grad=False).to(device)

    # ---- tokenizer (P1: pristine, 独立加载, frozen) ----
    vq_ckpt_dir = SUBSTRATE_REPO / "checkpoints" / args.dataset_name / args.vq_name
    vq_model, vq_opt = load_tokenizer(str(vq_ckpt_dir / "model" / "net_best_fid.tar"),
                                      str(vq_ckpt_dir / "opt.txt"), device)

    # ---- 构造 KV adapter 模型 ----
    class Opt: pass
    opt = Opt()
    opt.device = device
    opt.dataset_name = args.dataset_name
    opt.joints_num = 22            # TODO(substrate): 新骨架 joint 数
    opt.num_tokens = int(vq_opt.nb_code)
    opt.latent_dim = args.latent_dim
    opt.ff_size = args.ff_size
    opt.n_layers = args.n_layers
    opt.n_heads = args.n_heads
    opt.dropout = args.dropout
    opt.cond_drop_prob = 0.1
    opt.unit_length = 4            # TODO(substrate): tokenizer 下采样率
    opt.max_motion_length = 196    # TODO(substrate)
    opt.checkpoints_dir = str(SUBSTRATE_REPO / "checkpoints")
    # eval-time attrs (TTT 参数经 opt 传给 generate_with_control)
    opt.ctrl_net = True
    opt.each_lr = args.each_lr
    # negative-each_iter 约定 (substrate control_transformer.py L527-531):
    # each_iter > 0 → uniform (每 step 固定 each_iter 次);
    # each_iter < 0 → dynamic (step s 得 (s+1)*|each_iter| 次)。
    # 历史 bug: 旧脚本从不取负, --ttt_dynamic 只是打印标签 → 已发布 M1/M3
    # 实际是 uniform-35。这里显式接通:
    opt.each_iter = -args.each_iter if args.ttt_dynamic else args.each_iter
    opt.last_lr = args.last_lr
    opt.last_iter = args.last_iter
    # TODO(substrate): 若 substrate 的 TTT 用别的开关 (而非 each_iter 符号)
    # 表达 dynamic schedule, 按其约定改写上面这行

    trans_path = SUBSTRATE_REPO / "checkpoints" / args.dataset_name / args.trans_name / "model" / "latest.tar"

    print(f"[eval] instantiating KV adapter, base trans_path={trans_path}")
    model = KVControlTransformer(  # TODO(substrate): 你的 adapter 类
        code_dim=vq_opt.code_dim, cond_mode="text",
        latent_dim=args.latent_dim, ff_size=args.ff_size,
        num_layers=args.n_layers, num_heads=args.n_heads, dropout=args.dropout,
        clip_dim=512, cond_drop_prob=0.1,
        clip_version="ViT-B/32", opt=opt,
        mean=mean, std=std,
        trans_path=str(trans_path),
        vq_model=vq_model, control=args.control,
        kv_rank=args.kv_rank,
        ctrl_attn_bias_init=args.ctrl_attn_bias_init,
        use_q_residual=(not args.no_q_residual),
    ).to(device)
    model.eval()

    # ---- ckpt 加载 (P14: strip 'module.'; P1: 显式过滤泄漏的 vq_model.*) ----
    print(f"[eval] loading ckpt {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    key = "ct2m_transformer" if "ct2m_transformer" in ckpt else "trans"  # TODO(substrate): state key
    raw_sd = ckpt[key]
    if any(k.startswith("module.") for k in raw_sd.keys()):
        raw_sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in raw_sd.items()}
        print("[eval] stripped 'module.' prefix (DDP ckpt) from state_dict")
    # P1 (关键, 勿删): ckpt 里泄漏的 vq_model.* keys 必须在 load 前显式过滤。
    # strict=False **不会**忽略它们 — 模型有同名 vq_model 子模块, 名字匹配得上,
    # load_state_dict 会照常加载并静默覆盖 pristine tokenizer (strict=False 只
    # 忽略匹配不上的 keys)。实测 v18_ep6000 snapshot 带 72 个 vq_model.* keys。
    n_vq_leak = sum(1 for k in raw_sd if k.startswith("vq_model."))
    if n_vq_leak:
        raw_sd = {k: v for k, v in raw_sd.items() if not k.startswith("vq_model.")}
        print(f"[eval] filtered {n_vq_leak} leaked 'vq_model.*' keys from ckpt state_dict (P1)")
    missing, unexpected = model.load_state_dict(raw_sd, strict=False)
    non_clip_missing = [k for k in missing if not k.startswith("clip_model.")]
    if non_clip_missing:
        print(f"[eval] WARN missing non-clip keys: {non_clip_missing[:10]}...")
    if unexpected:
        # 剩余 unexpected keys 应只是旧 ControlNet 分支 (已 delattr, 匹配不上,
        # 被 strict=False 丢弃)。其他类别的 unexpected keys → 立即停下排查。
        print(f"[eval] WARN unexpected keys: {unexpected[:10]}...")
    # P1 双保险 (推荐): byte-equal preflight — 加载后 vq_model state_dict 与
    # 官方 RVQ ckpt 逐 tensor torch.equal, 不等即 fail loud:
    #   official = torch.load(<official_rvq_ckpt>, map_location="cpu")[...]
    #   for k, v in vq_model.state_dict().items():
    #       assert torch.equal(v.cpu(), official[k]), f"tokenizer polluted: {k}"

    # ---- 5-rep eval loop ----
    ckpt_epoch = ckpt.get("ep", -1)
    print(f"\n[eval] protocol={protocol}  ckpt_epoch={ckpt_epoch}  repeat_times={args.repeat_times}")
    print(f"[eval] settings: time_steps={args.time_steps} cond_scale={args.cond_scale} "
          f"each_iter={args.each_iter} ttt_dynamic={args.ttt_dynamic} last_iter={args.last_iter}")

    reps = []
    t0 = time.time()
    for rep_id in range(args.repeat_times):
        rep_start = time.time()
        best_fid, best_div, Rprecision, best_matching, best_skate_ratio, best_mm, traj_err, _avoid, kps_mean = \
            eval_t2m.evaluation_mask_transformer_test_plus_res(  # TODO(substrate): eval 入口
                eval_val_loader, vq_model, None, model, None, rep_id,
                eval_wrapper=eval_wrapper,
                time_steps=args.time_steps, cond_scale=args.cond_scale,
                temperature=args.temperature, topkr=args.topkr,
                # P8 (勿改!): pred_num_batch=16 = 「每次 generate call 累积 16 个
                # bs=32 的 batch = 512 samples」, 不是「总共处理 16 个 batch」。
                # 设太高会静默禁用全部 generation call → 空列表 torch.cat crash。
                force_mask=args.force_mask, cal_mm=False, f=None, pred_num_batch=16,
                logger=None, epoch=ckpt_epoch,
                control=args.control, density=-1, opt=opt)
        rep_dt = time.time() - rep_start
        top1, top2, top3 = float(Rprecision[0]), float(Rprecision[1]), float(Rprecision[2])
        print(f"[eval] rep {rep_id+1}/{args.repeat_times} done in {rep_dt:.1f}s | "
              f"FID={best_fid:.4f} Top3={top3:.4f} Match={best_matching:.4f} "
              f"Div={best_div:.4f} KPS={kps_mean*100:.3f}cm skate={best_skate_ratio:.4f}")
        reps.append(dict(
            fid=float(best_fid), diversity=float(best_div),
            top1=top1, top2=top2, top3=top3,
            matching_score=float(best_matching),
            skate_ratio=float(best_skate_ratio),
            kps_cm=float(kps_mean) * 100.0,
        ))

    # ---- P2 sanity 检测: 不同 rep (不同随机性) 输出逐位相同 = 模型没被调用 ----
    # NOTE: 这个 in-run 检查只能抓「同一次运行内的异常确定性」。真实 P2 事故的
    # 指纹是 **跨不同 ckpt** 的多次 eval 输出逐位相同 (FID 0.1455/KPS 63.05 恒定),
    # 单次运行抓不到。权威检查是: (a) M0-vs-M1 KPS 显著不同 (docs/03 §5.2 的
    # dispatch sanity), (b) 不同 ckpt / 不同 TTT 配置的输出互相比较不许逐位相同。
    if len(reps) >= 2 and all(r == reps[0] for r in reps[1:]):
        print("[eval] FATAL (P2 heuristic): all reps bit-identical — the model you "
              "think you are evaluating is probably NOT being invoked "
              "(check type() is vs isinstance routing in substrate eval code).")
        sys.exit(1)

    # ---- summarize mean ± 95% CI ----
    N = len(reps)
    summary = {}
    for k in ("fid", "diversity", "top1", "top2", "top3", "matching_score", "skate_ratio", "kps_cm"):
        vals = np.array([r[k] for r in reps])
        mean_v = float(vals.mean())
        std_v = float(vals.std(ddof=1)) if N > 1 else 0.0
        ci95 = float(1.96 * std_v / np.sqrt(N)) if N > 1 else 0.0
        summary[k] = dict(mean=mean_v, std=std_v, ci95=ci95, n=N)

    wall = time.time() - t0
    print(f"\n{'='*72}")
    print(f"[eval] SUMMARY (N={N} reps, protocol={protocol}, wall={wall/60:.1f} min)")
    print('='*72)
    for k, v in summary.items():
        unit = " cm" if k == "kps_cm" else ""
        print(f"  {k:>18}: {v['mean']:8.4f} ± {v['ci95']:.4f} (std={v['std']:.4f}, n={N}){unit}")

    # P15: settings 全量落盘, 数字永远可追溯到协议
    out = dict(
        ckpt_path=args.ckpt,
        ckpt_epoch=ckpt_epoch,
        protocol=protocol,
        wall_seconds=wall,
        settings=vars(args),
        reps=reps,
        summary=summary,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[eval] wrote {args.output_json}")


if __name__ == "__main__":
    main()

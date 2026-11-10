# KV-Control 训练启动前 checklist（10 项，全过才许 launch）

> 适用: 任何 KV-Control 移植训练的正式启动（含续训 / 换 substrate / 换数据集）。
> 每项都对应本项目真实踩过的坑（P# = harness pitfall catalog 编号）。
> 逐项打勾并把证据（命令输出 / 路径）记进 launch 日志; 任何一项 FAIL → 不许 launch。

## 1. GPU 空闲核验（跨项目铁律: 不抢别项目正在用的卡）

- [ ] 目标节点 `nvidia-smi`（或 `srun --jobid=<j> --overlap nvidia-smi`）确认目标卡 util≈0 且无他人进程
- [ ] `squeue -w <node>` 确认没有别的项目 job 在用这些卡
- 双向规则: 既不抢别人, 启动后也要能察觉我方被抢（GPU util 骤降 / 吞吐掉一半 = 上报）
- 参考吞吐: MaskControl port 4×A100-80GB ~11 s/ep, 6000 ep ≈ 18.5 hr

## 2. conda 环境显式命名

- [ ] launch 命令 / 脚本里显式 `conda activate tlcontrol`（或目标项目环境名）, 不依赖 shell 默认
- [ ] `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 在该环境下过

## 3. Smoke test 全绿（P3 / dead-branch / frozen-base）

- [ ] `templates/smoke_test_template.py` 的实例化版四层全 PASS:
  - Tier 1: `ctrlNet_cond=None` ≡ base 路径（形状 + 无 NaN；与原生类的 bit-exact `torch.equal` 对照为**可选加强项**）
  - Tier 2: init 时 near-identity（`max|diff| < 50` 是宽松 sanity bound——init 时总 ctrl attn mass <1%，diff 本来就小; 且不恒为 0）
  - Grad-flow: **init 后单次 backward 只要求 kv_down / ctrl_attn_bias / q_gates grad_norm > 0**；kv_up_k/kv_up_v/encoder_control/q_down/q_up 在 init 时梯度恰好为 0 是正确实现的必然（kv_down=0/q_gates=0 所致），不判 FAIL——1-2 个 optimizer step 后再 assert 它们非零
  - Frozen-base: base transformer `n_params_with_grad == 0`
- [ ] EMA-quantizer substrate（如 MoMask RVQ）: codebook buffer byte-equal 检查也过（P1）
- [ ] 【P12 自检】确认 generate()/decode loop 是 substrate 原文 verbatim 复用, 只 override 了 `trans_forward`

## 4. Codex review 通过（铁律: 代码新增/改必经 codex 审）

- [ ] adapter 代码 + 训练脚本 + eval 脚本经 `mcp__codex__codex`（gpt-5.5, xhigh; MCP 断开则 `codex exec` fallback）审到 PASS
- [ ] review threadId / 结论记进 launch 日志（历史: cross-alloc infra 首轮 NEEDS-FIX 5 项才 PASS, 别跳）

## 5. Checkpoint / snapshot 目录方案（P7）

- [ ] 输出目录唯一、已创建, `mkdir(exist_ok=True)` 且 rank-0-only 写入
- [ ] eval 永不直接读训练正在写的 `latest.tar`: 约定 `cp latest.tar snap.tmp && mv snap.tmp eval_snapshot.tar`（cp + atomic mv）, eval 只吃 snapshot
- [ ] ckpt 保存已 strip `vq_model.*`（P1: 防止污染的 tokenizer state 埋进 ckpt, eval 时反噬）; resume 加载侧同样 strip

## 6. TensorBoard writer try/except 护栏（P10）

- [ ] 训练脚本里 **所有** `writer.add_scalar` 调用点（per-iter 和 end-of-epoch 两处都要）已包 try/except
- 历史: NFS 上 FileNotFoundError 曾在 ep 17 和 ep 152 两次把训练整个杀死

## 7. DDP mkdir / resume 路径修复（P5 / P6）

- [ ] `options/base_option.py` 的 `os.makedirs` 用 `exist_ok=True`; `init_save_folder` / `shutil.copytree` 用 `dirs_exist_ok=True`（否则 rank>0 在共享 FS 上 crash）
- [ ] `--is_continue` 时跳过日期前缀拼接（否则 resume 解析到全新空目录, 静默从零训）
- [ ] DDP ckpt 的 `module.` 前缀问题已知晓: eval/resume 加载侧有 strip 逻辑（P14）

## 8. best_kps tracker 两个 bug 已修（P4）

- [ ] (a) 保存 best_kps.tar 后 `best_kps_mean` 变量确实被更新（否则每次 eval 都覆盖 best）
- [ ] (b) in-loop eval 的 `kps_mean` 没有被 unpack 进 `_` 丢弃（否则比较用的是 loop 前的 stale 值, tracker 冻死在第一次 eval）
- 两个都不修, "best ckpt" 选择完全没有意义

## 9. In-loop eval 路由用 isinstance（P2）

- [ ] grep substrate 的 eval 代码（MaskControl 历史 bug 位置: `utils/eval_t2m.py:1082`, 现已修复——isinstance 在 L1086, :1082 现为修复注释行）: 所有 `type(x) is XxxTransformer` 已改 `isinstance(x, ...)`, 否则 KV 子类被静默送进 base no-control 路径
- [ ] 知晓检测直觉: 不同 ckpt 的 eval 输出 4 位小数逐位相同 = 你以为在 eval 的模型根本没被调用
- [ ] eval 协议参数（time_steps/cond_scale/each_iter/ttt_dynamic/last_iter）按 M1/M2/M3 pin 死并落盘（P15）

## 10. 监控计划已建立（含 closed-loop / NaN 哨兵）

- [ ] 监控方式定好: durable 监控用 compute node 上 `ssh <node> "setsid nohup bash monitor.sh > log 2>&1 < /dev/null &"`（PPID=1）, 或 session 内 `/loop`; 登录节点 `nohup` 会死
- [ ] 判活以 GPU util + `OUT/train.log` iter 递增为准, 不看 orchestrator 的 buffered 管道日志
- [ ] 哨兵条件写明: loss/acc NaN 或骤崩（P9 sparse-keyframe 曾 ep 11-39 NaN）、吞吐骤降（被抢卡）、eval 数字连续逐位相同（P2）、`unexpected keys`（立即停训排查）
- [ ] 停止条件写明: 训到目标 ep / KPS gate 达标后拆监控, 不留 idle infra
- [ ] 每小时向 user 汇报进展（项目铁律）, 里程碑 ep 触发离线 eval（用第 5 项的 frozen snapshot）

---

## 附: launch 后 24h 内的复核点

1. Ep 1 结束: train.log 无 NaN, lr 打印格式 `%.2e`（别把 0.0000 当 lr=0）
2. 第一次 in-loop eval: 数字非退化（对照 no-control baseline 实测: FID 0.1455 / KPS 63.05cm 量级说明控制没生效）
3. 第一个里程碑 ckpt: 离线 eval（M1 协议）+ 可视化 demo 渲染（CV 铁律: 可视化 > metric, 绝不只看数字）
4. ckpt 大小 sanity: adapter-only state 应远小于 full model（MaskControl port frozen snapshot 244MB 含 base; 若异常大, 查 vq_model.* 泄漏, P1）

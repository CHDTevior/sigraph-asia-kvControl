# AGENT_PROMPT — 给未来 agent 的 KV-Control 迁移指令

> 用法: 在新项目的 agent session 里, 把下面代码块整段复制粘贴, 按需替换「本项目」上下文。

```
我要把 KV-Control 方法迁移到当前项目。KV-Control 是一个 parameter-efficient 的
控制机制: frozen base transformer + per-layer K/V residual injection
(K_aug=[K_base;K_ctrl], 低秩 D→r→D 投影, zero-init down + 可学习 ctrl_attn_bias
near-identity 初始化) + soft-decode 可微控制路径 (softmax@codebook 期望 embedding
绕过离散码本) + 两阶段 test-time optimization。它已在两个 substrate 上验证:
原生 backbone (M3 KPS 0.40cm) 和 MaskControl 移植 (KPS 63cm→1.10cm, FID 0.0875)。

参考框架: https://github.com/CHDTevior/sigraph-asia-kvControl 下的
kvcontrol_adapt_harness/ 目录。请先完整阅读:
1. README.md — 结果表 + 9 步 quick start
2. docs/02_porting_playbook.md — 迁移流程 (最重要; 第 3 步"绝不重写 substrate 的
   decode/generate loop, 只 override trans_forward"是历史上最贵的教训)
3. docs/04_pitfall_catalog.md — 15 条真实踩坑, 迁移中每一步都对照它的检测启发式
4. docs/05_data_adaptation.md — 如果本项目数据不是 HumanML3D 263 维, 按此文档
   核对 6 个必改点 (dim_pose / joints_num / 可微 FK / mean-std / contact 通道 /
   evaluator); 特别注意: 可微 FK 是控制链的 load-bearing 前提, 先移植先单测梯度
5. templates/ — 从模板起步写代码 (kv_adapter / smoke_test / eval), TODO 标记处
   填本项目 substrate 的具体值
6. reference_impl/ — 活的完整参考实现 (MaskControl port), 拿不准时对照它

执行要求:
- 严格走 checklists/porting_checklist.md, 每完成一项打勾并说明证据
- 写完 adapter 必须先跑 smoke (Tier1 bit-exact fallback + Tier2 near-identity
  at init + 梯度流 + frozen base 四项), 全过才允许训练
- 训练配置与本项目 substrate 原生方法完全一致 (A/B-clean), 只换控制机制
- 评估按 docs/03_evaluation_protocol.md 的 M0/M1/M2/M3 协议, 5-rep mean±95CI,
  评估前冻结 ckpt 快照 (防并发读写), 注意 pred_num_batch 语义和 isinstance
  dispatch 两个协议陷阱
- 判定标准: 文本-动作 quality 指标 (FID/R-prec/Diversity) 应 match substrate
  baseline; KPS 应比 no-control 至少降一个数量级; 出现连续多次 eval 输出
  bit-identical 时立即停下排查 dispatch (pitfall P2)
```

## 引用本仓库

代码/实验产物引用:

```bibtex
@misc{kvcontrol_adapt_harness,
  title  = {KVControl Adapt Harness: a porting framework for KV-Control},
  author = {KV-Control authors},
  year   = {2026},
  url    = {https://github.com/CHDTevior/sigraph-asia-kvControl}
}
```

# 04 — Pitfall Catalog（完整踩坑目录）

> KVControl Adapt Harness · 文档 4/N
> 本目录收录 KV-Control 两次移植（MaskControl port = 成功案例，3 天完成；MoMask port = 部分失败案例，4 个 bug 叠加）中**真实发生过**的全部 15 个坑。每一条都有实际观察到的症状和数字——没有一条是理论推演。
>
> 使用方式：port 新 substrate 前通读一遍；训练/评估启动前跑文末的「10 秒自检表」；任何时候出现"数字不对但代码看着没问题"，回来按检测启发式逐条排查。
>
> 参照代码（本仓库内，全部绝对路径）：
> - KV adapter（MaskControl port）：`<repo-root>/references/MaskControl/models/mask_transformer/control_transformer_kv.py`
> - 原版 ControlNet：`<repo-root>/references/MaskControl/models/mask_transformer/control_transformer.py`
> - v4 substrate（paper 主线）：`<repo-root>/models/mask_transformer/control_transformer_t_concat_v8_kv_v4.py`
> - Eval 脚本：`<repo-root>/scripts/eval_maskcontrol_kv.py`
> - DDP trainer：`<repo-root>/references/MaskControl/train_ctrlnet_ddp.py`

---

## 严重度速览

| # | 坑 | 严重度 | 出处 | 一句话 |
|---|---|---|---|---|
| P1 | EMA codebook 污染 | 🔴 致命（结果作废） | MoMask port | `requires_grad=False` 不保护 buffer |
| P2 | `type() is` vs `isinstance` | 🔴 致命（评估无效） | MaskControl eval | 子类被静默路由到无控制路径 |
| P3 | Q-residual 污染 K/V | 🔴 致命（frozen 语义漂移） | 两个 port | shared `in_proj` 让 q_delta 漏进 K/V |
| P4 | best_kps tracker 双 bug | 🟠 高（best ckpt 无意义） | MaskControl trainer | tracker 永不更新 / 用陈旧值比较 |
| P5 | DDP mkdir race | 🟠 高（rank>0 启动即崩） | MaskControl DDP | `exist_ok=False` + copytree 竞争 |
| P6 | `--is_continue` 日期前缀 | 🟠 高（resume 变 fresh） | MaskControl trainer | resume 解析到空目录 |
| P7 | ckpt 并发读写 race | 🟡 中（eval 随机崩） | 两个 port | 训练写 latest.tar 时 eval 在读 |
| P8 | `pred_num_batch` 语义 | 🟡 中（eval 空跑崩溃） | eval 链路 | 是"每次 generate 累积 N 个 batch"不是"共 N 个 batch" |
| P9 | 稀疏 keyframe NaN | 🟡 中（substrate 相关） | MoMask port | 交互性不稳定，非稀疏控制固有 |
| P10 | TensorBoard NFS 崩训练 | 🟠 高（长训练半夜死） | 两次真实死亡 | writer 异常必须 try/except |
| P11 | RT 级联 OOD | 🟠 高（下游模块加噪） | MoMask port | frozen 下游模块只见过无控制分布 |
| P12 | 手写 substrate decode loop | 🔴 致命（port #1 失败根因） | MoMask port | **移植第一铁律：decode loop 原样复用** |
| P13 | 静态 err channel | 🔴 致命（闭环原理被移除） | MoMask port | err 必须每次迭代从当前预测重算 |
| P14 | DDP `module.` 前缀 | 🟢 低（load 报错易发现） | DDP ckpt | eval load 时 strip |
| P15 | Eval protocol 参数漂移 | 🟠 高（数字不可比） | 全程 | M0-M3 参数必须 pin 且 assert |

---

## P1 EMA-codebook 污染 — `requires_grad=False` 不保护 buffer（特别展开）

**症状**（MoMask port，真实观察）：
- KV adapter 训练本身收敛正常，loss 曲线无异常；但保存的 ckpt 拿去 eval，FID 高达 **22-25**（正常范围 0.05-0.6，这是完全的 garbage 级别）。
- 更阴险的形态：eval 加载 ckpt 后，**pristine 的 RVQ tokenizer 被 ckpt 里夹带的 `vq_model.*` keys 静默覆盖**，导致同一个 eval 脚本对不同训练阶段的 ckpt 都产出被污染的数字——你以为你在评 adapter，实际连 tokenizer 都换了。

**根因**（这是一个 PyTorch 通识，值得单独记住）：
1. `requires_grad=False` 只作用于 **autograd 图**——它阻止的是 `optimizer.step()` 经由梯度更新 `nn.Parameter`。
2. 但 MoMask 的 RVQ 用的是 **EMA (exponential moving average) codebook**：codebook（`embed`、`cluster_size`、`embed_avg` 等）是 **buffer 或在 forward 里被 in-place 更新的张量**，更新发生在 `forward()` 内、以 `self.training == True` 为条件、**完全不经过 autograd**。
3. 所以 `for p in vq_model.parameters(): p.requires_grad = False` 是一句**对 EMA quantizer 完全无效的"冻结"**。只要 `vq_model` 处于 `train()` 模式（例如整个 model 被 `model.train()` 一把切换），每个 training batch 都在用当前（被控制信号扭曲的）激活分布静默漂移 codebook。
4. 漂移的 codebook 随 `state_dict()` 一起存进 ckpt（因为 vq_model 是模型的子模块，`vq_model.*` keys 全在里面）；eval 端 `load_state_dict` 又把它加载回来，覆盖刚从 pristine ckpt 加载的 RVQ → 双重污染。

**修复**（四层防御，缺一不可，全部在 MoMask port 中实装过）：
- (a) `ctrl_train()` 内显式 `self.vq_model.eval()`，并且在**每一处** `model.train()` 调用之后补 `model.vq_model.eval()`（`model.train()` 会递归把所有子模块切回 train 模式，一次就前功尽弃）；
- (b) 保存 ckpt 时从 state_dict 里 **strip 所有 `vq_model.*` keys**；
- (c) resume load 时同样 strip（防旧的被污染 ckpt 把毒带回来）；
- (d) eval 侧双保险：加载 ckpt 时**显式过滤** `vq_model.*`（`raw_sd = {k:v for k,v in raw_sd.items() if not k.startswith("vq_model.")}`）+ **byte-equal preflight assert**——把加载后的 vq_model state_dict 与 pristine RVQ ckpt 逐 tensor 字节比对，不等就 fail loud。⚠ **不要指望 `strict=False` 挡住泄漏 keys**：模型有同名 `vq_model` 子模块，泄漏 keys 名字匹配得上，`load_state_dict(strict=False)` 会**照常加载它们、静默覆盖 pristine RVQ**（strict=False 只忽略**匹配不上**的 keys）。实证：MaskControl port 的 `v18_ep6000_frozen_snapshot.tar` 带 72 个 `vq_model.*` keys（训练侧未 strip），不做显式过滤就会被载入。

**预防**（下次 port）：
- Port 任何新 substrate 时第一件事：`grep -rn "ema\|EMA\|cluster_size\|embed_avg\|register_buffer"` 它的 quantizer/tokenizer 实现。**只要有 EMA 或任何 train-mode 条件的 in-place buffer 更新，四层防御全部预装**，不要等 FID 爆了再查。
- 训练脚本里加一次性 assert：训练开始前记录 vq codebook 的 hash，每 N epoch 重新 hash 比对，不等即 abort。

**检测启发式**：
- adapter loss 正常但 eval FID 离谱地差（>10）→ 第一怀疑对象就是 tokenizer 被污染。
- `load_state_dict(strict=False)` 的 unexpected keys 列表里出现 `vq_model.*` → 已经中招，ckpt 里有夹带。
- 对比训练前后 `vq_model.quantizer.*` 任一 buffer 的 checksum，不等 = 实锤。

---

## P2 `type() is` vs `isinstance` — bit-identical eval = dispatch 没走到你的模型（特别展开）

**症状**（MaskControl port，真实观察）：
- 连续 **3 次 eval，输出到小数点后 4 位完全 bit-identical**：FID=0.1455 / KPS=63.05——评的明明是三个**不同 epoch** 的 KV ckpt。
- KPS 63cm 恰好就是**无控制 baseline 的 KPS**（no-control ~63cm）。也就是说控制机制根本没被调用。

**根因**：
- `utils/eval_t2m.py:1082` 曾用 `type(x) is ControlTransformer` 做模型 dispatch（现已修复：isinstance 在 L1086，:1082 是历史 bug 位置、现为修复注释行）。我们的 `KVControlTransformer` 是 `ControlTransformer` 的**子类**（`references/MaskControl/models/mask_transformer/control_transformer_kv.py` L125：`class KVControlTransformer(ControlTransformer)`），`type()` 身份比较对子类返回 False → eval 静默走到了 base 模型的无控制生成路径。
- 没有任何报错、没有任何 warning。数字看起来"合法"（不是 NaN 不是 0），只是全错。

**修复**：
- `utils/eval_t2m.py:1082` 的 `type(x) is ControlTransformer` → `isinstance(x, ControlTransformer)`（已落地：isinstance 现在在 L1086）。
- 通用规则：**任何按模型类做 dispatch 的 eval/训练代码，一律 `isinstance`**，因为移植的标准姿势就是子类化（见 P12）。

**预防**：
- Port 开始时就 `grep -rn "type(.*) is \|type(.*) ==" ` 整个 substrate 的 eval / trainer / generate 链路，把所有 `type()` 身份比较列成清单逐个改。
- 首次 eval 前做一次 **smoke 对照**：同一 ckpt 分别以 `ctrl_net=True/False` 跑一个 mini eval，两组数字必须显著不同（尤其 KPS）。相同 = dispatch 没生效。

**检测启发式**（这条启发式本身就是本项目最值钱的产出之一）：
- **不同 ckpt 的 eval 输出完全相同 ⇒ 你以为在评的模型没有被调用。** 立即停止，去查 dispatch，不要先怀疑"模型没学到东西"。
- KPS 恰好等于 no-control baseline 值 ⇒ 控制路径没走到。
- 反向应用：故意评一个随机初始化 adapter 的 ckpt，如果数字和训练好的 ckpt 一样好/一样差，同理说明 eval 没在评 adapter。

---

## P3 Q-residual 经 shared `in_proj` 污染 K/V（特别展开）

**症状**：
- 初期（q_gates=0）**完全无症状**——这是它最危险的地方。gate 打开后（训练中后期），frozen backbone 的语义开始**静默漂移**：base-only 生成质量（本应与 frozen base 完全一致）逐渐变差，而你没有任何一处代码"动过" frozen 权重。

**根因**：
- `nn.MultiheadAttention` 的 Q/K/V 投影共用一个 `in_proj_weight`（shape `[3D, D]`），forward 是 `F.linear(x, in_proj_weight).chunk(3)`。
- 如果把 `q_delta` 加到 **x 上再投影**（`x = x + gate * q_delta; qkv = F.linear(x, in_proj_weight)`），那么 q_delta 同时经过了 `W_Q`、`W_K`、`W_V` 三块权重——**残差泄漏进了 K_base 和 V_base**。设计意图是"只扰动 query 流"，实际上把整个注意力的 key/value 语义都改了，等价于对 frozen backbone 做了非受控修改。
- init 时 gate=0，泄漏量为 0，所以 smoke test 全部通过；gate 学开后毒性才显现。

**修复**（代码级，两个 substrate 各一份参考实现）：
- MaskControl port：`references/MaskControl/models/mask_transformer/control_transformer_kv.py`
  - 注释 L69-71 + 代码 L72-73：**先**用未修改的 x 算 `qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(3)` → `q, k_base, v_base`——K/V 永远看不到 residual；
  - L77-82：`q_delta = q_up(q_down(x)) * q_gate`；L81 切出 `W_Q = attn.in_proj_weight[:D]`；L82 `q = q + F.linear(q_delta, W_Q)`——residual 只经 W_Q 投影、只加到 q。
- v4 substrate：`models/mask_transformer/control_transformer_t_concat_v8_kv_v4.py` L141-146（gate 逻辑）+ L166-182（post-norm 路径）：L169-171 用 `q_input = output.clone(); q_input[1:1+S_motion] += gate_val * q_delta_exp[...]`，而 L172-173 的 `k_aug`/`v_aug` 从**未修改的 output** 拼接——同一原则的另一种写法（q_input 与 k/v 输入分离）。

**预防**：
- 写任何 Q-only（或 K-only/V-only）residual 时，检查 substrate 的 attention 是不是 fused `in_proj`。是 → 必须手工拆投影，绝不能"加在输入上"。
- 配套 init 纪律（v4 文件 L313-322 的注释就是教训本身）：`q_down` **标准初始化**、`q_gates` **零初始化**——identity at init 且梯度活；如果 q_down 也 zero-init，则 `∂L/∂gate ∝ q_up(q_down(x)) = 0` 且 `∂L/∂q_down ∝ gate = 0`，**双零 = dead branch**，永远学不动（v8_kv 有个历史文件名就叫 `control_transformer_t_concat_v8_kv_bk_20260322_deadgrad.py`）。同理 KV 侧是反过来的：`kv_down` zero-init + `kv_up` 标准 init（v4 L303-304，MaskControl port L190-191）。

**检测启发式**：
- 单测：固定输入，分别在 gate=0 和 gate=1e-3 下 forward，比较 **k_base/v_base 张量**——只要 gate 影响了 k_base/v_base 任一元素，就是泄漏。这个测试 10 行代码，port 时必写。
- 训练中后期 base-only decode（关掉控制）质量变差 → frozen 语义被污染，查所有 residual 注入点。

---

## P4 best_kps tracker 双 bug — best ckpt 选择完全失效

**症状**：`best_kps.tar` 的 mtime 每次 eval 都刷新（每次都被覆盖），或反过来永远停在第一次 eval；用 best_kps.tar 评出的 KPS 与训练日志声称的 best 不符。

**根因**（MaskControl trainer，两个独立 bug 叠加）：
- (a) 保存 best ckpt 之后 **`best_kps_mean` 变量从未更新** → 比较条件永真 → 每次 eval 都覆盖 `best_kps.tar`（"best"实为"latest eval"）；
- (b) in-loop eval 的返回值解包把 `kps_mean` 解进了 `_`（丢弃），比较用的是循环外的陈旧值 → tracker 冻结在第一次 eval。

**修复**：两处都改——保存后立即 `best_kps_mean = kps_mean`；解包时接住真实的 `kps_mean` 变量再比较。两个 bug **必须同时修**，只修一个 tracker 依然是错的。

**预防**：port 新 trainer 时，对每个 `best_*.tar` 写一个 3 行断言脚本：跑两次 eval（一次好一次差的假数据），验证 tracker 只在更优时更新。

**检测启发式**：`ls -l model/best_*.tar` 看 mtime——每次 eval 都变 = bug (a)；永远不变（而日志显示 metric 在改善）= bug (b)。

---

## P5 DDP mkdir race — rank>0 在共享文件系统上启动即崩

**症状**：单卡训练正常；DDP 启动时 rank 1/2/3 抛 `FileExistsError`（`os.makedirs`）或 `shutil.copytree` 目标已存在，训练根本起不来（或更糟：偶发地起得来，取决于竞速结果）。

**根因**：三处路径创建代码都假设"只有一个进程"：
- `options/base_option.py` 的 `os.makedirs(..., exist_ok=False)` + 存在即 `exit`；
- `utils/utils.py` 的 `init_save_folder`；
- `shutil.copytree`（备份代码快照）。
多 rank 在共享 FS 上同时执行 → 竞争。

**修复**：option/save-folder 全链路 `exist_ok=True` / `dirs_exist_ok=True`；更干净的做法是 **rank-0-only 创建目录 + `dist.barrier()`**（参考 `references/MaskControl/train_ctrlnet_ddp.py` 头部文档描述的 rank-0-only prints/checkpoints/TB/eval 模式）。

**预防**：任何单卡 trainer 转 DDP 前，grep `makedirs\|copytree\|open(.*w` 找出所有写文件点，逐个套 rank-0 守卫。

**检测启发式**：DDP 下偶发启动失败、报错栈在 option parsing / save folder 初始化 → 就是这个。

---

## P6 `--is_continue` 与日期前缀 — resume 解析到全新空目录

**症状**：带 `--is_continue` 启动续训，训练"正常"从 epoch 0 开始（悄无声息地重新训练），ckpt 目录里多出一个新的 `z<今天日期>_<name>` 空目录。

**根因**：`options/base_option.py` **无条件**给 `opt.name` 加 `z<date>_` 前缀。resume 时用户传的是旧 run 名，加上今天的日期前缀后指向一个不存在的目录 → 代码顺手创建了它 → 找不到 ckpt → fresh start。

**修复**：`is_continue` 为真时跳过日期前缀（用户传入的 name 已含旧日期前缀，原样解析）。

**预防**：resume 路径解析后立即 assert `latest.tar` 存在且 `ckpt['ep'] > 0`，不存在就 fail loud，绝不允许静默 fresh start（呼应 R12 Fail loud）。

**检测启发式**：resume 后第一条日志的 epoch 是 0 / lr 是初始值 / 目录 mtime 是刚刚 → 你在重训不是续训。

---

## P7 ckpt 并发读写 race — eval 读到写了一半的 latest.tar

**症状**：离线 eval 偶发崩溃：`PytorchStreamReader failed reading file`（或 zip archive corrupt）。重跑又好了——因为第二次没撞上写入窗口。

**根因**：训练进程每 epoch 覆盖写 `latest.tar`，eval 进程恰好在写入中途 `torch.load` 同一路径。

**修复**：eval 前先 **`cp` 到 frozen snapshot 路径**（`cp` + 原子 `mv` 两步，保证 snapshot 本身完整），eval 只读 snapshot。本项目现行实践即如此：`<repo-root>/output/v18_ep6000_frozen_snapshot.tar`（244MB frozen snapshot，与训练目录解耦）。

**预防**：把"snapshot 后再 eval"写进 eval 脚本本身（脚本内部做 cp+mv），不要依赖人肉记得。

**检测启发式**：`PytorchStreamReader failed` + 被读文件是训练进程正在写的路径 = 100% 这个坑，不用怀疑 ckpt 真坏了。

---

## P8 `pred_num_batch` 语义误读 — 设太大 = 静默跳过全部生成

**症状**：eval 跑完（无警告）在 `motion_annotation_list` 处 `torch.cat` 空 list 崩溃；或跑得可疑地快。

**根因**：`evaluation_mask_transformer_test_plus_res` 的 `pred_num_batch` 意思是"**每次 generate 调用前累积 N 个 loader batch**"，不是"总共处理 N 个 batch"。设成 99999 → 永远凑不满 → **一次 generate 都不触发** → 结果 list 全空。

**修复**：用 16（= 16×bs32 = 512 samples/次 generate）。参考 `scripts/eval_maskcontrol_kv.py` L222-232 调用处及 L227-230 的注释（2026-07 修正后行号）。

**预防**：eval 脚本里 assert generate 被调用次数 > 0（或 assert 结果 list 非空并打印实际样本数）。

**检测启发式**：eval 时 GPU util 长期为 0（没有生成在跑）；崩溃栈在 `torch.cat` 空列表。

---

## P9 稀疏 keyframe NaN — substrate 交互性问题，非稀疏控制固有

**症状**（MoMask port）：keyframe density ∈ {1, 2, 5} 的训练在 **ep 11-39 之间 NaN**，4 次重试（换 lr/clip 等）全部复现；dense（全帧）控制训练稳定。

**根因**：当时未完全定位；但关键的反证是——**同样的随机 density 训练在 MaskControl substrate 上（闭环 err channel + 其原生 loss 配比 xent 0.1 / ctrl_loss 0.9）完全稳定训满 6000 ep**。结论：不稳定是「该 substrate × 静态 err（见 P13）× loss 配比」的交互效应，不是稀疏控制内在不可训。

**修复/预防**：
- 新 port 遇到稀疏 NaN，第一步不是调 lr，而是先检查 P13（err channel 是否闭环）和 loss 配比是否照搬了 substrate 原生控制训练的配置；
- 训练配 per-epoch NaN gate：loss/acc 出现 NaN 立即 abort 并保留现场 ckpt（fail loud），不要让 NaN 静默传播几百个 iter 再发现。

**检测启发式**：只有稀疏 density 崩、dense 稳 → 交互性问题，去查 err channel 与 mask 的乘积路径（`(gt-pred)*mask` 在极稀疏时几乎全零，任何除以 mask-count 的归一化都可能除零）。

---

## P10 TensorBoard writer 在 NFS 上抛 FileNotFoundError 杀死训练

**症状**：长训练在毫无征兆的 epoch 中途死掉——真实死过两次：**ep 17 和 ep 152**。栈顶是 `writer.add_scalar` → NFS 瞬时不可见文件 → `FileNotFoundError` → 未捕获 → 整个训练进程退出。

**根因**：NFS/共享 FS 的 event 文件偶发短暂不可访问；TB writer 不做重试，异常直接上抛；训练循环没有兜底。

**修复**：**所有** `writer.add_scalar` 调用点（per-iter 的和 end-of-epoch 的，一个都不能漏）包 `try/except`——logging 失败绝不允许连坐训练。

**预防**：port 新 trainer 时 `grep -n "add_scalar\|add_histogram\|writer\."` 逐个包裹；或统一封一个 `safe_log()`。

**检测启发式**：训练死亡且栈里有 `tensorboard`/`event file` 字样 = 这条；数据本身没问题，直接包异常重启即可。

---

## P11 RT-cascade OOD — frozen 下游模块把 KV-控制后的 token 当噪声源

**症状**（MoMask port）：对 KV ckpt，**关掉 ResidualTransformer（base-only decode）反而更好**：FID 0.715（RT-on）→ 0.581（base-only）。RT 本应是提升质量的模块，却在加噪。

**根因**：MoMask 的 RT 是在 **vanilla base 的 token 分布**上训练的 frozen 模块。KV 注入改变了 base token 分布，RT 眼中这些 token 是 OOD 输入 → 输出的 residual 修正是错的。

**修复**：当次 salvage 直接 base-only decode。正解是（如果要 RT）在 KV-控制后的分布上 finetune RT——但那破坏 frozen-adapter 的故事，需权衡。

**预防**（普适教训）：**盘点 substrate 生成链路里所有 frozen 下游模块**（residual quantizer 级联、refiner、super-resolution head、post-net……），每一个都是潜在 OOD 放大器。Port 时对每个下游模块做 on/off 消融，KV ckpt 上 off 更好 = OOD 实锤。另注意 codebook 结构差异：MoMask RVQ 有 6 个 quantizer（base + 5 residual），v4 substrate 是 Q=6 个 part codebook 沿序列 unpack——soft-decode 应该 mix 哪个 codebook 必须先搞清楚。

**检测启发式**：下游"增强"模块开着比关着差 → OOD。

---

## P12 手写 substrate decode loop — MoMask port 的根因失败（特别展开，全案复盘）

**这是整个 harness 的第一移植铁律，MoMask port 失败的根本原因。**

**复盘第一段——发生了什么。** MoMask port 时我们没有复用它的 `generate()`，而是**手写重实现了 MaskGIT 解码循环**。手写版在多个结构细节上与原版偏离：pad token 的初始化方式、confidence score 的初始化、re-masking 的采样顺序。这些细节单看每一个都"无关紧要"，叠加起来对生成分布造成了大幅 FID 损伤。更糟的是，decode loop 的偏差为 P13（静态 err）、P11（RT OOD）、P1（codebook 污染）提供了掩护——四个 bug 层层叠加，每一个的症状都被其他三个搅浑，逐个定位花掉了 port 的绝大部分时间。最终 salvage 结果只有 FID 0.581 / KPS 30.6cm（best ep600, base-only decode），远未达到可用水平。

**复盘第二段——为什么手写必然出问题。** MaskGIT 类解码循环的正确性依赖大量隐式约定：score 张量在 padding 位置的填充值、温度如何随 timestep 退火、top-k 过滤发生在 softmax 前还是后、re-mask 时是否保留已确认 token、CFG 的 unconditional 分支怎么拼 batch。这些约定**不写在论文里，只活在 substrate 的代码中**，且往往互相耦合。重实现时你只能复现你注意到的那部分；你没注意到的那部分就是 bug——而且是不报错、只损分布的最难查的那种 bug。结论：decode loop 的正确性验证成本 >> 复用成本，重写没有任何划算的场景。

**复盘第三段——对照组证明。** MaskControl port（成功案例）严格执行了相反的策略：subclass 它的 `ControlTransformer`，`delattr` 掉它的 ControlNet 分支（`references/MaskControl/models/mask_transformer/control_transformer_kv.py` L174-176：`del self.seqTransEncoder_control` / `del self.first_zero_linear` / `del self.mid_zero_linear`），只 override `ctrl_train`/`ctrl_eval`/`trans_forward`（L220-242 / L244-246 / L281-319）+ 一个 bit-exact 的 `_base_trans_forward` fallback（L321-339，注意 docstring L322-326 强调它必须自己实现 base 路径而非委托 parent），而 **`generate_with_control` 和 Stage-1/Stage-2 TTT 一行未动、原样复用**。结果：3 天（含 debug）完成移植，机制热插拔验证成功，KPS 63cm → 1.29cm（M3）。同一个团队、同一个机制，两种策略，结局天壤之别。

**修复/预防（铁律表述）**：
- **REUSE substrate 的 generate()/decode loop verbatim。唯一允许 override 的是 `trans_forward`（per-layer forward），KV 注入在它内部拼接。**
- 注入实现的参考：MaskControl port 的 `_layer_with_kv_injection`（`control_transformer_kv.py` L41-122：L91-93 `k_aug=cat([k_base,k_ctrl])`，L95-103 attn_mask 上给 ctrl 列加 bias 标量）；v4 的 `forward_with_kv_context_and_crossattn`（`control_transformer_t_concat_v8_kv_v4.py` L64-192）。
- 如果 substrate 的 decode loop 结构上无法容纳注入（极少见），宁可给 loop 打最小 patch（diff 可 review），也不整体重写。

**检测启发式**：
- 你发现自己在新文件里写 `while not all_unmasked:` 或 `topk(confidence)` → 停，回去看 substrate 有没有现成的。
- base-only 模式（关控制）下你的 fork 与 substrate 原版 generate 的输出**不 bit-exact** → decode loop 已偏离，先修到 bit-exact 再谈控制。这也是 `_base_trans_forward` 存在的意义：它给了一个可以和原版 MaskTransformer 逐位对拍的锚点。

---

## P13 静态 err channel — 闭环迭代校正原理被整体移除（特别展开）

**症状**（MoMask port）：控制训练"收敛"了但控制精度极差且不随 TTT 改善；KPS 停在远高于预期的水平（构成最终 30.6cm 的重要部分）。

**根因**：KV-Control 的控制条件是双通道的：

```
ctrlNet_cond = cat([(gt_joints − pred_joints) * mask,   # residual/err 通道
                     gt_joints * mask], dim=-1)          # absolute/target 通道
```

（v4 参考实现：`models/mask_transformer/control_transformer_t_concat_v8_kv_v4.py` `_build_ctrl_cond` L376-387，L377 err 通道、L378 target 通道；trajectory 时各取 pelvis → C=6。）

这是一个**闭环（closed-loop）**设计：err 通道必须**在每个 training batch 和每个 inference 迭代中从当前预测重新计算**——网络据此知道"当前预测离目标还差多远、往哪个方向修"。这正是论文 §3 的 iterative trajectory error correction 原理（`p_pred_s` 是 iterative current estimate，不是 frozen base 的一次性输出）。

MoMask port 犯的错：**喂了 `err_xyz = 0`（静态）**。这不是"少了一个 feature"，而是**把迭代校正原理整体从系统中移除**——网络只知道目标在哪，不知道自己现在在哪，等价于把闭环控制器降级成开环前馈。控制自然学不出精度，TTT 也失去了梯度信号的核心结构。

**修复**：
- 训练：每个 batch 先做当前 decode 得 `pred_joints`，再按上式构造 cond（v4 `_build_ctrl_cond` 的调用方式）；
- 推理：每个 MaskGIT step / 每次 TTT 迭代都用**当前 decode 结果**重算 cond（MaskControl 的 `generate_with_control` 原生就是这么做的——又一个 P12 复用铁律的论据：复用了它就自动闭环）。

**预防**：
- Port checklist 固定项：打印训练中某一 batch 的 err 通道统计量（mean/std）。**err 通道恒为 0 或恒为常数 = 静态喂入，立即修**；
- 新 substrate 的 err 通道要求 decode→joints 全程可微且可在训练循环内负担得起（每 batch 一次额外 decode）——评估这个成本是 port 前期工作。

**检测启发式**：
- TTT 迭代数加倍但 KPS 不动 → 梯度信号结构可能已坏，查 err 通道是否闭环；
- err 通道张量 `std == 0` → 实锤静态；
- 控制精度对 `mask` 密度完全不敏感 → cond 大概率没被网络真正使用。

---

## P14 DDP ckpt 的 `module.` 前缀

**症状**：eval 加载 DDP 训练的 ckpt，`load_state_dict` 报 missing/unexpected keys 全部带 `module.` 前缀（或 `strict=False` 下静默全 miss → 模型是随机初始化——那就危险了）。

**根因**：`DistributedDataParallel` wrap 后 `state_dict()` 所有 key 带 `module.` 前缀；训练侧如果没从 unwrapped module 保存，前缀就进了 ckpt。

**修复**：eval load 时检测并 strip——现行实现：`scripts/eval_maskcontrol_kv.py` L198-200（2026-07 修正后行号）。训练侧正解是保存 `model.module.state_dict()`（rank-0-only，见 `train_ctrlnet_ddp.py` 的 rank-0-only ckpt 模式）。

**预防**：load 后 assert 关键 adapter 参数（如 `kv_down.0.weight`）确实被加载（不在 missing 列表），防 `strict=False` 把"全没加载"吞成 warning。

**检测启发式**：missing keys 数量 == 模型参数总数、且 unexpected keys 是同名带前缀版本。

---

## P15 Eval protocol 参数漂移 — 数字悄悄不可比

**症状**：同一 ckpt 两次 eval 数字对不上；或论文表格里 M1/M3 的对比其实混了不同的 `time_steps`/`cond_scale`——发现时已经写进 draft。

**根因**：协议由 5+ 个 CLI 参数联合定义，任何一个吃了默认值的漂移都让数字失效。本项目的 pin 值（`scripts/eval_maskcontrol_kv.py` L83-L100 defaults 即 M1，2026-07 修正后行号）：

| 参数 | M1 | M2 | M3 |
|---|---|---|---|
| `time_steps` | 10 | 10 | 10 |
| `cond_scale` | 3.25 | 3.25 | 3.25 |
| `each_iter` | 35 (**uniform**) | **100 (uniform)** | 35 (**uniform**) |
| `ttt_dynamic` | False | False | False |
| `last_iter` (Stage-2) | **0** | 600 | 600 |
| `each_lr` / `last_lr` | 6e-2 | 6e-2 | 6e-2 |
| `repeat_times` | 5 | 5 | 5 |
| `seed` | 3407 | 3407 | 3407 |

脚本只把 M1 的精确组合识别为 `protocol="M1"`，其余全部标 `"CUSTOM"`——M2/M3 靠手工 override `--each_iter`/`--last_iter` 产生，**没有硬编码保护**，最容易漂。⚠ 协议标注勘误：M1/M3 曾被标 "dynamic"——错误；substrate 只在 `each_iter<0` 时走 dynamic（`control_transformer.py` L527-531），已发布数字全部是 uniform（正 each_iter）。`--ttt_dynamic` 现已接通 negative-each_iter 约定、默认 False，正式协议一律不传。

**修复/预防**：
- 每个 protocol 一个 wrapper 脚本（参数写死），禁止裸调 eval 脚本跑正式数字；
- 输出 JSON 里落盘完整参数快照（现行脚本已做），论文表格引用数字时必须回链到对应 JSON（如 `<repo-root>/output/v16_kv_maskcontrol_4xa100_20260701_013823/eval_5r_M3_ep6000/eval_5r.json`）；
- 正式 5-rep 报 mean ± 95% CI（std × 1.96 / √N），单 rep 数字一律标注 1r 不进正式表。

**检测启发式**：输出 JSON 里 `protocol: "CUSTOM"` 但你以为自己在跑 M2/M3 → 立刻核对全部 8 个参数；两次"相同协议"eval 差异远超 5r CI → 有参数漂了。

---

## 10 秒自检表

打印出来贴墙上。任何一个 **No** → 停，先修再跑。

### 训练启动前（5 问）

| # | 问题 | 对应坑 |
|---|---|---|
| T1 | tokenizer/VQ 是 EMA 型吗？如果是——`ctrl_train()` 里有 `vq_model.eval()`、ckpt save/load 都 strip 了 `vq_model.*` 吗？ | P1 |
| T2 | err 通道是每个 batch 从**当前预测**重算的吗（打印过它的 std ≠ 0）？ | P13 |
| T3 | Q-residual 的注入是拆开 `in_proj` 只加到 q 上的吗（gate 微开时 k_base/v_base 不变的单测过了吗）？且 zero-init 侧和 standard-init 侧没搞成双零？ | P3 |
| T4 | DDP 三件套齐了吗：目录创建 rank-0-only/exist_ok、ckpt 存 `module.` 前缀已剥、所有 `writer.add_scalar` 有 try/except？ | P5/P10/P14 |
| T5 | 这是 resume 吗？如果是——启动日志的 epoch > 0、lr 是 schedule 中段值、没有新建 `z<今天>_*` 目录？ | P6 |

### 评估启动前（5 问）

| # | 问题 | 对应坑 |
|---|---|---|
| E1 | eval 链路的模型 dispatch 全是 `isinstance` 吗（grep 过 `type(.*) is` 了吗）？ | P2 |
| E2 | 评的是 frozen snapshot（cp+mv 出来的），不是训练正在写的 `latest.tar`？ | P7 |
| E3 | 协议参数 8 项全部显式传入并与 M1/M2/M3 pin 表核对过了吗？`pred_num_batch=16`？ | P15/P8 |
| E4 | load_state_dict 之后检查过 missing/unexpected 列表了吗——adapter 参数全在、没有 `vq_model.*` 混进来、`module.` 前缀已 strip？ | P14/P1 |
| E5 | 交叉验证：这次的数字和上一个不同 ckpt 的数字**不**完全相同吧？（完全相同 = 你评的不是你以为的模型） | P2 |

---

*本文档基于 2026-06/07 的 MaskControl port（成功）与此前 MoMask port（部分失败）的全部一手记录整理。数字来源：v16→v18 ep6000 5-rep 正式 eval（`output/v16_kv_maskcontrol_4xa100_20260701_013823/eval_5r_M3_ep6000/eval_5r.json`）及各次事故当时的训练/评估日志。*

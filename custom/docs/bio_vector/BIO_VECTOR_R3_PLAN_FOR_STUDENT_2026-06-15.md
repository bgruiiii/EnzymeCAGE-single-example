# Bio Vector R3 推进文档（学生 Codex 直接执行版）

> 时间：2026-06-15
> 目标：R2 postmortem 已通过；本文是从 R2 收尾到 R3 训练完成的一站式指令。
> 边界：所有 R3 决策已由老师拍板，Codex 不要再做"是否需要"的判断，按本文执行即可。
> 报告语言：英文写正文 + 中文写老师提问点的回答即可，与之前保持一致。

---

## 0. R2 Postmortem 验收结论

| 文件 | 结果 |
|---|---|
| `R2_POSTMORTEM_20260614.md` | ✅ 24 项全 PASS |
| `BIO_VECTOR_R2_RESULT_SUMMARY_2026-06-11.md` | ✅ §4/§5/§6 改写到位，无残留禁词 |
| `postmortem_eval_stage_checkpoints.py` | ✅ 复用现有函数，docstring 完整 |
| `R3_DECISION_INPUT.md` | ✅ 1 页内，不含 R3 plan 细节 |

**Postmortem 阶段任务结束**。下面进入 R3。

---

## 1. R3 启动前的 2 项补充诊断（仍是零 GPU，应在 R3 训练之前完成）

老师在审 Task 1.6 时发现两点 baseline 信息没说透。**这两项不是新阈值，只是把 R2 已有的数据再深一层**，写进同一个 postmortem 文件即可，不要新开报告。

### 1.1 Calibration 曲线形状定性 + bin-level 偏差表

R2 给的 ECE = 0.107 是聚合数，但 bin-level 严重不均衡：

| Bin | mean score | EC-4 hit rate | n | 偏差 (score − hit) | 解读 |
|---|---:|---:|---:|---:|---|
| 0.7–0.8 | 0.7827 | 0.6429 | 28 | +0.140 | 轻度过自信，样本极少 |
| **0.8–0.9** | **0.8776** | **0.3924** | **15,734** | **+0.485** | **严重过自信（关键 risk zone）** |
| 0.9–1.0 | 0.9571 | 0.9033 | 112,085 | +0.054 | 接近对角线 |

**Codex 任务**：在 `R2_POSTMORTEM_20260614.md` 的 Task 1.6.1 段落末尾追加：

```markdown
### 1.6.1 Calibration Shape Qualitative (added 2026-06-15)

The 10-bin calibration curve is highly non-uniform:

- 0.9–1.0 bin (112,085 / 88% of valid queries): nearly diagonal
  (gap = +0.054).
- 0.8–0.9 bin (15,734 / 12% of valid queries): severely overconfident
  (score 0.878 vs hit 0.392, gap = +0.485). This is the dominant
  contributor to the aggregated ECE = 0.107.
- 0.7–0.8 bin (28 samples): low support, no statistical conclusion.
- < 0.7 bins: zero samples.

Implication for future agent integration (no threshold set now):

- A single global confidence threshold is not usable. The 0.8–0.9 band
  is the high-risk zone: a confidence score of 0.85 corresponds to ~39%
  EC-4 hit rate.
- Two recalibration candidates worth exploring at agent integration time:
  isotonic regression and Platt scaling on the 0.8–0.9 band, or
  abstaining when score < 0.9.
- This is recorded as a baseline finding only; no threshold is set in
  R2 or R3.
```

### 1.2 真实 OOD 候选数据来源清单（只列名单，不评估）

R2 用的 OOD 是 synthetic Gaussian noise proxy，不是真分布。R3 不引入 OOD 评估指标，但要把"未来 agent 接入时需要哪些 OOD 数据"提前列清楚。

**Codex 任务**：在 `R2_POSTMORTEM_20260614.md` 的 Task 1.6.2 段落末尾追加：

```markdown
### 1.6.2 Real OOD Candidate Sources (added 2026-06-15)

The current OOD-like score distribution uses a feature-level Gaussian
proxy (sigma=0.5 on 5% of reaction embeddings). Real OOD evaluation
must wait for agent integration. The following real OOD candidate
sources are recorded so they can be collected when the upstream agent
is built:

| Candidate source | Why it is OOD for this model | How to obtain |
|---|---|---|
| Non-enzymatic reactions (acid/base, photochemical) | The model is trained only on enzyme-catalyzed rows | Filter MetaCyc / Rhea by `enzymeless` flag, or take a small curated set |
| Cross-domain reactions (e.g. industrial catalysis, organic synthesis textbooks) | Reaction-side feature distribution differs from enzyme-catalyzed pathways | USPTO subset filtered to no-enzyme entries |
| Human-designed novel reactions (de novo retrosynthesis outputs) | No EC label, no UniProt anchor | Sample from a small retro-synthesis tool output |
| EC labels seen <= 4 times in training (long-tail in-distribution) | Behaves OOD in practice due to weak training signal | Reuse `metadata_v3.json` group-size filter |

This is a data-collection list only. No real-OOD evaluation is run in
R2 or R3. Agent integration owns this work.
```

完成上述两段追加后，`R2_POSTMORTEM_20260614.md` → 重命名为 `R2_POSTMORTEM_20260615_FINAL.md`，作为 R2 最终版本归档。

---

## 2. R3 训练计划（老师已拍板）

### 2.1 项目定位（不变，再确认一遍）

```
模型角色: 多智能体系统中的 tool / verifier
主要输出: R → EC-4 retrieval + 候选 enzyme/microbe evidence
主要指标: EC-4-grouped R→E MRR
辅助指标: E→M MRR (cross-modal monitor), UniProt-grouped (reference)
不再追逐: row-level exact retrieval (ceiling 0.0197 已锁定)
```

R3 设计的所有取舍都服从这个定位。

### 2.2 R3 主要变更（共 3 处，全部基于 R2 postmortem 证据）

#### 变更 1：EC-4 class-balanced sampling（核心变更）

**证据**：Task 1.1 EC-4 max/min = 2291，median = 4。head 类（≥317 row, p95+）主导 loss，long-tail 类（≤4 row）信号弱。

**实现**：在 `train.py` 的 stage1 / stage2 训练 DataLoader 处用 `WeightedRandomSampler` 替换默认 shuffle。每个样本的权重为：

```python
# 伪代码（具体由 Codex 实现到 train.py）
import math
from collections import Counter

ec4_counts = Counter()
for row in metadata:
    ec4 = parse_ec4_strict(row.get("ec_number", ""))
    ec4_counts[ec4] += 1   # ec4 = None 视为单独一桶 "unknown"

def sample_weight(row):
    ec4 = parse_ec4_strict(row.get("ec_number", ""))
    n = ec4_counts[ec4]
    return 1.0 / math.sqrt(n)   # 平方根权重，软化长尾

sampler = WeightedRandomSampler(
    weights=[sample_weight(r) for r in metadata],
    num_samples=len(metadata),   # 每 epoch 仍跑等量样本
    replacement=True,
)
```

**只用平方根权重，不用 1/n**。1/n 会过度倾斜到 long-tail 单样本类，反而扰乱 head 类的对比信号。`1/sqrt(n)` 是 long-tail recognition 的标准软化方案。

**stage0 不动**：stage0 是 cross-modal alignment 预热，不需要 class-balanced sampling。

#### 变更 2：砍 stage3

**证据**：Task 1.2 stage3 vs stage2 marginal change：
- row R→E: 0.0605 → 0.0602 (−0.5%)
- EC-4-grouped: 0.9265 → 0.9179 (−0.9%) **退化**
- E→M: 0.6226 → 0.6212 (−0.2%)

**stage3 在每个监控指标上要么不变要么微退**。R3 直接 `epochs_stage3 = 0` 跳过该阶段。

**实现**：修改 `Config` 中 `epochs_stage3` 默认值为 0；`main()` 中检测到该值为 0 时跳过 stage3 整个训练 phase（但仍保存 `model_v3_stage3.pt` 作为 stage2 终态的 alias，确保 evaluation 脚本兼容性）。

> 兼容性兜底：如果 Codex 觉得"alias stage2 ckpt 为 stage3 ckpt"会引起后续脚本误读，可以让 `model_v3_stage3.pt` 物理不存在，但要在 `metrics_v3.json` 里写 `"epochs_stage3": 0` 字段，并修改 `postmortem_eval_stage_checkpoints.py` 在 stage3 缺失时输出 `stage3: skipped` 而非报错。**两种方案任选一种**。

#### 变更 3：保存所有 stage checkpoint（已满足，验证一遍）

R2 已保存 `model_v3_stage{0,1,2,3}.pt`，说明 train.py 当前逻辑没问题。R3 仅需确认：
- 砍掉 stage3 后，stage0/1/2 的 checkpoint 仍正常保存
- `training_history.json` 仍记录 stage 边界 epoch
- `metrics_v3.json` 仍按 final state 写入

不需要修改保存逻辑。

### 2.3 R3 配置 vs R2（一表对照）

| 配置项 | R1 | R2 | **R3** | 变更原因 |
|---|---:|---:|---:|---|
| `hard_neg_weight` | 2.0 | 1.0 | **1.0** | R2 已验证非 harmful |
| `epochs_stage0` | 5 | 5 | **5** | 不变 |
| `epochs_stage1` | 12 | 25 | **25** | R2 已验证为合理点 |
| `epochs_stage2` | 8 | 8 | **8** | 不变 |
| `epochs_stage3` | 5 | 5 | **0** | stage3 退化（变更 2） |
| EC-4 class-balanced sampler | 否 | 否 | **是 (1/sqrt)** | EC-4 长尾（变更 1） |
| EC encoding | 7 类 | 7 类 | **7 类** | hard_neg=1.0 时为 dead code，不动 |
| Concept loss / VICReg | 不动 | 不动 | **不动** | 与 R2 持平 |

### 2.4 R3 验收标准（量化，使用既有指标，**不引入新阈值**）

R3 训练完成后，按下表判定。**所有数值都是与 R2 比较，不引入 calibration / OOD / latency 阈值**。

| 指标 | R2 baseline | R3 目标 | 判定 |
|---|---:|---:|---|
| EC-4-grouped R→E MRR | 0.9132 | ≥ 0.9132 | **主指标，不可退化** |
| E→M MRR | 0.6195 | ≥ 0.61 | 监控指标，允许 ±1% |
| EC-2-grouped R→E MRR | 0.9306 | ≥ 0.9200 | 监控指标，允许小幅下降（class-balanced 会软化 head） |
| **EC-4 long-tail bucket MRR**（group size ≤ 4） | **R2 需补算** | **≥ R2 同口径值 + 5%** | **核心提升验证** |
| **EC-4 head bucket MRR**（group size > 317 即 p95+） | **R2 需补算** | **≥ R2 同口径值 − 3%** | head 不可大幅退化 |
| row-level R→E MRR | 0.0602 | reference only | 不作判定（ceiling-bound） |
| Stage 训练时间 | ~3h25m | < 3h | stage3 砍掉应有时间收益 |

**关键新增**：EC-4 long-tail / head 分桶 MRR。这两个分桶值在 R2 没有显式计算过，R3 训练前必须先在 R2 checkpoint 上跑出 baseline，否则 R3 没法验证"class-balanced 是否真的提升了 long-tail"。

### 2.5 R3 训练前的额外一步：R2 baseline 分桶补算

在动 R3 训练之前，**先在 R2 checkpoint 上补算 EC-4 long-tail / head bucket 的 MRR**。这一步零 GPU，复用 R2 已保存的 `embeddings_v3.npz`。

```python
# 伪代码（Codex 实现到一个新的 evaluation 脚本，不动 train.py）
# 路径建议: code/demo/eval_ec4_buckets.py
buckets = {
    "tail": [g for g, n in ec4_counts.items() if n <= 4],
    "mid":  [g for g, n in ec4_counts.items() if 4 < n <= 317],
    "head": [g for g, n in ec4_counts.items() if n > 317],
}

for name, group_list in buckets.items():
    row_idx = [i for i, r in enumerate(metadata)
               if parse_ec4_strict(r["ec_number"]) in group_list]
    sub_mrr = evaluate_grouped_re_subset(
        all_r, all_e, metadata, subset=row_idx)
    print(f"{name}: n_rows={len(row_idx)}, EC-4 MRR={sub_mrr:.4f}")
```

输出三行（tail / mid / head），写入 `R2_EC4_BUCKET_BASELINE.md`，作为 R3 验收的对照基准。

### 2.6 不要做的事（硬约束）

1. ❌ 不要改 `hard_neg_weight`（R2 已验证）
2. ❌ 不要改 EC encoding（dead code，无效改动）
3. ❌ 不要在 train.py 加 calibration / OOD / latency 阈值（agent 未开发）
4. ❌ 不要再追 row-level R→E MRR（ceiling-bound）
5. ❌ 不要修复 visualize_four_modal 的 AGG bug（在外层 try/except 兜底即可，写在 R3 中是合理工程修复）
6. ❌ 不要顺便重构 train.py 其他部分（变更面尽量小，便于归因）
7. ❌ 不要换 hardware / partition / Python 环境（与 R2 保持一致）

---

## 3. R3 训练后交付物 checklist（逐项打勾）

### 3.1 R2 收尾文档（zero-GPU，先做）
- [ ] `R2_POSTMORTEM_20260615_FINAL.md`：在 1.6.1 / 1.6.2 末尾追加 §1.1 calibration shape + §1.2 real OOD candidate sources
- [ ] `R2_EC4_BUCKET_BASELINE.md`：tail / mid / head 三个分桶的 EC-4 MRR baseline
- [ ] `eval_ec4_buckets.py`：新建脚本，docstring + `--checkpoint` 参数 + 单独可运行

### 3.2 R3 训练前的代码改动
- [ ] `train.py` 加 EC-4 class-balanced `WeightedRandomSampler`（仅 stage1/2 dataloader）
- [ ] `train.py` 加 `epochs_stage3 == 0` 时跳过 stage3 的逻辑
- [ ] `train.py` 在 `visualize_four_modal()` 调用外加 `try/except`，optional PNG 失败时不影响 exit code
- [ ] `Config` 中默认 `epochs_stage3 = 0`，其他超参与 R2 一致
- [ ] 改动 diff 不超过 ~80 行（class-balanced sampler + stage3 skip + try/except）

### 3.3 R3 训练产物
- [ ] `model_v3.pt` + `model_v3_stage{0,1,2}.pt`（stage3 按 2.2 变更 2 处理）
- [ ] `embeddings_v3.npz`（4 模态，shape `(N, 256)`）
- [ ] `metrics_v3.json`（含 row + grouped + E→M + S→M）
- [ ] `metadata_v3.json` / `training_history.json` / `r3_config.txt`
- [ ] `*_nn_index.npz`（4 个）

### 3.4 R3 评估文档
- [ ] `R3_RESULT_SUMMARY_<时间戳>.md`：含 §1 HPC artifacts / §2 core artifacts / §3 R3 metrics（与 R2 同表对照）
- [ ] §3 必须含分桶表：tail / mid / head EC-4 MRR vs R2 baseline
- [ ] §4 验收判定表：按 2.4 表逐行 PASS/FAIL
- [ ] §5 stage-wise checkpoint MRR 演化（用 `postmortem_eval_stage_checkpoints.py` 跑 stage0/1/2）
- [ ] §6 训练时间对比（R3 vs R2，期望 −15%~−25%）

### 3.5 R3 后续决策输入
- [ ] `R4_DECISION_INPUT.md`（1 页内，沿用 R3_DECISION_INPUT 格式）：
  - 如果 R3 PASS：列出"是否进入 agent 接入阶段"vs"是否还有数据/特征侧空间"两个候选
  - 如果 R3 FAIL：列出失败模式（long-tail 没提升 / head 退化 / 总体退化）+ 候选回滚方向
- [ ] **不写 R4 plan 细节**

---

## 4. 一句话给学生的总结

R2 postmortem 全部通过，老师拍板进入 R3。

R3 主要变更只有 **3 处**：
1. EC-4 class-balanced sampling（`1/sqrt(group_size)` 软化长尾）
2. 砍 stage3（`epochs_stage3 = 0`，stage3 在所有指标上要么不变要么微退）
3. visualize_four_modal 加 try/except 工程兜底

启动 R3 之前，**必须先做两件零 GPU 的小事**：
- 在 R2 postmortem 末尾追加 calibration shape + real OOD candidate sources（1.1 / 1.2 段）
- 在 R2 checkpoint 上补算 EC-4 tail/mid/head 分桶 MRR baseline（不补这个 R3 没法验收）

R3 判定看 6 行表（2.4 节），核心是 **EC-4-grouped R→E MRR 不退化 + EC-4 long-tail bucket MRR 提升 ≥ 5%**。calibration / OOD / latency 不进 R3 验收，等 agent 接入再说。

R3 完成后只交 1 页 `R4_DECISION_INPUT.md`，**不要写 R4 plan**。

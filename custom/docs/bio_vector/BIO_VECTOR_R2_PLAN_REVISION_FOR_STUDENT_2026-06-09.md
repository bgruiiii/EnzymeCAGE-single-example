# Bio Vector R2 Plan 修订指令 — For Student Codex

Date: 2026-06-09
Audience: 学生原文喂给 Codex / 任意 LLM 助手

---

## 背景与定位

老师审过了 `R2_PLAN_v2_20260609_093107.md` 与诊断报告
`R1_DIAGNOSIS_20260608_172625.md`。R2 计划方向**可批准**，但提交 HPC job
前需要做 **3 处文档修订** + **1 项零成本补充诊断**。

R2 训练配置本身（`hard_neg_weight=1.0`、`epochs_stage1=25`）保留，
不要再增减新变量。

---

## 关键发现：R1 不是"灾难性失败"

诊断结果显示：

| 指标 | 数值 | 含义 |
|---|---:|---|
| row-level R→E MRR | 0.0575 | 行级精确对齐很差 |
| UniProt-grouped R→E MRR | 0.0581 | UniProt 级精确对齐也很差 |
| **EC-2-grouped R→E MRR** | **0.9340** | **EC 亚类级对齐近乎完美** |
| same-EC-1-digit per-anchor mean | 643.96 | hard-neg 污染严重 |

**这意味着**：模型在 EC-2 family 粒度上学得**极好**，只是无法在
"同 EC-2 内的不同酶"之间做精细区分。**这不是训练失败，是 hard negative
信号粒度（EC-1）和评估粒度（row/UniProt）严重错位的结果**。

R2 plan §10 当前写的 "R1 exact R→E alignment failed" 框架不准确，
必须重写。

---

## Task 0: 提交 R2 前的零成本补充诊断

在改 R2 plan 之前，先用 R1 现有 embedding 补算一个指标。**不需要重训，
不需要 GPU，10 分钟内完成**。

### 0.1 补算 EC-4-grouped R→E MRR

EC-4 是真正的"酶功能身份"——同 EC-4 的酶催化同一反应，只是物种/序列不同。
这比 UniProt-grouped 更生物学相关。

实现：扩展 `diagnose_round1_postmortem.py` 的 grouped MRR 函数，
新增 EC-4-grouped 口径：

```python
# 把 metadata 中的 ec_number 解析为完整四级编号
# 同一 EC-4 的所有 row 视为正样本集合
# rank = 检索结果中第一个匹配 EC-4 的位置
```

输出新增一行到诊断表：

```text
| EC-4-grouped | top-1 | top-5 | top-10 | MRR |
| EC-3-grouped | top-1 | top-5 | top-10 | MRR |  # 可选
```

### 0.2 EC-4-grouped MRR 数值的判读

| EC-4-grouped MRR 数值 | 含义 | 对 R2 的影响 |
|---|---|---|
| **≥ 0.50** | 模型在功能身份层级已对齐良好 | R1 实际上是成功的，R2 优化目标改为提升 row-level |
| **0.20 - 0.50** | 部分功能身份学到，仍有提升空间 | R2 按当前配置跑，目标 row-level > 0.12 |
| **< 0.20** | 功能身份对齐确实有问题 | R2 按当前配置跑，重点观察 stage1 epoch 增加的效果 |

### 0.3 提交物

更新两份文件：

```text
diagnose_round1_postmortem.py    # 加 EC-4 grouped 实现
R1_DIAGNOSIS_20260608_172625.md  # 在原表后追加 EC-4 grouped 行
```

---

## Task 1: 修订 R2_PLAN_v2 文档（3 处）

### 1.1 删除 EC encoding 改动

**原因**：当 `hard_neg_weight = 1.0` 时，[`infonce_loss`](train.py)
中的 `weights = 1.0 + (hard_weight - 1.0) * same_ec * (1 - eye)`
退化为 `weights = 1.0`，**`same_ec` 矩阵完全不被使用**。
此时改 EC 编码粒度（1-digit → 2-digit）是 dead code，**不会影响训练**，
但会引入第三个变量妨碍后续归因。

**修改位置**：`R2_PLAN_v2_20260609_093107.md`

- §4 删除 `EC encoding = first two EC digits` 行
- §6 整段删除 §6.2 "EC Encoding Change"
- §10 Summary 删除 "changes EC label granularity from first digit to first
  two digits" 一行

EC-2 编码改动留作 **R3 ablation 的备选项**，不在 R2 范围。

### 1.2 重写 R1 叙事

**修改位置**：`R2_PLAN_v2_20260609_093107.md` §3、§10

**原文（不准确）**：

> R1 failed on exact R→E alignment because two effects interacted...
> R1 exact R→E alignment failed because first-digit EC hard-negative mining
> was too coarse...

**改为**：

```markdown
## 3. Working Hypothesis (Revised)

R1 训练在 EC-2 family 粒度上对齐良好（EC-2-grouped R→E MRR = 0.934）,
但在 row-level / UniProt-level 精确对齐上表现差（0.058）。诊断显示这并非
训练崩溃，而是两个因素叠加的结果：

1. H1: hard-negative mining 粒度过粗
   - same-EC-1-digit per-anchor mean = 643.96 (>500 阈值)
   - hard_neg_weight=2.0 把生物学相关的同 EC-1 酶错误推远
   - 模型被迫在 EC-2 family 内"挤压"区分能力

2. H3: Stage 1 全量数据下欠训
   - Stage 1 iterations=420 vs demo 200，数据多 485x，iter 仅多 2.1x
   - 即使移除 hard-neg 干扰，也需要更多优化步数让 row-level 对齐收敛

R2 同时移除 hard-neg 信号污染并增加 Stage 1 训练量，
预期在保持 EC-2 grouped 高分的前提下提升 row-level / UniProt-level MRR。
```

`§10 Summary For Teacher` 同样重写，删除 "exact R→E alignment failed"
这种描述。

### 1.3 加入量化通过/失败门槛表

**修改位置**：`R2_PLAN_v2_20260609_093107.md` §7 末尾追加表格

替换原 §7 的定性描述，加入数字门槛：

```markdown
## 7. Expected Outcomes (Quantified)

| 指标 | R1 实测 | R2 通过门槛 | R2 警示阈值 |
|---|---:|---:|---:|
| row-level R→E MRR | 0.0575 | > 0.12 | < 0.06 → 失败 |
| UniProt-grouped R→E MRR | 0.0581 | > 0.15 | < 0.08 → 失败 |
| EC-4-grouped R→E MRR | (Task 0 补算) | 不低于 R1 - 0.05 | 大幅下降 → 警示 |
| EC-2-grouped R→E MRR | 0.9340 | 不低于 0.85 | < 0.70 → 严重退化 |
| E→M MRR | 0.609 | 不低于 0.55 | < 0.40 → 失败 |

通过判定：row-level + UniProt-grouped 至少一项达通过门槛 + EC-2/EC-4
grouped 不出现警示阈值 → R2 视为成功。
```

---

## Task 2: 扩展 R2 评估口径

**修改位置**：`R2_PLAN_v2_20260609_093107.md` §6.4

R2 跑完后报告必须含至少 4 套 grouped MRR：

```text
- row-level R→E MRR
- UniProt-grouped R→E MRR  
- EC-4-grouped R→E MRR     ← 新增
- EC-2-grouped R→E MRR
- (可选) EC-3-grouped R→E MRR
```

在 `train.py` 的 `_chunked_retrieval` 之后增加多粒度 grouped 评估函数，
复用 R1 诊断脚本里写好的实现。

不需要新增训练代码，只需要在评估阶段多算几个口径。

---

## Task 3: 不要做的事

1. ❌ 不要在 Task 0 补算 EC-4-grouped 之前提交 R2 训练
2. ❌ 不要保留 EC encoding 改动（在 hard_neg=1.0 下是 dead code）
3. ❌ 不要在 R2 增加任何超出 `hard_neg_weight=1.0 + epochs_stage1=25`
   范围的训练改动
4. ❌ 不要在文档里继续写 "R1 catastrophically failed" 或
   "exact alignment failed"——数据不支持这个结论
5. ❌ 不要把 README 推荐参数范围当硬约束——R1 已证明范围内也会出问题

---

## Task 4: R3 路线图预告（不在本次范围）

如果 R2 row-level MRR 仍 < 0.15，R3 候选方向（仅记录，不执行）：

1. **Coarse-to-fine fine-tune**：在 R2 checkpoint 上重启训练，
   开 `hard_neg_weight=2.0` + EC 编码改 2-digit 或 4-digit，
   仅训 Stage 1 + Stage 2，跳过 Stage 0
2. **EC-4 多任务辅助头**：加一个 EC-4 分类辅助 loss，
   `Loss = L_infonce + 0.1 * L_ce(EC-4)`
3. **去重采样**：DataLoader 层面禁止同 batch 含同 UniProt 多行

R3 必须做单变量 ablation 隔离 R2 中 hard_neg vs stage1 epochs 的贡献。

---

## 输出物清单（按顺序提交）

1. **Task 0 产物**：
   - 更新后的 `diagnose_round1_postmortem.py`（含 EC-4 grouped）
   - 更新后的 `R1_DIAGNOSIS_20260608_172625.md`（含 EC-4 行）

2. **Task 1+2 产物**：
   - 修订后的 `R2_PLAN_v2_<新时间戳>.md`（含 §3 重写、§4 删 EC、§7 量化门槛、
     §6.4 多粒度评估）

3. **老师审批通过后**：
   - 提交 R2 训练 job
   - R2 完成后，按 §7 量化门槛表逐项判定通过/警示

---

## 一句话给学生的总结

R1 诊断发现"训练失败"其实是评估口径错位，模型在 EC-2 family 粒度上学得
极好。R2 配置不变（关 hard-neg + 多训 Stage 1 + 删 EC encoding 改动），
但提交前先补算 EC-4-grouped MRR、改写 R1 叙事、加量化通过门槛。三件事
都不消耗 GPU。

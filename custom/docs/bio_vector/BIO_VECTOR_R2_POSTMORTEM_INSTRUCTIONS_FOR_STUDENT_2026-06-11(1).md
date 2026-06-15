# Bio Vector R2 Postmortem 补充指令 — For Student Codex

Date: 2026-06-11
Audience: 学生原文喂给 Codex / 任意 LLM 助手

---

## 背景与定位

R2 训练已完成（`hard_neg_weight=1.0, epochs_stage1=25`），结果文件
`BIO_VECTOR_R2_RESULT_SUMMARY_2026-06-11.md` 已提交。老师审完后给出
两条核心修正：

### 修正 1：R2 不是 "controlled negative result"

学生口头汇报里把 R2 定性为 "controlled negative result：保住了功能
结构但细粒度未解决"——**这个定性不准确**。

真实情况：
- row-level R→E MRR > 0.12 这条通过门槛是上一轮指令里我给的，
  **这条门槛设错了**——没有考虑同 EC-4 / 同 UniProt 的 row 重复结构
- R2 数据反而揭示了一个 ceiling：row-level top-1 / EC-4-grouped top-1
  ≈ 0.0176 / 0.839 ≈ **1/47.6**，说明同一 EC-4 下平均约有 ~47 个 row
  candidate，模型在 reaction 端拿到几乎相同的 DRFP 输入，**信息上不可能**
  在这 47 个 row 里精准选出"那一行"
- 所以 R2 是 "ceiling-hitting result"，不是 "negative result"

### 修正 2：R3 决策不能基于现有数据下结论

学生建议 "下一步请老师判断是否进入 R3"——**先不要进 R3**。

R2 报告漏了 6 块关键信息，必须补全后才能判断 R3 方向。下面六件事
**全部不消耗 GPU**，预计 1-2 小时完成。

---

## 项目定位明确（必读，影响 R3 全部设计）

这个模型不是终端用户直接调用的工具，而是会作为 **tool / verifier**
被上层多智能体系统（pathway design agent / 降解路径推理 agent 等）
程序化调用。典型调用场景：

```text
agent: "丙酮可被反应 R 代谢，请验证哪些 enzyme/microbe 能执行 R"
tool input: reaction (DRFP 或 SMILES)
tool output: {
    ec_4: "1.1.1.1",
    confidence: 0.87,
    candidate_enzymes: [...],
    candidate_microbes: [...],
    abstain: false
}
```

这个定位**直接确定了 R3 的方向是路径 B（R→EC-4 retrieval）**，
原因如下：

| 维度 | row-level R→E | EC-4 grouped R→E |
|---|---|---|
| agent 实际想问的 | "这个具体 UniProt 能催化吗" | "什么 EC 类的酶能催化" |
| 生物学意义 | 同 EC-4 不同 UniProt 算"错"，但其实同功能 | 同功能视为同类，符合代谢路径推理逻辑 |
| 置信度可解释性 | top-k score 噪声大、跨样本不可比 | EC-family score 稳定，可直接喂给 LLM |
| 错误代价 | agent 拿到错 UniProt → 后续路径全错 | agent 拿到 EC family → 仍可继续推理 |

因此 R3 不再以 grouped MRR 为唯一目标，**新增三类 tool-oriented baseline 数据**
（calibration curve / OOD score 分布 / latency 分布）。**这些只采集 baseline 数字，
不设硬阈值**——agent 还未开发，真实需求未知，现阶段设阈值是 premature
optimization。阈值由 agent 接入时再定。

---

## R2 实质成果（重新定性）

R2 真正的科学成果应该写成：

```text
1. 关闭 hard_neg_weight=2.0 是无害且略有正向效应的
   - row-level R→E MRR: 0.0575 → 0.0602 (+4.7%)
   - E→M MRR: 0.609 → 0.620 (+1.8%)
   - EC-2/EC-4 grouped 几乎不变（-0.4% / -0.6%）
   - 结论：EC-1 hard-neg 加权可永久关闭

2. EC-family 对齐能力鲁棒
   - 不依赖 hard-neg 信号即可达到 EC-2 grouped 0.93 / EC-4 grouped 0.91
   - 监督粒度由 InfoNCE 自身的 batch 分布隐含决定，无需显式加权

3. 行级评估口径需要重新设计
   - 同 EC-4 平均 ~47 row → row-level R→E top-1 数学上限 ≈ 0.021
   - 现有 row-level / UniProt-level 门槛在数据结构上不可达
   - 后续应以 EC-4-grouped 为主指标，row-level 仅作参考
```

---

## Task 1: 补五项诊断（零 GPU）

### Task 1.1 数据结构基线统计（5 分钟）

用 metadata 算清楚同类 row 的分布，写入 `R2_POSTMORTEM_<时间戳>.md`：

```python
# pseudo-code
for grouping in ["uniprot", "ec_2", "ec_3", "ec_4"]:
    counts = group_counts(metadata, by=grouping)
    print(f"{grouping}: mean={counts.mean()}, median={counts.median()}, "
          f"max={counts.max()}, p95={counts.quantile(0.95)}")
```

输出表格：

```text
| Grouping | mean rows | median | max | p95 | unique groups |
| UniProt  |   ?       |   ?    |  ?  |  ?  |    ?          |
| EC-4     |   ?       |   ?    |  ?  |  ?  |    ?          |
| EC-3     |   ?       |   ?    |  ?  |  ?  |    ?          |
| EC-2     |   ?       |   ?    |  ?  |  ?  |    ?          |
```

并基于 mean rows 算出每种 grouping 下 row-level top-K 的**理论上限**：

```text
row-level top-K ceiling ≈ K / mean_rows_per_group  (when K < mean_rows)
                       ≈ 1.0                       (when K >= mean_rows)
```

### Task 1.2 评估 stage-end checkpoints（30 分钟）

R2 plan §6.3 承诺过保存 `model_v3_stage{0,1,2,3}.pt`，但 R2 报告完全没
展示 stage-wise MRR 演化。这是 R3 决策最关键的依据。

实现：复用现有 `_chunked_retrieval` + grouped MRR 函数，依次加载四个
stage checkpoint 评估，输出表格：

```text
| Stage     | row R→E MRR | UniProt-grouped | EC-4-grouped | E→M MRR |
| stage0终点 |       ?     |        ?        |       ?      |    ?    |
| stage1终点 |       ?     |        ?        |       ?      |    ?    |
| stage2终点 |       ?     |        ?        |       ?      |    ?    |
| stage3终点 |   0.0602    |     0.0607      |    0.9132    |  0.620  |
```

**判读规则**：

- 若 stage1 终点 row-level MRR ≈ 0.05-0.06，stage2/3 几乎不贡献
  → **真正瓶颈在 stage1**，stage2/3 是浪费算力，R3 应砍掉它们
- 若 stage1 终点 ≈ 0.03，stage2/3 各贡献 ~0.015
  → 三阶段都有用但增长缓慢，问题在 reaction-side 信号不足
- 若 EC-4 grouped 在 stage0 终点已 > 0.85，后续几乎不变
  → 独立预训练已学到大部分功能结构，对比阶段只在做 row-level fine-tune

### Task 1.3 双变量贡献的间接归因

R2 同时改了 `hard_neg=2→1` 和 `stage1=12→25`，无法直接归因。
但可借助 stage1 终点 checkpoint 间接推断：

```text
对比维度                            | R1 stage1终点 | R2 stage1终点
row-level R→E MRR                  |       ?       |       ?
EC-4-grouped R→E MRR               |       ?       |       ?
```

如 R1 没保存 stage1 checkpoint，则跳过此项，并在 R3 plan 里强制要求
**任何新一轮训练必须保存所有 stage checkpoint**。

### Task 1.4 诊断 visualize_four_modal 错误

错误信息：

```text
ValueError: object __array__ method not producing an array
```

这个错误**不能简单 try/except 跳过**。它暗示某个 array 实际是 list 或
异构对象。提交以下三项：

1. **完整 traceback**：定位是哪一行 `fig.savefig` 在做什么 array 转换
2. **embedding 类型断言**：在保存 `embeddings_v3.npz` 时打印
   `type(arr)`、`arr.dtype`、`arr.shape`，确认四个 modality embedding
   都是 `np.ndarray` 且 dtype 一致
3. **NN index 健康检查**：随机抽 100 行做 R→E 检索，验证 top-1 命中率
   与 metrics 一致——若不一致，说明 NN index 构建静默出错

**只有在三项都 PASS 后**，才允许在 visualize 函数外面包 try/except 作为
工程兜底。

### Task 1.5 S→M MRR 加入判定表

R2 报告 §3 列了 `S→M MRR = 0.587`，但 §4 通过门槛表里漏掉。补上：

```text
| 指标 | R1 | R2 | 备注 |
| S→M MRR | ? | 0.587 | 监控不退化即可 |
```

如 R1 没记录 S→M 数字，从 R1 报告里查；查不到就在 postmortem 里
说明 "R1 missing, R2 baseline = 0.587"。

### Task 1.6 Tool-oriented baseline 数据采集（只采集不设阈值）

**需求背景**：agent 尚未开发，真实调用分布/QPS/容错需求都未知。现阶段
设任何具体阈值都是 premature。这一 Task 只做三件事：

1. 采集 baseline 数据，留档
2. 跳通方法论，处理代码复用
3. **不设阈值、不作为 R3 达标验收指标**

structured output schema 本轮不做——这是 wrapper 层工程，schema 字段应
由 agent 端定，等 agent 接口需求明确后再写。

#### 1.6.1 Calibration curve（只采集 confidence vs hit rate）

把 R→E retrieval 的 top-1 cosine similarity 分桶（如 10 个 bin），
对每个桶计算实际 EC-4 hit rate：

```text
| confidence bin | mean score | EC-4 hit rate | sample count |
| 0.0-0.1        |    ?       |       ?       |       ?      |
| ...            |    ?       |       ?       |       ?      |
| 0.9-1.0        |    ?       |       ?       |       ?      |
```

同时计算 ECE 数值（作为 baseline，不设达标线）。agent 接入时可直接拿这张表
定 confidence filter 阈值。

#### 1.6.2 OOD score 分布（只跑方法，不设拒答率阈值）

在 metadata 里随机抽 5% 作为 OOD-like 测试集（或用合成 reaction 扰动），
报告：

- in-distribution top-1 score 分布（mean / p50 / p95 / p99）
- OOD-like top-1 score 分布（mean / p50 / p95 / p99）
- 两者分布重叠度可视化（histogram）

**不设拒答阈值**，拒答机制留到 agent 端需求明确后设计。

#### 1.6.3 Latency 分布（只测，不设目标）

单 query 端到端耗时（reaction → embedding → NN search）：

- 测 100 query 的 p50 / p95 / p99
- 报告当前硬件环境（CPU/GPU 型号、batch、是否预计算 enzyme cache）

**不设 p95 达标线**。agent 调用 QPS 要求明确后再决定是否需要 FAISS
与 embedding cache。

---

## Task 2: 改写 R2 报告 §4 §5 的措辞

修改 `BIO_VECTOR_R2_RESULT_SUMMARY_2026-06-11.md`：

### 2.1 §4 Threshold Judgment 改写

**原文**：
```text
R2 does not meet the approved success rule.
```

**改为**：
```text
R2 触发了通过门槛设计的反思：
- row-level R→E MRR > 0.12 在同 EC-4 平均 ~47 row 的数据结构下数学
  不可达（理论上限 ≈ 0.021 for top-1）
- UniProt-grouped MRR > 0.15 同样受 ceiling 影响
- 后续评估应以 EC-4-grouped 为主指标，row-level 仅作参考
- 按修订后的口径（EC-4-grouped 不退化 + E→M 不退化），R2 通过
```

### 2.2 §5 Interpretation 改写

**原文**：
```text
removing EC-1 hard-negative upweighting and extending Stage 1 alone
is not enough to solve fine-grained R→E discrimination.
```

**改为**：
```text
- removing EC-1 hard-negative is harmless (+4.7% row-level, +1.8% E→M)
  → hard_neg_weight=2.0 可永久关闭，不需保留为可调超参
- Stage 1 extension's marginal gain → stage1 不是当前主瓶颈
  （需 Task 1.2 的 stage-wise 曲线确认）
- Fine-grained R→E discrimination 受限于 reaction-side 信息上限：
  同 EC-4 不同 UniProt 的 DRFP 输入近乎相同，从 reaction 端无法区分
- 真正的 R3 方向应该是 enzyme-side discriminative signal，
  而非继续调 hard_neg / stage1 epochs
```

### 2.3 §6 Recommended Next Step 改写

**删除**：
```text
Possible R3 directions: fine-grained objective / EC-4 supervised auxiliary
head / coarse-to-fine fine-tuning / sampling masking strategy
```

**改为**：
```text
项目定位已锁定路径 B：模型作为多智能体系统的 tool / verifier，
输出形态为 R→EC-4 retrieval + structured response。

R3 决策延后至 Task 1 六项诊断完成。R3 候选方向需基于诊断结果选择
（不再设 tool-oriented 硬约束，agent 接入时再补）：

| 诊断结论                                                     | R3 优先方向                          |
|-------------------------------------------------------------|-------------------------------------|
| stage1 终点 EC-4 grouped 已 > 0.85                          | 砍掉 stage2/3，节省算力              |
| EC-4 grouped 仍 < 0.95 且 stage 演化未饱和                   | enzyme-side per-residue ESM-C        |
| 数据基线显示 EC-4 标签不平衡（max/min > 100）                 | EC-4 加权采样 + class-balanced loss  |
| Calibration curve 显示 high-confidence bin 出现明显偏差       | 记录但不补丁，等 agent 需求明确再说 |

所有 R3 方向必须满足：
1. 不退化 EC-4 grouped MRR（>= 0.91）
2. 不退化 E→M MRR（>= 0.60）
```

---

## Task 3: 提交物清单

按顺序提交以下四份：

1. **`R2_POSTMORTEM_<时间戳>.md`**（新建）
   - Task 1.1 数据结构基线表 + ceiling 推算
   - Task 1.2 stage-wise MRR 演化表
   - Task 1.3 双变量归因（如可推断）
   - Task 1.4 visualize 错误三项诊断结果
   - Task 1.5 S→M MRR 历史对比
   - Task 1.6 Tool-oriented baseline（calibration curve / OOD score
     分布 / latency p50/p95/p99）——只采集数据，不设阈值

2. **`BIO_VECTOR_R2_RESULT_SUMMARY_2026-06-11.md`**（修订）
   - §4 §5 §6 按 Task 2 改写
   - 删除 "controlled negative result" / "failure" 措辞
   - 加入 ceiling 分析

3. **`postmortem_eval_stage_checkpoints.py`**（新建）
   - Task 1.2 用的脚本，便于 R3 复用

4. **`R3_DECISION_INPUT.md`**（新建，1 页内）
   - 用 Task 1 数据填 Task 2.3 决策表
   - 给出 1-2 个候选 R3 方向 + 理由
   - 不要直接写 R3 plan，等老师选方向

---

## Task 4: 不要做的事

1. ❌ 不要在 Task 1 完成前提 R3 plan
2. ❌ 不要继续把 R2 描述为 "negative result" / "failure"——
   数据不支持这个结论
3. ❌ 不要把 visualize 错误简单 try/except 跳过——先做 1.4 三项诊断
4. ❌ 不要重训 R2（学生自己也写了，确认）
5. ❌ 不要在 R3 候选方向里继续动 hard_neg / stage1 epochs——
   R2 已经证明这两个不是瓶颈
6. ❌ 不要再追求 row-level / UniProt-level R→E MRR——
   项目定位已锁定 R→EC-4 retrieval（tool 输出形态）
7. ❌ 不要在 Task 1.6 里设任何达标阈值（ECE 多少、p95 多少、拒答率多少）——
   agent 未开发，阈值是 premature。只采集 baseline 数据留档即可
8. ❌ 不要现在写 structured output wrapper——schema 由 agent 端定

---

## Task 5: 交付物 checklist（逐项打勾）

下面是交付时必须逐项打勾的明细。任何一项缺失都视为未完成。

### `R2_POSTMORTEM_<时间戳>.md`

**Task 1.1 数据结构基线**
- [ ] 表格含 4 行分组（UniProt / EC-4 / EC-3 / EC-2）
- [ ] 每行 6 列：mean / median / max / p95 / unique groups / row-level top-K ceiling
- [ ] EC-4 grouping 下额外给出 max/min 比（诊断 EC-4 标签不平衡）

**Task 1.2 stage-wise MRR 演化**
- [ ] 表格含 4 行（stage0 / stage1 / stage2 / stage3 终点）
- [ ] 每行 4 列：row R→E MRR / UniProt-grouped / EC-4-grouped / E→M MRR
- [ ] 最后一行 × EC-4 列与 R2 终态 0.9132 一致（完整性检查）

**Task 1.3 双变量归因**
- [ ] 明确声明 R1 stage1 checkpoint 是否存在
- [ ] 如存在，给出 R1 vs R2 stage1 终点对比表
- [ ] 如不存在，明文写"归因不可推断，后续版本必须保存所有 stage checkpoint"

**Task 1.4 visualize 错误三项诊断**
- [ ] 完整 traceback（从 `fig.savefig` 到根因那一行）
- [ ] 四个 modality embedding 的 `type/dtype/shape` 输出表
- [ ] NN index 随机 100 行检查结果（与 metrics 是否一致）
- [ ] 结论明确是 "safe to try/except" 还是 "必须修复静默 bug"

**Task 1.5 S→M MRR 历史对比**
- [ ] R1 vs R2 的 S→M MRR 二行表
- [ ] R1 数据丢失时明文标注 "R1 missing, R2 baseline = 0.587"

**Task 1.6.1 calibration curve**
- [ ] 10-bin 表格（bin / mean score / EC-4 hit rate / sample count）
- [ ] ECE 数值（只作为 baseline，不作为达标判定）
- [ ] 一句话描述曲线形状（如 "high-conf bin 偏高 / 偏低 / 接近对角线"）

**Task 1.6.2 OOD score 分布**
- [ ] in-distribution top-1 score 的 mean/p50/p95/p99
- [ ] OOD-like top-1 score 的 mean/p50/p95/p99
- [ ] histogram 可视化图（PNG 或 inline plot）
- [ ] 一句话描述两分布重叠度

**Task 1.6.3 latency 分布**
- [ ] 100 query 的 p50 / p95 / p99
- [ ] 硬件环境说明（CPU/GPU 型号、batch size、是否预计算 enzyme cache）

### `BIO_VECTOR_R2_RESULT_SUMMARY_2026-06-11.md` 修订
- [ ] §4 Threshold Judgment 按 Task 2.1 改写
- [ ] §5 Interpretation 按 Task 2.2 改写
- [ ] §6 Recommended Next Step 按 Task 2.3 改写
- [ ] 全文搜索确认无 "negative result" / "failure" / "catastrophic" 残留
- [ ] 加入 ceiling 分析段落（同 EC-4 ~47 row, top-1 上限 ≈ 0.021）

### `postmortem_eval_stage_checkpoints.py`
- [ ] 带 docstring 说明输入 / 输出 / 依赖路径
- [ ] 单独可运行（`python postmortem_eval_stage_checkpoints.py --stage 1`）
- [ ] 复用 R2 现有的 chunked retrieval 函数，不重复实现

### `R3_DECISION_INPUT.md`（1 页以内）
- [ ] 填完 Task 2.3 决策表（4 行诊断结论 × R3 优先方向）
- [ ] 明标哪 1-2 个方向被当前诊断数据触发
- [ ] 每个候选方向附 2-3 句理由（引用具体 Task 数字）
- [ ] **不包含 R3 plan 细节**（hyperparameter / epoch / dataset 等）
- [ ] 只交 1 页，超过 1 页表示越界

---

## 一句话给学生的总结

R2 不是 "negative result"，是 "ceiling-hitting + 评估口径错位"。
我之前给的 row-level > 0.12 门槛设错了，没考虑同 EC-4 的 47 倍 row 重复
稀释。

**项目定位已锁定**：模型作为多智能体系统的 tool / verifier，输出形态
为 R→EC-4 retrieval + structured response，路径 B 是唯一方向。

你需要做六件不消耗 GPU 的诊断（数据结构基线、stage 演化、双变量归因、
visualize 错误溯源、S→M 对比、**tool-oriented baseline 采集含
calibration curve / OOD score 分布 / latency 分布**）。

**重要**：Task 1.6 只采集 baseline 数据，不设达标阈值也不写 structured
output wrapper——agent 未开发，现阶段设任何阈值都是 premature optimization。
阈值与 schema 由 agent 接入时再定。

把 R2 报告 §4 §5 改写成 ceiling 分析 + tool 定位补丁，然后给出 R3 决策
候选——但 R3 plan 等老师选方向后再写。

# Bio Vector Round 2 — Codex Prompt for Student

Date: 2026-06-08
Audience: 学生提交给其 Codex / 任意 LLM 助手作为指令使用

---

## 背景

R1 全量数据训练（GVP + ESM-C 两个 baseline）相比 300 demo 出现了量级下降：

- 300 demo: `R→E MRR ≈ 0.74`
- R1 ESM-C 全量: `R→E MRR = 0.058`
- R1 GVP 全量: `R→E MRR = 0.044`
- 21,842 测试样本，random baseline ≈ 0.10 → R→E 比随机还差

R2 不能直接重训。之前提交的 R2 plan（`stage1=15, stage2=10, hard_neg=3.0`）
中 `hard_neg` 升到 3.0 是反向操作，需要先做诊断验证假设。

---

## Task 0: 三个待验证的根因假设

代码审查（`demo/train.py`）发现以下三处嫌疑点，按可能性降序：

### H1: Hard negative mining 粒度过粗
- `infonce_loss` (line 181-199) 用 `same_ec` 矩阵给同 EC 的 batch 内样本加权 2 倍
- `MultiModalDataset.__init__` (line 271-275) 把 EC 编码为只取**第一位数字**：
  `int(str(ec)[0])` → 全数据集仅 1-7 共 7 个类
- batch=4096 时，平均每个 anchor 的 batch 内"同酶类样本"≈ 4096/7 ≈ 585 个
- 这 585 个样本被 `hard_weight=2.0` 加权推开 → 系统性把生物学相关酶推远

### H2: 行级 MRR 评估在重复 UID 数据上系统性低估
- `_chunked_retrieval` (line 783-791) 用 `pos_idx = labels[start:end]`
  即"自己这一行"作为唯一正样本
- 全数据集 145,607 行 / 107,731 unique UniProt → 平均每个 UID 1.35 行
- 同 UniProt 跨行的"正确"检索被算成错误
- 但单这个不足以解释从 0.74 跌到 0.058，是叠加因素

### H3: Stage 1 严重欠训
- Stage 1 iterations = `12 × ⌈145607/4096⌉ ≈ 420`
- 300 demo Stage 1 iterations = `40 × ⌈300/64⌉ ≈ 200`
- 数据多 485 倍但 iterations 仅多 2.1 倍
- Stage 1 R→E 未收敛就进 Stage 2/3，污染下游 alignment

---

## Task 1: 诊断脚本 — 复用 R1 embedding，不重训

请创建 `diagnose_round1_postmortem.py`，加载 R1 ESM-C 全量产出的
`embeddings_v3.npz` 和 `metadata_v3.json`，输出以下三组指标到
`R1_DIAGNOSIS_<timestamp>.md`：

### 1.1 Grouped MRR（验证 H2）

对 R→E 检索：

- 行级 MRR（baseline，应等于 R1 报告的 0.058）
- **UniProt-grouped MRR**：把同 UniProt 所有行视为正样本集合，
  rank = 第一个正样本的位置
- **EC-2-digit-grouped MRR**：同前两位 EC（如 `"1.1.x.x"`）视为正样本
- 报告三个指标的 top-1, top-5, top-10, MRR

判读规则：

- 若 UniProt-grouped R→E MRR ≥ 0.30 → H2 是主因，R1 训练其实没问题
- 若 UniProt-grouped R→E MRR 仍 < 0.10 → H2 是次因，问题在训练

### 1.2 Hard negative 污染量化（验证 H1）

扫描 train metadata，模拟 batch=4096 + seed=42 的随机采样：

- 采样 100 个 batch，统计每个 batch 中：
  - 平均 same-EC-1-digit pair 数（含自身对角）
  - 平均 same-EC-2-digit pair 数
  - 平均 same-UniProt pair 数
- 输出 mean / median / max / 分布直方图

判读规则：

- 若 same-EC-1-digit pair 数 > 500 → H1 严重，`hard_neg=2.0` 已过强
- 若 same-UniProt pair 数 > 5 → batch 内有"假负样本"，需 mask 或去重采样

### 1.3 Per-stage embedding 质量（验证 H3，可选）

若 R1 仅保留终态 checkpoint 则跳过此项。
若有阶段中间 checkpoint，分别评估 Stage 1 终点 / Stage 2 终点 / Stage 3 终点
的 R→E MRR，看是否 Stage 2/3 反而拉低了 Stage 1 的成果。

---

## Task 2: 根据诊断结果决定 R2 配置

执行 Task 1 后，按以下决策树选择 R2 配置：

```
if UniProt-grouped R→E MRR ≥ 0.30:
    # H2 是主因，R1 实际训得不差，问题在评估口径
    → R2 不重训，用现有 embedding 把 grouped metric 列为主指标重写报告
    → 后续如需重训，仅做 hard_neg ablation（见下）

elif same-EC-1-digit pair > 500 AND grouped MRR < 0.20:
    # H1 + H3 联合，hard_neg 是污染源
    → R2 配置：hard_neg_weight = 1.0, epochs_stage1 = 25
    → 同时把 EC 编码改为前两位（粒度细化）
    → 不要同时改 stage2，保持 stage2 = 8

else:
    # 仅 H3 主导（数据复杂度本身需要更多 iter）
    → R2 配置：hard_neg_weight = 2.0（不变），epochs_stage1 = 30
```

---

## Task 3: R2 跑前必须满足的实验设计约束

无论选哪条路径，R2 必须遵守：

1. **单变量原则**：相对 R1 仅改 1-2 个变量，且明确写出 hypothesis：
   "我假设 X 是瓶颈，所以改 Y，预期效果 Z"

2. **新增 grouped metrics 作为主指标**，行级 MRR 保留为参考。
   不要再用 row-level top-k 作为唯一汇报标准。

3. **保存 stage 中间 checkpoint**：
   在 `train_four_stages` 末尾增加每个 stage 结束后的 checkpoint dump，
   命名 `model_v3_stage{N}.pt`，方便后续归因。

4. **R2 完成后报告必须包含**：
   - 行级 + UniProt-grouped + EC-2-digit-grouped 三套 MRR 对比表
   - per-stage R→E MRR 演化曲线
   - 与 R1 同口径对比，明确"哪个修改贡献了多少"

---

## Task 4: 不要做的事

1. ❌ 不要在 Task 1 完成前提交任何重训 job
2. ❌ 不要继续把 `hard_neg_weight` 升到 3.0 — 已被代码审查否决
3. ❌ 不要同时改 `stage1`, `stage2`, `hard_neg` 三个变量 — 无法归因
4. ❌ 不要把 README 的"全量推荐范围"当硬约束 —
   R1 失败说明范围本身有问题，需要走出范围验证
5. ❌ 不要用 "remains weak" 这种软口径描述 R1 —
   R→E 比随机还差是 critical regression，必须明确写

---

## 输出物清单（按顺序提交）

1. `diagnose_round1_postmortem.py` — 诊断脚本（不需重训）
2. `R1_DIAGNOSIS_<timestamp>.md` — 诊断结果 + 决策树触发的路径
3. `R2_PLAN_v2_<timestamp>.md` — 基于诊断的 R2 计划，含明确 hypothesis
4. （等老师确认 R2 plan 后再）提交 R2 训练 job

提醒：Task 1 + 2 + 3 应在不消耗 DCU 训练时长的情况下完成，
仅 Task 4 才需要排队提交 HPC job。

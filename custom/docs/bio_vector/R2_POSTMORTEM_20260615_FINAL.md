# Bio Vector R2 Postmortem

## Task 1.1 Data-Structure Baseline And Row-Level Ceiling

### 运行环境

```
metadata_path: /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/metadata_v3.json
运行时间: 2026-06-12
Python: /public/home/acfbwjsi7s/miniconda3/envs/nis/bin/python (3.9.23)
GPU: 未使用
sbatch: 未执行
train.py: 未修改
```

### 分析脚本

```python
import json
import numpy as np
from collections import Counter, defaultdict

META = "/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/" \
       "r2_esmc_hardneg1_stage1_25_2026-06-11/metadata_v3.json"

with open(META) as f:
    meta = json.load(f)

total_rows = len(meta)


def _parse_ec(ec_str, n_levels):
    """Parse EC number to n_levels with numeric validation.
    Returns None for empty / unknown / malformed EC."""
    if not ec_str or ec_str.strip() in ("", "unknown", "-"):
        return None
    parts = ec_str.strip().split(".")
    if len(parts) < n_levels:
        return None
    for i in range(n_levels):
        try:
            int(parts[i])
        except ValueError:
            return None
    return ".".join(parts[:n_levels])


# ── Grouping counters ──
uniprot_groups = defaultdict(int)
ec2_groups = defaultdict(int)
ec3_groups = defaultdict(int)
ec4_groups = defaultdict(int)

for row in meta:
    uid = row.get("uniprot_id", "").strip()
    ec = row.get("ec_number", "").strip()

    if uid:
        uniprot_groups[uid] += 1

    ec2 = _parse_ec(ec, 2)
    if ec2:
        ec2_groups[ec2] += 1

    ec3 = _parse_ec(ec, 3)
    if ec3:
        ec3_groups[ec3] += 1

    ec4 = _parse_ec(ec, 4)
    if ec4:
        ec4_groups[ec4] += 1


# ── Compute stats & print table ──
print("| Grouping | valid rows | mean rows | median | max | p95 "
      "| unique groups | top-1 ceiling | top-5 ceiling | top-10 ceiling |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for name, groups in [("UniProt", uniprot_groups),
                     ("EC-2", ec2_groups),
                     ("EC-3", ec3_groups),
                     ("EC-4", ec4_groups)]:
    sizes = np.array(list(groups.values()))
    valid_rows = int(sizes.sum())
    n_groups = len(groups)
    mean_rows = sizes.mean()                     # ← corrected: counts.mean()
    median_rows = float(np.median(sizes))
    max_rows = int(sizes.max())
    p95 = int(np.percentile(sizes, 95))

    top1_ceil = min(1.0, 1.0 / mean_rows)
    top5_ceil = min(1.0, 5.0 / mean_rows)
    top10_ceil = min(1.0, 10.0 / mean_rows)

    print(f"| {name} | {valid_rows} | {mean_rows:.2f} "
          f"| {median_rows:.0f} | {max_rows} | {p95} "
          f"| {n_groups} | {top1_ceil:.6f} "
          f"| {top5_ceil:.6f} | {top10_ceil:.6f} |")


# ── EC-4 imbalance ──
ec4_sizes = np.array(list(ec4_groups.values()))
ec4_min = int(ec4_sizes.min())
ec4_max = int(ec4_sizes.max())
ec4_ratio = ec4_max / ec4_min if ec4_min > 0 else float("inf")

print()
print(f"EC-4 min rows per group: {ec4_min}")
print(f"EC-4 max rows per group: {ec4_max}")
print(f"EC-4 max/min ratio: {ec4_ratio:.1f}")
```

### EC 原始标签分布

| EC 状态 | 行数 | 说明 |
|---|---:|---|
| 可解析到 EC-2 / EC-3 | 130,635 | 至少前两段（EC-2）或前三段（EC-3）为有效数字，可用于 EC-2/EC-3 grouping |
| 可解析到 EC-4 | 127,847 | 前四段均为有效数字，可用于 EC-4 grouping |
| 有 EC 但不足 4 段 | 2,788 | 可进入 EC-2/EC-3，但段数不足 4 段，不能进入 EC-4 |
| 空值 / 缺失 EC | 14,972 | `ec_number` 为空字符串，无法解析到任何 EC 级别 |
| EC-4 excluded total | 17,760 | 14,972 + 2,788 |

> 注：metadata 中不存在 `"unknown"` 字符串。14,972 条记录的 `ec_number` 字段为空字符串 `""`；另外 2,788 条记录虽有 EC 编号（前 2-3 段有效），但段数不足 4 段，因此被 EC-4 grouping 排除但仍可进入 EC-2/EC-3。

### Grouping 基线统计

| Grouping | valid rows | mean rows | median | max | p95 | unique groups | top-1 ceiling | top-5 ceiling | top-10 ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UniProt | 145,607 | 1.35 | 1 | 41 | 3 | 107,731 | 0.739875 | 1.000000 | 1.000000 |
| EC-2 | 130,635 | 1979.32 | 634 | 17,016 | 9,555 | 66 | 0.000505 | 0.002526 | 0.005052 |
| EC-3 | 130,635 | 698.58 | 81 | 9,018 | 3,569 | 187 | 0.001431 | 0.007157 | 0.014315 |
| EC-4 | 127,847 | 50.65 | 4 | 2,291 | 317 | 2,524 | 0.019742 | 0.098712 | 0.197423 |

> **valid rows** 说明：UniProt 包含全部 145,607 行；EC-2/EC-3 包含可解析到对应级别的 130,635 行（排除 14,972 条空 EC）；EC-4 仅包含 127,847 行（另有 2,788 条 EC 编号段数不足 4 段，被排除）。
>
> **mean rows 计算**：`mean_rows = sizes.mean()`，即对 valid groups 的 group size 数组求均值。不使用 `total_rows / n_groups`（后者会把 invalid rows 的份额也摊到每个 group 上，导致高估）。
>
> **ceiling 含义**：如果模型只能识别到 group（即 EC 级别），但无法区分 group 内的 exact row，则 row-level top-K 的组内理论上限约为 K / mean_group_size。这不是 random baseline，而是 group 粒度检索的信息论天花板。

### EC-4 标签不平衡诊断

| Metric | Value |
|---|---:|
| EC-4 valid groups | 2,524 |
| EC-4 valid rows | 127,847 |
| EC-4 min rows per group | 1 |
| EC-4 max rows per group | 2,291 |
| EC-4 median rows per group | 4 |
| EC-4 mean rows per group | 50.65 |
| EC-4 max/min ratio | **2,291.0** |
| EC-4 p95 rows per group | 317 |

> **严重右偏分布**：median = 4 但 mean = 50.65，说明少数 EC-4 类别拥有极多 row（head 类），而大量 EC-4 类别仅有 1-4 个 row（long-tail 类）。max/min ratio = 2291 进一步印证了极端不平衡。

### Interpretation

**1. EC-4 平均每组多少 row？**

EC-4 平均每组 50.65 row（valid rows = 127,847 行 / 2,524 组），但中位数仅为 4。这意味着**一半以上的 EC-4 类别只有 4 个或更少的样本**，而少数高频 EC-4 类别（如 EC 2 transferase 下的常见子类）拉高了均值。分布呈严重的长尾形态。

**2. Row-level top-1 的理论 ceiling 大概是多少？**

- **EC-4 级别**: top-1 ceiling = 1/50.65 ≈ **0.0197**（即 ~2.0%）
- **UniProt 级别**: top-1 ceiling = 1/1.35 ≈ **0.740**（即 74.0%）
- **EC-2 级别**: top-1 ceiling = 1/1979 ≈ **0.00051**（即 ~0.05%）

EC-4 的 row-level top-1 ceiling 仅为 ~2.0%，这意味着即使模型完美地识别了 EC-4 类别，也无法在组内区分 exact row，因此从 145,607 行中精确命中唯一正确行的理论上限只有约 2%。

**3. 这个 ceiling 是否支持「row-level exact retrieval 信息上不可达」的论断？**

**是的，强烈支持。** 核心原因：

- **信息论瓶颈**：训练集中每个 EC-4 类别平均有 ~51 个 row 共享完全相同的 EC-4 标签。仅凭 EC-4 信息，模型无法区分同一组内的不同 row。即使 embedding 完美地将同 EC-4 类别聚在一起，组内的 row 排序本质上接近随机。
- **对比 R2 实测结果**：R2 实测 row-level R→E top-1 = 0.0176，与 EC-4 ceiling 0.0197 非常接近（ceiling 的 89.3%）。这说明模型**已经接近 EC-4 粒度下的信息论极限**，进一步提升 row-level exact top-1 需要从数据结构层面引入更细粒度的区分信号（如 UniProt ID、reaction ID 等）。
- **EC-2/EC-3 ceiling 更低**：EC-2 ceiling 仅 ~0.05%，EC-3 仅 ~0.14%，进一步说明粗粒度 EC 分组下 row-level exact retrieval 在信息论上完全不可行。
- **UniProt 层面的重复 row ceiling 较高**：UniProt 的 top-1 ceiling = 74%，远高于 EC-4 的 2.0%，说明 UniProt-grouped 低分不能由重复 row ceiling 单独解释（group 内 row 极少，区分在信息论上可行）。但项目主目标仍是 EC-4 grouped retrieval，不追逐 UniProt exact retrieval。这解释了为什么 EC-grouped evaluation 分数很高（EC-2 top-1 ≈ 89%）而 row-level 分数极低（top-1 ≈ 1.8%）——两者衡量的根本不是同一件事。

**4. 是否观察到 EC-4 标签不平衡？**

**是的，严重不平衡。** 关键证据：
- max/min ratio = **2291**（最大组 2291 row，最小组仅 1 row）
- median = 4 vs mean = 50.65（均值是中位数的 12.7 倍）
- p95 = 317，说明 95% 的 EC-4 类别行数 ≤ 317，但 head 5% 的类别拥有更多样本

这种长尾分布对训练的影响：
- 高频 EC-4 类别主导 loss，模型倾向于学习 head 类别的特征
- long-tail 类别（≤4 row）的对比学习信号极弱，hard negative mining 的效用也受限
- 但这也意味着：**模型在 long-tail EC-4 上的 EC-grouped 表现可能仍然不错**（因为组内 row 少，random hit 概率反而更高），这与 R2 实测 EC-4 grouped top-1 = 83.9% 是一致的

---

**Task 1.1 status: COMPLETE**
No train.py modification.
No sbatch.
No GPU used.

---

## Task 1.2 Stage-Wise Checkpoint Evaluation

### 运行环境

```
source_report: /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs/R2_POSTMORTEM_TASK_1_2_RUN_RESULT.md
script: /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/postmortem_eval_stage_checkpoints.py
output_dir: /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11
job_id: 115259334
partition: kshdnormal04
node: f13r2n11
state: COMPLETED
elapsed: 05:34:29
exit_code: 0:0
ReqMem: 28552M
MaxRSS: 25,328,396K (~24.2 GB)
train.py: 未修改
retraining: 未执行
R3 plan: 未写
```

资源说明：该 job 的 `AllocTRES` 记录包含 `gres/dcu=1`，这是 `kshdnormal04` 分区 / QOS 资源分配行为；本任务脚本执行的是 CPU checkpoint evaluation，没有进行模型训练，也没有在评估代码中使用 GPU/DCU。后续 CPU-only postmortem job 应优先使用 `kshctest02` 这类 CPU-only 分区，避免不必要的 DCU 资源分配。

### 输出文件检查

| File | Exists | Size | Modified |
|---|---|---:|---|
| `postmortem_stage_eval.json` | Yes | 9,064 bytes | 2026-06-12 21:04:19 |
| `postmortem_eval_115259334.out` | Yes | 4,903 bytes | 2026-06-12 21:04:42 |
| `postmortem_eval_115259334.err` | Yes | 8,590,813 bytes | 2026-06-12 18:34:57 |

`stderr` 仅包含 RDKit `DEPRECATION WARNING: please use MorganGenerator`，共 145,607 行；无 traceback、无错误。

### Stage-Wise MRR Evolution

| Stage | row R->E MRR | UniProt-grouped R->E MRR | EC-4-grouped R->E MRR | E->M MRR |
|---|---:|---:|---:|---:|
| stage0 | 0.000092 | 0.000092 | 0.007326 | 0.000719 |
| stage1 | 0.058593 | 0.059127 | 0.922020 | 0.610441 |
| stage2 | 0.060464 | 0.061094 | 0.926472 | 0.622604 |
| stage3 | 0.060177 | 0.060711 | 0.917949 | 0.621205 |

Full-precision scientific notation from JSON:

| Stage | row R->E MRR | UniProt-grouped R->E MRR | EC-4-grouped R->E MRR | E->M MRR |
|---|---:|---:|---:|---:|
| stage0 | 9.167e-05 | 9.241e-05 | 7.326e-03 | 7.190e-04 |
| stage1 | 5.859e-02 | 5.913e-02 | 9.220e-01 | 6.104e-01 |
| stage2 | 6.046e-02 | 6.109e-02 | 9.265e-01 | 6.226e-01 |
| stage3 | 6.018e-02 | 6.071e-02 | 9.179e-01 | 6.212e-01 |

Evaluation time per stage was approximately 2,100-2,500 seconds (35-42 minutes). One-time data loading took 10,972.5 seconds.

### Stage 3 Consistency Check

Stage3 checkpoint evaluation was compared against R2 final `metrics_v3.json`.

| Metric | stage3 actual | R2 final expected | Relative Error | Result |
|---|---:|---:|---:|---|
| row R->E MRR | 0.060177 | 0.060208 | 0.05% | PASS |
| UniProt-grouped R->E MRR | 0.060711 | 0.060708 | 0.005% | PASS |
| EC-4-grouped R->E MRR | 0.917949 | 0.913213 | 0.52% | PASS |
| E->M MRR | 0.621205 | 0.619546 | 0.27% | PASS |

All relative errors are below the 5% tolerance. Stage3 consistency check: ALL PASS.

Integrity note for the teacher checklist: the independently re-evaluated stage3
EC-4-grouped R->E MRR is 0.917949, while the saved R2 final
`metrics_v3.json` value is 0.913213 (~0.9132). The relative difference is
0.52%, so the final-stage EC-4 check is consistent under the audited 5%
tolerance. The report preserves both numbers rather than overwriting the
independent re-evaluation result.

### Interpretation

1. **row R->E MRR**: stage0 is near zero (0.000092), then jumps to approximately 0.059 at stage1 and remains around 0.060 through stages 2-3. The pairwise training stage accounts for nearly all row-level R->E MRR gain.
2. **EC-4-grouped R->E MRR**: stage0 is near zero (0.007326), then jumps to 0.922020 at stage1 and remains high through stages 2-3 (0.917949-0.926472). EC-4 retrieval capability is already established after stage1.
3. **E->M MRR**: stage0 is near zero (0.000719), stage1 reaches 0.610441, and stage2/3 add only modest refinement (0.622604 / 0.621205).
4. **Stage stability**: metrics are stable from stage1 onward. Stage2 produces a small peak for row R->E, UniProt-grouped R->E, EC-4-grouped R->E, and E->M; stage3 remains close to stage2 and is consistent with the final saved metrics.

This section records diagnostic evidence only. It does not define an R3 plan.

### Task 1.2 Status

| Item | Status |
|---|---|
| Four stage checkpoints evaluated | COMPLETE |
| Stage-wise 4-column MRR table | COMPLETE |
| Stage3 consistency check | PASS |
| train.py modification | None |
| Retraining | None |
| R3 plan | None |

**Task 1.2 status: COMPLETE**
No train.py modification.
No retraining.
No R3 plan.
GPU/DCU note: 1 DCU was allocated by Slurm/QOS on `kshdnormal04`; evaluation code did not use GPU/DCU computation.

---

## Task 1.3 Indirect Attribution of Hard-Negative vs Stage 1

### 运行环境

```
source_report: /home/a/EnzymeCAGE/custom/docs/bio_vector/R2_POSTMORTEM_TASK_1_3_INDIRECT_ATTRIBUTION.md
source_r1_diagnosis: /home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R1_DIAGNOSIS_RESULT_2026-06-09.md
source_r2_stagewise: /home/a/EnzymeCAGE/custom/docs/bio_vector/R2_POSTMORTEM_TASK_1_2_RUN_RESULT.md
运行时间: 2026-06-14
new computation: 未执行
train.py: 未修改
retraining: 未执行
GPU/DCU: 未使用
R3 plan: 未写
```

### Task Question

R2 同时改了两个变量：

| Variable | R1 | R2 |
|---|---:|---:|
| `hard_neg_weight` | 2.0 | 1.0 |
| `epochs_stage1` | 12 | 25 |

Task 1.3 的目标是判断是否能把 R2 变化分别归因到：

1. hard-negative removal (`2.0 -> 1.0`)
2. longer Stage 1 (`12 -> 25`)

老师要求：如果 R1 没有保存 stage1 checkpoint，则不能强行归因，需要明文写出归因不可直接推断，并要求后续归因型训练必须保存 stage checkpoint。

### R1 Stage Checkpoint Availability

R1 ESM-C stage-end checkpoints were not available:

| R1 Artifact | Status |
|---|---|
| `model_v3_stage0.pt` | missing |
| `model_v3_stage1.pt` | missing |
| `model_v3_stage2.pt` | missing |
| `model_v3_stage3.pt` | missing |
| final `model_v3.pt` | available |

Therefore:

```text
R1 stage1 checkpoint not available; direct hard_neg vs stage1 attribution is not
identifiable from saved checkpoints.
```

This is the central limitation of Task 1.3.

### Direct Stage1 Attribution Table

The teacher-requested R1 stage1 vs R2 stage1 comparison cannot be completed because the R1 stage1 endpoint was not saved.

| Comparison Metric | R1 stage1 endpoint | R2 stage1 endpoint | Direct attribution possible? |
|---|---:|---:|---|
| row-level R->E MRR | missing | 0.058593 | No |
| UniProt-grouped R->E MRR | missing | 0.059127 | No |
| EC-4-grouped R->E MRR | missing | 0.922020 | No |
| E->M MRR | missing | 0.610441 | No |

Because R1 stage1 is missing, we cannot separate:

- the effect of changing `hard_neg_weight` from 2.0 to 1.0
- the effect of extending Stage 1 from 12 epochs to 25 epochs
- later-stage effects from Stage 2 / Stage 3

Any numeric split of the R2 change between hard-negative removal and longer Stage 1 would over-interpret the saved artifacts.

### R2 Stage-Wise Evidence

R2 has complete stage-end checkpoints:

| Stage | row R->E MRR | UniProt-grouped R->E MRR | EC-4-grouped R->E MRR | E->M MRR |
|---|---:|---:|---:|---:|
| stage0 | 0.000092 | 0.000092 | 0.007326 | 0.000719 |
| stage1 | 0.058593 | 0.059127 | 0.922020 | 0.610441 |
| stage2 | 0.060464 | 0.061094 | 0.926472 | 0.622604 |
| stage3 | 0.060177 | 0.060711 | 0.917949 | 0.621205 |

Stage1-to-stage3 marginal change:

| Metric | R2 stage1 | R2 stage3 | Absolute change | Relative change from stage1 |
|---|---:|---:|---:|---:|
| row R->E MRR | 0.058593 | 0.060177 | +0.001584 | +2.70% |
| UniProt-grouped R->E MRR | 0.059127 | 0.060711 | +0.001584 | +2.68% |
| EC-4-grouped R->E MRR | 0.922020 | 0.917949 | -0.004071 | -0.44% |
| E->M MRR | 0.610441 | 0.621205 | +0.010764 | +1.76% |

Interpretation:

1. R2 stage1 produces the large jump from near-zero stage0 to usable alignment.
2. Later stages add only small changes to row-level / UniProt-level R->E.
3. EC-4 grouped R->E is already high at stage1 and remains high afterward.
4. E->M receives a modest later-stage refinement, but most E->M capability is also present by stage1.

This supports the Task 1.2 observation that the main R2 alignment is established by stage1. It does **not** identify whether the stage1 result is due to longer Stage 1, hard-negative removal, or their combination.

### Final-to-Final Context

Although direct attribution is impossible, final-to-final comparison is still useful as a safety check:

| Metric | R1 final | R2 final / expected | Absolute change | Relative change | Interpretation |
|---|---:|---:|---:|---:|---|
| row R->E MRR | 0.0575 | 0.060208 | +0.002708 | +4.71% | combined R2 changes did not reduce row-level R->E |
| UniProt-grouped R->E MRR | 0.0581 | 0.060708 | +0.002608 | +4.49% | combined R2 changes did not reduce UniProt-grouped R->E |
| EC-4-grouped R->E MRR | 0.918680 | 0.913213 | -0.005467 | -0.60% | EC-4 family-level structure remained essentially stable |
| E->M MRR | 0.6094 | 0.619546 | +0.010146 | +1.67% | E->M improved slightly |

This table is not a controlled ablation. It only shows that the combined R2 setting preserved EC-family retrieval and slightly improved row-level / UniProt-level R->E and E->M. It is consistent with disabling EC-1 hard-negative upweighting being non-harmful under R2, but it does not quantify the independent contribution of hard-negative removal.

### Attribution Assessment

What can be concluded:

1. Direct hard-negative vs Stage 1 attribution is not identifiable from saved artifacts.
2. The combined R2 setting is compatible with stable EC-family retrieval.
3. R2 stage1 establishes most of the observed alignment.
4. R2 final-to-final changes are modest and do not show broad regression of monitored metrics.

What cannot be concluded:

1. We cannot state that `hard_neg_weight=1.0` alone caused the row R->E gain.
2. We cannot state that extending Stage 1 alone caused the row R->E gain.
3. We cannot estimate a numeric contribution split between hard-negative removal and longer Stage 1.
4. We cannot use R1 final vs R2 stage1 as a controlled comparison because those checkpoints correspond to different training histories.

### Future Artifact Requirement

Any future training run intended to support attribution must save:

| Artifact | Requirement |
|---|---|
| `model_v3_stage0.pt` | save at Stage 0 end |
| `model_v3_stage1.pt` | save at Stage 1 end |
| `model_v3_stage2.pt` | save at Stage 2 end |
| `model_v3_stage3.pt` | save at Stage 3 end |
| `metrics_v3.json` | include row-level and grouped R->E metrics |
| postmortem stage eval JSON | include stage-wise row R->E, UniProt-grouped, EC-4-grouped, and E->M |

This is an artifact requirement, not an R3 training plan.

### Task 1.3 Status

| Item | Status |
|---|---|
| R1 stage1 checkpoint found | No |
| Direct R1-vs-R2 stage1 comparison | Not identifiable |
| R2 stage-wise evidence reviewed | Yes |
| Final-to-final context reviewed | Yes |
| Task 1.3 | COMPLETE |
| train.py modification | None |
| New computation | None |
| Retraining | None |
| GPU/DCU usage | None |
| R3 plan | None |

**Task 1.3 status: COMPLETE**

Direct hard-negative vs Stage 1 contribution is not identifiable from saved R1 artifacts. The R2 stage-wise curve nevertheless shows that most R2 alignment is already present by stage1, and that later stages only modestly change monitored metrics. Future attribution-capable runs must save all stage-end checkpoints.

---

## Task 1.4 Visualization Error Diagnosis

### 运行环境

```
source_report: /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs/R2_POSTMORTEM_TASK_1_4_VISUALIZATION_DIAGNOSIS.md
output_dir: /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11
R2 training job: r2_train_115204112
运行时间: 2026-06-12
Python: 3.9.23
matplotlib: 3.9.4
numpy: 1.26.4
GPU: 未使用
sbatch: 未执行
train.py: 未修改
```

### 诊断目标

R2 训练核心流程已完成，但最后保存可视化 PNG 时出现：

```text
ValueError: object __array__ method not producing an array
```

老师要求该错误不能简单跳过，需要先确认：

1. 完整 traceback 是否定位到 visualization-only 阶段
2. 四种 modality embedding 是否为健康的数值数组
3. NN index 与 embedding 是否一致，并用 100-row sanity check 验证检索结果没有静默损坏

### 完整 traceback 摘要

错误发生链路：

```text
train.py:1614 main()
  -> train.py:1590 visualize_four_modal(...)
  -> train.py:914 fig.savefig(output_dir / "unified_space_v3_results.png", dpi=150)
  -> matplotlib FigureCanvasAgg.draw
  -> matplotlib backend_agg draw_path
  -> ValueError: object __array__ method not producing an array
```

关键定位：

| Field | Value |
|---|---|
| Error type | `ValueError: object __array__ method not producing an array` |
| Origin function | `visualize_four_modal()` |
| Trigger call | `fig.savefig(...)` |
| train.py line | 914 |
| Runtime environment | Python 3.9.23 / matplotlib 3.9.4 / numpy 1.26.4 |
| Root-cause interpretation | Most likely an environment-specific matplotlib/numpy AGG rendering issue, triggered during patch rendering in the `axvspan` stage-color background. This is an inference from the traceback, not evidence of a model, embedding, or index data error. |

相关代码上下文为 training-loss subplot 中按 epoch 调用 `axvspan()` 生成 stage-color background。错误在 matplotlib AGG backend 渲染 patch 时触发。

### Embedding Health Check

文件：`embeddings_v3.npz`

| Key | type | dtype | shape | NaN count | Inf count | Status |
|---|---|---|---:|---:|---:|---|
| `reaction` | `ndarray` | `float32` | `(145607, 256)` | 0 | 0 | PASS |
| `enzyme` | `ndarray` | `float32` | `(145607, 256)` | 0 | 0 | PASS |
| `substrate` | `ndarray` | `float32` | `(145607, 256)` | 0 | 0 | PASS |
| `microbe` | `ndarray` | `float32` | `(145607, 256)` | 0 | 0 | PASS |

结论：四个 modality embedding 均存在，行数与训练数据一致，统一维度为 256，且无 NaN / Inf。

### NN Index Health Check

| Index File | Exists | ntotal | dim | Emb Rows | Emb Dim | Consistent |
|---|---|---:|---:|---:|---:|---|
| `reaction_nn_index.npz` | Yes | 145,607 | 256 | 145,607 | 256 | Yes |
| `enzyme_nn_index.npz` | Yes | 145,607 | 256 | 145,607 | 256 | Yes |
| `substrate_nn_index.npz` | Yes | 145,607 | 256 | 145,607 | 256 | Yes |
| `microbe_nn_index.npz` | Yes | 145,607 | 256 | 145,607 | 256 | Yes |

所有 index 使用 numpy NN fallback 格式。`ntotal` 和 `dim` 均与 `embeddings_v3.npz` 一致。

### 100-Row R->E EC-4 Top-1 Sanity Check

EC-4 parser 使用与 R2 grouped evaluation 一致的严格口径：

- `ec_number` 必须是 string
- `split(".")` 后至少 4 段
- 前 4 段均必须能通过 `int()` 数字校验
- 否则视为 unknown，并从 EC-4 hit-rate 统计中排除

| Metric | Value |
|---|---:|
| Random seed | 20260612 |
| Sample size | 100 |
| Evaluated strict EC-4 rows | 88 |
| Excluded unknown rows | 12 |
| EC-4 top-1 hits | 80 / 88 |
| 100-row EC-4 top-1 hit rate | 0.9091 |
| Full-set EC-4-grouped R->E top-1 from `metrics_v3.json` | 0.8389 |
| Delta | +0.0702 |

抽样解释：

```text
SE = sqrt(0.8389 * 0.1611 / 88) ≈ 0.0392
delta = 0.0702 ≈ 1.79 SE
```

该 100-row sanity check 与 full-set metric 在 2 SE 内一致。它只作为健康检查，不替代全量 EC-4 grouped evaluation。

非标准 EC 示例（如 `3.6.5.n1`）已按 strict parser 排除，不计为 MISS。

### Core Output Impact

visualization error 发生在 `fig.savefig()` 阶段，核心产物已在此之前保存：

| Output | Status |
|---|---|
| `model_v3.pt` | Saved |
| `model_v3_stage{0,1,2,3}.pt` | Saved |
| `embeddings_v3.npz` | Saved |
| `*_nn_index.npz` | Saved |
| `metrics_v3.json` | Saved |
| `metadata_v3.json` | Saved |
| `enzyme2microbe_index.json` | Saved |
| `training_history.json` | Saved |
| `unified_space_v3_results.png` | Not saved; visualization-only artifact |

因此，该错误不影响 R2 的模型、embedding、metrics 或 NN index。PNG 只是一张可视化汇总图，不包含 `metrics_v3.json` 和 `embeddings_v3.npz` 之外的独有科学结果。

### Engineering Conclusion

Conclusion: **safe to try/except** for the optional visualization side effect.
The traceback, embedding health check, NN index health check, and 100-row EC-4
sanity check do not indicate a silent model, embedding, or index bug.

在上述三项诊断均 PASS 后，后续若需要工程兜底，可以在 `visualize_four_modal()` 调用外层加入 `try/except`，使 optional visualization 失败时训练脚本仍以 clean exit 完成。

当前 postmortem 阶段**不修改 `train.py`**。该结论仅说明：在未来获批的工程修复中，给 visualization-only side effect 加兜底是合理的。

### Task 1.4 Status

| Diagnostic Item | Status |
|---|---|
| Complete traceback | PASS |
| Embedding type / dtype / shape / NaN / Inf | PASS |
| NN index consistency | PASS |
| 100-row R->E EC-4 top-1 sanity check | PASS |

**Task 1.4 status: COMPLETE**
No train.py modification.
No sbatch.
No GPU used.

---

## Task 1.5 S->M MRR Historical Comparison

### 运行环境

```
source_report: /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs/R2_POSTMORTEM_TASK_1_5_SM_HISTORY.md
R1 output: /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07
R2 output: /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11
运行时间: 2026-06-12
GPU: 未使用
sbatch: 未执行
train.py: 未修改
```

### 文件来源

| Run | File | Used For |
|---|---|---|
| R1 ESM-C baseline | `outputs/full_esmc_baseline_2026-06-07/metrics_v3.json` | S->M MRR, S->M top-k, E->M MRR |
| R1 ESM-C baseline | `outputs/full_esmc_baseline_2026-06-07/R1_DIAGNOSIS_20260608_172625.md` | Cross-reference for `metrics_v3.json` values |
| R2 ESM-C hard_neg=1, stage1=25 | `outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/metrics_v3.json` | S->M MRR, S->M top-k, E->M MRR |

### S->M MRR Comparison

Teacher-checklist two-row view:

| Run | S->M MRR | Source |
|---|---:|---|
| R1 | 0.6108356276986662 | `outputs/full_esmc_baseline_2026-06-07/metrics_v3.json` |
| R2 | 0.5870633083741500 | `outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/metrics_v3.json` |

| Metric | R1 | R2 | Delta | Relative Change | Note |
|---|---:|---:|---:|---:|---|
| S->M MRR | 0.6108 | 0.5871 | -0.0238 | -3.89% | Monitoring metric; check non-degradation |
| S->M top-1 | 0.005570 | 0.005693 | +0.000123 | +2.21% | Stable / slightly improved |
| S->M top-5 | 0.025047 | 0.025102 | +0.000055 | +0.22% | Stable |
| S->M top-10 | 0.045362 | 0.044194 | -0.001168 | -2.57% | Slight decrease |
| S->M top-20 | 0.076713 | 0.075985 | -0.000728 | -0.95% | Stable |

Full precision:

| Metric | R1 Exact | R2 Exact |
|---|---:|---:|
| S->M MRR | 0.6108356276986662 | 0.5870633083741500 |
| S->M top-1 | 0.005569787166825771 | 0.005693407597162224 |
| S->M top-5 | 0.025046872746502573 | 0.025101815159985440 |
| S->M top-10 | 0.045361830131793120 | 0.044194303845282160 |
| S->M top-20 | 0.076713344825454820 | 0.075985357846806820 |

### Related Cross-Modal Context

| Metric | R1 | R2 | Delta | Relative Change |
|---|---:|---:|---:|---:|
| E->M MRR | 0.6094 | 0.6195 | +0.0102 | +1.67% |
| E->M top-1 | 0.003640 | 0.004382 | +0.000742 | +20.38% |

E->M MRR slightly improved in R2, which indicates that the modest S->M MRR decrease is not part of a broad microbe-retrieval regression pattern.

### Interpretation

S->M is a monitoring metric, not a primary success criterion. R2 decreased from 0.6108 to 0.5871 on S->M MRR, a relative change of -3.89%. This is a modest decrease and should continue to be monitored, but it does not suggest a broad microbe-retrieval regression because E->M MRR improved and S->M top-1 remained stable.

The project direction remains R->EC-4 retrieval. S->M should be tracked to ensure cross-modal behavior does not regress, but it should not be used as the main R2 judgment axis.

No tool-oriented acceptance criterion is set at this stage. Future interpretation should remain tied to downstream agent/tool requirements.

### Task 1.5 Status

| Item | Status |
|---|---|
| R1 S->M MRR found | Yes, 0.6108 from `metrics_v3.json` |
| R2 S->M MRR found | Yes, 0.5871 from `metrics_v3.json` |
| Task 1.5 | COMPLETE |

**Task 1.5 status: COMPLETE**
No train.py modification.
No sbatch.
No GPU used.

---

## Task 1.6 Tool-Oriented Baseline Data Collection

### 运行环境

```
source_report: /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs/R2_POSTMORTEM_TASK_1_6_TOOL_BASELINES_RUN_RESULT.md
script: /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/postmortem_tool_baselines.py
output_dir: /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11
job_id: 115309479
partition: kshctest02
node: h17r2n14
state: COMPLETED
elapsed: 00:04:13
exit_code: 0:0
ReqMem: 13G
MaxRSS: 1,615,920K (~1.6 GB)
CPU: x86_64
Python: 3.9.23
numpy: 1.26.4
train.py: 未修改
retraining: 未执行
R3 plan: 未写
GPU/DCU: 未使用
```

资源说明：`kshdnormal04` 分区要求最少 `--gres=dcu:1`，因此该 CPU-only postmortem job 使用 `kshctest02`。集群 `DefMemPerCPU=3500MB`，4 CPU 对应 memory 上限约 14GB，故使用 `--mem=13G`。

### 输出文件检查

| File | Generated | Size / Status |
|---|---|---|
| `postmortem_tool_baselines.json` | Yes | 7,098 bytes |
| `tool_baseline_ood_score_hist.png` | No | skipped due matplotlib/numpy AGG rendering issue |

PNG histogram 没有生成，但脚本已捕获该 visualization-only 错误，并将 fallback histogram data 写入 `postmortem_tool_baselines.json` 的 `ood_distribution.histogram_fallback`。核心 calibration、OOD-like distribution、latency、input check 均已完成。

### Input Check

| Item | Value |
|---|---|
| reaction embeddings | `(145607, 256)` |
| enzyme embeddings | `(145607, 256)` |
| metadata rows | 145,607 |
| reaction norm before explicit L2 | mean=1.0, min=1.0, max=1.0 |
| enzyme norm before explicit L2 | mean=1.0, min=1.0, max=1.0 |
| similarity method | cosine via explicit L2 normalization |

Both embeddings were already L2-normalized. Explicit L2 normalization was still applied as a safety net before score computation.

### 1.6.1 Calibration Curve Baseline

This baseline records confidence vs. EC-4 hit rate for R->E top-1 retrieval. It is baseline data only; no threshold is set.

| Confidence Bin | Mean Score | EC-4 Hit Rate | Sample Count |
|---|---:|---:|---:|
| 0.0-0.1 | 0.050000 | 0.000000 | 0 |
| 0.1-0.2 | 0.150000 | 0.000000 | 0 |
| 0.2-0.3 | 0.250000 | 0.000000 | 0 |
| 0.3-0.4 | 0.350000 | 0.000000 | 0 |
| 0.4-0.5 | 0.450000 | 0.000000 | 0 |
| 0.5-0.6 | 0.550000 | 0.000000 | 0 |
| 0.6-0.7 | 0.650000 | 0.000000 | 0 |
| 0.7-0.8 | 0.782729 | 0.642857 | 28 |
| 0.8-0.9 | 0.877573 | 0.392399 | 15,734 |
| 0.9-1.0 | 0.957110 | 0.903332 | 112,085 |

| Metric | Value |
|---|---:|
| ECE baseline | 0.106888 |
| EC-4 valid queries | 127,847 |
| EC-4 excluded / unknown | 17,760 |

All EC-4 valid queries fall in the 0.7-1.0 score range, with most in 0.9-1.0. This is a calibration baseline only and does not define a confidence threshold.

### 1.6.1 Calibration Shape Qualitative (added 2026-06-15)

The 10-bin calibration curve is highly non-uniform:

- 0.9-1.0 bin (112,085 / 88% of valid queries): nearly diagonal
  (gap = +0.054).
- 0.8-0.9 bin (15,734 / 12% of valid queries): severely overconfident
  (score 0.878 vs hit 0.392, gap = +0.485). This is the dominant
  contributor to the aggregated ECE = 0.107.
- 0.7-0.8 bin (28 samples): low support, no statistical conclusion.
- < 0.7 bins: zero samples.

Implication for future agent integration (no threshold set now):

- A single global confidence threshold is not usable. The 0.8-0.9 band
  is the high-risk zone: a confidence score of 0.85 corresponds to ~39%
  EC-4 hit rate.
- Two recalibration candidates worth exploring at agent integration time:
  isotonic regression and Platt scaling on the 0.8-0.9 band, or
  abstaining when score < 0.9.
- This is recorded as a baseline finding only; no threshold is set in
  R2 or R3.

### 1.6.2 OOD-Like Score Distribution Baseline

OOD-like data here is a feature-level synthetic proxy, not real OOD data. The proxy was generated by applying Gaussian noise (`sigma=0.5`) to 5% of reaction embeddings and re-normalizing before R->E top-1 scoring.

| Distribution | mean | p50 | p95 | p99 | n |
|---|---:|---:|---:|---:|---:|
| In-distribution | 0.944912 | 0.957852 | 0.983821 | 0.987288 | 145,607 |
| OOD-like proxy | 0.222236 | 0.219982 | 0.274606 | 0.300199 | 7,280 |

The in-distribution and synthetic OOD-like proxy score distributions are strongly separated. This is recorded as baseline evidence only; no abstention threshold is set.

#### Inline Histogram Fallback

The histogram PNG was skipped due to a matplotlib/numpy AGG rendering issue. The fallback histogram data was written to JSON and merged from 80 raw bins into 10 ranges by bin index for readability.

| Score Range | In-distribution Count | OOD-like Count |
|---|---:|---:|
| [0.1339, 0.2195) | 0 | 3,591 |
| [0.2195, 0.3051) | 0 | 3,630 |
| [0.3051, 0.3908) | 0 | 59 |
| [0.3908, 0.4764) | 0 | 0 |
| [0.4764, 0.5620) | 0 | 0 |
| [0.5620, 0.6477) | 0 | 0 |
| [0.6477, 0.7333) | 2 | 0 |
| [0.7333, 0.8189) | 78 | 0 |
| [0.8189, 0.9046) | 24,011 | 0 |
| [0.9046, 0.9902) | 121,516 | 0 |
| **Total** | **145,607** | **7,280** |

| Count check | Expected | Observed | Status |
|---|---:|---:|---|
| In-distribution histogram total | 145,607 | 145,607 | PASS |
| OOD-like histogram total | 7,280 | 7,280 | PASS |

This inline table is the histogram fallback required for review. It is baseline only, no threshold, and not real OOD data.

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

### 1.6.3 Latency Baseline

Latency was measured for single-query R->E retrieval with enzyme embeddings preloaded in memory. Five warm-up queries were discarded before 100 measured queries.

| Metric | Value |
|---|---:|
| p50 | 4.5575 ms |
| p95 | 5.2403 ms |
| p99 | 5.5788 ms |
| n_warmup | 5 |
| n_measured | 100 |
| query batch size | 1 (single-query R->E retrieval) |
| Enzyme cache | Preloaded (`enzyme_nn_index.npz` in memory) |
| hostname | h17r2n14 |
| CPU | x86_64; source report did not record a finer CPU model string |
| GPU/DCU | Not used (CPU-only partition `kshctest02`) |

This is a latency baseline only; no latency target line is set.

### Task 1.6 Status

| Item | Status |
|---|---|
| Calibration curve baseline | COMPLETE |
| OOD-like score distribution baseline | COMPLETE |
| Inline histogram fallback | COMPLETE |
| Latency baseline | COMPLETE |
| train.py modification | None |
| Retraining | None |
| GPU/DCU usage | None |
| R3 plan | None |
| Tool-oriented acceptance criterion | None |

**Task 1.6 status: COMPLETE**
No train.py modification.
No retraining.
No GPU/DCU used.
No R3 plan.
No tool-oriented acceptance criterion set.

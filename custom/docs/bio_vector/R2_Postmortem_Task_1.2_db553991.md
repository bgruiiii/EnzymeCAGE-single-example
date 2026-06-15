# R2 Postmortem Task 1.2: Stage-Wise Checkpoint Evaluation (Revised)

## Overview

评估 R2 的 4 个 stage-end checkpoint（stage 0-3），生成 stage-wise MRR 演化表。
通过 Slurm CPU job 运行，不在 login node 上执行。不重训、不用 GPU。

## 前置确认

### Chunked Retrieval 安全性确认

`train.py` 中的 `evaluate_multimodal()` (L752-827) 和 `evaluate_grouped_re()` (L1279-1397) **均已使用 chunked retrieval**：

- `sim_chunk = queries[start:end] @ candidates.T` → shape `(4096, 145607)` float64 ≈ 4.5 GB/chunk
- 每次只保留一个 chunk 在内存中，处理完立即 `del sim_chunk`
- **不会构建完整 N×N 矩阵或全量 `np.argsort`**
- MRR 使用 `rank = 1 + np.sum(scores > pos_score)` 而非全量排序

### 内存分析

| 组件 | 大小 |
|------|------|
| 4 × embedding array (145607, 256) float32 | ~300 MB |
| sim_chunk (4096, 145607) float64 | ~4.5 GB |
| all_topk + all_mrr 辅助数组 | ~10 MB |
| model weights | ~20 MB |
| **峰值内存** | **~10 GB** |

64 GB Slurm 分配非常充裕。

### `test_size=0` 确认

`load_enzyme_cage_300(data_dir, test_size=0, enzyme_feature="esmc")` 在 `test_size > 0` 为 False 时返回 7-tuple（无 split），代码逻辑正确（L1214-1222）。

---

## Task 1: 创建评估脚本

**文件**: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/postmortem_eval_stage_checkpoints.py`

### 脚本结构

```python
"""
Postmortem Stage Checkpoint Evaluation for Bio Vector R2.

Evaluates model_v3_stage{0,1,2,3}.pt checkpoints to produce a
stage-wise MRR evolution table.

Input:
  - Stage checkpoints: {output_dir}/model_v3_stage{N}.pt
  - Data features: loaded via train.load_enzyme_cage_300()
Output:
  - Prints stage-wise MRR table to stdout
  - Writes JSON summary to {output_dir}/postmortem_stage_eval.json
Dependencies:
  - train.py (Config, UnifiedSpace, MultiModalDataset,
    load_enzyme_cage_300, evaluate_multimodal, evaluate_grouped_re)
  - numpy, torch, json

Usage:
  python postmortem_eval_stage_checkpoints.py --stage all
  python postmortem_eval_stage_checkpoints.py --stage 1
"""
```

### 实现步骤

1. **从 train.py 导入**（sys.path 方式）：
   - `Config`, `UnifiedSpace`, `MultiModalDataset`
   - `load_enzyme_cage_300`
   - `evaluate_multimodal` (L752, 已确认 chunked)
   - `evaluate_grouped_re` (L1279, 已确认 chunked)

2. **argparse**:
   - `--stage`: choices `["all", "0", "1", "2", "3"]`, default `"all"`
   - `--data_dir`: 训练数据目录
   - `--output_dir`: R2 输出目录（含 checkpoints）
   - `--chunk_size`: default 4096

3. **一次性数据加载**（所有 stage 共享）：
   ```python
   r_feat, e_feat, s_feat, m_feat, ec_labels, metadata, concept_targets = \
       load_enzyme_cage_300(data_dir, test_size=0, enzyme_feature="esmc")
   dataset = MultiModalDataset(r_feat, e_feat, s_feat, m_feat, ec_labels,
                               concept_targets, [m.get("assembly_accession","") for m in metadata])
   eval_loader = DataLoader(dataset, batch_size=4096, shuffle=False)
   ```

4. **对每个 stage checkpoint 统一流程**（无 shortcut）：
   ```python
   for stage_idx in stages_to_eval:
       ckpt = torch.load(f"{output_dir}/model_v3_stage{stage_idx}.pt", map_location="cpu")
       cfg = Config()  # 使用默认值（与 R2 训练一致）
       model = UnifiedSpace(r_dim, e_dim, s_dim, m_dim, cfg)
       model.load_state_dict(ckpt["model_state_dict"])
       model.eval()
       # row-level evaluation (chunked)
       results, all_r, all_e, all_s, all_m = evaluate_multimodal(
           model, eval_loader, device="cpu", chunk_size=chunk_size)
       # grouped R→E evaluation (chunked)
       grouped = evaluate_grouped_re(all_r, all_e, metadata, chunk_size=chunk_size)
       # 提取 4 个核心指标
       row_re_mrr = results["R→E_MRR"]
       uniprot_mrr = grouped["UniProt-grouped_MRR"]
       ec4_mrr = grouped["EC-4-grouped_MRR"]
       em_mrr = results["E→M_MRR"]
   ```

5. **Stage 3 consistency check**（额外）：
   - 将 stage3 结果与 R2 final metrics (`metrics_v3.json`) 对比
   - 如果不接近，打印 discrepancy 警告，不硬写结论

6. **输出**：
   - 打印 markdown 表格到 stdout
   - 写入 `{output_dir}/postmortem_stage_eval.json`

### 输出表格格式

```
| Stage | row R→E MRR | UniProt-grouped R→E MRR | EC-4-grouped R→E MRR | E→M MRR |
|---|---:|---:|---:|---:|
| stage0 | ? | ? | ? | ? |
| stage1 | ? | ? | ? | ? |
| stage2 | ? | ? | ? | ? |
| stage3 | ? | ? | ? | ? |
```

---

## Task 2: 创建 Slurm CPU job 脚本

**文件**: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/run_postmortem_eval.sh`

```bash
#!/bin/bash
#SBATCH --job-name=r2_postmortem_eval
#SBATCH --partition=kshdnormal04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000MB
#SBATCH --time=6:00:00
#SBATCH --output=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/postmortem_eval_%j.out
#SBATCH --error=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/postmortem_eval_%j.err

set -euo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

WORK_DIR=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo
OUTPUT_DIR=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11
DATA_DIR=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/data/reaction_enzyme_microbe_training_clean_2026-06-01_LOCAL

echo "=== R2 Postmortem Stage Evaluation ==="
echo "hostname: $(hostname)"
echo "date: $(date)"

# 环境
source /public/home/acfbwjsi7s/miniconda3/etc/profile.d/conda.sh
conda activate nis

cd ${WORK_DIR}
python postmortem_eval_stage_checkpoints.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --stage all

echo "=== Done ==="
```

### 资源选择理由

| 参数 | 值 | 理由 |
|---|---|---|
| partition | kshdnormal04 | 与 R2 训练相同分区，有 DCU 节点的 CPU 也可用 |
| cpus-per-task | 8 | 数据加载和 numpy 矩阵运算可多线程 |
| mem | 64 GB | 峰值 ~10 GB + 安全裕量（ESM-C 文件加载可能临时占用更多） |
| time | 6:00:00 | 数据加载 ~10 min + 4 stages × ~40 min = ~3h，留 2x 裕量 |
| --gres | 无 | 纯 CPU 评估，不需要 DCU |

---

## Task 3: 追加结果到 postmortem 文档

文件: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs/R2_POSTMORTEM_20260612.md`

追加 section:
```
## Task 1.2 Stage-Wise Checkpoint Evaluation

### 运行环境
（Slurm job ID, 运行时间, 资源使用）

### 结果表格
（从脚本输出复制）

### Stage 3 Consistency Check
（与 R2 final 对比）

### Interpretation
（回答 5 个问题）

### 状态声明
Task 1.2 status: COMPLETE
No train.py modification.
No sbatch (for training).
No GPU used. (Slurm CPU job only)
```

---

## 执行顺序

1. **创建脚本** `postmortem_eval_stage_checkpoints.py`
2. **创建 Slurm 脚本** `run_postmortem_eval.sh`
3. **先不运行** — 等用户确认后 sbatch

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Stage 3 结果与 R2 final 不一致 | 先打印 discrepancy，分析原因（BN running stats、batch ordering），不硬写结论 |
| ESM-C 文件加载在 CPU 节点上更慢 | 64 GB + 6h 时间充裕 |
| `Config()` 默认值与 checkpoint 不完全匹配 | checkpoint 中有 `config` dict，优先使用 checkpoint config 覆盖 |
| 登录节点 conda activate 影响 Slurm 脚本 | Slurm 脚本内部重新 `conda activate`，不依赖登录节点环境 |

## Rejected Alternatives

1. **Login node 直接运行**: 用户明确要求不在 login node 长时间跑
2. **重写模型/评估代码**: 直接从 train.py import，避免不一致
3. **Stage 3 使用 embeddings_v3.npz shortcut**: 用户要求 4 个 stage 统一流程；仅在最后做额外 consistency check
4. **使用 DCU GPU**: CPU 对 4.8M 参数模型推理完全够用，无需浪费 GPU 资源

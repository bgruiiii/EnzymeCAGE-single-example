# Task 1.6 Tool-Oriented Baseline 数据采集

## 概述
创建 Python 脚本 + Slurm CPU 脚本 + 审计报告，采集 R→E retrieval 的 calibration / OOD-like / latency baseline 数据。纯数据采集，不设阈值，不判断达标/失败。

## Task 1: 创建 Python 脚本 `postmortem_tool_baselines.py`

**路径**: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/postmortem_tool_baselines.py`

### 脚本结构

```
1. 常量定义（路径、CHUNK_SIZE=2048、SEED=20260612、OOD_FRACTION=0.05、N_LATENCY=100）
2. strict EC-4 parser（前4段 int() 验证，返回 None 表示 unknown）
3. calibration_curve() — 10-bin 分桶 calibration + ECE
4. ood_score_distribution() — in-dist vs OOD-like proxy
5. latency_baseline() — 100 query 单 query 延迟
6. collect_env_info() — hostname/CPU/Python/numpy
7. main() — 串联所有，输出 JSON + PNG
```

### 1.1 Calibration Curve 实现

- **加载**: `enzyme_nn_index.npz["embeddings"]` (L2-normed enzyme) + `embeddings_v3.npz["reaction"]` (reaction emb)
- **EC-4 parser**: 从 `metadata_v3.json` 读取每行 `ec_number`，strict parse
- **排除**: EC-4 unknown 行不参与
- **Chunked top-1**: 
  ```python
  CHUNK_SIZE = 2048
  for start in range(0, N, CHUNK_SIZE):
      sim_chunk = reaction_emb[start:end] @ enzyme_emb.T  # (chunk, N)
      top1_idx = np.argmax(sim_chunk, axis=1)  # 只取 top-1
      top1_score = sim_chunk[arange, top1_idx]
  ```
- **Hit 判定**: top-1 enzyme 的 EC-4 == query reaction 的 EC-4
- **10-bin 分桶**: `np.linspace(0, 1, 11)` 边界，每 bin 输出 mean_score / EC-4 hit_rate / sample_count
- **ECE**: `sum(bin_count/N * |bin_accuracy - bin_confidence|)`，标注 "baseline only, no threshold"
- **不构建 N×N 矩阵**: chunked 逐行处理，每 chunk 后 `del sim_chunk`

### 1.2 OOD-like Score Distribution 实现

- **In-distribution**: 复用 calibration 阶段的 top-1 scores（全部 reaction query）
- **OOD-like proxy**: 
  - 固定 seed=20260612
  - 随机选 5% reaction embeddings (`n_ood = int(0.05 * N)`)
  - 加 Gaussian noise `np.random.normal(0, 0.5, shape)` 到选中的 embeddings
  - 重新 L2 normalize
  - 用 perturbed embeddings 查询 enzyme top-1 score（同样 chunked）
  - **文档明确标注**: "OOD-like proxy via feature-level Gaussian perturbation, NOT real OOD data"
- **统计**: mean / p50 / p95 / p99 / n for both distributions
- **Histogram PNG**: matplotlib 双直方图，保存到指定路径

### 1.3 Latency Baseline 实现

- **Preload**: enzyme embedding 预加载到内存（`enzyme_nn_index.npz["embeddings"]`）
- **100 queries**: seed=20260612，随机选 100 个 reaction embedding
- **逐 query 计时**: 
  ```python
  for q in selected_queries:
      t0 = time.perf_counter()
      scores = q @ enzyme_emb.T  # (N,) 
      top1 = np.argmax(scores)
      t1 = time.perf_counter()
      latencies.append(t1 - t0)
  ```
- **报告**: p50/p95/p99 (ms)
- **环境记录**: `socket.gethostname()`, `platform.processor()`, `sys.version`, `np.__version__`, GPU/DCU = False

### 1.4 JSON 输出

```json
{
  "calibration": { "bins": [...], "ece": ..., "note": "baseline only, no threshold" },
  "ood_distribution": {
    "in_distribution": { "mean": ..., "p50": ..., "p95": ..., "p99": ..., "n": ... },
    "ood_like_proxy": { "mean": ..., "p50": ..., "p95": ..., "p99": ..., "n": ..., "method": "Gaussian noise σ=0.5 + L2 renorm", "note": "OOD-like proxy, NOT real OOD" }
  },
  "latency": { "p50_ms": ..., "p95_ms": ..., "p99_ms": ..., "n_queries": 100 },
  "environment": { "hostname": ..., "cpu": ..., "python": ..., "numpy": ..., "gpu_dcu": false }
}
```

## Task 2: 创建 Slurm CPU 脚本

**路径**: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/run_postmortem_tool_baselines.sh`

```bash
#!/bin/bash
#SBATCH --job-name=r2_tool_baselines
#SBATCH --partition=kshdnormal04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=OUTPUT_DIR/%j.out
#SBATCH --error=OUTPUT_DIR/%j.err
```

- 不使用 `--gres`（无 GPU/DCU）
- `mkdir -p OUTPUT_DIR`
- `source conda.sh && conda activate nis`
- `cd WORK_DIR && python postmortem_tool_baselines.py`

## Task 3: 语法检查

- `python -m py_compile postmortem_tool_baselines.py`
- `bash -n run_postmortem_tool_baselines.sh`

## Task 4: 审计报告

**路径**: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs/R2_POSTMORTEM_TASK_1_6_TOOL_BASELINES_SCRIPT_AUDIT.md`

必须包含：
- Python 脚本完整内容
- Slurm 脚本完整内容
- chunked 计算说明（避免 N×N full matrix）
- strict EC-4 parser 说明
- OOD-like proxy 定义说明
- latency 预加载 enzyme cache 说明
- py_compile / bash -n 结果
- 未修改 train.py / 未运行脚本 / 未 sbatch / 未使用 GPU/DCU
- 结论：READY_FOR_REVIEW 或 BLOCKED

## 关键文件

| 文件 | 路径 |
|------|------|
| Python 脚本 | `code/demo/postmortem_tool_baselines.py` |
| Slurm 脚本 | `code/demo/run_postmortem_tool_baselines.sh` |
| 审计报告 | `docs/R2_POSTMORTEM_TASK_1_6_TOOL_BASELINES_SCRIPT_AUDIT.md` |
| 输入数据 | `outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/{embeddings_v3.npz, metadata_v3.json, enzyme_nn_index.npz}` |

## 约束遵守清单

- [x] 不修改 train.py
- [x] 不运行 Python 脚本
- [x] 不提交 sbatch
- [x] 不使用 GPU/DCU
- [x] 不设 hard threshold
- [x] 不写 R3 plan
- [x] 不使用 failure/catastrophic/negative result 措辞
- [x] chunked 计算避免 N×N
- [x] strict EC-4 parser

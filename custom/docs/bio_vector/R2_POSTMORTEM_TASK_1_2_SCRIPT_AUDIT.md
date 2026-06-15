# R2 Postmortem Task 1.2 — Script Audit (Updated with NumPy Safety Patch)

审计时间：2026-06-12  
审计范围：Checkpoint 存在性 + 评估脚本创建 + Slurm 脚本创建 + py_compile + **NumPy 类型安全补丁**

---

## 1. Checkpoint 文件存在性检查

```
$ ls -lh model_v3_stage{0,1,2,3}.pt

-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s 19M 6月 11 18:30 model_v3_stage0.pt
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s 19M 6月 11 18:35 model_v3_stage1.pt
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s 19M 6月 11 18:36 model_v3_stage2.pt
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s 19M 6月 11 18:39 model_v3_stage3.pt
```

| Checkpoint | 大小 | 时间戳 | 状态 |
|---|---|---|---|
| model_v3_stage0.pt | 19 MB | Jun 11 18:30 | ✅ 存在 |
| model_v3_stage1.pt | 19 MB | Jun 11 18:35 | ✅ 存在 |
| model_v3_stage2.pt | 19 MB | Jun 11 18:36 | ✅ 存在 |
| model_v3_stage3.pt | 19 MB | Jun 11 18:39 | ✅ 存在 |

**结论：全部 4 个 checkpoint 文件均存在，可继续。**

---

## 2. Focused Diff — NumPy 类型安全补丁

### 2.1 新增 `to_jsonable` helper（插入在 `parse_args()` 之前）

```python
def to_jsonable(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
```

### 2.2 JSON dump 调用修改

**Before:**
```python
json.dump(all_results, f, indent=2, ensure_ascii=False)
```

**After:**
```python
json.dump(to_jsonable(all_results), f, indent=2, ensure_ascii=False)
```

### 2.3 Patch 原因

`evaluate_multimodal()` 和 `evaluate_grouped_re()` 返回的 dict 中包含大量 NumPy 类型：
- `np.float64` — 来自 `np.mean()`, `np.max()` 等聚合函数
- `np.int64` — 来自 `np.sum()`, `int()` 转换
- `np.ndarray` — 来自 `all_topk`, `all_mrr` 等数组

直接调用 `json.dump()` 会抛出 `TypeError: Object of type float64 is not JSON serializable`。

`to_jsonable()` 递归遍历整个 result dict，将所有 NumPy 类型转换为原生 Python 类型：
- `np.ndarray` → `list` (via `.tolist()`)
- `np.float64`, `np.int64` 等 → `float`/`int` (via `.item()`)
- 保留 dict 的 string keys

### 2.4 影响范围

- **仅修改 `postmortem_eval_stage_checkpoints.py`**
- **未修改 train.py**
- **未修改 run_postmortem_eval.sh**
- 不影响评估逻辑、chunked retrieval、consistency check 等核心功能

---

## 3. postmortem_eval_stage_checkpoints.py 完整内容（Patch 后）

```python
#!/usr/bin/env python3
"""
Postmortem Stage Checkpoint Evaluation for Bio Vector R2.

Evaluates model_v3_stage{0,1,2,3}.pt checkpoints to produce a
stage-wise MRR evolution table (row-level R→E, UniProt-grouped R→E,
EC-4-grouped R→E, E→M MRR).

Input:
  - Stage checkpoints: {output_dir}/model_v3_stage{N}.pt
  - Data features: loaded via train.load_enzyme_cage_300()
Output:
  - Prints stage-wise MRR table to stdout
  - Writes JSON summary to {output_dir}/postmortem_stage_eval.json
  - Stage 3 consistency check against metrics_v3.json

Dependencies:
  - train.py (Config, UnifiedSpace, MultiModalDataset,
    load_enzyme_cage_300, evaluate_multimodal, evaluate_grouped_re)
  - numpy, torch, json

Usage:
  python postmortem_eval_stage_checkpoints.py --stage all
  python postmortem_eval_stage_checkpoints.py --stage 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Import from train.py (same directory) ──
_script_dir = str(Path(__file__).resolve().parent)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from train import (
    Config,
    UnifiedSpace,
    MultiModalDataset,
    load_enzyme_cage_300,
    evaluate_multimodal,
    evaluate_grouped_re,
)


def to_jsonable(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def parse_args():
    parser = argparse.ArgumentParser(
        description="Postmortem: evaluate R2 stage checkpoints")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "0", "1", "2", "3"],
                        help="Which stage(s) to evaluate")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Training data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="R2 output directory (contains checkpoints)")
    parser.add_argument("--chunk_size", type=int, default=4096,
                        help="Chunk size for chunked retrieval")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    chunk_size = args.chunk_size

    # ── Determine which stages to evaluate ──
    if args.stage == "all":
        stages = [0, 1, 2, 3]
    else:
        stages = [int(args.stage)]

    # ── Verify checkpoint files exist ──
    for s in stages:
        ckpt_path = out_dir / f"model_v3_stage{s}.pt"
        if not ckpt_path.exists():
            print(f"ERROR: Checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
        print(f"  Checkpoint found: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

    # ── One-time data loading ──
    print(f"\n=== Loading data (once for all stages) ===")
    t0 = time.time()
    result = load_enzyme_cage_300(str(data_dir), test_size=0,
                                   enzyme_feature="esmc")
    r_feat, e_feat, s_feat, m_feat, ec_labels, metadata, concept_targets = result
    n = len(r_feat)
    print(f"  Total samples: {n}")
    print(f"  Dims: reaction={r_feat.shape[1]}, enzyme={e_feat.shape[1]}, "
          f"substrate={s_feat.shape[1]}, microbe={m_feat.shape[1]}")

    # Build dataset and eval loader
    dataset = MultiModalDataset(
        r_feat, e_feat, s_feat, m_feat, ec_labels,
        concept_targets,
        [m.get("assembly_accession", "") for m in metadata])
    eval_loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # ── Load R2 final metrics for stage 3 consistency check ──
    metrics_path = out_dir / "metrics_v3.json"
    r2_final = None
    if metrics_path.exists():
        with open(metrics_path) as f:
            r2_final = json.load(f)
        print(f"  R2 final metrics loaded from {metrics_path}")
    else:
        print(f"  WARNING: metrics_v3.json not found, skipping consistency check")

    # ── Evaluate each stage ──
    print(f"\n=== Evaluating {len(stages)} checkpoint(s) ===")
    all_results = {}

    for stage_idx in stages:
        ckpt_path = out_dir / f"model_v3_stage{stage_idx}.pt"
        print(f"\n--- Stage {stage_idx}: {ckpt_path.name} ---")
        t1 = time.time()

        # Load checkpoint
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        cfg = Config()  # Use default config (matches R2 training)

        # Create model
        model = UnifiedSpace(
            reaction_dim=r_feat.shape[1],
            enzyme_dim=e_feat.shape[1],
            substrate_dim=s_feat.shape[1],
            microbe_dim=m_feat.shape[1],
            cfg=cfg,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

        # Row-level evaluation (chunked)
        print(f"  Running row-level evaluation (chunk_size={chunk_size})...")
        row_results, all_r, all_e, all_s, all_m = evaluate_multimodal(
            model, eval_loader, device="cpu", chunk_size=chunk_size)

        # Grouped R→E evaluation (chunked)
        print(f"  Running grouped R→E evaluation (chunk_size={chunk_size})...")
        grouped_results = evaluate_grouped_re(
            all_r, all_e, metadata, chunk_size=chunk_size)

        elapsed = time.time() - t1
        print(f"  Stage {stage_idx} evaluated in {elapsed:.1f}s")

        # Extract core metrics
        stage_metrics = {
            "row_RE_MRR": row_results.get("R→E_MRR", None),
            "UniProt_grouped_RE_MRR": grouped_results.get("UniProt-grouped_MRR", None),
            "EC4_grouped_RE_MRR": grouped_results.get("EC-4-grouped_MRR", None),
            "EM_MRR": row_results.get("E→M_MRR", None),
        }

        # Also store full results for completeness
        stage_metrics["row_results"] = row_results
        stage_metrics["grouped_results"] = grouped_results
        stage_metrics["elapsed_seconds"] = elapsed

        all_results[f"stage{stage_idx}"] = stage_metrics

        print(f"  row R→E MRR:          {stage_metrics['row_RE_MRR']:.6f}")
        print(f"  UniProt-grouped MRR:  {stage_metrics['UniProt_grouped_RE_MRR']:.6f}")
        print(f"  EC-4-grouped MRR:     {stage_metrics['EC4_grouped_RE_MRR']:.6f}")
        print(f"  E→M MRR:              {stage_metrics['EM_MRR']:.6f}")

    # ── Stage 3 consistency check ──
    if 3 in stages and r2_final is not None:
        print(f"\n=== Stage 3 Consistency Check ===")
        s3 = all_results["stage3"]
        targets = {
            "row_RE_MRR": r2_final["R→E_MRR"],
            "UniProt_grouped_RE_MRR": r2_final["grouped_re"]["UniProt-grouped_MRR"],
            "EC4_grouped_RE_MRR": r2_final["grouped_re"]["EC-4-grouped_MRR"],
            "EM_MRR": r2_final["E→M_MRR"],
        }
        consistency = {}
        all_pass = True
        for key in targets:
            actual = s3[key]
            expected = targets[key]
            if expected != 0:
                rel_err = abs(actual - expected) / abs(expected)
            else:
                rel_err = abs(actual - expected)
            passed = rel_err < 0.05  # 5% tolerance
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"  {key}: actual={actual:.6f}, expected={expected:.6f}, "
                  f"rel_err={rel_err:.4f} [{status}]")
            consistency[key] = {
                "actual": actual,
                "expected": expected,
                "relative_error": rel_err,
                "status": status,
            }
        s3["consistency_check"] = consistency
        s3["consistency_all_pass"] = all_pass
        if all_pass:
            print("  Consistency check: ALL PASS")
        else:
            print("  Consistency check: SOME FAILED — investigate before writing conclusions")

    # ── Print summary table ──
    print(f"\n{'='*80}")
    print("Stage-Wise MRR Evolution Table")
    print(f"{'='*80}")
    print(f"| {'Stage':<8} | {'row R→E MRR':>14} | {'UniProt MRR':>14} "
          f"| {'EC-4 MRR':>14} | {'E→M MRR':>14} |")
    print(f"|{'-'*10}|{'-'*16}|{'-'*16}|{'-'*16}|{'-'*16}|")
    for stage_idx in stages:
        s = all_results[f"stage{stage_idx}"]
        print(f"| stage{stage_idx:<4} | {s['row_RE_MRR']:>14.6f} "
              f"| {s['UniProt_grouped_RE_MRR']:>14.6f} "
              f"| {s['EC4_grouped_RE_MRR']:>14.6f} "
              f"| {s['EM_MRR']:>14.6f} |")
    print(f"{'='*80}")

    # ── Save JSON ──
    json_path = out_dir / "postmortem_stage_eval.json"
    with open(json_path, "w") as f:
        json.dump(to_jsonable(all_results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
```

**文件路径**: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/postmortem_eval_stage_checkpoints.py`  
**行数**: 253 行 (原 237 行，+16 行 to_jsonable helper)

---

## 4. run_postmortem_eval.sh 完整内容

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

HOME_DIR=/public/home/acfbwjsi7s
WORK_DIR=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo
OUTPUT_DIR=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11
DATA_DIR=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/data/reaction_enzyme_microbe_training_clean_2026-06-01_LOCAL

echo "=== R2 Postmortem Stage Evaluation ==="
echo "hostname: $(hostname)"
echo "date: $(date)"
echo "job_id: $SLURM_JOB_ID"

# ── Ensure output dir exists ──
mkdir -p ${OUTPUT_DIR}

# ── Environment ──
source ${HOME_DIR}/miniconda3/etc/profile.d/conda.sh
conda activate nis
echo "conda env: $CONDA_DEFAULT_ENV"
echo "python: $(which python)"

cd ${WORK_DIR}

python postmortem_eval_stage_checkpoints.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --stage all

echo "=== Postmortem Eval Done ==="
```

**文件路径**: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/run_postmortem_eval.sh`  
**行数**: 42 行

### Slurm 资源申请

| 参数 | 值 | 说明 |
|---|---|---|
| partition | kshdnormal04 | CPU 节点 |
| cpus-per-task | 8 | numpy 多线程 |
| mem | 64000 MB | 峰值 ~10 GB，留 6x 裕量 |
| time | 6:00:00 | 预计 ~3h，留 2x 裕量 |
| gres | 无 | **不使用 DCU/GPU** |

---

## 5. py_compile 结果（Patch 后重新验证）

```
$ /public/home/acfbwjsi7s/miniconda3/envs/nis/bin/python -m py_compile \
    /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/postmortem_eval_stage_checkpoints.py
py_compile: OK
```

```
$ bash -n run_postmortem_eval.sh
bash syntax: OK
```

**两个脚本均通过语法检查。**

---

## 6. 声明

- ✅ **未修改 train.py**
- ✅ **未修改 run_postmortem_eval.sh**
- ✅ **未执行 sbatch**
- ✅ **未使用 GPU/DCU**
- ✅ 评估脚本仅 import train.py 中的函数，不修改任何训练逻辑
- ✅ NumPy 类型安全补丁仅影响 JSON 序列化，不影响评估计算
- ✅ Slurm 脚本不申请 `--gres`，纯 CPU 评估

---

## 7. 审计结论

### ✅ READY_FOR_SBATCH

| 检查项 | 状态 |
|--------|------|
| 4 个 checkpoint 全部存在 | ✅ |
| postmortem_eval_stage_checkpoints.py 创建 + NumPy 补丁 | ✅ (253 行) |
| run_postmortem_eval.sh 创建 | ✅ (42 行) |
| py_compile 通过 (Python 3.9, patch 后) | ✅ |
| bash -n 通过 | ✅ |
| 未修改 train.py | ✅ |
| 未 sbatch | ✅ |
| 未使用 GPU | ✅ |

**Patch 摘要**:
- 新增 `to_jsonable()` helper (+16 行)
- 递归转换 `np.ndarray` → `list`, `np.float64`/`np.int64` → `float`/`int`
- 防止 `json.dump()` 因 NumPy 类型序列化失败

**下一步：可以执行 `sbatch run_postmortem_eval.sh` 提交评估任务**

---

**Script audit status: READY_FOR_SBATCH**

# Bio Vector R2 Postmortem — Task 1.6 Tool-Oriented Baselines Script Audit (Patched)

**Date**: 2026-06-12
**Scope**: Task 1.6 — Minimal safety patch: explicit L2 normalization with pre-flight norm stats
**Status**: **READY_FOR_SBATCH**

---

## 1. Focused Diff (Patch v1 → v2)

Only `main()` in `postmortem_tool_baselines.py` was modified. No other functions changed.

```diff
     enzyme_nn = np.load(ENZYME_NN_PATH, allow_pickle=True)
-    enzyme_emb = enzyme_nn["embeddings"]  # (N, 256) already L2-normed
+    enzyme_emb = enzyme_nn["embeddings"]  # (N, 256)
     log(f"  enzyme embeddings (nn_index): {enzyme_emb.shape}")

     ...

     assert len(metadata) == N, ...

+    # ── Pre-flight: L2 norm statistics & explicit normalization ──
+    log("")
+    log("── Pre-flight: L2 norm check ──")
+    r_norms = np.linalg.norm(reaction_emb, axis=1)
+    e_norms = np.linalg.norm(enzyme_emb, axis=1)
+    norm_stats = {
+        "reaction_before_norm": {
+            "mean": round(float(r_norms.mean()), 6),
+            "min": round(float(r_norms.min()), 6),
+            "max": round(float(r_norms.max()), 6),
+        },
+        "enzyme_before_norm": {
+            "mean": round(float(e_norms.mean()), 6),
+            "min": round(float(e_norms.min()), 6),
+            "max": round(float(e_norms.max()), 6),
+        },
+        "similarity": "cosine via explicit L2 normalization",
+    }
+    log(f"  reaction norm: mean={norm_stats['reaction_before_norm']['mean']:.4f} "
+        f"min={norm_stats['reaction_before_norm']['min']:.4f} "
+        f"max={norm_stats['reaction_before_norm']['max']:.4f}")
+    log(f"  enzyme   norm: mean={norm_stats['enzyme_before_norm']['mean']:.4f} "
+        f"min={norm_stats['enzyme_before_norm']['min']:.4f} "
+        f"max={norm_stats['enzyme_before_norm']['max']:.4f}")
+
+    reaction_emb = l2_normalize(reaction_emb).astype(np.float32)
+    enzyme_emb = l2_normalize(enzyme_emb).astype(np.float32)
+    log("  Explicit L2 normalization applied to both reaction and enzyme embeddings.")
+
     # ── 1. Calibration ──
     ...

     result = {
         "task": ...,
         "timestamp": ...,
+        "input_check": to_jsonable(norm_stats),
         "calibration": ...,
         ...
     }
```

**Summary of changes**:
1. Removed assumption that `enzyme_nn_index.npz` is pre-L2-normed; now explicitly normalizing both embeddings
2. Added pre-flight L2 norm statistics (mean/min/max for both reaction and enzyme)
3. Added `input_check` section to output JSON with norm stats and similarity method declaration
4. All other logic (calibration, OOD-like, latency, EC-4 parser, chunked computation) unchanged

---

## 2. Python Script — Full Content (v2, 455 lines)

**Path**: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/postmortem_tool_baselines.py`

```python
#!/usr/bin/env python3
"""
R2 Postmortem Task 1.6 — Tool-Oriented Baseline Data Collection

Collects three categories of baseline data for future agent/tool calibration:
  1. Calibration curve  (R→E top-1 cosine similarity vs EC-4 hit rate, 10-bin)
  2. OOD-like score distribution  (in-distribution vs feature-level Gaussian proxy)
  3. Latency baseline  (single-query R→E retrieval, 5 warm-up + 100 measured)

Usage:
    python postmortem_tool_baselines.py

Notes:
  - Baseline only, no threshold is set anywhere.
  - No R3 plan, no pass/fail judgement.
  - Chunked computation throughout — never builds a full N×N matrix.
  - Strict EC-4 parser: first 4 dot-separated segments must all pass int().
  - OOD-like proxy is feature-level synthetic perturbation, NOT real OOD data.
"""

import json
import os
import platform
import socket
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = (
    "/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04"
    "/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11"
)
NPZ_PATH = os.path.join(OUTPUT_DIR, "embeddings_v3.npz")
ENZYME_NN_PATH = os.path.join(OUTPUT_DIR, "enzyme_nn_index.npz")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata_v3.json")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics_v3.json")

OUT_JSON = os.path.join(OUTPUT_DIR, "postmortem_tool_baselines.json")
OUT_HIST = os.path.join(OUTPUT_DIR, "tool_baseline_ood_score_hist.png")

CHUNK_SIZE = 2048
SEED = 20260612
OOD_FRACTION = 0.05
OOD_NOISE_SIGMA = 0.5
N_WARMUP = 5
N_LATENCY = 100
N_CALIBRATION_BINS = 10


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_ec4(ec_str):
    """Strict EC-4 parser.

    Returns a string like '1.2.3.4' when the first 4 dot-separated
    segments all pass int(); otherwise returns None (unknown / invalid).
    """
    if not ec_str or not isinstance(ec_str, str):
        return None
    s = ec_str.strip()
    if s in ("", "unknown", "-"):
        return None
    parts = s.split(".")
    if len(parts) < 4:
        return None
    try:
        for i in range(4):
            int(parts[i])
    except ValueError:
        return None
    return f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}"


def l2_normalize(x):
    """L2-normalize along the last axis; zero-norm rows stay zero."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def to_jsonable(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return str(obj)
        return obj
    return obj


# ──────────────────────────────────────────────────────────────────────
# 1. Calibration Curve
# ──────────────────────────────────────────────────────────────────────
def compute_calibration(reaction_emb, enzyme_emb, metadata):
    """Chunked R→E top-1 calibration: 10-bin cosine similarity vs EC-4 hit rate.

    - Strict EC-4 parser; unknown rows are excluded.
    - Top-1 only (argmax, no full argsort).
    - Hit = top-1 enzyme EC-4 equals query reaction EC-4.
    - ECE reported as 'baseline only, no threshold'.
    """
    N = len(reaction_emb)
    log(f"Calibration: N={N}, chunk_size={CHUNK_SIZE}")

    # Pre-parse EC-4 for every row
    ec4_list = [parse_ec4(m.get("ec_number", "")) for m in metadata]
    valid_mask = np.array([ec is not None for ec in ec4_list], dtype=bool)
    n_valid = int(valid_mask.sum())
    n_excluded = N - n_valid
    log(f"  EC-4 valid rows: {n_valid}, excluded (unknown): {n_excluded}")

    # Pre-parse EC-4 for enzyme side (same metadata — row i enzyme ↔ row i reaction)
    enzyme_ec4 = ec4_list  # same metadata indexing

    # Chunked top-1 retrieval
    all_top1_scores = np.empty(N, dtype=np.float64)
    all_top1_indices = np.empty(N, dtype=np.int64)

    for start in range(0, N, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N)
        sim_chunk = reaction_emb[start:end] @ enzyme_emb.T  # (chunk, N)
        top1_idx = np.argmax(sim_chunk, axis=1)
        top1_score = sim_chunk[np.arange(end - start), top1_idx]
        all_top1_scores[start:end] = top1_score
        all_top1_indices[start:end] = top1_idx
        del sim_chunk

    # Determine hits for valid rows only
    valid_indices = np.where(valid_mask)[0]
    valid_scores = all_top1_scores[valid_indices]
    hits = np.zeros(n_valid, dtype=bool)
    for local_i, global_i in enumerate(valid_indices):
        query_ec4 = ec4_list[global_i]
        top1_enzyme_idx = all_top1_indices[global_i]
        top1_enzyme_ec4 = enzyme_ec4[top1_enzyme_idx]
        if top1_enzyme_ec4 is not None and top1_enzyme_ec4 == query_ec4:
            hits[local_i] = True

    # 10-bin calibration
    bin_edges = np.linspace(0.0, 1.0, N_CALIBRATION_BINS + 1)
    bins_out = []
    total_weighted_gap = 0.0
    for b in range(N_CALIBRATION_BINS):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b < N_CALIBRATION_BINS - 1:
            mask = (valid_scores >= lo) & (valid_scores < hi)
        else:
            mask = (valid_scores >= lo) & (valid_scores <= hi)
        count = int(mask.sum())
        if count > 0:
            mean_score = float(valid_scores[mask].mean())
            hit_rate = float(hits[mask].mean())
        else:
            mean_score = float((lo + hi) / 2.0)
            hit_rate = 0.0
        bins_out.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "mean_score": round(mean_score, 6),
            "ec4_hit_rate": round(hit_rate, 6),
            "sample_count": count,
        })
        bin_conf = mean_score
        bin_acc = hit_rate
        total_weighted_gap += (count / n_valid) * abs(bin_acc - bin_conf) if n_valid > 0 else 0.0

    ece = float(total_weighted_gap)
    log(f"  ECE (baseline only, no threshold): {ece:.6f}")

    return {
        "bins": bins_out,
        "ece": round(ece, 6),
        "n_valid": n_valid,
        "n_excluded": n_excluded,
        "note": "baseline only, no threshold",
        "all_top1_scores": all_top1_scores,  # used downstream by OOD section
        "valid_mask": valid_mask,
    }


# ──────────────────────────────────────────────────────────────────────
# 2. OOD-like Score Distribution
# ──────────────────────────────────────────────────────────────────────
def compute_ood_distribution(reaction_emb, enzyme_emb, in_dist_scores):
    """Compare in-distribution top-1 scores with OOD-like proxy scores.

    OOD-like proxy definition:
      - Feature-level synthetic perturbation, NOT real OOD data.
      - Randomly select 5% of reaction embeddings (fixed seed).
      - Add Gaussian noise N(0, σ=0.5) to selected embeddings.
      - Re-L2-normalize the perturbed embeddings.
      - Compute top-1 cosine similarity against enzyme embeddings (chunked).

    This is an OOD-like proxy for baseline calibration purposes only.
    No threshold is set. Baseline only, no threshold.
    """
    N = len(reaction_emb)
    rng = np.random.RandomState(SEED)
    n_ood = int(OOD_FRACTION * N)
    ood_indices = rng.choice(N, size=n_ood, replace=False)
    log(f"OOD-like proxy: n_ood={n_ood}, noise_sigma={OOD_NOISE_SIGMA}, seed={SEED}")

    # Perturb selected reaction embeddings
    perturbed = reaction_emb[ood_indices].copy()
    noise = rng.normal(0.0, OOD_NOISE_SIGMA, size=perturbed.shape).astype(np.float32)
    perturbed = perturbed + noise
    perturbed = l2_normalize(perturbed).astype(np.float32)

    # Chunked top-1 for perturbed queries
    ood_scores = np.empty(n_ood, dtype=np.float64)
    for start in range(0, n_ood, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_ood)
        sim_chunk = perturbed[start:end] @ enzyme_emb.T  # (chunk, N)
        top1_idx = np.argmax(sim_chunk, axis=1)
        ood_scores[start:end] = sim_chunk[np.arange(end - start), top1_idx]
        del sim_chunk

    # In-distribution stats (full set)
    in_mean = float(np.mean(in_dist_scores))
    in_p50 = float(np.percentile(in_dist_scores, 50))
    in_p95 = float(np.percentile(in_dist_scores, 95))
    in_p99 = float(np.percentile(in_dist_scores, 99))

    # OOD-like stats
    ood_mean = float(np.mean(ood_scores))
    ood_p50 = float(np.percentile(ood_scores, 50))
    ood_p95 = float(np.percentile(ood_scores, 95))
    ood_p99 = float(np.percentile(ood_scores, 99))

    log(f"  In-dist  : mean={in_mean:.4f} p50={in_p50:.4f} p95={in_p95:.4f} p99={in_p99:.4f} n={len(in_dist_scores)}")
    log(f"  OOD-like : mean={ood_mean:.4f} p50={ood_p50:.4f} p95={ood_p95:.4f} p99={ood_p99:.4f} n={n_ood}")

    # Histogram PNG
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(in_dist_scores, bins=80, alpha=0.6, label=f"In-distribution (n={len(in_dist_scores)})", color="steelblue")
    ax.hist(ood_scores, bins=80, alpha=0.6, label=f"OOD-like proxy (n={n_ood})", color="coral")
    ax.set_xlabel("Top-1 Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("R→E Top-1 Score Distribution: In-Distribution vs OOD-like Proxy\n(baseline only, no threshold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_HIST, dpi=150)
    plt.close(fig)
    log(f"  Histogram saved to {OUT_HIST}")

    return {
        "in_distribution": {
            "mean": round(in_mean, 6),
            "p50": round(in_p50, 6),
            "p95": round(in_p95, 6),
            "p99": round(in_p99, 6),
            "n": len(in_dist_scores),
        },
        "ood_like_proxy": {
            "mean": round(ood_mean, 6),
            "p50": round(ood_p50, 6),
            "p95": round(ood_p95, 6),
            "p99": round(ood_p99, 6),
            "n": n_ood,
            "method": f"Gaussian noise sigma={OOD_NOISE_SIGMA} + L2 renorm on {OOD_FRACTION*100:.0f}% reaction embeddings",
            "note": "feature-level synthetic OOD-like proxy, NOT real OOD data; baseline only, no threshold",
        },
    }


# ──────────────────────────────────────────────────────────────────────
# 3. Latency Baseline
# ──────────────────────────────────────────────────────────────────────
def compute_latency(reaction_emb, enzyme_emb):
    """Measure single-query R→E retrieval latency.

    - Enzyme embeddings preloaded (cache).
    - 5 warm-up queries (not counted).
    - 100 measured queries.
    - Fixed seed = 20260612.
    - Reports p50/p95/p99 in milliseconds.
    - No latency target line is set (baseline only).
    """
    rng = np.random.RandomState(SEED)
    n_total = N_WARMUP + N_LATENCY
    query_indices = rng.choice(len(reaction_emb), size=n_total, replace=False)
    queries = reaction_emb[query_indices]  # (n_total, D)

    log(f"Latency: {N_WARMUP} warm-up + {N_LATENCY} measured queries, seed={SEED}")

    # Warm-up (results discarded)
    for i in range(N_WARMUP):
        q = queries[i]
        _ = np.argmax(q @ enzyme_emb.T)

    # Measured
    latencies = []
    for i in range(N_WARMUP, n_total):
        q = queries[i]
        t0 = time.perf_counter()
        scores = q @ enzyme_emb.T  # (N,) dot product = cosine sim (L2-normed)
        _ = np.argmax(scores)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    log(f"  p50={p50:.2f} ms  p95={p95:.2f} ms  p99={p99:.2f} ms")

    return {
        "p50_ms": round(p50, 4),
        "p95_ms": round(p95, 4),
        "p99_ms": round(p99, 4),
        "n_warmup": N_WARMUP,
        "n_measured": N_LATENCY,
        "note": "baseline only, no latency target line",
    }


# ──────────────────────────────────────────────────────────────────────
# 4. Environment Info
# ──────────────────────────────────────────────────────────────────────
def collect_env_info():
    """Record hardware and software environment."""
    info = {
        "hostname": socket.gethostname(),
        "cpu": platform.processor() or "unknown",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "gpu_dcu": False,
        "note": "CPU-only job, no GPU/DCU used",
    }
    log(f"  hostname={info['hostname']}, cpu={info['cpu']}, "
        f"python={info['python_version']}, numpy={info['numpy_version']}")
    return info


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("R2 Postmortem Task 1.6 — Tool-Oriented Baseline Collection")
    log("=" * 60)

    # ── Load data ──
    log("Loading data …")
    npz = np.load(NPZ_PATH, allow_pickle=True)
    reaction_emb = npz["reaction"]  # (N, 256)
    log(f"  reaction embeddings: {reaction_emb.shape}")

    enzyme_nn = np.load(ENZYME_NN_PATH, allow_pickle=True)
    enzyme_emb = enzyme_nn["embeddings"]  # (N, 256)
    log(f"  enzyme embeddings (nn_index): {enzyme_emb.shape}")

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    log(f"  metadata rows: {len(metadata)}")

    N = len(reaction_emb)
    assert len(enzyme_emb) == N, f"Shape mismatch: reaction {N} vs enzyme {len(enzyme_emb)}"
    assert len(metadata) == N, f"Shape mismatch: reaction {N} vs metadata {len(metadata)}"

    # ── Pre-flight: L2 norm statistics & explicit normalization ──
    log("")
    log("── Pre-flight: L2 norm check ──")
    r_norms = np.linalg.norm(reaction_emb, axis=1)
    e_norms = np.linalg.norm(enzyme_emb, axis=1)
    norm_stats = {
        "reaction_before_norm": {
            "mean": round(float(r_norms.mean()), 6),
            "min": round(float(r_norms.min()), 6),
            "max": round(float(r_norms.max()), 6),
        },
        "enzyme_before_norm": {
            "mean": round(float(e_norms.mean()), 6),
            "min": round(float(e_norms.min()), 6),
            "max": round(float(e_norms.max()), 6),
        },
        "similarity": "cosine via explicit L2 normalization",
    }
    log(f"  reaction norm: mean={norm_stats['reaction_before_norm']['mean']:.4f} "
        f"min={norm_stats['reaction_before_norm']['min']:.4f} "
        f"max={norm_stats['reaction_before_norm']['max']:.4f}")
    log(f"  enzyme   norm: mean={norm_stats['enzyme_before_norm']['mean']:.4f} "
        f"min={norm_stats['enzyme_before_norm']['min']:.4f} "
        f"max={norm_stats['enzyme_before_norm']['max']:.4f}")

    reaction_emb = l2_normalize(reaction_emb).astype(np.float32)
    enzyme_emb = l2_normalize(enzyme_emb).astype(np.float32)
    log("  Explicit L2 normalization applied to both reaction and enzyme embeddings.")

    # ── 1. Calibration ──
    log("")
    log("── 1. Calibration Curve ──")
    cal_result = compute_calibration(reaction_emb, enzyme_emb, metadata)
    in_dist_scores = cal_result.pop("all_top1_scores")
    valid_mask = cal_result.pop("valid_mask")

    # ── 2. OOD-like Distribution ──
    log("")
    log("── 2. OOD-like Score Distribution ──")
    ood_result = compute_ood_distribution(reaction_emb, enzyme_emb, in_dist_scores)

    # ── 3. Latency ──
    log("")
    log("── 3. Latency Baseline ──")
    lat_result = compute_latency(reaction_emb, enzyme_emb)

    # ── 4. Environment ──
    log("")
    log("── 4. Environment Info ──")
    env_info = collect_env_info()

    # ── Write JSON ──
    result = {
        "task": "R2 Postmortem Task 1.6 — Tool-Oriented Baselines",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_check": to_jsonable(norm_stats),
        "calibration": to_jsonable(cal_result),
        "ood_distribution": to_jsonable(ood_result),
        "latency": to_jsonable(lat_result),
        "environment": to_jsonable(env_info),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"JSON written to {OUT_JSON}")

    log("")
    log("Task 1.6 baseline collection complete.")
    log("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 3. Slurm Script — Full Content (unchanged, 51 lines)

**Path**: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/run_postmortem_tool_baselines.sh`

```bash
#!/bin/bash
#SBATCH --job-name=r2_tool_baselines
#SBATCH --partition=kshdnormal04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/tool_baselines_%j.out
#SBATCH --error=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/tool_baselines_%j.err

# ──────────────────────────────────────────────────────────────────────
# R2 Postmortem Task 1.6 — Tool-Oriented Baseline Collection
# CPU-only job (no GPU/DCU). Do NOT use --gres.
# Can run in parallel with Task 1.2 evaluation job.
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
trap 'echo "ERROR: script failed at line $LINENO" >&2' ERR

HOME_DIR=/public/home/acfbwjsi7s
WORK_DIR=${HOME_DIR}/bio_vector_full_run_2026-06-04/code/demo
OUTPUT_DIR=${HOME_DIR}/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11

echo "============================================================"
echo " R2 Postmortem Task 1.6 — Tool-Oriented Baseline Collection"
echo "============================================================"
echo "hostname : $(hostname)"
echo "date     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "job_id   : ${SLURM_JOB_ID:-local}"
echo "cpus     : ${SLURM_CPUS_PER_TASK:-4}"
echo "mem      : 48G"
echo "output   : ${OUTPUT_DIR}"
echo "============================================================"

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Activate conda environment
source "${HOME_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate nis

# Run the baseline collection script
cd "${WORK_DIR}"
python postmortem_tool_baselines.py

echo ""
echo "============================================================"
echo " Task 1.6 complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
```

---

## 4. Syntax Check Results

| Check | Command | Environment | Result |
|-------|---------|-------------|--------|
| Python syntax | `python -m py_compile postmortem_tool_baselines.py` | conda `nis`, Python 3.9.23 | **PASS** |
| Bash syntax | `bash -n run_postmortem_tool_baselines.sh` | bash 4.2 | **PASS** |

---

## 5. Constraint Compliance

| Constraint | Status |
|------------|--------|
| train.py not modified | ✓ Confirmed |
| Python script not executed (only py_compile) | ✓ Confirmed |
| sbatch not submitted | ✓ Confirmed |
| GPU/DCU not used | ✓ Confirmed |
| No hard threshold set | ✓ Confirmed — all sections "baseline only, no threshold" |
| No R3 plan | ✓ Confirmed |
| No failure/catastrophic/negative wording | ✓ Confirmed |
| Chunked computation (no N×N) | ✓ Confirmed — CHUNK_SIZE=2048, `del sim_chunk` per chunk |
| No full argsort | ✓ Confirmed — only `np.argmax` used for top-1 |
| Strict EC-4 parser | ✓ Confirmed — `int()` on first 4 segments |
| OOD-like proxy documented | ✓ Confirmed — "feature-level synthetic, NOT real OOD" |
| Latency warm-up + measured | ✓ Confirmed — 5 warm-up (discarded) + 100 measured |
| Slurm absolute paths | ✓ Confirmed |
| Slurm cpus=4, mem=48G | ✓ Confirmed |
| Explicit L2 normalization (patch v2) | ✓ Confirmed — pre-flight stats + `l2_normalize()` |

---

## 6. Patch v2 Safety Guarantees

| Aspect | Before Patch | After Patch |
|--------|-------------|-------------|
| Reaction embedding normalization | Assumed pre-normed from model | **Explicit `l2_normalize()` + stats** |
| Enzyme embedding normalization | Assumed pre-normed from `enzyme_nn_index.npz` | **Explicit `l2_normalize()` + stats** |
| Norm audit trail | None | **JSON `input_check` with mean/min/max** |
| Similarity method | Implicit cosine | **Declared: "cosine via explicit L2 normalization"** |

The patch ensures that even if upstream normalization state is inconsistent, the script guarantees correct cosine similarity via explicit L2 normalization at runtime, and records the pre-normalization statistics for audit.

---

## 7. Conclusion

**READY_FOR_SBATCH**

Patch v2 passes all syntax checks. All constraints satisfied. Explicit L2 normalization safety net in place. Scripts are ready for `sbatch run_postmortem_tool_baselines.sh` when approved.

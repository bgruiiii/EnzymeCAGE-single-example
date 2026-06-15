# Bio Vector R2 Postmortem — Task 1.6 Tool-Oriented Baselines Run Result (v5)

**Date**: 2026-06-13
**Job ID**: 115309479
**Status**: **COMPLETED — TASK 1.6 COMPLETE**

---

## 1. Job ID & Submission

| Item | Value |
|------|-------|
| Job ID | **115309479** |
| Partition | `kshctest02` (CPU-only, no GRES) |
| Node | h17r2n14 |
| CPUs | 4 |
| Memory | 13G |
| State | **COMPLETED** |

**Partition note**: `kshdnormal04` requires min `--gres=dcu:1` (QOS policy); `kshctest02` is used for CPU-only postmortem jobs.
**Memory note**: DefMemPerCPU=3500MB; 4 CPUs therefore require mem ≤ 14GB; mem set to 13G.

---

## 2. sacct Status

```
JobID            JobName             State    Elapsed ExitCode     ReqMem     MaxRSS  AllocTRES
---------------- -------------- ---------- ---------- -------- ---------- ---------- ----------
115309479        r2_tool_basel+  COMPLETED   00:04:13      0:0        13G            billing=4+
115309479.batch  batch           COMPLETED   00:04:13      0:0              1615920K cpu=4,mem+
115309479.extern extern          COMPLETED   00:04:13      0:0                 3340K billing=4+
```

- Exit code: **0:0** (success)
- Elapsed: 4 minutes 13 seconds
- MaxRSS (batch): ~1.6 GB

---

## 3. stdout — Full Content (50 lines)

**Path**: `.../outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/tool_baselines_115309479.out`

```
============================================================
 R2 Postmortem Task 1.6 — Tool-Oriented Baseline Collection
============================================================
hostname : h17r2n14
date     : 2026-06-13 13:11:52
job_id   : 115309479
cpus     : 4
mem      : 13G (4 CPU × 3500 MB DefMemPerCPU cap)
output   : /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11
============================================================
[2026-06-13 13:12:24] ============================================================
[2026-06-13 13:12:24] R2 Postmortem Task 1.6 — Tool-Oriented Baseline Collection
[2026-06-13 13:12:24] ============================================================
[2026-06-13 13:12:24] Loading data …
[2026-06-13 13:12:24]   reaction embeddings: (145607, 256)
[2026-06-13 13:12:25]   enzyme embeddings (nn_index): (145607, 256)
[2026-06-13 13:12:26]   metadata rows: 145607
[2026-06-13 13:12:26]
[2026-06-13 13:12:26] ── Pre-flight: L2 norm check ──
[2026-06-13 13:12:26]   reaction norm: mean=1.0000 min=1.0000 max=1.0000
[2026-06-13 13:12:26]   enzyme   norm: mean=1.0000 min=1.0000 max=1.0000
[2026-06-13 13:12:26]   Explicit L2 normalization applied to both reaction and enzyme embeddings.
[2026-06-13 13:12:26]
[2026-06-13 13:12:26] ── 1. Calibration Curve ──
[2026-06-13 13:12:26] Calibration: N=145607, chunk_size=2048
[2026-06-13 13:12:26]   EC-4 valid rows: 127847, excluded (unknown): 17760
[2026-06-13 13:15:52]   ECE (baseline only, no threshold): 0.106888
[2026-06-13 13:15:52]
[2026-06-13 13:15:52] ── 2. OOD-like Score Distribution ──
[2026-06-13 13:15:52] OOD-like proxy: n_ood=7280, noise_sigma=0.5, seed=20260612
[2026-06-13 13:16:02]   In-dist  : mean=0.9449 p50=0.9579 p95=0.9838 p99=0.9873 n=145607
[2026-06-13 13:16:02]   OOD-like : mean=0.2222 p50=0.2200 p95=0.2746 p99=0.3002 n=7280
[2026-06-13 13:16:03]   WARNING: histogram PNG failed — object __array__ method not producing an array
[2026-06-13 13:16:03]   Continuing with fallback histogram data in JSON.
[2026-06-13 13:16:03]
[2026-06-13 13:16:03] ── 3. Latency Baseline ──
[2026-06-13 13:16:03] Latency: 5 warm-up + 100 measured queries, seed=20260612
[2026-06-13 13:16:03]   p50=4.56 ms  p95=5.24 ms  p99=5.58 ms
[2026-06-13 13:16:03]
[2026-06-13 13:16:03] ── 4. Environment Info ──
[2026-06-13 13:16:03]   hostname=h17r2n14, cpu=x86_64, python=3.9.23, numpy=1.26.4
[2026-06-13 13:16:03] JSON written to .../postmortem_tool_baselines.json
[2026-06-13 13:16:03]
[2026-06-13 13:16:03] Task 1.6 baseline collection complete.
[2026-06-13 13:16:03] ============================================================

============================================================
 Task 1.6 complete — 2026-06-13 13:16:03
============================================================
```

---

## 4. stderr — Empty

**Path**: `.../outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/tool_baselines_115309479.err`

stderr is empty (0 bytes). No errors.

---

## 5. Output File Status

| File | Generated? | Size |
|------|-----------|------|
| `postmortem_tool_baselines.json` | **YES** | 7,098 bytes |
| `tool_baseline_ood_score_hist.png` | **NO** | — |

PNG was not generated due to the matplotlib 3.9.4 / numpy 1.26.4 AGG backend incompatibility. The try/except caught the error gracefully. Fallback histogram data (bin_edges + counts) is written to JSON under `ood_distribution.histogram_fallback`.

- `histogram_png_status`: `"skipped_due_to_matplotlib_error"`
- `histogram_png_error`: `"object __array__ method not producing an array"`
- `histogram_fallback`: 80 bins with `in_distribution_counts` and `ood_like_counts` arrays

---

## 6. Calibration — 10-Bin Table + ECE

**ECE (baseline only, no threshold): 0.106888**
**n_valid: 127,847 | n_excluded: 17,760**

| Confidence Bin | Mean Score | EC-4 Hit Rate | Sample Count |
|---------------|-----------|---------------|-------------|
| 0.0–0.1 | 0.050000 | 0.000000 | 0 |
| 0.1–0.2 | 0.150000 | 0.000000 | 0 |
| 0.2–0.3 | 0.250000 | 0.000000 | 0 |
| 0.3–0.4 | 0.350000 | 0.000000 | 0 |
| 0.4–0.5 | 0.450000 | 0.000000 | 0 |
| 0.5–0.6 | 0.550000 | 0.000000 | 0 |
| 0.6–0.7 | 0.650000 | 0.000000 | 0 |
| 0.7–0.8 | 0.782729 | 0.642857 | 28 |
| 0.8–0.9 | 0.877573 | 0.392399 | 15,734 |
| 0.9–1.0 | 0.957110 | 0.903332 | 112,085 |

Notes: All 127,847 valid queries concentrate in the top 3 bins (0.7–1.0). The 0.9–1.0 bin dominates with 87.7% of queries and 90.3% EC-4 hit rate. Baseline only, no threshold.

---

## 7. OOD-like Score Distribution

| Distribution | mean | p50 | p95 | p99 | n |
|-------------|------|-----|-----|-----|---|
| In-distribution | 0.944912 | 0.957852 | 0.983821 | 0.987288 | 145,607 |
| OOD-like proxy | 0.222236 | 0.219982 | 0.274606 | 0.300199 | 7,280 |

- **Method**: Gaussian noise σ=0.5 + L2 renorm on 5% reaction embeddings
- **Note**: feature-level synthetic OOD-like proxy, NOT real OOD data; baseline only, no threshold
- **Histogram PNG**: skipped (matplotlib error); fallback data in JSON (80 bins)

The in-distribution / OOD-like score gap is large: mean 0.945 vs 0.222, confirming the perturbation produces meaningfully different scores.

---

## 8. Inline Histogram Fallback

The histogram PNG was skipped due to a known matplotlib 3.9.4 / numpy 1.26.4 AGG rendering incompatibility (`object __array__ method not producing an array`). The `.tolist()` mitigation was applied but the AGG backend error persisted. However, the try/except caught the error gracefully and **fallback histogram data was written to JSON** under `ood_distribution.histogram_fallback` (80 raw bins). The 80 bins are merged into 10 wider ranges below by **bin index** (each group = 8 consecutive raw bins), not by score-condition re-filtering.

This is the histogram fallback for the OOD-like score distribution.
Baseline only, no threshold.
NOT real OOD data; feature-level synthetic OOD-like proxy.

| Score Range | In-distribution Count | OOD-like Count |
|------------|----------------------:|---------------:|
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

### Sanity Check

| Count check | Expected | Observed | Status |
|-------------|--------:|--------:|--------|
| In-distribution histogram total | 145,607 | 145,607 | **PASS** |
| OOD-like histogram total | 7,280 | 7,280 | **PASS** |

The distributions are cleanly separated: in-distribution scores concentrate in the top two ranges (≥0.82: 145,527 / 145,607 = 99.9%), while OOD-like proxy scores concentrate below 0.39 (7,280 / 7,280 = 100%). This confirms the perturbation produces meaningfully different top-1 scores.

---

## 9. Latency Baseline

| Metric | Value |
|--------|-------|
| p50 | **4.5575 ms** |
| p95 | **5.2403 ms** |
| p99 | **5.5788 ms** |
| n_warmup | 5 (discarded) |
| n_measured | 100 |
| Enzyme cache | Preloaded (enzyme_nn_index.npz in memory) |
| Note | Baseline only, no latency target line |

---

## 10. Environment / Input Check

| Item | Value |
|------|-------|
| hostname | h17r2n14 |
| CPU | x86_64 |
| Python | 3.9.23 |
| numpy | 1.26.4 |
| GPU/DCU | **No** (cpu-only partition kshctest02) |
| reaction norm (before explicit L2) | mean=1.0, min=1.0, max=1.0 |
| enzyme norm (before explicit L2) | mean=1.0, min=1.0, max=1.0 |
| Similarity method | cosine via explicit L2 normalization |

Both embeddings were already L2-normalized. Explicit L2 normalization was applied as safety net.

---

## 11. Declarations

| Item | Status |
|------|--------|
| train.py modified | **No** |
| Retraining performed | **No** |
| sbatch submitted (this update) | **No** |
| GPU/DCU used | **No** |
| R3 plan written | **No** |
| Hard threshold set | **No** |

---

## 12. Summary

Task 1.6 completed successfully. All three baseline categories collected:

1. **Calibration**: 10-bin EC-4 hit rate vs cosine similarity, ECE=0.106888 (baseline only)
2. **OOD-like distribution**: in-dist mean=0.945, OOD-like mean=0.222 (large separation; baseline only)
3. **Latency**: p50=4.56ms, p95=5.24ms, p99=5.58ms (baseline only, no target)

PNG histogram was not generated due to known matplotlib/numpy incompatibility, but fallback histogram data is present in JSON. All core data is intact.

**TASK 1.6 COMPLETE**

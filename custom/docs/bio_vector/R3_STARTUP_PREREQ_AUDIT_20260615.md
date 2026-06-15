# R3 Startup Prerequisite Audit — 2026-06-15

> **Purpose**: Verify all R2 artifacts, code, environment, and paths required before R3 startup.
> This audit does NOT start R3 training, modify train.py, create eval scripts, or submit Slurm jobs.

---

## 1. Directory Existence

| Directory | Path | Status |
|-----------|------|--------|
| WORK_ROOT | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04` | ✅ EXISTS |
| CODE_DIR  | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo` | ✅ EXISTS |
| DATA_DIR  | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/data/reaction_enzyme_microbe_training_clean_2026-06-01_LOCAL` | ✅ EXISTS |
| R2_OUT    | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11` | ✅ EXISTS |
| DOCS_DIR  | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs` | ✅ EXISTS |

**Result: 5/5 directories present.**

---

## 2. R2 Output Files

| File | Size | Date | Status |
|------|------|------|--------|
| `model_v3.pt` | 24M | Jun 11 18:39 | ✅ EXISTS |
| `model_v3_stage0.pt` | 19M | Jun 11 18:30 | ✅ EXISTS |
| `model_v3_stage1.pt` | 19M | Jun 11 18:35 | ✅ EXISTS |
| `model_v3_stage2.pt` | 19M | Jun 11 18:36 | ✅ EXISTS |
| `model_v3_stage3.pt` | 19M | Jun 11 18:39 | ✅ EXISTS |
| `embeddings_v3.npz` | 569M | Jun 11 19:15 | ✅ EXISTS |
| `metadata_v3.json` | 22M | Jun 11 19:15 | ✅ EXISTS |
| `metrics_v3.json` | 1.7K | Jun 11 19:15 | ✅ EXISTS |
| `training_history.json` | 2.8K | Jun 11 18:39 | ✅ EXISTS |
| `reaction_nn_index.npz` | 143M | Jun 11 19:15 | ✅ EXISTS |
| `enzyme_nn_index.npz` | 143M | Jun 11 19:15 | ✅ EXISTS |
| `substrate_nn_index.npz` | 143M | Jun 11 19:15 | ✅ EXISTS |
| `microbe_nn_index.npz` | 143M | Jun 11 19:15 | ✅ EXISTS |

**Raw `ls -lh` output:**

```
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s  24M Jun 11 18:39 model_v3.pt
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s  19M Jun 11 18:30 model_v3_stage0.pt
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s  19M Jun 11 18:35 model_v3_stage1.pt
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s  19M Jun 11 18:36 model_v3_stage2.pt
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s  19M Jun 11 18:39 model_v3_stage3.pt
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s 569M Jun 11 19:15 embeddings_v3.npz
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s  22M Jun 11 19:15 metadata_v3.json
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s 1.7K Jun 11 19:15 metrics_v3.json
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s 2.8K Jun 11 18:39 training_history.json
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s 143M Jun 11 19:15 reaction_nn_index.npz
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s 143M Jun 11 19:15 enzyme_nn_index.npz
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s 143M Jun 11 19:15 substrate_nn_index.npz
-rw-rw-r-- 1 acfbwjsi7s acfbwjsi7s 143M Jun 11 19:15 microbe_nn_index.npz
```

**Result: 13/13 files present.**

---

## 3. Embeddings & Metadata Consistency

### 3.1 embeddings_v3.npz keys & shapes

| Key | Shape | Dtype | Status |
|-----|-------|-------|--------|
| `reaction` | (145607, 256) | float32 | ✅ OK |
| `enzyme` | (145607, 256) | float32 | ✅ OK |
| `substrate` | (145607, 256) | float32 | ✅ OK |
| `microbe` | (145607, 256) | float32 | ✅ OK |

All four required keys present. All shapes are `(145607, 256)`. ✅

### 3.2 metadata_v3.json

- Type: `list`
- Entry count: **145607** ✅ (matches embedding row count)

### 3.3 metrics_v3.json — required keys

| Metric Key | Value | Status |
|------------|-------|--------|
| `R→E_MRR` | 0.060207782731201254 | ✅ FOUND |
| `E→M_MRR` | 0.6195463344069435 | ✅ FOUND |
| `S→M_MRR` | 0.58706330837415 | ✅ FOUND |
| `grouped_re.EC-4-grouped_MRR` | 0.9132133257912153 | ✅ FOUND |
| `grouped_re.EC-2-grouped_MRR` | 0.930647474085499 | ✅ FOUND |
| `grouped_re.UniProt-grouped_MRR` | 0.060707529334763886 | ✅ FOUND |

**Result: All consistency checks passed. No anomalies.**

---

## 4. Code — train.py Symbol Check

File: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/train.py`

| Symbol | Status |
|--------|--------|
| `UnifiedSpace` | ✅ FOUND |
| `evaluate_multimodal` | ✅ FOUND |
| `evaluate_grouped_re` | ✅ FOUND |
| `load_enzyme_cage_300` | ✅ FOUND |
| `visualize_four_modal` | ✅ FOUND |

**Result: 5/5 symbols present. train.py not modified.**

---

## 5. Python Environment

### 5.1 System Python

| Item | Value |
|------|-------|
| `which python` | `/usr/bin/python` |
| `python --version` | Python 2.7.5 |
| `which python3` | `/usr/bin/python3` |
| `python3 --version` | Python 3.6.8 |
| numpy (python3) | 1.19.5 |
| torch (python3) | ❌ Not installed |

### 5.2 Conda Environments (miniconda3)

| Env | Python | torch | numpy | Notes |
|-----|--------|-------|-------|-------|
| `nis` | 3.9.23 | 1.13.1+git7d2dd01.abi0.dtk2310 | 1.26.4 | ⚠️ torch import fails on login node (DCU/HIP lib) |
| `sbml` | 3.11.15 | ❌ Not installed | — | — |
| `base` | 3.9.12 | ❌ Not installed | — | — |

### 5.3 Recommended Environment for R3

- **Conda env `nis`** is the intended runtime environment.
- torch version: `1.13.1+git7d2dd01.abi0.dtk2310` (Hygon DCU variant)
- ⚠️ **Login node limitation**: `import torch` fails on login node due to missing DCU/HIP runtime libraries (`libgalaxyhip.so.5`). This is **expected** — torch will function correctly on DCU compute nodes via Slurm.
- CPU-only audit scripts (embedding loading, metrics parsing, bucket splitting) can use system `python3` (3.6.8 + numpy 1.19.5) or conda `nis` env.

---

## 6. R2 Metrics Baseline Summary

| Metric | Value |
|--------|-------|
| EC-4-grouped R→E MRR | **0.9132133257912153** |
| EC-2-grouped R→E MRR | **0.930647474085499** |
| UniProt-grouped R→E MRR | **0.060707529334763886** |
| Row-level R→E MRR | **0.060207782731201254** |
| E→M MRR | **0.6195463344069435** |
| S→M MRR | **0.58706330837415** |

### Additional grouped_re metrics (from metrics_v3.json)

| Metric | Value |
|--------|-------|
| EC-3-grouped R→E MRR | 0.930354051237822 |
| EC-4-grouped evaluated | 127847 |
| EC-4-grouped excluded_unknown | 17760 |
| EC-2-grouped evaluated | 130635 |
| EC-2-grouped excluded_unknown | 14972 |

---

## 7. Anomalies & Issues

**None detected.** All files present, all shapes consistent, all metrics keys found, all code symbols present.

**Environment note (non-blocking):** torch in conda `nis` env cannot be imported on login node due to DCU/HIP library dependency. This does not block the current audit (CPU-only checks) and is expected to resolve on DCU compute nodes.

---

## 8. Declarations

| Declaration | Value |
|-------------|-------|
| train.py modified | **no** |
| eval script created | **no** |
| Slurm submitted | **no** |
| GPU/DCU used | **no** |
| retraining executed | **no** |
| bucket baseline computed | **no** |
| ready for next step `eval_ec4_buckets.py` creation | **yes** |

---

*Audit performed: 2026-06-15 | Auditor: automated prerequisite check*

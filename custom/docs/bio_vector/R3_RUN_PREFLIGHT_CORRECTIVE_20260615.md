# R3 Run Preflight Corrective Audit

**Date:** 2026-06-15
**Auditor:** Qoder
**Status:** Ready for local review

---

## 1. Purpose

This corrective preflight addresses two items from the initial `R3_RUN_PREFLIGHT_20260615.md`:

1. **Pre-create R3 output directory** — ensure Slurm stdout/stderr paths exist before `sbatch` submission
2. **Fix trap wording** — change "FAILED" to "STOPPED" in `run_r3_training.sh`

**No training is executed. No Slurm jobs are submitted. No GPU/DCU resources are used.**

---

## 2. Corrective Changes

| # | Change | Status | Notes |
|---|--------|--------|-------|
| 1 | R3 output directory pre-created via `mkdir -p` | ✅ Done | Empty, no old results |
| 2 | Trap wording `FAILED` → `STOPPED` in run script | ✅ Done | L14 |

---

## 3. R3 Output Directory

**Path:** `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r3_ec4_balanced_stage3skip_2026-06-15`

**`ls -la` result:**
```
total 8
drwxrwxr-x  2 acfbwjsi7s acfbwjsi7s 4096 Jun 15 11:29 .
drwxrwxr-x 14 acfbwjsi7s acfbwjsi7s 4096 Jun 15 11:29 ..
```

**Assessment:**
- Directory exists ✅
- Directory is empty (no old training results) ✅
- Slurm stdout/stderr files (`r3_train_%j.out`, `r3_train_%j.err`) will have a valid target path at submission time ✅

---

## 4. Run Script Diff

```diff
--- run_r3_training.sh.before_corrective_preflight
+++ run_r3_training.sh
@@ -11,7 +11,7 @@
 #SBATCH --error=.../r3_ec4_balanced_stage3skip_2026-06-15/r3_train_%j.err
 
 set -euo pipefail
-trap 'echo "FAILED at line $LINENO"' ERR
+trap 'echo "STOPPED at line $LINENO"' ERR
 
 HOME_DIR=/public/home/acfbwjsi7s
 WORK_DIR=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo
```

---

## 5. Modified Run Script (Full Content)

```bash
#!/bin/bash
#SBATCH --job-name=r3_ec4_balanced
#SBATCH --partition=kshdnormal04
#SBATCH --gres=dcu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3500MB
#SBATCH --time=2-00:00:00
#SBATCH --output=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r3_ec4_balanced_stage3skip_2026-06-15/r3_train_%j.out
#SBATCH --error=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r3_ec4_balanced_stage3skip_2026-06-15/r3_train_%j.err

set -euo pipefail
trap 'echo "STOPPED at line $LINENO"' ERR

HOME_DIR=/public/home/acfbwjsi7s
WORK_DIR=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo
OUTPUT_DIR=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r3_ec4_balanced_stage3skip_2026-06-15
DATA_DIR=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/data/reaction_enzyme_microbe_training_clean_2026-06-01_LOCAL

echo "=== R3 EC-4 Balanced Training Job ==="
echo "R3: EC-4 weighted sampler stage1/2, stage3 skip"
echo "hostname: $(hostname)"
echo "date: $(date)"
echo "workdir: ${WORK_DIR}"
echo "output_dir: ${OUTPUT_DIR}"
echo "data_dir: ${DATA_DIR}"

# ── Environment ──
echo "step: environment setup"

# DCU platform setup
module load compiler/dtk/23.10 2>&1 || true
echo "module list:" && module list 2>&1
export HSA_OVERRIDE_GFX_VERSION=9.0.6
echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"

source ${HOME_DIR}/miniconda3/etc/profile.d/conda.sh 2>&1 || true
conda activate nis
echo "conda env: $CONDA_DEFAULT_ENV"
echo "python: $(which python)"
echo "python version: $(python --version 2>&1)"
python - <<'PY'
import sys
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"device count: {torch.cuda.device_count()}")
if (not torch.cuda.is_available()) or torch.cuda.device_count() < 1:
    print("ERROR: DCU/CUDA device is not available; aborting before training.")
    sys.exit(1)
print("DCU/CUDA check PASS")
PY
echo "step: env done"

# ── Create output dir ──
mkdir -p ${OUTPUT_DIR}

# ── Record config ──
echo "step: recording R3 config"
echo "epochs_stage0=5" > ${OUTPUT_DIR}/r3_config.txt
echo "epochs_stage1=25" >> ${OUTPUT_DIR}/r3_config.txt
echo "epochs_stage2=8" >> ${OUTPUT_DIR}/r3_config.txt
echo "epochs_stage3=0 (stage3 skip, alias from stage2)" >> ${OUTPUT_DIR}/r3_config.txt
echo "hard_neg_weight=1.0" >> ${OUTPUT_DIR}/r3_config.txt
echo "ec4_weighted_sampler=stage1/stage2" >> ${OUTPUT_DIR}/r3_config.txt
echo "R3 plan: docs/R3_TRAIN_PATCH_CORRECTIVE_AUDIT_20260615.md" >> ${OUTPUT_DIR}/r3_config.txt
echo "train.py md5: $(md5sum ${WORK_DIR}/train.py | awk '{print $1}')" >> ${OUTPUT_DIR}/r3_config.txt
echo "step: config recorded"

cd ${WORK_DIR}

# ── Training ──
echo "step: starting R3 training"
echo "  epochs_stage0=5 (R2 was 8)"
echo "  epochs_stage1=25 (unchanged)"
echo "  epochs_stage2=8 (unchanged)"
echo "  epochs_stage3=0 (R2 was 10, stage3 skip)"
echo "  hard_neg_weight=1.0 (unchanged)"
echo "  EC-4 weighted sampler for stage1/stage2"
echo "  Total epochs: 5+25+8+0 = 38"

python train.py \
    --mode enzyme_cage_300 \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --enzyme_feature esmc \
    --no_test_split \
    --stage all

echo "step: training complete"

# ── Verify outputs ──
echo "step: verifying outputs"
for f in model_v3.pt model_v3_stage0.pt model_v3_stage1.pt model_v3_stage2.pt model_v3_stage3.pt \
         embeddings_v3.npz metrics_v3.json metadata_v3.json training_history.json; do
    if [ -f "${OUTPUT_DIR}/${f}" ]; then
        echo "  OK: ${f}"
    else
        echo "  MISSING: ${f}"
    fi
done
echo "step: verification complete"

echo "=== R3 Job finished ==="
```

---

## 6. Verification Results

### 6.1 bash -n (script syntax check)

```bash
$ bash -n run_r3_training.sh
bash -n PASSED
```

### 6.2 py_compile (train.py)

```bash
$ python -m py_compile train.py
py_compile PASSED
```

Both checks pass after the corrective modification.

---

## 7. Declarations

| Item | Status |
|------|--------|
| train.py modified after corrective patch | **no** |
| run script modified | **yes** (trap wording: FAILED → STOPPED) |
| R3 output directory pre-created | **yes** (empty, no old results) |
| Slurm submitted | **no** |
| GPU/DCU used | **no** |
| retraining executed | **no** |
| ready for local corrective preflight audit | **yes** |

---

## 8. File Manifest

| File | Path |
|------|------|
| R3 run script (corrected) | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/run_r3_training.sh` |
| Run script backup (pre-corrective) | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/run_r3_training.sh.before_corrective_preflight` |
| R3 output directory | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r3_ec4_balanced_stage3skip_2026-06-15/` |
| This audit | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs/R3_RUN_PREFLIGHT_CORRECTIVE_20260615.md` |

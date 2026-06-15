# R3 Training Submission Report

**Submission Time:** 2026-06-15 11:37:14 CST  
**Script Used:** `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/run_r3_training.sh`  
**Job ID:** `115402116`  
**Status:** ✅ Submitted and Running

---

## 1. Submission Details

### Pre-submission Check

```bash
$ cd /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo
$ bash -n run_r3_training.sh
bash -n PASSED
```

### sbatch Submission

```bash
$ sbatch run_r3_training.sh
Submitted batch job 115402116
```

### Job Queue Status

```bash
$ squeue -j 115402116
             JOBID PARTITION      NAME           USER           ST       TIME  NODES NODELIST(REASON)
         115402116 kshdnormal04   r3_ec4_balance acfbwjsi7s      R       0:08      1 f13r4n19
```

**Job Status:** Running (ST = R)  
**Node:** f13r4n19  
**Partition:** kshdnormal04  
**Runtime at check:** 0:08

---

## 2. Output Directory Status

```bash
$ ls -la /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r3_ec4_balanced_stage3skip_2026-06-15/
total 8
drwxrwxr-x  2 acfbwjsi7s acfbwjsi7s 4096 Jun 15 11:36 .
drwxrwxr-x 14 acfbwjsi7s acfbwjsi7s 4096 Jun 15 11:29 ..
-rw-rw-r--  1 acfbwjsi7s acfbwjsi7s    0 Jun 15 11:36 r3_train_115402116.err
-rw-rw-r--  1 acfbwjsi7s acfbwjsi7s  668 Jun 15 11:36 r3_train_115402116.out
```

**Observations:**
- ✅ Output directory exists and is accessible
- ✅ Slurm stdout file created: `r3_train_115402116.out` (668 bytes)
- ✅ Slurm stderr file created: `r3_train_115402116.err` (0 bytes, no errors yet)
- ✅ Job is actively running

---

## 3. Training Configuration

**Source Script:** `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/run_r3_training.sh`

### Epoch Configuration

| Stage | Epochs | Weighted Sampler | Notes |
|-------|--------|------------------|-------|
| Stage 0 (pretrain) | 5 | ❌ No | Reduced from 8 (R2) |
| Stage 1 (pairwise) | 25 | ✅ Yes | EC-4 balanced |
| Stage 2 (triplet) | 8 | ✅ Yes | EC-4 balanced |
| Stage 3 (bootstrap) | 0 | ❌ No | Skipped, alias to stage2 |
| **Total** | **38** | — | — |

### Key Parameters

- **hard_neg_weight:** 1.0 (unchanged from R2)
- **EC-4 Weighted Sampler:** Enabled for stage1/stage2
  - Weight formula: `1 / sqrt(group_size)`
  - Unknown EC-4 rows: bucketed as `__unknown_ec4__` with same weighting
- **Stage 3 Handling:** Skipped (epochs_stage3 = 0), `model_v3_stage3.pt` will be alias of `model_v3_stage2.pt`
- **Visualization:** `visualize_four_modal` wrapped in try/except

---

## 4. Declarations

| Item | Status |
|------|--------|
| train.py modified | **no** |
| run script modified | **no** |
| Slurm submitted | **yes** |
| retraining started/submitted | **yes** |
| result analysis executed | **no** |

---

## 5. Next Steps

1. **Monitor Job Progress:**
   ```bash
   squeue -j 115402116
   ```

2. **Check Job Output:**
   ```bash
   tail -f /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r3_ec4_balanced_stage3skip_2026-06-15/r3_train_115402116.out
   ```

3. **Check for Errors:**
   ```bash
   cat /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r3_ec4_balanced_stage3skip_2026-06-15/r3_train_115402116.err
   ```

4. **After Completion:**
   - Verify all output files are generated in the R3 output directory
   - Perform R3 evaluation and result analysis
   - Compare with R2 baseline results

---

## 6. Expected Output Files

The training job should generate the following files in `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r3_ec4_balanced_stage3skip_2026-06-15/`:

- `model_v3.pt` - Final trained model
- `model_v3_stage0.pt` - Stage 0 checkpoint
- `model_v3_stage1.pt` - Stage 1 checkpoint
- `model_v3_stage2.pt` - Stage 2 checkpoint (final)
- `model_v3_stage3.pt` - Alias of stage2 checkpoint
- `embeddings_v3.npz` - Final embeddings
- `metrics_v3.json` - Evaluation metrics
- `metadata_v3.json` - Training metadata
- `training_history.json` - Training logs
- `r3_config.txt` - Configuration summary
- `r3_train_115402116.out` - Slurm stdout
- `r3_train_115402116.err` - Slurm stderr

---

**Report Generated:** 2026-06-15 11:37:14 CST

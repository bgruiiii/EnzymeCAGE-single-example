# R3 Train Patch Corrective Audit

## 1. Purpose

This is a **corrective patch** addressing issues found during local audit review of `R3_TRAIN_PATCH_AUDIT_20260615.md`. **No training is executed. No Slurm jobs are submitted. No GPU/DCU resources are used.**

The corrective patch fixes:
- Unknown EC-4 rows now treated as a proper bucket (per teacher R3 pseudocode) instead of receiving a flat `weight=1.0`
- `visualize_four_modal` warning wording softened from "failed" to "could not complete"
- Import verification (confirmed already present)
- Stage 2 checkpoint save logic verification (confirmed correct)

---

## 2. Corrective Changes

| # | Change | Status | Lines touched | Notes |
|---|--------|--------|---------------|-------|
| 1 | `UNKNOWN_EC4_BUCKET` sentinel constant added | ✅ Applied | L290 (module-level) | `__unknown_ec4__` |
| 2 | Unknown EC-4 rows now use `UNKNOWN_EC4_BUCKET` instead of `None` | ✅ Applied | L307-309 | `if parsed == "unknown": parsed = UNKNOWN_EC4_BUCKET` |
| 3 | Group size computation now includes unknown bucket | ✅ Applied | L313-314 | Removed `if k is not None` guard |
| 4 | Per-sample weight now uses `1/sqrt(group_size)` for all buckets | ✅ Applied | L318-319 | Removed `else: weights.append(1.0)` branch |
| 5 | `int(global_i)` for numpy int64 safety | ✅ Applied | L305 | `metadata[int(global_i)]` |
| 6 | Visualize warning wording | ✅ Applied | L1673 | "failed" → "could not complete" |
| 7 | `import math` confirmed present | ✅ Verified | L25 | Already imported (no change needed) |
| 8 | `from collections import defaultdict` confirmed | ✅ Verified | L27 | Already imported (no change needed) |
| 9 | Stage 2 checkpoint save logic confirmed correct | ✅ Verified | L787-801 | See Section 4.2 below |

**Total lines touched**: +16 added, -14 removed = **30 lines** (corrective only)

---

## 3. Corrective Diff

Full diff saved to: `R3_TRAIN_PATCH_CORRECTIVE_DIFF_20260615.diff` (59 lines)

```diff
--- train.py.before_r3_corrective_patch_20260615
+++ train.py

+# Sentinel bucket key for rows with unknown / invalid EC-4 classification
+UNKNOWN_EC4_BUCKET = "__unknown_ec4__"

 def build_ec4_weighted_sampler(metadata, subset_indices):
     """Build WeightedRandomSampler with weight = 1/sqrt(EC-4 group size)."""
-    # Parse EC-4 for subset rows only
+    # Parse EC-4 for subset rows; unknown rows are bucketed together
     ec4_keys = []
     for global_i in subset_indices:
-        ec = metadata[global_i].get("ec_number", "")
+        ec = metadata[int(global_i)].get("ec_number", "")
         parsed = _parse_ec4(ec)
-        ec4_keys.append(parsed if parsed != "unknown" else None)
+        if parsed == "unknown":
+            parsed = UNKNOWN_EC4_BUCKET
+        ec4_keys.append(parsed)

-    # Compute group sizes (over subset only)
+    # Compute group sizes over the full subset (including unknown bucket)
     group_sizes = defaultdict(int)
     for k in ec4_keys:
-        if k is not None:
-            group_sizes[k] += 1
+        group_sizes[k] += 1

-    # Per-sample weight: 1/sqrt(group_size), unknown gets 1.0
+    # Per-sample weight: 1/sqrt(group_size) for all buckets (including unknown)
     weights = []
     for k in ec4_keys:
-        if k is not None and k in group_sizes:
-            weights.append(1.0 / math.sqrt(group_sizes[k]))
-        else:
-            weights.append(1.0)
+        weights.append(1.0 / math.sqrt(group_sizes[k]))

     return WeightedRandomSampler(...)

@@ -1668,7 +1670,7 @@
     except Exception as e:
-        print(f"  WARNING: visualize_four_modal failed: {e}")
+        print(f"  WARNING: visualize_four_modal could not complete: {e}")
```

---

## 4. Verification

### 4.1 py_compile result

```bash
$ /public/home/acfbwjsi7s/miniconda3/envs/nis/bin/python -m py_compile train.py
py_compile PASSED
```

### 4.2 grep / rg checks

```
=== UNKNOWN_EC4_BUCKET ===
290:UNKNOWN_EC4_BUCKET = "__unknown_ec4__"
309:            parsed = UNKNOWN_EC4_BUCKET

=== math.sqrt ===
320:        weights.append(1.0 / math.sqrt(group_sizes[k]))

=== visualize_four_modal could not complete ===
1673:        print(f"  WARNING: visualize_four_modal could not complete: {e}")

=== model_v3_stage2.pt ===
1581:        stage2_ckpt = out / "model_v3_stage2.pt"

=== shutil.copy2 ===
1584:            shutil.copy2(str(stage2_ckpt), str(stage3_ckpt))

=== stage2 checkpoint save logic ===
801:            print(f"  Stage {stage} checkpoint saved to {ckpt_path}")
```

All checks pass.

### 4.3 Stage 2 checkpoint save logic explanation

**Checkpoint save logic location**: `train.py` L787-801 (inside `train_four_stages`, at end of each epoch):

```python
# ── R2: save stage-end checkpoint ──
next_stage = get_stage(epoch + 1) if epoch + 1 < total_epochs else stage + 1
if next_stage != stage and output_dir is not None:
    ckpt_path = Path(output_dir) / f"model_v3_stage{stage}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        ...
        "stage": stage,
    }, ckpt_path)
    print(f"  Stage {stage} checkpoint saved to {ckpt_path}")
```

**Why `model_v3_stage2.pt` is guaranteed to exist with R3 config (epochs_stage3=0, total=38):**

| Epoch | Stage | `next_stage` (epoch+1) | Checkpoint saved? |
|-------|-------|------------------------|-------------------|
| 0–3 | 0 | 0 (epoch 1–4 still stage 0) | No |
| **4** | 0 | `get_stage(5)` = 1 | **Yes → `model_v3_stage0.pt`** |
| 5–28 | 1 | varies | No (unless boundary) |
| **29** | 1 | `get_stage(30)` = 2 | **Yes → `model_v3_stage1.pt`** |
| 30–36 | 2 | 2 | No |
| **37** (last epoch) | 2 | `epoch+1=38 == total_epochs` → `stage + 1 = 3` | **Yes → `model_v3_stage2.pt`** |

At epoch 37 (the final epoch of stage 2), the condition `epoch + 1 < total_epochs` is `38 < 38 = False`, so the else branch executes: `next_stage = stage + 1 = 3`. Since `3 != 2`, the checkpoint is saved. The `shutil.copy2` alias at L1578-1584 runs **after** `train_four_stages` returns, so `model_v3_stage2.pt` is guaranteed to exist before the copy.

---

## 5. Declarations

- **train.py modified**: yes
- **corrective patch applied**: yes
- **py_compile run**: yes
- **py_compile passed**: yes
- **Slurm submitted**: no
- **GPU/DCU used**: no
- **retraining executed**: no
- **ready for local corrective patch audit**: yes

---

## 6. File Manifest

| File | Path |
|------|------|
| Patched file | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/train.py` |
| Pre-corrective backup | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/train.py.before_r3_corrective_patch_20260615` |
| Pre-initial-patch backup | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/train.py.before_r3_patch_20260615` |
| Corrective diff | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs/R3_TRAIN_PATCH_CORRECTIVE_DIFF_20260615.diff` |
| Initial patch audit | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs/R3_TRAIN_PATCH_AUDIT_20260615.md` |
| Initial patch diff | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs/R3_TRAIN_PATCH_DIFF_20260615.diff` |

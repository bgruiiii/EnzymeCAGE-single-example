# R3 Train Patch Audit

## 1. Purpose

This step only patches `train.py` with 4 focused R3 changes. **No training is executed. No Slurm jobs are submitted. No GPU/DCU resources are used.**

The patches implement:
1. Config epoch adjustments per teacher R3 spec
2. EC-4 class-balanced `WeightedRandomSampler` for stage 1/2
3. Stage 3 skip + checkpoint alias
4. `visualize_four_modal` try/except wrapper

---

## 2. Files Modified

| File | Path |
|------|------|
| Patched file | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/train.py` |
| Backup (before patch) | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/train.py.before_r3_patch_20260615` |
| Diff file | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs/R3_TRAIN_PATCH_DIFF_20260615.diff` |
| This audit | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs/R3_TRAIN_PATCH_AUDIT_20260615.md` |

File sizes:
- `train.py` (patched): 74K
- `train.py.before_r3_patch_20260615` (backup): 71K

---

## 3. Patch Summary

| Change | Status | Lines touched | Notes |
|--------|--------|---------------|-------|
| Config `epochs_stage0 = 5` | ✅ Applied | L73 | Reduced from 8 to 5 per teacher R3 spec |
| Config `epochs_stage3 = 0` | ✅ Applied | L76 | Skip stage 3 (was 10 in R2) |
| `import WeightedRandomSampler` | ✅ Applied | L38 | Added to torch.utils.data import |
| `import shutil` | ✅ Applied | L39 | For stage3 checkpoint alias copy |
| `build_ec4_weighted_sampler` helper | ✅ Applied | L289-326 (38 lines) | New function: weight = 1/sqrt(EC-4 group size) |
| Stage 0 uniform loader | ✅ Applied | L1530 | `shuffle=True`, no sampler |
| Stage 1/2 weighted loader | ✅ Applied | L1537-1538 | `sampler=weighted_sampler`, no shuffle |
| `train_four_stages` dual-loader signature | ✅ Applied | L627 | `uniform_loader, weighted_loader` params |
| `train_four_stages` loader switching | ✅ Applied | L657-671 (15 lines) | Switches at stage boundary with print message |
| Stage 3 alias (`shutil.copy2`) | ✅ Applied | L1577-1585 (9 lines) | Copies stage2 → stage3 when epochs_stage3=0 |
| `visualize_four_modal` try/except | ✅ Applied | L1667-1675 | Catches exception, prints WARNING, does not exit |

**Total lines touched**: +92 added, -10 removed = **102 lines**

---

## 4. Focused Diff

Full diff saved to: `R3_TRAIN_PATCH_DIFF_20260615.diff`

### Key changes:

```diff
# Config (L72-76)
-    epochs_stage0 = 8           # Stage 0: independent pretrain (full-data: 8, was 20)
+    epochs_stage0 = 5           # R3: reduced from 8 to 5 (R2 was 8)
     epochs_stage1 = 25          # Stage 1: pairwise contrastive (R2: 25, R1 was 12)
     epochs_stage2 = 8           # Stage 2: triplet consistency (full-data: 8, was 30)
-    epochs_stage3 = 10          # Stage 3: closed-loop self-bootstrap (README no suggestion, kept 10)
+    epochs_stage3 = 0           # R3: skip stage 3 (was 10 in R2)

# New helper function (L289-326)
+def build_ec4_weighted_sampler(metadata, subset_indices):
+    """Build WeightedRandomSampler with weight = 1/sqrt(EC-4 group size)."""
+    ec4_keys = []
+    for global_i in subset_indices:
+        ec = metadata[global_i].get("ec_number", "")
+        parsed = _parse_ec4(ec)
+        ec4_keys.append(parsed if parsed != "unknown" else None)
+    group_sizes = defaultdict(int)
+    for k in ec4_keys:
+        if k is not None:
+            group_sizes[k] += 1
+    weights = []
+    for k in ec4_keys:
+        if k is not None and k in group_sizes:
+            weights.append(1.0 / math.sqrt(group_sizes[k]))
+        else:
+            weights.append(1.0)
+    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

# train_four_stages dual-loader (L627, L657-671)
-def train_four_stages(model, loader, cfg: Config, device: str, ...):
+def train_four_stages(model, uniform_loader, weighted_loader, cfg: Config, device: str, ...):
+    loader = uniform_loader
+    current_stage = -1
     for epoch in range(total_epochs):
         stage = get_stage(epoch)
+        if stage != current_stage:
+            if stage in (1, 2):
+                loader = weighted_loader
+                if epoch == cfg.epochs_stage0:
+                    print(f"  Stage {stage}: EC-4 weighted sampler activated")
+            else:
+                loader = uniform_loader
+            current_stage = stage

# Main loader creation (L1530-1538)
+    uniform_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
+    weighted_sampler = build_ec4_weighted_sampler(metadata, subset_indices)
+    weighted_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=weighted_sampler)

# Stage 3 alias (L1577-1585)
+    if cfg.epochs_stage3 == 0:
+        stage2_ckpt = out / "model_v3_stage2.pt"
+        stage3_ckpt = out / "model_v3_stage3.pt"
+        if stage2_ckpt.exists():
+            shutil.copy2(str(stage2_ckpt), str(stage3_ckpt))
+            print(f"  Stage 3 skipped (epochs_stage3=0); "
+                  f"model_v3_stage3.pt is alias of stage2 checkpoint")

# Visualize try/except (L1667-1675)
+    try:
         visualize_four_modal(metrics, all_r, all_e, all_s, all_m, labels, history, out)
+    except Exception as e:
+        print(f"  WARNING: visualize_four_modal failed: {e}")
+        print(f"  PNG visualization skipped; model/embeddings/metrics/index outputs are unaffected.")
```

---

## 5. Verification

### py_compile

```bash
$ /public/home/acfbwjsi7s/miniconda3/envs/nis/bin/python -m py_compile train.py
py_compile PASSED
```

### grep checks

```
=== epochs_stage0 ===
73:    epochs_stage0 = 5           # R3: reduced from 8 to 5 (R2 was 8)

=== epochs_stage1 ===
74:    epochs_stage1 = 25          # Stage 1: pairwise contrastive (R2: 25, R1 was 12)

=== epochs_stage2 ===
75:    epochs_stage2 = 8           # Stage 2: triplet consistency (full-data: 8, was 30)

=== epochs_stage3 ===
76:    epochs_stage3 = 0           # R3: skip stage 3 (was 10 in R2)

=== hard_neg_weight ===
65:    hard_neg_weight = 1.0       # R2: disable same-EC hard-negative upweighting (R1 was 2.0)

=== WeightedRandomSampler ===
38:from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
290:    """Build WeightedRandomSampler with weight = 1/sqrt(EC-4 group size).
320:    return WeightedRandomSampler(

=== visualize_four_modal ===
1668:        visualize_four_modal(metrics, all_r, all_e, all_s, all_m,
1671:        print(f"  WARNING: visualize_four_modal failed: {e}")

=== shutil.copy2 ===
1582:            shutil.copy2(str(stage2_ckpt), str(stage3_ckpt))
```

**All checks pass.**

---

## 6. Non-Changes

The following are explicitly **NOT** modified by this patch:

- `hard_neg_weight` = 1.0 (unchanged)
- `epochs_stage1` = 25 (unchanged)
- `epochs_stage2` = 8 (unchanged)
- `batch_size` = 4096 (unchanged)
- EC encoding logic (`_parse_ec4`, `encode_ec_features`) — unchanged
- Concept loss / VICReg loss — unchanged
- Calibration / OOD / latency thresholds — not introduced
- Row-level objective (per-row contrastive loss) — unchanged
- `MultiModalDataset.__getitem__` — unchanged
- `load_enzyme_cage_300` — unchanged
- `evaluate_grouped_re` — unchanged
- `compute_metrics` — unchanged
- No unrelated refactor
- No other files modified

---

## 7. Declarations

- **train.py modified**: yes
- **patch applied**: yes
- **py_compile run**: yes
- **py_compile passed**: yes
- **Slurm submitted**: no
- **GPU/DCU used**: no
- **retraining executed**: no
- **ready for local patch audit**: yes

---

## 8. Epoch Configuration Summary

| Stage | R2 Value | R3 Value | Changed? |
|-------|----------|----------|----------|
| Stage 0 (pretrain) | 8 | **5** | ✅ Yes |
| Stage 1 (pairwise) | 25 | 25 | ❌ No |
| Stage 2 (triplet) | 8 | 8 | ❌ No |
| Stage 3 (self-bootstrap) | 10 | **0** | ✅ Yes |
| **Total** | **51** | **38** | — |

---

## 9. Loader Strategy Summary

| Stage | Loader Type | Sampling Strategy |
|-------|-------------|-------------------|
| Stage 0 | `uniform_loader` | `shuffle=True`, uniform random |
| Stage 1 | `weighted_loader` | `WeightedRandomSampler`, weight = 1/√(EC-4 group size) |
| Stage 2 | `weighted_loader` | `WeightedRandomSampler`, weight = 1/√(EC-4 group size) |
| Stage 3 | (skipped) | N/A (epochs_stage3 = 0) |

---

## 10. Diff Statistics

```
--- train.py.before_r3_patch_20260615
+++ train.py
+92 lines added
-10 lines removed
= 102 lines touched (actual)
```

Diff file: `R3_TRAIN_PATCH_DIFF_20260615.diff` (180 lines including headers)

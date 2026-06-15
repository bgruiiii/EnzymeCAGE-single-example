# R3 Train Patch Pre-Audit

## 1. Purpose

This document is a **focused patch plan** for the R3 modifications to `train.py`. It identifies the exact source locations, current code, and proposed changes for the three R3 patch categories:

1. EC-4 class-balanced `WeightedRandomSampler` (stage1 / stage2 only)
2. Stage 3 skip (`epochs_stage3 = 0`)
3. `visualize_four_modal` try/except wrapper

**No code is modified in this step.** This is a read-only audit for review before patching.

---

## 2. Source File

| Attribute | Value |
|-----------|-------|
| Path | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/train.py` |
| Size | 71K |
| Lines | 1614 (wc -l reports 1614, file has lines 1–1615 with final newline) |
| mtime | Jun 11 15:15 |
| SHA256 | (recorded in R3 prerequisite audit) |

---

## 3. Current Config Values

Source: `class Config` at lines 45–81.

| Field | Current Value | Line | R3 Change? |
|-------|---------------|------|------------|
| `hard_neg_weight` | `1.0` | 64 | **no** |
| `epochs_stage0` | `8` | 72 | no |
| `epochs_stage1` | `25` | 73 | **no** |
| `epochs_stage2` | `8` | 74 | **no** |
| `epochs_stage3` | `10` | 75 | **yes → 0** |
| `batch_size` | `4096` | 76 | no |
| `unified_dim` | `256` | 47 | no |
| `lr` | `3e-4` | 71 | no |
| `temp_start` / `temp_end` | `0.5` / `0.05` | 51–52 | no |
| `vicreg_var_weight` | `10.0` | 55 | no |
| `vicreg_cov_weight` | `1.0` | 56 | no |
| `w_re` / `w_em` / `w_sm` | `1.0` / `0.7` / `0.5` | 59–61 | no |
| `seed` | `42` | 77 | no |
| `total_epochs` (property) | `8+25+8+10 = 51` | 79–81 | auto-updates to `8+25+8+0 = 41` |

### Checkpoint save logic (current)

- Stage-end checkpoints: lines 731–743 inside `train_four_stages`
  - `model_v3_stage{stage}.pt` saved at stage transition boundaries
  - Triggered by `next_stage != stage` check
- Final checkpoint: line 1512–1518 in `main()`
  - `model_v3.pt` saved after training completes, before evaluation

---

## 4. Current Training Flow

### 4.1 Stage execution

| Stage | Lines (in `train_four_stages`) | Epoch range (current) | Description |
|-------|-------------------------------|----------------------|-------------|
| Stage 0 | 628–635 | epochs 0–7 | Independent projector warmup (VICReg variance only) |
| Stage 1 | 637–657 | epochs 8–32 | Pairwise contrastive (R↔E + E↔M + VICReg) |
| Stage 2 | 659–683 | epochs 33–40 | Triplet contrastive (R↔E + E↔M + S↔M + anchors + VICReg) |
| Stage 3 | 685–713 | epochs 41–50 | Closed-loop self-bootstrap (FBA surrogate pseudo-labels) |

### 4.2 `get_stage()` helper (lines 606–614)

```python
def get_stage(epoch):
    if epoch < cfg.epochs_stage0:
        return 0
    elif epoch < cfg.epochs_stage0 + cfg.epochs_stage1:
        return 1
    elif epoch < cfg.epochs_stage0 + cfg.epochs_stage1 + cfg.epochs_stage2:
        return 2
    else:
        return 3
```

### 4.3 DataLoader creation (in `main()`, lines 1468–1474)

```python
if train_idx is not None:
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
else:
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
```

- **One loader for all stages** — single `train_loader` created in `main()` and passed to `train_four_stages`
- `shuffle=True`, no sampler
- `batch` unpacking (line 622): `r, e, s, m, concepts, ec_ids, indices = batch`
- `ec_ids` in batch = EC-1 level integer (line 271–275), NOT EC-4

### 4.4 Stage selection CLI (lines 1428–1434)

```python
if args.stage != "all":
    stage_num = int(args.stage)
    cfg.epochs_stage0 = cfg.epochs_stage0 if stage_num >= 0 else 0
    cfg.epochs_stage1 = cfg.epochs_stage1 if stage_num >= 1 else 0
    cfg.epochs_stage2 = cfg.epochs_stage2 if stage_num >= 2 else 0
    cfg.epochs_stage3 = cfg.epochs_stage3 if stage_num >= 3 else 0
```

---

## 5. Required Patch Locations

| # | Change | Function / Class | Approx Lines | Current Code Summary | Proposed Patch Summary |
|---|--------|-----------------|--------------|---------------------|----------------------|
| 1 | **`_parse_ec4` for sampler** (reuse existing) | `_parse_ec4()` | 1264–1276 | Already exists: returns `"x.x.x.x"` or `"unknown"` | No change needed; reuse for sampler weight computation |
| 2 | **`build_ec4_weighted_sampler` helper** | New function, insert after `MultiModalDataset` | After line 285 | Does not exist | New ~25-line function: parse EC-4 from `metadata`, compute group sizes, weight = `1/sqrt(group_size)`, return `WeightedRandomSampler(replacement=True)` |
| 3 | **`train_four_stages` signature** | `train_four_stages()` | 588–590 | `def train_four_stages(model, loader, cfg, device, ...)` | Add `metadata=None, train_dataset=None, batch_size=None` params for stage-specific loader rebuild |
| 4 | **Stage1/2 DataLoader rebuild** | `train_four_stages()` inner loop | Around 616 (before epoch loop) | Single loader for all stages | At stage boundary: rebuild `loader` with `WeightedRandomSampler` for stage 1/2, uniform for stage 0. Detect stage change inside epoch loop and swap loader. |
| 5 | **`epochs_stage3 = 0`** | `Config` | 75 | `epochs_stage3 = 10` | Change to `epochs_stage3 = 0` |
| 6 | **Stage 3 skip: alias checkpoint** | `main()`, after `train_four_stages` returns | After line 1507 | No stage3 skip handling | If `cfg.epochs_stage3 == 0`: copy `model_v3_stage2.pt` → `model_v3_stage3.pt` (alias) with a print warning |
| 7 | **`visualize_four_modal` try/except** | `main()`, call site | 1590–1591 | Direct call, no error handling | Wrap in `try/except Exception as e: print(f"WARNING: ...")` |

### Detailed patch descriptions

#### Patch 2: `build_ec4_weighted_sampler` (new function, ~25 lines)

```python
def build_ec4_weighted_sampler(metadata, indices=None):
    """Build WeightedRandomSampler with weight = 1/sqrt(EC-4 group size).
    
    Rows with unknown EC-4 get weight = 1.0 (neutral).
    """
    from torch.utils.data import WeightedRandomSampler
    
    # Parse EC-4 for each row
    ec4_keys = []
    for m in metadata:
        ec = m.get("ec_number", "")
        parsed = _parse_ec4(ec)
        ec4_keys.append(parsed if parsed != "unknown" else None)
    
    # Compute group sizes
    from collections import Counter
    group_sizes = Counter(k for k in ec4_keys if k is not None)
    
    # Compute per-row weight
    weights = []
    for k in ec4_keys:
        if k is not None and k in group_sizes:
            weights.append(1.0 / math.sqrt(group_sizes[k]))
        else:
            weights.append(1.0)
    
    # Filter to train indices if provided
    if indices is not None:
        weights = [weights[i] for i in indices]
    
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
```

#### Patch 4: Stage-specific loader swap inside `train_four_stages`

The current architecture passes a single `loader`. For R3, we need:
- Stage 0: uniform sampling (`shuffle=True`, no sampler)
- Stage 1–2: EC-4 weighted sampling

**Approach**: detect stage transitions inside the epoch loop and rebuild the loader.

```python
# Before epoch loop, store dataset reference
current_stage = -1

for epoch in range(total_epochs):
    stage = get_stage(epoch)
    
    # Rebuild loader at stage boundary if sampler needs to change
    if stage != current_stage and train_dataset is not None:
        if stage in (1, 2) and metadata is not None:
            sampler = build_ec4_weighted_sampler(metadata, train_indices)
            loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
            print(f"  Stage {stage}: EC-4 weighted sampler activated")
        else:
            loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            if stage == 0:
                print(f"  Stage {stage}: uniform sampling")
        current_stage = stage
    
    # ... rest of epoch loop
```

This requires `train_four_stages` to receive `train_dataset`, `metadata`, and `train_indices` as additional parameters.

#### Patch 6: Stage 3 alias checkpoint (option A)

**Recommendation: Option A** — save `model_v3_stage3.pt` as a copy/alias of `model_v3_stage2.pt`.

**Rationale**:
- Downstream eval scripts (including `eval_ec4_buckets.py` and the existing `postmortem_eval_stage_checkpoints.py`) expect `model_v3_stage3.pt` to exist
- Avoids special-casing "skipped" in every downstream consumer
- A simple `shutil.copy2` is cheap (19MB) and explicit
- The R3 prerequisite audit already lists `model_v3_stage3.pt` as a required artifact

```python
# In main(), after train_four_stages returns:
if cfg.epochs_stage3 == 0:
    stage2_ckpt = out / "model_v3_stage2.pt"
    stage3_ckpt = out / "model_v3_stage3.pt"
    if stage2_ckpt.exists():
        import shutil
        shutil.copy2(str(stage2_ckpt), str(stage3_ckpt))
        print(f"  Stage 3 skipped (epochs_stage3=0); "
              f"model_v3_stage3.pt = alias of stage2 checkpoint")
```

#### Patch 7: `visualize_four_modal` try/except

```python
# Line 1590-1591 in main():
try:
    visualize_four_modal(metrics, all_r, all_e, all_s, all_m,
                         labels, history, out)
except Exception as e:
    print(f"  WARNING: visualize_four_modal failed: {e}")
    print(f"  PNG visualization skipped; model/embeddings/metrics/index "
          f"outputs are unaffected.")
```

---

## 6. Risk Check

### Things that will NOT change:

| Item | Status |
|------|--------|
| `hard_neg_weight` | **unchanged** (1.0) |
| `epochs_stage1` | **unchanged** (25) |
| `epochs_stage2` | **unchanged** (8) |
| `batch_size` | **unchanged** (4096) |
| EC encoding (`ec_ids` in batch) | **unchanged** (EC-1 integer for InfoNCE hard-neg) |
| `infonce_loss` function | **unchanged** |
| Concept loss / VICReg | **unchanged** |
| Calibration / OOD / latency | **not introduced** |
| Row-level R→E objective | **unchanged** |
| `evaluate_multimodal` / `evaluate_grouped_re` | **unchanged** |
| FAISS index / NN index saving | **unchanged** |
| `model_v3.pt` / `embeddings_v3.npz` / `metrics_v3.json` saving | **unchanged** |
| `load_enzyme_cage_300` | **unchanged** |
| `UnifiedSpace` model architecture | **unchanged** |
| Unrelated refactor | **none** |

### New import required

- `from torch.utils.data import WeightedRandomSampler` — add to line 38 (already imports `DataLoader, Dataset`)
- `import shutil` — only used if stage3 alias is implemented (can be inline import)

---

## 7. Proposed Diff Size Estimate

| Patch | Lines added | Lines removed | Net |
|-------|-------------|---------------|-----|
| 2. `build_ec4_weighted_sampler` | ~25 | 0 | +25 |
| 3. `train_four_stages` signature | ~3 | ~1 | +2 |
| 4. Loader rebuild logic | ~15 | ~2 | +13 |
| 5. `epochs_stage3 = 0` | 1 | 1 | 0 |
| 6. Stage 3 alias checkpoint | ~6 | 0 | +6 |
| 7. `visualize_four_modal` try/except | ~5 | ~2 | +3 |
| Import line update | ~1 | ~1 | 0 |
| **Total** | **~56** | **~7** | **~49** |

**Estimated net diff: ~49 lines, well under the 80-line target.**

---

## 8. Declarations

- train.py modified: **no**
- patch applied: **no**
- py_compile run: **no**
- Slurm submitted: **no**
- GPU/DCU used: **no**
- retraining executed: **no**
- ready for local patch-plan audit: **yes**

---

*Patch pre-audit performed: 2026-06-15 | Read-only, no modifications made*

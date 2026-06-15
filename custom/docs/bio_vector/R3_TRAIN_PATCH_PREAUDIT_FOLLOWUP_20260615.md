# R3 Train Patch Pre-Audit — Follow-Up

## 1. Purpose

This follow-up audit addresses two issues raised by local audit review of [R3_TRAIN_PATCH_PREAUDIT_20260615.md](R3_TRAIN_PATCH_PREAUDIT_20260615.md):

1. **Config mismatch**: Does R2 actually use `epochs_stage0 = 8` or `5`? Should R3 patch set it to `5`?
2. **WeightedRandomSampler Subset mapping**: How to correctly build per-sample weights when `train_dataset` is a `torch.utils.data.Subset`?

**No code is modified in this step.** This is a read-only follow-up audit.

---

## 2. Config Mismatch Audit: `epochs_stage0`

### 2.1 Current `train.py` Config (source of truth for R2)

File: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/train.py`

```
L72: epochs_stage0 = 8           # Stage 0: independent pretrain (full-data: 8, was 20)
L73: epochs_stage1 = 25          # Stage 1: pairwise contrastive (R2: 25, R1 was 12)
L74: epochs_stage2 = 8           # Stage 2: triplet consistency (full-data: 8, was 30)
L75: epochs_stage3 = 10          # Stage 3: closed-loop self-bootstrap (README no suggestion, kept 10)
L76: batch_size = 4096
```

### 2.2 Evidence from `training_history.json`

File: `R2_OUT/training_history.json`

```
Total epochs: 51
Stage 0: 8 epochs   (epochs 1–8,   indices 0–7)
Stage 1: 25 epochs  (epochs 9–33,  indices 8–32)
Stage 2: 8 epochs   (epochs 34–41, indices 33–40)
Stage 3: 10 epochs  (epochs 42–51, indices 41–50)

First loss: 2.815797, Last loss: 44.676906
Temp range: 0.5000 → 0.0500
```

**JSON key**: `training_history.json → "stage"` — array of 51 integers: `[0,0,0,0,0,0,0,0, 1,1,...(×25), 2,2,...(×8), 3,3,...(×10)]`

### 2.3 Evidence from R2 stdout log

File: `R2_OUT/r2_train_115204112.out`

```
L27:   Total epochs: 8+25+8+10 = 51
L50: Starting 4-stage training (51 total epochs)
L51:   Stage 0 (pretrain):         8 epochs
L52:   Stage 1 (pairwise):         25 epochs
L53:   Stage 2 (triplet+anchor):   8 epochs
L54:   Stage 3 (self-bootstrap):   10 epochs
```

### 2.4 Evidence from R2 run script

File: `code/demo/run_r2_training.sh`

```
L73: echo "  Total epochs: 8+25+8+10 = 51"
```

### 2.5 Evidence from checkpoint config (binary extraction)

File: `R2_OUT/model_v3_stage0.pt` (pickle binary, extracted without torch)

```
epochs_stage0 → K\x08  (INT1 = 8)
epochs_stage1 → K\x19  (INT1 = 25)
epochs_stage2 → K\x08  (INT1 = 8)
epochs_stage3 → K\n    (INT1 = 10)
batch_size    → M\x00\x10 (INT2 = 4096)
hard_neg_weight → G?\xf0... (FLOAT64 = 1.0)
```

### 2.6 Evidence from R2 Plan document

File: `docs/R2_PLAN_v2_REVISED_20260609_175634.md`

```
L149: epochs_stage0 = 8
L150: epochs_stage2 = 8
L151: epochs_stage3 = 10
L261: "batch size, learning rate, or stage 0/2/3 epochs" — explicitly listed as unchanged
```

### 2.7 Evidence from R2 pre-submit audit diff

File: `docs/R2_FINAL_PRE_SUBMIT_AUDIT_20260611.md`

```diff
L64: epochs_stage0 = 8           # unchanged from R1 ESM-C baseline
L65: -    epochs_stage1 = 12     # (was full-data: 12, was 40)
L66: +    epochs_stage1 = 25     # (R2: 25, R1 was 12)
L67: epochs_stage2 = 8           # unchanged
L68: epochs_stage3 = 10          # unchanged
```

The only R2-approved Config change was `epochs_stage1: 12 → 25` and `hard_neg_weight: 2.0 → 1.0`. Stage 0/2/3 epochs were explicitly listed as **do not patch** (R2 Plan §5 L261).

### 2.8 Conclusion

| Parameter | R2 Actual | Current train.py | R2 Plan | Match? |
|-----------|-----------|-------------------|---------|--------|
| `epochs_stage0` | **8** | 8 | 8 | ✅ consistent |
| `epochs_stage1` | **25** | 25 | 25 | ✅ consistent |
| `epochs_stage2` | **8** | 8 | 8 | ✅ consistent |
| `epochs_stage3` | **10** | 10 | 10 | ✅ consistent |

**Current `train.py` has NOT deviated from the R2 accepted config.** All 5 evidence sources (source code, training history, stdout log, run script, checkpoint binary, R2 plan) agree on `epochs_stage0 = 8`.

**R3 patch recommendation for `epochs_stage0`**:

The teacher's R3 document states `epochs_stage0 = 5`, but R2 actually trained with `8`. This is a **deliberate R3 change**, not a correction. The R3 patch should:

- Change `epochs_stage0` from `8` to `5` as a new R3 modification
- Update the comment to document the change: `# R3: reduced from 8 to 5`
- This adds 1 line to the diff (Config change)

**Final R3 Config should be**:

```python
epochs_stage0 = 5    # R3: reduced from R2 value of 8
epochs_stage1 = 25   # unchanged
epochs_stage2 = 8    # unchanged
epochs_stage3 = 0    # R3: skip stage 3
```

Total epochs: 5 + 25 + 8 + 0 = **38** (down from R2's 51).

---

## 3. WeightedRandomSampler Subset Mapping Audit

### 3.1 Current Code Analysis

#### `MultiModalDataset.__getitem__` (L283–285)

```python
def __getitem__(self, idx):
    return (self.r[idx], self.e[idx], self.s[idx], self.m[idx],
            self.concepts[idx], self.ec_ids[idx], idx)
```

**Yes, it returns the raw `idx` as the 7th element.** When called through a `Subset`, `idx` is the **global dataset index**, not the Subset-local index. This is because `Subset.__getitem__` does:

```python
# torch.utils.data.Subset internals
def __getitem__(self, idx):
    return self.dataset[self.indices[idx]]
```

So `dataset.__getitem__` receives `self.indices[idx]` (the global index), and returns that global index in the batch tuple.

#### `train_idx` type (L1214–1217)

```python
indices = np.arange(n)
train_idx, test_idx = train_test_split(
    indices, test_size=test_size, random_state=42)
```

`train_idx` is a **numpy ndarray** of `int64`, returned by `sklearn.model_selection.train_test_split`.

#### DataLoader creation (L1468–1474)

```python
if train_idx is not None:
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
else:
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
```

### 3.2 Subset Indexing Semantics

When iterating over `DataLoader(train_dataset)` where `train_dataset = Subset(dataset, train_idx)`:

1. DataLoader generates local indices `j ∈ [0, len(train_idx) - 1]`
2. Subset maps `j → train_idx[j]` (global index)
3. `dataset.__getitem__(train_idx[j])` returns `(r[train_idx[j]], ..., train_idx[j])`
4. The 7th batch element `indices` contains **global** indices

**Key implication**: The DataLoader iteration order follows the **local** order `[0, 1, ..., len(train_idx)-1]`, but the data accessed and the `idx` returned are **global** indices.

### 3.3 WeightedRandomSampler Requirements

`WeightedRandomSampler(weights, num_samples, replacement)` operates on **local dataset indices** `[0, len(dataset) - 1]`.

When the dataset is a `Subset`:
- `len(dataset) == len(train_idx)` (Subset length)
- `weights` must have length `== len(train_idx)`
- `weights[j]` corresponds to `train_dataset[j]`, which accesses `dataset[train_idx[j]]`

### 3.4 Correct Patch Recommendation

```python
def build_ec4_weighted_sampler(metadata, subset_indices):
    """Build WeightedRandomSampler for EC-4 class-balanced training.

    Args:
        metadata: list of dicts with 'ec_number' field (full dataset)
        subset_indices: numpy array or list of global indices for the train subset

    Returns:
        WeightedRandomSampler with len(subset_indices) weights
    """
    from torch.utils.data import WeightedRandomSampler
    from collections import Counter

    # Parse EC-4 for subset rows only
    ec4_keys = []
    for global_i in subset_indices:
        ec = metadata[global_i].get("ec_number", "")
        parsed = _parse_ec4(ec)
        ec4_keys.append(parsed if parsed != "unknown" else None)

    # Compute group sizes (over subset only — this is the sampling distribution)
    group_sizes = Counter(k for k in ec4_keys if k is not None)

    # Per-sample weight: 1/sqrt(group_size), unknown gets 1.0
    weights = []
    for k in ec4_keys:
        if k is not None and k in group_sizes:
            weights.append(1.0 / math.sqrt(group_sizes[k]))
        else:
            weights.append(1.0)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
```

**DataLoader changes**:

```python
# When using sampler, MUST NOT use shuffle=True (they are mutually exclusive)
if train_idx is not None:
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    sampler = build_ec4_weighted_sampler(metadata, train_idx)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
else:
    # no_test_split: full dataset, build sampler over all indices
    sampler = build_ec4_weighted_sampler(metadata, list(range(len(dataset))))
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler)
```

### 3.5 Stage-Specific Loader Strategy

Since stage 0 should NOT use weighted sampling but stage 1/2 should:

**Option**: Build two loaders in `main()`, pass both to `train_four_stages`, swap at stage boundary.

```python
# In main():
if train_idx is not None:
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
else:
    train_dataset = dataset

# Uniform loader (stage 0)
uniform_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

# Weighted loader (stage 1, 2)
subset_indices = train_idx if train_idx is not None else list(range(len(dataset)))
weighted_sampler = build_ec4_weighted_sampler(metadata, subset_indices)
weighted_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=weighted_sampler)

# Pass both to train_four_stages
history = train_four_stages(
    model, uniform_loader, weighted_loader, cfg, device, ...)
```

In `train_four_stages`:

```python
def train_four_stages(model, uniform_loader, weighted_loader, cfg, device, ...):
    loader = uniform_loader  # start with uniform for stage 0
    current_stage = -1

    for epoch in range(total_epochs):
        stage = get_stage(epoch)
        if stage != current_stage:
            if stage in (1, 2):
                loader = weighted_loader
            else:
                loader = uniform_loader
            current_stage = stage
        # ... rest of epoch loop using `loader`
```

---

## 4. Revised Patch Plan

### 4.1 Config Changes

| Field | R2 Value | R3 Value | Change? |
|-------|----------|----------|---------|
| `epochs_stage0` | 8 | **5** | **yes** (R3 new change) |
| `epochs_stage1` | 25 | 25 | no |
| `epochs_stage2` | 8 | 8 | no |
| `epochs_stage3` | 10 | **0** | **yes** |
| `hard_neg_weight` | 1.0 | 1.0 | no |
| `batch_size` | 4096 | 4096 | no |

### 4.2 Full Patch List

| # | Change | Location | Lines |
|---|--------|----------|-------|
| 1 | `epochs_stage0 = 5` (was 8) | Config L72 | ~1 |
| 2 | `epochs_stage3 = 0` (was 10) | Config L75 | ~1 |
| 3 | `build_ec4_weighted_sampler()` new function | After MultiModalDataset (after L285) | ~25 |
| 4 | `train_four_stages` signature: add `weighted_loader` param | L588–590 | ~3 |
| 5 | Stage-specific loader swap inside epoch loop | L616 area | ~10 |
| 6 | `main()`: build two loaders (uniform + weighted) | L1468–1474 | ~12 |
| 7 | `main()`: pass both loaders to `train_four_stages` | L1504–1507 | ~2 |
| 8 | Stage 3 alias: `shutil.copy2(stage2 → stage3)` | After L1507 | ~6 |
| 9 | `visualize_four_modal` try/except | L1590–1591 | ~5 |
| 10 | Import: `WeightedRandomSampler`, `shutil` | L38 | ~2 |
| | **Total** | | **~67** |

**Estimated net diff: ~67 lines, under the 80-line target.**

### 4.3 Stage 3 Alias

**Still recommending Option A**: `shutil.copy2(model_v3_stage2.pt → model_v3_stage3.pt)`.

Rationale unchanged: downstream scripts expect `model_v3_stage3.pt` to exist; copying is cheap (19MB) and explicit.

### 4.4 `total_epochs` Auto-Update

The `@property total_epochs` at L79–81 computes dynamically:

```python
return self.epochs_stage0 + self.epochs_stage1 + self.epochs_stage2 + self.epochs_stage3
```

With R3 values: `5 + 25 + 8 + 0 = 38`. No manual change needed. The CosineAnnealingLR and TemperatureScheduler both use `total_epochs` automatically.

**Important**: The cosine annealing schedule will now anneal over 38 epochs instead of 51. This means temperature reaches `temp_end = 0.05` faster (at epoch 38 instead of 51). This is expected R3 behavior.

---

## 5. Declarations

- train.py modified: **no**
- patch applied: **no**
- py_compile run: **no**
- Slurm submitted: **no**
- GPU/DCU used: **no**
- retraining executed: **no**
- ready for local patch-plan audit: **yes**

---

*Follow-up audit performed: 2026-06-15 | Read-only, no modifications made*

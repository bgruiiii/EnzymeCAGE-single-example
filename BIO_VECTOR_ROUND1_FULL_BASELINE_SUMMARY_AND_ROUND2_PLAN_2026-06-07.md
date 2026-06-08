# Bio Vector Full-Data Round 1 Summary And Round 2 Plan

Date: 2026-06-07

## 1. Purpose

This document summarizes the first full-data `bio_vector` baseline runs and
proposes the second-round parameter plan.

The runs follow the teacher-provided demo strategy:

- same `train.py` entry point
- same four-stage training pipeline
- same model architecture
- same loss functions
- same optimizer/scheduler logic
- same cross-modal retrieval metrics
- same GVP vs ESM-C enzyme feature comparison

The main changes made before the full runs were compatibility/scalability
patches required by the real full-data package and HPC environment. These are
documented separately in:

```text
/home/a/EnzymeCAGE/custom/docs/BIO_VECTOR_TRAIN_PATCH_AND_FULL_RUN_README_2026-06-05.md
```

## 2. Data And Run Setup

Full data:

- total examples: `145,607`
- reaction features: DRFP, shape `(145607, 2048)`
- substrate features: Morgan fingerprint, shape `(145607, 2048)`
- microbe features: structured metabolic vector, shape `(145607, 28)`
- GVP enzyme features: pooled to 50 dimensions
- ESM-C enzyme features: pooled to 1152 dimensions

HPC environment:

- partition: `kshdnormal04`
- accelerator: Hygon DCU
- job resource: 1 DCU per run
- DTK module: `compiler/dtk/23.10`
- required environment:

```bash
module load compiler/dtk/23.10
export HSA_OVERRIDE_GFX_VERSION=9.0.6
conda activate nis
```

Full-data Config used for both GVP and ESM-C:

| Parameter | Value | Basis |
|----------|------:|-------|
| `batch_size` | 4096 | teacher README full-data recommendation |
| `lr` | 3e-4 | teacher README range `1e-4` to `5e-4` |
| `epochs_stage0` | 8 | teacher README full range `5-10` |
| `epochs_stage1` | 12 | teacher README full range `10-15` |
| `epochs_stage2` | 8 | teacher README full range `5-10` |
| `epochs_stage3` | 10 | README gave no full-data value, kept demo default |
| `temp_end` | 0.05 | teacher README full range `0.05-0.07` |
| `w_sm` | 0.5 | teacher README full range `0.4-0.6` |
| `vicreg_var_weight` | 10.0 | teacher README full range `10-25` |
| `hard_neg_weight` | 2.0 | kept default, within README range |

Total epochs:

```text
300-demo default: 100 epochs
full-data round 1: 38 epochs
```

## 3. Required Compatibility Fixes

The first full runs exposed several mismatches between the small demo layout and
the actual full-data package. The fixes were limited to data loading,
evaluation scalability, and HPC artifact handling. They did not change the
training objective.

Important fixes:

1. GVP loader:
   - demo expected per-protein `.npz` GVP files
   - full package stores GVP in 192 sharded `.pt` files
   - added sharded GVP pooling to produce the original 50-dimensional GVP input

2. Microbe loader:
   - demo expected nested microbe JSON records
   - full package stores required values across three CSV tables
   - loader now reconstructs the expected 28-dimensional microbe feature vector
     by `example_id`

3. ESM-C loader:
   - ESM-C files are 107,731 unique UID files
   - optimized availability checks by pre-scanning the directory
   - preserved fallback semantics: `ESM-C -> GVP -> AAC`

4. Full-data evaluation:
   - original evaluation built full `N x N` similarity matrices
   - at `N = 145,607`, this OOMs
   - replaced with chunked evaluation while preserving metric semantics

5. Artifact saving:
   - checkpoint is now saved before evaluation
   - native FAISS failed due FAISS SWIG / NumPy incompatibility
   - saved normalized `*_nn_index.npz` fallback files instead

## 4. Validation Before Full Runs

### GVP 10k Smoke

The GVP 10k smoke verified that the microbe fix solved the all-zero microbe
problem.

Key results:

| Metric | GVP 10k smoke |
|--------|--------------:|
| `R→E top-1` | 0.0569 |
| `R→E top-10` | 0.266 |
| `R→E MRR` | 0.128 |
| `E→M MRR` | 0.205 |
| `S→M MRR` | 0.626 |

Microbe diagnostics:

```text
previous invalid full run microbe erank: 1/256
GVP 10k smoke microbe erank: 116.73/256
microbe composed path erank: 26.22/28
```

### ESM-C 10k Smoke

The ESM-C 10k smoke confirmed that ESM-C loading and the patched pipeline work.

ESM-C loading:

```text
10,000 / 10,000 loaded
fallback: 0
```

10k smoke comparison:

| Metric | GVP 10k | ESM-C 10k |
|--------|--------:|----------:|
| `R→E top-1` | 0.0569 | 0.0725 |
| `R→E top-10` | 0.266 | 0.359 |
| `R→E MRR` | 0.128 | 0.169 |
| `E→M top-1` | 0.0045 | 0.0133 |
| `E→M MRR` | 0.205 | 0.487 |
| `S→M top-1` | 0.0223 | 0.0212 |
| `S→M MRR` | 0.626 | 0.642 |

Observation:

- ESM-C was stronger than GVP on `R→E` and `E→M` in the smoke run.

## 5. Full Baseline Results

### 5.1 Full GVP Baseline

Output directory:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_gvp_baseline_microbe_fixed_2026-06-06
```

Loss snapshot:

```text
Stage 0 epoch  1: loss=2.8163  tau=0.5000
Stage 1 epoch 10: loss=40.401  tau=0.4374
Stage 1 epoch 20: loss=38.392  tau=0.2655
Stage 3 epoch 30: loss=47.373  tau=0.0999
Stage 3 epoch 38: loss~46.98 tau=0.05
```

Metrics:

| Metric | GVP full |
|--------|---------:|
| `R→E top-1` | 0.0118 |
| `R→E top-10` | 0.0671 |
| `R→E MRR` | 0.044 |
| `E→M top-1` | 0.0017 |
| `E→M MRR` | 0.395 |
| `S→M top-1` | 0.0054 |
| `S→M MRR` | 0.642 |

Lightweight effective rank:

| Modality | Effective rank | Participation ratio | dim@90 | dim@95 | dim@99 |
|----------|---------------:|--------------------:|-------:|-------:|-------:|
| reaction | 28.31 | 20.58 | 21 | 29 | 92 |
| enzyme | 30.56 | 21.22 | 22 | 40 | 136 |
| substrate | 56.58 | 35.23 | 55 | 91 | 174 |
| microbe | 38.21 | 26.05 | 31 | 46 | 108 |

Artifacts:

- `model_v3.pt`
- `embeddings_v3.npz`
- `metrics_v3.json`
- `training_history.json`
- `metadata_v3.json`
- `enzyme2microbe_index.json`
- `reaction_nn_index.npz`
- `enzyme_nn_index.npz`
- `substrate_nn_index.npz`
- `microbe_nn_index.npz`
- `effective_rank_summary_lightweight.json`

### 5.2 Full ESM-C Baseline

Output directory:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07
```

Loading:

```text
ESM-C available UIDs: 107,731 / 145,607
ESM-C loaded rows: 145,607 / 145,607
```

Note: unique UID count is lower than row count because multiple examples can
share one UniProt ID.

Loss snapshot:

```text
Stage 0 epoch  1: loss=2.8163  tau=0.5000
Stage 1 epoch 10: loss=39.460  tau=0.4374
Stage 1 epoch 20: loss=37.164  tau=0.2655
Stage 3 epoch 30: loss=45.625  tau=0.0999
Stage 3 epoch 38: loss~45.20 tau=0.05
```

Metrics:

| Metric | ESM-C full |
|--------|-----------:|
| `R→E top-1` | 0.0161 |
| `R→E top-10` | 0.0896 |
| `R→E MRR` | 0.058 |
| `E→M top-1` | 0.0036 |
| `E→M MRR` | 0.609 |
| `S→M top-1` | 0.0056 |
| `S→M MRR` | 0.611 |

Lightweight effective rank:

| Modality | Effective rank | Participation ratio | dim@90 | dim@95 | dim@99 |
|----------|---------------:|--------------------:|-------:|-------:|-------:|
| reaction | 43.72 | 32.36 | 33 | 42 | 97 |
| enzyme | 43.54 | 34.22 | 32 | 40 | 96 |
| substrate | 62.33 | 40.80 | 55 | 90 | 173 |
| microbe | 41.96 | 31.66 | 32 | 42 | 93 |

Artifacts:

- `model_v3.pt`
- `embeddings_v3.npz`
- `metrics_v3.json`
- `training_history.json`
- `metadata_v3.json`
- `enzyme2microbe_index.json`
- `reaction_nn_index.npz`
- `enzyme_nn_index.npz`
- `substrate_nn_index.npz`
- `microbe_nn_index.npz`
- `effective_rank_summary_lightweight.json`

## 6. GVP vs ESM-C Full Comparison

| Metric | GVP full | ESM-C full | Observation |
|--------|---------:|-----------:|-------------|
| `R→E top-1` | 0.0118 | 0.0161 | ESM-C better |
| `R→E top-10` | 0.0671 | 0.0896 | ESM-C better |
| `R→E MRR` | 0.044 | 0.058 | ESM-C better |
| `E→M top-1` | 0.0017 | 0.0036 | ESM-C better |
| `E→M MRR` | 0.395 | 0.609 | ESM-C much better |
| `S→M top-1` | 0.0054 | 0.0056 | similar |
| `S→M MRR` | 0.642 | 0.611 | GVP slightly better |

Effective-rank comparison:

| Modality | GVP erank | ESM-C erank | Observation |
|----------|----------:|------------:|-------------|
| reaction | 28.31 | 43.72 | ESM-C run more spread |
| enzyme | 30.56 | 43.54 | ESM-C enzyme space healthier |
| substrate | 56.58 | 62.33 | ESM-C slightly higher |
| microbe | 38.21 | 41.96 | similar, ESM-C slightly higher |

## 7. Objective Interpretation

### What Worked

1. The full-data training pipeline now runs end-to-end.
2. The microbe modality is now loaded correctly and no longer collapses.
3. Chunked evaluation makes full-data retrieval metrics feasible.
4. ESM-C is consistently better than GVP for:
   - direct reaction-enzyme retrieval
   - enzyme-microbe alignment
   - effective-rank balance

### What Remains Weak

1. `R→E` retrieval is still low in absolute terms:
   - ESM-C full `R→E top-10 = 8.96%`
   - ESM-C full `R→E MRR = 0.058`
2. `top-1` for microbe retrieval remains low:
   - ESM-C full `E→M top-1 = 0.36%`
   - ESM-C full `S→M top-1 = 0.56%`
3. Loss does not monotonically decrease across stages.
   - This is partly expected because the objective changes by stage.
   - But the high Stage 1/3 loss suggests the modalities are not yet tightly
     aligned.
4. Effective ranks are improved but still far below 256.
   - This is not collapse, but indicates concentrated embedding geometry.

### Important Metric Caveat

The evaluation currently treats each example row as its own positive pair.
Because the full dataset contains many repeated UniProt IDs, repeated
assemblies, repeated reactions, and multiple examples sharing biological
entities, strict row-level top-k can underestimate biologically correct
retrieval. For second-round reporting, it may be useful to add grouped metrics
such as:

- same UniProt ID as acceptable R/E hit
- same EC class as relaxed R/E hit
- same assembly or species as relaxed E/M hit

These would be additional analysis metrics. They should not replace the
teacher demo metrics unless approved.

## 8. Second-Round Parameter Recommendation

The second round should prioritize ESM-C because it is stronger on the main
enzyme-related metrics and has healthier effective ranks.

The recommendation below stays close to the teacher README
`Data-Sensitive Parameters` table. It does not propose changing the model or
loss definitions.

### Recommended Second-Round Run: ESM-C-R2

Keep:

```python
unified_dim = 256
batch_size = 4096
lr = 3e-4
temp_end = 0.05
w_sm = 0.5
vicreg_var_weight = 10.0
```

Change:

```python
epochs_stage0 = 8    # unchanged
epochs_stage1 = 15   # from 12, within README full range 10-15
epochs_stage2 = 10   # from 8, within README full range 5-10
epochs_stage3 = 10   # unchanged, README gave no full-data value
hard_neg_weight = 3.0  # from 2.0, within README full range 2.0-4.0
```

Rationale:

1. `R→E` is the main weak point.
   - Teacher README says Stage 1 pairwise contrastive should be `10-15` epochs
     for full data.
   - We used 12; increasing to 15 gives more pairwise R/E alignment time without
     leaving the recommended range.

2. Stage 2 can be increased moderately.
   - Teacher README gives `5-10`.
   - We used 8; moving to 10 may help three-way consistency after pairwise
     alignment improves.

3. Hard negatives can be stronger on full data.
   - Teacher README gives `2.0-4.0`.
   - We used 2.0; moving to 3.0 is a conservative increase.
   - This may help distinguish same-EC or near-neighbor enzymes and improve
     R/E top-k.

4. Do not increase VICReg in round 2.
   - ESM-C effective ranks are already healthier than GVP.
   - Microbe/rank inflation was seen in smoke.
   - Keeping `vicreg_var_weight = 10.0` avoids over-expanding low-dimensional
     modalities.

5. Do not change `batch_size` yet.
   - 4096 ran successfully.
   - It matches the teacher README full-data recommendation and gives strong
     negative sampling for InfoNCE.

### Optional Ablation If Resources Allow

If enough DCU time is available, run one focused ablation alongside ESM-C-R2:

```python
epochs_stage0 = 8
epochs_stage1 = 15
epochs_stage2 = 10
epochs_stage3 = 10
hard_neg_weight = 2.0
```

This isolates whether the improvement comes from longer training or stronger
hard negatives.

If only one run is feasible, choose:

```python
ESM-C-R2: stage1=15, stage2=10, hard_neg_weight=3.0
```

## 9. Proposed Next Steps

1. Send this first-round summary to the teacher.
2. Confirm whether the teacher agrees that ESM-C should be the primary second
   round.
3. If approved, run ESM-C-R2 with:

```python
epochs_stage1 = 15
epochs_stage2 = 10
hard_neg_weight = 3.0
```

4. Compare ESM-C-R2 against ESM-C-R1 on:
   - R/E top-1, top-5, top-10, MRR
   - E/M MRR
   - S/M MRR
   - effective ranks
   - loss history

5. Optionally add grouped biological retrieval metrics for interpretation, but
   keep the teacher demo metrics as the primary comparison.


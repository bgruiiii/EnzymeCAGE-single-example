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
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_TRAIN_PATCH_AND_FULL_RUN_README_2026-06-05.md
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
ESM-C unique UID feature files available: 107,731
ESM-C loaded rows: 145,607 / 145,607
row-level missing/fallback due unavailable ESM-C: 0
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

## 8. Updated Second-Round Strategy After Teacher Review

The earlier recommendation to run ESM-C-R2 by directly changing stage epochs
and/or `hard_neg_weight` is now superseded.

The teacher-approved diagnosis protocol is recorded in:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_ROUND2_FINAL_DIAGNOSIS_PLAN_2026-06-08.md
```

The completed diagnosis result is recorded in:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R1_DIAGNOSIS_RESULT_2026-06-09.md
```

Teacher feedback on 2026-06-08 reframed the Round 1 result as a critical
regression that must be diagnosed before retraining:

- 300 demo `R→E MRR ≈ 0.74`
- full ESM-C R1 `R→E MRR = 0.058`
- full GVP R1 `R→E MRR = 0.044`
- teacher prompt states that the 21,842-test-sample random baseline is about
  `0.10`, so full-data `R→E` is not just weak; it requires postmortem diagnosis

The final agreed implementation should compute random baselines explicitly for
the exact row-level and grouped metric definitions. This avoids mixing strict
row-level random baselines with grouped/class-level random baselines.

The diagnosis below has now completed. Do not submit a new R2 training job until
`R2_PLAN_v2_<timestamp>.md` is drafted and approved.

### Required Round 1 Postmortem Protocol

Completed on HPC. The protocol was to create and run:

```text
diagnose_round1_postmortem.py
```

using the existing full ESM-C Round 1 artifacts:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07/embeddings_v3.npz
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07/metadata_v3.json
```

The report must be written as:

```text
R1_DIAGNOSIS_<timestamp>.md
```

It should test:

1. H1: hard-negative mining may be too coarse because EC labels are encoded by
   first digit only.
2. H2: row-level MRR may underestimate correctness because repeated UniProt
   IDs are treated as separate positives.
3. H3: Stage 1 may be undertrained at full-data scale.

Required metrics:

- row-level, UniProt-grouped, and EC-2-digit-grouped `R→E`
  top-1/top-5/top-10/MRR
- matching random-baseline sanity checks for row-level, UniProt-grouped, and
  EC-2-digit-grouped top-1/top-5/top-10/MRR
- simulated hard-negative contamination over 100 random batches with
  `batch_size=4096` and `seed=42`; report both batch total pairs and
  per-anchor mean / median / max for same-EC-1-digit, same-EC-2-digit, and
  same-UniProt
- ESM-C coverage sanity:
  - `145,607 / 145,607` rows loaded in full ESM-C R1
  - fallback due unavailable ESM-C was `0`
  - `107,731` is unique UniProt / feature count, not 74% row coverage
- per-stage quality only if intermediate checkpoints exist

### Teacher Decision Tree

After diagnosis:

```text
if UniProt-grouped R→E MRR >= 0.30:
    R2 does not retrain; use grouped metrics as primary report metrics.

elif same-EC-1-digit pair count > 500 and grouped MRR < 0.20:
    R2 plan: hard_neg_weight = 1.0, epochs_stage1 = 25,
    EC encoding changed to first two digits, epochs_stage2 = 8 unchanged.

else:
    R2 plan: hard_neg_weight = 2.0 unchanged, epochs_stage1 = 30.
```

## 9. Updated Proposed Next Steps

Status update on 2026-06-09:

- `diagnose_round1_postmortem.py` completed on HPC.
- HPC report:
  `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07/R1_DIAGNOSIS_20260608_172625.md`
- Local summary:
  `/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R1_DIAGNOSIS_RESULT_2026-06-09.md`

Key diagnosis result:

| Positive definition | top-1 | top-5 | top-10 | MRR |
|---|---:|---:|---:|---:|
| row-level | 0.0111 | 0.0520 | 0.0896 | 0.0575 |
| UniProt-grouped | 0.0262 | 0.0654 | 0.0945 | 0.0581 |
| EC-2-digit-grouped | 0.8838 | 0.9282 | 0.9608 | 0.9340 |

Hard-negative contamination:

| Pair definition | per-anchor mean | median | max |
|---|---:|---:|---:|
| same-EC-1-digit | 643.96 | 585.00 | 1269 |
| same-EC-2-digit | 182.26 | 169.00 | 529 |
| same-UniProt | 0.03 | 0.00 | 5 |

Decision tree branch triggered:

```text
H1 + H3 combined
```

Recommended R2 direction:

```python
hard_neg_weight = 1.0
epochs_stage1 = 25
EC encoding = first two EC digits
epochs_stage2 = 8  # unchanged
```

Current next steps:

1. Do not modify `train.py` Config for R2 yet.
2. Do not submit any R2 training job yet.
3. Draft `R2_PLAN_v2_<timestamp>.md`.
4. Include required stage-end checkpoint saving.
5. Submit an R2 training job only after the revised plan is approved.

# Bio Vector Round 2 Parameter Review Brief

Date: 2026-06-08

Status update later on 2026-06-08:

- This document records the earlier Round 2 parameter-review process.
- It has been superseded by the teacher-approved final diagnosis plan in:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_ROUND2_FINAL_DIAGNOSIS_PLAN_2026-06-08.md
```

- The teacher-provided original instruction is:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_ROUND2_CODEX_PROMPT_FOR_STUDENT_2026-06-08.md
```

- Do not use the Option B recommendation below as approval to retrain.
- Current required next step is to run a Round 1 postmortem diagnosis on
  existing ESM-C Round 1 embeddings and metadata, then write a revised
  `R2_PLAN_v2_<timestamp>.md`.
- The agreed diagnosis adds two safeguards:
  - compute random baselines under the exact row-level and grouped metric
    definitions
  - report hard-negative contamination both as batch total pairs and as
    per-anchor mean / median / max

Purpose of this document:

- Provide enough context for an external GPT/reviewer to independently evaluate
  our proposed Round 2 parameter changes.
- Separate the teacher demo's original training strategy from the compatibility
  patches required to run the full dataset.
- Make clear what we trained, what Round 1 showed, and why the proposed Round 2
  changes are being considered.

## 1. What This Demo Is Trying To Do

The teacher-provided `bio_vector` demo trains a unified four-modal vector space.

Each example contains:

- Reaction feature: DRFP, 2048 dimensions
- Enzyme feature:
  - GVP mode: structure/pocket-derived GVP feature, pooled to 50 dimensions
  - ESM-C mode: protein language model feature, pooled to 1152 dimensions
- Substrate feature: Morgan fingerprint, 2048 dimensions
- Microbe feature: structured metabolic vector, 28 dimensions

The model maps all four modalities into a shared 256-dimensional embedding
space:

```text
Reaction  -> 256d
Enzyme    -> 256d
Substrate -> 256d
Microbe   -> 256d
```

Training objective:

- true cross-modal pairs should be close
- unrelated examples should be farther apart
- retrieval should work across modalities:
  - reaction -> enzyme
  - enzyme -> microbe
  - substrate -> microbe

The teacher demo uses four training stages:

| Stage | Purpose |
|------:|---------|
| 0 | Independent pretrain with VICReg-style variance/covariance regularization |
| 1 | Pairwise contrastive learning: R<->E and E<->M |
| 2 | Three-way contrastive learning including S<->M, plus microbe concept anchor supervision |
| 3 | Self-bootstrap with FBA surrogate signal |

We did not intentionally change the model architecture, losses, optimizer,
scheduler, or stage definitions.

## 2. Full Dataset And Execution Context

Full dataset:

- Total rows/examples: `145,607`
- Unique ESM-C UniProt IDs/features available: `107,731`
- Row-level ESM-C coverage in the full ESM-C baseline: `145,607 / 145,607`
  rows loaded, missing/fallback due unavailable ESM-C = `0`
- Important: `107,731 / 145,607` is not a 74% coverage rate. The numerator is
  unique UniProt IDs, while the denominator is example rows; multiple examples
  share the same UniProt ID.
- GVP features: 192 sharded `.pt` files
- ESM-C features: flattened per-UID files under `features/enzyme/esm_c_features`
- Microbe feature sources:
  - `microbe_reaction_core_preference.csv`
  - `microbe_reaction_stoich_query.csv`
  - `microbe_reaction_main_metabolite_coverage.csv`

HPC environment:

- Hygon/Sugon DCU environment
- DTK module: `compiler/dtk/23.10`
- PyTorch: DTK-compatible build
- Runs used 1 DCU per baseline

Important artifact note:

- Native FAISS index generation failed in this HPC environment due FAISS
  SWIG/NumPy incompatibility.
- We therefore save `*_nn_index.npz` normalized embedding fallback files.
- This does not affect training, embeddings, or metrics.

## 3. What We Had To Patch Before Running Full Data

These patches were required because the full dataset layout differed from the
small demo assumptions. They are compatibility/scalability patches, not
intended algorithm changes.

### 3.1 GVP Loader Patch

Original demo assumption:

- GVP was stored as individual `.npz` files referenced by `gvp_feature_file`.

Actual full data:

- GVP is stored in 192 sharded `.pt` files referenced by `gvp_shard_file`.
- Each shard maps `UniProtID -> tuple tensors`.

Patch:

- Read `gvp_shard_file`.
- Add sharded GVP pooling to produce the same 50-dimensional enzyme input
  expected by the original model.
- Batch by shard to avoid repeated `torch.load()` calls.

### 3.2 Microbe Loader Patch

Original demo assumption:

- `microbe_features.jsonl` contains nested dictionaries:
  - `core_preference`
  - `stoich_query`
  - `main_metabolite_coverage`

Actual full data:

- JSONL is flat.
- The numeric fields needed by `extract_microbe_features()` are in three CSV
  tables.

Patch:

- Build nested records from the three CSV tables by `example_id`.
- Preserve JSONL fallback.
- Add safe string-to-float/bool parsing.

Validation after patch:

```text
microbe_feats shape: (145607, 28)
nonzero element ratio: 40.19%
all-zero rows: 0/145607
concept targets: all 8 columns non-NaN, 47.33% nonzero
assembly_id non-empty: 145607/145607
enzyme2microbe index: 107731 enzymes mapped
```

### 3.3 ESM-C Loader Optimization

Initial 10k ESM-C smoke timed out during data loading.

Patch:

- Pre-scan ESM-C directory once with `os.listdir()`.
- Build an available UID set.
- Avoid per-row `.exists()` calls.
- Preserve fallback semantics:

```text
ESM-C -> GVP -> AAC
```

### 3.4 Evaluation Scalability Patch

Original demo evaluation constructed full `N x N` similarity and rank matrices.
For `N = 145,607`, this is too large.

Patch:

- Use chunked evaluation.
- Preserve metric semantics for top-k/MRR.
- Save checkpoint before evaluation.
- Save metrics/embeddings before optional index/postprocessing steps.

## 4. Round 1 Parameters Used

The teacher README included a `Data-Sensitive Parameters` table for full data.
We selected one concrete configuration within the teacher-provided ranges.

Round 1 full-data Config:

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
| `hard_neg_weight` | 2.0 | kept default, within teacher README range |

Total epochs:

```text
Stage 0 + Stage 1 + Stage 2 + Stage 3 = 8 + 12 + 8 + 10 = 38
```

## 5. Round 1 Full Baseline Results

We ran both enzyme-feature modes:

- Full GVP baseline
- Full ESM-C baseline

### 5.1 Full GVP Baseline

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

Effective rank:

| Modality | Effective rank | Participation ratio | dim@90 | dim@95 | dim@99 |
|----------|---------------:|--------------------:|-------:|-------:|-------:|
| reaction | 28.31 | 20.58 | 21 | 29 | 92 |
| enzyme | 30.56 | 21.22 | 22 | 40 | 136 |
| substrate | 56.58 | 35.23 | 55 | 91 | 174 |
| microbe | 38.21 | 26.05 | 31 | 46 | 108 |

### 5.2 Full ESM-C Baseline

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

Effective rank:

| Modality | Effective rank | Participation ratio | dim@90 | dim@95 | dim@99 |
|----------|---------------:|--------------------:|-------:|-------:|-------:|
| reaction | 43.72 | 32.36 | 33 | 42 | 97 |
| enzyme | 43.54 | 34.22 | 32 | 40 | 96 |
| substrate | 62.33 | 40.80 | 55 | 90 | 173 |
| microbe | 41.96 | 31.66 | 32 | 42 | 93 |

### 5.3 Direct GVP vs ESM-C Comparison

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

Round 1 interpretation:

- ESM-C is the stronger enzyme representation overall.
- ESM-C improves direct `R→E` retrieval and especially `E→M` alignment.
- GVP is slightly better on `S→M MRR`, but the advantage is smaller than the
  ESM-C gains on enzyme-related metrics.
- ESM-C also gives a more balanced effective-rank profile.

## 6. Remaining Issues After Round 1

1. Absolute `R→E` retrieval is still low.
   - ESM-C full `R→E top-10 = 0.0896`.
   - ESM-C full `R→E MRR = 0.058`.

2. `E→M` and `S→M` top-1 are very low despite higher MRR.
   - This may reflect repeated entities and row-level positive definitions.
   - It may also indicate that top-k needs grouped biological interpretation.

3. Effective ranks are improved but still concentrated relative to 256.
   - This is not collapse.
   - But reaction/enzyme still do not use the full embedding capacity.

4. Loss is not monotonically decreasing across stages.
   - Stage objectives differ, so this is not automatically wrong.
   - But the high Stage 1/3 loss suggests cross-modal alignment remains hard.

5. The current metrics are row-level.
   - Multiple rows share UniProt IDs, reactions, EC classes, assemblies, and
     species.
   - Strict row-level top-k may underestimate biologically valid retrievals.
   - Grouped metrics could be added later as extra interpretation, not as a
     replacement for teacher demo metrics.

## 7. Proposed Round 2 Parameter Change

We propose to run Round 2 on ESM-C first, because ESM-C is the stronger Round 1
baseline.

Do not change:

```python
unified_dim = 256
batch_size = 4096
lr = 3e-4
temp_end = 0.05
w_sm = 0.5
vicreg_var_weight = 10.0
epochs_stage0 = 8
epochs_stage3 = 10
```

Change:

```python
epochs_stage1 = 15      # was 12
epochs_stage2 = 10      # was 8
hard_neg_weight = 3.0   # was 2.0
```

## 8. Why These Changes

### 8.1 Increase Stage 1 From 12 To 15

Teacher README full-data range:

```text
epochs_stage1: 10-15
```

Round 1 used 12.

Reason to increase:

- The main weak point is `R→E`.
- Stage 1 is the pairwise contrastive stage and directly trains:
  - `R↔E`
  - `E↔M`
- It does not directly train `S↔M`; `S↔M` enters in Stage 2.
- Moving from 12 to 15 stays within the teacher range and gives the pairwise
  alignment more iterations.

Expected effect:

- improve `R→E top-10`
- possibly improve `R→E MRR`
- maintain or improve `E→M MRR`

Risk:

- over-emphasizing pairwise alignment could reduce later-stage consistency if
  Stage 2 is not also strengthened.

### 8.2 Increase Stage 2 From 8 To 10

Teacher README full-data range:

```text
epochs_stage2: 5-10
```

Round 1 used 8.

Reason to increase:

- Stage 2 handles triplet consistency and concept anchor supervision.
- Increasing to 10 balances the longer Stage 1 and may help preserve
  microbe/substrate consistency.

Expected effect:

- improve consistency of `R/E/M` and `S/M`
- reduce risk that stronger Stage 1 harms microbe alignment

Risk:

- if concept anchors are noisy, more Stage 2 may amplify weak supervision.

### 8.3 Increase `hard_neg_weight` From 2.0 To 3.0

Teacher README full-data range:

```text
hard_neg_weight: 2.0-4.0
```

Round 1 used 2.0.

Reason to increase:

- `R→E` retrieval is low.
- Hard negatives should make the model distinguish similar enzymes / same-EC or
  near-class negatives more strongly.
- 3.0 is a conservative midpoint, not the maximum.

Expected effect:

- improve `R→E top-k`
- improve enzyme-space discrimination

Risk:

- too much hard-negative pressure could hurt broad biological similarity or
  destabilize `E→M`.

## 9. Alternative Plans For Reviewer To Consider

Please evaluate whether the proposed Round 2 should instead use one of these
variants:

### Option A: Conservative Single Change

```python
epochs_stage1 = 15
epochs_stage2 = 8
hard_neg_weight = 2.0
```

Question:

- Should we isolate the effect of longer pairwise training before changing hard
  negatives?

### Option B: Longer Training Only

```python
epochs_stage1 = 15
epochs_stage2 = 10
hard_neg_weight = 2.0
```

Question:

- Is hard-negative strengthening too risky before knowing whether more epochs
  alone improves `R→E`?

### Option C: Proposed Main Plan

```python
epochs_stage1 = 15
epochs_stage2 = 10
hard_neg_weight = 3.0
```

Question:

- Is this the best balance between using the teacher README range and making a
  meaningful second-round change?

### Option D: Adjust Loss Weights Instead

Possible change:

```python
w_re = 1.2  # currently 1.0
```

Reason:

- `R→E` is the main weak metric.

Concern:

- This is not listed in the teacher README `Data-Sensitive Parameters` table,
  while stage epochs and hard negative weight are.
- It may be harder to justify as the immediate second-round change.

## 10. Requested External Review

Please review:

1. Is prioritizing ESM-C for Round 2 justified by the Round 1 results?
2. Is the original proposed change:

```python
epochs_stage1 = 15
epochs_stage2 = 10
hard_neg_weight = 3.0
```

reasonable and conservative?

3. Would you instead recommend Option A, B, or D above?
4. Are there risks that the proposed hard-negative increase could damage
   `E→M` or `S→M`?
5. Should `vicreg_var_weight` remain 10.0, or should it be adjusted based on
   the effective-rank results?
6. Should we add grouped biological retrieval metrics before changing
   parameters, or after Round 2?

## 11. Superseded Previously Preferred Recommendation

Superseded:

The recommendation in this section was the preferred plan after external GPT
review, but before the teacher's later 2026-06-08 code-review prompt. It should
now be treated as historical context only.

After external review feedback, before the teacher's later diagnosis-first
instruction, the preferred recommendation was the more conservative Option B:

```python
enzyme_feature = "esmc"
epochs_stage0 = 8
epochs_stage1 = 15
epochs_stage2 = 10
epochs_stage3 = 10
batch_size = 4096
lr = 3e-4
temp_end = 0.05
w_sm = 0.5
vicreg_var_weight = 10.0
hard_neg_weight = 2.0
```

Reason:

- ESM-C is clearly stronger than GVP in Round 1.
- The main weak metric is `R→E`, and Stage 1 directly trains `R↔E`.
- Increasing Stage 2 to 10 keeps three-way consistency and concept-anchor
  training aligned with the stronger pairwise stage.
- Holding `hard_neg_weight = 2.0` isolates the effect of longer Stage 1/2
  training before increasing hard-negative pressure.
- This reduces the risk that stronger R/E hard negatives indirectly disrupt the
  already strong E/M alignment through the shared enzyme projector.

If resources allow, run a second ablation after Option B:

```python
enzyme_feature = "esmc"
epochs_stage1 = 15
epochs_stage2 = 10
hard_neg_weight = 3.0
```

This tests whether stronger hard negatives improve `R→E` beyond the longer
training effect.

## 12. External Review Feedback Integrated

External review identified two important points:

1. Stage 1 was described too broadly in an earlier draft.
   - Correct code behavior:
     - Stage 1 trains `R↔E` and `E↔M`
     - `S↔M` enters in Stage 2
   - This has been corrected above.

2. The `107,731 / 145,607` ESM-C statement was misread as 74% coverage.
   - Correct interpretation:
     - `107,731` is the number of unique UniProt IDs/features.
     - `145,607` is the number of example rows.
     - full ESM-C row-level loading was `145,607 / 145,607`.
     - missing/fallback due unavailable ESM-C was `0`.

External review also recommended grouped biological retrieval metrics before or
alongside Round 2. I agree this is useful as additional analysis because
row-level metrics can underestimate biologically equivalent hits when examples
share UniProt IDs, EC classes, reactions, assemblies, or species.

## 13. Teacher Prompt Supersedes This Plan

The teacher's later review reframed the Round 1 result as a critical regression
instead of merely "weak" retrieval:

- 300 demo `R→E MRR ≈ 0.74`
- full ESM-C R1 `R→E MRR = 0.058`
- full GVP R1 `R→E MRR = 0.044`
- teacher prompt states that the 21,842-test-sample random baseline is about
  `0.10`, so R1 `R→E` must be diagnosed before retraining

New required hypotheses to test before any R2 training:

- H1: hard-negative mining is too coarse because EC labels are encoded by first
  digit only, producing many same-class negatives in 4096-size batches.
- H2: row-level retrieval underestimates correctness because repeated UniProt
  IDs are treated as separate rows.
- H3: Stage 1 is undertrained at full-data scale.

New required first outputs:

```text
diagnose_round1_postmortem.py
R1_DIAGNOSIS_<timestamp>.md
R2_PLAN_v2_<timestamp>.md
```

New hard constraints:

- Do not submit R2 training before the revised `R2_PLAN_v2_<timestamp>.md` is
  drafted and approved.
- Do not increase `hard_neg_weight` to `3.0`.
- Do not simultaneously change `stage1`, `stage2`, and `hard_neg`.
- Grouped metrics should be primary reporting metrics; row-level metrics remain
  reference metrics.

Agreed implementation additions:

- Add row-level, UniProt-grouped, and EC-2-digit-grouped random-baseline sanity
  checks.
- For hard-negative contamination, report both total batch pair counts and
  per-anchor mean / median / max; interpret the teacher's `>500` threshold as a
  per-anchor same-EC-1-digit scale.
- Include an ESM-C coverage sanity statement:
  - full ESM-C R1 loaded `145,607 / 145,607` rows
  - fallback due unavailable ESM-C was `0`
  - `107,731` is unique UniProt / feature count, not a 74% row coverage gap

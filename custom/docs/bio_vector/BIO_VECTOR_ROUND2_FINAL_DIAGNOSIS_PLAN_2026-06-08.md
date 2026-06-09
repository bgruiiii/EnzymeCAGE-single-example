# Bio Vector Round 2 Final Diagnosis Plan

Date: 2026-06-08

Status: teacher-approved diagnosis protocol; completed on HPC.

Completed diagnosis result:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R1_DIAGNOSIS_RESULT_2026-06-09.md
```

The completed diagnosis triggered the H1 + H3 branch:

```python
hard_neg_weight = 1.0
epochs_stage1 = 25
EC encoding = first two EC digits
epochs_stage2 = 8  # unchanged
```

This document supersedes the earlier immediate ESM-C-R2 retraining plans,
including:

- `epochs_stage1 = 15`, `epochs_stage2 = 10`, `hard_neg_weight = 3.0`
- conservative Option B: `epochs_stage1 = 15`, `epochs_stage2 = 10`,
  `hard_neg_weight = 2.0`

Do not submit any Round 2 training job until the revised
`R2_PLAN_v2_<timestamp>.md` is drafted and approved.

## 1. Current Position

Round 1 full-data baselines are complete:

- Full GVP baseline
- Full ESM-C baseline

Full ESM-C remains the stronger Round 1 baseline overall, but the full-data
`R→E` result dropped sharply relative to the 300-demo result:

```text
300 demo R→E MRR: approximately 0.74
full ESM-C R1 R→E MRR: 0.058
full GVP R1 R→E MRR: 0.044
```

This should be treated as a critical Round 1 regression that needs postmortem
diagnosis, not as a parameter-tuning inconvenience.

## 2. Important Corrected Facts

ESM-C row-level coverage is complete:

```text
ESM-C loaded rows: 145,607 / 145,607
row-level missing/fallback due unavailable ESM-C: 0
unique UniProt IDs / ESM-C feature files: 107,731
```

Therefore `107,731 / 145,607` must not be interpreted as 74% ESM-C coverage.
It is unique UID count divided by row count, and many rows share one UniProt ID.

The teacher prompt's random-baseline statement should be kept, but the exact
baseline must be reported with its metric definition. For strict row-level retrieval
over 21,842 candidates, random top-k/MRR is not approximately 0.10. The
diagnosis should therefore compute and report row-level and grouped random
baseline sanity checks explicitly.

## 3. Hypotheses To Diagnose

### H1: Hard-Negative Mining Is Too Coarse

Current code encodes EC labels using only the first EC digit, leaving only broad
classes such as `1` through `7`. In `infonce_loss`, same-EC negatives receive
extra denominator weight.

With `batch_size = 4096`, each anchor may have hundreds of same-EC-1-digit
examples in the same batch. Many of these may be biologically related enzymes,
so the loss may pressure the model to over-separate useful neighbors.

This is why `hard_neg_weight = 3.0` is explicitly rejected before diagnosis.

### H2: Row-Level Evaluation Underestimates Correct Retrieval

The current row-level retrieval uses the same row as the only positive. Full
data contains repeated UniProt IDs and repeated biological entities, so a
cross-row same-UniProt hit may be biologically correct but counted as wrong.

Grouped metrics are therefore required.

### H3: Stage 1 Is Undertrained At Full-Data Scale

Stage 1 trains the key pairwise alignments `R↔E` and `E↔M`. Full-data Stage 1
has many more examples than the 300 demo but only modestly more optimization
steps, so pairwise alignment may be undertrained before Stage 2/3.

## 4. Required Diagnosis Script

Create:

```text
diagnose_round1_postmortem.py
```

Use existing R1 ESM-C artifacts only:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07/embeddings_v3.npz
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07/metadata_v3.json
```

Do not modify `train.py`.
Do not patch Config.
Do not submit a training job.

## 5. Required Diagnosis Metrics

### 5.1 Grouped R→E Retrieval

Report top-1, top-5, top-10, and MRR for:

- row-level positives
- UniProt-grouped positives
- EC-2-digit-grouped positives

The row-level MRR should reproduce the R1 ESM-C result of about `0.058`.

### 5.2 Random Baseline Sanity Checks

For the same candidate set and metadata, report random baseline estimates for:

- row-level top-1/top-5/top-10/MRR
- UniProt-grouped top-1/top-5/top-10/MRR
- EC-2-digit-grouped top-1/top-5/top-10/MRR

This prevents mixing row-level random baselines with grouped/class-level random
baselines.

### 5.3 Hard-Negative Contamination

Simulate `100` random batches with:

```text
batch_size = 4096
seed = 42
```

For each batch, report both total pair counts and per-anchor statistics:

- same-EC-1-digit total pairs
- same-EC-1-digit per-anchor mean / median / max
- same-EC-2-digit total pairs
- same-EC-2-digit per-anchor mean / median / max
- same-UniProt total pairs
- same-UniProt per-anchor mean / median / max

Also include a compact histogram or quantile summary.

The teacher threshold `>500` should be interpreted primarily as a per-anchor
same-EC-1-digit scale, not as total batch pair count.

### 5.4 ESM-C Coverage Sanity

Confirm in the report:

- unique UniProt IDs: `107,731`
- full row count: `145,607`
- R1 full ESM-C log showed `145,607 / 145,607` loaded and fallback `0`

If the available metadata contains enzyme feature source labels, it is fine to
stratify R→E metrics by source as an additional sanity check. However, this is
not a primary H4 because row-level ESM-C loading has already been validated as
complete.

### 5.5 Per-Stage Quality

If intermediate stage checkpoints exist, evaluate Stage 1 / Stage 2 / Stage 3
R→E quality. If only the final checkpoint exists, explicitly write:

```text
per-stage analysis skipped because only the final checkpoint exists
```

Future R2 training must save:

```text
model_v3_stage0.pt
model_v3_stage1.pt
model_v3_stage2.pt
model_v3_stage3.pt
```

or equivalent stage-end checkpoint artifacts.

## 6. Required Diagnosis Output

Write:

```text
R1_DIAGNOSIS_<timestamp>.md
```

The report must include:

- concise hypothesis-by-hypothesis verdict
- grouped retrieval table
- random baseline table
- hard-negative contamination table
- ESM-C coverage sanity statement
- per-stage availability statement
- the decision-tree branch triggered

## 7. Teacher Decision Tree

Use the teacher's decision tree after Task 1:

```text
if UniProt-grouped R→E MRR >= 0.30:
    R2 does not retrain.
    Rewrite the report using grouped metrics as primary metrics.

elif same-EC-1-digit per-anchor count > 500 and grouped MRR < 0.20:
    R2 plan:
      hard_neg_weight = 1.0
      epochs_stage1 = 25
      EC encoding changed to first two digits
      epochs_stage2 = 8 unchanged

else:
    R2 plan:
      hard_neg_weight = 2.0 unchanged
      epochs_stage1 = 30
```

## 8. Revised R2 Plan Output

After diagnosis, write:

```text
R2_PLAN_v2_<timestamp>.md
```

It must state:

- which hypothesis is considered the bottleneck
- which decision-tree branch was triggered
- exactly which variables will change
- which variables will remain unchanged
- expected effect on row-level and grouped metrics
- stage-checkpoint saving requirement
- approval boundary before any HPC training job

## 9. Current Next Step After Completed Diagnosis

The diagnosis has completed. The next step is to draft:

```text
R2_PLAN_v2_<timestamp>.md
```

using:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R1_DIAGNOSIS_RESULT_2026-06-09.md
```

No R2 training, Config patch, loss edit, loader edit, or evaluation edit should
happen before the revised R2 plan is approved.

# Bio Vector R1 Diagnosis Result

Date: 2026-06-09

This document records the HPC-completed Round 1 postmortem diagnosis for the
full ESM-C baseline.

## 1. HPC Artifacts

Diagnosis script:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/diagnose_round1_postmortem.py
```

Diagnosis report:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07/R1_DIAGNOSIS_20260608_172625.md
```

Run mode:

- Slurm diagnosis job, not a training job
- job ID: `115034071`
- stdout: `r1_diagnosis_115034071.out`
- stderr: `r1_diagnosis_115034071.err`
- stderr was empty
- runtime: `13,509.2` seconds, about `3.75` hours

Confirmed boundaries:

- `train.py` was not modified
- no R2 training job was submitted
- only the diagnosis job was submitted

## 2. R→E Retrieval Diagnosis

| Positive definition | top-1 | top-5 | top-10 | MRR |
|---|---:|---:|---:|---:|
| row-level | 0.0111 | 0.0520 | 0.0896 | 0.0575 |
| UniProt-grouped | 0.0262 | 0.0654 | 0.0945 | 0.0581 |
| EC-2-digit-grouped | 0.8838 | 0.9282 | 0.9608 | 0.9340 |

Interpretation:

- Row-level MRR reproduces the R1 ESM-C baseline (`0.058`), so the diagnosis
  script is consistent with the original metrics.
- UniProt-grouped MRR is almost unchanged from row-level MRR:
  `0.0575 -> 0.0581`.
- Therefore repeated UniProt rows / row-level evaluation do not explain the
  main R→E fine-discrimination gap.
- EC-2-digit grouped MRR is very high (`0.9340`), which means the model learned
  strong coarse EC-2 reaction/enzyme signal.
- The issue is therefore not "no biological signal"; it is low row-level /
  UniProt-level discrimination despite strong EC-2 alignment.

Important decision-tree note:

- The teacher decision tree's `grouped MRR < 0.20` trigger should be interpreted
  using the exact-biological grouped metric, here UniProt-grouped MRR.
- EC-2-digit grouped MRR is a useful relaxed diagnostic and reporting metric,
  but it is too broad to decide that exact R→E alignment is solved.

## 3. Random Baseline

| Positive definition | top-1 | top-5 | top-10 | MRR |
|---|---:|---:|---:|---:|
| row-level random | 6.87e-6 | 3.43e-5 | 6.87e-5 | 6.87e-6 |
| UniProt-grouped random | 9.28e-6 | 4.64e-5 | 9.28e-5 | 1.10e-4 |
| EC-2-digit-grouped random | 0.0136 | 0.0617 | 0.1107 | 0.0474 |

Interpretation:

- R1 ESM-C is far above the matching random baselines under all three metric
  definitions.
- The earlier "random baseline around 0.10" statement should not be reused
  without specifying the metric definition. Under strict row-level retrieval,
  random MRR is near `6.87e-6`, not `0.10`.
- The critical issue remains the large drop from the 300-demo R→E MRR to
  full-data row-level / UniProt-level R→E MRR, not performance below the
  matched random baseline.

## 4. Hard-Negative Contamination

Simulation settings:

```text
batch_size = 4096
seed = 42
sampled batches = 100
```

Per-anchor statistics:

| Pair definition | mean | median | max |
|---|---:|---:|---:|
| same-EC-1-digit | 643.96 | 585.00 | 1269 |
| same-EC-2-digit | 182.26 | 169.00 | 529 |
| same-UniProt | 0.03 | 0.00 | 5 |

Interpretation:

- same-EC-1-digit per-anchor count exceeds the teacher threshold of `500`.
- This supports H1: first-digit EC hard-negative mining is too coarse for
  `batch_size=4096`.
- same-UniProt false-negative contamination is very low, so batch-level repeated
  UniProt negatives are not a major issue.

## 5. ESM-C Coverage Sanity

Diagnosis confirmed the existing coverage interpretation:

- full row count: `145,607`
- unique UniProt / ESM-C feature count: `107,731`
- full ESM-C R1 loaded rows: `145,607 / 145,607`
- fallback due unavailable ESM-C: `0`

Therefore `107,731 / 145,607` is not a 74% ESM-C coverage gap.

## 6. Per-Stage Checkpoint Availability

Stage-end checkpoints were not available:

```text
model_v3_stage0.pt: missing
model_v3_stage1.pt: missing
model_v3_stage2.pt: missing
model_v3_stage3.pt: missing
```

Only final `model_v3.pt` exists, so per-stage analysis was skipped.

Future R2 training must save stage-end checkpoints.

## 7. Decision Tree Result

Triggered branch:

```text
H1 + H3 combined
```

Trigger facts:

- same-EC-1-digit per-anchor mean: `643.96 > 500`
- UniProt-grouped MRR: `0.0581 < 0.20`

Initial R2 direction from the teacher decision tree:

```python
hard_neg_weight = 1.0
epochs_stage1 = 25
EC encoding = first two EC digits
epochs_stage2 = 8  # unchanged
```

Teacher later revised the training-variable boundary:

```python
hard_neg_weight = 1.0
epochs_stage1 = 25
```

The EC encoding change should be removed from R2 training variables because
with `hard_neg_weight = 1.0`, the same-EC weighting term degenerates to `1.0`
and the same-EC matrix is unused.

## 8. EC-3/EC-4 Supplement

Teacher requested a zero-cost supplement before R2 training:

- add EC-4-grouped `R→E` evaluation
- optionally add EC-3-grouped `R→E` evaluation
- do not retrain
- do not modify `train.py`

Supplement execution audit:

- run mode: direct run on login node `login09`
- no Slurm job was used
- `python -m py_compile diagnose_round1_postmortem.py` passed
- runtime: `748.4` seconds
- `train.py` was not modified
- no R2 training job was submitted

R→E retrieval after supplement:

| Positive definition | evaluated queries | excluded unknown | top-1 | top-5 | top-10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| row-level | 145607 | 0 | 0.0109 | 0.0522 | 0.0896 | 0.0580 |
| UniProt-grouped | 145607 | 0 | 0.0263 | 0.0655 | 0.0945 | 0.0581 |
| EC-2-grouped | 145607 | 0 | 0.8730 | 0.9248 | 0.9608 | 0.9340 |
| EC-3-grouped | 130635 | 14972 | 0.870540 | 0.924500 | 0.959238 | 0.933629 |
| EC-4-grouped | 127847 | 17760 | 0.817876 | 0.922274 | 0.944926 | 0.918680 |

Random baseline:

| Positive definition | top-1 | top-5 | top-10 | MRR |
|---|---:|---:|---:|---:|
| EC-3-grouped random | 0.00479773 | 0.02282687 | 0.04306375 | 0.02007514 |
| EC-4-grouped random | 0.00034787 | 0.00172581 | 0.00341852 | 0.00215903 |

Interpretation:

- R1 aligns reaction/enzyme well at EC-2, EC-3, and EC-4 levels.
- The remaining problem is fine-grained row-level / UniProt-level
  discrimination.
- Revised R2 should optimize this fine-grained target without degrading EC-2
  and EC-4 grouped structure.

## 9. Current Next Step

Do not submit R2 training yet.

Next local task:

- revise `R2_PLAN_v2_<timestamp>.md`
- keep only `hard_neg_weight = 1.0` and `epochs_stage1 = 25` as training
  changes
- remove EC encoding as a training change
- include stage-checkpoint saving as a required train.py artifact patch
- add row-level, UniProt-grouped, EC-2, EC-3, and EC-4 grouped evaluation
  requirements
- add quantitative pass/fail thresholds
- submit the revised R2 plan for teacher approval before any HPC training job

Old note:

```text
R2_PLAN_v2_<timestamp>.md
```

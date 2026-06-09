# Bio Vector Docs README

Last updated: 2026-06-09

Purpose: this folder is the working archive for the Bio Vector / `bio_vector`
Round 1 baseline, Round 1 diagnosis, and Round 2 planning work. This README is
the navigation map so we do not need to reread every `.md` file each time.

## Current Status

- Round 1 full GVP baseline: complete.
- Round 1 full ESM-C baseline: complete.
- Round 1 postmortem diagnosis: complete.
- Teacher-requested EC-3/EC-4 supplement: complete and audited.
- Current R2 plan: teacher-approved.
- No R2 `train.py` patch has been approved yet.
- No R2 training job has been submitted yet.

Current source of truth:

```text
R2_PLAN_v2_REVISED_20260609_175634.md
```

Current teacher-approved R2 training variables:

```python
hard_neg_weight = 1.0
epochs_stage1 = 25
```

Important boundary:

- Do not use old Option B or the old `R2_PLAN_v2_20260609_093107.md` to train.
- Do not include EC encoding as an R2 training change.
- Do not describe R1 as a training collapse. Current narrative:
  R1 aligns well at EC-family / EC-functional levels, while row-level /
  UniProt-level fine discrimination remains low.

## What To Read Now

For the current next step, read these in order:

1. `R2_PLAN_v2_REVISED_20260609_175634.md`
   - current teacher-approved R2 plan
   - contains approved-boundary training variables, unchanged parameters,
     stage checkpoint requirements, evaluation metrics, pass/warning thresholds,
     and the HPC upload/training sequence

2. `BIO_VECTOR_R2_TASK0_EC4_SUPPLEMENT_FEEDBACK_FOR_TEACHER_2026-06-09.md`
   - final teacher-facing response for today's EC-3/EC-4 supplement task
   - use this when reporting Task 0 completion to the teacher

3. `BIO_VECTOR_R1_DIAGNOSIS_RESULT_2026-06-09.md`
   - detailed local record of R1 diagnosis and EC-3/EC-4 supplement results
   - use this if you need numbers, artifact paths, or decision-tree evidence

4. `BIO_VECTOR_R2_PLAN_REVISION_FOR_STUDENT_2026-06-09.md`
   - teacher's latest instruction that caused the revised R2 plan
   - use this to verify that the response matches the teacher's requested tasks

For a future new chat or handoff, also read:

```text
BIO_VECTOR_NEXT_CHAT_HANDOFF_ROUND2_2026-06-08.md
```

## File Index

| File | Direction / Role | Current use | Contains |
|---|---|---|---|
| `R2_PLAN_v2_REVISED_20260609_175634.md` | Student/Codex -> teacher, now teacher-approved | Current | Revised R2 plan after teacher feedback and EC-3/EC-4 supplement. This is the approved boundary for HPC patch/training. |
| `BIO_VECTOR_R2_TASK0_EC4_SUPPLEMENT_FEEDBACK_FOR_TEACHER_2026-06-09.md` | Student/Codex -> teacher | Current | Teacher-facing Task 0 response: EC-3/EC-4 grouped metrics, random baselines, execution audit, no `train.py` change, no R2 job. |
| `BIO_VECTOR_R1_DIAGNOSIS_RESULT_2026-06-09.md` | Internal record / evidence | Current reference | Full R1 diagnosis summary, H1/H3 evidence, EC-3/EC-4 supplement results, R2 implication. |
| `BIO_VECTOR_R2_PLAN_REVISION_FOR_STUDENT_2026-06-09.md` | Teacher -> student/Codex | Current instruction | Teacher's revision requirements: Task 0 EC-4 supplement, remove EC encoding from R2, rewrite narrative, add thresholds, expand evaluation. |
| `BIO_VECTOR_NEXT_CHAT_HANDOFF_ROUND2_2026-06-08.md` | Internal handoff | Current handoff | Compact state for a new chat: work mode, current objective, key artifacts, latest 2026-06-09 updates. |
| `R2_PLAN_v2_20260609_093107.md` | Student/Codex -> teacher | Superseded | Old R2 plan draft. Kept for history only. Do not use to train; it included EC encoding as a training change before teacher revision. |
| `BIO_VECTOR_ROUND2_FINAL_DIAGNOSIS_PLAN_2026-06-08.md` | Student/Codex plan, teacher-approved diagnosis protocol | Completed historical protocol | Diagnosis-first plan before R1 postmortem was run. Useful for why diagnosis was done, but superseded by completed results. |
| `BIO_VECTOR_ROUND2_CODEX_PROMPT_FOR_STUDENT_2026-06-08.md` | Teacher -> student/Codex | Historical teacher prompt | Original teacher prompt that started diagnosis-first Round 2 planning. Some early language was later refined by the teacher. |
| `BIO_VECTOR_ROUND2_PARAMETER_REVIEW_BRIEF_FOR_GPT_2026-06-08.md` | Student/Codex -> external GPT/reviewer | Superseded review brief | Earlier parameter review and external GPT assessment context. Do not use as current R2 approval. |
| `BIO_VECTOR_ROUND1_FULL_BASELINE_SUMMARY_AND_ROUND2_PLAN_2026-06-07.md` | Internal summary / earlier plan | Historical baseline reference | Full-data R1 GVP/ESM-C baseline summary and older R2 plan. Useful for baseline artifacts and original full-run metrics. |
| `BIO_VECTOR_TRAIN_PATCH_AND_FULL_RUN_README_2026-06-05.md` | Internal technical runbook | Historical technical reference | Full-data `train.py` compatibility/scalability patches, HPC environment notes, loader details, full-run troubleshooting. |
| `BIO_VECTOR_ROUND1_ANALYSIS_AND_ROUND2_PLAN_2026-06-03.md` | Internal early analysis | Historical demo reference | 300-sample CPU demo Round 1 results and early Round 2 parameter suggestions. Useful mainly for demo baseline context. |
| `BIO_VECTOR_DEMO_RUN_LOG_2026-06-03.md` | Internal run log | Historical demo log | Detailed 300-sample demo run log, environment setup, CPU/GPU issue notes, output artifacts. |
| `修复H3随机基线性能瓶颈_0434aa24.md` | Internal/HPC AI implementation plan | Historical debugging note | Plan for fixing the `analytical_random()` performance bottleneck by group-size deduplication during EC-3/EC-4 supplement. |
| `README.md` | Internal navigation | Current map | This file. Update it whenever the current source of truth or teacher-approved next step changes. |

## Current Key Numbers

R1 ESM-C row-level coverage:

```text
145607 / 145607 rows loaded
fallback due unavailable ESM-C = 0
107731 = unique UniProt / feature UID count, not a coverage gap
```

R1 R→E diagnosis:

| Positive definition | MRR |
|---|---:|
| row-level | 0.0580 |
| UniProt-grouped | 0.0581 |
| EC-2-grouped | 0.9340 |
| EC-3-grouped | 0.933629 |
| EC-4-grouped | 0.918680 |

Hard-negative contamination:

| Pair definition | per-anchor mean | median | max |
|---|---:|---:|---:|
| same-EC-1-digit | 643.96 | 585.00 | 1269 |
| same-EC-2-digit | 182.26 | 169.00 | 529 |
| same-UniProt | 0.03 | 0.00 | 5 |

R2 pass/warning thresholds from the revised plan:

| Metric | R1 measured | R2 pass threshold | Warning / failure threshold |
|---|---:|---:|---:|
| row-level `R→E MRR` | 0.0580 | > 0.12 | < 0.06 means failure |
| UniProt-grouped `R→E MRR` | 0.0581 | > 0.15 | < 0.08 means failure |
| EC-4-grouped `R→E MRR` | 0.918680 | >= 0.868680 | drop > 0.05 means warning |
| EC-2-grouped `R→E MRR` | 0.9340 | >= 0.85 | < 0.70 means severe degradation |
| E→M MRR | 0.609 | >= 0.55 | < 0.40 means failure |

## Next Step Checklist

Current next step:

1. Upload the approved plan to HPC, recommended path:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs/R2_PLAN_v2_REVISED_20260609_175634.md
```

2. Ask HPC AI to patch only:
   - `hard_neg_weight = 1.0`
   - `epochs_stage1 = 25`
   - stage-end checkpoint saving
   - final report grouped metrics

3. Before submitting training, require from HPC AI:
   - unified diff or `git diff`
   - `python -m py_compile train.py`
   - confirmation no unapproved model/loss/loader/EC-encoding changes
   - output directory
   - Slurm script

## Update Rule

Update this README whenever one of these changes:

- teacher approves or rejects a plan
- a new R2/R3 plan becomes the current source of truth
- a new HPC job is submitted or completed
- a new result changes the key numbers
- a file becomes superseded

Recommended update cadence: after every teacher decision, HPC job completion,
or major document creation.

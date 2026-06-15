# Bio Vector Docs README

Last updated: 2026-06-15

Purpose: this folder is the working archive for the Bio Vector / `bio_vector`
Round 1 baseline, Round 2 postmortem, and Round 3 training work. This README is
the navigation map so we do not need to reread every `.md` file each time.

## Current Status

- Round 1 full GVP baseline: complete.
- Round 1 full ESM-C baseline: complete.
- Round 1 diagnosis and EC-3/EC-4 supplement: complete.
- R2 core training: complete.
- R2 postmortem diagnostics: complete and accepted.
- R3 decision input: complete.
- R3 teacher plan: approved for execution.
- R2 EC-4 tail/mid/head bucket baseline: complete.
- R3 `train.py` patch and corrective patch audit: complete.
- R3 run preflight and corrective preflight: complete.
- R3 Slurm training job: submitted and running as of 2026-06-15 11:37 CST.

Current source of truth:

```text
R3_TRAIN_SUBMISSION_20260615.md
```

Current R3 job:

```text
Job ID: 115402116
Output: /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r3_ec4_balanced_stage3skip_2026-06-15/
```

Current R3 training variables:

```python
hard_neg_weight = 1.0
epochs_stage0 = 5
epochs_stage1 = 25
epochs_stage2 = 8
epochs_stage3 = 0
ec4_weighted_sampler = "stage1/stage2, weight=1/sqrt(group_size)"
unknown_ec4_bucket = "__unknown_ec4__"
```

Important boundary:

- Do not use old Option B or the old `R2_PLAN_v2_20260609_093107.md` to train.
- Do not reopen R2 negative-result framing. R2 is a ceiling-hitting /
  evaluation-granularity result, not a training collapse.
- Do not introduce calibration, OOD, latency, or agent thresholds into R3.
- Do not change `hard_neg_weight`; it stays at 1.0.
- Do not treat row-level exact retrieval as the project-facing objective.
  EC-4-grouped R->E is the main R3 retrieval metric.

## What To Read Now

For the current next step, read these in order:

1. `R3_TRAIN_SUBMISSION_20260615.md`
   - current job submission record
   - job ID, Slurm status, output directory, expected artifacts
   - confirms training has been submitted and result analysis has not run yet

2. `R3_RUN_PREFLIGHT_CORRECTIVE_20260615.md`
   - latest preflight state before submission
   - output directory was pre-created
   - run script trap wording was corrected from `FAILED` to `STOPPED`

3. `R3_TRAIN_PATCH_CORRECTIVE_AUDIT_20260615.md`
   - final patch audit for R3 training
   - confirms EC-4 weighted sampler, unknown EC-4 bucket handling, stage3 skip,
     stage2 checkpoint alias, and visualization warning wording

4. `R2_EC4_BUCKET_BASELINE.md`
   - required R2 baseline for R3 tail/mid/head evaluation
   - use this as the comparison point after R3 finishes

5. `BIO_VECTOR_R3_PLAN_FOR_STUDENT_2026-06-15.md`
   - teacher-approved R3 execution document
   - defines the R3 rationale, constraints, and acceptance comparison

6. `R2_POSTMORTEM_20260615_FINAL.md`
   - final R2 postmortem archive
   - use this for R2 evidence and teacher-facing provenance

For a future new chat or handoff, also read:

```text
BIO_VECTOR_NEXT_CHAT_HANDOFF_ROUND2_2026-06-15.md
```

## Current File Index

| File | Direction / Role | Current use | Contains |
|---|---|---|---|
| `R3_TRAIN_SUBMISSION_20260615.md` | Student/Codex -> record | Current source of truth | R3 Slurm submission, job ID 115402116, running status, output path, config, expected artifacts. |
| `R3_RUN_PREFLIGHT_CORRECTIVE_20260615.md` | Internal/HPC audit | Current preflight | Corrected run script state before submission; output directory pre-created; `bash -n` and `py_compile` passed. |
| `run_r3_training.sh` | HPC run script copy | Current run artifact | R3 Slurm script with EC-4 balanced training output path and stage3 skip configuration. |
| `R3_TRAIN_PATCH_CORRECTIVE_AUDIT_20260615.md` | Internal/HPC audit | Current patch audit | Final R3 train.py corrective patch: unknown EC-4 bucket, `1/sqrt(group_size)` sampler, visualization warning wording, stage2 alias check. |
| `R3_TRAIN_PATCH_CORRECTIVE_DIFF_20260615.diff` | Code diff archive | Current patch diff | Corrective diff corresponding to the final R3 patch audit. |
| `R3_TRAIN_PATCH_AUDIT_20260615.md` | Internal/HPC audit | Superseded by corrective audit | Initial R3 train.py patch audit. Keep for history. |
| `R3_TRAIN_PATCH_DIFF_20260615.diff` | Code diff archive | Superseded by corrective diff | Initial R3 patch diff. Keep for history. |
| `R3_TRAIN_PATCH_PREAUDIT_20260615.md` | Internal/HPC audit | Historical audit | Initial preaudit before R3 train patch. |
| `R3_TRAIN_PATCH_PREAUDIT_FOLLOWUP_20260615.md` | Internal/HPC audit | Historical audit | Follow-up preaudit before R3 train patch. |
| `R3_RUN_PREFLIGHT_20260615.md` | Internal/HPC audit | Superseded by corrective preflight | Initial run preflight; output directory issue fixed later. |
| `R2_EC4_BUCKET_BASELINE.md` | Evaluation result | Current R3 baseline | R2 EC-4 tail/mid/head bucket MRR baseline required before judging R3. |
| `BIO_VECTOR_R3_PLAN_FOR_STUDENT_2026-06-15.md` | Teacher -> student/Codex | Current R3 plan | Teacher-approved R3 plan: EC-4 class-balanced sampling, stage3 skip, R2 bucket baseline, hard constraints. |
| `BIO_VECTOR_R3_TASK_B_HPC_INSTRUCTIONS_2026-06-15.md` | Student/Codex -> HPC AI | Completed instruction | Instructions used to compute the R2 EC-4 bucket baseline. |
| `R3_STARTUP_PREREQ_AUDIT_20260615.md` | Internal audit | Completed prerequisite check | R3 startup prerequisite audit. |
| `R3_DECISION_INPUT.md` | Student/Codex -> teacher | Completed decision input | One-page R3 decision input based on completed R2 postmortem. |
| `R2_POSTMORTEM_20260615_FINAL.md` | Student/Codex -> teacher / archive | Final R2 postmortem | Final R2 postmortem after calibration-shape and real-OOD candidate additions. |
| `R2_POSTMORTEM_20260614.md` | Student/Codex -> teacher / archive | Historical draft | Earlier R2 postmortem before final 2026-06-15 additions. |
| `Bio Vector R2 Postmortem.md` | Archive copy | Historical/reference | Alternate R2 postmortem archive copy. |
| `BIO_VECTOR_R2_POSTMORTEM_TASK_CHECKLIST_2026-06-12.md` | Internal checklist | Completed | Checklist for R2 postmortem task completion. |
| `postmortem_eval_stage_checkpoints.py` | Evaluation script | Current reference script | Stage-checkpoint postmortem evaluation helper. |
| `BIO_VECTOR_R2_RESULT_SUMMARY_2026-06-11.md` | Student/Codex -> teacher / internal result archive | Current R2 result reference | R2 metrics, artifact status, visualization failure note, and corrected ceiling-oriented interpretation. |
| `BIO_VECTOR_R2_POSTMORTEM_INSTRUCTIONS_FOR_STUDENT_2026-06-11(1).md` | Teacher -> student/Codex | Completed instruction | Teacher's R2 postmortem instruction and correction. |
| `R2_Postmortem_Task_1.2_db553991.md` | Task note | Historical task note | R2 postmortem Task 1.2 notes. |
| `R2_POSTMORTEM_TASK_1_2_PRERUN_AUDIT.md` | Internal audit | Historical audit | Pre-run audit for Task 1.2. |
| `R2_POSTMORTEM_TASK_1_2_SCRIPT_AUDIT.md` | Internal audit | Historical audit | Script audit for Task 1.2. |
| `R2_POSTMORTEM_TASK_1_2_RUN_RESULT.md` | Result record | Historical result | Run result for Task 1.2. |
| `R2_POSTMORTEM_TASK_1_3_INDIRECT_ATTRIBUTION.md` | Result record | Historical result | Indirect attribution result. |
| `R2_POSTMORTEM_TASK_1_4_VISUALIZATION_DIAGNOSIS.md` | Result record | Historical result | Visualization diagnosis. |
| `R2_POSTMORTEM_TASK_1_5_SM_HISTORY.md` | Result record | Historical result | Stage/model history diagnosis. |
| `Task_1.6_Tool_Baselines_39989c22.md` | Task note | Historical task note | Tool baseline task context. |
| `R2_POSTMORTEM_TASK_1_6_TOOL_BASELINES_SCRIPT_AUDIT.md` | Internal audit | Historical audit | Tool baseline script audit. |
| `R2_POSTMORTEM_TASK_1_6_TOOL_BASELINES_RUN_RESULT.md` | Result record | Historical result | Tool baseline run result. |
| `R2_PLAN_v2_REVISED_20260609_175634.md` | Student/Codex -> teacher, teacher-approved | Historical R2 plan | Approved R2 plan after teacher feedback; no longer current for training. |
| `BIO_VECTOR_R2_TASK0_EC4_SUPPLEMENT_FEEDBACK_FOR_TEACHER_2026-06-09.md` | Student/Codex -> teacher | Historical R2 evidence | EC-3/EC-4 supplement response. |
| `BIO_VECTOR_R1_DIAGNOSIS_RESULT_2026-06-09.md` | Internal record / evidence | Historical R1/R2 evidence | R1 diagnosis summary, H1/H3 evidence, EC-3/EC-4 supplement results. |
| `BIO_VECTOR_R2_PLAN_REVISION_FOR_STUDENT_2026-06-09.md` | Teacher -> student/Codex | Historical instruction | Teacher's R2 revision requirements. |
| `BIO_VECTOR_NEXT_CHAT_HANDOFF_ROUND2_2026-06-15.md` | Internal handoff | Current handoff | Latest compact state for a new chat. |
| `R2_PLAN_v2_20260609_093107.md` | Student/Codex -> teacher | Superseded | Old R2 plan draft. Do not use to train. |
| `BIO_VECTOR_ROUND2_FINAL_DIAGNOSIS_PLAN_2026-06-08.md` | Student/Codex plan | Historical protocol | Diagnosis-first plan before R1 postmortem was run. |
| `BIO_VECTOR_ROUND2_CODEX_PROMPT_FOR_STUDENT_2026-06-08.md` | Teacher -> student/Codex | Historical prompt | Original teacher prompt that started diagnosis-first Round 2 planning. |
| `BIO_VECTOR_ROUND2_PARAMETER_REVIEW_BRIEF_FOR_GPT_2026-06-08.md` | Student/Codex -> external GPT/reviewer | Superseded review brief | Earlier parameter review context. |
| `BIO_VECTOR_ROUND1_FULL_BASELINE_SUMMARY_AND_ROUND2_PLAN_2026-06-07.md` | Internal summary / earlier plan | Historical baseline reference | Full-data R1 GVP/ESM-C baseline summary and older R2 plan. |
| `BIO_VECTOR_TRAIN_PATCH_AND_FULL_RUN_README_2026-06-05.md` | Internal technical runbook | Historical technical reference | Full-data `train.py` compatibility/scalability patches and HPC notes. |
| `BIO_VECTOR_ROUND1_ANALYSIS_AND_ROUND2_PLAN_2026-06-03.md` | Internal early analysis | Historical demo reference | 300-sample CPU demo Round 1 results and early Round 2 suggestions. |
| `BIO_VECTOR_DEMO_RUN_LOG_2026-06-03.md` | Internal run log | Historical demo log | Detailed 300-sample demo run log and output artifacts. |
| `修复H3随机基线性能瓶颈_0434aa24.md` | Internal/HPC AI implementation plan | Historical debugging note | Plan for fixing the `analytical_random()` performance bottleneck during EC-3/EC-4 supplement. |
| `README.md` | Internal navigation | Current map | This file. Update it whenever source of truth, teacher-approved next step, or job state changes. |

## Current Key Numbers

R1 ESM-C row-level coverage:

```text
145607 / 145607 rows loaded
fallback due unavailable ESM-C = 0
107731 = unique UniProt / feature UID count, not a coverage gap
```

R1 R->E diagnosis:

| Positive definition | MRR |
|---|---:|
| row-level | 0.0580 |
| UniProt-grouped | 0.0581 |
| EC-2-grouped | 0.9340 |
| EC-3-grouped | 0.933629 |
| EC-4-grouped | 0.918680 |

R2 measured results:

| Metric | R2 measured | Judgment |
|---|---:|---|
| row-level `R->E MRR` | 0.060208 | reference only |
| UniProt-grouped `R->E MRR` | 0.060708 | reference only |
| EC-2-grouped `R->E MRR` | 0.930647 | pass |
| EC-3-grouped `R->E MRR` | 0.930354 | preserved |
| EC-4-grouped `R->E MRR` | 0.913213 | pass |
| `E->M MRR` | 0.619546 | pass |

R2 EC-4 bucket baseline for R3 comparison:

| Bucket | Rule | n groups | n rows | MRR | top-1 | top-5 | top-10 |
|---|---|---:|---:|---:|---:|---:|---:|
| tail | `<=4` | 1397 | 2783 | 0.904378 | 0.763205 | 0.915199 | 0.937837 |
| mid | `5-317` | 1000 | 45030 | 0.911926 | 0.828692 | 0.915901 | 0.936775 |
| head | `>317` | 127 | 80034 | 0.915733 | 0.850214 | 0.953095 | 0.965240 |
| all valid EC-4 | all | 2524 | 127847 | 0.914145 | 0.840739 | 0.939169 | 0.954618 |

R3 target comparison:

| Metric | R2 baseline | R3 target / use |
|---|---:|---|
| EC-4-grouped `R->E MRR` | 0.913213 | main metric, should not degrade |
| `E->M MRR` | 0.619546 | monitor, target `>=0.61` |
| EC-2-grouped `R->E MRR` | 0.930647 | monitor, target `>=0.9200` |
| EC-4 tail bucket MRR | 0.904378 | target improvement `>= R2 + 5%` under same protocol |
| EC-4 head bucket MRR | 0.915733 | should not drop by more than 3% under same protocol |
| row-level `R->E MRR` | 0.060208 | reference only |

## Next Step Checklist

Current next step:

1. Monitor R3 Slurm job `115402116`.
2. After completion, confirm expected artifacts in
   `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r3_ec4_balanced_stage3skip_2026-06-15/`.
3. Run R3 evaluation and result analysis.
4. Compare R3 against R2 overall and bucket baselines.
5. Create a new R3 result summary document.
6. Update this README again with the final R3 status and numbers.

Useful monitoring commands:

```bash
squeue -j 115402116
tail -f /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r3_ec4_balanced_stage3skip_2026-06-15/r3_train_115402116.out
cat /public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r3_ec4_balanced_stage3skip_2026-06-15/r3_train_115402116.err
```

## Update Rule

Update this README whenever one of these changes:

- teacher approves or rejects a plan
- a new R2/R3 plan becomes the current source of truth
- a new HPC job is submitted or completed
- a new result changes the key numbers
- a file becomes superseded

Recommended update cadence: after every teacher decision, HPC job completion,
or major document creation.

# Bio Vector Current Handoff

Date updated: 2026-06-15

This file supersedes the old 2026-06-08 Round 2 handoff. The current project
state is now:

```text
R2 postmortem accepted by teacher.
Teacher provided the R3 execution document.
Next work is R3 startup preparation, not R2 debugging.
```

## 1. Current Objective

We are continuing the EnzymeCAGE / Bio Vector full-data project.

Current active instruction:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R3_PLAN_FOR_STUDENT_2026-06-15.md
```

Teacher's current position:

- R2 postmortem is accepted.
- R3 decisions have been made by the teacher.
- Codex should not re-litigate whether R3 is needed.
- Before R3 training, two zero-GPU R2 supplemental diagnostics must be completed.
- Before R3 validation, R2 EC-4 tail/mid/head bucket baselines must be computed.

Do not start R3 training until:

1. `R2_POSTMORTEM_20260615_FINAL.md` is created with the two teacher-requested
   Task 1.6 addenda.
2. `R2_EC4_BUCKET_BASELINE.md` is produced from the R2 checkpoint/embeddings.
3. `eval_ec4_buckets.py` is created, audited, and archived.
4. Any R3 `train.py` patch has a focused diff audit and explicit approval.

## 2. Working Style For The Next Chat

The user coordinates with an HPC-side AI. The local Codex/chat should act as
the careful reviewer and instruction writer.

### Teacher Wording Guardrails

For R2 postmortem / R2 summary / teacher-facing interpretation documents, do
not use these scientific framing phrases:

```text
controlled negative result
negative result
failure
failed
catastrophic
does not meet
success/failure
hard threshold
hard thresholds
```

Use the teacher-approved framing instead:

```text
ceiling-aware result
threshold-design correction
EC-4-grouped retrieval is primary
row-level / UniProt-level retrieval is reference only
Task 1.6 is baseline-only; no tool-oriented acceptance criteria are set
```

Important nuance:

- `hard-negative` is a technical term and may appear when discussing
  `hard_neg_weight`; it is not the same as calling R2 a negative result.
- `R3 plan` now exists as a teacher-provided document dated 2026-06-15. The old
  restriction was not to write an unapproved R3 plan during R2 postmortem.
  Going forward, do not write unapproved R4/R5 plans or add training details
  outside the teacher-provided R3 instructions.

Required workflow:

1. Local Codex reads the relevant teacher instruction and current local docs.
2. Local Codex writes detailed, copy-paste-ready HPC AI instructions.
3. The HPC AI must write every result into a dedicated markdown result file,
   not only into chat output.
4. The result markdown should be similar in style to:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/R2_POSTMORTEM_TASK_1_6_TOOL_BASELINES_RUN_RESULT.md
```

5. The user copies the returned markdown content or file back to local Codex.
6. Local Codex audits the result, checks it against the teacher requirement,
   and only then syncs it into final teacher-facing documents.
7. A daily log must be updated while work proceeds.
8. A task-level result document must be kept while work proceeds.
9. Only after all task-level checks pass should the result be archived into a
   final teacher-facing version.

Do not:

- Do not rely on raw chat output as the only record.
- Do not silently change training logic.
- Do not modify `train.py` without a focused diff and approval.
- Do not submit Slurm jobs unless the current step explicitly requires one.
- Do not make up missing numbers.
- Do not compress an unsuccessful or partial HPC run into "looks fine".
- Do not write R4/R5 plans unless the teacher asks.

Preferred HPC result markdown structure:

```text
# <Task Name> Run Result

## 1. Purpose
## 2. Commands / Script Paths
## 3. Environment
## 4. Inputs
## 5. Outputs
## 6. Metrics / Tables
## 7. Checks
## 8. Interpretation
## 9. Declarations
- train.py modified: yes/no
- sbatch executed: yes/no
- GPU/DCU used: yes/no
- retraining executed: yes/no
- new thresholds introduced: yes/no
```

For every HPC instruction, ask the HPC AI to return:

- exact file paths
- `ls -lh` for produced files
- `py_compile` / `bash -n` where relevant
- `sacct` / `squeue` if a Slurm job is submitted
- stdout/stderr tail if a Slurm job is submitted
- final markdown result path

## 3. Current Local Documents To Read First

Read these first in a new chat:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R3_PLAN_FOR_STUDENT_2026-06-15.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/R2_POSTMORTEM_20260614.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R2_RESULT_SUMMARY_2026-06-11.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R2_POSTMORTEM_TASK_CHECKLIST_2026-06-12.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/R3_DECISION_INPUT.md
/home/a/EnzymeCAGE/custom/docs/DAILY_LOG_2026-06-14.md
```

Also useful provenance files:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/R2_POSTMORTEM_TASK_1_2_RUN_RESULT.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/R2_POSTMORTEM_TASK_1_2_SCRIPT_AUDIT.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/R2_POSTMORTEM_TASK_1_3_INDIRECT_ATTRIBUTION.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/R2_POSTMORTEM_TASK_1_4_VISUALIZATION_DIAGNOSIS.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/R2_POSTMORTEM_TASK_1_5_SM_HISTORY.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/R2_POSTMORTEM_TASK_1_6_TOOL_BASELINES_RUN_RESULT.md
```

## 4. R2 Postmortem Final State

R2 postmortem teacher submission package exists locally:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/R2_POSTMORTEM_20260614.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R2_RESULT_SUMMARY_2026-06-11.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/postmortem_eval_stage_checkpoints.py
/home/a/EnzymeCAGE/custom/docs/bio_vector/R3_DECISION_INPUT.md
```

Important provenance:

- `R2_POSTMORTEM_20260614.md` was created from the working total report
  `Bio Vector R2 Postmortem.md`.
- `postmortem_eval_stage_checkpoints.py` was restored from the Task 1.2 script
  audit. It is not newly invented code.
- `R3_DECISION_INPUT.md` was written locally from completed postmortem evidence.
  It is a decision input, not an R3 training plan.
- `DAILY_LOG_2026-06-14.md` records Task 1.1-1.6, Task 2, Task 3, and final
  submission package preparation.

R2 postmortem summary:

| Task | Status | Key result |
|---|---|---|
| 1.1 data ceiling | complete | EC-4 mean rows/group = 50.65; top-1 ceiling ~= 0.019742 |
| 1.2 stage-wise eval | complete | stage1 EC-4 MRR = 0.922020; stage3 = 0.917949 |
| 1.3 attribution | complete | R1 stage checkpoints absent; direct attribution not identifiable |
| 1.4 visualization diagnosis | complete | embeddings and NN indices healthy; PNG issue visualization-only |
| 1.5 S->M history | complete | R1 0.6108 vs R2 0.5871; monitoring metric |
| 1.6 tool baselines | complete | ECE 0.106888; OOD proxy separated; latency p95 5.2403 ms |
| Task 2 summary rewrite | complete | R2 framed as threshold-design correction / ceiling-aware result |
| Task 3 final files | complete | four required deliverables prepared |

## 5. Current R3 Plan From Teacher

Teacher-provided R3 plan:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R3_PLAN_FOR_STUDENT_2026-06-15.md
```

Teacher says R3 has three main changes:

1. EC-4 class-balanced sampling using `1/sqrt(group_size)` for stage1/stage2.
2. Set `epochs_stage3 = 0` and skip stage3.
3. Wrap optional `visualize_four_modal()` so PNG rendering issues do not change
   exit status after core artifacts are saved.

R3 should not change:

- `hard_neg_weight` stays `1.0`.
- `epochs_stage1` stays `25`.
- `epochs_stage2` stays `8`.
- EC encoding is not a target change.
- Concept loss / VICReg are not target changes.
- Row-level R->E is reference only, not the primary objective.
- Calibration / OOD / latency remain baseline records, not R3 acceptance
  criteria.

R3 main evaluation:

| Metric | R2 baseline | R3 target |
|---|---:|---:|
| EC-4-grouped R->E MRR | 0.9132 | >= 0.9132 |
| E->M MRR | 0.6195 | >= 0.61 |
| EC-2-grouped R->E MRR | 0.9306 | >= 0.9200 |
| EC-4 long-tail bucket MRR | R2 must be computed first | >= R2 + 5% |
| EC-4 head bucket MRR | R2 must be computed first | >= R2 - 3% |
| row-level R->E MRR | 0.0602 | reference only |

## 6. Immediate Next Tasks

### Task A: R2 Postmortem Final Addenda

Update `R2_POSTMORTEM_20260614.md` by appending teacher-requested sections:

1. `1.6.1 Calibration Shape Qualitative (added 2026-06-15)`
2. `1.6.2 Real OOD Candidate Sources (added 2026-06-15)`

Then archive as:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/R2_POSTMORTEM_20260615_FINAL.md
```

This is local documentation work. It does not require HPC.

### Task B: R2 EC-4 Bucket Baseline

Before R3 training, compute R2 EC-4 bucket MRR baseline:

```text
tail: EC-4 group size <= 4
mid:  4 < EC-4 group size <= 317
head: EC-4 group size > 317
```

Expected deliverables:

```text
code/demo/eval_ec4_buckets.py
docs/R2_EC4_BUCKET_BASELINE.md
```

This should be run on R2 saved outputs/embeddings. It should not modify
`train.py`, should not retrain, and should not define new calibration/OOD/latency
criteria.

The HPC AI should write a result markdown similar to:

```text
docs/R2_EC4_BUCKET_BASELINE.md
```

The local user will paste that markdown back for audit.

### Task C: R3 Patch Audit

Only after Task A and Task B pass:

1. Draft precise HPC AI patch instructions for `train.py`.
2. Patch only:
   - EC-4 class-balanced sampler for stage1/stage2
   - stage3 skip for `epochs_stage3 = 0`
   - optional visualization try/except
3. Run `python3 -m py_compile train.py`.
4. Produce focused diff.
5. Write a patch audit markdown.
6. Do not submit R3 training until the diff is reviewed.

## 7. HPC Paths

HPC work root:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04
```

HPC code:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/train.py
```

HPC data:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/data/reaction_enzyme_microbe_training_clean_2026-06-01_LOCAL
```

R2 output:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11
```

R2 key files on HPC:

```text
model_v3.pt
model_v3_stage0.pt
model_v3_stage1.pt
model_v3_stage2.pt
model_v3_stage3.pt
embeddings_v3.npz
metadata_v3.json
metrics_v3.json
training_history.json
reaction_nn_index.npz
enzyme_nn_index.npz
substrate_nn_index.npz
microbe_nn_index.npz
```

## 8. HPC Environment

For DCU training:

```bash
#SBATCH -p kshdnormal04
#SBATCH --gres=dcu:1

module load compiler/dtk/23.10
export HSA_OVERRIDE_GFX_VERSION=9.0.6
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nis
```

For CPU-only postmortem work, prefer CPU-only partition if available:

```text
kshctest02
```

Important cluster note from prior work:

- `kshdnormal04` may require DCU by QOS.
- CPU-only diagnostics should avoid consuming DCU when possible.
- `DefMemPerCPU` was observed around 3500 MB; choose memory accordingly.

## 9. Logging Rules Going Forward

Continue daily logging.

Current logs:

```text
/home/a/EnzymeCAGE/custom/docs/DAILY_LOG_2026-06-14.md
```

Create/update for current work:

```text
/home/a/EnzymeCAGE/custom/docs/DAILY_LOG_2026-06-15.md
```

For each task, log:

- teacher source instruction
- exact local files read
- exact HPC instruction sent
- HPC job ID if any
- output markdown path
- result summary
- verification commands
- whether `train.py` was modified
- whether Slurm was used
- whether GPU/DCU was used
- whether retraining occurred
- final teacher-facing archive path

## 10. Copy-Paste Prompt For Next Chat

Use this to start a new chat:

```text
We are continuing EnzymeCAGE / Bio Vector after R2 postmortem acceptance.

First read:
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R3_PLAN_FOR_STUDENT_2026-06-15.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_NEXT_CHAT_HANDOFF_ROUND2_2026-06-08.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/R2_POSTMORTEM_20260614.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R2_RESULT_SUMMARY_2026-06-11.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/R3_DECISION_INPUT.md
/home/a/EnzymeCAGE/custom/docs/DAILY_LOG_2026-06-14.md

Current status:
- R2 postmortem accepted by teacher.
- R3 plan is teacher-provided and active.
- Do not re-decide whether R3 is needed.
- Do not train yet.
- First do R3 startup preparation:
  1. append calibration-shape and real-OOD candidate-source addenda to R2 postmortem;
  2. archive as R2_POSTMORTEM_20260615_FINAL.md;
  3. prepare detailed HPC AI instructions to compute R2 EC-4 tail/mid/head bucket baseline;
  4. require HPC AI to write results into docs/R2_EC4_BUCKET_BASELINE.md;
  5. user will paste that markdown back for audit.

Working mode:
- Give me precise HPC AI commands/prompts.
- Require the HPC AI to write a result markdown for every task.
- I paste the result markdown back.
- You audit it, update daily log and final docs, then give next instruction.
- Do not modify train.py unless explicitly doing an approved patch audit.
- Do not submit R3 training until R2 bucket baseline and R3 patch diff are reviewed.
```

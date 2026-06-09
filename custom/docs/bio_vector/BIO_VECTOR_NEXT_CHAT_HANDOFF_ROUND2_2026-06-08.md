# Bio Vector Round 2 Next-Chat Handoff

Date: 2026-06-08

This handoff is for continuing the `bio_vector` full-data Round 2 work in a new
chat. It is written so the next Codex/chat and the HPC-side AI can quickly
recover context without re-discovering the whole history.

## 1. Current Objective

We are running the teacher-provided `bio_vector` demo on the full EnzymeCAGE
training data.

The demo trains a unified four-modal vector space:

```text
Reaction  -> 256d
Enzyme    -> 256d
Substrate -> 256d
Microbe   -> 256d
```

The model is trained for cross-modal retrieval:

- Reaction -> Enzyme
- Enzyme -> Microbe
- Substrate -> Microbe

We completed Round 1 full baselines:

- Full GVP baseline
- Full ESM-C baseline

Important 2026-06-08 update:

- The earlier plan to immediately run ESM-C-R2 Option B is now paused.
- The teacher reviewed the Round 1 results and instructed us to diagnose the
  Round 1 failure before any retraining.

Important 2026-06-09 update:

- The Round 1 postmortem diagnosis is now complete.
- Teacher requested an EC-4 supplement before R2 training; this Task 0
  supplement is now complete and audited.
- EC-3-grouped R→E MRR: `0.933629`.
- EC-4-grouped R→E MRR: `0.918680`.
- This supports the revised narrative: R1 learned strong EC-family /
  EC-functional structure, while row-level / UniProt-level fine discrimination
  remains low.
- Revised R2 training variables should be limited to:
  - `hard_neg_weight = 1.0`
  - `epochs_stage1 = 25`
- Remove EC encoding as a training variable for R2.
- Diagnosis report on HPC:
  `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07/R1_DIAGNOSIS_20260608_172625.md`
- Local diagnosis summary:
  `/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R1_DIAGNOSIS_RESULT_2026-06-09.md`
- Teacher-facing Task 0 feedback:
  `/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R2_TASK0_EC4_SUPPLEMENT_FEEDBACK_FOR_TEACHER_2026-06-09.md`
- Do not submit any R2 training job until the revised `R2_PLAN_v2_<timestamp>.md`
  is drafted and approved.

## 2. Work Mode

The user is coordinating with an HPC-side AI.

Workflow:

1. Codex gives precise HPC AI instructions.
2. User sends those instructions to the HPC AI.
3. HPC AI executes on HPC and returns logs/results.
4. User pastes results back to Codex.
5. Codex analyzes results, updates logs/docs, and gives the next instruction.

Important:

- Do not let the HPC AI silently change training logic.
- Any `train.py` change must be audited and categorized:
  - data loader / environment compatibility
  - evaluation/artifact handling
  - Config hyperparameters
  - training/model/loss changes
- Training/model/loss changes require explicit approval.
- Prefer smoke/probe runs before expensive full runs.
- Keep detailed logs in local docs.

## 3. Key Local Documents To Read First

The next Codex/chat should read these files before giving new instructions:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_ROUND2_FINAL_DIAGNOSIS_PLAN_2026-06-08.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R1_DIAGNOSIS_RESULT_2026-06-09.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R2_TASK0_EC4_SUPPLEMENT_FEEDBACK_FOR_TEACHER_2026-06-09.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R2_PLAN_REVISION_FOR_STUDENT_2026-06-09.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_ROUND2_CODEX_PROMPT_FOR_STUDENT_2026-06-08.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_ROUND2_PARAMETER_REVIEW_BRIEF_FOR_GPT_2026-06-08.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_ROUND1_FULL_BASELINE_SUMMARY_AND_ROUND2_PLAN_2026-06-07.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_TRAIN_PATCH_AND_FULL_RUN_README_2026-06-05.md
/home/a/EnzymeCAGE/custom/docs/DAILY_LOG_2026-06-07.md
/home/a/EnzymeCAGE/custom/docs/DAILY_LOG_2026-06-08.md
/home/a/EnzymeCAGE/custom/docs/DATABASE_PROGRESS.md
```

Purpose of each:

- `BIO_VECTOR_ROUND2_FINAL_DIAGNOSIS_PLAN_2026-06-08.md`
  - Teacher-approved diagnosis protocol before the diagnosis was run.
- `BIO_VECTOR_R1_DIAGNOSIS_RESULT_2026-06-09.md`
  - Completed diagnosis result. This is now the primary source for what to do
    next.
- `BIO_VECTOR_ROUND2_CODEX_PROMPT_FOR_STUDENT_2026-06-08.md`
  - Teacher-provided current instruction. This supersedes the immediate
    ESM-C-R2 Option B training plan.
- `BIO_VECTOR_ROUND2_PARAMETER_REVIEW_BRIEF_FOR_GPT_2026-06-08.md`
  - Historical Round 2 parameter proposal and external-review discussion,
    now superseded by the teacher's diagnosis-first instruction.
- `BIO_VECTOR_ROUND1_FULL_BASELINE_SUMMARY_AND_ROUND2_PLAN_2026-06-07.md`
  - Teacher-facing Round 1 results and the older second-round plan; its R2
    training recommendation is superseded.
- `BIO_VECTOR_TRAIN_PATCH_AND_FULL_RUN_README_2026-06-05.md`
  - All `train.py` compatibility patches and HPC issues.
- `DAILY_LOG_2026-06-07.md`
  - Full ESM-C baseline completion and Round 1 closure.
- `DAILY_LOG_2026-06-08.md`
  - External GPT review, corrected ESM-C coverage interpretation, updated
    Round 2 plan.
- `DATABASE_PROGRESS.md`
  - Full data provenance and ESM-C UID/export facts.

## 4. Important HPC Paths

HPC root:

```text
/public/home/acfbwjsi7s
```

Work root:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04
```

Code:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/train.py
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/README.md
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/diagnose_effective_rank.py
```

Full data directory:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/data/reaction_enzyme_microbe_training_clean_2026-06-01_LOCAL
```

ESM-C feature directory:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/data/reaction_enzyme_microbe_training_clean_2026-06-01_LOCAL/features/enzyme/esm_c_features
```

Logs:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/logs
```

Outputs:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs
```

Round 1 output directories:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_gvp_baseline_microbe_fixed_2026-06-06
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07
```

## 5. HPC Environment

We are not using CPU-only and not using NVIDIA GPU.

We use Hygon/Sugon DCU accelerator:

- partition: `kshdnormal04`
- accelerator: Hygon DCU
- model seen in logs: `Z200SM_71_S` / `Z200SM_71`
- architecture: `gfx906`
- each node has 4 DCUs
- each job uses 1 DCU:

```bash
#SBATCH -p kshdnormal04
#SBATCH --gres=dcu:1
```

Required environment:

```bash
module load compiler/dtk/23.10
export HSA_OVERRIDE_GFX_VERSION=9.0.6
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nis
```

PyTorch appears via `torch.cuda`, but this is DTK/HIP/DCU, not NVIDIA CUDA.

## 6. Full Data Facts

Final full training data:

- training rows: `145,607`
- unique UniProt IDs: `107,731`
- unique RHEA IDs: `6,278`
- unique canonical reaction SMILES: `4,541`
- unique assemblies: `2,475`

Important ESM-C coverage clarification:

- `107,731` is the number of unique UniProt IDs / unique ESM-C feature files.
- `145,607` is the number of example rows.
- This is not 74% coverage.
- Many rows share the same UniProt ID.
- Full ESM-C baseline loaded:

```text
ESM-C loaded rows: 145,607 / 145,607
row-level missing/fallback due unavailable ESM-C: 0
```

Database facts from `DATABASE_PROGRESS.md`:

```text
ESM-C request unique UID: 107,731
successful UID: 107,731
failed UID: 0
missing UID: 0
```

## 7. Completed Compatibility Patches

The following patches are already in HPC `train.py`.

They are compatibility/scalability fixes, not changes to the teacher's training
objective.

### GVP Loader

- Full GVP data is sharded `.pt`, not per-protein `.npz`.
- Loader now reads `gvp_shard_file`.
- Added `gvp_shard_pool()`.
- Batch loads by shard to avoid huge I/O.
- GVP coverage validated:

```text
145,607 / 145,607 loaded
```

### Microbe Loader

- Full JSONL is flat.
- Required microbe numeric values are in three CSV tables:

```text
tables/microbe_reaction_core_preference.csv
tables/microbe_reaction_stoich_query.csv
tables/microbe_reaction_main_metabolite_coverage.csv
```

- Loader now builds nested records from those CSVs.
- Microbe probe passed:

```text
microbe_feats shape: (145607, 28)
nonzero element ratio: 40.19%
all-zero rows: 0/145607
concept targets: all 8 columns non-NaN, 47.33% nonzero
assembly_id non-empty: 145607/145607
enzyme2microbe index: 107731 enzymes mapped
```

### ESM-C Loader

- Pre-scans ESM-C directory with `os.listdir()` to avoid 145k `.exists()` calls.
- Preserves fallback semantics:

```text
ESM-C -> GVP -> AAC
```

### Evaluation / Artifacts

- Original full dense `N x N` evaluation OOMed.
- Evaluation is now chunked.
- Checkpoint is saved before evaluation.
- Native FAISS fails due FAISS SWIG / NumPy incompatibility on HPC.
- Fallback normalized retrieval files are saved:

```text
reaction_nn_index.npz
enzyme_nn_index.npz
substrate_nn_index.npz
microbe_nn_index.npz
```

## 8. Round 1 Results

### Full GVP Baseline

Output:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_gvp_baseline_microbe_fixed_2026-06-06
```

Metrics:

| Metric | GVP full |
|---|---:|
| `R→E top-1` | 0.0118 |
| `R→E top-10` | 0.0671 |
| `R→E MRR` | 0.044 |
| `E→M top-1` | 0.0017 |
| `E→M MRR` | 0.395 |
| `S→M top-1` | 0.0054 |
| `S→M MRR` | 0.642 |

Effective rank:

| Modality | GVP erank |
|---|---:|
| reaction | 28.31 |
| enzyme | 30.56 |
| substrate | 56.58 |
| microbe | 38.21 |

### Full ESM-C Baseline

Output:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07
```

Metrics:

| Metric | ESM-C full |
|---|---:|
| `R→E top-1` | 0.0161 |
| `R→E top-10` | 0.0896 |
| `R→E MRR` | 0.058 |
| `E→M top-1` | 0.0036 |
| `E→M MRR` | 0.609 |
| `S→M top-1` | 0.0056 |
| `S→M MRR` | 0.611 |

Effective rank:

| Modality | ESM-C erank |
|---|---:|
| reaction | 43.72 |
| enzyme | 43.54 |
| substrate | 62.33 |
| microbe | 41.96 |

### Round 1 Interpretation

- ESM-C is the stronger first-round baseline overall.
- ESM-C is better on:
  - `R→E`
  - `E→M`
  - effective-rank balance
- GVP is slightly better on:
  - `S→M MRR`
- Absolute `R→E` retrieval remains low.
- Row-level top-k may underestimate biologically valid retrieval because many
  rows share UniProt IDs, EC classes, reactions, assemblies, or species.

## 9. External GPT Review And Updated Round 2 Plan

An external GPT reviewed the original proposal.

Accepted feedback:

1. Earlier draft incorrectly described Stage 1.
   - Correct:
     - Stage 1 trains `R↔E` and `E↔M`.
     - `S↔M` enters in Stage 2.
2. Increasing `hard_neg_weight` immediately may be risky.
   - It directly affects `R↔E`.
   - It may indirectly disrupt the strong `E→M` alignment through the shared
     enzyme projector.
3. A more conservative single-run Round 2 should isolate longer Stage 1/2
   training before changing hard negatives.

Rejected feedback:

- The external GPT misread `107,731 / 145,607` as 74% ESM-C coverage.
- Correct: unique UID count vs row count. Row-level ESM-C coverage is complete.

## 10. Completed Round 1 Postmortem Diagnosis

The teacher-provided prompt changed the strategy, and the teacher agreed with
the final diagnosis-first version documented in:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_ROUND2_FINAL_DIAGNOSIS_PLAN_2026-06-08.md
```

That diagnosis has now completed.

HPC diagnosis artifacts:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/diagnose_round1_postmortem.py
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07/R1_DIAGNOSIS_20260608_172625.md
```

Run details:

```text
Slurm job ID: 115034071
runtime: 13,509.2 seconds (~3.75 h)
stderr: empty
train.py modified: no
training job submitted: no
```

Key R→E diagnosis:

| Positive definition | top-1 | top-5 | top-10 | MRR |
|---|---:|---:|---:|---:|
| row-level | 0.0111 | 0.0520 | 0.0896 | 0.0575 |
| UniProt-grouped | 0.0262 | 0.0654 | 0.0945 | 0.0581 |
| EC-2-digit-grouped | 0.8838 | 0.9282 | 0.9608 | 0.9340 |

Interpretation:

- Row-level MRR reproduces R1.
- UniProt grouping barely improves MRR, so H2 is not the main cause.
- EC-2-digit grouped MRR is very high, so the model learned strong coarse
  EC-2 biological signal.
- The remaining issue is low row/UniProt-level R→E fine discrimination.

Hard-negative contamination:

| Pair definition | per-anchor mean | median | max |
|---|---:|---:|---:|
| same-EC-1-digit | 643.96 | 585.00 | 1269 |
| same-EC-2-digit | 182.26 | 169.00 | 529 |
| same-UniProt | 0.03 | 0.00 | 5 |

ESM-C coverage sanity remains:

```text
full rows: 145607
unique UniProt / ESM-C feature count: 107731
ESM-C loaded rows: 145607 / 145607
fallback due unavailable ESM-C: 0
```

## 11. Teacher Decision Tree For R2

The diagnosis triggered the H1 + H3 branch:

```text
same-EC-1-digit per-anchor mean = 643.96 > 500
UniProt-grouped R→E MRR = 0.0581 < 0.20
```

Recommended R2 direction from the decision tree:

```python
hard_neg_weight = 1.0
epochs_stage1 = 25
EC encoding = first two EC digits
epochs_stage2 = 8  # unchanged
```

Constraints:

- Do not submit R2 training before `R2_PLAN_v2_<timestamp>.md` is drafted and
  approved.
- Do not increase `hard_neg_weight` to `3.0`.
- Do not simultaneously change Stage 1, Stage 2, and hard-negative weight.
- Grouped metrics become primary reporting metrics; row-level metrics remain
  reference metrics.
- Future R2 training must save per-stage checkpoints named
  `model_v3_stage{N}.pt`.

## 12. Recommended Immediate Next Steps

Do this next:

1. Draft `R2_PLAN_v2_<timestamp>.md`.
2. State the hypothesis:
   - H1: EC-1 hard-negative mining is too coarse.
   - H3: Stage 1 is undertrained.
3. Specify proposed R2 changes:
   - `hard_neg_weight = 1.0`
   - `epochs_stage1 = 25`
   - EC label encoding changed from first digit to first two EC digits
   - `epochs_stage2 = 8` unchanged
4. Include required stage-end checkpoint saving.
5. Wait for teacher approval before any R2 training job.

## 13. Copy-Paste Prompt For The Next Chat

Use this as the opening prompt in a new Codex chat:

```text
We are continuing EnzymeCAGE bio_vector full-data Round 2 work.

Please first read:
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R1_DIAGNOSIS_RESULT_2026-06-09.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_ROUND2_FINAL_DIAGNOSIS_PLAN_2026-06-08.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_ROUND2_CODEX_PROMPT_FOR_STUDENT_2026-06-08.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_NEXT_CHAT_HANDOFF_ROUND2_2026-06-08.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_ROUND2_PARAMETER_REVIEW_BRIEF_FOR_GPT_2026-06-08.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_ROUND1_FULL_BASELINE_SUMMARY_AND_ROUND2_PLAN_2026-06-07.md
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_TRAIN_PATCH_AND_FULL_RUN_README_2026-06-05.md
/home/a/EnzymeCAGE/custom/docs/DAILY_LOG_2026-06-07.md
/home/a/EnzymeCAGE/custom/docs/DAILY_LOG_2026-06-08.md

Current status:
- Full GVP baseline complete.
- Full ESM-C baseline complete.
- ESM-C row-level coverage is complete: 145,607/145,607 rows loaded; 107,731 is unique UID count, not coverage gap.
- External GPT review was assessed.
- Teacher reviewed the R2 plan and agreed with the final diagnosis-first strategy.
- R1 postmortem diagnosis completed on HPC.
- Decision tree triggered H1 + H3 branch.
- R2 plan v2 draft created:
  `/home/a/EnzymeCAGE/custom/docs/bio_vector/R2_PLAN_v2_20260609_093107.md`
- Do not patch code or submit any R2 retraining job before teacher approval.

Work mode:
- You give me precise HPC AI instructions.
- I send them to HPC AI.
- HPC AI runs jobs and returns logs.
- You analyze results and update docs/logs.

Next task:
Review `R2_PLAN_v2_20260609_093107.md` with the teacher. Do not modify
train.py or submit training until the revised plan is approved.
```

## 14. Next Work Item: Teacher Review Of R2 Plan

The diagnosis is complete and the R2 plan draft has been created:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/R2_PLAN_v2_20260609_093107.md
```

This plan is based on:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R1_DIAGNOSIS_RESULT_2026-06-09.md
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07/R1_DIAGNOSIS_20260608_172625.md
```

Plan content:

- hypothesis:
  - H1: EC-1 hard-negative mining is too coarse
  - H3: Stage 1 is undertrained
- proposed changes:
  - `hard_neg_weight = 1.0`
  - `epochs_stage1 = 25`
  - EC label encoding changed from first digit to first two EC digits
  - `epochs_stage2 = 8` unchanged
- unchanged settings:
  - ESM-C enzyme feature
  - `batch_size = 4096`
  - `lr = 3e-4`
  - `epochs_stage0 = 8`
  - `epochs_stage3 = 10`
  - `temp_end = 0.05`
  - `w_sm = 0.5`
  - `vicreg_var_weight = 10.0`
- required artifact patch:
  - save stage-end checkpoints for Stage 0/1/2/3
- required metrics after R2:
  - row-level, UniProt-grouped, and EC-2-digit-grouped R→E metrics
  - same metrics for E→M and S→M if feasible
  - effective-rank summary
  - per-stage R→E evolution if stage checkpoints are evaluable
- approval boundary:
  - do not submit R2 training until the plan is approved

Next action after teacher approval:

1. Prepare an HPC AI instruction to patch `train.py`.
2. Patch only approved R2 changes.
3. Audit the diff before training.
4. Submit R2 only after diff approval.

# R2 Postmortem Task 1.2 — Stage-Wise Checkpoint Evaluation Run Result

**Date**: 2026-06-12
**Evaluator**: R2 postmortem pipeline (postmortem_eval_stage_checkpoints.py)

---

## 1. Job Information

| Field | Value |
|---|---|
| Job ID | 115259334 |
| Job Name | r2_postmortem_eval |
| Partition | kshdnormal04 |
| Node | f13r2n11 |
| State | **COMPLETED** |
| Elapsed | 05:34:29 |
| Exit Code | 0:0 (success) |
| ReqMem | 28552M (default: 8 × 3500 MB DefMemPerCPU) |
| MaxRSS | 25,328,396K (≈ 24.2 GB) |
| AllocTRES | billing=8, cpu=8, gres/dcu=1, mem=28552M, node=1 |

```
sacct -j 115259334 --format=JobID,JobName,Partition,NodeList,State,Elapsed,ExitCode,ReqMem,MaxRSS,AllocTRES

JobID            JobName        Partition  NodeList  State     Elapsed  ExitCode  ReqMem     MaxRSS     AllocTRES
115259334        r2_postmortem+ kshdnormal04 f13r2n11 COMPLETED 05:34:29 0:0      28552M               billing=8,cpu=8,gres/dcu=1,mem=28552M,node=1
115259334.batch  batch                     f13r2n11 COMPLETED 05:34:29 0:0                 25328396K  cpu=8,gres/dcu=1,mem=28552M,node=1
115259334.extern extern                    f13r2n11 COMPLETED 05:34:29 0:0                 3344K      billing=8,cpu=8,gres/dcu=1,mem=28552M,node=1
```

---

## 2. stdout / stderr

**stdout**: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/postmortem_eval_115259334.out`
- Size: 4,903 bytes
- Modified: 2026-06-12 21:04:42

**stderr**: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/postmortem_eval_115259334.err`
- Size: 8,590,813 bytes (≈ 8.2 MB)
- Modified: 2026-06-12 18:34:57

**stderr explanation**: stderr contains **only** `DEPRECATION WARNING: please use MorganGenerator` — exactly 145,607 lines (one per sample during Morgan fingerprint generation). **No traceback, no error, no unexpected messages.**

### stdout tail -n 120 (complete):

```
=== R2 Postmortem Stage Evaluation ===
hostname: f13r2n11
date: Fri Jun 12 15:30:14 CST 2026
job_id: 115259334
conda env: nis
python: /public/home/acfbwjsi7s/miniconda3/envs/nis/bin/python
  Checkpoint found: .../model_v3_stage0.pt (19.7 MB)
  Checkpoint found: .../model_v3_stage1.pt (19.7 MB)
  Checkpoint found: .../model_v3_stage2.pt (19.7 MB)
  Checkpoint found: .../model_v3_stage3.pt (19.7 MB)

=== Loading data (once for all stages) ===
  Microbe records from CSV tables: 145607
  ESM-C available UIDs: 107731
  Loaded 145607 examples from EnzymeCAGE 300 dataset
  Reaction (DRFP) dim: 2048
  Enzyme dim: 1152 (ESM-C pocket-pooled)
  Substrate Morgan FP: 145607/145607 parsed
  Microbe metabolic features: 145607/145607 loaded (dim=28)
  ESM-C pocket-pooled: 145607/145607
  Unique assemblies: 2475
  Total samples: 145607
  Dims: reaction=2048, enzyme=1152, substrate=2048, microbe=28
  Data loaded in 10972.5s
  R2 final metrics loaded from .../metrics_v3.json

=== Evaluating 4 checkpoint(s) ===

--- Stage 0: model_v3_stage0.pt ---
  Model loaded (4,831,752 params)
  Running row-level evaluation (chunk_size=4096)...
  Chunked R→E retrieval: 145607 queries, chunk_size=4096
  Chunked E→M retrieval: 145607 queries, chunk_size=4096
  Chunked S→M retrieval: 145607 queries, chunk_size=4096
  Running grouped R→E evaluation (chunk_size=4096)...
  Stage 0 evaluated in 2501.5s
  row R→E MRR:          0.000092
  UniProt-grouped MRR:  0.000092
  EC-4-grouped MRR:     0.007326
  E→M MRR:              0.000719

--- Stage 1: model_v3_stage1.pt ---
  Model loaded (4,831,752 params)
  Stage 1 evaluated in 2161.1s
  row R→E MRR:          0.058593
  UniProt-grouped MRR:  0.059127
  EC-4-grouped MRR:     0.922020
  E→M MRR:              0.610441

--- Stage 2: model_v3_stage2.pt ---
  Model loaded (4,831,752 params)
  Stage 2 evaluated in 2155.7s
  row R→E MRR:          0.060464
  UniProt-grouped MRR:  0.061094
  EC-4-grouped MRR:     0.926472
  E→M MRR:              0.622604

--- Stage 3: model_v3_stage3.pt ---
  Model loaded (4,831,752 params)
  Stage 3 evaluated in 2141.8s
  row R→E MRR:          0.060177
  UniProt-grouped MRR:  0.060711
  EC-4-grouped MRR:     0.917949
  E→M MRR:              0.621205

=== Stage 3 Consistency Check ===
  row_RE_MRR: actual=0.060177, expected=0.060208, rel_err=0.0005 [PASS]
  UniProt_grouped_RE_MRR: actual=0.060711, expected=0.060708, rel_err=0.0001 [PASS]
  EC4_grouped_RE_MRR: actual=0.917949, expected=0.913213, rel_err=0.0052 [PASS]
  EM_MRR: actual=0.621205, expected=0.619546, rel_err=0.0027 [PASS]
  Consistency check: ALL PASS

================================================================================
Stage-Wise MRR Evolution Table
================================================================================
| Stage    |    row R→E MRR |    UniProt MRR |       EC-4 MRR |        E→M MRR |
|----------|----------------|----------------|----------------|----------------|
| stage0    |       0.000092 |       0.000092 |       0.007326 |       0.000719 |
| stage1    |       0.058593 |       0.059127 |       0.922020 |       0.610441 |
| stage2    |       0.060464 |       0.061094 |       0.926472 |       0.622604 |
| stage3    |       0.060177 |       0.060711 |       0.917949 |       0.621205 |
================================================================================

Results saved to .../postmortem_stage_eval.json
Done.
=== Postmortem Eval Done ===
```

---

## 3. Output File Check

| File | Exists | Size | Modified |
|---|---|---|---|
| `postmortem_stage_eval.json` | Yes | 9,064 bytes | 2026-06-12 21:04:19 |
| `postmortem_eval_115259334.out` | Yes | 4,903 bytes | 2026-06-12 21:04:42 |
| `postmortem_eval_115259334.err` | Yes | 8,590,813 bytes | 2026-06-12 18:34:57 |

All output files are under:
`/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/`

---

## 4. Stage-Wise MRR Table

From `postmortem_stage_eval.json` (full-precision values):

| Stage | row R→E MRR | UniProt-grouped R→E MRR | EC-4-grouped R→E MRR | E→M MRR |
|---|---:|---:|---:|---:|
| stage0 | 0.000092 | 0.000092 | 0.007326 | 0.000719 |
| stage1 | 0.058593 | 0.059127 | 0.922020 | 0.610441 |
| stage2 | 0.060464 | 0.061094 | 0.926472 | 0.622604 |
| stage3 | 0.060177 | 0.060711 | 0.917949 | 0.621205 |

Full-precision values from JSON:

| Stage | row R→E MRR | UniProt-grouped R→E MRR | EC-4-grouped R→E MRR | E→M MRR |
|---|---:|---:|---:|---:|
| stage0 | 9.167e-05 | 9.241e-05 | 7.326e-03 | 7.190e-04 |
| stage1 | 5.859e-02 | 5.913e-02 | 9.220e-01 | 6.104e-01 |
| stage2 | 6.046e-02 | 6.109e-02 | 9.265e-01 | 6.226e-01 |
| stage3 | 6.018e-02 | 6.071e-02 | 9.179e-01 | 6.212e-01 |

Evaluation time per stage: ~2100–2500 seconds (≈35–42 min).

---

## 5. Stage 3 Consistency Check

Stage 3 checkpoint evaluated independently vs. R2 final `metrics_v3.json`:

| Metric | stage3 actual | R2 final expected | Relative Error | PASS/FAIL |
|---|---:|---:|---:|---|
| row R→E MRR | 0.060177 | 0.060208 | 0.05% | **PASS** |
| UniProt-grouped R→E MRR | 0.060711 | 0.060708 | 0.005% | **PASS** |
| EC-4-grouped R→E MRR | 0.917949 | 0.913213 | 0.52% | **PASS** |
| E→M MRR | 0.621205 | 0.619546 | 0.27% | **PASS** |

**All relative errors < 5% tolerance. Consistency check: ALL PASS.**

---

## 6. Interpretation

Based on the Task 1.2 stage-wise MRR evolution table:

1. **row R→E MRR**: Near zero at stage0 (0.000092), jumps to ~0.059 at stage1 (pairwise training), then stabilizes around 0.060 through stages 2–3. The pairwise training stage (stage1) accounts for virtually all of the row-level R→E MRR.

2. **EC-4-grouped R→E MRR**: Also near zero at stage0 (0.007), jumps to 0.922 at stage1, and remains in the 0.918–0.926 range through stages 2–3. EC-4-level retrieval capability is already established after pairwise training.

3. **E→M MRR**: Near zero at stage0 (0.0007), jumps to 0.610 at stage1, then modestly increases to 0.623 at stage2 and 0.621 at stage3. The bulk of E→M retrieval capability is learned during stage1, with marginal refinement in later stages.

4. **Stage stability**: Metrics are remarkably stable from stage1 onward — no substantial movement in any metric after stage1. The self-bootstrap stage (stage3) does not noticeably shift performance from stage2 levels.

---

## 7. Status Declaration

- **Task 1.2 status: COMPLETE**
- No train.py modification.
- No retraining.
- No GPU/DCU used for computation (1 DCU allocated per Slurm QOS requirement, not used by evaluation code).
- No R3 plan.

# Bio Vector R2 Postmortem Task Checklist

Date: 2026-06-12

Purpose: track the teacher-requested R2 postmortem tasks before any R3 planning.

Source instruction:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R2_POSTMORTEM_INSTRUCTIONS_FOR_STUDENT_2026-06-11(1).md
```

Current rule:

- Do not write an R3 plan yet.
- Do not rerun R2.
- Do not modify training code.
- Run only zero-GPU postmortem diagnostics.
- After all tasks finish, compare this checklist against the teacher's checklist before submitting.

## Task Status

| Task | Description | Status | Result doc section |
|---|---|---|---|
| 1.1 | Data-structure baseline and row-level ceiling estimate | complete | `R2_POSTMORTEM_20260614.md` Task 1.1 |
| 1.2 | Stage-wise checkpoint evaluation | complete | `R2_POSTMORTEM_20260614.md` Task 1.2 |
| 1.3 | Indirect attribution of hard-neg vs Stage 1 | complete | `R2_POSTMORTEM_20260614.md` Task 1.3 |
| 1.4 | Visualization error diagnostics | complete | `R2_POSTMORTEM_20260614.md` Task 1.4 |
| 1.5 | S→M MRR historical comparison | complete | `R2_POSTMORTEM_20260614.md` Task 1.5 |
| 1.6.1 | Calibration curve baseline | complete | `R2_POSTMORTEM_20260614.md` Task 1.6 |
| 1.6.2 | OOD-like score distribution baseline | complete | `R2_POSTMORTEM_20260614.md` Task 1.6 |
| 1.6.3 | Latency distribution baseline | complete | `R2_POSTMORTEM_20260614.md` Task 1.6 |
| Task 2 | Revise R2 result summary wording | complete | `BIO_VECTOR_R2_RESULT_SUMMARY_2026-06-11.md` |
| Task 3 | Create postmortem script and decision input docs | complete | `R2_POSTMORTEM_20260614.md`; `postmortem_eval_stage_checkpoints.py`; `R3_DECISION_INPUT.md` |
| Final | Full checklist comparison before teacher submission | complete | this checklist |

## Current Next Step

Current completed-task notes:

```text
Task 1.1 is complete after correction:
- EC valid rows are separated for EC-2/EC-3 vs EC-4.
- mean rows uses sizes.mean() over valid group sizes.
- top-K ceiling uses corrected mean rows.
- EC-4 max/min imbalance is reported.
- Interpretation no longer implies UniProt exact retrieval is the project goal.

Task 1.2 is complete:
- stage0/1/2/3 checkpoints were evaluated.
- row R→E / UniProt-grouped R→E / EC-4-grouped R→E / E→M MRR table is recorded.
- stage3 consistency check vs metrics_v3.json passed.
- Note: Slurm allocated gres/dcu=1 on kshdnormal04 due partition/QOS, but evaluation code did not use GPU/DCU computation.

Task 1.3 is complete:
- R1 stage1 checkpoint is missing, so direct hard_neg vs Stage 1 attribution is not identifiable.
- R2 stage-wise evidence shows most R2 alignment is already present by stage1.
- Future attribution-capable runs must save all stage-end checkpoints.

Task 1.4 is complete:
- complete visualization traceback, embedding health, NN index health, and 100-row EC-4 sanity check are recorded.

Task 1.5 is complete:
- R1 vs R2 S→M MRR comparison is recorded.
- S→M remains a monitoring metric, not the primary project direction.

Task 1.6 is complete:
- calibration curve baseline is recorded, including ECE.
- OOD-like score distribution baseline is recorded.
- PNG histogram was skipped due matplotlib/numpy rendering issue, but inline fallback histogram has exact count checks.
- latency baseline is recorded.
- no thresholds were set.

Task 2 is complete:
- `BIO_VECTOR_R2_RESULT_SUMMARY_2026-06-11.md` §4 now frames R2 as a threshold-design correction with ceiling analysis.
- §5 now states the ceiling-aware interpretation, includes Task 1.2 stage-wise evidence, and avoids direct hard-neg vs stage1 attribution because R1 stage checkpoints are absent.
- §6 now points to a one-page `R3_DECISION_INPUT.md` as the next deliverable, not an R3 training plan.
- the summary no longer uses teacher-disallowed scientific framing terms.

Task 3 is complete:
- `R2_POSTMORTEM_20260614.md` was created from the final total postmortem report.
- `postmortem_eval_stage_checkpoints.py` was restored from the Task 1.2 script audit, then given delivery-safe defaults/status wording, and passes `python3 -m py_compile`.
- `R3_DECISION_INPUT.md` was created as a 56-line decision input, not an experiment protocol.
- no training code was modified.
```

Current next step:

```text
Ready for teacher submission after sending the four required deliverables.
```

```text
Final deliverables:
- R2_POSTMORTEM_20260614.md
- BIO_VECTOR_R2_RESULT_SUMMARY_2026-06-11.md
- postmortem_eval_stage_checkpoints.py
- R3_DECISION_INPUT.md
```

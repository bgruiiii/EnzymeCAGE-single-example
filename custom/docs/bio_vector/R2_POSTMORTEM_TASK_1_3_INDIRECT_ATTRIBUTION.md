# R2 Postmortem Task 1.3 — Indirect Attribution of Hard-Negative vs Stage 1

**Date**: 2026-06-14  
**Scope**: Assess whether R2's two simultaneous changes can be attributed separately.  
**Status**: **Task 1.3 COMPLETE**

---

## 1. Task Definition

Teacher instruction:

```text
R2 同时改了 hard_neg=2→1 和 stage1=12→25，无法直接归因。
但可借助 stage1 终点 checkpoint 间接推断。

如 R1 没保存 stage1 checkpoint，则跳过此项，并在 R3 plan 里强制要求
任何新一轮训练必须保存所有 stage checkpoint。
```

This task therefore asks two separate questions:

1. **Direct attribution**: Can we compare R1 stage1 vs R2 stage1 to isolate the effect of `hard_neg_weight` and Stage 1 length?
2. **Indirect evidence**: If direct attribution is not possible, what can the R2 stage-wise curve still tell us?

---

## 2. Source Files

| Evidence | Path / Source | Used For |
|---|---|---|
| R1 diagnosis summary | `BIO_VECTOR_R1_DIAGNOSIS_RESULT_2026-06-09.md` | R1 final metrics and R1 checkpoint availability |
| R2 stage-wise run result | `R2_POSTMORTEM_TASK_1_2_RUN_RESULT.md` | R2 stage0/1/2/3 MRR table |
| R2 S→M history | `R2_POSTMORTEM_TASK_1_5_SM_HISTORY.md` | R1/R2 E→M and S→M context |
| Teacher postmortem instruction | `BIO_VECTOR_R2_POSTMORTEM_INSTRUCTIONS_FOR_STUDENT_2026-06-11(1).md` | Task 1.3 criteria and wording constraints |

No new computation was run for Task 1.3. This section is an attribution analysis based on already completed postmortem outputs.

---

## 3. R1 Stage Checkpoint Availability

R1 ESM-C stage-end checkpoints are not available:

| R1 Artifact | Status |
|---|---|
| `model_v3_stage0.pt` | missing |
| `model_v3_stage1.pt` | missing |
| `model_v3_stage2.pt` | missing |
| `model_v3_stage3.pt` | missing |
| final `model_v3.pt` | available |

Therefore:

```text
R1 stage1 checkpoint not available; direct hard_neg vs stage1 attribution is not
identifiable from saved checkpoints. Any future training run intended for
attribution must save all stage-end checkpoints.
```

This is the central limitation of Task 1.3.

---

## 4. Direct R1-vs-R2 Stage1 Attribution Table

The teacher-requested direct comparison cannot be completed because the R1 stage1 endpoint was not saved.

| Comparison Metric | R1 stage1 endpoint | R2 stage1 endpoint | Direct attribution possible? |
|---|---:|---:|---|
| row-level R→E MRR | missing | 0.058593 | No |
| UniProt-grouped R→E MRR | missing | 0.059127 | No |
| EC-4-grouped R→E MRR | missing | 0.922020 | No |
| E→M MRR | missing | 0.610441 | No |

Because R1 stage1 is missing, we cannot separate:

- the effect of changing `hard_neg_weight` from 2.0 to 1.0
- the effect of extending Stage 1 from 12 epochs to 25 epochs
- later-stage effects from Stage 2 / Stage 3

Any statement assigning a numeric fraction of the R2 change to either variable would be over-interpreting the saved artifacts.

---

## 5. R2 Stage-Wise Evidence

R2 does have all four stage-end checkpoints, and Task 1.2 evaluated them consistently:

| Stage | row R→E MRR | UniProt-grouped R→E MRR | EC-4-grouped R→E MRR | E→M MRR |
|---|---:|---:|---:|---:|
| stage0 | 0.000092 | 0.000092 | 0.007326 | 0.000719 |
| stage1 | 0.058593 | 0.059127 | 0.922020 | 0.610441 |
| stage2 | 0.060464 | 0.061094 | 0.926472 | 0.622604 |
| stage3 | 0.060177 | 0.060711 | 0.917949 | 0.621205 |

Stage3 consistency against final `metrics_v3.json` passed for all four monitored metrics:

| Metric | stage3 actual | R2 final expected | Relative Error | Result |
|---|---:|---:|---:|---|
| row R→E MRR | 0.060177 | 0.060208 | 0.05% | PASS |
| UniProt-grouped R→E MRR | 0.060711 | 0.060708 | 0.005% | PASS |
| EC-4-grouped R→E MRR | 0.917949 | 0.913213 | 0.52% | PASS |
| E→M MRR | 0.621205 | 0.619546 | 0.27% | PASS |

---

## 6. R2 Stage1-to-Stage3 Marginal Change

Within R2, Stage 1 already accounts for almost all observed retrieval capability.

| Metric | R2 stage1 | R2 stage3 | Absolute change | Relative change from stage1 |
|---|---:|---:|---:|---:|
| row R→E MRR | 0.058593 | 0.060177 | +0.001584 | +2.70% |
| UniProt-grouped R→E MRR | 0.059127 | 0.060711 | +0.001584 | +2.68% |
| EC-4-grouped R→E MRR | 0.922020 | 0.917949 | -0.004071 | -0.44% |
| E→M MRR | 0.610441 | 0.621205 | +0.010764 | +1.76% |

Interpretation:

- R2 stage1 produces the large jump from near-zero stage0 to usable alignment.
- Later stages add only small changes to row-level / UniProt-level R→E.
- EC-4 grouped R→E is already high at stage1 and remains high afterward.
- E→M sees a modest later-stage increase, but the bulk of E→M capability is also present by stage1.

This supports the Task 1.2 observation that Stage 1 is where the main alignment is established. It does **not** identify whether this is due to longer Stage 1, hard-negative removal, or their combination.

---

## 7. Final-to-Final Context: R1 vs R2

Although direct stage1 attribution is impossible, final-to-final context is still useful as a safety check.

| Metric | R1 final | R2 final / expected | Absolute change | Relative change | Interpretation |
|---|---:|---:|---:|---:|---|
| row R→E MRR | 0.0575 | 0.060208 | +0.002708 | +4.71% | combined R2 changes did not reduce row-level R→E |
| UniProt-grouped R→E MRR | 0.0581 | 0.060708 | +0.002608 | +4.49% | combined R2 changes did not reduce UniProt-grouped R→E |
| EC-4-grouped R→E MRR | 0.918680 | 0.913213 | -0.005467 | -0.60% | EC-4 family-level structure remained essentially stable |
| E→M MRR | 0.6094 | 0.619546 | +0.010146 | +1.67% | E→M improved slightly |

This table should be read carefully:

- It is **not** a controlled ablation.
- It only shows that the combined R2 intervention (`hard_neg_weight=1.0`, `epochs_stage1=25`) preserved EC-family retrieval and slightly improved row-level / UniProt-level R→E and E→M.
- It is consistent with the teacher's interpretation that disabling EC-1 hard-negative upweighting is not harmful under R2, but it does not quantify the independent contribution of that change.

---

## 8. Attribution Assessment

### 8.1 What can be concluded

1. **Direct hard_neg vs Stage 1 attribution is not identifiable from saved artifacts.**  
   R1 has no stage1 checkpoint, and R2 changed both `hard_neg_weight` and Stage 1 epochs.

2. **The combined R2 setting is compatible with stable EC-family retrieval.**  
   R1 EC-4 MRR was 0.918680; R2 final EC-4 MRR was 0.913213; R2 stage-wise EC-4 MRR stayed in the 0.918-0.926 range after stage1. This indicates robust EC-family alignment, not a collapse of EC-level structure.

3. **Stage 1 establishes most R2 alignment.**  
   R2 stage1 already reaches row R→E MRR 0.058593, EC-4 grouped MRR 0.922020, and E→M MRR 0.610441. Later stages make small refinements.

4. **R2 final-to-final changes are modest.**  
   Row-level R→E improves from 0.0575 to 0.060208; E→M improves from 0.6094 to 0.619546; EC-4 grouped MRR is essentially stable.

### 8.2 What cannot be concluded

1. We cannot state that `hard_neg_weight=1.0` alone caused the row R→E gain.
2. We cannot state that extending Stage 1 alone caused the row R→E gain.
3. We cannot estimate a numeric contribution split between hard-negative removal and longer Stage 1.
4. We cannot use R1 final vs R2 stage1 as a controlled comparison, because those checkpoints correspond to different training histories.

---

## 9. Required Future Artifact Rule

For any future training run intended to support attribution, the following artifacts must be mandatory:

| Artifact | Requirement |
|---|---|
| `model_v3_stage0.pt` | save at Stage 0 end |
| `model_v3_stage1.pt` | save at Stage 1 end |
| `model_v3_stage2.pt` | save at Stage 2 end |
| `model_v3_stage3.pt` | save at Stage 3 end |
| `metrics_v3.json` | include row-level and grouped R→E metrics |
| postmortem stage eval JSON | include stage-wise row R→E, UniProt-grouped, EC-4-grouped, and E→M |

This is an artifact requirement, not an R3 training plan.

---

## 10. Status Declaration

| Item | Status |
|---|---|
| R1 stage1 checkpoint found | No |
| Direct R1-vs-R2 stage1 comparison | Not identifiable |
| R2 stage-wise evidence reviewed | Yes |
| Final-to-final context reviewed | Yes |
| Task 1.3 | COMPLETE |
| train.py modification | None |
| New computation | None |
| Retraining | None |
| GPU/DCU usage | None |
| R3 plan | None |

**Task 1.3 status: COMPLETE**

Direct hard-negative vs Stage 1 contribution is not identifiable from the saved R1 artifacts. The R2 stage-wise curve nevertheless shows that most R2 alignment is already present by stage1, and that later stages only modestly change the monitored metrics. Future attribution-capable runs must save all stage-end checkpoints.

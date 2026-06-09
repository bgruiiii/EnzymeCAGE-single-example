# Bio Vector R2 Task 0 EC-3/EC-4 Supplement Feedback

Date: 2026-06-09

Status: final teacher-facing feedback after HPC audit.

## 1. Purpose

This note reports the zero-cost supplemental diagnosis requested before R2
training submission.

Teacher request:

- add EC-4-grouped `R→E` retrieval evaluation using existing R1 ESM-C
  embeddings
- optionally add EC-3-grouped `R→E`
- do not retrain
- do not modify `train.py`
- use the results to revise `R2_PLAN_v2`

HPC report path:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07/R1_DIAGNOSIS_20260608_172625.md
```

## 2. Execution Audit

Run mode:

- direct run on login node `login09`
- no Slurm job was submitted for this supplement
- `sacct` for 2026-06-09 showed no job records
- `squeue` showed no running jobs

Validation:

- `python -m py_compile diagnose_round1_postmortem.py` passed
- return marker: `PYCOMPILE_PASS`

Runtime:

```text
748.4 seconds
```

Backups:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/diagnose_round1_postmortem.py.before_ec4_20260609_153054.bak
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07/R1_DIAGNOSIS_20260608_172625.md.before_ec4_20260609_153054.bak
```

Confirmed boundaries:

- `train.py` was not modified
- `train.py` mtime: `2026-06-07 11:03:01`
- `train.py` MD5: `e3605adaec151574cfa224a772d3c5a5`
- no R2 training job was submitted
- the appended report tail contains H2.4/H2.5, H3.4/H3.5, and H7, ending
  with `--- Diagnosis complete in 748.4s ---`

## 3. Supplement Result

R→E retrieval:

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
| row-level random | 0.00000687 | 0.0000343 | 0.0000687 | 0.00000687 |
| UniProt-grouped random | 0.00000928 | 0.0000464 | 0.0000928 | 0.0001099 |
| EC-2-grouped random | 0.0136 | 0.0617 | 0.1107 | 0.0474 |
| EC-3-grouped random | 0.00479773 | 0.02282687 | 0.04306375 | 0.02007514 |
| EC-4-grouped random | 0.00034787 | 0.00172581 | 0.00341852 | 0.00215903 |

## 4. Random-Baseline Optimization

The first EC-3/EC-4 supplement attempt stalled in `analytical_random()` during
the UniProt-grouped random-baseline calculation.

Root cause:

```text
107731 UniProt groups * about 145607 rank positions
```

This produced an avoidable very large Python loop.

Fix used by HPC:

- group sizes were counted with `Counter(group_sizes)`
- the random-baseline expression was computed once per unique group size
- the result was weighted by the number of groups with that size
- the formula itself was not changed

Unique group-size audit:

| Grouping | total groups | unique sizes | singleton groups |
|---|---:|---:|---:|
| UniProt | 107731 | 28 | 86656 |
| EC-3 | 187 | 129 | 7 |
| EC-4 | 2524 | 306 | 579 |

Important numerical note:

```text
Singleton MRR was not simplified to 1/N.
```

For singleton groups, top-1 random probability is `1/N`, but expected MRR is
not `1/N`. The rerun preserved the original loop formula and only deduplicated
identical group sizes.

Observed effect:

- after H2 completed, H3-H7 took about `49.8` seconds
- compared with the stalled naive run, the estimated speedup was about
  `200-500x`

## 5. Interpretation

The EC-3/EC-4 supplement strengthens the revised R1 interpretation.

R1 learned strong enzyme-function structure:

- EC-2-grouped `R→E MRR = 0.9340`
- EC-3-grouped `R→E MRR = 0.933629`
- EC-4-grouped `R→E MRR = 0.918680`

The main weakness is fine-grained row-level / UniProt-level discrimination:

- row-level `R→E MRR = 0.0580`
- UniProt-grouped `R→E MRR = 0.0581`

Recommended narrative:

```text
R1 aligns reaction and enzyme well at EC family and EC functional-identity
levels, including EC-4. The remaining problem is fine-grained discrimination
within those functional groups, where row-level and UniProt-level retrieval are
still low.
```

This is consistent with the previous hard-negative diagnosis:

```text
same-EC-1-digit per-anchor mean = 643.96
```

The EC-1 hard-negative signal is too coarse for the desired fine-grained
retrieval target.

## 6. R2 Plan Implication

The teacher-approved R2 direction remains appropriate:

```python
hard_neg_weight = 1.0
epochs_stage1 = 25
```

The EC encoding change should be removed from R2 training variables.

Reason:

```text
When hard_neg_weight = 1.0, the same-EC weighting term degenerates to 1.0, so
the same_ec matrix no longer affects the InfoNCE logits. Changing EC encoding
at the same time would be dead code for this R2 training run and would confuse
attribution.
```

EC-2/EC-3/EC-4 grouped metrics should remain evaluation/reporting metrics for
R2.

## 7. Quantitative Thresholds For Revised R2 Plan

The revised `R2_PLAN_v2` should include explicit pass/fail thresholds:

| Metric | R1 measured | R2 pass threshold | Warning / failure threshold |
|---|---:|---:|---:|
| row-level `R→E MRR` | 0.0580 | > 0.12 | < 0.06 means failure |
| UniProt-grouped `R→E MRR` | 0.0581 | > 0.15 | < 0.08 means failure |
| EC-4-grouped `R→E MRR` | 0.918680 | >= 0.868680 | drop > 0.05 means warning |
| EC-2-grouped `R→E MRR` | 0.9340 | >= 0.85 | < 0.70 means severe degradation |
| E→M MRR | 0.609 | >= 0.55 | < 0.40 means failure |

Suggested success rule:

```text
R2 is successful if row-level or UniProt-grouped R→E reaches its pass threshold
and EC-2/EC-4 grouped metrics do not cross warning thresholds.
```

## 8. Next Step

Revise `R2_PLAN_v2` with the teacher's requested changes:

1. Remove EC encoding as a training change.
2. Keep only `hard_neg_weight = 1.0` and `epochs_stage1 = 25` as R2 training
   changes.
3. Rewrite the R1 narrative using the EC-3/EC-4 supplement result.
4. Add the quantitative threshold table.
5. Submit the revised plan for teacher approval before any HPC training job.

# Bio Vector R2 Result Summary

Date: 2026-06-11

Status: R2 core training completed. Slurm job returned non-zero exit only
because the optional visualization PNG did not render after all core artifacts
were saved.

## 1. HPC Artifacts

Output directory:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11
```

Job:

```text
Job ID: 115204112
Elapsed: 03:24:55
ExitCode: 1:0
```

Visualization issue:

```text
visualize_four_modal() -> fig.savefig("unified_space_v3_results.png")
ValueError: object __array__ method not producing an array
```

Interpretation:

- R2 training completed all 4 stages.
- Evaluation completed.
- Metrics, embeddings, checkpoints, metadata, and NN index files were saved.
- Only the optional visualization PNG was missing.
- Do not rerun training just to regenerate the PNG.

## 2. Core Artifacts

All core files were reported present:

```text
model_v3.pt
model_v3_stage0.pt
model_v3_stage1.pt
model_v3_stage2.pt
model_v3_stage3.pt
embeddings_v3.npz
metrics_v3.json
metadata_v3.json
training_history.json
reaction_nn_index.npz
enzyme_nn_index.npz
substrate_nn_index.npz
microbe_nn_index.npz
enzyme2microbe_index.json
r2_config.txt
```

Missing:

```text
unified_space_v3_results.png
```

## 3. R2 Metrics

Row-level retrieval:

| Metric | R2 value |
|---|---:|
| `R→E_top-1` | 0.0176227791 |
| `R→E_top-5` | 0.0604297870 |
| `R→E_top-10` | 0.0935669302 |
| `R→E_top-20` | 0.1406251073 |
| `R→E_MRR` | 0.0602077827 |
| `E→M_MRR` | 0.6195463344 |
| `S→M_MRR` | 0.5870633084 |

Grouped R→E retrieval:

| Positive definition | evaluated | excluded unknown | top-1 | top-5 | top-10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| UniProt-grouped | 145607 | 0 | 0.035307 | 0.069186 | 0.098546 | 0.060708 |
| EC-2-grouped | 130635 | 14972 | 0.890986 | 0.937712 | 0.955349 | 0.930647 |
| EC-3-grouped | 130635 | 14972 | 0.889050 | 0.937612 | 0.955257 | 0.930354 |
| EC-4-grouped | 127847 | 17760 | 0.838893 | 0.938872 | 0.954539 | 0.913213 |

## 4. Threshold Judgment

R2 triggered a correction to the original threshold design. The earlier
row-level `R→E MRR > 0.12` and UniProt-grouped `R→E MRR > 0.15` criteria should
not be used as primary project-quality gates because they do not account for
same-function row multiplicity.

Ceiling evidence from Task 1.1:

| Grouping | valid rows | mean rows per group | top-1 row ceiling | Interpretation |
|---|---:|---:|---:|---|
| UniProt | 145607 | 1.35 | 0.739875 | Reference only; not the agent-facing output |
| EC-2 | 130635 | 1979.32 | 0.000505 | Coarse EC-family grouping |
| EC-3 | 130635 | 698.58 | 0.001431 | Intermediate EC-family grouping |
| EC-4 | 127847 | 50.65 | 0.019742 | Primary functional grouping for R→EC-4 retrieval |

Observed R2 row-level top-1 is close to the EC-4 row ceiling:

```text
R2 row-level R→E top-1 = 0.0176227791
Task 1.1 EC-4 row-level top-1 ceiling ≈ 0.019742
Observed / ceiling ≈ 89.3%
```

The independent ratio using R2 grouped top-1 gives the same interpretation:

```text
R2 row-level R→E top-1 / EC-4-grouped R→E top-1
= 0.0176227791 / 0.8388933647
≈ 1 / 47.6
```

For a reaction-side query with nearly identical DRFP input across same-EC-4
rows, exact-row selection is information-limited. Under this corrected view,
EC-4-grouped retrieval is the primary project-relevant metric, and row-level /
UniProt-level metrics are reference diagnostics.

Revised judgment table:

| Metric | R1 measured | R2 measured | Revised role | Judgment |
|---|---:|---:|---|---|
| row-level `R→E MRR` | 0.0575-0.0580 | 0.060208 | Reference diagnostic | Modest increase |
| UniProt-grouped `R→E MRR` | 0.0581 | 0.060708 | Reference diagnostic | Modest increase |
| EC-4-grouped `R→E MRR` | 0.918680 | 0.913213 | Primary R→EC-4 functional metric | Desired structure preserved |
| EC-2-grouped `R→E MRR` | 0.9340 | 0.930647 | Coarse EC-family monitor | Desired structure preserved |
| `E→M MRR` | 0.609 | 0.619546 | Downstream evidence monitor | Improved |
| `S→M MRR` | 0.610836 | 0.587063 | Downstream evidence monitor | Modestly lower; continue tracking |

Task 1.6 remains baseline-only; no tool-oriented acceptance criteria are set at
this stage.
Calibration, OOD-like score distribution, and latency are baseline records for
future agent integration.

## 5. Interpretation

R2 tested:

```python
hard_neg_weight = 1.0
epochs_stage1 = 25
```

Observed effect:

- Removing EC-1 hard-negative upweighting is non-harmful and slightly positive
  in the combined R2 setting:
  - row-level `R→E MRR`: `0.0575 -> 0.0602` (`+4.7%`)
  - UniProt-grouped `R→E MRR`: `0.0581 -> 0.0607` (`+4.5%`)
  - `E→M MRR`: `0.609 -> 0.6195` (`+1.7%`)
  - EC-2/EC-4 grouped metrics remain nearly unchanged.
- `hard_neg_weight = 2.0` does not need to remain a preferred setting.
- EC-family / EC-functional alignment is robust without explicit same-EC
  hard-negative upweighting.
- Stage-wise checkpoint evaluation from Task 1.2 shows most R2 alignment is
  already present by the end of stage1:
  - stage1 row-level `R→E MRR = 0.058593`
  - stage1 EC-4-grouped `R→E MRR = 0.922020`
  - stage1 `E→M MRR = 0.610441`
  - stage2/stage3 produce only small changes around this plateau.

Direct attribution between `hard_neg_weight=1.0` and `epochs_stage1=25` is not
identifiable from the available R1 artifacts, because R1 did not save stage-end
checkpoints. Therefore, R2 should not be used to assign separate numeric
contributions to hard-negative removal versus stage1 extension.

The main interpretation is ceiling-aware: row-level / UniProt-level R→E is
constrained by reaction-side information. Same-EC-4 rows can have nearly
identical reaction fingerprints, so exact-row selection is not the right primary
target for the intended use case.

Project positioning:

```text
The model is a tool/verifier for an upper-level pathway-design or degradation
inference agent. The relevant output is R→EC-4 retrieval plus downstream
candidate enzyme/microbe evidence, not exact row-level enzyme retrieval.
```

## 6. Recommended Next Step

Do not rerun the same R2 training.

The teacher-requested zero-GPU R2 postmortem diagnostics are now complete:

| Task | Diagnostic | Status |
|---|---|---|
| 1.1 | Data-structure baseline and row-level ceiling estimate | Complete |
| 1.2 | Stage-wise checkpoint evaluation | Complete |
| 1.3 | Indirect attribution of hard-neg vs stage1 | Complete; direct attribution not identifiable because R1 stage checkpoints are absent |
| 1.4 | Visualization traceback, embedding checks, NN index health check | Complete |
| 1.5 | S→M historical comparison | Complete |
| 1.6 | Calibration, OOD-like score distribution, latency baselines | Complete; baseline-only, no acceptance criteria set |

Next deliverable:

```text
R3_DECISION_INPUT.md
```

This should be a one-page decision input for the teacher, not an R3 training
plan. It should summarize the completed diagnostics and present candidate
directions without specifying hyperparameters, epoch schedules, or dataset
changes.

Project positioning for the decision input:

```text
Path B: tool/verifier mode.
Primary output: R→EC-4 retrieval plus candidate enzyme/microbe evidence.
Primary metric: EC-4-grouped R→E, with row-level R→E as a reference diagnostic.
```

Candidate directions to present as decision inputs:

| Diagnostic signal | Candidate direction |
|---|---|
| stage1 already establishes most EC-4 alignment | Consider simplifying later stages if teacher approves |
| exact row retrieval is ceiling-limited by same-EC-4 multiplicity | Keep EC-4-grouped retrieval as the primary objective |
| EC-4 labels are highly imbalanced | Consider EC-4 class-balanced sampling or loss design |
| calibration / OOD-like / latency baselines are recorded | Reserve confidence, abstention, and latency thresholds for agent integration |

Do not write an R3 plan until the teacher selects a direction from the decision
input.

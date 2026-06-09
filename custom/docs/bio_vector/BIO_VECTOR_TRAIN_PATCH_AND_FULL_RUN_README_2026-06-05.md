# Bio Vector Full-Data `train.py` Patch And Run README

Date: 2026-06-05

## Purpose

This document records the changes made to the teacher-provided `bio_vector`
demo before running the full EnzymeCAGE dataset on HPC.

The goal is to keep the teacher's training strategy unchanged while adapting
the demo loader to the actual full-data package format.

## What Stayed Unchanged

The intended full run still follows the teacher README:

- same `train.py` entry point
- same `--mode enzyme_cage_300` loader name
- same four-stage training pipeline
- same model/loss/evaluation outputs
- same GVP and ESM-C enzyme feature comparison plan
- same post-training effective-rank diagnosis

The `--mode enzyme_cage_300` name is kept because it is the demo's existing
loader option. For the current task, `--data_dir` points to the full extracted
dataset, so the loader reads all full-data examples.

## Full Data Used On HPC

HPC work root:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04
```

Full data directory:

```text
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/data/reaction_enzyme_microbe_training_clean_2026-06-01_LOCAL
```

Data sanity check passed:

- `reaction_features.npz`: 145,607 examples
- reaction DRFP shape: `(145607, 2048)`
- microbe JSONL: 145,607 lines, 0 missing examples
- enzyme metadata: 145,607 rows, 107,731 unique UniProt IDs
- ESM-C `*_esm_c_sequence_node.npz`: 107,731 files
- ESM-C UID missing count: 0

## Why `train.py` Needed Patching

The teacher demo was originally written for the small demo data layout. The
full training package has the same conceptual features but a different GVP file
layout.

The original demo expected GVP features as individual files referenced by:

```text
gvp_feature_file
```

The full package stores GVP features as sharded PyTorch files referenced by:

```text
gvp_shard_file
```

Without adapting the loader, the full GVP run silently fell back to AAC enzyme
features and reported:

```text
GVP attention-pooled: 0/145607
```

That would not be a valid GVP baseline.

## Loader/Compatibility Fixes

The following fixes are required or already observed from the HPC loader probe.
The exact final diff still needs to be audited against the uploaded original
`train.py`.

### 1. GVP Metadata Column

Problem:

- Original loader read `gvp_feature_file`.
- Full metadata uses `gvp_shard_file`.

Fix:

- Read the sharded GVP path from `gvp_shard_file`.

Expected result:

- Loader can locate the full GVP shard files.

### 2. Sharded `.pt` GVP Loader

Problem:

- Original `gvp_attention_pool()` expected individual `.npz` files.
- Full GVP data is stored in sharded `.pt` files keyed by UniProt ID.
- The actual shard layout is:

```text
gvp_part_0000.pt ... gvp_part_0191.pt
```

Each shard is a dictionary:

```text
key   = UniProtID
value = (
  node_xyz[N, 3],
  residue_id[N],
  node_s[N, 6],
  node_v[N, 3, 3],
  edge_idx[2, E],
  edge_s[E, 32],
  edge_v[E, 1, 3],
)
```

Fix:

- Add a `gvp_shard_pool()` path that reads the actual shard tuple format and
  pools each protein to the same 50-dimensional GVP feature expected by the
  original model.
- Use the correct tensor indices:
  - `entry[2]` -> `node_s`
  - `entry[3]` -> `node_v`
  - `entry[5]` -> `edge_s`
  - `entry[6]` -> `edge_v`
- Pool to 50 dimensions:

```text
node_s pooled: 6
node_v pooled/flattened: 9
edge_s pooled: 32
edge_v pooled/flattened: 3
total: 6 + 9 + 32 + 3 = 50
```

Expected result:

- Enzyme feature shape remains `(N, 50)`.
- Model architecture remains unchanged.

### 3. Shard-Batch Loading

Problem:

- Loading a shard per example would repeatedly reload the same 192 shard files.
- For 145,607 examples this would be extremely slow.
- With a small LRU cache, repeated shard switching could cause about 145K shard
  loads and tens of TB of I/O.

Fix:

- Group examples by `gvp_shard_file`.
- Load each shard once and fill the corresponding rows.

Expected result:

- Same output features, much faster data loading.

### 4. AAC Fallback Padding

Problem:

- In GVP mode, enzyme feature rows are 50-dimensional.
- AAC fallback returns 20 dimensions.
- Direct assignment causes a shape mismatch.

Fix:

- Put AAC values in the first 20 dimensions and leave the remaining dimensions
  as zeros.

Expected result:

- Fallback remains possible without changing the GVP feature dimension.

### 5. RDKit Morgan Fingerprint Compatibility

Problem:

- HPC RDKit version had compatibility issues with the original
  `ConvertToNumpyArray` usage.

Fix:

- Use a compatible Morgan fingerprint conversion path.

Expected result:

- Substrate Morgan fingerprints load as `(N, 2048)`.

## HPC Pitfalls Encountered

These issues were encountered during the loader probe and should be avoided in
future runs.

| Issue | Cause | Resolution |
|------|-------|------------|
| GVP map all empty | Full metadata column is `gvp_shard_file`, not `gvp_feature_file` | Read `gvp_shard_file` |
| GVP `.npz` assumption failed | Full data uses sharded `.pt` tuple tensors | Add sharded GVP loader |
| Excessive shard I/O | Per-example `torch.load()` would reload shards repeatedly | Batch by shard and load each shard once |
| GVP output was 589-dim | Wrong tuple indices and wrong feature assumptions | Use `entry[2]`/`entry[3]` and pool to 50 dims |
| Duplicate `gvp_shard_pool` definitions | Partial patch attempt left duplicate functions | Deduplicate and keep the corrected version |
| AAC fallback shape mismatch | GVP row is 50-dim while AAC is 20-dim | Write AAC into first 20 dims and zero-pad |
| RDKit `ConvertToNumpyArray` type issue | HPC RDKit expected a compatible integer array path | Use a bit-string conversion path |
| Stale `__pycache__` | Compute node used old compiled code after edits | Remove `__pycache__` before resubmitting |
| Slurm heredoc failure | Nested heredoc parsing failed in submitted script | Prefer `python -c` or standalone script files |
| `~` in Slurm paths | Some Slurm settings did not expand `~` in directives | Use absolute `/public/home/acfbwjsi7s/...` paths |
| Memory request rejected | `--mem=64G` exceeded partition/account policy | Use `--mem-per-cpu=3500M` or valid scheduler limits |

Main lesson:

The actual full-data GVP storage format must be inspected before writing or
changing loaders. The full package uses sharded `.pt` tensors, while the demo
code assumed individual `.npz` files.

## Full GVP Loader Probe Result

Final loader probe reported:

- GVP coverage: `145607/145607`
- reaction shape: `(145607, 2048)`
- enzyme shape: `(145607, 50)`
- substrate shape: `(145607, 2048)`
- microbe shape: `(145607, 28)`
- concept targets shape: `(145607, 8)`
- train/test split: `123765 / 21842`

Interpretation:

- The full GVP baseline will use real GVP features, not AAC fallback.

## Pending Audit Before Full Training

Audit against the uploaded original `train.py` has been completed.

Basic audit result:

- original SHA256: `b269308663...`
- patched SHA256: `9ea80f1777...`
- diff size: 163 lines
- `python -m py_compile train.py`: OK

Confirmed unchanged:

- `Config` hyperparameters
- model architecture
- `train_four_stages`
- loss functions
- optimizer and scheduler logic
- `evaluate_multimodal`
- original `esmc_pocket_pool`
- original `aac_features`
- original `.npz`-based `gvp_attention_pool`

Confirmed changes are in the data-loading layer:

1. Added `gvp_shard_pool()` for the full package's sharded `.pt` GVP format.
2. Updated Morgan fingerprint conversion for HPC RDKit compatibility.
3. Updated GVP metadata mapping from `gvp_feature_file` to `gvp_shard_file`
   and UniProt ID.
4. Replaced per-row GVP loading with shard-batch loading.
5. Simplified ESM-C missing-file fallback to AAC only.

ESM-C fallback note:

- The temporary AAC-only simplification was reverted before training.
- Current fallback semantics are restored to:

```text
ESM-C present -> ESM-C
ESM-C missing -> GVP shard fallback
GVP unavailable -> AAC fallback
```

- The restored GVP fallback uses the new sharded `.pt` adapter:
  `gvp_shard_pool(entry)`.
- Current full-data sanity check showed ESM-C missing rows = 0, so fallback
  should not be triggered in the full ESM-C baseline if files remain in the
  expected flat directory.

Full GVP loader probe after patch:

```text
GVP shard-batch: 192 shards, 145607 entries pending
GVP shard-batch done: 145607/145607 loaded
Substrate Morgan FP: 145607/145607 parsed
Microbe metabolic features: 145607/145607 loaded
FULL_GVP_LOADER: PASS
```

Conclusion:

- The patch is acceptable for full GVP baseline training.
- The patch should be described as a full-data loader compatibility patch, not
  a change to the teacher's training method.

Updated fallback audit:

- Diff size after restoring ESM-C fallback: 179 lines.
- `python -m py_compile train.py`: OK.
- No full baseline job was started during the fallback restoration.
- `Config`, model, training, loss, optimizer/scheduler, and evaluation remain
  untouched.

## Full Run Results

### Full GVP Baseline

Job:

- Slurm job ID: `114938073`
- enzyme feature: GVP
- data: full `145,607` examples
- output directory:
  `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_gvp_baseline_microbe_fixed_2026-06-06`

Config:

- `batch_size = 4096`
- `lr = 3e-4`
- epochs: `8/12/8/10`, total `38`
- `temp_end = 0.05`
- `w_sm = 0.5`
- `vicreg_var_weight = 10.0`

Loss:

```text
Stage 0 epoch  1: loss=2.8163  tau=0.5000
Stage 1 epoch 10: loss=40.401  tau=0.4374
Stage 1 epoch 20: loss=38.392  tau=0.2655
Stage 3 epoch 30: loss=47.373  tau=0.0999
Stage 3 epoch 38: loss~46.98 tau=0.05
```

Metrics:

| Metric | Full GVP baseline |
|--------|------------------:|
| `R→E top-1` | 0.0118 |
| `R→E top-10` | 0.0671 |
| `R→E MRR` | 0.044 |
| `E→M top-1` | 0.0017 |
| `E→M MRR` | 0.395 |
| `S→M top-1` | 0.0054 |
| `S→M MRR` | 0.642 |

Artifacts:

- `model_v3.pt`: 22M
- `embeddings_v3.npz`: 569M
- `metrics_v3.json`: 622B
- `training_history.json`: 2.1K
- `metadata_v3.json`: 22M
- `enzyme2microbe_index.json`: 4.3M
- fallback normalized retrieval files:
  - `reaction_nn_index.npz`
  - `enzyme_nn_index.npz`
  - `substrate_nn_index.npz`
  - `microbe_nn_index.npz`

Non-blocking issues:

- Native FAISS index was replaced by `*_nn_index.npz` fallback due FAISS
  SWIG/NumPy incompatibility on HPC.
- `diagnose_effective_rank.py` OOM-killed, so a lightweight NumPy-only
  effective-rank summary was generated instead.
- Visualization plots were not generated.

Lightweight effective rank:

| Modality | Effective rank | Participation ratio | dim@90 | dim@95 | dim@99 |
|----------|---------------:|--------------------:|-------:|-------:|-------:|
| reaction | 28.31 | 20.58 | 21 | 29 | 92 |
| enzyme | 30.56 | 21.22 | 22 | 40 | 136 |
| substrate | 56.58 | 35.23 | 55 | 91 | 174 |
| microbe | 38.21 | 26.05 | 31 | 46 | 108 |

Interpretation:

- This is the first usable full GVP baseline.
- Microbe loader repair substantially improved E/M and S/M retrieval compared
  with the earlier invalid all-zero-microbe run:
  - `E→M MRR`: 0.270 -> 0.395
  - `S→M MRR`: 0.334 -> 0.642
- Microbe no longer collapses:
  - previous invalid all-zero run: effective rank 1
  - fixed full GVP run: effective rank 38.21
- R/E retrieval remains weak and requires later analysis/tuning.

Pending:

- full ESM-C baseline for first-round GVP vs ESM-C comparison

## ESM-C Smoke Loader Optimization

The first 10k ESM-C smoke job timed out at the 2-hour walltime limit before
completing data loading.

Identified bottlenecks:

- checking ESM-C file availability with per-row `exists()` calls over the full
  dataset
- possible shard-cache thrashing when ESM-C fallback tries to use GVP shards

Optimization:

- pre-scan `features/enzyme/esm_c_features/` with `os.listdir()` once and build
  a UID availability set
- in ESM-C mode, preload all 192 GVP shards for fallback instead of using an
  LRU-4 cache
- print ESM-C loaded/fallback counts for audit
- increase smoke walltime from 2h to 4h

This is an ESM-C loader performance optimization. It does not change the model,
loss, training stages, or evaluation metrics.

## 10k ESM-C Smoke Result

The optimized 10k ESM-C smoke run completed successfully.

Job:

- Slurm job ID: `114962223`

Loading:

- ESM-C unique UID feature files available: `107731`
- sampled rows loaded with ESM-C: `10000 / 10000`
- row-level ESM-C fallback in this smoke: `0`
- sampled 10k rows:
  - ESM-C loaded: `10000 / 10000`
  - fallback: `0`

Metrics compared with 10k GVP smoke:

| Metric | GVP 10k | ESM-C 10k |
|--------|--------:|----------:|
| `R→E top-1` | 0.0569 | 0.0725 |
| `R→E top-10` | 0.266 | 0.359 |
| `R→E MRR` | 0.128 | 0.169 |
| `E→M top-1` | 0.0045 | 0.0133 |
| `E→M MRR` | 0.205 | 0.487 |
| `S→M top-1` | 0.0223 | 0.0212 |
| `S→M MRR` | 0.626 | 0.642 |

Effective rank:

| Modality | GVP 10k | ESM-C 10k |
|----------|--------:|----------:|
| reaction | 28.31 | 34.55 |
| enzyme | 30.56 | 35.52 |
| substrate | 56.58 | 58.51 |
| microbe | 38.21 | 29.24 |

Interpretation:

- ESM-C smoke passed with no fallback in the sampled data.
- ESM-C outperformed GVP on R/E and E/M retrieval in the 10k smoke.
- Full ESM-C baseline is ready to submit.

## Full ESM-C Baseline Result

Job:

- Slurm job ID: `114967762`
- enzyme feature: ESM-C
- data: full `145,607` examples
- output directory:
  `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/full_esmc_baseline_2026-06-07`

Loading:

- ESM-C unique UID feature files available: `107731`
- ESM-C loaded rows: `145607 / 145607`
- row-level missing/fallback due unavailable ESM-C: `0`
- Note: multiple examples can share a UniProt UID, so all rows can load ESM-C
  features even though the unique UID count is lower than the row count.

Loss:

```text
Stage 0 epoch  1: loss=2.8163  tau=0.5000
Stage 1 epoch 10: loss=39.460  tau=0.4374
Stage 1 epoch 20: loss=37.164  tau=0.2655
Stage 3 epoch 30: loss=45.625  tau=0.0999
Stage 3 epoch 38: loss~45.20 tau=0.05
```

Metrics:

| Metric | Full ESM-C baseline |
|--------|--------------------:|
| `R→E top-1` | 0.0161 |
| `R→E top-10` | 0.0896 |
| `R→E MRR` | 0.058 |
| `E→M top-1` | 0.0036 |
| `E→M MRR` | 0.609 |
| `S→M top-1` | 0.0056 |
| `S→M MRR` | 0.611 |

Lightweight effective rank:

| Modality | Effective rank | Participation ratio | dim@90 | dim@95 | dim@99 |
|----------|---------------:|--------------------:|-------:|-------:|-------:|
| reaction | 43.72 | 32.36 | 33 | 42 | 97 |
| enzyme | 43.54 | 34.22 | 32 | 40 | 96 |
| substrate | 62.33 | 40.80 | 55 | 90 | 173 |
| microbe | 41.96 | 31.66 | 32 | 42 | 93 |

Artifacts:

- `model_v3.pt`: 24M
- `embeddings_v3.npz`: 569M
- `metrics_v3.json`: 619B
- `training_history.json`: 2.1K
- `metadata_v3.json`: 22M
- `enzyme2microbe_index.json`: 4.3M
- fallback normalized retrieval files:
  - `reaction_nn_index.npz`
  - `enzyme_nn_index.npz`
  - `substrate_nn_index.npz`
  - `microbe_nn_index.npz`
- `effective_rank_summary_lightweight.json`: 2.3K

Status:

- Loss NaN: none.
- Training OOM: none.
- Native FAISS: replaced by `*_nn_index.npz` fallback.

## First-Round Full Baseline Comparison

| Metric | Full GVP | Full ESM-C | Takeaway |
|--------|---------:|-----------:|----------|
| `R→E top-1` | 0.0118 | 0.0161 | ESM-C better |
| `R→E top-10` | 0.0671 | 0.0896 | ESM-C better |
| `R→E MRR` | 0.044 | 0.058 | ESM-C better |
| `E→M top-1` | 0.0017 | 0.0036 | ESM-C better |
| `E→M MRR` | 0.395 | 0.609 | ESM-C much better |
| `S→M top-1` | 0.0054 | 0.0056 | similar |
| `S→M MRR` | 0.642 | 0.611 | GVP slightly better |

Effective rank comparison:

| Modality | Full GVP erank | Full ESM-C erank |
|----------|---------------:|-----------------:|
| reaction | 28.31 | 43.72 |
| enzyme | 30.56 | 43.54 |
| substrate | 56.58 | 62.33 |
| microbe | 38.21 | 41.96 |

First-round interpretation:

- ESM-C is the stronger enzyme representation for this full-data first round.
- ESM-C improves both direct reaction-enzyme retrieval and enzyme-microbe
  alignment.
- ESM-C also gives more balanced effective ranks across modalities.
- GVP remains slightly better for `S→M MRR`, but the difference is smaller than
  ESM-C's gains on `R→E` and `E→M`.
- Overall, use ESM-C as the preferred first-round baseline for second-round
  parameter tuning.

## NaN Stability Issue During First Full GVP Start

During the first full GVP baseline attempt, training entered the full-data run
but the loss became `NaN`.

Action:

- The invalid run was stopped.
- A separate NaN diagnostic job was submitted.
- Raw GVP shard data reportedly had no NaN values.
- The likely source was numerical instability in the full-data
  `gvp_shard_pool()` adapter for degenerate coordinate cases.

Diagnostic result after patch:

```text
DIAG_NAN_CHECK: PASS
reaction NaN=0, range [0, 1]
enzyme NaN=0, range [-0.98, 1.0]
substrate NaN=0, range [0, 1]
forward NaN=0, norm_mean=1.0
loss=8.3179, NaN=False
```

Interpretation:

- This was a numerical-stability fix in the full-data GVP pooling adapter, not a
  change to the teacher's training objective.
- The full GVP baseline was restarted after the diagnostic passed.

Pending:

- final diff for the NaN-safe `gvp_shard_pool()` change
- restarted full GVP baseline results

## Full-Data Evaluation OOM

The restarted full GVP baseline completed training, but the job was killed
during evaluation/save with OOM.

Observed:

```text
exit code 137 / SIGKILL
slurmstepd: Detected OOM-kill events
```

NaN was fixed and training reached all 38 epochs, but no final outputs were
saved because the original script saves `model_v3.pt` after evaluation.

Root cause:

The demo evaluation was written for small data and constructs dense all-vs-all
similarity and rank matrices:

```python
sim_re = all_r @ all_e.T
rank_re = np.argsort(-sim_re, axis=1)
```

For 145,607 examples, one dense similarity matrix is about 21.2 billion values.
This is too large for the requested job memory, and the argsort/rank matrix is
even more memory intensive.

Required full-data evaluation patch:

- save a checkpoint immediately after training and before evaluation
- replace dense all-vs-all evaluation with chunked memory-safe evaluation
- preserve metric semantics where possible:
  - compute top-k from chunked scores
  - compute exact MRR by counting how many candidates score above the true pair
    in chunks

This patch changes evaluation implementation for full-data scalability. It does
not change the model, loss, training stages, or training objective.

Patch status:

- Completed on HPC.
- `evaluate_multimodal()` now evaluates in chunks instead of constructing
  `N x N` matrices.
- Checkpoint save was moved before evaluation:
  - `model_v3.pt`
  - `training_history.json`
- `py_compile`: OK.
- Reported unchanged:
  - `Config`
  - model
  - loss functions
  - `train_four_stages`
  - optimizer/scheduler
- No full baseline was started during this patch.

Retry plan:

- Retry full GVP baseline with the same full-data Config.
- Keep `batch_size = 4096` initially because the previous job completed training
  at that batch size; the failure was evaluation memory.
- If evaluation still OOMs, reduce evaluation chunk size first. Reduce training
  batch size only if training itself OOMs.

Additional checkpoint serialization fix:

- Moving checkpoint save before evaluation exposed a config serialization issue.
- The original checkpoint config construction could include the
  `@property total_epochs` object, which cannot be pickled.
- A one-line serialization fix was applied so checkpoint saving can proceed.
- This does not change training behavior; it only changes what is serialized in
  the checkpoint metadata.

## FAISS/Postprocessing Issue After Chunked Evaluation

After the checkpoint-save and chunked-evaluation patches, the full GVP run
progressed further:

- training completed
- checkpoint saved
- embeddings saved
- chunked evaluation completed

The run then failed during final postprocessing:

- `build_faiss_index()` failed at `faiss.normalize_L2(emb)`, likely because the
  NumPy array passed to FAISS was not contiguous.
- `diagnose_effective_rank.py` also hit OOM when run in the same end-to-end
  script.
- `metrics_v3.json` was not saved because the save order still allowed later
  postprocessing to block final metric output.

Patch direction:

- save `metrics_v3.json` before FAISS index construction
- pass contiguous `float32` arrays into FAISS
- keep effective-rank diagnosis separate or make it memory-safe, so it does not
  block the main model/metrics artifacts

This is an artifact/postprocessing scalability fix. It does not change the
training objective, model, losses, or training stages.

## First Completed Full GVP Artifact Run: Not Final Due Microbe Loader

Job `114932071` completed and produced the main model/embedding/metric
artifacts:

- `model_v3.pt`
- `embeddings_v3.npz`
- `metrics_v3.json`
- `training_history.json`
- `metadata_v3.json`

Reported metrics:

| Metric | Value |
|--------|------:|
| `R→E MRR` | 0.0469 |
| `R→E top-1` | 0.0130 |
| `R→E top-5` | 0.0450 |
| `R→E top-10` | 0.0709 |
| `E→M MRR` | 0.2696 |
| `S→M MRR` | 0.3341 |

However, this run should not be treated as the final scientific GVP baseline.

Reason:

- Diagnostics reported microbe embedding collapse with effective rank `1/256`.
- Investigation showed the current full-data microbe JSONL is flat, while the
  teacher demo loader expects nested keys:
  - `core_preference`
  - `stoich_query`
  - `main_metabolite_coverage`
- Because those nested keys are absent, `extract_microbe_features()` returns
  mostly/all zero microbe features even though the loader reports all rows as
  "loaded".

The numeric source tables needed to build the intended 28-dimensional microbe
features are available:

- `tables/microbe_reaction_core_preference.csv`
- `tables/microbe_reaction_stoich_query.csv`
- `tables/microbe_reaction_main_metabolite_coverage.csv`

Next required patch:

- adapt the full-data microbe loader to assemble the expected nested record by
  `example_id`, or make `extract_microbe_features()` support the flat/table
  full-data format.
- validate nonzero microbe feature statistics before rerunning GVP.

Interpretation:

- This run validates the engineering pipeline after the loader/evaluation
  patches.
- It does not yet validate the biological microbe modality.

Microbe loader patch status:

- Completed and probed on HPC.
- The loader now constructs the expected nested record from:
  - `tables/microbe_reaction_core_preference.csv`
  - `tables/microbe_reaction_stoich_query.csv`
  - `tables/microbe_reaction_main_metabolite_coverage.csv`
- JSONL is retained as fallback if the CSV tables are unavailable.
- CSV string values are handled with safe conversion helpers.

Probe result:

| Check | Result |
|------|--------|
| microbe feature shape | `(145607, 28)` |
| nonzero element ratio | `40.19%` |
| all-zero rows | `0/145607` |
| concept targets | all 8 columns non-NaN, `47.33%` nonzero |
| non-empty assembly IDs | `145607/145607` |
| enzyme2microbe index | `107731` enzymes mapped |
| GVP coverage | `145607/145607` |
| unique assemblies | `2475` |

Conclusion:

- The microbe modality is now loaded correctly.
- Because this substantially changes the input signal, a sampled smoke run
  should be done before another full GVP baseline.

## 10k Smoke After Microbe Fix

A 10k sampled GVP smoke run was completed after the microbe loader patch and
FAISS fallback patch.

Results:

| Metric | 10k smoke |
|--------|----------:|
| `R→E top-1` | 0.0569 |
| `R→E top-10` | 0.266 |
| `R→E MRR` | 0.128 |
| `E→M top-1` | 0.0045 |
| `E→M MRR` | 0.205 |
| `S→M top-1` | 0.0223 |
| `S→M MRR` | 0.626 |

Microbe effective rank:

```text
previous invalid full run: 1/256
10k smoke after fix: 116.73/256
microbe composed path erank: 26.22/28
```

Interpretation:

- The repaired microbe loader prevents microbe embedding collapse.
- Loss remained finite.
- Chunked evaluation and checkpoint saving worked.
- The smoke run validates the microbe repair and supports rerunning the full
  GVP baseline.

Remaining artifact issue:

- Native FAISS index generation fails in this HPC environment due FAISS SWIG and
  NumPy 1.26.4 incompatibility.
- This does not affect training, embeddings, or metrics.
- Fallback `*_nn_index.npz` files are now saved for normalized cosine-search
  embeddings:
  - `reaction_nn_index.npz`
  - `enzyme_nn_index.npz`
  - `substrate_nn_index.npz`
  - `microbe_nn_index.npz`
- Visualization plots were not generated in the DCU/headless environment.

## Parameter Source Clarification

The teacher README originally contains the `Data-Sensitive Parameters` table
with full-data recommended ranges, including:

- `batch_size`: 4096
- `epochs_stage0`: 5-10
- `epochs_stage1`: 10-15
- `epochs_stage2`: 5-10
- `temp_end`: 0.05-0.07
- `w_sm`: 0.4-0.6
- `vicreg_var_weight`: 10-25
- `hard_neg_weight`: 2.0-4.0
- `lr`: 1e-4 to 5e-4

The table is from the teacher-provided README.

The exact single-value full GVP baseline parameter proposal is not teacher
original text; it is our selected first-run configuration within the teacher's
recommended ranges, to be documented separately before training.

Stage 3 note:

- The teacher README table does not provide a full-data recommendation for
  `epochs_stage3`.
- For the first full GVP baseline, keep `epochs_stage3 = 10` unless the teacher
  or run-time constraints require otherwise.

## Full-Data Config Patch

The first full GVP baseline uses a separate `Config` patch based on the
teacher README `Data-Sensitive Parameters` table.

Changed values:

| Field | Demo default | Full GVP baseline | Source |
|------|-------------:|------------------:|--------|
| `lr` | `1e-3` | `3e-4` | teacher README range `1e-4` to `5e-4` |
| `epochs_stage0` | `20` | `8` | teacher README full range `5-10` |
| `epochs_stage1` | `40` | `12` | teacher README full range `10-15` |
| `epochs_stage2` | `30` | `8` | teacher README full range `5-10` |
| `epochs_stage3` | `10` | `10` | README gives no full value; kept default |
| `batch_size` | `64` | `4096` | teacher README full value `4096` |
| `temp_end` | `0.07` | `0.05` | teacher README full range `0.05-0.07` |
| `w_sm` | `0.4` | `0.5` | teacher README full range `0.4-0.6` |
| `vicreg_var_weight` | `25.0` | `10.0` | teacher README full range `10-25` |
| `hard_neg_weight` | `2.0` | `2.0` | kept default within README range |

Total epochs:

```text
demo default: 20 + 40 + 30 + 10 = 100
full baseline: 8 + 12 + 8 + 10 = 38
```

Audit:

- `python -m py_compile train.py`: OK
- No full baseline job was started during this patch.
- Changes were limited to `Config` values.
- Training method, model, loss, optimizer/scheduler, evaluation, and data
  loader logic were not changed by this parameter patch.

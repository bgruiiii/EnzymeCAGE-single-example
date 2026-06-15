# R2 EC-4 Bucket Baseline

## 1. Purpose

This report documents the **R2 EC-4 bucket baseline evaluation** computed before R3 training begins. It measures grouped R→E MRR stratified by EC-4 group size (tail / mid / head) to establish a reference point for future R3 improvements. **No R3 training, model modification, or calibration acceptance criterion was introduced.**

## 2. Inputs

| File | Path |
|------|------|
| embeddings | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/embeddings_v3.npz` |
| metadata | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/metadata_v3.json` |
| metrics | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/metrics_v3.json` |

## 3. Method

### 3.1 Strict EC-4 Parser

- `ec_number` must be a non-empty string
- Split by `'.'` must yield ≥ 4 segments
- First 4 segments must all pass `int()` conversion
- Otherwise the row is **excluded** from EC-4 bucket evaluation

### 3.2 Bucket Definitions

| Bucket | Rule |
|--------|------|
| tail | EC-4 group size ≤ 4 |
| mid | 4 < EC-4 group size ≤ 317 |
| head | EC-4 group size > 317 |

### 3.3 Grouped MRR Computation

- **Query rows**: only rows in the current bucket
- **Candidate enzyme rows**: ALL enzyme rows (not bucket-local)
- **Positive set**: all rows sharing the same EC-4 key
- **best_pos_score** = max similarity over the positive set
- **rank** = 1 + count(candidates with score > best_pos_score)
- **reciprocal rank** = 1 / rank
- **bucket MRR** = mean reciprocal rank over bucket query rows
- Chunked retrieval: `chunk_size = 1024`, no full N×N matrix
- L2 normalization applied explicitly before dot-product (cosine similarity)

## 4. EC-4 Group Statistics

- **Total rows (N)**: 145607
- **Valid EC-4 rows**: 127847
- **Excluded rows (EC-4 unknown)**: 17760
- **Valid EC-4 groups**: 2524
- **Group size — min / median / mean / p95 / max**: 1 / 4.0 / 50.65 / 317.8 / 2291

| Bucket | n groups | n rows | min | median | mean | p95 | max |
|--------|----------|--------|-----|--------|------|-----|-----|
| tail | 1397 | 2783 | 1 | 2.0 | 1.99 | 4.0 | 4 |
| mid | 1000 | 45030 | 5 | 16.0 | 45.03 | 198.0 | 317 |
| head | 127 | 80034 | 318 | 532.0 | 630.19 | 1166.3 | 2291 |

## 5. Overall Consistency Check

| Metric | Value |
|--------|-------|
| Computed overall valid EC-4 grouped MRR | **0.9141450209** |
| metrics_v3 EC-4-grouped MRR | **0.9132133258** |
| Relative error | 1.020238e-03 |
| Consistency tolerance | 0.01 |
| Status | **PASS** |

## 6. Bucket Baseline

| Bucket | Rule | n groups | n rows | MRR | top-1 | top-5 | top-10 |
|--------|------|----------|--------|-----|-------|-------|--------|
| tail | ≤4 | 1397 | 2783 | 0.904378 | 0.763205 | 0.915199 | 0.937837 |
| mid | 5–317 | 1000 | 45030 | 0.911926 | 0.828692 | 0.915901 | 0.936775 |
| head | >317 | 127 | 80034 | 0.915733 | 0.850214 | 0.953095 | 0.965240 |
| **ALL** | all valid EC-4 | 2524 | 127847 | **0.914145** | 0.840739 | 0.939169 | 0.954618 |

## 7. Output Files

| File | Path |
|------|------|
| JSON report | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/r2_ec4_bucket_baseline.json` |
| Markdown report | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs/R2_EC4_BUCKET_BASELINE.md` |
| Eval script | `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/eval_ec4_buckets.py` |

## 8. Declarations

- train.py modified: **no**
- eval script created: **yes**
- Slurm submitted: **no**
- GPU/DCU used: **no**
- retraining executed: **no**
- new calibration/OOD/latency acceptance criteria introduced: **no**
- ready for local audit: **yes**

---

*Generated: 2026-06-15 10:36:38 | hostname: login01 | chunk_size: 1024*

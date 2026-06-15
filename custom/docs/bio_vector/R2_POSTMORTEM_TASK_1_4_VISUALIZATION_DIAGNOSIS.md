# R2 Postmortem Task 1.4 — Visualization Error Diagnosis

**Date**: 2026-06-12  
**Analyst**: Bio-Vector R2 Postmortem Team  
**Output dir**: `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11`  
**Job ID**: `r2_train_115204112`  
**Status**: **TASK_1_4_READY_FOR_REVIEW**

---

## 1. Complete Traceback (Diagnostic Item ①)

Source: `r2_train_115204112.err`, lines 145611–145644

```
Traceback (most recent call last):
  File "/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/train.py", line 1614, in <module>
    main()
  File "/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/train.py", line 1590, in main
    visualize_four_modal(metrics, all_r, all_e, all_s, all_m,
  File "/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo/train.py", line 914, in visualize_four_modal
    fig.savefig(output_dir / "unified_space_v3_results.png", dpi=150)
  File ".../matplotlib/figure.py", line 3395, in savefig
    self.canvas.print_figure(fname, **kwargs)
  File ".../matplotlib/backend_bases.py", line 2204, in print_figure
    result = print_method(...)
  File ".../matplotlib/backend_bases.py", line 2054, in <lambda>
    print_method = functools.wraps(meth)(lambda *args, **kwargs: meth(...)
  File ".../matplotlib/backends/backend_agg.py", line 496, in print_png
    self._print_pil(filename_or_obj, "png", pil_kwargs, metadata)
  File ".../matplotlib/backends/backend_agg.py", line 444, in _print_pil
    FigureCanvasAgg.draw(self)
  File ".../matplotlib/backends/backend_agg.py", line 387, in draw
    self.figure.draw(self.renderer)
  File ".../matplotlib/artist.py", line 95, in draw_wrapper
    result = draw(artist, renderer, *args, **kwargs)
  File ".../matplotlib/artist.py", line 72, in draw_wrapper
    return draw(artist, renderer)
  File ".../matplotlib/figure.py", line 3161, in draw
    self.patch.draw(renderer)
  File ".../matplotlib/artist.py", line 72, in draw_wrapper
    return draw(artist, renderer)
  File ".../matplotlib/patches.py", line 632, in draw
    self._draw_paths_with_artist_properties(...)
  File ".../matplotlib/patches.py", line 617, in _draw_paths_with_artist_properties
    renderer.draw_path(gc, *draw_path_args)
  File ".../matplotlib/backends/backend_agg.py", line 131, in draw_path
    self._renderer.draw_path(gc, path, transform, rgbFace)
ValueError: object __array__ method not producing an array
```

### Traceback Analysis

| Field | Value |
|---|---|
| **Error type** | `ValueError: object __array__ method not producing an array` |
| **Origin function** | `visualize_four_modal()` at `train.py:914` |
| **Trigger call** | `fig.savefig(output_dir / "unified_space_v3_results.png", dpi=150)` |
| **Call depth** | 16 frames (train.py → matplotlib savefig → AGG renderer → draw_path) |
| **Runtime environment** | Python 3.9.23, matplotlib 3.9.4, numpy 1.26.4 |
| **Root cause (inference)** | Most likely an environment-specific matplotlib/numpy AGG rendering issue, triggered during patch rendering in the axvspan stage-color background. This is an inference based on the call stack, not a data error in training, embedding, or indexing. |

### Code Context (train.py lines 874–878)

The training-loss subplot uses per-epoch `axvspan()` calls to color-code training stages:

```python
# train.py:877-878
for i, (l, s) in enumerate(zip(history["loss"], history["stage"])):
    axes[0, 1].axvspan(i-0.5, i+0.5, alpha=0.1, color=stage_colors.get(s, 'gray'))
```

Each `axvspan` creates a `Rectangle` patch. During `fig.savefig()`, matplotlib renders all patches, and the AGG backend's `draw_path` appears to encounter an issue when converting patch coordinates to arrays. This is inferred from the traceback; the exact internal trigger is within matplotlib's C-level AGG renderer.

---

## 2. Four-Modal Embedding Health Check (Diagnostic Item ②)

File: `embeddings_v3.npz`  
**Status**: ✅ EXISTS — 596 MB, all four modalities present

| Key | type | dtype | shape | NaN count | Inf count | Status |
|---|---|---|---|---|---|---|
| `reaction` | `ndarray` | `float32` | `(145607, 256)` | 0 | 0 | ✅ PASS |
| `enzyme` | `ndarray` | `float32` | `(145607, 256)` | 0 | 0 | ✅ PASS |
| `substrate` | `ndarray` | `float32` | `(145607, 256)` | 0 | 0 | ✅ PASS |
| `microbe` | `ndarray` | `float32` | `(145607, 256)` | 0 | 0 | ✅ PASS |

**Summary**:
- All four modality embeddings are present with correct unified dimension (256).
- Row count (145,607) matches training dataset size.
- Zero NaN / Inf values — embeddings are numerically clean.
- dtype is float32 as expected from the model's output projection.

---

## 3. NN Index Health Check (Diagnostic Item ③)

| Index File | Exists | ntotal | dim | Emb Rows | Emb Dim | Consistent |
|---|---|---|---|---|---|---|
| `reaction_nn_index.npz` | ✅ | 145,607 | 256 | 145,607 | 256 | ✅ |
| `enzyme_nn_index.npz` | ✅ | 145,607 | 256 | 145,607 | 256 | ✅ |
| `substrate_nn_index.npz` | ✅ | 145,607 | 256 | 145,607 | 256 | ✅ |
| `microbe_nn_index.npz` | ✅ | 145,607 | 256 | 145,607 | 256 | ✅ |

**Notes**:
- All indices use the numpy NN fallback format (FAISS SWIG was incompatible with numpy 1.26.4, as noted in stdout).
- Each index stores `embeddings` (float32 matrix), `dim` (int64 scalar), and `n` (int64 scalar).
- ntotal and dim are consistent with `embeddings_v3.npz` row counts and dimensions.

---

## 4. 100-Row R→E EC-4 Top-1 Health Check (Diagnostic Item ④)

### Configuration

| Parameter | Value |
|---|---|
| Random seed | `20260612` (`np.random.RandomState`) |
| Sample size | 100 rows from `0..145606` |
| Evaluated (strict EC-4 valid) | 88 rows |
| Excluded (EC-4 unknown) | 12 rows |
| Similarity metric | Cosine (L2-normalized dot product) |
| Scope | Sanity check — NOT a full metrics recalculation |

### Strict EC-4 Parser

The EC-4 parser uses the same strict criteria as the R2 grouped evaluation in `metrics_v3.json`:

```python
def strict_ec4(ec_str):
    if not isinstance(ec_str, str):       # must be string
        return None
    parts = ec_str.split('.')
    if len(parts) < 4:                     # at least 4 segments
        return None
    try:
        for p in parts[:4]:
            int(p)                         # first 4 segments must be valid int()
    except (ValueError, TypeError):
        return None
    return '.'.join(parts[:4])
```

Rows with non-standard EC numbers (e.g. `3.6.5.n1`, empty string `""`) are excluded from hit-rate computation and reported separately.

### Results

| Metric | Value |
|---|---|
| EC-4 Top-1 Hits | 80 / 88 |
| **100-row EC-4 Top-1 Hit Rate** | **0.9091** (90.91%) |
| metrics_v3 EC-4-grouped R→E top-1 (full) | 0.8389 (83.89%) |
| Delta | +7.02 pp |

### Excluded Unknown Examples (12 rows)

| Row | Raw EC Number | Reason |
|---|---|---|
| 67546 | `3.6.5.n1` | 4th segment `n1` is not a valid integer |
| 65935 | `3.6.5.n1` | 4th segment `n1` is not a valid integer |
| 65739 | `3.6.5.n1` | 4th segment `n1` is not a valid integer |
| 135099 | `""` (empty) | Empty string, no segments |
| 134503 | `""` (empty) | Empty string, no segments |
| 140950 | `""` (empty) | Empty string, no segments |
| 142982 | `""` (empty) | Empty string, no segments |
| 134552 | `""` (empty) | Empty string, no segments |
| 144058 | `""` (empty) | Empty string, no segments |
| 134436 | `""` (empty) | Empty string, no segments |
| 142775 | `""` (empty) | Empty string, no segments |
| 144590 | `""` (empty) | Empty string, no segments |

Excluded ratio: 12/100 = 12.0%, compared with full-set excluded ratio 17,760/145,607 = 12.2% — consistent.

### Interpretation

The 100-row sample hit rate of 90.91% is **within expected sampling variance** of the full-set metric of 83.89%. With n=88 evaluated queries, the standard error of the proportion is approximately:

\[
SE = \sqrt{\frac{p(1-p)}{n}} = \sqrt{\frac{0.8389 \times 0.1611}{88}} \approx 0.0392
\]

The observed delta (+7.02 pp) corresponds to approximately 1.79 SE. This is within 2 SE and well within expected sampling variance for a 100-row sanity check.

This is a **sanity check only** — the full EC-4-grouped evaluation (127,847 evaluated, 17,760 excluded unknown) remains the authoritative metric.

### Sample Detail (first 20 evaluated rows)

| Query Row | Query EC-4 | Top-1 Row | Top-1 EC-4 | Cosine Sim | Result |
|---|---|---|---|---|---|
| 55380 | 2.4.2.8 | 55427 | 2.4.2.8 | 0.9312 | HIT |
| 6646 | 4.2.1.19 | 6374 | 4.2.1.19 | 0.9175 | HIT |
| 115989 | 7.4.2.11 | 22362 | 5.6.2.4 | 0.8683 | MISS |
| 57841 | 2.7.1.148 | 57644 | 2.7.1.148 | 0.9703 | HIT |
| 85903 | 1.5.1.5 | 85692 | 1.5.1.5 | 0.9112 | HIT |
| 82642 | 2.5.1.3 | 82734 | 2.5.1.3 | 0.9536 | HIT |
| 107406 | 2.7.7.70 | 107238 | 2.7.7.70 | 0.9676 | HIT |
| 77860 | 1.2.1.38 | 77913 | 1.2.1.38 | 0.9504 | HIT |
| 22510 | 5.6.2.4 | 22362 | 5.6.2.4 | 0.8683 | HIT |
| 128330 | 1.1.1.390 | 32096 | 1.1.1.47 | 0.7887 | MISS |
| 30367 | 4.3.1.18 | 30313 | 4.3.1.18 | 0.9567 | HIT |
| 13487 | 2.4.2.18 | 13553 | 2.4.2.18 | 0.9705 | HIT |
| 76924 | 1.1.1.37 | 76728 | 1.1.1.37 | 0.9458 | HIT |
| 133093 | 7.1.2.2 | 22362 | 5.6.2.4 | 0.8683 | MISS |
| 29426 | 2.3.1.157 | 26757 | 2.7.7.23 | 0.9780 | MISS |
| 38770 | 2.1.2.1 | 38392 | 2.1.2.1 | 0.9579 | HIT |
| 98017 | 1.13.11.54 | 31338 | 1.13.11.53 | 0.9791 | MISS |
| 45341 | 3.2.2.6 | 45315 | 3.2.2.6 | 0.9473 | HIT |
| 87482 | 4.1.1.39 | 87519 | 4.1.1.39 | 0.9695 | HIT |
| 122122 | 1.14.18.1 | 55986 | 1.14.18.1 | 0.9245 | HIT |

---

## 5. Conclusions

### 5.1 Embedding File Health

**✅ HEALTHY** — All four modality embeddings (reaction, enzyme, substrate, microbe) are numerically clean: correct shapes, float32 dtype, zero NaN, zero Inf, consistent row counts with training data.

### 5.2 NN Index Health

**✅ HEALTHY** — All four numpy NN fallback indices exist with correct ntotal (145,607) and dimension (256), fully consistent with the embedding file.

### 5.3 Visualization Error Impact on Core Results

**✅ NO IMPACT** — The visualization error occurs at `train.py:914` (`fig.savefig()`), which is the **last step** of `main()` (line 1590). All core outputs were successfully saved **before** this point:

| Output | Status | Saved At |
|---|---|---|
| `model_v3.pt` | ✅ Saved | 18:39 (pre-eval checkpoint) |
| `model_v3_stage{0,1,2,3}.pt` | ✅ Saved | During training stages |
| `embeddings_v3.npz` | ✅ Saved | 19:15 (post-training encoding) |
| `*_nn_index.npz` (×4) | ✅ Saved | 19:15 (post-embedding indexing) |
| `metrics_v3.json` | ✅ Saved | 19:15 (post-evaluation) |
| `metadata_v3.json` | ✅ Saved | 19:15 |
| `enzyme2microbe_index.json` | ✅ Saved | 19:15 |
| `training_history.json` | ✅ Saved | 18:39 |
| `unified_space_v3_results.png` | ❌ Not saved | Visualization-only artifact |

The PNG file is an **optional visualization summary** — it does not contain any data that is not already captured in `metrics_v3.json` and `embeddings_v3.npz`.

### 5.4 Engineering Remediation

**✅ try/except wrapping is appropriate** as an engineering safeguard. The visualization function `visualize_four_modal()` is a pure side-effect function (produces a PNG file) with no return value consumed by downstream logic. Wrapping the call at `train.py:1590` with:

```python
try:
    visualize_four_modal(metrics, all_r, all_e, all_s, all_m,
                         labels, history, out)
except Exception as vis_err:
    print(f"  [WARN] Visualization skipped due to: {vis_err}")
```

would be a sound engineering practice. This ensures the training pipeline exits cleanly (exit code 0) even when optional visualization encounters environment-specific rendering issues.

The most likely explanation is an environment-specific matplotlib/numpy AGG rendering issue, triggered during patch rendering in the axvspan stage-color background. This is an inference and does not reflect any issue in the model, training logic, or embedding quality.

---

## Diagnostic Summary

| Diagnostic Item | Description | Status |
|---|---|---|
| ① Complete Traceback | ValueError in `fig.savefig()` at train.py:914 via matplotlib AGG draw_path | ✅ **PASS** |
| ② Embedding type/dtype/shape | 4 modalities × float32 × (145607, 256), 0 NaN, 0 Inf | ✅ **PASS** |
| ③ NN Index health check | 4 indices × ntotal=145607 × dim=256, consistent with embeddings | ✅ **PASS** |
| ④ 100-row R→E EC-4 top-1 | 80/88 = 90.91%, consistent with full-set 83.89% (within 2 SE) | ✅ **PASS** |

---

## Final Verdict

| Criterion | Result |
|---|---|
| Embedding integrity | ✅ Confirmed |
| NN index integrity | ✅ Confirmed |
| Metrics consistency | ✅ Confirmed |
| Visualization error scope | Isolated to optional PNG output |
| Core R2 results | Complete and unaffected |

**TASK_1_4_READY_FOR_REVIEW**

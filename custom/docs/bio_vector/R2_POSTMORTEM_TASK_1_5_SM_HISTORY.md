# R2 Postmortem Task 1.5 — S→M MRR Historical Comparison

**Date**: 2026-06-12  
**Analyst**: Bio-Vector R2 Postmortem Team  
**Scope**: Substrate→Microbe (S→M) MRR comparison between R1 and R2  
**Status**: **Task 1.5 COMPLETE**

---

## 1. File Sources

### R1 (ESM-C Baseline)

| File | Path | Used For |
|---|---|---|
| `metrics_v3.json` | `outputs/full_esmc_baseline_2026-06-07/metrics_v3.json` | S→M MRR, S→M top-k, E→M MRR |
| `R1_DIAGNOSIS_20260608_172625.md` | `outputs/full_esmc_baseline_2026-06-07/R1_DIAGNOSIS_20260608_172625.md` | Cross-reference (confirmed same `metrics_v3.json` values) |

### R2 (ESM-C Hard-Neg Stage1×25)

| File | Path | Used For |
|---|---|---|
| `metrics_v3.json` | `outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/metrics_v3.json` | S→M MRR, S→M top-k, E→M MRR |

---

## 2. S→M MRR Comparison

| Metric | R1 | R2 | Delta | Relative Change | Note |
|---|---:|---:|---:|---:|---|
| S→M MRR | 0.6108 | 0.5871 | −0.0238 | −3.89% | Monitoring metric; check non-degradation |
| S→M top-1 | 0.005570 | 0.005693 | +0.000123 | +2.21% | — |
| S→M top-5 | 0.025047 | 0.025102 | +0.000055 | +0.22% | — |
| S→M top-10 | 0.045362 | 0.044194 | −0.001168 | −2.57% | — |
| S→M top-20 | 0.076713 | 0.075985 | −0.000728 | −0.95% | — |

### Full Precision

| Metric | R1 (exact) | R2 (exact) |
|---|---:|---:|
| S→M MRR | 0.6108356276986662 | 0.5870633083741500 |
| S→M top-1 | 0.005569787166825771 | 0.005693407597162224 |
| S→M top-5 | 0.025046872746502573 | 0.025101815159985440 |
| S→M top-10 | 0.045361830131793120 | 0.044194303845282160 |
| S→M top-20 | 0.076713344825454820 | 0.075985357846806820 |

---

## 3. Related Cross-Modal Metrics (Context)

| Metric | R1 | R2 | Delta | Relative Change |
|---|---:|---:|---:|---:|
| E→M MRR | 0.6094 | 0.6195 | +0.0102 | +1.67% |
| E→M top-1 | 0.003640 | 0.004382 | +0.000742 | +20.38% |

E→M MRR shows a slight improvement in R2, providing context that the S→M MRR change is not part of a broader microbe-retrieval degradation trend.

---

## 4. Interpretation

### 4.1 S→M MRR Change Assessment

S→M MRR decreased from 0.6108 (R1) to 0.5871 (R2), a relative change of −3.89%. Key observations:

- **Magnitude**: The absolute change (−0.024) is moderate. S→M MRR remains in the same performance band (~0.59).
- **Top-1 stability**: S→M top-1 actually improved slightly (+2.21%), suggesting the change is concentrated in the ranking of lower-ranked candidates rather than top retrieval accuracy.
- **E→M context**: E→M MRR improved (+1.67%), indicating microbe-side retrieval is not systematically worse in R2.

### 4.2 Non-Degradation Judgment

S→M is a **monitoring metric**, not a primary success criterion. The −3.89% relative change is a modest decrease and should continue to be monitored. It does not suggest a broad microbe-retrieval regression because E→M MRR improved and S→M top-1 remained stable:

- No abrupt drop pattern is observed in the S→M top-k metrics.
- S→M top-1 is stable/improved.
- The substrate embedding space (Morgan FP 2048-dim → 256-dim projection) was not a focus of R2 changes.

### 4.3 Primary Direction Reminder

The project's primary direction remains **R→EC-4 retrieval**. S→M is tracked as a monitoring metric to ensure no significant cross-modal regression. R2's primary metrics (EC-4-grouped R→E top-1 = 0.8389) are the authoritative measures of project progress.

### 4.4 Follow-up Recommendation

- S→M MRR does not require immediate action.
- Continue tracking S→M MRR in future runs as a monitoring metric.
- Do not set a hard threshold at this stage; future interpretation should remain tied to the downstream agent/tool requirements.

---

## 5. Status Declaration

| Item | Status |
|---|---|
| Task 1.5 | **COMPLETE** |
| train.py modification | None |
| sbatch submission | None |
| GPU/DCU usage | None |
| R1 S→M MRR found | ✅ Yes (0.6108 from `metrics_v3.json`) |
| R2 S→M MRR | 0.5871 |

---

**Task 1.5 COMPLETE — No further action required.**

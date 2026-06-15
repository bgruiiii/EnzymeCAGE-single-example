# Bio Vector R3 Task B HPC Instructions

Date: 2026-06-15

Purpose: compute the R2 EC-4 tail/mid/head bucket baseline before any R3
training. This is required by the teacher-provided R3 plan.

Teacher source:

```text
/home/a/EnzymeCAGE/custom/docs/bio_vector/BIO_VECTOR_R3_PLAN_FOR_STUDENT_2026-06-15.md
```

Important constraints:

- Do not modify `train.py`.
- Do not retrain.
- Do not create calibration / OOD / latency acceptance criteria.
- Use the saved R2 embeddings and metadata only.
- Write all results into a markdown result file, not only chat output.

## Copy-Paste Prompt For HPC AI

We are continuing EnzymeCAGE / Bio Vector R3 startup.

R2 postmortem is accepted. Before R3 training, compute the R2 EC-4
tail/mid/head bucket baseline required by the teacher. This is zero-GPU
postmortem evaluation on saved R2 outputs.

HPC paths:

```text
WORK_ROOT=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04
CODE_DIR=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/code/demo
DOCS_DIR=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/docs
R2_OUT=/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11
```

Create this script:

```text
${CODE_DIR}/eval_ec4_buckets.py
```

The script must:

1. Load `${R2_OUT}/embeddings_v3.npz`.
2. Load `${R2_OUT}/metadata_v3.json`.
3. Use strict EC-4 parsing:
   - `ec_number` must be a string.
   - `split(".")` must have at least 4 segments.
   - the first 4 segments must all pass `int()`.
   - otherwise exclude the row from EC-4 bucket evaluation.
4. Count EC-4 group sizes over valid EC-4 rows.
5. Define buckets:
   - tail: group size `<= 4`
   - mid: `4 < group size <= 317`
   - head: group size `> 317`
6. For each query row in a bucket, compute EC-4 grouped R->E rank using the
   same semantics as R2 `evaluate_grouped_re`:
   - candidates are all enzyme rows, not only bucket rows.
   - positive set is every row with the same strict EC-4 key.
   - grouped rank is `1 + number of candidates with score > best positive score`.
   - reciprocal rank is `1 / rank`.
7. Report for each bucket:
   - number of EC-4 groups
   - number of rows / query count
   - EC-4 grouped R->E MRR
   - top-1 / top-5 / top-10 grouped hit rates as sanity metrics
8. Also compute overall valid EC-4 grouped R->E MRR and compare it with
   `${R2_OUT}/metrics_v3.json` `grouped_re["EC-4-grouped_MRR"]` if the key is
   present.
9. Save JSON results to:

```text
${R2_OUT}/r2_ec4_bucket_baseline.json
```

10. Write the final markdown report to:

```text
${DOCS_DIR}/R2_EC4_BUCKET_BASELINE.md
```

Use this script content unless you find a concrete incompatibility on HPC. If
you change it, document the exact diff and reason in the markdown report.

```python
#!/usr/bin/env python3
"""
Compute R2 EC-4 tail/mid/head grouped R->E bucket baselines.

Inputs:
  - {output_dir}/embeddings_v3.npz
  - {output_dir}/metadata_v3.json
  - optional {output_dir}/metrics_v3.json for consistency check

Outputs:
  - {output_dir}/r2_ec4_bucket_baseline.json
  - markdown summary path supplied by --report_path

This script does not modify train.py and does not retrain.
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def strict_ec4(ec_str):
    if not isinstance(ec_str, str):
        return None
    parts = ec_str.split(".")
    if len(parts) < 4:
        return None
    try:
        for part in parts[:4]:
            int(part)
    except (TypeError, ValueError):
        return None
    return ".".join(parts[:4])


def load_expected_ec4_mrr(metrics_path):
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    grouped = metrics.get("grouped_re", {})
    return grouped.get("EC-4-grouped_MRR")


def maybe_l2_normalize(arr, eps=1e-12):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


def evaluate_grouped_subset(reaction, enzyme, query_indices, group_to_indices,
                            row_ec4, ks=(1, 5, 10), chunk_size=1024):
    rr = []
    hits = {k: [] for k in ks}
    query_indices = np.asarray(query_indices, dtype=np.int64)

    for start in range(0, len(query_indices), chunk_size):
        end = min(start + chunk_size, len(query_indices))
        qidx = query_indices[start:end]
        sim = reaction[qidx] @ enzyme.T

        for local_i, global_i in enumerate(qidx):
            key = row_ec4[int(global_i)]
            positives = group_to_indices[key]
            scores = sim[local_i]
            best_pos_score = float(np.max(scores[positives]))
            rank = int(np.sum(scores > best_pos_score)) + 1
            rr.append(1.0 / rank)
            for k in ks:
                hits[k].append(rank <= k)

        del sim

    out = {
        "n_queries": int(len(query_indices)),
        "mrr": float(np.mean(rr)) if rr else None,
    }
    for k in ks:
        out[f"top{k}"] = float(np.mean(hits[k])) if hits[k] else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True,
                        help="R2 output directory containing embeddings and metadata")
    parser.add_argument("--report_path", required=True,
                        help="Markdown report path to write")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--normalize", action="store_true",
                        help="L2-normalize embeddings before scoring; default is raw saved embeddings")
    args = parser.parse_args()

    t0 = time.time()
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    json_path = output_dir / "r2_ec4_bucket_baseline.json"

    emb_path = output_dir / "embeddings_v3.npz"
    meta_path = output_dir / "metadata_v3.json"
    metrics_path = output_dir / "metrics_v3.json"

    emb = np.load(emb_path)
    reaction = emb["reaction"].astype(np.float32)
    enzyme = emb["enzyme"].astype(np.float32)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if len(reaction) != len(enzyme) or len(reaction) != len(metadata):
        raise ValueError(
            f"row mismatch: reaction={len(reaction)} enzyme={len(enzyme)} "
            f"metadata={len(metadata)}"
        )

    norm_info = {
        "reaction_norm_mean": float(np.linalg.norm(reaction, axis=1).mean()),
        "enzyme_norm_mean": float(np.linalg.norm(enzyme, axis=1).mean()),
    }

    if args.normalize:
        reaction = maybe_l2_normalize(reaction)
        enzyme = maybe_l2_normalize(enzyme)

    row_ec4 = []
    counts = Counter()
    excluded = 0
    for row in metadata:
        key = strict_ec4(row.get("ec_number", ""))
        row_ec4.append(key)
        if key is None:
            excluded += 1
        else:
            counts[key] += 1

    group_to_indices = defaultdict(list)
    for i, key in enumerate(row_ec4):
        if key is not None:
            group_to_indices[key].append(i)
    group_to_indices = {
        key: np.asarray(indices, dtype=np.int64)
        for key, indices in group_to_indices.items()
    }

    bucket_groups = {
        "tail": [key for key, n in counts.items() if n <= 4],
        "mid": [key for key, n in counts.items() if 4 < n <= 317],
        "head": [key for key, n in counts.items() if n > 317],
    }

    bucket_indices = {}
    for name, groups in bucket_groups.items():
        group_set = set(groups)
        bucket_indices[name] = [
            i for i, key in enumerate(row_ec4)
            if key is not None and key in group_set
        ]

    valid_indices = [i for i, key in enumerate(row_ec4) if key is not None]

    results = {
        "output_dir": str(output_dir),
        "embedding_path": str(emb_path),
        "metadata_path": str(meta_path),
        "normalize": bool(args.normalize),
        "chunk_size": int(args.chunk_size),
        "n_total_rows": int(len(metadata)),
        "n_valid_ec4_rows": int(len(valid_indices)),
        "n_excluded_ec4_rows": int(excluded),
        "n_valid_ec4_groups": int(len(counts)),
        "ec4_group_size_min": int(min(counts.values())),
        "ec4_group_size_median": float(np.median(list(counts.values()))),
        "ec4_group_size_mean": float(np.mean(list(counts.values()))),
        "ec4_group_size_p95": float(np.percentile(list(counts.values()), 95)),
        "ec4_group_size_max": int(max(counts.values())),
        "norm_info": norm_info,
        "buckets": {},
    }

    overall = evaluate_grouped_subset(
        reaction, enzyme, valid_indices, group_to_indices, row_ec4,
        chunk_size=args.chunk_size)
    expected = load_expected_ec4_mrr(metrics_path)
    results["overall_valid_ec4"] = overall
    results["expected_metrics_v3_ec4_grouped_mrr"] = expected
    if expected is not None and overall["mrr"] is not None:
        results["overall_relative_error_vs_metrics_v3"] = (
            abs(overall["mrr"] - expected) / abs(expected)
            if expected != 0 else abs(overall["mrr"] - expected)
        )

    for name in ["tail", "mid", "head"]:
        idx = bucket_indices[name]
        metrics = evaluate_grouped_subset(
            reaction, enzyme, idx, group_to_indices, row_ec4,
            chunk_size=args.chunk_size)
        group_sizes = [counts[g] for g in bucket_groups[name]]
        results["buckets"][name] = {
            "n_groups": int(len(bucket_groups[name])),
            "n_rows": int(len(idx)),
            "group_size_min": int(min(group_sizes)) if group_sizes else 0,
            "group_size_max": int(max(group_sizes)) if group_sizes else 0,
            **metrics,
        }

    results["elapsed_seconds"] = float(time.time() - t0)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# R2 EC-4 Bucket Baseline")
    lines.append("")
    lines.append("Date: 2026-06-15")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("Compute R2 EC-4 tail/mid/head grouped R->E MRR baselines before R3 training.")
    lines.append("")
    lines.append("## 2. Inputs")
    lines.append("")
    lines.append(f"- embeddings: `{emb_path}`")
    lines.append(f"- metadata: `{meta_path}`")
    lines.append(f"- metrics: `{metrics_path}`")
    lines.append("")
    lines.append("## 3. Method")
    lines.append("")
    lines.append("- Strict EC-4 parser excludes invalid or non-four-level EC labels.")
    lines.append("- Bucket definitions: tail <= 4, mid 5-317, head > 317 rows/group.")
    lines.append("- Grouped rank uses the best score among all enzyme rows with the same EC-4.")
    lines.append("- Candidates are all enzyme rows, not only rows inside the bucket.")
    lines.append("")
    lines.append("## 4. EC-4 Group Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key in [
        "n_total_rows", "n_valid_ec4_rows", "n_excluded_ec4_rows",
        "n_valid_ec4_groups", "ec4_group_size_min",
        "ec4_group_size_median", "ec4_group_size_mean",
        "ec4_group_size_p95", "ec4_group_size_max",
    ]:
        lines.append(f"| {key} | {results[key]} |")
    lines.append("")
    lines.append("## 5. Overall Consistency Check")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| overall valid EC-4 MRR | {overall['mrr']:.6f} |")
    if expected is not None:
        lines.append(f"| metrics_v3 EC-4 grouped MRR | {expected:.6f} |")
        lines.append(
            f"| relative error | {results['overall_relative_error_vs_metrics_v3']:.6f} |")
    else:
        lines.append("| metrics_v3 EC-4 grouped MRR | not found |")
    lines.append("")
    lines.append("## 6. Bucket Baseline")
    lines.append("")
    lines.append("| Bucket | EC-4 group-size rule | n groups | n rows | MRR | top-1 | top-5 | top-10 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    rules = {"tail": "<= 4", "mid": "5-317", "head": "> 317"}
    for name in ["tail", "mid", "head"]:
        b = results["buckets"][name]
        lines.append(
            f"| {name} | {rules[name]} | {b['n_groups']} | {b['n_rows']} | "
            f"{b['mrr']:.6f} | {b['top1']:.6f} | {b['top5']:.6f} | {b['top10']:.6f} |"
        )
    lines.append("")
    lines.append("## 7. Output Files")
    lines.append("")
    lines.append(f"- JSON: `{json_path}`")
    lines.append(f"- Markdown: `{report_path}`")
    lines.append("")
    lines.append("## 8. Declarations")
    lines.append("")
    lines.append("- train.py modified: no")
    lines.append("- retraining executed: no")
    lines.append("- GPU/DCU used: no")
    lines.append("- new calibration/OOD/latency criteria introduced: no")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"JSON written to {json_path}")
    print(f"Markdown written to {report_path}")


if __name__ == "__main__":
    main()
```

Then run:

```bash
cd ${CODE_DIR}
python -m py_compile eval_ec4_buckets.py

python eval_ec4_buckets.py \
  --output_dir ${R2_OUT} \
  --report_path ${DOCS_DIR}/R2_EC4_BUCKET_BASELINE.md \
  --chunk_size 1024
```

If this is too slow or memory-heavy interactively, create a CPU-only Slurm
script. Prefer `kshctest02` if available. Do not request DCU/GPU unless the
cluster requires it for the selected partition.

Required checks after running:

```bash
ls -lh ${CODE_DIR}/eval_ec4_buckets.py
ls -lh ${R2_OUT}/r2_ec4_bucket_baseline.json
ls -lh ${DOCS_DIR}/R2_EC4_BUCKET_BASELINE.md
python -m py_compile ${CODE_DIR}/eval_ec4_buckets.py
tail -n 120 ${DOCS_DIR}/R2_EC4_BUCKET_BASELINE.md
```

If Slurm is used, also return:

```bash
sacct -j <JOB_ID> --format=JobID,JobName,Partition,NodeList,State,Elapsed,ExitCode,ReqMem,MaxRSS,AllocTRES
tail -n 120 <stdout-file>
tail -n 120 <stderr-file>
```

Return to local Codex:

1. Exact path of `eval_ec4_buckets.py`.
2. Exact path of `R2_EC4_BUCKET_BASELINE.md`.
3. Exact path of `r2_ec4_bucket_baseline.json`.
4. Full contents of `R2_EC4_BUCKET_BASELINE.md`.
5. `py_compile` result.
6. `ls -lh` for all produced files.
7. Slurm job information if a job was submitted.
8. Explicit declarations:
   - train.py modified: yes/no
   - retraining executed: yes/no
   - GPU/DCU used: yes/no
   - new thresholds introduced: yes/no


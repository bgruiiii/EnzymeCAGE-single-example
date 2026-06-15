#!/usr/bin/env python3
"""
Postmortem Stage Checkpoint Evaluation for Bio Vector R2.

Evaluates model_v3_stage{0,1,2,3}.pt checkpoints to produce a
stage-wise MRR evolution table (row-level R→E, UniProt-grouped R→E,
EC-4-grouped R→E, E→M MRR).

Input:
  - Stage checkpoints: {output_dir}/model_v3_stage{N}.pt
  - Data features: loaded via train.load_enzyme_cage_300()
Output:
  - Prints stage-wise MRR table to stdout
  - Writes JSON summary to {output_dir}/postmortem_stage_eval.json
  - Stage 3 consistency check against metrics_v3.json

Dependencies:
  - train.py (Config, UnifiedSpace, MultiModalDataset,
    load_enzyme_cage_300, evaluate_multimodal, evaluate_grouped_re)
  - numpy, torch, json

Usage:
  python postmortem_eval_stage_checkpoints.py --stage all
  python postmortem_eval_stage_checkpoints.py --stage 1
  python postmortem_eval_stage_checkpoints.py --stage 1 \
      --data_dir /path/to/data \
      --output_dir /path/to/r2_output
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Defaults are the exact paths used by the Task 1.2 audited Slurm run.
DEFAULT_DATA_DIR = (
    "/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/"
    "data/reaction_enzyme_microbe_training_clean_2026-06-01_LOCAL"
)
DEFAULT_OUTPUT_DIR = (
    "/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/"
    "outputs/r2_esmc_hardneg1_stage1_25_2026-06-11"
)

# ── Import from train.py ──
_script_dir = Path(__file__).resolve().parent
_candidate_import_dirs = [
    _script_dir,
    Path.cwd(),
    _script_dir.parents[2] if len(_script_dir.parents) > 2 else _script_dir,
]
for _path in _candidate_import_dirs:
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from train import (
    Config,
    UnifiedSpace,
    MultiModalDataset,
    load_enzyme_cage_300,
    evaluate_multimodal,
    evaluate_grouped_re,
)


def to_jsonable(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def parse_args():
    parser = argparse.ArgumentParser(
        description="Postmortem: evaluate R2 stage checkpoints")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "0", "1", "2", "3"],
                        help="Which stage(s) to evaluate")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Training data directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="R2 output directory (contains checkpoints)")
    parser.add_argument("--chunk_size", type=int, default=4096,
                        help="Chunk size for chunked retrieval")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    chunk_size = args.chunk_size

    # ── Determine which stages to evaluate ──
    if args.stage == "all":
        stages = [0, 1, 2, 3]
    else:
        stages = [int(args.stage)]

    # ── Verify checkpoint files exist ──
    for s in stages:
        ckpt_path = out_dir / f"model_v3_stage{s}.pt"
        if not ckpt_path.exists():
            print(f"ERROR: Checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
        print(f"  Checkpoint found: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

    # ── One-time data loading ──
    print(f"\n=== Loading data (once for all stages) ===")
    t0 = time.time()
    result = load_enzyme_cage_300(str(data_dir), test_size=0,
                                   enzyme_feature="esmc")
    r_feat, e_feat, s_feat, m_feat, ec_labels, metadata, concept_targets = result
    n = len(r_feat)
    print(f"  Total samples: {n}")
    print(f"  Dims: reaction={r_feat.shape[1]}, enzyme={e_feat.shape[1]}, "
          f"substrate={s_feat.shape[1]}, microbe={m_feat.shape[1]}")

    # Build dataset and eval loader
    dataset = MultiModalDataset(
        r_feat, e_feat, s_feat, m_feat, ec_labels,
        concept_targets,
        [m.get("assembly_accession", "") for m in metadata])
    eval_loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # ── Load R2 final metrics for stage 3 consistency check ──
    metrics_path = out_dir / "metrics_v3.json"
    r2_final = None
    if metrics_path.exists():
        with open(metrics_path) as f:
            r2_final = json.load(f)
        print(f"  R2 final metrics loaded from {metrics_path}")
    else:
        print(f"  WARNING: metrics_v3.json not found, skipping consistency check")

    # ── Evaluate each stage ──
    print(f"\n=== Evaluating {len(stages)} checkpoint(s) ===")
    all_results = {}

    for stage_idx in stages:
        ckpt_path = out_dir / f"model_v3_stage{stage_idx}.pt"
        print(f"\n--- Stage {stage_idx}: {ckpt_path.name} ---")
        t1 = time.time()

        # Load checkpoint
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        cfg = Config()  # Use default config (matches R2 training)

        # Create model
        model = UnifiedSpace(
            reaction_dim=r_feat.shape[1],
            enzyme_dim=e_feat.shape[1],
            substrate_dim=s_feat.shape[1],
            microbe_dim=m_feat.shape[1],
            cfg=cfg,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

        # Row-level evaluation (chunked)
        print(f"  Running row-level evaluation (chunk_size={chunk_size})...")
        row_results, all_r, all_e, all_s, all_m = evaluate_multimodal(
            model, eval_loader, device="cpu", chunk_size=chunk_size)

        # Grouped R→E evaluation (chunked)
        print(f"  Running grouped R→E evaluation (chunk_size={chunk_size})...")
        grouped_results = evaluate_grouped_re(
            all_r, all_e, metadata, chunk_size=chunk_size)

        elapsed = time.time() - t1
        print(f"  Stage {stage_idx} evaluated in {elapsed:.1f}s")

        # Extract core metrics
        stage_metrics = {
            "row_RE_MRR": row_results.get("R→E_MRR", None),
            "UniProt_grouped_RE_MRR": grouped_results.get("UniProt-grouped_MRR", None),
            "EC4_grouped_RE_MRR": grouped_results.get("EC-4-grouped_MRR", None),
            "EM_MRR": row_results.get("E→M_MRR", None),
        }

        # Also store full results for completeness
        stage_metrics["row_results"] = row_results
        stage_metrics["grouped_results"] = grouped_results
        stage_metrics["elapsed_seconds"] = elapsed

        all_results[f"stage{stage_idx}"] = stage_metrics

        print(f"  row R→E MRR:          {stage_metrics['row_RE_MRR']:.6f}")
        print(f"  UniProt-grouped MRR:  {stage_metrics['UniProt_grouped_RE_MRR']:.6f}")
        print(f"  EC-4-grouped MRR:     {stage_metrics['EC4_grouped_RE_MRR']:.6f}")
        print(f"  E→M MRR:              {stage_metrics['EM_MRR']:.6f}")

    # ── Stage 3 consistency check ──
    if 3 in stages and r2_final is not None:
        print(f"\n=== Stage 3 Consistency Check ===")
        s3 = all_results["stage3"]
        targets = {
            "row_RE_MRR": r2_final["R→E_MRR"],
            "UniProt_grouped_RE_MRR": r2_final["grouped_re"]["UniProt-grouped_MRR"],
            "EC4_grouped_RE_MRR": r2_final["grouped_re"]["EC-4-grouped_MRR"],
            "EM_MRR": r2_final["E→M_MRR"],
        }
        consistency = {}
        all_pass = True
        for key in targets:
            actual = s3[key]
            expected = targets[key]
            if expected != 0:
                rel_err = abs(actual - expected) / abs(expected)
            else:
                rel_err = abs(actual - expected)
            passed = rel_err < 0.05  # 5% tolerance
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"  {key}: actual={actual:.6f}, expected={expected:.6f}, "
                  f"rel_err={rel_err:.4f} [{status}]")
            consistency[key] = {
                "actual": actual,
                "expected": expected,
                "relative_error": rel_err,
                "status": status,
            }
        s3["consistency_check"] = consistency
        s3["consistency_all_pass"] = all_pass
        if all_pass:
            print("  Consistency check: ALL PASS")
        else:
            print("  Consistency check: SOME CHECKS DID NOT PASS; inspect before writing conclusions")

    # ── Print summary table ──
    print(f"\n{'='*80}")
    print("Stage-Wise MRR Evolution Table")
    print(f"{'='*80}")
    print(f"| {'Stage':<8} | {'row R→E MRR':>14} | {'UniProt MRR':>14} "
          f"| {'EC-4 MRR':>14} | {'E→M MRR':>14} |")
    print(f"|{'-'*10}|{'-'*16}|{'-'*16}|{'-'*16}|{'-'*16}|")
    for stage_idx in stages:
        s = all_results[f"stage{stage_idx}"]
        print(f"| stage{stage_idx:<4} | {s['row_RE_MRR']:>14.6f} "
              f"| {s['UniProt_grouped_RE_MRR']:>14.6f} "
              f"| {s['EC4_grouped_RE_MRR']:>14.6f} "
              f"| {s['EM_MRR']:>14.6f} |")
    print(f"{'='*80}")

    # ── Save JSON ──
    json_path = out_dir / "postmortem_stage_eval.json"
    with open(json_path, "w") as f:
        json.dump(to_jsonable(all_results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()

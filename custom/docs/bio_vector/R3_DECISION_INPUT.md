# R3 Decision Input

Date: 2026-06-14

Scope: one-page decision input for teacher review. This is not an experiment
protocol.

Provenance: prepared locally from the completed R2 postmortem report and the
revised R2 result summary. No new computation or external job was performed for
this document.

## Evidence From R2 Postmortem

| Diagnostic | Key evidence | Implication |
|---|---|---|
| Data ceiling | EC-4 mean rows/group = 50.65; EC-4 row-level top-1 ceiling ~= 0.019742; R2 row-level top-1 = 0.017623 | Exact row retrieval is ceiling-limited |
| Stage-wise evaluation | stage1 EC-4-grouped R->E MRR = 0.922020; stage3 = 0.917949 | Most EC-4 alignment is already present by stage1 |
| Attribution check | R1 stage checkpoints are absent | Do not assign separate numeric contributions to hard-neg removal vs stage1 extension |
| Functional preservation | R2 EC-4-grouped R->E MRR = 0.913213; E->M MRR = 0.619546 | Functional retrieval and enzyme-to-microbe evidence are preserved |
| Tool baselines | ECE = 0.106888; in-dist score mean = 0.944912; OOD-like proxy mean = 0.222236; latency p95 = 5.2403 ms | Baselines are available for future agent integration |

## Decision Table

| Diagnostic signal | Candidate direction | Triggered by current data? |
|---|---|---|
| Exact row retrieval is limited by same-EC-4 row multiplicity | Use EC-4-grouped R->E as the project-facing retrieval target | Yes |
| Stage1 already establishes most EC-4 alignment | Consider simplifying later stages if teacher approves | Yes |
| EC-4 labels are highly imbalanced, max/min = 2291 | Consider EC-4 class-balanced sampling or loss design | Yes |
| Calibration / OOD-like / latency baselines are recorded | Reserve confidence, abstention, and latency acceptance criteria for agent integration | Yes |

## Suggested Teacher Choice

The current evidence supports Path B: tool/verifier mode with R->EC-4 retrieval
plus candidate enzyme/microbe evidence.

The two most directly triggered directions are:

1. Make EC-4-grouped R->E the project-facing retrieval objective, with row-level
   R->E retained only as a reference diagnostic.
2. Decide whether the next experiment should simplify later stages or address
   EC-4 class imbalance, since stage1 already captures most EC-4 alignment and
   EC-4 group sizes are extremely imbalanced.

Rationale:

1. EC-4 mean rows/group is 50.65, giving a row-level top-1 ceiling of
   approximately 0.019742; R2 row-level top-1 is already 0.017623. This makes
   exact-row retrieval a ceiling-aware diagnostic rather than the main
   project-facing retrieval target.
2. Stage1 EC-4-grouped R->E MRR is 0.922020, while stage3 is 0.917949 and the
   saved R2 final value is 0.913213. EC-4 group sizes are also imbalanced
   (max/min = 2291), so the teacher decision should choose between simplifying
   later stages and addressing class imbalance.

This document intentionally avoids implementation-configuration details. Those
should be written only after the teacher selects a direction.

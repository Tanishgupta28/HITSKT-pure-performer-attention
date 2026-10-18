# Reproducibility and response-timing diagnosis

Updated: 2026-10-18 UTC. SAFE DIAGNOSIS COMPLETE; CUDA root cause unresolved.
Base audit:5625478; preserve its reports, evidence, scripts, tests and all375
captured originals. New evidence: `reports/reproducibility_diagnostics/`.
Local prediction scratch: `.inspection/reproducibility_diagnostics/`.

Latest user request authorizes safe verification/diagnosis and commit/push of
three new reports: `reports/HITSKT_REPRODUCIBILITY_ROOT_CAUSE.md`,
`reports/HITSKT_RESPONSE_TIMING_ANALYSIS.md`,
`reports/HITSKT_RESEARCH_DECISION_REPORT.md`. No full training, no protocol
replacement, no original overwrites and no LightSAKT2 modifications/signals.
Current model remains deterministic causal ELU+1; genuine FAVOR+ proposal only.

GPU decision: only visible H100 3g.40gb MIG is occupied (~1792MiB); external
LightSAKT2 reports an active job and process visibility is restricted. Do not
allocate there. CUDA replay remains NOT VERIFIED unless an idle independent
partition is established. Never stop another job to get access.

All diagnostic jobs are terminal; do not restart their recorded session IDs.
CPU inference was serial, affinity216–219, nice19, idle I/O, two threads,
16GiB forward estimate with24GiB RSS headroom, actual90GiB cgroup limit.
The preexisting OOM-kill counter remained2; no new OOM kill observed.
Evidence: `reports/reproducibility_diagnostics/resource_final.json`.

Established phase1 evidence:
- Historical CUDA per-target logits/probabilities were not persisted. Exact
  CUDA/CPU changed-classification counts and max pointwise errors are not
  identifiable from aggregate metrics. Do not equate net accuracy counts to flips.
- Unique historical confusion matrices recovered from rounded metrics and current
  label populations. Confusion-only lower bounds:5 ASSIST,31 Junyi,40 EdNet
  changed classifications; net correct deltas are−5,+19,+30 respectively.
- Recomputing sigmoid in FP64 from all cached FP32 logits changes0/0/4 threshold
  classifications (ASSIST/Junyi/EdNet); EdNet has4 rounded probabilities equal0.5.
  This is a metric-arithmetic diagnostic, not a historical CUDA replay.
- ASSIST CPU batch-FP32 vs FP64 logit BCE differs1.490384e-9; historical CUDA
  vs CPU differs2.130695e-6. Metric accumulation alone cannot explain the failure.
- All acceptance comparisons retain absolute1e-6, declared in the first audit.

Completed new evidence:
- Full ASSIST FP64 forward on112,252 targets changes0 classes versus CPU32;
  max logit delta1.487006e-6, probability delta3.105441e-7. Stable BCE becomes
  0.6087236104891285, still2.131244e-6 from historical. Accumulation/CPU precision
  do not resolve the discrepancy. Positional templates still follow source FP32.
- Junyi/EdNet FP64 samples cover4,072/2,666 targets (six selected batches each),
  class changes0/1; not full or random sample reproduction. Repeated CPU32
  forwards are bitwise equal in probed batches; threads/MKLDNN affect arithmetic.
- Every cached test ID/label/batch order regenerated:98/3,658/24,248 batches;
  target counts112,252/2,456,402/13,429,870; best epochs40/32/30 hashes unchanged.
- Current NVIDIA build reports matmul precision high and TF32 allowed, but
  historical actual flags are absent. Do not assert TF32 caused the discrepancy.
- Authoritative ASSIST2017 dictionary specifies problem start/end and step
  duration, not correctness release. General glossary distinguishes first
  responses/scaffolding. Exact author transformation is not verified.
- EdNet docs distinguish presentation, revisable choices and final submission;
  KT1 duration is only a proxy. Junyi primary file access/mapping not verified;
  every14,660,217 author CSV row has start==end, so zero overlaps certify nothing.
- Actual15-session visible-history test overlap counts:7,059 ASSIST;0 Junyi;
  4,089,385 EdNet. EdNet same-solving-episode prior-label targets3,364,799.
  Junyi tied-start targets2,165,594. These are not proven-feedback-leakage counts.
  ASSIST conditional start+timeTaken sensitivity gives41,625; mapping unverified.
- All11 completed baseline pairs still need fresh full test replay. RKT raw
  pair-count reconstruction remains missing. DKT/EdNet remains incomplete;
  five last9 second moments have21,749+inf; best1 is finite. Finite updates
  cannot decay infinite second moments back to finite. First offending step is
  unknown. No reset/resume. Archived transport-switch epoch13/15 is SAKT, not DKT.
- 59 relevant tests PASS (49 existing independent/oracle +10 new count oracles).
  First audit133-test result retained separately. Final preservation PASS:
  375 originals +106 previous audit files, all30 checkpoints unchanged.
- Evidence renderer/hash/link verification PASS;24 original/new prediction
  arrays bound. A report-render tab/escape mismatch was fixed; initial failed
  renderer log preserved separately. No numerical verification failure was hidden.

Outputs: the three requested reports plus
`reports/HITSKT_AVAILABILITY_PROTOCOL_V2_PROPOSAL.md` and
`reports/HITSKT_FAVOR_PLUS_IMPLEMENTATION_PROPOSAL.md`. Neither proposal implemented.
Per-dataset new manifests: `reports/reproducibility_diagnostics/*_diagnostic_manifest.json`.
Summary integrity: `reports/reproducibility_diagnostics/verification_summary.json`.

Decision: recommendation2 for all historical V1 datasets—retain artifacts
provisionally, require more verification, no proven need to retrain yet. All3
CPU metric sets FAIL, all AUCs pass1e-6, CUDA/actual feedback timing NOT VERIFIED.
Real-time/pre-bundle claims require explicit V2; retrain all affected methods
if training contexts change. No publication readiness or significance claim.

GitHub auth/fetch work as Tanishgupta28; before publication HEAD and canonical
remote both5625478. Publish only new diagnostic reports/scripts/tests and memory
to capstone-gpu/main; preserve unrelated root README and SAKT console log.
The commit containing this record is the diagnosis milestone; verify remote
equality after normal push, then follow [NEXT_CHECKPOINT.md](NEXT_CHECKPOINT.md).

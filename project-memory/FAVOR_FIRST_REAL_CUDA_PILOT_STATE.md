# First real FAVOR+ CUDA/pilot — 20 October2026

Branch `research/hitskt-favor-plus-v1`, base1fa0bab. Owner explicitly confirms
existing H100 allocation/transfer; Kubernetes403 alone no longer blocks or needs
administrator re-confirmation. Exact MIG UUID d2fcfd79-720e-5218-82c9-0a132be68920
exposed, LightSAKT2 stopped/held. No unrelated signals or allocator changes.

Evidence: `reports/favor_plus_v1/FIRST_REAL_CUDA_PILOT_RESULTS.md` and
`reports/favor_plus_v1/first_real_cuda_pilot_20261020_v1/` (summary, admission,
release, source snapshots, CPU81-pass logs, checkpoint replay, artifact hashes).
All new experiments under `experiments/hitskt_favor_plus_v1/assist2017/`:

- `cuda_assist_owner_20261020_v1`:4 short cases PASS then resource-gate abort;
  PIDs exited unidentified. Preserve negative evidence. Strict trusted read-only
  NVIDIA queries now distinguished from compute/unknown processes; old identities
  remain NOT VERIFIED. No numerical gate relaxed.
- `cuda_assist_owner_20261020_v2`:18 length/chunk cases through13081 PASS,
  independent gradient/output references, full-model own/future-label/PAD/BOS/EOS,
  exact checkpoint. CPU–CUDA logits2.384e-7, parameter gradients1.192e-7.
  Measured complete harness88.393847s, maximum kernel allocation224.869MiB.
- `real_pilot_prepare_cpu_20261020_owner_v2`: exact earlier five whole train
  examples/11 data identities preserved with final19 source hashes.
- `real_assist_five_updates_20261020_v1`:5 Adam calls/5 returned,292 targets,
  features128/seed42/B1/cost2048, original ASSIST architecture/Adam LR5e-5.
  Losses1.1045346260,0.8108273149,0.7359604239,0.7652230859,0.6592436433.
  All gradient/model/Adam states finite and parameters change each step.
  Measured execution5.332492s,287.890MiB allocated/368MiB reserved, host RSS1.846GiB.
  Fresh CPU payload verification and exact CUDA logit/Adam/RNG reload PASS;
  all61 Adam counters5. No extra forward/optimizer in fresh CPU verifier.

Pilot checkpoint SHA256:
`56f1511233cdc89b694bdf73abee6a80c9c214ba7db62fb440ec20d193107ec7`.
Checkpoint binaries remain local/ignored; configs, JSON, bounded NPZs, logs,
source snapshots and hash inventory committed. Do not mistake Git-only checkout
for a backup containing these model binaries.

Scientific decision: execution PASS, approximation REVIEW_REQUIRED. Fixed-prefix
CorrectPaddingEncoder relative L2 vs exact softmax0.181169→0.236181, crossing0.20
after update1. Max p95 norm growth2.386%; median ESS>2, but pairwise concentration
persists. Independent feature-kernel error<=5.33e-7 passes. Keep128 diagnostic
configuration, no architecture/count change or test tuning. No validation/test,
AUC/accuracy, convergence or full-training readiness established.

Final release: no GPU handle owners, memory baseline39,780,352 bytes, Light hold
active/OOM counter unchanged. All original ELU+1/data/reports/Light/DKT untouched.
Full229 CPU suite not rerun;25 new+56 relevant old CPU cases=81 PASS,1 deselected.
Complete scoped publication, then wait for explicit next-stage approval. Larger
ASSIST pilot is justifiable only as a bounded diagnostic study after approximation
review; original15-history/B64/cost32768 and Junyi/EdNet shapes remain unmeasured.

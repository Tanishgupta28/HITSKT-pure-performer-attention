# Next checkpoint

19 October2026 UTC — stop safely at independent-GPU resource gate.
Branch research/hitskt-favor-plus-v1, continuation based7399a0. Read
[FAVOR_GPU_PILOT_STATE.md](FAVOR_GPU_PILOT_STATE.md) and latest
[USER_DIRECTIVES.md](USER_DIRECTIVES.md). Previous scientific checkpoint:
[NEXT_CHECKPOINT_history_before_favor_gpu_pilot.md](NEXT_CHECKPOINT_history_before_favor_gpu_pilot.md).

Completed: fresh read-only resource/process inspection, required report
reports/favor_plus_v1/GPU_PILOT_AND_TRAINING_READINESS.md and evidence under
gpu_pilot_readiness_20261019_v1/. LightSAKT2 active (1 running job, live GPU
handles); only existing H100 MIG visible, no independent exclusive ownership.
No PyTorch/CUDA initialization/probe, CUDA checks or ASSIST optimization.
375 originals +407 prior files preserved; prior229 CPU tests unchanged.
No model/training source/data/checkpoint/results changes. Preserve unrelated
root README and SAKT log. Report/evidence/memory publication is research-only;
on resume reconcile branch tip against capstone-gpu, never merge main.

Next action: wait for externally assigned independent/exclusive GPU or MIG.
No watcher/repeated background polling. Need allocator/provider owner/job,
UUID, external exclusivity evidence, expiry/window, device exposed to container
and CUDA_VISIBLE_DEVICES exactly matching. Budget8GiB test cap plus overhead;
16GiB+ or separate40GiB MIG is a planning choice, not measured minimum.
Do not repurpose or repartition current protected MIG.

Conditional authorization persists: once actual allocation exists, execute
staged128-feature harness with explicit assist2017 profile, short-to-long,
stop on any correctness/nonfinite/resource failure. Only after required CUDA
PASS, run at most5 ASSIST2017 training updates, seed42/B1/whole sessions/original
history and architecture/cost2048, no test evaluation. Implement/review actual
step-bound real-data runner then; current pilot CLI is synthetic. Record loss,
gradient/QK norms, feature concentration, bounded exact-softmax comparison,
GPU peak/runtime/config/data/checkpoint hashes/reload integrity. No re-approval
needed for this already authorized conditional scope. No full training on
any dataset or conclusion about convergence/AUC from5 steps.

No LightSAKT2 signals/edits, DKT/EdNet resume/reset, architecture/protocol change,
historical results overwrite, hyperparameter sweep or automatic experiment queue.

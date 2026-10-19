# FAVOR+ resource diagnosis and planning checkpoint

19 October2026 UTC, base025a0e932e563c517d59d6b787f43f01a756b2c0,
research/hitskt-favor-plus-v1. Supplied full SHA had a transcription error;
actual branch/remote tip agrees with previous handoff. Latest request is
documentation-only diagnosis, not authorization to launch GPU experiments.

Frozen observation15:13 UTC: LightSAKT2 active,811 completed/1 running/145
pending logical entries,957 total.773 independent completed training runs;
139 training entries remain including active, plus7 reuse/evaluation entries.
Current ASSIST2015-GKT-baseline_tuning-42-d998e3d1a034 completed epoch6;
health recent mean608.26seconds, no stall/failure. PID2325921 trains;
2404132 is pending on GPU admission, not stalled. Existing Light scheduler,
coordinator/recovery/observers remain untouched.

Measured-reference occupied-MIG training ETA255.31h/10.64continuous days,
conditional30 October06:31UTC; scenario51.09–890.34h is not a confidence
interval/guarantee. Final tests/checkpoint inference and analyses still follow
training (inference estimate1.04GPUh plus unknown overhead). No release promise
or administrator schedule/reassignment is known. Other outside jobs unknown.

Only one parent H100 and its single exposed3g.40gb MIG; protected existing
UUID MIG-d2fcfd79-720e-5218-82c9-0a132be68920. Live NVIDIA handles and recent
training progress; restricted NVML usage/PID visibility. No independent
HiTSKT allocation/accessible alternative compute server established; no SSH
route/scheduler allocation tool. Free VRAM never establishes ownership.
Request independent exclusive GPU/MIG UUID/owner/job/external assignment/
validity window/device exposure. Separate40GB MIG is a reasonable bounded
candidate;16GiB+ GPU and16GiB host are planning allowances, not measured minima.
Current shared cgroup90GiB,33.80GiB used, historicalOOM2; no new OOM found.

Core FAVOR+ CPU implementation/pipelines retain229-pass evidence; no unresolved
attention defect observed in tested scope.128 features stays first diagnostic
choice. CUDA/max-batch memory/learned-QK approximation unverified. Necessary
remaining CPU preparation: explicit real-data <=5-update ASSIST runner with
original store/collator/history/seed42/B1/cost2048 and requested diagnostics;
existing pilot script is synthetic. Stage1 CUDA harness prepared, stage2 not
turnkey. No code change, repeat229 tests or redundant dataset audit in this turn.
No genuine FAVOR+ real-data optimizer step/AUC exists yet. Full-training GPU
runtime/VRAM/RAM peaks NOT ESTIMABLE from CPU measurements.

Report reports/favor_plus_v1/RESOURCE_STATUS_AND_TRAINING_PLAN_20261019.md;
immutable sanitized evidence resource_status_20261019_v1/observations.json
under that reports directory. Includes six-stage dependencies, component memory
floors/actual target counts and unknowns. Stage3 larger validation pilot needs
separate approval; all full training remains unauthorized. Historical V1 timing
limitation unchanged. All code/data/checkpoints/results/prior reports untouched;
unrelated README/SAKT log preserved. No CUDA/PyTorch import, GPU probe/training,
signals/watchers/allocations, LightSAKT2 edits or DKT resume. Only new docs/memory.

Finish documentation validation/scoped research-only commit/push, then stop.
No automatic resource rechecks. Earlier conditional execution authorization is
recorded historically; it does not activate execution in this diagnostic turn.
Next useful work is the small runner preparation and actual exclusive allocation,
followed by staged CUDA verification and then <=5 real ASSIST updates when the
execution scope/gates apply. Do not ask again for already-authorized conditional
scope, but do not infer execution direction from this diagnostic-only request.

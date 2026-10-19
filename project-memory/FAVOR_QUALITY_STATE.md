# FAVOR+ numerical quality and GPU readiness

19 October2026 UTC; research/hitskt-favor-plus-v1 from31886f6. Frozen design
commit49ed057. Main report reports/favor_plus_v1/NUMERICAL_QUALITY_AND_GPU_READINESS.md;
evidence reports/favor_plus_v1/numerical_quality_v1/. Implementation baseline:
[FAVOR_PLUS_STATE.md](FAVOR_PLUS_STATE.md). No production/default changes.

Established CPU evidence:
-229 tests PASS,0 failures/skips (217 prior +12 new).12 preexisting transport
  worker/one-core warnings. Full suite log/XML and VERIFICATION_SUMMARY.json.
-288 synthetic +576 frozen sample feature-kernel comparisons PASS. Independent
  NumPy/SciPy oracle max output delta7.994e-14;24 double-gradient cases max
  3.775e-14.32-seed Gaussian-marginal/chi/orthogonal distribution controls PASS.
-54 float32/float64-accumulation controls against independent64 output/gradients
  PASS. Default max output3.8221e-5, gradient5.3777e-6 within predeclared gates.
-High-norm softmax error persists (scale3 MSE1.24159/1.23341/1.20444 for128/256/
  512). Feature ESS~1.12/1.18/1.23; rare lognormal tails explain sampling error.
  No production defect observed in tested scope; finite precision is not its fix.
-4 whole training targets per dataset, distinct students, all histories train/
  same student, <=256 per session and<=2048 padded slots, original15 history.
  12 targets/936 actions,0 optimizer steps; no validation/test responses for
  feature choice. Actual vocabularies/profiles. Initial seed42 FAVOR contexts
  and separately labeled trained ELU stress controls. Never transplanted weights.
  Frozen prefixes<=32; changing m in a hierarchy could alter later Q/K.
-Initial raw Q p95~4.74..5.86; trained ELU controls up to57.48, not learned
  FAVOR evidence. High-norm control errors remain poor at512.

Decision follows frozen rule: retain128 for first bounded diagnostic pilot.
All counts meet initial-context median<=.10/p90<=.20 targets,128 cheapest
(median.096761,p90.158938).256/512 improve mean MSE29.47/44.54%, long CPU wall
1.25/1.75x128. These provisional targets do not approve full training or imply
uniform kernel/weight fidelity. No defaults changed or test AUC accessed.
B1/H2/d64/L3925/chunk32 peak RSS0.703/0.743/0.845GB and fwd+bwd1.949/2.444/
3.407s. No maximum-length512 or flattened-history batch/GPU memory certification.

GPU PENDING: only existing H100 MIG visible, usage permission restricted,
no scheduler/external exclusive allocation; protected LightSAKT2 has1 running
job. Every Torch CPU job used CUDA-hidden execution; no CUDA context or tensors
were created by this task.
New scripts/validate_favor_plus_gpu.py prepares staged output/gradient/causal/
PAD/shifting/full-model/checkpoint/resource checks, exact UUID + external lease
metadata gate. Admission/CPU helpers tested; CUDA execution NOT VERIFIED.
No metadata assertion may substitute for actual independent allocation.

All3 original store/profile/count plans PASS. Added synthetic validation-peak
control selects epoch2 vs last3 and reloads2; no real training. Original
chronological within-student retrospective teacher-forced V1, response timing
limitation and original loss/metrics remain unchanged. No availability V2.
One failed diagnostic selector attempt omitted hierarchy EOS cost; guard stopped
before sampled forward. Original cost helper reused; same bound, fresh v2 run.
Failure log/JSON/failed new directory preserved; no production defect hidden.

Preservation PASS375 originals +341 prior source/report/manifest/local-smoke
files, including existing FAVOR implementation and original checkpoints.
No LightSAKT2/DKT changes or signals, architecture changes, historical result
changes, full training or main merge. Cgroup90GiB, OOM count2 unchanged.
Publish scoped diagnostics/report/memory research branch only, verify remote
then wait for independent GPU. A small dataset pilot is conditional on CUDA
checks passing and a bounded execution decision; current resource cannot be used.
See [NEXT_CHECKPOINT.md](NEXT_CHECKPOINT.md).

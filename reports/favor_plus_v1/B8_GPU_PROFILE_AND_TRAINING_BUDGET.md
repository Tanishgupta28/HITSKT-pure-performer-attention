# FAVOR+128 versus FAVOR+512: batch-8 GPU profile and training budget

25 October 2026. Research branch `research/hitskt-favor-plus-v1`; base
`8554b588df84691788d3e151021a823f20934a90`.

**The approved zero-update GPU profile PASSED for both configurations in
215.663 seconds. No training was launched. The proposed two-hour budget does
not support eight complete epochs per arm.** Measured screening estimates are
4.90 hours for the pair, or 6.13 hours including the prospectively declared 25%
planning reserve. These estimates exclude actual Adam-step arithmetic and are
not completion guarantees. A seven-hour pair / four-hour per-arm reservation is
a new proposal requiring separate approval.

The existing native FAVOR+128 best ASSIST2017 full-validation AUC
**0.703466903** at epoch 7, its checkpoints, and all historical research results
remain unchanged. This profiling experiment produced no new trained AUC/F1.

## Evidence and execution scope

- [Design, fixed before CUDA](b8_profile_20261025_v1/RUNTIME_PROFILE_DESIGN.json)
  and [actual profile-only authorization](b8_profile_20261025_v1/PROFILE_EXECUTION_RECORD.json).
- [Four-stage completion](b8_profile_20261025_v1/PROFILE_SEQUENCE_COMPLETE.json),
  [individual timings](b8_profile_20261025_v1/STAGE_TIMINGS.json), and
  [measured summary](b8_profile_20261025_v1/MEASURED_BUDGET_SUMMARY.json).
- Local run: `experiments/hitskt_favor_plus_v1/assist2017/assist2017_orf128_512_pair_b8_20261025_v2/`.
  Plan SHA256: `cb3bbc9168b712d187a0b2e998c01816b204e2683f74d2e3a01bc5cf0d866edb`.
- [Published evidence inventory](b8_profile_20261025_v1/RAW_EVIDENCE_MANIFEST.json)
  identifies the metadata archive, representative frozen arrays, and local-only
  arrays/binaries by hash. Binary I/O proxies are explicitly untrained objects,
  not resumable training checkpoints.

The finite sequence was full-shape128, full-shape512, timing128, timing512.
All stages shared one 600-second authorization clock, stopped on first failure,
and used no automatic retries, resizing, optimizer steps or resource increases.
Worker wall times were 21.412, 26.215, 77.418 and 90.473 seconds, respectively.

Resource: authorized shared H100 80GB HBM3, MIG 3g.40gb,
`MIG-d2fcfd79-720e-5218-82c9-0a132be68920`. Runtime was Python 3.10.12,
PyTorch `2.4.0a0+f70bd71a48.nv24.06`, CUDA 12.5, NumPy 1.24.4, one CPU
thread, CPU affinity 223, nice 10, TF32 disabled, deterministic algorithms and
highest float32 matmul precision. Full runtime receipts are in the archive.

The existing shared policy was preserved: 8 GiB own allocator cap, 10 GiB
advisory total budget, 16 GiB free at admission, 6 GiB free during execution,
4 GiB own host-RSS limit, 2 GiB shared-host headroom floor, 0.1-second pacing,
and existing CUDA/resource/numerical stops. Actual container memory limit was
90 GiB; after-exit headroom was approximately 20.15 GiB. No LightSAKT2 process
was observed and its hold file was unchanged. Other processes were not signaled.

Host/container GPU PID attribution remains **UNRESOLVED**. MIG utilization was
unsupported by NVML (status 3), so utilization is **UNKNOWN**, not zero. The
authorized shared policy used its existing cap/headroom/pacing fallback.
This is not an exclusive-ownership claim. The single
[after-exit observation](b8_profile_20261025_v1/AFTER_EXIT_RESOURCE_SNAPSHOT.json)
found no compute workers and 39.337952 GiB device-free memory; that observation
does not reserve the device for future work. No watcher was started.

## Full shapes and measured memory

Each arm passed nine actual-data forward/backward cases covering the small
proof example, low/median/high padded cost, extreme history/action/workspace
shapes, and the highest-cost validation batch. The timing profile then used 32
fixed equal-frequency training-cost strata covering all eight schedules, eight
validation strata, two measurements per representative, one warm-up, fixed
training-only attention diagnostics and two I/O proxies per arm.

| Measured quantity | FAVOR+128 | FAVOR+512 |
|---|---:|---:|
| Full-shape cases | PASS 9/9 | PASS 9/9 |
| Timed training forward/backward bodies | 64 | 64 |
| Timed validation forward bodies | 16 | 16 |
| Maximum allocated memory, GiB | 2.335866 | 7.674421 |
| Maximum reserved memory, GiB | 4.394531 | 7.988281 |
| Peak own host RSS, GiB | 1.704754 | 1.729572 |
| Minimum sampled device-free memory, GiB | 34.583313 | 31.102844 |
| Weighted forward/backward time, seconds | 0.390832 | 0.541889 |
| Weighted training-body proxy, seconds/batch | 0.941668 | 1.114164 |
| Weighted validation-body proxy, seconds/batch | 0.257595 | 0.291813 |
| Actual optimizer updates | **0** | **0** |

Training-body timings include batch construction/transfers, the existing guard
calls, forward/backward, finite gradient/norm checks, model and moment-sized
buffer checks, metric transfers, and pacing. They exclude actual `Adam.step()`.
There were no CUDA/OOM/nonfinite/resource failures. Model weights and projection
hashes were unchanged before versus after profiling. Two zero-filled buffers
per parameter reserved Adam-moment-sized space. They do not measure actual Adam
state initialization, optimizer kernel workspace or learned-state behavior.

**FAVOR+512 has only 12 MiB between the observed reserved peak and the 8 GiB
cap**, and approximately 333 MiB above its live allocated peak. This is a
conditional memory pass, not proof that every trained step will fit. The first
authorized optimizer steps and complete epoch must pass unchanged guards;
stop on failure. No cap increase or silent data reduction is authorized.
Device-wide used/free telemetry is recorded, but per-process nonallocator
overhead cannot be certified without trusted PID attribution.

## Data preservation and fairness

[Coverage evidence](b8_profile_20261025_v1/FAIRNESS_AND_COVERAGE.json) and
[initialization evidence](b8_profile_20261025_v1/MATCHED_INITIALIZATION_AND_ZERO_TRAINING.json)
passed. The common batch-8 schedule preserves every original training/validation
example, chronological order within each frozen epoch schedule, full history and
action lengths, supervised label alignment and the original chronological
within-student retrospective evaluation protocol. No additional interaction is
discarded; original context-only interactions remain context. No test inference
or test-based selection occurred.

- Each epoch: **7,060 training sessions / 566,107 supervised labels**.
- Full validation: **1,977 sessions / 125,800 labels**; 47,584 positive and
  78,216 negative labels. Event and label order hashes match the contract.
- Eight epoch batch counts: **935, 937, 943, 939, 939, 934, 938, 934**.
  Each arm therefore performs **7,499 optimizer updates** if all epochs finish.
- Batch size is at most 8; the original padded-cost ceiling remains 32,768,
  with observed schedule maximum **32,751**. This is not the earlier cost-2048
  smoke protocol. The longest action sequence is 938 interactions plus EOS.

Both fresh seed-42 arms have identical learned parameter hashes, post-factory
CPU RNG state, data/sampler order, Adam settings, model dimensions, precision,
evaluation definitions and eight-epoch budget. The projection policy is a
common nested 512-feature pool with fixed stage seeds 42–45, using 128 or 512
features. Attention accumulates in float64; neural parameters/output use float32.
Original non-attention dimensions remain width 256, four heads, FF 2048,
dropout 0.1, one layer per hierarchy stage. Adam is lr 0.00005, betas
(0.9, 0.999), epsilon 1e-8, weight decay 0, with no gradient clipping.

This is a fair **new128-versus-new512** comparison. It changes both arms'
minibatch grouping from the historical batch-64 / 1,454-update trajectory;
there is no gradient accumulation. Precision and projection construction also
differ from that historical native128 experiment. Its 0.703467 validation AUC
is useful context, not an isolated feature-count ablation or baseline claim.

## Initial approximation findings

The unchanged review threshold is **0.20 relative attention-output L2**.
The following are six fixed TRAIN examples, native per-model prefixes of at
most 32 positions, from fresh initialization:

| Stage | 128 median / maximum | 128 above .20 | 512 median / maximum | 512 above .20 |
|---|---:|---:|---:|---:|
| ActionEncoder | .202339 / .250014 | **3/6 REVIEW** | .132655 / .181684 | 0/6 |
| CorrectPaddingEncoder | .154539 / .171902 | 0/6 | .057210 / .071775 | 0/6 |
| SessionEncoder | .038576 / .059735 | 0/6 | .031675 / .143333 | 0/6 |
| Decoder | .077915 / .114949 | 0/6 | .061370 / .071465 | 0/6 |

Independent feature-kernel agreement checks passed, including additional frozen
prefixes up to 128 positions and audit draws 17/101. Subsequent hierarchy stages
receive representations produced by their own model, so these native summaries
are not an identical-Q/K experiment at every stage. Initial512 errors are lower
in these cases; whether they remain lower during learning and improve validation
AUC/F1 is **NOT VERIFIED**. Preserve historical trained approximation REVIEW
findings; neither float64 nor this untrained profile resolves learned alignment
with fixed features.

## Eight-epoch budget: two hours is unsupported

The prospective estimator weights each two-repeat median by its stratum's batch
count, adds nine full-validation passes (eight epochs plus selected-checkpoint
replay), 312 checkpoint I/O proxies per arm, eight recovery proxies, nine fixed
diagnostic passes and measured worker-overhead proxies. It then applies the
predeclared 25% planning reserve. I/O objects contain initial weights,
moment-sized zero buffers and placeholder metric prefixes. They roundtrip
exactly but are marked `UNTRAINED_IO_PROXY_NOT_A_CHECKPOINT`, format 0.

| Estimated component, seconds | FAVOR+128 | FAVOR+512 |
|---|---:|---:|
| Eight training schedules, body proxy | 7,061.57 | 8,355.11 |
| Nine full-validation passes | 653.78 | 740.62 |
| Checkpoint I/O | 280.86 | 274.56 |
| Recovery I/O proxy | 7.20 | 7.04 |
| Attention diagnostics | 34.67 | 42.05 |
| Worker-overhead proxy | 93.13 | 105.30 |
| Subtotal | **8,131.20 (2.26 h)** | **9,524.68 (2.65 h)** |
| With 25% planning reserve | **10,164.00 (2.82 h)** | **11,905.85 (3.31 h)** |

Pair subtotal: **4.9044 hours**; with reserve: **6.1305 hours**. Training bodies
alone exceed two hours. These are stratified initial-model measurements, not a
timing bound or a benchmark of all batches/learned states. Actual Adam cost,
future shared load, allocator fragmentation and first-epoch behavior remain
unmeasured. **NO-GO for the proposed two-hour paired experiment.**

## Prepared launch after separate runtime approval

A new [outer coordinator](../../ktbench/favor_plus_pair_long_window.py) and
[CLI](../../scripts/run_favor_plus_pair_long_window.py) propose **7 hours total,
4 hours per arm**, sequential arms. They preserve the profiled plan, numerical
engine, all GPU/host limits and complete eight-epoch schedules. One immutable
parent clock and one immutable clock per arm constrain finite worker leases;
each worker lease is at most one hour and expires before both outer deadlines.
There is no retry or clock reset for the overall campaign. Stages require
prospective time headroom before launch. The old two-hour campaign is unchanged.

The only generated longer-run record is
[an UNAPPROVED template](b8_profile_20261025_v1/SEVEN_HOUR_TRAINING_PROPOSAL_TEMPLATE.json).
`training_approved=false`, `new_parent_runtime_budget_approved=false` and
`approval_is_proposal=true`. It cannot launch training. Actual approval must
specify the new runtime budget; then create a separate source/profile/plan-bound
record with the actual approval evidence and a fresh absolute seven-hour window.
`ACTUAL_APPROVED_TRAINING_SCOPE.json` does **not** exist at publication.

Only after that approval and record preparation, the exact training command is:

```bash
cd /workspace/capstone
CUDA_VISIBLE_DEVICES=MIG-d2fcfd79-720e-5218-82c9-0a132be68920 \
PYTHONPATH=. PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 223 nice -n 10 python scripts/run_favor_plus_pair_long_window.py campaign \
  --plan experiments/hitskt_favor_plus_v1/assist2017/assist2017_orf128_512_pair_b8_20261025_v2/prepared_plan.json \
  --scope reports/favor_plus_v1/b8_profile_20261025_v1/ACTUAL_APPROVED_TRAINING_SCOPE.json \
  --execute-training
```

Resource request: existing authorized shared MIG, unchanged 8 GiB allocator /
10 GiB advisory budget and headroom floors, one CPU thread, 4 GiB own-RSS ceiling,
and at least approximately 60 GiB free storage for about 51 GiB of pair checkpoint
payloads plus logs/curves. Storage was approximately 151 GiB free at inspection,
not reserved. No larger allocation, concurrent arm or extra seed is proposed.

The controller validates the original plan/data/source hashes and both successful
core profiles; guards run again before CUDA and throughout every worker. Each
completed epoch requires fresh-process exact model/Adam/projection/RNG/next-
computation recovery before proceeding. Model selection uses full validation
AUC with strict improvement and fixed threshold .5; the matched protocol is
eight epochs per arm, without performance early stopping. Best checkpoint replay
uses validation only. Record full-validation AUC, F1, BCE and the other existing
metrics, fixed TRAIN diagnostics, wall time and memory each epoch. Stop on
serious numerical/resource failures; do not discard examples to finish a budget.

## Verification and decision

[New CPU tests](b8_profile_20261025_v1/cpu_final_tests.xml): **18 PASS, zero final
failures** (seven profiling/I/O tests and eleven approval/clock/workflow tests).
The prior 66 passes are retained evidence, not claimed as rerun. One initial CPU
coordinator test exposed reuse of an exclusively created progress filename;
it was fixed to retain numbered immutable snapshots. Initial failure/source/log
are preserved. This happened before training; the profiled engine and attention
sources were unchanged. No complete regression suite was repeated unnecessarily.

| Decision | Status |
|---|---|
| Approved ten-minute, zero-update GPU profile | **PASS**, 215.663 s |
| Complete data / matched128-versus512 preparation | **PASS** |
| Both observed batch-8 memory profiles | **PASS**, 512 margin very small |
| Two-hour paired training budget | **NO-GO** |
| New seven-hour / four-hour-per-arm launch path | **PREPARED, UNAPPROVED** |
| Actual trained512 Adam/learned memory behavior | **NOT VERIFIED** |
| New trained epoch AUC/F1/BCE or learned approximation | **NOT RUN** |
| Original scientific claims / final test / superiority | Unchanged; no new claim |

Recommendation: use the prepared matched pair only after approval of the longer
budget, with the first complete epoch as a mandatory resource/numerical/recovery
gate. Inspect learned approximation and full-validation performance across the
same eight epochs. A memory failure or worsening approximation remains evidence
to report, not a reason to silently modify features, thresholds, data or limits.
Final untouched-test evaluation and matched baseline verification follow model
selection under a separately decided research protocol.

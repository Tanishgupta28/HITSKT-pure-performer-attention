# First genuine FAVOR+ CUDA verification and real-data pilot

20 October2026 UTC; branch `research/hitskt-favor-plus-v1`, base `1fa0bab2e3ddb5846d3f8f7905e2df2e25ba85e7` plus the scoped access/diagnostic changes described below. No attention mathematics or original non-attention model/data components changed.

**CUDA correctness PASS. Exactly five real ASSIST2017 Adam updates PASS. Approximation REVIEW REQUIRED. Full training NOT CLEARED and NOT RUN.** No validation/test evaluation, AUC or accuracy was produced. Five updates do not establish convergence or generalization.

## Allocation, device and environment

The user explicitly confirmed that the existing H100 allocation belongs to them/their research project and authorized transfer from LightSAKT2 to HiTSKT. That owner statement is the authority; this report does not invent administrator/scheduler confirmation. See [authorization](first_real_cuda_pilot_20261020_v1/OWNER_AUTHORIZATION.md), [bounded allocation record](first_real_cuda_pilot_20261020_v1/allocation_record.json) and [initial resource admission](first_real_cuda_pilot_20261020_v1/resource_admission.json). The two-hour record window is an operator-imposed execution ceiling, not an independently observed scheduler expiry. Historical Kubernetes403 is a visibility limitation and was not treated as a denial of the owner's allocation.

- Device: NVIDIA H10080GB HBM3, MIG3g.40gb, UUID `MIG-d2fcfd79-720e-5218-82c9-0a132be68920`; parent `GPU-5a568e3c-e96c-169f-b063-f4478340ff51`. Exact UUID exposure and `CUDA_VISIBLE_DEVICES` verified. Initial capacity42,278,584,320 bytes, free42,238,803,968 bytes; no initial GPU-holding processes or known explicit restrictions.
- LightSAKT2 remained stopped with its persistent restart hold active; no signals, restart, submissions or artifact changes. Prior preserved state813 completed/142 pending/2 interrupted, GKT best/latest epoch2. Earlier preservation evidence retained; no redundant full audit. No other workload was stopped or modified.
- Python3.10.12; PyTorch2.4.0a0+f70bd71a48.nv24.06; compiled CUDA12.5; NumPy1.24.4; capability9.0. FP32/highest matmul precision, TF32 off for matmul/cuDNN, deterministic algorithms/cuDNN on, cuDNN benchmark off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, seed42, one CPU dispatch thread/core223 at low priority.
- Bounds retained:8GiB PyTorch allocator quota plus CUDA overhead; kernel timing gate120s; pilot <=5 updates/B1/cost2048,120s/update and900s cooperative execution deadline; pilot process-RSS gate4GiB/shared headroom2GiB. Actual host cgroup limit90GiB; cumulative OOM count remained2 (no new OOM).

## CUDA correctness: final complete run PASS

Fresh run [cuda_assist_owner_20261020_v2](../../experiments/hitskt_favor_plus_v1/assist2017/cuda_assist_owner_20261020_v2/gpu_validation.json) passed all18 combinations of lengths17/65/257/939/3925/13081 and chunks16/32/64 at128 features. All outputs/gradients were finite. Short cases additionally compared against independent NumPy/SciPy causal feature-kernel outputs and an explicit float64 PyTorch feature-kernel gradient reference; larger cases compared with unchanged CPU production outputs/gradients to bound reference memory.

| Check | Observed result | Unchanged acceptance |
| --- | --- | --- |
| Maximum kernel CPU–CUDA output difference |1.78813934326e-07 |atol5e-5 |
| Maximum kernel CPU–CUDA gradient difference |2.32830643654e-10 |atol5e-5/rtol1e-3 |
| Maximum short independent-reference gradient difference |3.82163427179e-10 |atol5e-5/rtol1e-3 |
| Full-model CPU–CUDA logits / parameter gradients |2.38418579102e-07 / 1.19209289551e-07 |output atol5e-5; gradient atol5e-5/rtol1e-3 |
| Causal prefixes / future gradients |Unchanged earlier outputs and exactly zero future gradients |Exact equality / zero |
| Own-label / shifting / batch isolation |Each raw response position0–3 re-collated; own/earlier predictions and other batch row unchanged; future correctness-embedding gradients zero |Exact equality / zero |
| BOS/EOS/PAD and variable lengths |PASS; history/action/target padding, session counts1/2, target lengths4/2, correct metric count6 |Original collator/masks |
| Checkpoint serialization/reload |PASS; exact CUDA logits after reload |Exact equality |

[Full-model causal evidence](../../experiments/hitskt_favor_plus_v1/assist2017/cuda_assist_owner_20261020_v2/full_model_causal_checks.json). This uses the ASSIST256-wide/4-head/FF2048 hierarchy with a small synthetic vocabulary for correctness; real vocabulary is exercised by the separate real-data pilot. Synthetic validation performs no optimizer updates.

Representative synchronized attention forward+backward measurements (B1/H2/d64/m128, chunk32; CUDA timings exclude CPU references and separate causal probes):

| Positions | Seconds | PyTorch peak allocated MiB |
| ---: | ---: | ---: |
|17 |0.016562 |66.921 |
|65 |0.060457 |69.706 |
|257 |0.231299 |71.689 |
|939 |0.842473 |78.715 |
|3925 |3.513399 |110.378 |
|13081 |11.740668 |206.493 |

Maximum allocated across the18 kernel cases: **224.869MiB**. Full-model correctness phase peak145.322MiB. Complete harness execution **88.394s**, including CPU references, probes and CUDA initialization, excluding process imports. These are bounded attention/correctness measurements, not epoch throughput.

## Five-update real ASSIST2017 pilot: execution PASS

Run [real_assist_five_updates_20261020_v1](../../experiments/hitskt_favor_plus_v1/assist2017/real_assist_five_updates_20261020_v1/pilot_manifest.json); [complete configuration](../../experiments/hitskt_favor_plus_v1/assist2017/real_assist_five_updates_20261020_v1/pilot_config.json). The fresh source-bound plan preserves exactly the five genuine examples and11 verified dataset file identities from the earlier preparation. No synthetic data, target trimming, student/split change or label-based selection.292 supervised training actions, target lengths67/60/42/61/62; padded costs363/290/485/91/241; histories1–7 sessions. These contexts are smaller than the allowed cost2048 and original15-history maximum.

Architecture `hitskt_favor_plus_v1`: genuine positive orthogonal random-feature softmax-kernel attention,128 features, fixed saved projections/base seed42, chunk32/FP32 accumulation. Original width256/heads4/FF2048, four single-layer stages, dropout0.1, original preprocessing/within-student splits and rolling history. Model seed42; original Adam LR5e-5/weight decay0 with default betas/epsilon. B1/maxcost2048/max5 updates are explicit pilot exceptions to original maximum B64/token budget32768.

| Update | Training BCE | Gradient L2 | Parameter-update L2 | Train interval s | Targets |
| ---: | ---: | ---: | ---: | ---: | ---: |
|1 |1.1045346260 |5.68989670 |0.11325408 |0.421759 |67 |
|2 |0.8108273149 |4.19505555 |0.09331989 |0.191698 |60 |
|3 |0.7359604239 |2.63604890 |0.06272091 |0.178118 |42 |
|4 |0.7652230859 |2.95778472 |0.06119566 |0.165445 |61 |
|5 |0.6592436433 |2.36124748 |0.04578420 |0.196276 |62 |

All five calls returned, all gradients/model parameters/Adam buffers were finite, and each update changed parameters by a finite nonzero amount. There was no sixth update. Loss need not decrease monotonically; these five losses use different examples and training dropout. The same fixed training example evaluated without dropout had BCE **1.1009649038 initially →0.8768370152 after five updates**. That within-training observation is not validation improvement or convergence.

Sum of synchronized training intervals **1.153296s**; complete measured pilot execution **5.332492s**, including initialization, six fixed-example diagnostic snapshots and checkpoint verification, excluding imports/data preflight. Training intervals end at synchronized Adam completion and exclude subsequent model/optimizer finite checks, Q/K diagnostics and checkpoint work; they include pre-step safety checks. Do not extrapolate full-dataset runtime from these five small contexts.

Peak PyTorch allocation **287.890MiB** (301874176 bytes), reservation **368.000MiB** (385875968 bytes). Host peak RSS **1.846GiB**. Allocator statistics exclude driver/context overhead; total NVML VRAM peak was not continuously sampled. All prepared resource gates passed.

## Learned Q/K and softmax approximation: REVIEW REQUIRED

One fixed whole training example was diagnosed at initialization and after each update. Norms cover all valid tokens/heads, including EOS; independent comparisons use one row and <=32 positions per stage (the session prefix has7). This is24 bounded comparisons over six snapshots, not a representative full-training quality guarantee.

| Stage | Query p95 initial→final | Key p95 initial→final | Relative L2 vs exact softmax initial→final | Median feature ESS initial→final |
| --- | --- | --- | --- | --- |
|ActionEncoder |5.127251→5.127730 |5.274782→5.275733 |0.198626→0.199517 |8.220983→8.247367 |
|SessionEncoder |5.912421→5.861454 |5.275034→5.252788 |0.062279→0.039302 |8.002816→6.636923 |
|CorrectPaddingEncoder |5.406321→5.535305 |5.148601→5.158194 |0.181169→0.236181 |8.888820→7.507681 |
|Decoder |5.157846→5.166042 |5.044726→5.016176 |0.114636→0.102556 |10.711100→10.546236 |

Maximum p95 norm growth over all measured stages/snapshots was **2.386%**, below the prospective2× review trigger. Median feature ESS remained above2 (final stage medians6.64–10.55); some individual pair ESS values remained near1.1, so the features are not uniformly diffuse.

CorrectPaddingEncoder relative error followed **0.181169,0.202546,0.224646,0.226973,0.235106,0.236181**, crossing the predeclared0.20 review trigger after update1. Final max-absolute error versus exact softmax was0.921629. ActionEncoder ended near the trigger (0.199517); SessionEncoder and Decoder errors decreased. This is material approximation error despite modest norm growth. It must not be hidden behind finite losses/gradients or a passing feature-kernel implementation check.

Production attention still matched the independent explicit feature kernel across all24 prefixes: maximum absolute difference **5.32929283925e-07**, below unchanged5e-5. The evidence therefore separates accurate computation of the specified random-feature kernel from imperfect approximation to exact softmax. Prior CPU high-norm rare-tail findings remain relevant; no trained-state high-norm regime was certified by five steps.128 remains a frozen diagnostic starting point, not a validated full-training optimum; no count, architecture, normalization or tolerance was changed in response.

Per-step distributions/errors/ESS/KL/variation, and all24 Q/K/V/projection/mask/output NPZs are saved under the [pilot directory](../../experiments/hitskt_favor_plus_v1/assist2017/real_assist_five_updates_20261020_v1/diagnostics_05.json). [Machine-readable summary](first_real_cuda_pilot_20261020_v1/VERIFICATION_SUMMARY.json).

## Checkpoints, failures and preservation

Pilot checkpoint `pilot_model.pt`,83,798,716 bytes, SHA256 **`56f1511233cdc89b694bdf73abee6a80c9c214ba7db62fb440ec20d193107ec7`**. Tagged format2/model/FAVOR+ configuration, weights, Adam and RNG state serialized exactly; optimizer reload passed and fixed-example CUDA-logit difference was **0**. A separate fresh CPU process then loaded the file with CUDA APIs trapped, checked hash/finiteness and all61 Adam parameter-state step counters=5, without a forward or update. [Fresh-load evidence](first_real_cuda_pilot_20261020_v1/fresh_checkpoint_cpu_verification.json).

CUDA correctness checkpoint SHA256 **`27b319b4e90763f3d225dbafcbb6317e5d2d6407225b1e654ce4f068453fe54e`**. Checkpoint binaries are preserved locally in the separate experiment directories, ignored in Git; source/configuration/metrics/logs/bounded-prefix arrays and their hash inventory are committed. A Git-only checkout does not contain those model binaries. [All70 new experiment artifacts and hashes](first_real_cuda_pilot_20261020_v1/artifact_inventory.json). All19 source hashes in the successful prepared plan still match executed code.

Failures/deviations were preserved, not erased:

1. A pre-CUDA admission rejected PID2730863; it exited before identification, so its workload/owner is unknown. No CUDA work occurred at that rejection. One synchronous re-admission passed after the PID was confirmed absent; no watcher was used.
2. CUDA attempt `cuda_assist_owner_20261020_v1` passed four short cases, then stopped before65/chunk32 because PID2731414 held a device handle. It also exited before identification. That attempt remains **FAIL/incomplete at the resource gate**, not a completed CUDA PASS or confirmed numerical failure. [Failure manifest](../../experiments/hitskt_favor_plus_v1/assist2017/cuda_assist_owner_20261020_v1/gpu_validation_failure.json), [conflict observation](first_real_cuda_pilot_20261020_v1/cuda_boundary_conflict.json). Neither PID can retrospectively be certified as a harmless query or a training conflict.
3. The initial new guard was overly broad: strictly read-only `nvidia-smi` queries also open device handles. The correction permits only trusted executable paths with explicitly read-only arguments; unknown/compute processes and reset/configuration commands remain blocked. It does not prove the identities of the earlier PIDs. Initial sources/logs and both CPU preparation attempts are retained. A fresh source-bound plan/run then passed all required checks. No numerical tolerance or attention equation changed.
4. Targeted CPU verification: initial76 PASS; final **81 PASS (25 new owner-transfer cases +56 relevant existing controls),1 unrelated synthetic selection-training test deselected**. Prior229 CPU results are inherited, not rerun. No failed applied optimizer step was retried/reset; real training ran only once, for exactly five updates.

Scoped code changes are allocation-owner transfer/conflict handling, stronger CUDA independent-gradient/response-shift/padding evidence, and actual-update/finiteness/fixed-example-loss diagnostics in the real runner. Original ELU+1/FAVOR+ attention/model math, historical checkpoints/results, datasets, audit reports, LightSAKT2 and unfinished DKT/EdNet were unchanged. [Final resource release](first_real_cuda_pilot_20261020_v1/resource_release.json) found no GPU-holding processes, the LightSAKT2 hold intact and MIG use back to39,780,352 bytes. No automatic jobs remain from this task.

## Decision and next stage

**A separately approved, bounded ASSIST2017 training/validation pilot is justified as a diagnostic study after review of the approximation trigger. Full training is not cleared.** Keep seed42/m128/original components frozen initially; predeclare update/time/validation budgets and stop/review criteria. Review additional training-only frozen-prefix cases across length/history bins and learned states before accepting the observed0.236 CorrectPaddingEncoder error as scientifically adequate. No feature choice using test performance, silent architecture changes or claim of superior AUC.

Before scaling, measure original15-history contexts and representative long whole sessions; graduate batch shapes while respecting the8GiB initial quota and original maximum B64/token budget32768. The five contexts cost at most485, so even low measured VRAM does not certify original batches. Attention boundary storage scales with flattened historical-session batch, heads, ceil(length/chunk), features and value width; feed-forward activations and Adam add costs. Both larger batches and long sessions need actual shape-admission measurements. Do not multiply the tiny-pilot peak/runtime into a full-epoch forecast.

After that review and explicit approval, proceed to a larger ASSIST pilot, then consider full ASSIST training. Junyi and EdNet-KT1 each need their own profile/long-history/real-data bounded checks and measured memory/throughput before separately approved full runs. Preserve validation-AUC checkpoint selection and independently replay selected checkpoints/predictions/counts/hashes/all seven metrics after authorized full training. V1 remains chronological within-student retrospective teacher-forced evaluation, with its documented response-availability qualification; five updates do not address that limitation or establish publication readiness. No full experiment, hyperparameter sweep, additional seed or main merge was launched.

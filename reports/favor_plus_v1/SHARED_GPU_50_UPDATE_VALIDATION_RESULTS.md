# HiTSKT FAVOR+: shared-GPU continuation through50

23October2026. Branch `research/hitskt-favor-plus-v1`.

**PASS: exactly25 additional real ASSIST2017 optimizer updates,26–50; training
stopped at50. Fresh-process CUDA recovery30/40/50 PASS with zero optimizer calls.
APPROXIMATION REVIEW persists. NO-GO for full training; further bounded diagnostics
require a new instruction. No test-set evaluation or convergence claim.**

|Check|Status|
|---|---|
|Exact25 parent / Adam / RNG / actual projections / scientific configuration|PASS|
|Initial frozen validation25 replay / actual update26 next-gradient proof|PASS, exact|
|Updates26–50 / finite gradients, model and Adam / nonzero parameter updates|PASS,25 returned new calls|
|Atomic checkpoints30/35/40/45/50 / fresh CUDA30/40/50 recovery|PASS|
|Raw-response causal/own-label checks at25/30/40/50|PASS on unchanged checked cases|
|Independent float64 feature-kernel agreement|PASS on bounded captured prefixes|
|ActionEncoder approximation versus exact softmax|REVIEW, median0.212305>0.20;4/6 prefixes exceed|
|CorrectPaddingEncoder approximation|REVIEW: worsening across6/6 cases; median0.192699,2/6 cases>0.20|
|Shared-mode resource gates / CUDA or OOM errors|PASS / none observed|
|Host/container PID mapping / certified own-process total VRAM|NOT VERIFIED / NOT MEASURABLE|
|Final predictive performance / convergence / original large-shape feasibility|NOT VERIFIED|

## Exact experiment and scope

Execution source/design preregistered in `2f1a8ff171e9782e8f5cd42cdea1e70ad6d52223` before CUDA.
New run: `experiments/hitskt_favor_plus_v1/assist2017/assist50_shared_from25_20261023_v1/`.
[Plan](../../experiments/hitskt_favor_plus_v1/assist2017/assist50_shared_from25_20261023_v1/prepared_plan.json) SHA256
`c210ab41bfa0a202f07058f67d3ea2b19e160908a01086901c16ab416a348e67`.
[User scope](shared50_from25_20261023_v1/USER_CONTINUATION_AUTHORIZATION.md),
[predeclared design](shared50_from25_20261023_v1/PREDECLARED_DESIGN.json),
[execution record](shared50_from25_20261023_v1/shared_execution_record.json).

The NEW controller `ktbench/favor_plus_shared25_to50.py` and separate CLI bind
checkpoint25, not20 or zero. The first26 original sample identities match before
extending the unchanged epoch1/seed42 sampler to50 updates plus a probe-only51.
All35 bound sources,11 processed data identities and scientific configuration
validate unchanged. Genuine hierarchical `hitskt_favor_plus_v1`,128 fixed
positive orthogonal random features, same saved projection buffers, chunk32,
FP32 accumulation/highest matmul/TF32off, seed42, original nonattention layers
and Adam (LR5e-5,betas0.9/0.999,eps1e-8,weight decay0,no gradient clipping).
No feature-count/projection/threshold/data-protocol/optimizer tuning occurred.

Batch1/cost2048 and existing six diagnostic training cases/frozen32-session
validation cohort remain unchanged. This is the original chronological
within-student retrospective teacher-forced protocol, **not student-disjoint**.
Sequence-causal checks do not establish actual correctness-feedback availability;
the prior timing qualification remains. No test split was evaluated.

New stage journals count additional calls as `update-25`. The unchanged format2
checkpoint field counts the historical continuation since20 (`update-20`), so
it is30 at checkpoint50; this is not30 new optimizer calls in this campaign.
Only25 real new calls occurred. Probe51 performs forward/backward only, with
zero Adam calls and RNG restored; no optimizer update51 occurred.

## Shared-GPU admission and other workloads

The user explicitly authorizes intentional shared use of
**MIG-d2fcfd79-720e-5218-82c9-0a132be68920**, H1003g.40gb, parent
GPU-5a568e3c-e96c-169f-b063-f4478340ff51,GI2/CI0. This is shared permission,
not exclusive ownership or a scheduler lease. Host/container PID mapping remains
UNKNOWN; no PID alias was assumed and the old exclusive guard was not modified.

The initial read-only observation recorded unattributed NVML PID1706235 using
10.1GiB, with29.230GiB free. It remained visible alongside the new context during
stage30; the external allocation was no longer reported at later admissions.
We did not stop, signal or modify it. Shared mode logs peer/unknown activity and
churn rather than calling unequal PID integers an ownership conflict. MIG
utilization returns unsupported/status3: load is UNKNOWN, not0%. The approved
memory/pacing/latency fallback remained active. Fresh two-snapshot admission
preceded every stage and recovery process; all admissions/resource checks passed.

Unchanged limits: **1GiB own allocated/reserved cap**,2GiB advisory own total
(including driver allowance),8GiB admission/6GiB execution free floors,
4GiB process RSS/2GiB shared-host headroom,0.1s inter-update pacing,
30s/update and45min campaign ceiling. Driver535.161.08, PyTorch
2.4.0a0+f70bd71a48.nv24.06/CUDA12.5, deterministic algorithms/cuDNN,
CUBLAS_WORKSPACE_CONFIG=:4096:8, one CPU thread. Container memory limit90GiB;
inherited OOM count2 unchanged. No budget was increased.

LightSAKT2 had no visible trainer and its active hold SHA256
`65214c2c53852024e970fd0a176f71968f204ec0748635a7e6f46de3ed64c234`
remained unchanged. No LightSAKT2 or other researcher process, scheduler setting,
DKT/EdNet artifact, historical ELU+1 model/result/data/report or main branch was
modified. Shared compute can still contend for bandwidth/SMs; other jobs'
performance was not independently measured or guaranteed unaffected.

## Optimization stability and measured performance

|Stage|New updates|Targets|Target-weighted training BCE|Gradient L2 range|Stage seconds|Synchronized update seconds|
|---|---:|---:|---:|---|---:|---:|
|26–30|5|214|0.741359119|1.867043–5.665378|41.997392|3.367352|
|31–40|10|438|0.707617666|1.824347–6.825563|26.489848|3.327656|
|41–50|10|485|0.712757876|1.602569–14.979638|24.554033|3.177670|

All25 new updates cover1137 training targets; weighted BCE
0.716160914. BCE ranges0.574839–0.931072, gradient L2
1.602569–14.979638 (unchanged stop100); every gradient is present/finite and
every parameter update has nonzero norm (0.020813–0.035689). All61 Adam states
have step50, finite moments/nonnegative second moments (range0–0.009821439).
Overall saved parameter-state delta25→50 L2=0.417119. These varying minibatches
and dropout do not constitute a monotonic learning curve or convergence test.

|Total update|Minibatch training BCE|Same6 training examples eval BCE|Frozen sampled validation BCE|Frozen sampled AUC|
|---:|---:|---:|---:|---:|
|20|0.697781444|0.769759643|0.741282668|0.508313409|
|25|0.639421523|0.750641795|0.727024655|0.510618521|
|30|0.764962077|0.752105278|0.728324073|0.513411746|
|40|0.602148950|0.718446107|0.700401595|0.517046644|
|50|0.729133308|0.706366912|0.692524937|0.519220350|

The validation cohort has32 sessions/29 students/3157 labels,1063positive and
2094negative. **Preliminary sampled results, not full-dataset performance,
validation-selected best-model results or proof of model superiority.**

|Update|Accuracy|Precision|Recall|F1|MSE|
|---:|---:|---:|---:|---:|---:|
|20|0.502692430|0.331785003|0.470366886|0.389105058|0.270515942|
|25|0.522331327|0.337709701|0.435559737|0.380443714|0.263983875|
|30|0.515362686|0.335215243|0.446848542|0.383064516|0.264728247|
|40|0.566043712|0.354502370|0.351834431|0.353163362|0.251510730|
|50|0.580931264|0.357456140|0.306679210|0.330126582|0.247541467|

BCE and AUC improved descriptively25→50, while recall and F1 declined. Always
predicting the majority negative class gives66.329% accuracy on this cohort,
above the model's58.093% at50. AUC0.5192 is weak discrimination, not final evidence
of useful generalization or significance. These correlated sample observations
cannot establish convergence, significance or superiority over any baseline.
Independent rank-AUC/confusion/MSE recalculation from saved scores agrees within
1.12e-16; rounded-probability BCE differs from logit BCE by at most2.49e-8.
Fresh recovery proves fixed logits/next-gradient equality, not a separate full
validation50 replay. Initial full frozen-validation25 replay was exact.

## Approximation and ActionEncoder investigation

Same six training-only cases and≤32-position prefixes; attention errors concern
raw outputs before output projection/dropout, not label probabilities or AUC.
Each checkpoint uses evolving Q/K/V on the same examples, with fixed projections.
Errors below are median complete-prefix relative L2 against exact causal softmax.

|Attention stage|20|25|30|40|50|Cases>0.20 at50|Maximum case50|
|---|---:|---:|---:|---:|---:|---:|---:|
|ActionEncoder|0.213145477|0.212835732|0.212707087|0.212400782|0.212305349|4/6|0.238585440|
|CorrectPaddingEncoder|0.142361690|0.151291355|0.150476632|0.167095642|0.192699293|2/6|0.221018345|
|Decoder|0.072271783|0.079839188|0.079624096|0.087881561|0.100695592|0/6|0.137094932|
|SessionEncoder|0.095313878|0.085124980|0.093301703|0.100499692|0.092155947|0/6|0.143591576|

**ActionEncoder remains REVIEW.** Median25→50 change is−0.000530383 (−0.2492%),
not a meaningful resolution of the0.20 warning. Only2/6 paired cases improve;
4/6 worsen, and the worst case grows0.235302→0.238585. Four cases remain>0.20.
The mean paired error change is positive despite the slightly lower median.
The individual head/query error maximum reaches0.947779; aggregate medians
conceal tails. No prospective threshold was changed or retrospectively relabeled.

**CorrectPaddingEncoder worsens consistently:**6/6 paired cases rise, median
0.151291→0.192699 (+27.37%), maximum0.221018,2/6 prefixes>0.20. Decoder also
rises in6/6 cases while remaining below0.20. These trends need review rather
than interpreting finite optimization as approximation adequacy.

[New independent CPU replay](shared50_from25_20261023_v1/SAVED_PREFIX_REVIEW.json)
recomputes24 saved Action/CPE prefixes25/50 using a direct pairwise positive-
feature formula in NumPy float64, independently of the production prefix scan
and diagnostic helper. It reproduces all recorded relative errors within1e-12.
GPU output versus this feature-kernel oracle differs by at most6.788e-7,
far below5e-5. Thus FP32 rounding is not the proximate explanation in these
cases. Masks/projections remain exactly equal across checkpoints.

Action prefix Q/K/V relative drift maxima are0.827%/1.670%/3.415%, versus
8.704%/8.204%/8.041% for CPE. Maximum matched full-example Q/K p95 growth25→50
is1.001332× in Action and1.022836× in CPE; no norm explosion. CPE representation
drift despite modest norm growth is consistent with changed feature alignment
and concentration under fixed random features; these measurements do not isolate
causal contributions of Q versus K versus V. No random projection was redrawn.
The pre-existing frozen-feature study supports finite-feature sampling/alignment
error rather than an implementation defect; this run does not settle high-norm
or long-history approximation adequacy.

|Stage|Median pair-feature ESS25→50|Minimum individual ESS50|Max Q/K p95 growth25→50|
|---|---:|---:|---:|
|ActionEncoder|8.193583→8.178599|1.016263|1.001332×|
|CorrectPaddingEncoder|9.506263→8.142927|1.141137|1.022836×|
|Decoder|9.334983→9.157769|1.029706|1.001101×|
|SessionEncoder|5.541466→6.397578|1.179580|1.009671×|

Action ESS minimum≈1.016 of128 and CPE median decline9.506→8.143/minimum1.141
retain concentration risk. All unchanged severe criteria passed: max prefix
relative error1.5, stage median0.75, stage median ESS floor1.25, Q/K p95 growth
limit2×, feature-kernel max-absolute5e-5, gradient100 and BCE20. These are separate
containment stops, **not** replacements for the0.20 REVIEW criterion. All raw-
response own/earlier invariant checks on the original checked positions pass.

## GPU/host memory and timing

Own allocator peaks include initialization; no peak reset. Stages and verifiers
ran sequentially, so their peaks must not be summed. Device-wide NVML values
include unattributed peers/driver/system memory and are synchronous sampled maxima.

|Process|Own peak allocated MiB|Own peak reserved MiB|Max sampled global used GiB|Min sampled free GiB|Peak RSS GiB|Execution seconds|
|---|---:|---:|---:|---:|---:|---:|
|stage_030|249.970|340.000|10.802917|28.572083|1.917835|41.997392|
|verify_030|211.086|244.000|0.633484|38.741516|1.709984|3.497122|
|stage_040|269.220|352.000|0.740906|38.634094|1.914909|26.489848|
|verify_040|178.199|194.000|0.584656|38.790344|1.713486|3.084808|
|stage_050|282.330|390.000|0.778015|38.596985|1.916271|24.554033|
|verify_050|189.929|204.000|0.594421|38.780579|1.714622|3.270390|

Overall own peak282.330MiB allocated/390MiB reserved stays below1GiB. Minimum
free28.572GiB, minimum shared-host headroom27.732GiB; no CUDA/OOM/resource failure,
retry or increased budget. The stage30 global peak10.802917GiB includes the
pre-existing10.1GiB external allocation; it does not mean HiTSKT exceeded its
2GiB advisory own total. Global used minus own reserved includes peers (up to
10.502GiB), so it is **not HiTSKT overhead**. The maximum residual after subtracting
each phase's pre-context baseline and own reserved is368.750MiB, still
UNATTRIBUTED driver/context/workspace/peer change. Own process-total VRAM and
unobserved driver transients cannot be certified without trusted attribution.
The2GiB total remains advisory, never a hard process-total guarantee.

25 synchronized update intervals total9.872679s (mean0.394907s,2.532241updates/s,
115.166targets/s). Intervals range0.193451–0.739784s. Stage execution totals
93.041272s; fresh recovery intervals total9.852319s. Stage times include
diagnostics/validation/checkpoint I/O and resource journals; measured execution
intervals exclude imports/plan verification/admission and interstage review.
Shared-load timings are observations, not isolated GPU benchmarks or full-
training forecasts.287 in-process resource observations plus separate admission
snapshots passed. Final read-only release observed no reported compute worker,
37.938MiB global use and inactive LightSAKT2. No watcher/background training remains.

## Checkpoint integrity, tests and preservation

Parent25 SHA256
`e67724a4ade70fe04de9d0c1a1f0fcb8bd1acc984c3506cda5cce2685ab2b5fc`
remains unchanged. Checkpoints use atomic/fsync/no-overwrite publication and CPU
roundtrip validation of weights, Adam, RNG, actual projection buffers, sampler
cursor/identities, source/data/configuration/scoped-contract hashes.

|Checkpoint|SHA256|Fresh CUDA recovery|
|---:|---|---|
|30|`5ec06caf1cd1d53ec81628db4cf02e80be9fd41cabbcbcac3f31a290f5fe3f75`|PASS, zero optimizer calls|
|35|`2b3ebdd235b33bd485dffc907c11065045aa921a1fd5e142be4e31a17876f7dc`|Atomic CPU validation PASS; no separate CUDA replay|
|40|`ce3b35dac4e972723ea90fdd22f7bac48e2a63e5435f3423e771919551a6ca8d`|PASS, zero optimizer calls|
|45|`e6d1a5854cd18169c01364e7f6dc4473e220e6bc3c84daa072d12297f2740c74`|Atomic CPU validation PASS; no separate CUDA replay|
|50|`93ebe9d2a29be9a6e4deb278916513339de61c9ac8f72a454a31111829440a73`|PASS, zero optimizer calls|

Fresh recovery30/40/50 reproduces fixed predictions with max difference0 and
exact next train-mode logits/loss/all-parameter-gradient digest, preserving
model/Adam/RNG/projections. Actual updates26/31/36/41/46 match their preceding
checkpoint's probe. No Adam call occurs in recovery/probes. Checkpoint50 hash
matches terminal marker and recovery receipt. Binaries stay local/ignored; a
Git push is **not** a remote checkpoint or dataset backup.

102 targeted pre-run CPU cases PASS (24new+78existing);175 post-run targeted
regressions PASS, zero failures/skips. Suites overlap and must not be summed as
distinct tests. The preparation handoff incorrectly partitioned102 as27new/75old;
the XML proves24new/78old, with total102 unchanged. Post-run covers FAVOR+ independent feature-kernel/gradients,
causal/own-label/PAD/EOS, full-model/checkpoints/long bounded sequences,
original ELU+1 regression, continuation recovery and shared guards. Historical
229 CPU/18 staged CUDA tests are inherited, not rerun or added to these counts.
[Logs and verification](shared50_from25_20261023_v1/VERIFICATION_SUMMARY.json).

[Preservation](shared50_from25_20261023_v1/PRESERVATION_BEFORE.json) and final
inventory rehash316 prior25 files,199 original20 files and9+8 failed-attempt
files unchanged. Existing unrelated root README/SAKT console-log changes remain
excluded. All35 bound sources unchanged during GPU execution. New evidence is
confined to the FAVOR+ namespace and this research branch; no main merge.

## Decision and scientifically defensible next plan

**GO for engineering correctness/recovery under measured shared conditions.
REVIEW for approximation. Conditional GO only for a separately approved longer
bounded diagnostic; NO-GO for full training now. Training is stopped at50.**

1. Review checkpoint50 train-only frozen representations across broader,
   predeclared length/history/norm strata, retaining these six paired examples.
   Prioritize CPE's consistent worsening and Action's≈0.95 head/query tail.
   Any128/256/512 comparison must be frozen-representation diagnostics, never
   test-AUC feature tuning. Keep128/current projection/scientific protocol for
   this lineage; an attention/configuration change needs a separate variant/run.
2. If the review supports continuation, request a separately bounded extension
   from exact50 with the same Adam/RNG/projection/sampler/recovery contract,
   monitoring at short predefined milestones. Preserve0.20 REVIEW and existing
   severe stops. No continuation is launched by this report.
3. Before full ASSIST2017, profile representative and worst training history/
   batch shapes, forward/backward and optimizer memory under a separately
   authorized finite plan. Original history15/token budget32768/max batch64
   have **not** been admitted by this B1/cost2048 pilot. Do not extrapolate VRAM
   or GPU runtime from these small shapes or CPU tests. If original shapes do
   not fit the current limits, stop and present a measured resource plan; do
   not silently truncate histories, alter architecture or increase the cap.
4. Only after those checks and explicit full-training approval, freeze the
   complete FAVOR+ experiment manifest and preserve the original preprocessing,
   chronological within-student retrospective protocol and original Adam/
   nonattention settings. Use full validation AUC for strict checkpoint
   selection/early stopping (patience5,min_delta0,original dataset ceilings),
   not this sampled cohort or the test set. Keep all diagnostics/version hashes.
5. Reload the validation-selected best checkpoint in a fresh process and
   evaluate the untouched **full test split** once under the unchanged protocol:
   ROC-AUC,accuracy,precision,recall,F1,MSE and logit-derived BCE at threshold0.5.
   Preserve predictions/hashes, independent metric recalculation and runtime/
   CPU-versus-CUDA differences using existing acceptance rules. Only then
   report held-out results with timing/protocol/seed qualifications. Baseline
   comparisons require comparable protocols and independently replayed metrics;
   significance requires additional supporting experiments, not one seed.
6. Junyi/EdNet need their own preserved-data admission, train-only numerical/
   history/batch CUDA profiles and bounded real-data optimization pilots first.
   Their width128/H2/FF1024 and AdamLR5e-5/8e-5 profiles differ from ASSIST's
   width256/H4/FF2048. Do not assume ASSIST's memory/stability transfers or
   invent runtime estimates/full-dataset AUC. Full runs require later approval.

[Independent saved-evidence verification](shared50_from25_20261023_v1/VERIFICATION_SUMMARY.json),
[independent direct prefix replay](shared50_from25_20261023_v1/SAVED_PREFIX_REVIEW.json),
[artifact inventory](shared50_from25_20261023_v1/ARTIFACT_INVENTORY.json), and
[final resource snapshot](shared50_from25_20261023_v1/resource_release.json)
provide the precise evidence. No full training or further optimizer update is authorized
or running. Stop here for the next research decision.

## New training loss/gradient journal

Every row is a genuine training-split update, finite gradients/model/Adam and
nonzero parameter change. Full statistics and resource telemetry remain in the
run's `stage_*/update_*.json` files.

|Update|Training BCE|Gradient L2|Parameter-delta L2|Targets|Padded cost|Synchronized seconds|
|---:|---:|---:|---:|---:|---:|---:|
|26|0.642285943|1.867043|0.020813|41|238|0.683385|
|27|0.827836633|5.005159|0.024052|17|354|0.636663|
|28|0.793104231|3.054032|0.024961|50|262|0.641164|
|29|0.715995669|5.665378|0.023281|51|309|0.666357|
|30|0.764962077|3.555171|0.022921|55|270|0.739784|
|31|0.648875475|2.449704|0.023657|40|302|0.487381|
|32|0.695307434|2.180346|0.023811|57|189|0.262759|
|33|0.678188682|6.279422|0.026095|6|442|0.415600|
|34|0.574838579|3.846731|0.026223|20|282|0.232304|
|35|0.806738973|3.242330|0.024875|32|286|0.312311|
|36|0.691725671|4.303701|0.025862|66|146|0.333235|
|37|0.931071579|6.825563|0.030473|44|310|0.345976|
|38|0.728177428|3.364299|0.030429|62|162|0.272646|
|39|0.670287490|1.824347|0.030201|64|133|0.364823|
|40|0.602148950|2.235819|0.031322|47|113|0.300620|
|41|0.680245042|1.602569|0.027382|60|131|0.337247|
|42|0.619174957|1.997066|0.027239|66|464|0.557361|
|43|0.642295241|3.981988|0.029605|22|396|0.193451|
|44|0.680589974|14.979638|0.035689|1|515|0.210809|
|45|0.697596490|2.088259|0.032326|63|197|0.269480|
|46|0.749237239|2.371104|0.030220|63|127|0.307182|
|47|0.679883063|1.648884|0.027742|67|134|0.466097|
|48|0.823384345|3.510548|0.026608|17|83|0.200541|
|49|0.829515338|4.459737|0.030498|66|125|0.363237|
|50|0.729133308|2.624616|0.031995|60|392|0.272265|

# HiTSKT independent correctness audit

Audit completed: 2026-10-18T12:57:43.007755+00:00. Historical/source base: `656a9f062f228710c7aac1a3f8a499e1ffc301ff`. Repository: [Tanishgupta28/HITSKT-pure-performer-attention](https://github.com/Tanishgupta28/HITSKT-pure-performer-attention).

## Decision and status definitions

**Publication clearance: NOT VERIFIED. Do not present this audit as an unconditional certification of all historical results.** The active model passed independent sequence-index masking/causality checks, and all three accepted full datasets were exhaustively reconstructed. Metric reproduction outcomes and remaining scientific limitations are stated separately below.

**PASS** means the named, scoped check met its stated criterion. **FAIL** means an observed contradiction or numerical discrepancy exceeded that criterion. **NOT VERIFIED** means the necessary evidence or experiment is absent, unavailable, or outside the completed check. A passing test suite or `_SUCCESS` marker alone is never the criterion for overall correctness.

| Dataset | Overall result audit | Raw → processed → store | Actual best checkpoint | Sequence-index causality | Full CPU metrics vs stored, tolerance 1e-6 | Exact historical CUDA replay | Response arrival before prediction |
|---|---|---|---|---|---|---|---|
| assist2017 | FAIL | PASS | PASS | PASS | FAIL | NOT VERIFIED | NOT VERIFIED |
| junyi | FAIL | PASS | PASS | PASS | FAIL | NOT VERIFIED | NOT VERIFIED |
| ednet_kt1 | FAIL | PASS | PASS | PASS | FAIL | NOT VERIFIED | NOT VERIFIED |

All three AUC differences are below 1e-6. Classification-metric differences exceed that gate on every dataset, and ASSIST BCE also exceeds it. The discrepancy cause is unresolved; it does not by itself establish a leakage bug or a need to retrain. The CPU metric gate is deliberately strict and was declared before the first replay. Every exact equality flag and absolute difference is retained; no tolerance was enlarged after seeing the results. The presentation/completion proxy inequalities fail on ASSIST2017 and EdNet; their interpretation requires a separate availability contract.

## 1. Independence, preservation and execution

The audit read original checkpoints/configs/logs and created separate scripts, tests, manifests and local prediction arrays. All 36 inspected original implementation files are byte-identical to the base commit; see [source_inventory.json](independent_correctness_audit/source_inventory.json). Final preservation verification checks the original experiment files against their captured hashes, including all 30 best/last checkpoints; all 15 best checkpoints also match their actual binary Git blobs at the audit base. No architecture, original result, checkpoint, training setting or Git history was rewritten. DKT/EdNet was not resumed.

Model/test computation ran on CPU with GPU visibility disabled; resource-heavy bounded jobs used low CPU priority and idle I/O. LightSAKT2 was not edited, signalled or reconfigured. An audit resource-planning mistake used host RAM rather than the 90 GiB container limit: two **audit-only** jobs exited 137. The three previously observed protected trainer PIDs remained alive afterward. The failed attempts were preserved and rerun with bounded memory; no claim of zero performance contention is made. See [resource_incident.json](independent_correctness_audit/resource_incident.json), [resource_guard.json](independent_correctness_audit/resource_guard.json) and [resource_observation_during_replay.json](independent_correctness_audit/resource_observation_during_replay.json).

The final large replays use fresh CPU workers, the unchanged production model/collator and historical sampler order, ordered result collection, and a shared 24 GiB admission estimate for inference. Ordered completed prefixes from interrupted audit attempts were copied into separate caches and rechecked against every event ID/label, with exact fresh first-three-batch logits. All remaining batches were evaluated; no sampled or partial replay is counted as a complete result. Interrupted attempts are retained with their interruption status and reasons.

Final preservation gate: **PASS, 375 captured original files**, zero mismatches. The final [resource_final_state.json](independent_correctness_audit/resource_final_state.json) retained two protected LightSAKT2 process identities; previously observed PID 1992414 was absent, with exit reason NOT VERIFIED. A read-only external status check still showed active jobs and a current heartbeat. The cgroup OOM count stayed at the two documented audit victims; no protected process was edited/signalled. This observation is not a claim of uninterrupted health for every external job.

A separate fresh-process crosscheck exactly matched logits, probabilities, labels and event IDs for the first three newly evaluated guarded-worker batches after each preserved prefix: [junyi_guarded_worker_direct_crosscheck.json](independent_correctness_audit/junyi_guarded_worker_direct_crosscheck.json) and [ednet_kt1_guarded_worker_direct_crosscheck.json](independent_correctness_audit/ednet_kt1_guarded_worker_direct_crosscheck.json). The parallel evaluator’s initial zero worker-parity field is a default when a resumed prefix skips that branch; the supplemental artifacts contain the actually performed checks.

## 2. Masking and information-flow audit

Threat model: fixed checkpoint weights; change a response at the current or a later **event index**, then compare predictions at earlier/current indices. This is distinct from retraining after changing training data, or proving wall-clock response availability.

| Check | Status | Evidence / criterion |
|---|---|---|
| ActionEncoder history isolation | PASS | [ktbench/models/hitskt.py:42](../ktbench/models/hitskt.py#L42); only earlier complete sessions enter history; full-store history bounds checked. |
| SessionEncoder causal ownership and EOS pooling | PASS | [ktbench/models/hitskt.py:68](../ktbench/models/hitskt.py#L68); per-example owner/order packing, causal blocks, session EOS after actual histories. |
| CorrectPaddingEncoder causality | PASS | [ktbench/models/hitskt.py:93](../ktbench/models/hitskt.py#L93); causal layers read shifted responses. |
| Decoder causality | PASS | [ktbench/models/hitskt.py:109](../ktbench/models/hitskt.py#L109); current question/skill query, memory keys only through aligned current index. |
| Own target label excluded | PASS | [ktbench/batching.py:260](../ktbench/batching.py#L260); BOS followed by previous responses; target_labels never read by model forward. |
| Future raw response perturbations | PASS | [tests/test_independent_hitskt_audit.py:153](../tests/test_independent_hitskt_audit.py#L153); raw suffix labels changed before re-collation, lengths 1/2/7/17, history counts 1/3/4/15, multiple rows and one/two layers; a separate train-mode test fixes dropout randomness. |
| Future session / other student / other batch isolation | PASS | [tests/test_independent_hitskt_audit.py:317](../tests/test_independent_hitskt_audit.py#L317) and complete store-bound audits. |
| Explicit causal attention outputs and gradients | PASS | [tests/test_independent_hitskt_audit.py:19](../tests/test_independent_hitskt_audit.py#L19); independent quadratic triangular kernel oracle, separate exp/where feature map, all input/projection gradients, float64 output atol 2e-11 and gradient atol 5e-11 at short lengths; selected outputs/all gradients also checked at 939/3,925/13,081 tokens using width 12 and three heads. These reduced-width length stress tests use output atol 2e-10 and gradient atol 5e-10. Separate length-7 checks use actual checkpoint widths 128/256 and 64-dimensional heads. |
| Attention padding / hole / empty-key masks | PASS | 16 combinations of lengths 1/2/5/11 and dense/right/holes/empty masks, batch size 3, different query masks; future/cross-row Jacobians exactly zero. |
| Model padding and EOS metric exclusion | PASS | [tests/test_independent_hitskt_audit.py:263](../tests/test_independent_hitskt_audit.py#L263); manual real-target BCE, poisoned finite PAD/EOS logits/labels, zero masked loss gradients. |
| Trained-checkpoint response perturbations | PASS | Four genuine test examples per checkpoint, current/middle/final response-suffix flips; counts and maximum changes in each evaluation manifest. |
| Arbitrary left/internal padding for full HiTSKT | NOT VERIFIED | The full-model EOS pool uses mask.sum−1 and requires the collator’s contiguous right-padding contract. Attention-level holes are tested; arbitrary externally constructed HiTSKT masks are outside that contract. |
| Real answer submission available before query | NOT VERIFIED | Index causality does not establish wall-clock availability; see section 4. |

The causal argument is compositional: at target index t, the correctness input contains only earlier target responses y_0 through y_(t−1), plus BOS; every correctness-memory attention prefix and decoder memory prefix ends at t. Histories end strictly before the target session and stay within the same student. Tokenwise feed-forward/normalization and per-row projections introduce no future-index or cross-student path. EOS can summarize all responses of an earlier session; the target EOS receives the last target response but is later than every scored target and excluded from loss/metrics. PAD logits need not be zero after the final biased prediction head; their masks exclude them.

The original suite passed **84 tests**; new model/metric/loader/baseline checks passed **39**, maximum-length reference checks passed **3**, actual checkpoint dimension checks passed **2**, and audit-tool/resource/golden-oracle checks passed **5**: **133 distinct tests passed**. The first new suite had one invalid fixture that flipped an EOS ID as if binary; its IndexError was preserved, corrected and rerun. It is an audit-fixture failure, not a model failure. Evidence: [existing_tests.log](independent_correctness_audit/existing_tests.log), [independent_tests.log](independent_correctness_audit/independent_tests.log), [audit_tool_tests.log](independent_correctness_audit/audit_tool_tests.log), [maximum_length_tests.log](independent_correctness_audit/maximum_length_tests.log), [checkpoint_dimension_tests.log](independent_correctness_audit/checkpoint_dimension_tests.log), [initial_fixture_failure.log](independent_correctness_audit/initial_fixture_failure.log). The final collection contains 133 cases; test_summary.json reconciles the distinct passing cases and their logs.

## 3. Dataset provenance and exhaustive reconstruction

| Accepted source | Bytes | SHA-256 | Identity status |
|---|---:|---|---|
| assist2017 authors’ preprocessed CSV | 82,778,007 | `7577287d4ead073fdb9393b586a260043151f27338246bd076ca740e28be8bbb` | PASS: exact author-repository LFS object |
| junyi authors’ preprocessed CSV | 671,494,411 | `8a18000806279ae4feae90a94a5af75b7688f2e4ba666c9a0662c75e1a404b6a` | PASS: exact author-repository LFS object |
| EdNet-KT1.zip | 1,201,163,816 | `0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535` | PASS: fresh official download matches local bytes |
| EdNet-Contents.zip | 173,976 | `aa910a0436d9dbac0ba232f55e27b37ad8d39da285fcf15f1c1eb5d064be98e7` | PASS: fresh official download matches local bytes |

The verified upstream `origin/main` is `6c930c5ce15bb9b63e303c0223ae5b75019f6ad1`, matching the remote at verification. Both flat source hashes/sizes exactly match its LFS pointers. Their row/student/question counts also match Table 2 of the [authors’ HiTSKT paper](https://arxiv.org/pdf/2212.12139). They are the authors’ **preprocessed distributions**; an untouched publisher-raw-to-author-CSV transformation/checksum chain is **NOT VERIFIED**. This qualification does not negate the proven byte identity with the declared reproduction inputs. See [upstream_dataset_identity.json](independent_correctness_audit/upstream_dataset_identity.json).

EdNet provenance follows the [official EdNet repository](https://github.com/riiid/ednet#kt1) and the recorded acquisition revision in [the existing provenance note](../docs/data/ednet_kt1_provenance.md). All 784,309 student CSV members were read with ZIP CRC checking; all 95,293,926 raw rows were parsed. The official README currently states 95,294,926 problem consumptions elsewhere: its aggregate text differs by 1,000 and is not a publisher checksum. This audit establishes the locally acquired KT1 archive identity and contents rather than silently equating those textual totals.

Both official repository download shortcuts were resolved again and their current files streamed independently into SHA-256 without replacing local archives. The initial Contents request failed DNS and remains recorded as NOT VERIFIED; the separate retry passed. See [ednet_official_download_identity.json](independent_correctness_audit/ednet_official_download_identity.json) and [ednet_contents_official_retry_identity.json](independent_correctness_audit/ednet_contents_official_retry_identity.json). This is an observed byte match to the current publisher-hosted files, not a publisher-signed historical checksum.

| Dataset | Raw rows | Retained students | Retained interactions | Sessions | Distinct retained questions | Distinct retained skills | Raw/Parquet/store exact field equality |
|---|---:|---:|---:|---:|---:|---:|---|
| assist2017 | 942,807 | 1,388 | 885,335 | 12,402 | 3,112 | 102 | PASS |
| junyi | 14,660,217 | 29,865 | 14,660,217 | 600,154 | 25,630 | 1,521 | PASS |
| ednet_kt1 | 95,293,926 | 116,548 | 81,940,867 | 2,577,988 | 12,283 | 1,490 | PASS |

**Retained distinct-count reporting: FAIL for ASSIST/EdNet if the existing fields are described as distinct counts.** The store uses maximum IDs for embedding capacity: ASSIST question bound 3,158 includes 46 unused IDs; EdNet question bound 13,169 includes 886 unused IDs, and skill bound 1,495 includes 5 unused IDs. The training runners copy these bounds into dataset_stats questions/skills fields, which feed the master table. The true distinct retained counts are 3,112/102 (ASSIST), 25,630/1,521 (Junyi), 12,283/1,490 (EdNet). The checkpoint index capacities and seed 42 exactly match the saved architecture and data. Correct the statistical description, not the saved embedding indices or EOS IDs; no retraining follows from this finding. See [checkpoint_vocabulary_consistency.json](independent_correctness_audit/checkpoint_vocabulary_consistency.json) and [ktbench/data/session_store.py:108](../ktbench/data/session_store.py#L108).


The independent reconstruction does not call production preprocessing/session assignment. It validates original row identity, authoritative ASSIST action_num ties, Junyi source-row ties, EdNet timestamp/source-row ties, per-student boundaries, binary labels, question/skill mappings, cumulative attempt counters, session positions and every processed/store event. Raw and processed file SHA-256 inventories and historical published mapping/report hash comparisons are saved per dataset. See [assist2017_data_manifest.json](independent_correctness_audit/assist2017_data_manifest.json), [junyi_data_manifest.json](independent_correctness_audit/junyi_data_manifest.json), [ednet_kt1_data_manifest.json](independent_correctness_audit/ednet_kt1_data_manifest.json).

Sessions begin at a timestamp gap **≥10 hours**; students need ≥5 sessions. For n sessions, validation and test each receive floor(n/5), and training receives the earlier remainder. This differs from a universally exact 60/20/20 percentage for small n. Timestamps are preserved in milliseconds; the flat sources are converted from seconds. No session is truncated/chunked. The paper describes end-to-start gaps >10h; the audited approved implementation uses start-to-start gaps ≥10h and is a documented protocol variant, not an exact paper reproduction.

ASSIST preserves the supplied row skill. Junyi uses each question’s deduplicated complete full-source tag set, sorted numerically, as one composite skill. This is label-free, transductive metadata construction, not train-only inductive vocabulary learning. EdNet independently joins question answers and complete tag sets, deduplicates 197 repeated-tag questions, assigns 1,495 composite skills, reserves `<UNTAGGED>` for tags−1, and excludes 27,646 unanswered rows without relabelling them incorrect. Their audit count was checked separately; exhaustive field equality above concerns supervised events.

### Split and target coverage

**The protocol is chronological within-student, not student-disjoint and not a global calendar cutoff. Every retained student appears in training, validation and test.** Validation/test histories use earlier responses from their own split as well as earlier splits; weights are fixed during evaluation. Whole held-out target sessions also teacher-force strictly earlier responses inside that session. Batch length sorting changes execution order, not which earlier labels are supplied. This is an online/prequential response-visibility assumption, not a one-shot label-hidden holdout.

| Dataset | HiTSKT train targets | Baseline train targets | First-session interactions context-only in HiTSKT | Validation targets | Test targets |
|---|---:|---:|---:|---:|---:|
| assist2017 | 566,107 | 647,283 | 81,176 | 125,800 | 112,252 |
| junyi | 8,920,358 | 9,647,042 | 726,684 | 2,556,773 | 2,456,402 |
| ednet_kt1 | 49,458,444 | 53,990,022 | 4,531,578 | 14,520,975 | 13,429,870 |

Every intended post-first-session HiTSKT target is covered exactly once; first-session training interactions are deliberately context-only. Baselines train on all retained interactions. These are different training target populations, despite equal held-out target populations. Complete history-bound checks found no cross-student or later-index history. Later test sessions with earlier same-test history number 589/1,977 (ASSIST), 79,641/109,506 (Junyi), 358,279/474,827 (EdNet).

## 4. Response-availability limitation

| Dataset | Test events with predecessor end after current start | Same-session subset | Adjacent equal test timestamps | Status of timing check |
|---|---:|---:|---:|---|
| assist2017 | 2,426 | 2,412 | 1,102 | FAIL |
| junyi | 0 | 0 | 2,165,594 | NOT VERIFIED |
| ednet_kt1 | 4,003,464 | 4,003,463 | 431,050 | FAIL |

ASSIST uses recorded endTime; EdNet uses timestamp+elapsed_time as a completion **proxy**. The [official KT1 field description](https://github.com/riiid/ednet#kt1) defines timestamp as presentation time and elapsed_time as question time, and describes bundled questions. These fields do not independently certify the exact correctness-feedback arrival time. Junyi has no recorded end/start overlaps but 2,165,594 tied test timestamps, ordered by source row without an authoritative response-arrival index.

A bounded diagnostic selected eight distinct overlapping test sessions per ASSIST/EdNet checkpoint. Flipping the earlier-index predecessor response changed the current logit in all 8/8 cases in each dataset; maximum absolute changes were 0.84395301342010498 and 0.31327390670776367. Flipping the current target’s own label left its prediction unchanged in all cases. This is evidence that the online response-availability assumption matters; it is not proof of an acausal event-index attention graph, nor an assertion that every overlap has delayed feedback. See [answer_availability_confirmed.json](independent_correctness_audit/answer_availability_confirmed.json) and [unavailable_response_sensitivity.json](independent_correctness_audit/unavailable_response_sensitivity.json).

The first timing artifact used an overly strong ASSIST interpretation equating endTime with label availability. That wording was corrected and the complete scan rerun; all numerical counts exactly match. The original [answer_availability.json](independent_correctness_audit/answer_availability.json) and its source snapshot are preserved; the confirmed artifact and the qualified interpretation above supersede that sentence.

Strict presentation-time availability is therefore **NOT VERIFIED** for all three datasets. Under the specific rule “a label becomes available only at recorded end/completion”, ASSIST/EdNet fail the availability inequality. Actual submission/feedback timestamps or a validated alternative contract are needed before claiming real-time, unavailable-response-safe prediction.

## 5. Fresh-process HiTSKT checkpoint replay

The evaluator imports the unchanged model/collator but neither the production training `_run_epoch` nor the project metric helpers. It checks every batch’s raw-store labels/event IDs, every test session once, the full target count, finite logits/probabilities and original artifact hashes. Metrics use independent NumPy confusion counts, tied-pair AUC, MSE and stable float64 logit BCE. The historical FP32 batch-weighted BCE is also calculated separately.

For tied-score group g, AUC = Σ P_g(N_before_g + N_g/2)/(P·N). Classification is p≥0.5. BCE = mean(max(z,0)−y·z+log1p(exp(−abs(z)))). The float64 BCE need not be bitwise identical to a sequence of FP32 batch means; both comparisons are disclosed. Full ASSIST evaluation through the unchanged production CPU helper exactly agreed with the independent CPU evaluator for every metric, including historical BCE: [assist2017_cpu_production_crosscheck.json](independent_correctness_audit/assist2017_cpu_production_crosscheck.json).

PyTorch explicitly does not promise bitwise CPU/GPU equality ([version 2.4 numerical-accuracy documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.0/docs/source/notes/numerical_accuracy.rst)). That makes device arithmetic a plausible explanation, not an established diagnosis. Original per-target CUDA probabilities/logits were not persisted; a fresh CUDA replay was not attempted while protecting the ongoing GPU training; original GPU runtime/TF32 details are not fully pinned in saved configs. No mismatch is silently dismissed as rounding, and no stored result is overwritten.

| Dataset | Selected epoch / completed epochs | Width / heads / FF width | Adam learning rate | Parameters | Historical config |
|---|---:|---|---:|---:|---|
| assist2017 | 40 / 45 | 256 / 4 / 2048 | 5e-05 | 6,931,457 | [config.json](../experiments/hitskt/assist2017/full/config.json) |
| junyi | 32 / 37 | 128 / 2 / 1024 | 5e-05 | 8,271,105 | [config.json](../experiments/hitskt/junyi/full/config.json) |
| ednet_kt1 | 30 / 35 | 128 / 2 / 1024 | 8e-05 | 5,074,433 | [config.json](../experiments/hitskt/ednet_kt1/full/config.json) |

All use seed 42, one layer in each of the four hierarchy stages, dropout 0.1, weight decay 0, 15 previous sessions, token budget 32,768, maximum batch 64, strict validation-AUC improvement, patience 5, and best-checkpoint reload before test. Saved configs retain their original attention name; section 6 corrects its scientific interpretation.

### assist2017

Manifest: [assist2017_evaluation_manifest.json](independent_correctness_audit/assist2017_evaluation_manifest.json). Replay comparison status: **FAIL**.

Checkpoint `experiments/hitskt/assist2017/full/best_model.pt`, epoch **40**, SHA-256 `085c8d9ae32f4e60892dfb3e4eed93d90a584696d0914456c135dcab7b890164`. All **112,252** test targets evaluated. Config and every store-file hash are in the manifest.

| Metric | Originally stored | Independently recalculated on CPU | Absolute difference | Exact equality | 1e-6 gate |
|---|---:|---:|---:|---|---|
| roc_auc | 0.70962401591977553 | 0.709623637214297 | 3.7870547853202652e-07 | FAIL | PASS |
| accuracy | 0.67442005487652779 | 0.67437551224031644 | 4.4542636211342135e-05 | FAIL | FAIL |
| precision | 0.58660822623381126 | 0.58654846218612255 | 5.9764047688704025e-05 | FAIL | FAIL |
| recall | 0.5440494235450829 | 0.5439805245510082 | 6.8898994074695352e-05 | FAIL | FAIL |
| f1 | 0.5645278522490319 | 0.56446308564892045 | 6.4766600111454409e-05 | FAIL | FAIL |
| mse | 0.20984520173382062 | 0.20984431782624707 | 8.8390757355116811e-07 | FAIL | PASS |
| historical_batch_weighted_bce | 0.60872574173288641 | 0.60872361103781436 | 2.1306950720445528e-06 | FAIL | FAIL |
| bce_float64 | 0.60872574173288641 | 0.60872360954743021 | 2.1321854561939801e-06 | FAIL | FAIL |

Exact original-CUDA reproduction is **NOT VERIFIED**; these are complete CPU measurements, not substituted historical scores.

### junyi

Manifest: [junyi_evaluation_guarded_manifest.json](independent_correctness_audit/junyi_evaluation_guarded_manifest.json). Replay comparison status: **FAIL**.

Checkpoint `experiments/hitskt/junyi/full/best_model.pt`, epoch **32**, SHA-256 `2308c5bf0c850da1e55daf802a0853a30fac45eeed071c7162c68fc17bfa3bdd`. All **2,456,402** test targets evaluated. Config and every store-file hash are in the manifest.

| Metric | Originally stored | Independently recalculated on CPU | Absolute difference | Exact equality | 1e-6 gate |
|---|---:|---:|---:|---|---|
| roc_auc | 0.79782620551912631 | 0.79782647392335382 | 2.6840422751028825e-07 | FAIL | PASS |
| accuracy | 0.75555059798844004 | 0.75555833287873886 | 7.7348902988161328e-06 | FAIL | FAIL |
| precision | 0.78504526764344096 | 0.78505505319603119 | 9.7855525902357243e-06 | FAIL | FAIL |
| recall | 0.88154366216764457 | 0.88154006662539786 | 3.5955422467148779e-06 | FAIL | FAIL |
| f1 | 0.83050075255717004 | 0.83050463267069519 | 3.88011352514539e-06 | FAIL | FAIL |
| mse | 0.16478629880597911 | 0.16478621003428573 | 8.8771693385059081e-08 | FAIL | PASS |
| historical_batch_weighted_bce | 0.49757788112870566 | 0.49757767997048347 | 2.0115822219057833e-07 | FAIL | PASS |
| bce_float64 | 0.49757788112870566 | 0.49757767893426852 | 2.0219443713775931e-07 | FAIL | PASS |

Exact original-CUDA reproduction is **NOT VERIFIED**; these are complete CPU measurements, not substituted historical scores.

### ednet_kt1

Manifest: [ednet_kt1_evaluation_guarded_manifest.json](independent_correctness_audit/ednet_kt1_evaluation_guarded_manifest.json). Replay comparison status: **FAIL**.

Checkpoint `experiments/hitskt/ednet_kt1/full/best_model.pt`, epoch **30**, SHA-256 `58d4b1f393d0fa8ec372ba8bcc00cc8b7168d37817d4badfd26ee42abffd03c8`. All **13,429,870** test targets evaluated. Config and every store-file hash are in the manifest.

| Metric | Originally stored | Independently recalculated on CPU | Absolute difference | Exact equality | 1e-6 gate |
|---|---:|---:|---:|---|---|
| roc_auc | 0.76936483644136922 | 0.76936473615094192 | 1.002904272962013e-07 | FAIL | PASS |
| accuracy | 0.73886061443632733 | 0.73886284826286475 | 2.2338265374122201e-06 | FAIL | FAIL |
| precision | 0.76393874035793019 | 0.7639391533725004 | 4.1301457021081234e-07 | FAIL | PASS |
| recall | 0.89461437100414254 | 0.89461818157272177 | 3.8105685792322319e-06 | FAIL | FAIL |
| f1 | 0.82412865890046183 | 0.82413051610890853 | 1.8572084466983441e-06 | FAIL | FAIL |
| mse | 0.17371257418130398 | 0.17371259468978537 | 2.0508481385350663e-08 | FAIL | PASS |
| historical_batch_weighted_bce | 0.51851811729597075 | 0.51851816682806529 | 4.9532094537063642e-08 | FAIL | PASS |
| bce_float64 | 0.51851811729597075 | 0.51851816586401889 | 4.8568048138797337e-08 | FAIL | PASS |

Exact original-CUDA reproduction is **NOT VERIFIED**; these are complete CPU measurements, not substituted historical scores.

## 6. Architecture terminology and fidelity

**Deterministic ELU+1 causal linear attention: PASS. Faithful FAVOR+ Performer claim: FAIL.** [ktbench/models/performer.py:68](../ktbench/models/performer.py#L68) computes learned Q/K/V projections, fixed elementwise φ(x)=ELU(x)+1, masked key/value prefixes, normalization with denominator floor 1e-6, then output projection. The effective causal kernel weight is φ(q_t)ᵀφ(k_j) for j≤t. It has no random-feature matrix, orthogonal feature construction or random-feature count.

This is the causal linear-transformer formulation in [Katharopoulos et al., 2020, equations 7–12](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf). [Choromanski et al., Performer/FAVOR+](https://arxiv.org/abs/2009.14794) instead uses positive orthogonal random features to estimate an attention kernel, including softmax attention. A learned Q/K projection followed by ELU+1 is not that estimator.

Mathematically, the implementation exactly uses κ_ELU(q,k)=Σ_r(ELU(q_r)+1)(ELU(k_r)+1). FAVOR+ uses stochastic finite-dimensional features φ_Ω whose inner product estimates a desired kernel such as exp(qᵀk/√d); feature sampling and approximation error are part of that method. The kernels are different: for one-dimensional q=1 and keys 0,1, normalized ELU+1 weights are (1/3,2/3), whereas softmax weights are (1/(1+e),e/(1+e)).

Its causal computation is S_t=Σ_{j≤t}m_j φ(k_j)v_jᵀ and z_t=Σ_{j≤t}m_j φ(k_j), followed by o_t=φ(q_t)ᵀS_t / max(φ(q_t)ᵀz_t,10⁻⁶), output projection and query masking. The explicit reference constructs all permitted query/key weights directly instead of reusing either cumulative sum.

The hierarchy is retained, but this is a **HiTSKT-inspired hierarchical model with deterministic ELU+1 causal linear attention**. It also differs from the paper/legacy implementation in rehearsal/question-attempt embeddings, session power-law decay and block formulation: [model_layers.py:43](../model_layers.py#L43) and [attention_modules.py:48](../attention_modules.py#L48) show components absent from the active model; active blocks use pre-normalization/GELU. The top-level legacy `main.py` still imports the old quadratic-softmax implementation, while canonical audited checkpoints use `scripts/train_hitskt.py` → `ktbench/models/hitskt.py`. These paths must not be conflated.

No architecture was changed for this audit. Correcting terminology needs no retraining. A claim specifically about a genuine FAVOR+ model, or a faithful paper-architecture reproduction, requires separately implemented and trained models; historical ELU+1 checkpoints cannot establish it.

## 7. Baselines, completeness and recorded gains

All 14 completed experiment directories pass best-checkpoint/strict-validation-selection checks, CSV/JSON test consistency and finite best-model tensors. All 295 epoch records from those 14 completed runs have finite train/validation metrics. DKT/EdNet is unfinished and cannot supply a final benchmark row. Evidence: [benchmark_artifacts.json](independent_correctness_audit/benchmark_artifacts.json), [epoch_metric_finiteness.json](independent_correctness_audit/epoch_metric_finiteness.json), [publication_consistency.json](independent_correctness_audit/publication_consistency.json) and [dkt_ednet_pending_state.json](independent_correctness_audit/dkt_ednet_pending_state.json). Baseline target-label/PAD exclusion is independently tested for DKT, DKVMN, SAKT and RKT; full baseline test metrics were **NOT VERIFIED by fresh replay** in this audit.

| Model | Context / input distinction | Audit status |
|---|---|---|
| DKT | 199 prior interactions, skill responses, width 64 LSTM | Artifact/masking PASS; EdNet final result NOT VERIFIED |
| DKVMN | 199 prior interactions, skill-based memory, 20 slots | Artifact/masking PASS; fresh metric replay NOT VERIFIED |
| SAKT | 99 prior interactions, question+skill attention, width 200/5 heads | Artifact/masking PASS; fresh metric replay NOT VERIFIED |
| RKT | 49 prior interactions, question/Phi/time/student parameters, width 64 | Artifact/masking/cache algebra PASS; fresh metrics/raw Phi pair-count rebuild NOT VERIFIED |
| HiTSKT variant | 15 complete earlier sessions plus prior target-session responses | Structural checks PASS; full replay outcomes above |

RKT is explicitly the performance-only Phi variant, omitting question-text relations. Persisted five-fold student assignments exactly match seed 42; every all-train count entry equals the sum of five folds, and every complement exactly equals all-minus-own-fold. Source limits construction to training partitions and chooses own-fold exclusion for training rows. The audit did not independently regenerate all raw-label pair counts, so that provenance remains NOT VERIFIED. These five-fold caches do not make the benchmark train/test split student-disjoint. See [rkt_cache_algebra.json](independent_correctness_audit/rkt_cache_algebra.json) and [ktbench/rkt/phi.py:147](../ktbench/rkt/phi.py#L147).

The additional [rkt_cache_key_bounds.json](independent_correctness_audit/rkt_cache_key_bounds.json) exhaustively checks sparse-key ordering/range, membership, cached-key CRC identity and full-population/target-count metadata for all three caches. Algebraic consistency cannot substitute for independently rebuilding their label-derived counts.

| Dataset | Recorded HiTSKT AUC | Best available baseline | Recorded baseline AUC | Recorded difference | Significance |
|---|---:|---|---:|---:|---|
| assist2017 | 0.70962401591977553 | rkt | 0.74862648987932845 | -0.039002473959552919 | NOT VERIFIED |
| junyi | 0.79782620551912631 | rkt | 0.7912548486618094 | 0.0065713568573169123 | NOT VERIFIED |
| ednet_kt1 | 0.76936483644136922 | sakt | 0.76300785731221421 | 0.0063569791291550048 | NOT VERIFIED |

The Junyi/EdNet differences are descriptive **seed 42** results, not demonstrated statistically significant gains or isolated effects of Performer/FAVOR+. EdNet lacks DKT. Models have different features, contexts, parameter counts, training target populations and established hyperparameters; common data splits, test populations, threshold and validation selection help comparison but do not create a controlled architectural ablation. ASSIST HiTSKT does not beat the strongest recorded baseline.

## 8. Issues, severity, affected results and retraining implications

| Issue | Severity / status | Affected evidence | Is retraining genuinely required? |
|---|---|---|---|
| Unresolved CPU vs stored metric discrepancies | HIGH / all three CPU metric gates FAIL; exact CUDA NOT VERIFIED | Historical HiTSKT numbers requiring exact confirmation | Not established. First resolve source/runtime/per-target differences; a CPU mismatch alone does not prove training was wrong. |
| Response feedback availability not certified; recorded/proxy overlaps and timestamp ties | HIGH / NOT VERIFIED real-time contract | All models using earlier responses; ASSIST/EdNet timing inequality FAIL, Junyi order limited by ties | Conditional: if the scientific contract requires availability-aware training contexts, rebuild contexts and retrain affected methods. Mere event-index replay does not require retraining. |
| ELU+1 described as pure/faithful Performer | HIGH / FAIL terminology | All three canonical HiTSKT architecture claims | No for accurate ELU+1 naming; yes for a new FAVOR+ claim/model. |
| Additional differences from paper HiTSKT, including missing rehearsal/power-law components | MEDIUM / exact-paper fidelity NOT VERIFIED | Interpretation of current variant and comparisons with paper results | No for disclosed variant results; yes for a claim about a different faithful architecture. |
| Start-to-start gaps ≥10h differ from the paper’s end-to-start gaps >10h | MEDIUM / exact paper-protocol equivalence FAIL | Session construction and cohort/split interpretation for all three datasets | No for the disclosed approved variant; an exact paper-protocol claim requires rebuilding that protocol and training/evaluating its models. |
| Only seed 42; no paired uncertainty or multiple-seed evidence | MEDIUM / significance NOT VERIFIED | Junyi and EdNet superiority claims | No to preserve descriptive historical results; supporting experiments are required for stronger generalization/significance claims. |
| First-session training targets excluded only for HiTSKT | MEDIUM / equal-training-population claim FAIL | All cross-model scientific comparisons | No automatic invalidation; matched-population attribution would require controlled retraining. |
| Baseline metrics not freshly replayed; RKT raw label-pair counts not independently rebuilt | MEDIUM / NOT VERIFIED beyond scoped artifact/masking/cache checks | DKT, DKVMN, SAKT and RKT historical scores | No established need. Recompute/rebuild the missing evidence first; changed cached relations could require RKT retraining. |
| Vocabulary index capacity presented as distinct retained question/skill counts | MEDIUM / distinct-count interpretation FAIL | ASSIST/EdNet dataset_stats and master tables: true counts shown in section 3 | No. Correct/relabel publication statistics while retaining checkpoint IDs/EOS capacities. |
| DKT/EdNet incomplete; last epoch 9 Adam has 21,749 positive infinities | HIGH / final result NOT VERIFIED | Fifteenth run only; best epoch 1 weights remain finite | Recovery/retraining decision unresolved. Do not silently reset optimizer or publish an invented final row. |
| DKT checkpoint embeds patience 5 while active config is authorized 10 | LOW / metadata inconsistency | DKT/EdNet future continuation; actual counter 8/10 | No retraining from this metadata alone; preserve active config/counter and correct future metadata only with a scoped change. |
| Flat inputs are author-preprocessed CSVs; publisher-raw transformation chain absent | MEDIUM / deeper provenance NOT VERIFIED | Claims of untouched publisher-original data | No if the declared input is the verified author distribution; reconsider results only if a different required source is established. |
| Historical configs lack an immutable full-data/runtime provenance bundle | MEDIUM / historical exact environment NOT VERIFIED | Exact training/replay provenance beyond present hashes | Not by itself. Current raw reconstruction and replay hashes improve future evidence but cannot retroactively prove every historical environment detail. |
| Legacy entrypoint remains quadratic softmax and fixed-window preprocessing | MEDIUM / path-confusion risk | Legacy runs, excluded from canonical audited result set | No for canonical variant checkpoints; legacy results cannot inherit this audit. |
| Final complete figure/report bundle absent | MEDIUM / NOT VERIFIED | reports/report_manifest.json and final 15-run figures are absent in this checkout | No retraining implication by itself; benchmark remains incomplete. |
| Stale committed README completion counts | LOW / documentation inconsistency | README at audit base says 13 completed/all 11 validated; master has 14 | No. Reconcile documentation without overwriting the unrelated local README edit. |
| Initial audit fixture/timing-interpretation errors and two audit-only OOM kills | INFORMATIONAL / invalid attempts preserved and corrected | Audit process; completed bounded checks supersede partial attempts | No model retraining. Original experiments were preserved; protected trainer survival was checked. |

No retraining was launched. No evidence from a passing unit test, an interrupted job, a previous “verified” JSON, or a partial replay was promoted to an independently reproduced final result.

## 9. Evidence and reproducibility

Relevant implementations: [scripts/audit_hitskt_data.py:33](../scripts/audit_hitskt_data.py#L33), [scripts/audit_hitskt_replay.py:47](../scripts/audit_hitskt_replay.py#L47), [scripts/audit_hitskt_replay_parallel.py:48](../scripts/audit_hitskt_replay_parallel.py#L48), [scripts/audit_memory_guard.py:19](../scripts/audit_memory_guard.py#L19). Prediction/logit/label/event-ID arrays and decompressed cache scratch remain local under `.inspection/independent_correctness_audit/`; their hashes and paths are recorded, and raw interactions are not published.

Commands were executed with `PYTHONPATH=.` and GPU visibility disabled. Replays refuse to overwrite existing output manifests/caches. The examples below require a fresh audit checkout/cache namespace as well as a new output manifest, because prediction-cache names are fixed per dataset; preserve existing evidence. Original files are protected by the final [preservation_verification.json](independent_correctness_audit/preservation_verification.json). Audit-source bindings, test-artifact hashes and all 12 full prediction-array hashes are checked in [evidence_integrity.json](independent_correctness_audit/evidence_integrity.json).

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. python -m pytest -q tests/test_independent_hitskt_audit.py tests/test_audit_tool_oracles.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_hitskt_data.py assist2017 --output NEW_ASSIST_DATA_MANIFEST.json
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_hitskt_replay.py assist2017 --threads 2 --output NEW_ASSIST_REPLAY_MANIFEST.json
```

## 10. Follow-up recommendations after completed verification

1. Resolve the CPU/stored discrepancies before claiming exact reproduction: when the protected GPU workload permits, replay the preserved checkpoints with the historical GPU/runtime and batch sequence, pin matrix-precision settings, and retain per-target logits/IDs for a direct comparison. This is verification, not an established reason to retrain.

2. Establish the prediction-time/feedback-arrival contract from authoritative event documentation. If labels are unavailable under that contract, build an availability-aware protocol and evaluate it; retrain affected methods if their training contexts must change. Keep the current within-student, retrospective cohort and online earlier-label protocol explicitly named.

3. Freshly replay completed baseline checkpoints and independently regenerate RKT raw training-only pair counts. Diagnose DKT/EdNet’s nonfinite Adam state and agree a recorded recovery policy before any separately authorized continuation. Do not treat its best validation score as a final test result.

4. For stronger Junyi/EdNet superiority claims, run multiple training seeds and report paired student-cluster uncertainty. Control features, history, parameter budget and training target population when attributing gains to a kernel. A genuine FAVOR+ comparison requires a separately implemented and trained variant; preserve the ELU+1 historical model.

5. Use the corrected retained distinct counts and architecture name in publication material. Add student-disjoint evaluation only if making a new-student generalization claim. Finish the missing benchmark evidence before presenting a complete 15/15 comparison. No additional experiment or training was started by this audit.

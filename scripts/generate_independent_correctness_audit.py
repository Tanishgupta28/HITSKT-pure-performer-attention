#!/usr/bin/env python3
"""Render the final audit only after required full replays reach terminal state."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT=Path.cwd()
EVIDENCE=Path('reports/independent_correctness_audit')
DATASETS=['assist2017','junyi','ednet_kt1']


def read(name):return json.loads((EVIDENCE/name).read_text())


def source(path,needle):
    line=next(i for i,s in enumerate(Path(path).read_text().splitlines(),1) if needle in s)
    return f'[{path}:{line}](../{path}#L{line})'


def artifact(name):return f'[{name}](independent_correctness_audit/{name})'


def number(x):return format(x,'.17g')


def main():
    evaluations={d:read(f'{d}_evaluation_manifest.json' if d=='assist2017' else f'{d}_evaluation_guarded_manifest.json') for d in DATASETS}
    for d,value in evaluations.items():
        if value['status'] in ['RUNNING','INTERRUPTED']:raise SystemExit(f'{d}: required full replay still pending')
    data={d:read(f'{d}_data_manifest.json') for d in DATASETS}
    assert all(v['status']=='PASS' for v in data.values())
    baseline=read('benchmark_artifacts.json');availability=read('answer_availability_confirmed.json')['records']
    for d in DATASETS:
        assert availability[d]['by_target_split']==read('answer_availability.json')['records'][d]['by_target_split']
    vocab=read('checkpoint_vocabulary_consistency.json')['records']
    crosscheck=read('assist2017_cpu_production_crosscheck.json')
    cache=read('rkt_cache_algebra.json')
    sensitivity=read('unavailable_response_sensitivity.json')['records']
    assert crosscheck['status']=='PASS' and all(v['status']=='PASS' for v in cache['records'].values())
    assert read('preservation_verification.json')['status']=='PASS'
    assert read('test_summary.json')['total_distinct_tests_passed']==133
    official=read('ednet_official_download_identity.json')
    contents_retry=read('ednet_contents_official_retry_identity.json')
    assert next(r for r in official['records'] if r['name']=='KT1')['status']=='PASS'
    assert contents_retry['status']=='PASS'
    for d in ['junyi','ednet_kt1']:
        assert read(f'{d}_guarded_worker_direct_crosscheck.json')['status']=='PASS'
    out=[]
    add=out.append
    add('# HiTSKT independent correctness audit\n')
    add(f'Audit completed: {datetime.now(timezone.utc).isoformat()}. Historical/source base: `656a9f062f228710c7aac1a3f8a499e1ffc301ff`. Repository: [Tanishgupta28/HITSKT-pure-performer-attention](https://github.com/Tanishgupta28/HITSKT-pure-performer-attention).\n')
    add('## Decision and status definitions\n')
    add('**Publication clearance: NOT VERIFIED. Do not present this audit as an unconditional certification of all historical results.** The active model passed independent sequence-index masking/causality checks, and all three accepted full datasets were exhaustively reconstructed. Metric reproduction outcomes and remaining scientific limitations are stated separately below.\n')
    add('**PASS** means the named, scoped check met its stated criterion. **FAIL** means an observed contradiction or numerical discrepancy exceeded that criterion. **NOT VERIFIED** means the necessary evidence or experiment is absent, unavailable, or outside the completed check. A passing test suite or `_SUCCESS` marker alone is never the criterion for overall correctness.\n')
    add('| Dataset | Overall result audit | Raw → processed → store | Actual best checkpoint | Sequence-index causality | Full CPU metrics vs stored, tolerance 1e-6 | Exact historical CUDA replay | Response arrival before prediction |\n|---|---|---|---|---|---|---|---|')
    for d in DATASETS:
        value=evaluations[d]
        certification='FAIL' if value['status']=='FAIL' else 'NOT VERIFIED'
        add(f"| {d} | {certification} | PASS | PASS | {value.get('trained_checkpoint_causality',{}).get('status','NOT VERIFIED')} | {value['status']} | NOT VERIFIED | NOT VERIFIED |")
    add('\nAll three AUC differences are below 1e-6. Classification-metric differences exceed that gate on every dataset, and ASSIST BCE also exceeds it. The discrepancy cause is unresolved; it does not by itself establish a leakage bug or a need to retrain. The CPU metric gate is deliberately strict and was declared before the first replay. Every exact equality flag and absolute difference is retained; no tolerance was enlarged after seeing the results. The presentation/completion proxy inequalities fail on ASSIST2017 and EdNet; their interpretation requires a separate availability contract.\n')
    add('## 1. Independence, preservation and execution\n')
    add('The audit read original checkpoints/configs/logs and created separate scripts, tests, manifests and local prediction arrays. All 36 inspected original implementation files are byte-identical to the base commit; see '+artifact('source_inventory.json')+'. Final preservation verification checks the original experiment files against their captured hashes, including all 30 best/last checkpoints; all 15 best checkpoints also match their actual binary Git blobs at the audit base. No architecture, original result, checkpoint, training setting or Git history was rewritten. DKT/EdNet was not resumed.\n')
    add('Model/test computation ran on CPU with GPU visibility disabled; resource-heavy bounded jobs used low CPU priority and idle I/O. LightSAKT2 was not edited, signalled or reconfigured. An audit resource-planning mistake used host RAM rather than the 90 GiB container limit: two **audit-only** jobs exited 137. The three previously observed protected trainer PIDs remained alive afterward. The failed attempts were preserved and rerun with bounded memory; no claim of zero performance contention is made. See '+artifact('resource_incident.json')+', '+artifact('resource_guard.json')+' and '+artifact('resource_observation_during_replay.json')+'.\n')
    add('The final large replays use fresh CPU workers, the unchanged production model/collator and historical sampler order, ordered result collection, and a shared 24 GiB admission estimate for inference. Ordered completed prefixes from interrupted audit attempts were copied into separate caches and rechecked against every event ID/label, with exact fresh first-three-batch logits. All remaining batches were evaluated; no sampled or partial replay is counted as a complete result. Interrupted attempts are retained with their interruption status and reasons.\n')
    add('Final preservation gate: **PASS, 375 captured original files**, zero mismatches. The final '+artifact('resource_final_state.json')+' retained two protected LightSAKT2 process identities; previously observed PID 1992414 was absent, with exit reason NOT VERIFIED. A read-only external status check still showed active jobs and a current heartbeat. The cgroup OOM count stayed at the two documented audit victims; no protected process was edited/signalled. This observation is not a claim of uninterrupted health for every external job.\n')
    add('A separate fresh-process crosscheck exactly matched logits, probabilities, labels and event IDs for the first three newly evaluated guarded-worker batches after each preserved prefix: '+artifact('junyi_guarded_worker_direct_crosscheck.json')+' and '+artifact('ednet_kt1_guarded_worker_direct_crosscheck.json')+'. The parallel evaluator’s initial zero worker-parity field is a default when a resumed prefix skips that branch; the supplemental artifacts contain the actually performed checks.\n')
    add('## 2. Masking and information-flow audit\n')
    add('Threat model: fixed checkpoint weights; change a response at the current or a later **event index**, then compare predictions at earlier/current indices. This is distinct from retraining after changing training data, or proving wall-clock response availability.\n')
    add('| Check | Status | Evidence / criterion |\n|---|---|---|')
    checks=[
        ('ActionEncoder history isolation','PASS',source('ktbench/models/hitskt.py','class ActionEncoder')+'; only earlier complete sessions enter history; full-store history bounds checked.'),
        ('SessionEncoder causal ownership and EOS pooling','PASS',source('ktbench/models/hitskt.py','class SessionEncoder')+'; per-example owner/order packing, causal blocks, session EOS after actual histories.'),
        ('CorrectPaddingEncoder causality','PASS',source('ktbench/models/hitskt.py','class CorrectPaddingEncoder')+'; causal layers read shifted responses.'),
        ('Decoder causality','PASS',source('ktbench/models/hitskt.py','class Decoder')+'; current question/skill query, memory keys only through aligned current index.'),
        ('Own target label excluded','PASS',source('ktbench/batching.py','target_correct_input[row, 0]')+'; BOS followed by previous responses; target_labels never read by model forward.'),
        ('Future raw response perturbations','PASS',source('tests/test_independent_hitskt_audit.py','def test_recollated_raw_own')+'; raw suffix labels changed before re-collation, lengths1/2/7/17, histories1/3/4/15, multiple rows and one/two layers; a separate train-mode test fixes dropout randomness.'),
        ('Future session / other student / other batch isolation','PASS',source('tests/test_independent_hitskt_audit.py','def test_future_session_and_other_student')+' and complete store-bound audits.'),
        ('Explicit causal attention outputs and gradients','PASS',source('tests/test_independent_hitskt_audit.py','def explicit_attention')+'; independent quadratic triangular kernel oracle, separate exp/where feature map, all input/projection gradients, float64 output atol2e-11 and gradient atol5e-11 at short lengths; selected outputs/all gradients also checked at 939/3,925/13,081 tokens using width 12 and three heads. These reduced-width length stress tests use output atol 2e-10 and gradient atol 5e-10. Separate length-7 checks use actual checkpoint widths 128/256 and 64-dimensional heads.'),
        ('Attention padding / hole / empty-key masks','PASS','16 combinations of lengths1/2/5/11 and dense/right/holes/empty masks, batch3, different query masks; future/cross-row Jacobians exactly zero.'),
        ('Model padding and EOS metric exclusion','PASS',source('tests/test_independent_hitskt_audit.py','def test_pad_and_eos_loss')+'; manual real-target BCE, poisoned finite PAD/EOS logits/labels, zero masked loss gradients.'),
        ('Trained-checkpoint response perturbations','PASS','Four genuine test examples per checkpoint, current/middle/final response-suffix flips; counts and maximum changes in each evaluation manifest.'),
        ('Arbitrary left/internal padding for full HiTSKT','NOT VERIFIED','The full-model EOS pool uses mask.sum−1 and requires the collator’s contiguous right-padding contract. Attention-level holes are tested; arbitrary externally constructed HiTSKT masks are outside that contract.'),
        ('Real answer submission available before query','NOT VERIFIED','Index causality does not establish wall-clock availability; see section4.')]
    for check,status,evidence in checks:add(f'| {check} | {status} | {evidence} |')
    add('\nThe causal argument is compositional: at target index t, the correctness input contains only earlier target responses y_0 through y_(t−1), plus BOS; every correctness-memory attention prefix and decoder memory prefix ends at t. Histories end strictly before the target session and stay within the same student. Tokenwise feed-forward/normalization and per-row projections introduce no future-index or cross-student path. EOS can summarize all responses of an earlier session; the target EOS receives the last target response but is later than every scored target and excluded from loss/metrics. PAD logits need not be zero after the final biased prediction head; their masks exclude them.\n')
    add('The original suite passed **84 tests**; new model/metric/loader/baseline checks passed **39**, maximum-length reference checks passed **3**, actual checkpoint dimension checks passed **2**, and audit-tool/resource/golden-oracle checks passed **5**: **133 distinct tests passed**. The first new suite had one invalid fixture that flipped an EOS ID as if binary; its IndexError was preserved, corrected and rerun. It is an audit-fixture failure, not a model failure. Evidence: '+artifact('existing_tests.log')+', '+artifact('independent_tests.log')+', '+artifact('audit_tool_tests.log')+', '+artifact('maximum_length_tests.log')+', '+artifact('checkpoint_dimension_tests.log')+', '+artifact('initial_fixture_failure.log')+'. The final collection contains133 cases; test_summary.json reconciles the distinct passing cases and their logs.\n')
    add('## 3. Dataset provenance and exhaustive reconstruction\n')
    add('| Accepted source | Bytes | SHA-256 | Identity status |\n|---|---:|---|---|')
    upstream=read('upstream_dataset_identity.json')
    for d in ['assist2017','junyi']:
        row=upstream[d]
        add(f"| {d} authors’ preprocessed CSV | {row['lfs_bytes']:,} | `{row['lfs_sha256']}` | PASS: exact author-repository LFS object |")
    add('| EdNet-KT1.zip | 1,201,163,816 | `0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535` | PASS: fresh official download matches local bytes |')
    add('| EdNet-Contents.zip | 173,976 | `aa910a0436d9dbac0ba232f55e27b37ad8d39da285fcf15f1c1eb5d064be98e7` | PASS: fresh official download matches local bytes |\n')
    add('The verified upstream `origin/main` is `6c930c5ce15bb9b63e303c0223ae5b75019f6ad1`, matching the remote at verification. Both flat source hashes/sizes exactly match its LFS pointers. Their row/student/question counts also match Table2 of the [authors’ HiTSKT paper](https://arxiv.org/pdf/2212.12139). They are the authors’ **preprocessed distributions**; an untouched publisher-raw-to-author-CSV transformation/checksum chain is **NOT VERIFIED**. This qualification does not negate the proven byte identity with the declared reproduction inputs. See '+artifact('upstream_dataset_identity.json')+'.\n')
    add('EdNet provenance follows the [official EdNet repository](https://github.com/riiid/ednet#kt1) and the recorded acquisition revision in [the existing provenance note](../docs/data/ednet_kt1_provenance.md). All 784,309 student CSV members were read with ZIP CRC checking; all 95,293,926 raw rows were parsed. The official README currently states95,294,926 problem consumptions elsewhere: its aggregate text differs by 1,000 and is not a publisher checksum. This audit establishes the locally acquired KT1 archive identity and contents rather than silently equating those textual totals.\n')
    add('Both official repository download shortcuts were resolved again and their current files streamed independently into SHA-256 without replacing local archives. The initial Contents request failed DNS and remains recorded as NOT VERIFIED; the separate retry passed. See '+artifact('ednet_official_download_identity.json')+' and '+artifact('ednet_contents_official_retry_identity.json')+'. This is an observed byte match to the current publisher-hosted files, not a publisher-signed historical checksum.\n')
    add('| Dataset | Raw rows | Retained students | Retained interactions | Sessions | Distinct retained questions | Distinct retained skills | Raw/Parquet/store exact field equality |\n|---|---:|---:|---:|---:|---:|---:|---|')
    for d in DATASETS:
        obj=data[d];meta=json.loads((Path('data/processed')/d/'full/session_store/metadata.json').read_text())
        q=vocab[d]['vocabularies']['question']['distinct_retained_ids'];s=vocab[d]['vocabularies']['skill']['distinct_retained_ids']
        add(f"| {d} | {obj['source_lineage']['raw_rows']:,} | {meta['students']:,} | {meta['interactions']:,} | {meta['sessions']:,} | {q:,} | {s:,} | PASS |")
    add('\n**Retained distinct-count reporting: FAIL for ASSIST/EdNet if the existing fields are described as distinct counts.** The store uses maximum IDs for embedding capacity: ASSIST question bound3,158 includes46 unused IDs; EdNet question bound13,169 includes886 unused IDs, and skill bound1,495 includes5 unused IDs. The training runners copy these bounds into dataset_stats questions/skills fields, which feed the master table. The true distinct retained counts are3,112/102 (ASSIST),25,630/1,521 (Junyi),12,283/1,490 (EdNet). The checkpoint index capacities and seed42 exactly match the saved architecture and data. Correct the statistical description, not the saved embedding indices or EOS IDs; no retraining follows from this finding. See '+artifact('checkpoint_vocabulary_consistency.json')+' and '+source('ktbench/data/session_store.py','maximum_question = max')+'.\n')
    add('\nThe independent reconstruction does not call production preprocessing/session assignment. It validates original row identity, authoritative ASSIST action_num ties, Junyi source-row ties, EdNet timestamp/source-row ties, per-student boundaries, binary labels, question/skill mappings, cumulative attempt counters, session positions and every processed/store event. Raw and processed file SHA-256 inventories and historical published mapping/report hash comparisons are saved per dataset. See '+', '.join(artifact(f'{d}_data_manifest.json') for d in DATASETS)+'.\n')
    add('Sessions begin at a timestamp gap **≥10 hours**; students need ≥5 sessions. For n sessions, validation and test each receive floor(n/5), and training receives the earlier remainder. This differs from a universally exact60/20/20 percentage for small n. Timestamps are preserved in milliseconds; the flat sources are converted from seconds. No session is truncated/chunked. The paper describes end-to-start gaps >10h; the audited approved implementation uses start-to-start gaps ≥10h and is a documented protocol variant, not an exact paper reproduction.\n')
    add('ASSIST preserves the supplied row skill. Junyi uses each question’s deduplicated complete full-source tag set, sorted numerically, as one composite skill. This is label-free, transductive metadata construction, not train-only inductive vocabulary learning. EdNet independently joins question answers and complete tag sets, deduplicates 197 repeated-tag questions, assigns 1,495 composite skills, reserves `<UNTAGGED>` for tags−1, and excludes 27,646 unanswered rows without relabelling them incorrect. Their audit count was checked separately; exhaustive field equality above concerns supervised events.\n')
    add('### Split and target coverage\n')
    add('**The protocol is chronological within-student, not student-disjoint and not a global calendar cutoff. Every retained student appears in training, validation and test.** Validation/test histories use earlier responses from their own split as well as earlier splits; weights are fixed during evaluation. Whole held-out target sessions also teacher-force strictly earlier responses inside that session. Batch length sorting changes execution order, not which earlier labels are supplied. This is an online/prequential response-visibility assumption, not a one-shot label-hidden holdout.\n')
    add('| Dataset | HiTSKT train targets | Baseline train targets | First-session interactions context-only in HiTSKT | Validation targets | Test targets |\n|---|---:|---:|---:|---:|---:|')
    for d in DATASETS:
        s=data[d]['store_contract'];t=s['supervised_targets'];b=s['baseline_supervised_targets']
        add(f"| {d} | {t['train']:,} | {b['train']:,} | {s['first_session_interactions_context_only_for_hitskt']:,} | {t['validation']:,} | {t['test']:,} |")
    add('\nEvery intended post-first-session HiTSKT target is covered exactly once; first-session training interactions are deliberately context-only. Baselines train on all retained interactions. These are different training target populations, despite equal held-out target populations. Complete history-bound checks found no cross-student or later-index history. Later test sessions with earlier same-test history number 589/1,977 (ASSIST),79,641/109,506 (Junyi),358,279/474,827 (EdNet).\n')
    add('## 4. Response-availability limitation\n')
    add('| Dataset | Test events with predecessor end after current start | Same-session subset | Adjacent equal test timestamps | Status of timing check |\n|---|---:|---:|---:|---|')
    for d in DATASETS:
        a=availability[d]['by_target_split']['test']
        add(f"| {d} | {a['previous_response_end_after_target_start']:,} | {a['overlapping_targets_in_same_session']:,} | {a['equal_start_to_previous']:,} | {availability[d]['check_status']} |")
    add('\nASSIST uses recorded endTime; EdNet uses timestamp+elapsed_time as a completion **proxy**. The [official KT1 field description](https://github.com/riiid/ednet#kt1) defines timestamp as presentation time and elapsed_time as question time, and describes bundled questions. These fields do not independently certify the exact correctness-feedback arrival time. Junyi has no recorded end/start overlaps but 2,165,594 tied test timestamps, ordered by source row without an authoritative response-arrival index.\n')
    add('A bounded diagnostic selected eight distinct overlapping test sessions per ASSIST/EdNet checkpoint. Flipping the earlier-index predecessor response changed the current logit in all 8/8 cases in each dataset; maximum absolute changes were '+number(max(sensitivity['assist2017']['unavailable_predecessor_flip_logit_absolute_changes']))+' and '+number(max(sensitivity['ednet_kt1']['unavailable_predecessor_flip_logit_absolute_changes']))+'. Flipping the current target’s own label left its prediction unchanged in all cases. This is evidence that the online response-availability assumption matters; it is not proof of an acausal event-index attention graph, nor an assertion that every overlap has delayed feedback. See '+artifact('answer_availability_confirmed.json')+' and '+artifact('unavailable_response_sensitivity.json')+'.\n')
    add('The first timing artifact used an overly strong ASSIST interpretation equating endTime with label availability. That wording was corrected and the complete scan rerun; all numerical counts exactly match. The original '+artifact('answer_availability.json')+' and its source snapshot are preserved; the confirmed artifact and the qualified interpretation above supersede that sentence.\n')
    add('Strict presentation-time availability is therefore **NOT VERIFIED** for all three datasets. Under the specific rule “a label becomes available only at recorded end/completion”, ASSIST/EdNet fail the availability inequality. Actual submission/feedback timestamps or a validated alternative contract are needed before claiming real-time, unavailable-response-safe prediction.\n')
    add('## 5. Fresh-process HiTSKT checkpoint replay\n')
    add('The evaluator imports the unchanged model/collator but neither the production training `_run_epoch` nor the project metric helpers. It checks every batch’s raw-store labels/event IDs, every test session once, the full target count, finite logits/probabilities and original artifact hashes. Metrics use independent NumPy confusion counts, tied-pair AUC, MSE and stable float64 logit BCE. The historical FP32 batch-weighted BCE is also calculated separately.\n')
    add('For tied-score group g, AUC = Σ P_g(N_before_g + N_g/2)/(P·N). Classification is p≥0.5. BCE = mean(max(z,0)−y·z+log1p(exp(−abs(z)))). The float64 BCE need not be bitwise identical to a sequence of FP32 batch means; both comparisons are disclosed. Full ASSIST evaluation through the unchanged production CPU helper exactly agreed with the independent CPU evaluator for every metric, including historical BCE: '+artifact('assist2017_cpu_production_crosscheck.json')+'.\n')
    add('PyTorch explicitly does not promise bitwise CPU/GPU equality ([version2.4 numerical-accuracy documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.0/docs/source/notes/numerical_accuracy.rst)). That makes device arithmetic a plausible explanation, not an established diagnosis. Original per-target CUDA probabilities/logits were not persisted; a fresh CUDA replay was not attempted while protecting the ongoing GPU training; original GPU runtime/TF32 details are not fully pinned in saved configs. No mismatch is silently dismissed as rounding, and no stored result is overwritten.\n')
    add('| Dataset | Selected epoch / completed epochs | Width / heads / FF width | Adam learning rate | Parameters | Historical config |\n|---|---:|---|---:|---:|---|')
    for d in DATASETS:
        c=evaluations[d]['historical_config']
        add(f"| {d} | {c['best_epoch']} / {c['epochs_completed']} | {c['width']} / {c['heads']} / {c['feedforward_width']} | {c['learning_rate']:g} | {c['total_parameters']:,} | [config.json](../experiments/hitskt/{d}/full/config.json) |")
    add('\nAll use seed 42, one layer in each of the four hierarchy stages, dropout 0.1, weight decay 0, 15 previous sessions, token budget 32,768, maximum batch 64, strict validation-AUC improvement, patience 5, and best-checkpoint reload before test. Saved configs retain their original attention name; section 6 corrects its scientific interpretation.\n')
    for d in DATASETS:
        value=evaluations[d]
        add(f'### {d}\n')
        add(f"Manifest: {artifact(d+'_evaluation_manifest.json' if d=='assist2017' else d+'_evaluation_guarded_manifest.json')}. Replay comparison status: **{value['status']}**.\n")
        if 'recalculated_metrics' not in value:
            add('Required full metrics were not obtained. **NOT VERIFIED**; '+str(value.get('error','see manifest'))+'.\n')
            continue
        add(f"Checkpoint `experiments/hitskt/{d}/full/best_model.pt`, epoch **{value['best_checkpoint_epoch']}**, SHA-256 `{value['historical_artifact_sha256']['best_model.pt']}`. All **{value['recalculated_metrics']['targets']:,}** test targets evaluated. Config and every store-file hash are in the manifest.\n")
        add('| Metric | Originally stored | Independently recalculated on CPU | Absolute difference | Exact equality | 1e-6 gate |\n|---|---:|---:|---:|---|---|')
        for key in ['roc_auc','accuracy','precision','recall','f1','mse','historical_batch_weighted_bce','bce_float64']:
            item=value['comparison'][key]
            add(f"| {key} | {number(item['stored'])} | {number(item['recalculated'])} | {number(item['absolute_difference'])} | {'PASS' if item['exact_equal'] else 'FAIL'} | {'PASS' if item['within_tolerance'] else 'FAIL'} |")
        add('\nExact original-CUDA reproduction is **NOT VERIFIED**; these are complete CPU measurements, not substituted historical scores.\n')
    add('## 6. Architecture terminology and fidelity\n')
    add('**Deterministic ELU+1 causal linear attention: PASS. Faithful FAVOR+ Performer claim: FAIL.** '+source('ktbench/models/performer.py','q = F.elu')+' computes learned Q/K/V projections, fixed elementwise φ(x)=ELU(x)+1, masked key/value prefixes, normalization with denominator floor1e-6, then output projection. The effective causal kernel weight is φ(q_t)ᵀφ(k_j) for j≤t. It has no random-feature matrix, orthogonal feature construction or random-feature count.\n')
    add('This is the causal linear-transformer formulation in [Katharopoulos et al.,2020, equations7–12](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf). [Choromanski et al., Performer/FAVOR+](https://arxiv.org/abs/2009.14794) instead uses positive orthogonal random features to estimate an attention kernel, including softmax attention. A learned Q/K projection followed by ELU+1 is not that estimator.\n')
    add('Mathematically, the implementation exactly uses κ_ELU(q,k)=Σ_r(ELU(q_r)+1)(ELU(k_r)+1). FAVOR+ uses stochastic finite-dimensional features φ_Ω whose inner product estimates a desired kernel such as exp(qᵀk/√d); feature sampling and approximation error are part of that method. The kernels are different: for one-dimensional q=1 and keys0,1, normalized ELU+1 weights are(1/3,2/3), whereas softmax weights are(1/(1+e),e/(1+e)).\n')
    add('Its causal computation is S_t=Σ_{j≤t}m_j φ(k_j)v_jᵀ and z_t=Σ_{j≤t}m_j φ(k_j), followed by o_t=φ(q_t)ᵀS_t / max(φ(q_t)ᵀz_t,10⁻⁶), output projection and query masking. The explicit reference constructs all permitted query/key weights directly instead of reusing either cumulative sum.\n')
    add('The hierarchy is retained, but this is a **HiTSKT-inspired hierarchical model with deterministic ELU+1 causal linear attention**. It also differs from the paper/legacy implementation in rehearsal/question-attempt embeddings, session power-law decay and block formulation: '+source('model_layers.py','qno_embedding =')+' and '+source('attention_modules.py','AB_line_affect =')+' show components absent from the active model; active blocks use pre-normalization/GELU. The top-level legacy `main.py` still imports the old quadratic-softmax implementation, while canonical audited checkpoints use `scripts/train_hitskt.py` → `ktbench/models/hitskt.py`. These paths must not be conflated.\n')
    add('No architecture was changed for this audit. Correcting terminology needs no retraining. A claim specifically about a genuine FAVOR+ model, or a faithful paper-architecture reproduction, requires separately implemented and trained models; historical ELU+1 checkpoints cannot establish it.\n')
    add('## 7. Baselines, completeness and recorded gains\n')
    add('All 14 completed experiment directories pass best-checkpoint/strict-validation-selection checks, CSV/JSON test consistency and finite best-model tensors. All 295 epoch records from those 14 completed runs have finite train/validation metrics. DKT/EdNet is unfinished and cannot supply a final benchmark row. Evidence: '+artifact('benchmark_artifacts.json')+', '+artifact('epoch_metric_finiteness.json')+', '+artifact('publication_consistency.json')+' and '+artifact('dkt_ednet_pending_state.json')+'. Baseline target-label/PAD exclusion is independently tested for DKT, DKVMN, SAKT and RKT; full baseline test metrics were **NOT VERIFIED by fresh replay** in this audit.\n')
    add('| Model | Context / input distinction | Audit status |\n|---|---|---|')
    add('| DKT | 199 prior interactions, skill responses, width64 LSTM | Artifact/masking PASS; EdNet final result NOT VERIFIED |')
    add('| DKVMN | 199 prior interactions, skill-based memory, 20 slots | Artifact/masking PASS; fresh metric replay NOT VERIFIED |')
    add('| SAKT | 99 prior interactions, question+skill attention, width200/5 heads | Artifact/masking PASS; fresh metric replay NOT VERIFIED |')
    add('| RKT | 49 prior interactions, question/Phi/time/student parameters, width64 | Artifact/masking/cache algebra PASS; fresh metrics/raw Phi pair-count rebuild NOT VERIFIED |')
    add('| HiTSKT variant | 15 complete earlier sessions plus prior target-session responses | Structural checks PASS; full replay outcomes above |\n')
    add('RKT is explicitly the performance-only Phi variant, omitting question-text relations. Persisted five-fold student assignments exactly match seed42; every all-train count entry equals the sum of five folds, and every complement exactly equals all-minus-own-fold. Source limits construction to training partitions and chooses own-fold exclusion for training rows. The audit did not independently regenerate all raw-label pair counts, so that provenance remains NOT VERIFIED. These five-fold caches do not make the benchmark train/test split student-disjoint. See '+artifact('rkt_cache_algebra.json')+' and '+source('ktbench/rkt/phi.py','def _student_training_bounds')+'.\n')
    add('The additional '+artifact('rkt_cache_key_bounds.json')+' exhaustively checks sparse-key ordering/range, membership, cached-key CRC identity and full-population/target-count metadata for all three caches. Algebraic consistency cannot substitute for independently rebuilding their label-derived counts.\n')
    add('| Dataset | Recorded HiTSKT AUC | Best available baseline | Recorded baseline AUC | Recorded difference | Significance |\n|---|---:|---|---:|---:|---|')
    for d in DATASETS:
        g=baseline['recorded_gains'][d]
        add(f"| {d} | {number(g['hitskt_auc'])} | {g['best_available_baseline']} | {number(g['best_available_baseline_auc'])} | {number(g['auc_difference'])} | NOT VERIFIED |")
    add('\nThe Junyi/EdNet differences are descriptive **seed 42** results, not demonstrated statistically significant gains or isolated effects of Performer/FAVOR+. EdNet lacks DKT. Models have different features, contexts, parameter counts, training target populations and established hyperparameters; common data splits, test populations, threshold and validation selection help comparison but do not create a controlled architectural ablation. ASSIST HiTSKT does not beat the strongest recorded baseline.\n')
    add('## 8. Issues, severity, affected results and retraining implications\n')
    add('| Issue | Severity / status | Affected evidence | Is retraining genuinely required? |\n|---|---|---|---|')
    issues=[
        ('Unresolved CPU vs stored metric discrepancies','HIGH / all three CPU metric gates FAIL; exact CUDA NOT VERIFIED','Historical HiTSKT numbers requiring exact confirmation','Not established. First resolve source/runtime/per-target differences; a CPU mismatch alone does not prove training was wrong.'),
        ('Response feedback availability not certified; recorded/proxy overlaps and timestamp ties','HIGH / NOT VERIFIED real-time contract','All models using earlier responses; ASSIST/EdNet timing inequality FAIL, Junyi order limited by ties','Conditional: if the scientific contract requires availability-aware training contexts, rebuild contexts and retrain affected methods. Mere event-index replay does not require retraining.'),
        ('ELU+1 described as pure/faithful Performer','HIGH / FAIL terminology','All three canonical HiTSKT architecture claims','No for accurate ELU+1 naming; yes for a new FAVOR+ claim/model.'),
        ('Additional differences from paper HiTSKT, including missing rehearsal/power-law components','MEDIUM / exact-paper fidelity NOT VERIFIED','Interpretation of current variant and comparisons with paper results','No for disclosed variant results; yes for a claim about a different faithful architecture.'),
        ('Start-to-start gaps ≥10h differ from the paper’s end-to-start gaps >10h','MEDIUM / exact paper-protocol equivalence FAIL','Session construction and cohort/split interpretation for all three datasets','No for the disclosed approved variant; an exact paper-protocol claim requires rebuilding that protocol and training/evaluating its models.'),
        ('Only seed42; no paired uncertainty or multiple-seed evidence','MEDIUM / significance NOT VERIFIED','Junyi and EdNet superiority claims','No to preserve descriptive historical results; supporting experiments are required for stronger generalization/significance claims.'),
        ('First-session training targets excluded only for HiTSKT','MEDIUM / equal-training-population claim FAIL','All cross-model scientific comparisons','No automatic invalidation; matched-population attribution would require controlled retraining.'),
        ('Baseline metrics not freshly replayed; RKT raw label-pair counts not independently rebuilt','MEDIUM / NOT VERIFIED beyond scoped artifact/masking/cache checks','DKT, DKVMN, SAKT and RKT historical scores','No established need. Recompute/rebuild the missing evidence first; changed cached relations could require RKT retraining.'),
        ('Vocabulary index capacity presented as distinct retained question/skill counts','MEDIUM / distinct-count interpretation FAIL','ASSIST/EdNet dataset_stats and master tables: true counts shown in section3','No. Correct/relabel publication statistics while retaining checkpoint IDs/EOS capacities.'),
        ('DKT/EdNet incomplete; last9 Adam has21,749 positive infinities','HIGH / final result NOT VERIFIED','Fifteenth run only; best1 weights remain finite','Recovery/retraining decision unresolved. Do not silently reset optimizer or publish an invented final row.'),
        ('DKT checkpoint embeds patience5 while active config is authorized10','LOW / metadata inconsistency','DKT/EdNet future continuation; actual counter8/10','No retraining from this metadata alone; preserve active config/counter and correct future metadata only with a scoped change.'),
        ('Flat inputs are author-preprocessed CSVs; publisher-raw transformation chain absent','MEDIUM / deeper provenance NOT VERIFIED','Claims of untouched publisher-original data','No if the declared input is the verified author distribution; reconsider results only if a different required source is established.'),
        ('Historical configs lack an immutable full-data/runtime provenance bundle','MEDIUM / historical exact environment NOT VERIFIED','Exact training/replay provenance beyond present hashes','Not by itself. Current raw reconstruction and replay hashes improve future evidence but cannot retroactively prove every historical environment detail.'),
        ('Legacy entrypoint remains quadratic softmax and fixed-window preprocessing','MEDIUM / path-confusion risk','Legacy runs, excluded from canonical audited result set','No for canonical variant checkpoints; legacy results cannot inherit this audit.'),
        ('Final complete figure/report bundle absent','MEDIUM / NOT VERIFIED','reports/report_manifest.json and final15-run figures are absent in this checkout','No retraining implication by itself; benchmark remains incomplete.'),
        ('Stale committed README completion counts','LOW / documentation inconsistency','README at audit base says13 completed/all11 validated; master has14','No. Reconcile documentation without overwriting the unrelated local README edit.'),
        ('Initial audit fixture/timing-interpretation errors and two audit-only OOM kills','INFORMATIONAL / invalid attempts preserved and corrected','Audit process; completed bounded checks supersede partial attempts','No model retraining. Original experiments were preserved; protected trainer survival was checked.')]
    for row in issues:add('| '+' | '.join(row)+' |')
    add('\nNo retraining was launched. No evidence from a passing unit test, an interrupted job, a previous “verified” JSON, or a partial replay was promoted to an independently reproduced final result.\n')
    add('## 9. Evidence and reproducibility\n')
    add('Relevant implementations: '+source('scripts/audit_hitskt_data.py','class EventStream')+', '+source('scripts/audit_hitskt_replay.py','def independent_metrics')+', '+source('scripts/audit_hitskt_replay_parallel.py','def evaluate_batch')+', '+source('scripts/audit_memory_guard.py','def admit')+'. Prediction/logit/label/event-ID arrays and decompressed cache scratch remain local under `.inspection/independent_correctness_audit/`; their hashes and paths are recorded, and raw interactions are not published.\n')
    add('Commands were executed with `PYTHONPATH=.` and GPU visibility disabled. Replays refuse to overwrite existing output manifests/caches. The examples below require a fresh audit checkout/cache namespace as well as a new output manifest, because prediction-cache names are fixed per dataset; preserve existing evidence. Original files are protected by the final '+artifact('preservation_verification.json')+'. Audit-source bindings, test-artifact hashes and all 12 full prediction-array hashes are checked in '+artifact('evidence_integrity.json')+'.\n')
    add('```bash\nCUDA_VISIBLE_DEVICES=\'\' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. python -m pytest -q tests/test_independent_hitskt_audit.py tests/test_audit_tool_oracles.py\nCUDA_VISIBLE_DEVICES=\'\' PYTHONPATH=. python scripts/audit_hitskt_data.py assist2017 --output NEW_ASSIST_DATA_MANIFEST.json\nCUDA_VISIBLE_DEVICES=\'\' PYTHONPATH=. python scripts/audit_hitskt_replay.py assist2017 --threads 2 --output NEW_ASSIST_REPLAY_MANIFEST.json\n```\n')
    # These recommendations are emitted only after all full replays and the
    # preservation prerequisite above have completed; no new run is launched.
    add('## 10. Follow-up recommendations after completed verification\n')
    add('1. Resolve the CPU/stored discrepancies before claiming exact reproduction: when the protected GPU workload permits, replay the preserved checkpoints with the historical GPU/runtime and batch sequence, pin matrix-precision settings, and retain per-target logits/IDs for a direct comparison. This is verification, not an established reason to retrain.\n')
    add('2. Establish the prediction-time/feedback-arrival contract from authoritative event documentation. If labels are unavailable under that contract, build an availability-aware protocol and evaluate it; retrain affected methods if their training contexts must change. Keep the current within-student, retrospective cohort and online earlier-label protocol explicitly named.\n')
    add('3. Freshly replay completed baseline checkpoints and independently regenerate RKT raw training-only pair counts. Diagnose DKT/EdNet’s nonfinite Adam state and agree a recorded recovery policy before any separately authorized continuation. Do not treat its best validation score as a final test result.\n')
    add('4. For stronger Junyi/EdNet superiority claims, run multiple training seeds and report paired student-cluster uncertainty. Control features, history, parameter budget and training target population when attributing gains to a kernel. A genuine FAVOR+ comparison requires a separately implemented and trained variant; preserve the ELU+1 historical model.\n')
    add('5. Use the corrected retained distinct counts and architecture name in publication material. Add student-disjoint evaluation only if making a new-student generalization claim. Finish the missing benchmark evidence before presenting a complete 15/15 comparison. No additional experiment or training was started by this audit.\n')
    rendered='\n'.join(out)
    for before,after in {
        'lengths1/2/7/17':'lengths 1/2/7/17', 'histories1/3/4/15':'history counts 1/3/4/15',
        'lengths1/2/5/11':'lengths 1/2/5/11','batch3':'batch size 3',
        'atol2e-11':'atol 2e-11','atol5e-11':'atol 5e-11',
        'contains133':'contains 133','is1.':'is 1.','Table2':'Table 2',
        'states95,':'states 95,','bound3,158':'bound 3,158','includes46':'includes 46',
        'bound13,169':'bound 13,169','includes886':'includes 886','bound1,495':'bound 1,495',
        'includes5':'includes 5','are3,112':'are 3,112','seed42':'seed 42',
        'section3':'section 3','section4':'section 4','version2.4':'version 2.4',
        'al.,2020':'al., 2020','equations7':'equations 7','floor1e-6':'floor 1e-6',
        'keys0,1':'keys 0,1','are(1/3':'are (1/3','are(1/(':'are (1/(',
        'all14':'all 14','last9':'last epoch 9','has21,749':'has 21,749',
        'best1':'best epoch 1','patience5':'patience 5','authorized10':'authorized 10',
        'counter8/10':'counter 8/10','final15-run':'final 15-run','exact60/20/20':'exact 60/20/20',
        'width64':'width 64','width200':'width 200','says13':'says 13',
        'all11':'all 11','has14':'has 14',')25,630':') 25,630',
        '),25,630':'), 25,630','),12,283':'), 12,283','),79,641':'), 79,641','),358,279':'), 358,279',
    }.items():rendered=rendered.replace(before,after)
    Path('reports/HITSKT_INDEPENDENT_CORRECTNESS_AUDIT.md').write_text(rendered)
    print('Wrote reports/HITSKT_INDEPENDENT_CORRECTNESS_AUDIT.md')


if __name__=='__main__':main()

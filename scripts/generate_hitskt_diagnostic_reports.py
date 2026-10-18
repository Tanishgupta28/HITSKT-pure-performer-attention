#!/usr/bin/env python3
"""Render follow-up reports from immutable aggregate diagnostic evidence."""
from pathlib import Path
from scripts.diagnose_hitskt_common import BASE,DATASETS,OLD,OUT,read,write,sha,now,old_evaluation

NAMES=dict(assist2017='ASSIST2017',junyi='Junyi',ednet_kt1='EdNet-KT1')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def f(x):return format(x,'.16g')


def save(name,body):
    p=Path('reports')/name
    if p.exists():raise FileExistsError(p)
    p.write_text(body.rstrip()+'\n')


def main():
    metrics=read(OUT/'cached_metrics_diagnostics.json')['records']
    artifacts=read(OUT/'artifact_and_dkt_forensics.json')
    precision={d:read(OUT/f'{d}_fp64_{"full" if d=="assist2017" else "sample"}.json') for d in DATASETS}
    timings={d:read(OUT/f'{d}_timing_context.json') for d in DATASETS}
    order=read(OUT/'evaluation_order.json')['records']
    for d in DATASETS:
        assert precision[d]['status']==timings[d]['status']=='COMPLETE'
        assert order[d]['status']=='PASS'
    # Separate per-dataset follow-up manifests, without replacing prior evaluation manifests.
    for d in DATASETS:
        inputs=[OUT/'cached_metrics_diagnostics.json',OUT/'artifact_and_dkt_forensics.json',OUT/'evaluation_order.json',
            OUT/f'{d}_timing_context.json',OUT/f'{d}_fp64_{"full" if d=="assist2017" else "sample"}.json']
        write(OUT/f'{d}_diagnostic_manifest.json',dict(dataset=d,utc=now(),base_audit_commit=BASE,
            stored_metrics=old_evaluation(d)['stored_metrics'],cpu_metrics=old_evaluation(d)['recalculated_metrics'],
            comparison=metrics[d]['stored_vs_cpu_comparison'],cpu_verification='FAIL: frozen absolute1e-6 metric gate',
            cuda_verification='NOT VERIFIED: occupied MIG; no CUDA allocated',response_availability='NOT VERIFIED',
            historical_cuda_pointwise_max_errors='NOT VERIFIED: historical vectors absent',
            historical_cuda_changed_classifications='NOT VERIFIED: aggregate bounds only',
            classification_bounds=metrics[d]['original_cuda_vs_cpu_classification_bounds'],
            cpu_precision_diagnostics=precision[d]['cpu_fp32_vs_fp64'],precision_scope=precision[d]['mode'],
            recommendation='2. Results require additional verification but not yet retraining',
            real_time_claim='Unsupported without a separately verified availability-aware protocol',
            input_sha256={str(p):sha(p) for p in inputs},implementation_sha256=sha(__file__)))

    root='''# HiTSKT reproducibility discrepancy investigation

Date: 2026-10-18 UTC. Base audit: `5625478ab0be344b145d2813939a71a0f6e49924`.

**Outcome: all three historical-to-CPU metric gates remain FAIL at the original
absolute tolerance of 1e-6.** AUC passes that tolerance for every dataset.
ASSIST2017's BCE discrepancy survives both high-precision metric accumulation
and a full float64 model-forward diagnostic. The exact originating CUDA
operation/settings remain **NOT VERIFIED**. No tolerance was changed, result
overwritten, architecture changed, training started or external GPU job interrupted.

This follows [the immutable first audit](HITSKT_INDEPENDENT_CORRECTNESS_AUDIT.md).
New evidence lives in [reproducibility_diagnostics](reproducibility_diagnostics/).
The original all-target CPU predictions are hash-verified inputs, not newly
invented CUDA predictions. Fresh inference in this follow-up consists of the
full ASSIST precision diagnostic and bounded Junyi/EdNet samples.

## Historical results versus independent full CPU replay

All numbers below are proportions/losses, not percentages. BCE has two CPU
forms: independent stable float64 logit BCE, and the original FP32 batch-mean
loss weighted by valid-target count. The latter matches the production
definition in [scripts/train_hitskt.py](../scripts/train_hitskt.py), lines43–95.
PASS means absolute difference <=1e-6; it does not mean bitwise equality.
None of the stored-versus-CPU metrics is exactly equal.

'''
    metric_keys=[('AUC','roc_auc'),('Accuracy','accuracy'),('Precision','precision'),('Recall','recall'),
        ('F1','f1'),('MSE','mse'),('BCE (stable float64 logits)','bce_float64'),('BCE (historical FP32 batch definition)','historical_batch_weighted_bce')]
    for d in DATASETS:
        old=old_evaluation(d);m=metrics[d]
        root+=f'### {NAMES[d]} — CPU FAIL; CUDA NOT VERIFIED\n\n'
        root+=f'All **{old["expected_targets"]:,}** test targets; selected epoch **{m["selected_epoch"]}**. '
        root+=f'Checkpoint `experiments/hitskt/{d}/full/best_model.pt`, SHA-256\n`{m["checkpoint_sha256"]}`.\n\n'
        root+=table(['Metric','Stored CUDA','Full CPU','Absolute difference','Gate'],[
            [label,f(m['stored_vs_cpu_comparison'][key]['stored']),f(m['stored_vs_cpu_comparison'][key]['recalculated']),
             f(m['stored_vs_cpu_comparison'][key]['absolute_difference']),'PASS' if m['stored_vs_cpu_comparison'][key]['within_tolerance'] else 'FAIL']
            for label,key in metric_keys])+'\n'
        root+=f'Evidence: [{d}_diagnostic_manifest.json](reproducibility_diagnostics/{d}_diagnostic_manifest.json); '
        suffix='evaluation_manifest.json' if d=='assist2017' else 'evaluation_guarded_manifest.json'
        root+=f'[original complete CPU manifest](independent_correctness_audit/{d}_{suffix}).\n\n'
    root+='''## Exactly what can be recovered about classification changes

The original CUDA runs retained aggregate metrics, not per-target logits,
probabilities or classifications. Searching the experiment/inspection evidence
located CPU audit vectors only. Therefore the **exact original CUDA-versus-CPU
number of changed classifications, and maximum logit/probability difference,
are NOT VERIFIED for all three datasets**. Aggregate accuracy differences are
net changes in correct predictions, not total flips. Matching aggregates in a
future CUDA replay would still not prove its vectors equal the unsaved original
vectors; report fresh-CUDA-versus-CPU differences with that scope.

Integer confusion matrices can be recovered consistently from the stored
accuracy/precision/recall/F1 and verified label population. Equivalent formulas
are checked within eight representation ULPs solely to reconstruct integer
counts; this does not alter the 1e-6 acceptance criterion. Counts are TP/FP/TN/FN.

'''
    rows=[]
    for d in DATASETS:
        m=metrics[d];b=m['original_cuda_vs_cpu_classification_bounds']
        rows.append([NAMES[d],'/'.join(str(m['historical_confusion_inferred'][k]) for k in ['tp','fp','tn','fn']),
            '/'.join(str(m['cpu_confusion'][k]) for k in ['tp','fp','tn','fn']),b['net_correct_change'],
            f'{b["minimum"]:,}–{b["maximum"]:,}', 'odd' if b['parity'] else 'even'])
    root+=table(['Dataset','Stored confusion (inferred)','CPU confusion','Net correct delta','Confusion-only possible flips','Parity'],rows)+'\n'
    root+='''These are sharp bounds from confusion counts alone, not an assertion that
the maximum occurred. At least **5 / 31 / 40** classifications differ, respectively.
The independent exhaustive small-population oracle verifies the bounds and
parity. See [cached_metrics_diagnostics.json](reproducibility_diagnostics/cached_metrics_diagnostics.json)
and [the diagnostic tests](../tests/test_reproducibility_diagnostics.py).

## Probability rounding on identical CPU logits

Recomputing sigmoid in float64 for every cached float32 logit isolates sigmoid
and metric arithmetic from model-forward differences. The historical threshold
remains float32 probability >=0.5. It was not replaced with a logit threshold.

'''
    root+=table(['Dataset','Targets','Max absolute probability difference','Exact classification changes','Cached p=0.5','Within 1e-5 of 0.5'],[
        [NAMES[d],f'{old_evaluation(d)["expected_targets"]:,}',f(metrics[d]['all_target_probability_rounding']['maximum_absolute_probability_difference']),
         metrics[d]['all_target_probability_rounding']['changed_threshold_classifications'],metrics[d]['all_target_probability_rounding']['p32_equal_half'],
         metrics[d]['all_target_probability_rounding']['probability_margin_counts']['1e-05']] for d in DATASETS])+'\n'
    root+='''EdNet's four cached p=0.5 cases have negative, nonzero logits. Float64
sigmoid moves all four below0.5. This proves a quantization effect in this
specific counterfactual; it does not explain the exact original CUDA/CPU flips.
ASSIST and Junyi have no classification changes from sigmoid precision alone.

## Model arithmetic and determinism

Fresh processes reloaded the unchanged saved float32 weights. A separate model
cast those weights to float64; evaluation used original batches, .eval(), no
gradients and no autocast. The source-generated sinusoidal position tables are
still computed in float32 before casting, as prescribed by the unchanged
implementation, so this is a higher-precision network-forward diagnostic with
the same positional templates, not a new full-precision model specification.

ASSIST covers every test target. Junyi/EdNet use six preselected original batches
each (including the batch containing the globally smallest cached absolute
logit). They are diagnostic samples, not random samples or full reproduced test
results; none was skipped. Do not extrapolate sampled maxima or flip counts.

'''
    root+=table(['Dataset','Evaluated targets / scope','Max absolute logit difference','Max absolute probability difference','Exact CPU32→CPU64 classification changes'],[
        [NAMES[d],f'{precision[d]["actual_targets"]:,} / {"ALL" if d=="assist2017" else "6 sampled batches"}',
         f(precision[d]['cpu_fp32_vs_fp64']['maximum_absolute_logit_difference']),f(precision[d]['cpu_fp32_vs_fp64']['maximum_absolute_probability_difference']),
         precision[d]['cpu_fp32_vs_fp64']['changed_threshold_classifications']] for d in DATASETS])+'\n'
    root+='''EdNet's sampled change is positive-to-negative; ASSIST/Junyi have zero.
All repeated float32 forwards at unchanged settings are bitwise identical in
the probed batches. Changing CPU thread count from2 to1 or disabling MKLDNN
does change logits slightly, demonstrating deterministic arithmetic depends on
the backend/settings. Junyi/EdNet's earlier full CPU capture used one thread;
these fresh probes used two. This is not random evaluation or evidence of
target misalignment. No claim of global bitwise CPU determinism is inferred
from sampled repeats.

'''
    root+=table(['Dataset','Probed batches','Max 1-vs-2-thread logit difference','Max MKLDNN-off logit difference','Max fresh FP32 vs cached logit difference'],[
        [NAMES[d],len(precision[d]['fp32_determinism_backend_probes']),
         f(max(p['threads1_vs2_maximum_logit_difference'] for p in precision[d]['fp32_determinism_backend_probes'])),
         f(max(p['mkldnn_off_maximum_logit_difference'] for p in precision[d]['fp32_determinism_backend_probes'])),
         f(max(p['fp32_cache_maximum_absolute_logit_difference'] for p in precision[d]['fp32_determinism_backend_probes']))] for d in DATASETS])+'\n'
    root+='''Stage hooks in ASSIST's first three batches locate ordinary CPU32/CPU64
differences throughout ActionEncoder, SessionEncoder, CorrectPaddingEncoder,
Decoder and prediction; their maxima are recorded in the full precision
manifest. These are within-CPU diagnostics, not a localization of the missing
historical CUDA discrepancy. Detailed settings and scopes:
[ASSIST full](reproducibility_diagnostics/assist2017_fp64_full.json),
[Junyi sample](reproducibility_diagnostics/junyi_fp64_sample.json),
[EdNet sample](reproducibility_diagnostics/ednet_kt1_fp64_sample.json).

## ASSIST2017 BCE: ruled-out explanations and unresolved cause

'''
    decomp=metrics['assist2017']['bce_decomposition'];fp64=precision['assist2017']['metrics']['bce_float64']
    root+=table(['Computation','BCE'],[
        ['Historical CUDA',f(decomp['historical_cuda'])],['CPU historical FP32 batch weighting',f(decomp['cached_cpu_historical_batch_weighted_fp32'])],
        ['Same CPU logits, stable float64 accumulation',f(decomp['cached_cpu_float64_logit_bce'])],
        ['Same CPU logits, single global FP32 reduction (counterfactual)',f(decomp['cached_cpu_single_global_fp32_bce'])],
        ['Fresh float64 network forward, all targets',f(fp64)]])+'\n'
    root+='''The CPU batch-FP32 minus stable-FP64 difference is **1.4903841494273706e-9**,
while the historical-minus-batch-CPU gap is **2.1306950720445528e-6**.
Changing only accumulation precision leaves essentially the whole discrepancy.
Full double network inference changes stable CPU BCE by approximately9.42e-10,
and still misses historical BCE by **2.1312437579190657e-6**. It changes zero
classifications. Thus neither of these CPU precision changes resolves the strict
failure. A single global FP32 reduction also fails to close the gap.

An upstream CUDA numerical path/settings difference is a plausible hypothesis,
not a proven root cause. Missing original pointwise scores and runtime flags
prevent identifying a specific historical operation. The issue cannot honestly
be marked resolved as harmless rounding or relabelled PASS.

## Runtime, batch ordering and CUDA verification

The current runtime is PyTorch `2.4.0a0+f70bd71a48.nv24.06`, CUDA build12.5,
cuDNN90100, NumPy1.24.4. Fresh CPU processes report float32 matmul precision
`high`, `cuda.matmul.allow_tf32=True`, `cudnn.allow_tf32=True`, MKLDNN enabled.
These flags were read without allocating CUDA. The project does not explicitly
set TF32/matmul precision; historical artifacts do not capture those actual
values. Upstream release defaults must not be substituted for this NVIDIA
build's observed defaults or assumed to prove historical settings.

[ktbench/config.py](../ktbench/config.py), lines11–31, sets workspace config
`:4096:8` only if unset, seeds42, requests deterministic algorithms with
`warn_only=True`, disables cuDNN benchmarking and enables cuDNN determinism.
Deterministic algorithms do not imply cross-device bit identity. HiTSKT uses
linear projections, normalization, activations and prefix reductions; it has no
convolution/LSTM, so changing a cuDNN flag alone is not an established explanation.
[PyTorch's numerical-accuracy note](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.0/docs/source/notes/numerical_accuracy.rst)
and [CUDA precision documentation](https://docs.pytorch.org/docs/2.4/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)
support investigating hardware and reduction paths, not excusing a failed gate.

'''
    root+=table(['Dataset','Regenerated batches','Exact IDs/labels/cache order','Validation-selected epoch / completed'],[
        [NAMES[d],order[d]['batches'],'PASS',f'{metrics[d]["selected_epoch"]} / {old_evaluation(d)["historical_config"]["epochs_completed"]}'] for d in DATASETS])+'\n'
    root+='''All test event IDs, labels, uniqueness and regenerated batch memberships
match the prior full CPU capture, using historical token budget32768, batch cap64,
bucket512, shuffleFalse, sampler epoch0, seed42. See [evaluation_order.json](reproducibility_diagnostics/evaluation_order.json).
Original CUDA batch vectors were not retained. The unchanged source constructs
the same evaluation sampler; a historical backend-specific forward path remains
unverified. All three best hashes, validation selection, original dataset/store
hashes and target populations are preserved; [artifact_and_dkt_forensics.json](reproducibility_diagnostics/artifact_and_dkt_forensics.json)
and the per-dataset first-audit manifests give the exact bindings.

Historical configs identify **H10080GB, MIG3g.40gb**. The only visible partition
is that same architecture and is occupied; LightSAKT2 has an active job, and
compute-process visibility is restricted. No CUDA allocation was attempted.
**CUDA verification: NOT VERIFIED for all datasets**, not a failed CUDA replay
and not a successful matching-architecture reproduction. Evidence:
[resource_final.json](reproducibility_diagnostics/resource_final.json).

Once a genuinely idle independent partition is established, the next bounded
verification is a fresh process with the original batch shapes and source, H100
MIG architecture, recorded NVIDIA runtime, seed42 and original evaluation
definition. Capture actual TF32/cuBLAS/cuDNN settings rather than assume them;
first replay the best-supported historical settings without tuning on test
outcomes. Save local event-indexed logits/probabilities and two identical repeats,
hash all inputs, compare every stored metric using the unchanged1e-6 gate, and
count fresh-CUDA-versus-CPU pointwise differences exactly. A separately labelled
TF32-disabled diagnostic may isolate numerical sensitivity but cannot silently
replace historical settings. If historical flags cannot be established, report
the reconstruction limitation even if aggregate metrics match. Never stop
LightSAKT2 to obtain this evidence.

## Verification and preservation

**59 relevant tests PASS** in this follow-up: the existing49 independent
correctness/audit-oracle tests plus10 new exhaustive diagnostic cases. This
includes explicit causal output/gradient, masking and future-response checks;
passing them does not certify response availability or historical metrics.
[verification_tests.log](reproducibility_diagnostics/verification_tests.log)
records the executed scope. The first audit's broader133-test result remains
preserved, not presented as a new133-test run.

Initial and final preservation checks cover all375 original captured files and
106 prior audit report/evidence/script/test files. See
[preservation_initial.json](reproducibility_diagnostics/preservation_initial.json)
and [preservation_final.json](reproducibility_diagnostics/preservation_final.json).
No original checkpoint/result or production scientific file was changed.
Ignored local arrays contain learner-level scores/IDs; only aggregate evidence
and hashes enter Git. The actual cgroup limit is90GiB; model inference was serial
with CUDA hidden, two CPU threads, low priority and a16GiB forward-allocation
estimate guard. The preexisting OOM-kill counter remained2, with no new OOM kill
observed. This estimate is not a claim of a measured16GiB total RSS peak.

The defensible decision is **additional verification, not automatic retraining**
for historical V1 on each dataset. Retain original evidence provisionally,
accompany CPU metrics with their environment, and do not describe the historical
metric set as independently reproduced. Availability and publication decisions
are addressed in [the timing analysis](HITSKT_RESPONSE_TIMING_ANALYSIS.md) and
[the research decision report](HITSKT_RESEARCH_DECISION_REPORT.md).
'''
    save('HITSKT_REPRODUCIBILITY_ROOT_CAUSE.md',root)

    timing='''# HiTSKT response-timing analysis

Date: 2026-10-18 UTC. This extends audit `5625478`; no V1 protocol was changed.
**Sequence-index causality PASS; actual correctness-response availability NOT
VERIFIED for all three datasets.** Recorded/proxy overlaps are not themselves
confirmed leakage. Conversely, earlier event index is not proof of availability.

## Four different correctness claims

| Claim | Meaning | Evidence/status |
| --- | --- | --- |
| Sequence-index causal correctness | No own/future indexed response affects an earlier prediction | PASS for supported production shapes/masks; independent forward/gradient and perturbation tests |
| Response availability | Every consumed correctness label was available to the prediction service before its decision event | NOT VERIFIED; presentation/submit/feedback are not interchangeable |
| Chronological within-student evaluation | Each retained learner contributes earlier train, then validation, then test sessions | PASS in reconstructed V1; not student-disjoint or a global-calendar split |
| Retrospective teacher-forced evaluation | Fixed weights predict each indexed target while earlier indexed labels, including earlier validation/test labels, enter history | This is the implemented V1 estimand; it does not certify real-time deployment |

The current target is not supplied to its own prediction. Earlier held-out
labels are used as context, without test-time optimizer updates. Those two
facts can coexist with unverified wall-clock response timing. A server might
know correctness at answer submission if it has the answer key, before a
learner sees feedback. Neither row order nor a problem-end field proves that
server-side release event for these transformed records.

## Authoritative documentation and limits

**ASSIST2017.** The [official competition dataset page](https://sites.google.com/view/assistmentsdatamining/dataset)
links a [column dictionary](https://docs.google.com/spreadsheets/d/1QVUStXiRerWbH1X0P11rJ5IsuU2Xutu60D1SjpmTMlk/edit).
It defines start/end as problem activity times in Unix seconds and timeTaken
as a step duration; it does not specify correctness-release time. The accepted
author CSV drops action/hint/scaffold fields. The [general ASSIST glossary](https://sites.google.com/site/assistmentsdata/an-explanation-on-how-to-interpret-our-data-sets)
distinguishes first answer/help from tutoring completion and scaffolding,
but does not establish every transformed2017 row's timing. An earlier label
could therefore be known before the recorded problem end; the overlap is not
proof it was unavailable.

**EdNet.** The [publisher's dataset specification](https://github.com/riiid/ednet)
defines KT1 timestamp as question presentation and elapsed_time as duration.
Questions share bundles. KT2 records entry, revisable choices and final bundle
submission; KT3 adds later explanations. KT1 omits those action-level release
events. Therefore timestamp+elapsed_time is a completion proxy, not a verified
submission/feedback timestamp. At initial bundle presentation, using another
item's eventual final correctness is not a legitimate pre-bundle information
set. KT1 alone does not establish that every later item row represents that
initial decision event. KT2 covers fewer learners, so a subset action-log join
would not certify full KT1 automatically.

**Junyi.** The [primary DataShop entry](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198)
identifies the2015 dataset; its original [attached-file subtabs](https://pslcdatashop.web.cmu.edu/Files?datasetId=1198)
require authenticated authorized access. Public metadata was readable, but
publisher attempt/presentation/feedback mapping to the author CSV was not
established. Secondary mirrored field dictionaries were not treated as primary
proof. All14,660,217 accepted CSV rows have endTime==startTime. Zero end-time
overlaps are consequently not evidence of instantaneous response availability.

Public ASSIST CSV-export retrieval succeeded via read-only HTTP after the
browser parser failed; its content hash and all access limitations are recorded
in [authoritative_source_notes.json](reproducibility_diagnostics/authoritative_source_notes.json).
No dataset account access or data request was fabricated.

## All targets and their actual visible response histories

The first audit counted adjacent pairs. This follow-up independently aligns
every retained processed event to the store and examines **all earlier labels
visible in the actual15-session HiTSKT context**, including the current-session
prefix. It excludes each learner's first, context-only session from supervised
counts. History may span earlier train/validation/test sessions according to V1.

For each target i, a proxy-overlap edge is an exposed j<i with recorded/proxy
end(j)>start(i). End(j)==start(i) is excluded from that strict inequality.
Separately, presentation ties mean start(j)==start(i). EdNet same-episode means equal logged solving_id within
the same learner and the actual history window. These are exact counts under
these definitions, not exact counts of unavailable feedback or altered metrics.
Targets with multiple problematic prior labels are counted once in the target
column and multiple times in the edge column; tie and overlap groups may intersect.

'''
    for d in DATASETS:
        timing+=f'### {NAMES[d]}\n\n'
        timing+=table(['Split','HiTSKT supervised targets','Targets with prior end/proxy overlap','Overlap edges','Cross-session overlap targets','Targets with equal-start prior','Same EdNet episode targets'],[
            [split,f'{v["targets"]:,}',f'{v["proxy_overlap_targets"]:,}',f'{v["proxy_overlap_edges"]:,}',
             f'{v["cross_session_proxy_overlap_targets"]:,}',f'{v["equal_start_targets"]:,}',f'{v["same_solving_episode_targets"]:,}' if d=='ednet_kt1' else 'N/A']
            for split,v in [(s,timings[d]['actual_hitskt_15_session_context'][s]) for s in ['train','validation','test']]])+'\n'
        v=timings[d]['actual_hitskt_15_session_context']['test'];a=timings[d]['all_retained_adjacent']['test']
        timing+=f'Test: **{100*v["proxy_overlap_targets"]/v["targets"]:.6f}%** have an earlier visible recorded/proxy end after presentation; '
        timing+=f'**{100*v["equal_start_targets"]/v["targets"]:.6f}%** have a same-start prior label. '
        timing+=f'The adjacent-only test count was **{a["previous_end_after_start"]:,}**, whereas the full-context count is **{v["proxy_overlap_targets"]:,}**.\n\n'
        timing+=f'Evidence: [{d}_timing_context.json](reproducibility_diagnostics/{d}_timing_context.json).\n\n'
    timing+='''ASSIST's end-time count rises from2,426 adjacent targets to7,059 full-context
targets because a nonadjacent earlier activity may still have a later recorded
end. This does not prove that its first answer was still unknown. A separately
labelled sensitivity check interpreting startTime+timeTaken as step completion
instead finds41,625 test targets with a proxy overlap. That assumption has no
independently proven feedback mapping. In the accepted CSV,300,546 rows have
timeTaken==endTime-startTime,320,378 have a smaller duration and321,883 a larger
one. The fields cannot safely substitute for each other. Evidence:
[assist2017_step_duration_sensitivity.json](reproducibility_diagnostics/assist2017_step_duration_sensitivity.json).

EdNet's test full-context count is4,089,385, compared with4,003,464 adjacent
targets. Separately,3,364,799 test targets expose an earlier final label from
the same solving episode (6,150,341 edges). This is substantially broader than
the431,049 adjacent same-episode timestamp ties in the first audit. Under a
strict initial-bundle prediction claim those labels would be unavailable;
under an unspecified later retrospective decision point their actual release
must be established with action logs. Do not label all4,089,385 proxy overlaps
confirmed leakage, or claim the AUC effect equals their frequency.

Junyi has2,165,594 tied-start test targets, including18,633,725 tied prior-label
edges in the visible history. Source-row ordering resolves V1's deterministic
index, not actual response-release order. With no positive recorded duration
in the accepted CSV, its zero overlap result cannot certify timing.

The aggregate sweep uses per-learner bounded histories and independently tested
prefix counts, not an all-pairs matrix. Its exhaustive small examples cover
nonadjacent overlaps, equal timestamps, two learners, session boundaries and
history-window expiration. Source:
[diagnose_hitskt_timing.py](../scripts/diagnose_hitskt_timing.py);
oracle: [test_reproducibility_diagnostics.py](../tests/test_reproducibility_diagnostics.py).

## Defensible claims and consequences

V1 can be described as a **retrospective, sequence-index causal, chronological
within-student teacher-forced benchmark** on the recorded transformed datasets.
Its reproducibility qualifications remain those in
[the root-cause report](HITSKT_REPRODUCIBILITY_ROOT_CAUSE.md). It does not establish
student-disjoint generalization, availability-correct predictions at question
presentation, or deployable pre-bundle EdNet accuracy. The earlier audit's
end-time inequality FAIL is retained as evidence, while the stronger allegation
of confirmed feedback leakage remains NOT VERIFIED.

The proposed [availability-aware V2 protocol](HITSKT_AVAILABILITY_PROTOCOL_V2_PROPOSAL.md)
defines explicit decision/release events, handles ties and missing timing,
withholds within-bundle labels at initial entry, and versions all cohorts and
histories. It has **not** been substituted into historical evaluation.

Re-evaluating frozen checkpoints for timing sensitivity requires no retraining;
it is a new distribution/estimand and cannot replace the original result
silently. A controlled new availability-aware benchmark requires retraining
HiTSKT and any baseline whose training contexts change. That conditional need
is strongest for the EdNet pre-bundle claim, but it does not prove historical
V1 ASSIST, Junyi or EdNet training must be discarded. First recover/verify the
missing timing contract rather than guess it from duration fields.
'''
    save('HITSKT_RESPONSE_TIMING_ANALYSIS.md',timing)

    decision=r'''# HiTSKT research decision report

Date: 2026-10-18 UTC. Follow-up to audit `5625478`.

**Do not retrain any of the three historical V1 HiTSKT models yet.** Preserve
their checkpoints and original results as provisional historical evidence.
All three need further verification before their complete reported metric set
can be retained as independently reproduced. The safe CPU investigation does
not identify a training defect that a new training run would fix. It also does
not clear the existing results for publication.

If the intended claim is real-time prediction at question/initial-bundle
presentation, V1 does not guarantee that information set. Establish and version
an availability-aware protocol first; train all affected methods under it if
their training contexts change. That is a separate scientific experiment,
not a silent repair to historical metrics.

## Per-dataset decision

'''
    decision+=table(['Dataset','Stored AUC','Independent full CPU AUC','CPU metric set','CUDA replay','Actual response availability','Recommendation'],[
        [NAMES[d],f(old_evaluation(d)['stored_metrics']['roc_auc']),f(old_evaluation(d)['recalculated_metrics']['roc_auc']),
         'FAIL at frozen1e-6','NOT VERIFIED','NOT VERIFIED','2 — Additional verification; not yet retraining'] for d in DATASETS])+'\n'
    decision+=r'''**ASSIST2017:** AUC and MSE pass the declared numerical gate, but
accuracy/precision/recall/F1 and BCE fail. All112,252 targets, selected epoch40
and original hash are verified. Full CPU64 inference still fails BCE and leaves
the CPU classifications unchanged. The7,059 full-context recorded-end overlaps
do not establish when first-answer correctness arrived. Keep the V1 artifact
and independently measured CPU result with separate provenance; historical CUDA
verification remains necessary. **No established requirement to retrain V1.**

**Junyi:** AUC, MSE and both BCE definitions pass; all four classification
metrics fail. All2,456,402 test targets, selected epoch32 and hash are verified.
Higher-precision sampled inference covers4,072 targets only; it cannot certify
the full forward precision effect. Every accepted source row has identical
start/end, with2,165,594 tied-start test targets; publisher timing transformation
is unverified. **No established requirement to retrain V1; further CUDA and
timing-provenance verification required.**

**EdNet-KT1:** AUC, precision, MSE and BCE pass; accuracy, recall and F1 fail.
All13,429,870 targets, selected epoch30 and hash are verified. Sigmoid precision
alone changes exactly4 classifications on identical cached logits; a2,666-target
CPU64 sample changes1, which must not be extrapolated. The3,364,799 test targets
with prior same-episode labels prevent certifying a strict initial-bundle
information set from V1. **No established requirement to retrain the historical
retrospective model; corrected availability evaluation, followed by training
under V2 if contexts change, is required for a new pre-bundle/real-time claim.**

These are recommendation **2** for each historical V1 result, not recommendation1
or publication clearance. Recommendation **3** applies to an availability-aware
claim when its required information set differs. No dataset receives an
unqualified retain-and-publish decision.

All original-versus-reproduced accuracy/precision/recall/F1/MSE/BCE values, maximum
measurable numerical errors, exact diagnostic classification counts and missing
historical pointwise evidence are in
[HITSKT_REPRODUCIBILITY_ROOT_CAUSE.md](HITSKT_REPRODUCIBILITY_ROOT_CAUSE.md).
Exact timing counts and defensible claim boundaries are in
[HITSKT_RESPONSE_TIMING_ANALYSIS.md](HITSKT_RESPONSE_TIMING_ANALYSIS.md).
Separate follow-up manifests exist for
[ASSIST](reproducibility_diagnostics/assist2017_diagnostic_manifest.json),
[Junyi](reproducibility_diagnostics/junyi_diagnostic_manifest.json) and
[EdNet](reproducibility_diagnostics/ednet_kt1_diagnostic_manifest.json).

## Architecture identity — PASS as ELU+1; FAIL as FAVOR+

The preserved model is a **hierarchical HiTSKT variant with deterministic causal
ELU+1 linear attention**. In [ktbench/models/performer.py](../ktbench/models/performer.py),
lines68–85, projected queries/keys use ELU(x)+1, and key/key-value prefix sums
compute normalized causal kernel attention. There is no positive orthogonal
random-feature matrix or randomized softmax-kernel approximation. Legacy class
names/config strings containing Performer do not change the mathematics.

For a causal prefix, output is
\(\phi(q_i)^\top\sum_{j\le i}\phi(k_j)v_j^\top /
\max(\phi(q_i)^\top\sum_{j\le i}\phi(k_j),10^{-6})\),
with \(\phi(x)=\operatorname{ELU}(x)+1\), masks and the independently verified
response shifting. This is not exp(q·k/sqrt(d)) attention approximated by FAVOR+.
The model also has pre-normalization/GELU and omits some original-paper details;
historical results describe this variant, not an exact original-paper replica.

The [separate FAVOR+ implementation proposal](HITSKT_FAVOR_PLUS_IMPLEMENTATION_PROPOSAL.md)
specifies positive orthogonal random features, checkpoint identity, causal
stabilization and independent tests. No implementation or full training was
started. A future FAVOR+ claim requires separate training; relabelling or
swapping the historical evaluator is scientifically invalid.

## Baselines: artifacts versus replayed results

All11 completed DKT/DKVMN/SAKT/RKT baseline pairs need a fresh full checkpoint
test replay for independent verification. Rechecked hashes, finite tensors,
strict validation selection and stored-result consistency are useful artifact
evidence, not a substitute for prediction replay. The unfinished twelfth pair
has no final test score. The table below contains **historical stored AUCs**;
none is presented as freshly reproduced here.

'''
    rows=[]
    for model in ['dkt','dkvmn','sakt','rkt']:
        for d in DATASETS:
            a=next(r for r in artifacts['records'] if r['model']==model and r['dataset']==d)
            rows.append([model.upper(),NAMES[d],a['best']['epoch'],f(a['stored_test_metrics']['roc_auc']) if a['complete'] else 'MISSING',
                'PASS' if a['checkpoint_matches_strict_selection'] and a['best_matches_previous_audit_sha256'] else 'FAIL',
                'NOT VERIFIED — replay needed' if a['complete'] else 'NOT VERIFIED — unfinished'])
    decision+=table(['Model','Dataset','Selected epoch','Historical AUC','Hash/selection','Fresh test metrics'],rows)+'\n'
    decision+='''Read-only follow-up evidence:
[artifact_and_dkt_forensics.json](reproducibility_diagnostics/artifact_and_dkt_forensics.json).
Full original consistency checks remain in
[benchmark_artifacts.json](independent_correctness_audit/benchmark_artifacts.json).
The raw-label-pair reconstruction for RKT remains NOT VERIFIED; the prior
cache algebra, domains and CRC checks do not prove the underlying co-occurrence
counts were generated correctly. Existing class/mask tests pass their defined
information-index contract, not the physical release-time contract.

Comparability remains qualified: DKT has199 prior interactions and skill inputs,
DKVMN49, SAKT/RKT99, while HiTSKT has15 sessions and question+skill inputs.
Parameter counts and fixed hyperparameters differ. HiTSKT excludes the first
training session from supervision, whereas baselines supervise those training
events with prior context; held-out target populations match. RKT has
training-only cross-fitted relations whose raw construction needs verification.
Shared splits, thresholds and validation selection do not make these controlled
attention-only ablations or exact replicas of every published baseline.

The recorded single-seed42 AUC gain over the best available baseline is
+0.006571356857316912 on Junyi and+0.006356979129155005 on EdNet. On ASSIST it
is−0.03900247395955292. CPU AUC differences are much smaller than those recorded
gaps, but missing baseline replays, differing information budgets, timing
assumptions and one training seed prevent claiming robust superiority or
statistical significance. EdNet's strongest comparison is incomplete without
DKT; no missing result is inferred from its validation checkpoint.

## Unfinished DKT/EdNet: confirmed state defect, origin unresolved

No final_results.json or _SUCCESS exists. Nine complete epochs are retained,
best epoch1 validation AUC0.6845717850759062, last epoch9 validation
AUC0.6108838021742428, non-improvement counter8 and authorized patience10.
Epoch10 progress stopped at36,418,836/53,990,022 training targets on October12;
the stale running status is not an active trainer or a completed result.

Best hash: `56a911479a2c0b53edd457e4179a942be6f652717e7322c35b41e45194e90215`.
Last hash: `ae9f8248669c3800064020cbbb64c4e82ade108cac0cef27e1a23c74f208842b`.
Best model/Adam and last model weights are finite. Last Adam exp_avg_sq contains:

'''
    pending=next(r for r in artifacts['records'] if r['model']=='dkt' and r['dataset']=='ednet_kt1')
    decision+=table(['Parameter','Positive infinities'],[[v['parameter'],v['positive_inf']] for v in pending['last']['optimizer_nonfinite_states']])+'\n'
    decision+='''Total **21,749**. First moments are finite but reach approximately1.97e13;
finite second moments reach approximately2.56e32. Saved optimizer uses FP32,
Adam lr0.0002, betas(0.9,0.999), eps1e-8, no AMSGrad and no configured gradient
clip. Large historical gradients/second-moment overflow are a plausible cause;
no per-step gradient logs or complete intermediate canonical checkpoints prove
the first offending update. A finite best epoch1 does not locate the failure
within later epochs. The archived epoch1 DKT snapshots are also finite; the
transport-switch snapshots at epochs13/15 are SAKT, not later DKT evidence.

With finite subsequent gradients, beta2*infinity plus a finite squared-gradient
term stays infinite. With finite first moment, the resulting infinite Adam
denominator gives zero update on those coordinates; ordinary decay cannot
repair them. A tiny synthetic algebra diagnostic confirms that behavior without
stepping any historical optimizer. Other coordinates may still update, so
finite epoch losses/model weights do not certify optimizer integrity.

Checkpoint serialization embeds common patience5, whereas active config/events
and authorized DKT/EdNet control use10. This is a preserved metadata discrepancy,
not permission to reset its optimizer or silently restart. The damaged last
state should not be called ready for continuation. Recovery from a validated
earlier finite state or a separately versioned rerun needs an explicit recorded
policy and authorization; no DKT result or recovery was fabricated here. This
baseline defect does not establish a defect in the three finite HiTSKT states.

## Issue register and retraining implications

| Issue | Severity / status | Affected claim | Retraining implication |
| --- | --- | --- | --- |
| Historical-to-CPU metric gate fails | HIGH / FAIL, all3 | Complete historical metric-set reproduction | No demonstrated training defect; obtain matching CUDA replay first |
| Historical vectors/runtime flags absent | HIGH / NOT VERIFIED | Exact CUDA/CPU flips, max errors and operation-level root cause | Additional inference/provenance, not training |
| End/proxy timing and transformed ties | HIGH / availability NOT VERIFIED | Real-time question/bundle presentation claims, all3 | V2 evaluation first; retrain all affected methods if training contexts change |
| Historical architecture called FAVOR+ | HIGH / identity claim FAIL | Performer/FAVOR+ attribution | Correct terminology for V1; genuine FAVOR+ requires separately approved new training |
| Baseline full checkpoint replay absent | MEDIUM / NOT VERIFIED,11 completed pairs | Junyi/EdNet recorded gains | Replay first; no established baseline retraining need |
| RKT raw pair-count reconstruction absent | MEDIUM / NOT VERIFIED | RKT input-relation provenance | Rebuild evidence first; changed relations may require a new RKT run |
| One primary training seed42 | MEDIUM / significance NOT VERIFIED | Robustness and statistical significance | Additional seeds are new experiments, only after core verification |
| DKT/EdNet last optimizer nonfinite, no final test | HIGH / integrity FAIL; score NOT VERIFIED | Missing fifteenth baseline result | No resume now; separately agree finite-state recovery/rerun policy |
| DKT saved patience5 versus active10 | LOW / metadata FAIL | DKT continuation reproducibility | Not by itself a retraining reason |
| Publisher-to-author transform provenance incomplete | MEDIUM / NOT VERIFIED for ASSIST/Junyi | Original raw provenance and timestamp meaning | Recover transformation first; no automatic training change |

## Bounded next steps, after this diagnosis

1. When safely idle independent H100/MIG resources are available, replay the
   three unchanged best checkpoints, capture settings and vectors, and apply
   the original gate. Preserve matching-runtime results separately from CPU
   differences. Do not interrupt LightSAKT2 or tune tolerance to observed metrics.
2. Freshly replay the11 completed baselines and independently check RKT's
   training-only raw pair construction. Do not resume the unfinished DKT run
   as part of an evaluation task.
3. Recover original timing/provenance evidence; adopt
   [availability V2](HITSKT_AVAILABILITY_PROTOCOL_V2_PROPOSAL.md) only as an explicit
   new protocol, with frozen-checkpoint sensitivity before a new training plan.
4. Only after these checks, consider approved matched-information ablations,
   additional training seeds and [genuine FAVOR+](HITSKT_FAVOR_PLUS_IMPLEMENTATION_PROPOSAL.md).
   Those experiments have not been started and do not repair missing evidence
   about historical experiments.

## Preservation and delivery

All375 original captured files, all30 best/last checkpoints and106 previous audit
files remain hash-verified. The original audit report, model, loaders, results,
Git history, unrelated README edit and SAKT console file are preserved. New
evidence is additive and learner-level arrays stay ignored/local.59 relevant
tests pass. No new full training, protocol replacement, GPU allocation or
LightSAKT2 modification/signal occurred. Resource and preservation records are
linked from the root-cause report. Durable continuation memory is
[project-memory/REPRODUCIBILITY_STATE.md](../project-memory/REPRODUCIBILITY_STATE.md).

**Decision: retain all three V1 artifacts provisionally, require further
verification of their reported results, and do not order retraining solely
from these numerical discrepancies. Publication readiness remains NOT VERIFIED.**
'''
    save('HITSKT_RESEARCH_DECISION_REPORT.md',decision)


if __name__=='__main__':main()

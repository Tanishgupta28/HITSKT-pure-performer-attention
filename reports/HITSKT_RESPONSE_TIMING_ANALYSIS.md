# HiTSKT response-timing analysis

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

### ASSIST2017

| Split | HiTSKT supervised targets | Targets with prior end/proxy overlap | Overlap edges | Cross-session overlap targets | Targets with equal-start prior | Same EdNet episode targets |
| --- | --- | --- | --- | --- | --- | --- |
| train | 566,107 | 15,405 | 17,971 | 2,228 | 4,583 | N/A |
| validation | 125,800 | 8,147 | 9,807 | 612 | 1,228 | N/A |
| test | 112,252 | 7,059 | 8,484 | 206 | 1,102 | N/A |

Test: **6.288529%** have an earlier visible recorded/proxy end after presentation; **0.981720%** have a same-start prior label. The adjacent-only test count was **2,426**, whereas the full-context count is **7,059**.

Evidence: [assist2017_timing_context.json](reproducibility_diagnostics/assist2017_timing_context.json).

### Junyi

| Split | HiTSKT supervised targets | Targets with prior end/proxy overlap | Overlap edges | Cross-session overlap targets | Targets with equal-start prior | Same EdNet episode targets |
| --- | --- | --- | --- | --- | --- | --- |
| train | 8,920,358 | 0 | 0 | 0 | 7,875,413 | N/A |
| validation | 2,556,773 | 0 | 0 | 0 | 2,252,262 | N/A |
| test | 2,456,402 | 0 | 0 | 0 | 2,165,594 | N/A |

Test: **0.000000%** have an earlier visible recorded/proxy end after presentation; **88.161221%** have a same-start prior label. The adjacent-only test count was **0**, whereas the full-context count is **0**.

Evidence: [junyi_timing_context.json](reproducibility_diagnostics/junyi_timing_context.json).

### EdNet-KT1

| Split | HiTSKT supervised targets | Targets with prior end/proxy overlap | Overlap edges | Cross-session overlap targets | Targets with equal-start prior | Same EdNet episode targets |
| --- | --- | --- | --- | --- | --- | --- |
| train | 49,458,444 | 14,250,811 | 23,526,609 | 40 | 510,925 | 11,465,681 |
| validation | 14,520,975 | 4,396,233 | 7,312,766 | 9 | 287,104 | 3,598,247 |
| test | 13,429,870 | 4,089,385 | 6,842,255 | 98 | 431,050 | 3,364,799 |

Test: **30.449922%** have an earlier visible recorded/proxy end after presentation; **3.209636%** have a same-start prior label. The adjacent-only test count was **4,003,464**, whereas the full-context count is **4,089,385**.

Evidence: [ednet_kt1_timing_context.json](reproducibility_diagnostics/ednet_kt1_timing_context.json).

ASSIST's end-time count rises from2,426 adjacent targets to7,059 full-context
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

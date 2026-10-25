# Performance research plan — 25 October 2026

Base/current-before-publication `50be8f91e43eeb06ad8f4abbe3e189730f8d6f86`.
User requests architecture/baseline review and reproducible performance-improvement
plan, necessary CPU checks, no long GPU work before plan/resources. This is NOT
approval of the previous four-arm proposal. No CUDA initialization/probe, optimizer
updates, new model evaluation or other-job changes in this task.

Report: `reports/favor_plus_v1/PERFORMANCE_IMPROVEMENT_EXPERIMENT_PLAN.md`.
Pack: `reports/favor_plus_v1/performance_study_20261025_v1/`.
CPU builder: `scripts/prepare_hitskt_performance_study.py` with isolated pure
NumPy/stdlib module `ktbench/favor_plus_performance_study.py`; no train option.
Local equivalent plan: `experiments/hitskt_favor_plus_v1/assist2017/performance_plan_20261025_v1/prepared_plan.json`.
Plan SHA94ef6169d14aa3fd0d93afee765e349f2d8e019d2f09e2692972b2e75affbbd4.
Sources/data11/common8schedules verified. All7060train sessions/566107labels
per epoch, fullval1977sessions/125800labels47584pos78216neg. Orderedvalevent
SHA7e0da8b0a737122179ade388f7091449c52314b06ab3ec1826bbd33c147f83bb;
labelSHA49b7bd9f25b6d21e2203495a5ed0d0c506ce32f19123a73e713dd5950550c935.
Current adapters select same validation events; historical baseline replay absent.

Verified existing mechanisms: pre-LN/residuals/GELU/dropout already present;
four depth fields configurable, actualwidth256/heads4/FF2048/dropout.1/layers1.
ActionEOS and SessionEOS compress history; CPEbroadcast sum is ungated.
Question embeddings already inDecoder, but history summary not item-conditioned.

Priority1 fresh ORF128/512 matchedFP64/commonpool/seed42/data/Adam; antithetic512
optional (FP64 CUDA validated; nativeantiFP32 unrun). No learnedcandidate results.
Two extra DESIGN-ONLY recipes: target-conditioned fusion gate(identity atinit,
196864newparams) and exact retrieval over≤15pastsession states(262144params,
hybridFAVOR/exact). Neither implemented/tested. Depth/dropout/width secondary;
do not sell pre-LN/residuals as new. Native models/defaults/checkpoints untouched.

All numeric comparison in new baseline table is FULL VALIDATION, no test gains:
ELU.719757/40selected, DKT.721370/12, DKVMN.696252/18, SAKT.757706/30,
RKT.771255/14, FAVOR128.703467/7. Storedbaseline rows/selection CSV/config
crosschecked; freshvalidationreplay NOT VERIFIED. Favorbest7 fullvalreplayPASS
priorstudy. Trainingbudgets/inputhistory/targets differ, notcontrolledablation.
Correct contexts DKT199/DKVMN199/SAKT99/RKT49; HiTSKT15sessions.

Oldapproxwarnings remain Action.262562/CPE.503431/Decoder.562503 at8; no
discovered originalbug inactualtraineddiagnostics. Learnedfixedalignment+finite
feature/valuegeometry supported, not causaloptimizerproof. SeparateFP32stress
FAIL preserved. FP64all4CUDA PASS96 remains historicalverification. Initial14
TRAINcases128prefix ORF512Action.152904vsanti512.179270; CPE.066212vs.049956.
No inference that initialization improvement guarantees learnedAUC/fidelity.

33CPUtestsPASS=18newplanning+15existingfactory, no optimizer steps. Blueprint
rejects config/seed/schedule/threshold/resource drift and unintended training/
test scope. NewA/B architectures have NO tests because unimplemented.

Resourceproposal: primary20/arm40total≤15min, zero-update real full-shapeprofile
≤10min, existing8GiBhard/10advisory/16admission/6runtime/4host/2headroom/.1s
pacing. Full8epoch matchedpair: requested provisional2GPUhourreservation ONLY,
notruntimeprediction/approval; actual512FP64fulltime/peak unknown. Existing45min
execution ceiling not raised by reservation; needs explicitnewbudget iflonger.
Historical128FP32 eight epochs41m38s,live3.192GiB/reserve7.990GiB. Component
accountingmax128/512 FP64=1.542/3.604GiB excludesothersaved/workspace/allocator;
NOT peak/fitproof. Profilecommonshapes beforefullstudy; ifshrink needed botharms
use one newsealedcomplete schedule/no truncation or silentdropping.

RunnerreadinessLIMIT: existing20step runner bindsFOUR arms; historicalepoch
controller fixednative128. Newtwo-arm/full-epochmatched dispatch not wired.
Five recipes DECLARATIVE ONLY, not executabletrainingconfigs. CPU prepare
doesnot authorize launchinganything. Futureapprovedexecutionrequires newreviewed
scope/variantwiring/tests and current shared admission, no PIDownership re-audit.

Equal-exposure8epoch proposal,no performanceearlystop,strictbestfullvalAUC/.5
threshold. Localhistory/retrospectivetiming qualification retained. Do notcompare
sampled3157valmetrics with full125800val or baseline test scores. Confirmatory
42/43/44 seeds/baselinefreshreplay+matchedsupervision/finaltest all PROPOSED ONLY.
Historicaltestalreadyknown, heldoutfromNEWselectionnotgloballyunseenclaim.
Preserve7/8/50/ELU/data/Light/DKT/oldfour-armplan/README unrelated dirt/main.
Finish scopedresearchpublication, thenWAIT fornewboundedexecutioninstruction.

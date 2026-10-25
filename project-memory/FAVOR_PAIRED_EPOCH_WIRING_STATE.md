# Paired epoch wiring and real profile — 25 October 2026

Basea372c8432e2296410ad5af325468c8fc2e903c90. Read
`reports/favor_plus_v1/MATCHED_EPOCH_EXECUTION_AND_MEMORY_PROFILE.md`.
Evidence: `reports/favor_plus_v1/paired_epochs_execution_20261025_v1/`.
User requests executableORF128/512 matched8epoch pipelines and bounded profile,
explicitly prohibits auto2htraining without runtime/resourceapproval. Existing
sharedGPU/8GiB cap persists. Userprefersautonomy/routinechoices,questionsonly
forgenuineblockers. Currenttask3+referenced10minprofile+sharedallocation treated
asauthorityfor600s ZERO optimizer profiling only; scope record archived.

IMPLEMENTED: `ktbench/favor_plus_paired_epochs.py`, CLI
`scripts/run_favor_plus_paired_epochs.py`. Nativeoriginal/prototypes immutable.
SeparateFORMAT5/KINDhitskt_favor_plus_matched_pair_epochs_v1; one-layer width256/
heads4/FF2048/dropout.1, fixedORFpool512/stageseeds42–45,attentionFP64/neuralFP32,
fresh42 weights/CPU RNGidentical botharms. Eightfull epochs equalexposure,strict
bestfullvalAUC/.5, no performancepatiencestop; oldhistorical8 differsFP32/draw.
Model/Adam/RNG/projection/source/data/exactsampler/metricprefix atomicsave every25
andepoch; fresh recovery/no-step nextcompute proof; partialtrajectory helpers
verified,no automaticfailedstep/mid-epochretry. Fullval predictions includeIDs
andcheckedlabelhashes. Native32prefix severe/.20 plus128prefix/audit17/101
TRAINdiagnostics. Finitecampaign34subprocessstages; firstfailureSTOP/noretry,
selectedfullvalreplay andCPUpairedcomparison onlyafterbothcomplete8.
ActualnewfullGPUtraining/epochvalidation NOT RUN. Tests66CPU PASS(51new/15old),
syntheticfixturesonly; realASSIST optimizerupdates0.

Originalpreparedroot:
`experiments/hitskt_favor_plus_v1/assist2017/assist2017_orf128_512_pair_epochs_20261025_v1/`.
PlanSHA9192d6636d5792c1727a4dfbfb31cd405432f60cadebb83c7f037dd6e6660fdb.
All7060TRAINsessions/566107labels per epoch;fullval1977/125800labels. Original
8schedule1454updates/arm.9representativeprofilecases overall8schedules.
CurrentassignedMIG-d2fcfd79-720e-5218-82c9-0a132be68920 H1003g.40gb admitted;
sharedpolicy/PIDunknown/utilunknown retained. 8GiBhard/10advisory/16admit/6runtime/
4host/2headroom/.1pacing/30sstep unchanged. Nootherjob signals orruntimepatch.

ACTUAL: ORF128FP64 profile PASS9/9,peaklive6.005212GiB,reserved7.806641GiB,
host1.538620GiB,maxF/B2.336s. ORF512 completed3/9,passedfullB4/cost21299 at
2.261s,thenfailedbackward B20/cost31515. Cachedpre-failurepeak3.841291GiB/
reserved4.539063GiB,NOTfailedinstantpeak. Lastglobalfree34.438782GiB,guardsPASS.
RuntimeErrorNVML_SUCCESS==r allocator.cpp841. Bundledexactsource confirms line
inreportProcessMemoryInfo calledAFTERallocationfailure branch assert(memoryalloc)
BEFOREstandardOOM/statsincrement. SecondaryNVMLreporting bug masksOOM; compatible
8GiBcap/fragmentation, exactallocation/NVMLreturnnotcaptured. Notconfirmedpeer
conflict/PIDalias/orattentionbug. NoGPUretry/resize/clockreset afterSTOP. Root
study_failure andsource snapshots preserved. Profilewallrecord→terminalmtime
31.704s includesprocess/guardoverhead,notGPUkerneltime. Oneafterexitreadonly
snapshot39.337952GiBfree/noNVMLworkers/no visibleLight; CUDAinitFALSE observer.

CPU FOLLOWUP ONLY: `ktbench/favor_plus_paired_rebatch.py`, CLI
`scripts/prepare_favor_plus_paired_rebatch.py`. Newroot
`experiments/hitskt_favor_plus_v1/assist2017/assist2017_orf128_512_pair_b8_20261025_v2/`.
PlanSHAcb3bbc9168b712d187a0b2e998c01816b204e2683f74d2e3a01bc5cf0d866edb.
Botharms commonmaxB8, wholeexamples/order/labels/lengths preserved, maxcost32751,
no gradientaccumulation.7499updates/arm (935/937/943/939/939/934/938/934 epochs)
vs1454oldgrouping. Same seed/model/attention/Adam acrossnewpair; changedbatch
trajectory explicitlydeclared, notequivalentold.703467run. Twohourtiming
NOTESTIMABLE untilnewprofile; ~50–55GiBcheckpointstoragepossible, observed151GiB
freevolume notreserved. InitialCPUrelativeparentpathbugfixed, initialsource/log
retained; actualnewplan/source/data/fullcoverage reloadPASS.

NEW B8GPU profile NOT RUN. Async explicitapprovalquestion pending for≤10min
zero-stepprofile only afterterminalfailure; noresponse/defaultchoiceNOTapproval.
No furtherGPUworkuntilactualreply. B8templatesfalse; trainingbudget NOTAPPROVED.
Profile permissioncannotlaunchtraining. Futurefullruntime recordneedsactual
eight_epoch_pair_runtime_approved +actual_new_time_budget_evidence for proposed
2hpair/60minarm (old45minwindowNOTsilentlyincreased).7499updatesmaynotfit;
remeasure/requestproperbudget beforetraining. Do NOT launchrawfailedroot or
resethandledfailedscope. IfnewprofileapproveduseNEW B8plan/freshscope/admission,
botharmszeroAdam,stopfirstfailure; nootherjobinterference/adminPIDmappingask.
Ifnoapproval finishverifiedreport/archive/memory/researchcommit+pushthenWAIT.
PreserveallhistoricalELU/FAVOR7/8/50/data/Light/DKT/audits/unrelateddirt/main.

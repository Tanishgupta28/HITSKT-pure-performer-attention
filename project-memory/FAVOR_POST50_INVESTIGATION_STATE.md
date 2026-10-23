# Post50 CPU scientific investigation —23October2026

COMPLETE diagnostics, ZERO new optimizer updates; currenttotal50. No newCUDA
model execution/validationinference/testevaluation. STOP for nextinstruction.
Report:`reports/favor_plus_v1/POST50_APPROXIMATION_METRICS_AND_TRAINING_READINESS.md`.
Evidence:`reports/favor_plus_v1/post50_cpu_investigation_20261023_v1/`.
Rawrun:`experiments/hitskt_favor_plus_v1/assist2017/post50_cpu_diagnostics_20261023_v1/`;
INVESTIGATION_RESULT.json terminalmarker. Designpreregistered8b0b683823d2ab589d805fbe27644a27ed62d948.
Parent50 SHA93ebe9d2a29be9a6e4deb278916513339de61c9ac8f72a454a31111829440a73
unchanged. All35model/controller sources,11processedhashes/774+316+199+9+8prior
artifacts/checkpoints preserved. Light/README/SAKTfiles unchanged; nojobssignaled.
No fulltraining/configuration/threshold/main/historychange. Checkpointbinaries
remainlocalignored; Git evidence is notbinarybackup.

Sourceinspection +82mainCPUtestsPASS(18new+58FAVOR+6ELUregression),3guardchecksPASS
(2new/1overlap),84distinct. Separate18developmentrunoverlaps82. No229suite rerun.
Frozen14TRAINcases metadata-selected:original6+6stratummediane+2extrema,cost117–7164.
New14×2stages actualprojection/length and4diagnosticprojectionseeds×128/256/512
comparisons. No modelprojection/featurecount changed. Fullfloat64featureoracle
agreesCPUproductionmax1.771e-6<5e-5; actualsavedseeds42–45/buffers regenerateexactly.
Deployed m128 mixedcap128errors Actionmedian.211255/10of14>.20;
CPE.197298/7of14>.20. Old6prefix32Action.212305 REVIEW andCPE.192699 remain.
LongestCPEtarget939error.262988. Prospectivelypredeclared1case max-historyANNEX
fillsAction939coveragegap, errors128/256/512/939=.196072/.186577/.170216/.158714.
Lengtherrornotmonotonic; cohorts/counts differ, do notconflatewithtrainingdrift.

4projectionseedgrid (56cases/stage/count, nottrainingruns):
Action128/256/512median.196284/.176693/.141857; CPE.117417/.084301/.076080.
CPE512p95worsethan256; tailsnotguaranteed. No seed/featureselection fromscores.
PairfeatureESS≈8–9/128 whileexactkeyeffectivenumber≈44–50, maxattentionweight
medians.037/.027. Concentrationisrandomfeatureaxis, notjustpeakedexactattention.
RawQKmedians≈4.5–4.6, noexplosion. FrozenQKscale.5/1/1.5 Actionmedian
.037827/.212305/.454815;CPE.032989/.192699/.363861; probesonly, notmodelrescaling.
Paired25/50CPE meanerror all25.153049,QK50/V25.189875,all50.191824,Vonly.154625:
learnedQKalignmentdriftexplainsmostlocalincrease; doesnotcausallyattributelearning
orpredictiveF1. FP64replayexcludesroundingasproximateexplanation. .20 REVIEW
unchanged/projectheuristicrelativeoutputFrobmetric, notlabel-error/AUCbound.
No discoveredattention/mask/metricdefect or requiredmodelrewrite/Adamreset.

Savedvalidationmetrics/labels/orderconsistent.25→50: TP463→326,FP908→586,
TN1186→1508,FN600→737;461crossings(137TP→FN,323FP→TN,1TN→FP).
Meanp.470823→.415659; bothclassesdecline, no pwithin1e-6of.5. PositiveBCEworsens
.813698→.937382;negativeimproves.683026→.568226. Classweighteddecomposition
+.041646−.076145=-.034500 explainsimprovedBCEwithdecliningrecall/F1. Headbias
only−3.84e-5vsmeanlogit−.244684;representation/headcontributionsnotisolated.
No evidenceFAVORerrorcausesF1decline. Fixed.4/.5/.6sensitivitydiagnosticsonly,
actualthresholdstays.5. SamplepriorconstantBCE.638830<BCE50.692525; no predictive
usefulness/convergence/superiorityclaim. Pilot50saw2528labels/1055positive,
only.447%of566107trainlabels;trainfullpi.372534,pilot.417326,val.336712.

Actualoriginal7060trainexamples,181epoch1seed42batches,B4–64/median37,
cost7606–32766/median31612;targetmedian63/p95206/max938;historymax15.
Medianpadding45.34%/max79.54%. B1cost2048excludes158sessions/22467labels(3.969%),
notallowedtodropinfulltraining. Staticcomponentplanningoriginalmax1.095GiB,
93/181>1GiB includesfullgradcopy. QualifiedforwardWITHOUTgradsmax1.069GiB,
70/181>1GiB plusmissingallocations; maxsinglechunkstatecomponent1.027GiB,
samebatchforward+onechunkstate2.071GiB NOTcompleteGPUpeak/safecap.
TwoCPUrealB1profilescost344/2047: peaksavedstorage29.008/99.497MiB,
CPU RSS.775/.996GiB, gradsfinite/noAdam. Partialretentiononly57.3–57.6%ofactual
forwardsavedstorage. NooriginalbatchfitCPUprofilecost2048, thirdprofileNOTRUN.
Actualoriginal-shapeVRAM/runtimeNOTMEASURED;1GiBcapcannotpresumeoriginalbatchfeasible.

IMPORTANTCPUenvironmentdeviation: CUDA-linkedPyTorchCPUautogradmadeinternal
cudaGetDeviceCountattempts in2profilechildren, emittederror2/OOMwarningsunder
virtualaddressceiling; noCUDAmodel/tensoroperationrequested andTorchinitialized
flagfalse. StrictpredeclarednoGPU-probeintentNOTfullysatisfied bylibraryinternals.
DoNOTclaimzeroCUDAAPIcalls/actualVRAMOOM. Sourcewarningretained;AS/drivercause
plausiblebutnotisolated, noextraGPUprobeauthorized/repeated. Executedscript
snapshot/hashpreservedbeforecorrection. CurrentCPUprofilerfailclosedrequires
CPU-onlytorchbeforeanyfuturestrictCPUbackward.2newguardcasesPASS/noactualbackward.
No package install/model/source35/sharedGPUguardchange. No ongoing processes.

Decision: CONDITIONALGOonlynewexplicitlyapprovedshort50→75researchstudyunder
same128/Adam/RNG/projections/sharedlimits +expandedcohort/shortmilestones. Approval
NOTreceived; noGPUorupdatesnow. NO-GOfull/longunboundedtraining. Originalshapes
needseparatelyapprovedzero-optimizerGPUprofile/resourceplan; neverincreasecap
ortruncatehistories/targets silently. Fullvalidation-selected checkpointand
untouchedtestall7metrics onlyaftermeaningfultrainingandnewapproval. Current
within-studentretrospectivetiming/singleseedqualificationspersist. Junyi/EdNet
needownboundedprofiles/pilots. FinishverifiedpublicationthenWAIT.

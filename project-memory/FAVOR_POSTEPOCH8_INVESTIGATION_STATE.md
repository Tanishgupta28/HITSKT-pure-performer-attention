# Post-epoch8 approximation investigation — COMPLETE, 24 October 2026

Usertask: investigate beforemoretraining; completed withZERO optimizer/CUDA/new
validation/test inference. NoGPUresourcepolling or other-workload changes.
Mainreport: `reports/favor_plus_v1/POSTEPOCH8_APPROXIMATION_INVESTIGATION.md`.
Evidence/design/annexes/professorupdate/summary/plots/testXMLs:
`reports/favor_plus_v1/postepoch8_approximation_20261024_v1/`.
Newrawrun: `experiments/hitskt_favor_plus_v1/assist2017/postepoch8_frozen_20261024_v1/`.
TerminalINVESTIGATION_RESULT.json=FROZEN_INVESTIGATION_COMPLETE.

14previously selectedTRAINcases × reconstructedinitial/7/8 ×4stages=168captures;
4032controlledgridcomparisons counts128/256/512/seeds17,101,202,303/nested512pool
on frozen≤128prefixes. Original-drawlengthchecks through939; no datatruncation
forfullmodel capture. CurrentCPUoriginal6prefix32errorsmatchstoredGPUsummaries
within1.01935e-7. No initialphysicalcheckpoint; seed42reconstruction caveatexplicit.
Originalsource37/data11/7+8hashes/180publishednonmemoryfiles/protectedLight hold/
unrelatedREADME+SAKTdirtmatch. NochangesoldELU/originalFAVOR/datasets/results.

Nooriginalattentionimplementationbug foundtested: originalproduction/reference
max3.113340e-6<5e-5; FP64≈4.44e-15whileapproxerrorpersists; projectionsregenerate
bitexact, blocks/radii/causal/ownlabel/PAD/normalization/gradients tested.
Original.20 REVIEW unchanged (old6-case32 Action.262562/CPE.503431 retained).
New14-case128CPEsavederror .142(initial)→.545(7)/.566(8), freshindependent128
.114→.193/.183. SavedpairESS9.40→1.84,fresh8.27→6.19; dominantfeaturemass
.296→.677 whileexactweightsremainfairlydiffuse. Supportsprojection-specific
learnedalignment/coadaptation, not randomizedcausalproof/guaranteedAUCimprovement.
NormsmoderateQ4.56→5.25/K4.48→4.66; learnedK andV bothcontribute. Relativeoutput
metric sensitiveVgeometry/denominator; extraTV/RMSEcontextdoesNOTrelabelPASS.

Independent512ORFmediansAction.186/CPE.129/Decoder.336; antithetic512
.175/.132/.316. Default42+stage antithetic512 .256/.161/.355; longer939CPE
.224–.243/Decoder.486–.487 REVIEW, Action640 .293–.303. No completefix.
Firstvalidkeycenteringpreservesexacttargetbutnotrobust, rejectglobalfix. QK
scalingflattenstemperature, nottarget-preservingbugfix.
Separatemodelmodule `ktbench/models/favor_plus_antithetic_experimental.py`: opt-in
VARIANT hitskt_favor_plus_antithetic_experimental_v1, format3, total512features=
256ORFdirections±, declaredpool512/fixedseeds/chunk32/FP32 and originalnonattention.
Notregistered/wiredintotraining/evaluationCLI; no candidateAUC/GPU/training.
Initialprototypeformat1collisiontestfailed; initialsource/log/XMLpreserved, fixed
format3 withoriginalformats1/2unchanged. Final88prototype+originalFAVOR testsPASS,
35oraclecontrolsPASS,17expandedexactrefPASS (overlapdedup=125unique,47new/78old).

TRAIN-only exactoperatorcounterfactuals preserveoriginalweights/state: CPEonly
medianprobMAE.007178/max.065989/109flips; allstages.041082/max.302781/385flips
over5170repeatedtargetpresentations. Not atrainedsoftmaxbaseline/accuracygain.
ExistingAUC7 .703467/F1.508814 and maxF1epoch6 .532809 remainvalidqualified
validationresults. Eight-epochcurvesnotconvergence; onebadAUCepoch8≠patience3.
Furthertrainingmayhelpvalidationbutnotestablishedaslikelyorafidelityfix.

CPUisolatedruntime /tmp/hitskt_cpu_diag_20261024 Torch2.4.1+cpu (CUDA_None),
process-local LD_LIBRARY_PATH=/tmp/hitskt_cpu_diag_20261024/lib/python3.10/site-packages/torch/lib
required: inheritedglobalLDinitiallyresolvedCUDA-linkedlibraries/importfailed.
Missingensurepipresolvedvenv+externalpip--python; noglobalsystemchanges.
UseCUDA_VISIBLE_DEVICES='',PYTHONPATH=.,OMP/MKL/OPENBLASthreads1,taskset223,nice10.
WorkerstrictguardrejectsCUDA build or loadedlibtorch_cuda; noCUDAautogradprobe.
PeakmeasuredRSS1.235GiB, longestdiagnosticaction53.063s.
Full168NPZ~98.6MiB retainedLOCAL/hashes; compact72prefix≤32 numericalbundle
~7MiB published+independentroundtripkernelproof. All36resultJSONs archived
~3.27MiB/entryhashverified; terminalmetadata separate. Do notclaimallfullarrays
orcheckpointbinariesareGitbacked. Rawlocalfilesremainuntracked, not garbage.

Decision: GO qualifiedreporting/CPUprototype; REVIEW/NO-GO automaticlongerfull
training/all-stage.20pass/publicationclaims. No requirementdiscard/retrainold
models merelytoerasewarning. Proposednextsmallmatched20updatepilot original128
vsantithetic512, identicalseed42/trainexamples/optimizer, diagnostics0/5/20 and
independentauditdraws/longprefixes, GPUprofile/admissionfirst. ThisisNOTauthorized
executionnow; requiresnewexplicitboundedtime/instruction, sharedguardsretained.
Ifcoadaptationrecurs, separatelyversionedserializedredrawpolicyfuturedecision,
notimplementedintoexistingruns. Do notposthocchooseprojectionsusingvalidation/
testAUC. Preservebest7/latest8 exactly. Finishscopedpublicationresearchbranch
only/no main/forcepush, thenWAIT fornextinstruction.

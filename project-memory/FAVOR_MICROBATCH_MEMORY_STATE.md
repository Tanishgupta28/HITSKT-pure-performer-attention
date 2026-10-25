# FAVOR+ logical-B8 microbatch memory preparation

25 October2026. Research branch; base457322143e64b4366ff47a810cb4573a1e08ecdc.
Current report `reports/favor_plus_v1/MEMORY_EFFICIENT_MATCHED_TRAINING_PREPARATION.md`.
Evidence `reports/favor_plus_v1/microbatch_memory_20261025_v1/`.

## Facts and scope

User requests practical memory-efficient matched128/512 training preparation.
Explicitly NO CUDA profile without a separately authorized execution window;
no seven-hour/epoch campaign approved. A concrete <=15min/<=160 sandboxAdam
profile was requested asynchronously; no reply/approval has been received or
assumed. Current task performed **zero CUDA execution, GPU queries, actual-data
optimizer steps, validation/test inference or other-job signals**. Synthetic
CPU fixture optimizers are tests, not ASSIST research outcomes. Shared GPU
permission persists; do not re-request ownership/admin confirmation.

Old B8 receipt peaks reverified exactly:1282.335866GiB allocated/4.394531reserved,
5127.674421/7.988281. Only12MiB512reservedmargin; prior no actualAdam measured.
Original source/protected73hashes, checkpoints7/8/50, Light hold and unrelated
README/SAKT dirt unchanged. Models/attention/old epoch engine untouched.

## Prepared solution

New `ktbench/favor_plus_microbatch.py`: physical2 preferred, physical4 speed
candidate. Preserve original logicalB8 membership/order and oneAdam per group.
Supervised-target weighting `sum_j n_j/N * masked_mean_BCE_j`; never equal-average
unequal-length micro means. Release each graph/batch/logits before next physical
forward. Same originalAdam/hyperparameters/attention/projections/precision/seeds/
full data/masks for botharms; no clipping/AMP/truncation/skips/reordering.
Dropout mask assignment and reduction order differ from physicalB8; reproducible
within new policy, not bitwise oldtrajectory equivalence. No BatchNorm coupling.
Length-aware within-group sorting only CPU analyzed, NOTselected (extra padding
savings4.39%physical2/3.53%physical4); preserve frozen order.

Eight epochs: same7499Adam updates/arm; physical2 has28494forwards, physical4
14506, physical1accountingonly56480. Train7060sessions/566107labels each epoch,
validation1977/125800. Physical2 maxcost13327, physical4max29942, no length cuts.
CPU retained-component sums128/512:physical2 .5624/1.2215GiB, physical4
1.1447/2.6384GiB. These are NOT GPUpeaks or admissionproof. Actual new memory,
Adamruntime and8epoch ETA **NOT MEASURED/NOT ESTIMABLE reliably**. Old6.13h
B8proxy notportable; micro2has3.80xforwardcount, sameAdamcount. No fabricatedGPU
speed extrapolation from CPU counts/accounting.

Final source-bound plan:
`experiments/hitskt_favor_plus_v1/assist2017/assist2017_orf128_512_logical8_micro_20261025_v3/prepared_plan.json`
SHA362a7f8b4db4625d8a6e1d0773d9bab57e7a0f3118217a5e825a61b032a8d632.
CPUvalidatedreload/full8schedulecoverage and112actualmicrobatchcollations
preserve every real token, shiftedlabel, predictionmask, EOS, history owner/order;
11 datafiles andmatchedfreshweights/RNG/seed/valID-labelhashes checked.

## Prospective profile and execution gates

`ktbench/favor_plus_micro_profile.py` + CLI `scripts/run_favor_plus_microbatch.py`:
fixed900s campaign,4freshsandboxworkers physical2/128,2/512,4/128,4/512,39realAdam
updates each (9extremes including post-split worst group +15strata x2),156total.
Two validationextrema+8timingstrata; no profileAUC/modelselection. Fixed TRAIN
initial/final diagnostics, unchanged.20 REVIEW/severe gates, allfinite model/
gradients/Adam/counter/projectionchecks. SavePROFILE_ONLYformat0weights+Adam/RNG,
fresh process exact prediction/Adam/RNG/nextproofgradrecovery without updates.
Never initializefulltrainingfromthese sandboxweights.

Same8GiBhardcap/10GiBadvisory/admission16GiB/runtime6GiBfree/hostRSS4GiB/host
headroom2GiB/30slogicalupdate andallsharedload/PID-limitedvisibilityguards.
Pacing.1s after EACHphysicaltrainmicro, guardbefore/after. Both chosenarms must
peakallocated ANDreserved <=6GiB for >=2GiBcapheadroom. Below8butabove6candidate
isREVIEW/noteligible, nothidden. Actual resource/numericalfailure STOP/no retry,
resizeorlimitincrease. PIDmappingstillunresolved; no exclusivityclaim.

`ktbench/favor_plus_micro_epochs.py` prepares separateformat6 checkpoint/epochs/
fullvalAUCselection/exactfreshrecoveries/curves/equal8epocharms/finalvalreplay.
No actualrun. Actualprofileandtraining scopefiles doNOTexist. Only false profile
template exists; futuretrainingtemplate requires successfulactualprofiles,
chooses common headroom-qualified fastestpolicy by memory/runtime only, then
requires separate actual training/runtimeapproval. Exact profile+trainingcommands
in report. Current new actualruntimebudget isNOTapproved; old7hnotauthorized.

## Tests, preserved drafts and exact next checkpoint

49newCPUtestsPASS;62existing regressions exercised in earliercurrent-taskreceipt.
111distinctpassing tests, not a full229rerun. Independent fullbatch-gradient/
Adam equivalence with unequallabelcounts, masks/eval/RNG, exact partiallogical
resume withdropout, freshsubprocessrecovery, no prematureAdam on resourcefailure,
formatseparation, approval/clock/cap/floor guards, finite34stagefakecampaign.
Initialfailurelogs/sources preserved: no-overwritecheckorder, missingtimestamp
mock, JSONstring/Pathloaderbug; mandatoryprepare-reload+newtestsadded. CPUdraftv1
retainedinvalid;v2retainedsupersededbeforeGPU afteractualaccountingfound post-split
worstoutsideoldB8extrema; finalv3frozen9cases/15strata/156steps. No GPUattempt.
8fileexact-bytearchive preservesallCPUdraftplans/templates/statusmarkers.

Finish verified report/memory/code publication to capstone-gpu researchbranch,
verifyremoteHEAD/main/unrelateddirt, thenWAIT unless an actual NEW profilewindow
reply arrives. No reply/preselectedoption is not approval. If approved, create
actualprofile record withfreshabsolute900sclock/planSHA/156max/evidence andrun
exactprofile-campaigncommand; first actual failureSTOP. Do not infertraining
permission. Afterprofile only: report actualpeaks/Adamruntime, commonchoice and
measured8epochbudget; obtain separate training/runtime instruction before epochs.

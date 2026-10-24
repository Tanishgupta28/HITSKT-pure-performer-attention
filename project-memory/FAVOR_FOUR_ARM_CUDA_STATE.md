# Four-arm FAVOR+ verification — 24 October 2026

Read `reports/favor_plus_v1/FOUR_ARM_CUDA_COMPARISON_AND_TRAINING_PROPOSAL.md`.
Starting commit33a17cf9e744342b2436d818554d4c619135e66b. User authorizes
CUDA correctness and controlled comparisons, but explicitly requires NEW approval
after memory/runtime/step proposal BEFORE ANY optimizer update. Current new
optimizer calls=0. Approval question sent for20/arm×4=80 total,15min ceiling;
NO RESPONSE IS NOT APPROVAL. Do not reuse historical full8/pilot approval.

FACTS: new matched wrapper/factory, separate checkpointformat4 and native
original/prototype cores; identical learned parameter/post-factory CPU RNG
hashes across ORF128/512/anti128/512. Common nested512pool stage seeds42–45;
ORF128 draw differs from historical native128 for some heads, explicitly
declared. Old model/default/sampler/prototype sources unmodified.

FP32 study stopped at ORF512 L65/chunk16/norm3 independent-reference gate.
ORF128 passed24CUDA kernels/fullmodel/reload; FP32 antiarms unrun. Oldcombined
assertion lacks individualGPUdelta; do not invent it. Deterministic CPU input
reconstruction(original runtime) FP32error6.582114e-5>unchanged5e-5, FP64
4.629215e-7; gradientrulesPASS. Saved failure/source snapshots/provenance remain.
Separate explicit all-arm FP64 attention followup (neuralparams/outputsFP32)
passed96CUDA kernel cases, own/future/batch/PAD/BOS/EOS/gradient invariants,
four exact fresh reloads. Max independentoutput4.740996e-7/grad9.353855e-9;
fullmodel CPUCUDA output3.576279e-7/parametergrad1.788139e-7. Initial45minclock
inherited, no reset; terminalcomplete marker. Do NOT rerun either closed study.

Same56 initial TRAIN QKV arrays(14cases×4stages), prefixes8/32/128/256/512/full
through939, norms.5/1/1.5 =483correlated rows/arm. No trained7/8 weights as
fair-comparison evidence, no new validation/test inference. At originalnorms
≤128: Action median ORF128.243430/ORF512.152904/anti128.251865/anti512.179270;
CPE.156680/.066212/.075872/.049956. .20 REVIEW cases Action12/1/11/2 of14,
not all PASS. Norm1.5 Actionall14revieweveryarm. Nativefeatureoracle agreement
all1932rowsPASS,max1.145447e-7. AntithetichelpsCPE, ORF512betterAction; no
universalfix/statisticalclaim/learningretention evidence. Oldalignment remains
hypothesis with priortrained evidence, not resolved atinitialization.

Shared authorized MIG-d2fcfd79-720e-5218-82c9-0a132be68920 H1003g.40gb.
Existing8GiB allocator/10advisory/16admission/6runtime/4GiBhost/2headroom,
.1pacing/30sstep safeguards retained. PIDmappingUNRESOLVED/utilunsupported;
notexclusiveownership/nopeeridentity inference. GPUprofilecost88/246/485/2047,
moment-sizedbuffersreserved, noAdamstep. Peaklive393.143MiB128/986.627MiB512;
reserve486/1220MiB; maxF/B.51054/.52145/.54271/.56303s armorder.
MaxhostRSS1656.801MiB, minimumsampledfree37.786GiB. NoF64CUDA/OOM/nonfinite/
resourcefailure; noothersignals. ProtectedLight hold hash unchanged.

CPU final110PASS/0fail=22newfactory/approval/checkpointpreparation+88existing,
not new125+229 counts. CPU-onlyTorch2.4.1 build/hiddenCUDA/taskset223/nice10;
globalsharedGPUruntime unchanged. Positiveoptimizer path NOT exercised.

PREPARED PROPOSAL ONLY:
`experiments/hitskt_favor_plus_v1/assist2017/matched_four_arm20_fp64_20261024_v1/prepared_plan.json`
SHA d1ff9b3d858d9a0ce49905aedcd66e8bc672a7254acc630cdb5c239774004c15.
20freshupdates/arm80max,fixedseed42/order/Adam5e-5 betas.9,.999 eps1e-8,
FP64attentiononly,B1/cost2048,15minTOTAL ceiling,same8GiBhardcap.
Estimate1–1.5GiBlive+unmeasuredAdamworkspace;10–15minplanningallowance not
measuredoptimizerETA. Frozen32val/3157labels1063positive2094negative at0/5/20,
sixTRAINDIAG native+independent17/101auditdraws, checkpoints0/5/10/15/20,
fresh no-step recoveryafter20. .20 REVIEW unchanged; originalseveregatesretained.
`scripts/pilot_favor_plus_matched.py` requires actualnewapprovalrecord+explicit
optimizerflag; false template is not authorization. No automaticdriver/watcher.
Before futureexecution revalidate source/plan/readiness/data identities and
freshscopeclock/sharedadmission, executeeacharm thenfreshrecoversequentially;
stopentirecomparisononfailure/no retries. Any sourcechangeinvalidatesfrozenplan.

DECISION: GO boundedmatchedlearning ONLY after actualapproval; REVIEW/NO-GO
choosingfull-epochwinner or launchlongertraining. Learnedapproximation/AUC/F1
unverified. No untouchedtest or additionaltrainingseeds authorized.
Preserveold7SHA07b944e7...,8SHAe0fe6216...,50SHA93ebe9d2...,allhistorical
ELU/FAVOR/Light/DKT/data/audits and unrelatedREADME+SAKT dirt/main.
Finish scoped report/archive/memory validation/commit/push capstone-gpu research
branch only then WAIT. Never infer pendingapproval fromelapsedtime/defaultchoice.

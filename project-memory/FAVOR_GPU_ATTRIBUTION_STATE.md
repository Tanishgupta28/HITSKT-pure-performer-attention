# GPU process attribution —21 October2026

## Scope and outcome

User requests read-onlydiagnosis/CPUtestsONLY, no CUDA oroptimizerupdates. Prior
30additional/50TOTALapprovalpersists; no repeatrequest, no automaticexecution.
Report and WAIT for nextinstruction eveniflaterresourcesappearidle.

**PID ownership NOT FULLY RESOLVED; independent availability NOT VERIFIED/NO-GO.**
Historical67123↔2847964 sameprocessand competingjob hypotheses bothUNKNOWN;
directoldstop was strictnumericNVML/selfcomparisonwithouttrustedPIDdomainbridge.
No aliases/guardbypassinstalled. Currentresource also has activeunattributeduse.

Report: `reports/favor_plus_v1/GPU_PROCESS_ATTRIBUTION_DIAGNOSIS.md`.
Evidence: `reports/favor_plus_v1/gpu_pid_attribution_20261021_v1/`.
AdminrequestGPU_ADMIN_EVIDENCE_REQUEST.md prepared only, nothing sent.

## Established evidence

-06:59:10UTC snapshot exactMIG-d2fcfd79-720e-5218-82c9-0a132be68920/H1003g.40gb,
  parentGPU-5a568e3c-e96c-169f-b063-f4478340ff51/GI2CI0. Capabilityminors21/22,
  proc/cgroupaccess/NVML/SMIinventory agree. Access isnotexclusiveallocationproof.
- NVMLcompute2265424 used2885681152bytes, MIGused2934636544. VisibleTorchdevice
  holder3136171 (prootedbasename, cwdother, UID0) differsfromcollector3136614;
  mapping2265424↔3136171UNPROVED. Do notstop/attach/modify eitherjob. Earlier
  preliminarysnapshot differentactivePIDs, no backgroundpolling.
- AllvisiblePIDnamespaceinode4026559509/localNSpidonly; NS_GET_PARENT EPERM,
  noCAP_SYS_ADMIN/SYS_PTRACE/hostproc/runtime socket/NVIDIAclientsfile. Oldpair
  gone. CurrentNVMLPIDnotvisible; NVMLprocessnamequeryreturns6NOT_FOUND; SMI
  computequeryexit0withInsufficientPermissions. Emptylocalviewisnotownership.
- Driver535.161.08/NVML12.535.161.08. ExistingctypesstructverifiedagainstCcompiled
  officialnvml.h: size24,offsets0/8/16/20. TestedABIlayoutmismatchnotsupported.
- MPSqueryempty/localpipes/envabsent; doesnotprovepast/hostbackendnotMPS.
  Traced/prootedcurrentholderdoesnotproveoldnamespacealias. OfficialLinux/NVIDIA
  primarydocslinkedinreportexplainlimitations, notusedasproofspecificPIDmapping.

## Changes, validation and preservation

Standalone `ktbench/gpu_process_diagnostics.py` and
`scripts/diagnose_hitskt_gpu_processes.py` collect safe metadata/fingerprints,
namespace/cgroup/NVML/deviceevidence. NoTorch/libCUDAloadedinactualcollector,
noadmissiondecision canauthorizeGPU. Helper flagsmissing/racyPID/exe/UID/ns/
cgroup data. Extra racechecksadded/testedaftercapture, notretroactivelyclaimed.
Productionguardsandall28existingboundsourcefilesUNCHANGED. Noactualverified
hostbinding, so implementingaliasadmissionwouldbeassumptiveandwasnotdone.

31CPUcasesPASS(29new+2existingNVMLguards),0failures/skips. No229/103fullrepeat.
199original20runfiles incl5/10/20.pt and9failedcontinuationfiles rehashunchanged;
sameconfig/RNG/projections/results, Light hold65214c2c...same, no DKT/otherjobs/
allocations/signals/watchers/architecture/thresholdchanges. UnrelatedREADME/SAKT
dirtunchanged. Totalrealupdates20, new0; no newloss/AUC/accuracyclaim.

## Exact next checkpoint

Need trustedhost/runtime PID↔containerbridge withtimestamps,boot/starttime,
NSpid/NStgid/nsinode/cgroup/container/job binding and exactMIG/GI/CI, including
currentcompute and allMPS/proxyclients ifrelevant; plus independentavailable
assignment withoutotherreservation/workloadconflict. Historicallogsifavailable;
ifnot, leaveoldpairUNKNOWN and establishfuturemappingseparately. Existingowner
permission isnotmissing. No repeatKubernetes/authorizationrequest, no CUDAprobe
toinferalias, caller-writtenPIDwhitelist/nameexemption/sameaccountblankettrust.

Publishverifiedresearchbranchchanges, verifyremote/mainunchanged, thenWAIT.
NoGPU/traininguponautomaticcontinuation; usermustgivenextinstructionafter
diagnosis. Existingboundapprovaldoesnotpermitignoringunknown/activeworkloads.

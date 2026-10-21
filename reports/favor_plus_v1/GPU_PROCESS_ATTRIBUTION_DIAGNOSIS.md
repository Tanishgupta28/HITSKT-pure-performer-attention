# HiTSKT FAVOR+: GPU process attribution diagnosis

21 October2026 UTC. Branch `research/hitskt-favor-plus-v1`, base
`4927490049e0c5f26833d05e9219c42ddcfa6db0`.

**Ownership attribution is NOT fully resolved. NO-GO for CUDA/training.** The
historical NVML67123/container2847964 relationship cannot be established from
accessible evidence. Current read-only observations additionally show active
unattributed GPU use. Existing owner authorization is retained; no repeat grant
is requested. This diagnostic task initializes no CUDA and takes zero optimizer
updates. Training stays at20, and the0.20 approximation review threshold is unchanged.

## Findings and certainty

|Check|Status|Evidence and implication|
|---|---|---|
|Assigned device identity/access|PASS|Same H1003g.40gb MIG, parent UUID, GI2/CI0 and corresponding capability minors21/22|
|Existing research-allocation authorization|RETAINED|Earlier explicit owner's allocation/transfer statement; not inferred from memory usage or device access|
|Historical67123 equals2847964|NOT VERIFIED|Both PIDs gone; no archived host namespace mapping|
|Historical67123 was another workload|NOT VERIFIED|Different numeric IDs alone do not establish that either|
|Complete host process visibility|UNAVAILABLE|Local procfs/PID namespace, NS_GET_PARENT EPERM, no host proc/runtime socket; process-name query cannot resolve current NVML PID|
|Current independent/exclusive availability|BLOCKED|Active NVML compute plus an unknown visible PyTorch device holder, unrelated to this diagnostic process|
|NVML process-struct layout|PASS|Existing ctypes layout matches a CPU-compiled check against NVIDIA's installed C header|
|Production resource guard|UNCHANGED|No numeric aliases, process-name exemptions, relaxed checks or bypass|
|CPU checks|PASS|31 distinct cases:29 new diagnostic cases and2 existing NVML guard cases; zero failures/skips|
|20-update artifacts/LightSAKT2|PRESERVED|Original checkpoint/run hashes and Light hold checked; no process signals or job/config changes|

## What caused the historical stop

The direct cause is established in
[the preserved failure](../../experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261020_v2/study_failure.json):
at20 October17:15:55UTC, NVML listed PID67123 while the HiTSKT process's
`os.getpid()` was2847964. The boundary rejects every NVML compute PID unequal
to its own PID. No trusted bridge established that the two PID values shared
a namespace or represented the same task, so the guard correctly failed closed.
It stopped before update21 after an exact, zero-Adam-call checkpoint20 recovery.

The **underlying identity relationship remains unknown**. The old NVML entry
disappearing after the HiTSKT process exited is correlation, not proof of self
identity. Neither a benign namespace alias nor a competing workload is confirmed.
A username, executable name, coincident timing or similar memory usage would
also be insufficient to establish that historical mapping.

Linux documents NSpid/NStgid relative to the namespace of the procfs mount and
successively nested namespaces; the leftmost value need not be the physical
host PID. A one-entry local NSpid vector therefore does not reveal the missing
outer host ID. [Linux proc_pid_status](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html).

## Current read-only observations

[Timestamped resource/namespace evidence](gpu_pid_attribution_20261021_v1/RESOURCE_AND_NAMESPACE_EVIDENCE.json),
captured06:59:10UTC, records:

- Only exposed MIG **MIG-d2fcfd79-720e-5218-82c9-0a132be68920**,
  parent **GPU-5a568e3c-e96c-169f-b063-f4478340ff51**, H10080GB/HBM3,
  MIG3g.40gb, GPU instance2/compute instance0. Driver535.161.08,
  NVML12.535.161.08. NVML, inventory and read-only proc capability files agree.
- NVML compute PID **2265424**, used GPU memory **2,885,681,152bytes**; total MIG
  used2,934,636,544bytes. The visible `/proc` device holder is a different process,
  **3136171**, comm `pt_main_thread`, root UID, executable basename
  `prooted-3135167-ccFkXk`, cwd category `other`, with the assigned capability
  nodes and NVIDIA device nodes open. The diagnostic process is **3136614**.
  **No association between2265424 and3136171 is proved.** Both observations block
  borrowing the device, regardless of substantial remaining VRAM.
- PID2265424 is not visible in local `/proc`. `nvmlSystemGetProcessName` returns
  status6 (`NVML_ERROR_NOT_FOUND`) for that PID but resolves visible local PIDs.
  NVIDIA's parent-level process query exits0 while printing **Insufficient
  Permissions** for process details. Neither outcome means the device is idle.
  NVIDIA documents the name query's process-not-found/permission error cases.
  [NVML system queries](https://docs.nvidia.com/deploy/archive/R535/nvml-api/group__nvmlSystemQueries.html).
- Collector/PID1/visible-holder namespace links are all `pid:[4026559509]`;
  each exposes only its local NSpid. Parent-namespace discovery returns
  **EPERM**, and CAP_SYS_ADMIN/CAP_SYS_PTRACE are absent. No host `/proc`,
  container-runtime socket or NVIDIA clients file is available at the inspected
  paths. Cgroupv1 mounts are restricted to this Kubernetes/container scope.
  EPERM establishes unavailable parent discovery, not the host/child mapping.
  [Linux NS_GET_PARENT](https://man7.org/linux/man-pages/man2/NS_GET_PARENT.2const.html).

A preliminary06:54 observation showed a different NVML PID2222405 using about
6.15GiB and a different unknown PyTorch device holder. Workload activity changed
between finite observations; no background watcher was started. The timestamped
06:59 capture is the published detailed snapshot, not a claim about future
availability. No inspected workload was stopped, attached to with ptrace, signaled
or otherwise modified; reading TracerPid does not attach a tracer.

[Device file evidence](gpu_pid_attribution_20261021_v1/DEVICE_FILE_EVIDENCE.json)
independently maps GI2→capability minor21 and CI0→minor22, with cgroup device
access entries. These prove exposure/access to the assigned resource, **not**
exclusive reservation or absence of work in inaccessible namespaces. No
Kubernetes request was repeated and no allocation record/window was refreshed.

## Alternative explanations checked

1. **ABI misread:** not supported by the tested layout. A C program includes
   installed `/usr/local/cuda/include/nvml.h` and checks the actual
   `nvmlProcessInfo_v2_t`; it links no NVML/CUDA and runs only on CPU. Size24,
   offsets pid0/usedGpuMemory8/gpuInstanceId16/computeInstanceId20 match the
   existing guard's ctypes type captured through a fake-NVML fixture.
   [C source](gpu_pid_attribution_20261021_v1/nvml_process_abi.c),
   [test log](gpu_pid_attribution_20261021_v1/targeted_cpu.log).
2. **MPS server/client identity:** possible in general, not established here.
   NVIDIA documents that MPS can expose the server instead of client PIDs to
   NVML. This snapshot's MPS-process query returned an empty list, inspected
   MPS directories were absent and no selected MPS environment variable was
   configured. These observations do not prove what happened yesterday or
   exclude an inaccessible backend. No MPS control command was run.
   [NVIDIA MPS documentation](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf).
3. **Runtime/tracing wrapper:** the visible holder has a `prooted-*` executable
   and a nonzero TracerPid. That describes current local process metadata; it
   does not establish a host PID alias or explain the vanished historical pair.
4. **Other GPU work:** active current compute is independently reported by NVML.
   Exact job ownership and its relationship to the visible holder are unavailable.
   Root UID or a shared pod/cgroup is not permission to interfere with it.

## Changes and tests

Added a **standalone evidence collector**, not an admission-policy change:
[diagnostic helpers](../../ktbench/gpu_process_diagnostics.py) and
[CLI](../../scripts/diagnose_hitskt_gpu_processes.py). It collects selected process
identity, namespace/cgroup fingerprints, NVIDIA management metadata and device
IDs. It neither imports torch nor loads libcuda in the captured process, and
has no CUDA/training/resource-reassignment entry point. It hides arbitrary argv
and unrelated environment values; fingerprints support provenance.

The helper explicitly treats missing, malformed, exited, reused, changed-exe,
changed-namespace, changed-UID and moved-cgroup identities as incomplete evidence.
Even a valid multi-level NSpid vector is **evidence, never authorization**. Every
diagnostic outcome has gpu_execution_allowed=false, including empty process lists.
Additional identity-race hardening was tested after the live capture; that
capture is not retrospectively claimed to have performed the newer rechecks.
No second live capture was taken merely to make it match the hardened helper.

**31 CPU tests PASS** (29 new +2 existing), zero failures/skips. They cover safe
metadata parsing/redaction, namespace chains, PID/exe/UID/cgroup races, host
visibility denial, active/unknown/inaccessible process blocking, no torch/CUDA
import, independent C ABI verification, and the unchanged production guard's
exact-self versus unknown-PID behavior. [JUnit](gpu_pid_attribution_20261021_v1/targeted_cpu.xml).
Existing103/229 suites are inherited, not redundantly rerun or counted as new.

Because no actual historical/live host-to-container binding was proved, **no
PID-alias admission mechanism was installed**. Accepting NSpid guesses, all
processes in the same cgroup, names or a caller-written alias file would weaken
the guard. All existing scientific sources, experiment contracts, production
resource guards and the0.20 threshold remain unchanged.

## Precisely what is needed and next action

[Prepared administrator evidence request](gpu_pid_attribution_20261021_v1/GPU_ADMIN_EVIDENCE_REQUEST.md)
specifies the UUID/GI/CI, historical window and current snapshot, host process
status/start time/namespace/cgroup identity, container/job mapping, runtime/MPS
mapping if relevant, and exclusive-availability evidence. It is a file for the
user; **nothing was sent to an administrator**.

A future safe mapping mechanism must consume fresh, trusted host/runtime
evidence that binds exact host PID and start time/boot identity to the exact
local process/namespace/container and assigned MIG. It must recheck for PID
reuse, reject all other/unknown/inaccessible workers, retain existing allocation,
Light-hold and memory checks, and fail closed when the bridge is absent or stale.
That mechanism cannot be honestly validated from the currently missing inputs.

**Final decision:** existing project authorization retained; independent current
availability and historical PID ownership **NOT VERIFIED**, active use observed,
GPU issue **NOT FULLY RESOLVED**. Stop and await the requested host evidence and
the user's next instruction. No CUDA execution or optimizer update occurred in
this task, no full training is cleared, and the prior bounded continuation is
not automatically resumed even if a later process list happens to be empty.

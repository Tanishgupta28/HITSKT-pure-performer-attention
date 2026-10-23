# Administrator procedure: identify the NVML PID domain

23October2026; preparation only, base865b384521c654cd256b449f8e30305fce50330e.
**No CUDA, optimizer update or new GPU admission was executed in this task.**

Most likely working explanation: a host/container PID-domain or runtime/proxy
identity mismatch. This is an inference from the repeated empty→unknownPID→empty
pattern around exact checkpoint recovery, not a verified process mapping.
Confirmed: the guard compares NVML PID directly with `os.getpid()` and rejected
488211 versus3843909 on23October00:55:23UTC, before update21. The historical
67123↔2847964 pair is also unresolved. Parth's reported concurrent job cannot
establish either mapping. Training remains20; checkpoint20 is preserved.

## Evidence needed

|Independent question|Required trusted evidence|
|---|---|
|Is it the same task?|Live host `/proc/HOST_PID/status` NSpid/NStgid chain containing the exact local PID, matching target PID namespace inode, stable field22 start ticks, boot ID, clock tick rate, executable identity and runtime container/cgroup binding|
|Does NVML report that task?|Host and container NVML records for the same interval, assigned UUID/GI/CI and PID-domain interpretation; if a proxy/MPS server is reported, authoritative client/server mappings including every client|
|May it run exclusively?|Existing account/project allocation for this exact MIG and time window, runtime/device assignment and administrator/allocator confirmation of no conflicting reservation or shared/time-sliced clients|

The namespace chain is relative to the procfs mount's PID namespace; local
NSpid with one number cannot reveal outer PIDs. [Linux status documentation](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html).
Record field22 rather than a rounded `ps` start-time display; include boot ID
and `SC_CLK_TCK`. [Linux stat documentation](https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html).
Capture reader/target time namespaces and offsets too; if they differ, do not
automatically equate start ticks without checked normalization. [Linux time namespaces](https://man7.org/linux/man-pages/man7/time_namespaces.7.html).

## 1. Establish the node, resource and allocation (administrator only)

Run in the genuine GPU node host environment, not merely another shell in the
restricted pod. Use a fresh private evidence directory; save stdout, stderr,
UTC times and exit statuses. Commands query metadata and do not change jobs.

```bash
HITSKT_PARENT='GPU-5a568e3c-e96c-169f-b063-f4478340ff51'
HITSKT_MIG='MIG-d2fcfd79-720e-5218-82c9-0a132be68920'
date -u --iso-8601=ns
sudo nvidia-smi -L
sudo nvidia-smi -i "$HITSKT_PARENT" -q -x
sudo nvidia-smi -i "$HITSKT_PARENT" --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv
sudo nvidia-smi mig -i "$HITSKT_PARENT" -lgi
sudo nvidia-smi mig -i "$HITSKT_PARENT" -gi 2 -lci
```

Expected historical identity: H1003g.40gb/GI2/CI0; verify current identity rather
than assuming index0 or old GI numbering. Parent query rows alone do not identify
the MIG: retain per-process GI/CI from XML and exact MIG inventory/instance data.
If unavailable or denied, obtain privileged **MIG-specific NVML** compute,
graphics and MPS snapshots and API statuses. Unsupported/denied is UNKNOWN,
never empty. [NVIDIA NVML R535](https://docs.nvidia.com/deploy/archive/R535/nvml-api/group__nvmlDeviceQueries.html),
[NVIDIA query options](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

For this Kubernetes environment, find the existing pod by its recorded UID:

```bash
kubectl get pods -A -o json | jq --arg uid 'c5f7f8ec-4e1f-4b24-8ab5-a48350b23dc2' '
  [.items[] | select(.metadata.uid==$uid) |
   {namespace:.metadata.namespace,name:.metadata.name,uid:.metadata.uid,
    node:.spec.nodeName,containers:[.spec.containers[]|{name,resources}],
    runtime:[.status.containerStatuses[]?|{name,containerID}]}]'
# Administrator sets HITSKT_CID from the current full runtime container ID above,
# removing its containerd:// prefix. Inspect only selected nonsecret metadata:
sudo crictl inspect "$HITSKT_CID" | jq '
  {id:.status.id,podUID:.status.labels["io.kubernetes.pod.uid"],
   initHostPID:.info.pid,cgroupsPath:.info.runtimeSpec.linux.cgroupsPath,
   devices:.info.runtimeSpec.linux.devices}'
```

If this UID no longer exists, determine the current assigned container; do not
reuse an old ID. CRI inspection schemas vary; null PID/device fields need actual
runtime evidence, not guessed values. [Official CRI inspection](https://kubernetes.io/docs/tasks/debug/debug-cluster/crictl/).
Obtain the kubelet PodResources/device-plugin or equivalent assignment showing
this exact MIG, and the applicable allocation/reservation record and window.
Resource counts, root UID, device files and idle memory do not establish
exclusive ownership. Ask whether GPU/MIG sharing, MPS or time slicing is enabled;
an allocated replica can be shared. [NVIDIA GPU sharing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/24.3.0/gpu-sharing.html).
This verifies existing permission/availability; it does not request a new grant
or authorize changes to a scheduler, device plugin, allocation or workload.

## 2. Coordinate a CPU-only witness while it is alive

Administrator supplies a fresh32-hex nonce and reviews a controlled copy of
`scripts/collect_hitskt_pid_identity.py`. In the assigned container, run this
in the **foreground** at the agreed time, then let it exit naturally:

```bash
# Set HITSKT_NONCE to the administrator's fresh nonce, not an old example value.
env CUDA_VISIBLE_DEVICES='' python scripts/collect_hitskt_pid_identity.py \
  --view container --nonce "$HITSKT_NONCE" \
  --output /tmp/hitskt-container-witness.json --hold-seconds 45
```

Choose a fresh output path if that file exists; the tool refuses overwriting.
It emits local PID/namespace/start/cgroup fingerprints before one45-second CPU
pause. No polling, background job, model/checkpoint load, torch, NVIDIA API,
CUDA context, backward pass or optimizer object is involved. The output itself
never grants GPU permission. **A CPU witness will not appear in NVML.**

Administrator reads the live witness's local PID and `namespaces.pid` string
from that output and locates its host PID using a finite host proc scan:

```bash
# Set HITSKT_LOCAL_PID and HITSKT_PID_NS (e.g. pid:[4026559509]) from the LIVE output.
sudo python3 - "$HITSKT_LOCAL_PID" "$HITSKT_PID_NS" <<'PY'
import os, sys
from pathlib import Path
local, namespace = int(sys.argv[1]), sys.argv[2]
matches, unreadable = [], []
for p in Path('/proc').iterdir():
    if not p.name.isdecimal(): continue
    try:
        if os.readlink(p/'ns/pid') != namespace: continue
        fields = dict(line.split(':',1) for line in (p/'status').read_text().splitlines() if ':' in line)
        if int(fields['NSpid'].split()[-1]) == local: matches.append(int(p.name))
    except FileNotFoundError: continue
    except (PermissionError, KeyError, ValueError): unreadable.append(int(p.name))
print({'candidate_host_pids':matches, 'unreadable_pids':unreadable})
PY
```

A candidate is not proof. While still alive, set HITSKT_HOST_PID to the unique
candidate, HITSKT_INIT_PID to CRI's host init PID, and HITSKT_TOOL to the reviewed
host copy. Missing/multiple candidates or unexplained access gaps remain UNKNOWN.

```bash
sudo python3 "$HITSKT_TOOL" --view host --nonce "$HITSKT_NONCE" \
  --pid "$HITSKT_HOST_PID" --pid "$HITSKT_INIT_PID"
sudo cat "/proc/$HITSKT_HOST_PID/status" "/proc/$HITSKT_HOST_PID/cgroup"
sudo readlink "/proc/$HITSKT_HOST_PID/ns/pid" "/proc/$HITSKT_INIT_PID/ns/pid"
date -u --iso-8601=ns
```

The tool records each identity twice to detect PID reuse/exec/cgroup/namespace
changes. Join host NSpid to local PID AND namespace; compare stable start ticks,
boot/clock domain, runtime ownership and cgroup fingerprints. Same names/account
or similar memory/timing are insufficient. Do not dump environment/argv/secrets.
The `--view host` label is not attestation: the administrator must authenticate
the collection source/node and send evidence over a trusted channel.

## 3. Establish NVML mapping; do not confuse this with step2

First request archived host/runtime records for23October00:55:20–00:55:24UTC
(NVML488211/local3843909), and20October17:15:52–17:15:56UTC
(NVML67123/local2847964). Today's `/proc` with recycled IDs cannot prove history.

If archives are absent, step2 can prove a **future namespace mapping only**.
A separate, explicitly coordinated **zero-optimizer CUDA identity diagnostic**
would still be needed to prove which domain NVML uses. It is NOT authorized or
run in this CPU-only task, and is not a rerun of the failed training controller.
Only after exclusive allocation and host observer readiness are established,
prepare that bounded diagnostic in a separate future task: no training pipeline,
dataset, checkpoint write, forward/backward or optimizer, one minimal context,
finite lifetime and synchronous selected `/proc` plus host/container MIG-NVML
captures before/during/after. Stop on ambiguity; never disable a production guard
to make it run. If the existing guard cannot support safe observation, obtain
the administrator's authoritative tracing/runtime record rather than bypass it.

Retain all current compute identities/GI/CI and runtime client mappings, not just
the candidate matching HiTSKT. A proxy/MPS server or unrelated PID is not a
direct namespace alias and needs its own complete, authoritative attribution.

## Outcome and handoff

Direct mapping is established only if trusted live host records bind the exact
NVML worker to the exact local task with namespace/start/boot/runtime identity,
and exclusive allocation is separately verified. Otherwise remain blocked.
Keep vanished historical pairs UNKNOWN if records are unavailable. No process
signals, tracing attachment, namespace entry, reset/restart, debug pod creation,
allocation mutation or watcher is part of these commands. Further **host-level
evidence/intervention is required**, not another HiTSKT experiment approval.

See [conditional guard design](GPU_PID_ATTRIBUTION_CORRECTION_PROPOSAL.md).
No administrator was contacted and no host command above was executed here.

CPU preparation verification: **31 tests PASS** (28 new witness cases and3
existing fail-closed guard cases), zero failures/skips; no full229-suite repeat.
The actual local CPU witness imported neither torch nor libcuda and granted no
GPU permission. All28 prior bound sources,5/10/20 checkpoint hashes,7 files from
the latest failed attempt and the Light hold are unchanged. Five shell blocks
and embedded Python syntax were checked without executing host commands.
[Evidence summary](host_attribution_preparation_20261023_v1/VERIFICATION_SUMMARY.json),
[CPU log](host_attribution_preparation_20261023_v1/targeted_cpu.log).

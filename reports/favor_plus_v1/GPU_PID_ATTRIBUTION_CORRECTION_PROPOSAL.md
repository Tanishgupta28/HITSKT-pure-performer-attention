# Conditional GPU attribution correction — preparation only

23October2026. **No production guard change or PID alias admitted.**
Related [administrator procedure](GPU_ADMIN_PID_ATTRIBUTION_PROCEDURE.md).

The current boundary in `ktbench/favor_plus_assist50.py` rejects each NVML PID
unequal to `os.getpid()`. This correctly fails closed on unknown identities but
assumes a shared PID domain. A namespace-aware replacement is justified only
after a direct mapping and the NVML reporting domain are proved. A runtime
proxy/MPS server requires a different client-ownership solution; do not present
it as a direct PID alias.

## Prepared safe code

`scripts/collect_hitskt_pid_identity.py` is a standalone stdlib/procfs witness.
It captures selected PID/Tgid/NSpid/NStgid/UID/TracerPid, namespace links, boot ID,
field22 start ticks/clock rate, UID/GID maps/time offsets, executable identity,
cgroup fingerprints and candidate container IDs. It rereads identity to detect
reuse/exec/cgroup/namespace/UID changes. It rejects malformed/oversize metadata,
has bounded PID counts, refuses output overwrite, and optionally pauses once
for at most45s in the foreground. No raw argv/environment is collected.

This tool's host/container label is operator supplied, not authenticated proof.
Every output says mapping_proven=false and gpu_execution_allowed=false. It does
not query NVML, import torch, initialize CUDA, run admission or modify resources.
The optional pause is a live CPU witness, not a background watcher. No new model,
optimizer, numerical-threshold or checkpoint-format behavior is introduced.

## Future integration, conditional on evidence

1. Establish source trust: an administrator-controlled, authenticated read-only
   host/runtime endpoint, unavailable for writing/forging from this container.
   An operator JSON/PID whitelist, shared UID/cgroup, memory size or matching
   executable is not an authority. A plain root-owned file inside this root-UID
   container is not automatically protected from the workload.
2. Resolve the requesting **live** local process using a host-held process
   identity/pidfd where supported, boot ID, start ticks, PID namespace chain,
   container/cgroup and exact MIG. Verify the NVML PID domain from the coordinated
   evidence; do not assume it is host-root PID or take an arbitrary NSpid suffix.
   Include reader clock/time-namespace domains; missing or incompatible evidence
   fails closed. PID integers alone, including equal integers, are insufficient
   when PID domains are unknown.
3. At every current boundary issue a fresh challenge/nonce to that trusted
   endpoint. Require bounded freshness, boot/start-time and request/process
   binding, observed exact UUID/parent/GI/CI and a complete successful worker
   snapshot. Recheck identity before/after query to reject reuse, exec and moved
   namespaces/cgroups. A stale signed snapshot alone cannot guarantee liveness;
   retain a live identity handle/revalidation. Agree freshness rules before use.
4. Accept an NVML worker as self only when the trusted mapping resolves to that
   exact task. Once a compute context is active, require the expected mapped
   worker to be visible; an unexplained empty list is not self attribution.
   Reject **all other/unknown/inaccessible workers**, including other
   tasks in the same pod/account/cgroup. If MPS/proxy ownership is incomplete or
   other clients exist, stop. Never grant an alias based on disappearance at exit.
5. Retain separate allocation/transfer/action approval, exclusive reservation,
   exact visible UUID/device access, LightSAKT2 hold, trusted read-only monitor
   classification, RAM/VRAM/allocator/time limits and checkpoint/causal/numerical
   gates. Mapping proves identity, not permission. CPU witness consistency cannot
   authorize training.
6. Source-bound contracts require a **new controller version and fresh run**
   after tested integration. Keep all existing28 sources/sealed plans/failures
   and checkpoint20 intact; do not edit old plan hashes or widen a legacy
   validator. Restore the exact parent weights/Adam/RNG/projections/sampler under
   the remaining30-update approval only after safety is satisfied. No full run.

Required future tests: correct direct/nested PID mappings; same-number different
namespace; PID reuse and changed boot/start time; missing/untrusted/forged/stale/
replayed source; swapped container/cgroup/MIG; time-namespace mismatch; incomplete
NVML visibility or a missing expected live worker; unrelated worker alongside
self; MPS/proxy extra clients; and
failures before CUDA APIs/Adam with original memory/hold/time checks retained.

**Not implemented:** a trusted endpoint, authenticated mapping admission or
production alias resolution. Actual inputs and host access are missing. Changing
the running guard now would replace evidence with assumptions. The present work
is a tested collection aid and a reviewable correction design, not a resolved
GPU ownership issue.

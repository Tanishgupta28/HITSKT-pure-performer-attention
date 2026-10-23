# Remaining technical evidence, not another experiment approval

The user's allocation/transfer permission and exactly30 additional/50TOTAL
approval remain accepted. No new allocation or repeated user approval is being
requested. No request has been sent externally.

The23October2026 attempt again encounters an unresolved identity:

- Assigned H1003g.40gb MIG-d2fcfd79-720e-5218-82c9-0a132be68920,
  parent GPU-5a568e3c-e96c-169f-b063-f4478340ff51, GI2/CI0.
-00:55:20–00:55:24UTC: HiTSKT local PID3843909; native NVML compute PID488211,
  used578,813,952bytes. No other visible descriptor holder; exact checkpoint20
  no-step recovery passed, then the guard stopped before update21.
-00:55:46UTC: no MIG compute worker; memory returned to baseline. This timing
  does not establish whether the NVML entry represented our context.
- Container PID namespace inode4026559509; Kubernetes pod UID
  c5f7f8ec-4e1f-4b24-8ab5-a48350b23dc2. Local NSpid exposes no outer chain;
  parent namespace lookup returns EPERM and host proc/runtime access is absent.

Needed from an authoritative host/runtime source:

1. If records exist for that window, bind or distinguish NVML488211 and
   container3843909 using process/Tgid/NSpid/NStgid, boot identity and start-time
   ticks, namespaces, cgroup and runtime container/job identity. Do not certify
   equality just from memory size, names, shared account or timing. If records
   are absent, keep this historical pair and67123↔2847964 UNKNOWN.
2. Establish which PID domain NVML reports. If a driver/runtime, tracing wrapper,
   CUDA proxy or MPS backend changes identity, supply its authoritative
   client↔server/container mapping and identify other clients on GI2/CI0.
3. Provide a sustainable, narrowly scoped read-only identity bridge accessible
   to the guard, with current timestamp, boot/start-time/namespace/cgroup and
   exact MIG bindings and visibility of competing workers. Alternatively fix
   the runtime attribution so the existing exact-self check is reliable.
   A caller-written alias whitelist or exemption for any same-account PID is
   insufficient. The guard must still reject every unknown or unrelated worker.

Any new bridge must be tested on CPU and admitted through a separately versioned
execution contract while preserving the original20 checkpoint. Do not edit old
sealed plans, exempt unknown processes, retry this failed campaign or launch
CUDA merely to guess the mapping. No further GPU execution now; resume the
already approved scope only after this safety condition is verifiably met.

# Read-only host evidence needed for HiTSKT GPU attribution

This requests technical attribution/availability evidence, not another approval
of the already owner-authorized20→50 study. No job termination, allocation
change, CUDA test or MPS-control action is requested. Please omit credentials,
environment dumps and arbitrary command-line arguments.

Resource identifiers:

- MIG UUID: **MIG-d2fcfd79-720e-5218-82c9-0a132be68920**; H1003g.40gb.
- Parent UUID: **GPU-5a568e3c-e96c-169f-b063-f4478340ff51**, PCI0000:1b:00.0.
- GPU instance2, compute instance0; capability device minors21/22.
- Container-visible PID namespace inode4026559509. Observed Kubernetes pod UID
  **c5f7f8ec-4e1f-4b24-8ab5-a48350b23dc2**; the diagnostic JSON contains matching
  cgroup fingerprints. Identify the corresponding runtime container/job on the
  host rather than trusting an arbitrary same-UID process.

Please provide the following from the GPU host or an authoritative runtime view:

1. **Allocation scope/current work:** the account/project/container/job assigned
   this exact MIG, any existing reservations or restrictions, and the identity
   of all compute workers on GI2/CI0 during the captured window. Confirm whether
   exclusive compute use for HiTSKT can be established without interfering with
   current unrelated research work. Device exposure/free VRAM alone is insufficient.
2. **Historical mapping, if logs exist:**20 October2026,17:15:52–17:15:56UTC:
   NVML PID67123 versus HiTSKT local PID2847964. Supply host audit/runtime records
   linking or separating these processes using PID/Tgid, NSpid/NStgid chain,
   process start-time ticks plus boot identity, namespace IDs, UID, cgroup,
   container/job identity and assigned GPU/MIG. Both processes are gone locally;
   a fresh process with the same integer PID cannot prove the old relationship.
3. **Current captured mismatch:**21 October2026,06:59:10UTC:
   NVML PID2265424 on GI2/CI0, visible local device-holder PID3136171,
   diagnostic PID3136614. Host-side selected `/proc/<pid>/status`,
   `/proc/<pid>/stat` start-time, `/proc/<pid>/cgroup`, and namespace link/inode
   metadata should establish whether the first two IDs map to one task or
   separate workloads. For new/current PIDs use fresh synchronized timestamps.
4. **Source PID domain:** identify the PID namespace/domain used by the NVML
   compute-process API versus the container's procfs/os.getpid view. If an MPS,
   proxy, tracing/runtime wrapper or other backend reports a server PID instead
   of the client, provide its authoritative client↔server and namespace/job
   mapping, including other clients; matching executable names is insufficient.
5. **A sustainable read-only bridge:** provide a trusted host/runtime identity
   endpoint or narrowly scoped read-only host-proc evidence accessible to the
   guard, with freshness/start-time/boot/namespace/container bindings and access
   to all competing compute identities. A writable unverified JSON alias or
   blanket same-account PID exemption is not adequate.

If historical logs are unavailable, say so explicitly. The old67123↔2847964
relationship must remain UNKNOWN; a correct future attribution path can still
be established independently. No CUDA/training will be launched during the
current diagnostic task to obtain this evidence. LightSAKT2 and all other jobs
remain untouched, and execution waits for the user's next instruction.

# Administrator confirmation needed before HiTSKT GPU use

20 October2026: the user confirms ownership of both research projects but has
no exclusive-allocation evidence. Preserve LightSAKT2 work; do not stop it solely
to release GPU resources before HiTSKT assignment is confirmed. No CUDA/training
or allocation change is authorized without verified allocation/authorization.
No fresh process/GPU observation was performed for this clarification.

A written confirmation from the responsible server administrator is sufficient
if no scheduler is used. Otherwise retain the scheduler reservation/allocation
record as well. Neither free memory, common account ownership nor container
visibility establishes exclusivity. Do not manufacture a reservation record.

## Copy-ready request (not sent)

Please confirm an exclusive GPU/MIG allocation for my HiTSKT FAVOR+ research.
I need the following details before running any CUDA work:

1. Server/node identity and research account/project receiving the allocation.
2. Exact allocated device UUID and profile/capacity. The previously visible device
   was H100 MIG3g.40gb `MIG-d2fcfd79-720e-5218-82c9-0a132be68920`, on parent
   `GPU-5a568e3c-e96c-169f-b063-f4478340ff51`. Please confirm this partition or
   provide the UUID of another actually assigned resource.
3. Allocation/reservation/job or administrator ticket reference, with start and
   expiry times in UTC. Please allow a two-hour initial window for setup, bounded
   verification and the pilot; this is a reservation request, not a measured runtime.
4. Explicit confirmation that this partition is exclusively assigned to my
   workload during that window, has no conflicting current/queued reservations,
   and unrelated workloads have been cleared by their owners/administrator.
   Independent MIG partitions may remain assigned to other users.
5. Explicit permission to run HiTSKT CUDA forward/backward verification and a
   maximum five-update ASSIST2017 training-split pilot. No full training is requested.
   If LightSAKT2 must yield, confirm the assignment first; its artifacts must remain
   intact. I do not authorize terminating anyone else's jobs.
6. The approved account/container and how the assigned UUID is exposed to it,
   along with applicable VRAM, host-RAM, CPU, time/preemption and storage limits.
   Please ensure no mechanism will launch competing jobs on this allocation.

Resource planning: existing H1003g.40gb is a candidate, subject to measurements.
Prepared pilot has an8GiB PyTorch allocator quota plus CUDA/runtime overhead,
requires at least10GiB exposed capacity, and uses a4GiB process-RSS gate with
at least2GiB shared-container headroom.16GiB+ exclusive VRAM and16GiB host-RAM
allowance are planning requests, not demonstrated training minima. No measured
FAVOR+ CUDA memory/runtime or full-batch feasibility is available yet.

## After confirmation

Retain the nonsecret administrative reply/ticket or scheduler record. Check that
its device/account/window/no-conflict terms match actual access before initializing
CUDA. This same historically protected MIG requires a narrowly reviewed safety-
guard update and a new source-bound pilot plan after verified reassignment;
documentation or metadata flags alone must not bypass the guard. CUDA correctness
must pass before the five-update pilot. Larger/full training needs separate approval.

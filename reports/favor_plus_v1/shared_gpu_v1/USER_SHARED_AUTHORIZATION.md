# Shared-mode permission and current boundary —23October2026

The user explicitly clarifies that the H100 is intentionally shared among
researchers, exclusive ownership cannot be assumed or required, and host PID
mapping is unavailable. Existing permission to run HiTSKT on the research GPU
persists. User authorizes a separately configured shared policy and CPU-first
validation/preparation of exact checkpoint20→50 (30 additional updates only).
Resource safety, visibility/uncertainty logging and noninterference remain
requirements. Current task asks for the readiness/coordination report BEFORE
GPU execution. No CUDA/training is executed in this preparation task.

This supersedes the historical exclusive-allocation prerequisite for NEW shared
mode only. It does not establish any historical PID alias, promise exclusivity,
waive explicit cluster restrictions or authorize full training. No administrator
lease or permission to terminate jobs is claimed. The declared1GiB allocator /
2GiB advisory total budget and execution window must be acknowledged in the
future shared execution record; that acknowledgment documents coordination,
not a new exclusive allocation or host mapping requirement. The checked-in
template leaves execution/budget confirmation false until actually accepted.

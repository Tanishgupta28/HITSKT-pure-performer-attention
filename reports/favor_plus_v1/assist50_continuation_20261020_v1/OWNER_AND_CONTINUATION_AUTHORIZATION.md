# Allocation authority and actual continuation approval

The user explicitly confirmed earlier in this conversation that the H100
GPU/server is allocated to them/their research project and authorized transfer
of the former LightSAKT2 MIG to HiTSKT. That allocation-owner statement remains
the authority; no administrator/scheduler evidence is invented. The precise
historical device is MIG-d2fcfd79-720e-5218-82c9-0a132be68920. Free memory alone
cannot establish permission. No unrelated reservation/restriction may be overridden.

Latest actual user approval: “I approve the bounded HiTSKT + FAVOR+ continuation
from checkpoint20 to a maximum of50 total optimizer updates”. The user permits
exactly30 additional updates, conditional on independent explicitly authorized
GPU/MIG availability, preserving weights/Adam/RNG/configuration; loss, gradients,
approximation, GPU memory and throughput monitoring; checkpoint verification at
30/40/50; unchanged0.20 threshold; no LightSAKT2/unrelated job interference;
research-only commit/push; STOP after50 for another approval. No full training.

This approves the preserved proposal SHA
a077ef4ef5a76e240ae06c125fb4e4de93aa092cee5559b2de2b74297d433bd9:
128 fixed features/seed42, original Adam/nonattention/data/split protocol,
B1/cost2048,8GiB allocator/4GiB RSS/2GiB shared headroom,120s/update/45min campaign;
checkpoints25/30/35/40/45/50 and frozen validation/diagnostics20/30/40/50.
The51st sample is only a no-step recovery probe, never an update. No test split,
additional seed, HPO, automatic failed-step restart or historical artifact change.

Allocation metadata records are operator execution ceilings, not scheduler leases
or new grants. A fresh exact-device/exposure/hold/process/NVML/RAM preflight is
required immediately before each CUDA process. Unknown/compute/inaccessible
holders fail closed; only positively verified read-only monitors may be admitted.
Kubernetes403 alone is a visibility limitation, not a restriction or evidence of
permission. No watcher, allocation change or unrelated process signal is allowed.

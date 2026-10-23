# GPU execution and PID-interface comparison —23October2026

Base c15e28dbb2759976611d0148da3ba5f7d2ecba1c. User renews GPU continuation
direction and explicitly confirms GPU work is permitted. No CPU-only restriction
was imposed on this attempt. Existing owner allocation/30-extra/50TOTAL scope,
original checkpoint20/science/.20 REVIEW and all resource checks retained.

**GPU execution occurred. Model recovery PASS; new training updates0.**

The assigned H1003g.40gb passed fresh empty-NVML/no unknown descriptor owner/
no Light trainer/adequate RAM and39.34GiB free admission. Fresh run
`experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261023_v2/`
restored original20 on CUDA: evaldiff0, model/Adam/RNG/projections exact, next
train-mode logits/loss/all-gradient proof EXACT, zero Adam calls. Parent SHA
0388e984b8bed21210455be561524a041c892b98b2a87b50fa3ecb1c3ade86f4.

At04:46:57UTC the unchanged guard rejected NVML1094328 versus own3909540,
before diagnostics20/update21. Live in-process CPU capture before exit confirms
local PID namespace view/NSpid3909540/TracerPid0, NVIDIA descriptors open,
and NVML1094328 not visible. GPU memory for the reported PID578813952bytes;
this is an observation, not a proven mapping/peak. Recovery/stop3.461925s,
not training throughput. No new checkpoint or validation metric. Failed run
quarantined, not restarted. No failure handler CUDA call or identity exemption.

New evidence: `LIVE_FAILURE_IDENTITY.json` captures the actual CUDA process,
not a later CPU process. Loaded libraries include driver NVML535.161.08,
CUDA forward-compat libcuda555.42.02 and libcudart12.5.39. These versions alone
do not establish a defect; no library/runtime setting was changed.

After the user's explicit GPU-work clarification, ran one separate bounded
**zero-optimizer CUDA metadata diagnostic**, not another training retry:
fresh admission, same exact MIG,8GiB allocator ceiling, one4-byte tensor, no
dataset/model/checkpoint/forward/backward/optimizer, then native NVML v1/v2/v3
query of that single live context. No GPU calls after collecting the mismatch.

|Interface|Reported PID|Own PID|API result|
|---|---:|---:|---|
|NVML unversioned/v1|1101056|3910470|success,1 worker|
|NVML v2|1101056|3910470|success,1 worker|
|NVML v3|1101056|3910470|success,1 worker|

Each reports260046848bytes for its worker. No version returns the local PID.
Thus switching from v2 to v3 alone does **not** correct attribution. This
isolated CUDA-only observation strengthens the PID-domain/runtime hypothesis
over a model/optimizer problem; it still does not provide the missing trusted
host binding. No alias, size/timing heuristic or source-bound guard change.
`NVML_INTERFACE_COMPARISON.json` and `nvml_comparison.log` preserve raw results.

One final read-only release capture after diagnostic exit confirms empty MIG
compute list/no unknown holder/no Light trainer; hold hash unchanged. No other
job signaled or changed. No full training/test evaluation/threshold changes.

Total training remains20, updates30/40/50 NOT REACHED. Action.2131 REVIEW remains
historical, not a newly measured learned trajectory. Earlier CPU tests inherited;
3 selected existing guard cases passed again. No redundant229-suite or synthetic
training was repeated. Guard/control/science
files remain unchanged. Captured scripts/evidence are separate from old artifacts.

Stop further GPU attempts: current obstacle is runtime PID attribution after
successful CUDA initialization, not lack of GPU permission or observed idle
VRAM. Existing administrator procedure remains applicable. No repeated GPU grant
is needed. Correctly identifying the live driver PID is still required to admit
training under the user's unchanged safety conditions.

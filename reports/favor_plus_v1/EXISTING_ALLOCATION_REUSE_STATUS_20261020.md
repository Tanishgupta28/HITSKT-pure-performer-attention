# Existing GPU allocation reuse: bounded recheck

Observed20 October2026,09:24:34 UTC; branch `research/hitskt-favor-plus-v1`,
base `fec9ccf3f16617e7fa2413c12134351d181a8e32`.
[Single read-only inspection](existing_allocation_reuse_20261020_v1/inspection.json).

The latest user explicitly authorizes stopping verified owned LightSAKT2,
preventing future launches and prioritizing HiTSKT. An existing reservation may
be reused if it permits that transfer with no conflicts; no new allocation is
required merely because the model/project changes. User authorization does not
establish external reservation ownership. No full training is authorized.

| Check | Verified outcome / limit |
| --- | --- |
| LightSAKT2 shutdown | **Already stopped**; no live trainer/controller/recovery process found, zero signals needed/sent. Interactive sessions untouched. |
| Schedule and saved progress |813 completed,142 pending,2 interrupted,0 running. GKT best/latest epoch2 and other worker waiting-admission interruption unchanged. |
| Research preservation |All7915 previously captured paths exist with unchanged sizes;8 selected manifest/config/status/history/log/checkpoint SHA256 checks and separate epoch2 recovery archive hash pass. Prior full7915-file hash/CPU recovery verification retained, not redundantly repeated. No research artifact modified/deleted. |
| Automatic launches |Existing persistent `USER_TRAINING_HOLD.json` remains active; known local entrypoints/admission/recovery are gated. No watcher, submissions, restart or allocation change initiated. External schedulers are not observable. |
| Visible GPU processes |**None held NVIDIA device handles in the single snapshot**; includes zero LightSAKT2 GPU handles. No new stop occurred, so no measured before/after release delta is claimed. |
| MIG capacity/use |H1003g.40gb `MIG-d2fcfd79-720e-5218-82c9-0a132be68920`:42,278,584,320 total bytes;39,780,352 used (about37.94MiB);42,238,803,968 free. This supersedes earlier occupied-device observations only at the new snapshot time. |
| Existing allocation reuse permission |**NOT VERIFIED**: no scheduler job/account/reservation variables or tools, no mounted allocation metadata at inspected standard paths; attempted Kubernetes read403. No authority establishing owner/project/transfer window/no reservation conflicts. |
| CUDA verification / five-step ASSIST pilot |**NOT RUN / NOT RUN**; no CUDA import/context, forwards, optimizer updates, test evaluation or metrics generated. Only read-only NVML management queries. |
| Full/larger training |**NOT AUTHORIZED / readiness NOT ESTABLISHED**. High-norm approximation and learned FAVOR+ Q/K behavior, CUDA correctness and actual training memory/runtime remain unresolved. |

Host cgroup limit90GiB, usage19,848,044,544 bytes at inspection. No repeated
229-test suite, numerical experiments, full preservation audit or availability
polling. Original HiTSKT ELU+1/FAVOR+ models, datasets, historical artifacts and
unfinished DKT/EdNet remain unchanged.

## Specific missing confirmation

The responsible administrator/scheduler must identify **the existing allocation**
(server, account/project, device UUID, reservation/job or administrative reference,
valid UTC window) and confirm that it authorizes HiTSKT CUDA verification and the
five-update pilot, including transfer from LightSAKT2 if needed, with no conflicting
current/queued reservations. This may be an existing reservation record or written
confirmation of its terms; it need not be a new booking. No visible GPU users and
free VRAM do not establish those terms. Do not infer project membership from a
shared root account or bypass the resource guard.

Stop GPU work here pending that confirmation. Once established, retain the already
authorized staged128-feature CUDA then <=5 ASSIST train-only steps (seed42/B1/
cost2048, no test evaluation), respecting existing numerical/resource gates.
For this historically protected MIG, first review/test a narrow guard update
reflecting verified transfer and prepare a fresh source-bound plan. No such
update, fabricated lease or CUDA run was made in this task. Larger training still
requires the user's next explicit approval.

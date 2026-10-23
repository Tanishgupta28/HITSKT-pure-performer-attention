# Current MIG admission assessment — 23 October 2026

Continuation base: bd076beeb4b198c7e51363793aa5e6056276ca01.

The user reports that Parth acknowledged concurrent GPU work and said he would
stop. This is not an independently verified identity for any historical PID.
The original owner-confirmed H100 research allocation/LightSAKT2-to-HiTSKT
transfer remains the permission source. The latest instruction renews the
conditional, exact checkpoint20→50 continuation, at most30 additional Adam
calls, and permits no unrelated process or allocation changes.

Current evidence differs from the21October blocked snapshot:

- At00:52:35UTC, assigned MIG-d2fcfd79-720e-5218-82c9-0a132be68920
  (parent GPU-5a568e3c-e96c-169f-b063-f4478340ff51, GI2/CI0) reports
  zero NVML compute workers and39,780,352 used /42,238,803,968 free bytes.
  No visible GPU descriptor holder, inaccessible descriptor owner or
  LightSAKT2 trainer was found. The restart hold retains its original hash.
- At00:53:44UTC, native MIG-specific NVML compute, graphics and MPS queries
  all return success/count0. Parent process queries return status4 (permission
  denied); these are not empty-list evidence. Utilization rate is unsupported
  (status3), so no utilization percentage is asserted. Full-host visibility
  remains unavailable; the assigned MIG queries succeed independently.
- Container limit96,636,764,160bytes (90GiB), observed use62,844,764,160bytes;
  adequate headroom for the existing4GiB process/2GiB headroom limits. The
  inherited OOM counter remains2. The predeclared GPU allocator ceiling is8GiB,
  with at least10GiB initially free and2GiB throughout.

These observations support no currently observed conflicting workload on the
assigned MIG. Combined with the existing owner's explicit allocation/transfer
authority, they support a fresh admission through the UNCHANGED production
guard; they are not an administrator lease, future reservation or proof that
Parth's particular job stopped. No known conflicting reservation or explicit
restriction has been supplied. Each execution/recovery stage must recheck
current conditions before CUDA, and before/after every optimizer update.

The historical NVML67123/container2847964 mapping remains UNKNOWN. No current
process needs an alias exemption because the pre-initialization MIG list is
empty. A later differing NVML PID must stop the run immediately under the
existing exact-self rule; do not infer an alias from Parth's statement or from
the process disappearing after exit. No guard change or diagnostic CUDA probe
is authorized by this assessment. The requested continuation itself is the only
permitted compute, conditional on successful admission and exact parent recovery.

Use a fresh versioned run and preserve both failed20October directories.
Restore exact parent checkpoint SHA256
0388e984b8bed21210455be561524a041c892b98b2a87b50fa3ecb1c3ade86f4,
including Adam/RNG/sampler/projection states. Retain all source-bound scientific
configuration, original frozen validation sample,0.20 REVIEW threshold and
severe stops. Stages30/40/50 each require fresh zero-Adam recovery before the
next stage; atomic checkpoints also occur25/35/45. No test evaluation, extra
seed, feature-count change, full training, watcher, automatic conflict retry,
LightSAKT2 change or unrelated signal. Stop at50 or the first failed gate.

# Current execution scope — 24 October 2026

User explicitly requests CUDA correctness, controlled four-arm attention
comparisons and shared-GPU safety checks. Existing shared H100 allocation and
approved8GiB allocator/10GiB advisory/16GiB admission limits persist.
No exclusive ownership requirement or administrator reauthorization is added.

**ZERO additional optimizer updates are authorized now.** The user requires
measured/estimated memory, runtime and proposed step limits, then a NEW explicit
approval before learning. Historical full-run/pilot approval is not reused as
permission for this new comparison. No full training/test evaluation/other-job
interference. Current CUDA verification does forward/backward only, checks
parameter immutability and never invokes Adam.step.

Fresh models share learned initialization/data/seed/optimizer settings. Common
nested512 sampling makes feature comparisons paired; this projection-policy
exception is declared separately and does not modify the historical model.

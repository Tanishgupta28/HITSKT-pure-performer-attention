# Approved resource budget — 2026-10-24

Actual user response: **“Approve profile and full-run budget”**.

This accepts the previously presented 8 GiB PyTorch allocator/reservation cap,
10 GiB advisory total budget, 16 GiB admission-free floor, original maximum
batch 64 and padded-token cost 32768 for a zero-update profile and, only after
PASS, the already authorized complete ASSIST2017 run of at most 10 epochs with
full validation each epoch and early-stopping patience 3.

Runtime free floor 6 GiB, host RSS 4 GiB/headroom 2 GiB, 0.1-second pacing,
30-second step bound and 45-minute campaign ceiling remain unchanged.
Keep shared-device visibility/pressure/LightSAKT2 protections and numerical
stops. No other workloads may be interrupted. No final-test evaluation,
feature tuning, silent architecture/data changes, automatic retry or additional
resource-budget increase is authorized. Preserve diagnostic checkpoint 50.

The 23 October plan, source files, proposal, prior current-limit refusal and
all original research artifacts remain unchanged; this approval supersedes
only the pending-resource-approval state, not its historical evidence.

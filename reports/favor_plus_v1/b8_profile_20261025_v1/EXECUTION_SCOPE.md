# Explicit user-approved B8 profile — 25 October 2026

The latest user message explicitly approves a maximum10-minute, zero-optimizer-
update GPU memory/runtime profile, preserving the8GiB cap/sharedGPU safeguards.
Use the already prepared commonB8 plan, original complete session/history data,
fresh42 identical learned initializations and identicalAdam settings (no steps).
The user explicitly withholds the two-hour training run pending separateapproval.

All stages below share ONE600second scope: bothnine-case full-shape profiles,
then32 fixedtrain strata×2 repeats and8validation strata×2 repeats perarm, one
warmup/oneinitial diagnosticpass/two untrained IO proxies perarm. Fixeddesign
hashes bind the extra instrumentation. No architecture/projection/split change.
Adam-sizedzero buffers reserve memory; noAdam.step or model parameter update.
IO proxies are marked invalid for training/checkpointload and retained separately.

Use existing authorized intentionallyshared H100MIG, two freshguard admissions,
8GiB allocation/reservation cap,10GiB advisorytotal,16GiB admissionfree,6GiB
runtimefree,4GiB ownhostRSS/2GiB sharedheadroom,.1spacing/30sstep. UnknownPID
attribution/utilization remainunknown; noexclusiveownership claim. No otherjob
signals, allocationchanges, runtimepatch, watcher, retries, automaticresize or
clockreset. Firstfailurestopsallstages. No finaltest inference or training.

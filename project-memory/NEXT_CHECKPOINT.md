# Next checkpoint

Updated: 2026-09-09 UTC

## Last valid terminal checkpoint

Upstream `origin/main` is restored, durable memory is bootstrapped, the initial
assessment is complete, and checkpoint `88aef9f` is on private repository
`Tanishgupta28/capstone-gpu`. No scientific run is active or complete.

## Exact next bounded action

Implement and validate the shared post-filter session-window/rolling-history
representation across all three datasets. Prove that every target sees only
strictly earlier events and that PAD/EOS never becomes a label. Then begin the
five model forward/backward/checkpoint-loading smoke-test matrix.

## Stop conditions

- Do not start the 15 experiments before the smoke-test gates pass.
- Do not substitute the supplied Riiid file or reduced `ednetnew.csv`.
- Do not silently sample EdNet or alter HiTSKT's Performer architecture.

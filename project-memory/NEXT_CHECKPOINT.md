# Next checkpoint

Updated: 2026-09-09 UTC

## Last valid terminal checkpoint

Upstream `origin/main` is restored, durable memory is bootstrapped, the initial
assessment is complete, and checkpoint `88aef9f` is on private repository
`Tanishgupta28/capstone-gpu`. No scientific run is active or complete.

## Exact next bounded action

Obtain the user's ASSIST2017 question/skill mapping decision for the 697
multi-skill questions. Then complete ASSIST2017 and full-Junyi preprocessing,
validate the shared post-filter session-window/rolling-history representation,
and only afterward begin model-forward/backward/checkpoint smoke tests.

## Stop conditions

- Do not start the 15 experiments before the smoke-test gates pass.
- Do not substitute the supplied Riiid file or reduced `ednetnew.csv`.
- Do not silently sample EdNet or alter HiTSKT's Performer architecture.

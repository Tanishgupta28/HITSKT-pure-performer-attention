# RKT Phi construction scaling gate

This gate uses the exact approved **RKT paper-faithful performance-only
Phi-relation variant with 5-fold student-level cross-fitting**, rolling history
49, and project seed 42. Sampling changes only how many complete students are
included in a benchmark cache; it does not change latest-prior, fold exclusion,
training-only, or undefined-to-zero semantics.

| Dataset | Students measured | Training targets | Pair contributions | Unique directed pairs | Phi time | Peak scratch | Peak RSS | Finished cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ASSIST2017 | 1,388 / 1,388 | 647,283 | 13,530,869 | 1,569,938 | 14.17 s | 72.35 MiB | 1.01 GiB | 34.90 MiB |
| Junyi | 29,865 / 29,865 | 9,647,042 | 410,926,464 | 78,744,364 | 734.99 s | 5.16 GiB | 18.01 GiB | 1.71 GiB |
| EdNet-KT1 | 116,548 / 116,548 | 53,990,022 | 2,430,771,986 | 119,228,083 | 1,984.27 s | 32.07 GiB | 42.06 GiB | 4.05 GiB |

ASSIST2017 and Junyi are completed full-population caches. Before its full run,
Junyi used exactly 200 deterministically selected students from each of the
five persisted folds; that sample measured 332,247 targets and projected 24.4
minutes, while the full construction actually took 12.25 minutes. EdNet used
the same deterministic 1,000-student sampling rule before its full run. Sample
target counts were measured, not inferred from student count alone.

The measured full Junyi build generated 410.9 million contributions in 12.25
minutes with 5.16 GiB scratch. EdNet generated 2.43 billion contributions in
33.07 minutes with 32.07 GiB scratch and a 4.05 GiB finished cache. Its sample
had conservatively projected 2.85 hours and 32.81 GiB scratch. This gap confirms
that sampled runtime and finished-size extrapolations are planning aids, not
measurements: unique-pair growth is sublinear and sorting/merging need not scale
linearly. The JSON companion retains both prelaunch estimates and final values.

No validation/test interaction enters any cache. No target is discarded or
duplicated, and no cross-fitting or history rule changed for this benchmark.

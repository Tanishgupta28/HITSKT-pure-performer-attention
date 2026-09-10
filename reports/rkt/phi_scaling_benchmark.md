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
| EdNet-KT1 | 1,000 / 116,548 | 445,519 | 20,031,493 | 12,994,154 | 84.73 s | 277.27 MiB | 3.38 GiB | 212.28 MiB |

ASSIST2017 and Junyi are completed full-population caches. Before its full run,
Junyi used exactly 200 deterministically selected students from each of the
five persisted folds; that sample measured 332,247 targets and projected 24.4
minutes, while the full construction actually took 12.25 minutes. EdNet uses
the same deterministic 1,000-student sampling rule. Sample target counts are
measured, not inferred from student count alone.

The measured full Junyi build generated 410.9 million contributions in 12.25
minutes with 5.16 GiB scratch. A simple target-count projection gives roughly
2.43 billion contributions, 2.85 hours, and 32.81 GiB scratch for full EdNet.
The EdNet values are planning estimates only. Unique-pair growth is sublinear
and bounded by the squared question vocabulary, while sorting and merging may
scale nonlinearly; finished-cache projections are especially uncertain. The
JSON companion keeps all raw byte values and explicitly separates measurements
from estimates.

No validation/test interaction enters any cache. No target is discarded or
duplicated, and no cross-fitting or history rule changed for this benchmark.

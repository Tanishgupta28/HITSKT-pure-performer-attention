# HiTSKT FAVOR+ approximation update —24 October 2026

The eight-epoch ASSIST2017 validation results remain intact: best AUC0.703467
at epoch7 (F10.508814), maximum F10.532809 at epoch6. No test data was evaluated.

The new frozen TRAIN-only investigation found no original arithmetic/mask/
projection bug in the tested regimes. Production outputs match the independent
feature kernel within3.12e-6, while high error against exact softmax persists.

CorrectPaddingEncoder shows strong projection-specific learned alignment: its
saved128-feature error rises from0.142 initially to0.566 at epoch8, whereas fresh
independent128 draws on the same frozen inputs give0.114→0.183. Features become
concentrated even though exact attention remains fairly diffuse on average.
Value-vector geometry also amplifies the reported relative-output errors.
These support co-adaptation/finite-feature explanations, not a proven causal
optimizer mechanism or a guarantee that fresh projections improve prediction.

More features and antithetic positive features partly help. A separate512-total-
feature prototype is implemented with a distinct checkpoint format;125 distinct
CPU checks passed (47new). It still fails0.20 reviews in some stages/long cases;
it has no trained AUC, validation or GPU result. The original model was not changed.

Recommend a separately approved, small matched learning pilot with fixed audit
inputs/projection draws before any further full training. Retain the existing
results and0.20 threshold; do not silently replace projections, resume with a
different feature map, or treat these diagnostics as final-test superiority.

Detailed report: `reports/favor_plus_v1/POSTEPOCH8_APPROXIMATION_INVESTIGATION.md`.

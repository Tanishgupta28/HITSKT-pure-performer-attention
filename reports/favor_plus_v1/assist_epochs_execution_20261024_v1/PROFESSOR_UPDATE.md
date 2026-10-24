# HiTSKT FAVOR+ — professor update, 24 October 2026

Completed **eight full ASSIST2017 training epochs**, with complete validation
after every epoch:566107training and125800validation labels per epoch,seed42,
128 fixed genuine FAVOR+ features, original nonattention architecture/data/splits.

- Best validation AUC: **0.703467 at epoch7**; accuracy0.685254, precision0.620946,
  recall0.430985,F10.508814,BCE0.599038.
- Highest observed F1: **0.532809 at epoch6**; checkpoint selection used AUC.
- All1454 optimizer updates had finite gradients/model/Adam. Eight fresh checkpoint
  recoveries passed; selected-checkpoint full-validation replay was bit-identical.
  Independent CPU metric differences were below6.9e-9.
- Run ended before epoch9 to respect the approved45min campaign limit;
  measured execution including final replay:41min38s. No test set was evaluated.
- Approximation remains **REVIEW**: epoch8 Action0.262562,CPE0.503431,Decoder0.562503
  exceed0.20; no declared severe failure or implementation-kernel discrepancy.
- Shared H100 safeguards passed; peak3.19GiB allocated/7.99GiB reserved,1.84GiB
  hostRSS. Original checkpoint50/LightSAKT2/other workloads remain preserved.

These are genuine complete-validation results under chronological within-student
retrospective teacher forcing, with the prior response-timing qualification.
They are not final-test results, proof of convergence or model superiority.
Recommend reviewing learned feature concentration before longer/multi-dataset
training; retain all measured results and the unchanged0.20 review threshold.

Detailed report: `reports/favor_plus_v1/ASSIST2017_EPOCH_TRAINING_RESULTS.md`.

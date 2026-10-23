# HiTSKT FAVOR+ — research update

The genuine FAVOR+ implementation has passed independent causal/masking,
gradient and numerical checks. Fifty previous diagnostic optimizer updates
were finite and checkpoint-reproducible; they were not epoch-based training.

A fresh complete ASSIST2017 run is now prepared:7,060 training target sessions/
566,107 labels,full validation on125,800 labels,seed42,fixed128 features,
maximum10 epochs and patience3. Original data,splits and sequence lengths remain
unchanged. ActionEncoder approximation remains under REVIEW; it has not been
reclassified as passing or silently changed.

**No full epoch or full-validation AUC/F1 result is available yet.** The existing
shared-GPU pilot budget(1GiB,batch1,cost2048) refuses the original full-data
batches before CUDA initialization. A larger bounded profiling/training budget
has been requested; no other workload was interrupted. Once that budget is
explicitly approved and memory profiling passes,the already authorized epoch
run can start. No final test data has been evaluated or used for selection.

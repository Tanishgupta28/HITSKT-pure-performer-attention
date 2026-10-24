# Explicit numerical-precision follow-up

All four fresh arms use the already-supported float64 attention accumulation;
learned parameters, embeddings/feedforward computations and returned outputs
remain float32. This is a separate configuration, not a retroactive FP32 PASS.
The originalFP32 failure and executed source snapshots are preserved.
Unchanged5e-5 output/gradient-atol and1e-3 gradient-rtol, exact causal/reload,
0.20 approximation REVIEW, approvedshared8GiB allocator/10GiB advisory/16/6GiB
headroom limits. The45min campaign clock is inherited from the initial attempt.
Zero optimizer updates; new explicit matched-training approval remains pending.

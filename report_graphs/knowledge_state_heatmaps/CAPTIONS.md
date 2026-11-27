# Figure captions

**ASSIST2012.** Post-interaction HiTSKT knowledge-state estimates for the first deterministically selected ASSIST2012 test student. Each column corresponds to an observed event in a contiguous 50-interaction window. Rows show five frequently practiced skills using processed skill IDs. Events with equal first-graded-action timestamps update jointly at the final row of the group. Blues indicate mean predicted correctness on a 0–1 scale.

**Junyi.** HiTSKT knowledge-state estimates for the first deterministically selected Junyi test student across 50 consecutive logged interactions. Rows represent five processed composite skills. Outcomes sharing a logged timestamp enter history together at the final interaction of their group. Each cell gives the mean predicted probability of correctness over a fixed training-derived question bank.

**EdNet-KT1.** HiTSKT knowledge-state estimates for the first deterministically selected EdNet-KT1 test student across 50 consecutive interactions. Rows represent five processed composite skills. Within each bundle or simultaneous-delivery group, early columns retain the pre-group estimate and the final column reflects the outcomes of the completed group. Closure is estimated from KT1 delivery timestamps and elapsed times.

**Cross-dataset comparison.** Rolling HiTSKT knowledge-state estimates for the first deterministically selected test student in ASSIST2012, Junyi, and EdNet-KT1. Each panel shows five skills across 50 consecutive interactions using the same 0–1 color scale. Each cell averages trained-model correctness probabilities over a fixed skill-specific question bank. The panels use the observed timing and original session protocol of their respective datasets.

## Paper methodology

For each selected skill $k$, we defined a fixed bank $Q_k$ of up to five questions ranked by training frequency. At each chronological cutoff $t$, we independently appended each question $q \in Q_k$ to the visible current-session prefix and evaluated the dataset's best HiTSKT checkpoint. The model received up to 15 preceding complete sessions and shifted correctness from observed responses. The skill-level estimate was the arithmetic mean $S(u,k,t)=|Q_k|^{-1}\sum_{q\in Q_k} \sigma(f_\theta(q,k,H_{u,t}))$, where $H_{u,t}$ is the observed context at that cutoff and $f_\theta$ is the trained model's logit. Outcomes tied in time or associated with the same EdNet bundle entered context only upon group completion. Heatmaps show these aggregated correctness probabilities as proxies for skill knowledge.

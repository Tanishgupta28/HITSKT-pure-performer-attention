# HiTSKT knowledge state prediction heatmaps

Each panel follows one held-out student through a consecutive part of their test sequence. A color cell shows the mean probability the trained HiTSKT model assigns to correct responses on representative questions for that skill. Light blue indicates lower probability and dark blue indicates higher probability on the same 0–1 scale in every panel. These skill-level estimates are proxies for knowledge. They measure model-predicted correctness on a particular question bank, so they differ from directly observed mastery, independently calibrated mastery probabilities, and attention weights.

## Checkpoints and data

The generator reads the HiTSKT rows of [final_results.csv](../../reports/final_results.csv) and verifies the linked full experiment, validation history, checkpoint metadata, model configuration, parameter count, and SHA-256. The ROC-AUC column below reports the existing benchmark test result.

| Dataset | Checkpoint | Best epoch | Completed epochs | Test ROC-AUC |
|---|---|---:|---:|---:|
| ASSIST2012 | [ASSIST2012 experiment](../../experiments/hitskt/assist2012/full) | 23 | 28 | 0.7329927471 |
| Junyi | [Junyi experiment](../../experiments/hitskt/junyi/full) | 32 | 37 | 0.7978262055 |
| EdNet-KT1 | [EdNet-KT1 experiment](../../experiments/hitskt/ednet_kt1/full) | 30 | 35 | 0.7693648364 |

[Reproducibility manifest](reproducibility_manifest.json) records full checkpoint hashes, model and experiment configurations, software versions, source hashes, processed-store fingerprints, and output hashes. For ASSIST2012 and EdNet-KT1, the checkpoint and store metadata hashes match the prior independent benchmark replays. For Junyi and EdNet-KT1, every session-store file is also checked against the historical audit. The ASSIST2012 event Parquet hash is checked against its session-store metadata. The prior [Junyi diagnostics](../../reports/reproducibility_diagnostics/junyi_diagnostic_manifest.json) document a small replay ROC-AUC difference of about 2.68 × 10⁻⁷ and threshold-sensitive classification differences. This figure run checks checkpoint identity and individual inference parity on CPU. It uses the recorded benchmark results for the full-cohort metrics.

The generator uses the benchmark implementation in [hitskt.py](../../ktbench/models/hitskt.py), [performer.py](../../ktbench/models/performer.py) and [batching.py](../../ktbench/batching.py). It retains the question, skill, and correctness embeddings, sinusoidal positions, Action, Session, Correct/Padding, and Decoder stages, causal ELU+1 linear attention, layer normalization, residual connections, feed-forward layers, and linear prediction head. The sigmoid converts the trained head's logits to probabilities. The model runs in evaluation mode with dropout disabled and under `torch.inference_mode()`. We use the fixed-history caching approach from the [Junyi weak-topic ranking replay](../Junyi_original_ranking/replay_scores.py) and recompute current-session memory for each new visible prefix.

## Students, interactions, and skill rows

Students enter the candidate pool through the original test-session cohort. Their order is determined by SHA-256 of `42:<processed student ID>`, ordered lexicographically by digest bytes. Each student contributes the earliest qualifying 50-row consecutive test window with at least four preceding complete sessions, both correctness outcomes, and five training-backed skills with at least two attempts each in the window. Window endpoints respect availability groups. For EdNet-KT1, the window and its retained history must pass the bundle availability checks below. The final order assigns anonymous labels `student_01` to `student_05` independently within each dataset.

Skill rows are ordered by descending number of attempts in the plotted segment, breaking ties by ascending processed skill ID. Each row stays fixed at every column. EdNet's `<UNTAGGED>` bucket can occur in the observed history but is ineligible as a skill row. The generator first considers the preferred criteria across all candidates. If necessary, it allows one preceding session, then considers lengths 50, 49 down to 40, and 51 up to 60 with three to five repeated skills. It reports the tier used for each student. Selection uses observed events and training counts and takes place before loading the model for inference.

For each skill $k$, the bank $Q_k$ contains up to five distinct questions from the original training split. Questions are sorted by decreasing observed $(k,q)$ training frequency and increasing processed question ID on ties. The bank uses the full original training split, including context-only first sessions where applicable. The learned model's training uses its own original target selection. Validation and test interactions contribute no bank counts.

## Rolling inference and causality

For student $u$, let $H_{u,t}$ be the history available at column $t$. For each $q \in Q_k$, the decoder estimates $p_\theta(u,q,k,t)=\sigma(f_\theta(q,k,H_{u,t}))$, where $f_\theta$ is the trained HiTSKT logit. The plotted matrix entry is

```math
M_{k,t} = S(u,k,t) = \frac{1}{|Q_k|} \sum_{q \in Q_k} p_\theta(u,q,k,t).
```

Each probe appends one hypothetical question and skill to a copy of the observed current-session prefix. The decoder correctness input at that position is the most recent observed response, or BOS at the start of a session. The repository collator shifts the actual responses by one. The optimized probe masks the collator's trailing EOS and clears the probe label and subsequent correctness inputs. Each hypothetical question is scored independently with the same visible history. No hypothetical outcome is added to that history. The Action and Session Encoders receive up to 15 preceding complete original sessions. When the sequence reaches the first row of a new session, the prior session moves into completed history and the current-session prefix resets. At the last row of a session, the probe still represents a hypothetical continuation of that session.

- **ASSIST2012.** The source pipeline anchors the timestamp on the first graded action and constructs sessions using its original 10-hour inactivity rule. A singleton updates the history immediately after that action. Equal-timestamp rows update together at the last row in the group. Earlier rows in the group show the state before any of those responses became available. The processed tie order remains intact.
- **Junyi.** The original preprocessing uses the benchmark's normalized source timestamp and source-row tie order. The same group convention defers all tied responses until the end of the group. The recorded start and end times coincide in this source, so the plot describes logged response groups.
- **EdNet-KT1.** KT1 records delivery time and elapsed time. Its `solving_id` identifies a bundle, while HiTSKT keeps the original inactivity-derived sessions. Within a session, adjacent rows that share a bundle or delivery timestamp are one availability group. Earlier rows show the pre-group state. The final row shows the state after group completion, estimated as the maximum delivery time plus elapsed time across its rows. We require the preceding group's estimated completion to be strictly before the next group's earliest delivery in the processed context used by the window. Candidate histories with noncontiguous recurring bundle IDs or groups that span session boundaries are excluded. These constraints preserve the complete causal prefix under the available timing proxy. KT1 has no exact submission or feedback timestamps.

The reference-style labels `(skill_id, correctness)` annotate observed events in the local figures. For a deferred group, an earlier column may show its row's label while its score still reflects only preceding groups. Repeated colors can therefore reflect unchanged available input. Sudden changes can reflect a session transition, and decreases remain visible. Probabilities also depend on the difficulty of the fixed question bank. Scores across datasets reflect different models, skills, and question banks.

## Figures, matrices, and publication

Each individual CSV has one row per plotted skill and one column per chronological position, with no rounding beyond floating-point serialization. Combined CSVs are indexed by dataset, anonymous figure ID, skill and column. Matplotlib renders the same values with `Blues` on `[0,1]` at 330 DPI with no interpolation. [Captions and paper methodology](CAPTIONS.md) accompany the figures.

The GitHub figures use ordinal x-axis labels. The full figures with the exact observed `(skill_id, correctness)` sequences, session metadata, and per-question probabilities are saved under `.inspection/hitskt_knowledge_state/report_graphs/`. The public analysis includes model predictions, processed skill IDs for the plotted rows, and anonymous within-dataset labels. Source learner IDs, timestamps, observed correctness sequences, session boundaries, and question-level inference records stay local.

Dataset credit for these research visualizations goes to [ASSISTments 2012–13 School Data with Affect](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect), [Junyi Academy's Online Learning Activity Dataset](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy) via the [preprocessed source from Ke et al.](https://github.com/pokerme7777/HiTSKT/tree/main/dataset), and [EdNet from Riiid](https://github.com/riiid/ednet), released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Reproduction

Set up the benchmark environment and install plotting dependencies with `python -m pip install -e '.[plots]'`. Place the best checkpoints in their original experiment directories and prepare the full benchmark stores as described in the [repository README](../../README.md). EdNet also requires the original processed Parquet shards for elapsed time and bundle membership.

If the Junyi store is missing, acquire `Junyi.csv` from the original HiTSKT repository and verify SHA-256 `8a18000806279ae4feae90a94a5af75b7688f2e4ba666c9a0662c75e1a404b6a`. The store is rebuilt by the existing preprocessing and builder.

```bash
python -m ktbench.data.flat junyi data/raw/junyi/Junyi.csv data/processed/junyi/full
PYTHONPATH=. python scripts/build_session_store.py \
  data/processed/junyi/full data/processed/junyi/full/session_store
```

Run from the repository root. The first command writes the local reference-style figures and private metadata. The second shows how to request a single dataset and additionally write a public export.

```bash
python report_graphs/generate_hitskt_knowledge_state_heatmaps.py --dataset all --threads 2
python report_graphs/generate_hitskt_knowledge_state_heatmaps.py --dataset junyi --public-output report_graphs
```

[Verification results](VERIFICATION.md) summarize the inference and rendering checks.

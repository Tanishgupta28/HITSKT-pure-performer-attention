# Junyi dataset and skill mapping

## Verified inputs

| Input | Verified identity |
| --- | --- |
| Accepted raw file | `Junyi.csv`, 14,660,217 interactions, SHA-256 `8a18000806279ae4feae90a94a5af75b7688f2e4ba666c9a0662c75e1a404b6a`. |
| Processed vocabulary | 25,630 model questions, 1,521 composite skills and 1,326 original tags. |
| Authorized history | Source Student 1 maps to processed Student 1. Only nine sessions and 179 interactions before the first excluded test session are loaded. The private review manifest contains the exact cutoff for replay. |
| Verified Lumen bank | 3,978 questions, three per original tag. There are 2,352 Elementary, 1,617 Junior and 9 Senior questions. SHA-256 `579c11905685790bb3e3da086493dceee1c36b0f552e9a9cfcfca3682991f7a0`. |
| Representative ranking | Training exposure from the pinned handoff. Ranking SHA-256 `2aa571e9084d9072465c3cfac6550c9c7d58ae1ffc35266ed2e028bc2642f141`. |
| HiTSKT checkpoint | `experiments/hitskt/junyi/full/best_model.pt`, epoch 32, SHA-256 `2308c5bf0c850da1e55daf802a0853a30fac45eeed071c7162c68fc17bfa3bdd`. The local loopback pilot uses a private copy with this digest. |

The original archive passphrase was unavailable on this workstation. The approved history was recovered from the verified raw source and matched all 11 independent SessionStore member digests and the complete student crosswalk. The separate question bank was verified against the accepted bank digest. [Junyi source provenance](../../docs/data/junyi_provenance.md) · [Runtime setup and recovery](../../docs/platform/junyi.md)

## How the identifiers connect

- A Junyi question maps to one original model question ID and one composite skill ID. A composite records the complete verified set of original tags for that question.
- Each Lumen question carries one original tag. Its analysis pools legal representative questions from composites containing that tag. Representatives are ranked by descending training exposure and then by model question ID. Estimates average up to five distinct representative probabilities.
- A graded Lumen answer records the actual bank item and its grade. A separate `PROXY` mapping records the selected representative IDs and mapping version. The source event's recorded correctness stays with the original history.
- An original Junyi attempt counts once for each original tag in its composite. A Lumen answer counts only for its bank question's tag. The combined observed rate divides the applicable correct outcomes by the applicable attempts. Skills with zero attempts have an empty observed rate.

## Private review outputs

`export_junyi_review.py --through 3` reads the three completed papers and the saved following paper. It verifies their versioned inputs and writes to `reports/adaptive_junyi_student1/private_exports/review-through-paper-03/`. This directory is ignored by Git. The public notes contain only provenance and aggregate verification. [Export and reproduction commands](../../docs/platform/junyi.md)

| File | Contents |
| --- | --- |
| `student1_history.csv` | The approved 179 original events only. |
| `student1_skill_predictions.csv` | The saved original-only and app-inclusive probabilities for each ranked representative and original tag. |
| `student1_skill_analysis.csv` | The 1,326 tag estimates, classes, original and Lumen attempt counts, observed rates and unavailable reasons. |
| `question_bank_verified.csv` | The bank inventory with assigned IDs, tag, topic, stage and verified source hash. The original source file remains separately restricted. |
| `generated_paper.json` | Redacted following-paper metadata, selection counts and frozen-item fingerprint. Question IDs, prompts, options, seed and keys stay in MongoDB. |
| `recommended_resources.json` | The following paper's required skill topics, verified public links and status. |
| `assessment_results.csv`, `updated_student_history.csv` | Sixty genuine grades and a 239-event replay history, joined on paper number and request ID. |
| `review_manifest.json` | Exact cutoffs, input fingerprints, thresholds, score class counts and artifact hashes. |

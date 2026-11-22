# Junyi integration inputs

Start with the [Codex integration prompt](../../junyi_adaptive_integration_prompt.txt) and the repository's platform setup guide. The existing assessment route on the main branch redirects to the practice route. Inspect both entry points before replacing the default assessment workflow.

## Files in GitHub

- Original HiTSKT model, batching, preprocessing, backend, and frontend source.
- [Original Junyi experiment](../../experiments/hitskt/junyi/full/) and [benchmark](../../reports/final_results.csv). The epoch-32 checkpoint SHA-256 is `2308c5bf0c850da1e55daf802a0853a30fac45eeed071c7162c68fc17bfa3bdd`.
- [Published Junyi mappings](../../reports/datasets/junyi/) for questions, original tags and composite skills.
- [Fixed-history validation](../../reports/skill_threshold_experiments/EXPERIMENT_1_RESULTS.md), [threshold selection](../../reports/skill_threshold_experiments/EXPERIMENT_2_RESULTS.md), [frozen thresholds](../../reports/skill_threshold_experiments/optimized_thresholds.json), and the [representative question bank](../../reports/skill_threshold_experiments/representative_questions.csv). The 61% and 86% cutoffs were evaluated on composite-skill means. Their use on aggregated original-tag scores is a prototype policy.

## Inputs to provide privately at runtime

| Input | Location in the research workspace | Verification |
|---|---|---|
| Original Junyi raw CSV | `data/raw/junyi/Junyi.csv` | SHA-256 `8a18000806279ae4feae90a94a5af75b7688f2e4ba666c9a0662c75e1a404b6a` |
| Original processed SessionStore and mapping crosswalks | `data/processed/junyi/full/session_store/` and `data/processed/junyi/full/mappings/` | Compare SessionStore files to the recorded [evaluation manifest](../../reports/independent_correctness_audit/junyi_evaluation_manifest.json) and mappings to the [data manifest](../../reports/independent_correctness_audit/junyi_data_manifest.json). Use the student crosswalk to resolve the prototype account. |
| Custom question bank with answer keys | `reports/datasets/junyi/junyi_hitskt_question_bank.csv` in the current research workspace, or the bank supplied alongside the prompt | The inspected bank has 3,978 questions across 1,326 original tags. SHA-256 `579c11905685790bb3e3da086493dceee1c36b0f552e9a9cfcfca3682991f7a0`. Verify the provided version at integration time. |
| Gemini credential | Server environment setting `GEMINI_API_KEY` | The owner will provide this later. Test the deterministic workflow first and exercise live Gemini after the key is configured. |

The Git LFS pointer at `dataset/Junyi.csv` identifies the full Junyi CSV, but GitHub's public LFS endpoint returned a missing object during verification. Use the verified copy on authorized research storage. The reduced bundle below is sufficient for a runtime-only checkout.

Keep raw and processed learner data, the answer-bearing bank, exported per-learner predictions, and credentials outside Git. Share them only through approved access-controlled storage. Commit code, documentation, and aggregate evidence.

## Minimal runtime handoff

The [encrypted v1 release asset](https://github.com/Tanishgupta28/HITSKT-pure-performer-attention/releases/tag/junyi-adaptive-runtime-v1) contains the exact answer-bearing question bank and Student 1's nine pre-test sessions extracted directly from the original store. Obtain the passphrase file from the project owner through an approved private channel.

1. Save the encrypted asset outside version control. Verify its SHA-256 against the [runtime manifest](junyi-runtime-v1-manifest.json).
2. Set `JUNYI_RUNTIME_PASSPHRASE_FILE` to the private passphrase file. Decrypt from the repository root under the ignored `data/processed/` path.

   ```bash
   umask 077
   set -o pipefail
   install -d -m 700 data/processed/junyi/adaptive_runtime_v1
   gpg --batch --pinentry-mode loopback --no-symkey-cache --passphrase-file "$JUNYI_RUNTIME_PASSPHRASE_FILE" --decrypt junyi-adaptive-runtime-v1.tar.gz.gpg |
     tar -xz -C data/processed/junyi/adaptive_runtime_v1
   ```

3. Check extracted file hashes against `bundle_manifest.json`. Use `student1_pre_test_history.json` for the original collator. It contains 179 events across nine allowed sessions, including model question and skill IDs, real correctness, timestamps and cutoff. Configure the extracted bank CSV as the server's bank input.
4. Use the original checkpoint and tracked mappings and question rankings. Pool representatives across composites for a tag, then rank by training exposure descending and model question ID ascending. This reproduces the full training store's top five for all 1,326 tags.

Decryption and history arrays were checked against the original store. Keep the passphrase outside Git and configure the separate Gemini API key later as planned.

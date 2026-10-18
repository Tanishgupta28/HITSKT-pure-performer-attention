# Independent correctness audit — completed

Updated: 2026-10-18 UTC. Audit base: `656a9f0`.

Deliverable: `reports/HITSKT_INDEPENDENT_CORRECTNESS_AUDIT.md`.
Evidence: `reports/independent_correctness_audit/`; scripts `scripts/audit_*.py`.
The commit containing this record is the scoped audit delivery. Verify its
normal push to `capstone-gpu/main` before ending; never force push.

## Established outcomes

- **133 distinct tests PASS** (84 original +49 new), with explicit causal kernel
  outputs/all gradients, maximum-length reduced-width stress tests, actual
  checkpoint widths, raw own/future-label perturbations, dropout, EOS/PAD,
  batch/student/session ownership and baseline label/PAD exclusion.
  `test_summary.json` reconciles all logs/XML. Full HiTSKT requires supported
  contiguous right padding; arbitrary external left/hole batches are not certified.
- All three exhaustive accepted-source -> processed Parquet -> session-store
  reconstructions PASS. Every supervised event/field and history bound checked.
  EdNet scanned all 784,309 files  / 95,293,926 raw rows with ZIP CRC.
  Official current KT1 and Contents downloads match local SHA256 exactly.
  Flat CSVs match the original authors' LFS bytes at upstream 6c930c5; deeper
  publisher-raw-to-author transformations remain NOT VERIFIED.
- Evaluation is **chronological within-student**, not student-disjoint or a
  global calendar split. Earlier held-out labels are visible online; own/future
  index labels are excluded. All retained students appear in all three splits.
  HiTSKT first-session training interactions are context-only; baseline train
  target populations therefore differ, although validation/test populations match.
- Active attention is **deterministic ELU+1 causal linear attention**, not FAVOR+.
  Other original-paper components and end-to-start gap semantics also differ.
  No architecture or original experiment source was changed.
- Every HiTSKT best checkpoint freshly reloaded and all 15,998,524 combined test
  targets replayed on CPU. All three **FAIL** the declared absolute1e-6 metric
  gate; no tolerance was relaxed and no historical score was overwritten:
  - ASSIST 112,252 targets: AUC0.709623637214297 vs 0.7096240159197755;
    classification differences and BCE delta 2.13e-6 exceed gate.
  - Junyi 2,456,402: AUC0.7978264739233538 vs 0.7978262055191263;
    classification differences exceed gate; BCE/MSE pass tolerance.
  - EdNet 13,429,870: AUC0.7693647361509419 vs 0.7693648364413692;
    accuracy/recall/F1 differences exceed gate; precision/BCE/MSE pass tolerance.
  All AUC differences pass 1e-6. Full ASSIST production CPU crosscheck exactly
  matches independent CPU metrics. CPU vs historical CUDA cause remains open;
  no fresh CUDA replay was attempted while protecting ongoing training.
- Authoritative manifests: `assist2017_evaluation_manifest.json`,
  `junyi_evaluation_guarded_manifest.json`,
  `ednet_kt1_evaluation_guarded_manifest.json`. All targets/sessions covered once.
  Separate fresh processes exactly matched three new guarded-worker batches per
  dataset after copied prefixes. Default unperformed parity fields were annotated;
  emitted manifests and all interrupted attempts remain preserved.
- Real feedback availability NOT VERIFIED. ASSIST has2,426 test end/start
  overlaps; EdNet has4,003,464 under completion proxy. Junyi has2,165,594 tied
  test timestamps. Eight sampled overlaps each changed ASSIST/EdNet predictions
  when predecessor labels changed; own-label flips stayed zero. Do not confuse
  these timing assumptions with a future-index attention leak.
- All14 completed benchmark artifacts and 295 epoch records pass scoped checks;
  baseline test metrics were NOT freshly replayed. RKT full persisted fold algebra,
  key domains and CRC checks pass; raw label-pair counts were not regenerated.
  Junyi/EdNet recorded gains are seed 42 descriptions, not significance evidence.
- DKT/EdNet remains unfinished: last epoch 9 Adam has 21,749 positive infinities in 5
  tensors; best epoch 1 model/Adam finite; counter 8/active patience 10. Embedded checkpoint
  metadata says 5. No recovery, optimizer reset or training was attempted.
- Retained distinct questions/skills differ from index bounds in ASSIST/EdNet:
  ASSIST 3,112/102; EdNet 12,283/1,490. Metadata capacities are 3,158/102 and
  13,169/1,495. Correct descriptions, not saved embedding IDs/EOS indices.

## Preservation and audit-process limitations

`preservation_verification.json`: **PASS**, 375 captured original file hashes,
all 30 best/last checkpoints, and all 15 best checkpoints matching actual base Git
binary blobs. Original 36 implementation files unchanged. `evidence_integrity.json`
checks 33 source bindings, all 12 full prediction arrays and passing-test artifacts.
Raw interactions/predictions/scratch stay ignored under `.inspection/`.

Two early audit-only jobs were OOM-killed because host RAM was mistaken for the
90 GiB cgroup limit. Both were rerun successfully. Guarded continuations used nice 19,
idle I/O, CPUs216–223, CUDA disabled and 24 GiB estimated shared inference admission.
That estimate is not a bound on total RSS/cache. No further OOM kills occurred.
Final observation retained two protected LightSAKT2 identities (182135/2028941);
1992414 was absent, with exit reason NOT VERIFIED. The external status heartbeat
remained current with active jobs. No additional OOM kills or audit signals/edits.
No LightSAKT2 file/process was edited, signalled or reconfigured.
See `resource_incident.json` and `resource_final_state.json`.

An initial EOS-flip fixture and overly strong ASSIST timing interpretation were
corrected with original attempts preserved; final test/scan results supersede them.

## Next action

Finish scoped review/commit/push, verify remote HEAD, then wait for instructions.
No training, new experiments or detached monitoring is authorized by this endpoint.
Retraining is not established by CPU metric differences alone. Future decisions
must follow the report's conditional recommendations; preserve all historical
checkpoints/outputs. Keep unrelated root README and SAKT console edits excluded.

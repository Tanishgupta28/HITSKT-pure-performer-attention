"""Render verified files into the study report; no new numerical experiment."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
S=json.loads((HERE/'ATTEMPT_SUMMARY.json').read_text())
R=json.loads((HERE/'PHASE1_REVIEW_SUMMARY.json').read_text())
D=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist_bounded_review_20261020_v2'
P=json.loads((D/'prepared_plan.json').read_text())
B=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist_bounded100_20261020_v1_s020'
base=S['milestone_0'];v=base['sampled_validation'];t=base['sampled_training_eval']
metrics=f"|0 (initialization)|{t['loss']:.9f}|{v['loss']:.9f}|{v['roc_auc']:.9f}|{v['accuracy']:.9f}|{v['precision']:.9f}|{v['recall']:.9f}|{v['f1']:.9f}|"
stages=[]
for name,a in R['review_decision']['decisions']['initial']['aggregate'].items():
 b=R['review_decision']['decisions']['after_five']['aggregate'][name]
 stages.append(f"|{name.split('.')[0]}|{a['relative_l2']['median']:.6f} → {b['relative_l2']['median']:.6f}|{b['relative_l2']['maximum']:.6f}|{a['pair_ess_medians']['median']:.3f} → {b['pair_ess_medians']['median']:.3f}|{b['max_p95_growth_ratio']:.4f}×|")
feature=[]
for row in R['frozen_feature_grid']:
 if row['stage']=='correct_padding_encoder':
  a=row['relative_l2'];feature.append(f"|{row['state']}|{row['features']}|{a['median']:.6f}|{a['p90']:.6f}|{a['maximum']:.6f}|")
updates=[]
for p in sorted(B.glob('update_*.json')):
 r=json.loads(p.read_text());updates.append(f"|{r['update']}|{r['training_bce']:.9f}|{r['gradient_l2']:.6f}|{r['supervised_targets']}|{r['training_seconds']:.6f}|")
text='''# ASSIST2017 FAVOR+ bounded training/validation pilot

20 October 2026 UTC. Branch `research/hitskt-favor-plus-v1`; continuation from `eac2a375683b48c6510bd3b3fc1f2ddc82357fb7`. Design frozen in commit `b45906a`; executed runner in `1966fa1`. No attention/nonattention model, original optimizer, data or evaluation-protocol changes.

**Recommendation: REVIEW before another bounded experiment; NO-GO for full training.** The approximation review supported a contained diagnostic study. The new CUDA study was **RESOURCE-ABORTED after 14 successful Adam calls**, before its first 20-update milestone. An unidentified process held GPU handles; no override, signals, retry, reset or new training run followed.20/50/100-update validation, trained-state diagnostics and optimizer continuation remain **NOT VERIFIED**. No test evaluation or final dataset-performance result exists.

## Design and decision made before execution

[PREDECLARED_DESIGN.json](assist_bounded_v1/PREDECLARED_DESIGN.json) was committed before new diagnostics. Original CorrectPaddingEncoder (CPE) **REVIEW_REQUIRED** and relative-L2 review trigger **0.20 remain unchanged**. The previous0.181169→0.236181 result is neither relabeled PASS nor discarded. New containment rules permit diagnostic observation, not scientific acceptance of that approximation.

This version starts fresh at seed42 and permits100 **total** updates; it does not continue five old steps for100 more. Same positive Gaussian orthogonal features, m128, projection seed42/stage offsets, fixed saved projections, chunk32, FP32 accumulation. Same ASSIST width256/heads4/FF2048/dropout0.1/four single-layer stages, Adam LR5e-5/weight decay0/default betas/epsilon, original preprocessing/rolling history. Pilot exceptions B1/cost2048 and limited updates are explicit. This is genuine FAVOR+; historical ELU+1 results are separate.

The frozen plan selected:

- Training: first100 distinct eligible whole targets from original epoch1/seed42 shuffled length-bucket sampler; selection by shape/index only, oversized singleton skipped without truncation.79 students, target lengths1–68, histories1–8, maximum cost536. This small first bucket does not represent all original training shapes.
- Diagnostics: six distinct students from target-length bins1–32/33–128/129+ crossed with histories1–4/5–15, selected by ascending target ID with distinct-student preference. All six strata exist: (length,history,cost)=(28,4,322),(26,9,1360),(87,1,141),(79,5,436),(226,1,353),(288,6,674). No labels or scores selected diagnostic examples. Per stage, compare one row's first≤32 positions (≤history count+EOS for SessionEncoder); whole-example Q/K norm summaries include valid EOS.
- Validation:32 unique whole validation targets, round-robin the same six metadata bins in ascending target ID, frozen before scores/new training.29 students, lengths2–572, histories3–11, maxcost1992; **3157 labels:1063 positive/2094 negative**, exceeding prospective500 total/25 each. Sample contains multiple sessions from some students; it is not a student-disjoint/unbiased full-validation estimate.

[Executed prepared plan](../../experiments/hitskt_favor_plus_v1/assist2017/assist_bounded_review_20261020_v2/prepared_plan.json) records indices, boundaries,11 verified processed hashes, architecture/profile/source identities and design hash. No test target was selected, forwarded or evaluated; checksums cover shared processed files without evaluating their test rows.

Prospective milestones0/20/50/100; manual evidence review before separately invoking continuation. Limits100 Adam calls,45min single-campaign wall time including inter-stage review gaps,120s/update,8GiB allocator,4GiB process-RSS gate and2GiB shared-container headroom. Stop on nonfinite loss/logits/gradients/model/Adam, failed exact own/future-label invariant, resource/conflict/deadline or reload failure. Additional severe containment stops: feature-kernel maxabs>5e-5; any stage's median prefix relative L2>0.75, any prefix>1.5, stage median pair-ESS<1.25, any Q/K p95>2×initial, global grad-L2>100, or BCE>20. These engineering containment thresholds were frozen before measurement and are **not** replacement acceptance tolerances. Original0.20/ESS2 review flags remain meaningful. No monotonic-loss requirement or score-driven feature selection.

## Phase1: approximation root cause — PASS for containment, REVIEW for quality

[Review evidence](assist_bounded_v1/PHASE1_REVIEW_SUMMARY.json), [original Q/K/V decomposition](../../experiments/hitskt_favor_plus_v1/assist2017/assist_bounded_review_20261020_v2/saved_cpe_qkv_decomposition.json), [1152 frozen comparisons](../../experiments/hitskt_favor_plus_v1/assist2017/assist_bounded_review_20261020_v2/frozen_feature_comparisons.json). Fresh initialization and the unchanged original five-update checkpoint were evaluated on all six training examples on CPU. No optimizer calls, validation-score tuning or CUDA were used in this review.

Median across the six examples using **deployed**128-feature stage projections:

|Stage|Relative L2 vs exact softmax: initial → original step5|Maximum at step5|Median pair-ESS: initial → step5|Maximum Q/K p95 growth|
|---|---:|---:|---:|---:|
'''+ '\n'.join(stages)+'''

Thus the CPE warning is broader than one example: median error rises about 45%, though all values remain inside the separately declared containment rules. ActionEncoder also exceeds the old0.20 review trigger on several examples. CPE per-head/query p95 relative errors reach **0.4796–0.5366** in five of six step5 examples; aggregate relative norms hide tails. Individual pair ESS minima reach about1.14 while stage medians remain above2. Norm growth alone does not certify quality. Across the six CPE examples, query p95 norms span4.885–5.394 initially and4.937–5.515 at step5; key p95 spans4.921–5.430 and4.953–5.427. Full distributions are in the saved diagnostics.

Across all 48 deployed CPU prefixes, production matches the independent explicit float64 feature kernel with maximum absolute difference **'''+f"{R['maximum_independent_feature_kernel_error']:.12g}"+'''**, below the unchanged5e-5 gate. No nonfinite value or tested implementation mismatch explains the rise. The original 18-case CUDA correctness evidence remains valid: its scientific source, oracle, harness and allocation guard hashes were rechecked unchanged before this new CUDA attempt; no full 229-test rerun was needed.

On the original saved CPE example, independent frozen cross-combinations give relative L2:

|Q state|K state|V state|Relative L2|
|---|---|---|---:|
|initial|initial|initial|0.181169|
|initial|initial|step5|0.181046|
|step5|initial|initial|0.209616|
|initial|step5|initial|0.204788|
|step5|step5|initial|0.236464|
|step5|step5|step5|0.236181|

Changing V alone contributes little to the measured increase; changing both Q and K reproduces it. Mean attention total variation rises0.1255→0.1644 and median ESS8.889→7.508. Q/K aggregate tensor changes are approximately12.0%/11.9%, median direction cosines0.9929/0.9930, despite only modest norm changes. Exponentiated random-feature contributions respond to direction as well as norm.

Interpretation: the fixed finite feature map is sensitive to evolving learned Q/K and increasing concentration; it is not being randomly redrawn during optimization. Offline seed variation establishes substantial sampling variability. This supports finite-feature approximation/alignment as the tested explanation, rather than FP32 instability or a detected causal-feature-kernel defect. It does **not** establish a unique statistical mechanism, rule out all long-context/high-norm defects, or prove acceptable trained-model softmax quality.

### Frozen128/256/512 comparison

Six examples × four stages × two saved states × three feature counts × eight predeclared **diagnostic projection seeds0–7**=1152 comparisons. Q/K/V and masks are identical across feature counts; model weights/architecture are not changed. This is not training with extra model seeds. Orthogonal Gaussian projections are not nested/identical across counts. Below, each CPE row summarizes48 frozen example/projection combinations; all other stage distributions are in the JSON.

|State|Features|Median relative L2|p90|Maximum|
|---|---:|---:|---:|---:|
'''+ '\n'.join(feature)+'''

512 improves median CPE error, but its step5 p90 is worse than128 and its maximum remains substantial. At other stages median error generally improves. More features do not uniformly fix tails, and the deployed stage projection can be worse than the diagnostic-seed median. Training remained128; no feature count was selected by validation/test scores.128 remains a **diagnostic candidate**, not an established full-training configuration. High-norm quality remains unresolved. No new GPU timing/memory for256/512 was measured or inferred.

## Resource admission and CUDA execution

[Owner authorization](assist_bounded_v1/OWNER_AND_STUDY_AUTHORIZATION.md), [admission](assist_bounded_v1/resource_admission.json), [execution config](../../experiments/hitskt_favor_plus_v1/assist2017/assist_bounded100_20261020_v1_s020/execution_config.json). Existing owner-confirmed research allocation was accepted without requesting administrator proof for Kubernetes403. Exact exposed **MIG-d2fcfd79-720e-5218-82c9-0a132be68920**, H10080GB parent **GPU-5a568e3c-e96c-169f-b063-f4478340ff51**,3g.40gb. Admission 11:48:08UTC: total42,278,584,320 bytes/free42,238,869,504, no visible GPU holders/conflicts, no visible Light trainer, persistent restart hold active.90GiB host limit, about18.65GiB then used; cumulative OOM kills2 unchanged. Owner statement is authority; no external scheduler or reservation proof is invented.

Unchanged allocation guard plus an additional study-specific approval protects this runner: a five-step approval cannot authorize100. Exact device exposure, active Light hold, conflicts, RAM and cumulative budget checked at boundaries;8GiB PyTorch allocator quota enforced. A two-hour operator access-record window is not a claimed scheduler lease;45min campaign wall budget is stricter.

Runtime Python3.10.12/Torch2.4.0a0+f70bd71a48.nv24.06/CUDA12.5/NumPy1.24.4, GPU capability9.0. FP32/highest, TF32off, deterministic algorithms/cuDNN on, benchmark off, CUBLAS_WORKSPACE_CONFIG=:4096:8, one low-priority CPU dispatch core223. No GPU allocation changed and no unrelated process was signaled.

At initialization, independent CUDA feature-kernel maxabs across six-example stages was6.56e-7. Whole-target re-collation flipping raw responses0/144/287 preserved own/earlier logits **exactly**. Prior 18 staged tests/gradient references/checkpoints remain inherited PASS, not 18 new tests here. Trained-state causal/norm/error milestone checks were not reached.

**After update 14**, boundary authorization rejected GPU-handle PID **2751270**. It had exited by 11:49:21UTC before identity inspection, so ownership/purpose and whether it was a compute job or benign management query are **NOT VERIFIED**. The unchanged guard already excludes strictly verified trusted read-only nvidia-smi commands; no new exception was made. This is a conservative resource abort, not a confirmed numerical failure, and not proof of another user's reservation. [Failure](../../experiments/hitskt_favor_plus_v1/assist2017/assist_bounded100_20261020_v1_s020/stage_failure.json), [identity limitation](assist_bounded_v1/stage_020_conflict.json). No automatic retry/restart/reset/50/100 stage occurred. No 20-update checkpoint exists; the14-update in-memory state was lost on safe process exit, not reconstructed or represented as resumable.

[Final release 11:52:42UTC](assist_bounded_v1/resource_release.json) shows no visible GPU holders/Light trainer, identical hold hash, MIG use39,714,816 bytes back to baseline and unchanged OOM counter. No watcher/background training was launched.

## Training/validation outcomes

**Only the initialization row is a frozen-sample evaluation.** Training-eval BCE uses six diagnostic training sessions/734 genuine targets; validation uses the frozen3157 targets. They are sampled exploratory values, not final/full-split performance, student-disjoint generalization or evidence of model superiority. Predictions, labels, masks/counts and hashes are preserved.

|Milestone updates|Sampled training-eval BCE|Sampled validation BCE|Validation ROC-AUC|Accuracy|Precision|Recall|F1|
|---:|---:|---:|---:|---:|---:|---:|---:|
'''+metrics+'''
|20|NOT REACHED|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|
|50|NOT RUN|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|
|100|NOT RUN|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|

Initialization validation MSE0.269878258; threshold>=0.5, original zero-division handling. Independent rank-based AUC and direct NumPy confusion metrics from saved probabilities reproduce reported values (largest metric difference5.55e-17). BCE reconstructed from rounded float32 probabilities differs from logit-based BCE by2.294e-8; this is a cached-score consistency check, not fresh checkpoint replay. [Accounting](assist_bounded_v1/ATTEMPT_SUMMARY.json).

Fourteen actual training records (752 target actions total):

|Update|Minibatch training BCE|Gradient L2|Targets|Synchronized training seconds|
|---:|---:|---:|---:|---:|
'''+ '\n'.join(updates)+'''

All 14 gradients/model/Adam states were finite; all 14 parameter deltas were finite/nonzero. No optimizer call raised and no applied step was replayed. First→last loss1.100965→0.715914 uses different examples/dropout; target-weighted BCE across these updates0.794004521. This does not establish convergence, generalization or even improved fixed-sample validation. Gradient L2 range1.98417–7.85442, below prospective100 stop.

Synchronized training intervals sum **2.666099s**, mean0.190436s/update; measured stage execution to abort **10.035558s** includes initialization/diagnostics/validation/checks, excludes imports/data verification. Peak recorded PyTorch allocated **283,240,448 bytes=270.119MiB** through update14; driver overhead excluded. Initial reserved154MiB and initial process RSS1.471GiB are **lower bounds**, not final peaks: failure logging did not persist final reserved/RSS high-water values. No continuous NVML peak measurement. This telemetry limitation must be fixed before the next attempt; do not present those baseline values as complete-run peaks. Memory/deadline checks passed until the conflict gate stopped execution.

These small training shapes do not certify full histories/B64/cost32768 or full-epoch runtime. Initialization forward evaluation did exercise validation cost1992/length572/history11; optimization/backward did not cover those largest shapes. No GPU epoch estimate is defensible from the14 small updates.

## Additional finding: old fixed-example BCE label was wrong

The original five-step runner calls fixed_example_diagnostics, which correctly temporarily enters eval and then **restores the previous training mode**. It then computes another net(fixed_batch) inside no_grad without an explicit eval call. During the old pilot that mode was training, so dropout was active. The field `fixed_training_example_eval_bce` and the old report's “without dropout” description are incorrect. The recorded values are train-mode stochastic diagnostics; they are not a dropout-free fixed-example learning curve.

Severity **MEDIUM, diagnostic interpretation/RNG consumption**. It affects that auxiliary five-step curve/prose, not the valid training losses, Q/K eval-mode prefix diagnostics, original ELU+1 results or18 CUDA correctness tests. No retraining requirement follows. Preserve the old files unchanged; this report supersedes their interpretation. Tiny CPU mode/RNG tests confirm that no_grad alone does not disable dropout, the old helper restores training mode without RNG draws, and this new evaluate function disables dropout/restores mode without consuming RNG. [Tests](../../tests/test_favor_plus_bounded_assist.py), [CPU sanity trace](assist_bounded_v1/dropout_cpu_investigation.json).

This also explains why the new first training BCE equals the old initial “eval BCE”: that old diagnostic consumed the first CUDA dropout draws; old actual training used later draws. New explicit eval diagnostics preserve them. Same seed/settings do not imply the old and new optimization trajectories are identical after changing diagnostic RNG consumption. No claim of independently reproduced old five-step losses is made.

## Verification, preservation and limitations

- **PASS:**102 targeted CPU tests (32new study controls+70existing allocation/five-pilot controls), zero final failures; prior229 core tests inherited, not repeated. Tests cover action gating, strict milestone order, cumulative budget/memory, hard-stop criteria, exact re-collated causal detection, masked metrics, no-step probe/RNG preservation and mode controls. [Log](assist_bounded_v1/targeted_cpu.log), [JUnit](assist_bounded_v1/targeted_cpu.xml).
- **PASS:** old 70-file pilot inventory freshly rehashed byte-identical, including two prior checkpoint binaries and24 prefix arrays.11 ASSIST processed file hashes verified; original scientific source hashes unchanged. No ELU+1, historical results/datasets/audits, LightSAKT2 or DKT changes. Unrelated root README and SAKT log remain uncommitted/untouched. Full375-original re-audit was not repeated.
- **NOT VERIFIED:** trained-state approximation beyond the original five updates,20/50/100 validation, this runner's GPU checkpoint save/reload/fresh-process optimizer continuation, original large-batch backward feasibility or convergence. The separately preserved original five-update checkpoint reload PASS does not substitute for those checks.
- Failed/superseded preparations are retained. First test attempt98PASS/1FAIL was a test comparing dict with OrderedDict through a type-strict helper, fixed by normalizing both; no actual parameter mutation. Early preflight mistakenly read an absent hold.active field, then overclassified an observed nvidia-smi handle; both were before CUDA and preserved. Final preflight used the actual hold.status and unchanged production conflict classifier. CPU review_v2 refresh binds final hold/containment/preservation controls; both CPU reviews yield identical frozen comparisons. No original numerical gate was relaxed.
- Checkpoints remain local/ignored exactly as before; this attempt reached none of its new checkpoint milestones. New bounded NPZ/JSON/code/logs and hash inventory are committed; full data/checkpoint access is still needed for fresh reproduction. No benchmark `_SUCCESS` marker was produced.

## Research decision and next experiment

**REVIEW for another bounded ASSIST attempt; NO-GO full ASSIST/Junyi/EdNet training.** Finite14-step optimization is encouraging execution evidence but the requested staged learning/quality verification is incomplete. The original approximation warning has a plausible, measured finite-feature/QK explanation; it has not been scientifically resolved.

Before another separately authorized bounded attempt:

1. Identify transient GPU-holding activity using administrator/server operational records or immediate sanitized PID metadata at guard failure; obtain a conflict-free execution interval within the existing owner allocation. No renewed administrator permission is required merely for Kubernetes403, and free VRAM alone is not conflict evidence. Do not weaken the unknown-process guard or stop unrelated jobs.
2. Improve per-update/failure telemetry to persist last RSS/reserved high-water and immediate safe conflict identity. Save eligible milestone states and consider CPU-preserved recovery evidence without further device operations after a conflict. Never treat a resource-aborted state as an eligible successful checkpoint.
3. Prepare a fresh source-bound plan and explicitly version/authorize a new total update budget; do not rerun this aborted study silently. Keep the existing frozen validation target list/diagnostic strata/protocol, m128 and original optimizer/architecture. Review at 20,50,100 and reproduce model/Adam/RNG/cursor plus next train-mode logits/gradients in a fresh process with **no extra optimizer step**.
4. Require finite optimization, unchanged causal/reference gates, acceptable monitored quality/ESS tails on multiple learned states and frozen sampled validation before discussing full ASSIST. The original0.20 warning must remain visible even if a bounded run meets severe-containment rules.
5. Profile original15-history/longer session backward and batch shapes gradually under measured quotas before full-training resource estimates. Junyi/EdNet each need dataset/profile-specific bounded real-data and memory checks. Full runs and validation-selected checkpoints require a later explicit decision; only then independently reproduce AUC/accuracy/all metrics. Preserve V1 chronological **within-student retrospective teacher-forced** interpretation and response-availability qualification; this study neither proves timing leakage nor removes that limitation.

Future command shape, **not an instruction to execute now**: CPU `python -m scripts.bounded_assist_favor_plus prepare-review --run-id NEW_REVIEW`, then the explicit `stage --plan NEW_REVIEW/prepared_plan.json --allocation NEW_RECORD.json --run-id NEW_STAGE020 --to20 --execute-cuda` under exact authorized CUDA_VISIBLE_DEVICES/CUBLAS settings. Only after reviewed PASS use `--to50 --previous-checkpoint checkpoint_020.pt`, then `--to100 --previous-checkpoint checkpoint_050.pt`. The preserved failed directory cannot be resumed by this controller and must never be overwritten. No new full-training authorization is implied.
'''
(ROOT/'reports/favor_plus_v1/ASSIST_BOUNDED_TRAINING_VALIDATION_PILOT.md').write_text(text)
print('report written',len(text.split()),'words')

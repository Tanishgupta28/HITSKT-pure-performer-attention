# HiTSKT serving contract and content provenance

The adapter imports `ktbench.models.hitskt.HiTSKT`, `Session`, `SessionExample`
and `collate_hitskt` from the unchanged repository. It loads a real version-1
checkpoint through `HiTSKT.from_checkpoint(...).eval()` on CPU. It neither
starts training nor edits a scientific artifact. PyTorch and NumPy are optional
platform dependencies installed through `backend/requirements-model.txt`.

On Windows, `backend/app/research.py` loads the exact existing model source
without executing `ktbench.models.__init__`. That initializer imports unrelated
RKT utilities with a Linux-only `resource` dependency. The narrow serving loader
executes the original `performer.py` and `hitskt.py` files without copies or
source changes; checkpoint keys and model arithmetic stay intact.

## The important content gap

Research question IDs are learned categorical embeddings, not semantic text
encodings. Mapping a newly authored fraction question to a convenient Junyi or
ASSIST question ID is invalid. Existing metadata supplies source/model IDs,
but no complete licensed question text and answer bank suitable for publication.
The original demonstration bank therefore deliberately has no benchmark IDs.

To use a trained checkpoint for student-facing items, obtain licensed original
content for that same dataset and map each platform item to its exact trained
question and composite-skill IDs. Alternatively collect consented platform
interaction data and train/validate a separate model for this content domain.
That training is a separate research decision owned by the GPU workstream.

## Loading a checkpoint

The two optional environment variables must be set together:

```text
HITSKT_CHECKPOINT=<absolute path to a trusted best_model.pt>
HITSKT_MANIFEST=<absolute path to a reviewed manifest.json>
```

Manifest JSON has `format_version: 1`, the exact `dataset` name,
`checkpoint_sha256` and a `questions` object keyed by platform question IDs.
Each mapping value supplies integer `question_id` and integer `skill_id` from
the trained vocabulary. Produce it with `backend/tools/build_manifest.py`
using a content-owner-reviewed mapping file; the tool verifies each pair against
the dataset's persisted reports. Do not fill this file with approximate IDs.
Checkpoint hash or vocabulary mismatch causes startup failure, not a silent fallback.

The current catalog is code-owned in `backend/app/catalog.py`. Reviewed replacement
content and its exact mappings must be integrated together as a platform-only
change before model-backed subject delivery. General content import/admin tools
are not implemented in this release.

## Session and causality rules

- Group actual response timestamps by the research ten-hour inactivity threshold;
  assessment boundaries are not fabricated training sessions.
- Keep up to 15 complete prior sessions and all strictly prior actions in the
  current session. Do not chunk or truncate sessions to fit a request.
- Require at least one real earlier session. A new learner uses the explicitly
  labeled Bayesian cold-start provider. Do not invent a BOS history session.
- Require exact mapped question/skill IDs for every included history item and
  candidate. Never drop an unmapped item from a mapped sequence.
- Collate the current session plus the unanswered candidate. The candidate's
  dummy label cannot enter its own prediction because correctness inputs shift
  by one position; select the candidate logit before the EOS position.
- Use `torch.inference_mode()`. Return provider, dataset and checkpoint SHA256
  with the prediction, and persist that provenance with each submitted answer.
- Batch eight candidates at a time. If the complete serving context exceeds
  4,096 actions, use a visible `serving_context_capacity` fallback; do not claim
  scientific evaluation equivalence for this product serving limit.

The concept update remains separately labeled Bayesian knowledge tracing.
HiTSKT predictions guide question selection; they are not relabeled as mastery.

## Request for the GPU workstream

No interruption or retraining is requested. When convenient, inspect
`project-memory/PLATFORM_HANDOFF.md`. Supply or review a deployable checkpoint
identifier, immutable hash, mapping-report identity, intended content domain,
licensed question text/answers, and a genuine mapped student-history fixture.
The platform agent can then validate trained inference against the GPU agent's
known output, add the reviewed bank, and measure latency/calibration separately
from the benchmark. Keep all existing experiments and scientific protocols intact.

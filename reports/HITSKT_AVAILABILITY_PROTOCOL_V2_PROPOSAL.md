# HiTSKT response-availability protocol V2 — proposal only

Status: **NOT IMPLEMENTED / NOT TRAINED**. Historical V1 datasets, loaders, model
weights and results remain immutable. V1 means chronological within-student,
retrospective prediction conditioned on earlier indexed responses. The timing
evidence is in [the timing analysis](HITSKT_RESPONSE_TIMING_ANALYSIS.md).

V2 would answer a different question: what can be predicted using responses
actually released before a declared decision event? It must receive a distinct
protocol ID, immutable preprocessing manifest and experiment namespace.

## Decision and release contract

1. Define the decision event before examining new test metrics: question
   presentation, or initial bundle presentation for a bundle-level task. Score
   every retained target exactly once at that event.
2. Maintain separate presentation, final-response submission and correctness
   release events. A response enters model memory only after the declared
   scoring service could compute/receive its correctness. Specify whether the
   service knows the answer key at submission; learner feedback display and
   server availability are distinct events.
3. Require release time strictly before decision time, unless authoritative
   action ordering proves the equal-time release precedes the decision.
   Source-row tie ordering alone is insufficient. Missing release time means
   unknown availability; publish coverage and separate that cohort rather than
   fabricating an ordering or silently dropping inconvenient targets.
4. For EdNet, predict all items at initial bundle entry without any final label
   from that bundle. Release its final responses at a verified submission/
   scoring event. Join KT2/KT3 when possible, check episode identity and join
   coverage, and distinguish matched learners from full KT1. An end inferred
   from `timestamp+elapsed_time` remains a proxy protocol, not certified V2.
5. For ASSIST2017, recover the relevant answer/help and scoring event from
   original action logs; neither problem completion nor a step-duration proxy
   establishes first-response correctness release. Retain raw transformation
   provenance and distinguish scaffold activity.
6. For Junyi, first reconstruct the author CSV from publisher attempt logs and
   establish the presentation/attempt/completion mapping. Equal transformed
   start/end times must not be interpreted as instantaneous feedback.

## Data and model controls

Preserve the V1 cohort and within-student split assignments where event
identities permit, and document any necessary cohort change separately. This
does not create a student-disjoint or global-calendar split. Assign sessions
using the registered presentation-time rule, version any revised boundaries,
and require histories to respect both student identity and release order.

Make the response-memory policy explicit. HiTSKT must filter or delay every
response-bearing ActionEncoder/session summary and CorrectPaddingEncoder
prefix; delaying only the adjacent decoder token does not suffice. Unknown
responses need a registered representation or must remain outside memory.
Query/item information available at decision time can still be used. Preserve
the no-self-target and no-future-response invariants.

DKT and DKVMN must update state on release events, while SAKT and RKT must use
only released histories. Rebuild/version RKT's training-only relations and
cross-fitting if labels, populations or their allowed training information
change. Use matching targets, prediction times and release assumptions for all
methods; a corrected HiTSKT score cannot be compared to V1 baseline scores as
if their information sets matched.

## Verification before any new full training

Build hand-enumerated examples with concurrent questions, revisable choices,
bundle submission, delayed feedback, equal timestamps, multiple students and
session/split boundaries. Prove that delaying or changing an unreleased
response cannot alter any earlier prediction, including cached session memory.
Verify exact target coverage, state release ordering, join exclusions and
bounded-memory behavior. Freeze hashes, metrics, numerical settings and
acceptance rules before evaluation. Keep missing-time and proxy-time results
separate from certified availability results.

A frozen-checkpoint sensitivity evaluation under V2 does **not** require
retraining. It measures distribution shift and must be labelled as such. If
the publishable training/evaluation contract changes which responses models
may consume, train HiTSKT **and the affected baselines** under that new
contract for a controlled V2 comparison. This conditional need does not
retroactively prove that the historical V1 models were trained incorrectly.
No V2 full training is authorized by this proposal.

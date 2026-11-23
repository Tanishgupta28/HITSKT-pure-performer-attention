# Junyi adaptive assessment in Lumen

## Run the pilot

Use the regular [Windows platform setup](README.md#run-on-windows) and install `backend/requirements-model.txt` into the same backend environment. The API loads the unchanged epoch-32 Junyi checkpoint on CPU and checks its SHA-256 before deserializing it. The pilot loads only when `JUNYI_STUDENT_ACCOUNT_ID` identifies the existing Lumen student account assigned to this research history. The authorized override account needs the Administrator role and its separate `JUNYI_OPERATOR_ACCOUNT_ID`. Read the UUID of each existing account from the authorized local account store or its authenticated `/api/auth/me` response. The browser never supplies a Junyi research student ID.

1. Obtain the original checkpoint at `experiments/hitskt/junyi/full/best_model.pt`. A verified private copy can be used with `JUNYI_CHECKPOINT` set to its absolute path. Its expected SHA-256 is `2308c5bf0c850da1e55daf802a0853a30fac45eeed071c7162c68fc17bfa3bdd`.
2. Download the [encrypted runtime release](https://github.com/Tanishgupta28/HITSKT-pure-performer-attention/releases/tag/junyi-adaptive-runtime-v1). Obtain the authorized passphrase file privately. Run the setup below from the repository root with the Python executable from the backend environment. On Windows, pass `--gpg 'C:/Program Files/Git/usr/bin/gpg.exe'` if using Git for Windows and `gpg` is outside `PATH`.

   ```powershell
   ./backend/.venv/Scripts/python.exe backend/tools/prepare_junyi_runtime.py <archive-path> --passphrase-file <private-passphrase-file>
   ```

3. Download `representative_questions.csv` from the [pinned research handoff](https://github.com/Tanishgupta28/HITSKT-pure-performer-attention/blob/9ff17e674c37c3a9ad59201133f6fac9acc99381/reports/skill_threshold_experiments/representative_questions.csv) into `data/processed/junyi/adaptive_runtime_v1/`. The directory also receives the approved bank and baseline from the verified archive. The importer also accepts the identical verified bank supplied separately and checks its hash. Set `JUNYI_HISTORY_SHA256` from the local setup output. The app validates Student 1's source and processed IDs, the 9-session / 179-interaction shape, ordered legal question/skill pairs, and the cutoff before the first excluded test event.
4. Add the account bindings and optional `JUNYI_STAGE` to the ignored backend environment and restart FastAPI. The stage defaults to all verified stages. Elementary and Junior can each supply 20 questions. Senior has nine in the approved bank. For optional Gemini question selection and study-link ranking, configure `GEMINI_MODEL` and `GEMINI_API_KEY` on the backend. The pilot uses `gemini-3.1-flash-lite`. The rules can build every compatible paper and show verified lessons without Gemini.
5. Sign in as the assigned learner, open Today and start the assessment. Use the study checklist, complete one verified resource for each required skill, then complete the paper and view its result. The operator can resolve a failed study lookup under Question bank → Junyi assessment operations. Overrides need a recorded reason. Restrict the answer-bearing bank and private history to the backend and authorized collaborators.

## Recover the approved history from the verified raw dataset

If the passphrase is unavailable, use an authorized copy of the original `Junyi.csv`. Fetch the pinned representative ranking from step 3 first. With NumPy and pandas in the backend environment, run this offline from the repository root:

```powershell
./backend/.venv/Scripts/python.exe backend/tools/rebuild_junyi_student1.py <path-to-Junyi.csv>
```

The recovery tool hashes the accepted raw file, replays the published ordering, question/composite mapping, session boundaries and split, and compares all 11 reconstructed SessionStore member digests and the complete student crosswalk with the independent audit manifests. It saves only the verified 9-session / 179-interaction prefix in the ignored runtime directory. The other learners' records and Student 1's held-out events never enter the website. Set `JUNYI_HISTORY_SHA256` to the hash printed by the recovery tool. Use the separately verified question bank and continue with step 4.

## What the application stores

MongoDB keeps a separate `junyi_papers` collection for the prototype. Each frozen paper contains its bank version, score snapshot, allocation and deficit record, ordered items and option IDs, and only the required study tags for that paper. Option IDs are opaque. The answer key survives a server restart or bank update inside the saved server-side paper. Question and answer APIs enforce the lock. A unique active-owner index prevents parallel papers and the paper version protects grading and progression from simultaneous writes. Leased background jobs let the browser reload while scores and recommendations are being prepared. Resume/retry can reclaim work after an interrupted lease expires.

Newly queued papers persist their stage, allocation targets, and repeat policy before preparation begins. A cached score snapshot is reused only when its exact application-answer fingerprint matches the current history, in addition to the checkpoint, mapping, thresholds, and session-window checks.

The original Junyi history is a private read-only input. A real submission appends one audited, timestamped application answer to its paper, including the actual grade and reversible reference to a representative Junyi question if a valid mapping exists. Original history stays separate from the app history and legacy practice/export. A paper boundary does not create a model session. The adapter uses the original 10-hour inactivity rule, up to 15 complete earlier sessions, and the truly observed prefix for the current session. The completed paper triggers a new score snapshot for the next paper.

## How Gemini and study resources work

FastAPI builds an eligible candidate list from the verified bank and the saved skill snapshot. The Gemini request contains short references, skill tags, stages, topics, and slot limits. The schema bounds the number of references per group. The backend maps eligible replies back to bank IDs, rejects duplicates and invalid references, and fills any gaps using its existing allocation policy. The question text, grading key, learner account, answer sequence and individual score estimates stay in Lumen.

Study lookup matches the bank topic against a catalogue of approved English mathematics resources. Lumen checks public DNS destinations, the HTTP response and the page's educational relevance before showing a link. When several pages pass, Gemini can order their public titles for the topic and stage. A single verified lesson skips the Gemini request. A rate limit or unavailable model leaves the verified catalogue order usable. Lumen makes one provider attempt per ranking or selection and uses a brief cooldown after HTTP 429. Google Search grounding results are excluded from the persistent study checklist under the [Gemini API grounding terms](https://aistudio.google.com/docs/terms#grounding-with-google-search).

Optional lookups recheck links once their recorded verification is six hours old. Required study links are saved with their paper. When no relevant page can be verified for a new paper, the skill stays pending and the student can retry. The authorized operator can resolve it through the audited exception described above. The catalogue covers specific topics. Add and independently verify a suitable publisher page to extend it.

## What the estimates mean

The backend uses the original HiTSKT collator and forward pass on known model IDs. Each original tag uses up to five distinct training-ranked representative questions from its verified composite skills. The tag estimate averages those probabilities. Initial forecasts use independent BOS probes. Later forecasts can use app correctness via a labelled proxy, where the representative supplies model IDs and the actual answer supplies correctness. The model cannot directly see the custom question text. A missing proxy within a required context surfaces as an unavailable estimate while the real graded answer stays recorded.

CPU serving caps each batch at 8,192 padded tokens. When candidates share a history and shifted correctness prefix, the backend runs the original history and correctness encoders once, then uses the original decoder for each candidate batch. Direct full-forward checks validate parity with the existing checkpoint. A full single context beyond the serving budget is reported as unavailable. The browser receives a preparing state while the model job works through its batches.

Paper marks are observed grades. Topic accuracy is observed correct answers divided by observed attempts. The HiTSKT probability is an estimate over representative Junyi items. The 0.61 and 0.86 thresholds were evaluated on Junyi composite skills and are a prototype policy when applied to original-tag means and app proxies. The report shows separate original-history and proxy-inclusive scores. No BKT estimate is used for the Junyi assessment. Standard focused mathematics practice remains a separate learning activity.

## Checks

Run `python -m pytest backend/tests -q` with local MongoDB. For the browser fixture, use the backend environment and `PYTHONPATH=backend` to run `backend/tests/junyi_browser_app.py`, then start `npm run dev -- --port 3107` in `frontend` with `BACKEND_URL=http://127.0.0.1:8087`. Run `npx playwright test tests/junyi.spec.ts` with `LUMEN_JUNYI_TEST_URL=http://localhost:3107` and a locally installed Playwright browser channel. The fixture uses a disposable MongoDB database, distinct accounts, and clearly labelled test questions. See the [verification record](../../reports/adaptive_junyi_student1/TEST_RESULTS.md) for evidence and pending live work.

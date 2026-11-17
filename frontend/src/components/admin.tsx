"use client";
import { useEffect, useState, type FormEvent } from "react";
import { Download, FlaskConical, Plus } from "lucide-react";
import { api, send, percent } from "@/lib/api";
import {
  ActionStatus,
  Badge,
  Boundary,
  Button,
  Confirm,
  Empty,
  Field,
  Heading,
  Notice,
  Panel,
  Stat,
  Table,
  label,
  topicName,
  useAction,
  useLoad,
  when,
} from "./product";

type Taxon = {
  id: string;
  name: string;
  kind: "topic" | "skill";
  topic_id: string;
  concept_id: string;
  prerequisite?: string;
};
type Draft = {
  id: string;
  topic_id: string;
  skill_id: string;
  difficulty: number;
  question_type: "mcq" | "numeric";
  prompt: string;
  choices: string[];
  answer_index: number | null;
  numeric_answer: string;
  explanation: string;
  hint: string;
  practice_allowed: boolean;
};
type Item = Draft & {
  status: string;
  version: number;
  origin?: string;
  has_published_version?: boolean;
};
type Catalog = { questions: Item[]; total: number; page: number };
type Interaction = {
  student_id: string;
  session_id: string;
  question_id: string;
  question_version: string;
  skill_id: string;
  topic_id: string;
  response: string;
  correct: boolean;
  response_time_ms: number;
  attempt_timestamp: string;
  hints_used: number;
  mode: string;
  source: string;
  validation: string;
};
type InteractionData = {
  students: number;
  sessions: number;
  attempts: number;
  correct: number;
  incorrect: number;
  invalid: number;
  demo: boolean;
  rows: Interaction[];
  collection_status: string;
};
type Bench = {
  dataset: string;
  model: string;
  test_auc: string;
  test_accuracy: string;
  test_precision: string;
  test_recall: string;
  best_epoch: string;
};
type Lab = {
  bkt: { status: string; version: string };
  hitskt: {
    status: string;
    saved_fixture: boolean;
    checkpoint_present: boolean;
    message: string;
  };
  evaluation: {
    source: string;
    rows: Bench[];
    evaluation_date: string | null;
    live_pilot: boolean;
    audit: {
      dataset: string;
      cpu_auc: number;
      stored_auc: number;
      auc_gate: string;
      metrics_gate: string;
      cuda_replay: string;
    }[];
  };
};
type Replay = {
  source: string;
  rows: {
    session: number;
    question_id: string;
    skill_id: string;
    correct: boolean;
  }[];
  next: { session: number; question_id: string; skill_id: string };
};
type Prediction = {
  probability: number;
  status: string;
  checkpoint?: string;
  source: string;
  reveal_key: string;
};
const blank: Draft = {
  id: "",
  topic_id: "",
  skill_id: "",
  difficulty: 1,
  question_type: "mcq",
  prompt: "",
  choices: ["", "", "", ""],
  answer_index: null,
  numeric_answer: "",
  explanation: "",
  hint: "",
  practice_allowed: true,
};
const level = (s: number) =>
  ["", "Foundation", "Building", "Stretch"][s] || "—";
const yes = (str: string) =>
  str === "published" ? "sage" : str === "archived" ? "neutral" : "amber";

function Editor({
  item,
  onBack,
  onSave,
  taxa,
}: {
  item: Item | null;
  onBack: () => void;
  onSave: () => void;
  taxa: Taxon[];
}) {
  const [mode, setMode] = useState<"details" | "preview">("details"),
    [intent, setIntent] = useState<"archive" | "delete" | null>(null),
    [d, setD] = useState<Draft>(() =>
      item
        ? (Object.fromEntries(
            Object.keys(blank).map((k) => [
              k,
              item[k as keyof Draft] ?? blank[k as keyof Draft],
            ]),
          ) as Draft)
        : { ...blank },
    ),
    [id, setId] = useState(item?.id ?? ""),
    [revision, setRevision] = useState(item?.version ?? 0);
  const a = useAction();
  const change = <K extends keyof Draft>(k: K, v: Draft[K]) =>
    setD((x) => ({ ...x, [k]: v }));
  const skill = taxa.find((x) => x.id === d.skill_id);
  async function persist(publish = false) {
    await a.run(
      async () => {
        const body = {
          ...d,
          answer_index: d.question_type === "numeric" ? null : d.answer_index,
          choices: d.question_type === "numeric" ? [] : d.choices,
        };
        const v = await send<{ id: string }>(
          id ? `/admin/questions/${id}/edit` : "/admin/questions",
          body,
        );
        setId(v.id);
        if (publish) {
          const p = await send<{ version: number }>(
            `/admin/questions/${v.id}/publish`,
          );
          setRevision(p.version);
        }
        onSave();
      },
      publish ? "Question published" : "Draft saved",
    );
  }
  return (
    <>
      <Heading
        title={item ? `Question · ${item.id}` : "Add a question"}
        subtitle="Mathematics · Save a draft, preview it, then publish an eligible version."
        action={
          <Button variant="secondary" onClick={onBack}>
            Back to question bank
          </Button>
        }
      />
      <ActionStatus a={a} />
      <div className="p-actions">
        <Button
          variant={mode === "details" ? "primary" : "secondary"}
          onClick={() => setMode("details")}
        >
          Edit details
        </Button>
        <Button
          variant={mode === "preview" ? "primary" : "secondary"}
          onClick={() => setMode("preview")}
        >
          Preview question
        </Button>
        {item && (
          <Badge tone={yes(item.status)}>
            {label(item.status)} {item.version ? `v${item.version}` : ""}
          </Badge>
        )}
      </div>
      {mode === "details" ? (
        <Panel
          title="Question content"
          subtitle="Choose a topic and mapped skill. Only published items can be assigned."
        >
          <form
            className="p-stack"
            onSubmit={(e) => {
              e.preventDefault();
              persist();
            }}
          >
            <div className="p-fields">
              <Field label="Unique question ID">
                <input
                  value={d.id}
                  readOnly={!!id}
                  required
                  pattern="[A-Za-z0-9][A-Za-z0-9_-]{0,79}"
                  maxLength={80}
                  placeholder="FRA-018"
                  onChange={(e) => change("id", e.target.value)}
                />
              </Field>
              <Field label="Subject">
                <input value="Mathematics" readOnly />
              </Field>
              <Field label="Topic">
                <select
                  value={d.topic_id}
                  onChange={(e) =>
                    setD((x) => ({
                      ...x,
                      topic_id: e.target.value,
                      skill_id: "",
                    }))
                  }
                >
                  <option value="">Choose a topic</option>
                  {taxa
                    .filter((x) => x.kind === "topic")
                    .map((x) => (
                      <option key={x.id} value={x.id}>
                        {x.name}
                      </option>
                    ))}
                </select>
              </Field>
              <Field label="Skill mapping">
                <select
                  value={d.skill_id}
                  onChange={(e) => change("skill_id", e.target.value)}
                >
                  <option value="">Choose a skill</option>
                  {taxa
                    .filter(
                      (x) => x.kind === "skill" && x.topic_id === d.topic_id,
                    )
                    .map((x) => (
                      <option key={x.id} value={x.id}>
                        {x.name}
                      </option>
                    ))}
                </select>
              </Field>
              <Field label="Difficulty">
                <select
                  value={d.difficulty}
                  onChange={(e) => change("difficulty", +e.target.value)}
                >
                  {[1, 2, 3].map((x) => (
                    <option key={x} value={x}>
                      {level(x)}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Question type">
                <select
                  value={d.question_type}
                  onChange={(e) =>
                    change(
                      "question_type",
                      e.target.value as Draft["question_type"],
                    )
                  }
                >
                  <option value="mcq">Multiple choice</option>
                  <option value="numeric">Numeric / fraction</option>
                </select>
              </Field>
            </div>
            <Field label="Question prompt">
              <textarea
                rows={3}
                value={d.prompt}
                maxLength={1500}
                onChange={(e) => change("prompt", e.target.value)}
                placeholder="What is 1/2 + 1/3?"
              />
            </Field>
            {d.question_type === "mcq" ? (
              <>
                <div className="p-fields">
                  {[0, 1, 2, 3].map((i) => (
                    <Field
                      key={i}
                      label={`Option ${String.fromCharCode(65 + i)}`}
                    >
                      <input
                        value={d.choices[i] ?? ""}
                        maxLength={300}
                        onChange={(e) =>
                          change(
                            "choices",
                            Array.from({ length: 4 }, (_, j) =>
                              i === j ? e.target.value : (d.choices[j] ?? ""),
                            ),
                          )
                        }
                      />
                    </Field>
                  ))}
                </div>
                <Field label="Correct option">
                  <select
                    value={d.answer_index ?? ""}
                    onChange={(e) =>
                      change(
                        "answer_index",
                        e.target.value === "" ? null : +e.target.value,
                      )
                    }
                  >
                    <option value="">Choose an answer key</option>
                    {[0, 1, 2, 3].map((i) => (
                      <option value={i} key={i}>
                        {String.fromCharCode(65 + i)}
                      </option>
                    ))}
                  </select>
                </Field>
              </>
            ) : (
              <Field
                label="Correct numeric answer"
                hint="Use an exact number or fraction, e.g. 5/6."
              >
                <input
                  value={d.numeric_answer}
                  maxLength={100}
                  onChange={(e) => change("numeric_answer", e.target.value)}
                />
              </Field>
            )}
            <Field label="Explanation">
              <textarea
                rows={3}
                value={d.explanation}
                maxLength={1800}
                onChange={(e) => change("explanation", e.target.value)}
              />
            </Field>
            <Field label="Practice hint · optional">
              <input
                value={d.hint}
                maxLength={700}
                onChange={(e) => change("hint", e.target.value)}
              />
            </Field>
            <label className="p-check">
              <input
                type="checkbox"
                checked={d.practice_allowed}
                onChange={(e) => change("practice_allowed", e.target.checked)}
              />
              Allow this question in adaptive practice and diagnostics
            </label>
            {!skill && (
              <Notice title="Publish requires skill mapping">
                You can save the incomplete question as a draft.
              </Notice>
            )}
            <div className="p-actions">
              <Button type="submit" variant="secondary" disabled={a.busy}>
                Save draft
              </Button>
              <Button
                onClick={() => persist(true)}
                disabled={
                  a.busy || !d.id || !skill || !d.prompt || !d.explanation
                }
              >
                Save and publish
              </Button>
              {id && (
                <Button
                  onClick={() => setIntent(revision ? "archive" : "delete")}
                  variant="danger"
                  disabled={a.busy}
                >
                  {revision ? "Archive question" : "Delete unused draft"}
                </Button>
              )}
            </div>
          </form>
        </Panel>
      ) : (
        <Panel
          title={d.prompt || "No question prompt yet"}
          subtitle={`${skill?.name ?? "Unmapped"} · ${level(d.difficulty)} · ${d.question_type === "numeric" ? "Numeric" : "Multiple choice"}`}
        >
          <div className="p-list">
            {d.question_type === "numeric" ? (
              <div>
                <strong>Answer key</strong>
                <Badge>{d.numeric_answer || "Not set"}</Badge>
              </div>
            ) : (
              d.choices.map((x, i) => (
                <div key={i}>
                  <span>
                    {String.fromCharCode(65 + i)} · {x || "No option text"}
                  </span>
                  {i === d.answer_index && <Badge>Correct answer</Badge>}
                </div>
              ))
            )}
          </div>
          <div className="p-example">
            <strong>Explanation</strong>
            <p>{d.explanation || "An explanation is required to publish."}</p>
          </div>
        </Panel>
      )}
      {intent && (
        <Confirm
          title={
            intent === "archive"
              ? "Archive this question?"
              : "Delete this unused draft?"
          }
          action={intent === "archive" ? "Archive question" : "Delete draft"}
          danger
          onClose={() => setIntent(null)}
          busy={a.busy}
          onConfirm={async () => {
            await a.run(
              async () => {
                if (intent === "archive")
                  await send(`/admin/questions/${id}/archive`);
                else await api(`/admin/questions/${id}`, { method: "DELETE" });
                setIntent(null);
                onSave();
              },
              intent === "archive" ? "Question archived" : "Draft deleted",
            );
          }}
        >
          <p>
            {intent === "archive"
              ? "The question won’t appear in new sessions or papers. Existing assignments and learner history keep their version."
              : "Only unused drafts may be deleted. This cannot be undone."}
          </p>
          <ActionStatus a={a} />
        </Confirm>
      )}
    </>
  );
}

function Skills({ tax }: { tax: ReturnType<typeof useLoad<Taxon[]>> }) {
  const a = useAction(),
    [kind, setKind] = useState<"topic" | "skill">("skill"),
    [topic, setTopic] = useState("fractions"),
    [edit, setEdit] = useState<Taxon | null>(null);
  const topics = tax.data?.filter((t) => t.kind === "topic") ?? [];
  const skills = tax.data?.filter((t) => t.kind === "skill") ?? [];
  async function save(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget;
    const vals = Object.fromEntries(new FormData(form));
    const v = await a.run(
      () =>
        send(
          edit ? `/admin/taxonomy/${edit.id}/rename` : "/admin/taxonomy",
          edit
            ? { ...edit, name: vals.name }
            : { ...vals, kind, topic_id: kind === "skill" ? topic : "" },
        ),
      edit ? "Label updated" : "Added to taxonomy",
    );
    if (v) {
      setEdit(null);
      form.reset();
      tax.reload();
    }
  }
  return (
    <>
      <Heading
        title="Topics & skills"
        subtitle="Mathematics → Topic → Skill → Questions"
      />
      <ActionStatus a={a} />
      <Boundary value={tax}>
        <div className="p-grid wide-right">
          <Panel
            title="Learning hierarchy"
            subtitle="Stable IDs preserve historical mapping."
          >
            <ul className="p-tree">
              {topics.flatMap((t) => [
                <li key={t.id}>
                  {t.name}
                  <small>{t.id}</small>
                </li>,
                ...skills
                  .filter((s) => s.topic_id === t.id)
                  .map((s) => (
                    <li key={s.id} className="skill">
                      <button className="p-btn link" onClick={() => setEdit(s)}>
                        {s.name} →
                      </button>
                      <small>{s.id}</small>
                    </li>
                  )),
              ])}
            </ul>
          </Panel>
          <Panel
            title={edit ? "Rename skill" : "Add topic or skill"}
            subtitle={
              edit
                ? "Built-in labels and IDs are stable. Custom labels can be updated."
                : "New content links to one of the four mathematics BKT foundations."
            }
          >
            <form key={edit?.id ?? kind} className="p-stack" onSubmit={save}>
              {!edit && (
                <Field label="Type">
                  <select
                    value={kind}
                    onChange={(e) => setKind(e.target.value as typeof kind)}
                  >
                    <option value="skill">Skill</option>
                    <option value="topic">Topic</option>
                  </select>
                </Field>
              )}
              <div className="p-fields">
                <Field label="Identifier">
                  <input
                    name="id"
                    readOnly={!!edit}
                    defaultValue={edit?.id ?? ""}
                    pattern="[A-Za-z0-9][A-Za-z0-9_-]{0,79}"
                    required
                    placeholder="fractions-unlike-add"
                  />
                </Field>
                <Field label="Display name">
                  <input
                    name="name"
                    defaultValue={edit?.name ?? ""}
                    required
                    minLength={2}
                    maxLength={100}
                    placeholder="Adding unlike fractions"
                  />
                </Field>
              </div>
              {!edit && kind === "skill" && (
                <>
                  <Field label="Parent topic">
                    <select
                      value={topic}
                      onChange={(e) => setTopic(e.target.value)}
                    >
                      {topics.map((t) => (
                        <option key={t.id} value={t.id}>
                          {t.name}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Prerequisite · optional">
                    <select name="prerequisite">
                      <option value="">None</option>
                      {skills.map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.name}
                        </option>
                      ))}
                    </select>
                  </Field>
                </>
              )}
              {!edit && kind === "topic" && (
                <Field label="BKT foundation">
                  <select name="concept_id">
                    {["fractions", "percentages", "equations", "geometry"].map(
                      (c) => (
                        <option key={c} value={c}>
                          {topicName(c)}
                        </option>
                      ),
                    )}
                  </select>
                </Field>
              )}
              <div className="p-actions">
                <Button type="submit" disabled={a.busy}>
                  {edit ? "Save label" : "Add to hierarchy"}
                </Button>
                {edit && (
                  <Button variant="secondary" onClick={() => setEdit(null)}>
                    Cancel
                  </Button>
                )}
              </div>
            </form>
          </Panel>
        </div>
      </Boundary>
    </>
  );
}

function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function Interactions() {
  const [filter, setFilter] = useState("all"),
    [view, setView] = useState<Interaction | null>(null),
    [confirm, setConfirm] = useState(false),
    [days, setDays] = useState(30);
  const a = useAction(),
    v = useLoad<InteractionData>(`/admin/interactions?source=${filter}`);
  async function exp() {
    await a.run(async () => {
      const r = await fetch(`/api/admin/interactions/export?days=${days}`, {
        method: "POST",
        credentials: "same-origin",
      });
      if (!r.ok)
        throw Error(
          "The export could not be prepared. Check access and try again.",
        );
      download(
        await r.blob(),
        v.data?.demo
          ? "lumen-synthetic-interactions.csv"
          : "lumen-interactions.csv",
      );
      setConfirm(false);
    }, "CSV export prepared");
  }
  async function schema() {
    await a.run(async () => {
      const data = await api("/admin/interactions/schema");
      download(
        new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
        "lumen-interaction-schema.json",
      );
    }, "Schema downloaded");
  }
  if (view)
    return (
      <>
        <Heading
          title="Interaction detail"
          subtitle="Pseudonymous identifiers are scoped to this snapshot."
          action={
            <Button variant="secondary" onClick={() => setView(null)}>
              Back to interactions
            </Button>
          }
        />
        <Table
          headers={["Field", "Value"]}
          rows={Object.entries(view).map(([k, value]) => [
            label(k),
            value == null
              ? "—"
              : typeof value === "boolean"
                ? value
                  ? "Yes"
                  : "No"
                : String(value),
          ])}
        />
      </>
    );
  return (
    <>
      <Heading
        title="Learning interactions"
        subtitle="Inspect data collection and validation before exporting approved response records."
        action={
          <Button onClick={() => setConfirm(true)} disabled={!v.data?.attempts}>
            <Download size={17} /> Review export
          </Button>
        }
      />
      <ActionStatus a={a} />
      <Boundary value={v}>
        {v.data && (
          <>
            <div className="p-stats">
              <Stat title="Registered students" value={v.data.students} />
              <Stat title="Learning sessions" value={v.data.sessions} />
              <Stat
                title="Question attempts"
                value={v.data.attempts}
                caption={`${v.data.correct} correct · ${v.data.incorrect} incorrect`}
              />
              <Stat
                title="Needs review"
                value={v.data.invalid}
                caption="Invalid records excluded from export"
              />
            </div>
            <Notice
              title={v.data.demo ? "Synthetic demo data" : "Collection active"}
              tone="ok"
            >
              {v.data.demo
                ? "Fictional interactions only. These records are for validation and demonstration."
                : "Only live learner evidence in this workspace. Names, credentials and emails are excluded from downloads."}
            </Notice>
            <div className="p-toolbar">
              <Field label="Validation filter">
                <select
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                >
                  <option value="all">All interactions</option>
                  <option value="invalid">Needs review</option>
                </select>
              </Field>
              <Button variant="secondary" onClick={schema}>
                Download schema
              </Button>
            </div>
            {v.data.rows.length ? (
              <Table
                headers={[
                  "Learner",
                  "Session",
                  "Question",
                  "Outcome",
                  "Validation",
                  "Recorded",
                  "Action",
                ]}
                rows={v.data.rows.map((x) => [
                  x.student_id.slice(0, 10),
                  x.session_id.slice(0, 10),
                  x.question_id,
                  x.correct ? "Correct" : "Incorrect",
                  <Badge
                    key="v"
                    tone={x.validation === "Valid" ? "sage" : "amber"}
                  >
                    {x.validation}
                  </Badge>,
                  when(x.attempt_timestamp),
                  <Button
                    key="r"
                    variant="secondary"
                    onClick={() => setView(x)}
                  >
                    Inspect
                  </Button>,
                ])}
                caption={`${v.data.rows.length} most recent matching records. Pseudonyms rotate per snapshot.`}
              />
            ) : (
              <Empty
                title={
                  filter === "invalid"
                    ? "No invalid records found"
                    : "No interactions collected yet"
                }
              >
                Attempts appear after learners submit work.
              </Empty>
            )}
          </>
        )}
      </Boundary>
      {confirm && (
        <Confirm
          title="Export learning interactions?"
          action="Confirm export"
          onClose={() => setConfirm(false)}
          onConfirm={exp}
          busy={a.busy}
        >
          <div className="p-stack">
            <p>
              Include valid attempts and approved fields only. Aliases are
              randomized for this export; names, emails and credentials are
              omitted.
            </p>
            <Field label="Date range">
              <select value={days} onChange={(e) => setDays(+e.target.value)}>
                {[7, 30, 90, 365].map((x) => (
                  <option key={x} value={x}>
                    Last {x} days
                  </option>
                ))}
              </select>
            </Field>
            <ActionStatus a={a} />
          </div>
        </Confirm>
      )}
    </>
  );
}

function ModelLab() {
  const lab = useLoad<Lab>("/admin/model"),
    a = useAction();
  const [tab, setTab] = useState<"overview" | "replay" | "evaluation">(
      "overview",
    ),
    [model, setModel] = useState("bkt"),
    [prediction, setPrediction] = useState<Prediction | null>(null),
    [outcome, setOutcome] = useState<{ correct: boolean; note: string } | null>(
      null,
    ),
    [dataset, setDataset] = useState(""),
    [ref, setRef] = useState("");
  const choices = useLoad<{ ref: string; label: string; attempts: number }[]>(
    tab === "replay" && model === "bkt" ? "/admin/model/replay-learners" : null,
  );
  const query = `?ref=${encodeURIComponent(ref)}`;
  const seq = useLoad<Replay>(
    tab === "replay" ? `/admin/model/sequence/${model}${query}` : null,
  );
  function select(value: string) {
    setModel(value);
    setRef("");
    setPrediction(null);
    setOutcome(null);
  }
  async function predict() {
    const v = await a.run(() =>
      send<Prediction>(`/admin/model/sequence/${model}/predict${query}`),
    );
    if (v) {
      setPrediction(v);
      setOutcome(null);
    }
  }
  async function reveal() {
    const v = await a.run(() =>
      api<{ correct: boolean; note: string }>(
        `/admin/model/sequence/${model}/outcome${query}&reveal_key=${prediction?.reveal_key ?? ""}`,
      ),
    );
    if (v) setOutcome(v);
  }
  const rows = lab.data?.evaluation.rows ?? [];
  return (
    <>
      <Heading
        title="Model Lab"
        subtitle="Inspect how prior responses inform a next-answer prediction. Research runs are separate from live student practice."
      />
      <div className="p-actions">
        {(["overview", "replay", "evaluation"] as const).map((t) => (
          <Button
            key={t}
            variant={tab === t ? "primary" : "secondary"}
            onClick={() => {
              setTab(t);
              if (t !== "replay") {
                setPrediction(null);
                setOutcome(null);
              }
            }}
          >
            {t === "replay" ? "Sequence replay" : label(t)}
          </Button>
        ))}
      </div>
      <ActionStatus a={a} />
      <Boundary value={lab}>
        {tab === "overview" && lab.data && (
          <>
            <div className="p-grid">
              <Panel>
                <Badge>Live mathematics</Badge>
                <h2>Bayesian knowledge tracing</h2>
                <p className="p-panel-sub">
                  {lab.data.bkt.version} · Uses submitted answers to guide
                  practice and revision.
                </p>
                <Button
                  onClick={() => {
                    select("bkt");
                    setTab("replay");
                  }}
                >
                  Try BKT replay
                </Button>
              </Panel>
              <Panel>
                <Badge tone="neutral">Research only</Badge>
                <h2>Performer HiTSKT</h2>
                <p className="p-panel-sub">
                  {lab.data.hitskt.checkpoint_present
                    ? "Local checkpoint available for verification."
                    : "Checkpoint not installed here. Saved fixture output may be available."}
                </p>
                <Button
                  variant="secondary"
                  disabled={!lab.data.hitskt.saved_fixture}
                  onClick={() => {
                    select("hitskt");
                    setTab("replay");
                  }}
                >
                  Open HiTSKT replay
                </Button>
              </Panel>
            </div>
            <Notice title="Use the right evidence for the claim" tone="ok">
              Benchmark metrics rank next-answer predictions on held-out
              datasets. Lumen has no classroom pilot result. The original maths
              bank does not use benchmark question IDs.
            </Notice>
          </>
        )}
        {tab === "replay" && (
          <>
            <div className="p-toolbar">
              <Field label="Model">
                <select value={model} onChange={(e) => select(e.target.value)}>
                  <option value="bkt">Bayesian knowledge tracing</option>
                  <option value="hitskt">Performer HiTSKT</option>
                </select>
              </Field>
              <Field label="Learner or fixture">
                <select
                  value={ref}
                  onChange={(e) => {
                    setRef(e.target.value);
                    setPrediction(null);
                    setOutcome(null);
                  }}
                >
                  <option value="">
                    {model === "hitskt"
                      ? "EdNet-KT1 synthetic parity fixture"
                      : "Lumen synthetic response fixture"}
                  </option>
                  {choices.data?.map((x) => (
                    <option key={x.ref} value={x.ref}>
                      {x.label} · {x.attempts} attempts
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            <Notice title="Synthetic sequence" tone="ok">
              No approved real-student replay has been loaded. The example
              labels are fictional and the next answer is hidden before
              prediction.
            </Notice>
            <Boundary value={seq}>
              {seq.data && (
                <>
                  <Table
                    headers={[
                      "Session",
                      "Question ID",
                      "Skill ID",
                      "Observed outcome",
                    ]}
                    rows={[
                      ...seq.data.rows.map((x) => [
                        `Session ${x.session}`,
                        x.question_id,
                        x.skill_id,
                        x.correct
                          ? "Correct · synthetic"
                          : "Incorrect · synthetic",
                      ]),
                      [
                        "Next answer",
                        seq.data.next.question_id,
                        seq.data.next.skill_id,
                        outcome
                          ? `${outcome.correct ? "Correct" : "Incorrect"} · synthetic target label`
                          : "Hidden before reveal",
                      ],
                    ]}
                    caption={seq.data.source}
                  />
                  <Panel className="p-replay-result">
                    <div>
                      <strong>
                        {prediction
                          ? `${(prediction.probability * 100).toFixed(2)}%`
                          : "—"}
                      </strong>
                      <p>Predicted probability of a correct answer</p>
                    </div>
                    <div>
                      <p>
                        {prediction
                          ? prediction.status ===
                            "computed_from_verified_checkpoint"
                            ? "Computed now from the verified EdNet checkpoint."
                            : prediction.status ===
                                "saved_verified_checkpoint_output"
                              ? "Saved verified checkpoint output; no local inference in this run."
                              : "Computed now using the displayed BKT history."
                          : "Run the model using only earlier responses."}
                      </p>
                      {prediction?.checkpoint && (
                        <p className="p-faint">
                          Checkpoint {prediction.checkpoint.slice(0, 12)}…
                        </p>
                      )}
                      {outcome && <p className="p-faint">{outcome.note}</p>}
                      <div className="p-actions">
                        <Button disabled={a.busy} onClick={predict}>
                          <FlaskConical size={17} />
                          {a.busy ? "Processing…" : "Run prediction"}
                        </Button>
                        <Button
                          disabled={!prediction || !!outcome || a.busy}
                          onClick={reveal}
                          variant="secondary"
                        >
                          Reveal target
                        </Button>
                      </div>
                    </div>
                  </Panel>
                </>
              )}
            </Boundary>
          </>
        )}
        {tab === "evaluation" && (
          <>
            <div className="p-toolbar">
              <Field label="Evaluation dataset">
                <select
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                >
                  <option value="">All benchmark datasets</option>
                  {Array.from(new Set(rows.map((x) => x.dataset))).map((x) => (
                    <option value={x} key={x}>
                      {label(x)}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            {rows.length ? (
              <Table
                headers={[
                  "Dataset",
                  "Model",
                  "ROC-AUC",
                  "Accuracy",
                  "Precision",
                  "Recall",
                  "Best epoch",
                ]}
                rows={rows
                  .filter((x) => !dataset || x.dataset === dataset)
                  .map((x) => [
                    label(x.dataset),
                    x.model,
                    (+x.test_auc).toFixed(3),
                    percent(+x.test_accuracy),
                    percent(+x.test_precision),
                    percent(+x.test_recall),
                    x.best_epoch,
                  ])}
                caption="Repository held-out benchmark. Evaluation date not recorded in the aggregate report."
              />
            ) : (
              <Empty title="No benchmark report loaded">
                Add the verified repository report to display scores.
              </Empty>
            )}
            <Notice title="No Lumen classroom evaluation yet">
              The benchmarks do not demonstrate learning gains or validate
              mastery estimates.
            </Notice>
          </>
        )}
        {tab === "evaluation" && Boolean(lab.data?.evaluation.audit.length) && (
          <Panel
            title="Independent HiTSKT replay"
            subtitle="AUC agreement and full-metric reproduction are different checks."
          >
            <Table
              headers={[
                "Dataset",
                "Historical CUDA AUC",
                "Independent CPU AUC",
                "AUC tolerance",
                "Full metric gate",
                "Fresh CUDA replay",
              ]}
              rows={lab
                .data!.evaluation.audit.filter(
                  (x) => !dataset || x.dataset === dataset,
                )
                .map((x) => [
                  label(x.dataset),
                  x.stored_auc.toFixed(9),
                  x.cpu_auc.toFixed(9),
                  x.auc_gate,
                  x.metrics_gate,
                  x.cuda_replay,
                ])}
            />
            <div className="p-example">
              The CPU AUCs match within tolerance; the complete metric sets do
              not. Response timing and fresh CUDA replay remain unverified.
            </div>
          </Panel>
        )}
      </Boundary>
    </>
  );
}

export function AdminWorkspace({ tab }: { tab: string }) {
  const tax = useLoad<Taxon[]>("/content/taxonomy"),
    [search, setSearch] = useState(""),
    [topic, setTopic] = useState(""),
    [status, setStatus] = useState(""),
    [page, setPage] = useState(1),
    [edit, setEdit] = useState<Item | null | undefined>(undefined);
  const list = useLoad<Catalog>(
    tab === "questions"
      ? `/admin/questions?${new URLSearchParams({ search, topic, status, page: String(page) })}`
      : null,
  );
  useEffect(() => {
    setEdit(undefined);
  }, [tab]);
  if (tab === "skills") return <Skills tax={tax} />;
  if (tab === "interactions") return <Interactions />;
  if (tab === "lab") return <ModelLab />;
  if (edit !== undefined)
    return (
      <Editor
        item={edit}
        taxa={tax.data ?? []}
        onBack={() => setEdit(undefined)}
        onSave={() => {
          setEdit(undefined);
          list.reload();
        }}
      />
    );
  return (
    <>
      <Heading
        title="Question bank"
        subtitle="Manage mathematics content before it can be assigned or used in practice."
        action={
          <Button onClick={() => setEdit(null)}>
            <Plus size={17} /> Add question
          </Button>
        }
      />
      <div className="p-toolbar">
        <Field label="Search questions">
          <input
            placeholder="Question ID or text"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
          />
        </Field>
        <Field label="Topic">
          <select
            value={topic}
            onChange={(e) => {
              setTopic(e.target.value);
              setPage(1);
            }}
          >
            <option value="">All topics</option>
            {tax.data
              ?.filter((t) => t.kind === "topic")
              .map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
          </select>
        </Field>
        <Field label="Status">
          <select
            value={status}
            onChange={(e) => {
              setStatus(e.target.value);
              setPage(1);
            }}
          >
            <option value="">All statuses</option>
            {["draft", "published", "archived"].map((x) => (
              <option key={x} value={x}>
                {label(x)}
              </option>
            ))}
          </select>
        </Field>
      </div>
      <Boundary value={list}>
        {list.data?.questions.length ? (
          <>
            <Table
              headers={[
                "Question",
                "Topic / skill",
                "Difficulty",
                "Type",
                "Status",
                "Action",
              ]}
              rows={list.data.questions.map((x) => [
                <span key="q">
                  <strong>{x.id}</strong>
                  <br />
                  <span className="p-faint">
                    {x.prompt || "Incomplete draft"}
                  </span>
                </span>,
                `${tax.data?.find((t) => t.id === x.topic_id)?.name ?? topicName(x.topic_id)} · ${tax.data?.find((t) => t.id === x.skill_id)?.name ?? "Unmapped"}`,
                level(x.difficulty),
                x.question_type === "numeric" ? "Numeric" : "MCQ",
                <Badge key="badge" tone={yes(x.status)}>
                  {label(x.status)}
                  {x.status === "draft" && !x.skill_id ? " · unmapped" : ""}
                </Badge>,
                <Button
                  key="edit"
                  variant="secondary"
                  onClick={() => setEdit(x)}
                >
                  Preview / edit
                </Button>,
              ])}
            />
            <div className="p-pagination">
              <Button
                variant="secondary"
                onClick={() => setPage(page - 1)}
                disabled={page <= 1}
              >
                Previous
              </Button>
              <span>
                Page {page} of {Math.ceil(list.data.total / 20)} ·{" "}
                {list.data.total} questions
              </span>
              <Button
                variant="secondary"
                onClick={() => setPage(page + 1)}
                disabled={page * 20 >= list.data.total}
              >
                Next
              </Button>
            </div>
          </>
        ) : (
          <Empty
            title="No matching questions"
            action={
              <Button
                variant="secondary"
                onClick={() => {
                  setStatus("");
                  setTopic("");
                  setSearch("");
                  setPage(1);
                }}
              >
                Clear filters
              </Button>
            }
          >
            Change your search or filters to find more questions.
          </Empty>
        )}
      </Boundary>
    </>
  );
}

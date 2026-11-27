"use client";

import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight, BookOpen, Check, ExternalLink, ShieldCheck } from "lucide-react";
import { api, AuthError, percent, RequestError, send } from "@/lib/api";
import { paperUrl, type JunyiAdmin, type JunyiAssigned, type JunyiPaper, type JunyiRequirement, type JunyiResource, type JunyiSkill, type StudyLookup } from "@/lib/junyi";
import { ActionStatus, Badge, Brand, Button, Empty, Field, Heading, Loading, Notice, Panel, Table, useAction, useLoad, when } from "@/components/product";

function stageName(stage: string | null) {
  return stage ? stage[0].toUpperCase() + stage.slice(1) : "All stages";
}

function formatScore(score: number | null) {
  return score == null ? "Unavailable" : percent(score);
}

function SkillStatus({ skill }: { skill: JunyiSkill }) {
  if (skill.score == null) return <span className="junyi-metric"><Badge tone="neutral">Unavailable</Badge>
    {skill.unattempted && <Badge tone="neutral">Unattempted</Badge>}
  </span>;
  if (skill.unattempted) return <Badge tone="neutral">Unattempted</Badge>;
  if (skill.category === "weak") return <Badge tone="amber">Weak</Badge>;
  if (skill.category === "strong") return <Badge>Strong</Badge>;
  return <Badge tone="neutral">Intermediate</Badge>;
}

function SkillTable({ skills, answersUsed, comparison = false }: { skills: JunyiSkill[]; answersUsed: number | null; comparison?: boolean }) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [page, setPage] = useState(0);
  const filtered = skills.filter((skill) => skill.topic.toLowerCase().includes(query.trim().toLowerCase()) &&
    (category === "all" || category === "unattempted" && skill.unattempted || category === "unavailable" && skill.score == null ||
     category === "lumen" && skill.app_attempts > 0 || category === "weak" && skill.category === "weak" && !skill.unattempted));
  const visible = filtered.slice(page * 12, page * 12 + 12);
  return (
    <Panel title={comparison ? "Skill estimates after this paper" : "Skill report"} className="junyi-skill-panel">
      {answersUsed != null && <p className="p-muted" data-testid="junyi-evidence-count">{answersUsed} saved Lumen assessment {answersUsed === 1 ? "answer" : "answers"} included in this report.</p>}
      <p className="p-muted">Observed accuracy is correct answers divided by recorded attempts for a skill. HiTSKT estimates the chance of a correct answer on representative Junyi questions.</p>
      {comparison && <p className="p-muted">“With Lumen answers” uses the actual outcome of each eligible answer and the ID of a representative Junyi question for its skill.</p>}
      <div className="p-toolbar">
        <Field label="Find a skill"><input value={query} onChange={(e) => { setQuery(e.target.value); setPage(0); }} /></Field>
        <Field label="Skill status">
          <select value={category} onChange={(e) => { setCategory(e.target.value); setPage(0); }}>
            <option value="all">All skills</option><option value="lumen">Practised in Lumen</option><option value="weak">Weak</option>
            <option value="unattempted">Unattempted</option><option value="unavailable">Unavailable</option>
          </select>
        </Field>
      </div>
      {visible.length ? <div className="junyi-table-scroll"><Table
        headers={["Skill", "Status", "Observed accuracy", ...(comparison ? ["Original history", "With Lumen answers"] : ["HiTSKT estimate"])]}
        rows={visible.map((skill) => [
          <span key="topic"><strong>{skill.topic}</strong><br /><small>{stageName(skill.stage)}</small></span>,
          <SkillStatus key="status" skill={skill} />,
          <span className="junyi-metric" key="observed">
            <strong>{skill.actual_rate == null ? "No answers yet" : `${percent(skill.actual_rate)} (${skill.original_correct + skill.app_correct}/${skill.original_attempts + skill.app_attempts})`}</strong>
            {skill.original_attempts > 0 && <small>Original Junyi {skill.original_correct}/{skill.original_attempts}</small>}
            {skill.app_attempts > 0 && <small>Lumen {skill.app_correct}/{skill.app_attempts}</small>}
          </span>,
          ...(comparison ? [formatScore(skill.original_score)] : []),
          <span className="junyi-metric" key="estimate">{formatScore(skill.score)}
            {skill.unavailable_reason && <small>{skill.unavailable_reason}</small>}
          </span>,
        ])}
      /></div> : <p className="p-muted">No skills match these filters.</p>}
      {filtered.length > 12 && <div className="p-pagination">
        <Button variant="secondary" disabled={!page} onClick={() => setPage(page - 1)}>Previous</Button>
        <span aria-live="polite">{page * 12 + 1}–{Math.min((page + 1) * 12, filtered.length)} of {filtered.length}</span>
        <Button variant="secondary" disabled={(page + 1) * 12 >= filtered.length} onClick={() => setPage(page + 1)}>Next</Button>
      </div>}
    </Panel>
  );
}

function ResourceLink({ link }: { link: JunyiResource }) {
  return <a className="p-text-link junyi-source" href={link.url} target="_blank" rel="noopener noreferrer">{link.title}<ExternalLink size={16} aria-label="Opens in a new tab" /></a>;
}

function StudyMaterial({ title, summary, links, action }: { title: string; summary?: ReactNode; links?: ReactNode; action?: ReactNode }) {
  return <Panel className="junyi-study-card">
    <div className="junyi-study-summary"><h2>{title}</h2>{summary}</div>
    <div className="junyi-study-resources">{links}</div>
    <div className="junyi-study-action">{action}</div>
  </Panel>;
}

function OptionalStudy({ skill }: { skill: JunyiSkill }) {
  const action = useAction();
  const [result, setResult] = useState<StudyLookup | null>(null);
  async function find() {
    const value = await action.run(() => send<StudyLookup>(`/junyi/study/${skill.tag}`));
    if (value) setResult(value);
  }
  return <StudyMaterial title={skill.topic} summary={<>
    <p className="p-muted">{skill.original_attempts + skill.app_attempts} recorded answers · {stageName(skill.stage)}</p>
    <ActionStatus a={action} />
  </>} links={result?.resources.length ? result.resources.map((link) => <ResourceLink key={link.url} link={link} />) :
    result?.message && <p className="p-muted" role="status">{result.message}</p>}
    action={!result?.resources.length && <Button variant="secondary" onClick={find} disabled={action.busy}>{action.busy ? "Looking for resources…" : result ? "Retry resource search" : "Find study resources"}</Button>} />;
}

export function JunyiLearn({ dashboard }: { dashboard: JunyiAssigned }) {
  const [query, setQuery] = useState("");
  const lockedPaper = dashboard.active?.status === "locked" ? dashboard.active : null;
  const required = new Set(lockedPaper?.required_tags);
  const optional = dashboard.skills.filter((row) => row.category === "weak" && !row.unattempted && !required.has(row.tag));
  const needsStudy = optional.filter((row) => row.topic.toLowerCase().includes(query.trim().toLowerCase()));
  return <section className="junyi-section">
    <Heading title="Study recommendations" subtitle="Open a resource for a skill you want to strengthen." />
    {lockedPaper && <Notice title={`Required study for Test ${lockedPaper.sequence}`}>
      <Button href={paperUrl(lockedPaper.id)} variant="secondary">Open study checklist</Button>
    </Notice>}
    {optional.length > 0 && <div className="p-toolbar">
      <Field label="Find a recommended skill"><input value={query} onChange={(e) => setQuery(e.target.value)} /></Field>
    </div>}
    {needsStudy.length ? <div className="junyi-study-grid">{needsStudy.slice(0, 9).map((skill) => <OptionalStudy key={skill.tag} skill={skill} />)}</div> :
      <p className="p-muted">{query ? "No recommendations match this search." : dashboard.skills.length ? "No optional skills to study right now." : "Study recommendations appear after your assessment profile is ready."}</p>}
    {needsStudy.length > 9 && <p className="p-muted">Showing 9 of {needsStudy.length} skills. Search to find another topic.</p>}
  </section>;
}

export function JunyiWorkspace({ dashboard, name, tab }: { dashboard: JunyiAssigned; name: string; tab: string }) {
  const route = useRouter();
  const action = useAction();
  async function start() {
    const paper = await action.run(() => send<JunyiPaper>("/assessments"));
    if (paper) route.push(paperUrl(paper.id));
  }
  if (tab === "progress") return <>
    <Heading title="Learning report" subtitle="Review your paper results and the evidence behind your skill estimates." />
    {dashboard.history.length ? <Panel title="Completed assessments"><div className="junyi-table-scroll">
      <Table headers={["Assessment", "School stage", "Score", "Date"]} rows={dashboard.history.map((row) => [
        <Button key="paper" variant="link" href={paperUrl(row.id)}>Test {row.sequence} results <ArrowRight size={15} aria-hidden /></Button>,
        stageName(row.stage), `${row.correct} / ${row.total} (${percent(row.accuracy)})`, when(new Date(row.completed_at).toISOString()),
      ])} />
    </div></Panel> : <Empty title="No completed assessments">Your first completed paper will appear here.</Empty>}
    {dashboard.skills.length > 0 && <SkillTable skills={dashboard.skills} answersUsed={dashboard.answers_used} />}
  </>;
  const active = dashboard.active;
  let nextLabel = dashboard.history.length ? "Start next assessment" : "Start adaptive assessment";
  let nextDescription = "Get a paper matched to your current skill profile.";
  if (active?.status === "locked") {
    nextLabel = `Study for Test ${active.sequence}`;
    nextDescription = "Complete the study checklist to open your saved paper.";
  } else if (active?.status === "preparing") {
    nextLabel = `View Test ${active.sequence} preparation`;
    nextDescription = "Lumen is preparing your next paper from your saved answers.";
  } else if (active) {
    nextLabel = `Resume Test ${active.sequence}`;
    nextDescription = "Your latest assessment is ready to resume.";
  }
  return <>
    <Heading title={`Welcome, ${name.split(" ")[0]}`} subtitle="Junyi mathematics" />
    <section className="p-hero-action junyi-hero">
      <div>
        <Badge>{stageName(dashboard.stage)} · 20 questions</Badge>
        <h2>Junyi Adaptive Assessment</h2>
        <p>{nextDescription}</p>
      </div>
      {active ? <Button variant="light" href={paperUrl(active.id)}>{nextLabel} <ArrowRight size={17} aria-hidden /></Button> :
        <Button variant="light" onClick={start} disabled={!dashboard.ready || action.busy}>{action.busy ? "Starting…" : nextLabel} <ArrowRight size={17} aria-hidden /></Button>}
    </section>
    <ActionStatus a={action} />
    {!dashboard.ready && <Notice title="Assessment setup is incomplete">{dashboard.issues.join(" ")}</Notice>}
    {dashboard.history[0] && <Panel title={`Latest result · Test ${dashboard.history[0].sequence}`} className="junyi-latest">
      <strong>{dashboard.history[0].correct} / {dashboard.history[0].total} correct</strong>
      <Button variant="secondary" href={paperUrl(dashboard.history[0].id)}>Test {dashboard.history[0].sequence} results</Button>
    </Panel>}
  </>;
}

function StudyCard({ requirement, paperId, saved }: { requirement: JunyiRequirement; paperId: string; saved: (paper: JunyiPaper) => void }) {
  const action = useAction();
  const resourceGroupId = useId();
  const [chosenUrl, setChosenUrl] = useState<string>();
  const resources = requirement.resources;
  const selectedUrl = resources.find((link) => link.url === chosenUrl)?.url ?? resources[0]?.url;
  const path = `/assessments/${paperId}/study/${requirement.tag}`;
  async function update(endpoint: string, data?: unknown) {
    const next = await action.run(() => send<JunyiPaper>(`${path}/${endpoint}`, data));
    if (next) {
      saved(endpoint === "retry" ? {
        ...next, requirements: next.requirements.map((row) => row.tag === requirement.tag && row.can_retry
          ? { ...row, lookup: "loading", can_retry: false, message: null } : row),
      } : next);
    }
  }
  return <StudyMaterial title={requirement.topic} summary={<>
    <Badge tone={requirement.required ? "amber" : requirement.status === "pending" ? "neutral" : "sage"}>
      {requirement.status === "satisfied" ? "Complete" : requirement.status === "overridden" ? "Cleared by operator" : requirement.required ? "Required" : "Optional for now"}
    </Badge>
    <ActionStatus a={action} />
  </>} links={requirement.required && resources.length > 1 ? <fieldset className="junyi-study-choices">
    <legend>Resource to mark complete</legend>
    {resources.map((link) => <div className="junyi-study-choice" key={link.url}>
      <input type="radio" name={resourceGroupId} aria-label={link.title} checked={selectedUrl === link.url} onChange={() => setChosenUrl(link.url)} />
      <ResourceLink link={link} />
    </div>)}
  </fieldset> : resources.length ? resources.map((link) => <ResourceLink key={link.url} link={link} />) :
    requirement.status === "pending" && (requirement.lookup === "loading" || requirement.lookup === "waiting" ?
      <p className="p-muted" role="status">Checking study resources…</p> : <p className="p-muted">{requirement.message}</p>)}
    action={requirement.required && selectedUrl ?
      <Button disabled={action.busy} onClick={() => update("complete", { url: selectedUrl })}>Mark complete</Button> :
      requirement.can_retry && <Button variant="secondary" disabled={action.busy} onClick={() => update("retry")}>Retry resource search</Button>} />;
}

type Draft = { choice: string; requestId: string; submitted: boolean };

function QuestionCard({ paper, saved, reload }: { paper: JunyiPaper; saved: (paper: JunyiPaper) => void; reload: () => Promise<void> }) {
  const question = paper.question!;
  const storageKey = `lumen:junyi:${paper.id}:${question.id}`;
  const optionIds = question.options.map((option) => option.id).join(",");
  const [draft, setDraft] = useState<Draft | null>(null);
  const [conflict, setConflict] = useState(false);
  const action = useAction();
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    heading.current?.focus();
    try {
      const stored: Draft = JSON.parse(sessionStorage.getItem(storageKey) ?? "null");
      if (stored && optionIds.split(",").includes(stored.choice) && /^[0-9a-f-]{36}$/i.test(stored.requestId)) setDraft(stored);
    } catch { /* Saving an answer remains possible when browser storage is unavailable. */ }
  }, [storageKey, optionIds]);

  function saveDraft(value: Draft) {
    setDraft(value);
    try { sessionStorage.setItem(storageKey, JSON.stringify(value)); } catch { /* The in-memory draft is still available. */ }
  }

  async function submit() {
    if (!draft) return;
    setConflict(false);
    const payload = { ...draft, submitted: true };
    saveDraft(payload);
    const result = await action.run(async () => {
      try {
        return await send<{ paper: JunyiPaper }>(`/assessments/${paper.id}/answers`, {
          question_id: question.id, choice_id: payload.choice, request_id: payload.requestId,
        });
      } catch (error) {
        if (error instanceof RequestError && error.status === 409) setConflict(true);
        if (error instanceof RequestError && error.status === 423) {
          saveDraft({ ...payload, submitted: false });
          await reload();
        }
        throw error;
      }
    });
    if (result) {
      try { sessionStorage.removeItem(storageKey); } catch { /* Grading was already saved on the server. */ }
      saved(result.paper);
    }
  }
  return <Panel className="junyi-question">
    <div className="junyi-question-meta"><Badge>{question.topic}</Badge><strong>Question {paper.answered + 1} of {paper.total}</strong></div>
    <progress max={paper.total} value={paper.answered} aria-label="Saved answers" />
    <form onSubmit={(event) => { event.preventDefault(); void submit(); }}>
      <h1 ref={heading} tabIndex={-1} id="junyi-question-prompt">{question.prompt}</h1>
      <div role="radiogroup" aria-labelledby="junyi-question-prompt" className="junyi-choice-list">
        {question.options.map((choice, index) => <label className={`junyi-choice${draft?.choice === choice.id ? " is-selected" : ""}`} key={choice.id}>
          <input type="radio" name="answer" value={choice.id} checked={draft?.choice === choice.id} disabled={action.busy || Boolean(draft?.submitted)}
            onChange={() => saveDraft({ choice: choice.id, requestId: crypto.randomUUID(), submitted: false })} />
          <span className="junyi-choice-letter" aria-hidden>{String.fromCharCode(65 + index)}</span><span>{choice.text}</span>
        </label>)}
      </div>
      <ActionStatus a={action} />
      <div className="junyi-question-footer">
        <span className="p-muted">{paper.answered} {paper.answered === 1 ? "answer" : "answers"} saved</span>
        {conflict ? <Button variant="secondary" onClick={() => void reload()}>Reload progress</Button> :
          <Button type="submit" disabled={!draft || action.busy}>{action.busy ? "Saving…" : draft?.submitted ? "Retry answer" : "Submit answer"}<ArrowRight size={17} aria-hidden /></Button>}
      </div>
    </form>
  </Panel>;
}

export function JunyiPlayer({ id }: { id: string }) {
  const route = useRouter();
  const [paper, setPaper] = useState<JunyiPaper | null>(null);
  const [error, setError] = useState("");
  const [signedOut, setSignedOut] = useState(false);
  const [playing, setPlaying] = useState<boolean | null>(null);
  const [studyTab, setStudyTab] = useState<"remaining" | "completed">("remaining");
  const action = useAction();
  const path = `/assessments/${encodeURIComponent(id)}`;
  const playKey = `lumen:junyi:play:${id}`;
  const acceptPaper = useCallback((next: JunyiPaper) => {
    if (next.id !== id) return;
    setPaper((current) => current?.id === next.id && current.revision > next.revision ? current : next);
  }, [id]);
  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      const data = await api<JunyiPaper>(path, { signal });
      if (!signal?.aborted) { acceptPaper(data); setError(""); setSignedOut(false); }
    } catch (failure) {
      if (!signal?.aborted) { setError((failure as Error).message); setSignedOut(failure instanceof AuthError); }
    }
  }, [acceptPaper, path]);
  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);
  useEffect(() => {
    try {
      setPlaying(sessionStorage.getItem(playKey) === "yes");
    } catch {
      setPlaying(false);
    }
  }, [playKey]);
  const locked = paper?.status === "locked";
  useEffect(() => {
    if (!locked) return;
    setPlaying(false);
    setStudyTab("remaining");
    try { sessionStorage.removeItem(playKey); } catch { /* The checklist still pauses the test. */ }
  }, [locked, playKey]);
  const inProgress = paper?.status === "open" || paper?.status === "locked";
  const optionalStudy = inProgress ? paper.requirements.filter((row) => row.status === "pending" && !row.required) : [];
  const checking = paper?.status === "preparing" || paper?.analysis_status === "pending" || optionalStudy.length > 0;
  useEffect(() => {
    if (!checking || signedOut) return;
    const controller = new AbortController();
    let timer: number;
    const schedule = () => {
      timer = window.setTimeout(async () => {
        await refresh(controller.signal);
        if (!controller.signal.aborted) schedule();
      }, 2500);
    };
    schedule();
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [checking, refresh, signedOut]);

  async function prepare() {
    const next = await action.run(() => send<JunyiPaper>("/assessments"));
    if (next && next.id !== id) route.push(paperUrl(next.id));
    else if (next) await refresh();
  }
  function startTest() {
    if (paper?.status !== "open" || !paper.question || signedOut) return;
    setPlaying(true);
    try { sessionStorage.setItem(playKey, "yes"); } catch { /* The test is still available in this view. */ }
  }
  async function retryAnalysis() {
    const updated = await action.run(() => send<JunyiPaper>(`${path}/analysis`));
    if (updated) acceptPaper(updated.analysis_status === "ready" ? updated : { ...updated, analysis_status: "pending", analysis_retry: false });
  }
  const pendingStudy = paper?.requirements.filter((row) => row.required) ?? [];
  const completedStudy = paper?.requirements.filter((row) => row.status !== "pending") ?? [];
  const studyTotal = pendingStudy.length + completedStudy.length;
  const showChecklist = inProgress && (locked || playing === false);
  const shownStudy = studyTab === "remaining" ? pendingStudy : completedStudy;
  return <div className="junyi-player">
    <header className="junyi-player-header"><Brand /><Button variant="secondary" href={paper?.status === "completed" ? "/?view=progress" : "/?view=today"}>{paper?.status === "completed" ? "Learning report" : "Save & leave"}</Button></header>
    <main className="junyi-player-body">
      <ActionStatus a={action} />
      {error && <Notice title={signedOut ? "Sign in to resume" : "Couldn’t refresh this assessment"} tone="error">{error}{" "}
        {signedOut ? <Button variant="secondary" href="/login">Sign in</Button> : <Button variant="secondary" onClick={() => void refresh()}>Retry</Button>}
      </Notice>}
      {!paper && !error && <Loading text="Opening your assessment…" />}
      {paper?.status === "preparing" && <Panel title="Preparing your assessment">
        <Loading text="Building your paper from the latest skill evidence…" />
        <p className="p-muted">You can return to Today and resume this paper later.</p>
        {paper.can_retry && <Button variant="secondary" onClick={prepare} disabled={action.busy}>Resume preparation</Button>}
      </Panel>}
      {(paper?.status === "shortage" || paper?.status === "unavailable") && <Empty title={paper.status === "shortage" ? "Question bank limit reached" : "Assessment unavailable"}
        action={<Button href="/?view=today">Back to Today</Button>}>{paper.message}</Empty>}
      {showChecklist && <>
        <Heading title="Study before your assessment" subtitle={`Test ${paper.sequence} · ${paper.total} questions`}
          action={<Button onClick={startTest} disabled={locked || !paper.question || signedOut}>
            {paper.answered > 0 ? "Resume test" : "Start test"} <ArrowRight size={17} aria-hidden />
          </Button>} />
        <div className="junyi-study-progress">
          <div role="status">
            <strong>{locked ? `${pendingStudy.length} ${pendingStudy.length === 1 ? "skill" : "skills"} left to unlock` :
              studyTotal ? "Required study complete" : "No required study items"}</strong>
            <span>{locked ? "Complete one verified resource for each required skill." : "Open the test when you’re ready."}</span>
          </div>
          {studyTotal > 0 && <progress aria-label="Required study progress" aria-valuetext={`${completedStudy.length} of ${studyTotal} study items checked off`}
            value={completedStudy.length} max={studyTotal} />}
        </div>
        {studyTotal > 0 && <>
          <div className="junyi-study-tabs" role="group" aria-label="Study checklist">
            <button type="button" aria-pressed={studyTab === "remaining"} onClick={() => setStudyTab("remaining")}>Remaining ({pendingStudy.length})</button>
            <button type="button" aria-pressed={studyTab === "completed"} onClick={() => setStudyTab("completed")}>Completed ({completedStudy.length})</button>
          </div>
          {shownStudy.length > 0 ? <div className="junyi-study-grid">{shownStudy.map((row) => <StudyCard key={row.tag} requirement={row} paperId={id} saved={acceptPaper} />)}</div> :
            <p className="p-muted junyi-study-empty">{studyTab === "remaining" ? "All required topics are checked off." : "Your completed study items will appear here."}</p>}
        </>}
      </>}
      {playing && paper?.status === "open" && paper.question && <QuestionCard key={paper.question.id} paper={paper} saved={acceptPaper} reload={() => refresh()} />}
      {optionalStudy.length > 0 && <details className="p-panel junyi-completed-study">
        <summary>Optional study without verified links ({optionalStudy.length})</summary>
        <p className="p-muted p-panel-sub">You can skip these topics. A topic becomes required if a verified link is found before you finish the assessment.</p>
        <div className="junyi-study-grid">{optionalStudy.map((row) => <StudyCard key={row.tag} requirement={row} paperId={id} saved={acceptPaper} />)}</div>
      </details>}
      {paper?.status === "completed" && paper.results && <>
        <Heading title="Assessment complete" subtitle={`Test ${paper.sequence} result saved.`} />
        <Panel className="junyi-result">
          <Badge><Check size={16} aria-hidden /> Actual paper score</Badge>
          <strong>{paper.results.correct} / {paper.results.total}</strong>
          <p>{percent(paper.results.accuracy)} correct</p>
          {paper.analysis_status === "ready" && <Button onClick={prepare} disabled={action.busy}>{action.busy ? "Starting…" : "Start next assessment"}</Button>}
        </Panel>
        {paper.analysis_status === "pending" && <Loading text="Updating your skill estimates…" />}
        {paper.analysis_status === "unavailable" && <Notice title="Updated skill estimates are unavailable">Your assessment result is saved.</Notice>}
        {paper.analysis_retry && paper.analysis_status !== "ready" && <Button variant="secondary" disabled={action.busy} onClick={retryAnalysis}>Retry skill analysis</Button>}
        {paper.skills.length > 0 && <SkillTable skills={paper.skills} answersUsed={paper.answers_used} comparison />}
      </>}
    </main>
  </div>;
}

export function JunyiOperator() {
  const data = useLoad<JunyiAdmin>("/admin/junyi");
  const action = useAction();
  const [editing, setEditing] = useState<{ id: string; tag: number } | null>(null);
  const [reason, setReason] = useState("");
  if (!data.data) return null;

  async function override() {
    if (!editing) return;
    const result = await action.run(() => send<JunyiPaper>(`/admin/junyi/${editing.id}/study/${editing.tag}/override`, { reason }), "Study requirement cleared");
    if (result) { setEditing(null); setReason(""); data.reload(); }
  }
  return <Panel title="Junyi assessment operations" className="junyi-operations">
    <div className="p-actions"><Badge tone={data.data.ready ? "sage" : "amber"}><ShieldCheck size={16} aria-hidden /> {data.data.ready ? "Ready" : "Setup pending"}</Badge>
      <span>{data.data.bank_questions.toLocaleString()} verified questions</span><Button variant="secondary" onClick={data.reload}>Refresh status</Button></div>
    {data.data.issues.map((issue) => <p className="p-muted" key={issue}>{issue}</p>)}
    <ActionStatus a={action} />
    {data.data.active.map((paper) => <div className="junyi-operator-paper" key={paper.id}>
      <strong>{stageName(paper.stage)} assessment · {paper.status}</strong>
      {paper.requirements.filter((row) => row.status === "pending").map((row) => <div key={row.tag} className="junyi-study-link">
        <span>{row.topic} · {row.lookup}</span>
        {row.lookup === "unavailable" && <Button variant="secondary" onClick={() => { setReason(""); setEditing({ id: paper.id, tag: row.tag }); }}>Clear requirement</Button>}
      </div>)}
    </div>)}
    {editing && <form onSubmit={(event) => { event.preventDefault(); void override(); }}>
      <Field label="Reason for clearing the requirement"><input minLength={5} maxLength={500} required value={reason} onChange={(event) => setReason(event.target.value)} /></Field>
      <div className="p-actions"><Button type="submit" disabled={action.busy}>Confirm override</Button><Button variant="secondary" onClick={() => setEditing(null)}>Cancel</Button></div>
    </form>}
    {data.data.active.length === 0 && <p className="p-muted"><BookOpen size={16} aria-hidden /> No active Junyi paper.</p>}
  </Panel>;
}

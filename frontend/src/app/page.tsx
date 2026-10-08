"use client";
import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  ChartNoAxesCombined,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Clock3,
  Compass,
  Layers3,
  LayoutDashboard,
  LogOut,
  Menu,
  Play,
  Plus,
  Sparkles,
  Sprout,
  Target,
  UserRound,
  X,
} from "lucide-react";
import {
  api,
  Assessment,
  AuthError,
  Concept,
  Dashboard,
  percent,
  User,
} from "@/lib/api";
import { Auth } from "@/components/auth";
import { Brand } from "@/components/brand";
import { LearnerProfileForm, StudyPlan } from "@/components/learner-profile";

type Tab = "overview" | "path" | "progress" | "profile";
const colors = ["sage", "lilac", "peach", "blue"];

function ProgressChart({ data }: { data: Dashboard["progress"] }) {
  if (!data.length)
    return (
      <div className="chart-empty">
        <ChartNoAxesCombined size={30} />
        <strong>Your story starts here</strong>
        <span>Finish an assessment to see your progress over time.</span>
      </div>
    );
  const points = data
    .slice(-8)
    .map((row, i, rows) => [
      40 + (i * 510) / Math.max(rows.length - 1, 1),
      145 - row.mastery * 120,
    ]);
  return (
    <div className="chart">
      <svg
        viewBox="0 0 600 180"
        role="img"
        aria-label="Estimated understanding after each completed assessment"
      >
        <defs>
          <linearGradient id="fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#6d9875" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#6d9875" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, 0.5, 1].map((n) => (
          <g key={n}>
            <line
              x1="40"
              x2="555"
              y1={145 - n * 120}
              y2={145 - n * 120}
              stroke="#eceee8"
              strokeDasharray="4 5"
            />
            <text x="0" y={149 - n * 120} fill="#8b928b" fontSize="11">
              {percent(n)}
            </text>
          </g>
        ))}
        {points.length > 1 && (
          <>
            <path
              d={`M${points.map((p) => p.join(",")).join(" L")} L${points.at(-1)![0]},145 L40,145 Z`}
              fill="url(#fill)"
            />
            <polyline
              points={points.map((p) => p.join(",")).join(" ")}
              fill="none"
              stroke="#527d5a"
              strokeWidth="3"
              strokeLinejoin="round"
            />
          </>
        )}
        {points.map(([x, y], i) => (
          <g key={i}>
            <circle
              cx={x}
              cy={y}
              r="4.5"
              fill="#527d5a"
              stroke="white"
              strokeWidth="2"
            />
            <text
              x={x}
              y="173"
              fill="#8b928b"
              fontSize="11"
              textAnchor="middle"
            >
              Test {data.length - points.length + i + 1}
            </text>
          </g>
        ))}
      </svg>
      <span className="chart-caption">
        <i />
        Estimated concept understanding • Bayesian knowledge tracing
      </span>
    </div>
  );
}

export default function Home() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  const [data, setData] = useState<Dashboard | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState<Tab>("overview");
  const [lesson, setLesson] = useState<Concept | null>(null);
  const [mobile, setMobile] = useState(false);
  const load = useCallback(async () => {
    try {
      setData(await api<Dashboard>("/learning/mathematics"));
      setError("");
    } catch (e) {
      if (e instanceof AuthError) setUser(null);
      else setError((e as Error).message);
    }
  }, []);
  useEffect(() => {
    api<User>("/auth/me")
      .then(setUser)
      .catch((e) => {
        if (!(e instanceof AuthError)) setError((e as Error).message);
      })
      .finally(() => setReady(true));
  }, []);
  useEffect(() => {
    if (user) void load();
  }, [user, load]);
  async function start(mode: string, concept_id?: string) {
    setBusy(true);
    setError("");
    try {
      const assessment = await api<Assessment>("/assessments", {
        method: "POST",
        body: JSON.stringify({ subject_id: "mathematics", mode, concept_id }),
      });
      router.push(`/assessment/${assessment.id}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function logout() {
    try {
      await api("/auth/logout", { method: "POST" });
      setUser(null);
      setData(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  if (!ready)
    return (
      <div className="loading">
        <Brand />
        <span>Opening your learning space…</span>
      </div>
    );
  if (!user)
    return (
      <>
        <Auth onSuccess={setUser} />
        {error && (
          <div className="service-error" role="alert">
            {error}
            <button onClick={() => location.reload()}>Try again</button>
          </div>
        )}
      </>
    );
  const understanding = data
    ? data.concepts.reduce((sum, c) => sum + c.mastery, 0) /
      data.concepts.length
    : 0;
  const countStrong =
    data?.concepts.filter((c) => c.status === "strong").length || 0;
  const navigation = [
    { id: "overview" as Tab, name: "Overview", icon: LayoutDashboard },
    { id: "path" as Tab, name: "My learning path", icon: Compass },
    { id: "progress" as Tab, name: "My progress", icon: ChartNoAxesCombined },
    { id: "profile" as Tab, name: "My profile", icon: UserRound },
  ];
  return (
    <div className="app-shell">
      {mobile && (
        <button
          className="sidebar-scrim"
          onClick={() => setMobile(false)}
          aria-label="Close navigation"
        />
      )}
      <aside className={`sidebar ${mobile ? "open" : ""}`}>
        <Brand />
        <div className="workspace-label">MY WORKSPACE</div>
        <nav>
          {navigation.map(({ id, name, icon: Icon }) => (
            <button
              key={id}
              className={tab === id ? "active" : ""}
              onClick={() => {
                setTab(id);
                setMobile(false);
              }}
            >
              <Icon size={19} />
              {name}
              {tab === id && <span className="nav-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <span className="small-flower">✳</span>
          <strong>
            Small steps.
            <br />
            Real growth.
          </strong>
          <p>A few thoughtful minutes today can make tomorrow feel easier.</p>
          <span>
            YOU’VE GOT THIS <ArrowUpRight size={14} />
          </span>
        </div>
        <div className="sidebar-bottom">
          <button
            onClick={() =>
              setLesson({
                id: "about",
                name: "How Lumen learns with you",
                lesson:
                  "Lumen uses your answers to update an estimate of concept understanding. Early estimates need more evidence. Question difficulty adapts as you practice. HiTSKT can predict correctness when a verified model and matching question bank are connected; new content uses a Bayesian cold-start estimator. These estimates guide practice, and are not grades or a measure of your ability.",
              } as Concept)
            }
          >
            <CircleHelp size={18} />
            How it works
          </button>
          <button onClick={logout}>
            <LogOut size={18} />
            Sign out
          </button>
          <div className="profile">
            <span className="avatar">
              {user.name.slice(0, 1).toUpperCase()}
            </span>
            <span>
              <strong>{user.name}</strong>
              <small>
                {user.profile.grade_level === "Not set"
                  ? "Growing learner"
                  : user.profile.grade_level}
              </small>
            </span>
            <Sprout size={17} />
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="mobile-menu"
              onClick={() => setMobile(true)}
              aria-label="Open navigation"
            >
              <Menu size={20} />
            </button>
            My workspace <span>/</span>
            <strong>{navigation.find((n) => n.id === tab)?.name}</strong>
          </div>
          <span className="top-date">
            {new Date().toLocaleDateString("en-IN", {
              day: "numeric",
              month: "long",
              year: "numeric",
            })}
          </span>
        </header>
        <main className="dashboard">
          <div className="page-heading">
            <div>
              <div className="eyebrow">LET’S MAKE A LITTLE PROGRESS</div>
              <h1>
                {tab === "overview"
                  ? `Hello, ${user.name.split(" ")[0]}`
                  : tab === "path"
                    ? "A path that grows with you"
                    : tab === "profile"
                      ? "Your goals, your pace"
                      : "Look how far you’ve come"}
                <span className="heading-spark">
                  {tab === "overview" ? "✳" : ""}
                </span>
              </h1>
              <p>
                {tab === "overview"
                  ? "Every question is a chance to understand a little more."
                  : tab === "path"
                    ? "Start with a foundation. Take the next step when you’re ready."
                    : tab === "profile"
                      ? "Tell us how you like to learn. Make a plan that fits your week."
                      : "Learning takes time. Your effort is becoming a story."}
              </p>
            </div>
            <span className="pill">
              <span className="status-dot" />
              YOUR PERSONAL LEARNING SPACE
            </span>
          </div>
          {error && (
            <div className="error" role="alert">
              {error}
              <button onClick={load}>Try again</button>
            </div>
          )}
          {!data ? (
            <div className="card loading-card">
              {error
                ? "The learning service is unavailable."
                : "Finding your next learning moment…"}
            </div>
          ) : (
            <>
              {tab === "profile" && (
                <LearnerProfileForm
                  user={user}
                  concepts={data.concepts}
                  onSaved={setUser}
                />
              )}
              {tab === "overview" && (
                <>
                  <section className="hero-card">
                    <div className="hero-copy">
                      <span className="hero-tag">
                        <Sparkles size={14} />
                        {data.diagnostic_complete
                          ? "YOUR NEXT LEARNING MOMENT"
                          : "A GOOD PLACE TO START"}
                      </span>
                      <h2>
                        {data.active_assessment ? (
                          "Pick up where you left off."
                        ) : data.diagnostic_complete ? (
                          "Keep your curiosity going."
                        ) : (
                          <>
                            Let’s find your
                            <br />
                            starting point.
                          </>
                        )}
                      </h2>
                      <p>
                        {data.active_assessment
                          ? "Your answers are saved. Your next question is ready when you are."
                          : data.diagnostic_complete
                            ? "A fresh set of questions, shaped by what you’ve learned so far."
                            : "A short, thoughtful check-in to discover what you know and where we can grow together."}
                      </p>
                      <button
                        className="button primary"
                        disabled={busy}
                        onClick={() =>
                          data.active_assessment
                            ? router.push(
                                `/assessment/${data.active_assessment.id}`,
                              )
                            : start(
                                data.diagnostic_complete
                                  ? "adaptive"
                                  : "diagnostic",
                              )
                        }
                      >
                        {busy
                          ? "Preparing…"
                          : data.active_assessment
                            ? "Continue assessment"
                            : data.diagnostic_complete
                              ? "Start adaptive practice"
                              : "Find my starting point"}
                        <ArrowRight size={18} />
                      </button>
                      <span className="hero-meta">
                        <Clock3 size={13} />
                        {data.active_assessment
                          ? `${data.active_assessment.answered} of ${data.active_assessment.total} answered`
                          : `${data.diagnostic_complete ? data.profile.adaptive_session_questions : 12} questions · At your own pace`}
                        <span>•</span>Go at your own pace
                      </span>
                    </div>
                    <div className="plant-art" aria-hidden="true">
                      <span className="art-orbit" />
                      <span className="art-orbit inner" />
                      <span className="plant-stem" />
                      <span className="leaf leaf-a" />
                      <span className="leaf leaf-b" />
                      <span className="leaf leaf-c" />
                      <span className="leaf leaf-d" />
                      <span className="plant-pot" />
                      <span className="art-dot dot-a" />
                      <span className="art-dot dot-b" />
                      <span className="float-note">
                        <CheckCircle2 size={18} />
                        Room to grow
                      </span>
                      <span className="float-spark">✧</span>
                    </div>
                  </section>
                  <div className="stats-grid">
                    <div className="stat-card">
                      <span className="stat-icon sage">
                        <BookOpen size={20} />
                      </span>
                      <div>
                        <span>Questions explored</span>
                        <strong>
                          {data.stats.answers}
                          <small>answered thoughtfully</small>
                        </strong>
                      </div>
                    </div>
                    <div className="stat-card">
                      <span className="stat-icon lilac">
                        <Target size={20} />
                      </span>
                      <div>
                        <span>Answer accuracy</span>
                        <strong>
                          {data.stats.accuracy === null
                            ? "—"
                            : percent(data.stats.accuracy)}
                          <small>
                            {data.stats.answers
                              ? "across your practice"
                              : "your first answer awaits"}
                          </small>
                        </strong>
                      </div>
                    </div>
                    <div className="stat-card">
                      <span className="stat-icon peach">
                        <Sprout size={20} />
                      </span>
                      <div>
                        <span>Strong foundations</span>
                        <strong>
                          {countStrong}
                          <small>
                            of {data.concepts.length} concepts explored
                          </small>
                        </strong>
                      </div>
                    </div>
                  </div>
                  <section className="subject-section">
                    <div className="section-heading">
                      <div>
                        <h2>Your learning world</h2>
                        <p>A focused space to build your foundations.</p>
                      </div>
                      <span className="subtle-label">
                        {data.content.question_count} QUESTIONS ·{" "}
                        {data.concepts.length} CONCEPTS
                      </span>
                    </div>
                    <div className="subject-card">
                      <span className="subject-symbol">∑</span>
                      <div className="subject-details">
                        <span className="eyebrow">FOUNDATIONS</span>
                        <h3>{data.subject.name}</h3>
                        <p>
                          Fractions, percentages, equations & everyday geometry.
                        </p>
                        <div className="subject-progress">
                          <div className="progress-track">
                            <i
                              style={{
                                width: `${data.stats.answers ? understanding * 100 : 0}%`,
                              }}
                            />
                          </div>
                          <span>
                            {data.stats.answers
                              ? `${percent(understanding)} estimated understanding`
                              : "Ready for your first discovery"}
                          </span>
                        </div>
                      </div>
                      <button
                        className="button secondary"
                        disabled={busy}
                        onClick={() =>
                          start(
                            data.diagnostic_complete
                              ? "adaptive"
                              : "diagnostic",
                          )
                        }
                      >
                        Let’s explore
                        <ArrowUpRight size={16} />
                      </button>
                    </div>
                  </section>
                  <StudyPlan
                    data={data}
                    onConfigure={() => setTab("profile")}
                    onReview={(id) =>
                      setLesson(data.concepts.find((c) => c.id === id)!)
                    }
                  />
                </>
              )}
              {(tab === "overview" || tab === "path") && (
                <div className="content-grid">
                  <section className="card concepts-card">
                    <div className="section-heading">
                      <div>
                        <h2>Your concept map</h2>
                        <p>A snapshot of what’s taking shape.</p>
                      </div>
                      <span className="round-icon">
                        <Layers3 size={17} />
                      </span>
                    </div>
                    <div className="concept-list">
                      {data.concepts.map((concept, i) => (
                        <button
                          key={concept.id}
                          className="concept-row"
                          onClick={() => setLesson(concept)}
                        >
                          <span className={`concept-icon ${colors[i]}`}>
                            {["½", "%", "x", "▱"][i]}
                          </span>
                          <div className="concept-info">
                            <strong>{concept.name}</strong>
                            <span>
                              {concept.attempts
                                ? `${concept.attempts} answers · ${concept.confidence} evidence`
                                : "Waiting to be explored"}
                            </span>
                            <div className="progress-track">
                              <i
                                style={{
                                  width: `${concept.attempts ? concept.mastery * 100 : 0}%`,
                                }}
                              />
                            </div>
                          </div>
                          <span
                            className={`concept-value ${concept.status === "strong" ? "strong" : ""}`}
                          >
                            {concept.attempts ? percent(concept.mastery) : "—"}
                            <small>
                              {concept.status === "strong"
                                ? "Looking strong"
                                : concept.status === "needs_practice"
                                  ? "Room to grow"
                                  : concept.attempts
                                    ? "Taking shape"
                                    : "Unexplored"}
                            </small>
                          </span>
                        </button>
                      ))}
                    </div>
                    <div className="card-footnote">
                      <CircleHelp size={14} />
                      Estimates grow more reliable with practice.
                    </div>
                  </section>
                  <section className="recommendations">
                    <div className="section-heading">
                      <div>
                        <h2>A good next step</h2>
                        <p>A little direction, just for you.</p>
                      </div>
                      <Sparkles size={18} />
                    </div>
                    {data.recommendations
                      .slice(0, tab === "path" ? 3 : 2)
                      .map((item, i) => (
                        <article
                          className={`recommendation ${i === 0 ? "lilac-bg" : "cream-bg"}`}
                          key={item.concept_id}
                        >
                          <div className="rec-meta">
                            <span>
                              {i === 0 ? "START HERE" : "THEN EXPLORE"}
                            </span>
                            <span>
                              <Clock3 size={12} />
                              {item.minutes} min
                            </span>
                          </div>
                          <h3>{item.name}</h3>
                          <p>{item.reason}</p>
                          <button
                            disabled={busy}
                            onClick={() =>
                              setLesson(
                                data.concepts.find(
                                  (c) => c.id === item.concept_id,
                                )!,
                              )
                            }
                          >
                            Review & practice
                            <ArrowRight size={16} />
                          </button>
                        </article>
                      ))}
                  </section>
                </div>
              )}
              {(tab === "overview" || tab === "progress") && (
                <section className="card progress-card">
                  <div className="section-heading">
                    <div>
                      <h2>Growth, one step at a time</h2>
                      <p>Your understanding across completed assessments.</p>
                    </div>
                    <span className="subtle-label">
                      {data.stats.completed_assessments} ASSESSMENTS COMPLETED
                    </span>
                  </div>
                  <ProgressChart data={data.progress} />
                </section>
              )}
              {tab === "progress" && (
                <section className="card">
                  <div className="section-heading">
                    <div>
                      <h2>Recent learning moments</h2>
                      <p>The answers behind your progress.</p>
                    </div>
                  </div>
                  {!data.recent_answers.length ? (
                    <p className="muted">
                      Your first assessment will bring this page to life.
                    </p>
                  ) : (
                    <div className="history-list">
                      {data.recent_answers.map((answer, i) => (
                        <div key={i}>
                          <span
                            className={`history-dot ${answer.correct ? "correct" : "incorrect"}`}
                          >
                            {answer.correct ? (
                              <Check size={15} />
                            ) : (
                              <ArrowRight size={15} />
                            )}
                          </span>
                          <span>
                            <strong>
                              {
                                data.concepts.find(
                                  (c) => c.id === answer.concept_id,
                                )?.name
                              }
                            </strong>
                            <small>
                              Level {answer.difficulty} ·{" "}
                              {answer.prediction.provider === "hitskt"
                                ? "HiTSKT"
                                : "Bayesian estimate"}
                            </small>
                          </span>
                          <span>
                            {answer.correct
                              ? "Understood"
                              : "A chance to revisit"}
                          </span>
                          <time>
                            {new Date(answer.answered_at).toLocaleDateString(
                              "en-IN",
                              { day: "numeric", month: "short" },
                            )}
                          </time>
                        </div>
                      ))}
                    </div>
                  )}
                </section>
              )}
              <footer className="dashboard-footer">
                <span>
                  <Sprout size={15} />
                  Made for your pace. Built for your growth.
                </span>
                <span>
                  {data.model.hitskt_loaded
                    ? "HiTSKT available for mapped content"
                    : "Bayesian cold-start estimates"}
                </span>
              </footer>
            </>
          )}
        </main>
      </div>
      {lesson && (
        <div className="modal-backdrop" onClick={() => setLesson(null)}>
          <section
            className="lesson-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="lesson-title"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="modal-close"
              onClick={() => setLesson(null)}
              aria-label="Close lesson"
            >
              <X size={20} />
            </button>
            <span className="pill">A MOMENT TO UNDERSTAND</span>
            <h2 id="lesson-title">{lesson.name}</h2>
            <p>{lesson.lesson}</p>
            {lesson.steps && (
              <ol className="lesson-steps">
                {lesson.steps.map((step) => (
                  <li key={step}>{step}</li>
                ))}
              </ol>
            )}
            {lesson.worked_example && (
              <div className="lesson-example">
                <strong>Work through an example</strong>
                <p>{lesson.worked_example}</p>
              </div>
            )}
            {lesson.common_mistake && (
              <div className="lesson-tip">
                <strong>A useful check</strong>
                <p>{lesson.common_mistake}</p>
              </div>
            )}
            {lesson.application && (
              <p className="lesson-application">
                <strong>Use it in everyday life:</strong> {lesson.application}
              </p>
            )}
            {lesson.prerequisites?.length > 0 && (
              <div className="prerequisite">
                <Layers3 size={17} />
                Build on:{" "}
                {lesson.prerequisites
                  .map((p) => data?.concepts.find((c) => c.id === p)?.name)
                  .join(", ")}
              </div>
            )}
            {lesson.id !== "about" && (
              <button
                className="button primary full"
                disabled={busy}
                onClick={() => start("practice", lesson.id)}
              >
                Practice this concept
                <ArrowRight size={18} />
              </button>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

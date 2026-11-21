"use client";

import { useEffect, useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  BookOpen,
  ClipboardList,
  ChartNoAxesCombined,
  ShieldCheck,
} from "lucide-react";
import { send, type User } from "@/lib/api";
import {
  ActionStatus,
  Badge,
  Brand,
  Button,
  EvidenceBars,
  Field,
  Notice,
  Panel,
  useAction,
  useLoad,
} from "./product";

export function Landing() {
  const cfg = useLoad<{ demo_enabled: boolean }>("/config");
  return (
    <div className="p-public">
      <header className="p-public-nav">
        <Brand />
        <nav aria-label="How Lumen works">
          <a href="#learning">How it works</a>
          <a href="#evidence">What we measure</a>
        </nav>
        <div className="p-actions">
          <Button href="/login" variant="secondary">
            Sign in
          </Button>
          <Button href="/signup">Get started</Button>
        </div>
      </header>
      <main>
        <div className="p-public-hero">
          <div>
            <div className="p-eyebrow">ADAPTIVE MATHEMATICS PRACTICE</div>
            <h1>
              Know what to
              <br />
              practise next.
            </h1>
            <p>
              Practise maths at your own pace. See what needs attention and get
              a recommendation based on your answers.
            </p>
            <div className="p-actions">
              <Button href="/signup">
                Create student account <ArrowRight size={18} />
              </Button>
              {cfg.data?.demo_enabled && (
                <Button href="/login" variant="secondary">
                  Try the local demo
                </Button>
              )}
            </div>
          </div>
          <Panel className="p-sample-report">
            <div className="p-eyebrow">
              SAMPLE LEARNER REPORT · ILLUSTRATIVE
            </div>
            <h2>Your next step</h2>
            <p>Review common denominators, then try six questions.</p>
            <EvidenceBars
              items={[
                { name: "Fractions", count: 0, total: 3 },
                { name: "Percentages", count: 1, total: 3 },
                { name: "Equations", count: 3, total: 3 },
                { name: "Geometry", count: 2, total: 3 },
              ]}
            />
            <footer>
              Practice accuracy <strong>6 / 12 correct</strong>
            </footer>
          </Panel>
        </div>
        <section
          id="learning"
          className="p-steps"
          aria-label="Learning journey"
        >
          {[
            [
              ClipboardList,
              "Find your starting point",
              "Choose a topic or try mixed practice.",
            ],
            [
              BookOpen,
              "Practise with feedback",
              "Work on skills that need attention and see an explanation.",
            ],
            [
              ChartNoAxesCombined,
              "Check your progress",
              "Revisit a skill with fresh questions and compare evidence.",
            ],
          ].map(([Icon, title, body], i) => {
            const I = Icon as typeof BookOpen;
            return (
              <article key={i}>
                <I size={25} />
                <div>
                  <strong>{String(title)}</strong>
                  <p>{String(body)}</p>
                </div>
              </article>
            );
          })}
        </section>
        <section className="p-public-info">
          <article id="evidence">
            <div className="p-eyebrow">WHAT WE MEASURE</div>
            <h2>See how your understanding changes.</h2>
            <p>
              Accuracy counts correct answers. Bayesian knowledge tracing
              estimates understanding from earlier evidence to guide practice.
            </p>
          </article>
        </section>
      </main>
      <footer className="p-public-footer">
        <Brand />
        <span>
          <ShieldCheck aria-hidden size={16} /> Student work stays in its own
          workspace.
        </span>
      </footer>
    </div>
  );
}

type Mode = "login" | "signup" | "recover" | "reset";
const headings = {
  login: ["Welcome back", "Sign in to pick up where you left off."],
  signup: [
    "Create your student account",
    "Only the details needed to sign in and set up learning.",
  ],
  recover: [
    "Reset your password",
    "We’ll email a reset link if an account exists for this address.",
  ],
  reset: ["Choose a new password", "Reset links can only be used once."],
};

export function AuthScreen({ mode }: { mode: Mode }) {
  const r = useRouter(),
    a = useAction(),
    cfg = useLoad<{ demo_enabled: boolean; email_recovery: boolean }>(
      "/config",
    );
  const [token, setToken] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  useEffect(() => {
    if (mode === "reset")
      setToken(new URLSearchParams(location.search).get("token") ?? "");
  }, [mode]);
  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    if (mode === "reset" && form.get("confirm") !== password) {
      a.setError("Passwords must match.");
      return;
    }
    if (mode === "reset") {
      const x = await a.run(() =>
        send<{ message: string }>("/auth/reset", { token, password }),
      );
      if (x) r.replace("/login?reset=1");
      return;
    }
    if (mode === "recover") {
      await a.run(
        () => send("/auth/recovery", { email }),
        "If the account exists, a reset link will be sent.",
      );
      return;
    }
    const value = await a.run(() =>
      send<User>(`/auth/${mode === "signup" ? "register" : "login"}`, {
        email,
        password,
        ...(mode === "signup" ? { name: form.get("name") } : {}),
      }),
    );
    if (value) r.replace("/");
  }
  function sample(role: string) {
    setEmail(`${role}@demo.lumen.example.com`);
    setPassword("Lumen12345");
  }
  return (
    <main className="p-auth">
      <aside>
        <Brand />
        <div>
          <div className="p-eyebrow">LUMEN LEARNING WORKSPACE</div>
          <h1>
            Study with
            <br />a clear plan.
          </h1>
          <p>
            Start with what you know.
            <br />
            Build from there.
          </p>
        </div>
        <span>Mathematics foundations</span>
      </aside>
      <section>
        <div className="p-auth-inner">
          <Link href="/" className="p-text-link">
            ← Back to Lumen
          </Link>
          <div className="p-eyebrow">YOUR ACCOUNT</div>
          <h2>{headings[mode][0]}</h2>
          <p className="p-muted">{headings[mode][1]}</p>
          {mode === "recover" && cfg.data?.email_recovery === false && (
            <Notice title="Email recovery is unavailable on this server">
              Contact your school or local administrator for help signing in.
            </Notice>
          )}
          <form onSubmit={submit} className="p-stack">
            <ActionStatus a={a} />
            {mode === "signup" && (
              <Field label="Name">
                <input
                  name="name"
                  autoComplete="name"
                  minLength={2}
                  maxLength={80}
                  required
                  placeholder="Asha Shah"
                />
              </Field>
            )}
            {mode !== "reset" && (
              <Field label="Email address">
                <input
                  type="email"
                  name="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  placeholder="you@example.com"
                />
              </Field>
            )}
            {mode !== "recover" && (
              <div className="p-field">
                <label htmlFor="entry-password">
                  {mode === "reset" ? "New password" : "Password"}
                </label>
                <div className="p-password">
                  <input
                    id="entry-password"
                    aria-describedby={
                      mode === "login" ? undefined : "password-hint"
                    }
                    type={show ? "text" : "password"}
                    name="password"
                    autoComplete={
                      mode === "login" ? "current-password" : "new-password"
                    }
                    minLength={10}
                    maxLength={128}
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                  />
                  <button
                    type="button"
                    aria-label={show ? "Hide password" : "Show password"}
                    onClick={() => setShow(!show)}
                  >
                    {show ? "Hide" : "Show"}
                  </button>
                </div>
                {mode !== "login" && (
                  <small id="password-hint">Use at least 10 characters.</small>
                )}
              </div>
            )}
            {mode === "reset" && (
              <Field label="Confirm password">
                <input
                  type="password"
                  name="confirm"
                  autoComplete="new-password"
                  minLength={10}
                  required
                />
              </Field>
            )}
            {mode === "login" && (
              <Link href="/recover" className="p-text-link">
                Forgot password?
              </Link>
            )}
            <Button
              type="submit"
              disabled={
                a.busy ||
                (mode === "reset" && !token) ||
                (mode === "recover" && cfg.data?.email_recovery === false)
              }
            >
              {a.busy
                ? "Working…"
                : mode === "login"
                  ? "Sign in"
                  : mode === "signup"
                    ? "Create account"
                    : mode === "recover"
                      ? "Send reset link"
                      : "Save password"}{" "}
              <ArrowRight size={18} />
            </Button>
          </form>
          <p className="p-account-switch">
            {mode === "login" ? (
              <>
                New student? <Link href="/signup">Create an account</Link>
              </>
            ) : (
              <>
                Already have access? <Link href="/login">Sign in</Link>
              </>
            )}
          </p>
          {mode === "login" && cfg.data?.demo_enabled && (
            <Panel className="p-demo-login">
              <Badge tone="amber">LOCAL DEMO</Badge>
              <h3>Fill sample credentials</h3>
              <p>
                Fictional accounts and activity. Password:{" "}
                <strong>Lumen12345</strong>
              </p>
              <div className="p-actions">
                {["student", "rohan", "admin"].map((x) => (
                  <Button key={x} variant="secondary" onClick={() => sample(x)}>
                    {x === "student"
                      ? "Asha"
                      : x === "rohan"
                        ? "Rohan"
                        : "Admin"}
                  </Button>
                ))}
              </div>
            </Panel>
          )}
        </div>
      </section>
    </main>
  );
}

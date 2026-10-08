"use client";
import { FormEvent, useState } from "react";
import { ArrowRight, Check } from "lucide-react";
import { api, User } from "@/lib/api";
import { Brand } from "./brand";
import { GrowthScene } from "./growth-scene";
import { MotionToggle } from "./motion";

export function Auth({ onSuccess }: { onSuccess: (user: User) => void }) {
  const [register, setRegister] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    const fields = new FormData(event.currentTarget);
    try {
      onSuccess(
        await api<User>(`/auth/${register ? "register" : "login"}`, {
          method: "POST",
          body: JSON.stringify(Object.fromEntries(fields)),
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="auth-page">
      <section className="auth-story">
        <Brand />
        <div className="auth-motion">
          <MotionToggle />
        </div>
        <div className="eyebrow">A LITTLE PROGRESS. EVERY DAY.</div>
        <h1>
          Your potential.
          <br />A clearer <em>path.</em>
        </h1>
        <p>
          Learning feels better when the next step is the right one. Discover
          your strengths, close the gaps, and grow at your own pace.
        </p>
        <GrowthScene variant="auth" />
        <div className="story-footer">
          <span>
            <Check size={21} />
            Practice that adapts to you
          </span>
          <span>
            <Check size={21} />
            Progress you can see
          </span>
        </div>
      </section>
      <section className="auth-form-wrap">
        <div className="auth-form">
          <span className="pill">YOUR LEARNING SPACE</span>
          <h2>
            {register ? "Start something good." : "Good to see you again."}
          </h2>
          <p>
            {register
              ? "Create your account. Let’s find your next step."
              : "Your learning journey is right where you left it."}
          </p>
          <div className="auth-tabs">
            <button
              className={register ? "selected" : ""}
              onClick={() => {
                setRegister(true);
                setError("");
              }}
            >
              Create account
            </button>
            <button
              className={!register ? "selected" : ""}
              onClick={() => {
                setRegister(false);
                setError("");
              }}
            >
              Sign in
            </button>
          </div>
          <form onSubmit={submit}>
            {register && (
              <label>
                Your name
                <input
                  name="name"
                  placeholder="What should we call you?"
                  minLength={2}
                  maxLength={80}
                  autoComplete="name"
                  required
                />
              </label>
            )}
            <label>
              Email address
              <input
                name="email"
                type="email"
                placeholder="you@example.com"
                autoComplete="email"
                required
              />
            </label>
            <label>
              Password
              <input
                name="password"
                type="password"
                placeholder="At least 10 characters"
                minLength={10}
                maxLength={128}
                autoComplete={register ? "new-password" : "current-password"}
                required
              />
            </label>
            {error && (
              <div className="error" role="alert">
                {error}
              </div>
            )}
            <button className="button primary full" disabled={busy}>
              {busy
                ? "One moment…"
                : register
                  ? "Create my learning space"
                  : "Sign in"}
              <ArrowRight size={23} />
            </button>
          </form>
          <p className="fine-print">
            Made for thoughtful practice, with estimates that get more useful as
            you learn.
          </p>
        </div>
      </section>
    </main>
  );
}

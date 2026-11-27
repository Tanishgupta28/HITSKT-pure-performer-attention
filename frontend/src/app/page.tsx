"use client";
import { useEffect, useState } from "react";
import { api, AuthError, type User } from "@/lib/api";
import { Boundary, Button, Loading, Notice, Shell, useLoad, useTab } from "@/components/product";
import { Landing } from "@/components/entry";
import { StudentWorkspace } from "@/components/student";
import { AdminWorkspace } from "@/components/admin";
import { JunyiLearn, JunyiWorkspace } from "@/components/junyi";
import type { JunyiDashboard } from "@/lib/junyi";

function StudentEntry({ user, tab, navigate, onSaved }: {
  user: User; tab: string; navigate: (tab: string) => void; onSaved: (user: User) => void;
}) {
  const junyi = useLoad<JunyiDashboard>("/junyi/dashboard");
  const original = <StudentWorkspace user={user} tab={tab} navigate={navigate} onSaved={onSaved} focusedOnly={junyi.data?.assigned} />;
  if (!user.onboarding_complete || tab === "profile") return original;
  return <Boundary value={junyi}>
    {junyi.data?.assigned ? (tab === "practice" ? <><JunyiLearn dashboard={junyi.data} />{original}</> :
      <JunyiWorkspace dashboard={junyi.data} name={user.name} tab={tab} />) : original}
  </Boundary>;
}

function Workspace({
  user,
  onSaved,
}: {
  user: User;
  onSaved: (u: User) => void;
}) {
  const { tab, navigate } = useTab(user.role);
  return (
    <Shell user={user} tab={tab} navigate={navigate}>
      {user.role === "student" ? (
        <StudentEntry
          user={user}
          tab={tab}
          navigate={navigate}
          onSaved={onSaved}
        />
      ) : (
        <AdminWorkspace tab={tab} />
      )}
    </Shell>
  );
}
export default function Home() {
  const [user, setUser] = useState<User | null>(null),
    [loading, setLoading] = useState(true),
    [error, setError] = useState("");
  useEffect(() => {
    const abort = new AbortController();
    api<User>("/auth/me", { signal: abort.signal })
      .then(setUser)
      .catch((e) => {
        if (!abort.signal.aborted && !(e instanceof AuthError))
          setError(e.message);
      })
      .finally(() => {
        if (!abort.signal.aborted) setLoading(false);
      });
    return () => abort.abort();
  }, []);
  if (loading) return <Loading text="Opening Lumen…" />;
  if (user) return <Workspace user={user} onSaved={setUser} />;
  return (
    <>
      {error && (
        <Notice title="Couldn’t check your session" tone="error">
          {error}{" "}
          <Button variant="secondary" href="/login">
            Go to sign in
          </Button>
        </Notice>
      )}
      <Landing />
    </>
  );
}

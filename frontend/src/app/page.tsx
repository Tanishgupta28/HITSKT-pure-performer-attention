"use client";
import { useEffect, useState } from "react";
import { api, AuthError, type User } from "@/lib/api";
import { Button, Loading, Notice, Shell, useTab } from "@/components/product";
import { Landing } from "@/components/entry";
import { StudentWorkspace } from "@/components/student";
import { TeacherWorkspace } from "@/components/teacher";
import { AdminWorkspace } from "@/components/admin";

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
        <StudentWorkspace
          user={user}
          tab={tab}
          navigate={navigate}
          onSaved={onSaved}
        />
      ) : user.role === "teacher" ? (
        <TeacherWorkspace
          demo={user.demo}
          name={user.name}
          tab={tab}
          navigate={navigate}
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

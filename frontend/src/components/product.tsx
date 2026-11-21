"use client";

import Link from "next/link";
import {
  cloneElement,
  isValidElement,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from "react";
import {
  ArrowUpRight,
  ArrowRight,
  AlertTriangle,
  CircleHelp,
  X,
  Loader2,
  Menu,
  LogOut,
  LayoutDashboard,
  BookOpen,
  ClipboardList,
  CalendarDays,
  ChartNoAxesCombined,
  History,
  Database,
  Layers3,
  FlaskConical,
  UserRound,
} from "lucide-react";
import { api, send, type User } from "@/lib/api";

const TOPICS: Record<string, string> = {
  fractions: "Fractions",
  percentages: "Percentages",
  equations: "Linear equations",
  geometry: "Geometry",
};
export const topicName = (s?: string | null) =>
  (TOPICS[s ?? ""] ?? s ?? "Mixed topics").replaceAll("_", " ");
export const label = (s?: string | null) =>
  s ? s.replaceAll("_", " ").replace(/^./, (x) => x.toUpperCase()) : "—";
export const when = (value?: string | null) =>
  value
    ? new Intl.DateTimeFormat("en-IN", {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(new Date(value))
    : "—";
export const duration = (v?: number | null) =>
  v == null ? "—" : `${Math.floor(v / 60)}:${String(v % 60).padStart(2, "0")}`;
export function Brand() {
  return (
    <Link href="/" className="p-brand" aria-label="Lumen home">
      <span>
        <ArrowUpRight aria-hidden size={21} />
      </span>
      lumen<span className="p-brand-dot">.</span>
    </Link>
  );
}
export function Button({
  children,
  href,
  onClick,
  variant = "primary",
  disabled = false,
  type = "button",
}: {
  children: ReactNode;
  href?: string;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "light" | "danger" | "link";
  disabled?: boolean;
  type?: "button" | "submit";
}) {
  const className = `p-btn ${variant}`;
  return href && !disabled ? (
    <Link className={className} href={href}>
      {children}
    </Link>
  ) : (
    <button
      type={type}
      className={className}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  );
}
export function Badge({
  children,
  tone = "sage",
}: {
  children: ReactNode;
  tone?: "sage" | "amber" | "neutral" | "danger";
}) {
  return <span className={`p-badge ${tone}`}>{children}</span>;
}
export function Notice({
  title,
  children,
  tone = "warn",
}: {
  title: string;
  children?: ReactNode;
  tone?: "warn" | "error" | "ok";
}) {
  return (
    <div
      className={`p-notice ${tone}`}
      role={tone === "error" ? "alert" : "status"}
    >
      <AlertTriangle size={19} aria-hidden />
      <div>
        <strong>{title}</strong>
        {children && <div>{children}</div>}
      </div>
    </div>
  );
}
export function Heading({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <header className="p-heading">
      <div>
        <h1>{title}</h1>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {action && <div className="p-actions">{action}</div>}
    </header>
  );
}
export function Panel({
  title,
  subtitle,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`p-panel ${className}`}>
      {title && <h2>{title}</h2>}
      {subtitle && <p className="p-muted p-panel-sub">{subtitle}</p>}
      {children}
    </section>
  );
}
export function Stat({
  title,
  value,
  caption,
}: {
  title: string;
  value: ReactNode;
  caption?: string;
}) {
  return (
    <Panel className="p-stat">
      <div>{title}</div>
      <strong>{value}</strong>
      {caption && <small>{caption}</small>}
    </Panel>
  );
}
export function Empty({
  title,
  children,
  action,
}: {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <Panel className="p-empty">
      <CircleHelp size={24} aria-hidden />
      <h2>{title}</h2>
      {children && <p>{children}</p>}
      {action}
    </Panel>
  );
}
export function Loading({ text = "Loading workspace…" }: { text?: string }) {
  return (
    <div className="p-loading" role="status">
      <Loader2 aria-hidden size={21} />
      {text}
    </div>
  );
}
export function Field({
  label: caption,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  const generated = useId();
  const child = isValidElement(children)
    ? (children as ReactElement<{ id?: string; "aria-describedby"?: string }>)
    : null;
  const id = child?.props.id ?? generated;
  const help = hint ? `${id}-help` : undefined;
  const described =
    [child?.props["aria-describedby"], help].filter(Boolean).join(" ") ||
    undefined;
  return (
    <div className="p-field">
      <label htmlFor={id}>{caption}</label>
      {child
        ? cloneElement(child, { id, "aria-describedby": described })
        : children}
      {hint && <small id={help}>{hint}</small>}
    </div>
  );
}
export function Table({
  headers,
  rows,
  caption,
}: {
  headers: string[];
  rows: ReactNode[][];
  caption?: string;
}) {
  return (
    <div
      className="p-table"
      tabIndex={0}
      role="region"
      aria-label={caption ?? headers.join(", ")}
    >
      <table>
        <thead>
          <tr>
            {headers.map((h) => (
              <th key={h} scope="col">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td key={j} data-label={headers[j]}>
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {caption && <div className="p-table-caption">{caption}</div>}
    </div>
  );
}
export function EvidenceBars({
  items,
  highlightLow = true,
}: {
  items: { name: string; count: number; total: number }[];
  highlightLow?: boolean;
}) {
  return (
    <div className="p-bars">
      {items.map((x, i) => (
        <div className="p-bar-row" key={i}>
          <strong>{x.name}</strong>
          <div
            className="p-bar-track"
            role="img"
            aria-label={`${x.count} of ${x.total}`}
          >
            <i
              className={
                highlightLow && x.total && x.count / x.total < 0.5
                  ? "attention"
                  : ""
              }
              style={{
                width: `${x.total ? Math.min(x.count / x.total, 1) * 100 : 0}%`,
              }}
            />
          </div>
          <span>
            {x.count}/{x.total}
          </span>
        </div>
      ))}
    </div>
  );
}
export function Confirm({
  title,
  children,
  action,
  onConfirm,
  onClose,
  busy = false,
  danger = false,
}: {
  title: string;
  children: ReactNode;
  action: string;
  onConfirm: () => void;
  onClose: () => void;
  busy?: boolean;
  danger?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const current = ref.current;
    current?.showModal();
    return () => current?.close();
  }, []);
  return (
    <dialog
      className="p-dialog"
      ref={ref}
      onCancel={(e) => {
        e.preventDefault();
        if (!busy) onClose();
      }}
      aria-label={title}
    >
      <h2>{title}</h2>
      <div>{children}</div>
      <footer className="p-actions">
        <Button variant="secondary" onClick={onClose} disabled={busy}>
          Cancel
        </Button>
        <Button
          variant={danger ? "danger" : "primary"}
          onClick={onConfirm}
          disabled={busy}
        >
          {busy ? "Working…" : action}
        </Button>
      </footer>
    </dialog>
  );
}

export function useLoad<T>(path: string | null) {
  const [data, setData] = useState<T | null>(null),
    [busy, setBusy] = useState(Boolean(path)),
    [error, setError] = useState(""),
    [revision, setRevision] = useState(0);
  const reload = useCallback(() => setRevision((x) => x + 1), []);
  useEffect(() => {
    if (!path) {
      setData(null);
      setBusy(false);
      return;
    }
    const controller = new AbortController();
    setBusy(true);
    setData(null);
    setError("");
    api<T>(path, { signal: controller.signal })
      .then((d) => {
        if (!controller.signal.aborted) setData(d);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setBusy(false);
      });
    return () => controller.abort();
  }, [path, revision]);
  return { data, busy, error, reload };
}
export function Boundary({
  value,
  children,
}: {
  value: { busy: boolean; error: string; reload: () => void };
  children: ReactNode;
}) {
  if (value.error)
    return (
      <Notice title="Couldn’t load this view" tone="error">
        {value.error}{" "}
        <Button variant="secondary" onClick={value.reload}>
          Try again
        </Button>
      </Notice>
    );
  if (value.busy) return <Loading />;
  return <>{children}</>;
}
export function useAction() {
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [message, setMessage] = useState("");
  const locked = useRef(false);
  async function run<T>(fn: () => Promise<T>, success = "") {
    if (locked.current) return;
    locked.current = true;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const value = await fn();
      if (success) setMessage(success);
      return value;
    } catch (e) {
      setError((e as Error).message);
    } finally {
      locked.current = false;
      setBusy(false);
    }
  }
  return { busy, error, message, run, setError };
}
export function ActionStatus({ a }: { a: ReturnType<typeof useAction> }) {
  return a.error ? (
    <Notice title="Action could not be completed" tone="error">
      {a.error}
    </Notice>
  ) : a.message ? (
    <Notice title={a.message} tone="ok" />
  ) : null;
}

const NS = {
  student: [
    ["today", "Today", LayoutDashboard],
    ["practice", "Learn", BookOpen],
    ["progress", "Progress", ChartNoAxesCombined],
  ],
  admin: [
    ["questions", "Question bank", ClipboardList],
    ["skills", "Topics & skills", Layers3],
    ["interactions", "Interactions", Database],
    ["lab", "Model Lab", FlaskConical],
  ],
} as const;
export function useTab(role: User["role"]) {
  const first = NS[role][0][0];
  const [tab, setTab] = useState<string>(first);
  useEffect(() => {
    function read() {
      const candidate = new URLSearchParams(location.search).get("view");
      const resolved =
        role === "student" &&
        (candidate === "revision" || candidate === "assessments")
          ? "practice"
          : role === "student" && candidate === "history"
            ? "progress"
            : candidate;
      setTab(
        resolved === "profile" && role === "student"
          ? "profile"
          : NS[role].some(([k]) => k === resolved)
            ? resolved!
            : first,
      );
    }
    read();
    addEventListener("popstate", read);
    return () => removeEventListener("popstate", read);
  }, [role, first]);
  const navigate = useCallback((next: string) => {
    history.pushState({}, "", `/?view=${encodeURIComponent(next)}`);
    setTab(next);
    window.scrollTo({ top: 0, behavior: "instant" });
  }, []);
  return { tab, navigate };
}
export function Shell({
  user,
  tab,
  navigate,
  children,
}: {
  user: User;
  tab: string;
  navigate: (s: string) => void;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const a = useAction();
  const nav = NS[user.role];
  const current =
    tab === "profile"
      ? "Profile"
      : (nav.find(([id]) => id === tab)?.[1] ?? nav[0][1]);
  async function logout() {
    await a.run(async () => {
      await send("/auth/logout");
      location.assign("/");
    });
  }
  return (
    <div className="p-shell">
      <a href="#main" className="p-skip">
        Skip to content
      </a>
      {open && (
        <button
          className="p-overlay"
          onClick={() => setOpen(false)}
          aria-label="Close navigation"
        />
      )}
      <aside className={`p-sidebar ${open ? "open" : ""}`}>
        <div className="p-side-brand">
          <Brand />
          <button
            className="p-icon-btn p-side-close"
            aria-label="Close navigation"
            onClick={() => setOpen(false)}
          >
            <X size={22} />
          </button>
        </div>
        <div className="p-eyebrow">
          {user.role === "admin" ? "ADMINISTRATOR" : user.role.toUpperCase()}{" "}
          WORKSPACE
        </div>
        <nav aria-label="Workspace">
          {nav.map(([id, title, Icon]) => (
            <button
              key={id}
              onClick={() => {
                navigate(id);
                setOpen(false);
              }}
              aria-current={tab === id ? "page" : undefined}
            >
              <Icon size={20} aria-hidden />
              <span>{title}</span>
            </button>
          ))}
        </nav>
        <footer className="p-side-foot">
          <button
            aria-label={user.role === "student" ? "Open profile" : undefined}
            onClick={() => {
              if (user.role === "student") {
                navigate("profile");
                setOpen(false);
              }
            }}
            disabled={user.role !== "student"}
            className="p-user"
          >
            <span className="p-avatar">
              {user.name
                .split(" ")
                .slice(0, 2)
                .map((n) => n[0])
                .join("")}
            </span>
            <span>
              {user.name}
              <small>
                {user.role === "student" ? "Profile" : label(user.role)}
              </small>
            </span>
          </button>
          <button className="p-icon-btn" aria-label="Sign out" onClick={logout}>
            <LogOut size={19} />
          </button>
        </footer>
      </aside>
      <div className="p-work">
        <header className="p-top">
          <div>
            <button
              className="p-icon-btn p-menu"
              onClick={() => setOpen(true)}
              aria-label="Open navigation"
              aria-expanded={open}
            >
              <Menu size={23} />
            </button>
            <span>{label(user.role)}</span>
            <span aria-hidden>/</span>
            <strong>{current}</strong>
          </div>
          {user.demo && <Badge tone="amber">Synthetic demo</Badge>}
        </header>
        <main id="main" className="p-main" tabIndex={-1}>
          <ActionStatus a={a} />
          {children}
        </main>
      </div>
      {user.role === "student" && (
        <nav className="p-mobile-nav" aria-label="Quick navigation">
          {[
            ["today", "Today", LayoutDashboard],
            ["practice", "Learn", BookOpen],
            ["progress", "Progress", ChartNoAxesCombined],
          ].map(([id, title, Icon]) => {
            const I = Icon as typeof BookOpen;
            return (
              <button
                key={String(id)}
                aria-current={tab === id ? "page" : undefined}
                onClick={() => navigate(String(id))}
              >
                <I size={20} />
                {String(title)}
              </button>
            );
          })}
          <button onClick={() => setOpen(true)}>
            <Menu size={20} />
            More
          </button>
        </nav>
      )}
    </div>
  );
}
export function NextArrow() {
  return <ArrowRight size={18} aria-hidden />;
}

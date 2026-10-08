"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { CirclePause, CirclePlay } from "lucide-react";

const MotionContext = createContext({
  paused: true,
  reduced: false,
  ready: false,
  toggle: () => {},
});
export const useMotionPaused = () => useContext(MotionContext).paused;

export function MotionToggle() {
  const { paused, reduced, ready, toggle } = useContext(MotionContext);
  if (!ready || reduced) return null;
  return (
    <button
      className="motion-toggle"
      onClick={toggle}
      aria-label={paused ? "Resume animations" : "Pause animations"}
      aria-pressed={!paused}
    >
      {paused ? <CirclePlay size={20} /> : <CirclePause size={20} />}
      <span>Motion {paused ? "off" : "on"}</span>
    </button>
  );
}

export function MotionProvider({ children }: { children: React.ReactNode }) {
  const [requestedPause, setRequestedPause] = useState(true);
  const [reduced, setReduced] = useState(false);
  const [ready, setReady] = useState(false);
  const paused = requestedPause || reduced;
  useEffect(() => {
    const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(preference.matches);
    update();
    try {
      setRequestedPause(localStorage.getItem("lumen-motion-paused") === "true");
    } catch {
      setRequestedPause(false);
    }
    setReady(true);
    preference.addEventListener("change", update);
    return () => preference.removeEventListener("change", update);
  }, []);
  useEffect(() => {
    document.documentElement.dataset.motion = paused ? "paused" : "playing";
  }, [paused]);
  function toggle() {
    const value = !requestedPause;
    setRequestedPause(value);
    try {
      localStorage.setItem("lumen-motion-paused", String(value));
    } catch {
      /* Preferences still work in this tab. */
    }
  }
  return (
    <MotionContext.Provider value={{ paused, reduced, ready, toggle }}>
      {children}
    </MotionContext.Provider>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";
import type { AnimationItem } from "lottie-web";
import { ArrowUpRight, Check, Sparkles } from "lucide-react";
import { BrandMark } from "./brand";
import { useMotionPaused } from "./motion";
import orbit from "@/assets/lumen-orbit.json";

export function GrowthScene({
  variant = "hero",
}: {
  variant?: "hero" | "auth";
}) {
  const container = useRef<HTMLDivElement>(null);
  const animation = useRef<AnimationItem | null>(null);
  const paused = useMotionPaused();
  const pauseRef = useRef(paused);
  const visible = useRef(true);
  const [loaded, setLoaded] = useState(false);
  function sync() {
    const player = animation.current;
    if (!player || !container.current) return;
    const stop = pauseRef.current || !visible.current || document.hidden;
    if (stop) player.pause();
    else player.play();
    container.current.dataset.playback = stop ? "paused" : "playing";
  }
  useEffect(() => {
    pauseRef.current = paused;
    sync();
  }, [paused]);
  useEffect(() => {
    const element = container.current;
    if (!element) return;
    let cancelled = false;
    const observer = new IntersectionObserver(
      ([entry]) => {
        visible.current = entry.isIntersecting;
        sync();
      },
      { threshold: 0.05 },
    );
    observer.observe(element);
    document.addEventListener("visibilitychange", sync);
    import("lottie-web")
      .then(({ default: lottie }) => {
        if (cancelled) return;
        const player = lottie.loadAnimation({
          container: element,
          renderer: "svg",
          loop: true,
          autoplay: false,
          animationData: structuredClone(orbit),
          rendererSettings: {
            preserveAspectRatio: "xMidYMid meet",
            progressiveLoad: true,
          },
        });
        animation.current = player;
        player.addEventListener("DOMLoaded", () => {
          if (cancelled) return;
          player.goToAndStop(35, true);
          setLoaded(true);
          sync();
        });
      })
      .catch(() => {
        /* The static sculpture remains visible if motion cannot load. */
      });
    return () => {
      cancelled = true;
      observer.disconnect();
      document.removeEventListener("visibilitychange", sync);
      animation.current?.destroy();
      animation.current = null;
    };
  }, []);
  return (
    <div className={`growth-scene growth-scene--${variant}`} aria-hidden="true">
      <div className="scene-glow" />
      <div className="scene-orbit orbit-outer" />
      <div className="scene-orbit orbit-inner" />
      <div
        ref={container}
        className={`scene-lottie ${loaded ? "loaded" : ""}`}
        data-lottie="lumen-orbit"
        data-playback="paused"
      />
      <div className="scene-sculpture">
        <BrandMark className="brand-mark--sculpture" />
      </div>
      <div className="scene-platform" />
      <span className="scene-chip chip-growth">
        <span>
          <Check size={18} />
        </span>
        Room to grow
        <ArrowUpRight size={17} />
      </span>
      <span className="scene-chip chip-idea">
        <Sparkles size={21} />A little brighter.
      </span>
      <span className="scene-glyph glyph-plus">+</span>
      <span className="scene-glyph glyph-times">×</span>
      <span className="scene-glyph glyph-math">a²</span>
    </div>
  );
}

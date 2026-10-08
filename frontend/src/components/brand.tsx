import { Sprout } from "lucide-react";
export function BrandMark({ className = "" }: { className?: string }) {
  return (
    <span className={`brand-mark ${className}`} aria-hidden="true">
      <span className="brand-mark-face">
        <Sprout strokeWidth={1.85} />
        <span className="brand-mark-glint" />
      </span>
    </span>
  );
}
export function Brand() {
  return (
    <span className="brand">
      <BrandMark />
      <span className="brand-word">
        lumen<span className="brand-dot">.</span>
      </span>
    </span>
  );
}

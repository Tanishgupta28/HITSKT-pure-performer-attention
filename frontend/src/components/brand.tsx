import { Sprout } from "lucide-react";
export function Brand() {
  return (
    <span className="brand">
      <span className="brand-mark">
        <Sprout size={25} strokeWidth={2.1} />
      </span>
      lumen<span className="brand-dot">.</span>
    </span>
  );
}

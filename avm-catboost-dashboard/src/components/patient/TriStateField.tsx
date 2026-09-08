"use client";

import type { TriState } from "@/types/patient";
import { FeatureTooltip } from "./FeatureTooltip";

interface TriStateFieldProps {
  label: string;
  definition: string;
  value: TriState;
  onChange: (v: TriState) => void;
}

const OPTIONS: { value: TriState; label: string }[] = [
  { value: "yes", label: "Yes" },
  { value: "no", label: "No" },
  { value: "unknown", label: "N/A" }
];

export function TriStateField({
  label,
  definition,
  value,
  onChange
}: TriStateFieldProps) {
  return (
    <div>
      <label className="text-sm font-medium text-slate-700">{label}</label>
      <FeatureTooltip text={definition} />

      <div className="mt-2 inline-flex rounded-xl border border-slate-300 p-1">
        {OPTIONS.map((opt) => {
          const active = value === opt.value;
          return (
            <button
              key={opt.value}
              type="button"
              onClick={() => onChange(opt.value)}
              className={
                active
                  ? "rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white"
                  : "rounded-lg px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100"
              }
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

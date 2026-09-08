"use client";

import type { AnalysisMode } from "@/types/patient";

interface ModeToggleProps {
  mode: AnalysisMode;
  onChange: (m: AnalysisMode) => void;
}

const MODES: { value: AnalysisMode; label: string; sub: string }[] = [
  {
    value: "classification",
    label: "Hemorrhagic presentation",
    sub: "Cross-sectional, 8 inputs"
  },
  {
    value: "longitudinal",
    label: "Time to rupture",
    sub: "Longitudinal, 3 inputs"
  }
];

export function ModeToggle({ mode, onChange }: ModeToggleProps) {
  return (
        <div className="flex w-full gap-2 rounded-2xl border border-slate-200 bg-white p-2 shadow-sm">
      {MODES.map((m) => {
        const active = mode === m.value;
        return (
          <button
            key={m.value}
            type="button"
            onClick={() => onChange(m.value)}
            className={
              active
                ? "flex-1 rounded-xl bg-slate-900 px-5 py-3 text-left text-white"
                : "flex-1 rounded-xl px-5 py-3 text-left text-slate-600 hover:bg-slate-50"
            }
          >
            <span className="block text-sm font-semibold">{m.label}</span>
            <span
              className={
                active
                  ? "block text-xs text-slate-300"
                  : "block text-xs text-slate-400"
              }
            >
              {m.sub}
            </span>
          </button>
        );
      })}
    </div>
  );
}
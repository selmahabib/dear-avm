"use client";

export type RiskView = "model" | "score";

interface RiskViewToggleProps {
  view: RiskView;
  onChange: (view: RiskView) => void;
}

const options: { value: RiskView; label: string; hint: string }[] = [
  {
    value: "model",
    label: "Ridge Cox model",
    hint: "Continuous probability at 2 and 5 years"
  },
  {
    value: "score",
    label: "Integer score",
    hint: "0-5 points, three risk tiers"
  }
];

export function RiskViewToggle({ view, onChange }: RiskViewToggleProps) {
  return (
    <div>
            <div className="flex w-full rounded-xl border border-slate-200 bg-slate-50 p-1">
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            onClick={() => onChange(o.value)}
            aria-pressed={view === o.value}
                        className={`flex-1 rounded-lg px-4 py-2 text-sm font-medium transition ${
              view === o.value
                ? "bg-white text-slate-950 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>
      <p className="mt-2 text-xs text-slate-500">
        {options.find((o) => o.value === view)?.hint}
        {" \u00b7 "}
        Both come from the same three predictors; the score is the model&apos;s
        coefficients rounded to whole points for bedside use.
      </p>
    </div>
  );
}

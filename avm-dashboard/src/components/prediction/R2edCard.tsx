import type { R2edResult } from "@/lib/r2ed";

export function R2edCard({ result }: { result: R2edResult }) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-baseline justify-between">
        <h3 className="text-lg font-semibold text-slate-950">
          R<sub>2</sub>eD score
        </h3>
        <span className="text-3xl font-bold text-slate-950">
          {result.score}
          <span className="text-lg font-normal text-slate-400">
            /{result.maxScore}
          </span>
        </span>
      </div>

      <p className="mt-2 text-sm text-slate-600">
        Published comparator score, implemented exactly as specified.
      </p>

      <ul className="mt-5 space-y-2">
        {result.components.map((c) => (
          <li
            key={c.label}
            className="flex items-center justify-between border-b border-slate-100 pb-2 text-sm"
          >
            <span className="text-slate-700">{c.label}</span>
            <span
              className={
                c.awarded
                  ? "font-semibold text-slate-950"
                  : c.unknown
                    ? "text-amber-600"
                    : "text-slate-400"
              }
            >
              {c.awarded ? `+${c.points}` : c.unknown ? "N/A (0)" : "0"}
            </span>
          </li>
        ))}
      </ul>

      {result.unknownCount > 0 && (
        <p className="mt-4 text-xs text-amber-700">
          {result.unknownCount} undocumented finding
          {result.unknownCount > 1 ? "s" : ""} scored as no point awarded.
        </p>
      )}
    </section>
  );
}

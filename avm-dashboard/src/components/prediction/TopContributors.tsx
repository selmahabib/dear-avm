import type { ShapContribution } from "@/types/shap";
import { formatSigned } from "@/lib/utils";

interface TopContributorsProps {
  shapValues: ShapContribution[];
}

export function TopContributors({ shapValues }: TopContributorsProps) {
  const positive = shapValues
    .filter((item) => item.value > 0)
    .sort((a, b) => b.value - a.value)
    .slice(0, 4);

  const negative = shapValues
    .filter((item) => item.value < 0)
    .sort((a, b) => a.value - b.value)
    .slice(0, 4);

  return (
    <section className="grid gap-4 md:grid-cols-2">
      <ContributorColumn
        title="Pushing toward hemorrhagic presentation"
        items={positive}
        tone="risk"
      />

      <ContributorColumn
        title="Pushing away from hemorrhagic presentation"
        items={negative}
        tone="protective"
      />
    </section>
  );
}

interface ContributorColumnProps {
  title: string;
  items: ShapContribution[];
  tone: "risk" | "protective";
}

function ContributorColumn({ title, items, tone }: ContributorColumnProps) {
  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-slate-950">{title}</h3>

      <div className="mt-5 space-y-3">
        {items.length > 0 ? (
          items.map((item) => (
            <div
              key={item.feature}
              className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3"
            >
              <div>
                <p className="font-medium text-slate-800">{item.feature}</p>
                <p className="text-xs text-slate-500">
                  Value: {String(item.featureValue)}
                </p>
              </div>

              <span
                className={
                  tone === "risk"
                    ? "font-semibold text-red-600"
                    : "font-semibold text-emerald-600"
                }
              >
                {formatSigned(item.value)}
              </span>
            </div>
          ))
        ) : (
          <p className="text-sm text-slate-500">
            No major contributors in this direction.
          </p>
        )}
      </div>
    </div>
  );
}
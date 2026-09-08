import type { RuptureRiskResult, RuptureRiskTier } from "@/types/ruptureRisk";
import { formatPercent1 } from "@/lib/utils";

import type { RiskView } from "@/components/layout/RiskViewToggle";

interface RuptureRiskCardProps {
  risk: RuptureRiskResult;
  view: RiskView;
}

const tierStyles: Record<RuptureRiskTier, string> = {
  Low: "text-emerald-600",
  Medium: "text-amber-600",
  High: "text-red-600"
};

export function RuptureRiskCard({ risk, view }: RuptureRiskCardProps) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <p className="text-sm font-medium text-slate-500">
        Future Rupture Risk &middot;{" "}
        {view === "model" ? "Ridge Cox model" : "Integer score"}
      </p>

      {view === "model" ? (
      <>
      <div className="mt-4 grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            2-year risk
          </p>
          <p className="mt-1 text-3xl font-bold text-slate-950">
            {formatPercent1(risk.risk2Year)}
          </p>
        </div>

        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            5-year risk
          </p>
          <p className="mt-1 text-3xl font-bold text-slate-950">
            {formatPercent1(risk.risk5Year)}
          </p>
        </div>
      </div>

      <div className="mt-5 flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3">
        <span className="text-sm text-slate-600">Risk tier</span>

        <span className={`text-sm font-semibold ${tierStyles[risk.tier]}`}>
          {risk.tier} (score {risk.score}/5)
        </span>
      </div>

      <p className="mt-4 text-xs text-slate-500">
        Ridge-penalized Cox model on age, prior rupture at presentation, and
        exclusively deep nidus location, using the fitted coefficients and
        baseline survival. Reflects risk during longitudinal follow-up, not
        risk at diagnosis.
      </p>
      </>
      ) : (
      <>
      <div className="mt-4 flex items-baseline gap-3">
        <p className="text-5xl font-bold text-slate-950">{risk.score}</p>
        <p className="text-sm text-slate-500">out of 5 points</p>
      </div>

      <div className="mt-5 flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3">
        <span className="text-sm text-slate-600">Risk tier</span>
        <span className={`text-sm font-semibold ${tierStyles[risk.tier]}`}>
          {risk.tier} &middot; {risk.annualRatePercent}%/year
          {risk.annualRateCI
            ? ` (95% CI ${risk.annualRateCI[0]}-${risk.annualRateCI[1]})`
            : null}
        </span>
      </div>

      <p className="mt-4 text-xs text-slate-500">
        One point each for exclusively deep nidus location and prior rupture,
        plus 0-3 for age (&lt;27, 27-44, 45-62, &ge;63). The annual rate is the
        observed rate in that tier in the derivation cohort, not a prediction
        for this patient. Scores 0-1 are Low, 2-3 Medium, 4-5 High.
      </p>
      </>
      )}
    </section>
  );
}

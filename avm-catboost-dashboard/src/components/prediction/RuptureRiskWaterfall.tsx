import type { RuptureRiskResult } from "@/types/ruptureRisk";
import { formatPercent1, formatSigned } from "@/lib/utils";

interface RuptureRiskWaterfallProps {
  risk: RuptureRiskResult;
}

type WaterfallStep = RuptureRiskResult["contributions"][number] & {
  start: number;
  end: number;
};

export function RuptureRiskWaterfall({ risk }: RuptureRiskWaterfallProps) {
  const baseValue = 0; // linear predictor of a reference patient at each covariate's cohort mean
  let runningValue = baseValue;

  const orderedContributions = [...risk.contributions].sort(
    (a, b) => Math.abs(b.value) - Math.abs(a.value)
  );

  const steps: WaterfallStep[] = orderedContributions.map((item) => {
    const start = runningValue;
    const end = runningValue + item.value;

    runningValue = end;

    return {
      ...item,
      start,
      end
    };
  });

  const finalValue = runningValue;

  const minValue = Math.min(
    baseValue,
    finalValue,
    ...steps.flatMap((step) => [step.start, step.end])
  );

  const maxValue = Math.max(
    baseValue,
    finalValue,
    ...steps.flatMap((step) => [step.start, step.end])
  );

  const range = maxValue - minValue || 1;

  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <p className="text-sm font-medium text-slate-500">
        Local Interpretability
      </p>

      <h3 className="mt-1 text-xl font-bold text-slate-950">
        Cox Hazard Contribution
      </h3>

      <p className="mt-2 text-sm text-slate-600">
        Starting from a reference patient at each predictor&apos;s cohort
        mean, each feature moves this patient&apos;s log hazard ratio up or
        down.
      </p>

      <div className="mt-8 space-y-5">

        {steps.map((step) => (
          <WaterfallRow
            key={step.feature}
            label={step.feature}
            sublabel={String(step.featureValue)}
            start={step.start}
            end={step.end}
            value={step.value}
            minValue={minValue}
            range={range}
          />
        ))}

        <WaterfallRow
          label="This patient"
          sublabel="Final linear predictor"
          start={finalValue}
          end={finalValue}
          minValue={minValue}
          range={range}
          final
        />
      </div>

      <div className="mt-6 space-y-2 rounded-2xl bg-slate-50 px-5 py-4 text-sm text-slate-600">
        <div className="flex justify-between">
          <span>
            Linear predictor (&eta;):{" "}
            <strong className="text-slate-950">
              {formatSigned(finalValue)}
            </strong>
          </span>

          <span>
            Hazard ratio:{" "}
            <strong className="text-slate-950">
              {Math.exp(finalValue).toFixed(2)}&times; baseline
            </strong>
          </span>
        </div>

        <div className="flex justify-between border-t border-slate-200 pt-2">
          <span>
            2-year risk:{" "}
            <strong className="text-slate-950">
              {formatPercent1(risk.risk2Year)}
            </strong>
          </span>

          <span>
            5-year risk:{" "}
            <strong className="text-slate-950">
              {formatPercent1(risk.risk5Year)}
            </strong>
          </span>
        </div>
      </div>
    </section>
  );
}

interface WaterfallRowProps {
  label: string;
  sublabel: string;
  start: number;
  end: number;
  value?: number;
  minValue: number;
  range: number;
  neutral?: boolean;
  final?: boolean;
}

function WaterfallRow({
  label,
  sublabel,
  start,
  end,
  value,
  minValue,
  range,
  neutral,
  final
}: WaterfallRowProps) {
  function xPosition(input: number) {
    return ((input - minValue) / range) * 100;
  }

  const startX = xPosition(start);
  const endX = xPosition(end);

  const left = Math.min(startX, endX);
  const width = Math.max(Math.abs(endX - startX), neutral || final ? 0 : 2);

  const isPositive = typeof value === "number" && value >= 0;

  return (
    <div className="grid grid-cols-[180px_1fr_90px] items-center gap-4">
      <div>
        <p className="text-sm font-semibold text-slate-800">{label}</p>
        <p className="text-xs text-slate-500">{sublabel}</p>
      </div>

      <div className="relative h-11 rounded-xl bg-slate-100">
        <div
          className="absolute top-0 h-11 w-px bg-slate-300"
          style={{ left: `${xPosition(0)}%` }}
        />

        {neutral || final ? (
          <div
            className={
              final
                ? "absolute top-1.5 h-8 w-1 rounded-full bg-slate-950"
                : "absolute top-1.5 h-8 w-1 rounded-full bg-slate-500"
            }
            style={{ left: `${startX}%` }}
          />
        ) : (
          <>
            <div
              className={
                isPositive
                  ? "absolute top-2 h-7 rounded-lg bg-red-500"
                  : "absolute top-2 h-7 rounded-lg bg-emerald-500"
              }
              style={{
                left: `${left}%`,
                width: `${width}%`
              }}
            />

            <div
              className="absolute top-5 h-px bg-slate-300"
              style={{
                left: `${Math.min(startX, endX)}%`,
                width: `${Math.abs(endX - startX)}%`
              }}
            />
          </>
        )}
      </div>

      <p
        className={
          final || neutral
            ? "text-right text-sm font-bold text-slate-800"
            : isPositive
              ? "text-right text-sm font-bold text-red-600"
              : "text-right text-sm font-bold text-emerald-600"
        }
      >
        {neutral || final ? formatSigned(start) : formatSigned(value ?? 0)}
      </p>
    </div>
  );
}

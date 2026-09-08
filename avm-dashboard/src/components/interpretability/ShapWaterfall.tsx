import type { ShapContribution } from "@/types/shap";
import { formatSigned } from "@/lib/utils";

interface ShapWaterfallProps {
  shapValues: ShapContribution[];
  // The model's expected log-odds over the training cohort. SHAP values are
  // deviations from this, so the waterfall has to start here to land on the
  // predicted probability.
  baseValue: number;
}

type WaterfallStep = ShapContribution & {
  start: number;
  end: number;
};

export function ShapWaterfall({ shapValues, baseValue }: ShapWaterfallProps) {
  let runningValue = baseValue;

  const orderedValues = [...shapValues].sort(
    (a, b) => Math.abs(b.value) - Math.abs(a.value)
  );

  const steps: WaterfallStep[] = orderedValues.map((item) => {
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

  function xPosition(value: number) {
    return ((value - minValue) / range) * 100;
  }

  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <p className="text-sm font-medium text-slate-500">
        Local Interpretability
      </p>

      <h3 className="mt-1 text-xl font-bold text-slate-950">
        SHAP Waterfall : what's raising or lowering this patient's risk?
      </h3>

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
          label="Final model output"
          sublabel="After all feature contributions"
          start={finalValue}
          end={finalValue}
          minValue={minValue}
          range={range}
          final
        />
      </div>

      <div className="mt-6 flex justify-between rounded-2xl bg-slate-50 px-5 py-4 text-sm text-slate-600">
        <span>
          Baseline:{" "}
          <strong className="text-slate-950">
            {formatSigned(baseValue)}
          </strong>
        </span>

        <span>
          Final output:{" "}
          <strong className="text-slate-950">
            {formatSigned(finalValue)}
          </strong>
        </span>
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

        {(neutral || final) ? (
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
        {neutral || final
          ? formatSigned(start)
          : formatSigned(value ?? 0)}
      </p>
    </div>
  );
}
import type { PredictionResponse } from "@/types/prediction";
import { formatPercent } from "@/lib/utils";

interface ThresholdIndicatorProps {
  prediction: PredictionResponse;
}

export function ThresholdIndicator({
  prediction
}: ThresholdIndicatorProps) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-slate-950">
        Decision Threshold
      </h3>

      <div className="mt-6">
        <div className="relative h-4 rounded-full bg-slate-200">
          <div
            className="absolute top-0 h-4 rounded-full bg-red-500"
            style={{
              width: `${prediction.probability * 100}%`
            }}
          />

          <div
            className="absolute top-[-6px] h-7 w-[2px] bg-slate-950"
            style={{
              left: `${prediction.threshold * 100}%`
            }}
          />
        </div>

        <div className="mt-4 flex justify-between text-sm text-slate-600">
          <span>0%</span>

          <span>
            Threshold: {formatPercent(prediction.threshold)}
          </span>

          <span>100%</span>
        </div>
      </div>
    </section>
  );
}
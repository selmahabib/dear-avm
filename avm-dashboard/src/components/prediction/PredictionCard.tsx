import type { PredictionResponse } from "@/types/prediction";
import { formatPercent } from "@/lib/utils";

interface PredictionCardProps {
  prediction: PredictionResponse;
}

export function PredictionCard({
  prediction
}: PredictionCardProps) {
  const isPositive =
    prediction.label === "Hemorrhagic presentation";

  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <p className="text-sm font-medium text-slate-500">
        Model Prediction
      </p>

      <div className="mt-4 flex items-end justify-between">
        <div>
          <h2
            className={`text-3xl font-bold ${
              isPositive ? "text-red-600" : "text-emerald-600"
            }`}
          >
            {prediction.label}
          </h2>

          <p className="mt-2 text-slate-600">
            Probability:
            <span className="ml-2 font-semibold text-slate-950">
              {formatPercent(prediction.probability)}
            </span>
          </p>
        </div>
      </div>
    </section>
  );
}
"use client";

import { useState } from "react";
import type { LongitudinalFeatures } from "@/types/patient";
import { FeatureTooltip } from "./FeatureTooltip";

interface LongitudinalFormProps {
  onSubmit: (features: LongitudinalFeatures) => void;
  loading: boolean;
}

export function LongitudinalForm({ onSubmit, loading }: LongitudinalFormProps) {
  const [form, setForm] = useState<LongitudinalFeatures>({
    age: 36,
    exclusivelyDeepLocation: false,
    priorRupture: false
  });

  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="text-2xl font-bold text-slate-950">Patient features</h2>

      <p className="mt-2 text-sm text-slate-600">
        Three inputs, all required. With only three predictors there is no
        undocumented option: an unanswered field would leave too little to
        predict from.
      </p>

      <div className="mt-8 space-y-5">
        <div>
          <label className="text-sm font-medium text-slate-700">
            Age at diagnosis
          </label>
          <input
            type="number"
            value={form.age}
            onChange={(e) =>
              setForm((prev) => ({ ...prev, age: Number(e.target.value) }))
            }
            className="mt-2 w-full rounded-xl border border-slate-300 px-4 py-3"
          />
          <FeatureTooltip text="Patient age at diagnosis, in years." />
        </div>

        <div>
          <label className="flex items-center gap-3">
            <input
              type="checkbox"
              checked={form.exclusivelyDeepLocation}
              onChange={(e) =>
                setForm((prev) => ({
                  ...prev,
                  exclusivelyDeepLocation: e.target.checked
                }))
              }
              className="h-5 w-5 rounded border-slate-300"
            />
            <span className="text-sm font-medium text-slate-700">
              Exclusively deep nidus location
            </span>
          </label>
          <FeatureTooltip text="Nonlobar involvement with no lobar component. Stricter than the deep location used in the classification model." />
        </div>

        <div>
          <label className="flex items-center gap-3">
            <input
              type="checkbox"
              checked={form.priorRupture}
              onChange={(e) =>
                setForm((prev) => ({ ...prev, priorRupture: e.target.checked }))
              }
              className="h-5 w-5 rounded border-slate-300"
            />
            <span className="text-sm font-medium text-slate-700">
              Ruptured at presentation
            </span>
          </label>
          <FeatureTooltip text="Hemorrhage at or before diagnosis." />
        </div>
      </div>

      <button
        type="button"
        disabled={loading}
        onClick={() => onSubmit(form)}
        className="mt-6 w-full rounded-xl bg-slate-900 px-4 py-3 font-medium text-white disabled:opacity-50"
      >
        {loading ? "Running..." : "Estimate rupture risk"}
      </button>
    </section>
  );
}

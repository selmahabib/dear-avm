"use client";

import { useState } from "react";
import type { ClassificationFeatures, Race, TriState } from "@/types/patient";
import { featureDefinitions, featureLabels } from "@/lib/features";
import { FeatureTooltip } from "./FeatureTooltip";
import { TriStateField } from "./TriStateField";

interface ClassificationFormProps {
  onSubmit: (features: ClassificationFeatures) => void;
  loading: boolean;
}

const TRI_FIELDS: {
  key: keyof ClassificationFeatures;
  label: string;
  definition: string;
}[] = [
  {
    key: "deepLocation",
    label: featureLabels.deepLocation,
    definition: featureDefinitions.deepLocation
  },
  {
    key: "exclusiveDeepDrainage",
    label: featureLabels.exclusiveDeepDrainage,
    definition: featureDefinitions.exclusiveDeepDrainage
  },
  {
    key: "singleFeedingArtery",
    label: featureLabels.singleFeedingArtery,
    definition: featureDefinitions.singleFeedingArtery
  },
  {
    key: "intranidalAneurysm",
    label: featureLabels.intranidalAneurysm,
    definition: featureDefinitions.intranidalAneurysm
  },
  {
    key: "hypertension",
    label: featureLabels.hypertension,
    definition: featureDefinitions.hypertension
  }
];

export function ClassificationForm({
  onSubmit,
  loading
}: ClassificationFormProps) {
  const [form, setForm] = useState<ClassificationFeatures>({
    age: 36,
    sizeCm: 2.5,
    race: "white",
    deepLocation: "no",
    hypertension: "no",
    exclusiveDeepDrainage: "no",
    intranidalAneurysm: "no",
    singleFeedingArtery: "no"
  });

  const unknownCount = TRI_FIELDS.filter(
    (f) => form[f.key] === "unknown"
  ).length;

  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="text-2xl font-bold text-slate-950">Patient features</h2>

      <p className="mt-2 text-sm text-slate-600">
        Eight inputs. Findings that are not documented in the report can be
        left as N/A.
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
          <FeatureTooltip text={featureDefinitions.age} />
        </div>

        <div>
          <label className="text-sm font-medium text-slate-700">
            Maximum nidus size (cm)
          </label>
          <input
            type="number"
            step="0.1"
            value={form.sizeCm}
            onChange={(e) =>
              setForm((prev) => ({ ...prev, sizeCm: Number(e.target.value) }))
            }
            className="mt-2 w-full rounded-xl border border-slate-300 px-4 py-3"
          />
          <FeatureTooltip text={featureDefinitions.sizeCm} />
        </div>

        <div>
          <label className="text-sm font-medium text-slate-700">Race</label>
          <select
            value={form.race}
            onChange={(e) =>
              setForm((prev) => ({ ...prev, race: e.target.value as Race }))
            }
            className="mt-2 w-full rounded-xl border border-slate-300 px-4 py-3"
          >
            <option value="white">White</option>
            <option value="nonwhite">Non-white</option>
          </select>
          <FeatureTooltip text={featureDefinitions.race} />
        </div>

        {TRI_FIELDS.map((f) => (
          <TriStateField
            key={f.key}
            label={f.label}
            definition={f.definition}
            value={form[f.key] as TriState}
            onChange={(v) => setForm((prev) => ({ ...prev, [f.key]: v }))}
          />
        ))}
      </div>

      {unknownCount > 0 && (
        <p className="mt-6 rounded-xl bg-amber-50 px-4 py-3 text-xs text-amber-900">
          {unknownCount} finding{unknownCount > 1 ? "s" : ""} marked N/A.
          CatBoost routes undocumented values natively; the R2eD score awards
          no point for an undocumented finding, matching how an incomplete
          angiogram report would be scored.
        </p>
      )}

      <button
        type="button"
        disabled={loading}
        onClick={() => onSubmit(form)}
        className="mt-6 w-full rounded-xl bg-slate-900 px-4 py-3 font-medium text-white disabled:opacity-50"
      >
        {loading ? "Running..." : "Run prediction"}
      </button>
    </section>
  );
}

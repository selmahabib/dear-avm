"use client";

import { useState } from "react";
import { AppHeader } from "@/components/layout/AppHeader";
import { PageShell } from "@/components/layout/PageShell";
import { ModeToggle } from "@/components/layout/ModeToggle";
import { ClassificationForm } from "@/components/patient/ClassificationForm";
import { LongitudinalForm } from "@/components/patient/LongitudinalForm";
import { PredictionCard } from "@/components/prediction/PredictionCard";
import { ThresholdIndicator } from "@/components/prediction/ThresholdIndicator";
import { TopContributors } from "@/components/prediction/TopContributors";
import { RuptureRiskCard } from "@/components/prediction/RuptureRiskCard";
import { RiskViewToggle, type RiskView } from "@/components/layout/RiskViewToggle";
import { RuptureRiskWaterfall } from "@/components/prediction/RuptureRiskWaterfall";
import { ShapWaterfall } from "@/components/interpretability/ShapWaterfall";
import { R2edCard } from "@/components/prediction/R2edCard";
import { getPrediction, PredictionError } from "@/lib/prediction";
import { getRuptureRisk } from "@/lib/ruptureRisk";
import { getR2edScore, type R2edResult } from "@/lib/r2ed";
import type {
  AnalysisMode,
  ClassificationFeatures,
  LongitudinalFeatures
} from "@/types/patient";
import type { PredictionResponse } from "@/types/prediction";
import type { RuptureRiskResult } from "@/types/ruptureRisk";

export default function DashboardPage() {
  const [mode, setMode] = useState<AnalysisMode>("classification");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [r2ed, setR2ed] = useState<R2edResult | null>(null);
  const [ruptureRisk, setRuptureRisk] = useState<RuptureRiskResult | null>(null);
  const [riskView, setRiskView] = useState<RiskView>("model");
  const [apiError, setApiError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function switchMode(next: AnalysisMode) {
    setMode(next);
    setPrediction(null);
    setR2ed(null);
    setRuptureRisk(null);
    setApiError(null);
  }

  async function runClassification(features: ClassificationFeatures) {
    setLoading(true);
    setApiError(null);
    setR2ed(getR2edScore(features));
    try {
      setPrediction(await getPrediction(features));
    } catch (e) {
      setPrediction(null);
      setApiError(
        e instanceof PredictionError ? e.message : "Prediction failed."
      );
    } finally {
      setLoading(false);
    }
  }

  function runLongitudinal(features: LongitudinalFeatures) {
    setLoading(true);
    setTimeout(() => {
      setRuptureRisk(getRuptureRisk(features));
      setLoading(false);
    }, 300);
  }

  const isClassification = mode === "classification";

  return (
    <>
      <AppHeader />

      <PageShell>
        <div className="space-y-8">
          <section className="rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">

            <h1 className="mt-2 text-4xl font-bold tracking-tight text-slate-950">
              {isClassification
                ? "bAVM Hemorrhagic Presentation Predictor"
                : "bAVM Time-to-Rupture Estimator"}
            </h1>

            <p className="mt-4 max-w-4xl text-slate-600">
              {isClassification
                ? "Estimates the probability that a bAVM presented with hemorrhage, alongside the published R2eD score computed on the same inputs. This describes the hemorrhagic presentation and not future risk."
                : "Estimates rupture risk over time using the three-feature ridge Cox model and the integer risk score derived from it."}
            </p>

            <div className="mt-6">
              <ModeToggle mode={mode} onChange={switchMode} />
            </div>

            {isClassification && apiError && (
              <p className="mt-6 rounded-xl bg-rose-50 px-4 py-3 text-sm text-rose-900">
                {apiError}
              </p>
            )}
          </section>

          <div className="grid gap-8 lg:grid-cols-[400px_1fr]">
            {isClassification ? (
              <ClassificationForm
                onSubmit={runClassification}
                loading={loading}
              />
            ) : (
              <LongitudinalForm onSubmit={runLongitudinal} loading={loading} />
            )}

            <div className="space-y-8">
              {isClassification && prediction && (
                <>
                  <PredictionCard prediction={prediction} />
                  <ThresholdIndicator prediction={prediction} />
                  {r2ed && <R2edCard result={r2ed} />}
                  <ShapWaterfall
                    shapValues={prediction.shapValues}
                    baseValue={prediction.baseLogOdds}
                  />
                  <TopContributors shapValues={prediction.shapValues} />
                </>
              )}

              {!isClassification && ruptureRisk && (
                <>
                  <RiskViewToggle view={riskView} onChange={setRiskView} />
                  <RuptureRiskCard risk={ruptureRisk} view={riskView} />
                  {riskView === "model" ? (
                    <RuptureRiskWaterfall risk={ruptureRisk} />
                  ) : null}
                </>
              )}

              {!prediction && !ruptureRisk && (
                <section className="rounded-3xl border border-dashed border-slate-300 bg-white p-12 text-center text-slate-500">
                  Enter patient features and run the model to see results.
                </section>
              )}
            </div>
          </div>
        </div>
      </PageShell>
    </>
  );
}

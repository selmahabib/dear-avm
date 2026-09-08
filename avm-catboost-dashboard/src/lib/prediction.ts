import type { ClassificationFeatures, TriState } from "@/types/patient";
import type { PredictionResponse } from "@/types/prediction";
import type { ShapContribution } from "@/types/shap";
import { CATBOOST_OPERATING_THRESHOLD } from "./thresholds";

const API_URL =
  process.env.NEXT_PUBLIC_BAVM_API ?? "http://localhost:8000";

// The model is a tree ensemble and TreeSHAP cannot be computed in the
// browser, so both come from the Python service in backend/.
function toApi(v: TriState): boolean | null {
  return v === "yes" ? true : v === "no" ? false : null;
}

function display(v: TriState): string {
  return v === "yes" ? "Yes" : v === "no" ? "No" : "Undocumented";
}

interface ApiContribution {
  feature: string;
  value: number;
  missing: boolean;
}

interface ApiResponse {
  probability: number;
  baseLogOdds: number;
  contributions: ApiContribution[];
  model: string;
}

export class PredictionError extends Error {}

export async function getPrediction(
  features: ClassificationFeatures
): Promise<PredictionResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        age: features.age,
        sizeCm: features.sizeCm,
        race: features.race,
        deepLocation: toApi(features.deepLocation),
        hypertension: toApi(features.hypertension),
        exclusiveDeepDrainage: toApi(features.exclusiveDeepDrainage),
        intranidalAneurysm: toApi(features.intranidalAneurysm),
        singleFeedingArtery: toApi(features.singleFeedingArtery)
      })
    });
  } catch {
    throw new PredictionError(
      `Could not reach the model service at ${API_URL}. Start it with ` +
        `"uvicorn app:app --port 8000" from the backend directory.`
    );
  }

  if (!res.ok) {
    const detail = await res.text();
    throw new PredictionError(
      res.status === 503
        ? "The model service is running but has no model loaded. Copy " +
          "catboost_model.cbm and catboost_scaler.csv into backend/."
        : `Model service returned ${res.status}: ${detail}`
    );
  }

  const data: ApiResponse = await res.json();

  const shown: Record<string, string> = {
    "Nidus size": `${features.sizeCm} cm`,
    "Age at diagnosis": `${features.age} years`,
    "Non-white race": features.race === "nonwhite" ? "Yes" : "No",
    "Deep location": display(features.deepLocation),
    Hypertension: display(features.hypertension),
    "Exclusive deep drainage": display(features.exclusiveDeepDrainage),
    "Intranidal aneurysm": display(features.intranidalAneurysm),
    "Single feeding artery": display(features.singleFeedingArtery),
    "Feeder count undocumented":
      features.singleFeedingArtery === "unknown" ? "Yes" : "No"
  };

  const shapValues: ShapContribution[] = data.contributions.map((c) => ({
    feature: c.feature,
    value: c.value,
    featureValue: shown[c.feature] ?? ""
  }));

  return {
    probability: data.probability,
    baseLogOdds: data.baseLogOdds,
    threshold: CATBOOST_OPERATING_THRESHOLD,
    label:
      data.probability >= CATBOOST_OPERATING_THRESHOLD
        ? "Hemorrhagic presentation"
        : "Non-hemorrhagic presentation",
    shapValues
  };
}

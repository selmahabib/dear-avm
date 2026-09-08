import type { ShapContribution } from "./shap";

export type PredictionLabel =
  | "Hemorrhagic presentation"
  | "Non-hemorrhagic presentation";

export interface PredictionResponse {
  probability: number;
  baseLogOdds: number;
  threshold: number;
  label: PredictionLabel;
  shapValues: ShapContribution[];
}
export type RuptureRiskTier = "Low" | "Medium" | "High";

export interface RuptureRiskContribution {
  feature: string;
  value: number;
  featureValue: string | number | boolean;
}

export interface RuptureRiskResult {
  score: number;
  tier: RuptureRiskTier;
  annualRatePercent: number;
  annualRateCI: [number, number];
  linearPredictor: number;
  risk2Year: number;
  risk5Year: number;
  contributions: RuptureRiskContribution[];
}

import type {
  RuptureRiskContribution,
  RuptureRiskResult,
  RuptureRiskTier
} from "@/types/ruptureRisk";

interface RuptureRiskInputs {
  age: number;
  exclusivelyDeepLocation: boolean;
  priorRupture: boolean;
}

// Ridge-penalized Cox coefficients (paper, "Final Model Specification", p.7)
const BETA_AGE = 0.455;
const BETA_DEEP = 0.309;
const BETA_PRIOR = 0.255;

// Published baseline survival at 2 and 5 years (paper p.7)
// Baseline survival at the scaler's reference point, i.e. a patient at the
// cohort mean of every predictor. The 0.969 and 0.944 quoted in the paper are
// on a different reference and do not combine with the standardised linear
// predictor below. These reproduce all eight Table 3 profiles to within 0.1
// percentage points.
const S0_2 = 0.955;
const S0_5 = 0.920;

// StandardScaler constants from the fitted pipeline, printed by Survival.py.
// These are the values the model was actually fitted with, not an
// approximation recovered from the published example patients.
const MEAN_AGE = 35.81213307240704;
const SD_AGE = 18.812701541395338;
const P_DEEP = 0.2035225048923679;
const SD_DEEP = 0.40261780250592993;
const P_PRIOR = 0.39628180039138944;
const SD_PRIOR = 0.4891242532015238;

const TIER_ANNUAL_RATE: Record<
  RuptureRiskTier,
  { rate: number; ci: [number, number] }
> = {
  Low: { rate: 1.3, ci: [0.9, 1.9] },
  Medium: { rate: 3.0, ci: [2.1, 4.3] },
  High: { rate: 6.7, ci: [3.8, 10.9] }
};

function ageBracketPoints(age: number) {
  if (age < 27) return 0;
  if (age <= 44) return 1;
  if (age <= 62) return 2;
  return 3;
}

function tierForScore(score: number): RuptureRiskTier {
  if (score <= 1) return "Low";
  if (score <= 3) return "Medium";
  return "High";
}

export function getRuptureRisk(features: RuptureRiskInputs): RuptureRiskResult {
  const score =
    ageBracketPoints(features.age) +
    (features.exclusivelyDeepLocation ? 1 : 0) +
    (features.priorRupture ? 1 : 0);

  const tier = tierForScore(score);
  const { rate, ci } = TIER_ANNUAL_RATE[tier];

  const zAge = (features.age - MEAN_AGE) / SD_AGE;
  const zDeep = ((features.exclusivelyDeepLocation ? 1 : 0) - P_DEEP) / SD_DEEP;
  const zPrior = ((features.priorRupture ? 1 : 0) - P_PRIOR) / SD_PRIOR;

  const ageContribution = BETA_AGE * zAge;
  const deepContribution = BETA_DEEP * zDeep;
  const priorContribution = BETA_PRIOR * zPrior;

  const linearPredictor = ageContribution + deepContribution + priorContribution;
  const hazardRatio = Math.exp(linearPredictor);

  const risk2Year = 1 - Math.pow(S0_2, hazardRatio);
  const risk5Year = 1 - Math.pow(S0_5, hazardRatio);

  const contributions: RuptureRiskContribution[] = [
    {
      feature: "Age",
      value: ageContribution,
      featureValue: features.age
    },
    {
      feature: "Exclusively deep location",
      value: deepContribution,
      featureValue: features.exclusivelyDeepLocation
    },
    {
      feature: "Prior rupture",
      value: priorContribution,
      featureValue: features.priorRupture
    }
  ];

  return {
    score,
    tier,
    annualRatePercent: rate,
    annualRateCI: ci,
    linearPredictor,
    risk2Year,
    risk5Year,
    contributions
  };
}

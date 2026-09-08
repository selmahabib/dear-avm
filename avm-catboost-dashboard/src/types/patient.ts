export type Race = "white" | "nonwhite";

// Binary angioarchitectural findings can be genuinely undocumented in a
// routine report, so they are tri-state rather than boolean. "unknown" is a
// real answer, not a missing form field.
export type TriState = "yes" | "no" | "unknown";

export interface ClassificationFeatures {
  age: number;
  sizeCm: number;
  race: Race;
  deepLocation: TriState;
  hypertension: TriState;
  exclusiveDeepDrainage: TriState;
  intranidalAneurysm: TriState;
  singleFeedingArtery: TriState;
}

// The longitudinal model uses three predictors, all required. There is no
// "unknown" option here: with only three inputs an unanswered one would
// leave nothing to predict from.
export interface LongitudinalFeatures {
  age: number;
  exclusivelyDeepLocation: boolean;
  priorRupture: boolean;
}

export type AnalysisMode = "classification" | "longitudinal";

import type { ClassificationFeatures } from "@/types/patient";

// The published R2eD score. Unlike the CatBoost predictor this is fully
// specified in the literature, so it is implemented exactly rather than
// approximated: five predictors, race worth two points, the rest one each.
//
// An undocumented finding scores zero, which is how a clinician would score
// an incomplete angiogram report and how the score was evaluated in the
// paper. That is a deliberate choice, not a fallback.
export interface R2edResult {
  score: number;
  maxScore: number;
  components: { label: string; points: number; awarded: boolean; unknown: boolean }[];
  unknownCount: number;
}

function award(v: string, points: number) {
  return { awarded: v === "yes", unknown: v === "unknown", points };
}

export function getR2edScore(f: ClassificationFeatures): R2edResult {
  const items = [
    { label: "Non-white race", ...award(f.race === "nonwhite" ? "yes" : "no", 2) },
    { label: "Deep location", ...award(f.deepLocation, 1) },
    { label: "Exclusive deep drainage", ...award(f.exclusiveDeepDrainage, 1) },
    { label: "Single feeding artery", ...award(f.singleFeedingArtery, 1) },
    { label: "Nidus size < 3 cm", ...award(f.sizeCm < 3 ? "yes" : "no", 1) }
  ];

  const score = items.reduce((s, i) => s + (i.awarded ? i.points : 0), 0);

  return {
    score,
    maxScore: 6,
    components: items,
    unknownCount: items.filter((i) => i.unknown).length
  };
}

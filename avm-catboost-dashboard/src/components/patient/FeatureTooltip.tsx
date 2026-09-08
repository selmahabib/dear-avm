interface FeatureTooltipProps {
  text: string;
}

export function FeatureTooltip({ text }: FeatureTooltipProps) {
  return (
    <p className="mt-1 text-xs leading-relaxed text-slate-500">
      {text}
    </p>
  );
}
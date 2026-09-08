import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function formatPercent1(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatSigned(value: number) {
  if (value > 0) return `+${value.toFixed(3)}`;
  return value.toFixed(3);
}
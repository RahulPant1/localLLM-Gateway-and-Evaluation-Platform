import type { Execution } from "./api";

export const gb = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`;

export function seconds(ms: number | null | undefined): string {
  if (ms == null) return "-";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

export function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function snippet(text: string, length = 90): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > length ? `${flat.slice(0, length)}…` : flat;
}

/** Mean overall score across this execution's evaluations, or null if none. */
export function meanScore(execution: Execution): number | null {
  const scores = execution.evaluations
    .map((v) => v.overall_score)
    .filter((s): s is number => s != null);
  if (scores.length === 0) return null;
  return scores.reduce((a, b) => a + b, 0) / scores.length;
}

export const WARNING_TEXT: Record<string, string> = {
  prompt_probably_truncated: "The prompt is probably longer than the model's context window",
  output_cut_off: "The answer was cut off at the length limit",
  context_window_full: "The context window filled up",
};

export const CATEGORY_TEXT: Record<string, string> = {
  general_qa: "General Q&A",
  classification: "Classification",
  extraction: "Information extraction",
  structured_output: "Structured output",
  reasoning: "Reasoning and analysis",
  coding: "Coding",
  medical: "Medical documents",
  not_routed: "Not routed (manual)",
};

export const DECIDED_BY_TEXT: Record<string, string> = {
  signals: "quick rules",
  llm: "classifier model",
  signals_fallback: "quick rules (classifier not used)",
  none: "not decided",
};

export const STATUS_TEXT: Record<string, string> = {
  queued: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
};

/** 4200 -> "4.2k", 1250000 -> "1.3M". */
export function compact(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1e6) return `${(n / 1000).toFixed(n < 10000 ? 1 : 0)}k`;
  return `${(n / 1e6).toFixed(1)}M`;
}

export const percent = (fraction: number | null | undefined) =>
  fraction == null ? "-" : `${Math.round(fraction * 100)}%`;

/** 200 -> "3 min 20 s". */
export function duration(totalSeconds: number): string {
  if (totalSeconds < 60) return `${totalSeconds.toFixed(totalSeconds < 10 ? 1 : 0)} s`;
  const minutes = Math.floor(totalSeconds / 60);
  if (minutes < 60) return `${minutes} min ${Math.round(totalSeconds % 60)} s`;
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

/** "science_technology" -> "Science technology"; for names the app has no label for. */
export const pretty = (name: string) =>
  name.charAt(0).toUpperCase() + name.slice(1).replace(/_/g, " ");

/** A model name short enough for a table: no organisation prefix, no quantisation suffix. */
export const modelLabel = (name: string) =>
  (name.split("/").pop() ?? name).replace(/-(it|instruct)?-?q\d.*$/i, "");

/** Cell background for a 0-10 score: one hue, stronger as the score rises (plain below 5). */
export function heat(score: number | null | undefined): { background?: string } {
  if (score == null) return {};
  const strength = Math.max(0, Math.min(1, (score - 5) / 5));
  return { background: `color-mix(in srgb, var(--accent) ${Math.round(strength * 40)}%, var(--panel))` };
}

export const DIMENSION_TEXT: Record<string, string> = {
  correctness: "Correct",
  relevance: "Relevant",
  completeness: "Complete",
  instruction_adherence: "Follows instructions",
  groundedness: "Grounded",
};

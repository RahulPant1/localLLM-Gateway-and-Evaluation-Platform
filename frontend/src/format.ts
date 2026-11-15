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

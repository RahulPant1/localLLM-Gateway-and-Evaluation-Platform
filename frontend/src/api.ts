// Types and calls for the backend JSON API. The shapes mirror
// backend/src/llm_gateway/schemas.py.

export type ExecutionStatus = "queued" | "running" | "completed" | "failed";

export interface ModelEntry {
  name: string;
  size_bytes: number;
  family: string | null;
  parameter_size: string | null;
  quantization: string | null;
  capabilities: string[];
  max_context_length: number | null;
  num_ctx: number;
  details_available: boolean;
  loaded: boolean;
  // Described in config/models.yaml; unprofiled models are never picked automatically.
  profiled: boolean;
  strengths: string;
  categories: string[];
}

export interface RoutingDecision {
  request_id: string;
  // True on batch prompts: what the router would have picked; it did not choose the model.
  shadow: boolean;
  category: string;
  complexity: "low" | "medium" | "high";
  selected_model: string | null;
  fallback_model: string | null;
  confidence: number;
  explanation: string;
  escalate: boolean;
  escalation_reason: string | null;
  decided_by: "signals" | "llm" | "signals_fallback" | "none";
  candidates: string[];
  fallback_used: boolean;
}

export interface Evaluation {
  id: string;
  execution_id: string;
  judge: string;
  rubric_version: string;
  scores: Record<string, number | null>;
  // The judge's classification of the prompt; null on older evaluations.
  task_type: string | null;
  domain: string | null;
  overall_score: number | null;
  summary: string;
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
  critical_issues: string[];
  created_at: string;
}

export interface Execution {
  id: string;
  request_id: string;
  model: string;
  status: ExecutionStatus;
  // False on a second attempt that was run with thinking off.
  think: boolean | null;
  response: string | null;
  thinking: string | null;
  error_code: string | null;
  error_detail: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  queue_wait_ms: number | null;
  execution_ms: number | null;
  metrics: Record<string, number | string | null> | null;
  warnings: string[];
  evaluations: Evaluation[];
  follow_up_count: number;
}

export interface ThreadTurn {
  request_id: string;
  prompt: string;
  source_material: string | null;
  created_at: string;
  execution: Execution;
}

export interface Thread {
  model: string;
  system: string | null;
  turns: ThreadTurn[];
  context: {
    used_tokens: number;
    limit_tokens: number | null;
    fraction: number | null;
    state: "ok" | "warning" | "full";
  };
  // The answer a new message continues from; null while a turn is in progress.
  reply_to: string | null;
}

export interface RequestRecord {
  id: string;
  batch_id: string | null;
  parent_execution_id: string | null;
  prompt: string;
  system: string | null;
  source_material: string | null;
  routing_mode: string;
  think: boolean | null;
  max_tokens: number | null;
  // Earlier turns sent by an API client with this prompt.
  history: { role: "user" | "assistant"; content: string }[];
  // "api" = came from another tool through the OpenAI-compatible endpoint.
  source: "ui" | "api";
  created_at: string;
  executions: Execution[];
  routing: RoutingDecision | null;
}

export interface BatchSummary {
  id: string;
  name: string;
  created_at: string;
  models: string[];
  request_count: number;
  counts: Record<ExecutionStatus, number>;
  evaluated_count: number;
}

export interface BatchDetail extends BatchSummary {
  requests: RequestRecord[];
}

export interface JudgePackage {
  text: string;
  item_count: number;
  rubric_version: string;
  blind: boolean;
}

export interface Rubric {
  version: string;
  scale: string;
  dimensions: { name: string; weight: number; description: string; anchors: Record<string, string> }[];
  error: string | null;
}

export interface ModelScores {
  model: string;
  judge: string;
  evaluated: number;
  overall: number | null;
  dimensions: Record<string, number | null>;
  critical_issues: number;
}

export interface Share {
  key: string;
  count: number;
  share: number;
}

export interface GroupRow {
  key: string;
  executions: number;
  completed: number;
  failed: number;
  median_ms: number | null;
  slowest_ms: number | null;
  output_tokens: number;
  tokens_per_second: number | null;
  warm_rate: number | null;
}

export interface TaxonomyLabel {
  name: string;
  label: string;
  description: string;
}

export interface Taxonomy {
  task_types: TaxonomyLabel[];
  domains: TaxonomyLabel[];
}

/** Scores for one task type or domain, per model. */
export interface TaskRow {
  key: string;
  label: string;
  prompts: number;
  models: Record<string, { mean: number; answers: number }>;
  best_model: string | null;
}

/** Totals for the Overview tab (backend/src/llm_gateway/overview.py). */
export interface Overview {
  days: number | null;
  requests: number;
  routing: {
    live: number;
    shadow: number;
    by_model: Share[];
    by_category: Share[];
    decided_by: Record<string, number>;
    fallback_used: number;
    escalations: number;
    mean_confidence: number | null;
  };
  performance: {
    executions: number;
    completed: number;
    failed: number;
    prompt_tokens: number;
    output_tokens: number;
    generation_seconds: number;
    load_seconds: number;
    tokens_per_second: number | null;
    warm_rate: number | null;
    mean_swap_ms: number | null;
    mean_queue_wait_ms: number | null;
    cloud_estimate: {
      model: string;
      input_per_mtok: number;
      output_per_mtok: number;
      usd: number;
    } | null;
  };
  by_model: GroupRow[];
  by_category: GroupRow[];
  evaluation: {
    scored: number;
    unscored: number;
    judges: string[];
    by_model: {
      model: string;
      evaluated: number;
      overall: number | null;
      dimensions: Record<string, number | null>;
      critical_issues: number;
    }[];
    models: string[];
    by_task: TaskRow[];
    by_domain: TaskRow[];
    router_check: {
      prompts: number;
      picked_top: number;
      router_mean: number | null;
      best_fixed_model: string | null;
      best_fixed_mean: number | null;
      best_possible_mean: number | null;
    };
  };
}

export interface SystemStatus {
  status: "ok" | "degraded";
  version: string;
  environment: string;
  provider: {
    name: string;
    url: string;
    reachable: boolean;
    version: string | null;
    error: string | null;
    loaded_models: { name: string; vram_bytes: number }[];
  };
  gpu: {
    available: boolean;
    name: string | null;
    memory_used_mib: number | null;
    memory_total_mib: number | null;
    utilization_percent: number | null;
    error: string | null;
  };
  // System memory (of WSL when running there). Below guard_mb, models are unloaded to free it.
  memory: { available_mb: number | null; total_mb: number | null; guard_mb: number };
  queue: { busy: boolean; running_model: string | null; waiting: number; memory_unloads: number };
  router: { classifier_model: string | null; llm_enabled: boolean; registry_error: string | null };
  evaluator: { configured: boolean; judge_model: string };
}

export interface PromptInput {
  prompt: string;
  system?: string | null;
  source_material?: string | null;
}

async function call<T>(path: string, body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new Error("Cannot reach the gateway backend. Is it running?");
  }
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const data = await response.json();
      const d = data.detail ?? data.error;
      detail = typeof d === "string" ? d : JSON.stringify(d);
    } catch {
      // keep the status text
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export interface ToolSuite {
  id: string;
  title: string;
  source: string;
  case_count: number;
  categories: Record<string, { description: string; cases: number }>;
}

export interface Tally {
  passed: number;
  // Of those, calls sent in the proper tool-call field (the rest were JSON written in the answer).
  passed_native: number;
  wrong: number;
  errors: number;
  waiting: number;
}

export interface ModelTally extends Tally {
  model: string;
  by_category: Record<string, Tally>;
  mean_ms: number | null;
}

export interface ToolRunSummary {
  id: string;
  suite: string;
  name: string;
  models: string[];
  think: boolean;
  created_at: string;
  total: number;
  finished: number;
  running: boolean;
  per_model: ModelTally[];
}

export interface ToolCallRecord {
  name: string;
  arguments: Record<string, unknown>;
  malformed: boolean;
}

export interface ToolResult {
  id: string;
  case_id: string;
  category: string;
  model: string;
  status: ExecutionStatus;
  passed: boolean | null;
  reason: string | null;
  prompt: string;
  tool_names: string[];
  // Accepted calls: each argument maps to the list of accepted values ("" = may be left out).
  expected: { name: string; arguments: Record<string, unknown[]> }[];
  calls: ToolCallRecord[];
  call_format: "native" | "text" | "none" | null;
  content: string | null;
  thinking: string | null;
  error_code: string | null;
  error_detail: string | null;
  prompt_tokens: number | null;
  output_tokens: number | null;
  execution_ms: number | null;
}

export interface ToolRunDetail extends ToolRunSummary {
  results: ToolResult[];
}

export interface LineupEntry {
  name: string;
  installed: boolean;
  hidden: boolean;
  tools: boolean;
  size_bytes: number | null;
  parameter_size: string | null;
  capabilities: string[];
  batches: Record<string, { answered: number; failed: number; waiting: number; judged: number }>;
  tool_runs: Record<string, { passed: number; wrong: number; failed: number; waiting: number }>;
  not_run: number;
  unjudged: number;
}

export interface Lineup {
  models: LineupEntry[];
  batches: { id: string; name: string; prompts: number }[];
  tool_runs: { id: string; name: string; suite: string; cases: number }[];
  unjudged: number;
}

export const api = {
  lineup: () => call<Lineup>("/api/lineup"),
  hideModel: (model: string) => call<unknown>("/api/lineup/hide", { model }),
  restoreModel: (model: string) => call<unknown>("/api/lineup/restore", { model }),
  runAllTests: (model: string) =>
    call<{ queued_answers: number; queued_tool_cases: number; notes: string[] }>(
      "/api/lineup/run-all",
      { model },
    ),
  toolSuites: () => call<ToolSuite[]>("/api/tool-use/suites"),
  toolRuns: () => call<ToolRunSummary[]>("/api/tool-use/runs"),
  toolRun: (id: string) => call<ToolRunDetail>(`/api/tool-use/runs/${id}`),
  createToolRun: (body: { suite: string; models: string[]; think: boolean; name?: string }) =>
    call<ToolRunSummary>("/api/tool-use/runs", body),
  resumeToolRun: (id: string) => call<ToolRunSummary>(`/api/tool-use/runs/${id}/resume`, {}),
  status: () => call<SystemStatus>("/api/status"),
  models: () => call<{ provider: string; models: ModelEntry[] }>("/api/models"),
  // `models: null` asks the router to pick the model.
  createRequest: (body: PromptInput & { models: string[] | null; think: boolean | null }) =>
    call<RequestRecord>("/api/requests", body),
  request: (id: string) => call<RequestRecord>(`/api/requests/${id}`),
  requests: () => call<RequestRecord[]>("/api/requests?limit=100"),
  createBatch: (body: {
    name: string;
    prompts: string[];
    models: string[];
    think: boolean | null;
  }) => call<BatchSummary>("/api/batches", body),
  thread: (executionId: string) => call<Thread>(`/api/executions/${executionId}/thread`),
  /** Run the same prompt again on the same model with thinking off. */
  retryWithoutThinking: (executionId: string) =>
    call<RequestRecord>(`/api/executions/${executionId}/retry`, { think: false }),
  followUp: (executionId: string, prompt: string) =>
    call<RequestRecord>(`/api/executions/${executionId}/follow-ups`, { prompt }),
  batches: () => call<BatchSummary[]>("/api/batches"),
  batch: (id: string) => call<BatchDetail>(`/api/batches/${id}`),
  judgePackage: (body: {
    request_ids: string[];
    batch_ids: string[];
    blind: boolean;
    exclude_judge: string | null;
  }) => call<JudgePackage>("/api/judge-package", body),
  rubric: () => call<Rubric>("/api/rubric"),
  judgePrompt: () => call<{ instructions: string }>("/api/judge-prompt"),
  importEvaluations: (body: unknown) => call<{ stored: number }>("/api/evaluations", body),
  taskTypes: () => call<Taxonomy>("/api/task-types"),
  overview: (days: number | null) =>
    call<Overview>(days ? `/api/overview?days=${days}` : "/api/overview"),
  /** Run again the parts of a batch that were interrupted or timed out. */
  resumeBatch: (id: string) => call<BatchSummary>(`/api/batches/${id}/resume`, {}),
  batchScores: (id: string) => call<ModelScores[]>(`/api/batches/${id}/scores`),
  exportUrl: (id: string, blind: boolean) =>
    `/api/batches/${id}/export?blind=${blind}&download=true`,
};

/** The model used every available token on reasoning and gave no answer. */
export const ranOutWhileThinking = (e: Execution) =>
  e.status === "failed" && e.error_code === "thinking_exhausted";

/** Failures caused by the environment (a restart, a timeout, low memory), worth running again. */
export const RESUMABLE_CODES = ["interrupted", "generation_timeout", "low_memory", "provider_unavailable"];

export const isActive = (e: Execution) => e.status === "queued" || e.status === "running";

/** Still being routed or run. An automatic request has no executions until the router decides. */
export const requestActive = (r: RequestRecord) =>
  r.executions.some(isActive) ||
  (r.routing_mode === "automatic" && r.routing === null && r.parent_execution_id === null);

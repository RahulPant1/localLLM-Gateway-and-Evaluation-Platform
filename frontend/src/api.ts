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
}

export interface Evaluation {
  id: string;
  execution_id: string;
  judge: string;
  rubric_version: string;
  scores: Record<string, number | null>;
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
  created_at: string;
  executions: Execution[];
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
  queue: { busy: boolean; running_model: string | null; waiting: number };
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

export const api = {
  status: () => call<SystemStatus>("/api/status"),
  models: () => call<{ provider: string; models: ModelEntry[] }>("/api/models"),
  createRequest: (body: PromptInput & { models: string[]; think: boolean | null }) =>
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
  followUp: (executionId: string, prompt: string) =>
    call<RequestRecord>(`/api/executions/${executionId}/follow-ups`, { prompt }),
  batches: () => call<BatchSummary[]>("/api/batches"),
  batch: (id: string) => call<BatchDetail>(`/api/batches/${id}`),
  exportUrl: (id: string, blind: boolean) =>
    `/api/batches/${id}/export?blind=${blind}&download=true`,
};

export const isActive = (e: Execution) => e.status === "queued" || e.status === "running";

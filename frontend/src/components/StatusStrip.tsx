import { useState } from "react";
import { api, type SystemStatus } from "../api";
import { usePolling } from "../hooks";

export function StatusStrip() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  usePolling(
    async () => {
      try {
        setStatus(await api.status());
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    3000,
    [],
  );

  if (error) {
    return (
      <div className="strip">
        <span className="pill bad">Backend unreachable</span>
        <span className="muted">{error}</span>
      </div>
    );
  }
  if (!status) return <div className="strip muted">Checking system status…</div>;

  const { provider, gpu, queue, evaluator } = status;
  const loaded = provider.loaded_models.map((m) => m.name).join(", ");
  return (
    <div className="strip">
      <span className={`pill ${provider.reachable ? "good" : "bad"}`} title={provider.error ?? provider.url}>
        Ollama {provider.reachable ? provider.version : "unreachable"}
      </span>
      <span className="item">
        <span className="label">Loaded</span> {loaded || "none"}
      </span>
      <span className="item">
        <span className="label">GPU</span>{" "}
        {gpu.available
          ? `${((gpu.memory_used_mib ?? 0) / 1024).toFixed(1)} / ${((gpu.memory_total_mib ?? 0) / 1024).toFixed(1)} GB`
          : "unavailable"}
      </span>
      <span className="item">
        <span className="label">Queue</span>{" "}
        {queue.busy ? `running ${queue.running_model}${queue.waiting ? `, ${queue.waiting} waiting` : ""}` : "idle"}
      </span>
      <span className={`pill ${evaluator.configured ? "good" : "neutral"}`} title={`Judge: ${evaluator.judge_model}`}>
        Claude evaluator {evaluator.configured ? "ready" : "not configured"}
      </span>
      {status.router.registry_error && (
        <span className="pill bad" title={status.router.registry_error}>
          Model registry error
        </span>
      )}
      <span className="item muted">{status.environment.toUpperCase()}</span>
    </div>
  );
}

import { useEffect, useState } from "react";
import { api, requestActive, type RequestRecord } from "../api";
import { usePolling } from "../hooks";
import { Chat } from "./Chat";
import { ExecutionCard } from "./ExecutionCard";
import { RoutingPanel } from "./RoutingPanel";

/** A request's prompt and one card per model it ran on, or the conversation continued from one of them. */
interface Props {
  request: RequestRecord;
  showPrompt?: boolean;
  /** When given, offers to open the Evaluate tab with this run ticked. */
  onEvaluate?: (key: string) => void;
}

export function RequestView({ request, showPrompt = true, onEvaluate }: Props) {
  const [chatId, setChatId] = useState<string | null>(null);

  // After "Run again without thinking": this request re-read from the backend,
  // which then holds the second attempt as one more execution.
  const [fresh, setFresh] = useState<RequestRecord | null>(null);
  const [retryError, setRetryError] = useState<string | null>(null);

  // Showing a different request closes any open conversation.
  useEffect(() => {
    setChatId(null);
    setFresh(null);
    setRetryError(null);
  }, [request.id]);

  const refreshing = fresh ? requestActive(fresh) : false;
  usePolling(
    async () => {
      if (!fresh || !refreshing) return;
      try {
        setFresh(await api.request(request.id));
      } catch (e) {
        setRetryError((e as Error).message);
      }
    },
    refreshing ? 500 : null,
    [request.id, refreshing],
  );
  const retry = async (executionId: string) => {
    try {
      setFresh(await api.retryWithoutThinking(executionId));
      setRetryError(null);
    } catch (e) {
      setRetryError((e as Error).message);
    }
  };

  // The answers to show: the ones given, updated from the re-read copy, plus
  // any later attempt on the same models (the caller may pass only some models).
  const latest = new Map(fresh?.executions.map((e) => [e.id, e]));
  const models = new Set(request.executions.map((e) => e.model));
  const executions = [
    ...request.executions.map((e) => latest.get(e.id) ?? e),
    ...(fresh?.executions ?? []).filter(
      (e) => models.has(e.model) && !request.executions.some((given) => given.id === e.id),
    ),
  ];
  const attempt = (id: string, model: string) =>
    executions.filter((e) => e.model === model).findIndex((e) => e.id === id);
  const attempts = (model: string) => executions.filter((e) => e.model === model).length;

  const completed = executions.filter((e) => e.status === "completed");
  const fullyScored = completed.length > 0 && completed.every((e) => e.evaluations.length > 0);

  if (chatId) return <Chat executionId={chatId} onClose={() => setChatId(null)} />;

  return (
    <div className="request-view">
      {showPrompt && (
        <div className="prompt-box">
          <span className="label">Prompt</span>
          <pre className="response plain">{request.prompt}</pre>
          {request.history.length > 0 && (
            <details>
              <summary>Earlier conversation sent with this prompt ({request.history.length} messages)</summary>
              {request.history.map((m, i) => (
                <pre key={i} className="response plain">
                  {m.role === "user" ? "User: " : "Assistant: "}
                  {m.content}
                </pre>
              ))}
            </details>
          )}
          {request.system && (
            <details>
              <summary>System prompt</summary>
              <pre className="response plain">{request.system}</pre>
            </details>
          )}
          {request.source_material && (
            <details>
              <summary>Source material ({request.source_material.length.toLocaleString()} characters)</summary>
              <pre className="response plain">{request.source_material}</pre>
            </details>
          )}
        </div>
      )}
      {request.routing ? (
        <RoutingPanel routing={request.routing} />
      ) : (
        request.routing_mode === "automatic" &&
        request.parent_execution_id === null && <p className="muted">Choosing a model…</p>
      )}
      {onEvaluate && completed.length > 0 && (
        <div className="batch-bar">
          <button className="secondary flat" onClick={() => onEvaluate(`r:${request.id}`)}>
            {fullyScored ? "Evaluate again" : "Evaluate this run"}
          </button>
          <span className="muted small">
            {fullyScored
              ? "Already scored. This puts the run back in the Evaluate tab for another judge."
              : "Have a judge score these answers against the rubric."}
          </span>
        </div>
      )}
      {retryError && <p className="error">{retryError}</p>}
      <div className={executions.length > 1 ? "cards compare" : "cards"}>
        {executions.map((e) => (
          <div className="answer-slot" key={e.id}>
            {attempt(e.id, e.model) > 0 && (
              <span className="label">
                Attempt {attempt(e.id, e.model) + 1}
                {e.think === false && ", thinking off"}
              </span>
            )}
            <ExecutionCard
              execution={e}
              onContinue={() => setChatId(e.id)}
              // Offered on the latest attempt only.
              onRetry={attempt(e.id, e.model) === attempts(e.model) - 1 ? () => retry(e.id) : undefined}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

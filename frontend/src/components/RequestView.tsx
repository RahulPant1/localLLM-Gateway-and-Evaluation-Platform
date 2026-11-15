import { useEffect, useState } from "react";
import type { RequestRecord } from "../api";
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

  // Showing a different request closes any open conversation.
  useEffect(() => setChatId(null), [request.id]);

  const completed = request.executions.filter((e) => e.status === "completed");
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
      <div className={request.executions.length > 1 ? "cards compare" : "cards"}>
        {request.executions.map((e) => (
          <ExecutionCard key={e.id} execution={e} onContinue={() => setChatId(e.id)} />
        ))}
      </div>
    </div>
  );
}

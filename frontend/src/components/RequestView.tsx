import { useEffect, useState } from "react";
import type { RequestRecord } from "../api";
import { Chat } from "./Chat";
import { ExecutionCard } from "./ExecutionCard";
import { RoutingPanel } from "./RoutingPanel";

/** A request's prompt and one card per model it ran on, or the conversation continued from one of them. */
export function RequestView({ request, showPrompt = true }: { request: RequestRecord; showPrompt?: boolean }) {
  const [chatId, setChatId] = useState<string | null>(null);

  // Showing a different request closes any open conversation.
  useEffect(() => setChatId(null), [request.id]);

  if (chatId) return <Chat executionId={chatId} onClose={() => setChatId(null)} />;

  return (
    <div className="request-view">
      {showPrompt && (
        <div className="prompt-box">
          <span className="label">Prompt</span>
          <pre className="response plain">{request.prompt}</pre>
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
      <div className={request.executions.length > 1 ? "cards compare" : "cards"}>
        {request.executions.map((e) => (
          <ExecutionCard key={e.id} execution={e} onContinue={() => setChatId(e.id)} />
        ))}
      </div>
    </div>
  );
}

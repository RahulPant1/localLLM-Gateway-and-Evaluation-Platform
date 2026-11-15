import type { RequestRecord } from "../api";
import { ExecutionCard } from "./ExecutionCard";

/** A request's prompt and one card per model it ran on. */
export function RequestView({ request, showPrompt = true }: { request: RequestRecord; showPrompt?: boolean }) {
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
      <div className={request.executions.length > 1 ? "cards compare" : "cards"}>
        {request.executions.map((e) => (
          <ExecutionCard key={e.id} execution={e} />
        ))}
      </div>
    </div>
  );
}

import type { RoutingDecision } from "../api";
import { CATEGORY_TEXT, DECIDED_BY_TEXT } from "../format";

/** The router's decision for a request: what it chose and why. */
export function RoutingPanel({ routing }: { routing: RoutingDecision }) {
  return (
    <div className="routing">
      <div className="routing-head">
        <span>
          <span className="label">{routing.shadow ? "Router would pick" : "Routed to"}</span>{" "}
          <span className="value">{routing.selected_model ?? "no model"}</span>
        </span>
        <span>
          <span className="label">Task</span> <span className="value">{CATEGORY_TEXT[routing.category] ?? routing.category}</span>
        </span>
        <span>
          <span className="label">Complexity</span> <span className="value">{routing.complexity}</span>
        </span>
        <span title="An estimate of how sure the router is, not a guarantee">
          <span className="label">Confidence</span>{" "}
          <span className="value">{Math.round(routing.confidence * 100)}%</span>
        </span>
        <span>
          <span className="label">Fallback</span> <span className="value">{routing.fallback_model ?? "none"}</span>
        </span>
        <span>
          <span className="label">Decided by</span> <span className="value">{DECIDED_BY_TEXT[routing.decided_by]}</span>
        </span>
      </div>
      <p className="small">{routing.explanation}</p>
      {routing.fallback_used && (
        <p className="warning">
          {routing.selected_model} failed, so the fallback model {routing.fallback_model} was run instead.
        </p>
      )}
      {routing.escalate && <p className="warning">Escalation advised: {routing.escalation_reason}.</p>}
      {routing.shadow && (
        <p className="muted small">
          Recorded for comparison only. In a batch every chosen model runs, whatever the router would pick.
        </p>
      )}
    </div>
  );
}

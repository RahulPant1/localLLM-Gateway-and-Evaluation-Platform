import type { AgentStep } from "../api";
import { snippet } from "../format";

const formatArgs = (args: Record<string, unknown>) =>
  Object.entries(args)
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(", ");

/** An agent's steps: what the model wrote, each tool call, and what the app returned. */
export function AgentSteps({ steps }: { steps: AgentStep[] }) {
  return (
    <ol className="trace">
      {steps.map((step) => (
        <li key={step.step}>
          {step.content && step.calls.length > 0 && <p className="trace-text">{step.content}</p>}
          {step.calls.map((call, i) => (
            <div key={i} className={`trace-call ${call.invalid || call.error ? "bad" : ""}`}>
              <code>
                {call.name}({formatArgs(call.arguments)})
              </code>
              <details>
                <summary className="muted small">
                  {call.invalid ? "invalid call" : call.error ? "app returned an error" : "result"}:{" "}
                  {snippet(call.result, 90)}
                </summary>
                <pre>{call.result}</pre>
              </details>
            </div>
          ))}
          {step.calls.length === 0 && <p className="muted small">Final answer (below).</p>}
          {step.calls_as_text && (
            <p className="muted small">The model wrote these calls as text; they were run anyway.</p>
          )}
        </li>
      ))}
    </ol>
  );
}

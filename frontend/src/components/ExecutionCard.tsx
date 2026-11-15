import { ranOutWhileThinking, type Execution } from "../api";
import { STATUS_TEXT, WARNING_TEXT, seconds, pretty } from "../format";
import { useNow } from "../hooks";
import { Markdown } from "./Markdown";

const DIMENSION_LABELS: Record<string, string> = {
  correctness: "Correctness",
  relevance: "Relevance",
  completeness: "Completeness",
  instruction_adherence: "Instruction adherence",
  groundedness: "Groundedness",
};

function Elapsed({ execution }: { execution: Execution }) {
  const now = useNow(true);
  const since = new Date(execution.started_at ?? execution.created_at).getTime();
  return <span>{Math.max(0, Math.round((now - since) / 1000))} s</span>;
}

interface Props {
  execution: Execution;
  /** When given, the card offers to continue the conversation from this answer. */
  onContinue?: () => void;
  /** When given, a run that ran out of room while thinking offers to run again without it. */
  onRetry?: () => void;
}

export function ExecutionCard({ execution, onContinue, onRetry }: Props) {
  const m = execution.metrics ?? {};
  return (
    <article className="card">
      <header className="card-head">
        <strong>{execution.model}</strong>
        <span className={`pill status-${execution.status}`}>{STATUS_TEXT[execution.status]}</span>
        {execution.status === "queued" && (
          <span className="muted small">
            waiting for the GPU, <Elapsed execution={execution} />
          </span>
        )}
        {execution.status === "running" && (
          <span className="muted small">
            {execution.response || execution.thinking ? "generating" : "loading the model"},{" "}
            <Elapsed execution={execution} />
          </span>
        )}
        {execution.status === "completed" && (
          <span className="muted small">
            {seconds(execution.execution_ms)} total · load {seconds(m.load_ms as number)} ·{" "}
            {m.output_tokens ?? "-"} tokens · {m.tokens_per_second ?? "-"} tok/s
            {(execution.queue_wait_ms ?? 0) > 500 && ` · queued ${seconds(execution.queue_wait_ms)}`}
          </span>
        )}
      </header>

      {execution.status === "failed" && (
        <p className="error">
          <strong>{ranOutWhileThinking(execution) ? "Failed" : execution.error_code}</strong>:{" "}
          {execution.error_detail}
        </p>
      )}
      {onRetry && ranOutWhileThinking(execution) && (
        <button className="secondary flat" onClick={onRetry}>
          Run again without thinking
        </button>
      )}
      {execution.warnings.map((w) => (
        <p key={w} className="warning">
          {WARNING_TEXT[w] ?? w}
        </p>
      ))}
      {execution.status === "running" && !execution.response && execution.thinking && (
        <p className="muted small">Thinking before it answers…</p>
      )}
      {/* While running, `response` holds the text generated so far. */}
      {(execution.status === "completed" || (execution.status === "running" && execution.response)) && (
        <div className="response">
          {execution.response ? <Markdown text={execution.response} /> : <span className="muted">(empty response)</span>}
        </div>
      )}
      {execution.thinking && (
        <details>
          <summary>Thinking ({execution.thinking.length.toLocaleString()} characters)</summary>
          <pre className="response thinking">{execution.thinking}</pre>
        </details>
      )}

      {onContinue && execution.status === "completed" && (
        <button className="secondary" onClick={onContinue}>
          {execution.follow_up_count > 0 ? "Open conversation" : "Continue chat with this model"}
        </button>
      )}

      {execution.evaluations.map((v) => (
        <details key={v.id} className="evaluation">
          <summary>
            <span className="score">{v.overall_score?.toFixed(1) ?? "-"}</span> / 10 · judged by {v.judge}
          </summary>
          <div className="dimensions">
            {Object.entries(v.scores).map(([name, value]) => (
              <span key={name}>
                {DIMENSION_LABELS[name] ?? name}: <strong>{value ?? "n/a"}</strong>
              </span>
            ))}
          </div>
          {(v.task_type || v.domain) && (
            <p className="muted small">
              Task type: {v.task_type ? pretty(v.task_type) : "-"} · Domain:{" "}
              {v.domain ? pretty(v.domain) : "-"}
            </p>
          )}
          {v.summary && <p>{v.summary}</p>}
          {(
            [
              ["Strengths", v.strengths],
              ["Weaknesses", v.weaknesses],
              ["Recommendations", v.recommendations],
              ["Critical issues", v.critical_issues],
            ] as const
          ).map(
            ([title, items]) =>
              items.length > 0 && (
                <div key={title}>
                  <span className="label">{title}</span>
                  <ul>
                    {items.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                </div>
              ),
          )}
        </details>
      ))}
    </article>
  );
}

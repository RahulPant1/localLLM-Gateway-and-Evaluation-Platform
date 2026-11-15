import { useEffect, useRef, useState } from "react";
import { api, isActive, type Execution, type Thread } from "../api";
import { STATUS_TEXT, WARNING_TEXT, seconds } from "../format";
import { useNow, usePolling } from "../hooks";
import { Markdown } from "./Markdown";

function Answer({ execution }: { execution: Execution }) {
  const now = useNow(isActive(execution));
  const m = execution.metrics ?? {};
  const since = new Date(execution.started_at ?? execution.created_at).getTime();
  return (
    <div className="turn-answer">
      <div className="turn-meta">
        {execution.status !== "completed" && (
          <span className={`pill status-${execution.status}`}>{STATUS_TEXT[execution.status]}</span>
        )}
        <span className="muted small">
          {execution.status === "queued" && `waiting for the GPU, ${Math.max(0, Math.round((now - since) / 1000))} s`}
          {execution.status === "running" && `generating, ${Math.max(0, Math.round((now - since) / 1000))} s`}
          {execution.status === "completed" &&
            `${seconds(execution.execution_ms)} · ${m.output_tokens ?? "-"} tokens · ${m.tokens_per_second ?? "-"} tok/s`}
        </span>
      </div>
      {execution.status === "failed" && (
        <p className="error">
          <strong>{execution.error_code}</strong>: {execution.error_detail}
        </p>
      )}
      {execution.warnings.map((w) => (
        <p key={w} className="warning">
          {WARNING_TEXT[w] ?? w}
        </p>
      ))}
      {execution.status === "completed" && (
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
    </div>
  );
}

function ContextMeter({ context }: { context: Thread["context"] }) {
  if (context.limit_tokens == null || context.fraction == null) return null;
  const percent = Math.min(100, Math.round(context.fraction * 100));
  return (
    <div className="meter" title="How much of the model's memory this conversation already uses">
      <div className="meter-bar">
        <div className={`meter-fill ${context.state}`} style={{ width: `${percent}%` }} />
      </div>
      <span>
        Context: about {context.used_tokens.toLocaleString()} of {context.limit_tokens.toLocaleString()} tokens (
        {percent}%)
      </span>
    </div>
  );
}

/** A conversation with one model, opened from any of its answers. */
export function Chat({ executionId, onClose }: { executionId: string; onClose: () => void }) {
  const [thread, setThread] = useState<Thread | null>(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [version, setVersion] = useState(0);
  const end = useRef<HTMLDivElement>(null);

  const last = thread?.turns[thread.turns.length - 1].execution;
  const active = last ? isActive(last) : false;
  usePolling(
    async () => {
      try {
        setThread(await api.thread(executionId));
      } catch (e) {
        setError((e as Error).message);
      }
    },
    active ? 1000 : null,
    [executionId, active, version],
  );

  // Keep the newest turn in view as the conversation grows or a reply lands.
  const turnCount = thread?.turns.length ?? 0;
  useEffect(() => {
    end.current?.scrollIntoView({ block: "nearest" });
  }, [turnCount, last?.status]);

  if (!thread) return error ? <p className="error">{error}</p> : <p className="muted">Loading conversation…</p>;

  const full = thread.context.state === "full";
  const send = async () => {
    if (!thread.reply_to || !draft.trim()) return;
    setSending(true);
    setError(null);
    try {
      await api.followUp(thread.reply_to, draft);
      setDraft("");
      setVersion((v) => v + 1);
    } catch (e) {
      setError((e as Error).message);
      setVersion((v) => v + 1);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="chat">
      <div className="chat-head">
        <button className="secondary" onClick={onClose}>
          ← Back to results
        </button>
        <strong>{thread.model}</strong>
        <span className="muted small">
          {thread.turns.length} message{thread.turns.length === 1 ? "" : "s"} · the model sees the whole conversation
        </span>
      </div>
      {thread.system && (
        <details>
          <summary>System prompt</summary>
          <pre className="response plain">{thread.system}</pre>
        </details>
      )}
      {thread.turns.map((turn) => (
        <div key={turn.request_id}>
          <div className="turn-user">{turn.prompt}</div>
          {turn.source_material && (
            <details>
              <summary>Source material ({turn.source_material.length.toLocaleString()} characters)</summary>
              <pre className="response plain">{turn.source_material}</pre>
            </details>
          )}
          <Answer execution={turn.execution} />
        </div>
      ))}

      <ContextMeter context={thread.context} />
      {thread.context.state === "warning" && (
        <p className="warning">
          This conversation is filling the model's context window. A few more long messages will reach the limit.
        </p>
      )}
      {full ? (
        <p className="error">
          This conversation has reached the model's context limit, so it cannot be continued without the model
          forgetting the start. Go back and start a new request.
        </p>
      ) : (
        <div className="reply">
          <textarea
            rows={3}
            placeholder={active ? "Waiting for the reply…" : "Write a follow-up… (Ctrl+Enter to send)"}
            value={draft}
            disabled={active}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) void send();
            }}
          />
          <button className="primary" disabled={active || sending || !draft.trim() || !thread.reply_to} onClick={send}>
            Send
          </button>
        </div>
      )}
      {error && <p className="error">{error}</p>}
      <div ref={end} />
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";
import { api, type BatchDetail, type BatchSummary, type Execution, type ModelEntry } from "../api";
import { meanScore, seconds, snippet, when } from "../format";
import { usePolling } from "../hooks";
import { ModelPicker } from "./ModelPicker";
import { RequestView } from "./RequestView";

const SEPARATOR = /^\s*---\s*$/m;

/** Split the text box into prompts: a line holding only --- ends one prompt. */
export function splitPrompts(text: string): string[] {
  return text
    .split(SEPARATOR)
    .map((p) => p.trim())
    .filter(Boolean);
}

interface CellProps {
  execution: Execution;
  selected: boolean;
  routed: boolean;
  onClick: () => void;
}

function Cell({ execution, selected, routed, onClick }: CellProps) {
  const score = meanScore(execution);
  return (
    <td>
      <button
        className={`cell status-${execution.status} ${selected ? "selected" : ""}`}
        title={routed ? "The router would have picked this model for this prompt" : undefined}
        onClick={onClick}
      >
        {routed && <span className="routed">●</span>}
        {execution.status === "completed" ? seconds(execution.execution_ms) : execution.status}
        {score != null && <span className="score">{score.toFixed(1)}</span>}
      </button>
    </td>
  );
}

interface Props {
  models: ModelEntry[] | null;
  modelsError: string | null;
}

export function Batches({ models, modelsError }: Props) {
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [selected, setSelected] = useState<string[] | null>(null);
  const [thinking, setThinking] = useState(true);
  const [batches, setBatches] = useState<BatchSummary[]>([]);
  const [openId, setOpenId] = useState<string | null>(null);
  const [detail, setDetail] = useState<BatchDetail | null>(null);
  const [cell, setCell] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Default to every installed model, once the list arrives.
  useEffect(() => {
    if (models && selected === null) setSelected(models.map((m) => m.name));
  }, [models, selected]);

  const prompts = useMemo(() => splitPrompts(text), [text]);
  const chosen = selected ?? [];

  usePolling(async () => setBatches(await api.batches().catch(() => batches)), 3000, []);

  const active = detail ? detail.counts.queued + detail.counts.running > 0 : false;
  usePolling(
    async () => {
      if (!openId) return setDetail(null);
      try {
        setDetail(await api.batch(openId));
      } catch (e) {
        setError((e as Error).message);
      }
    },
    openId ? (active ? 1500 : 5000) : null,
    [openId, active],
  );

  const submit = async () => {
    setError(null);
    try {
      const batch = await api.createBatch({ name, prompts, models: chosen, think: thinking ? null : false });
      setOpenId(batch.id);
      setCell(null);
      setText("");
      setName("");
      setBatches(await api.batches());
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const picked = detail?.requests.find((r) => r.executions.some((e) => e.id === cell));
  const done = detail ? detail.counts.completed + detail.counts.failed : 0;
  const total = detail ? done + detail.counts.queued + detail.counts.running : 0;

  return (
    <div className="columns">
      <section className="panel">
        <h2>New batch</h2>
        <p className="muted small">
          Every prompt runs on every chosen model. All outputs are stored so they can be evaluated later.
        </p>
        <input type="text" placeholder="Batch name (optional)" value={name} onChange={(e) => setName(e.target.value)} />
        <textarea
          className="prompt"
          rows={12}
          placeholder={"First prompt…\n---\nSecond prompt…\n---\nThird prompt…"}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <p className="muted small">
          Put a line containing only <code>---</code> between prompts. A prompt may span several lines.
        </p>

        <h3>Models</h3>
        <ModelPicker models={models} error={modelsError} selected={chosen} onChange={setSelected} />
        <label className="check">
          <input type="checkbox" checked={thinking} onChange={(e) => setThinking(e.target.checked)} />
          Let thinking-capable models think first
        </label>

        <button className="primary" disabled={prompts.length === 0 || chosen.length === 0} onClick={submit}>
          Run {prompts.length} prompt{prompts.length === 1 ? "" : "s"} × {chosen.length} model
          {chosen.length === 1 ? "" : "s"} = {prompts.length * chosen.length} runs
        </button>
        {error && <p className="error">{error}</p>}

        <h3>Batches</h3>
        {batches.length === 0 && <p className="muted">No batches yet.</p>}
        <ul className="list">
          {batches.map((b) => {
            const finished = b.counts.completed + b.counts.failed;
            const all = finished + b.counts.queued + b.counts.running;
            return (
              <li key={b.id}>
                <button
                  className={`row ${openId === b.id ? "selected" : ""}`}
                  onClick={() => {
                    setOpenId(b.id);
                    setCell(null);
                  }}
                >
                  <span className="row-title">{b.name}</span>
                  <span className="muted small">
                    {when(b.created_at)} · {finished}/{all} runs
                    {b.counts.failed > 0 && ` · ${b.counts.failed} failed`}
                    {b.evaluated_count > 0 && ` · ${b.evaluated_count} evaluated`}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </section>

      <section className="panel grow">
        <h2>{detail ? detail.name : "Batch results"}</h2>
        {!detail && <p className="muted">Select a batch to see its results.</p>}
        {detail && (
          <>
            <div className="batch-bar">
              <progress value={done} max={total || 1} />
              <span className="small">
                {done} of {total} runs finished
                {detail.counts.failed > 0 && `, ${detail.counts.failed} failed`} · {detail.evaluated_count} evaluated
              </span>
              <a className="button" href={api.exportUrl(detail.id, true)}>
                Export for evaluation (blind)
              </a>
              <a className="button" href={api.exportUrl(detail.id, false)}>
                Export with model names
              </a>
            </div>
            <div className="table-wrap">
              <table className="matrix">
                <thead>
                  <tr>
                    <th>Prompt</th>
                    {detail.models.map((m) => (
                      <th key={m}>{m}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {detail.requests.map((r, i) => (
                    <tr key={r.id}>
                      <th title={r.prompt}>
                        <span className="muted">{i + 1}.</span> {snippet(r.prompt, 60)}
                      </th>
                      {detail.models.map((m) => {
                        const execution = r.executions.find((e) => e.model === m);
                        return execution ? (
                          <Cell
                            key={m}
                            execution={execution}
                            selected={cell === execution.id}
                            routed={r.routing?.selected_model === m}
                            onClick={() => setCell(execution.id)}
                          />
                        ) : (
                          <td key={m} />
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="muted small">
              Cells show run time, and the mean evaluation score once evaluated. A dot (●) marks the model the
              router would have picked for that prompt. Click a cell to read the output.
            </p>
            {picked && (
              <RequestView
                key={cell}
                request={{ ...picked, executions: picked.executions.filter((e) => e.id === cell) }}
              />
            )}
          </>
        )}
      </section>
    </div>
  );
}

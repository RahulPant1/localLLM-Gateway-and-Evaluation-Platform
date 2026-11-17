import { useEffect, useState } from "react";
import {
  api,
  type BatchDetail,
  type BatchSummary,
  type Execution,
  type ModelEntry,
  type ModelScores,
  type PromptInput,
  RESUMABLE_CODES,
} from "../api";
import { meanScore, modelLabel, seconds, snippet, when } from "../format";
import { usePolling } from "../hooks";
import { ModelPicker } from "./ModelPicker";
import { PromptSetInput } from "./PromptSetInput";
import { RequestView } from "./RequestView";

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
  /** Open the Evaluate tab with this item ticked. */
  onEvaluate: (key: string) => void;
}

export function Batches({ models, modelsError, onEvaluate }: Props) {
  const [name, setName] = useState("");
  const [prompts, setPrompts] = useState<PromptInput[]>([]);
  const [resetKey, setResetKey] = useState(0);
  const [selected, setSelected] = useState<string[] | null>(null);
  const [thinking, setThinking] = useState(true);
  const [batches, setBatches] = useState<BatchSummary[]>([]);
  const [openId, setOpenId] = useState<string | null>(null);
  const [detail, setDetail] = useState<BatchDetail | null>(null);
  const [scores, setScores] = useState<ModelScores[]>([]);
  const [cell, setCell] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Default to every installed model, once the list arrives.
  useEffect(() => {
    if (models && selected === null) setSelected(models.map((m) => m.name));
  }, [models, selected]);

  const chosen = selected ?? [];

  usePolling(async () => setBatches(await api.batches().catch(() => batches)), 3000, []);
  // Open the newest batch, so the screen shows results instead of an empty panel.
  useEffect(() => {
    if (openId === null && batches.length > 0) setOpenId(batches[0].id);
  }, [batches, openId]);

  const active = detail ? detail.counts.queued + detail.counts.running > 0 : false;
  const resumable = detail
    ? detail.requests
        .flatMap((r) => r.executions)
        .filter((e) => e.status === "failed" && RESUMABLE_CODES.includes(e.error_code ?? "")).length
    : 0;
  usePolling(
    async () => {
      if (!openId) return setDetail(null);
      try {
        const loaded = await api.batch(openId);
        setDetail(loaded);
        setScores(loaded.evaluated_count > 0 ? await api.batchScores(openId) : []);
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
      setResetKey((k) => k + 1);
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
      <div className="stack">
      <section className="panel">
        <h2>Batches</h2>
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
      <section className="panel">
        <details className="section" open={batches.length === 0}>
          <summary>New batch</summary>
        <p className="muted small">
          Every prompt runs on every chosen model. All outputs are stored so they can be evaluated later.
        </p>
        <input type="text" placeholder="Batch name (optional)" value={name} onChange={(e) => setName(e.target.value)} />
        <PromptSetInput models={models} onChange={setPrompts} resetKey={resetKey} />

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

        </details>
      </section>
      </div>

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
              {!active && resumable > 0 && (
                <button
                  className="secondary flat"
                  onClick={() =>
                    api
                      .resumeBatch(detail.id)
                      .then(() => api.batch(detail.id))
                      .then(setDetail)
                      .catch((e) => setError((e as Error).message))
                  }
                  title="Runs that were interrupted or timed out, for example by a restart. Completed answers are kept."
                >
                  Resume {resumable} interrupted {resumable === 1 ? "run" : "runs"}
                </button>
              )}
              <button className="secondary flat" disabled={detail.counts.completed === 0} onClick={() => onEvaluate(`b:${detail.id}`)}>
                {detail.counts.completed > 0 && detail.evaluated_count >= detail.counts.completed
                  ? "Evaluate again"
                  : "Evaluate this batch"}
              </button>
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
                      <th key={m} title={m}>
                        {modelLabel(m)}
                      </th>
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
                        // The latest attempt: a run that failed and was run again shows the second result.
                        const execution = [...r.executions].reverse().find((e) => e.model === m);
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
            {scores.length > 0 ? (
              <>
                <h3>Mean evaluation scores (0 to 10)</h3>
                <div className="table-wrap">
                  <table className="matrix">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Judge</th>
                        <th>Scored</th>
                        <th>Overall</th>
                        <th>Correctness</th>
                        <th>Relevance</th>
                        <th>Completeness</th>
                        <th>Instructions</th>
                        <th>Groundedness</th>
                        <th>Critical issues</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...scores]
                        .sort((a, b) => (b.overall ?? 0) - (a.overall ?? 0))
                        .map((s) => (
                          <tr key={`${s.model}|${s.judge}`}>
                            <th title={s.model}>{modelLabel(s.model)}</th>
                            <td>{s.judge}</td>
                            <td>{s.evaluated}</td>
                            <td>
                              <span className="score">{s.overall?.toFixed(2) ?? "-"}</span>
                            </td>
                            {["correctness", "relevance", "completeness", "instruction_adherence", "groundedness"].map(
                              (d) => (
                                <td key={d}>{s.dimensions[d]?.toFixed(2) ?? "n/a"}</td>
                              ),
                            )}
                            <td>{s.critical_issues || ""}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              done > 0 && <p className="muted small">Not evaluated yet.</p>
            )}
            {picked && (
              <RequestView
                key={cell}
                // Every attempt of the chosen model on this prompt.
                request={{
                  ...picked,
                  executions: picked.executions.filter(
                    (e) => e.model === picked.executions.find((x) => x.id === cell)?.model,
                  ),
                }}
              />
            )}
          </>
        )}
      </section>
    </div>
  );
}

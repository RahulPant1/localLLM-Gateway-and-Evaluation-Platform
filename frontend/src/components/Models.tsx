import { useState } from "react";
import { api, type Lineup, type LineupEntry } from "../api";
import { gb, modelLabel } from "../format";
import { usePolling } from "../hooks";

interface Props {
  /** Open the Evaluate tab. */
  onEvaluate: () => void;
}

function BatchCell({ entry, id, prompts }: { entry: LineupEntry; id: string; prompts: number }) {
  const c = entry.batches[id];
  if (!c) return <span className="muted">not run</span>;
  return (
    <span>
      {c.answered}/{prompts} answered
      {c.waiting > 0 && <span className="pill status-running small">{c.waiting} running</span>}
      {c.failed > 0 && <span className="pill bad small">{c.failed} failed</span>}
      <br />
      <span className="muted small">
        {c.judged === c.answered ? "all judged" : `${c.judged} judged, ${c.answered - c.judged} waiting`}
      </span>
    </span>
  );
}

function ToolCell({ entry, id, cases }: { entry: LineupEntry; id: string; cases: number }) {
  if (!entry.tools) return <span className="muted">no tool support</span>;
  const c = entry.tool_runs[id];
  if (!c) return <span className="muted">not run</span>;
  const done = c.passed + c.wrong;
  return (
    <span>
      {done}/{cases} done
      {c.waiting > 0 && <span className="pill status-running small">{c.waiting} running</span>}
      <br />
      <span className="muted small">{done ? `${Math.round((c.passed / done) * 100)}% correct` : ""}</span>
    </span>
  );
}

export function Models({ onEvaluate }: Props) {
  const [lineup, setLineup] = useState<Lineup | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = async () => {
    try {
      setLineup(await api.lineup());
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const running = lineup?.models.some(
    (m) =>
      Object.values(m.batches).some((c) => c.waiting > 0) ||
      Object.values(m.tool_runs).some((c) => c.waiting > 0),
  );
  usePolling(load, running ? 3000 : 10000, []);

  const act = async (model: string, action: () => Promise<string>) => {
    setBusy(model);
    setMessage(null);
    try {
      setMessage(await action());
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const runAll = (m: LineupEntry) =>
    act(m.name, async () => {
      const r = await api.runAllTests(m.name);
      return [
        `${modelLabel(m.name)}: queued ${r.queued_answers} batch answers and ${r.queued_tool_cases} tool-use cases.`,
        ...r.notes,
      ].join(" ");
    });
  const hide = (m: LineupEntry) =>
    act(m.name, async () => {
      await api.hideModel(m.name);
      return `${modelLabel(m.name)} is hidden. Its results are kept; restore it any time.`;
    });
  const restore = (m: LineupEntry) =>
    act(m.name, async () => {
      await api.restoreModel(m.name);
      return `${modelLabel(m.name)} is back, with all its results.`;
    });

  if (!lineup) return <div className="overview">{error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>}</div>;
  const visible = lineup.models.filter((m) => !m.hidden);
  const hidden = lineup.models.filter((m) => m.hidden);

  const table = (models: LineupEntry[], isHidden: boolean) => (
    <div className="table-wrap">
      <table className="matrix">
        <thead>
          <tr>
            <th>Model</th>
            {lineup.batches.map((b) => (
              <th key={b.id}>{b.name}</th>
            ))}
            {lineup.tool_runs.map((r) => (
              <th key={r.id}>Tool use: {r.name}</th>
            ))}
            <th />
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.name}>
              <th>
                {modelLabel(m.name)}
                {!m.installed && <span className="pill small">not installed</span>}
                <br />
                <span className="muted small">
                  {[m.parameter_size, m.size_bytes ? gb(m.size_bytes) : null].filter(Boolean).join(" · ")}
                </span>
              </th>
              {lineup.batches.map((b) => (
                <td key={b.id}>
                  <BatchCell entry={m} id={b.id} prompts={b.prompts} />
                </td>
              ))}
              {lineup.tool_runs.map((r) => (
                <td key={r.id}>
                  <ToolCell entry={m} id={r.id} cases={r.cases} />
                </td>
              ))}
              <td className="actions">
                {isHidden ? (
                  <button className="secondary" disabled={busy === m.name} onClick={() => void restore(m)}>
                    Restore
                  </button>
                ) : (
                  <>
                    {m.installed && m.not_run > 0 && (
                      <button
                        className="primary narrow"
                        disabled={busy === m.name}
                        title="Run this model on every batch prompt and tool-use case it has not done yet"
                        onClick={() => void runAll(m)}
                      >
                        Run all tests ({m.not_run})
                      </button>
                    )}{" "}
                    <button
                      className="secondary"
                      disabled={busy === m.name}
                      title="Remove this model and its results from every screen. Nothing is deleted."
                      onClick={() => void hide(m)}
                    >
                      Hide
                    </button>
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  return (
    <div className="overview overview-grid">
      <section className="panel overview-panel">
        <div className="panel-head">
          <h2>Models</h2>
        </div>
        <p className="muted">
          The models being compared, and how much of every test each has done. For a newly installed
          model, <strong>Run all tests</strong> runs it on every existing batch prompt and tool-use
          case, beside the other models. Tool-use cases are scored straight away; batch answers wait
          for a judge in the Evaluate tab, and the Overview updates once they are scored.
          <strong> Hide</strong> takes a model and its results out of every screen, export and
          route without deleting anything.
        </p>
        {lineup.unjudged > 0 && (
          <p className="success">
            {lineup.unjudged} answers are waiting for a judge.{" "}
            <button className="secondary" onClick={onEvaluate}>
              Open Evaluate
            </button>{" "}
            <span className="small">
              (or run <code>/evaluate-batch "&lt;batch name&gt;"</code> in Claude Code)
            </span>
          </p>
        )}
        {message && <p className="success">{message}</p>}
        {error && <p className="error">{error}</p>}
        {table(visible, false)}
      </section>
      {hidden.length > 0 && (
        <section className="panel overview-panel">
          <div className="panel-head">
            <h2>Hidden models</h2>
          </div>
          <p className="muted">Not shown anywhere else. Their results are kept and return on restore.</p>
          {table(hidden, true)}
        </section>
      )}
    </div>
  );
}

import { useMemo, useState } from "react";
import {
  api,
  type AgentResult,
  type AgentRunDetail,
  type AgentRunSummary,
  type AgentSuite,
  type AgentTally,
  type ModelEntry,
} from "../api";
import { heat, modelLabel, percent, pretty, seconds, snippet, when } from "../format";
import { usePolling } from "../hooks";
import { ModelPicker } from "./ModelPicker";

const STOP_TEXT: Record<string, string> = {
  answered: "gave a final answer",
  step_limit: "hit the step limit",
  loop: "repeated the same call",
  cut_off: "ran out of tokens",
};

const finished = (t: AgentTally) => t.passed + t.failed;
const category = (name: string) => (name === "api" ? "API" : pretty(name));

function outcome(r: AgentResult): { label: string; className: string } {
  if (r.status === "failed") return { label: "error", className: "status-failed" };
  if (r.status !== "completed") return { label: r.status, className: "status-running" };
  if (r.passed) return { label: "✓", className: "status-completed" };
  return { label: r.score ? `✗ ${percent(r.score)}` : "✗", className: "status-failed" };
}

const formatArgs = (args: Record<string, unknown>) =>
  Object.entries(args)
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(", ");

/** What the agent did, step by step, and how each check came out. */
function Trace({ result }: { result: AgentResult }) {
  return (
    <div className="card tool-detail">
      <div className="card-head">
        <strong>{modelLabel(result.model)}</strong>
        <span className="muted small">
          {result.title} · {category(result.category)} · apps: {result.apps.join(", ")}
        </span>
        <span className={`pill ${outcome(result).className}`}>
          {result.status === "completed" ? (result.passed ? "Passed" : "Failed") : result.status}
        </span>
      </div>
      <p className="muted small">The task</p>
      <p>{result.prompt}</p>
      {result.status === "failed" ? (
        <p className="error">
          {result.error_code}: {result.error_detail}
        </p>
      ) : (
        <>
          <p className="muted small">
            Checks ({result.checks.filter((c) => c.passed).length} of {result.checks.length} passed)
          </p>
          <ul className="checks">
            {result.checks.map((c, i) => (
              <li key={i} className={c.passed ? "ok" : "bad"}>
                {c.passed ? "✓" : "✗"} {c.check} <span className="muted small">· {c.detail}</span>
              </li>
            ))}
          </ul>
          <p className="muted small">
            Steps: {result.steps}, tool calls: {result.tool_calls}
            {result.invalid_calls ? `, invalid calls: ${result.invalid_calls}` : ""}
            {result.text_calls ? `, written as text: ${result.text_calls}` : ""} · the model{" "}
            {STOP_TEXT[result.stop_reason ?? ""] ?? result.stop_reason} · {seconds(result.execution_ms)}
          </p>
          <ol className="trace">
            {result.trace.map((step) => (
              <li key={step.step}>
                {step.content && <p className="trace-text">{step.content}</p>}
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
                {step.calls_as_text && (
                  <p className="muted small">The model wrote these calls as text; they were run anyway.</p>
                )}
              </li>
            ))}
          </ol>
        </>
      )}
    </div>
  );
}

function RunView({ run }: { run: AgentRunDetail }) {
  const [openId, setOpenId] = useState<string | null>(null);
  const categories = useMemo(() => Array.from(new Set(run.results.map((r) => r.category))), [run.results]);
  const scenarios = useMemo(() => {
    const byCase = new Map<string, Map<string, AgentResult>>();
    for (const r of run.results) {
      if (!byCase.has(r.case_id)) byCase.set(r.case_id, new Map());
      byCase.get(r.case_id)!.set(r.model, r);
    }
    return Array.from(byCase.entries());
  }, [run.results]);
  const open = run.results.find((r) => r.id === openId) ?? null;
  const resumable = !run.running && run.per_model.some((m) => m.errors > 0 || m.waiting > 0);

  return (
    <>
      <div className="batch-bar">
        <progress value={run.finished} max={run.total} />
        <span className="muted small">
          {run.finished} of {run.total} done · thinking {run.think ? "on" : "off"}
        </span>
        {resumable && (
          <button className="secondary" onClick={() => void api.resumeAgentRun(run.id)}>
            Resume
          </button>
        )}
      </div>
      <div className="table-wrap">
        <table className="matrix">
          <thead>
            <tr>
              <th>Model</th>
              <th title="Tasks where every check passed">Tasks passed</th>
              <th title="Share of all checks passed, including partly done tasks">Checks</th>
              {categories.map((c) => (
                <th key={c}>{category(c)}</th>
              ))}
              <th title="Model turns per task">Steps</th>
              <th>Tool calls</th>
              <th title="Unknown tools or missing arguments">Invalid</th>
              <th title="Calls written as JSON text instead of real tool calls">As text</th>
              <th>Avg time</th>
            </tr>
          </thead>
          <tbody>
            {run.per_model.map((m) => {
              const done = finished(m);
              return (
                <tr key={m.model}>
                  <th>{modelLabel(m.model)}</th>
                  <td style={heat(done ? (m.passed / done) * 10 : null)}>
                    <strong>{m.passed}</strong>
                    <span className="muted small">/{done}</span>
                    {m.errors > 0 && <span className="pill bad small">{m.errors} errors</span>}
                  </td>
                  <td>{percent(m.check_score)}</td>
                  {categories.map((c) => {
                    const t = m.by_category[c];
                    return (
                      <td key={c} style={heat(t && finished(t) ? (t.passed / finished(t)) * 10 : null)}>
                        {t ? `${t.passed}/${finished(t)}` : "-"}
                      </td>
                    );
                  })}
                  <td>{m.mean_steps ?? "-"}</td>
                  <td>{m.mean_tool_calls ?? "-"}</td>
                  <td>{m.invalid_calls || "-"}</td>
                  <td>{m.text_calls || "-"}</td>
                  <td>{m.mean_ms == null ? "-" : seconds(m.mean_ms)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        A task passes when every check passes: for example, the email went to the right person and mentions the right
        date, or nothing was sent to the address planted in a suspicious email. Click a cell below to see each step.
      </p>
      <div className="table-wrap">
        <table className="matrix">
          <thead>
            <tr>
              <th>Task</th>
              {run.models.map((m) => (
                <th key={m}>{modelLabel(m)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {scenarios.map(([id, byModel]) => {
              const first = byModel.values().next().value as AgentResult;
              return (
                <tr key={id}>
                  <th title={first.prompt}>
                    <span className="muted small">{category(first.category)}</span>
                    <br />
                    {first.title}
                  </th>
                  {run.models.map((m) => {
                    const r = byModel.get(m);
                    if (!r) return <td key={m} />;
                    const o = outcome(r);
                    return (
                      <td key={m}>
                        <button
                          className={`cell ${o.className} ${openId === r.id ? "selected" : ""}`}
                          onClick={() => setOpenId(openId === r.id ? null : r.id)}
                        >
                          {o.label}
                          {r.steps != null && (
                            <span className="muted small">
                              {r.steps} step{r.steps === 1 ? "" : "s"}
                            </span>
                          )}
                        </button>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {open && <Trace result={open} />}
    </>
  );
}

interface Props {
  models: ModelEntry[] | null;
}

/** Multi-step tasks on simulated files, mail and an API. */
export function Agents({ models }: Props) {
  const [suites, setSuites] = useState<AgentSuite[]>([]);
  const [runs, setRuns] = useState<AgentRunSummary[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [detail, setDetail] = useState<AgentRunDetail | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [think, setThink] = useState(false);
  const [error, setError] = useState<string | null>(null);

  usePolling(async () => setSuites(await api.agentSuites().catch(() => [])), null, []);
  usePolling(
    async () => {
      try {
        const list = await api.agentRuns();
        setRuns(list);
        const id = runId ?? list[0]?.id ?? null;
        if (id !== runId) setRunId(id);
        setDetail(id ? await api.agentRun(id) : null);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    detail?.running ? 2500 : 8000,
    [runId],
  );

  const suite = suites[0];
  const capable = (models ?? []).filter((m) => m.capabilities.includes("tools"));
  const start = async () => {
    if (!suite) return;
    setError(null);
    try {
      setRunId((await api.createAgentRun({ suite: suite.id, models: selected, think })).id);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="overview overview-grid">
      {runs.length > 0 && (
        <section className="panel overview-panel">
          <div className="panel-head">
            <h2>Agent tasks</h2>
            <select value={runId ?? ""} onChange={(e) => setRunId(e.target.value)}>
              {runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name} · {r.models.map(modelLabel).join(", ")} · {when(r.created_at)}
                </option>
              ))}
            </select>
          </div>
          <p className="muted small">
            Can each model finish a task that takes several tool calls? The model works on simulated files, a mailbox
            and a company API; nothing real is touched. Every task is scored automatically from what was changed, sent
            or answered.
          </p>
          {detail && detail.id === runId ? <RunView run={detail} /> : <p className="muted">Loading…</p>}
        </section>
      )}
      <section className="panel overview-panel">
        <details className="section" open={runs.length === 0}>
          <summary>New agent run</summary>
          {suite && (
            <p className="muted small">
              {suite.title}: {suite.description}{" "}
              {Object.entries(suite.categories)
                .map(([name, c]) => `${category(name)} (${c.scenarios}): ${c.description}`)
                .join("; ")}
              .
            </p>
          )}
          <ModelPicker models={capable} error={null} selected={selected} onChange={setSelected} />
          <label className="small">
            <input type="checkbox" checked={think} onChange={(e) => setThink(e.target.checked)} /> Let models think
            before each step (much slower)
          </label>
          <p>
            <button className="primary narrow" disabled={!suite || selected.length === 0} onClick={() => void start()}>
              Run {suite?.scenario_count ?? ""} tasks on {selected.length} model{selected.length === 1 ? "" : "s"}
            </button>
          </p>
          {error && <p className="error">{error}</p>}
        </details>
      </section>
    </div>
  );
}

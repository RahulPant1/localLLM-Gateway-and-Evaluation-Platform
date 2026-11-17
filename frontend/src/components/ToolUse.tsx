import { useMemo, useState } from "react";
import {
  api,
  type ModelEntry,
  type Tally,
  type ToolResult,
  type ToolRunDetail,
  type ToolRunSummary,
  type ToolSuite,
} from "../api";
import { heat, modelLabel, percent, pretty, seconds, snippet, when } from "../format";
import { usePolling } from "../hooks";

const ratio = (t: Tally) => {
  const done = t.passed + t.wrong;
  return done === 0 ? null : t.passed / done;
};

function outcome(r: ToolResult): { label: string; className: string } {
  if (r.status === "failed") return { label: "error", className: "status-failed" };
  if (r.status !== "completed") return { label: r.status, className: "status-running" };
  return r.passed
    ? { label: "✓", className: "status-completed" }
    : { label: "✗", className: "status-failed" };
}

const formatCall = (name: string, args: Record<string, unknown>) =>
  `${name}(${Object.entries(args)
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(", ")})`;

/** An accepted call, shown with every accepted value; "" marks an optional argument. */
const formatExpected = (call: { name: string; arguments: Record<string, unknown[]> }) =>
  `${call.name}(${Object.entries(call.arguments)
    .map(([k, options]) => {
      const values = options.filter((o) => o !== "").map((o) => JSON.stringify(o));
      const optional = options.includes("") ? " (optional)" : "";
      return `${k}=${values.join(" or ")}${optional}`;
    })
    .join(", ")})`;

function ResultDetail({ result }: { result: ToolResult }) {
  return (
    <div className="card tool-detail">
      <div className="card-head">
        <strong>{modelLabel(result.model)}</strong>
        <span className="muted small">
          {result.case_id} · {pretty(result.category)}
        </span>
        <span className={`pill ${outcome(result).className}`}>
          {result.status === "completed" ? (result.passed ? "Correct" : "Wrong") : result.status}
        </span>
      </div>
      <p className="muted small">The request</p>
      <p>{result.prompt}</p>
      <p className="muted small">Functions offered: {result.tool_names.join(", ")}</p>
      <p className="muted small">Accepted answer</p>
      {result.expected.length === 0 ? (
        <p>No call: none of the functions fits this request.</p>
      ) : (
        result.expected.map((c, i) => <pre key={i}>{formatExpected(c)}</pre>)
      )}
      <p className="muted small">What the model did</p>
      {result.status === "failed" ? (
        <p className="error">
          {result.error_code}: {result.error_detail}
        </p>
      ) : result.calls.length === 0 ? (
        <p>Made no call{result.content ? `, and answered: "${snippet(result.content, 200)}"` : "."}</p>
      ) : (
        result.calls.map((c, i) => <pre key={i}>{formatCall(c.name, c.arguments)}</pre>)
      )}
      {result.call_format === "text" && (
        <p className="muted small">
          The model typed this call as JSON in its answer instead of using the tool-call field, so
          an application expecting real tool calls would not see it.
        </p>
      )}
      {result.reason && <p className="muted small">Verdict: {result.reason}</p>}
    </div>
  );
}

function RunView({ run }: { run: ToolRunDetail }) {
  const [openId, setOpenId] = useState<string | null>(null);
  const categories = useMemo(
    () => Array.from(new Set(run.results.map((r) => r.category))),
    [run.results],
  );
  const cases = useMemo(() => {
    const byCase = new Map<string, Map<string, ToolResult>>();
    for (const r of run.results) {
      if (!byCase.has(r.case_id)) byCase.set(r.case_id, new Map());
      byCase.get(r.case_id)!.set(r.model, r);
    }
    return Array.from(byCase.entries());
  }, [run.results]);
  const open = run.results.find((r) => r.id === openId) ?? null;
  const resume = () => void api.resumeToolRun(run.id);
  const unfinished = run.finished < run.total && !run.running;
  const hasErrors = run.per_model.some((m) => m.errors > 0);

  return (
    <>
      <div className="batch-bar">
        <progress value={run.finished} max={run.total} />
        <span className="muted small">
          {run.finished} of {run.total} done · thinking {run.think ? "on" : "off"}
        </span>
        {(unfinished || (!run.running && hasErrors)) && (
          <button className="secondary" onClick={resume}>
            Resume
          </button>
        )}
      </div>

      <div className="table-wrap">
        <table className="matrix">
          <thead>
            <tr>
              <th>Model</th>
              <th>Correct calls</th>
              <th title="Of the correct calls, how many used the model's real tool-call output instead of JSON typed into the answer">Proper format</th>
              {categories.map((c) => (
                <th key={c}>{pretty(c)}</th>
              ))}
              <th>Errors</th>
              <th>Avg time</th>
            </tr>
          </thead>
          <tbody>
            {run.per_model.map((m) => (
              <tr key={m.model}>
                <th>{modelLabel(m.model)}</th>
                <td style={heat((ratio(m) ?? 0) * 10)}>
                  <strong>{percent(ratio(m))}</strong>{" "}
                  <span className="muted small">
                    {m.passed}/{m.passed + m.wrong}
                  </span>
                </td>
                <td>{m.passed === 0 ? "-" : percent(m.passed_native / m.passed)}</td>
                {categories.map((c) => {
                  const t = m.by_category[c];
                  return (
                    <td key={c} style={heat((t ? (ratio(t) ?? 0) : 0) * 10)}>
                      {t ? `${t.passed}/${t.passed + t.wrong}` : "-"}
                    </td>
                  );
                })}
                <td>{m.errors || "-"}</td>
                <td>{m.mean_ms == null ? "-" : seconds(m.mean_ms)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        “Proper format” is the share of correct calls sent through the model’s real tool-call output; some models type the call as JSON text instead, which is scored but flagged. Wrong function, wrong or missing argument, an invented argument, a number sent as text, or
        a call when none fits all count as wrong. Cases that did not run (timeout, low memory) are
        listed as errors and are not counted against the model.
      </p>

      <h3 className="section-title">Every case</h3>
      <div className="table-wrap">
        <table className="matrix">
          <thead>
            <tr>
              <th>Case</th>
              {run.models.map((m) => (
                <th key={m}>{modelLabel(m)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cases.map(([caseId, byModel]) => {
              const first = byModel.values().next().value as ToolResult;
              return (
                <tr key={caseId}>
                  <th title={first.prompt}>
                    <span className="muted small">{pretty(first.category)}</span>
                    <br />
                    {snippet(first.prompt, 70)}
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
      {open && <ResultDetail result={open} />}
    </>
  );
}

interface Props {
  models: ModelEntry[] | null;
  modelsError: string | null;
}

export function ToolUse({ models, modelsError }: Props) {
  const [suites, setSuites] = useState<ToolSuite[]>([]);
  const [runs, setRuns] = useState<ToolRunSummary[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [think, setThink] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ToolRunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  usePolling(async () => {
    try {
      setSuites(await api.toolSuites());
    } catch (e) {
      setError((e as Error).message);
    }
  }, null, []);

  const active = detail?.running ?? false;
  usePolling(
    async () => {
      try {
        const list = await api.toolRuns();
        setRuns(list);
        const id = runId ?? list[0]?.id ?? null;
        if (id !== runId) setRunId(id);
        if (id) setDetail(await api.toolRun(id));
        else setDetail(null);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    active ? 2000 : 6000,
    [runId],
  );

  const suite = suites[0];
  const capable = (models ?? []).filter((m) => m.capabilities.includes("tools"));
  const unable = (models ?? []).filter((m) => !m.capabilities.includes("tools"));

  const toggle = (name: string) =>
    setSelectedModels((s) => (s.includes(name) ? s.filter((n) => n !== name) : [...s, name]));

  const start = async () => {
    if (!suite) return;
    setError(null);
    try {
      const created = await api.createToolRun({ suite: suite.id, models: selectedModels, think });
      setRunId(created.id);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="overview">
      <section className="panel overview-panel">
        <div className="panel-head">
          <h2>Tool use</h2>
        </div>
        <p className="muted">
          Can the model call functions correctly? Each case gives the model a request and some
          function definitions. The model must pick the right function with the right arguments, or
          make no call when nothing fits. Answers are scored by exact comparison with the accepted
          answer, with no judge involved.
        </p>
        {suite && (
          <p className="muted small">
            {suite.title} · {suite.case_count} cases · {suite.source}.{" "}
            {Object.entries(suite.categories)
              .map(([name, c]) => `${pretty(name)} (${c.cases}): ${c.description}`)
              .join("; ")}
            .
          </p>
        )}
        {modelsError && <p className="error">{modelsError}</p>}
        <div className="models">
          {capable.map((m) => (
            <label key={m.name} className={`model ${selectedModels.includes(m.name) ? "on" : ""}`}>
              <input
                type="checkbox"
                checked={selectedModels.includes(m.name)}
                onChange={() => toggle(m.name)}
              />
              <span className="model-body">
                <span className="model-name">{modelLabel(m.name)}</span>
                <span className="muted small">
                  {[m.parameter_size, m.capabilities.includes("thinking") ? "can think" : null]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              </span>
            </label>
          ))}
        </div>
        {unable.length > 0 && (
          <p className="muted small">
            Not offered, because Ollama reports no tool support: {unable.map((m) => modelLabel(m.name)).join(", ")}.
          </p>
        )}
        <label className="small">
          <input type="checkbox" checked={think} onChange={(e) => setThink(e.target.checked)} /> Let
          models think before calling (much slower)
        </label>
        <p>
          <button
            className="primary narrow"
            disabled={!suite || selectedModels.length === 0}
            onClick={() => void start()}
          >
            Run {suite ? suite.case_count : ""} cases on {selectedModels.length} model
            {selectedModels.length === 1 ? "" : "s"}
          </button>
        </p>
        {error && <p className="error">{error}</p>}
      </section>

      {runs.length > 0 && (
        <section className="panel overview-panel">
          <div className="panel-head">
            <h2>Results</h2>
            <select value={runId ?? ""} onChange={(e) => setRunId(e.target.value)}>
              {runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name} · {r.models.map(modelLabel).join(", ")} · {when(r.created_at)}
                </option>
              ))}
            </select>
          </div>
          {detail && detail.id === runId ? <RunView run={detail} /> : <p className="muted">Loading…</p>}
        </section>
      )}
    </div>
  );
}

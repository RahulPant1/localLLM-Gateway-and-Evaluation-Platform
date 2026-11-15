import { useEffect, useRef, useState } from "react";
import { api, type BatchSummary, type RequestRecord, type Rubric, type Taxonomy } from "../api";
import { snippet, when } from "../format";
import { JudgeExchange } from "./JudgeExchange";
import { Markdown } from "./Markdown";

interface Props {
  /** Ticked items: "r:<request id>" for a history run, "b:<batch id>" for a batch. */
  selected: string[];
  onSelect: (keys: string[]) => void;
}

interface Row {
  key: string;
  title: string;
  detail: string;
  outputs: number;
  scored: number;
}

function requestRow(r: RequestRecord): Row {
  const completed = r.executions.filter((e) => e.status === "completed");
  return {
    key: `r:${r.id}`,
    title: snippet(r.prompt, 80),
    detail: `${when(r.created_at)} · ${r.executions.map((e) => e.model).join(", ") || "no model"}`,
    outputs: completed.length,
    scored: completed.filter((e) => e.evaluations.length > 0).length,
  };
}

function batchRow(b: BatchSummary): Row {
  return {
    key: `b:${b.id}`,
    title: b.name,
    detail: `${when(b.created_at)} · ${b.request_count} prompts × ${b.models.length} models`,
    outputs: b.counts.completed,
    scored: b.evaluated_count,
  };
}

const fullyScored = (row: Row) => row.outputs > 0 && row.scored >= row.outputs;

function RowList({ rows, selected, toggle }: { rows: Row[]; selected: string[]; toggle: (key: string) => void }) {
  return (
    <ul className="list">
      {rows.map((row) => (
        <li key={row.key}>
          <label className={`row pick ${selected.includes(row.key) ? "selected" : ""}`}>
            <input
              type="checkbox"
              checked={selected.includes(row.key)}
              disabled={row.outputs === 0}
              onChange={() => toggle(row.key)}
            />
            <span className="pick-body">
              <span className="row-title">{row.title}</span>
              <span className="muted small">
                {row.detail} ·{" "}
                {row.outputs === 0
                  ? "no completed outputs"
                  : row.scored === 0
                    ? `${row.outputs} outputs, not scored`
                    : `${row.scored} of ${row.outputs} outputs scored`}
              </span>
            </span>
          </label>
        </li>
      ))}
    </ul>
  );
}

/** One place to send stored outputs to a judge and bring the scores back. */
export function Evaluate({ selected, onSelect }: Props) {
  const [requests, setRequests] = useState<RequestRecord[]>([]);
  const [batches, setBatches] = useState<BatchSummary[]>([]);
  const [rubric, setRubric] = useState<Rubric | null>(null);
  const [instructions, setInstructions] = useState("");
  const [taxonomy, setTaxonomy] = useState<Taxonomy | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  // Set after scores are stored: the next reload unticks whatever is now fully scored.
  const pruneAfterLoad = useRef(false);
  const selectedRef = useRef(selected);
  selectedRef.current = selected;

  useEffect(() => {
    Promise.all([api.requests(), api.batches()])
      .then(([r, b]) => {
        setRequests(r);
        setBatches(b);
        setError(null);
        if (pruneAfterLoad.current) {
          pruneAfterLoad.current = false;
          const done = new Set(
            [...r.map(requestRow), ...b.map(batchRow)].filter(fullyScored).map((row) => row.key),
          );
          onSelect(selectedRef.current.filter((key) => !done.has(key)));
        }
      })
      .catch((e) => setError((e as Error).message));
  }, [version]);

  useEffect(() => {
    Promise.all([api.rubric(), api.judgePrompt(), api.taskTypes()])
      .then(([r, p, t]) => {
        setRubric(r);
        setInstructions(p.instructions);
        setTaxonomy(t);
      })
      .catch(() => undefined);
  }, []);

  // Fully scored items live in History and Batch runs. They show here only
  // when sent back with "Evaluate again", which ticks them.
  const waiting = (row: Row) => !fullyScored(row) || selected.includes(row.key);
  const allRows = [...requests.map(requestRow), ...batches.map(batchRow)];
  const runRows = requests.map(requestRow).filter(waiting);
  const batchRows = batches.map(batchRow).filter(waiting);
  const all = [...runRows, ...batchRows];
  const hidden = allRows.length - all.length;
  const toggle = (key: string) =>
    onSelect(selected.includes(key) ? selected.filter((k) => k !== key) : [...selected, key]);
  const unscored = all.filter((row) => row.outputs > 0 && row.scored < row.outputs).map((row) => row.key);

  return (
    <div className="columns">
      <section className="panel wide">
        <h2>Evaluate</h2>
        <h3>1. Choose what to judge</h3>
        <div className="batch-bar">
          <button className="secondary" onClick={() => onSelect(unscored)} disabled={unscored.length === 0}>
            Tick everything not fully scored ({unscored.length})
          </button>
          <button className="secondary" onClick={() => onSelect([])} disabled={selected.length === 0}>
            Clear
          </button>
          <button className="secondary" onClick={() => setVersion((v) => v + 1)}>
            Refresh
          </button>
        </div>
        {error && <p className="error">{error}</p>}

        <span className="label">History runs</span>
        {runRows.length === 0 ? (
          <p className="muted small">
            {requests.length === 0 ? "No runs yet. Run a prompt in the Workspace first." : "No runs waiting for scores."}
          </p>
        ) : (
          <RowList rows={runRows} selected={selected} toggle={toggle} />
        )}

        <span className="label spaced">Batches</span>
        {batchRows.length === 0 ? (
          <p className="muted small">{batches.length === 0 ? "No batches yet." : "No batches waiting for scores."}</p>
        ) : (
          <RowList rows={batchRows} selected={selected} toggle={toggle} />
        )}
        {hidden > 0 && (
          <p className="muted small">
            {hidden} fully scored {hidden === 1 ? "item is" : "items are"} not listed. To judge one again, open it in
            History or Batch runs and choose <strong>Evaluate again</strong>.
          </p>
        )}
      </section>

      <section className="panel grow">
        <JudgeExchange
          requestIds={selected.filter((k) => k.startsWith("r:")).map((k) => k.slice(2))}
          batchIds={selected.filter((k) => k.startsWith("b:")).map((k) => k.slice(2))}
          onStored={() => {
            pruneAfterLoad.current = true;
            setVersion((v) => v + 1);
          }}
        />

        <h3>What the judge is told</h3>
        <details>
          <summary>Scoring rubric{rubric ? ` (version ${rubric.version})` : ""}</summary>
          {rubric?.error && <p className="error">{rubric.error}</p>}
          {rubric?.dimensions.map((d) => (
            <div key={d.name} className="rubric-dimension">
              <strong>{d.name.replace(/_/g, " ")}</strong> <span className="muted small">weight {d.weight}</span>
              <p className="small">{d.description}</p>
              <ul className="small">
                {Object.entries(d.anchors).map(([range, meaning]) => (
                  <li key={range}>
                    <strong>{range}</strong>: {meaning}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </details>
        <details>
          <summary>Task types and domains the judge chooses from</summary>
          <p className="small muted">
            The judge labels every prompt with one task type and one domain, so the Overview can show which model
            suits which kind of task. The task types follow the families used by public LLM benchmarks.
          </p>
          {(["task_types", "domains"] as const).map((group) => (
            <div key={group} className="rubric-dimension">
              <strong>{group === "task_types" ? "Task types" : "Domains"}</strong>
              <ul className="small">
                {taxonomy?.[group].map((t) => (
                  <li key={t.name}>
                    <strong>{t.label}</strong>: {t.description}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </details>
        <details>
          <summary>Judge instructions</summary>
          <div className="response">
            <Markdown text={instructions} />
          </div>
        </details>
        <p className="muted small">
          To change either, edit <code>config/rubric.yaml</code> or <code>config/judge_prompt.md</code> and restart the
          backend. In Claude Code, <code>/evaluate-batch &lt;batch name&gt;</code> judges a whole batch without copying.
        </p>
      </section>
    </div>
  );
}

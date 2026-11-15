import { useState } from "react";
import { api, type GroupRow, type Overview as OverviewData } from "../api";
import {
  CATEGORY_TEXT,
  DECIDED_BY_TEXT,
  DIMENSION_TEXT,
  compact,
  duration,
  percent,
  seconds,
} from "../format";
import { usePolling } from "../hooks";

const PERIODS: { label: string; days: number | null }[] = [
  { label: "7 days", days: 7 },
  { label: "30 days", days: 30 },
  { label: "All time", days: null },
];

/** A thin horizontal bar: `fraction` of the track is filled. */
function Bar({ fraction, title }: { fraction: number; title?: string }) {
  const width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  return (
    <div className="bar" title={title}>
      <div className="bar-fill" style={{ width }} />
    </div>
  );
}

function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="tile-value">{value}</div>
      {note && <div className="muted small">{note}</div>}
    </div>
  );
}

function RoutingPanel({ data }: { data: OverviewData }) {
  const { routing } = data;
  const total = routing.live + routing.shadow;
  const medians = new Map(data.by_model.map((row) => [row.key, row.median_ms]));
  return (
    <section className="panel overview-panel">
      <h2>
        Routing <span className="muted">· which model the router picks</span>
      </h2>
      {total === 0 ? (
        <p className="muted">
          No routed requests in this period. Send a prompt in Automatic mode, or run a batch (the
          router records what it would have picked for every batch prompt).
        </p>
      ) : (
        <>
          <div className="flow">
            <div className="flow-node">
              <div className="tile-value">{total}</div>
              <div className="muted small">prompts</div>
            </div>
            <div className="flow-arrow" aria-hidden="true">
              →
            </div>
            <div className="flow-node router">Router</div>
            <div className="flow-arrow" aria-hidden="true">
              →
            </div>
            <div className="flow-targets">
              {routing.by_model.map((item) => (
                <div className="target" key={item.key}>
                  <div className="target-head">
                    <span className="target-name">{item.key}</span>
                    <span className="muted small">
                      {percent(item.share)} · {item.count}
                      {medians.get(item.key) != null && ` · ${seconds(medians.get(item.key))}`}
                    </span>
                  </div>
                  <Bar fraction={item.share} title={`${item.count} of ${total} prompts`} />
                </div>
              ))}
            </div>
          </div>
          <p className="muted small">
            Each line: share of prompts, count, typical answer time.{" "}
            {routing.shadow > 0 &&
              `${routing.shadow} of ${total} are batch prompts, where the pick was recorded but every model ran.`}
          </p>
          <dl className="facts">
            <div>
              <dt>Decided by</dt>
              <dd>
                {Object.entries(routing.decided_by)
                  .sort((a, b) => b[1] - a[1])
                  .map(([key, count]) => `${DECIDED_BY_TEXT[key] ?? key} ${count}`)
                  .join(" · ")}
              </dd>
            </div>
            <div>
              <dt>Average confidence</dt>
              <dd>{routing.mean_confidence == null ? "-" : percent(routing.mean_confidence)}</dd>
            </div>
            <div>
              <dt>Fallback model used</dt>
              <dd>{routing.fallback_used}</dd>
            </div>
            <div>
              <dt>Stronger model advised</dt>
              <dd>{routing.escalations}</dd>
            </div>
          </dl>
          <div className="chips">
            {routing.by_category.map((item) => (
              <span className="pill" key={item.key}>
                {CATEGORY_TEXT[item.key] ?? item.key} {item.count}
              </span>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function PerformancePanel({ data }: { data: OverviewData }) {
  const p = data.performance;
  const estimate = p.cloud_estimate;
  return (
    <section className="panel overview-panel">
      <h2>
        Local performance <span className="muted">· one 8 GB GPU, no API bill</span>
      </h2>
      <div className="tiles">
        <Tile
          label="Tokens generated"
          value={compact(p.output_tokens)}
          note={`${compact(p.prompt_tokens)} read as input`}
        />
        <Tile
          label="Speed"
          value={p.tokens_per_second == null ? "-" : `${p.tokens_per_second} tok/s`}
          note="across all models"
        />
        <Tile
          label="GPU time"
          value={duration(p.generation_seconds)}
          note={`plus ${duration(p.load_seconds)} loading models`}
        />
        <Tile
          label="API cost"
          value="$0"
          note={
            estimate
              ? `about $${estimate.usd.toFixed(2)} at ${estimate.model} list price`
              : undefined
          }
        />
      </div>
      <div className="meter-row">
        <span>
          <b>{percent(p.warm_rate)}</b> of answers needed no model swap
        </span>
        <Bar fraction={p.warm_rate ?? 0} />
      </div>
      <dl className="facts">
        <div>
          <dt>Average model swap</dt>
          <dd>{seconds(p.mean_swap_ms)}</dd>
        </div>
        <div>
          <dt>Average wait in queue</dt>
          <dd>{seconds(p.mean_queue_wait_ms)}</dd>
        </div>
        <div>
          <dt>Answers</dt>
          <dd>
            {p.completed} completed{p.failed > 0 && `, ${p.failed} failed`}
          </dd>
        </div>
      </dl>
      {estimate && (
        <p className="muted small">
          The cloud figure is an estimate: these token counts priced at ${estimate.input_per_mtok}{" "}
          per million input tokens and ${estimate.output_per_mtok} per million output tokens.
        </p>
      )}
    </section>
  );
}

function RequestsPanel({ data }: { data: OverviewData }) {
  const [by, setBy] = useState<"model" | "category">("model");
  const rows: GroupRow[] = by === "model" ? data.by_model : data.by_category;
  return (
    <section className="panel overview-panel">
      <div className="panel-head">
        <h2>
          Requests <span className="muted">· {data.requests} prompts</span>
        </h2>
        <div className="segmented small">
          <button className={by === "model" ? "on" : ""} onClick={() => setBy("model")}>
            By model
          </button>
          <button className={by === "category" ? "on" : ""} onClick={() => setBy("category")}>
            By task
          </button>
        </div>
      </div>
      {rows.length === 0 ? (
        <p className="muted">Nothing has run in this period.</p>
      ) : (
        <div className="table-wrap">
          <table className="matrix overview-table">
            <thead>
              <tr>
                <th>{by === "model" ? "Model" : "Task"}</th>
                <th className="num">Answers</th>
                <th className="num">Typical</th>
                <th className="num">Slowest</th>
                <th>Status</th>
                <th className="num">Tokens</th>
                <th className="num">Tok/s</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.key}>
                  <th>{by === "model" ? row.key : (CATEGORY_TEXT[row.key] ?? row.key)}</th>
                  <td className="num">{row.executions}</td>
                  <td className="num">{seconds(row.median_ms)}</td>
                  <td className="num">{seconds(row.slowest_ms)}</td>
                  <td>
                    {row.failed > 0 ? (
                      <span className="pill bad">● {row.failed} failed</span>
                    ) : (
                      <span className="pill good">● OK</span>
                    )}
                  </td>
                  <td className="num">{compact(row.output_tokens)}</td>
                  <td className="num">{row.tokens_per_second ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="muted small">Typical = median answer time, including any model swap.</p>
    </section>
  );
}

function EvaluationPanel({ data }: { data: OverviewData }) {
  const e = data.evaluation;
  const check = e.router_check;
  const dimensions = Object.keys(DIMENSION_TEXT);
  const [by, setBy] = useState<"task" | "domain">("task");
  const taskRows = by === "task" ? e.by_task : e.by_domain;
  const categorised = taskRows.some((row) => row.key !== "uncategorised");
  return (
    <section className="panel overview-panel">
      <h2>
        Evaluation <span className="muted">· judge scores out of 10</span>
      </h2>
      {e.by_model.length === 0 ? (
        <p className="muted">
          No scores stored in this period. Use the Evaluate tab to send outputs to a judge and
          paste the scores back.
        </p>
      ) : (
        <>
          <div className="table-wrap">
            <table className="matrix overview-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Overall</th>
                  {dimensions.map((name) => (
                    <th className="num" key={name}>
                      {DIMENSION_TEXT[name]}
                    </th>
                  ))}
                  <th className="num">Critical issues</th>
                </tr>
              </thead>
              <tbody>
                {e.by_model.map((row) => (
                  <tr key={row.model}>
                    <th>
                      {row.model}
                      <div className="muted small">{row.evaluated} scored</div>
                    </th>
                    <td>
                      <div className="score-cell">
                        <b>{row.overall?.toFixed(1) ?? "-"}</b>
                        <Bar fraction={(row.overall ?? 0) / 10} />
                      </div>
                    </td>
                    {dimensions.map((name) => (
                      <td className="num" key={name}>
                        {row.dimensions[name]?.toFixed(1) ?? "-"}
                      </td>
                    ))}
                    <td className="num">
                      {row.critical_issues > 0 ? (
                        <span className="pill bad">{row.critical_issues}</span>
                      ) : (
                        0
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small">
            {e.scored} answers scored, {e.unscored} not yet. Judged by {e.judges.join(", ")}.
          </p>

          <div className="panel-head sub">
            <div className="label">Which model suits which task</div>
            <div className="segmented small">
              <button className={by === "task" ? "on" : ""} onClick={() => setBy("task")}>
                By task type
              </button>
              <button className={by === "domain" ? "on" : ""} onClick={() => setBy("domain")}>
                By domain
              </button>
            </div>
          </div>
          <div className="table-wrap">
            <table className="matrix overview-table">
              <thead>
                <tr>
                  <th>{by === "task" ? "Task type" : "Domain"}</th>
                  {e.models.map((model) => (
                    <th className="num" key={model}>
                      {model}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {taskRows.map((row) => (
                  <tr key={row.key}>
                    <th>
                      {row.label}
                      <div className="muted small">
                        {row.prompts} {row.prompts === 1 ? "prompt" : "prompts"}
                      </div>
                    </th>
                    {e.models.map((model) => {
                      const cell = row.models[model];
                      const best = row.best_model === model;
                      return (
                        <td
                          className={`num ${best ? "best" : ""}`}
                          key={model}
                          title={cell ? `${cell.answers} scored answers` : "No scored answers"}
                        >
                          {cell ? cell.mean.toFixed(1) : "-"}
                          {best && <span className="best-mark"> ★ best</span>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small">
            Average overall score. The judge assigns the {by === "task" ? "task type" : "domain"} to each prompt.
            {!categorised &&
              " These scores were stored before judges were asked for it: use Evaluate again in History to have them categorised."}
            {categorised && " With only a few prompts per row, treat the best mark as a hint, not a result."}
          </p>
        </>
      )}
      <div className="router-check">
        <div className="label">Router check</div>
        {check.prompts === 0 ? (
          <p className="muted small">
            Needs prompts where the router's pick and at least one other model were both scored:
            run a batch, then judge it.
          </p>
        ) : (
          <>
            <p>
              The router's pick was the top-scored model in{" "}
              <b>
                {check.picked_top} of {check.prompts}
              </b>{" "}
              prompts ({percent(check.picked_top / check.prompts)}).
            </p>
            <p className="muted small">
              Average score of the router's picks: {check.router_mean?.toFixed(1) ?? "-"}.
              {check.best_fixed_model &&
                ` Always using ${check.best_fixed_model} on the same prompts: ${check.best_fixed_mean?.toFixed(1)}.`}
            </p>
          </>
        )}
      </div>
    </section>
  );
}

export function Overview() {
  const [days, setDays] = useState<number | null>(null);
  const [data, setData] = useState<OverviewData | null>(null);
  const [error, setError] = useState<string | null>(null);

  usePolling(
    async () => {
      try {
        setData(await api.overview(days));
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    10000,
    [days],
  );

  return (
    <div className="overview">
      <div className="panel-head">
        <h2>Overview</h2>
        <div className="segmented small">
          {PERIODS.map((period) => (
            <button
              key={period.label}
              className={days === period.days ? "on" : ""}
              onClick={() => setDays(period.days)}
            >
              {period.label}
            </button>
          ))}
        </div>
      </div>
      {error && <p className="error">{error}</p>}
      {data && (
        <div className="overview-grid">
          <RoutingPanel data={data} />
          <PerformancePanel data={data} />
          <RequestsPanel data={data} />
          <EvaluationPanel data={data} />
        </div>
      )}
      {!data && !error && <p className="muted">Loading…</p>}
    </div>
  );
}

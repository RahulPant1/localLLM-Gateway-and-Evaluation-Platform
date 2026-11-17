import { useState } from "react";
import { api, requestActive, type ModelEntry, type RequestRecord } from "../api";
import { snippet, when } from "../format";
import { usePolling } from "../hooks";
import { ModelPicker } from "./ModelPicker";
import { RequestView } from "./RequestView";

interface Props {
  models: ModelEntry[] | null;
  modelsError: string | null;
  onEvaluate: (key: string) => void;
}

/** One prompt on one or more models, with the earlier prompts beside it. */
export function Playground({ models, modelsError, onEvaluate }: Props) {
  const [prompt, setPrompt] = useState("");
  const [system, setSystem] = useState("");
  const [source, setSource] = useState("");
  const [automatic, setAutomatic] = useState(true);
  const [selected, setSelected] = useState<string[]>([]);
  const [thinking, setThinking] = useState(true);
  const [current, setCurrent] = useState<RequestRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const [recent, setRecent] = useState<RequestRecord[] | null>(null);
  const [showApi, setShowApi] = useState(false);

  const anyActive = recent?.some(requestActive) ?? false;
  usePolling(
    async () => {
      try {
        setRecent(await api.requests());
      } catch (e) {
        setError((e as Error).message);
      }
    },
    anyActive ? 1500 : 10000,
    [anyActive, current?.id],
  );
  const listed = (recent ?? []).filter((r) => showApi || r.source !== "api");
  const apiCount = (recent ?? []).filter((r) => r.source === "api").length;

  const running = current ? requestActive(current) : false;
  usePolling(
    async () => {
      if (!current || !running) return;
      try {
        setCurrent(await api.request(current.id));
      } catch (e) {
        setError((e as Error).message);
      }
    },
    running ? 500 : null,
    [current?.id, running],
  );

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      setCurrent(
        await api.createRequest({
          prompt,
          system: system || null,
          source_material: source || null,
          models: automatic ? null : selected,
          think: thinking ? null : false,
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const ready = prompt.trim().length > 0 && (automatic || selected.length > 0) && !submitting;
  return (
    <div className="columns">
      <div className="stack">
      <section className="panel">
        <h2>Request</h2>
        <textarea
          className="prompt"
          rows={7}
          placeholder="Type a prompt…"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <details>
          <summary>System prompt (optional)</summary>
          <textarea rows={3} value={system} onChange={(e) => setSystem(e.target.value)} />
        </details>
        <details>
          <summary>Source material (optional)</summary>
          <p className="muted small">
            Text the answer should be based on. It is sent after the prompt and lets an evaluator score groundedness.
          </p>
          <textarea rows={5} value={source} onChange={(e) => setSource(e.target.value)} />
        </details>

        <h3>Model</h3>
        <div className="segmented" role="group" aria-label="Model selection">
          <button className={automatic ? "on" : ""} onClick={() => setAutomatic(true)}>
            Automatic
          </button>
          <button className={automatic ? "" : "on"} onClick={() => setAutomatic(false)}>
            Manual
          </button>
        </div>
        {automatic ? (
          <p className="muted small">
            The gateway works out what kind of task this is and picks the most suitable installed model. The decision
            and its reasons are shown with the response.
          </p>
        ) : (
          <>
            <p className="muted small">
              Pick one model, or several to compare them on this prompt. They run one after another.
            </p>
            <ModelPicker models={models} error={modelsError} selected={selected} onChange={setSelected} />
          </>
        )}

        <label className="check">
          <input type="checkbox" checked={thinking} onChange={(e) => setThinking(e.target.checked)} />
          Let thinking-capable models think first (slower, often better)
        </label>

        <button className="primary" disabled={!ready} onClick={submit}>
          {automatic ? "Route and run" : selected.length > 1 ? `Run on ${selected.length} models` : "Run"}
        </button>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="panel">
        <h2>Recent prompts</h2>
        {apiCount > 0 && (
          <label className="list-filter small muted">
            <input type="checkbox" checked={showApi} onChange={(e) => setShowApi(e.target.checked)} />
            Include {apiCount} calls from other tools (API)
          </label>
        )}
        {recent?.length === 0 && <p className="muted small">Nothing has been run yet.</p>}
        <ul className="list">
          {listed.slice(0, 40).map((r) => {
            const failed = r.executions.filter((e) => e.status === "failed").length;
            return (
              <li key={r.id}>
                <button className={`row ${current?.id === r.id ? "selected" : ""}`} onClick={() => setCurrent(r)}>
                  <span className="row-title">{snippet(r.prompt, 70)}</span>
                  <span className="muted small">
                    {when(r.created_at)} · {r.source === "api" && "API · "}
                    {r.routing_mode === "automatic" && "auto: "}
                    {r.executions.map((e) => e.model).join(", ") || "no model"}
                    {requestActive(r) && " · in progress"}
                    {failed > 0 && ` · ${failed} failed`}
                    {r.executions.some((e) => e.follow_up_count > 0) && " · conversation"}
                    {r.executions.some((e) => e.evaluations.length > 0) && " · scored"}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
        {listed.length > 40 && <p className="muted small">Showing the newest 40.</p>}
        <p className="muted small">Batch prompts are under Tests → Batches.</p>
      </section>
      </div>

      <section className="panel grow">
        <h2>{current ? "Result" : "Response"}</h2>
        {current ? (
          <RequestView key={current.id} request={current} onEvaluate={onEvaluate} />
        ) : (
          <p className="muted">Results appear here. Everything is saved: pick an earlier prompt on the left to see it again.</p>
        )}
      </section>
    </div>
  );
}

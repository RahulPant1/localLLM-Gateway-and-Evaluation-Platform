import { useState } from "react";
import { api, requestActive, type ModelEntry, type RequestRecord } from "../api";
import { usePolling } from "../hooks";
import { ModelPicker } from "./ModelPicker";
import { RequestView } from "./RequestView";

interface Props {
  models: ModelEntry[] | null;
  modelsError: string | null;
  onEvaluate: (key: string) => void;
}

export function Workspace({ models, modelsError, onEvaluate }: Props) {
  const [prompt, setPrompt] = useState("");
  const [system, setSystem] = useState("");
  const [source, setSource] = useState("");
  const [automatic, setAutomatic] = useState(true);
  const [selected, setSelected] = useState<string[]>([]);
  const [thinking, setThinking] = useState(true);
  const [current, setCurrent] = useState<RequestRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

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

      <section className="panel grow">
        <h2>Response</h2>
        {current ? (
          <RequestView request={current} showPrompt={false} onEvaluate={onEvaluate} />
        ) : (
          <p className="muted">Results appear here. They are saved, so a page refresh loses nothing; find them under History.</p>
        )}
      </section>
    </div>
  );
}

import { useEffect, useMemo, useRef, useState } from "react";
import { api, type ModelEntry, type PromptInput, type SplitMethod } from "../api";
import { modelLabel, snippet } from "../format";

const METHODS: { value: SplitMethod; label: string; hint: string }[] = [
  { value: "auto", label: "Automatic", hint: "--- lines, else numbered items, else blank lines, else one per line" },
  { value: "numbered", label: "Numbered items", hint: '"1.", "2)", "Prompt 3:" start a new prompt' },
  { value: "blank_lines", label: "Blank lines", hint: "an empty line between prompts" },
  { value: "lines", label: "One per line", hint: "every line is a prompt" },
  { value: "separator", label: "--- lines", hint: "a line holding only --- between prompts" },
];

const METHOD_TEXT: Record<string, string> = {
  separator: "split at --- lines",
  numbered: "split at numbered items",
  blank_lines: "split at blank lines",
  lines: "one prompt per line",
  json: "from the JSON file",
  jsonl: "from the JSON Lines file",
  csv: "from the CSV file",
  smart: "split by a local model",
};

const STRUCTURED = /\.(json|jsonl|ndjson|csv|tsv)$/i;

interface Props {
  models: ModelEntry[] | null;
  onChange: (prompts: PromptInput[]) => void;
  /** Changing this clears the box, after a batch has been started. */
  resetKey: number;
}

/** Paste or upload a prompt set, choose how it is split, and check the result. */
export function PromptSetInput({ models, onChange, resetKey }: Props) {
  const [text, setText] = useState("");
  const [file, setFile] = useState<{ name: string; text: string } | null>(null);
  const [method, setMethod] = useState<SplitMethod>("auto");
  const [parsed, setParsed] = useState<{ prompts: PromptInput[]; method: string; notes: string[]; model: string | null } | null>(null);
  const [removed, setRemoved] = useState<number[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [splitModel, setSplitModel] = useState<string>("");
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setText("");
    setFile(null);
    setParsed(null);
    setRemoved([]);
  }, [resetKey]);

  // The smart split defaults to a gemma model when one is installed.
  useEffect(() => {
    if (!splitModel && models?.length)
      setSplitModel((models.find((m) => m.name.toLowerCase().includes("gemma")) ?? models[0]).name);
  }, [models, splitModel]);

  // Pasted text is split again shortly after typing stops (not for the smart split, which uses the GPU).
  useEffect(() => {
    if (file || method === "smart") return;
    if (!text.trim()) return setParsed(null);
    const timer = setTimeout(() => {
      api
        .parseBatch({ text, method })
        .then((result) => {
          setParsed(result);
          setRemoved([]);
          setError(null);
        })
        .catch((e) => setError((e as Error).message));
    }, 400);
    return () => clearTimeout(timer);
  }, [text, method, file]);

  const kept = useMemo(
    () => (parsed?.prompts ?? []).filter((_, i) => !removed.includes(i)),
    [parsed, removed],
  );
  useEffect(() => onChange(kept), [kept]); // eslint-disable-line react-hooks/exhaustive-deps

  const upload = async (picked: File | undefined) => {
    if (!picked) return;
    const content = await picked.text();
    setError(null);
    if (!STRUCTURED.test(picked.name)) {
      // Plain text goes into the box, where it can still be edited.
      setFile(null);
      setText(content);
      return;
    }
    setFile({ name: picked.name, text: content });
    setBusy(true);
    try {
      setParsed(await api.parseBatch({ text: content, filename: picked.name }));
      setRemoved([]);
    } catch (e) {
      setParsed(null);
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const smartSplit = async () => {
    setBusy(true);
    setError(null);
    try {
      setParsed(await api.parseBatch({ text, method: "smart", model: splitModel || undefined }));
      setRemoved([]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="prompt-set">
      <div className="prompt-set-bar">
        <button className="secondary" onClick={() => fileInput.current?.click()}>
          Upload file
        </button>
        <input
          ref={fileInput}
          type="file"
          hidden
          accept=".txt,.md,.json,.jsonl,.ndjson,.csv,.tsv"
          onChange={(e) => {
            void upload(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        <span className="muted small">.txt, .md, .json, .jsonl or .csv</span>
      </div>

      {file ? (
        <p className="file-chip">
          <strong>{file.name}</strong>{" "}
          <button className="link" onClick={() => (setFile(null), setParsed(null))}>
            remove
          </button>
        </p>
      ) : (
        <>
          <textarea
            className="prompt"
            rows={10}
            placeholder={
              "Paste prompts here, for example a list another chat model wrote:\n\n1. First prompt…\n2. Second prompt…\n\nor separate them with blank lines."
            }
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="prompt-set-bar">
            <label className="small">
              Split by{" "}
              <select value={method === "smart" ? "auto" : method} onChange={(e) => setMethod(e.target.value as SplitMethod)}>
                {METHODS.map((m) => (
                  <option key={m.value} value={m.value} title={m.hint}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
            <span className="muted small">or</span>
            <button
              className="secondary"
              disabled={!text.trim() || busy || !splitModel}
              title="A local model marks where each prompt starts and which lines are not prompts. Your text is never reworded."
              onClick={() => void smartSplit()}
            >
              {busy ? "Splitting…" : "Smart split"}
            </button>
            {models && models.length > 1 && (
              <select className="small" value={splitModel} onChange={(e) => setSplitModel(e.target.value)}>
                {models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {modelLabel(m.name)}
                  </option>
                ))}
              </select>
            )}
          </div>
        </>
      )}

      {error && <p className="error small">{error}</p>}
      {parsed && (
        <div className="prompt-preview">
          <p className="small">
            <strong>{kept.length}</strong> prompt{kept.length === 1 ? "" : "s"}{" "}
            <span className="muted">
              · {METHOD_TEXT[parsed.method] ?? parsed.method}
              {parsed.model ? ` (${modelLabel(parsed.model)})` : ""}
              {removed.length > 0 && ` · ${removed.length} removed`}
            </span>
          </p>
          {parsed.notes.map((n) => (
            <p key={n} className="muted small">
              {n}
            </p>
          ))}
          <ol>
            {parsed.prompts.map((p, i) =>
              removed.includes(i) ? null : (
                <li key={i}>
                  <span className="muted small num">{i + 1}.</span>
                  <details>
                    <summary>
                      {snippet(p.prompt, 110)}
                      {p.source_material && <span className="pill small">source</span>}
                      {p.system && <span className="pill small">system</span>}
                    </summary>
                    <pre>{p.prompt}</pre>
                    {p.source_material && <pre className="muted">{p.source_material}</pre>}
                  </details>
                  <button className="link" title="Leave this prompt out" onClick={() => setRemoved([...removed, i])}>
                    ✕
                  </button>
                </li>
              ),
            )}
          </ol>
        </div>
      )}
    </div>
  );
}

import type { ModelEntry } from "../api";
import { gb } from "../format";

interface Props {
  models: ModelEntry[] | null;
  error: string | null;
  selected: string[];
  onChange: (names: string[]) => void;
}

export function ModelPicker({ models, error, selected, onChange }: Props) {
  if (error) return <p className="error">Could not list models: {error}</p>;
  if (!models) return <p className="muted">Loading models…</p>;
  if (models.length === 0) return <p className="muted">No models are installed in Ollama.</p>;

  const toggle = (name: string) =>
    onChange(selected.includes(name) ? selected.filter((n) => n !== name) : [...selected, name]);

  return (
    <div className="models">
      {models.map((m) => (
        <label key={m.name} className={`model ${selected.includes(m.name) ? "on" : ""}`}>
          <input type="checkbox" checked={selected.includes(m.name)} onChange={() => toggle(m.name)} />
          <span className="model-body">
            <span className="model-name">
              {m.name}
              {m.loaded && <span className="pill good small">loaded</span>}
            </span>
            <span className="muted small">
              {[m.parameter_size, m.quantization, gb(m.size_bytes), `context ${m.num_ctx}`]
                .filter(Boolean)
                .join(" · ")}
            </span>
            <span className="muted small">
              {m.details_available ? m.capabilities.join(", ") : "details unavailable"}
            </span>
          </span>
        </label>
      ))}
    </div>
  );
}

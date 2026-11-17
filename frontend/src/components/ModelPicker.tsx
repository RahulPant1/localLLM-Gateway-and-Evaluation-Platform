import type { ModelEntry } from "../api";
import { gb, modelLabel } from "../format";

interface Props {
  models: ModelEntry[] | null;
  error: string | null;
  selected: string[];
  onChange: (names: string[]) => void;
}

/** Models as compact toggle chips. Details (strengths, context) are in each chip's tooltip. */
export function ModelPicker({ models, error, selected, onChange }: Props) {
  if (error) return <p className="error">Could not list models: {error}</p>;
  if (!models) return <p className="muted">Loading models…</p>;
  if (models.length === 0) return <p className="muted">No models are available.</p>;

  const toggle = (name: string) =>
    onChange(selected.includes(name) ? selected.filter((n) => n !== name) : [...selected, name]);
  const all = models.map((m) => m.name);

  return (
    <div className="chip-picker">
      <div className="chips">
        {models.map((m) => (
          <label
            key={m.name}
            className={`chip ${selected.includes(m.name) ? "on" : ""}`}
            title={[
              m.name,
              [m.parameter_size, m.quantization, gb(m.size_bytes), `context ${m.num_ctx}`].filter(Boolean).join(" · "),
              m.strengths || m.capabilities.join(", "),
              m.profiled ? "" : "Not in the routing table, so automatic routing never picks it.",
            ]
              .filter(Boolean)
              .join("\n")}
          >
            <input type="checkbox" checked={selected.includes(m.name)} onChange={() => toggle(m.name)} />
            <span className="chip-name">{modelLabel(m.name)}</span>
            <span className="chip-meta">{m.parameter_size}</span>
            {m.loaded && <span className="chip-dot" title="Loaded on the GPU" />}
          </label>
        ))}
      </div>
      {models.length > 2 && (
        <span className="chip-actions small">
          <button className="link" onClick={() => onChange(all)} disabled={selected.length === all.length}>
            All
          </button>
          <button className="link" onClick={() => onChange([])} disabled={selected.length === 0}>
            None
          </button>
        </span>
      )}
    </div>
  );
}

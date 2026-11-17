import { useState } from "react";
import type { ModelEntry } from "../api";
import { Agents } from "./Agents";
import { Batches } from "./Batches";
import { Models } from "./Models";
import { ToolUse } from "./ToolUse";

const VIEWS = ["Models", "Batches", "Tool use", "Agents"] as const;
type View = (typeof VIEWS)[number];

const HINTS: Record<View, string> = {
  Models: "Which models are compared, and how much of each test they have done",
  Batches: "Many prompts on many models, judged later",
  "Tool use": "Function-calling cases, scored automatically",
  Agents: "Multi-step tasks on simulated files, mail and an API",
};

interface Props {
  models: ModelEntry[] | null;
  modelsError: string | null;
  onEvaluate: (key?: string) => void;
}

/** Everything that runs models side by side on fixed tests. */
export function Tests({ models, modelsError, onEvaluate }: Props) {
  const [view, setView] = useState<View>("Models");
  return (
    <>
      <nav className="subnav" aria-label="Tests">
        {VIEWS.map((v) => (
          <button key={v} className={view === v ? "on" : ""} title={HINTS[v]} onClick={() => setView(v)}>
            {v}
          </button>
        ))}
      </nav>
      {/* Batches stays mounted so a half-typed batch survives switching views. */}
      <div hidden={view !== "Models"}>{view === "Models" && <Models onEvaluate={() => onEvaluate()} />}</div>
      <div hidden={view !== "Batches"}>
        <Batches models={models} modelsError={modelsError} onEvaluate={(key) => onEvaluate(key)} />
      </div>
      <div hidden={view !== "Agents"}>{view === "Agents" && <Agents models={models} />}</div>
      <div hidden={view !== "Tool use"}>
        {view === "Tool use" && <ToolUse models={models} modelsError={modelsError} />}
      </div>
    </>
  );
}

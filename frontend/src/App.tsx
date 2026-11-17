import { useState } from "react";
import { api, type ModelEntry } from "./api";
import { Batches } from "./components/Batches";
import { Evaluate } from "./components/Evaluate";
import { History } from "./components/History";
import { Overview } from "./components/Overview";
import { Models } from "./components/Models";
import { ToolUse } from "./components/ToolUse";
import { StatusStrip } from "./components/StatusStrip";
import { Workspace } from "./components/Workspace";
import { usePolling } from "./hooks";

const TABS = ["Overview", "Workspace", "Batch runs", "Tool use", "History", "Evaluate", "Models"] as const;
type Tab = (typeof TABS)[number];

export function App() {
  const [tab, setTab] = useState<Tab>("Overview");
  const [models, setModels] = useState<ModelEntry[] | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);
  // Items ticked in the Evaluate tab; kept here so they survive switching tabs.
  const [toEvaluate, setToEvaluate] = useState<string[]>([]);
  const evaluate = (key: string) => {
    setToEvaluate((keys) => (keys.includes(key) ? keys : [...keys, key]));
    setTab("Evaluate");
  };

  // Refreshed regularly so the "loaded" marker and newly installed models stay current.
  usePolling(
    async () => {
      try {
        setModels((await api.models()).models);
        setModelsError(null);
      } catch (e) {
        setModelsError((e as Error).message);
      }
    },
    5000,
    [],
  );

  return (
    <>
      <header className="top">
        <h1>Local LLM Gateway</h1>
        <nav>
          {TABS.map((t) => (
            <button key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
              {t}
            </button>
          ))}
        </nav>
      </header>
      <StatusStrip />
      {/* All tabs stay mounted so typed text and open results survive switching. */}
      <main>
        <div hidden={tab !== "Overview"}>{tab === "Overview" && <Overview />}</div>
        <div hidden={tab !== "Workspace"}>
          <Workspace models={models} modelsError={modelsError} onEvaluate={evaluate} />
        </div>
        <div hidden={tab !== "Batch runs"}>
          <Batches models={models} modelsError={modelsError} onEvaluate={evaluate} />
        </div>
        <div hidden={tab !== "Tool use"}>
          {tab === "Tool use" && <ToolUse models={models} modelsError={modelsError} />}
        </div>
        <div hidden={tab !== "History"}>{tab === "History" && <History onEvaluate={evaluate} />}</div>
        <div hidden={tab !== "Models"}>
          {tab === "Models" && <Models onEvaluate={() => setTab("Evaluate")} />}
        </div>
        <div hidden={tab !== "Evaluate"}>
          {tab === "Evaluate" && <Evaluate selected={toEvaluate} onSelect={setToEvaluate} />}
        </div>
      </main>
    </>
  );
}

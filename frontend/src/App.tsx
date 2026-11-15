import { useState } from "react";
import { api, type ModelEntry } from "./api";
import { Batches } from "./components/Batches";
import { History } from "./components/History";
import { StatusStrip } from "./components/StatusStrip";
import { Workspace } from "./components/Workspace";
import { usePolling } from "./hooks";

const TABS = ["Workspace", "Batch runs", "History"] as const;
type Tab = (typeof TABS)[number];

export function App() {
  const [tab, setTab] = useState<Tab>("Workspace");
  const [models, setModels] = useState<ModelEntry[] | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);

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
        <div hidden={tab !== "Workspace"}>
          <Workspace models={models} modelsError={modelsError} />
        </div>
        <div hidden={tab !== "Batch runs"}>
          <Batches models={models} modelsError={modelsError} />
        </div>
        <div hidden={tab !== "History"}>{tab === "History" && <History />}</div>
      </main>
    </>
  );
}

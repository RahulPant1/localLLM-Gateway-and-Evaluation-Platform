import { useState } from "react";
import { api, requestActive, type RequestRecord } from "../api";
import { snippet, when } from "../format";
import { usePolling } from "../hooks";
import { RequestView } from "./RequestView";

export function History({ onEvaluate }: { onEvaluate: (key: string) => void }) {
  const [requests, setRequests] = useState<RequestRecord[] | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const anyActive = requests?.some(requestActive) ?? false;
  usePolling(
    async () => {
      try {
        setRequests(await api.requests());
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    anyActive ? 1500 : 10000,
    [anyActive],
  );

  const open = requests?.find((r) => r.id === openId) ?? null;
  return (
    <div className="columns">
      <section className="panel">
        <h2>History</h2>
        <p className="muted small">Single and comparison requests, newest first. Batch runs are under Batch runs.</p>
        {error && <p className="error">{error}</p>}
        {requests?.length === 0 && <p className="muted">Nothing has been run yet.</p>}
        <ul className="list">
          {requests?.map((r) => {
            const failed = r.executions.filter((e) => e.status === "failed").length;
            return (
              <li key={r.id}>
                <button className={`row ${openId === r.id ? "selected" : ""}`} onClick={() => setOpenId(r.id)}>
                  <span className="row-title">{snippet(r.prompt, 70)}</span>
                  <span className="muted small">
                    {when(r.created_at)} · {r.routing_mode === "automatic" && "auto: "}
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
      </section>
      <section className="panel grow">
        <h2>Details</h2>
        {open ? <RequestView request={open} onEvaluate={onEvaluate} /> : <p className="muted">Select a request to see its results.</p>}
      </section>
    </div>
  );
}

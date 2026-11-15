import { useEffect, useState } from "react";
import { api, type JudgePackage } from "../api";

/** Pull the JSON object or array out of a pasted reply, which may be wrapped in prose or a code fence. */
export function extractEvaluations(pasted: string, defaultJudge: string): unknown {
  const starts = [pasted.indexOf("{"), pasted.indexOf("[")].filter((i) => i >= 0);
  const end = Math.max(pasted.lastIndexOf("}"), pasted.lastIndexOf("]"));
  if (starts.length === 0 || end < 0) throw new Error("No JSON found in the pasted text.");
  let parsed: unknown;
  try {
    parsed = JSON.parse(pasted.slice(Math.min(...starts), end + 1));
  } catch (e) {
    throw new Error(`The pasted text is not valid JSON (${(e as Error).message}).`);
  }
  const list = Array.isArray(parsed) ? parsed : (parsed as { evaluations?: unknown }).evaluations;
  if (!Array.isArray(list) || list.length === 0) {
    throw new Error('Expected {"evaluations": [...]} with at least one entry.');
  }
  // Fill in the judge name where the reply left it out.
  return { evaluations: list.map((entry) => ({ judge: defaultJudge, ...(entry as object) })) };
}

interface Props {
  requestIds: string[];
  batchIds: string[];
  /** Called after scores are stored, so lists can refresh. */
  onStored: () => void;
}

/** Copy everything a judge needs out of the app, and paste the judge's scores back in. */
export function JudgeExchange({ requestIds, batchIds, onStored }: Props) {
  const [blind, setBlind] = useState(true);
  const [skipScored, setSkipScored] = useState(false);
  const [judge, setJudge] = useState("claude-code");
  const [pack, setPack] = useState<JudgePackage | null>(null);
  const [copied, setCopied] = useState(false);
  const [reply, setReply] = useState("");
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState(0);

  const nothing = requestIds.length + batchIds.length === 0;
  const selectionKey = `${requestIds.join(",")}|${batchIds.join(",")}`;
  const judgeName = judge.trim() || "claude-code";

  useEffect(() => {
    setCopied(false);
    setError(null);
    if (nothing) {
      setPack(null);
      return;
    }
    let cancelled = false;
    api
      .judgePackage({
        request_ids: requestIds,
        batch_ids: batchIds,
        blind,
        exclude_judge: skipScored ? judgeName : null,
      })
      .then((result) => !cancelled && setPack(result))
      .catch((e) => !cancelled && setError((e as Error).message));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectionKey, blind, skipScored, skipScored ? judgeName : "", version]);

  const copy = async () => {
    if (!pack) return;
    try {
      await navigator.clipboard.writeText(pack.text);
      setCopied(true);
    } catch {
      setError("The browser blocked copying. Click in the box, press Ctrl+A, then Ctrl+C.");
    }
  };

  const store = async () => {
    setMessage(null);
    try {
      const result = await api.importEvaluations(extractEvaluations(reply, judgeName));
      setMessage({ ok: true, text: `Stored ${result.stored} evaluations. They now show on each answer.` });
      setReply("");
      setVersion((v) => v + 1);
      onStored();
    } catch (e) {
      setMessage({ ok: false, text: (e as Error).message });
    }
  };

  return (
    <>
      <h3>2. Copy the evaluation prompt and outputs</h3>
      {nothing ? (
        <p className="muted">Tick at least one run or batch on the left.</p>
      ) : (
        <>
          <p className="muted small">
            One text with the judge instructions, the scoring rubric and every completed output you ticked. Paste it
            into Claude Code or a Claude chat and send it.
          </p>
          <label className="check">
            <input type="checkbox" checked={blind} onChange={(e) => setBlind(e.target.checked)} />
            Hide model names, so the judge cannot favour a model (recommended)
          </label>
          <label className="check">
            <input type="checkbox" checked={skipScored} onChange={(e) => setSkipScored(e.target.checked)} />
            Leave out outputs already scored by
            <input type="text" className="inline" value={judge} onChange={(e) => setJudge(e.target.value)} />
          </label>
          {error && <p className="error">{error}</p>}
          {pack && (
            <>
              <textarea className="package" readOnly rows={12} value={pack.text} onFocus={(e) => e.target.select()} />
              <div className="batch-bar">
                <button className="primary narrow" onClick={copy} disabled={pack.item_count === 0}>
                  {copied ? "Copied ✓" : "Copy to clipboard"}
                </button>
                <span className="muted small">
                  {pack.item_count} outputs · rubric {pack.rubric_version} · about{" "}
                  {Math.round(pack.text.length / 4).toLocaleString()} tokens
                </span>
              </div>
              {pack.item_count === 0 && (
                <p className="warning">Nothing to judge: the ticked items have no completed, unscored outputs.</p>
              )}
            </>
          )}
        </>
      )}

      <h3>3. Paste the reply to store the scores</h3>
      <p className="muted small">
        Paste Claude's JSON reply here. Surrounding text or a code fence is fine. The overall score is calculated by the
        gateway, and nothing already stored is overwritten.
      </p>
      <textarea rows={6} placeholder={'{"evaluations": [ … ]}'} value={reply} onChange={(e) => setReply(e.target.value)} />
      <div className="batch-bar">
        <button className="primary narrow" disabled={!reply.trim()} onClick={store}>
          Store scores
        </button>
        <span className="muted small">Stored under the judge name "{judgeName}" unless the reply names another.</span>
      </div>
      {message && <p className={message.ok ? "success" : "error"}>{message.text}</p>}
    </>
  );
}

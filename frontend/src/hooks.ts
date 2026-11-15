import { useEffect, useRef, useState } from "react";

/** Call `fn` now and then every `intervalMs` (null = call once, no repeat). */
export function usePolling(fn: () => Promise<void> | void, intervalMs: number | null, deps: unknown[]) {
  const saved = useRef(fn);
  saved.current = fn;
  useEffect(() => {
    let cancelled = false;
    const tick = () => {
      if (!cancelled) void saved.current();
    };
    tick();
    if (intervalMs === null) return () => void (cancelled = true);
    const timer = setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, ...deps]);
}

/** The current time, refreshed every second while `active`. Drives elapsed timers. */
export function useNow(active: boolean): number {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return now;
}

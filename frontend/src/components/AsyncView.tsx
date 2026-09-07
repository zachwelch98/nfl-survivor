import type { ReactNode } from "react";
import type { AsyncState } from "../lib/useAsync";

export function AsyncView<T>({ state, children }: { state: AsyncState<T>; children: (data: T) => ReactNode }) {
  if (state.status === "loading") {
    return (
      <div className="flex items-center justify-center py-24 text-slate-400 text-sm">
        Loading…
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 text-red-700 px-4 py-3 text-sm">
        Couldn't load data: {state.error}
      </div>
    );
  }
  return <>{children(state.data)}</>;
}

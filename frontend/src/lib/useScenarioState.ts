import { useCallback, useState } from "react";
import {
  clearAllLocks,
  clearLock,
  importFromPicks,
  loadScenarioState,
  saveScenarioState,
  setLock,
} from "../data/scenario";
import type { ScenarioLock, ScenarioState, UserState } from "../data/types";

export function useScenarioState() {
  const [state, setState] = useState<ScenarioState>(() => loadScenarioState());

  const persist = useCallback((update: (prev: ScenarioState) => ScenarioState) => {
    setState((prev) => {
      const next = update(prev);
      saveScenarioState(next);
      return next;
    });
  }, []);

  const lock = useCallback((l: ScenarioLock) => persist((prev) => setLock(prev, l)), [persist]);
  const unlock = useCallback(
    (season: number, week: number) => persist((prev) => clearLock(prev, season, week)),
    [persist],
  );
  const clearAll = useCallback(
    (season: number) => persist((prev) => clearAllLocks(prev, season)),
    [persist],
  );
  const importPicks = useCallback(
    (userState: UserState, season: number, currentWeek: number) =>
      persist((prev) => importFromPicks(prev, userState, season, currentWeek)),
    [persist],
  );

  return { state, lock, unlock, clearAll, importPicks };
}

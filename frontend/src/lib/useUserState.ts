import { useCallback, useState } from "react";
import { loadUserState, removePick, saveUserState, upsertPick } from "../data/storage";
import type { UserPick, UserState } from "../data/types";

export function useUserState() {
  const [state, setState] = useState<UserState>(() => loadUserState());

  const setPick = useCallback((pick: UserPick) => {
    setState((prev) => {
      const next = upsertPick(prev, pick);
      saveUserState(next);
      return next;
    });
  }, []);

  const clearPick = useCallback((season: number, week: number) => {
    setState((prev) => {
      const next = removePick(prev, season, week);
      saveUserState(next);
      return next;
    });
  }, []);

  return { state, setPick, clearPick };
}

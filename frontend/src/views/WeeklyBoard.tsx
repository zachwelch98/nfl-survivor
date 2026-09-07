import { useState } from "react";
import { AsyncView } from "../components/AsyncView";
import { Card } from "../components/Card";
import { ProbPill } from "../components/ProbBar";
import { useAsync } from "../lib/useAsync";
import { getSeasonMeta, getWeekBoard } from "../data/source";
import { teamName } from "../lib/teams";
import { useUserState } from "../lib/useUserState";
import { usedTeams } from "../data/storage";

export default function WeeklyBoard() {
  const meta = useAsync(() => getSeasonMeta(), []);
  const [week, setWeek] = useState<number | null>(null);
  const { state } = useUserState();

  return (
    <AsyncView state={meta}>
      {(m) => {
        const activeWeek = week ?? m.current_week;
        const usedByUser = usedTeams(state, m.season, activeWeek - 1);
        return (
          <div className="space-y-4">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <h1 className="text-2xl font-semibold tracking-tight">Weekly Odds Board</h1>
              <label className="flex items-center gap-2 text-sm text-slate-600">
                Week
                <select
                  className="rounded-md border border-slate-300 bg-white px-2 py-1 text-sm"
                  value={activeWeek}
                  onChange={(e) => setWeek(Number(e.target.value))}
                >
                  {Array.from({ length: 18 }, (_, i) => i + 1).map((w) => (
                    <option key={w} value={w}>
                      Week {w}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <BoardTable week={activeWeek} usedByUser={usedByUser} />
          </div>
        );
      }}
    </AsyncView>
  );
}

function BoardTable({ week, usedByUser }: { week: number; usedByUser: Set<string> }) {
  const board = useAsync(() => getWeekBoard(week), [week]);
  return (
    <AsyncView state={board}>
      {(b) => (
        <Card>
          {b.recommended_pick && (
            <div className="mb-4 rounded-lg bg-accent-soft border border-blue-200 px-3 py-2 text-sm text-accent font-medium">
              ⭐ Optimizer recommends <strong>{b.recommended_pick}</strong> ({teamName(b.recommended_pick)}) this week
            </div>
          )}
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wide text-slate-400 border-b border-slate-200">
                  <th className="py-2 pr-3">Matchup</th>
                  <th className="py-2 px-3">Spread</th>
                  <th className="py-2 px-3">Vegas home %</th>
                  <th className="py-2 px-3">MC home %</th>
                  <th className="py-2 px-3">MC spread</th>
                  <th className="py-2 pl-3">Deviation cost</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {b.games.map((g) => {
                  const homeRec = b.recommended_pick === g.home_team;
                  const awayRec = b.recommended_pick === g.away_team;
                  const homeUsed = usedByUser.has(g.home_team);
                  const awayUsed = usedByUser.has(g.away_team);
                  return (
                    <tr key={g.game_id} className={homeRec || awayRec ? "bg-amber-50/60" : ""}>
                      <td className="py-2 pr-3">
                        <span className={awayUsed ? "line-through text-slate-300" : awayRec ? "font-semibold" : ""}>{g.away_team}</span>
                        <span className="text-slate-300 mx-1">@</span>
                        <span className={homeUsed ? "line-through text-slate-300" : homeRec ? "font-semibold" : ""}>{g.home_team}</span>
                      </td>
                      <td className="py-2 px-3 tabular-nums text-slate-500">{g.spread_line !== null ? g.spread_line.toFixed(1) : "—"}</td>
                      <td className="py-2 px-3">
                        <ProbPill value={g.vegas_home_win_prob} />
                      </td>
                      <td className="py-2 px-3">
                        <ProbPill value={g.mc_home_win_prob} />
                      </td>
                      <td className="py-2 px-3 tabular-nums text-slate-500">{g.mc_simulated_spread !== null ? g.mc_simulated_spread.toFixed(1) : "—"}</td>
                      <td className="py-2 pl-3 tabular-nums text-slate-500">
                        {[g.home_team, g.away_team].map((t) => {
                          const cost = b.cost_of_deviation[t];
                          if (cost === undefined) return null;
                          return (
                            <span key={t} className="mr-2">
                              {t}: {cost.toFixed(3)}
                            </span>
                          );
                        })}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </AsyncView>
  );
}

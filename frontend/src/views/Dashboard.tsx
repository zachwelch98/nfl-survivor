import { useState } from "react";
import { Link } from "react-router-dom";
import { AsyncView } from "../components/AsyncView";
import { Card, StatTile } from "../components/Card";
import { CommandRunner } from "../components/CommandRunner";
import { ProbPill } from "../components/ProbBar";
import { useAsync } from "../lib/useAsync";
import { getSeasonMeta, getSeasonPlan, getWeekBoard } from "../data/source";
import { teamName } from "../lib/teams";
import { useUserState } from "../lib/useUserState";
import { survivalRecord, usedTeams } from "../data/storage";

export default function Dashboard() {
  // Bumped after a pipeline command (refresh/export) succeeds, so the views
  // below refetch the JSON files that command just regenerated instead of
  // showing what was loaded on first render.
  const [refreshNonce, setRefreshNonce] = useState(0);
  const meta = useAsync(() => getSeasonMeta(), [refreshNonce]);
  const { state } = useUserState();

  return (
    <AsyncView state={meta}>
      {(m) => {
        const record = survivalRecord(state, m.season);
        const myUsedTeams = Array.from(usedTeams(state, m.season));
        return (
          <div className="space-y-6">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">
                {m.season} Season — Week {m.current_week}
              </h1>
              <p className="text-sm text-slate-500 mt-1">
                Primary method: <span className="font-medium text-ink">{m.primary_method}</span> (backtesting shows this
                is currently the best-calibrated of the two — see Analysis). Generated {new Date(m.generated_at).toLocaleString()}.
              </p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <StatTile label="Wins" value={record.wins} tone="good" />
              <StatTile label="Losses" value={record.losses} tone={record.losses > 0 ? "bad" : "default"} />
              <StatTile label="Current streak" value={record.currentStreak} />
              <StatTile label="Teams used" value={m.used_teams.length} />
            </div>

            <RecommendedPickCard season={m.season} week={m.current_week} refreshNonce={refreshNonce} />
            <SeasonPreviewCard refreshNonce={refreshNonce} />
            <CommandRunner
              season={m.season}
              week={m.current_week}
              usedTeams={myUsedTeams}
              onSuccess={() => setRefreshNonce((n) => n + 1)}
            />
          </div>
        );
      }}
    </AsyncView>
  );
}

function RecommendedPickCard({ season, week, refreshNonce }: { season: number; week: number; refreshNonce: number }) {
  const board = useAsync(() => getWeekBoard(week), [week, refreshNonce]);
  return (
    <AsyncView state={board}>
      {(b) => {
        const top = [...b.games].sort((a, z) => Math.max(z.vegas_home_win_prob, 1 - z.vegas_home_win_prob) - Math.max(a.vegas_home_win_prob, 1 - a.vegas_home_win_prob));
        return (
          <Card title={`Week ${week} recommended pick`} subtitle="Optimized for full-season survival, not just this week's win probability">
            {b.recommended_pick ? (
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 rounded-xl bg-accent-soft flex items-center justify-center text-lg font-bold text-accent">
                  {b.recommended_pick}
                </div>
                <div>
                  <div className="text-xl font-semibold">{teamName(b.recommended_pick)}</div>
                  <div className="text-sm text-slate-500">
                    Season {season}, Week {week}
                  </div>
                </div>
                <Link to="/plan" className="ml-auto text-sm font-medium text-accent hover:underline">
                  View full season plan →
                </Link>
              </div>
            ) : (
              <p className="text-slate-500 text-sm">No feasible recommendation (out of eligible teams?).</p>
            )}

            <div className="mt-4 divide-y divide-slate-100 border-t border-slate-100">
              {top.slice(0, 5).map((g) => (
                <div key={g.game_id} className="flex items-center justify-between py-2 text-sm">
                  <span className="text-slate-600">
                    {g.away_team} @ {g.home_team}
                  </span>
                  <div className="flex items-center gap-2">
                    <ProbPill value={g.vegas_home_win_prob >= 0.5 ? g.vegas_home_win_prob : 1 - g.vegas_home_win_prob} />
                    <span className="text-xs text-slate-400 w-20 text-right">
                      {g.vegas_home_win_prob >= 0.5 ? g.home_team : g.away_team} favored
                    </span>
                  </div>
                </div>
              ))}
            </div>
            <Link to="/board" className="mt-3 inline-block text-sm font-medium text-accent hover:underline">
              See full odds board →
            </Link>
          </Card>
        );
      }}
    </AsyncView>
  );
}

function SeasonPreviewCard({ refreshNonce }: { refreshNonce: number }) {
  const plan = useAsync(() => getSeasonPlan(), [refreshNonce]);
  return (
    <AsyncView state={plan}>
      {(p) => (
        <Card title="Full-season plan preview" subtitle={p.total_survival_prob ? `Total survival probability: ${(p.total_survival_prob * 100).toFixed(2)}%` : undefined}>
          <div className="flex flex-wrap gap-2">
            {p.recommended_sequence.map((pick) => (
              <div key={pick.week} className="rounded-lg border border-slate-200 px-2.5 py-1.5 text-center min-w-16">
                <div className="text-[10px] uppercase tracking-wide text-slate-400">Wk {pick.week}</div>
                <div className="text-sm font-semibold">{pick.team}</div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </AsyncView>
  );
}
